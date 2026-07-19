"""BYO runtime: prove ChorusGraph features work without ChorusGraph.

Uses only ``prismshine.wiring`` + ``ShineGate`` — the contract any LangGraph /
custom orchestrator user implements.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismshine.gate import ShineGate
from prismshine.runtime import assert_adapter, check_adapter
from prismshine.wiring import (
    ShineNotWiredError,
    append_trace,
    is_shine_wired,
    make_dict_adapter,
    mark_shine_wired,
    post_llm_check,
    pre_llm_check,
    record_cache,
    record_llm_error,
    record_retrieval,
    require_shine_wiring,
    shine_verify_node,
    wrap_llm,
)


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def _gate():
    return ShineGate.build(embedder=_embed)


# ---------------------------------------------------------------------------
# Fake orchestrator — no ChorusGraph, no LangGraph
# ---------------------------------------------------------------------------


class FakeApp:
    """Minimal graph stand-in: users mark wiring + call wrap_llm / verify node."""

    def __init__(self):
        self.state: dict = {}


def test_dict_adapter_conforms():
    adapter = make_dict_adapter(_gate())
    assert check_adapter(adapter) == []
    assert_adapter(adapter)


def test_require_shine_wiring_any_runtime():
    app = FakeApp()
    with pytest.raises(ShineNotWiredError):
        require_shine_wiring(None, _gate())
    require_shine_wiring(app, _gate(), attach_node=True)
    assert is_shine_wired(app)
    assert hasattr(app, "_prismshine_node_factory")


def test_pre_llm_empty_retrieval_halts_without_chorusgraph():
    gate = _gate()
    state = {
        "question": "What was revenue?",
        "docs": [],
        "declared_sections": ["must_ground"],
        "trace": [record_retrieval("retrieve", n_chunks=0, top_k=3)],
    }
    decision = pre_llm_check(gate, state)
    assert decision.should_halt
    assert decision.verdict is not None
    assert "EMPTY_RETRIEVAL" in {s.id for s in decision.verdict.signatures}


def test_wrap_llm_halts_before_model_call():
    gate = _gate()
    called = {"n": 0}

    def model(system, user):
        called["n"] += 1
        return "hallucinated"

    state = {
        "question": "What was revenue?",
        "docs": [],
        "declared_sections": ["must_ground"],
        "trace": [record_retrieval("retrieve", n_chunks=0)],
    }
    wrapped = wrap_llm(model, gate, state_factory=lambda: state)
    out = wrapped("sys", "What was revenue?")
    assert called["n"] == 0
    assert "don't have" in out.lower() or out != "hallucinated"


def test_cache_miss_skip_and_llm_error_via_trace_helpers():
    gate = _gate()
    adapter = make_dict_adapter(gate)
    state = {
        "question": "q",
        "answer": "guess",
        "docs": [{"chunk_id": "c1", "text": "stale"}],
        "declared_sections": ["must_ground"],
        "trace": [
            record_cache("cache", "MISS"),
            record_llm_error("gen", error="429"),
        ],
    }
    v = adapter.post_llm_hook(state)
    ids = {s.id for s in v.signatures}
    assert "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS" in ids
    assert "LLM_ERROR" in ids


def test_trace_incomplete_via_verify_node():
    gate = _gate()
    node = shine_verify_node(gate)
    out = node(
        {
            "question": "q",
            "answer": "hello world today",
            "docs": ["hello world today"],
            "consumes": ["docs"],
            "expect_trace_kinds": ["retrieval"],
            "trace": [],
        }
    )
    ids = {s["id"] for s in out["shine_verdict"]["signatures"]}
    assert "TRACE_INCOMPLETE" in ids
    assert out["shine_route"] in {"pass", "flag", "block", "regenerate"}
    assert "shine_actions" in out


def test_grounded_pass_with_verify_node():
    gate = _gate()
    node = shine_verify_node(gate)
    text = "Revenue was $1000 in Q1 for Acme Corp."
    out = node(
        {
            "question": "What was revenue?",
            "answer": text,
            "docs": [{"chunk_id": "c1", "text": text}],
            "trace": [record_retrieval("r", n_chunks=1, top_k=1)],
            "declared_sections": ["must_ground"],
        }
    )
    assert out["shine_verdict"]["decision"] in {"pass", "flag"}


def test_parallel_ambiguity_without_chorusgraph():
    gate = _gate()
    state = {
        "question": "q",
        "answer": "a",
        "docs": [{"chunk_id": "1", "text": "a"}, {"chunk_id": "2", "text": "b"}],
        "parallel_hops": True,
        "trace": [
            record_retrieval("r1", n_chunks=1),
            record_retrieval("r2", n_chunks=1),
            {"hop": "g", "kind": "llm", "status": "ok", "detail": {}},
        ],
    }
    v = post_llm_check(gate, state).verdict
    assert v is not None
    assert "PARALLEL_PRELOAD_AMBIGUITY" in {s.id for s in v.signatures}


def test_hit_revalidate_ignored_without_chorusgraph():
    gate = _gate()
    state = {
        "question": "q",
        "answer": "x",
        "docs": [{"chunk_id": "1", "text": "x"}],
        "trace": [record_cache("c", "HIT_REUSE", must_revalidate=True)],
    }
    v = post_llm_check(gate, state).verdict
    assert v is not None
    assert "HIT_REVALIDATE_IGNORED" in {s.id for s in v.signatures}


def test_append_trace_and_mark_wired():
    app = FakeApp()
    mark_shine_wired(app, interceptor=True)
    assert is_shine_wired(app)
    st = append_trace({}, record_llm_error("llm", error="boom"))
    assert st["trace"][0]["status"] == "error"


def test_capability_parity_matrix():
    """Documented feature set: each row must be reachable via wiring alone."""
    features = {
        "EMPTY_RETRIEVAL": record_retrieval("r", n_chunks=0),
        "LLM_ERROR": record_llm_error("g", error="500"),
        "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS": None,  # multi-step below
        "HIT_REVALIDATE_IGNORED": record_cache("c", "HIT_REUSE", must_revalidate=True),
        "TRACE_INCOMPLETE": None,
    }
    gate = _gate()
    # EMPTY
    d = pre_llm_check(
        gate,
        {
            "question": "q",
            "docs": [],
            "declared_sections": ["must_ground"],
            "trace": [features["EMPTY_RETRIEVAL"]],
        },
    )
    assert "EMPTY_RETRIEVAL" in {s.id for s in d.verdict.signatures}
    # LLM_ERROR
    d2 = post_llm_check(
        gate,
        {
            "question": "q",
            "answer": "x",
            "docs": [{"chunk_id": "1", "text": "x"}],
            "trace": [features["LLM_ERROR"]],
        },
    )
    assert "LLM_ERROR" in {s.id for s in d2.verdict.signatures}
    # HIT_REVALIDATE
    d3 = post_llm_check(
        gate,
        {
            "question": "q",
            "answer": "x",
            "docs": [{"chunk_id": "1", "text": "x"}],
            "trace": [features["HIT_REVALIDATE_IGNORED"]],
        },
    )
    assert "HIT_REVALIDATE_IGNORED" in {s.id for s in d3.verdict.signatures}
