"""P0–P3 improvement suite."""

from __future__ import annotations

import numpy as np
import pytest

from prismshine.actions import actions_for_verdict, clarify_conflict_question
from prismshine.evidence.builder import bundle_from_dict
from prismshine.forensics.engine import run_forensics
from prismshine.gate import ShineGate
from prismshine.handbook.loader import load_handbook
from prismshine.integrations.chorusgraph import (
    ShineNotWiredError,
    attach_interceptors,
    is_shine_wired,
    require_shine,
    shine_node,
    ChorusGraphAdapter,
)
from prismshine.integrations.langgraph import LangGraphAdapter, shine_langgraph_node
from prismshine.models import SignatureHit
from prismshine.runtime import assert_adapter, check_adapter


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


HB = load_handbook()


def _run(**kwargs):
    b, _ = bundle_from_dict(kwargs)
    return run_forensics(b, HB)


# --- P0 ---


def test_require_shine_fails_without_wiring():
    class Bare:
        pass

    with pytest.raises(ShineNotWiredError):
        require_shine(Bare(), ShineGate.build(embedder=_embed), prefer="interceptor")


def test_require_shine_attaches_interceptor():
    class Compiled:
        def __init__(self):
            self.hooks = None

        def register_interceptor(self, **kwargs):
            self.hooks = kwargs

    c = Compiled()
    gate = ShineGate.build(embedder=_embed)
    require_shine(c, gate, prefer="both")
    assert is_shine_wired(c)
    assert c.hooks is not None
    assert hasattr(c, "_prismshine_node_factory")


def test_trace_incomplete_fires():
    r = _run(
        question="q",
        preload=[{"text": "x", "chunk_id": "1"}],
        node_state={"consumes": ["docs"], "expect_trace_kinds": ["retrieval"]},
        trace=[],
    )
    assert "TRACE_INCOMPLETE" in {h.id for h in r.hits}


def test_llm_error_and_empty():
    r = _run(
        question="q",
        answer="hi",
        preload=[{"text": "x", "chunk_id": "1"}],
        trace=[{"hop": "g", "kind": "llm", "status": "error", "detail": {"error": "429"}}],
    )
    assert "LLM_ERROR" in {h.id for h in r.hits}
    r2 = _run(
        question="q",
        answer="",
        preload=[{"text": "x", "chunk_id": "1"}],
        trace=[{"hop": "g", "kind": "llm", "status": "empty", "detail": {"empty": True}}],
    )
    assert "LLM_EMPTY_COMPLETION" in {h.id for h in r2.hits}
    r3 = _run(
        question="q",
        answer="I cannot help with that.",
        preload=[{"text": "x", "chunk_id": "1"}],
        trace=[
            {
                "hop": "g",
                "kind": "llm",
                "status": "ok",
                "detail": {"finish_reason": "content_filter"},
            }
        ],
    )
    assert "LLM_REFUSAL" in {h.id for h in r3.hits}


# --- P1 ---


def test_retrieval_skipped_after_cache_miss():
    r = _run(
        question="q",
        answer="guess",
        preload=[{"text": "stale", "chunk_id": "1"}],
        trace=[
            {"hop": "c", "kind": "cache", "status": "ok", "detail": {"decision": "MISS"}},
            {"hop": "g", "kind": "llm", "status": "ok"},
        ],
    )
    assert "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS" in {h.id for h in r.hits}


def test_hit_revalidate_ignored():
    r = _run(
        question="q",
        preload=[{"text": "x", "chunk_id": "1"}],
        trace=[
            {
                "hop": "c",
                "kind": "cache",
                "status": "ok",
                "detail": {"decision": "HIT_REUSE", "must_revalidate": True},
            }
        ],
    )
    assert "HIT_REVALIDATE_IGNORED" in {h.id for h in r.hits}


def test_parallel_ambiguity():
    r = _run(
        question="q",
        preload=[{"text": "a", "chunk_id": "1"}, {"text": "b", "chunk_id": "2"}],
        node_state={"parallel_hops": True},
        trace=[
            {"hop": "r1", "kind": "retrieval", "status": "ok", "detail": {"n_chunks": 1}},
            {"hop": "r2", "kind": "retrieval", "status": "ok", "detail": {"n_chunks": 1}},
            {"hop": "g", "kind": "llm", "status": "ok"},
        ],
    )
    assert "PARALLEL_PRELOAD_AMBIGUITY" in {h.id for h in r.hits}


def test_chorusgraph_adapter_auto_ledger():
    gate = ShineGate.build(embedder=_embed)
    adapter = ChorusGraphAdapter(gate)
    assert check_adapter(adapter) == []
    state = {
        "question": "What was revenue?",
        "reply": "Revenue was $1000.",
        "docs": [{"chunk_id": "c1", "text": "Revenue was $1000."}],
        "ledger_steps": [
            {
                "hop": "r",
                "kind": "retrieval",
                "status": "ok",
                "scores": {"constructive_score": 0.9},
                "detail": {"n_chunks": 2, "top_k": 2},
            }
        ],
    }
    bundle = adapter.extract_bundle(state)
    v = adapter.post_llm_hook(state)
    assert v.decision in {"pass", "flag", "block", "regenerate"}
    out = adapter.enforce(v, state)
    assert "shine_actions" in out


# --- P2 ---


def test_runtime_conformance_chorus_and_langgraph():
    gate = ShineGate.build(embedder=_embed)
    assert_adapter(ChorusGraphAdapter(gate))
    assert_adapter(LangGraphAdapter(gate))


def test_langgraph_trace_incomplete():
    gate = ShineGate.build(embedder=_embed)
    node = shine_langgraph_node(gate)
    out = node(
        {
            "question": "q",
            "answer": "hello world today",
            "docs": ["hello world today"],
            "consumes": ["docs"],
            "expect_trace_kinds": ["retrieval"],
            # no trace -> TRACE_INCOMPLETE
        }
    )
    sigs = out["shine_verdict"]["signatures"]
    assert any(s["id"] == "TRACE_INCOMPLETE" for s in sigs)


# --- P3 ---


def test_clarify_action_on_conflict():
    hit = SignatureHit(
        id="CONFLICTING_PRELOAD_FACTS",
        severity="error",
        advice="x",
        evidence={
            "subject": "person a",
            "relation": "kinship",
            "value_a": "brother",
            "value_b": "sister",
        },
    )
    q = clarify_conflict_question(hit)
    assert "brother" in q and "sister" in q
    gate = ShineGate.build(embedder=_embed)
    b, _ = bundle_from_dict(
        {
            "question": "Who is A?",
            "answer": "Person A is my brother.",
            "preload": [
                {"chunk_id": "h1", "text": "Person A is my brother.", "source": "history"},
                {"chunk_id": "h2", "text": "Person A is my sister.", "source": "history"},
            ],
        }
    )
    v = gate.verify(b)
    assert any(s.id == "CONFLICTING_PRELOAD_FACTS" for s in v.signatures)
    assert any("conflicting information" in a.lower() for a in v.advice)
    acts = actions_for_verdict(v)
    assert any(a["type"] == "ask_user_clarify" for a in acts)


def test_shine_node_marks_wired():
    gate = ShineGate.build(embedder=_embed)

    class C:
        pass

    c = C()
    node = shine_node(gate, compiled=c)
    assert is_shine_wired(c)
    assert getattr(node, "_prismshine_shine_node", False)


def test_span_pin_env(monkeypatch):
    from prismshine.grounding.spans import PINNED_MODEL_ENV, SpanClassifier

    monkeypatch.setenv(PINNED_MODEL_ENV, "org/pinned-lettuce-test")
    clf = SpanClassifier(allow_lexical_fallback=True)
    assert clf.model_id == "org/pinned-lettuce-test"
    assert clf.model_candidates == ("org/pinned-lettuce-test",)


def test_domain_calibration_receipt():
    from prismshine.calibrate import calibrate_synthetic
    from prismshine.evidence.builder import bundle_from_dict

    gate = ShineGate.build(embedder=_embed, profile="clinical")
    assert gate.policy.threshold_status == "proposal"
    b, _ = bundle_from_dict(
        {
            "question": "dose?",
            "answer": "Give 5 mg daily.",
            "preload": [{"chunk_id": "1", "text": "Give 5 mg daily."}],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "detail": {"n_chunks": 1},
                }
            ],
        }
    )
    report = calibrate_synthetic([b], gate=gate, seed=3)
    assert report.mode == "synthetic"
    assert gate.policy.threshold_status == "validated-synthetic"
    assert "validated-synthetic" in gate.capabilities()["threshold_status"]
