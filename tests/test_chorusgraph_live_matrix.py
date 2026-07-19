"""Live ChorusGraph integration matrix — real Graph + call_llm + injected failures.

Requires chorusgraph>=1.3.0. Skips cleanly if unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

chorusgraph = pytest.importorskip("chorusgraph")
from chorusgraph import END, START, Graph  # noqa: E402
from chorusgraph.core.node import NodeContext  # noqa: E402

from prismshine.gate import ShineGate
from prismshine.integrations.chorusgraph import (
    ChorusGraphAdapter,
    require_shine,
    shine_node,
)
from prismshine.integrations.runtime_base import pull_ledger_steps


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def _gate() -> ShineGate:
    return ShineGate.build(embedder=_embed)


def _step(hop: str, kind: str, *, status: str = "ok", detail: dict | None = None) -> dict:
    """JSON-safe ledger step (channel state cannot carry LedgerStep objects)."""
    return {
        "hop": hop,
        "kind": kind,
        "status": status,
        "detail": dict(detail or {}),
    }


def _build_graph(*, empty_retrieval: bool = False):
    """Minimal retrieve → answer → shine graph using ctx.call_llm."""

    def retrieve(ctx: NodeContext):
        state = dict(ctx.read())
        if empty_retrieval:
            state["docs"] = []
            step = _step(
                "retrieve",
                "retrieval",
                status="empty",
                detail={"n_chunks": 0, "top_k": 3},
            )
        else:
            state["docs"] = state.get("docs") or [
                {"chunk_id": "c1", "text": "Revenue was $1000 in Q1 for Acme Corp."}
            ]
            step = _step(
                "retrieve",
                "retrieval",
                detail={"n_chunks": len(state["docs"]), "top_k": 3},
            )
        state["ledger_steps"] = list(state.get("ledger_steps") or []) + [step]
        state["question"] = state.get("question") or state.get("message") or "q"
        state["declared_sections"] = ["must_ground"]
        return ctx.publish(artifact={**state, "raw_output": "retrieved"}, category_slug="general")

    def answer(ctx: NodeContext):
        state = dict(ctx.read())

        def model(_system: str, _user: str) -> str:
            return "Revenue was $1000 in Q1 for Acme Corp."

        text = ctx.call_llm("sys", state.get("question", "q"), model=model)
        state["reply"] = text
        return ctx.publish(artifact={**state, "raw_output": text}, category_slug="general")

    g = Graph(tenant_id="shine-matrix", graph_id="live-matrix")
    g.add_node("retrieve", retrieve, category_slug="general", consumes=["docs"])
    g.add_node("answer", answer, category_slug="general", consumes=["docs"])
    gate = _gate()
    # Legacy dict node — Graph.add_node auto-wraps via dict_node_adapter
    g.add_node("shine", shine_node(gate, answer_key="reply"), category_slug="general")
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "shine")
    g.add_edge("shine", END)
    compiled = g.compile()
    require_shine(compiled, gate, prefer="both", already_has_shine_node=True)
    return compiled, gate


def test_live_empty_retrieval_halts_before_llm():
    compiled, _ = _build_graph(empty_retrieval=True)
    result = compiled.invoke(
        {"question": "What was revenue?", "message": "What was revenue?", "docs": []}
    )
    assert isinstance(result, dict)
    # Interceptor halt → NodeInterrupt; or shine_node block if graph continued
    verdict = result.get("shine_verdict")
    if verdict:
        ids = {s["id"] for s in verdict.get("signatures", [])}
        assert "EMPTY_RETRIEVAL" in ids
        assert verdict["decision"] in {"block", "regenerate"}
    else:
        # Partial / interrupt path — must not deliver invented grounded prose
        reply = str(result.get("reply") or "")
        assert "Revenue was $1000" not in reply
        # Either fallback text or node-error / interrupt artifact
        blob = str(result)
        assert (
            "I don't have" in blob
            or "llm_intercept" in blob
            or "__node_error__" in result
            or result.get("__partial__")
        )


def test_live_grounded_pass_or_flag():
    compiled, _ = _build_graph(empty_retrieval=False)
    result = compiled.invoke(
        {
            "question": "What was revenue?",
            "message": "What was revenue?",
            "docs": [{"chunk_id": "c1", "text": "Revenue was $1000 in Q1 for Acme Corp."}],
        }
    )
    verdict = result.get("shine_verdict")
    assert verdict is not None, f"shine_node did not write verdict; keys={sorted(result)}"
    assert verdict["decision"] in {"pass", "flag"}
    assert "EMPTY_RETRIEVAL" not in {s["id"] for s in verdict.get("signatures", [])}


def test_live_adapter_pulls_last_ledger():
    compiled, _ = _build_graph(empty_retrieval=False)
    compiled.invoke(
        {
            "question": "What was revenue?",
            "message": "What was revenue?",
            "docs": [{"chunk_id": "c1", "text": "Revenue was $1000 in Q1 for Acme Corp."}],
        }
    )
    assert compiled.last_ledger is not None
    steps = pull_ledger_steps(compiled)
    assert steps, "expected auto-pull from compiled.last_ledger"


def test_live_cache_miss_skip_retrieval_signature():
    gate = _gate()
    adapter = ChorusGraphAdapter(gate)
    state = {
        "question": "q",
        "reply": "guessed answer without retrieval",
        "docs": [{"chunk_id": "c1", "text": "stale"}],
        "declared_sections": ["must_ground"],
        "ledger_steps": [
            _step("cache", "cache", detail={"decision": "MISS"}),
            _step("gen", "llm", status="ok"),
        ],
    }
    v = adapter.post_llm_hook(state)
    ids = {s.id for s in v.signatures}
    assert "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS" in ids


def test_live_llm_error_signature():
    gate = _gate()
    adapter = ChorusGraphAdapter(gate)
    state = {
        "question": "q",
        "reply": "partial",
        "docs": [{"chunk_id": "c1", "text": "x"}],
        "ledger_steps": [
            _step("gen", "llm", status="error", detail={"error": "429 rate limit"}),
        ],
    }
    v = adapter.post_llm_hook(state)
    assert "LLM_ERROR" in {s.id for s in v.signatures}


def test_live_trace_incomplete():
    gate = _gate()
    node = shine_node(gate)
    out = node(
        {
            "question": "q",
            "reply": "hello world today",
            "docs": ["hello world today"],
            "consumes": ["docs"],
            "expect_trace_kinds": ["retrieval"],
            "ledger_steps": [],
        }
    )
    ids = {s["id"] for s in out["shine_verdict"]["signatures"]}
    assert "TRACE_INCOMPLETE" in ids
