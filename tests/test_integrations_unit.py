"""Unit tests for integration helpers (no live sibling required)."""

from __future__ import annotations

from prismshine.evidence.adapters.chorusgraph import bundle_from_chorusgraph
from prismshine.evidence.adapters.langgraph import bundle_from_langgraph
from prismshine.gate import ShineGate
from prismshine.integrations.chorusgraph import shine_node
from prismshine.integrations.langgraph import shine_langgraph_node, shine_route
import numpy as np

from prismshine.integrations.prismguard import consume_guard_verdict, guard_compatible


def fake_embedder(texts: list[str]) -> np.ndarray:
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


def test_chorusgraph_adapter_includes_history_memory():
    b, fb = bundle_from_chorusgraph(
        state={
            "question": "Who is A?",
            "reply": "A is my brother.",
            "docs": [{"chunk_id": "d1", "text": "Profile note."}],
            "history": ["Person A is my brother."],
            "memory": [{"text": "Person A lives nearby.", "subject": "A"}],
        }
    )
    sources = {c.source for c in b.preload}
    assert "history" in sources
    assert "memory" in sources


def test_shine_node_writes_verdict():
    gate = ShineGate.build(embedder=fake_embedder)
    node = shine_node(gate)
    out = node(
        {
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
    )
    assert "shine_verdict" in out
    assert out["shine_verdict"]["decision"] in {"pass", "flag", "block", "regenerate"}


def test_langgraph_node_and_route():
    gate = ShineGate.build(embedder=fake_embedder)
    node = shine_langgraph_node(gate)
    state = node(
        {
            "question": "q",
            "answer": "Revenue was $1000.",
            "docs": ["Revenue was $1000."],
        }
    )
    assert shine_route(state) in {"pass", "flag", "block", "regenerate"}


def test_guard_consume_and_compat():
    gate = ShineGate.build(embedder=fake_embedder)
    b, _ = bundle_from_langgraph(
        {"question": "q", "answer": "x", "docs": ["x data here"]}
    )
    b2 = consume_guard_verdict(b, {"decision": "flag", "zone": "gray", "fused_score": 0.4})
    assert any(t.kind == "guard" for t in b2.trace)
    v = gate.verify(b2)
    assert any(s.id == "GUARD_GRAY_INPUT" for s in v.signatures) or "guard" not in v.dormant_families
    compat = guard_compatible(v)
    assert "resolution_gate" in compat
