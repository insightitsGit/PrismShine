"""FIX-1 through FIX-7 acceptance tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from prismshine.evidence.adapters.chorusgraph import bundle_from_chorusgraph
from prismshine.evidence.builder import bundle_from_dict
from prismshine.forensics.engine import run_forensics
from prismshine.fusion import fuse
from prismshine.gate import ShineGate
from prismshine.grounding.contradiction import screen_contradictions
from prismshine.grounding.copycheck import copycheck
from prismshine.grounding.judge import EscalationBudget, JudgeResult
from prismshine.models import Signal
from prismshine.policy import resolve_policy


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_fix1_numeric_tolerance_override_survives_verify():
    gate = ShineGate.build(embedder=_embed, overrides={"numeric_tolerance": 0.5})
    b, _ = bundle_from_dict(
        {
            "question": "price?",
            "answer": "The discounted price is $1400.",
            "preload": [{"chunk_id": "1", "text": "Original price was $1000 (40% off)."}],
            "trace": [{"hop": "r", "kind": "retrieval", "status": "ok", "detail": {"n_chunks": 1}}],
        }
    )
    v = gate.verify(b)
    assert not any("unmatched_currency" in (s.reason or "") for s in v.spans)


def test_fix1_bands_override_blocks_mildly_dirty_bundle():
    gate = ShineGate.build(
        embedder=_embed,
        overrides={"bands": (0.01, 0.02, 0.03)},
    )
    b, _ = bundle_from_dict(
        {
            "question": "revenue?",
            "answer": "Revenue was $1000 in Q1 with minor extra wording.",
            "preload": [{"chunk_id": "1", "text": "Company revenue was $1000 in Q1."}],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "scores": {"constructive_score": 0.9},
                    "detail": {"n_chunks": 1},
                },
                {
                    "hop": "w",
                    "kind": "guard",
                    "status": "ok",
                    "detail": {"verdict": "gray"},
                },
            ],
            "node_state": {"guard_verdict": "gray"},
        }
    )
    v_default = ShineGate.build(embedder=_embed).verify(b)
    v_tight = gate.verify(b)
    assert v_default.decision in {"pass", "flag"}
    assert v_tight.decision == "block"


def test_fix2_escalation_budget_counts_all_verifies():
    budget = EscalationBudget(budget=0.10, window=100)
    allows = 0
    for i in range(100):
        budget.observe()
        if i >= 80 and budget.allow():
            allows += 1
    assert allows <= 10

    recovery = EscalationBudget(budget=0.10, window=20)
    for _ in range(20):
        recovery.observe()
        recovery.allow()
    assert not recovery.allow()
    for _ in range(20):
        recovery.observe()
    assert recovery.allow()


def test_fix3_numpy_vector_384_injected():
    class Rec:
        def __init__(self, chunk_id: str, vec):
            self.chunk_id = chunk_id
            self.vector_384 = vec
            self.encoder_artifact_id = "test-artifact"
            self.partition = "p1"
            self.version = 1

    class Stack:
        def get_chunk_vectors(self, chunk_ids, partition=None):
            return [Rec(cid, np.ones(384)) for cid in chunk_ids]

    state = {
        "question": "q",
        "reply": "answer",
        "docs": [{"chunk_id": "c1", "text": "preload text"}],
        "chunk_ids": ["c1"],
    }
    bundle, _ = bundle_from_chorusgraph(state=state, stack=Stack(), chunk_ids=["c1"])
    chunk = next(c for c in bundle.preload if c.chunk_id == "c1")
    assert chunk.vector is not None
    assert len(chunk.vector) == 384
    assert chunk.vector_space == "raw-384@test-artifact"


def test_fix4_tier0_cache_avoids_repeat_forensics():
    gate = ShineGate.build(embedder=_embed)
    base = {
        "question": "q",
        "preload": [{"chunk_id": "1", "text": "Shared preload context."}],
        "trace": [{"hop": "r", "kind": "retrieval", "status": "ok", "detail": {"n_chunks": 1}}],
    }
    b1, _ = bundle_from_dict({**base, "answer": "Answer variant one."})
    b2, _ = bundle_from_dict({**base, "answer": "Answer variant two."})
    calls = {"n": 0}
    real = run_forensics

    def counting(bundle, handbook):
        calls["n"] += 1
        return real(bundle, handbook)

    with patch("prismshine.gate.run_forensics", side_effect=counting):
        gate.verify(b1)
        gate.verify(b2)
    assert calls["n"] == 1
    assert len(gate._tier0_cache) == 1
    for _ in range(300):
        gate._tier0_cache_put(f"k{_}", object())
    assert len(gate._tier0_cache) <= gate._tier0_cache_maxsize


def test_fix6_judge_does_not_wash_contradiction_cue():
    pol = resolve_policy(profile="finance")
    det_signals = [
        Signal(name="grounding.risk_coverage", tier=2, value=0.35, weight=0.25),
        Signal(name="grounding.contradiction_cue", tier=2, value=1.0, weight=0.30),
    ]
    without = fuse(det_signals, [], pol)
    with_judge = fuse(
        det_signals
        + [Signal(name="grounding.judge_risk", tier=4, value=0.0, weight=0.45)],
        [],
        pol,
        judge_present=True,
    )
    assert with_judge.fused_score >= without.fused_score
    assert with_judge.decision != "pass"


def test_fix7_opposite_pairs_respect_word_boundaries():
    false_pos = screen_contradictions(
        ["The passenger completed the trip."],
        [("c1", "failure rates were low in the study.")],
    )
    assert false_pos == []
    true_pos = screen_contradictions(
        ["Revenue increased in Q1."],
        [("c1", "Revenue decreased in Q2.")],
    )
    assert len(true_pos) == 1
    assert "opposite:increased/decreased" in true_pos[0].reason


def test_fix11_currency_product_not_derivable():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Combined value is $1200000.",
            "preload": [{"chunk_id": "1", "text": "Price A is $1000. Price B is $1200."}],
        }
    )
    r = copycheck(b)
    assert any(f.raw.startswith("$") or "$" in f.raw for f in r.unmatched)
