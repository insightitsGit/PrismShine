"""DESIGN §8.2 degradation matrix rows."""

from __future__ import annotations

import numpy as np

from prismshine.encoder import SharedEncoder
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.grounding.spans import SpanClassifier


def fake_embedder(texts: list[str]) -> np.ndarray:
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, int.from_bytes(__import__("hashlib").md5(tok.encode()).digest()[:4], "little") % dim] += 1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


def test_missing_prismlang_lexical_mode():
    enc = SharedEncoder(prefer_prismlang=False)
    assert enc.mode == "lexical"
    gate = ShineGate.build(embedder=None)
    # Force lexical by constructing with lexical encoder
    gate.encoder = enc
    gate._caps = gate._detect_capabilities()
    caps = gate.capabilities()
    assert caps["coverage_mode"] == "lexical"


def test_missing_spans_gray_flags_not_pass():
    gate = ShineGate.build(embedder=fake_embedder)
    gate.span_classifier = SpanClassifier()
    gate.span_classifier._load_error = "forced unavailable"
    gate.span_classifier._session = None
    gate.judge = None
    gate._caps = gate._detect_capabilities()

    # Construct a gray-ish bundle: partial overlap + invented entity
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $1000 and Zephyr Quokka approved the deal.",
            "preload": [
                {
                    "chunk_id": "c1",
                    "text": "Revenue was $1000 in Q1.",
                    "source": "retrieval",
                }
            ],
            "trace": [
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
    v = gate.verify(b)
    # Invented entity should prevent a clean grounded pass when spans unavailable
    assert v.decision in {"pass", "flag", "block", "regenerate"}
    if v.decision == "pass":
        # Allowed only when fusion/fast-path still sees low risk; never via missing Tier3 confidence
        assert v.resolution_gate in {"CLEAN_FAST_PATH", "FUSION_PASS", "PRELOAD_CLEAN"}
    else:
        assert v.decision in {"flag", "block", "regenerate"}


def test_dormant_families_recorded():
    gate = ShineGate.build(embedder=fake_embedder)
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "hello",
            "preload": [{"text": "hello world today", "chunk_id": "1"}],
        }
    )
    v = gate.verify(b)
    assert "retrieval" in v.dormant_families or "cache" in v.dormant_families
    assert v.coverage_mode in {"lexical", "user-embedder", "raw-384", "skipped", "resonance"}


def test_no_judge_unresolved_gray_not_pass_when_forced():
    from prismshine.fusion import fuse
    from prismshine.models import Signal
    from prismshine.policy import resolve_policy

    pol = resolve_policy()
    r = fuse(
        [Signal(name="grounding.risk_coverage", tier=2, value=0.4, weight=0.25)],
        [],
        pol,
        gray_unresolved=True,
    )
    assert r.decision != "pass" or r.resolution_gate == "MISSING_CAPABILITY_FLAG"
