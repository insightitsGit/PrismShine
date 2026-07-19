from __future__ import annotations

import numpy as np

from prismshine.encoder import SharedEncoder
from prismshine.evidence.builder import bundle_from_dict
from prismshine.grounding.contradiction import screen_contradictions
from prismshine.grounding.coverage import coverage_check


def embedder(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, int.from_bytes(__import__("hashlib").md5(tok.encode()).digest()[:4], "little") % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_lexical_fallback_mode():
    enc = SharedEncoder(prefer_prismlang=False)
    assert enc.mode == "lexical"
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "The cat sat on the mat quietly.",
            "preload": [{"text": "The cat sat on the mat quietly.", "chunk_id": "1"}],
        }
    )
    r = coverage_check(b, enc)
    assert r.coverage_mode == "lexical"
    assert r.coverage > 0.5


def test_vector_coverage_with_user_embedder():
    enc = SharedEncoder(embedder=embedder)
    text = "Acme Corp reported strong quarterly revenue growth."
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": text,
            "preload": [{"text": text, "chunk_id": "1"}],
        }
    )
    r = coverage_check(b, enc, tau_sent=0.5)
    assert r.coverage >= 0.5
    assert r.coverage_mode in {"user-embedder", "raw-384", "resonance"}


def test_contradiction_cue_negation():
    cues = screen_contradictions(
        ["The drug is safe for children."],
        [("c1", "The drug is not safe for children.")],
        sentence_offsets=[(0, 30)],
    )
    assert cues
    assert "negation" in cues[0].reason


def test_containment_support_short_extractive():
    from prismshine.grounding.coverage import containment_support

    ctx = [
        "Claudia Rivas Vega (born 15 June 1989) is a Mexican triathlete. "
        "She competed at the 2015 Pan American Games."
    ]
    assert containment_support("triathlete", ctx) == 1.0
    assert containment_support("BorgWarner", [
        "Norge was once a division of BorgWarner and later Fedders."
    ]) == 1.0
    # Hallucinated entity not in context
    assert containment_support("General Motors", ctx) == 0.0


def test_short_extractive_answer_not_coverage_collapse():
    """HaluEval-style 1-word answer copied from preload must not collapse."""
    enc = SharedEncoder(embedder=embedder)
    b, _ = bundle_from_dict(
        {
            "question": "What kind of athlete is she?",
            "answer": "triathlete",
            "preload": [
                {
                    "chunk_id": "1",
                    "text": (
                        "Claudia Rivas Vega (born 15 June 1989) is a Mexican "
                        "triathlete. She represents Mexico in competitions."
                    ),
                }
            ],
        }
    )
    r = coverage_check(b, enc, tau_sent=0.62)
    assert r.coverage >= 0.75
    assert r.signals[0].detail.get("containment_hits", 0) >= 1

    from prismshine.gate import ShineGate

    gate = ShineGate.build(embedder=embedder)
    # healthy retrieval so collapse path is the one under test
    b2, _ = bundle_from_dict(
        {
            "question": "What kind of athlete is she?",
            "answer": "triathlete",
            "preload": [
                {
                    "chunk_id": "1",
                    "text": (
                        "Claudia Rivas Vega (born 15 June 1989) is a Mexican "
                        "triathlete. She represents Mexico in competitions."
                    ),
                    "source": "retrieval",
                }
            ],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "scores": {"constructive_score": 0.95},
                    "detail": {"n_chunks": 1, "top_k": 1},
                }
            ],
        }
    )
    v = gate.verify(b2)
    assert v.resolution_gate != "T2_COVERAGE_COLLAPSE"
    assert v.decision == "pass"