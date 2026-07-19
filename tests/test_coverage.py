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
