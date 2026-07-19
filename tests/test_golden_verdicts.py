from __future__ import annotations

import numpy as np

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate


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


def _stable(v):
    d = v.model_dump(mode="json")
    d.pop("verdict_id", None)
    # signals detail order stable via pydantic
    return d


def test_determinism_same_bundle():
    gate = ShineGate.build(embedder=fake_embedder)
    b, _ = bundle_from_dict(
        {
            "run_id": "g1",
            "question": "What was revenue?",
            "answer": "Revenue was $1000 in Q1 for Acme Corp.",
            "preload": [
                {
                    "chunk_id": "c1",
                    "text": "Revenue was $1000 in Q1 for Acme Corp.",
                    "source": "retrieval",
                }
            ],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "scores": {"constructive_score": 0.95},
                    "detail": {"n_chunks": 3, "top_k": 3},
                }
            ],
        }
    )
    a = gate.verify(b)
    b2 = gate.verify(b)
    assert _stable(a) == _stable(b2)
    assert a.evidence_hash == b2.evidence_hash


def test_fatal_early_exit_skips_grounding():
    gate = ShineGate.build(embedder=fake_embedder)
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Anything invented.",
            "preload": [{"text": "real data", "chunk_id": "1"}],
            "declared_sections": ["must_ground"],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "empty",
                    "detail": {"n_chunks": 0},
                }
            ],
        }
    )
    v = gate.verify(b)
    assert v.tier_reached == 0
    assert v.resolution_gate == "HANDBOOK:EMPTY_RETRIEVAL"
    assert v.decision in {"block", "regenerate"}


def test_pregeneration_mode():
    gate = ShineGate.build(embedder=fake_embedder)
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": None,
            "preload": [{"text": "ok", "chunk_id": "1"}],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "detail": {"n_chunks": 3, "top_k": 3},
                    "scores": {"constructive_score": 0.9},
                }
            ],
        }
    )
    v = gate.verify(b)
    assert v.coverage_mode == "skipped"
    assert v.tier_reached == 0
