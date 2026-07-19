from __future__ import annotations

import numpy as np
import pytest

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate


def fake_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words embedder for tests (no network)."""
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int.from_bytes(__import__("hashlib").md5(tok.encode()).digest()[:4], "little") % dim
            out[i, h] += 1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


@pytest.fixture
def embedder():
    return fake_embedder


@pytest.fixture
def gate(embedder):
    return ShineGate.build(profile="default", embedder=embedder)


def make_bundle(**kwargs):
    base = {
        "run_id": "t",
        "question": "What was revenue?",
        "answer": "Revenue was $1000 in Q1.",
        "preload": [
            {
                "chunk_id": "c1",
                "text": "Company revenue was $1000 in Q1.",
                "source": "retrieval",
            }
        ],
        "trace": [
            {
                "hop": "retrieve",
                "kind": "retrieval",
                "status": "ok",
                "scores": {"constructive_score": 0.9},
                "detail": {"n_chunks": 3, "top_k": 3},
            }
        ],
        "declared_sections": ["must_ground"],
    }
    base.update(kwargs)
    b, _ = bundle_from_dict(base)
    return b
