"""Latency harness — soft targets for CI (CPU variance tolerant)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate


def _embed(texts):
    dim = 32
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, int.from_bytes(__import__("hashlib").md5(tok.encode()).digest()[:4], "little") % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


@pytest.mark.benchmark
def test_tier0_and_fast_path_latency():
    gate = ShineGate.build(embedder=_embed)
    pre, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": None,
            "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
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
    t0 = time.perf_counter()
    for _ in range(20):
        gate.verify(pre)
    tier0_ms = (time.perf_counter() - t0) * 1000 / 20

    full, _ = bundle_from_dict(
        {
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
    # warm
    gate.verify(full)
    t1 = time.perf_counter()
    for _ in range(20):
        # bust cache by unique run_id — actually cache keys on content; reuse ok for p50 path
        gate.verify(full)
    fast_ms = (time.perf_counter() - t1) * 1000 / 20

    # Soft budgets (CI machines vary); record via assertion messages
    assert tier0_ms < 50, f"Tier0 p50 {tier0_ms:.2f}ms (target <2ms local)"
    assert fast_ms < 100, f"fast path {fast_ms:.2f}ms (target <25ms local)"
