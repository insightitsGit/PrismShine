"""Deterministic hash embedder for offline benches (no network)."""

from __future__ import annotations

import hashlib

import numpy as np


def hash_embedder(texts: list[str], *, dim: int = 32) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % dim
            out[i, h] += 1.0
        n = float(np.linalg.norm(out[i]) or 1.0)
        out[i] /= n
    return out
