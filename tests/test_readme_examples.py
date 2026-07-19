"""README examples must import-run (anti-drift)."""

from __future__ import annotations

import numpy as np

from prismshine import EvidenceBundle, PreloadChunk, ShineGate, ShineVerdict


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_readme_basic_verify():
    gate = ShineGate.build(profile="default", embedder=_embed)
    bundle = EvidenceBundle(
        run_id="demo",
        question="What was revenue?",
        answer="Revenue was $1000 in Q1.",
        preload=[
            PreloadChunk(
                chunk_id="c1",
                text="Revenue was $1000 in Q1.",
                source="retrieval",
            )
        ],
    )
    verdict = gate.verify(bundle)
    assert isinstance(verdict, ShineVerdict)
    assert verdict.decision in {"pass", "flag", "block", "regenerate"}
    assert verdict.resolution_gate
    assert verdict.evidence_hash


def test_readme_capabilities():
    gate = ShineGate.build(embedder=_embed)
    caps = gate.capabilities()
    assert "tiers" in caps
    assert caps["tiers"]["0"] is True
