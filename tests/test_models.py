"""Golden serialization and evidence_hash stability."""

from __future__ import annotations

from prismshine.evidence.builder import bundle_from_dict
from prismshine.hashing import canonical_bytes, evidence_hash
from prismshine.models import EvidenceBundle, PreloadChunk, ShineVerdict, SignatureHit


def _sample_bundle(**overrides) -> EvidenceBundle:
    base = {
        "run_id": "r1",
        "tenant_id": "t1",
        "question": "What is revenue?",
        "answer": "Revenue was $1,000.",
        "preload": [
            {
                "chunk_id": "c1",
                "text": "Revenue was $1,000 in Q1.",
                "vector": [0.1, 0.2, 0.3],
                "vector_space": "raw-384",
                "source": "retrieval",
            }
        ],
        "trace": [
            {
                "hop": "retrieve",
                "kind": "retrieval",
                "status": "ok",
                "scores": {"constructive_score": 0.9},
            }
        ],
        "declared_sections": ["must_ground"],
    }
    base.update(overrides)
    bundle, _ = bundle_from_dict(base)
    return bundle


def test_canonical_bytes_stable():
    a = _sample_bundle()
    b = _sample_bundle()
    assert canonical_bytes(a) == canonical_bytes(b)
    assert evidence_hash(a) == evidence_hash(b)


def test_canonical_bytes_key_order_independent():
    b1, _ = bundle_from_dict(
        {
            "question": "q",
            "run_id": "r",
            "answer": "a",
            "preload": [{"chunk_id": "c", "text": "t", "source": "retrieval"}],
        }
    )
    # Reconstruct with same content
    b2 = EvidenceBundle(
        run_id="r",
        question="q",
        answer="a",
        preload=[PreloadChunk(chunk_id="c", text="t", source="retrieval")],
    )
    assert evidence_hash(b1) == evidence_hash(b2)


def test_builder_requires_question_and_preload():
    import pytest

    with pytest.raises(ValueError, match="question"):
        bundle_from_dict({"preload": [{"text": "x"}]})
    with pytest.raises(ValueError, match="preload"):
        bundle_from_dict({"question": "q", "preload": []})


def test_capability_feedback_specific():
    bundle, fb = bundle_from_dict(
        {
            "question": "q",
            "preload": [{"text": "hello", "chunk_id": "1"}],
        }
    )
    joined = " ".join(fb)
    assert "lexical" in joined
    assert "no trace" in joined
    assert bundle.answer is None
    assert any("pre-generation" in f for f in fb)


def test_shine_verdict_fields():
    v = ShineVerdict(
        decision="pass",
        resolution_gate="CLEAN_FAST_PATH",
        fused_score=0.1,
        confidence=0.8,
        evidence_hash="abc",
        verdict_id="v1",
        signatures=[
            SignatureHit(id="X", severity="info", advice="note"),
        ],
    )
    assert v.tier_reached == 0
    assert v.coverage_mode == "skipped"
    data = v.model_dump()
    for key in (
        "decision",
        "resolution_gate",
        "fused_score",
        "confidence",
        "signatures",
        "spans",
        "tier_reached",
        "coverage_mode",
        "strictness_effective",
        "dormant_families",
        "evidence_hash",
        "verdict_id",
        "signals",
        "advice",
    ):
        assert key in data
