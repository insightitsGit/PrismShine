from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prismshine.calibrate import (
    calibrate_labeled,
    calibrate_synthetic,
    calibrate_dir,
    synthetic_negatives,
)
from prismshine.cache import TieredVerdictStore, MemoryVerdictStore, SqliteVerdictStore
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.grounding.judge import CachedJudge, EscalationBudget, JudgeResult, build_judge
from prismshine.grounding.spans import SpanClassifier
from prismshine.handbook.loader import load_handbook, format_advice
from prismshine.models import ShineVerdict, Span


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_synthetic_calibration():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $1000.",
            "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
        }
    )
    gate = ShineGate.build(embedder=_embed)
    report = calibrate_synthetic([b], gate=gate, seed=2)
    assert report.mode == "synthetic"
    assert report.n_samples >= 2
    overlay = report.to_yaml_overlay()
    assert "calibration_version" in overlay


def test_labeled_calibration():
    pos, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $9999 and Zephyr Quokka.",
            "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
        }
    )
    neg, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $1000.",
            "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
        }
    )
    gate = ShineGate.build(embedder=_embed)
    report = calibrate_labeled([(pos, True), (neg, False)], gate=gate)
    assert report.auroc is None or 0 <= report.auroc <= 1


def test_calibrate_dir(tmp_path: Path):
    p = tmp_path / "b.json"
    p.write_text(
        json.dumps(
            {
                "question": "q",
                "answer": "Revenue was $1000.",
                "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
            }
        ),
        encoding="utf-8",
    )
    report = calibrate_dir(tmp_path, mode="synthetic", gate=ShineGate.build(embedder=_embed))
    assert report.n_samples >= 2


def test_span_classifier_lexical():
    sc = SpanClassifier()
    sc._session = "lexical"
    sc.artifact_id = "test-lexical"
    sc._load_error = None
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "The unicorn quantum foam exploded yesterday.",
            "preload": [{"text": "Revenue was steady.", "chunk_id": "1"}],
        }
    )
    r = sc.classify(
        b,
        candidate_spans=[
            Span(start=0, end=len(b.answer or ""), text=b.answer or "", reason="x", tier=2)
        ],
    )
    assert r.available
    assert r.unsupported_span_ratio >= 0


def test_judge_budget_and_cache():
    budget = EscalationBudget(0.1)
    assert budget.allow()
    calls = {"n": 0}

    def fake(claims, context):
        calls["n"] += 1
        return JudgeResult(risk=0.2, claim_support=[])

    j = CachedJudge(fake)
    assert j(["a"], "ctx").risk == 0.2
    assert j(["a"], "ctx").risk == 0.2
    assert calls["n"] == 1
    with pytest.raises(ValueError):
        build_judge("nope")


def test_handbook_merge_and_advice(tmp_path: Path):
    overlay = tmp_path / "ovr.yaml"
    overlay.write_text(
        """
handbook_version: "0.1.1"
signatures:
  - id: EMPTY_RETRIEVAL
    severity: error
    detector: retrieval.empty
    advice: "custom {hop}"
""",
        encoding="utf-8",
    )
    hb = load_handbook(overlay)
    assert hb.handbook_version == "0.1.1"
    sig = hb.by_id()["EMPTY_RETRIEVAL"]
    assert sig.severity == "error"
    assert "r1" in format_advice(sig.advice, hop="r1")


def test_tiered_store(tmp_path: Path):
    store = TieredVerdictStore(
        memory=MemoryVerdictStore(2),
        sqlite=SqliteVerdictStore(tmp_path / "t.db"),
    )
    v = ShineVerdict(
        decision="pass",
        resolution_gate="X",
        fused_score=0.1,
        confidence=0.5,
        evidence_hash="h",
        verdict_id="1",
    )
    store.put("k", v)
    assert store.get("k").decision == "pass"


def test_coverage_collapse_gate():
    gate = ShineGate.build(embedder=_embed)
    b, _ = bundle_from_dict(
        {
            "question": "What happened?",
            "answer": "Completely unrelated astronomy facts about nebulae and quasars.",
            "preload": [
                {
                    "chunk_id": "c1",
                    "text": "Quarterly revenue filing for Acme Corp was submitted on time.",
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
    v = gate.verify(b)
    assert v.decision in {"flag", "block", "regenerate", "pass"}
    assert v.tier_reached >= 0


def test_synthetic_negatives_modes():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Alice Smith paid $50 on 2024-01-01.",
            "preload": [
                {"text": "Alice Smith paid $50 on 2024-01-01.", "chunk_id": "1"}
            ],
        }
    )
    pairs = synthetic_negatives([b], seed=3)
    assert any(y for _, y in pairs)
