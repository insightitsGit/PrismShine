"""Regression tests for the design-gap closures."""

from __future__ import annotations

import numpy as np

from prismshine.calibrate import calibrate_synthetic
from prismshine.evidence.builder import bundle_from_dict
from prismshine.forensics.engine import run_forensics
from prismshine.gate import ShineGate
from prismshine.grounding.copycheck import copycheck
from prismshine.grounding.contradiction import screen_contradictions
from prismshine.handbook.loader import load_handbook
from prismshine.integrations.chorusgraph import on_fact_corrected, shine_node
from prismshine.regen import build_repair_feedback, next_route


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_capabilities_announce_limits():
    gate = ShineGate.build(embedder=_embed)
    caps = gate.capabilities()
    assert caps["pass_means"] == "grounded_in_preload_not_world_true"
    assert caps["buffered_display"] is True
    assert caps["span_backend"] in {"onnx", "lexical", "unavailable"}
    assert caps["threshold_status"] in {
        "proposal",
        "validated-synthetic",
        "validated-labeled",
    }
    assert any("PASS means" in n for n in caps["notes"])


def test_domain_packs_merge_severity():
    hb = load_handbook(domain="clinical")
    assert "clinical" in hb.handbook_version or hb.handbook_version.startswith("0.1")
    staged = hb.by_id()["STAGED_FACT_SERVED"]
    assert staged.severity == "error"
    fin = load_handbook(domain="finance")
    assert fin.by_id()["MARGINAL_CACHE_HIT"].severity == "error"


def test_clinical_profile_loads_pack():
    gate = ShineGate.build(profile="clinical", embedder=_embed)
    assert gate.policy.contradiction_forces_judge is True
    assert gate.policy.mandatory_tier3 is True
    assert "clinical" in gate.handbook.handbook_version or any(
        s.severity == "error" and s.id == "STAGED_FACT_SERVED"
        for s in gate.handbook.signatures
    )


def test_structured_json_copycheck():
    b, _ = bundle_from_dict(
        {
            "question": "revenue?",
            "answer": '{"revenue": 9999, "currency": "USD"}',
            "preload": [
                {"chunk_id": "1", "text": '{"revenue": 1000, "currency": "USD"}'}
            ],
        }
    )
    r = copycheck(b, numeric_tolerance=0.0)
    assert r.unmatched_ratio > 0
    assert any("revenue" in f.raw for f in r.unmatched)


def test_structured_json_match():
    b, _ = bundle_from_dict(
        {
            "question": "revenue?",
            "answer": '{"revenue": 1000, "currency": "USD"}',
            "preload": [
                {"chunk_id": "1", "text": "Revenue was 1000 USD in the filing."}
            ],
        }
    )
    r = copycheck(b)
    assert r.unmatched_ratio == 0.0 or len(r.matched) >= 1


def test_contradiction_high_stakes_cannot_pass_without_judge():
    gate = ShineGate.build(profile="clinical", embedder=_embed, judge=None)
    b, _ = bundle_from_dict(
        {
            "question": "Is the drug safe?",
            "answer": "The drug is safe for children.",
            "preload": [
                {
                    "chunk_id": "c1",
                    "text": "The drug is not safe for children.",
                    "source": "retrieval",
                }
            ],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "scores": {"constructive_score": 0.95},
                    "detail": {"n_chunks": 2, "top_k": 2},
                }
            ],
        }
    )
    # Ensure cue fires
    cues = screen_contradictions(
        ["The drug is safe for children."],
        [("c1", "The drug is not safe for children.")],
    )
    assert cues
    v = gate.verify(b)
    assert v.decision != "pass"


def test_regen_bound_and_feedback():
    assert next_route("regenerate", 0) == "regenerate"
    assert next_route("regenerate", 1) == "flag"
    fb = build_repair_feedback(
        spans=[{"text": "bad fact", "reason": "unmatched_number"}],
        advice=["fix the number"],
        signatures=["EMPTY_RETRIEVAL"],
    )
    assert "prompt_suffix" in fb
    assert "bad fact" in fb["prompt_suffix"]


def test_shine_node_regen_prompt():
    gate = ShineGate.build(embedder=_embed)

    class _ForceRegen:
        """Wrap gate.verify to force regenerate once for protocol test."""

        def __init__(self, g):
            self._g = g

        def verify(self, bundle):
            v = self._g.verify(bundle)
            return v.model_copy(
                update={
                    "decision": "regenerate",
                    "advice": ["retry with spans"],
                    "spans": v.spans,
                }
            )

    node = shine_node(_ForceRegen(gate), max_regenerate=1)  # type: ignore[arg-type]
    out = node(
        {
            "question": "q",
            "reply": "x",
            "docs": [{"chunk_id": "1", "text": "x data"}],
        }
    )
    assert out["shine_route"] == "regenerate"
    assert "shine_repair_prompt" in out
    out2 = node(
        {
            "question": "q",
            "reply": "x",
            "docs": [{"chunk_id": "1", "text": "x data"}],
            "_shine_regen_attempts": 1,
        }
    )
    assert out2["shine_route"] == "flag"
    assert out2.get("shine_regen_exhausted") is True


def test_dual_rail_cache_predates_without_prevention():
    """Prevention disabled: CACHE_PREDATES_FACT_UPDATE must still fire (100%)."""
    from prismshine.handbook.loader import load_handbook

    hb = load_handbook()
    # Simulate: no invalidation called, but detection still sees stale hit
    b, _ = bundle_from_dict(
        {
            "question": "Who is A?",
            "answer": "Person A is my brother.",
            "preload": [
                {
                    "chunk_id": "h1",
                    "text": "Person A is my sister.",
                    "source": "history",
                }
            ],
            "trace": [
                {
                    "hop": "cache",
                    "kind": "cache",
                    "status": "ok",
                    "detail": {
                        "decision": "HIT_REUSE",
                        "created_at": "2026-01-01T00:00:00",
                        "tags": ["person_a"],
                    },
                }
            ],
            "node_state": {
                "fact_corrections": [
                    {
                        "subject": "person_a",
                        "valid_from": "2026-02-01T00:00:00",
                    }
                ]
            },
        }
    )
    # Prevention path no-ops (no cache object) — detection must still catch
    on_fact_corrected(cache=None, sidecar=None, stack=None, subjects=["person_a"])
    hits = run_forensics(b, hb)
    assert "CACHE_PREDATES_FACT_UPDATE" in {h.id for h in hits.hits}


def test_calibration_receipt_updates_status():
    gate = ShineGate.build(embedder=_embed)
    assert gate.policy.threshold_status == "proposal"
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $1000.",
            "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
        }
    )
    report = calibrate_synthetic([b], gate=gate, seed=1)
    assert report.mode == "synthetic"
    assert gate.policy.threshold_status == "validated-synthetic"
    assert gate.capabilities()["threshold_status"] == "validated-synthetic"
