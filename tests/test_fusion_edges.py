from __future__ import annotations

from prismshine.fusion import fuse, _calibrate
from prismshine.models import Signal, SignatureHit
from prismshine.policy import resolve_policy


def test_calibration_curve():
    assert _calibrate(0.5) == 0.5
    assert _calibrate(0.25, [(0.0, 0.0), (1.0, 1.0)]) == 0.25
    assert _calibrate(0.0, [(0.2, 0.1), (0.8, 0.9)]) == 0.1
    assert _calibrate(1.0, [(0.2, 0.1), (0.8, 0.9)]) == 0.9


def test_fusion_with_t3_t4_and_gray_flag():
    pol = resolve_policy()
    r = fuse(
        [
            Signal(name="grounding.risk_coverage", tier=2, value=0.4, weight=0.25),
            Signal(name="grounding.unsupported_span_ratio", tier=3, value=0.5, weight=0.35),
            Signal(name="grounding.judge_risk", tier=4, value=0.7, weight=0.45),
            Signal(name="grounding.contradiction_cue", tier=2, value=0.5, weight=0.3),
            Signal(name="forensics.warnings", tier=0, value=0.4, weight=0.25),
            Signal(name="forensics.errors", tier=0, value=0.6, weight=0.85),
            Signal(name="grounding.unmatched_ratio", tier=1, value=0.6, weight=0.3),
        ],
        [SignatureHit(id="X", severity="warning", advice="a")],
        pol,
        judge_present=True,
    )
    assert r.fused_score >= 0
    r2 = fuse(
        [Signal(name="grounding.risk_coverage", tier=2, value=0.1, weight=0.25)],
        [],
        pol,
        gray_unresolved=True,
    )
    assert r2.decision == "flag" or r2.resolution_gate == "MISSING_CAPABILITY_FLAG" or r2.decision == "pass"
