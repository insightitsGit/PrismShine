"""Weighted fusion, bands, gate naming, confidence."""

from __future__ import annotations

from dataclasses import dataclass

from prismshine.models import Decision, Signal, SignatureHit
from prismshine.policy import EffectivePolicy


@dataclass
class FusionResult:
    fused_score: float
    decision: Decision
    resolution_gate: str
    confidence: float
    band: str


def _calibrate(value: float, curve: list[tuple[float, float]] | None = None) -> float:
    """Identity by default; piecewise-linear curve if provided."""
    v = max(0.0, min(1.0, value))
    if not curve:
        return v
    curve = sorted(curve, key=lambda p: p[0])
    if v <= curve[0][0]:
        return curve[0][1]
    if v >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:], strict=False):
        if x0 <= v <= x1:
            t = (v - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return v


def fuse(
    signals: list[Signal],
    signatures: list[SignatureHit],
    policy: EffectivePolicy,
    *,
    has_fatal_halt: bool = False,
    early_gate: str | None = None,
    calibration: dict[str, list[tuple[float, float]]] | None = None,
    gray_unresolved: bool = False,
    judge_present: bool = False,
) -> FusionResult:
    if early_gate:
        decision: Decision = "block"
        if "REGENERATE" in early_gate or early_gate.endswith(":REGENERATE"):
            decision = "regenerate"
        sev = next((s.severity for s in signatures if s.severity == "fatal"), None)
        if sev == "fatal":
            decision = "block"
        return FusionResult(
            fused_score=1.0 if has_fatal_halt else 0.9,
            decision=decision,
            resolution_gate=early_gate,
            confidence=0.95,
            band="act",
        )

    calib = calibration or {}
    w = policy.weights
    contrib = 0.0

    fatal = [s for s in signatures if s.severity == "fatal"]
    if fatal:
        return FusionResult(
            fused_score=1.0,
            decision="block",
            resolution_gate=f"HANDBOOK:{fatal[0].id}",
            confidence=0.95,
            band="act",
        )

    by_name = {s.name: s for s in signals}
    # warnings agg
    warn = by_name.get("forensics.warnings")
    if warn:
        contrib += w["warnings"] * _calibrate(warn.value, calib.get(warn.name))

    err = by_name.get("forensics.errors")
    if err:
        contrib += 0.85 * _calibrate(err.value, calib.get(err.name))

    t1 = by_name.get("grounding.unmatched_ratio")
    if t1:
        contrib += w["t1"] * _calibrate(t1.value, calib.get(t1.name))

    t4 = by_name.get("grounding.judge_risk")
    t3 = by_name.get("grounding.unsupported_span_ratio")
    t2 = by_name.get("grounding.risk_coverage")
    cue = by_name.get("grounding.contradiction_cue")

    if t4 is not None and judge_present:
        contrib += w["t4"] * _calibrate(t4.value, calib.get(t4.name))
    else:
        if t2:
            contrib += w["t2"] * _calibrate(t2.value, calib.get(t2.name))
        if cue:
            # unresolved contradiction cues
            contrib += w["contradiction"] * _calibrate(cue.value, calib.get(cue.name))
        if t3:
            contrib += w["t3"] * _calibrate(t3.value, calib.get(t3.name))

    fused = max(0.0, min(1.0, contrib))
    b_pass, b_gray, b_act = policy.bands

    if fused < b_pass:
        decision = "pass"
        gate = "FUSION_PASS"
        band = "pass"
    elif fused < b_gray:
        decision = "flag" if gray_unresolved else "flag"
        gate = "FUSION_GRAY"
        band = "gray"
    elif fused < b_act:
        decision = "flag"
        gate = "FUSION_ACT"
        band = "act"
    else:
        decision = "block"
        gate = "FUSION_BLOCK"
        band = "block"

    # Name gate from dominant signal when crossing
    if t1 and t1.value >= 0.5 and fused >= b_gray:
        gate = "T1_UNMATCHED_FACTS"
    if t2 and t2.value >= (1.0 - policy.tau_floor) and fused >= b_act:
        gate = "T2_COVERAGE_FAIL"
    if cue and cue.value > 0 and fused >= b_gray:
        gate = "T2_CONTRADICTION_CUE"
    if t3 and t3.value >= 0.3:
        gate = "T3_SPANS_CONFIRMED"
    if t4 and judge_present and t4.value >= 0.5:
        gate = "T4_JUDGE"

    # confidence: distance from nearest band boundary, discounted by disagreement
    boundaries = [b_pass, b_gray, b_act]
    dist = min(abs(fused - b) for b in boundaries)
    disagreement = 0.0
    vals = [s.value for s in signals if s.weight > 0]
    if len(vals) >= 2:
        disagreement = float(max(vals) - min(vals))
    confidence = max(0.0, min(1.0, (dist / 0.25) * (1.0 - 0.5 * disagreement)))

    if gray_unresolved and decision == "pass":
        decision = "flag"
        gate = "MISSING_CAPABILITY_FLAG"

    return FusionResult(
        fused_score=fused,
        decision=decision,
        resolution_gate=gate,
        confidence=confidence,
        band=band,
    )
