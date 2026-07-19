"""Strictness layers, domain profiles, and threshold matrix."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from prismshine.models import Strictness

STRICTNESS_ORDER: list[Strictness] = ["lenient", "standard", "strict", "paranoid"]

ThresholdStatus = Literal["proposal", "validated-synthetic", "validated-labeled"]

# DESIGN §5.5 default threshold matrix — status is proposal until calibrate receipts land
PROFILE_MATRIX: dict[str, dict[str, Any]] = {
    "default": {
        "numeric_tolerance": 0.005,
        "date_granularity": "day",
        "tau_sent": 0.62,
        "tau_sent_jl": 0.80,
        "tau_floor": 0.20,
        "tau_tok": 0.50,
        "bands": (0.25, 0.55, 0.75),
        "tier4_budget": 0.10,
        "escalate_derived": False,
        "jl_allowed": True,
        "contradiction_forces_tier3": True,
        "contradiction_forces_judge": False,
        "mandatory_tier3": False,
        "threshold_status": "proposal",
    },
    "clinical": {
        "numeric_tolerance": 0.0,
        "date_granularity": "day",
        "tau_sent": 0.72,
        "tau_sent_jl": 0.99,
        "tau_floor": 0.30,
        "tau_tok": 0.35,
        "bands": (0.15, 0.40, 0.60),
        "tier4_budget": 0.25,
        "escalate_derived": True,
        "jl_allowed": False,
        "contradiction_forces_tier3": True,
        "contradiction_forces_judge": True,
        "mandatory_tier3": True,
        "threshold_status": "proposal",
    },
    "finance": {
        "numeric_tolerance": 0.0,
        "date_granularity": "day",
        "tau_sent": 0.70,
        "tau_sent_jl": 0.85,
        "tau_floor": 0.25,
        "tau_tok": 0.40,
        "bands": (0.18, 0.45, 0.65),
        "tier4_budget": 0.15,
        "escalate_derived": True,
        "jl_allowed": True,
        "contradiction_forces_tier3": True,
        "contradiction_forces_judge": True,
        "mandatory_tier3": True,
        "threshold_status": "proposal",
    },
    "legal": {
        "numeric_tolerance": 0.005,
        "date_granularity": "day",
        "tau_sent": 0.68,
        "tau_sent_jl": 0.82,
        "tau_floor": 0.25,
        "tau_tok": 0.45,
        "bands": (0.20, 0.48, 0.68),
        "tier4_budget": 0.15,
        "escalate_derived": False,
        "jl_allowed": True,
        "contradiction_forces_tier3": True,
        "contradiction_forces_judge": False,
        "mandatory_tier3": False,
        "threshold_status": "proposal",
    },
}

STRICTNESS_BAND_SHIFT = {
    "lenient": 0.08,
    "standard": 0.0,
    "strict": -0.07,
    "paranoid": -0.13,
}


@dataclass
class EffectivePolicy:
    profile: str
    strictness: Strictness
    strictness_effective: Strictness
    numeric_tolerance: float
    tau_sent: float
    tau_sent_jl: float
    tau_floor: float
    tau_tok: float
    bands: tuple[float, float, float]
    tier4_budget: float
    escalate_derived: bool
    jl_allowed: bool
    halt_on_fatal: bool = True
    mandatory_tier3: bool = False
    contradiction_forces_tier3: bool = True
    contradiction_forces_judge: bool = False
    threshold_status: ThresholdStatus = "proposal"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "fatal": 1.0,
            "warnings": 0.25,
            "t1": 0.30,
            "t2": 0.25,
            "contradiction": 0.30,
            "t3": 0.35,
            "t4": 0.45,
        }
    )
    extras: dict[str, Any] = field(default_factory=dict)


def bump_strictness(current: Strictness, steps: int = 1) -> Strictness:
    idx = STRICTNESS_ORDER.index(current)
    return STRICTNESS_ORDER[min(len(STRICTNESS_ORDER) - 1, idx + steps)]


def resolve_policy(
    profile: str = "default",
    strictness: Strictness = "standard",
    overrides: dict[str, Any] | None = None,
    *,
    dynamic_bump: int = 0,
    halt_on_fatal: bool = True,
) -> EffectivePolicy:
    base = deepcopy(PROFILE_MATRIX.get(profile, PROFILE_MATRIX["default"]))
    effective = bump_strictness(strictness, dynamic_bump) if dynamic_bump else strictness
    shift = STRICTNESS_BAND_SHIFT[effective]
    bands = tuple(max(0.0, min(1.0, b + shift)) for b in base["bands"])
    b0, b1, b2 = bands
    b1 = max(b1, b0 + 0.05)
    b2 = max(b2, b1 + 0.05)
    bands = (b0, b1, min(b2, 0.99))

    mandatory = bool(base.get("mandatory_tier3")) or (effective == "paranoid")
    pol = EffectivePolicy(
        profile=profile if profile in PROFILE_MATRIX else "default",
        strictness=strictness,
        strictness_effective=effective,
        numeric_tolerance=float(base["numeric_tolerance"]),
        tau_sent=float(base["tau_sent"]),
        tau_sent_jl=float(base["tau_sent_jl"]),
        tau_floor=float(base["tau_floor"]),
        tau_tok=float(base["tau_tok"]),
        bands=bands,  # type: ignore[arg-type]
        tier4_budget=float(base["tier4_budget"]),
        escalate_derived=bool(base["escalate_derived"]),
        jl_allowed=bool(base["jl_allowed"]),
        halt_on_fatal=halt_on_fatal,
        mandatory_tier3=mandatory,
        contradiction_forces_tier3=bool(base.get("contradiction_forces_tier3", True)),
        contradiction_forces_judge=bool(base.get("contradiction_forces_judge", False)),
        threshold_status=base.get("threshold_status", "proposal"),  # type: ignore[arg-type]
    )
    if overrides:
        for key, val in overrides.items():
            if hasattr(pol, key) and val is not None:
                setattr(pol, key, val)
            else:
                pol.extras[key] = val
            if key == "bands" and val is not None:
                pol.bands = tuple(val)  # type: ignore[assignment]
            if key == "weights" and isinstance(val, dict):
                pol.weights.update(val)
    return pol


def apply_calibration_receipt(
    policy: EffectivePolicy,
    *,
    thresholds: dict[str, float] | None = None,
    status: ThresholdStatus = "validated-synthetic",
) -> EffectivePolicy:
    """Apply calibrate() overlay thresholds and mark receipt status."""
    if thresholds:
        if "tau_sent" in thresholds:
            policy.tau_sent = float(thresholds["tau_sent"])
        if "tau_floor" in thresholds:
            policy.tau_floor = float(thresholds["tau_floor"])
        if "tau_tok" in thresholds:
            policy.tau_tok = float(thresholds["tau_tok"])
        b0, b1, b2 = policy.bands
        if "fused_pass" in thresholds:
            b0 = float(thresholds["fused_pass"])
        if "fused_flag" in thresholds:
            b1 = float(thresholds["fused_flag"])
        if "fused_act" in thresholds:
            b2 = float(thresholds["fused_act"])
        # Keep ordered bands
        b0, b1, b2 = sorted((b0, b1, b2))
        if b0 == b1:
            b1 = min(b0 + 0.05, 0.99)
        if b1 == b2:
            b2 = min(b1 + 0.05, 1.0)
        policy.bands = (b0, b1, b2)
    policy.threshold_status = status
    return policy
