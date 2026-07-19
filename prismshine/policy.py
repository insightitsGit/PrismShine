"""Strictness layers, domain profiles, and threshold matrix."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from prismshine.models import Strictness

STRICTNESS_ORDER: list[Strictness] = ["lenient", "standard", "strict", "paranoid"]

# DESIGN §5.5 default threshold matrix
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
    # ensure monotonic
    b0, b1, b2 = bands
    b1 = max(b1, b0 + 0.05)
    b2 = max(b2, b1 + 0.05)
    bands = (b0, b1, min(b2, 0.99))

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
        mandatory_tier3=(effective == "paranoid"),
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
