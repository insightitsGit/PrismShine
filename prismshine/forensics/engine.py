"""Tier-0 forensics engine: run handbook detectors over an EvidenceBundle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from prismshine.forensics.detectors import REGISTRY, get_detector
from prismshine.handbook.schema import Handbook
from prismshine.models import EvidenceBundle, SignatureHit, Signal


FAMILY_EVIDENCE: dict[str, Callable[[EvidenceBundle], bool]] = {
    "retrieval": lambda b: any(t.kind == "retrieval" for t in b.trace),
    "tools": lambda b: any(t.kind == "tool" for t in b.trace),
    "context": lambda b: True,
    "cache": lambda b: any(t.kind == "cache" for t in b.trace),
    "memory": lambda b: (
        any(t.kind == "memory" for t in b.trace)
        or any(c.source == "memory" for c in b.preload)
        or bool(b.node_state.get("memory_conflicts"))
        or bool(b.node_state.get("staged_facts"))
        or bool(b.node_state.get("preload_conflicts"))
        or any(c.source == "history" for c in b.preload)
    ),
    "guardrun": lambda b: (
        any(t.kind == "guard" for t in b.trace)
        or bool(b.node_state.get("guard_verdict"))
        or b.node_state.get("hop_budget_exhausted") is True
        or b.node_state.get("anti_thrash") is True
    ),
}


def _family_of(detector: str) -> str:
    return detector.split(".", 1)[0]


@dataclass
class ForensicsResult:
    hits: list[SignatureHit] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    dormant_families: list[str] = field(default_factory=list)
    fatal: bool = False


def dormant_families(bundle: EvidenceBundle) -> list[str]:
    out: list[str] = []
    for family, has_evidence in FAMILY_EVIDENCE.items():
        if not has_evidence(bundle):
            out.append(family)
    return out


def run_forensics(bundle: EvidenceBundle, handbook: Handbook) -> ForensicsResult:
    dormant = dormant_families(bundle)
    hits: list[SignatureHit] = []
    for sig in handbook.signatures:
        if sig.deprecated:
            continue
        family = _family_of(sig.detector)
        # CONFLICTING_PRELOAD_FACTS still runs without cortex via lexicon —
        # memory family is considered active when history/memory preload exists.
        if family in dormant and sig.id != "CONFLICTING_PRELOAD_FACTS":
            # still allow context family always; memory lexicon path:
            if family == "memory" and sig.id == "CONFLICTING_PRELOAD_FACTS":
                pass
            elif family != "context":
                continue
        try:
            detector = get_detector(sig.detector)
        except KeyError:
            continue
        # Skip low_fidelity / encoder mismatch unless answer encoder known /
        # jl chunks — detectors themselves no-op when inapplicable.
        found = detector(bundle, dict(sig.params), sig)
        hits.extend(found)

    # Recompute dormant after — CONFLICTING may need history which activates memory
    dormant = dormant_families(bundle)
    fatal = any(h.severity == "fatal" for h in hits)
    signals: list[Signal] = []
    warnings = [h for h in hits if h.severity == "warning"]
    errors = [h for h in hits if h.severity in {"error", "fatal"}]
    if fatal:
        top = next(h for h in hits if h.severity == "fatal")
        signals.append(
            Signal(
                name=f"forensics.{top.id.lower()}",
                tier=0,
                value=1.0,
                weight=1.0,
                detail={"signature": top.id},
            )
        )
    if errors:
        agg = max(h.signal_value for h in errors)
        signals.append(
            Signal(
                name="forensics.errors",
                tier=0,
                value=agg,
                weight=0.85,
                detail={"ids": [h.id for h in errors]},
            )
        )
    if warnings:
        signals.append(
            Signal(
                name="forensics.warnings",
                tier=0,
                value=min(1.0, sum(h.signal_value for h in warnings) / max(len(warnings), 1)),
                weight=0.25,
                detail={"ids": [h.id for h in warnings]},
            )
        )
    return ForensicsResult(
        hits=hits, signals=signals, dormant_families=dormant, fatal=fatal
    )


__all__ = ["ForensicsResult", "run_forensics", "dormant_families", "REGISTRY"]
