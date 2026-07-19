"""PrismGuard symmetry: consume gray input + expose output-side check."""

from __future__ import annotations

from typing import Any, Callable

from prismshine.gate import ShineGate
from prismshine.models import EvidenceBundle, ShineVerdict


def consume_guard_verdict(
    bundle: EvidenceBundle, guard_verdict: dict[str, Any] | Any
) -> EvidenceBundle:
    """Inject Guard input verdict into bundle so GUARD_GRAY_INPUT can fire."""
    state = dict(bundle.node_state)
    if hasattr(guard_verdict, "model_dump"):
        gv = guard_verdict.model_dump(mode="json")
    elif isinstance(guard_verdict, dict):
        gv = dict(guard_verdict)
    else:
        gv = {
            "decision": getattr(guard_verdict, "decision", None),
            "resolution_gate": getattr(guard_verdict, "resolution_gate", None),
            "fused_score": getattr(guard_verdict, "fused_score", None),
            "zone": getattr(guard_verdict, "zone", None),
        }
    decision = str(gv.get("decision") or "").lower()
    score = gv.get("fused_score")
    is_gray = (
        gv.get("zone") == "gray"
        or gv.get("gray") is True
        or decision in {"flag", "gray", "escalate"}
        or (score is not None and 0.25 <= float(score) < 0.55)
    )
    if is_gray:
        gv["gray"] = True
        gv["zone"] = "gray"
    state["guard_verdict"] = gv
    trace = list(bundle.trace)
    from prismshine.models import TraceStep

    trace.append(
        TraceStep(
            hop="prismguard",
            kind="guard",
            status="ok",
            detail={
                "decision": gv.get("decision"),
                "resolution_gate": gv.get("resolution_gate"),
                "zone": gv.get("zone"),
                "gray": gv.get("gray"),
            },
        )
    )
    return bundle.model_copy(update={"node_state": state, "trace": trace})


def as_output_gate(
    gate: ShineGate,
) -> Callable[[EvidenceBundle], dict[str, Any]]:
    """Expose Shine as a Guard-compatible output check."""

    def _check(bundle: EvidenceBundle) -> dict[str, Any]:
        verdict = gate.verify(bundle)
        return guard_compatible(verdict)

    return _check


def guard_compatible(verdict: ShineVerdict) -> dict[str, Any]:
    """Map ShineVerdict to Guard-like decision vocabulary."""
    return {
        "decision": verdict.decision,
        "resolution_gate": verdict.resolution_gate,
        "fused_score": verdict.fused_score,
        "confidence": verdict.confidence,
        "evidence_hash": verdict.evidence_hash,
        "advice": list(verdict.advice),
        "signatures": [s.id for s in verdict.signatures],
    }
