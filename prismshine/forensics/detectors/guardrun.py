"""Guard / run family detectors."""

from __future__ import annotations

from typing import Any

from prismshine.handbook.loader import format_advice
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit


def gray_input(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind != "guard":
            continue
        decision = (
            step.detail.get("decision")
            or step.detail.get("verdict")
            or step.status
        )
        zone = step.detail.get("zone") or step.detail.get("band")
        is_gray = (
            str(decision).lower() in {"gray", "flag", "escalate"}
            or str(zone).lower() == "gray"
            or step.detail.get("gray") is True
        )
        if is_gray:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        guard_gate=step.detail.get("resolution_gate", decision),
                    ),
                    evidence={"trace_index": i, "hop": step.hop, "decision": decision},
                    signal_value=sig.signal_value,
                )
            )
    guard_state = bundle.node_state.get("guard_verdict") or {}
    if isinstance(guard_state, dict) and (
        guard_state.get("zone") == "gray" or guard_state.get("gray") is True
    ):
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(
                    sig.advice,
                    guard_gate=guard_state.get("resolution_gate", "GUARD_GRAY"),
                ),
                evidence=dict(guard_state),
                signal_value=sig.signal_value,
            )
        )
    return hits


def hop_budget(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    used = bundle.node_state.get("hop_count")
    limit = bundle.node_state.get("hop_limit")
    exhausted = bundle.node_state.get("hop_budget_exhausted") is True
    if exhausted or (
        used is not None and limit is not None and int(used) >= int(limit)
    ):
        return [
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(sig.advice, used=used, limit=limit),
                evidence={"used": used, "limit": limit},
                signal_value=sig.signal_value,
            )
        ]
    return []


def anti_thrash(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    if bundle.node_state.get("anti_thrash") is True:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(
                    sig.advice,
                    hop=bundle.node_state.get("anti_thrash_hop", "unknown"),
                    action=bundle.node_state.get("anti_thrash_action", "repeated"),
                ),
                evidence={
                    "hop": bundle.node_state.get("anti_thrash_hop"),
                    "action": bundle.node_state.get("anti_thrash_action"),
                },
                signal_value=sig.signal_value,
            )
        )
    for i, step in enumerate(bundle.trace):
        if step.detail.get("anti_thrash") is True:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        hop=step.hop,
                        action=step.detail.get("action", "repeated"),
                    ),
                    evidence={"trace_index": i, "hop": step.hop},
                    signal_value=sig.signal_value,
                )
            )
    return hits
