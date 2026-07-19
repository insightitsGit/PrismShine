"""Tool / API family detectors."""

from __future__ import annotations

from typing import Any

from prismshine.handbook.loader import format_advice
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit


def _has_error_marker(state: dict[str, Any]) -> bool:
    for key in ("error", "tool_error", "last_error", "errors"):
        val = state.get(key)
        if val:
            return True
    return False


def error_swallowed(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind != "tool" or step.status != "error":
            continue
        proceeded = bundle.answer is not None or any(
            t.kind == "llm" and t.status == "ok" for t in bundle.trace
        )
        if proceeded and not _has_error_marker(bundle.node_state):
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop=step.hop),
                    evidence={"trace_index": i, "hop": step.hop},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def empty_result(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    section = params.get("applies_to_sections", "must_ground")
    must = (
        not bundle.declared_sections
        or section in bundle.declared_sections
        or "must_ground" in bundle.declared_sections
    )
    if not must:
        return []
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind != "tool" or step.status != "ok":
            continue
        payload = step.detail.get("payload", step.detail.get("result"))
        empty = payload in (None, "", [], {}, ())
        if empty or step.detail.get("empty") is True:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop=step.hop),
                    evidence={"trace_index": i, "hop": step.hop},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def timeout(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind == "tool" and step.status == "timeout":
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop=step.hop),
                    evidence={"trace_index": i, "hop": step.hop},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def schema_drift(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    missing = list(bundle.node_state.get("missing_keys") or [])
    consumes = list(bundle.node_state.get("consumes") or [])
    for key in consumes:
        if key not in bundle.node_state or bundle.node_state.get(key) in (None, "", [], {}):
            if key not in missing:
                missing.append(key)
    tool_hops = [t.hop for t in bundle.trace if t.kind == "tool"]
    hop = tool_hops[-1] if tool_hops else "unknown"
    for key in missing:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(sig.advice, key=key, hop=hop),
                evidence={"key": key, "hop": hop},
                signal_value=sig.signal_value,
            )
        )
    return hits
