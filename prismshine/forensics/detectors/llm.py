"""LLM-hop failure detectors (provider errors mapped into TraceStep kind=llm)."""

from __future__ import annotations

from typing import Any

from prismshine.handbook.loader import format_advice
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit


def error(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind != "llm":
            continue
        if step.status in {"error", "timeout"}:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        hop=step.hop,
                        status=step.status,
                        detail=step.detail.get("error")
                        or step.detail.get("message")
                        or step.status,
                    ),
                    evidence={
                        "trace_index": i,
                        "hop": step.hop,
                        "status": step.status,
                        "detail": step.detail,
                    },
                    signal_value=sig.signal_value,
                )
            )
    return hits


def empty_completion(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind != "llm":
            continue
        empty = (
            step.status == "empty"
            or step.detail.get("empty") is True
            or (
                step.status == "ok"
                and step.detail.get("completion") in ("", None, [])
                and step.detail.get("tokens_out", 1) == 0
            )
        )
        if empty:
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
    # Also: answer present but blank after an llm hop
    if bundle.answer is not None and not str(bundle.answer).strip():
        if any(t.kind == "llm" for t in bundle.trace):
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop="generate"),
                    evidence={"answer_empty": True},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def refusal(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    """Safety/policy refusal finish reasons — not a hallucination, but blocks grounding."""
    hits: list[SignatureHit] = []
    markers = {
        "content_filter",
        "refusal",
        "safety",
        "blocked",
        "policy",
    }
    for i, step in enumerate(bundle.trace):
        if step.kind != "llm":
            continue
        reason = str(
            step.detail.get("finish_reason")
            or step.detail.get("stop_reason")
            or step.detail.get("refusal")
            or ""
        ).lower()
        if any(m in reason for m in markers) or step.detail.get("refused") is True:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop=step.hop, reason=reason),
                    evidence={"trace_index": i, "hop": step.hop, "reason": reason},
                    signal_value=sig.signal_value,
                )
            )
    return hits
