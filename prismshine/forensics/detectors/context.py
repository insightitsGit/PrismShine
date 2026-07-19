"""Context-assembly family detectors."""

from __future__ import annotations

import re
from typing import Any

from prismshine.handbook.loader import format_advice
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit


def truncated(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    budget = bundle.context_budget
    if budget is None:
        return []
    if budget.truncated or budget.truncated_tail:
        return [
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(
                    sig.advice,
                    used=budget.used_tokens,
                    limit=budget.limit_tokens,
                    truncated_tail=budget.truncated_tail or budget.truncated,
                ),
                evidence={
                    "used_tokens": budget.used_tokens,
                    "limit_tokens": budget.limit_tokens,
                    "truncated": budget.truncated,
                },
                signal_value=sig.signal_value,
            )
        ]
    if (
        budget.limit_tokens is not None
        and budget.used_tokens is not None
        and budget.used_tokens > budget.limit_tokens
    ):
        return [
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(
                    sig.advice,
                    used=budget.used_tokens,
                    limit=budget.limit_tokens,
                    truncated_tail=True,
                ),
                evidence={
                    "used_tokens": budget.used_tokens,
                    "limit_tokens": budget.limit_tokens,
                },
                signal_value=sig.signal_value,
            )
        ]
    return []


def missing_state_key(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    keys = list(bundle.node_state.get("missing_keys") or [])
    referenced = list(bundle.node_state.get("prompt_keys") or [])
    for key in referenced:
        if key not in bundle.node_state or bundle.node_state.get(key) in (None, "", [], {}):
            if key not in keys:
                keys.append(key)
    for key in keys:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(sig.advice, key=key),
                evidence={"key": key},
                signal_value=sig.signal_value,
            )
        )
    return hits


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def duplication(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    thr = float(params.get("jaccard_threshold", 0.9))
    hits: list[SignatureHit] = []
    chunks = bundle.preload
    for i in range(len(chunks)):
        ti = _tokens(chunks[i].text)
        if not ti:
            continue
        for j in range(i + 1, len(chunks)):
            tj = _tokens(chunks[j].text)
            if not tj:
                continue
            score = len(ti & tj) / len(ti | tj)
            if score >= thr:
                hits.append(
                    SignatureHit(
                        id=sig.id,
                        title=sig.title,
                        severity=sig.severity,
                        scope=sig.scope,
                        advice=format_advice(
                            sig.advice,
                            a=chunks[i].chunk_id,
                            b=chunks[j].chunk_id,
                            score=score,
                        ),
                        evidence={
                            "a": chunks[i].chunk_id,
                            "b": chunks[j].chunk_id,
                            "jaccard": score,
                        },
                        signal_value=sig.signal_value,
                    )
                )
    return hits


def low_fidelity_space(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for chunk in bundle.preload:
        if chunk.vector_space.startswith("jl-64") or chunk.vector_space == "jl-64":
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, chunk_id=chunk.chunk_id),
                    evidence={"chunk_id": chunk.chunk_id, "vector_space": chunk.vector_space},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def encoder_mismatch(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    answer_artifact = bundle.node_state.get("answer_encoder_artifact") or bundle.node_state.get(
        "encoder_artifact_id"
    )
    if not answer_artifact:
        return []
    hits: list[SignatureHit] = []
    for chunk in bundle.preload:
        space = chunk.vector_space or ""
        # vector_space may be "raw-384@model_id"
        chunk_artifact = None
        if "@" in space:
            chunk_artifact = space.split("@", 1)[1]
        chunk_artifact = chunk_artifact or chunk.metadata.get("encoder_artifact_id")
        if chunk_artifact and chunk_artifact != answer_artifact and chunk.vector is not None:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        chunk_id=chunk.chunk_id,
                        chunk_artifact=chunk_artifact,
                        answer_artifact=answer_artifact,
                    ),
                    evidence={
                        "chunk_id": chunk.chunk_id,
                        "chunk_artifact": chunk_artifact,
                        "answer_artifact": answer_artifact,
                    },
                    signal_value=sig.signal_value,
                )
            )
    return hits
