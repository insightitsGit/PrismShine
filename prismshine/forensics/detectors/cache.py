"""Cache-family detectors."""

from __future__ import annotations

from typing import Any

from prismshine.handbook.loader import format_advice
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit


def _cache_steps(bundle: EvidenceBundle):
    for i, step in enumerate(bundle.trace):
        if step.kind == "cache":
            yield i, step


def stale_reuse(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for i, step in _cache_steps(bundle):
        kind = step.detail.get("decision") or step.detail.get("kind")
        if kind != "HIT_REUSE":
            continue
        entry_version = step.detail.get("entry_partition_version", step.detail.get("partition_version"))
        current = step.detail.get("current_partition_version")
        if entry_version is None or current is None:
            continue
        if int(entry_version) < int(current):
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        hop=step.hop,
                        entry_version=entry_version,
                        current_version=current,
                    ),
                    evidence={
                        "trace_index": i,
                        "hop": step.hop,
                        "entry_version": entry_version,
                        "current_version": current,
                    },
                    signal_value=sig.signal_value,
                )
            )
    return hits


def predates_fact_update(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    thr = float(params.get("similarity_threshold", 0.55))
    corrections = list(bundle.node_state.get("fact_corrections") or [])
    hits: list[SignatureHit] = []
    for i, step in _cache_steps(bundle):
        kind = step.detail.get("decision") or step.detail.get("kind")
        if kind not in {"HIT_REUSE", "HIT_AS_CONTEXT"}:
            continue
        created_at = step.detail.get("created_at")
        if created_at is None:
            continue
        tags = set(step.detail.get("tags") or [])
        query_sim = step.detail.get("correction_similarity")
        for corr in corrections:
            valid_from = corr.get("valid_from")
            subject = corr.get("subject", "")
            if valid_from is None:
                continue
            related = subject in tags or subject in (step.detail.get("subjects") or [])
            if query_sim is not None:
                related = related or float(query_sim) >= thr
            if corr.get("related") is True:
                related = True
            if related and str(created_at) < str(valid_from):
                hits.append(
                    SignatureHit(
                        id=sig.id,
                        title=sig.title,
                        severity=sig.severity,
                        scope=sig.scope,
                        advice=format_advice(
                            sig.advice,
                            hop=step.hop,
                            created_at=created_at,
                            valid_from=valid_from,
                            subject=subject,
                        ),
                        evidence={
                            "trace_index": i,
                            "hop": step.hop,
                            "created_at": created_at,
                            "valid_from": valid_from,
                            "subject": subject,
                        },
                        signal_value=sig.signal_value,
                    )
                )
    return hits


def marginal_hit(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    eps = float(params.get("epsilon", 0.01))
    hits: list[SignatureHit] = []
    for i, step in _cache_steps(bundle):
        kind = step.detail.get("decision") or step.detail.get("kind")
        if kind != "HIT_REUSE":
            continue
        score = step.scores.get("verify_score", step.detail.get("verify_score"))
        threshold = step.detail.get("threshold", step.scores.get("threshold"))
        if score is None or threshold is None:
            continue
        score_f = float(score)
        thr_f = float(threshold)
        if thr_f <= score_f < thr_f + eps:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        hop=step.hop,
                        score=score_f,
                        epsilon=eps,
                        threshold=thr_f,
                    ),
                    evidence={
                        "trace_index": i,
                        "hop": step.hop,
                        "score": score_f,
                        "threshold": thr_f,
                    },
                    signal_value=sig.signal_value,
                )
            )
    return hits


def context_misuse(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    """Fires when node_state marks a sole-support fact from HIT_AS_CONTEXT."""
    hits: list[SignatureHit] = []
    sole = list(bundle.node_state.get("cache_sole_support_facts") or [])
    cache_hops = [
        t.hop
        for t in bundle.trace
        if t.kind == "cache"
        and (t.detail.get("decision") or t.detail.get("kind")) == "HIT_AS_CONTEXT"
    ]
    if not sole or not cache_hops:
        return []
    hop = cache_hops[-1]
    for fact in sole:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(sig.advice, hop=hop, fact=fact),
                evidence={"hop": hop, "fact": fact},
                signal_value=sig.signal_value,
            )
        )
    return hits
