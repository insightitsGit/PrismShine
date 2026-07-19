"""Retrieval-family detectors."""

from __future__ import annotations

from typing import Any

from prismshine.handbook.loader import format_advice
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit


def _must_ground(bundle: EvidenceBundle, params: dict[str, Any]) -> bool:
    section = params.get("applies_to_sections", "must_ground")
    if not bundle.declared_sections:
        return True
    return section in bundle.declared_sections or "must_ground" in bundle.declared_sections


def empty(bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef) -> list[SignatureHit]:
    if not _must_ground(bundle, params):
        return []
    hits: list[SignatureHit] = []
    min_chunks = int(params.get("min_chunks", 1))
    for i, step in enumerate(bundle.trace):
        if step.kind != "retrieval":
            continue
        n = int(step.detail.get("n_chunks", step.detail.get("chunk_count", -1)))
        if step.status == "empty" or (n >= 0 and n < min_chunks):
            n_val = 0 if step.status == "empty" and n < 0 else max(n, 0)
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop=step.hop, n=n_val),
                    evidence={"trace_index": i, "hop": step.hop, "n": n_val},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def low_relevance(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    min_score = float(params.get("min_score", 0.55))
    score_key = str(params.get("score_key", "constructive_score"))
    for i, step in enumerate(bundle.trace):
        if step.kind != "retrieval" or step.status in {"empty", "error", "timeout"}:
            continue
        n = int(step.detail.get("n_chunks", step.detail.get("chunk_count", 0)))
        if n <= 0 and not step.scores:
            continue
        scores = [float(v) for k, v in step.scores.items() if k == score_key]
        if not scores:
            scores = [float(v) for v in step.scores.values()]
        if not scores:
            continue
        max_score = max(scores)
        if max_score < min_score:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        hop=step.hop,
                        score_key=score_key,
                        max_score=max_score,
                        min_score=min_score,
                    ),
                    evidence={
                        "trace_index": i,
                        "hop": step.hop,
                        "max_score": max_score,
                    },
                    signal_value=sig.signal_value,
                )
            )
    return hits


def error(bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind != "retrieval":
            continue
        if step.status in {"error", "timeout"}:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop=step.hop, status=step.status),
                    evidence={"trace_index": i, "hop": step.hop, "status": step.status},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def category_mismatch(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    query_cat = bundle.node_state.get("query_category") or bundle.node_state.get(
        "inferred_category"
    )
    for i, step in enumerate(bundle.trace):
        if step.kind != "retrieval":
            continue
        rule_chain = step.detail.get("rule_chain") or []
        chunk_cat = None
        if isinstance(rule_chain, list) and rule_chain:
            last = rule_chain[-1]
            if isinstance(last, dict):
                chunk_cat = last.get("category") or last.get("slug")
            else:
                chunk_cat = str(last)
        chunk_cat = chunk_cat or step.detail.get("category")
        if query_cat and chunk_cat and str(query_cat) != str(chunk_cat):
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        hop=step.hop,
                        chunk_cat=chunk_cat,
                        query_cat=query_cat,
                    ),
                    evidence={
                        "trace_index": i,
                        "hop": step.hop,
                        "chunk_cat": chunk_cat,
                        "query_cat": query_cat,
                    },
                    signal_value=sig.signal_value,
                )
            )
    return hits


def partial(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    min_expected = int(params.get("min_chunks_expected", 3))
    for i, step in enumerate(bundle.trace):
        if step.kind != "retrieval" or step.status != "ok":
            continue
        top_k = int(step.detail.get("top_k", min_expected))
        n = int(step.detail.get("n_chunks", step.detail.get("chunk_count", top_k)))
        if n < top_k and n < min_expected:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        hop=step.hop,
                        n=n,
                        top_k=top_k,
                        min_chunks_expected=min_expected,
                    ),
                    evidence={"trace_index": i, "hop": step.hop, "n": n, "top_k": top_k},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def skipped_after_cache_miss(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    """Cache MISS but no subsequent retrieval hop — graph answered without fresh docs."""
    hits: list[SignatureHit] = []
    for i, step in enumerate(bundle.trace):
        if step.kind != "cache":
            continue
        decision = step.detail.get("decision") or step.detail.get("kind")
        if decision != "MISS":
            continue
        later_retrieval = any(
            t.kind == "retrieval" and j > i for j, t in enumerate(bundle.trace)
        )
        skipped = step.detail.get("retrieval_skipped") is True or (
            not later_retrieval
            and bundle.node_state.get("planned_retrieval") is True
        )
        # Also fire when MISS is followed only by llm with no retrieval
        if not later_retrieval and any(
            t.kind == "llm" and j > i for j, t in enumerate(bundle.trace)
        ):
            skipped = True
        if skipped:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(sig.advice, hop=step.hop),
                    evidence={"trace_index": i, "hop": step.hop, "decision": decision},
                    signal_value=sig.signal_value,
                )
            )
    return hits
