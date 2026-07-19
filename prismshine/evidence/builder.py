"""EvidenceBundle construction, validation, and capability feedback."""

from __future__ import annotations

from typing import Any

from prismshine.models import (
    ContextBudget,
    EvidenceBundle,
    PreloadChunk,
    TraceStep,
)


def bundle_from_dict(d: dict[str, Any]) -> tuple[EvidenceBundle, list[str]]:
    """Validate minimal contract and return (bundle, capability feedback)."""
    if not isinstance(d, dict):
        raise TypeError("bundle_from_dict expects a dict")
    question = d.get("question")
    if not question or not isinstance(question, str):
        raise ValueError("EvidenceBundle requires a non-empty 'question' string")

    preload_raw = d.get("preload") or []
    if not isinstance(preload_raw, list) or len(preload_raw) == 0:
        raise ValueError(
            "EvidenceBundle requires non-empty 'preload' with at least one chunk text"
        )
    for i, chunk in enumerate(preload_raw):
        if not isinstance(chunk, dict) or not str(chunk.get("text") or "").strip():
            raise ValueError(f"preload[{i}] must include non-empty 'text'")

    preload = [
        PreloadChunk(
            chunk_id=str(c.get("chunk_id") or f"c{i}"),
            text=str(c["text"]),
            vector=c.get("vector"),
            vector_space=str(c.get("vector_space") or ("raw-384" if c.get("vector") else "none")),
            source=c.get("source") or "retrieval",
            retrieval_score=c.get("retrieval_score"),
            metadata=dict(c.get("metadata") or {}),
        )
        for i, c in enumerate(preload_raw)
    ]

    trace = [
        TraceStep(
            hop=str(t.get("hop") or f"hop{i}"),
            kind=t.get("kind") or "other",
            status=t.get("status") or "ok",
            scores=dict(t.get("scores") or {}),
            duration_ms=t.get("duration_ms"),
            detail=dict(t.get("detail") or {}),
        )
        for i, t in enumerate(d.get("trace") or [])
    ]

    budget = None
    if d.get("context_budget"):
        budget = ContextBudget(**d["context_budget"])

    bundle = EvidenceBundle(
        run_id=str(d.get("run_id") or "run"),
        tenant_id=d.get("tenant_id"),
        question=question,
        answer=d.get("answer"),
        preload=preload,
        trace=trace,
        node_state=dict(d.get("node_state") or {}),
        declared_sections=list(d.get("declared_sections") or []),
        context_budget=budget,
    )
    return bundle, capability_feedback(bundle)


def capability_feedback(bundle: EvidenceBundle) -> list[str]:
    feedback: list[str] = []
    has_vectors = any(c.vector for c in bundle.preload)
    spaces = {c.vector_space for c in bundle.preload}
    if not has_vectors:
        feedback.append(
            "no vectors -> Tier 2 lexical mode; add PreloadChunk.vector (raw-384) to enable cosine coverage"
        )
    elif spaces <= {"jl-64", "none"} or ("jl-64" in spaces and "raw-384" not in spaces):
        feedback.append(
            "only jl-64 vectors -> Tier 2 runs with stricter tau_sent + LOW_FIDELITY_SPACE; add raw-384 vectors for full-fidelity coverage"
        )
    else:
        feedback.append("preload vectors present -> Tier 2 vector coverage enabled")

    if not bundle.trace:
        feedback.append(
            "no trace -> retrieval/tool/cache detector families dormant; add TraceStep entries to enable Tier-0 forensics"
        )
    else:
        kinds = {t.kind for t in bundle.trace}
        for family, kind in (
            ("retrieval", "retrieval"),
            ("tools", "tool"),
            ("cache", "cache"),
            ("memory", "memory"),
            ("guard", "guard"),
        ):
            if kind not in kinds:
                feedback.append(
                    f"no {kind}-kind trace steps -> {family} detector family dormant; add {kind} TraceSteps to enable"
                )

    sources = {c.source for c in bundle.preload}
    if "history" not in sources and "memory" not in sources:
        feedback.append(
            "no history/memory preload chunks -> conversation and memory grounding may false-positive; adapters MUST populate source=history|memory"
        )

    if bundle.answer is None:
        feedback.append(
            "answer=None -> pre-generation mode (Tier 0 only); supply answer for grounding tiers 1-4"
        )

    if bundle.context_budget is None:
        feedback.append(
            "no context_budget -> CONTEXT_TRUNCATED detector dormant; add ContextBudget to enable truncation detection"
        )

    return feedback
