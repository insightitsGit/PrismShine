"""Detector registry: dotted names → callable(bundle, params, sig) -> list[SignatureHit]."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prismshine.forensics.detectors import (
    cache,
    context,
    guardrun,
    llm,
    memory,
    retrieval,
    tools,
)
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit

DetectorFn = Callable[
    [EvidenceBundle, dict[str, Any], SignatureDef], list[SignatureHit]
]

REGISTRY: dict[str, DetectorFn] = {
    "retrieval.empty": retrieval.empty,
    "retrieval.low_relevance": retrieval.low_relevance,
    "retrieval.error": retrieval.error,
    "retrieval.category_mismatch": retrieval.category_mismatch,
    "retrieval.partial": retrieval.partial,
    "retrieval.skipped_after_cache_miss": retrieval.skipped_after_cache_miss,
    "tools.error_swallowed": tools.error_swallowed,
    "tools.empty_result": tools.empty_result,
    "tools.timeout": tools.timeout,
    "tools.schema_drift": tools.schema_drift,
    "context.truncated": context.truncated,
    "context.missing_state_key": context.missing_state_key,
    "context.duplication": context.duplication,
    "context.low_fidelity_space": context.low_fidelity_space,
    "context.encoder_mismatch": context.encoder_mismatch,
    "context.trace_incomplete": context.trace_incomplete,
    "context.parallel_ambiguity": context.parallel_ambiguity,
    "cache.stale_reuse": cache.stale_reuse,
    "cache.predates_fact_update": cache.predates_fact_update,
    "cache.marginal_hit": cache.marginal_hit,
    "cache.context_misuse": cache.context_misuse,
    "cache.revalidate_ignored": cache.revalidate_ignored,
    "memory.conflict_served": memory.conflict_served,
    "memory.staged_fact": memory.staged_fact,
    "memory.expired_fact": memory.expired_fact,
    "memory.conflicting_preload": memory.conflicting_preload,
    "guardrun.gray_input": guardrun.gray_input,
    "guardrun.hop_budget": guardrun.hop_budget,
    "guardrun.anti_thrash": guardrun.anti_thrash,
    "llm.error": llm.error,
    "llm.empty_completion": llm.empty_completion,
    "llm.refusal": llm.refusal,
}


def get_detector(name: str) -> DetectorFn:
    if name not in REGISTRY:
        raise KeyError(f"Unknown detector: {name}")
    return REGISTRY[name]
