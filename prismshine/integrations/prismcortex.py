"""PrismCortex event subscription → cache invalidation orchestration."""

from __future__ import annotations

import logging
from typing import Any, Callable

from prismshine.integrations.chorusgraph import on_fact_corrected

logger = logging.getLogger(__name__)


def bind_memory_invalidation(
    memory: Any,
    *,
    cache: Any | None = None,
    sidecar: Any | None = None,
    stack: Any | None = None,
    partition: str | None = None,
    threshold: float = 0.55,
) -> Callable[[], None]:
    """Subscribe to Memory.on_event; on accommodate/forget run consistency hooks.

    Returns an unsubscribe callable.
    """

    def _callback(event: Any) -> None:
        kind = getattr(event, "kind", None) or (
            event.get("kind") if isinstance(event, dict) else None
        )
        if kind not in {"accommodate", "forget"}:
            return
        subject = getattr(event, "subject", None) or (
            event.get("subject") if isinstance(event, dict) else None
        )
        # Prefer vector from event when present; else tag invalidation only
        vector = getattr(event, "vector", None) or (
            event.get("vector") if isinstance(event, dict) else None
        )
        subjects = [str(subject)] if subject else []
        on_fact_corrected(
            cache=cache,
            sidecar=sidecar,
            stack=stack,
            partition=partition,
            query_vector=list(vector) if vector is not None else None,
            threshold=threshold,
            subjects=subjects,
        )
        # Also try native bind_cache_revalidate if available on a service wrapper
        if hasattr(memory, "bind_cache_revalidate") and sidecar is not None:
            try:
                memory.bind_cache_revalidate(sidecar, threshold=threshold)
            except Exception as exc:  # noqa: BLE001
                logger.debug("bind_cache_revalidate: %s", exc)

    if not hasattr(memory, "on_event"):
        raise RuntimeError("memory object has no on_event; require prismcortex>=0.3.0")
    unsub = memory.on_event(_callback)
    if callable(unsub):
        return unsub
    return lambda: None


def hit_meta_to_trace_detail(hit_meta: Any) -> dict[str, Any]:
    """Normalize cache HitMeta into TraceStep.detail fields for detectors."""
    if hit_meta is None:
        return {}
    if isinstance(hit_meta, dict):
        return {
            "created_at": hit_meta.get("created_at"),
            "tags": hit_meta.get("tags"),
            "llm_model": hit_meta.get("llm_model"),
            "similarity": hit_meta.get("similarity"),
        }
    return {
        "created_at": getattr(hit_meta, "created_at", None),
        "tags": getattr(hit_meta, "tags", None),
        "llm_model": getattr(hit_meta, "llm_model", None),
        "similarity": getattr(hit_meta, "similarity", None),
    }
