"""ChorusGraph ledger + state + chunk vectors → EvidenceBundle."""

from __future__ import annotations

import logging
from typing import Any

from prismshine.evidence.builder import bundle_from_dict
from prismshine.models import EvidenceBundle

logger = logging.getLogger(__name__)


def _normalize_kind(raw: str | None) -> str:
    k = (raw or "other").lower()
    mapping = {
        "retrieve": "retrieval",
        "retrieval": "retrieval",
        "tool": "tool",
        "tools": "tool",
        "cache": "cache",
        "cache.decision": "cache",
        "llm": "llm",
        "memory": "memory",
        "guard": "guard",
    }
    if k in mapping:
        return mapping[k]
    if k.startswith("cache"):
        return "cache"
    if k.startswith("retriev"):
        return "retrieval"
    if k.startswith("llm"):
        return "llm"
    if k.startswith("tool"):
        return "tool"
    return "other"


def bundle_from_chorusgraph(
    *,
    state: dict[str, Any],
    ledger_steps: list[Any] | None = None,
    question_key: str = "question",
    answer_key: str = "reply",
    run_id: str | None = None,
    tenant_id: str | None = None,
    stack: Any | None = None,
    chunk_ids: list[str] | None = None,
    partition: str | None = None,
    declared_sections: list[str] | None = None,
) -> tuple[EvidenceBundle, list[str]]:
    """Build a bundle from ChorusGraph run artifacts."""
    question = str(state.get(question_key) or state.get("query") or "")
    answer = state.get(answer_key)
    if answer is not None:
        answer = str(answer)

    preload: list[dict[str, Any]] = []
    # docs / context from state
    docs = state.get("docs") or state.get("chunks") or state.get("preload") or []
    if isinstance(docs, str):
        docs = [{"text": docs, "chunk_id": "docs"}]
    for i, d in enumerate(docs):
        if isinstance(d, str):
            preload.append({"chunk_id": f"c{i}", "text": d, "source": "retrieval"})
        elif isinstance(d, dict):
            preload.append(
                {
                    "chunk_id": str(d.get("chunk_id") or d.get("id") or f"c{i}"),
                    "text": str(d.get("text") or d.get("content") or ""),
                    "source": d.get("source") or "retrieval",
                    "vector": d.get("vector"),
                    "vector_space": d.get("vector_space") or ("raw-384" if d.get("vector") else "none"),
                    "retrieval_score": d.get("score") or d.get("retrieval_score"),
                    "metadata": dict(d.get("metadata") or {}),
                }
            )

    # history + memory (correctness requirement)
    for i, msg in enumerate(state.get("history") or state.get("messages") or []):
        if isinstance(msg, dict):
            text = str(msg.get("content") or msg.get("text") or "")
        else:
            text = str(msg)
        if text:
            preload.append(
                {"chunk_id": f"hist{i}", "text": text, "source": "history"}
            )
    for i, mem in enumerate(state.get("memory") or state.get("recalls") or []):
        if isinstance(mem, dict):
            text = str(mem.get("text") or mem.get("value") or "")
            meta = {k: v for k, v in mem.items() if k not in {"text", "value"}}
        else:
            text = str(mem)
            meta = {}
        if text:
            preload.append(
                {
                    "chunk_id": f"mem{i}",
                    "text": text,
                    "source": "memory",
                    "metadata": meta,
                }
            )

    # Warm-index vectors
    if stack is not None and chunk_ids and hasattr(stack, "get_chunk_vectors"):
        try:
            records = stack.get_chunk_vectors(chunk_ids, partition=partition)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_chunk_vectors failed: %s", exc)
            records = []
        by_id = {getattr(r, "chunk_id", None): r for r in records}
        for p in preload:
            rec = by_id.get(p["chunk_id"])
            if rec is None:
                continue
            try:
                vec = getattr(rec, "vector_384", None)
                if vec is None:
                    vec = getattr(rec, "vector", None)
                if vec is not None:
                    p["vector"] = list(vec)
                    art = getattr(rec, "encoder_artifact_id", None)
                    p["vector_space"] = f"raw-384@{art}" if art else "raw-384"
                    p.setdefault("metadata", {})["partition"] = getattr(
                        rec, "partition", partition
                    )
                    p["metadata"]["version"] = getattr(rec, "version", None)
                    if art:
                        p["metadata"]["encoder_artifact_id"] = art
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "warm-index vector injection failed for chunk %s: %s",
                    p.get("chunk_id"),
                    exc,
                )

    if not preload:
        preload = [{"chunk_id": "empty", "text": "(no preload)", "source": "system"}]

    trace: list[dict[str, Any]] = []
    for i, step in enumerate(ledger_steps or state.get("ledger") or []):
        if hasattr(step, "model_dump"):
            d = step.model_dump()
        elif isinstance(step, dict):
            d = step
        else:
            d = {
                "hop": getattr(step, "hop", None) or getattr(step, "node", f"hop{i}"),
                "kind": getattr(step, "kind", "other"),
                "detail": getattr(step, "detail", {}) or {},
                "scores": getattr(step, "scores", {}) or {},
                "status": getattr(step, "status", "ok"),
                "duration_ms": getattr(step, "duration_ms", None),
            }
        detail = dict(d.get("detail") or {})
        # ChorusGraph LedgerStep uses node=; shine TraceStep uses hop=
        hop = d.get("hop") or d.get("node") or f"hop{i}"
        kind = _normalize_kind(d.get("kind") or detail.get("kind"))
        status = d.get("status") or detail.get("status") or "ok"
        if kind == "retrieval" and detail.get("n_chunks") == 0:
            status = "empty"
        if kind == "cache" and "decision" not in detail and "kind" in detail:
            detail["decision"] = detail["kind"]
        if kind == "cache" and "decision" not in detail and d.get("cache_hit") is not None:
            detail["decision"] = "HIT_REUSE" if d.get("cache_hit") else "MISS"
        trace.append(
            {
                "hop": str(hop),
                "kind": kind,
                "status": status,
                "scores": dict(d.get("scores") or {}),
                "duration_ms": d.get("duration_ms"),
                "detail": detail,
            }
        )

    # Drop non-JSON / ledger payload keys — they belong in trace, not node_state hash.
    _skip_state = {
        "ledger_steps",
        "_ledger_steps",
        "ledger",
        "route_ledger",
        "docs",
        "chunks",
        "preload",
        "history",
        "messages",
        "memory",
        "recalls",
        "prism_sequence",
        "vector_hops",
    }
    node_state: dict[str, Any] = {}
    for k, v in state.items():
        if k in _skip_state or k.startswith("_"):
            continue
        if _is_jsonish(v):
            node_state[k] = v
    consumes = state.get("consumes") or []
    if consumes:
        node_state["consumes"] = list(consumes)
        missing = [
            k
            for k in consumes
            if k not in state or state.get(k) in (None, "", [], {})
        ]
        if missing:
            node_state["missing_keys"] = missing
    if state.get("expect_trace_kinds"):
        node_state["expect_trace_kinds"] = list(state["expect_trace_kinds"])
    if state.get("parallel_hops") is not None:
        node_state["parallel_hops"] = state["parallel_hops"]
    if state.get("answer_source_hop") is not None:
        node_state["answer_source_hop"] = state["answer_source_hop"]

    data = {
        "run_id": run_id or str(state.get("run_id") or "chorusgraph"),
        "tenant_id": tenant_id or state.get("tenant_id"),
        "question": question or "(missing question)",
        "answer": answer,
        "preload": preload,
        "trace": trace,
        "node_state": node_state,
        "declared_sections": declared_sections or list(state.get("declared_sections") or []),
        "context_budget": state.get("context_budget"),
    }
    return bundle_from_dict(data)


def _is_jsonish(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_jsonish(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_jsonish(v) for k, v in value.items())
    return False
