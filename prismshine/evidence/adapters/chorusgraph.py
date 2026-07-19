"""ChorusGraph ledger + state + chunk vectors → EvidenceBundle."""

from __future__ import annotations

from typing import Any

from prismshine.evidence.builder import bundle_from_dict
from prismshine.models import EvidenceBundle


def _normalize_kind(raw: str | None) -> str:
    k = (raw or "other").lower()
    mapping = {
        "retrieve": "retrieval",
        "retrieval": "retrieval",
        "tool": "tool",
        "tools": "tool",
        "cache": "cache",
        "llm": "llm",
        "memory": "memory",
        "guard": "guard",
    }
    return mapping.get(k, "other")


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
            by_id = {getattr(r, "chunk_id", None): r for r in records}
            for p in preload:
                rec = by_id.get(p["chunk_id"])
                if rec is None:
                    continue
                vec = getattr(rec, "vector_384", None) or getattr(rec, "vector", None)
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
        except Exception:  # noqa: BLE001
            pass

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
                "hop": getattr(step, "hop", f"hop{i}"),
                "kind": getattr(step, "kind", "other"),
                "detail": getattr(step, "detail", {}),
                "scores": getattr(step, "scores", {}),
                "status": getattr(step, "status", "ok"),
                "duration_ms": getattr(step, "duration_ms", None),
            }
        detail = dict(d.get("detail") or {})
        kind = _normalize_kind(d.get("kind") or detail.get("kind"))
        status = d.get("status") or detail.get("status") or "ok"
        if kind == "retrieval" and detail.get("n_chunks") == 0:
            status = "empty"
        if kind == "cache" and "decision" not in detail and "kind" in detail:
            detail["decision"] = detail["kind"]
        trace.append(
            {
                "hop": str(d.get("hop") or f"hop{i}"),
                "kind": kind,
                "status": status,
                "scores": dict(d.get("scores") or {}),
                "duration_ms": d.get("duration_ms"),
                "detail": detail,
            }
        )

    node_state = dict(state)
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
