"""LangGraph state → EvidenceBundle."""

from __future__ import annotations

from typing import Any

from prismshine.evidence.builder import bundle_from_dict
from prismshine.models import EvidenceBundle


def bundle_from_langgraph(
    state: dict[str, Any],
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    docs_key: str = "docs",
    run_id: str | None = None,
) -> tuple[EvidenceBundle, list[str]]:
    question = str(state.get(question_key) or "")
    answer = state.get(answer_key)
    if answer is not None:
        answer = str(answer)

    preload: list[dict[str, Any]] = []
    docs = state.get(docs_key) or state.get("context") or []
    if isinstance(docs, str):
        docs = [docs]
    for i, d in enumerate(docs):
        if isinstance(d, str):
            preload.append({"chunk_id": f"d{i}", "text": d, "source": "retrieval"})
        elif isinstance(d, dict):
            preload.append(
                {
                    "chunk_id": str(d.get("chunk_id") or d.get("id") or f"d{i}"),
                    "text": str(d.get("text") or d.get("page_content") or d.get("content") or ""),
                    "source": d.get("source") or "retrieval",
                    "vector": d.get("vector") or d.get("embedding"),
                    "vector_space": d.get("vector_space")
                    or ("raw-384" if d.get("vector") or d.get("embedding") else "none"),
                    "retrieval_score": d.get("score"),
                    "metadata": dict(d.get("metadata") or {}),
                }
            )

    for i, msg in enumerate(state.get("messages") or state.get("history") or []):
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if content:
            preload.append(
                {"chunk_id": f"m{i}", "text": str(content), "source": "history"}
            )

    if not preload:
        preload = [{"chunk_id": "empty", "text": "(no preload)", "source": "system"}]

    trace = list(state.get("trace") or [])
    data = {
        "run_id": run_id or str(state.get("run_id") or "langgraph"),
        "tenant_id": state.get("tenant_id"),
        "question": question or "(missing question)",
        "answer": answer,
        "preload": preload,
        "trace": trace,
        "node_state": dict(state),
        "declared_sections": list(state.get("declared_sections") or []),
        "context_budget": state.get("context_budget"),
    }
    return bundle_from_dict(data)
