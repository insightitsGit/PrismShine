"""Actionable agent UX templates derived from verdicts / signatures."""

from __future__ import annotations

from typing import Any

from prismshine.models import ShineVerdict, SignatureHit


def clarify_conflict_question(hit: SignatureHit) -> str:
    """Ask-the-user template for CONFLICTING_PRELOAD_FACTS / MEMORY_CONFLICT_SERVED."""
    ev = hit.evidence or {}
    subject = ev.get("subject") or "this"
    relation = ev.get("relation") or "fact"
    a = ev.get("value_a") or "?"
    b = ev.get("value_b") or "?"
    return (
        f"I have conflicting information about {subject} ({relation}): "
        f'"{a}" vs "{b}". Which is correct? '
        "Please answer with one value so I can update memory and continue."
    )


def actions_for_verdict(verdict: ShineVerdict) -> list[dict[str, Any]]:
    """Structured actions an agent/UI can execute (not just advice strings)."""
    actions: list[dict[str, Any]] = []
    for hit in verdict.signatures:
        if hit.id in {"CONFLICTING_PRELOAD_FACTS", "MEMORY_CONFLICT_SERVED"}:
            actions.append(
                {
                    "type": "ask_user_clarify",
                    "signature": hit.id,
                    "question": clarify_conflict_question(hit),
                    "evidence": dict(hit.evidence),
                }
            )
        if hit.id in {"EMPTY_RETRIEVAL", "RETRIEVAL_ERROR", "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS"}:
            actions.append(
                {
                    "type": "repair_retrieval",
                    "signature": hit.id,
                    "hop": hit.evidence.get("hop"),
                    "advice": hit.advice,
                }
            )
        if hit.id in {"TOOL_ERROR_SWALLOWED", "TOOL_TIMEOUT", "TOOL_EMPTY_RESULT"}:
            actions.append(
                {
                    "type": "retry_tool",
                    "signature": hit.id,
                    "hop": hit.evidence.get("hop"),
                    "advice": hit.advice,
                }
            )
        if hit.id in {"LLM_ERROR", "LLM_EMPTY_COMPLETION"}:
            actions.append(
                {
                    "type": "retry_llm",
                    "signature": hit.id,
                    "hop": hit.evidence.get("hop"),
                    "advice": hit.advice,
                }
            )
    if verdict.decision == "regenerate":
        actions.append(
            {
                "type": "regenerate_with_feedback",
                "spans": [s.model_dump() for s in verdict.spans],
                "advice": list(verdict.advice),
            }
        )
    if verdict.decision == "block":
        actions.append(
            {
                "type": "use_fallback",
                "message": "I don't have reliable grounded data for that.",
            }
        )
    return actions
