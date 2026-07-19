"""LangGraph plugin: node factory + conditional-edge router."""

from __future__ import annotations

from typing import Any, Callable, Literal

from prismshine.evidence.adapters.langgraph import bundle_from_langgraph
from prismshine.gate import ShineGate

Route = Literal["pass", "flag", "block", "regenerate"]


def shine_langgraph_node(
    gate: ShineGate,
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    pre_generation: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        bundle, feedback = bundle_from_langgraph(
            state,
            question_key=question_key,
            answer_key=answer_key,
        )
        if pre_generation:
            bundle = bundle.model_copy(update={"answer": None})
        verdict = gate.verify(bundle)
        attempts = int(state.get("_shine_regen_attempts") or 0)
        route: Route = verdict.decision  # type: ignore[assignment]
        out: dict[str, Any] = {
            "shine_verdict": verdict.model_dump(mode="json"),
            "shine_feedback": feedback,
            "shine_route": route,
            "shine_advice": list(verdict.advice),
        }
        if verdict.decision == "block":
            out[answer_key] = state.get("shine_fallback") or (
                "I don't have reliable grounded data for that."
            )
        elif verdict.decision == "regenerate":
            if attempts < 1:
                out["_shine_regen_attempts"] = attempts + 1
                out["shine_repair_feedback"] = {
                    "spans": [s.model_dump() for s in verdict.spans],
                    "advice": verdict.advice,
                }
            else:
                out["shine_route"] = "flag"
        return out

    return _node


def shine_route(state: dict[str, Any]) -> str:
    """Conditional-edge router for pass/flag/block/regenerate."""
    return str(state.get("shine_route") or "pass")


def route_map(
    *,
    pass_to: str = "end",
    flag_to: str = "end",
    block_to: str = "end",
    regenerate_to: str = "generate",
) -> dict[str, str]:
    return {
        "pass": pass_to,
        "flag": flag_to,
        "block": block_to,
        "regenerate": regenerate_to,
    }
