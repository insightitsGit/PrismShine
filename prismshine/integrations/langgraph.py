"""LangGraph plugin: node factory + conditional-edge router."""

from __future__ import annotations

from typing import Any, Callable, Literal

from prismshine.evidence.adapters.langgraph import bundle_from_langgraph
from prismshine.gate import ShineGate
from prismshine.regen import build_repair_feedback, next_route

Route = Literal["pass", "flag", "block", "regenerate"]


def shine_langgraph_node(
    gate: ShineGate,
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    pre_generation: bool = False,
    max_regenerate: int = 1,
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
        route = next_route(verdict.decision, attempts, max_attempts=max_regenerate)
        out: dict[str, Any] = {
            "shine_verdict": verdict.model_dump(mode="json"),
            "shine_feedback": feedback,
            "shine_route": route,
            "shine_advice": list(verdict.advice),
        }
        if route == "block":
            out[answer_key] = state.get("shine_fallback") or (
                "I don't have reliable grounded data for that."
            )
        elif route == "regenerate":
            repair = build_repair_feedback(
                spans=verdict.spans,
                advice=list(verdict.advice),
                signatures=[s.id for s in verdict.signatures],
            )
            out["_shine_regen_attempts"] = attempts + 1
            out["shine_repair_feedback"] = repair
            out["shine_repair_prompt"] = repair["prompt_suffix"]
        elif verdict.decision == "regenerate":
            out["shine_route"] = "flag"
            out["shine_regen_exhausted"] = True
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
