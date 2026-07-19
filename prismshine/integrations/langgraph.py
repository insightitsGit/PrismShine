"""LangGraph plugin: same Shine features as ChorusGraph, via generic wiring."""

from __future__ import annotations

from typing import Any, Callable, Literal

from prismshine.evidence.adapters.langgraph import bundle_from_langgraph
from prismshine.gate import ShineGate
from prismshine.models import EvidenceBundle
from prismshine.regen import next_route
from prismshine.runtime import GateRuntimeAdapter, assert_adapter, pull_ledger_steps
from prismshine.wiring import (
    ShineDecision,
    ShineNotWiredError,
    append_trace,
    is_shine_wired,
    mark_shine_wired,
    post_llm_check,
    pre_llm_check,
    record_llm_empty,
    record_llm_error,
    require_shine_wiring,
    shine_verify_node,
    wrap_llm,
)

Route = Literal["pass", "flag", "block", "regenerate"]


def shine_langgraph_node(
    gate: ShineGate,
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    pre_generation: bool = False,
    max_regenerate: int = 1,
    compiled: Any | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Post- (or pre-) generation node — same contract as ChorusGraph ``shine_node``."""
    node = shine_verify_node(
        gate,
        answer_key=answer_key,
        question_key=question_key,
        max_regenerate=max_regenerate,
        pre_generation=pre_generation,
    )
    if compiled is not None:
        mark_shine_wired(compiled, node=True)
    return node


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


def require_shine(
    compiled: Any,
    gate: ShineGate | None = None,
    *,
    answer_key: str = "answer",
    already_has_shine_node: bool = False,
) -> Any:
    """Fail-fast wiring for LangGraph / any dict-state graph (no interceptors)."""
    return require_shine_wiring(
        compiled,
        gate,
        answer_key=answer_key,
        already_has_shine_node=already_has_shine_node,
    )


def shine_before_hook(
    gate: ShineGate,
    *,
    fallback: str | None = None,
    answer_key: str = "answer",
    question_key: str = "question",
) -> Callable[[dict[str, Any]], ShineDecision]:
    """Pre-generation check — call before your LLM node."""

    def _before(state: dict[str, Any]) -> ShineDecision:
        return pre_llm_check(
            gate,
            state,
            fallback=fallback,
            answer_key=answer_key,
            question_key=question_key,
        )

    return _before


def shine_after_hook(
    gate: ShineGate,
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    fallback: str | None = None,
) -> Callable[[dict[str, Any]], ShineDecision]:
    """Post-generation check — call after your LLM node."""

    def _after(state: dict[str, Any]) -> ShineDecision:
        # Allow mapping provider failures already placed on state
        st = dict(state)
        if st.get("llm_error") or st.get("provider_error"):
            st = append_trace(
                st,
                record_llm_error(
                    str(st.get("hop") or "llm"),
                    error=str(st.get("llm_error") or st.get("provider_error")),
                ),
            )
        if st.get("empty_completion"):
            st = append_trace(st, record_llm_empty(str(st.get("hop") or "llm")))
        return post_llm_check(
            gate,
            st,
            answer_key=answer_key,
            question_key=question_key,
            fallback=fallback,
        )

    return _after


def wrap_langgraph_llm(
    model: Callable[[str, str], str],
    gate: ShineGate,
    *,
    state_factory: Callable[[], dict[str, Any]],
    answer_key: str = "answer",
    question_key: str = "question",
    fallback: str | None = None,
) -> Callable[[str, str], str]:
    """Provider-boundary wrap — LangGraph equivalent of ChorusGraph ``ctx.call_llm``."""
    return wrap_llm(
        model,
        gate,
        state_factory=state_factory,
        answer_key=answer_key,
        question_key=question_key,
        fallback=fallback,
    )


class LangGraphAdapter(GateRuntimeAdapter):
    def __init__(
        self, gate: ShineGate, *, answer_key: str = "answer", question_key: str = "question"
    ) -> None:
        from prismshine.wiring import bundle_from_state

        def _extract(run: Any) -> EvidenceBundle:
            state = run if isinstance(run, dict) else getattr(run, "state", None) or {}
            if not isinstance(state, dict):
                state = dict(state)
            if not state.get("trace"):
                ledger = pull_ledger_steps(run) or pull_ledger_steps(state)
                if ledger:
                    state = {**state, "trace": list(ledger)}
            # Prefer generic state mapper (history/memory/TRACE_INCOMPLETE keys)
            try:
                return bundle_from_state(
                    state, answer_key=answer_key, question_key=question_key
                )
            except Exception:  # noqa: BLE001
                bundle, _ = bundle_from_langgraph(state, answer_key=answer_key)
                return bundle

        super().__init__(gate, _extract, answer_key=answer_key)
        assert_adapter(self)


__all__ = [
    "shine_langgraph_node",
    "shine_route",
    "route_map",
    "require_shine",
    "shine_before_hook",
    "shine_after_hook",
    "wrap_langgraph_llm",
    "LangGraphAdapter",
    "ShineNotWiredError",
    "is_shine_wired",
    "next_route",
    "wrap_llm",
]
