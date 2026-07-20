"""ChorusGraph plugin: shine_node, interceptors, require_shine, consistency hooks."""

from __future__ import annotations

import logging
from typing import Any, Callable

from prismshine.evidence.adapters.chorusgraph import bundle_from_chorusgraph
from prismshine.gate import ShineGate
from prismshine.models import EvidenceBundle, ShineVerdict
from prismshine.runtime import GateRuntimeAdapter, assert_adapter, enforce_verdict, pull_ledger_steps
from prismshine.wiring import (
    ShineDecision,
    ShineNotWiredError,
    is_shine_wired,
    mark_shine_wired,
    on_fact_corrected as _on_fact_corrected_generic,
)

logger = logging.getLogger(__name__)


def _intercept_decision():
    try:
        from chorusgraph import InterceptDecision  # type: ignore

        return InterceptDecision
    except Exception:  # noqa: BLE001
        try:
            from chorusgraph.core.intercept import InterceptDecision  # type: ignore

            return InterceptDecision
        except Exception:  # noqa: BLE001
            return None


def _mark_attached(compiled: Any, *, interceptor: bool = False, node: bool = False) -> None:
    mark_shine_wired(compiled, interceptor=interceptor, node=node)


def _to_intercept(decision: ShineDecision, Decision: Any) -> Any:
    if Decision is None:
        return {
            "action": decision.action,
            "fallback": decision.fallback,
            "hop": decision.hop,
            "shine_verdict": decision.verdict.model_dump(mode="json")
            if decision.verdict
            else None,
        }
    if decision.action == "halt":
        return Decision.halt(fallback=decision.fallback)
    if decision.action == "reroute" and decision.hop:
        return Decision.reroute(decision.hop)
    return Decision.proceed()


def require_shine(
    compiled: Any,
    gate: ShineGate | None = None,
    *,
    prefer: str = "both",
    answer_key: str = "reply",
    fallback: str | None = None,
    already_has_shine_node: bool = False,
) -> Any:
    """Fail-fast wiring helper (P0).

    Ensures at least one of:
      - LLM interceptors (pre-gen halt via ctx.call_llm)
      - shine_node path (guaranteed post-gen)

    ``prefer``: ``interceptor`` | ``node`` | ``both`` (default attaches interceptor
    when available and marks node factory on ``compiled._prismshine_node_factory``).
    Set ``already_has_shine_node=True`` if you already ``add_node(shine_node(...))``.
    """
    if compiled is None:
        raise ShineNotWiredError("compiled graph is None")
    if gate is None:
        raise ShineNotWiredError("require_shine needs a ShineGate instance")

    want_interceptor = prefer in {"interceptor", "both"}
    want_node = prefer in {"node", "both"}

    if already_has_shine_node:
        _mark_attached(compiled, node=True)

    if want_interceptor:
        if hasattr(compiled, "register_interceptor"):
            attach_interceptors(
                compiled, gate, answer_key=answer_key, fallback=fallback
            )
        elif prefer == "interceptor":
            raise ShineNotWiredError(
                "compiled has no register_interceptor; need chorusgraph>=1.3.0 "
                "or prefer='node' with shine_node"
            )

    if want_node:
        factory = shine_node(gate, answer_key=answer_key, compiled=compiled)
        try:
            setattr(compiled, "_prismshine_node_factory", factory)
        except Exception:  # noqa: BLE001
            pass
        _mark_attached(compiled, node=True)

    if not is_shine_wired(compiled):
        raise ShineNotWiredError(
            "PrismShine wiring check failed. Use require_shine(compiled, gate) or "
            "attach_interceptors() / add_node('shine', shine_node(gate)). "
            "Raw provider SDK calls bypass interceptors — shine_node is the guaranteed path."
        )
    return compiled


def shine_node(
    gate: ShineGate,
    *,
    answer_key: str = "reply",
    question_key: str = "question",
    regenerate_target: str | None = None,
    max_regenerate: int = 1,
    compiled: Any | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Post-generation node: verify, write state+ledger fields, route by decision."""

    if compiled is not None:
        _mark_attached(compiled, node=True)

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        ledger = (
            pull_ledger_steps(state)
            or state.get("_ledger_steps")
            or state.get("ledger_steps")
            or []
        )
        stack = state.get("_stack") or state.get("stack")
        bundle, feedback = bundle_from_chorusgraph(
            state=state,
            ledger_steps=ledger,
            question_key=question_key,
            answer_key=answer_key,
            stack=stack,
            chunk_ids=state.get("chunk_ids"),
            partition=state.get("partition"),
        )
        # Propagate consumes for TRACE_INCOMPLETE
        if state.get("consumes") and "consumes" not in bundle.node_state:
            bundle = bundle.model_copy(
                update={"node_state": {**bundle.node_state, "consumes": state["consumes"]}}
            )
        verdict = gate.verify(bundle)
        out = enforce_verdict(
            verdict, state, answer_key=answer_key, max_regenerate=max_regenerate
        )
        out["shine_feedback"] = feedback
        if out.get("shine_route") == "regenerate" and regenerate_target:
            fb = out.get("shine_repair_feedback") or {}
            fb["target"] = regenerate_target
            out["shine_repair_feedback"] = fb
        return out

    # Mark factory as shine node for require_shine introspection
    setattr(_node, "_prismshine_shine_node", True)
    return _node


def _state_from_ctx(ctx: Any, state: Any | None = None) -> dict[str, Any]:
    if isinstance(state, dict):
        return dict(state)
    if state is not None and not isinstance(state, dict):
        try:
            return dict(state)
        except Exception:  # noqa: BLE001
            pass
    if ctx is not None and hasattr(ctx, "read"):
        try:
            view = ctx.read()
            if isinstance(view, dict):
                return dict(view)
        except Exception:  # noqa: BLE001
            pass
    raw = getattr(ctx, "state", None)
    return dict(raw) if isinstance(raw, dict) else {}


def shine_before_hook(gate: ShineGate, *, fallback: str | None = None) -> Callable[..., Any]:
    """Pre-generation interceptor: Tier-0 only (answer=None).

    ChorusGraph ADR-008 calls ``before_llm(ctx, state)``. Decision semantics
    match ``prismshine.wiring.pre_llm_check`` / ``ShineDecision``.
    """

    Decision = _intercept_decision()

    def _before(ctx: Any = None, state: Any = None, **kwargs: Any) -> Any:
        state_map = _state_from_ctx(ctx, state if state is not None else kwargs.get("state"))
        ledger = (
            pull_ledger_steps(ctx)
            or pull_ledger_steps(state_map)
            or getattr(ctx, "ledger_steps", None)
            or state_map.get("ledger_steps")
            or []
        )
        bundle, _ = bundle_from_chorusgraph(
            state=state_map,
            ledger_steps=list(ledger) if ledger else [],
            answer_key="__none__",
        )
        bundle = bundle.model_copy(update={"answer": None})
        verdict = gate.verify(bundle)
        if verdict.decision == "block":
            sd = ShineDecision.halt(
                fallback=fallback or "I don't have the data for that.",
                verdict=verdict,
            )
        elif verdict.decision == "regenerate":
            hop = None
            if verdict.signatures:
                hop = verdict.signatures[0].evidence.get("hop")
            if hop:
                sd = ShineDecision.reroute(str(hop), verdict=verdict)
            else:
                sd = ShineDecision.halt(
                    fallback=fallback
                    or (verdict.advice[0] if verdict.advice else "Repair needed."),
                    verdict=verdict,
                )
        else:
            sd = ShineDecision.proceed(verdict)
        return _to_intercept(sd, Decision)

    return _before


def shine_after_hook(gate: ShineGate, *, answer_key: str = "reply") -> Callable[..., Any]:
    """Post-generation interceptor: full verify.

    ChorusGraph ADR-008 calls ``after_llm(ctx, state, output)``.
    Same ``ShineDecision`` contract as ``prismshine.wiring.post_llm_check``.
    """

    Decision = _intercept_decision()

    def _after(ctx: Any = None, state: Any = None, output: Any = None, **kwargs: Any) -> Any:
        from prismshine.wiring import append_trace, record_llm_empty, record_llm_error

        state_map = _state_from_ctx(ctx, state if state is not None else kwargs.get("state"))
        response = output if output is not None else kwargs.get("response") or kwargs.get("answer")
        if response is not None:
            state_map = {**state_map, answer_key: response}
        if kwargs.get("llm_error") or kwargs.get("provider_error"):
            err = kwargs.get("llm_error") or kwargs.get("provider_error")
            state_map = append_trace(
                state_map,
                record_llm_error(str(kwargs.get("hop") or "llm"), error=str(err)),
            )
        if kwargs.get("empty_completion"):
            state_map = append_trace(
                state_map, record_llm_empty(str(kwargs.get("hop") or "llm"))
            )

        ledger = (
            pull_ledger_steps(ctx)
            or pull_ledger_steps(state_map)
            or state_map.get("ledger_steps")
            or state_map.get("trace")
            or []
        )
        bundle, _ = bundle_from_chorusgraph(
            state=state_map,
            ledger_steps=list(ledger) if ledger else [],
            answer_key=answer_key,
        )
        verdict = gate.verify(bundle)
        if verdict.decision == "block":
            sd = ShineDecision.halt(
                fallback=state_map.get("shine_fallback")
                or "I don't have reliable grounded data for that.",
                verdict=verdict,
            )
        else:
            sd = ShineDecision.proceed(verdict)
        return _to_intercept(sd, Decision)

    return _after


def attach_interceptors(compiled: Any, gate: ShineGate, **kwargs: Any) -> None:
    """Register before/after LLM hooks on a compiled ChorusGraph."""
    if not hasattr(compiled, "register_interceptor"):
        raise RuntimeError(
            "compiled graph has no register_interceptor; require chorusgraph>=1.3.0"
        )
    compiled.register_interceptor(
        before_llm=shine_before_hook(gate, fallback=kwargs.get("fallback")),
        after_llm=shine_after_hook(gate, answer_key=kwargs.get("answer_key", "reply")),
    )
    _mark_attached(compiled, interceptor=True)


def on_fact_corrected(
    *,
    cache: Any | None = None,
    sidecar: Any | None = None,
    stack: Any | None = None,
    partition: str | None = None,
    query_vector: list[float] | None = None,
    threshold: float = 0.55,
    subjects: list[str] | None = None,
) -> None:
    """Consistency hook — delegates to generic ``wiring.on_fact_corrected``.

    Also tries ChorusGraph's module-level ``mark_revalidate`` when sidecar has
    no method of its own.
    """
    if sidecar is not None and not hasattr(sidecar, "mark_revalidate"):
        try:
            from chorusgraph import mark_revalidate  # type: ignore

            class _SidecarProxy:
                def mark_revalidate(self, **kwargs: Any) -> Any:
                    return mark_revalidate(sidecar, **kwargs)

            sidecar = _SidecarProxy()
        except Exception:  # noqa: BLE001
            pass
    _on_fact_corrected_generic(
        cache=cache,
        sidecar=sidecar,
        stack=stack,
        partition=partition,
        query_vector=query_vector,
        threshold=threshold,
        subjects=subjects,
    )


class ChorusGraphAdapter(GateRuntimeAdapter):
    """RuntimeAdapter for ChorusGraph runs/state dicts."""

    def __init__(self, gate: ShineGate, *, answer_key: str = "reply") -> None:
        def _extract(run: Any) -> EvidenceBundle:
            state = run if isinstance(run, dict) else getattr(run, "state", None) or {}
            if not isinstance(state, dict):
                state = dict(state)
            ledger = pull_ledger_steps(run) or pull_ledger_steps(state)
            stack = state.get("_stack") or state.get("stack") or getattr(run, "stack", None)
            bundle, _ = bundle_from_chorusgraph(
                state=state,
                ledger_steps=ledger,
                answer_key=answer_key,
                stack=stack,
                chunk_ids=state.get("chunk_ids"),
                partition=state.get("partition"),
            )
            return bundle

        super().__init__(gate, _extract, answer_key=answer_key)
        assert_adapter(self)


__all__ = [
    "shine_node",
    "shine_before_hook",
    "shine_after_hook",
    "attach_interceptors",
    "require_shine",
    "is_shine_wired",
    "ShineNotWiredError",
    "on_fact_corrected",
    "ChorusGraphAdapter",
    "ShineVerdict",
]
