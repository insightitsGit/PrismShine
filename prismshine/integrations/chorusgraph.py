"""ChorusGraph plugin: shine_node, interceptors, consistency hooks."""

from __future__ import annotations

import logging
from typing import Any, Callable

from prismshine.evidence.adapters.chorusgraph import bundle_from_chorusgraph
from prismshine.gate import ShineGate
from prismshine.models import ShineVerdict

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


def shine_node(
    gate: ShineGate,
    *,
    answer_key: str = "reply",
    question_key: str = "question",
    regenerate_target: str | None = None,
    max_regenerate: int = 1,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Post-generation node: verify, write state+ledger fields, route by decision.

    Guaranteed path — works even when nodes bypass ctx.call_llm.
    Use ctx.call_llm in generator nodes to also enable interceptor pre-gen halt.
    """

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        ledger = state.get("_ledger_steps") or state.get("ledger_steps") or []
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
        verdict = gate.verify(bundle)
        attempts = int(state.get("_shine_regen_attempts") or 0)
        out = {
            "shine_verdict": verdict.model_dump(mode="json"),
            "shine_feedback": feedback,
            "shine_advice": list(verdict.advice),
            "_ledger_append": {
                "kind": "shine.verdict",
                "detail": {
                    "decision": verdict.decision,
                    "resolution_gate": verdict.resolution_gate,
                    "fused_score": verdict.fused_score,
                    "evidence_hash": verdict.evidence_hash,
                    "signatures": [s.id for s in verdict.signatures],
                },
            },
        }
        if verdict.decision == "block":
            out[answer_key] = state.get("shine_fallback") or (
                "I don't have reliable grounded data for that."
            )
            out["shine_route"] = "block"
        elif verdict.decision == "regenerate" and attempts < max_regenerate:
            out["_shine_regen_attempts"] = attempts + 1
            out["shine_route"] = "regenerate"
            out["shine_repair_feedback"] = {
                "spans": [s.model_dump() for s in verdict.spans],
                "advice": verdict.advice,
                "target": regenerate_target or "generate",
            }
        elif verdict.decision == "regenerate":
            out["shine_route"] = "flag"
            out[answer_key] = state.get(answer_key)
        else:
            out["shine_route"] = verdict.decision
        return out

    return _node


def shine_before_hook(gate: ShineGate, *, fallback: str | None = None) -> Callable[..., Any]:
    """Pre-generation interceptor: Tier-0 only (answer=None)."""

    Decision = _intercept_decision()

    def _before(ctx: Any = None, **kwargs: Any) -> Any:
        state = getattr(ctx, "state", None) or kwargs.get("state") or {}
        if not isinstance(state, dict):
            state = dict(state) if state else {}
        ledger = getattr(ctx, "ledger_steps", None) or state.get("ledger_steps") or []
        bundle, _ = bundle_from_chorusgraph(
            state=state,
            ledger_steps=list(ledger) if ledger else [],
            answer_key="__none__",
        )
        # force pre-gen
        bundle = bundle.model_copy(update={"answer": None})
        verdict = gate.verify(bundle)
        if Decision is None:
            return {"shine_verdict": verdict.model_dump(mode="json"), "proceed": verdict.decision == "pass"}
        if verdict.decision == "block":
            return Decision.halt(
                fallback=fallback
                or "I don't have the data for that."
            )
        if verdict.decision == "regenerate":
            hop = None
            if verdict.signatures:
                hop = verdict.signatures[0].evidence.get("hop")
            if hop and hasattr(Decision, "reroute"):
                return Decision.reroute(hop)
            return Decision.halt(fallback=fallback or verdict.advice[:1] and verdict.advice[0])
        return Decision.proceed()

    return _before


def shine_after_hook(gate: ShineGate, *, answer_key: str = "reply") -> Callable[..., Any]:
    """Post-generation interceptor: full verify; Tier-0 reused via evidence hash."""

    Decision = _intercept_decision()

    def _after(ctx: Any = None, **kwargs: Any) -> Any:
        state = getattr(ctx, "state", None) or kwargs.get("state") or {}
        if not isinstance(state, dict):
            state = dict(state) if state else {}
        # inject LLM output if provided
        response = kwargs.get("response") or kwargs.get("answer")
        if response is not None:
            state = {**state, answer_key: response}
        ledger = getattr(ctx, "ledger_steps", None) or state.get("ledger_steps") or []
        bundle, _ = bundle_from_chorusgraph(
            state=state,
            ledger_steps=list(ledger) if ledger else [],
            answer_key=answer_key,
        )
        verdict = gate.verify(bundle)
        if Decision is None:
            return {"shine_verdict": verdict.model_dump(mode="json")}
        if verdict.decision == "block":
            return Decision.halt(
                fallback=state.get("shine_fallback")
                or "I don't have reliable grounded data for that."
            )
        return Decision.proceed()

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
    """Best-effort consistency: invalidate cache, mark revalidate, bump partition."""
    try:
        if cache is not None and query_vector is not None and hasattr(cache, "invalidate_where"):
            cache.invalidate_where(query_vector, tau_evict=threshold)
        if cache is not None and subjects and hasattr(cache, "invalidate_tags"):
            cache.invalidate_tags(subjects)
    except Exception as exc:  # noqa: BLE001
        logger.debug("cache invalidation failed: %s", exc)
    try:
        if sidecar is not None:
            try:
                from chorusgraph import mark_revalidate  # type: ignore

                mark_revalidate(sidecar, query_vector=query_vector, threshold=threshold)
            except Exception:  # noqa: BLE001
                if hasattr(sidecar, "mark_revalidate"):
                    sidecar.mark_revalidate(query_vector=query_vector, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        logger.debug("mark_revalidate failed: %s", exc)
    try:
        if stack is not None and partition and hasattr(stack, "bump_partition_version"):
            stack.bump_partition_version(partition)
    except Exception as exc:  # noqa: BLE001
        logger.debug("bump_partition_version failed: %s", exc)


__all__ = [
    "shine_node",
    "shine_before_hook",
    "shine_after_hook",
    "attach_interceptors",
    "on_fact_corrected",
    "ShineVerdict",
]
