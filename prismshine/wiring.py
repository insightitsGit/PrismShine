"""Orchestrator-agnostic wiring — same Shine features without ChorusGraph.

Any runtime (LangGraph, custom agents, plain callables) can implement the
ChorusGraph feature set by:

1. Mapping run state → ``EvidenceBundle`` (``DictStateAdapter`` / ``bundle_from_dict``)
2. Appending JSON-safe ``trace`` steps via the helpers below
3. Calling ``pre_llm_check`` / ``post_llm_check`` (or ``wrap_llm``) around the provider
4. Calling ``enforce_verdict`` / ``shine_verify_node`` after generation
5. Marking wiring with ``mark_shine_wired`` / ``require_shine_wiring``

ChorusGraph is a convenience plugin over this contract, not a hard dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from prismshine.actions import actions_for_verdict
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.models import EvidenceBundle, ShineVerdict, TraceStep
from prismshine.runtime import (
    GateRuntimeAdapter,
    assert_adapter,
    check_adapter,
    enforce_verdict,
    pull_ledger_steps,
)

ShineAction = Literal["proceed", "halt", "reroute"]


@dataclass(frozen=True)
class ShineDecision:
    """Runtime-agnostic intercept outcome (maps to ChorusGraph InterceptDecision)."""

    action: ShineAction = "proceed"
    fallback: Any = None
    hop: str | None = None
    verdict: ShineVerdict | None = None

    @classmethod
    def proceed(cls, verdict: ShineVerdict | None = None) -> ShineDecision:
        return cls(action="proceed", verdict=verdict)

    @classmethod
    def halt(
        cls, fallback: Any = None, verdict: ShineVerdict | None = None
    ) -> ShineDecision:
        return cls(action="halt", fallback=fallback, verdict=verdict)

    @classmethod
    def reroute(cls, hop: str, verdict: ShineVerdict | None = None) -> ShineDecision:
        return cls(action="reroute", hop=hop, verdict=verdict)

    @property
    def should_halt(self) -> bool:
        return self.action == "halt"

    @property
    def should_reroute(self) -> bool:
        return self.action == "reroute"


# ---------------------------------------------------------------------------
# Trace builders (JSON-safe — safe to put in LangGraph / channel state)
# ---------------------------------------------------------------------------


def make_trace_step(
    hop: str,
    kind: str,
    *,
    status: str = "ok",
    detail: Mapping[str, Any] | None = None,
    scores: Mapping[str, float] | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe TraceStep dict for any runtime's ``state['trace']``."""
    return {
        "hop": hop,
        "kind": kind,
        "status": status,
        "detail": dict(detail or {}),
        "scores": dict(scores or {}),
        "duration_ms": duration_ms,
    }


def record_retrieval(
    hop: str,
    *,
    n_chunks: int,
    top_k: int | None = None,
    scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    status = "empty" if n_chunks <= 0 else "ok"
    detail: dict[str, Any] = {"n_chunks": n_chunks}
    if top_k is not None:
        detail["top_k"] = top_k
    return make_trace_step(hop, "retrieval", status=status, detail=detail, scores=scores)


def record_cache(
    hop: str,
    decision: str,
    *,
    must_revalidate: bool = False,
    created_at: str | None = None,
    scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {"decision": decision}
    if must_revalidate:
        detail["must_revalidate"] = True
    if created_at:
        detail["created_at"] = created_at
    return make_trace_step(hop, "cache", detail=detail, scores=scores)


def record_llm_ok(hop: str = "llm", **detail: Any) -> dict[str, Any]:
    return make_trace_step(hop, "llm", status="ok", detail=detail)


def record_llm_error(
    hop: str = "llm",
    *,
    error: str,
    status: str = "error",
) -> dict[str, Any]:
    """Map provider 5xx / auth / rate-limit into an llm TraceStep."""
    return make_trace_step(hop, "llm", status=status, detail={"error": error})


def record_llm_empty(hop: str = "llm") -> dict[str, Any]:
    return make_trace_step(hop, "llm", status="empty", detail={"empty": True})


def record_llm_refusal(
    hop: str = "llm",
    *,
    finish_reason: str = "content_filter",
) -> dict[str, Any]:
    return make_trace_step(
        hop, "llm", status="ok", detail={"finish_reason": finish_reason}
    )


def append_trace(state: dict[str, Any], step: dict[str, Any] | TraceStep) -> dict[str, Any]:
    """Return a shallow-copied state with ``step`` appended to ``trace``."""
    if isinstance(step, TraceStep):
        step_d = step.model_dump(mode="json")
    else:
        step_d = dict(step)
    steps = list(state.get("trace") or state.get("ledger_steps") or [])
    steps.append(step_d)
    return {**state, "trace": steps, "ledger_steps": steps}


# ---------------------------------------------------------------------------
# Pre / post LLM checks (any runtime)
# ---------------------------------------------------------------------------


def _jsonish(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_jsonish(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _jsonish(v) for k, v in value.items())
    return False


def bundle_from_state(
    state: Mapping[str, Any],
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    answer: Any = ...,
) -> EvidenceBundle:
    """Build a bundle from a plain state dict (docs/history/trace/consumes)."""
    data = dict(state)
    if answer is not ...:
        data[answer_key] = answer
    if "preload" not in data:
        docs = data.get("docs") or data.get("context") or []
        preload: list[Any] = []
        if isinstance(docs, str):
            docs = [docs]
        for i, d in enumerate(docs or []):
            if isinstance(d, str):
                preload.append({"chunk_id": f"d{i}", "text": d, "source": "retrieval"})
            elif isinstance(d, dict):
                preload.append(
                    {
                        "chunk_id": str(d.get("chunk_id") or d.get("id") or f"d{i}"),
                        "text": str(
                            d.get("text")
                            or d.get("page_content")
                            or d.get("content")
                            or ""
                        ),
                        "source": d.get("source") or "retrieval",
                        "vector": d.get("vector") or d.get("embedding"),
                        "metadata": dict(d.get("metadata") or {}),
                    }
                )
        for i, msg in enumerate(data.get("history") or data.get("messages") or []):
            if isinstance(msg, dict):
                text = str(msg.get("content") or msg.get("text") or "")
            else:
                text = str(getattr(msg, "content", None) or msg)
            if text:
                preload.append({"chunk_id": f"h{i}", "text": text, "source": "history"})
        for i, mem in enumerate(data.get("memory") or data.get("recalls") or []):
            if isinstance(mem, dict):
                text = str(mem.get("text") or mem.get("value") or "")
                meta = {k: v for k, v in mem.items() if k not in {"text", "value"}}
            else:
                text = str(mem)
                meta = {}
            if text:
                preload.append(
                    {
                        "chunk_id": f"m{i}",
                        "text": text,
                        "source": "memory",
                        "metadata": meta,
                    }
                )
        data["preload"] = preload or [
            {"chunk_id": "empty", "text": "(no preload)", "source": "system"}
        ]
    if not data.get("trace"):
        ledger = pull_ledger_steps(data)
        if ledger:
            data["trace"] = list(ledger)
    if "question" not in data and question_key in data:
        data["question"] = data[question_key]
    if answer_key != "answer" and answer_key in data and "answer" not in data:
        data["answer"] = data[answer_key]
    ns = dict(data.get("node_state") or {})
    for k in (
        "consumes",
        "expect_trace_kinds",
        "parallel_hops",
        "answer_source_hop",
        "missing_keys",
    ):
        if k in data and k not in ns:
            ns[k] = data[k]
    clean_ns: dict[str, Any] = {}
    for k, v in {**data, **ns}.items():
        if k in {
            "preload",
            "trace",
            "docs",
            "ledger_steps",
            "history",
            "messages",
            "memory",
            "answer",
            "reply",
            "question",
            "context",
            "recalls",
        }:
            continue
        if _jsonish(v):
            clean_ns[k] = v
    data["node_state"] = clean_ns
    data.setdefault("run_id", data.get("run_id") or "runtime")
    data.setdefault("question", data.get("question") or "(missing question)")
    b, _ = bundle_from_dict(data)
    return b


def pre_llm_check(
    gate: ShineGate,
    state: Mapping[str, Any],
    *,
    fallback: str | None = None,
    answer_key: str = "answer",
    question_key: str = "question",
) -> ShineDecision:
    """Tier-0 only (answer=None). Halt/reroute on fatal preload failures."""
    bundle = bundle_from_state(
        state, answer_key=answer_key, question_key=question_key, answer=None
    )
    verdict = gate.verify(bundle)
    if verdict.decision == "block":
        return ShineDecision.halt(
            fallback=fallback or "I don't have the data for that.",
            verdict=verdict,
        )
    if verdict.decision == "regenerate":
        hop = None
        if verdict.signatures:
            hop = verdict.signatures[0].evidence.get("hop")
        if hop:
            return ShineDecision.reroute(str(hop), verdict=verdict)
        return ShineDecision.halt(
            fallback=fallback
            or (verdict.advice[0] if verdict.advice else "Repair needed."),
            verdict=verdict,
        )
    return ShineDecision.proceed(verdict)


def post_llm_check(
    gate: ShineGate,
    state: Mapping[str, Any],
    *,
    answer: str | None = None,
    answer_key: str = "answer",
    question_key: str = "question",
    fallback: str | None = None,
) -> ShineDecision:
    """Full verify after generation."""
    st = dict(state)
    if answer is not None:
        st[answer_key] = answer
    bundle = bundle_from_state(st, answer_key=answer_key, question_key=question_key)
    verdict = gate.verify(bundle)
    if verdict.decision == "block":
        return ShineDecision.halt(
            fallback=fallback
            or st.get("shine_fallback")
            or "I don't have reliable grounded data for that.",
            verdict=verdict,
        )
    return ShineDecision.proceed(verdict)


def wrap_llm(
    model: Callable[[str, str], str],
    gate: ShineGate,
    *,
    state_factory: Callable[[], Mapping[str, Any]],
    answer_key: str = "answer",
    question_key: str = "question",
    fallback: str | None = None,
    on_decision: Callable[[ShineDecision], None] | None = None,
) -> Callable[[str, str], str]:
    """Wrap any ``(system, user) -> str`` model with pre/post Shine checks.

    Works for LangGraph nodes, custom agents, or bare callables — no ChorusGraph.
    On halt, returns ``decision.fallback`` (does not call the model if pre-halt).
    """

    def wrapped(system: str, user: str) -> str:
        state = dict(state_factory())
        state.setdefault("question", state.get(question_key) or user)
        pre = pre_llm_check(
            gate,
            state,
            fallback=fallback,
            answer_key=answer_key,
            question_key=question_key,
        )
        if on_decision:
            on_decision(pre)
        if pre.should_halt:
            return str(pre.fallback or fallback or "")
        if pre.should_reroute:
            return str(pre.fallback or f"[reroute:{pre.hop}]")
        out = model(system, user)
        state = append_trace(state, record_llm_ok("llm"))
        state[answer_key] = out
        post = post_llm_check(
            gate,
            state,
            answer=out,
            answer_key=answer_key,
            question_key=question_key,
            fallback=fallback,
        )
        if on_decision:
            on_decision(post)
        if post.should_halt and post.fallback is not None:
            return str(post.fallback)
        return out

    setattr(wrapped, "_prismshine_wrapped", True)
    return wrapped


# ---------------------------------------------------------------------------
# Verify node + wiring markers (any graph library)
# ---------------------------------------------------------------------------


def shine_verify_node(
    gate: ShineGate,
    *,
    answer_key: str = "answer",
    question_key: str = "question",
    max_regenerate: int = 1,
    pre_generation: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Dict-in/dict-out node usable from LangGraph, custom graphs, or tests."""

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        st = dict(state)
        if not st.get("trace"):
            ledger = pull_ledger_steps(st)
            if ledger:
                st["trace"] = list(ledger)
        if pre_generation:
            decision = pre_llm_check(
                gate, st, answer_key=answer_key, question_key=question_key
            )
        else:
            decision = post_llm_check(
                gate, st, answer_key=answer_key, question_key=question_key
            )
        verdict = decision.verdict
        assert verdict is not None
        out = enforce_verdict(
            verdict, st, answer_key=answer_key, max_regenerate=max_regenerate
        )
        out["shine_actions"] = actions_for_verdict(verdict)
        return out

    setattr(_node, "_prismshine_shine_node", True)
    return _node


_WIRED: dict[int, dict[str, bool]] = {}


class ShineNotWiredError(RuntimeError):
    """Raised when Shine wiring was required but not marked on a graph/app."""


def mark_shine_wired(
    target: Any,
    *,
    interceptor: bool = False,
    node: bool = False,
) -> None:
    """Mark that Shine is attached (interceptor and/or post-gen node)."""
    key = id(target) if target is not None else 0
    cur = _WIRED.get(key, {"interceptor": False, "node": False})
    if interceptor:
        cur["interceptor"] = True
    if node:
        cur["node"] = True
    _WIRED[key] = cur
    if target is not None:
        try:
            setattr(target, "_prismshine_attached", dict(cur))
        except Exception:  # noqa: BLE001
            pass


def is_shine_wired(target: Any) -> bool:
    meta = getattr(target, "_prismshine_attached", None) or _WIRED.get(id(target), {})
    return bool(meta.get("interceptor") or meta.get("node"))


def require_shine_wiring(
    target: Any,
    gate: ShineGate | None = None,
    *,
    attach_node: bool = True,
    answer_key: str = "answer",
    already_has_shine_node: bool = False,
) -> Any:
    """Fail-fast wiring check for *any* runtime (LangGraph, custom, etc.).

    Attaches a ``shine_verify_node`` factory on ``target._prismshine_node_factory``
    when ``attach_node`` is True. Does not assume ChorusGraph interceptors.
    """
    if target is None:
        raise ShineNotWiredError("target graph/app is None")
    if gate is None:
        raise ShineNotWiredError("require_shine_wiring needs a ShineGate instance")
    if already_has_shine_node:
        mark_shine_wired(target, node=True)
    if attach_node:
        factory = shine_verify_node(gate, answer_key=answer_key)
        try:
            setattr(target, "_prismshine_node_factory", factory)
        except Exception:  # noqa: BLE001
            pass
        mark_shine_wired(target, node=True)
    if not is_shine_wired(target):
        raise ShineNotWiredError(
            "PrismShine wiring check failed. Call mark_shine_wired(...) after "
            "attaching wrap_llm / shine_verify_node, or pass already_has_shine_node=True. "
            "See docs/INTEGRATION.md § BYO runtime."
        )
    return target


class DictStateAdapter(GateRuntimeAdapter):
    """RuntimeAdapter over plain dict state (LangGraph-compatible keys)."""

    def __init__(
        self,
        gate: ShineGate,
        *,
        answer_key: str = "answer",
        question_key: str = "question",
    ) -> None:
        self._question_key = question_key

        def _extract(run: Any) -> EvidenceBundle:
            state = run if isinstance(run, dict) else getattr(run, "state", None) or {}
            if not isinstance(state, dict):
                state = dict(state)
            return bundle_from_state(
                state, answer_key=answer_key, question_key=question_key
            )

        super().__init__(gate, _extract, answer_key=answer_key)
        assert_adapter(self)

    def pre_llm_hook(self, run: Any) -> ShineVerdict:
        state = run if isinstance(run, dict) else getattr(run, "state", {}) or {}
        decision = pre_llm_check(self.gate, state, answer_key=self.answer_key)
        assert decision.verdict is not None
        return decision.verdict

    def post_llm_hook(self, run: Any) -> ShineVerdict:
        state = run if isinstance(run, dict) else getattr(run, "state", {}) or {}
        decision = post_llm_check(self.gate, state, answer_key=self.answer_key)
        assert decision.verdict is not None
        return decision.verdict


def make_dict_adapter(
    gate: ShineGate, *, answer_key: str = "answer", question_key: str = "question"
) -> DictStateAdapter:
    """Factory for a RuntimeAdapter that works with any dict-shaped state."""
    return DictStateAdapter(gate, answer_key=answer_key, question_key=question_key)


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
    """Best-effort consistency: invalidate cache, mark revalidate, bump partition.

    Duck-typed — works with PrismCache, ChorusGraph sidecar, or any object
    exposing the same methods. Missing objects are no-ops.
    """
    import logging

    log = logging.getLogger(__name__)
    try:
        if cache is not None and query_vector is not None and hasattr(cache, "invalidate_where"):
            cache.invalidate_where(query_vector, tau_evict=threshold)
        if cache is not None and subjects and hasattr(cache, "invalidate_tags"):
            cache.invalidate_tags(subjects)
    except Exception as exc:  # noqa: BLE001
        log.debug("cache invalidation failed: %s", exc)
    try:
        if sidecar is not None and hasattr(sidecar, "mark_revalidate"):
            sidecar.mark_revalidate(query_vector=query_vector, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        log.debug("mark_revalidate failed: %s", exc)
    try:
        if stack is not None and partition and hasattr(stack, "bump_partition_version"):
            stack.bump_partition_version(partition)
    except Exception as exc:  # noqa: BLE001
        log.debug("bump_partition_version failed: %s", exc)


__all__ = [
    "ShineDecision",
    "ShineNotWiredError",
    "make_trace_step",
    "record_retrieval",
    "record_cache",
    "record_llm_ok",
    "record_llm_error",
    "record_llm_empty",
    "record_llm_refusal",
    "append_trace",
    "bundle_from_state",
    "pre_llm_check",
    "post_llm_check",
    "wrap_llm",
    "shine_verify_node",
    "mark_shine_wired",
    "is_shine_wired",
    "require_shine_wiring",
    "DictStateAdapter",
    "make_dict_adapter",
    "on_fact_corrected",
    "check_adapter",
]
