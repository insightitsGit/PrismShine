"""ChorusGraph + PrismShine runtime shim (no Guard).

Proves the wired-runtime moat: ledger-aware Tier-0 pre-gen halt (R1) +
post-answer ShineGate grounding (H1). Competitors that ignore ``evidence``
cannot catch R1 failures.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel

from prismshine.bench.embed import hash_embedder
from prismshine.gate import ShineGate

os.environ.setdefault("CHORUSGRAPH_ALLOW_HASH_EMBEDDER", "1")

app = FastAPI()
gate = ShineGate.build(embedder=hash_embedder)

_compiled: Any | None = None
_stack: Any | None = None
_mode = "uninitialized"
_caps: dict[str, Any] = {}


class EvalRequest(BaseModel):
    id: str
    track: str
    question: str
    context: list[str] = []
    answer: str | None = None
    evidence: dict[str, Any] | None = None
    gold: str | None = None


EvalRequest.model_rebuild()


def _json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    skip = {"_stack", "stack"}
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k in skip or str(k).startswith("_"):
            continue
        if hasattr(v, "__dataclass_fields__") and type(v).__name__ == "ChorusStack":
            continue
        out[k] = v
    return out


def _shine_pre_node(shine_gate: ShineGate, stack: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    from prismshine.evidence.adapters.chorusgraph import bundle_from_chorusgraph

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        ledger = state.get("ledger_steps") or state.get("trace") or []
        bundle, _ = bundle_from_chorusgraph(
            state=state,
            ledger_steps=list(ledger),
            question_key="question",
            answer_key="__none__",
            stack=stack,
            chunk_ids=state.get("chunk_ids"),
            partition=state.get("partition"),
        )
        bundle = bundle.model_copy(update={"answer": None})
        ns = dict(bundle.node_state or {})
        if state.get("consumes"):
            ns["consumes"] = state["consumes"]
        if state.get("expect_trace_kinds"):
            ns["expect_trace_kinds"] = state["expect_trace_kinds"]
        if state.get("declared_sections"):
            bundle = bundle.model_copy(update={"declared_sections": list(state["declared_sections"])})
        if ns != bundle.node_state:
            bundle = bundle.model_copy(update={"node_state": ns})
        verdict = shine_gate.verify(bundle)
        halt = verdict.decision in {"block", "flag", "regenerate"}
        clean = _json_safe_state(state)
        return {
            **clean,
            "shine_verdict": verdict.model_dump(mode="json"),
            "shine_halt": halt,
            "stack_decision": "halt" if halt else "pass",
            "stack_label": "runtime_fail" if halt else "runtime_ok",
            "stack_risk": float(verdict.fused_score),
            "resolution_gate": verdict.resolution_gate,
            "tier_reached": verdict.tier_reached,
        }

    setattr(_node, "_prismshine_shine_node", True)
    return _node


def _shine_post_node(shine_gate: ShineGate, stack: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    from prismshine.integrations.chorusgraph import shine_node

    inner = shine_node(shine_gate, answer_key="answer", question_key="question")

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        state_in = {**state, "_stack": stack}
        out = inner(state_in)
        verdict = out.get("shine_verdict") or {}
        if isinstance(verdict, dict):
            decision = str(verdict.get("decision") or "pass")
            fused = float(verdict.get("fused_score") or 0.0)
            gate_name = verdict.get("resolution_gate")
            tier = verdict.get("tier_reached")
        else:
            decision = str(getattr(verdict, "decision", None) or "pass")
            fused = float(getattr(verdict, "fused_score", 0.0) or 0.0)
            gate_name = getattr(verdict, "resolution_gate", None)
            tier = getattr(verdict, "tier_reached", None)
        hallucinated = decision != "pass"
        clean = _json_safe_state(state)
        return {
            **clean,
            **{k: v for k, v in out.items() if k not in {"_stack", "stack"} and not str(k).startswith("_")},
            "stack_decision": decision,
            "stack_label": "hallucinated" if hallucinated else "grounded",
            "stack_risk": fused,
            "resolution_gate": gate_name or out.get("resolution_gate"),
            "tier_reached": tier,
        }

    setattr(_node, "_prismshine_shine_node", True)
    return _node


def _build_graph(shine_gate: ShineGate) -> tuple[Any, Any, dict[str, Any]]:
    from chorusgraph import END, START, ChorusStack, Graph
    from chorusgraph.core.node import dict_node_adapter
    from prismshine.integrations.chorusgraph import require_shine
    from prismshine.wiring import is_shine_wired

    stack = ChorusStack.defaults(
        tenant_id="chorus-shine",
        enable_memory=False,
        ledger_path=":memory:",
        sidecar_path=":memory:",
    )

    def route(state: dict[str, Any]) -> str:
        track = str(state.get("track") or "")
        if track == "R1":
            return "r1"
        return "h1"

    def router_node(state: dict[str, Any]) -> dict[str, Any]:
        return _json_safe_state(state)

    g = Graph(tenant_id="chorus-shine", graph_id="runtime-evaluate")
    g.add_node("router", dict_node_adapter(router_node, hop="router"))
    g.add_node("shine_pre", dict_node_adapter(_shine_pre_node(shine_gate, stack), hop="shine_pre"))
    g.add_node("shine_post", dict_node_adapter(_shine_post_node(shine_gate, stack), hop="shine_post"))
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route, {"r1": "shine_pre", "h1": "shine_post"})
    g.add_edge("shine_pre", END)
    g.add_edge("shine_post", END)

    compiled = g.compile(stack=stack)
    require_shine(compiled, shine_gate, prefer="both", already_has_shine_node=True)

    sidecar = stack.resolve_sidecar()
    cache = stack.resolve_cache()
    try:
        from prismshine.integrations.chorusgraph import on_fact_corrected

        on_fact_corrected(cache=cache, sidecar=sidecar, stack=stack, partition="bench", subjects=["warmup"])
        consistency: Any = True
    except Exception as exc:  # pragma: no cover
        consistency = f"error:{type(exc).__name__}"

    caps = {
        "graph": "START→router→shine_pre|shine_post→END",
        "require_shine": is_shine_wired(compiled),
        "guard": False,
        "ledger_sink": type(stack.resolve_ledger()).__name__ if hasattr(stack, "resolve_ledger") else "attached",
        "sidecar": type(sidecar).__name__,
        "cache_backend": type(cache).__name__,
        "on_fact_corrected": consistency,
        "bump_partition_version": hasattr(stack, "bump_partition_version"),
    }
    return compiled, stack, caps


def _ensure_runtime() -> None:
    global _compiled, _stack, _mode, _caps
    if _compiled is not None or _mode != "uninitialized":
        return
    try:
        _compiled, _stack, _caps = _build_graph(gate)
        _mode = "chorusgraph+prismshine"
    except ImportError as exc:
        _mode = f"degraded:import:{exc.name or type(exc).__name__}"
        _caps = {"error": str(exc)}
    except Exception as exc:
        _mode = f"degraded:init:{type(exc).__name__}"
        _caps = {"error": f"{type(exc).__name__}:{exc}"}


def _initial_state(req: EvalRequest) -> dict[str, Any]:
    evidence = dict(req.evidence or {})
    docs = evidence.pop("docs", req.context) or req.context or []
    trace = list(evidence.pop("trace", []) or [])
    node_state = dict(evidence.pop("node_state", {}) or {})
    return _json_safe_state(
        {
            "run_id": req.id,
            "session_id": req.id,
            "track": req.track,
            "question": req.question,
            "text": req.question,
            "answer": req.answer,
            "reply": req.answer,
            "docs": docs,
            "context": req.context,
            "trace": trace,
            "ledger_steps": trace,
            **node_state,
            **evidence,
        }
    )


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_runtime()
    return {
        "status": "ok" if _compiled is not None else "degraded",
        "system": "chorus-shine",
        "runtime": _mode,
        "chorus_caps": _caps,
        "grounding": "prismshine.ShineGate",
        "guard": False,
    }


@app.post("/stack_evaluate")
@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    started = time.perf_counter()
    _ensure_runtime()
    if _compiled is None:
        return {
            "id": req.id,
            "track": req.track,
            "decision": "error",
            "label": "n/a",
            "risk": 1.0,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "llm_calls": 0,
            "cost_usd": 0.0,
            "resolution_gate": "DEGRADED",
            "components": {"runtime": {"status": _mode}, "grounding": {"status": "unavailable"}},
            "saw_evidence": bool(req.evidence),
            "error": _caps.get("error"),
        }

    out = _compiled.invoke(_initial_state(req))
    if not isinstance(out, dict):
        out = dict(getattr(out, "state", None) or {})

    decision = str(out.get("stack_decision") or out.get("decision") or "pass")
    label = str(out.get("stack_label") or out.get("label") or "grounded")
    risk = float(out.get("stack_risk") if out.get("stack_risk") is not None else out.get("risk") or 0.0)
    resolution = out.get("resolution_gate")

    if req.track == "R1":
        runtime = {
            "decision": decision,
            "signatures": [
                s.get("id") if isinstance(s, dict) else getattr(s, "id", None)
                for s in (out.get("shine_verdict") or {}).get("signatures", [])
            ]
            if isinstance(out.get("shine_verdict"), dict)
            else [],
        }
        grounding = {"status": "pre_generation_only", "tier_reached": out.get("tier_reached")}
    else:
        runtime = {"status": "chorusgraph_shine_post"}
        grounding = {
            "tier_reached": out.get("tier_reached"),
            "coverage_mode": (out.get("shine_verdict") or {}).get("coverage_mode")
            if isinstance(out.get("shine_verdict"), dict)
            else None,
        }

    return {
        "id": req.id,
        "track": req.track,
        "decision": decision,
        "label": label,
        "risk": max(0.0, min(1.0, risk)),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "llm_calls": 0,
        "cost_usd": 0.0,
        "resolution_gate": str(resolution) if resolution else None,
        "components": {"guard": False, "runtime": runtime, "grounding": grounding},
        "saw_evidence": bool(req.evidence),
    }
