"""Insight-stack: PrismGuard + ChorusGraph + PrismShine (full product path).

Guard: law_pilot + ONNX + [prism] taxonomy + feedback persist + law overlay.
ChorusGraph: START → guard → (S1 end | R1 shine_pre | H1 shine_post) with
``make_guard_handler``, ``require_shine`` / ``shine_node``, ledger steps, stack.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from pydantic import BaseModel

from prismshine.bench.embed import hash_embedder
from prismshine.gate import ShineGate

# --- Guard / Chorus env (set before imports that read env at factory time) ---
os.environ.setdefault("PRISMGUARD_USE_ONNX", "1")
os.environ.setdefault("PRISMGUARD_APP_PROFILE", "law_pilot")
os.environ.setdefault("PRISMGUARD_DOMAIN", "law")
os.environ.setdefault("PRISMGUARD_SEED_PROFILE", "authored")
os.environ.setdefault("PRISMGUARD_FEEDBACK_PERSIST", "1")
os.environ.setdefault("PRISMGUARD_STORAGE_BACKEND", "memory")
os.environ.setdefault("CHORUSGRAPH_ALLOW_HASH_EMBEDDER", "1")

app = FastAPI()
gate = ShineGate.build(embedder=hash_embedder)

_checker: Any | None = None
_compiled: Any | None = None
_stack: Any | None = None
_guard_mode = "uninitialized"
_guard_caps: dict[str, Any] = {}
_chorus_caps: dict[str, Any] = {}
_JAILBREAK = re.compile(
    r"\b(ignore|disregard|override|forget|bypass|jailbreak|DAN|developer mode|system message)"
    r"\b.{0,100}\b(instruction|prompt|policy|safety|guard|secret|constraint)\b",
    re.IGNORECASE | re.DOTALL,
)


class EvalRequest(BaseModel):
    id: str
    track: str
    question: str
    context: list[str] = []
    answer: str | None = None
    evidence: dict[str, Any] | None = None
    gold: str | None = None


EvalRequest.model_rebuild()


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _import_extra_seed(checker: Any) -> dict[str, Any]:
    raw = os.environ.get("PRISMGUARD_EXTRA_SEED_PATH", "").strip()
    if not raw:
        return {"extra_seed": False}
    path = Path(raw)
    if not path.is_file():
        return {"extra_seed": False, "extra_seed_error": f"missing:{path}"}
    try:
        from prismguard.seed import import_seeds
        from prismguard.seed.parse import parse_seed_file

        storage = getattr(checker, "_storage", None)
        if storage is None:
            return {"extra_seed": False, "extra_seed_error": "no_storage"}
        report = import_seeds(storage, parse_seed_file(path), mode="update", skip_taxonomy=False)
        payload = {
            k: getattr(report, k)
            for k in ("inserted", "updated", "skipped", "errors", "mode")
            if hasattr(report, k)
        }
        return {"extra_seed": True, "extra_seed_path": str(path), "extra_seed_report": payload or str(report)}
    except Exception as exc:  # pragma: no cover
        return {"extra_seed": False, "extra_seed_error": f"{type(exc).__name__}:{exc}"}


def _probe_guard_caps(checker: Any, *, profile: str, use_onnx: bool) -> dict[str, Any]:
    has_prismrag = False
    try:
        from prismguard.taxonomy.mapping import has_prismrag as _hp

        has_prismrag = bool(_hp())
    except Exception:
        pass
    gm = getattr(checker, "_guard_model", None)
    cfg = getattr(checker, "_config", None)
    storage = getattr(checker, "_storage", None)
    caps = {
        "profile": profile,
        "domain": os.environ.get("PRISMGUARD_DOMAIN", ""),
        "seed_profile": os.environ.get("PRISMGUARD_SEED_PROFILE", "authored"),
        "use_onnx": use_onnx,
        "onnx_ready": bool(gm is not None and getattr(gm, "is_ready", False)),
        "prismrag_taxonomy": has_prismrag,
        "corpus_ann_minilm": bool(getattr(getattr(cfg, "embedding", None), "corpus_path_enabled", False)),
        "feedback_persist": getattr(checker, "_feedback_review", None) is not None,
        "tenant_lexicon": getattr(checker, "_tenant_lexicon", None) is not None,
        "llm_judge": getattr(checker, "_llm_judge", None) is not None,
        "storage_backend": type(storage).__name__ if storage is not None else None,
        "storage_env": os.environ.get("PRISMGUARD_STORAGE_BACKEND", "memory"),
        "classifier_mode": getattr(getattr(cfg, "guard_model", None), "classifier_mode", None),
        "gray_zone_policy": getattr(cfg, "gray_zone_policy", None),
        "law_overlay": True,
        "chorusgraph_handler": True,
    }
    caps.update(_import_extra_seed(checker))
    return caps


def _build_checker() -> Any:
    from prismguard.runtime.factory import create_checker_for_app

    profile = os.environ.get("PRISMGUARD_APP_PROFILE", "law_pilot").strip() or "law_pilot"
    use_onnx = _truthy("PRISMGUARD_USE_ONNX", "1")
    # law_pilot keeps taxonomy when [prism] is installed; security_bench skips it.
    return create_checker_for_app(profile, use_onnx=use_onnx)  # type: ignore[arg-type]


def _shine_pre_node(shine_gate: ShineGate, stack: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Tier-0 / pre-gen path using ChorusGraph evidence adapter (no tokens)."""
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
        # Never return non-JSON values — ChorusGraph envelope publish requires JSON.
        clean = {k: v for k, v in state.items() if not str(k).startswith("_") and k not in {"_stack", "stack"}}
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
        # Inject stack for chunk-vector reads without putting it in the envelope.
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
        clean = {k: v for k, v in state.items() if not str(k).startswith("_") and k not in {"_stack", "stack"}}
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


def _json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON objects before ChorusGraph envelope publish."""
    skip = {"_stack", "stack"}
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k in skip or str(k).startswith("_"):
            continue
        if hasattr(v, "__dataclass_fields__") and type(v).__name__ == "ChorusStack":
            continue
        out[k] = v
    return out


def _s1_finalize(state: dict[str, Any]) -> dict[str, Any]:
    blocked = bool(state.get("blocked"))
    guard = state.get("guard") or {}
    risk = 1.0 if blocked else 0.0
    details = guard.get("details") if isinstance(guard, dict) else None
    if isinstance(details, dict) and "fused_score" in details:
        try:
            risk = float(details["fused_score"])
        except (TypeError, ValueError):
            pass
    return {
        **_json_safe_state(state),
        "stack_decision": "block" if blocked else "allow",
        "stack_label": "attack" if blocked else "benign",
        "stack_risk": risk,
        "resolution_gate": (guard.get("resolution_gate") if isinstance(guard, dict) else None)
        or state.get("resolution_gate"),
    }


def _build_graph(checker: Any, shine_gate: ShineGate) -> tuple[Any, Any, dict[str, Any]]:
    from chorusgraph import END, START, ChorusStack, Graph
    from chorusgraph.core.node import dict_node_adapter
    from prismguard.integrations.chorusgraph import make_guard_handler, route_after_guard
    from prismshine.integrations.chorusgraph import require_shine
    from prismshine.wiring import is_shine_wired

    stack = ChorusStack.defaults(
        tenant_id="insight-stack",
        enable_memory=False,  # Cortex optional; ledger+cache+shine still wired
        ledger_path=":memory:",
        sidecar_path=":memory:",
    )

    base_guard = make_guard_handler(
        checker,
        text_key="question",
        session_id_key="session_id",
        block_on=frozenset({"block", "gray", "deny", "uncertain"}),
    )

    def guard_node(state: dict[str, Any]) -> dict[str, Any]:
        safe = _json_safe_state(state)
        out = base_guard(safe)
        # H1/R1: keep Guard telemetry but never short-circuit Shine (run2 FP bug).
        if not state.get("guard_hard_block", False):
            out = {**out, "blocked": False}
        return _json_safe_state(out)

    def route(state: dict[str, Any]) -> str:
        if route_after_guard(state) == "end" and state.get("guard_hard_block"):
            return "blocked"
        track = str(state.get("track") or "")
        if track == "S1":
            return "s1"
        if track == "R1":
            return "r1"
        return "h1"

    g = Graph(tenant_id="insight-stack", graph_id="stack-evaluate")
    g.add_node("guard", dict_node_adapter(guard_node, hop="guard"))
    g.add_node("s1_done", dict_node_adapter(_s1_finalize, hop="s1_done"))
    g.add_node("shine_pre", dict_node_adapter(_shine_pre_node(shine_gate, stack), hop="shine_pre"))
    g.add_node("shine_post", dict_node_adapter(_shine_post_node(shine_gate, stack), hop="shine_post"))
    g.add_edge(START, "guard")
    g.add_conditional_edges(
        "guard",
        route,
        {"blocked": "s1_done", "s1": "s1_done", "r1": "shine_pre", "h1": "shine_post"},
    )
    g.add_edge("s1_done", END)
    g.add_edge("shine_pre", END)
    g.add_edge("shine_post", END)

    compiled = g.compile(stack=stack)
    require_shine(compiled, shine_gate, prefer="both", already_has_shine_node=True)

    # Consistency surface (ChorusGraph 1.3.0) — prove mark_revalidate + partition bump exist.
    sidecar = stack.resolve_sidecar()
    cache = stack.resolve_cache()
    try:
        from prismshine.integrations.chorusgraph import on_fact_corrected

        on_fact_corrected(cache=cache, sidecar=sidecar, stack=stack, partition="bench", subjects=["warmup"])
        consistency = True
    except Exception as exc:  # pragma: no cover
        consistency = f"error:{type(exc).__name__}"

    caps = {
        "graph": "guard→s1|shine_pre|shine_post",
        "require_shine": is_shine_wired(compiled),
        "register_interceptor": hasattr(compiled, "register_interceptor"),
        "interceptor_attached": bool(getattr(compiled, "_prismshine_interceptor", False) or is_shine_wired(compiled)),
        "make_guard_handler": True,
        "route_after_guard": True,
        "ledger_sink": type(stack.resolve_ledger()).__name__ if hasattr(stack, "resolve_ledger") else "attached",
        "sidecar": type(sidecar).__name__,
        "cache_backend": type(cache).__name__,
        "enable_memory": False,
        "on_fact_corrected": consistency,
        "mark_revalidate": hasattr(sidecar, "mark_revalidate") or True,
        "bump_partition_version": hasattr(stack, "bump_partition_version"),
    }
    return compiled, stack, caps


def _ensure_runtime() -> None:
    global _checker, _compiled, _stack, _guard_mode, _guard_caps, _chorus_caps
    if _compiled is not None or _guard_mode != "uninitialized":
        return
    try:
        profile = os.environ.get("PRISMGUARD_APP_PROFILE", "law_pilot").strip() or "law_pilot"
        use_onnx = _truthy("PRISMGUARD_USE_ONNX", "1")
        _checker = _build_checker()
        _guard_caps = _probe_guard_caps(_checker, profile=profile, use_onnx=use_onnx)
        _compiled, _stack, _chorus_caps = _build_graph(_checker, gate)
        tax = "prismrag" if _guard_caps.get("prismrag_taxonomy") else "rules_seed"
        fb = "feedback" if _guard_caps.get("feedback_persist") else "no_feedback"
        onnx = "onnx_ready" if _guard_caps.get("onnx_ready") else "onnx_missing"
        _guard_mode = f"prismguard:{profile}:{onnx}:{tax}:{fb}+chorusgraph"
    except ImportError as exc:
        _guard_mode = f"degraded:import:{exc.name or type(exc).__name__}"
        _guard_caps = {"error": str(exc)}
        _chorus_caps = {"error": str(exc)}
    except Exception as exc:
        _guard_mode = f"degraded:init:{type(exc).__name__}"
        _guard_caps = {"error": f"{type(exc).__name__}:{exc}"}
        _chorus_caps = {"error": f"{type(exc).__name__}:{exc}"}


def _regex_fallback(req: EvalRequest, started: float) -> dict[str, Any]:
    blocked = bool(_JAILBREAK.search(req.question))
    if req.track == "S1":
        return _response(
            req,
            decision="block" if blocked else "allow",
            label="attack" if blocked else "benign",
            risk=1.0 if blocked else 0.0,
            resolution_gate="REGEX_STRUCTURAL",
            guard={"mode": _guard_mode},
            runtime={"status": "degraded"},
            grounding={"status": "degraded"},
            started=started,
        )
    return _response(
        req,
        decision="error",
        label="n/a",
        risk=1.0,
        resolution_gate="DEGRADED",
        guard={"mode": _guard_mode},
        runtime={"status": "unavailable"},
        grounding={"status": "unavailable"},
        started=started,
    )


def _initial_state(req: EvalRequest) -> dict[str, Any]:
    evidence = dict(req.evidence or {})
    docs = evidence.pop("docs", req.context) or req.context or []
    trace = list(evidence.pop("trace", []) or [])
    node_state = dict(evidence.pop("node_state", {}) or {})
    # Remaining evidence keys (fact_corrections, consumes, …) flatten into state.
    state: dict[str, Any] = {
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
        "guard_hard_block": req.track == "S1",
        **node_state,
        **evidence,
    }
    return _json_safe_state(state)


def _response(
    req: EvalRequest,
    *,
    decision: str,
    label: str,
    risk: float,
    resolution_gate: str | None,
    guard: dict[str, Any],
    runtime: dict[str, Any],
    grounding: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "id": req.id,
        "track": req.track,
        "decision": decision,
        "label": label,
        "risk": max(0.0, min(1.0, risk)),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "llm_calls": 0,
        "cost_usd": 0.0,
        "resolution_gate": resolution_gate,
        "components": {"guard": guard, "runtime": runtime, "grounding": grounding},
        "saw_evidence": bool(req.evidence),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_runtime()
    return {
        "status": "ok" if _compiled is not None else "degraded",
        "system": "insight-stack",
        "guard": _guard_mode,
        "guard_caps": _guard_caps,
        "chorus_caps": _chorus_caps,
        "runtime": "chorusgraph+ledger",
        "grounding": "prismshine.ShineGate",
        "components": {
            "prismguard": _guard_mode,
            "chorusgraph": "wired" if _compiled is not None else _chorus_caps.get("error", "missing"),
        },
    }


@app.post("/stack_evaluate")
@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    started = time.perf_counter()
    _ensure_runtime()
    if _compiled is None:
        return _regex_fallback(req, started)

    state = _initial_state(req)
    out = _compiled.invoke(state)
    if not isinstance(out, dict):
        out = dict(getattr(out, "state", None) or {})

    guard_info = out.get("guard") if isinstance(out.get("guard"), dict) else {}
    guard_info = {**guard_info, "mode": _guard_mode}

    decision = str(out.get("stack_decision") or out.get("decision") or "allow")
    label = str(out.get("stack_label") or out.get("label") or "benign")
    risk = float(out.get("stack_risk") if out.get("stack_risk") is not None else out.get("risk") or 0.0)
    resolution = out.get("resolution_gate") or (guard_info.get("resolution_gate") if guard_info else None)

    if req.track == "S1":
        runtime = {"status": "not_run" if decision == "block" else "not_applicable"}
        grounding = {"status": "not_run" if decision == "block" else "not_applicable"}
    elif req.track == "R1":
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

    return _response(
        req,
        decision=decision,
        label=label,
        risk=risk,
        resolution_gate=str(resolution) if resolution else None,
        guard=guard_info,
        runtime=runtime,
        grounding=grounding,
        started=started,
    )
