"""Insight-stack Stack-suite shim: PrismGuard -> PrismShine ledger-aware gate."""

from __future__ import annotations

import re
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from prismshine.bench.embed import hash_embedder
from prismshine.gate import ShineGate
from prismshine.wiring import post_llm_check, pre_llm_check

app = FastAPI()
gate = ShineGate.build(embedder=hash_embedder)
_checker: Any | None = None
_guard_mode = "uninitialized"
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
    gold: str | None = None  # accepted but never used for a decision


def _guard() -> Any | None:
    global _checker, _guard_mode
    if _checker is not None or _guard_mode != "uninitialized":
        return _checker
    try:
        from prismguard.runtime.factory import create_checker_rules_only

        _checker = create_checker_rules_only()
        _guard_mode = "prismguard:rules_only"
    except ImportError:
        _guard_mode = "degraded:structural_regex"
    except Exception as exc:  # a broken optional dependency must be visible, not masked
        _guard_mode = f"degraded:prismguard_init:{type(exc).__name__}"
    return _checker


def _guard_result(prompt: str) -> tuple[bool, float, str, dict[str, Any]]:
    checker = _guard()
    if checker is None:
        blocked = bool(_JAILBREAK.search(prompt))
        return blocked, 1.0 if blocked else 0.0, "REGEX_STRUCTURAL", {"mode": _guard_mode}
    result = checker.check(prompt)
    decision = str(getattr(result, "decision", "")).lower()
    blocked = decision in {"block", "deny"} or "block" in decision
    risk = float(getattr(result, "fused_score", 1.0 if blocked else 0.0))
    resolution = str(getattr(result, "resolution_gate", "PRISMGUARD"))
    return blocked, max(0.0, min(1.0, risk)), resolution, {
        "mode": _guard_mode, "decision": decision,
        "matched_category": getattr(result, "matched_category", None),
    }


def _state(req: EvalRequest) -> dict[str, Any]:
    evidence = dict(req.evidence or {})
    state = {
        "run_id": req.id,
        "question": req.question,
        "answer": req.answer,
        "docs": evidence.pop("docs", req.context),
        "context": req.context,
        "trace": evidence.pop("trace", []),
        "node_state": evidence.pop("node_state", {}),
    }
    state.update(evidence)
    return state


def _response(
    req: EvalRequest, *, decision: str, label: str, risk: float, resolution_gate: str | None,
    guard: dict[str, Any], runtime: dict[str, Any], grounding: dict[str, Any], started: float,
) -> dict[str, Any]:
    return {
        "id": req.id, "track": req.track, "decision": decision, "label": label,
        "risk": max(0.0, min(1.0, risk)),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "llm_calls": 0, "cost_usd": 0.0, "resolution_gate": resolution_gate,
        "components": {"guard": guard, "runtime": runtime, "grounding": grounding},
        "saw_evidence": bool(req.evidence),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    _guard()
    return {
        "status": "ok", "system": "insight-stack", "guard": _guard_mode,
        "runtime": "wiring+ledger_evidence", "grounding": "prismshine.ShineGate",
        "components": {"prismguard": _guard_mode, "chorusgraph": "not_required"},
    }


@app.post("/stack_evaluate")
@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    started = time.perf_counter()
    blocked, guard_risk, guard_gate, guard_info = _guard_result(req.question)
    if blocked:
        return _response(
            req, decision="block", label="attack" if req.track == "S1" else "n/a",
            risk=guard_risk, resolution_gate=guard_gate, guard=guard_info,
            runtime={"status": "not_run"}, grounding={"status": "not_run"}, started=started,
        )
    if req.track == "S1":
        return _response(
            req, decision="allow", label="benign", risk=guard_risk, resolution_gate=guard_gate,
            guard=guard_info, runtime={"status": "not_applicable"},
            grounding={"status": "not_applicable"}, started=started,
        )

    state = _state(req)
    if req.track == "R1":
        pre = pre_llm_check(gate, state)
        verdict = pre.verdict
        assert verdict is not None
        runtime_fail = pre.should_halt or verdict.decision in {"block", "flag", "regenerate"}
        return _response(
            req, decision="halt" if runtime_fail else "pass",
            label="runtime_fail" if runtime_fail else "runtime_ok", risk=float(verdict.fused_score),
            resolution_gate=verdict.resolution_gate, guard=guard_info,
            runtime={"decision": pre.action, "signatures": [s.id for s in verdict.signatures]},
            grounding={"status": "pre_generation_only"}, started=started,
        )

    post = post_llm_check(gate, state, answer=req.answer or "")
    verdict = post.verdict
    assert verdict is not None
    hallucinated = verdict.decision != "pass"
    return _response(
        req, decision=verdict.decision, label="hallucinated" if hallucinated else "grounded",
        risk=float(verdict.fused_score), resolution_gate=verdict.resolution_gate,
        guard=guard_info, runtime={"status": "healthy_content_path"},
        grounding={"tier_reached": verdict.tier_reached, "coverage_mode": verdict.coverage_mode},
        started=started,
    )
