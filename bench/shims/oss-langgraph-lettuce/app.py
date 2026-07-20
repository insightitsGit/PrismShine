"""LangGraph-shaped LettuceDetect shim (evidence-blind).

Closest open-source *product* peer to PrismShine's effect-side Tier-3
(span-level unsupported detection). Ignores runtime ledger ``evidence``.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
app = FastAPI()
_detector: Any | None = None
_mode = "uninitialized"


class EvalRequest(BaseModel):
    id: str
    track: str
    question: str
    context: list[str] = []
    answer: str | None = None
    evidence: dict[str, Any] | None = None
    gold: str | None = None


def _get_detector() -> Any:
    global _detector, _mode
    if _detector is not None:
        return _detector
    from lettucedetect.models.inference import HallucinationDetector

    _detector = HallucinationDetector(method="transformer", model_path=MODEL_ID)
    _mode = f"lettucedetect:{MODEL_ID}"
    return _detector


def _response(req: EvalRequest, **values: Any) -> dict[str, Any]:
    return {
        "id": req.id,
        "track": req.track,
        "llm_calls": 0,
        "cost_usd": 0.0,
        "resolution_gate": values.pop("resolution_gate", None),
        "components": values.pop("components", {}),
        "saw_evidence": False,
        **values,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _get_detector()
        status = "ok"
    except Exception as exc:  # pragma: no cover
        status = "degraded"
        _mode_local = f"error:{type(exc).__name__}"
        return {
            "status": status,
            "system": "oss-langgraph-lettuce",
            "runtime": "sequential_node_simulation,evidence_ignored",
            "grounding": _mode_local,
            "guard": False,
        }
    return {
        "status": status,
        "system": "oss-langgraph-lettuce",
        "runtime": "sequential_node_simulation,evidence_ignored",
        "grounding": _mode,
        "guard": False,
    }


@app.post("/stack_evaluate")
@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    started = time.perf_counter()
    if req.track == "R1":
        return _response(
            req,
            decision="pass",
            label="runtime_ok",
            risk=0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
            resolution_gate="EVIDENCE_IGNORED",
            components={"runtime": "evidence_ignored", "grounding": "not_run"},
        )
    detector = _get_detector()
    contexts = list(req.context or [])
    answer = req.answer or ""
    spans = detector.predict(
        context=contexts,
        question=req.question,
        answer=answer,
        output_format="spans",
    )
    if not isinstance(spans, list):
        spans = list(spans or [])
    halluc = len(spans) > 0
    # Risk ≈ unsupported char mass / answer length (cap 1.0)
    unsupported = sum(max(0, int(s.get("end", 0)) - int(s.get("start", 0))) for s in spans if isinstance(s, dict))
    risk = min(1.0, unsupported / max(len(answer), 1)) if halluc else 0.0
    if halluc and risk < 0.15:
        risk = 0.55  # any span → at least mid risk
    return _response(
        req,
        decision="flag" if halluc else "pass",
        label="hallucinated" if halluc else "grounded",
        risk=risk,
        latency_ms=(time.perf_counter() - started) * 1000,
        resolution_gate="LETTUCEDETECT_SPANS",
        components={
            "runtime": "evidence_ignored",
            "grounding": {"model": MODEL_ID, "n_spans": len(spans), "spans": spans[:8]},
        },
    )
