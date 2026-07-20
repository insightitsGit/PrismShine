"""Light guard -> HHEM shim representing a minimal sequential LangGraph node flow.

No LangGraph dependency is required: the graph shape is simply guard -> hhem.
Like a standard content-only graph node, this service intentionally ignores
runtime ledger evidence.
"""

from __future__ import annotations

import re
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = "vectara/hallucination_evaluation_model"
app = FastAPI()
_model: Any | None = None
_INJECTION = re.compile(
    r"\b(ignore|disregard|override|bypass|jailbreak|DAN|developer mode|system message)\b",
    re.IGNORECASE,
)


class EvalRequest(BaseModel):
    id: str
    track: str
    question: str
    context: list[str] = []
    answer: str | None = None
    evidence: dict[str, Any] | None = None
    gold: str | None = None


def _get_model() -> Any:
    global _model
    if _model is None:
        from transformers import AutoModelForSequenceClassification

        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    return _model


def _response(req: EvalRequest, **values: Any) -> dict[str, Any]:
    return {
        "id": req.id, "track": req.track, "llm_calls": 0, "cost_usd": 0.0,
        "resolution_gate": values.pop("resolution_gate", None),
        "components": values.pop("components", {}),
        "saw_evidence": False,
        **values,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok", "system": "oss-langgraph-hhem", "guard": "light_regex",
        "runtime": "sequential_node_simulation,evidence_ignored", "grounding": MODEL_ID,
    }


@app.post("/stack_evaluate")
@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    started = time.perf_counter()
    attack = bool(_INJECTION.search(req.question))
    if req.track == "S1":
        return _response(
            req, decision="block" if attack else "allow", label="attack" if attack else "benign",
            risk=1.0 if attack else 0.0, latency_ms=(time.perf_counter() - started) * 1000,
            resolution_gate="REGEX_GUARD",
            components={"guard": "light_regex", "runtime": "not_applicable", "grounding": "not_run"},
        )
    if req.track == "R1":
        return _response(
            req, decision="pass", label="runtime_ok", risk=0.0,
            latency_ms=(time.perf_counter() - started) * 1000, resolution_gate="EVIDENCE_IGNORED",
            components={"guard": "light_regex", "runtime": "evidence_ignored", "grounding": "not_run"},
        )
    model = _get_model()
    score = float(model.predict([("\n".join(req.context), req.answer or "")])[0])
    risk = 1.0 - score
    label = "grounded" if score >= 0.5 else "hallucinated"
    return _response(
        req, decision="pass" if label == "grounded" else "flag", label=label, risk=risk,
        latency_ms=(time.perf_counter() - started) * 1000, resolution_gate="HHEM",
        components={"guard": "light_regex", "runtime": "evidence_ignored",
                    "grounding": {"model": MODEL_ID, "score": score}},
    )
