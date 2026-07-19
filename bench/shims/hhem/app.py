"""Vectara HHEM-2.1-Open bench shim — common /evaluate contract.

HHEM returns a consistency score in [0,1] (1 = answer consistent with premise).
risk = 1 - score; label hallucinated when score < 0.5 (published default).
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = "vectara/hallucination_evaluation_model"

app = FastAPI()
_model = None


def _get_model():
    global _model
    if _model is None:
        from transformers import AutoModelForSequenceClassification

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
    return _model


class EvalRequest(BaseModel):
    id: str
    question: str
    context: list[str]
    answer: str
    evidence: dict | None = None  # ignored: HHEM cannot consume runtime evidence


@app.get("/health")
def health() -> dict:
    _get_model()
    return {"status": "ok", "system": "hhem-2.1-open", "model": MODEL_ID}


@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict:
    model = _get_model()
    premise = "\n".join(req.context)
    t0 = time.perf_counter()
    score = float(model.predict([(premise, req.answer)])[0])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "id": req.id,
        "risk": 1.0 - score,
        "label": "grounded" if score >= 0.5 else "hallucinated",
        "decision": "pass" if score >= 0.5 else "flag",
        "latency_ms": latency_ms,
        "llm_calls": 0,
        "cost_usd": 0.0,
    }
