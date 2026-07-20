"""LangGraph-shaped MiniLM cosine faithfulness shim (evidence-blind).

Represents the common DIY agent pattern: sequential nodes + embedding
similarity as a grounding check. Ignores runtime ledger ``evidence``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
_model: Any | None = None


class EvalRequest(BaseModel):
    id: str
    track: str
    question: str
    context: list[str] = []
    answer: str | None = None
    evidence: dict[str, Any] | None = None
    gold: str | None = None


def _embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype=np.float64)


def _faithfulness(context: list[str], answer: str) -> tuple[float, str]:
    if not context or not answer:
        return 1.0, "hallucinated"
    embeddings = _embed([*context, answer])
    context_mean = embeddings[:-1].mean(axis=0)
    norm = np.linalg.norm(context_mean)
    if norm:
        context_mean /= norm
    similarity = float(np.clip(np.dot(context_mean, embeddings[-1]), -1.0, 1.0))
    risk = max(0.0, min(1.0, 1.0 - similarity))
    return risk, "hallucinated" if risk >= 0.45 else "grounded"


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
    return {
        "status": "ok",
        "system": "oss-langgraph-minilm",
        "runtime": "sequential_node_simulation,evidence_ignored",
        "grounding": "MiniLM cosine, risk=1-sim, threshold=0.45",
        "guard": False,
    }


@app.post("/stack_evaluate")
@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    started = time.perf_counter()
    # R1: content-only graphs cannot inspect ledger evidence.
    if req.track == "R1":
        return _response(
            req,
            decision="pass",
            label="runtime_ok",
            risk=0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
            resolution_gate="EVIDENCE_IGNORED",
            components={
                "runtime": "evidence_ignored",
                "grounding": "not_run",
            },
        )
    risk, label = _faithfulness(req.context, req.answer or "")
    return _response(
        req,
        decision="pass" if label == "grounded" else "flag",
        label=label,
        risk=risk,
        latency_ms=(time.perf_counter() - started) * 1000,
        resolution_gate="MINILM_COSINE",
        components={
            "runtime": "evidence_ignored",
            "grounding": {"model": "all-MiniLM-L6-v2", "risk": risk},
        },
    )
