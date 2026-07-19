"""PrismShine bench shim — common /evaluate contract (content-only track).

Runs the default profile fast path (Tiers 0-3, zero LLM calls) with a real
sentence-transformers embedder (same MiniLM family prismlang uses).
"""

from __future__ import annotations

import os
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.grounding.splitter import split_sentences

MODEL_NAME = os.environ.get("SHINE_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
PROFILE = os.environ.get("SHINE_PROFILE", "default")
CALIBRATION = os.environ.get("PRISMSHINE_CALIBRATION")

_st_model = None


def _embedder(texts: list[str]) -> np.ndarray:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(MODEL_NAME, device="cpu")
    return np.asarray(_st_model.encode(texts, normalize_embeddings=True), dtype=np.float64)


app = FastAPI()
gate = ShineGate.build(
    profile=PROFILE, embedder=_embedder, calibration_path=CALIBRATION
)
# warm the encoder so first-sample latency is not model load time
_embedder(["warmup sentence"])


class EvalRequest(BaseModel):
    id: str
    question: str
    context: list[str]
    answer: str
    evidence: dict | None = None


@app.get("/health")
def health() -> dict:
    caps = gate.capabilities()
    return {
        "status": "ok",
        "system": "prismshine-fast",
        "profile": PROFILE,
        "calibration_version": gate.calibration_version,
        "threshold_status": gate.policy.threshold_status,
        "coverage_mode": caps["coverage_mode"],
        "span_backend": caps["span_backend"],
        "tiers": caps["tiers"],
    }


@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict:
    t0 = time.perf_counter()
    data = {
        "run_id": req.id,
        "question": req.question,
        "answer": req.answer,
        # Sentence-granular chunks: short answers must be able to match a single
        # supporting sentence, not fight the cosine of a whole paragraph.
        "preload": [
            {"chunk_id": f"c{i}-{j}", "text": sent, "source": "retrieval"}
            for i, c in enumerate(req.context)
            for j, sent in enumerate(split_sentences(c) or [c])
        ],
        # Content-only track: synthesize a healthy retrieval step (no ledger available).
        "trace": [
            {
                "hop": "retrieve",
                "kind": "retrieval",
                "status": "ok",
                "scores": {"constructive_score": 0.9},
                "detail": {"n_chunks": len(req.context), "top_k": len(req.context)},
            }
        ],
    }
    if req.evidence:
        data.update(req.evidence)
    bundle, _ = bundle_from_dict(data)
    verdict = gate.verify(bundle)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "id": req.id,
        "risk": float(verdict.fused_score),
        "label": "grounded" if verdict.decision == "pass" else "hallucinated",
        "decision": verdict.decision,
        "resolution_gate": verdict.resolution_gate,
        "tier_reached": verdict.tier_reached,
        "spans": [s.model_dump() for s in verdict.spans[:10]],
        "latency_ms": latency_ms,
        "llm_calls": 0,
        "cost_usd": 0.0,
    }
