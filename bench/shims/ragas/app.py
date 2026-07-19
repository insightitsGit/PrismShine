"""RAGAS faithfulness bench shim — common /evaluate contract.

Uses a pinned local Ollama model (reproducible, $0) as the judge LLM.
faithfulness = fraction of answer claims supported by context (1 = grounded).
risk = 1 - faithfulness; label hallucinated when faithfulness < 0.5.
"""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")

app = FastAPI()
_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness

        llm = LangchainLLMWrapper(
            ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0)
        )
        _scorer = Faithfulness(llm=llm)
    return _scorer


class EvalRequest(BaseModel):
    id: str
    question: str
    context: list[str]
    answer: str
    evidence: dict | None = None  # ignored: RAGAS cannot consume runtime evidence


@app.get("/health")
def health() -> dict:
    import httpx

    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    models = [m.get("name") for m in r.json().get("models", [])]
    return {
        "status": "ok" if any(OLLAMA_MODEL in (m or "") for m in models) else "model_missing",
        "system": "ragas-faithfulness",
        "judge_model": OLLAMA_MODEL,
        "ollama_models": models,
    }


@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict:
    from ragas.dataset_schema import SingleTurnSample

    scorer = _get_scorer()
    sample = SingleTurnSample(
        user_input=req.question,
        response=req.answer,
        retrieved_contexts=list(req.context),
    )
    t0 = time.perf_counter()
    try:
        score = asyncio.run(scorer.single_turn_ascore(sample))
        score = float(score) if score == score else 0.5  # NaN -> unknown midpoint
        error = None
    except Exception as exc:  # noqa: BLE001
        score, error = 0.5, str(exc)[:300]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    out = {
        "id": req.id,
        "risk": 1.0 - score,
        "label": "grounded" if score >= 0.5 else "hallucinated",
        "decision": "pass" if score >= 0.5 else "flag",
        "latency_ms": latency_ms,
        # statement extraction + NLI verdict = 2 LLM round-trips per sample
        "llm_calls": 2,
        "cost_usd": 0.0,
    }
    if error:
        out["error"] = error
    return out
