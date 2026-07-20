"""OSS LLM Guard + MiniLM Stack-suite shim (intentionally evidence-blind)."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
_scanner: Any | None = None
_scanner_state = "uninitialized"
_model: Any | None = None
_INJECTION = re.compile(
    r"\b(ignore|disregard|override|bypass|jailbreak|DAN|developer mode|system message)"
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


def _get_scanner() -> Any | None:
    global _scanner, _scanner_state
    if _scanner is not None or _scanner_state != "uninitialized":
        return _scanner
    if os.environ.get("STACK_FORCE_REGEX_GUARD") == "1":
        _scanner_state = "degraded:forced_regex"
        return None
    try:
        from llm_guard.input_scanners import PromptInjection

        _scanner = PromptInjection()
        _scanner_state = "llm-guard:PromptInjection"
    except ImportError:
        _scanner_state = "degraded:regex_no_llm_guard"
    except Exception as exc:
        _scanner_state = f"degraded:llm_guard_init:{type(exc).__name__}"
    return _scanner


def _is_attack(text: str) -> tuple[bool, float]:
    scanner = _get_scanner()
    if scanner is None:
        blocked = bool(_INJECTION.search(text))
        return blocked, 1.0 if blocked else 0.0
    # LLM Guard's scan API returns (sanitized_prompt, is_valid, risk_score).
    result = scanner.scan(text)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("Unexpected llm-guard PromptInjection scan result")
    valid = bool(result[1])
    score = float(result[2]) if len(result) > 2 else (0.0 if valid else 1.0)
    return not valid, max(0.0, min(1.0, score))


def _embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype=np.float64)


def _faithfulness(context: list[str], answer: str) -> tuple[float, str]:
    if not context:
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
        "id": req.id, "track": req.track, "llm_calls": 0, "cost_usd": 0.0,
        "resolution_gate": values.pop("resolution_gate", None),
        "components": values.pop("components", {}),
        "saw_evidence": False,  # this comparator cannot see runtime ledgers
        **values,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    _get_scanner()
    return {
        "status": "ok", "system": "oss-llmguard", "guard": _scanner_state,
        "grounding": "MiniLM cosine, risk=1-sim, threshold=0.45", "runtime": "evidence_ignored",
    }


@app.post("/stack_evaluate")
@app.post("/evaluate")
def evaluate(req: EvalRequest) -> dict[str, Any]:
    started = time.perf_counter()
    attack, risk = _is_attack(req.question)
    guard = {"scanner": _scanner_state}
    if req.track == "S1":
        return _response(
            req, decision="block" if attack else "allow", label="attack" if attack else "benign",
            risk=risk, latency_ms=(time.perf_counter() - started) * 1000,
            resolution_gate="PROMPT_INJECTION", components={"guard": guard, "runtime": "ignored",
                                                            "grounding": "not_run"},
        )
    if req.track == "R1":
        # Do not infer a runtime failure from data the package was not given.
        return _response(
            req, decision="pass", label="runtime_ok", risk=0.0,
            latency_ms=(time.perf_counter() - started) * 1000, resolution_gate="EVIDENCE_IGNORED",
            components={"guard": guard, "runtime": "evidence_ignored", "grounding": "not_run"},
        )
    risk, label = _faithfulness(req.context, req.answer or "")
    return _response(
        req, decision="flag" if label == "hallucinated" else "pass", label=label, risk=risk,
        latency_ms=(time.perf_counter() - started) * 1000, resolution_gate="MINILM_COSINE",
        components={"guard": guard, "runtime": "evidence_ignored",
                    "grounding": {"threshold": 0.45, "risk": "1-cosine"}},
    )
