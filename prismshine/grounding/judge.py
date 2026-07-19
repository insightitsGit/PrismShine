"""Tier 4: opt-in LLM judge protocol + reference implementations."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    risk: float
    claim_support: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class Judge(Protocol):
    def __call__(self, claims: list[str], context: str) -> JudgeResult: ...


class EscalationBudget:
    """Hard-cap judge traffic fraction."""

    def __init__(self, budget: float = 0.10) -> None:
        self.budget = budget
        self._total = 0
        self._escalated = 0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            self._total += 1
            rate = self._escalated / self._total
            if rate >= self.budget and self._total > 10:
                return False
            self._escalated += 1
            return True

    @property
    def rate(self) -> float:
        with self._lock:
            return self._escalated / max(self._total, 1)


class CachedJudge:
    def __init__(self, inner: Judge) -> None:
        self.inner = inner
        self._cache: dict[str, JudgeResult] = {}
        self._lock = threading.Lock()

    def __call__(self, claims: list[str], context: str) -> JudgeResult:
        key = hashlib.sha256(
            json.dumps({"claims": claims, "context": context}, sort_keys=True).encode()
        ).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        result = self.inner(claims, context)
        with self._lock:
            self._cache[key] = result
        return result


class OpenAIJudge:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def __call__(self, claims: list[str], context: str) -> JudgeResult:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install prismshine[judge-openai]") from exc
        client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        prompt = (
            "You are an entailment judge. For each claim, say supported|unsupported "
            "given ONLY the context. Reply JSON: "
            '{"claims":[{"claim":"...","label":"supported|unsupported","risk":0-1}],'
            '"overall_risk":0-1}\n\n'
            f"CONTEXT:\n{context}\n\nCLAIMS:\n"
            + "\n".join(f"- {c}" for c in claims)
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"overall_risk": 0.5, "claims": [], "raw_text": text}
        return JudgeResult(
            risk=float(data.get("overall_risk", 0.5)),
            claim_support=list(data.get("claims") or []),
            raw=data,
        )


class GeminiJudge:
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def __call__(self, claims: list[str], context: str) -> JudgeResult:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install prismshine[judge-gemini]") from exc
        client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        prompt = (
            "Entailment judge. Return JSON with overall_risk 0-1 and per-claim labels.\n"
            f"CONTEXT:\n{context}\n\nCLAIMS:\n"
            + "\n".join(f"- {c}" for c in claims)
        )
        resp = client.models.generate_content(model=self.model, contents=prompt)
        text = getattr(resp, "text", None) or str(resp)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"overall_risk": 0.5, "claims": [], "raw_text": text}
        return JudgeResult(
            risk=float(data.get("overall_risk", 0.5)),
            claim_support=list(data.get("claims") or []),
            raw=data,
        )


def build_judge(provider: str | None, model: str | None = None) -> Judge | None:
    if not provider:
        return None
    p = provider.lower()
    if p in {"openai", "judge-openai"}:
        return CachedJudge(OpenAIJudge(model=model or "gpt-4o-mini"))
    if p in {"gemini", "judge-gemini", "google"}:
        return CachedJudge(GeminiJudge(model=model or "gemini-2.0-flash"))
    raise ValueError(f"Unknown judge provider: {provider}")
