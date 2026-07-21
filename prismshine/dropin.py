"""PrismGuard-style drop-in for chat / RAG apps.

Lazy singleton gate, env-driven enforce mode, and a single
``validate_grounding`` call that attaches verdict meta to the reply.

Env flags (same spirit as PrismGuard)::

    PRISMSHINE_ENFORCE=0|off|shadow   # observe only — never rewrite / halt
    PRISMSHINE_ENFORCE=block          # default: halt only on decision=block
    PRISMSHINE_ENFORCE=flag|strict    # also halt on flag (noisier)

Tier-3 ONNX (headline receipt path)::

    pip install "prismshine[spans]"
    python -m prismshine.tools.ensure_span_onnx --export
    set PRISMSHINE_SPAN_ONNX=.../model.onnx
    set PRISMSHINE_SPAN_TOKENIZER=.../tokenizer.json
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.models import ShineVerdict, normalize_chunk_source

EnforceMode = Literal["shadow", "block", "flag"]

_GATE: ShineGate | None = None
_GATE_LOCK = threading.Lock()
_GATE_KWARGS: dict[str, Any] = {}


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def enforce_mode_from_env(default: EnforceMode = "block") -> EnforceMode:
    """Resolve ``PRISMSHINE_ENFORCE`` to shadow | block | flag."""
    raw = (os.environ.get("PRISMSHINE_ENFORCE") or "").strip().lower()
    if raw in {"", "default"}:
        return default
    if raw in {"0", "off", "false", "no", "shadow", "observe"}:
        return "shadow"
    if raw in {"block", "blocks", "hard"}:
        return "block"
    if raw in {"flag", "flags", "strict", "all", "1", "true", "yes", "on"}:
        # "1"/true → flag is intentional for people who set ENFORCE=1 expecting
        # hard mode; default install leaves the env unset (= block-only).
        return "flag"
    return default


def get_gate(*, reset: bool = False, **build_kwargs: Any) -> ShineGate:
    """Lazy singleton ``ShineGate`` (PrismGuard checker pattern)."""
    global _GATE, _GATE_KWARGS
    with _GATE_LOCK:
        if reset or _GATE is None or (build_kwargs and build_kwargs != _GATE_KWARGS):
            kwargs = dict(build_kwargs)
            if "profile" not in kwargs:
                kwargs["profile"] = os.environ.get("PRISMSHINE_PROFILE", "default")
            if "strictness" not in kwargs:
                kwargs["strictness"] = os.environ.get("PRISMSHINE_STRICTNESS", "standard")
            _GATE = ShineGate.build(**kwargs)
            _GATE_KWARGS = kwargs
        return _GATE


def _contexts_to_preload(contexts: list[Any]) -> list[dict[str, Any]]:
    preload: list[dict[str, Any]] = []
    for i, item in enumerate(contexts or []):
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            preload.append(
                {
                    "chunk_id": f"c{i}",
                    "text": text,
                    "source": "retrieval",
                }
            )
            continue
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item.get("snippet") or "").strip()
            if not text:
                continue
            raw_source = item.get("source")
            preload.append(
                {
                    "chunk_id": str(item.get("chunk_id") or item.get("id") or f"c{i}"),
                    "text": text,
                    "source": normalize_chunk_source(raw_source),
                    "metadata": {
                        **dict(item.get("metadata") or {}),
                        **({"source_raw": raw_source} if raw_source not in (None, "") else {}),
                    },
                }
            )
            continue
        text = str(item).strip()
        if text:
            preload.append({"chunk_id": f"c{i}", "text": text, "source": "retrieval"})
    if not preload:
        preload.append({"chunk_id": "empty", "text": "(no preload)", "source": "system"})
    return preload


def should_halt(verdict: ShineVerdict, mode: EnforceMode) -> bool:
    if mode == "shadow":
        return False
    if mode == "block":
        return verdict.decision == "block"
    return verdict.decision in {"block", "flag", "regenerate"}


def validate_grounding(
    *,
    question: str,
    answer: str,
    contexts: list[Any],
    run_id: str = "chat",
    enforce: EnforceMode | str | None = None,
    gate: ShineGate | None = None,
    fallback: str | None = None,
) -> dict[str, Any]:
    """Drop-in post-answer check for chat / RAG pipelines.

    Returns a dict safe to merge onto the assistant reply meta::

        result = validate_grounding(question=q, answer=a, contexts=kb_or_web)
        if result["blocked"]:
            a = result["answer"]  # fallback substitution
        meta["prismshine"] = result["meta"]
    """
    mode: EnforceMode
    if enforce is None:
        mode = enforce_mode_from_env()
    elif isinstance(enforce, str):
        key = enforce.strip().lower()
        if key in {"env", "default", ""}:
            mode = enforce_mode_from_env()
        elif key in {"0", "off", "false", "no", "shadow", "observe"}:
            mode = "shadow"
        elif key in {"flag", "flags", "strict", "all", "1", "true", "yes", "on"}:
            mode = "flag"
        else:
            mode = "block"
    else:
        mode = enforce

    g = gate or get_gate()
    preload = _contexts_to_preload(list(contexts or []))
    bundle, feedback = bundle_from_dict(
        {
            "run_id": run_id,
            "question": question,
            "answer": answer,
            "preload": preload,
        }
    )
    verdict = g.verify(bundle)
    caps = g.capabilities()
    halt = should_halt(verdict, mode)
    safe_fallback = (
        fallback
        if fallback is not None
        else "I don't have enough grounded evidence to answer that confidently."
    )
    out_answer = safe_fallback if halt else answer
    meta = {
        "decision": verdict.decision,
        "resolution_gate": verdict.resolution_gate,
        "evidence_hash": verdict.evidence_hash,
        "fused_score": verdict.fused_score,
        "confidence": verdict.confidence,
        "tier_reached": verdict.tier_reached,
        "span_backend": caps.get("span_backend"),
        "coverage_mode": verdict.coverage_mode,
        "enforce_mode": mode,
        "enforced": halt,
        "signatures": [s.id for s in verdict.signatures],
        "advice": list(verdict.advice),
        "capability_notes": feedback,
        "pass_means": caps.get("pass_means"),
    }
    if caps.get("span_backend") != "onnx":
        meta["span_note"] = (
            "Tier-3 ONNX not loaded — lexical/unavailable path "
            "(pip install 'prismshine[spans]' && python -m prismshine.tools.ensure_span_onnx --export)"
        )
    return {
        "ok": not halt,
        "blocked": halt,
        "decision": verdict.decision,
        "resolution_gate": verdict.resolution_gate,
        "evidence_hash": verdict.evidence_hash,
        "answer": out_answer,
        "verdict": verdict,
        "meta": meta,
        "enforce_mode": mode,
    }
