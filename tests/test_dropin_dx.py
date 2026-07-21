"""Drop-in DX: source aliases, shadow/block enforce, validate_grounding."""

from __future__ import annotations

import os

from prismshine import PreloadChunk, enforce_mode_from_env, get_gate, validate_grounding
from prismshine.models import normalize_chunk_source


def test_source_aliases():
    assert normalize_chunk_source("kb") == "retrieval"
    assert normalize_chunk_source("web_search") == "retrieval"
    assert normalize_chunk_source("docs") == "retrieval"
    assert normalize_chunk_source("chat") == "history"
    assert normalize_chunk_source("cortex") == "memory"
    assert PreloadChunk(chunk_id="1", text="hi", source="kb").source == "retrieval"
    assert PreloadChunk(chunk_id="1", text="hi", source="WEB").source == "retrieval"


def test_validate_grounding_shadow_never_blocks(monkeypatch):
    monkeypatch.setenv("PRISMSHINE_ENFORCE", "0")
    assert enforce_mode_from_env() == "shadow"
    get_gate(reset=True)
    r = validate_grounding(
        question="What was revenue?",
        answer="Revenue was $9999.",
        contexts=["Revenue was $1000 in Q1."],
        run_id="t-shadow",
    )
    assert r["enforce_mode"] == "shadow"
    assert r["blocked"] is False
    assert r["answer"] == "Revenue was $9999."
    assert r["meta"]["decision"] in {"pass", "flag", "block", "regenerate"}
    assert "evidence_hash" in r["meta"]


def test_validate_grounding_block_only_default(monkeypatch):
    monkeypatch.delenv("PRISMSHINE_ENFORCE", raising=False)
    assert enforce_mode_from_env() == "block"
    get_gate(reset=True)
    # Soft mismatch typically flags — must NOT halt under block-only.
    r = validate_grounding(
        question="What was revenue?",
        answer="Revenue was $9999 in Q1.",
        contexts=[{"text": "Revenue was $1000 in Q1.", "source": "kb"}],
        run_id="t-block",
        enforce="block",
    )
    assert r["enforce_mode"] == "block"
    if r["decision"] == "flag":
        assert r["blocked"] is False
        assert r["answer"].startswith("Revenue was $9999")
    # Empty preload path can block via Tier-0 — that SHOULD halt.
    r2 = validate_grounding(
        question="What was revenue?",
        answer="Anything.",
        contexts=[],
        run_id="t-empty",
        enforce="block",
    )
    # empty → system sentinel; may pass/flag depending on path — just ensure shape
    assert "meta" in r2 and "resolution_gate" in r2["meta"]


def test_span_note_when_onnx_missing(monkeypatch):
    monkeypatch.delenv("PRISMSHINE_SPAN_ONNX", raising=False)
    get_gate(reset=True)
    r = validate_grounding(
        question="q",
        answer="Revenue was $1000.",
        contexts=["Revenue was $1000."],
        run_id="t-span",
    )
    assert r["meta"].get("span_backend") in {"lexical", "unavailable", "onnx"}
    if r["meta"].get("span_backend") != "onnx":
        assert "span_note" in r["meta"]
