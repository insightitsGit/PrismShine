"""Pydantic data models for EvidenceBundle, signals, and ShineVerdict."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Decision = Literal["pass", "flag", "block", "regenerate"]
ChunkSource = Literal["retrieval", "tool", "memory", "history", "cache", "system"]
TraceKind = Literal["retrieval", "tool", "cache", "llm", "memory", "guard", "other"]
TraceStatus = Literal["ok", "error", "empty", "timeout", "skipped"]
Severity = Literal["fatal", "error", "warning", "info"]
VectorSpace = Literal["raw-384", "jl-64", "none"]
CoverageMode = Literal["raw-384", "user-embedder", "lexical", "skipped", "resonance"]
Strictness = Literal["lenient", "standard", "strict", "paranoid"]

# Integrator-friendly aliases → canonical ChunkSource (kb/web/docs → retrieval, etc.)
_CHUNK_SOURCE_ALIASES: dict[str, ChunkSource] = {
    "retrieval": "retrieval",
    "retrieve": "retrieval",
    "rag": "retrieval",
    "kb": "retrieval",
    "knowledge": "retrieval",
    "knowledge_base": "retrieval",
    "knowledge-base": "retrieval",
    "docs": "retrieval",
    "doc": "retrieval",
    "document": "retrieval",
    "documents": "retrieval",
    "web": "retrieval",
    "web_search": "retrieval",
    "web-search": "retrieval",
    "search": "retrieval",
    "snippet": "retrieval",
    "snippets": "retrieval",
    "context": "retrieval",
    "passage": "retrieval",
    "passages": "retrieval",
    "chunk": "retrieval",
    "chunks": "retrieval",
    "vectorstore": "retrieval",
    "vector_store": "retrieval",
    "tool": "tool",
    "tools": "tool",
    "function": "tool",
    "function_call": "tool",
    "memory": "memory",
    "mem": "memory",
    "cortex": "memory",
    "history": "history",
    "chat": "history",
    "conversation": "history",
    "message": "history",
    "messages": "history",
    "cache": "cache",
    "cached": "cache",
    "system": "system",
    "empty": "system",
    "sentinel": "system",
    "internal": "system",
}

_TRACE_KIND_ALIASES: dict[str, TraceKind] = {
    "retrieval": "retrieval",
    "retrieve": "retrieval",
    "rag": "retrieval",
    "tool": "tool",
    "tools": "tool",
    "cache": "cache",
    "llm": "llm",
    "generate": "llm",
    "generation": "llm",
    "memory": "memory",
    "guard": "guard",
    "prismguard": "guard",
    "other": "other",
}


def normalize_chunk_source(value: Any, *, default: ChunkSource = "retrieval") -> ChunkSource:
    """Map common integrator labels onto canonical ``ChunkSource`` values."""
    if value is None or value == "":
        return default
    key = str(value).strip().lower().replace(" ", "_")
    if key in _CHUNK_SOURCE_ALIASES:
        return _CHUNK_SOURCE_ALIASES[key]
    # Unknown labels default to retrieval (most drop-in chat/RAG paths) rather than
    # raising — original string is preserved via metadata if callers need it.
    return default


def normalize_trace_kind(value: Any, *, default: TraceKind = "other") -> TraceKind:
    if value is None or value == "":
        return default
    key = str(value).strip().lower().replace(" ", "_")
    return _TRACE_KIND_ALIASES.get(key, default)


class Span(BaseModel):
    start: int
    end: int
    text: str
    reason: str
    tier: int = 0


class ContextBudget(BaseModel):
    limit_tokens: int | None = None
    used_tokens: int | None = None
    truncated: bool = False
    truncated_tail: bool = False


class PreloadChunk(BaseModel):
    chunk_id: str
    text: str
    vector: list[float] | None = None
    vector_space: str = "none"
    source: ChunkSource = "retrieval"
    retrieval_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, v: Any) -> ChunkSource:
        return normalize_chunk_source(v)


class TraceStep(BaseModel):
    hop: str
    kind: TraceKind = "other"
    status: TraceStatus = "ok"
    scores: dict[str, float] = Field(default_factory=dict)
    duration_ms: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v: Any) -> TraceKind:
        return normalize_trace_kind(v)


class EvidenceBundle(BaseModel):
    run_id: str
    tenant_id: str | None = None
    question: str
    answer: str | None = None
    preload: list[PreloadChunk] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    node_state: dict[str, Any] = Field(default_factory=dict)
    declared_sections: list[str] = Field(default_factory=list)
    context_budget: ContextBudget | None = None


class Signal(BaseModel):
    name: str
    tier: int
    value: float
    weight: float = 1.0
    spans: list[Span] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class SignatureHit(BaseModel):
    id: str
    title: str = ""
    severity: Severity
    scope: str = "preload"
    advice: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    signal_value: float = 1.0


class ShineVerdict(BaseModel):
    decision: Decision
    resolution_gate: str
    fused_score: float
    confidence: float
    signatures: list[SignatureHit] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    tier_reached: int = 0
    coverage_mode: str = "skipped"
    strictness_effective: str = "standard"
    dormant_families: list[str] = Field(default_factory=list)
    evidence_hash: str
    verdict_id: str
    signals: list[Signal] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
