"""Pydantic data models for EvidenceBundle, signals, and ShineVerdict."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Decision = Literal["pass", "flag", "block", "regenerate"]
ChunkSource = Literal["retrieval", "tool", "memory", "history", "cache", "system"]
TraceKind = Literal["retrieval", "tool", "cache", "llm", "memory", "guard", "other"]
TraceStatus = Literal["ok", "error", "empty", "timeout", "skipped"]
Severity = Literal["fatal", "error", "warning", "info"]
VectorSpace = Literal["raw-384", "jl-64", "none"]
CoverageMode = Literal["raw-384", "user-embedder", "lexical", "skipped", "resonance"]
Strictness = Literal["lenient", "standard", "strict", "paranoid"]


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


class TraceStep(BaseModel):
    hop: str
    kind: TraceKind = "other"
    status: TraceStatus = "ok"
    scores: dict[str, float] = Field(default_factory=dict)
    duration_ms: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


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
