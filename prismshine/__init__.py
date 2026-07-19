"""PrismShine — anti-hallucination verdict engine for the Insight agent stack."""

from prismshine.gate import ShineGate
from prismshine.models import (
    EvidenceBundle,
    PreloadChunk,
    ShineVerdict,
    Signal,
    SignatureHit,
    Span,
    TraceStep,
)

__version__ = "0.1.0"

__all__ = [
    "ShineGate",
    "EvidenceBundle",
    "PreloadChunk",
    "TraceStep",
    "ShineVerdict",
    "Signal",
    "Span",
    "SignatureHit",
    "__version__",
]
