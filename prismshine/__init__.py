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
from prismshine.runtime import RuntimeAdapter, check_adapter
from prismshine.wiring import (
    DictStateAdapter,
    ShineDecision,
    make_dict_adapter,
    pre_llm_check,
    post_llm_check,
    require_shine_wiring,
    shine_verify_node,
    wrap_llm,
)

__version__ = "0.2.1"

__all__ = [
    "ShineGate",
    "EvidenceBundle",
    "PreloadChunk",
    "TraceStep",
    "ShineVerdict",
    "Signal",
    "Span",
    "SignatureHit",
    "RuntimeAdapter",
    "check_adapter",
    "ShineDecision",
    "DictStateAdapter",
    "make_dict_adapter",
    "pre_llm_check",
    "post_llm_check",
    "wrap_llm",
    "shine_verify_node",
    "require_shine_wiring",
    "__version__",
]
