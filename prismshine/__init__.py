"""PrismShine — anti-hallucination verdict engine for the Insight agent stack.

Unified pipeline: Tier-0 trace forensics (Handbook signatures over runtime
evidence) fused with Tier-1..4 grounding verification (copy-checks, vector
coverage on reused embeddings, ONNX span classifier, opt-in LLM judge) into
one auditable ShineVerdict.

NOT YET IMPLEMENTED — this package is a design-stage scaffold.
See docs/DESIGN.md for the full architecture and public API contract.

Planned public API (docs/DESIGN.md §7.1):
    from prismshine import ShineGate, EvidenceBundle, ShineVerdict
"""

__version__ = "0.0.1"
