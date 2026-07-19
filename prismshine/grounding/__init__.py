"""Grounding verification tiers (docs/DESIGN.md §5).

Planned modules:
    copycheck.py     — Tier 1: typed fact extraction (numbers/dates/IDs/entities) + normalized
                       matching + arithmetic closure for derived figures
    coverage.py      — Tier 2: sentence split + support scoring against REUSED preload vectors
                       (raw 384-d space; composite support for comparative/aggregative
                       sentences; optional resonance interference mode)
    contradiction.py — Tier 2 screen: negation asymmetry + opposite-verb lexicon vs the
                       best-supporting chunk; hits promote to Tier 3 (never silent pass)
    spans.py         — Tier 3: ONNX token-classification span detector (no LLM)
    judge.py         — Tier 4: pluggable LLM judge protocol (opt-in escalation only)
"""
