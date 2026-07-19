"""EvidenceBundle construction: builders, validation, and runtime adapters.

Planned modules (docs/DESIGN.md §7):
    builder.py            — EvidenceBundle builders/validators
    adapters/chorusgraph.py — Route Ledger + node state -> EvidenceBundle (vectors reused)
    adapters/generic.py     — plain dict -> EvidenceBundle (standalone mode)
    adapters/langgraph.py   — LangGraph state -> EvidenceBundle
"""
