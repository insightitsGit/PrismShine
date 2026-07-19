"""Handbook: declarative failure-signature catalog (YAML) + schema + loader.

Signatures are data, detectors are code (docs/DECISIONS.md ADR-5).
Schema and initial catalog: docs/HANDBOOK.md.

Planned modules:
    schema.py   — signature pydantic schema
    loader.py   — YAML load / merge (builtin -> domain pack -> tenant) / version pinning
    builtin/    — core.yaml (+ licensed domain packs: clinical, finance, legal)
"""
