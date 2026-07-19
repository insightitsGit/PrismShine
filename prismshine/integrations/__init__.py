"""Ecosystem integrations (docs/INTEGRATION.md).

Runtime plugin contract: docs/DESIGN.md §8.1. The core NEVER imports from this
package — runtimes plug into the neutral EvidenceBundle/ShineVerdict contract.

Planned modules:
    chorusgraph.py — richest plugin: shine_node(), ShineInterceptor (pre+post generation),
                     ledger write-back, on_fact_corrected/on_source_updated hooks
    langgraph.py   — LangGraph plugin: shine node factory + conditional-edge router,
                     state-dict evidence extraction
    prismguard.py  — output-gate exposure + GUARD_GRAY_INPUT signal consumption
    prismcortex.py — memory conflict / staging signals into forensics
"""
