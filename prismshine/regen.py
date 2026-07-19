"""Bounded regenerate protocol (ADR-7) — core, usable by integrations."""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_ATTEMPTS = 1


def build_repair_feedback(
    *,
    spans: list[Any],
    advice: list[str],
    signatures: list[str] | None = None,
    max_span_chars: int = 400,
) -> dict[str, Any]:
    """Structured repair payload for the generator hop.

    Integrations MUST append ``prompt_suffix`` to the next generation prompt,
    increment the attempt counter, and after ``max_attempts`` degrade
    ``regenerate`` -> ``flag`` (never unbounded retry).
    """
    span_bits: list[str] = []
    for s in spans[:8]:
        if hasattr(s, "model_dump"):
            d = s.model_dump()
        elif isinstance(s, dict):
            d = s
        else:
            d = {"text": str(s), "reason": "unsupported"}
        text = str(d.get("text") or "")[:120]
        reason = str(d.get("reason") or "unsupported")
        span_bits.append(f"- [{reason}] {text}")
    span_block = "\n".join(span_bits) if span_bits else "- (no spans)"
    advice_block = "\n".join(f"- {a}" for a in advice[:8]) if advice else "- (none)"
    sig_block = ", ".join(signatures or []) or "(none)"
    prompt_suffix = (
        "\n\n[PrismShine repair feedback - revise the answer so every claim is "
        "supported by the provided evidence. Do not invent numbers, entities, or facts.]\n"
        f"Signatures: {sig_block}\n"
        f"Unsupported / unmatched spans:\n{span_block}\n"
        f"Advice:\n{advice_block}\n"
    )[: max_span_chars + 800]
    return {
        "spans": [
            (s.model_dump() if hasattr(s, "model_dump") else s) for s in spans
        ],
        "advice": list(advice),
        "signatures": list(signatures or []),
        "prompt_suffix": prompt_suffix,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
    }


def next_route(
    decision: str,
    attempts: int,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> str:
    """Map verdict decision + attempt count to a route label."""
    if decision == "regenerate":
        if attempts < max_attempts:
            return "regenerate"
        return "flag"
    return decision
