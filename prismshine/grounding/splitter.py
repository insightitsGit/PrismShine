"""Deterministic zero-dep sentence splitter."""

from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?<=\n)\s*")
_BOILERPLATE = re.compile(
    r"^(hi|hello|thanks|thank you|regards|best|sure|ok|okay)[.!]?\s*$",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    return parts if parts else [text]


def is_boilerplate(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 12:
        return True
    if _BOILERPLATE.match(s):
        return True
    content = re.findall(r"[a-zA-Z]{3,}", s)
    return len(content) < 2


COMPOSITE_CUES = (
    "more than",
    "less than",
    "compared to",
    "compared with",
    "total",
    "average",
    "both",
    "sum of",
    "combined",
    "versus",
    " vs ",
)


def is_composite(sentence: str) -> bool:
    lower = sentence.lower()
    if any(c in lower for c in COMPOSITE_CUES):
        return True
    # enumeration pattern: "A, B, and C"
    if re.search(r",\s+[^,]+,\s+and\s+", lower):
        return True
    return False
