"""Tier-2 contradiction-cue screen (negation asymmetry + opposite lexicon)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from prismshine.models import Span

NEGATION_CUES = (" not ", " never ", " no longer ", " without ", "n't ", " cannot ", " can't ")

OPPOSITE_PAIRS: list[tuple[str, str]] = [
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("approve", "deny"),
    ("approved", "denied"),
    ("safe", "unsafe"),
    ("allow", "forbid"),
    ("allowed", "forbidden"),
    ("rise", "fall"),
    ("rising", "falling"),
    ("accept", "reject"),
    ("accepted", "rejected"),
    ("include", "exclude"),
    ("success", "failure"),
    ("pass", "fail"),
]


@dataclass
class ContradictionCue:
    sentence: str
    sentence_index: int
    chunk_id: str
    reason: str
    span: Span


def _has_negation(text: str) -> bool:
    padded = f" {text.lower()} "
    return any(n in padded for n in NEGATION_CUES) or bool(
        re.search(r"\bno\b", padded)
    )


def _opposite_hit(a: str, b: str) -> str | None:
    al, bl = a.lower(), b.lower()
    for x, y in OPPOSITE_PAIRS:
        if (x in al and y in bl) or (y in al and x in bl):
            return f"opposite:{x}/{y}"
    return None


def screen_contradictions(
    sentences: list[str],
    best_chunk_texts: list[tuple[str, str]],
    *,
    sentence_offsets: list[tuple[int, int]] | None = None,
    extra_pairs: list[tuple[str, str]] | None = None,
) -> list[ContradictionCue]:
    """
    For each sentence that has a best-supporting chunk, detect negation
    asymmetry or opposite-verb/adjective cues.
    best_chunk_texts: parallel list of (chunk_id, chunk_text) per sentence.
    """
    pairs = list(OPPOSITE_PAIRS)
    if extra_pairs:
        pairs.extend(extra_pairs)
    cues: list[ContradictionCue] = []
    for i, sent in enumerate(sentences):
        if i >= len(best_chunk_texts):
            break
        chunk_id, chunk_text = best_chunk_texts[i]
        if not chunk_text:
            continue
        neg_s = _has_negation(sent)
        neg_c = _has_negation(chunk_text)
        reason = None
        if neg_s != neg_c:
            reason = "negation_asymmetry"
        else:
            # local opposite check using extended pairs
            al, bl = sent.lower(), chunk_text.lower()
            for x, y in pairs:
                if (x in al and y in bl) or (y in al and x in bl):
                    reason = f"opposite:{x}/{y}"
                    break
        if reason:
            if sentence_offsets and i < len(sentence_offsets):
                start, end = sentence_offsets[i]
            else:
                start, end = 0, len(sent)
            cues.append(
                ContradictionCue(
                    sentence=sent,
                    sentence_index=i,
                    chunk_id=chunk_id,
                    reason=reason,
                    span=Span(
                        start=start,
                        end=end,
                        text=sent,
                        reason=f"contradiction_cue:{reason}",
                        tier=2,
                    ),
                )
            )
    return cues
