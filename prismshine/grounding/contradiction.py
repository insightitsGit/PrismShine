"""Tier-2 contradiction-cue screen (negation asymmetry + opposite lexicon)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from prismshine.models import Span

# Phrase-level negation (order matters: longer first)
NEGATION_PHRASES = (
    " not ",
    " never ",
    " no longer ",
    " without ",
    "n't ",
    " cannot ",
    " can't ",
    " does not ",
    " do not ",
    " did not ",
    " is not ",
    " are not ",
    " was not ",
    " were not ",
    " isn't ",
    " aren't ",
    " wasn't ",
    " weren't ",
    " won't ",
    " wouldn't ",
    " shouldn't ",
    " couldn't ",
    " no evidence ",
    " fails to ",
    " failed to ",
    " rather than ",
    " instead of ",
)

# Antonym / polarity pairs — clinical, finance, legal, general
OPPOSITE_PAIRS: list[tuple[str, str]] = [
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("approve", "deny"),
    ("approved", "denied"),
    ("safe", "unsafe"),
    ("safer", "riskier"),
    ("allow", "forbid"),
    ("allowed", "forbidden"),
    ("permitted", "prohibited"),
    ("rise", "fall"),
    ("rising", "falling"),
    ("accept", "reject"),
    ("accepted", "rejected"),
    ("include", "exclude"),
    ("included", "excluded"),
    ("success", "failure"),
    ("successful", "unsuccessful"),
    ("pass", "fail"),
    ("passed", "failed"),
    ("profit", "loss"),
    ("profitable", "unprofitable"),
    ("gain", "loss"),
    ("bullish", "bearish"),
    ("long", "short"),
    ("credit", "debit"),
    ("surplus", "deficit"),
    ("guilty", "innocent"),
    ("liable", "not liable"),
    ("valid", "invalid"),
    ("effective", "ineffective"),
    ("efficacious", "inefficacious"),
    ("indicated", "contraindicated"),
    ("therapeutic", "toxic"),
    ("beneficial", "harmful"),
    ("recommend", "advise"),
    ("recommended", "contraindicated"),
    ("true", "false"),
    ("correct", "incorrect"),
    ("present", "absent"),
    ("available", "unavailable"),
    ("open", "closed"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("start", "stop"),
    ("before", "after"),
    ("higher", "lower"),
    ("more", "less"),
    ("above", "below"),
    ("positive", "negative"),
    ("confirm", "deny"),
    ("confirmed", "denied"),
    ("support", "oppose"),
    ("supported", "opposed"),
]

# Explicit polarity phrase pairs (answer vs preload)
POLARITY_PHRASE_PAIRS: list[tuple[str, str]] = [
    ("is safe", "is not safe"),
    ("are safe", "are not safe"),
    ("is effective", "is not effective"),
    ("is approved", "is not approved"),
    ("is indicated", "is contraindicated"),
    ("was approved", "was denied"),
    ("did increase", "did decrease"),
    ("has increased", "has decreased"),
    ("showed profit", "showed loss"),
    ("is guilty", "is innocent"),
    ("is liable", "is not liable"),
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
    if any(n in padded for n in NEGATION_PHRASES):
        return True
    return bool(re.search(r"\bno\b", padded))


def _polarity_phrase_hit(a: str, b: str) -> str | None:
    al, bl = a.lower(), b.lower()
    for x, y in POLARITY_PHRASE_PAIRS:
        if (x in al and y in bl) or (y in al and x in bl):
            return f"polarity:{x}/{y}"
    return None


def _opposite_hit(a: str, b: str, pairs: list[tuple[str, str]]) -> str | None:
    al, bl = a.lower(), b.lower()
    for x, y in pairs:
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
    asymmetry, polarity phrases, or opposite-verb/adjective cues.
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
            reason = _polarity_phrase_hit(sent, chunk_text)
            if reason is None:
                reason = _opposite_hit(sent, chunk_text, pairs)
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
