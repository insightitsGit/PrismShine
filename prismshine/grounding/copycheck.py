"""Tier 1: typed fact extraction, normalized matching, arithmetic closure."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from prismshine.models import EvidenceBundle, Signal, Span

_NUM_RE = re.compile(
    r"(?P<currency>[$€£])?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|kg|mg|g|ml|usd|eur|gbp)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?P<d>\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_ID_RE = re.compile(
    r"\b(?P<id>(?:[A-Z]{2,}-\d{2,}|\d{3}-\d{2}-\d{4}|[A-Z]{3}\d{4,}|"
    r"[A-Z0-9]{2,}-\d{3,}[A-Z0-9]*))\b"
)
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

WEIGHTS = {"number": 3.0, "currency": 3.0, "date": 2.0, "id": 3.0, "entity": 1.0}


@dataclass
class Fact:
    kind: str
    raw: str
    normalized: str
    value: float | None = None
    unit: str | None = None
    start: int = 0
    end: int = 0
    derived: bool = False


@dataclass
class CopyCheckResult:
    unmatched_ratio: float
    unmatched: list[Fact] = field(default_factory=list)
    derived: list[Fact] = field(default_factory=list)
    matched: list[Fact] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)


def _parse_number(num_str: str) -> float:
    return float(num_str.replace(",", ""))


def _normalize_date(text: str) -> str | None:
    text = text.strip()
    for fmt in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def extract_facts(text: str, lexicon: set[str] | None = None) -> list[Fact]:
    facts: list[Fact] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0).strip()
        try:
            val = _parse_number(m.group("num"))
        except ValueError:
            continue
        currency = m.group("currency")
        unit = (m.group("unit") or currency or "").lower() or None
        if unit == "%":
            unit = "percent"
        kind = "currency" if currency or (unit in {"usd", "eur", "gbp", "$", "€", "£"}) else "number"
        norm = f"{val:.6g}"
        if unit:
            norm = f"{norm}|{unit}"
        facts.append(
            Fact(
                kind=kind,
                raw=raw,
                normalized=norm,
                value=val,
                unit=unit,
                start=m.start(),
                end=m.end(),
            )
        )
    for m in _DATE_RE.finditer(text):
        raw = m.group("d")
        iso = _normalize_date(raw)
        if not iso:
            continue
        facts.append(
            Fact(
                kind="date",
                raw=raw,
                normalized=iso,
                start=m.start(),
                end=m.end(),
            )
        )
    for m in _ID_RE.finditer(text):
        raw = m.group("id")
        facts.append(
            Fact(
                kind="id",
                raw=raw,
                normalized=raw.upper(),
                start=m.start(),
                end=m.end(),
            )
        )
    for m in _ENTITY_RE.finditer(text):
        raw = m.group(1)
        if lexicon and raw.lower() not in {x.lower() for x in lexicon}:
            # still include capitalized multi-word entities
            pass
        facts.append(
            Fact(
                kind="entity",
                raw=raw,
                normalized=raw.lower(),
                start=m.start(),
                end=m.end(),
            )
        )
    if lexicon:
        lower_text = text.lower()
        for term in lexicon:
            idx = lower_text.find(term.lower())
            if idx >= 0:
                facts.append(
                    Fact(
                        kind="entity",
                        raw=text[idx : idx + len(term)],
                        normalized=term.lower(),
                        start=idx,
                        end=idx + len(term),
                    )
                )
    return facts


def _units_compatible(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True
    aliases = {
        "$": "usd",
        "usd": "usd",
        "€": "eur",
        "eur": "eur",
        "£": "gbp",
        "gbp": "gbp",
        "%": "percent",
        "percent": "percent",
    }
    aa = aliases.get(a.lower(), a.lower())
    bb = aliases.get(b.lower(), b.lower())
    return aa == bb


def _numeric_match(
    fact: Fact, preload_facts: list[Fact], tolerance: float
) -> bool:
    if fact.value is None:
        return False
    for pf in preload_facts:
        if pf.value is None:
            continue
        if not _units_compatible(fact.unit, pf.unit):
            continue
        if pf.value == 0:
            if fact.value == 0:
                return True
            continue
        if abs(fact.value - pf.value) / abs(pf.value) <= tolerance:
            return True
        if tolerance == 0 and fact.value == pf.value:
            return True
    return False


def _arithmetic_closure(fact: Fact, preload_nums: list[Fact]) -> bool:
    if fact.value is None:
        return False
    vals = [p for p in preload_nums if p.value is not None]
    n = len(vals)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = vals[i].value, vals[j].value
            assert a is not None and b is not None
            if fact.unit and vals[i].unit and fact.unit != vals[i].unit:
                if fact.unit not in {"percent", "%"}:
                    continue
            candidates = [a + b, a - b, b - a, a * b]
            if b != 0:
                candidates.append(a / b)
                candidates.append((a - b) / b * 100.0)  # percent change
            if a != 0:
                candidates.append(b / a)
            for c in candidates:
                if abs(c - fact.value) <= max(1e-6, abs(fact.value) * 0.005):
                    return True
    return False


def copycheck(
    bundle: EvidenceBundle,
    *,
    numeric_tolerance: float = 0.005,
    lexicon: set[str] | None = None,
    escalate_derived: bool = False,
) -> CopyCheckResult:
    if not bundle.answer:
        return CopyCheckResult(unmatched_ratio=0.0)

    answer_facts = extract_facts(bundle.answer, lexicon=lexicon)
    preload_text = "\n".join(c.text for c in bundle.preload)
    preload_facts = extract_facts(preload_text, lexicon=lexicon)
    preload_norms = {f.normalized for f in preload_facts}
    preload_nums = [f for f in preload_facts if f.value is not None]

    matched: list[Fact] = []
    unmatched: list[Fact] = []
    derived: list[Fact] = []

    for fact in answer_facts:
        if fact.kind in {"number", "currency"}:
            if _numeric_match(fact, preload_nums, numeric_tolerance):
                matched.append(fact)
            elif _arithmetic_closure(fact, preload_nums):
                fact.derived = True
                derived.append(fact)
            else:
                unmatched.append(fact)
        elif fact.normalized in preload_norms or fact.raw.lower() in preload_text.lower():
            matched.append(fact)
        else:
            unmatched.append(fact)

    total_w = 0.0
    unmatched_w = 0.0
    for fact in answer_facts:
        if fact.derived and not escalate_derived:
            continue
        w = WEIGHTS.get(fact.kind, 1.0)
        total_w += w
        if fact in unmatched or (fact.derived and escalate_derived):
            unmatched_w += w

    ratio = (unmatched_w / total_w) if total_w > 0 else 0.0
    spans = [
        Span(
            start=f.start,
            end=f.end,
            text=f.raw,
            reason=f"unmatched_{f.kind}",
            tier=1,
        )
        for f in unmatched
    ]
    signals = [
        Signal(
            name="grounding.unmatched_ratio",
            tier=1,
            value=ratio,
            weight=0.30,
            spans=spans,
            detail={
                "unmatched_count": len(unmatched),
                "derived_count": len(derived),
                "matched_count": len(matched),
            },
        )
    ]
    if derived:
        signals.append(
            Signal(
                name="grounding.derived_fact_count",
                tier=1,
                value=0.0 if not escalate_derived else min(1.0, len(derived) / 5),
                weight=0.0 if not escalate_derived else 0.15,
                detail={"count": len(derived), "facts": [d.raw for d in derived]},
            )
        )
    return CopyCheckResult(
        unmatched_ratio=ratio,
        unmatched=unmatched,
        derived=derived,
        matched=matched,
        signals=signals,
        spans=spans,
    )
