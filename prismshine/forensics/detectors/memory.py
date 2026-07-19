"""Memory-family detectors including exclusive-relation lexicon."""

from __future__ import annotations

import re
from typing import Any

from prismshine.handbook.loader import format_advice
from prismshine.handbook.schema import SignatureDef
from prismshine.models import EvidenceBundle, SignatureHit

# Exclusive relation families: values within a set are mutually exclusive.
EXCLUSIVE_RELATIONS: dict[str, set[str]] = {
    "kinship": {
        "brother",
        "sister",
        "father",
        "mother",
        "son",
        "daughter",
        "uncle",
        "aunt",
        "nephew",
        "niece",
        "husband",
        "wife",
        "spouse",
        "partner",
    },
    "marital": {"married", "single", "divorced", "widowed", "engaged"},
    "vital": {"alive", "dead", "deceased"},
    "employment": {"employed", "unemployed", "retired"},
    "approval": {"approved", "denied", "rejected", "accepted"},
}

_SUBJECT_PAT = re.compile(
    r"(?P<subject>[A-Z][a-zA-Z0-9_.-]*(?:\s+[A-Z][a-zA-Z0-9_.-]*)*)\s+"
    r"(?:is|was|'s)\s+(?:my\s+)?(?P<value>[a-z]+)",
    re.IGNORECASE,
)


def conflict_served(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    conflicts = list(bundle.node_state.get("memory_conflicts") or [])
    for c in conflicts:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(
                    sig.advice,
                    subject=c.get("subject", ""),
                    relation=c.get("relation", ""),
                ),
                evidence=dict(c),
                signal_value=sig.signal_value,
            )
        )
    return hits


def staged_fact(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    for fact in bundle.node_state.get("staged_facts") or []:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(sig.advice, subject=fact.get("subject", "")),
                evidence=dict(fact),
                signal_value=sig.signal_value,
            )
        )
    for chunk in bundle.preload:
        if chunk.source == "memory" and chunk.metadata.get("staged") is True:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice, subject=chunk.metadata.get("subject", chunk.chunk_id)
                    ),
                    evidence={"chunk_id": chunk.chunk_id, **chunk.metadata},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def expired_fact(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    query_time = bundle.node_state.get("query_time") or bundle.node_state.get("as_of")
    for fact in bundle.node_state.get("expired_facts") or []:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(
                    sig.advice,
                    subject=fact.get("subject", ""),
                    valid_to=fact.get("valid_to"),
                    query_time=query_time,
                ),
                evidence=dict(fact),
                signal_value=sig.signal_value,
            )
        )
    for chunk in bundle.preload:
        if chunk.source == "memory" and chunk.metadata.get("expired") is True:
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        subject=chunk.metadata.get("subject", chunk.chunk_id),
                        valid_to=chunk.metadata.get("valid_to"),
                        query_time=query_time,
                    ),
                    evidence={"chunk_id": chunk.chunk_id, **chunk.metadata},
                    signal_value=sig.signal_value,
                )
            )
    return hits


def _extract_exclusive_assertions(texts: list[str]) -> list[tuple[str, str, str]]:
    """Return (subject, relation_family, value) triples."""
    found: list[tuple[str, str, str]] = []
    value_to_family = {
        v: fam for fam, values in EXCLUSIVE_RELATIONS.items() for v in values
    }
    for text in texts:
        for m in _SUBJECT_PAT.finditer(text):
            subject = m.group("subject").strip()
            value = m.group("value").lower()
            fam = value_to_family.get(value)
            if fam:
                found.append((subject.lower(), fam, value))
        # also scan bare kinship phrases: "my brother" with nearby proper noun is harder;
        # handle "Person A is my sister" style already covered.
        lower = text.lower()
        for fam, values in EXCLUSIVE_RELATIONS.items():
            for v in values:
                if f"my {v}" in lower or f"is {v}" in lower or f"was {v}" in lower:
                    # try to find a capitalized subject in the same sentence
                    for sent in re.split(r"[.!?]", text):
                        if v in sent.lower():
                            subj_m = re.search(
                                r"\b([A-Z][a-zA-Z0-9_.-]*(?:\s+[A-Z][a-zA-Z0-9_.-]*)*)\b",
                                sent,
                            )
                            if subj_m:
                                found.append((subj_m.group(1).lower(), fam, v))
    return found


def conflicting_preload(
    bundle: EvidenceBundle, params: dict[str, Any], sig: SignatureDef
) -> list[SignatureHit]:
    hits: list[SignatureHit] = []
    # Cortex path
    for c in bundle.node_state.get("preload_conflicts") or []:
        hits.append(
            SignatureHit(
                id=sig.id,
                title=sig.title,
                severity=sig.severity,
                scope=sig.scope,
                advice=format_advice(
                    sig.advice,
                    subject=c.get("subject", ""),
                    relation=c.get("relation", ""),
                    value_a=c.get("value_a", ""),
                    value_b=c.get("value_b", ""),
                ),
                evidence=dict(c),
                signal_value=sig.signal_value,
            )
        )

    texts = [
        c.text
        for c in bundle.preload
        if c.source in {"history", "memory", "retrieval", "system"}
    ]
    assertions = _extract_exclusive_assertions(texts)
    by_key: dict[tuple[str, str], set[str]] = {}
    for subject, fam, value in assertions:
        by_key.setdefault((subject, fam), set()).add(value)
    for (subject, fam), values in by_key.items():
        if len(values) >= 2:
            vals = sorted(values)
            hits.append(
                SignatureHit(
                    id=sig.id,
                    title=sig.title,
                    severity=sig.severity,
                    scope=sig.scope,
                    advice=format_advice(
                        sig.advice,
                        subject=subject,
                        relation=fam,
                        value_a=vals[0],
                        value_b=vals[1],
                    ),
                    evidence={
                        "subject": subject,
                        "relation": fam,
                        "value_a": vals[0],
                        "value_b": vals[1],
                        "all_values": vals,
                    },
                    signal_value=sig.signal_value,
                )
            )
    return hits
