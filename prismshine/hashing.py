"""Canonical serialization and content-addressed hashing."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from prismshine.models import EvidenceBundle


def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 12)
    if isinstance(value, dict):
        return {k: _normalize(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON bytes with sorted keys and normalized floats."""
    if isinstance(obj, EvidenceBundle):
        data = obj.model_dump(mode="json")
    elif hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
    else:
        data = obj
    normalized = _normalize(data)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def evidence_hash(bundle: EvidenceBundle) -> str:
    return hashlib.sha256(canonical_bytes(bundle)).hexdigest()


def content_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(canonical_bytes(part))
        h.update(b"\x00")
    return h.hexdigest()
