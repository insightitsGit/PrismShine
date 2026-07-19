"""Generic dict → EvidenceBundle adapter."""

from __future__ import annotations

from typing import Any

from prismshine.evidence.builder import bundle_from_dict
from prismshine.models import EvidenceBundle


def from_dict(data: dict[str, Any]) -> tuple[EvidenceBundle, list[str]]:
    """Build an EvidenceBundle from plain dicts (standalone / custom runtimes)."""
    return bundle_from_dict(data)
