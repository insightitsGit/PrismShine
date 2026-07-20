"""Datasets for the ChorusGraph+PrismShine runtime suite (H1 / R1 / P1).

No S1 — prompt injection belongs to PrismGuard / the internal stack suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.stack.datasets import build_h1, build_r1


def build_all(n_h1: int = 40, cache_dir: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return hallucination + runtime tracks; P1 is derived by the runner."""
    return {
        "H1": build_h1(n_h1, cache_dir),
        "R1": build_r1(),
    }
