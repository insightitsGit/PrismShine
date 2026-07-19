"""Verdict records, replay helpers, and HRI rolling metrics."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from prismshine.hashing import evidence_hash
from prismshine.models import EvidenceBundle, ShineVerdict


def new_verdict_id() -> str:
    # deterministic-enough for uniqueness; not part of golden equality
    return uuid.uuid4().hex


@dataclass
class VerdictRecord:
    verdict: ShineVerdict
    recorded_at: float
    bundle_hash: str
    meta: dict[str, Any] = field(default_factory=dict)


class AuditLog:
    def __init__(self, maxlen: int = 10_000) -> None:
        self._records: deque[VerdictRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._fused_window: deque[float] = deque(maxlen=500)
        self._sig_counts: dict[str, int] = {}
        self._escalations = 0
        self._total = 0

    def record(self, bundle: EvidenceBundle, verdict: ShineVerdict, **meta: Any) -> VerdictRecord:
        rec = VerdictRecord(
            verdict=verdict,
            recorded_at=time.time(),
            bundle_hash=evidence_hash(bundle),
            meta=meta,
        )
        with self._lock:
            self._records.append(rec)
            self._fused_window.append(verdict.fused_score)
            self._total += 1
            if verdict.tier_reached >= 3:
                self._escalations += 1
            for sig in verdict.signatures:
                self._sig_counts[sig.id] = self._sig_counts.get(sig.id, 0) + 1
        return rec

    def replay(self, bundle_hash: str) -> list[ShineVerdict]:
        with self._lock:
            return [r.verdict for r in self._records if r.bundle_hash == bundle_hash]

    def hri(self) -> float:
        """Hallucination Risk Index 0–100."""
        with self._lock:
            if not self._fused_window:
                return 0.0
            avg_fused = sum(self._fused_window) / len(self._fused_window)
            esc_rate = self._escalations / max(self._total, 1)
            top_sig = max(self._sig_counts.values()) / max(self._total, 1) if self._sig_counts else 0.0
            score = 100.0 * (0.6 * avg_fused + 0.25 * esc_rate + 0.15 * top_sig)
            return max(0.0, min(100.0, score))

    def metrics(self) -> dict[str, Any]:
        return {
            "hri": self.hri(),
            "total": self._total,
            "escalations": self._escalations,
            "signature_counts": dict(self._sig_counts),
        }
