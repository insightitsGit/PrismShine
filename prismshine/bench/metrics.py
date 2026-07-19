"""Shared classification / latency / cost metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, *, y_true: bool, y_pred: bool) -> None:
        if y_true and y_pred:
            self.tp += 1
        elif y_true and not y_pred:
            self.fn += 1
        elif (not y_true) and y_pred:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-12)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / max(total, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Mann–Whitney AUROC; None if only one class present."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    total = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                total += 1.0
            elif p == n:
                total += 0.5
    return total / (len(pos) * len(neg))


@dataclass
class CostModel:
    """Compare Shine ms/check vs LLM-judge $/call (literature proxy)."""

    judge_usd_per_call: float = 0.002
    shine_usd_per_cpu_check: float = 0.0

    def compare(self, *, n_checks: int, judge_escalation_rate: float) -> dict[str, Any]:
        judge_all = n_checks * self.judge_usd_per_call
        shine = n_checks * self.shine_usd_per_cpu_check + (
            n_checks * judge_escalation_rate * self.judge_usd_per_call
        )
        return {
            "n_checks": n_checks,
            "judge_usd_per_1k_all_traffic": round(1000 * self.judge_usd_per_call, 4),
            "shine_usd_per_1k": round(1000 * (shine / max(n_checks, 1)), 4),
            "judge_escalation_rate": round(judge_escalation_rate, 4),
            "savings_vs_judge_all_usd_per_1k": round(
                1000 * self.judge_usd_per_call - 1000 * (shine / max(n_checks, 1)), 4
            ),
            "notes": [
                "Judge cost is a configurable proxy (default $0.002/call).",
                "Shine fast path assumes $0 CPU marginal; only escalations bill.",
                "Competitor cells without a local run stay 'literature / not run'.",
            ],
        }


@dataclass
class CaseResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


def rate(passed: Iterable[bool]) -> float:
    xs = list(passed)
    return sum(1 for x in xs if x) / max(len(xs), 1)
