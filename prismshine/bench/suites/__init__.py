"""Benchmark suite runners."""

from __future__ import annotations

from prismshine.bench.suites.cause import run_cause_suite
from prismshine.bench.suites.consistency import run_consistency_suite
from prismshine.bench.suites.grounding import run_grounding_suite
from prismshine.bench.suites.latency import run_latency_suite

SUITE_RUNNERS = {
    "cause": run_cause_suite,
    "grounding": run_grounding_suite,
    "latency": run_latency_suite,
    "consistency": run_consistency_suite,
}

__all__ = [
    "SUITE_RUNNERS",
    "run_cause_suite",
    "run_grounding_suite",
    "run_latency_suite",
    "run_consistency_suite",
]
