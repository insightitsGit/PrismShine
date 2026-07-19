"""B3 latency / cost suite (soft CI budgets)."""

from __future__ import annotations

import pytest

from prismshine.bench.suites.latency import run_latency_suite


@pytest.mark.benchmark
def test_tier0_and_fast_path_latency():
    result = run_latency_suite(iterations=20)
    assert result.passed, result.as_dict()
    assert result.metrics["tier0_p50_ms"] < 50
    assert result.metrics["fast_p50_ms"] < 100
