"""B4 consistency dual-rail."""

from __future__ import annotations

import pytest

from prismshine.bench.suites.consistency import run_consistency_suite


@pytest.mark.benchmark
def test_consistency_dual_rail():
    result = run_consistency_suite()
    assert result.passed, result.as_dict()
    assert result.gates["detection_catch_rate_when_prevention_off"] == 1.0
