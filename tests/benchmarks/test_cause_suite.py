"""B1 cause-side suite — POSITIONING ≥90% catch gate."""

from __future__ import annotations

import pytest

from prismshine.bench.suites.cause import run_cause_suite


@pytest.mark.benchmark
def test_cause_side_catch_rate():
    result = run_cause_suite()
    assert result.passed, result.as_dict()
    assert result.gates["catch_rate"] >= 0.90
    assert result.gates["pre_gen_model_calls"] == 0
