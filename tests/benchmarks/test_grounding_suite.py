"""B2 grounding quality — synthetic hard cases + span baseline."""

from __future__ import annotations

import pytest

from prismshine.bench.suites.grounding import run_grounding_suite


@pytest.mark.benchmark
def test_grounding_synthetic_f1():
    result = run_grounding_suite()
    assert result.passed, result.as_dict()
    assert result.gates["synthetic_f1"] >= 0.85
