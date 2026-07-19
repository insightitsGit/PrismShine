"""Backward-compatible alias — grounding suite owns the quality gate."""

from __future__ import annotations

from prismshine.bench.suites.grounding import run_grounding_suite


def test_synthetic_detection_f1():
    result = run_grounding_suite()
    assert result.metrics["synthetic"]["f1"] >= 0.5
