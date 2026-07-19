"""Orchestrate benchmark suites and write receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from prismshine import __version__
from prismshine.bench.report import BenchReport, SuiteResult
from prismshine.bench.suites import SUITE_RUNNERS
from prismshine.gate import ShineGate

ALL_SUITES = ("cause", "grounding", "latency", "consistency")


def run_suite(name: str, *, gate: ShineGate | None = None) -> SuiteResult:
    key = name.strip().lower()
    if key == "cause_side":
        key = "cause"
    if key == "latency_cost":
        key = "latency"
    if key not in SUITE_RUNNERS:
        raise ValueError(
            f"Unknown suite {name!r}; choose from {sorted(SUITE_RUNNERS)} or 'all'"
        )
    return SUITE_RUNNERS[key](gate=gate)


def run_bench(
    suites: Iterable[str] | None = None,
    *,
    report_dir: str | Path | None = None,
    gate: ShineGate | None = None,
) -> BenchReport:
    requested = list(suites) if suites is not None else list(ALL_SUITES)
    if any(s.lower() == "all" for s in requested):
        requested = list(ALL_SUITES)

    report = BenchReport(version=__version__)
    for name in requested:
        suite = run_suite(name, gate=gate)
        report.suites[suite.name] = suite

    if report_dir is not None:
        report.write(report_dir)
    return report
