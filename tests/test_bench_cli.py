"""CLI + runner smoke for prismshine bench."""

from __future__ import annotations

from pathlib import Path

from prismshine.bench import run_bench
from prismshine.cli import main


def test_run_bench_writes_receipts(tmp_path: Path):
    report = run_bench(["cause"], report_dir=tmp_path)
    assert report.passed
    assert (tmp_path / "bench_report.json").is_file()
    assert (tmp_path / "bench_report.md").is_file()
    assert (tmp_path / "cause_side.json").is_file()


def test_cli_bench(tmp_path: Path):
    code = main(["bench", "--suite", "consistency", "--report", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "consistency.json").is_file()
