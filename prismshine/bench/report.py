"""Write machine-readable JSON + human markdown receipts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SuiteResult:
    name: str
    passed: bool
    gates: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    cases: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    competitor_baseline: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "gates": self.gates,
            "metrics": self.metrics,
            "cases": self.cases,
            "notes": self.notes,
            "competitor_baseline": self.competitor_baseline
            or {
                "status": "literature / not run",
                "detail": (
                    "External products (RAGAS, LettuceDetect, Blue Guardrails) are "
                    "not executed here. Claims vs them require a local adapter run "
                    "or published numbers cited separately."
                ),
            },
        }


@dataclass
class BenchReport:
    suites: dict[str, SuiteResult] = field(default_factory=dict)
    version: str = "0.1.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.suites.values()) if self.suites else False

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "passed": self.passed,
            "suites": {k: v.as_dict() for k, v in self.suites.items()},
        }

    def to_markdown(self) -> str:
        lines = [
            "# PrismShine benchmark receipt",
            "",
            f"- Created: `{self.created_at}`",
            f"- Overall: **{'PASS' if self.passed else 'FAIL'}**",
            "",
        ]
        for name, suite in self.suites.items():
            lines.append(f"## {name}")
            lines.append("")
            lines.append(f"- Suite: **{'PASS' if suite.passed else 'FAIL'}**")
            if suite.gates:
                lines.append("- Gates:")
                for g, val in suite.gates.items():
                    lines.append(f"  - `{g}`: `{val}`")
            if suite.metrics:
                lines.append("- Metrics:")
                for k, v in suite.metrics.items():
                    lines.append(f"  - `{k}`: `{v}`")
            if suite.notes:
                lines.append("- Notes:")
                for n in suite.notes:
                    lines.append(f"  - {n}")
            cb = suite.competitor_baseline or {}
            if cb:
                lines.append(
                    f"- Competitor baseline: `{cb.get('status', 'literature / not run')}`"
                )
            lines.append("")
        lines.append(
            "Claims without a green receipt are banned - see `docs/POSITIONING.md`."
        )
        return "\n".join(lines) + "\n"

    def write(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict()
        json_path = root / "bench_report.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        (root / "bench_report.md").write_text(self.to_markdown(), encoding="utf-8")
        for name, suite in self.suites.items():
            (root / f"{name}.json").write_text(
                json.dumps(suite.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
        return json_path
