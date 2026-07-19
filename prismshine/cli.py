"""Console script: prismshine capabilities | verify | calibrate | bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prismshine import __version__
from prismshine.bench.runner import ALL_SUITES, run_bench
from prismshine.calibrate import calibrate_dir
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate


def cmd_capabilities(args: argparse.Namespace) -> int:
    gate = ShineGate.build(
        profile=args.profile,
        strictness=args.strictness,
        handbook=args.handbook,
    )
    print(json.dumps(gate.capabilities(), indent=2, sort_keys=True))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.bundle)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "bundle" in data:
        data = data["bundle"]
    bundle, feedback = bundle_from_dict(data)
    gate = ShineGate.build(
        profile=args.profile,
        strictness=args.strictness,
        handbook=args.handbook,
    )
    verdict = gate.verify(bundle)
    out = verdict.model_dump(mode="json")
    if args.verbose:
        out["_feedback"] = feedback
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if verdict.decision in {"pass", "flag"} else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    gate = ShineGate.build(profile=args.profile, strictness=args.strictness)
    report = calibrate_dir(args.dir, mode=args.mode, gate=gate)
    payload = {
        "mode": report.mode,
        "n_samples": report.n_samples,
        "auroc": report.auroc,
        "precision_at_bands": report.precision_at_bands,
        "recall_at_bands": report.recall_at_bands,
        "thresholds": report.thresholds,
        "version": report.version,
        "overlay": report.to_yaml_overlay(),
        "notes": report.notes,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    suites = [args.suite] if args.suite != "all" else list(ALL_SUITES)
    report_dir = Path(args.report)
    report = run_bench(suites, report_dir=report_dir)
    # Windows consoles may be cp1252 — keep stdout ASCII-safe
    text = report.to_markdown().encode("ascii", errors="replace").decode("ascii")
    print(text)
    print(f"Wrote receipts to {report_dir.resolve()}")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prismshine",
        description=f"PrismShine anti-hallucination gate (v{__version__})",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_caps = sub.add_parser("capabilities", help="Print gate capability report")
    p_caps.add_argument("--profile", default="default")
    p_caps.add_argument("--strictness", default="standard")
    p_caps.add_argument("--handbook", default="builtin")
    p_caps.set_defaults(func=cmd_capabilities)

    p_ver = sub.add_parser("verify", help="Verify a bundle JSON file")
    p_ver.add_argument("bundle", help="Path to EvidenceBundle JSON")
    p_ver.add_argument("--profile", default="default")
    p_ver.add_argument("--strictness", default="standard")
    p_ver.add_argument("--handbook", default="builtin")
    p_ver.add_argument("-v", "--verbose", action="store_true")
    p_ver.set_defaults(func=cmd_verify)

    p_cal = sub.add_parser("calibrate", help="Fit thresholds from a directory of bundles")
    p_cal.add_argument("dir", help="Directory of JSON bundles")
    p_cal.add_argument("--mode", choices=["synthetic", "labeled"], default="synthetic")
    p_cal.add_argument("--profile", default="default")
    p_cal.add_argument("--strictness", default="standard")
    p_cal.add_argument("--out", default=None, help="Write report JSON to path")
    p_cal.set_defaults(func=cmd_calibrate)

    p_bench = sub.add_parser(
        "bench",
        help="Run receipt-backed benchmark suites (see docs/BENCHMARKS.md)",
    )
    p_bench.add_argument(
        "--suite",
        default="all",
        choices=["all", "cause", "grounding", "latency", "consistency"],
        help="Suite to run (default: all)",
    )
    p_bench.add_argument(
        "--report",
        default="benchmarks/reports",
        help="Directory for JSON/MD receipts",
    )
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
