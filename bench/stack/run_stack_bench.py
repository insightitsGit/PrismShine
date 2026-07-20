"""HTTP orchestrator for the Stack suite."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from bench.stack.datasets import build_all


def _f1(records: list[dict[str, Any]], positive: str) -> dict[str, float | int]:
    tp = sum(r["gold"] == positive and r.get("label") == positive for r in records)
    fp = sum(r["gold"] != positive and r.get("label") == positive for r in records)
    fn = sum(r["gold"] == positive and r.get("label") != positive for r in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }


def _runtime_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    failing = [r for r in records if r["gold"] == "runtime_fail"]
    clean = [r for r in records if r["gold"] == "runtime_ok"]
    caught = sum(r.get("label") == "runtime_fail" for r in failing)
    false_alarms = sum(r.get("label") == "runtime_fail" for r in clean)
    return {
        "n_runtime_fail": len(failing),
        "n_runtime_ok": len(clean),
        "catch_rate": round(caught / len(failing), 4) if failing else 0.0,
        "false_alarm": round(false_alarms / len(clean), 4) if clean else 0.0,
    }


def _latency_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    values = sorted(float(r.get("latency_ms") or 0.0) for r in records) or [0.0]
    return {
        "n": len(records),
        "p50_ms": round(statistics.median(values), 2),
        "p95_ms": round(values[min(len(values) - 1, int(0.95 * len(values)))], 2),
        "llm_calls_total": sum(int(r.get("llm_calls") or 0) for r in records),
        "cost_usd_total": round(sum(float(r.get("cost_usd") or 0.0) for r in records), 6),
    }


def _request(client: httpx.Client, url: str, sample: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: sample[key]
        for key in ("id", "track", "question", "context", "answer", "evidence", "gold")
        if key in sample
    }
    primary = client.post(f"{url.rstrip('/')}/stack_evaluate", json=payload)
    if primary.status_code == 404:
        primary = client.post(f"{url.rstrip('/')}/evaluate", json=payload)
    primary.raise_for_status()
    return primary.json()


def run_system(
    name: str, url: str, datasets: dict[str, list[dict[str, Any]]], out_dir: Path, timeout: float
) -> dict[str, list[dict[str, Any]]]:
    results = {track: [] for track in datasets}
    raw_path = out_dir / f"raw_{name}.jsonl"
    with httpx.Client(timeout=timeout) as client, raw_path.open("w", encoding="utf-8") as fh:
        for track, samples in datasets.items():
            for sample in samples:
                started = time.perf_counter()
                try:
                    body = _request(client, url, sample)
                except Exception as exc:  # network/service errors remain explicit
                    body = {
                        "id": sample["id"], "track": track, "decision": "error", "label": "n/a",
                        "risk": 1.0, "latency_ms": (time.perf_counter() - started) * 1000,
                        "llm_calls": 0, "cost_usd": 0.0, "resolution_gate": None,
                        "components": {}, "saw_evidence": False, "error": str(exc)[:300],
                    }
                record = {**body, "gold": sample["gold"], "system": name, "track": track}
                results[track].append(record)
                fh.write(json.dumps(record) + "\n")
    return results


def _score(results: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float | int]]:
    all_records = [record for track in results.values() for record in track]
    return {
        "S1": _f1(results["S1"], "attack"),
        "H1": _f1(results["H1"], "hallucinated"),
        # Evidence-aware by construction; never combine with content F1.
        "R1_evidence_aware": _runtime_metrics(results["R1"]),
        "P1": _latency_metrics(all_records),
    }


def _scoreboard(summary: dict[str, Any]) -> str:
    lines = [
        "# Stack suite scoreboard",
        "",
        "R1 is an **evidence-aware** runtime track and is intentionally separate from S1/H1.",
        "",
        "| system | S1 attack F1 | H1 hallucination F1 | R1 catch rate (evidence-aware) | R1 false alarm | P1 p50 ms | P1 p95 ms | LLM calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, scores in summary["systems"].items():
        s1, h1, r1, p1 = scores["S1"], scores["H1"], scores["R1_evidence_aware"], scores["P1"]
        lines.append(
            f"| {name} | {s1['f1']} | {h1['f1']} | {r1['catch_rate']} | {r1['false_alarm']} "
            f"| {p1['p50_ms']} | {p1['p95_ms']} | {p1['llm_calls_total']} |"
        )
    lines += [
        "",
        "P1 is derived from all evaluated S1/H1/R1 requests. Latency is shim-internal.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, help="JSON mapping of system name to base URL")
    parser.add_argument("--n-h1", type=int, default=40, help="HaluEval source rows (two samples each)")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="bench/stack/results/run")
    args = parser.parse_args()

    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = build_all(n_h1=args.n_h1)
    summary: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {track: len(samples) for track, samples in datasets.items()},
        "r1_note": "Evidence-aware track; compare only within its separately labeled column.",
        "systems": {},
    }
    for name, url in targets.items():
        try:
            health = httpx.get(f"{url.rstrip('/')}/health", timeout=30).json()
            print(f"health {name}: {health}")
        except Exception as exc:  # health should not prevent the run receipt
            print(f"health {name}: FAILED {exc}")
        records = run_system(name, url, datasets, out_dir, args.timeout)
        summary["systems"][name] = _score(records)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "scoreboard.md").write_text(_scoreboard(summary), encoding="utf-8")
    print(f"Wrote {out_dir / 'summary.json'} and {out_dir / 'scoreboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
