"""Comparative benchmark runner (docs/BENCHMARKS.md § Comparative suite).

Tracks:
  B1 content-only  — HaluEval QA pairs (grounded + hallucinated per row)
  B2 numbers slice — digit-perturbed grounded answers (fabricated-figure detection)

Usage:
  python run_bench.py --targets targets.json --n 100 --ragas-limit 30 --out results/run1

targets.json: {"prismshine-fast": "http://<fqdn>:8000", "hhem": "...", "ragas": "..."}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import urllib.request
from pathlib import Path

import httpx

HALUEVAL_QA = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
WARMUP = 5  # excluded from latency stats (models pre-baked, small warmup suffices)
SEED = 42


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def load_halueval(cache_dir: Path, n_rows: int) -> list[dict]:
    cache = cache_dir / "halueval_qa.jsonl"
    if not cache.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading HaluEval QA -> {cache}")
        with urllib.request.urlopen(HALUEVAL_QA, timeout=120) as r:
            cache.write_bytes(r.read())
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(SEED).shuffle(rows)
    return rows[:n_rows]


def build_b1(rows: list[dict]) -> list[dict]:
    """One grounded + one hallucinated sample per HaluEval row."""
    samples = []
    for i, row in enumerate(rows):
        ctx = [row["knowledge"]]
        q = row["question"]
        samples.append(
            {"id": f"b1-{i}-g", "track": "B1", "question": q, "context": ctx,
             "answer": row["right_answer"], "label": "grounded"}
        )
        samples.append(
            {"id": f"b1-{i}-h", "track": "B1", "question": q, "context": ctx,
             "answer": row["hallucinated_answer"], "label": "hallucinated"}
        )
    return samples


_DIGIT = re.compile(r"\d+")


def build_b2(rows: list[dict], limit: int) -> list[dict]:
    """Fabricated-figure slice: perturb a digit in numeric grounded answers."""
    rng = random.Random(SEED)
    samples = []
    for i, row in enumerate(rows):
        ans = row["right_answer"]
        m = _DIGIT.search(ans)
        if not m or _DIGIT.search(row["knowledge"]) is None:
            continue
        val = int(m.group(0))
        fabricated = _DIGIT.sub(str(val + rng.randint(2, 9) * max(1, 10 ** (len(m.group(0)) - 1))), ans, count=1)
        ctx = [row["knowledge"]]
        samples.append(
            {"id": f"b2-{i}-g", "track": "B2", "question": row["question"],
             "context": ctx, "answer": ans, "label": "grounded"}
        )
        samples.append(
            {"id": f"b2-{i}-h", "track": "B2", "question": row["question"],
             "context": ctx, "answer": fabricated, "label": "hallucinated"}
        )
        if len(samples) >= 2 * limit:
            break
    return samples


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def prf1(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["gold"] == "hallucinated" and r["label"] == "hallucinated")
    fp = sum(1 for r in results if r["gold"] == "grounded" and r["label"] == "hallucinated")
    tn = sum(1 for r in results if r["gold"] == "grounded" and r["label"] == "grounded")
    fn = sum(1 for r in results if r["gold"] == "hallucinated" and r["label"] == "grounded")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / max(len(results), 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4)}


def auroc(results: list[dict]) -> float:
    """Rank-based AUROC over risk scores (positive class = hallucinated)."""
    pos = [r["risk"] for r in results if r["gold"] == "hallucinated"]
    neg = [r["risk"] for r in results if r["gold"] == "grounded"]
    if not pos or not neg:
        return 0.0
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def latency_stats(results: list[dict]) -> dict:
    lats = sorted(r["latency_ms"] for r in results[WARMUP:])
    if not lats:
        lats = sorted(r["latency_ms"] for r in results) or [0.0]
    return {
        "p50_ms": round(statistics.median(lats), 2),
        "p95_ms": round(lats[min(len(lats) - 1, int(0.95 * len(lats)))], 2),
        "mean_ms": round(statistics.fmean(lats), 2),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_system(name: str, url: str, samples: list[dict], out_dir: Path, timeout: float) -> list[dict]:
    results = []
    raw_path = out_dir / f"raw_{name}.jsonl"
    with httpx.Client(timeout=timeout) as client, raw_path.open("w", encoding="utf-8") as fh:
        for k, s in enumerate(samples):
            payload = {"id": s["id"], "question": s["question"],
                       "context": s["context"], "answer": s["answer"]}
            try:
                resp = client.post(f"{url}/evaluate", json=payload)
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:  # noqa: BLE001
                body = {"id": s["id"], "risk": 0.5, "label": "grounded",
                        "latency_ms": 0.0, "error": str(exc)[:200]}
            rec = {**body, "gold": s["label"], "track": s["track"], "system": name}
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if (k + 1) % 25 == 0:
                print(f"  {name}: {k + 1}/{len(samples)}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="JSON file {system: base_url}")
    ap.add_argument("--n", type=int, default=100, help="HaluEval rows (x2 samples) for B1")
    ap.add_argument("--b2", type=int, default=25, help="numeric pairs for B2")
    ap.add_argument("--ragas-limit", type=int, default=30, help="max samples for ragas (slow judge)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="results/run")
    args = ap.parse_args()

    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(__file__).parent / "data"

    rows = load_halueval(cache, args.n)
    b1 = build_b1(rows)
    b2 = build_b2(load_halueval(cache, 2000), args.b2)
    print(f"B1: {len(b1)} samples | B2: {len(b2)} samples | systems: {list(targets)}")

    # health checks
    for name, url in targets.items():
        try:
            h = httpx.get(f"{url}/health", timeout=120).json()
            print(f"health {name}: {h}")
        except Exception as exc:  # noqa: BLE001
            print(f"health {name}: FAILED {exc}")

    summary: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "n_b1": len(b1), "n_b2": len(b2), "systems": {}}
    for name, url in targets.items():
        for track, samples in (("B1", b1), ("B2", b2)):
            subset = samples[: 2 * args.ragas_limit] if name.startswith("ragas") else samples
            print(f"== {name} / {track} ({len(subset)} samples) ==")
            res = run_system(f"{name}_{track}", url, subset, out_dir, args.timeout)
            entry = {"n": len(res), **prf1(res), "auroc": auroc(res), **latency_stats(res),
                     "llm_calls_total": sum(int(r.get("llm_calls") or 0) for r in res),
                     "errors": sum(1 for r in res if r.get("error"))}
            summary["systems"].setdefault(name, {})[track] = entry
            print(f"   {entry}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # markdown scoreboard
    lines = ["# Comparative benchmark scoreboard", "",
             f"Created: {summary['created']}  |  B1 n={summary['n_b1']}  B2 n={summary['n_b2']}", "",
             "| system | track | n | F1 | precision | recall | AUROC | p50 ms | p95 ms | LLM calls | errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, tracks in summary["systems"].items():
        for track, e in tracks.items():
            lines.append(
                f"| {name} | {track} | {e['n']} | {e['f1']} | {e['precision']} | {e['recall']} "
                f"| {e['auroc']} | {e['p50_ms']} | {e['p95_ms']} | {e['llm_calls_total']} | {e['errors']} |"
            )
    lines += ["", "Latency is shim-internal (network excluded). RAGAS runs a reduced subset "
              "(pinned local judge is slow); its row is comparable on quality, not throughput.",
              "Fairness rules: docs/BENCHMARKS.md § Comparative suite."]
    (out_dir / "scoreboard.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'scoreboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
