"""Calibrate effect thresholds on HaluEval B1 (library calibrate path).

Produces a marked calibrated receipt — not a bench-only threshold hack.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "runner"))

from run_bench import build_b1, load_halueval  # noqa: E402

from prismshine.calibrate import (  # noqa: E402
    _decision_f1,
    apply_overlay_to_gate,
    calibrate_labeled,
)
from prismshine.evidence.builder import bundle_from_dict  # noqa: E402
from prismshine.gate import ShineGate  # noqa: E402
from prismshine.grounding.splitter import split_sentences  # noqa: E402


def _hash_embedder(texts: list[str]) -> np.ndarray:
    dim = 64
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % dim
            out[i, h] += 1.0
        n = float(np.linalg.norm(out[i]) or 1.0)
        out[i] /= n
    return out


def _to_pairs(samples: list[dict]) -> list:
    pairs = []
    for s in samples:
        data = {
            "run_id": s["id"],
            "question": s["question"],
            "answer": s["answer"],
            "preload": [
                {"chunk_id": f"c{i}-{j}", "text": sent, "source": "retrieval"}
                for i, c in enumerate(s["context"])
                for j, sent in enumerate(split_sentences(c) or [c])
            ],
            "trace": [
                {
                    "hop": "retrieve",
                    "kind": "retrieval",
                    "status": "ok",
                    "detail": {"n_chunks": max(1, len(s["context"]))},
                }
            ],
        }
        b, _ = bundle_from_dict(data)
        pairs.append((b, s["label"] == "hallucinated"))
    return pairs


def main() -> int:
    cache = ROOT / "bench" / "runner" / "data"
    out_dir = ROOT / "benchmarks" / "reports" / "b1_calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = build_b1(load_halueval(cache, 100))
    rng = random.Random(42)
    rng.shuffle(samples)
    mid = len(samples) // 2
    train, test = samples[:mid], samples[mid:]
    train_pairs = _to_pairs(train)
    test_pairs = _to_pairs(test)

    # Coarse grid for runtime; product path still lives in calibrate.fit_effect_thresholds
    gate = ShineGate.build(profile="default", embedder=_hash_embedder)
    pre_f1 = _decision_f1(gate, test_pairs)
    report = calibrate_labeled(
        train_pairs,
        gate=gate,
        version="cal-halueval-b1-0.1",
        apply_to_gate=True,
        fit_coverage=True,
        tau_sent_grid=[0.45, 0.55, 0.62, 0.72],
        tau_floor_grid=[0.05, 0.10, 0.15, 0.20],
    )
    # apply same overlay fresh for clarity
    apply_overlay_to_gate(gate, report.to_yaml_overlay())
    post_f1 = _decision_f1(gate, test_pairs)

    payload = {
        "track": "B1-calibrated",
        "dataset": "HaluEval-QA",
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "embedder": "hash-64 (local; Azure shim uses MiniLM — redeploy with PRISMSHINE_CALIBRATION)",
        "pre_calibrate_f1_test": round(pre_f1, 4),
        "calibrated_f1_test": round(post_f1, 4),
        "f1_lift": round(post_f1 - pre_f1, 4),
        "thresholds": report.thresholds,
        "threshold_status": gate.policy.threshold_status,
        "calibration_version": report.version,
        "overlay": report.to_yaml_overlay(),
        "notes": [
            "Calibrated row — not the default headline vs HHEM.",
            "Fitting lives in prismshine.calibrate (library), not a bench fork.",
            *report.notes,
        ],
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
