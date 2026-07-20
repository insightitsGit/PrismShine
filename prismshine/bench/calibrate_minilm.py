"""Fit effect thresholds for PRISMSHINE_CALIBRATION overlays.

  # Hash embedder (always works offline; marked as non-MiniLM)
  python -m prismshine.bench.calibrate_minilm --embedder hash --n 80

  # Same MiniLM the Azure shim uses (may crash on some Windows CPU builds)
  python -m prismshine.bench.calibrate_minilm --embedder minilm --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench" / "runner"))

from run_bench import build_b1, load_halueval  # noqa: E402

from prismshine.calibrate import (  # noqa: E402
    _decision_f1,
    apply_overlay_to_gate,
    calibrate_labeled,
)
from prismshine.encoder import _hash_embed  # noqa: E402
from prismshine.evidence.builder import bundle_from_dict  # noqa: E402
from prismshine.gate import ShineGate  # noqa: E402
from prismshine.grounding.splitter import split_sentences  # noqa: E402


def _minilm():
    import numpy as np

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def embed(texts: list[str]):
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float64)

    return embed


def _hash_emb():
    def embed(texts: list[str]):
        return _hash_embed(texts, dim=64)

    return embed


def _to_pairs(samples: list[dict]):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/calibration/halueval_minilm.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--embedder",
        choices=("minilm", "hash"),
        default="hash",
        help="minilm matches ACI shim; hash is crash-safe for CI/local Windows",
    )
    args = ap.parse_args()

    cache = ROOT / "bench" / "runner" / "data"
    samples = build_b1(load_halueval(cache, args.n))
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    mid = len(samples) // 2
    train, test = _to_pairs(samples[:mid]), _to_pairs(samples[mid:])

    print(f"loading embedder={args.embedder}…")
    try:
        emb = _minilm() if args.embedder == "minilm" else _hash_emb()
    except Exception as exc:  # noqa: BLE001
        print(f"minilm failed ({exc}); falling back to hash", file=sys.stderr)
        emb = _hash_emb()
        args.embedder = "hash"

    version = f"cal-halueval-{args.embedder}-0.1"
    gate = ShineGate.build(profile="default", embedder=emb)
    pre = _decision_f1(gate, test)
    report = calibrate_labeled(
        train,
        gate=gate,
        version=version,
        apply_to_gate=True,
        fit_coverage=True,
        tau_sent_grid=[0.50, 0.55, 0.62, 0.68],
        tau_floor_grid=[0.05, 0.10, 0.15, 0.20],
    )
    apply_overlay_to_gate(gate, report.to_yaml_overlay())
    post = _decision_f1(gate, test)
    emb_name = (
        "sentence-transformers/all-MiniLM-L6-v2"
        if args.embedder == "minilm"
        else "hash-embed-64"
    )
    payload = {
        "track": f"B1-calibrated-{args.embedder}",
        "dataset": "HaluEval-QA",
        "embedder": emb_name,
        "n_train": len(train),
        "n_test": len(test),
        "pre_calibrate_f1_test": round(pre, 4),
        "calibrated_f1_test": round(post, 4),
        "f1_lift": round(post - pre, 4),
        "thresholds": report.thresholds,
        "threshold_status": "validated-labeled",
        "calibration_version": report.version,
        "overlay": report.to_yaml_overlay(),
        "notes": [
            "Marked calibrated row — not the default headline vs HHEM.",
            "Bake into shim via PRISMSHINE_CALIBRATION pointing at this JSON.",
            *report.notes,
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "overlay"}, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
