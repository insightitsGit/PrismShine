# Baseline snapshot — 2026-07-19

First real comparative / RAGTruth / cause receipts before effect-side threshold work.
Keep this folder immutable; add new dated folders for later runs.

## Headline numbers

| Track | Result |
|---|---|
| **B1 HaluEval** (n=200, Azure shims, default profile, lexical Tier-3) | HHEM F1 **0.7458** · Shine F1 **0.5306** · Shine p50 **27 ms** vs HHEM **164 ms** |
| **B1 failure mode** | Shine: 80 FP via `T2_COVERAGE_COLLAPSE`, 21 FN via `CLEAN_FAST_PATH` |
| **RAGTruth** (in-process, 100 rows, `wandb/RAGTruth-processed`) | Shine F1 **0.3866** · P **0.25** · R **0.85** · AUROC **0.53** |
| **B3 cause** (injected failures) | Catch **1.0** (9/9) · false alarm on clean **false** · pre-gen calls **0** |
| **B1 calibrated** (local hash embedder, library `calibrate`) | Test F1 **0.65 → 0.65** (no lift); overlay saved under `b1_calibrated/` |

## Files

| Path | What |
|---|---|
| `scoreboard.md` / `summary.json` | Comparative B1 Shine vs HHEM |
| `raw_*_B1.jsonl` | Per-sample labels / risk / gates |
| `b3_cause/` | Cause-side receipt |
| `b1_ragtruth/` | Grounding suite + RAGTruth metrics |
| `b1_calibrated/calibration.json` | First calibrate overlay attempt |

## Conditions

- Content-only track (no ledger on B1)
- Shine shim: MiniLM embedder, `span_backend=lexical` (no ONNX pin)
- RAGAS skipped (hung on prior full run)
- Default uncalibrated profile for headline B1 vs HHEM
