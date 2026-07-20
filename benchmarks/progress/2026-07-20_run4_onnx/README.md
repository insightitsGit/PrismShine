# run4 — ACI v4 with LettuceDetect ONNX + entity/fast-path

Date: 2026-07-20
Image: `bench/prismshine:v4` (lean ACR staging context ~1.04 GiB, single ONNX bake)
Health: `span_backend=onnx`, `calibration_version=identity-0-placeholder`
Containers stopped after bench (shine/hhem/ragas).

## Scoreboard (B1 n=200, B2/Bsum n=50)

| system | B1 F1 | B2 F1 | Bsum F1 | B1 p50 ms |
|---|---|---|---|---|
| prismshine-fast | **0.831** | **1.000** | **0.600** | 90 |
| hhem-2.1-open | 0.746 | 0.926 | 0.474 | 216 |

## vs run3 (pre-ONNX / containment-only)

| track | Shine run3 | Shine run4 | HHEM |
|---|---|---|---|
| B1 | 0.719 | **0.831** | 0.746 |
| B2 | 1.000 | 1.000 (0 FP) | 0.926 |
| Bsum | 0.565 | 0.600 | 0.474 |

## Notes

- B1: Shine now ahead of HHEM by ~8.5 F1 pts (was behind by ~2.7).
- B2 gate (F1 ≥ 0.99 + zero FP): **pass**.
- MiniLM overlay still identity placeholder (local calibrate crash); not blocking ONNX Tier-3.
- ragas-bench left stopped (not in targets).
