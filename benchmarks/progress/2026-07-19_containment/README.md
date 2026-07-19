# Containment snapshot — 2026-07-19

Post–Tier-2 preload containment redeploy (`bench/prismshine:v3` on Azure).
Compare to `../2026-07-19_baseline/`.

## B1 HaluEval (n=200, Azure MiniLM Shine vs HHEM)

| System | F1 | P | R | AUROC | p50 ms |
|---|---|---|---|---|---|
| **prismshine-fast (new)** | **0.7186** | 0.8955 | 0.60 | **0.838** | ~0.2* |
| hhem | 0.7458 | 0.8571 | 0.66 | 0.7934 | ~280 |
| shine baseline (old) | 0.5306 | 0.4483 | 0.65 | 0.4275 | ~27 |

\*Shim-internal timer after warm image; treat as “fast path”, not wall-clock marketing.

### Delta vs baseline Shine
- F1 **+0.19** (0.53 → 0.72)
- FP **80 → 7**
- Within **~2.7 F1** of HHEM (POSITIONING gate: ≤5 pts)
- AUROC **beats HHEM** (0.84 vs 0.79)

### Remaining errors
- FP: 7× `T2_COVERAGE_COLLAPSE`
- FN: 24× `CLEAN_FAST_PATH` + 16× `FUSION_PASS`

## Cause (B3)
Still **PASS** — catch 1.0, pre-gen 0.

## Git / image
- Commit: containment + calibrate on `main`
- Image: `prismshinebenchbx7k2m.azurecr.io/bench/prismshine:v3`
