# Improvement batch — 2026-07-20 (design apply)

Implements the design-ordered backlog against run3 standing.

## Standing (pre-redeploy)

run3 / `2026-07-19_comparative_aci`: B2 win, B1 −2.7 F1 vs HHEM, Bsum lead, 0 LLM calls, cause-side moat.

## Applied in code

| Item | Design layer | Change |
|---|---|---|
| Proper-noun whole-phrase match | Tier 1 | Entities match only as word-bounded phrases (no surname-in-Olivia) |
| Multi-word entity hard floor | Tier 1 → gate | Novel multi-word PNs floor at `T1_UNMATCHED_HARD_FACT` |
| Narrow `CLEAN_FAST_PATH` | §3 early exit | PASS only if **extractive** (containment) and no novel multi-word PNs |
| ONNX export tool | Tier 3 | `python -m prismshine.tools.export_span_onnx` → pin `PRISMSHINE_SPAN_ONNX` |
| Hub candidate | Tier 3 | Prefer `KRLabsOrg/lettucedect-base-modernbert-en-v1` |
| MiniLM calibrate | §5.5 | `python -m prismshine.bench.calibrate_minilm` → `PRISMSHINE_CALIBRATION` |
| B2 gate reword | BENCHMARKS | F1 ≥ 0.99 + zero FP (was unattainable ≥15 pts at ceiling) |

## Local smoke (hash embedder, HaluEval 200)

| | run3 Azure baseline | after this batch (local hash) |
|---|---|---|
| F1 | 0.719 | **0.862** |
| FN | 40 | **16** |
| FP | 7 | 11 |

Azure MiniLM + ONNX + calibrated row still required for official claim.

## Still operator steps

1. Export ONNX + bake into `prismshine:v4` with `PRISMSHINE_SPAN_ONNX`
2. Run `calibrate_minilm` → bake `PRISMSHINE_CALIBRATION`
3. 3-run median B1/B2/Bsum scoreboard
4. Optional GPU RAGAS row
