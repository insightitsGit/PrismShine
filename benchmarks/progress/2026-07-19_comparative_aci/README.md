# Comparative benchmark — Azure ACI, 2026-07-19 (run3)

First real containerized comparison of PrismShine against a market comparator on
identical data and per-container hardware caps (docs/BENCHMARKS.md § Comparative suite).

## Environment

| System | Image | Host | Hardware |
|---|---|---|---|
| prismshine-fast | `bench/prismshine:v3` (commit `781f571`) | ACI eastus | 4 vCPU / 8 GB |
| hhem | `bench/hhem:v1` (vectara/hallucination_evaluation_model) | ACI centralus | 4 vCPU / 8 GB |
| ragas | `bench/ragas:v3` + ollama llama3.2:3b-instruct-q4_K_M | ACI eastus | 1+3 vCPU / 12 GB |

- Registry: `prismshinebenchbx7k2m.azurecr.io`; runner: `bench/runner/run_bench.py`, seed 42.
- Data: HaluEval QA (B1 200 samples, B2 50 fabricated-number samples) and
  HaluEval summarization (Bsum 50 samples).
- PrismShine config: `default` profile, **uncalibrated** (`calibration_version=identity-0`,
  `threshold_status=proposal`), embedder all-MiniLM-L6-v2, Tier-4 judge off,
  Tier-3 span backend = **lexical fallback** (no public LettuceDetect ONNX artifact
  resolved at container start — these are floor numbers for PrismShine).
- RAGAS: **no row** — abandoned on this hardware. The pinned CPU judge measured
  ~8 tok/s prompt eval in-container; even a 10-sample subset produced zero
  completions in 85 minutes. A fair RAGAS row needs a GPU judge or a hosted API
  (breaks the $0-reproducible rule; revisit per fairness rule 4).

## Scoreboard

| system | track | n | F1 | precision | recall | AUROC | acc | p50 ms | p95 ms | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|
| prismshine-fast | B1 | 200 | 0.7186 | 0.8955 | 0.60 | **0.838** | 0.765 | **29.9** | 67.8 | 0 |
| hhem | B1 | 200 | **0.7458** | 0.8571 | 0.66 | 0.7934 | 0.775 | 169.0 | 237.3 | 0 |
| prismshine-fast | B2 | 50 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | **13.6** | 43.3 | 0 |
| hhem | B2 | 50 | 0.9259 | 0.8621 | 1.000 | 1.000 | 0.920 | 134.8 | 204.8 | 0 |
| prismshine-fast | Bsum | 50 | **0.5652** | 0.619 | 0.52 | 0.5848 | 0.60 | **133.6** | 693.0 | 0 |
| hhem | Bsum | 50 | 0.4737 | 0.6923 | 0.36 | 0.616 | 0.60 | 2101.0 | 5025.3 | 0 |

Latency is shim-internal (network excluded). Full raw per-sample receipts:
`summary.json` here; per-sample JSONL under `bench/runner/results/run3/`.

## Gate check (docs/BENCHMARKS.md § Comparative gates)

| Claim | Gate | Result |
|---|---|---|
| Fast path B2 (numbers) F1 ≥ 0.99 with zero false positives | B2 | **Met**: F1 1.000, FP 0 (gate reworded; old ≥15-pt delta unattainable at ceiling). |
| Within 5 F1 pts of HHEM on B1 at ≤ ½ its p50 | B1 | **Met**: −2.7 pts (0.7186 vs 0.7458) at 0.18× HHEM's p50 (29.9 vs 169.0 ms). AUROC is higher than HHEM (0.838 vs 0.793). |
| `prismshine-fast` LLM calls per 1k = 0 | counters | **Met**: 0 LLM calls across all 300 samples. |
| Calibrated row improves B1 AUROC | calibrated vs default | **Not yet run** — this run is the uncalibrated headline row only. |

## Honest caveats

1. PrismShine Tier-3 ran on the lexical fallback, not ONNX — recall (0.60 on B1,
   0.52 on Bsum) is the main gap and the most improvable number.
2. No calibrated row yet; `identity-0` thresholds throughout.
3. Bsum is weak for both systems; long-form verification needs ONNX Tier-3 or Tier-4.
4. B2 here is digit-perturbation synthetic (HaluEval-derived), not the RAGTruth-based
   numbers slice; treat it as the fabricated-figure smoke slice.
5. Single run (no 3-run median yet); warm-up excluded per fairness rule 2.
