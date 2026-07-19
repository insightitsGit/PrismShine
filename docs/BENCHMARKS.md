# PrismShine benchmarks — receipts before claims

Family rule: **public accuracy / cost / moat claims require a green receipt** from this harness (`docs/POSITIONING.md`).

## Why these suites (not just RAGTruth)

| Suite | What it proves | Competitor coverage |
|---|---|---|
| `cause` / `cause_side` | Tier-0 catches injected runtime failures; pre-gen halt avoids tokens | **Shine-only** — encoders/judges never see the ledger |
| `grounding` | Effect-side P/R/F1 on hard synthetic cases (+ optional RAGTruth) | Apples-to-apples vs span baseline |
| `latency` / `latency_cost` | p50/p95 ms + $/1k vs LLM-judge proxy + escalation rate | Cost leadership story |
| `consistency` | Dual-rail: prevention off → detection still 100% on stale cache | Shine-only |
| `compare` (containerized) | **Real measured numbers vs. 2 market products** on identical data + hardware | HHEM-2.1-Open, RAGAS faithfulness (see §Comparative suite) |

## Run

```bash
# All suites → benchmarks/reports/
prismshine bench --suite all --report benchmarks/reports

# Single suite
prismshine bench --suite cause --report benchmarks/reports

# Pytest (CI)
pytest tests/benchmarks -m benchmark
# or without marker filter:
pytest tests/benchmarks
```

Optional full grounding (network) + pinned ONNX Tier-3:

```bash
set PRISMSHINE_BENCH_FULL=1
set PRISMSHINE_SPAN_ONNX=C:\path\to\model.onnx
set PRISMSHINE_SPAN_TOKENIZER=C:\path\to\tokenizer.json
pip install datasets "prismshine[spans]"
prismshine bench --suite grounding --report benchmarks/reports
```

Feedback loop (FP/FN → calibrate):

```bash
prismshine feedback bundle.json --label hallucination --out benchmarks/feedback.jsonl
prismshine calibrate benchmarks/feedback.jsonl --mode feedback --profile clinical --out cal.json
```

Offline hard-effect cases (negation / entity / finance / legal) always run inside the grounding suite via `prismshine.bench.ragtruth.hard_effect_pairs`.

## Gates (must pass for marketing)

| Claim | Gate | Receipt field |
|---|---|---|
| ≥90% injected runtime failures caught | `cause_side.gates.catch_rate >= 0.90` | `cause_side.json` |
| Pre-gen saves tokens | `pre_gen_model_calls == 0` | `cause_side.json` |
| Synthetic hard-case F1 | `grounding.gates.synthetic_f1 >= 0.85` | `grounding.json` |
| Within 5 F1 of span SotA proxy | `within_5pts_of_span_baseline` | `grounding.json` |
| Fast path budget | soft CI `<100ms` p50; local target `<25ms` | `latency_cost.json` |
| Judge escalation | target `≤0.10` (soft CI `≤0.25` without live judge) | `latency_cost.json` |
| Stale-cache dual-rail | detection catch rate `1.0` when prevention off | `consistency.json` |

## Output layout

```
benchmarks/reports/
  bench_report.json      # aggregate
  bench_report.md        # human summary
  cause_side.json
  grounding.json
  latency_cost.json
  consistency.json
```

## Honest competitor cells

Reports never invent RAGAS / Blue Guardrails / LettuceDetect Hub scores. Cells are either:

- measured in-process (Shine + `SpanClassifier` baseline),
- **measured in the containerized comparative suite below**, or
- `"literature / not run"`.

## Comparative suite (`bench/` — containerized, real numbers vs. the market)

The in-process suites above prove Shine against itself; this suite produces the
head-to-head numbers. Each system runs in its **own container** on identical hardware,
over identical data, behind one common HTTP contract.

### Chosen comparators (2 external + 1 internal ablation)

| System | Container | Represents | Why chosen |
|---|---|---|---|
| **Vectara HHEM-2.1-Open** | `bench-hhem` | Encoder-classifier SotA (the "fast model" competitor) | Open weights (HF `vectara/hallucination_evaluation_model`), ~110M params, CPU-friendly, the de-facto public baseline. Apples-to-apples with our no-judge fast path. |
| **RAGAS `faithfulness`** | `bench-ragas` | The LLM-judge path (the "expensive but accurate" competitor) | Most-adopted open-source RAG eval; what teams actually deploy. LLM pinned via an Ollama sidecar (llama3.1:8b-instruct-q4) for $0 reproducible runs; optional OpenAI mode for the best-case-competitor number. |
| LettuceDetect standalone | `bench-lettuce` | Our own Tier-3 in isolation | Ablation: PrismShine minus Tiers 0–2 — isolates what forensics + copy-check + coverage add on top of the span model we embed. |

PrismShine runs twice: `prismshine-fast` (default profile, T0–T3, **zero LLM calls**) and
`prismshine-judge` (T4 against the same Ollama sidecar RAGAS uses, so LLM cost is symmetric).

### Topology

```
docker compose -f bench/compose.yaml up
  bench-runner (orchestrator, datasets, scoring, reports)
    │ POST /evaluate  (one common contract per sample)
    ├── prismshine-fast    cpus:4 mem:8g
    ├── prismshine-judge   cpus:4 mem:8g
    ├── bench-hhem         cpus:4 mem:8g
    ├── bench-ragas        cpus:4 mem:8g
    ├── bench-lettuce      cpus:4 mem:8g
    └── ollama             (pinned llama3.1:8b-instruct-q4)
  volumes: ./bench/data (ro), ./bench/results (rw)
```

Common contract (each system gets a ~60-line FastAPI shim):

```
POST /evaluate
  {"id", "question", "context": [...], "answer",
   "evidence": {…optional EvidenceBundle extras: trace, node_state…}}
→ {"id", "risk": 0-1, "label": "hallucinated|grounded",
   "spans": [...opt...], "latency_ms", "llm_calls", "cost_usd"}
```

Competitors ignore `evidence` (they can't consume ledgers) — that asymmetry is the moat,
reported honestly via the track split below.

### Tracks

| Track | Data | Who competes | Proves |
|---|---|---|---|
| **B1 content-only** | RAGTruth test split + FaithBench slice | all 5 | fair-fight F1/AUROC (+ span F1 where supported) |
| **B2 numbers slice** | fabricated/derived-figure subset (B1 + synthetic) | all 5 | Tier-1 exact-match vs. embedding/LLM number-blindness |
| **B3 injected failures** | ~50 clean runs × handbook failure injections | Shine full-evidence vs. others content-only | cause-side category gap: catch ≥90%, false-fire ≤2% |
| **B4 cost & latency** | 1k replay mix, 85% clean / 15% dirty | all 5 | p50/p95, LLM calls per 1k, $/1k, tier-resolution histogram |

### Fairness rules (print in every report)

1. Same context strings, same order, same hardware caps per container.
2. First 20 samples per system excluded (warm-up); 3 runs, median reported.
3. All versions pinned by image digest and printed in the report header, incl.
   `PRISMSHINE_SPAN_ONNX` artifact id and `calibration_version`.
4. Competitors at published defaults; PrismShine `default` profile uncalibrated for the
   headline row, plus one `prismshine calibrate`-fitted row clearly marked.
5. B3 is labeled "evidence-aware track" — never presented as like-for-like accuracy.
6. No cherry-picking: every track/system pair appears in the summary, including losses.

### Comparative gates (extends the table above)

| Claim | Gate |
|---|---|
| Fast path beats HHEM on B2 (numbers) F1 by ≥ 15 pts | B2 scoreboard |
| Fast path within 5 F1 pts of HHEM on B1 at ≤ ½ its p50 latency | B1 + B4 |
| ≥ 90% B3 catch rate, ≤ 2% false-fire on clean runs | B3 scoreboard |
| `prismshine-fast` LLM calls per 1k = **0**; judge mode ≤ 100 | B4 counters |
| Calibrated row improves B1 AUROC by a reportable delta | calibrated vs. default rows |

### Build order

1. Land `handoffs/handoff-fix1.md` P0 items first — FIX-1/2/3 change scores and costs.
2. `bench/` skeleton: `compose.yaml`, `runner/` (httpx + pandas), per-system Dockerfile + shim.
3. Reuse `tests/benchmarks` label logic for the B2 generator; reuse
   `tests/test_action_matrix.py` scenario shapes for the B3 injector.
4. Full B1–B4 pass ≈ 2–3 h on a 16-core box (RAGAS/Ollama dominates).

## Competitive advantages these suites support

1. **Cause-side + pre-gen halt** — category gap vs all effect-only tools.
2. **Consistency contract** — stale cache after fact correction.
3. **Cost** — CPU fast path; judge only on residual gray zone.
4. **Runtime-agnostic wiring** — same cause suite uses `prismshine.wiring` (no ChorusGraph required).

See also: `docs/POSITIONING.md`, `docs/INTEGRATION.md` §8 (BYO runtime).
