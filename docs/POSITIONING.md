# PrismShine — Market Positioning & Targets

Snapshot of the anti-hallucination market (Jul 2026) and where PrismShine stands if implemented per `DESIGN.md`. Keep this current as the market moves; claims below marked **[target]** are design goals that must be proven by the benchmark suite before being used publicly (family rule: claims ship with receipts — see [`docs/BENCHMARKS.md`](BENCHMARKS.md)).

## Competitive landscape

| Dimension | LLM-as-judge (RAGAS, DeepEval, Blue Guardrails) | Encoder tools (LettuceDetect, MiniCheck, Lynx, HHEM) | Geometric (Groundlens) | Cloud APIs (Azure, Google, Bedrock) | Eval platforms (Phoenix, FutureAGI) | **PrismShine (per design)** |
|---|---|---|---|---|---|---|
| Cause-side detection (broken preload, tool errors, stale cache) | ✗ | ✗ | ✗ | ✗ | partial, offline | **✓ deterministic, free** |
| Pre-generation halt (save the tokens) | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Cache/state consistency (stale-answer bypass) | ✗ can't see it | ✗ | ✗ | ✗ | ✗ | **✓ contract, dual rails (§6.1)** |
| Marginal cost per check | 1 LLM call | GPU/CPU inference | ~free | per-call pricing | per-call/platform | **~free; opt-in judge on gray zone only [target ≤10%]** |
| Deterministic, replayable verdicts | ✗ | mostly | ✓ | ✗ | ✗ | **✓ content-addressed** |
| Span-level output | some | ✓ | ✗ | ✓ | some | ✓ |
| Named audit gate + evidence pointer | ✗ | ✗ | partial | ✗ | trace-linked | **✓** |
| Self-hosted / offline | varies | ✓ | ✓ | ✗ | varies | ✓ |

Market context: a 2026 public benchmark (PlaceboBench) measured six major tools at 53–62% message-level accuracy — barely above guessing; the one high scorer (94%, Blue Guardrails) uses LLM verification agents, i.e. the expensive path. Domain calibration is the single biggest published quality lever (AUROC ~0.76 → 0.90+), which PrismShine ships as `prismshine calibrate`.

## Standing (if implemented cleanly)

- **Category-creator:** only product doing runtime-integrated hallucination *prevention* (cause-side forensics + pre-generation halt + consistency contract). Structural moat — competitors must become runtimes (or adopt the same ledger contract) to copy it.
- **Cost leader:** majority path is Tier 0–2, CPU, zero network, zero marginal cost; pre-gen halting *saves* generation spend. No competitor can claim verification that pays for itself.
- **Best-in-class auditability:** named gates, evidence pointers, content-addressed replay — the regulated-vertical (legal/clinical/finance) differentiator.
- **Peer-competitive raw detection:** Tier-3 adoption = LettuceDetect-class parity at v0, with a data-advantaged path to lead (ledger traces with Tier-0-labeled causes are proprietary training data — ADR-8).

## Honest weaknesses (do not paper over)

1. Raw detection accuracy will not lead at v0; LLM-agent products win cue-less contradictions unless Tier 4 is enabled. Counter: expanded contradiction lexicon + polarity phrases force Tier-3 (and flag without ONNX/judge); pin `PRISMSHINE_SPAN_ONNX` for real span SotA; domain `calibrate` + `feedback` JSONL close the gap with receipts. Never market "highest accuracy" without `grounding.json` / RAGTruth (`PRISMSHINE_BENCH_FULL=1`).
2. The full moat requires **wiring** (trace / ledger steps + pre/post hooks via `prismshine.wiring` or ChorusGraph plugins). Without a ledger, Shine is a strong grounding checker — competitive, not category-defining. Richest out-of-the-box inside ChorusGraph; LangGraph/custom achieve parity when authors emit the same evidence.
3. Distribution: eval platforms own integration mindshare; beachhead = Insight ecosystem + regulated verticals.
4. English-only at v0; multilingual is post-v0.

## Targets to claim the standing (benchmark gates before launch)

| Target | Gate | Receipt |
|---|---|---|
| Fast path p50 < 25 ms (T0+T1+T2) local | latency harness soft CI `<100ms`; local target `<25ms` | `benchmarks/reports/latency_cost.json` via `prismshine bench --suite latency` |
| Example-level detection within 5 F1 pts of encoder SotA on RAGTruth | Tier-2+3 eval (+ optional `PRISMSHINE_BENCH_FULL=1`) | `benchmarks/reports/grounding.json` |
| ≥ 90% of injected runtime failures caught by Tier-0 signatures | cause-side injection suite | `benchmarks/reports/cause_side.json` |
| Judge escalation ≤ 10% of traffic on default profile | traffic replay / mixed corpus proxy | `benchmarks/reports/latency_cost.json` (`judge_escalation_rate`) |
| Zero stale-cache serves after correction in the consistency-contract test matrix | dual-rail (prevention off → detection 100%) | `benchmarks/reports/consistency.json` |
| Domain calibration lifts AUROC ≥ 0.10 over generic defaults on each domain pack | `prismshine calibrate` report | calibration overlay JSON (`threshold_status` leaves `proposal`) |

Run all receipts: `prismshine bench --suite all --report benchmarks/reports` (details in [`docs/BENCHMARKS.md`](BENCHMARKS.md)).

**Latest comparative receipt (2026-07-20, Azure ACI v4 + ONNX Tier-3, vs HHEM-2.1-Open):** B1 QA F1 **0.831** (HHEM 0.746), B2 fabricated-numbers F1 **1.000 / 0 FP** (HHEM 0.926), Bsum F1 0.600 (HHEM 0.474), B1 p50 90 ms vs 216 ms, zero LLM calls, `span_backend=onnx`. Details: [`benchmarks/progress/2026-07-20_run4_onnx/`](../benchmarks/progress/2026-07-20_run4_onnx/README.md). In-process suites (cause/grounding/latency/consistency) **PASS** under `benchmarks/reports/` after v0.2.0.

### Go-live readiness (enterprise open-source pip) — v0.2.0

| Layer | Status | Verdict |
|---|---|---|
| Effect-side vs HHEM (HaluEval B1/B2) | Ahead on F1 + latency; B2 gate green | Competitive fast grounding checker |
| Cost story (0 LLM on default) | Proven on ACI | Claim with receipt |
| Cause / consistency / latency receipts | **PASS** in `benchmarks/reports/` | Claim with those receipts |
| FIX-1–14 + ONNX ensure + wiring/judge demos | Shipped in **0.2.0** | Code-ready enterprise install |
| Calibrated overlay | `cal-halueval-hash-0.1` validated-labeled | Marked row OK; MiniLM bake for ACI parity |
| Moat wiring | Demo + INTEGRATION.md | Competitive when wired into runtime |
| vs LLM-judge (RAGAS / Blue) | Not measured | Do not claim |
| PyPI publish | `0.2.0` ready to publish | Soft GA after pip publish |
| 3-run median ACI | `run_bench.py --runs 3` ready; not re-run | Optional before big marketing |

**Bottom line:** **v0.2.0 is enterprise-ready open source** for the self-hosted verifier lane. Category-creator / beats-judges claims still need production wiring, optional 3-run ACI median, and a fair judge comparator.
