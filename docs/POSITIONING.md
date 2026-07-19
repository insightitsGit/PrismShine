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
- **Best-in-class auditability:** named gates, evidence pointers, content-addressed replay — the regulated-vertical (legal/clinical/finance) differentiator; strongest when sold with PrismGuard + PrismCortex as an auditable input→memory→output chain.
- **Peer-competitive raw detection:** Tier-3 adoption = LettuceDetect-class parity at v0, with a data-advantaged path to lead (ledger traces with Tier-0-labeled causes are proprietary training data — ADR-8).

## Honest weaknesses (do not paper over)

1. Raw detection accuracy will not lead at v0; LLM-agent products win cue-less contradictions unless Tier 4 is enabled. Counter: Tier-1 beats pure NLI on fabricated numbers; contradiction cues route hard cases to the right tier. Never market "highest accuracy" without benchmark receipts.
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
