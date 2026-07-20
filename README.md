# PrismShine

**Anti-hallucination verdict engine for the Insight agent stack.**

PrismShine catches hallucinations from both ends in a single unified pipeline:

- **Cause side (trace forensics)** — reads the runtime evidence that no external tool can see (ChorusGraph Route Ledger, node states, cache-gate decisions, retrieval scores, tool results) and detects broken preloads *before* they become hallucinations: empty retrieval, swallowed tool errors, truncated context, stale cache reuse, memory conflicts. Deterministic, zero LLM calls, driven by a versioned failure-signature **Handbook**.
- **Effect side (grounding verification)** — verifies the final answer against the exact preload the LLM received, using a tiered ladder: lexical copy-checks (numbers/entities/dates) → vector coverage (reusing embeddings the runtime already has) → ONNX span classifier → LLM judge *only* for the residual gray zone.

Both sides fuse into one auditable **ShineVerdict** with a named resolution gate — the same audit-grade decision style as PrismGuard.

> **PASS ≠ true.** A `pass` means the answer is grounded in the provided preload, not that the preload is world-correct. Poisoned or wrong retrieval can still PASS. See [`docs/LIMITS.md`](docs/LIMITS.md).
>
> **Buffered only (v0).** PrismShine verifies the completed answer before display. It does not mid-stream verify tokens.
>
> **Moat requires wiring.** Standalone dicts give a strong grounding checker. Cause-side halt + cache/memory consistency need ChorusGraph (or equivalent) `trace`, interceptors, and Cortex/cache hooks — checklist in `docs/LIMITS.md`.

## Design principles

1. **One pipeline, one verdict.** Forensic and grounding signals are fused; there are no separate "phases" a developer has to wire.
2. **Minimum LLM calls.** Tiers 0–2 are free/deterministic; Tier 3 is a small local ONNX model; Tier 4 (LLM judge) is an opt-in escalation reached by a small fraction of traffic.
3. **Zero extra embedding cost.** Context vectors are reused from the runtime (retrieval/warm index); only the answer is encoded, once, with the already-loaded local ONNX encoder. No API embedding calls anywhere. Verdicts are content-address cached.
4. **Audit-grade.** Every verdict names its gate, its firing signatures, and the evidence hash — replayable, ledger-attached, compliance-ready.
5. **Cause before effect.** A broken preload is flagged (and can halt generation) before tokens are ever spent.
6. **The consistency contract.** Every state mutation in the stack (fact corrections, source updates, model changes) has BOTH an invalidation path and a deterministic detection backstop — no stale cache entry, warm-index row, or cached verdict may keep answering from pre-mutation state (DESIGN.md §6.1). Only a verifier that lives inside the runtime can guarantee this; stateless external checkers cannot even see these artifacts.
7. **Zero hard sibling dependencies.** `pip install prismshine` works standalone on any stack (LangGraph, custom runtimes); every Insight sibling is an optional extra that lights up more capability when present, detected at build time and recorded in every verdict — degradation is always transparent, and a missing capability can never manufacture a false PASS (DESIGN.md §8.2).

## Documentation

| Doc | Contents |
|---|---|
| [`docs/LIMITS.md`](docs/LIMITS.md) | Scope boundaries: PASS≠truth, streaming, moat wiring, threshold receipts |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Full architecture: unified pipeline, data model, scoring math, module layout, performance budget |
| [`docs/prismshine-architecture.png`](docs/prismshine-architecture.png) | One-page visual of the pipeline, plugins, early exits, and support systems |
| [`docs/HANDBOOK.md`](docs/HANDBOOK.md) | Failure-signature taxonomy: schema + initial catalog of deterministic detectors |
| [`docs/INTEGRATION.md`](docs/INTEGRATION.md) | Integration points with ChorusGraph, PrismGuard, PrismCortex, LangGraph, and standalone use |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture decision records |
| [`docs/UPSTREAM.md`](docs/UPSTREAM.md) | Coordinated version bumps needed in sibling libraries (prismlib, ChorusGraph, PrismCortex, prismlang) |
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | Market comparison, standing, honest weaknesses, and the benchmark gates required before claims go public |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Receipt-backed suites: cause-side, grounding, latency/cost, consistency |
| [`handoffs/`](handoffs/) | Self-contained work orders, one per sibling repo, for implementing the upstream changes (report-back format for verification) |
| [`kb/README.md`](kb/README.md) | Knowledge base of the 12 sibling Insight libraries (verified from source) |

## Ecosystem position

```
User query ──► PrismGuard (input firewall)
                  │
                  ▼
             ChorusGraph (runtime) ──► retrieval / tools / PrismCortex memory
                  │                          │
                  │   ledger, node state,    │  chunks + existing vectors
                  │   cache decisions        ▼
                  ├────────────────► PrismShine EvidenceBundle
                  ▼                          │
                LLM answer ─────────────────►│
                                             ▼
                                    ShineGate.verify()
                                             │
                              ShineVerdict: pass / flag / block / regenerate
                                    (gate + signatures + spans, ledger-attached)
```

## Install

```bash
pip install -e ".[dev]"
# optional extras: coverage (prismlang), spans, chorusgraph, langgraph, guard, judge-openai, judge-gemini
```

Any runtime (LangGraph / custom) gets the same Shine features as ChorusGraph via `prismshine.wiring` — see `docs/INTEGRATION.md` §8.

### Enterprise / production checklist

1. **Tier-3 ONNX** (not bundled in the wheel — ~1GB):  
   `python -m prismshine.tools.ensure_span_onnx --export` then pin `PRISMSHINE_SPAN_ONNX` / `PRISMSHINE_SPAN_TOKENIZER`.
2. **Domain calibration** (marked row, not the uncalibrated headline):  
   `python -m prismshine.bench.calibrate_minilm --embedder minilm` (or `--embedder hash` offline) → `PRISMSHINE_CALIBRATION=...`.
3. **Wiring moat** (cause-side halt + consistency): run `python examples/enterprise_wiring_demo.py`, then integrate via `docs/INTEGRATION.md`.
4. **Tier-4 judge** (opt-in): `pip install 'prismshine[judge-openai]'` + `ShineGate.build(judge="openai")` — see `examples/tier4_judge_demo.py`.
5. **Receipts before claims**: `prismshine bench --suite all --report benchmarks/reports` and comparative `bench/runner/run_bench.py --runs 3` vs HHEM.

## Quick start

```python
from prismshine import EvidenceBundle, PreloadChunk, ShineGate

gate = ShineGate.build(profile="default")
bundle = EvidenceBundle(
    run_id="demo",
    question="What was revenue?",
    answer="Revenue was $1000 in Q1.",
    preload=[
        PreloadChunk(
            chunk_id="c1",
            text="Revenue was $1000 in Q1.",
            source="retrieval",
        )
    ],
)
verdict = gate.verify(bundle)
print(verdict.decision, verdict.resolution_gate, verdict.evidence_hash)
print(gate.capabilities())
```

## CLI

```bash
prismshine capabilities
prismshine verify bundle.json --profile default
prismshine calibrate ./samples --mode synthetic
prismshine bench --suite all --report benchmarks/reports
```

## Profiles & handbook packs

```python
gate = ShineGate.build(profile="clinical")  # merges builtin clinical.yaml pack
# finance / legal likewise. Thresholds remain "proposal" until:
#   prismshine calibrate ./samples --mode synthetic
```

`gate.capabilities()` reports `span_backend` (`onnx`|`lexical`|`unavailable`), `threshold_status`, and `pass_means`.

Pin Tier-3 for CI reproducibility: `PRISMSHINE_SPAN_MODEL` (HF model id), `PRISMSHINE_SPAN_ONNX` (local `.onnx`), and optionally `PRISMSHINE_SPAN_TOKENIZER`. Without a pin/artifact the gate honestly reports `span_backend=lexical`. Run `prismshine calibrate` (or `feedback` → `calibrate --mode feedback`) so `threshold_status` leaves `proposal` before accuracy claims. See `docs/BENCHMARKS.md`.

## Status

**0.1.0** implemented. Upstream siblings shipped (prismlang 0.1.2, prismlib 0.5.0, prismlib-plus 0.8.0, prismcortex 0.3.0, chorusgraph 1.3.0). Design authority: `docs/DESIGN.md`. Honest limits: `docs/LIMITS.md`.
