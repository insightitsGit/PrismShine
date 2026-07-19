# PrismShine

**Anti-hallucination verdict engine for the Insight agent stack.**

PrismShine catches hallucinations from both ends in a single unified pipeline:

- **Cause side (trace forensics)** — reads the runtime evidence that no external tool can see (ChorusGraph Route Ledger, node states, cache-gate decisions, retrieval scores, tool results) and detects broken preloads *before* they become hallucinations: empty retrieval, swallowed tool errors, truncated context, stale cache reuse, memory conflicts. Deterministic, zero LLM calls, driven by a versioned failure-signature **Handbook**.
- **Effect side (grounding verification)** — verifies the final answer against the exact preload the LLM received, using a tiered ladder: lexical copy-checks (numbers/entities/dates) → vector coverage (reusing embeddings the runtime already has) → ONNX span classifier → LLM judge *only* for the residual gray zone.

Both sides fuse into one auditable **ShineVerdict** with a named resolution gate — the same audit-grade decision style as PrismGuard.

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
| [`docs/DESIGN.md`](docs/DESIGN.md) | Full architecture: unified pipeline, data model, scoring math, module layout, performance budget |
| [`docs/prismshine-architecture.png`](docs/prismshine-architecture.png) | One-page visual of the pipeline, plugins, early exits, and support systems |
| [`docs/HANDBOOK.md`](docs/HANDBOOK.md) | Failure-signature taxonomy: schema + initial catalog of deterministic detectors |
| [`docs/INTEGRATION.md`](docs/INTEGRATION.md) | Integration points with ChorusGraph, PrismGuard, PrismCortex, LangGraph, and standalone use |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture decision records |
| [`docs/UPSTREAM.md`](docs/UPSTREAM.md) | Coordinated version bumps needed in sibling libraries (prismlib, ChorusGraph, PrismCortex, prismlang) |
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | Market comparison, standing, honest weaknesses, and the benchmark gates required before claims go public |
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

## Status

Design complete; **all upstream sibling releases shipped and verified** (prismlang 0.1.2, prismlib 0.5.0, prismlib-plus 0.8.0, prismcortex 0.3.0, chorusgraph 1.3.0 — see `docs/UPSTREAM.md` for the verification record). Next step: PrismShine implementation, targeting the native sibling APIs — remaining open questions in `docs/DESIGN.md` §13.
