# PrismShine — Architecture Decision Records

## ADR-1: One unified pipeline, no user-visible phases

**Decision:** Cause-side forensics (trace/preload analysis) and effect-side grounding (answer verification) are tiers of a single `ShineGate.verify()` pipeline producing one `ShineVerdict`. Pre-generation checking is the same pipeline invoked with `answer=None`, not a separate API.

**Why:** The user experience must be "add one gate, get one verdict." Fused scoring is also strictly better: a marginal coverage score plus a `MARGINAL_CACHE_HIT` warning should block where either alone would pass. Separate phases would force developers to combine verdicts themselves.

**Consequence:** Tier-0 results are keyed by evidence hash so the pre-generation invocation's work is reused post-generation without re-running detectors.

## ADR-2: Deterministic-first tier ladder; LLM judge is opt-in escalation only

**Decision:** Tiers 0–2 are free/deterministic and always available; Tier 3 is a local ONNX encoder (no LLM); Tier 4 (LLM judge) is disabled unless explicitly configured, and only receives gray-zone traffic.

**Why:** Ecosystem goal is minimum LLM calls / maximum performance. Market evidence (2026): encoder classifiers (LettuceDetect-class) beat GPT-4-as-judge on RAGTruth at a fraction of cost; deterministic geometric first stages (Groundlens pattern) clear most traffic sub-second. Mirrors PrismGuard's tiered design — house style.

## ADR-3: Verification runs in RAW embedding space (384-d), not JL-64

**Decision:** Tier-2 coverage compares answer-sentence vectors to preload chunk vectors in raw 384-d space. 64-d JL vectors are accepted only as a degraded fallback with a stricter threshold and an explicit `LOW_FIDELITY_SPACE` signal.

**Why:** JL projection is lossy by design and tenant-seeded (isolation, not fidelity). ChorusGraph itself verifies cache hits on raw 384-d for the same reason (coarse 64-d recall, raw-space verify). Verification is a precision task; precision demands the raw space.

## ADR-4: Zero embedding-API calls; reuse runtime vectors; encode only the answer, locally, once

**Decision:** Context vectors are always carried through from the runtime (retrieval results, warm chunk index) and never recomputed. The answer is encoded once, sentence-batched, on the shared prismlang ONNX session. Chunks arriving without vectors are encoded once and written back. No configuration of PrismShine performs a network embedding call.

**Why:** The vectors already exist — recomputing them is pure waste. Local ONNX answer-encoding costs milliseconds and keeps the pipeline offline-capable and deterministic. This is the direct answer to the "can we reuse preload embeddings?" design question: yes for context (100%), and the answer side is a single unavoidable but near-free local encode, skipped entirely when Tiers 0–1 short-circuit.

## ADR-5: Handbook = data, detectors = code

**Decision:** Failure signatures are declarative YAML entries (id, severity, params, advice) bound to a small set of pure detector functions. Domain packs and tenant overrides are YAML merges, never code forks.

**Why:** Signatures must be versionable, diffable, auditable, and testable as fixtures. Regulated customers need to review "what does the firewall check?" without reading Python. PrismGuard's rules and prismrag's mapping tables set the same precedent.

## ADR-6: Verdicts are content-addressed and replayable

**Decision:** `evidence_hash = SHA-256(canonical bundle)`; verdict cache key includes profile id and handbook version; verdicts are deterministic given (bundle, profile, handbook, model artifacts).

**Why:** Matches PrismCortex's determinism contract and makes verdicts usable as compliance artifacts ("replay this decision"). Also the mechanism that makes repeated verification free.

## ADR-7: Regenerate is a bounded, feedback-carrying loop

**Decision:** `decision="regenerate"` returns unsupported spans + firing signatures as structured repair feedback for the generator; the integration enforces max retries (default 1) and falls back to `flag` — never an unbounded loop.

**Why:** Regeneration without feedback just re-rolls the dice; regeneration without bounds fights ChorusGraph's anti-thrash design (ADR-007 there). One informed retry captures most of the win.

## ADR-8: Adopt a Tier-3 model first, train later

**Decision:** v0 ships with an adopted MIT-licensed span-classifier (LettuceDetect-class) exported to ONNX as the `[spans]` extra. Training a custom model on RAGTruth + ChorusGraph-native traces is a post-v0 track.

**Why:** Adoption de-risks v0 and validates the tier architecture; the proprietary-data advantage (real ledger traces with known-cause hallucinations from Tier-0 labels) is exactly what makes a *later* custom model differentiating.

## ADR-9: Layered strictness with per-technique calibrated thresholds

**Decision:** No global sensitivity number. Thresholds live in a per-technique matrix owned by domain profiles; developers normally touch only a 4-level `strictness` knob that shifts fusion bands; experts can override any threshold per tenant/section; per-request dynamic modifiers (guard gray input, resonance EMERGENCY phase) step strictness up automatically. All raw signals pass through per-technique calibration curves before fusion, and a `prismshine calibrate` harness fits domain/tenant calibration — including a zero-label synthetic-perturbation mode that fabricates hallucinated negatives deterministically from the developer's own grounded traffic.

**Why:** Score distributions differ per comparison technique (raw-384 cosine ≠ JL-64 cosine ≠ token probability ≠ copy-check tolerance) — one knob applied to raw scores is meaningless. Calibrated signals keep fusion weights stable when models change. Market evidence: domain calibration is the single largest quality lever (AUROC ~0.76 → 0.90+). The layered model keeps the common case simple ("pick a profile, maybe turn the knob") without capping expert control. Full details: DESIGN.md §5.5.

## ADR-10: Contradiction cues, arithmetic closure, and composite support are core, not extensions

**Decision:** Three deterministic guards are first-class pipeline features from v0: (1) the Tier-2 contradiction-cue screen — negation asymmetry / opposite-verb lexicon checked against each well-supported sentence's best chunk, promoting hits to Tier 3 instead of passing them; (2) Tier-1 arithmetic closure — unmatched numbers tested as pairwise sum/difference/product/ratio/percent of preload numbers and reclassified `derived` on a hit; (3) Tier-2 composite support — comparative/aggregative sentences scored against the union of their top-k supporting chunks. Contradiction cues live in `grounding/contradiction.py`; closure and composite support inside `copycheck.py` / `coverage.py`.

**Why:** They neutralize the two dominant failure modes of geometric grounding — false negatives on in-register contradictions (cosine is blind to negation) and false positives on legitimately derived/synthesized claims — at zero LLM cost and near-zero latency, using only artifacts the pipeline already computes (best-support chunk, preload number set). Leaving them as "future mitigations" would ship a verifier whose most predictable errors were known at design time.

## ADR-11: Zero hard sibling dependencies; capability detection with transparent degradation

**Decision:** Core install is numpy + pydantic + pyyaml only. Every sibling library (chorusgraph, prismcortex, prismlang, prismlib, prismresonance, prismguard) and every heavy dependency (onnxruntime, judge SDKs) is an optional extra. `ShineGate.build()` performs capability detection and assembles the pipeline from what exists; verdicts record what ran (`tier_reached`, `coverage_mode`, dormant detector families); missing deciders resolve gray zones to `flag`, never `pass`; and pluggable protocols (`Embedder`, `Judge`, `VerdictStore`) let developers substitute components they already have. Full degradation matrix: DESIGN.md §8.2.

**Why:** PrismShine must be adoptable by any stack (LangGraph, custom runtimes) without buying into the Insight ecosystem first — the ecosystem should be the *upgrade path* (each sibling installed lights up more capability), not the entry fee. Transparent degradation preserves the audit-grade contract: a verdict from a minimal install is honest about being weaker, and absence of a capability can never manufacture a false PASS.
