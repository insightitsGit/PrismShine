# MASTER HANDOFF 1 — Implement PrismShine v0.1.0

**Repo:** `C:\code\PrismShine` (github.com/insightitsGit/PrismShine)
**Deliverable:** the `prismshine` Python package, fully implemented per the approved design — no stubs, no faked implementations, every documented API import-tested.
**Design authority (read in this order):** `docs/DESIGN.md` → `docs/HANDBOOK.md` → `docs/INTEGRATION.md` → `docs/DECISIONS.md` (ADR-1…11) → `docs/UPSTREAM.md`. The architecture visual is `docs/prismshine-architecture.png`. This handoff consolidates them into one work order; where this file and the design docs conflict, the design docs win — flag the conflict in the report-back.
**Status of dependencies:** ALL upstream sibling releases are shipped, verified, and on PyPI (Jul 18, 2026): `prismlang 0.1.2`, `prismlib 0.5.0`, `prismlib-plus 0.8.0`, `prismcortex 0.3.0`, `chorusgraph 1.3.0`. Implement against the **native APIs** (exact signatures in §8 below); keep `hasattr` feature detection only as ADR-11 degradation for older third-party pins.

---

## 0. What PrismShine is (one paragraph of context)

Anti-hallucination verdict engine. One `ShineGate.verify(bundle)` pass fuses **cause-side forensics** (Tier 0: deterministic failure signatures over runtime traces — empty retrieval, swallowed tool errors, stale cache, memory conflicts) with **effect-side grounding** (Tier 1 lexical copy-checks → Tier 2 vector coverage reusing runtime embeddings → Tier 3 local ONNX span classifier → opt-in Tier 4 LLM judge) into a single auditable `ShineVerdict` (`pass / flag / block / regenerate`, named resolution gate, spans, evidence hash). Deterministic-first: 0 LLM calls and 0 embedding-API calls on the majority path. Runtime-agnostic core; ChorusGraph is the richest plugin, never required.

## 1. Ground rules (non-negotiable)

1. **No faked implementations.** Every function does what its docstring says. If something must be deferred, it raises `NotImplementedError` with a clear message AND is excluded from public exports and docs — never silently returns a fake success.
2. **Core purity:** `prismshine` core modules import only `numpy`, `pydantic`, `pyyaml`, stdlib. No module outside `prismshine/integrations/` may import a sibling library or runtime. Enforce with a lint test (`tests/test_import_purity.py`) that walks the AST of every core module.
3. **Zero network in core.** The ONLY network call in the whole library is the opt-in Tier-4 judge. No embedding API calls under any configuration (ADR-4). Tier-2 answer encoding uses a local in-process session only.
4. **Determinism:** same `EvidenceBundle` + same config ⇒ byte-identical `ShineVerdict` (excluding timestamps/verdict_id), across runs and platforms. Golden-verdict tests enforce this.
5. **Missing capability may never manufacture confidence** (ADR-11): if the tier that would resolve a gray zone is unavailable, resolve to `flag`, never `pass`.
6. **README examples must be import-tested in CI** (family anti-drift rule). Add `tests/test_readme_examples.py`.
7. Python `>=3.11`, pydantic v2 models, `ruff` clean, coverage gate **80** (already in `pyproject.toml`). Follow the existing `pyproject.toml` — extras and floors are already correct; do not lower floors.
8. Do not push to git without the owner's permission. Commit locally with clear messages per milestone.

## 2. Package layout (build exactly this; scaffold already exists)

```
prismshine/
├── __init__.py            # public exports: ShineGate, EvidenceBundle, PreloadChunk, TraceStep,
│                          #   ShineVerdict, Signal, Span, SignatureHit, __version__
├── models.py              # all pydantic models + profiles (§3)
├── gate.py                # ShineGate: build(), verify(), averify(), capabilities(), early exits (§4)
├── evidence/
│   ├── builder.py         # bundle_from_dict + contract validation + capability feedback (§5)
│   └── adapters/
│       ├── chorusgraph.py # ledger + state + chunk vectors → bundle (§8.1)
│       ├── langgraph.py   # LangGraph state → bundle (§8.2)
│       └── generic.py     # plain dicts → bundle
├── handbook/
│   ├── schema.py          # signature schema (HANDBOOK.md §1)
│   ├── loader.py          # YAML load/merge/version pinning (builtin → domain pack → tenant overrides)
│   └── builtin/core.yaml  # the full v0.1 catalog (HANDBOOK.md §2 — every signature listed there)
├── forensics/
│   ├── engine.py          # run detectors over bundle → SignatureHits (Tier 0)
│   └── detectors/         # retrieval.py, tools.py, context.py, cache.py, memory.py, guardrun.py
├── grounding/
│   ├── copycheck.py       # Tier 1 (§6.1)
│   ├── coverage.py        # Tier 2 support/composite (§6.2)
│   ├── contradiction.py   # Tier 2 cue screen (§6.2)
│   ├── spans.py           # Tier 3 ONNX wrapper (§6.3)
│   └── judge.py           # Tier 4 protocol + openai/gemini reference impls (§6.4)
├── encoder.py             # SharedEncoder (§7)
├── fusion.py              # weighted fusion, bands, gate naming, confidence (§6.5)
├── policy.py              # strictness layers, domain profiles, threshold matrix (§6.6)
├── cache.py               # content-addressed verdict cache: memory + SQLite (§9)
├── audit.py               # verdict records, replay, HRI rolling metrics
├── calibrate.py           # calibration harness (§11)
├── cli.py                 # `prismshine calibrate|capabilities|verify` (entry point already declared)
├── config.py              # PRISMSHINE_* env + programmatic config
└── integrations/
    ├── chorusgraph.py     # shine_node, interceptor hooks, ledger sink, consistency hooks (§8.1)
    ├── langgraph.py       # node factory + conditional-edge router (§8.2)
    ├── prismguard.py      # output-gate wrapper + GUARD_GRAY_INPUT consumption
    └── prismcortex.py     # on_event subscription → invalidation orchestration (§8.3)
```

## 3. Data models (`models.py`) — exact contract

Implement `PreloadChunk`, `TraceStep`, `EvidenceBundle`, `Signal`, `Span`, `SignatureHit`, `ShineVerdict` exactly as specified in DESIGN §4 (field names, types, defaults). Key points implementers miss:

- `EvidenceBundle.answer: str | None` — `None` triggers pre-generation mode (Tier 0 only).
- `PreloadChunk.source` ∈ `{"retrieval","tool","memory","history","cache","system"}` — adapters MUST populate history + memory chunks (DESIGN §12.3; this is a correctness requirement with fixture coverage).
- `PreloadChunk.vector_space` carries the encoder artifact id when known (e.g. `"raw-384@<model_id>"`), enabling `ENCODER_VERSION_MISMATCH`.
- `ShineVerdict` includes ALL of: `decision`, `resolution_gate`, `fused_score`, `confidence`, `signatures`, `spans`, `tier_reached`, `coverage_mode`, `strictness_effective`, `dormant_families`, `evidence_hash`, `verdict_id`, `signals`, `advice`.
- `evidence_hash` = SHA-256 over a canonical JSON serialization of the bundle (sorted keys, normalized floats) — implement one `canonical_bytes(bundle)` helper, reuse it for the verdict cache key.

## 4. Pipeline orchestration (`gate.py`)

- `ShineGate.build(profile=..., handbook=..., judge=None, embedder=None, verdict_store=None, strictness="standard", overrides=None)` — performs **capability detection** (what is importable/configured), assembles the tier pipeline, logs one capability report; `gate.capabilities()` returns it.
- `verify(bundle)` order: Tier 0 → early-exit check → Tier 1 → Tier 2 → fusion probe → (gray?) Tier 3 → (still gray AND judge?) Tier 4 → fusion → verdict. `averify` is the async twin (thread offload for ONNX is fine; no event-loop blocking > ~10 ms).
- **Early exits (must-have, they are the p50 story):**
  - Tier-0 fatal + `halt_on_fatal` → immediate verdict, gate = `HANDBOOK:<SIGNATURE_ID>`; grounding tiers skipped entirely (a broken preload cannot be "passed" by good coverage).
  - Tier 0 clean + Tier 1 zero unmatched + Tier 2 coverage ≥ pass threshold → PASS, gate `CLEAN_FAST_PATH` (expected majority).
  - Tier-2 catastrophic coverage collapse (coverage < `τ_floor` with healthy Tier 0) → act band, gate `T2_COVERAGE_COLLAPSE`. If Tier 0 was UNHEALTHY, the verdict names the forensic gate instead (cause attribution — DESIGN §5.2).
- **Pre-generation mode:** `answer=None` ⇒ Tier 0 only, returns a preload verdict; post-generation verify reuses Tier-0 results via `evidence_hash` (cache them keyed by hash). Tier-0 outcome protocol per DESIGN §3.1 (pass/flag/block/regenerate semantics per moment; every fired signature written to the verdict regardless of outcome).
- **Regenerate is bounded** (ADR-7): default 1 attempt, feedback = unsupported spans + signature advice; then degrade to `block` or `flag` per policy. The loop itself lives in the integrations (enforcement is a plugin capability); the gate only emits the decision + feedback payload.

## 5. Evidence builder (`evidence/builder.py`)

- `bundle_from_dict(d)` validates the minimal contract (question + preload texts; answer optional) and returns `(bundle, feedback)` where feedback lists what the provided data enables/disables ("no vectors → Tier 2 lexical mode; no trace → retrieval/tool/cache detector families dormant; add X to enable Y"). Feedback strings must be specific, not generic.
- Features follow **data**, not products (ADR-11): a caller who populates `trace` from custom logging gets full Tier-0 forensics with zero Insight libraries installed.

## 6. The tiers — math and thresholds (implement exactly; all constants profile-tunable)

### 6.1 Tier 1 — copy-check (`grounding/copycheck.py`)

- Typed fact extraction from the answer: numbers (unit/currency normalization), dates (→ ISO, day granularity), IDs/codes (regex families), capitalized entities (+ optional tenant lexicon). Match = normalized form appears in any preload chunk (numeric tolerance per profile, default ±0.5%, exact for clinical/finance).
- **Arithmetic closure** before flagging an unmatched number: pairwise sum/diff/product/ratio/percent-change over preload numbers (same-unit where units known). Hit ⇒ reclassify `derived` (info signal, no risk contribution by default; strict profiles may escalate derived facts to Tier 3).
- Signal: `unmatched_ratio = unmatched_weighted / total_weighted` (derived excluded). Weights: numbers/currency 3.0, dates 2.0, IDs 3.0, entities 1.0.

### 6.2 Tier 2 — coverage + composite + contradiction cues (`grounding/coverage.py`, `contradiction.py`)

- Deterministic rule-based sentence splitter (zero-dep). Encode sentences once via SharedEncoder (§7); memoize by sentence hash.
- `support(s_i) = max_j cos(v(s_i), c_j)`; `coverage = Σ w_i · 1[support ≥ τ_sent] / Σ w_i`; `risk_coverage = 1 − coverage`. Sentence weights: fact-bearing 2.0, boilerplate 0.25.
- **Composite support** for comparative/aggregative sentences (cue words: "more than", "compared to", "total", "average", "both", enumerations): `support_comp = cos(v(s_i), normalize(Σ top-k c_j))`, take `max(support, support_comp)`.
- **Contradiction-cue screen** on sentences that would pass: negation asymmetry (not/never/no longer/without on exactly one side vs best-supporting chunk) + opposite-verb/adjective lexicon (increase/decrease, approve/deny, safe/unsafe; domain packs extend). Cue hit ⇒ strip supported status, promote as mandatory Tier-3 candidate with a `contradiction_cue` span. Keep the lexicon small and high-precision.
- **Space rule:** raw-384 is the verification space. JL-64-only chunks: stricter `τ_sent` + `LOW_FIDELITY_SPACE` signal (ADR-3). Numbers are NEVER judged by cosine — Tier 1 owns numeric fidelity.
- Optional resonance mode: when chunks carry phase metadata, support = `Re⟨q,p⟩ − λ·|Im⟨q,p⟩|`.
- Lexical fallback mode (`coverage_mode="lexical"`, no encoder available): token-overlap scoring with stricter promotion to Tier 3.

### 6.3 Tier 3 — span classifier (`grounding/spans.py`) — `[spans]` extra

- Adopt a LettuceDetect-class token-classification model (MIT), exported to ONNX, run via onnxruntime (ADR-8: adopt first, train later). Model artifact downloaded via huggingface-hub on first use, cached, artifact id recorded (participates in verdict cache key).
- Per-token unsupported probability → spans from contiguous tokens above `τ_tok` (default 0.5). Signal: `unsupported_span_ratio = unsupported_chars / answer_chars`.
- Runs ONLY on gray-zone verdicts + mandatory candidates (contradiction cues, uncovered fact-bearing sentences). Handle the model's context window (~4–8k tokens) by chunking preload text.

### 6.4 Tier 4 — judge (`grounding/judge.py`) — opt-in only

- `Judge` protocol: any callable `(claims, context) -> JudgeResult`. Reference impls for `[judge-openai]` / `[judge-gemini]`. Claim-level entailment prompt; verdict cached content-addressed; escalation budget enforced per profile (default ≤10% of traffic — track and hard-cap).

### 6.5 Fusion (`fusion.py`)

Weighted linear fusion clamped to [0,1], default weights per DESIGN §5.4 table (fatal 1.0 short-circuit; warnings agg 0.25; T1 0.30; T2 risk_coverage 0.25; contradiction cue unresolved 0.30; T3 0.35; T4 0.45 replacing T2/T3 when present). Default bands `.25/.55/.75`. Every band crossing names its gate. `confidence` = distance from nearest band boundary, discounted by signal disagreement. Signals pass through per-technique calibration curves before fusion (identity curve by default; calibration harness fits real ones).

### 6.6 Strictness / profiles (`policy.py`)

Three layers + dynamic modifiers, precedence `overrides > profile > strictness > defaults` (DESIGN §5.5). Ship the full default threshold matrix from DESIGN §5.5 as data (profiles: default/clinical/finance/legal — clinical/finance/legal live in the licensed handbook packs but the *mechanism* is OSS). Strictness knob shifts bands (lenient +0.08, strict −0.07, paranoid −0.13 + mandatory Tier 3). Dynamic: `GUARD_GRAY_INPUT` or resonance EMERGENCY/ALERT ⇒ strictness +1 for that request, recorded as `strictness_effective`.

## 7. SharedEncoder (`encoder.py`)

- Preference order: (1) user-supplied `Embedder` (`Callable[[list[str]], np.ndarray]`); (2) `prismlang.encoder` via **`get_session()`** (0.1.2 public API — provably one ONNX session per process) with **`model_id()`** recorded; (3) lexical fallback (no encoding).
- Chunks arriving without vectors: encode once, write back into the bundle. Answer sentences: one batched encode, memoized by sentence hash. **Zero embedding-API calls, ever.**

## 8. Integrations — exact shipped APIs to code against

### 8.1 ChorusGraph (`integrations/chorusgraph.py`) — `[chorusgraph]`, floor 1.3.0

| Need | Shipped API (verified) |
|---|---|
| Pre/post-gen hooks | `compiled.register_interceptor(before_llm=hook, after_llm=hook)`; hooks return `InterceptDecision.proceed() / .halt(fallback=...) / .reroute(hop)`; fire inside `NodeContext.call_llm`. **Caveat:** inert for nodes calling raw provider SDKs — document "use `ctx.call_llm`", and always provide `shine_node` as the guaranteed path |
| Post-gen node | build `shine_node(gate, answer_key=...)`: bundle from ledger+state, verify, write `state["shine_verdict"]` + ledger step, route per decision (bounded regenerate) |
| Preload vectors | `ChorusStack.get_chunk_vectors(chunk_ids, partition=...)` → `ChunkVectorRecord(chunk_id, vector_384, partition, version, encoder_artifact_id)` |
| Ledger write-back | `LedgerStep(kind="shine.verdict", detail={...})` — stable third-party contract |
| `MISSING_STATE_KEY` | `add_node(..., consumes=[...])` metadata exposed on NodeContext/ledger |
| Cache-gate revalidation | `chorusgraph.mark_revalidate(sidecar, query_vector=..., threshold=...)` (top-level export) |
| Warm-index invalidation | `ChorusStack.bump_partition_version(partition) -> int` |
| Cache decision age | decision/backend expose entry `created_at` → normalize into cache-kind `TraceStep.detail` |
| Cortex event wiring | `CortexMemoryService.on_event(cb)` and `bind_cache_revalidate(sidecar, threshold=0.55)` exist natively — Shine's hook orchestrates PrismCache eviction alongside |

### 8.2 LangGraph (`integrations/langgraph.py`) — `[langgraph]`

Node factory (`shine_langgraph_node(gate)`) + conditional-edge router applying `pass/flag/block/regenerate` via state flags; evidence from graph state + retriever results (carry vectors when the retriever exposes them, else encode-once write-back). Pre-gen: a node placed before the generator calling `verify(answer=None)`.

### 8.3 PrismCortex + PrismCache consistency hooks (`integrations/prismcortex.py`)

- Subscribe: `unsub = mem.on_event(callback)`; `MemoryEvent(kind ∈ {accommodate, conflict_opened, conflict_resolved, forget}, subject, relation, old_value, new_value, valid_from, source_event_id, tenant_id)`.
- On `accommodate`/`forget`: (a) `cache.invalidate_where(vector, τ_evict=0.55)` and/or `cache.invalidate_tags(subjects)` (PrismCache 0.5.0/0.8.0 — identical signatures); (b) `mark_revalidate(...)` on the cache-gate sidecar; (c) `bump_partition_version(...)` on the warm index. All best-effort prevention; the `CACHE_PREDATES_FACT_UPDATE` detection signature stays on regardless (dual-rail, DESIGN §6.1).
- Hit metadata for detection: `cache.last_hit_meta` (thread-local) or `on_hit` callback → `HitMeta(created_at, tags, llm_model, similarity)`.

### 8.4 PrismGuard (`integrations/prismguard.py`) — `[guard]`

Consume Guard's input verdict (from ledger/state) → `GUARD_GRAY_INPUT` signature + dynamic strictness. Expose the gate as an output-side check with Guard-compatible decision vocabulary.

## 9. Verdict cache (`cache.py`)

Key: `SHA-256(canonical(preload_ids + preload_hash) ‖ answer_norm ‖ profile_id ‖ handbook_version ‖ calibration_version ‖ model_artifact_ids)`. Backends: in-memory LRU + SQLite (`VerdictStore` protocol for BYO). Self-invalidating by construction — no active invalidation of Shine's own cache, ever (ADR-6).

## 10. Handbook (`handbook/`) + forensics (`forensics/`)

- Schema, loader (merge order: builtin → domain pack → tenant overrides), version pinning per HANDBOOK.md §1. Detectors are pure functions `detect(bundle, params) -> list[SignatureHit]`; signatures are YAML data binding detector + params + severity + advice (ADR-5).
- Implement **every** signature in HANDBOOK.md §2 (retrieval, tool/API, context-assembly, cache, memory, guard/run families — including `CONFLICTING_PRELOAD_FACTS` with the exclusive-relation lexicon and `ENCODER_VERSION_MISMATCH`). Every signature ships with ≥2 fixture bundles (fires / adjacent must-NOT-fire). `advice` strings reference concrete evidence fields — generic advice is banned.

## 11. Calibration harness (`calibrate.py` + CLI)

- Labeled mode: 20–100 `(bundle, is_hallucination)` pairs → fit per-technique thresholds + calibration curves → emit profile-overlay YAML + report (AUROC, precision/recall at bands).
- Synthetic-perturbation mode (v0 priority): from grounded bundles, generate hallucinated negatives deterministically (number/date/entity swaps to values absent from preload, chunk drops, cross-chunk splices). Zero LLM calls.
- Calibration artifacts versioned; version participates in verdict cache key.

## 12. Config & CLI

- `config.py`: `PRISMSHINE_PROFILE`, `PRISMSHINE_STRICTNESS`, `PRISMSHINE_HANDBOOK_PATH`, `PRISMSHINE_VERDICT_DB`, `PRISMSHINE_JUDGE_*`, `PRISMSHINE_DISABLE_TIER3`, etc. Programmatic config always wins over env.
- `cli.py`: `prismshine capabilities` (print the gate capability report), `prismshine verify <bundle.json> [--profile ...]` (file-based verify, JSON verdict out), `prismshine calibrate <dir> [--mode synthetic|labeled]`.

## 13. Testing requirements (definition of done)

| Suite | Requirement |
|---|---|
| Unit | every detector (fire + must-not-fire fixtures); copy-check normalization table-tests; arithmetic closure; contradiction cues; fusion bands; strictness layering/precedence |
| Golden verdicts | canonical bundles → snapshot `ShineVerdict`s; byte-identical across runs (determinism) |
| Import purity | no core module imports siblings/runtimes; `pip install prismshine` (bare) yields a working gate |
| Degradation matrix | one test per row of DESIGN §8.2 (missing prismlang → lexical mode; missing onnxruntime → gray→flag; etc.); verify `coverage_mode`/`dormant_families` recorded |
| Integration (extras installed) | ChorusGraph demo graph with injected failures (empty retrieval, tool exception, truncation, stale cache) → correct signatures + gates; interceptor pre-gen halt saves the LLM call; LangGraph flow; Cortex correction event → eviction + revalidation + detection backstop test (prevention disabled → `CACHE_PREDATES_FACT_UPDATE` must still catch 100%) |
| README | all README examples import-run in CI |
| Benchmarks | RAGTruth subset: Tier 2+3 example-level F1 within 5 pts of encoder SotA; latency harness: T0 < 2 ms, fast path < 25 ms p50 (CPU); judge escalation ≤ 10% on default profile against the synthetic traffic set |
| Coverage | ≥ 80 (gate already configured) |

## 14. Performance budgets (hard targets, measured in CI where feasible)

Tier 0 only < 2 ms · fast path (T0+T1+T2 pass) < 25 ms p50 incl. one batched encode · +Tier 3 < 150 ms CPU · Tier 4 = judge latency, ≤10% traffic, cached.

## 15. Implementation order (milestones — commit per milestone)

1. **M1 core contract:** `models.py`, `canonical_bytes`/hashing, `evidence/builder.py` + generic adapter, config. *Gate: golden serialization tests.*
2. **M2 Tier 0:** handbook schema/loader/core.yaml + all detector families + engine. *Gate: full fixture matrix green.*
3. **M3 Tiers 1–2:** copycheck (+closure), splitter, SharedEncoder (all 3 modes), coverage (+composite), contradiction screen. *Gate: unit + lexical-fallback tests.*
4. **M4 gate + fusion + policy + cache + audit:** `verify()`/`averify()` end-to-end with early exits, pre-gen mode, verdict cache, HRI. *Gate: golden verdicts + determinism + degradation matrix.*
5. **M5 Tier 3 + Tier 4:** ONNX span classifier (`[spans]`), judge protocol + reference impls, escalation budget. *Gate: RAGTruth benchmark numbers recorded.*
6. **M6 integrations + CLI + calibration:** chorusgraph (node + interceptor + consistency hooks), langgraph, prismcortex, prismguard, `cli.py`, `calibrate.py`. *Gate: integration suite + README import tests + latency harness.*

Ship as `0.1.0`. OSS scope for v0: full pipeline Tiers 0–3, core handbook, SQLite store, all integrations. Licensed-tier surfaces (domain handbook packs, Postgres store, console hooks) are stubbed as *load points* only (loader accepts extra packs) — do NOT implement licensing enforcement in this handoff.

## 16. Open decisions — pre-made so you don't stall (deviate only with written rationale)

| # | Decision | Call for v0 |
|---|---|---|
| 1 | Tier-3 model | adopt LettuceDetect weights → ONNX (`[spans]`); training our own is out of scope |
| 2 | ChorusGraph attach | RESOLVED — both interceptor (ADR-008 hooks) and `shine_node`; node is the guaranteed path |
| 3 | Handbook distribution | in-wheel (`handbook/builtin/`) |
| 4 | Sentence splitter | rule-based, deterministic, zero-dep |
| 5 | Regenerate protocol | 1 retry, feedback = spans + advice, then degrade per policy |
| 6 | JL fallback threshold | ship 0.80 default, mark "needs calibration" in capability report |
| 7 | Calibration v0 | synthetic-perturbation mode required; labeled mode required (it's simpler); per-domain perturbation review deferred |
| 8 | Threshold matrix | ship DESIGN §5.5 values as-is; benchmark suite records deltas, does not auto-adjust |
| 9 | Cue lexicons | small/high-precision core list in code; domain packs extend via YAML |
| 10 | Feedback loop (`prismshine feedback`) | defer to v0.x — not in this handoff |

## 17. Report back (for verification when you return)

- Module-by-module: implemented / deviated (with why) / deferred (with `NotImplementedError` + doc exclusion confirmed)
- Test counts per suite + pass status + coverage %
- Benchmark numbers: RAGTruth F1 (Tier 2 alone, Tier 2+3), latency p50/p95 per path, judge escalation rate
- The exact public API surface (`prismshine.__all__`) as shipped
- Capability report output (`prismshine capabilities`) for: bare install, `[coverage]`, `[coverage,spans]`, full extras
- Any design-doc conflicts found and how they were resolved
- Confirmation: zero embedding-API calls (grep + test), import purity green, README import tests green
