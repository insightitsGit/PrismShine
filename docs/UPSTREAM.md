# PrismShine — Upstream Coordination

> **STATUS: ALL SHIPPED & VERIFIED (Jul 18, 2026).** Every item below landed and was verified against source, tests, and PyPI: prismlang **0.1.2**, prismlib **0.5.0**, prismlib-plus **0.8.0**, prismcortex **0.3.0**, chorusgraph **1.3.0** (all green test suites: 42/203/225/44/403 passed respectively). The tables below are kept for the rationale record; the "Fallback until shipped" columns are now history — PrismShine implementation should target the native APIs, keeping `hasattr` feature detection only as ADR-11 degradation for third-party users pinned to older versions.
>
> Notable implementation choices vs. spec: ChorusGraph's interceptor landed as **provider-boundary hooks (ADR-008)** — `CompiledGraph.register_interceptor(before_llm=, after_llm=)` + `InterceptDecision.proceed/halt/reroute` firing inside `NodeContext.call_llm` — rather than scheduler-level wrapping; hooks are inert for nodes that call a raw model directly (see INTEGRATION §1). The warm-index read path shipped as `ChorusStack.get_chunk_vectors() -> ChunkVectorRecord` (raw 384-d vector, partition, version, `encoder_artifact_id`), covering handoff items 6 and 7 together.

Changes needed (or desired) in sibling Insight libraries for first-class PrismShine coupling. Nothing here **blocks** PrismShine v0 — every row has a fallback — but shipping these upstream turns best-effort behavior into guaranteed behavior. Ordered by impact.

---

## 1. prismlib / prismlib-plus — proposed 0.5.0 / 0.8.0 (PrismCache API additions)

| # | Change | Enables in PrismShine | Fallback until shipped |
|---|---|---|---|
| 1.1 | **Selective invalidation**: `PrismCache.invalidate_where(vector, threshold: float) -> int` (evict entries whose stored query vector is within threshold) | correction-driven eviction (`on_fact_corrected`, INTEGRATION §6) without nuking the tenant cache | `invalidate_all()` on exclusive-relation corrections — correct but hit-rate-destructive |
| 1.2 | **Tagged entries**: optional `tags: list[str]` on cache writes + `invalidate_tags(tags) -> int` (subject/entity tags) | precise eviction by subject (e.g. "person_a"), cheaper than vector scans | vector-similarity eviction (1.1) or full invalidation |
| 1.3 | **Entry metadata exposure**: `created_at` timestamp + tags readable on cache hits (in the hit result / metrics API) | `CACHE_PREDATES_FACT_UPDATE` detection needs the entry's creation time to compare against correction `valid_from` | ChorusGraph-side sidecar timestamps where available; otherwise signature degrades to warning on undated hits |
| 1.4 | **Hit event hook**: optional callback `on_hit(entry_meta)` | zero-copy feed of cache decisions into the EvidenceBundle outside ChorusGraph (standalone prismlib users) | inside ChorusGraph the ledger already records decisions; standalone users lose this signal |

Also worth fixing in the same release (ecosystem hygiene, not Shine-specific): the `prismlib` vs `prismlib-plus` `prism` import collision, and the `get_metrics()` README drift.

## 2. ChorusGraph — proposed 1.3.0

| # | Change | Enables in PrismShine | Fallback until shipped |
|---|---|---|---|
| 2.1 | **LLM-hop interceptor API**: pre/post hooks on the scheduler around generator hops (`CompiledGraph.register_interceptor(...)` or equivalent) | pre-generation halting (Tier-0 preload verdict *before* tokens are spent) as a supported API instead of a node-graph pattern | `shine_node` placed before/after the generator — works, but pre-gen halting requires graph authors to wire an extra routing edge |
| 2.2 | **Cache-gate revalidation flag**: public API to mark sidecar entries "must revalidate" (force `HIT_REVALIDATE` on next hit) | correction-driven soft invalidation of the L1/L2 gate without eviction | seed_cache_entry overwrite tricks or eviction; coarser |
| 2.3 | **Warm-index partition version bump**: public `bump_partition_version(partition)` (ADR-005 versioning exists internally) | `on_fact_corrected` / `on_source_updated` prevention rail; makes `STALE_CACHE_REUSE` detection exact | re-index the partition; heavier |
| 2.4 | **Custom ledger step types**: documented API for third-party steps (`LedgerStep(kind="shine.verdict", ...)`) so verdicts are first-class in `get_run`/`list_runs`/audit CLI | verdict write-back visible in all ledger tooling | write verdicts as generic detail payloads on existing step kinds; readable but not queryable by kind |
| 2.5 | **Declared state-key consumption**: optional node metadata declaring which state keys a generator's prompt template consumes | exact `MISSING_STATE_KEY` detection (template referenced a key that was empty at generation time) | heuristic: diff prompt text against state keys; more false negatives |
| 2.6 | **Public read path for warm-index chunk vectors + partition version** (confirm existing API is public/stable, or expose one) | the zero-re-embed guarantee inside ChorusGraph: adapter carries raw 384-d vectors + partition version into the bundle | if internal-only: encode-once write-back path (costs the one encode the design tries to avoid) |
| 2.7 | **Encoder artifact id stamped on warm-index entries** (from `prismlang.encoder.model_id()`, item 4.1) | exact `ENCODER_VERSION_MISMATCH` across model upgrades that survive a warm index | assume current-process model id for all runtime vectors; blind across upgrades |
| 2.8 | **Entry `created_at` in cache-decision ledger detail** | `CACHE_PREDATES_FACT_UPDATE` on cache-*gate* hits (prismlib items 1.3 cover PrismCache hits) | signature degrades to warning on undated gate hits |

## 3. PrismCortex — proposed 0.3.0

| # | Change | Enables in PrismShine | Fallback until shipped |
|---|---|---|---|
| 3.1 | **Correction event hook**: subscribe to `ACCOMMODATE`/conflict-open events (`Memory.on_event(callback)` or a `MeshBroadcast`-port emission) | push-based `on_fact_corrected` — the prevention rail fires at correction time, not at next-query time | poll `Memory.conflicts()` + compare graph versions per run; detection rail (`CACHE_PREDATES_FACT_UPDATE`) still guarantees correctness |
| 3.2 | **Correction metadata in recall provenance**: expose `valid_from` of the *superseding* fact on recalled facts | precise timestamp comparison for stale-cache detection | derivable from bitemporal queries (`subgraph_at`), just more calls |

## 4. prismlang — proposed 0.1.2 (patch)

| # | Change | Enables in PrismShine | Fallback until shipped |
|---|---|---|---|
| 4.1 | **Encoder artifact id**: expose model/artifact version on the encoder module (e.g. `encoder.model_id()`), and accept an externally shared ORT session handle | `ENCODER_VERSION_MISMATCH` detection; guaranteed session sharing (one ONNX session per process) | hash the model file on disk for an id; session sharing already works de facto when host initializes first |

## 5. PrismGuard — no work order (boundary note only)

Nothing required for PrismShine v0 (the `GUARD_GRAY_INPUT` signal reads Guard's existing verdict). One future boundary item worth tracking on the Guard roadmap: **scanning retrieved content for injected instructions** (the "grounded-but-poisoned preload" scope boundary, DESIGN.md §12.5) is input-side territory that naturally belongs to PrismGuard, not PrismShine. No timeline pressure.

## Release sequencing recommendation

1. **PrismShine v0 develops against current versions** using the fallbacks — no upstream wait.
2. **prismlib 1.1–1.3 and ChorusGraph 2.2–2.3, 2.6** are the highest-value bumps (they complete the consistency contract's prevention rails and the zero-re-embed read path) — target them while PrismShine is in development so v0.1 can require them.
3. ChorusGraph **2.1 (interceptor)** decides open question #2 in DESIGN §13 — coordinate early because it shapes `integrations/chorusgraph.py`.
4. Cortex 3.1 and prismlang 4.1 are quality-of-life; schedule opportunistically.

Compatibility floor for PrismShine v0 (updated post-ship): extras now target **`chorusgraph>=1.3.0`, `prismlib-plus>=0.8.0`, `prismcortex>=0.3.0`, `prismlang>=0.1.2`** — the versions with native PrismShine APIs. Feature detection (`hasattr`) remains in the adapters for graceful behavior on older pins (ADR-11), but is no longer the primary path.

## Dependency-safety audit of the wave (verified from each repo's pyproject, Jul 2026)

**No version conflicts are possible from these bumps.** Every cross-dependency in the family is an open floor (`>=`) with no upper pins; all five releases are additive minor bumps, so every existing floor remains satisfied (`prismlang>=0.1.1` ← 0.1.2, `prismlib-plus>=0.7.0` ← 0.8.0, etc.). Shared third-party floors also co-resolve: `numpy` ≥1.24/≥1.26, `pydantic` ≥2.0/≥2.5, `onnxruntime` ≥1.17 everywhere, Python ≥3.10/≥3.11.

**One real hazard, fixed in this wave — the `prism` import collision.** `prismlib` and `prismlib-plus` both ship the `prism` package. `chorusgraph[cortex]` → `prismcortex[prism]` → `prismlib`, while ChorusGraph core requires `prismlib-plus`: both get installed into the same `prism` directory and install order decides which cache implementation wins. Benign while the copies were near-identical; unacceptable once the new cache APIs land. Fixes: PrismCortex 0.3.0 adds a `prism-plus` extra (handoff item 3) and ChorusGraph 1.3.0 switches its `cortex` extra to it (handoff coordination note). Long-term hygiene fix (out of this wave): rename one package's import root.

**Noted, no action:** `prismlang` hard-depends on `langgraph`, so `prismshine[coverage]` transitively installs langgraph even for non-LangGraph users. Harmless (extras are opt-in, and PrismShine's BYO-`Embedder` protocol avoids prismlang entirely if desired); moving langgraph to a prismlang extra would be a nice-to-have in some future prismlang 0.2.0, not requested now.
