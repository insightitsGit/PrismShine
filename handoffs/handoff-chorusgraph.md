# Handoff: ChorusGraph 1.2.0 → 1.3.0 — runtime hooks for PrismShine coupling

> **STATUS: SHIPPED & VERIFIED (Jul 18, 2026)** — all 8 items landed: ADR-008 provider-boundary interceptors (`register_interceptor` + `InterceptDecision`, firing in `NodeContext.call_llm` — note: inert for nodes calling raw provider SDKs), `mark_revalidate`, `bump_partition_version`, third-party `LedgerStep.kind/detail`, `consumes=` node metadata, `get_chunk_vectors() -> ChunkVectorRecord` (raw 384-d + version + `encoder_artifact_id`), cache-decision `created_at`, `cortex` extra → `prismcortex[prism-plus,gemini]>=0.3.0`, floors bumped to prismlang>=0.1.2 / prismlib-plus>=0.8.0. Tests green (403 passed); on PyPI. Verification record in `docs/UPSTREAM.md`.

**Repo:** `C:\code\ChorusGraph` (github.com/insightitsGit/ChorusGraph)
**Requested by:** PrismShine (anti-hallucination verdict engine, design at `C:\code\PrismShine\docs\` — read `DESIGN.md` §3.1, §6.1, §8.1 and `INTEGRATION.md` §1 for full context)
**Priority:** HIGH — item 1 below decides PrismShine's primary integration architecture.

## Context (why)

PrismShine verifies agent answers in two moments: **pre-generation** (Tier-0 forensics over the Route Ledger and node state — halt before tokens are spent if the preload is broken, e.g. retrieval returned 0 chunks) and **post-generation** (grounding verification). It also needs to invalidate/revalidate cache-gate entries when facts are corrected (consistency contract). ChorusGraph is the richest runtime plugin; these hooks make the coupling first-class instead of pattern-based.

## Requested changes

### 1. LLM-hop interceptor API  (required — architectural decision)

A supported way to run callbacks around generator hops in `CompiledGraph`:

```python
compiled.register_interceptor(
    before_llm=Callable[[NodeContext, state], InterceptDecision],  # proceed | halt(fallback) | reroute(hop)
    after_llm=Callable[[NodeContext, state, output], InterceptDecision],
)
```

- `before_llm` must run after the hop's prompt inputs are resolved (so the interceptor can see the final preload) and before the provider call.
- Halt must integrate with existing interrupt semantics (`GraphInterrupt` or a clean sibling), and reroute must respect recursion limits + anti-thrash (ADR-007).
- If a scheduler-level interceptor is architecturally wrong for BSP super-steps, propose the closest alternative (e.g. a wrap-node contract with guaranteed placement) — document the reasoning; PrismShine has a node-based fallback either way.

### 2. Cache-gate revalidation flag  (required)

Public API to mark sidecar/cache entries "must revalidate": next hit on a marked entry returns `HIT_REVALIDATE` instead of `HIT_REUSE`, regardless of scores. Suggested: `sidecar.mark_revalidate(entry_ids | vector, threshold)` mirroring `seed_cache_entry` ergonomics.

### 3. Public warm-index partition version bump  (required)

The `index(partition, version)` versioning exists internally (ADR-005). Expose `bump_partition_version(partition: str) -> int` (returns new version) so external events (fact corrections, source updates) can invalidate warm chunk vectors without a re-index.

### 4. Custom ledger step kinds  (required)

Documented support for third-party `LedgerStep` kinds (e.g. `kind="shine.verdict"`) with arbitrary `detail` payloads, queryable via `get_run`/`list_runs` and visible in `chorusgraph-audit`. If already possible, add tests + docs making it a stable contract.

### 5. Declared state-key consumption  (nice-to-have)

Optional node metadata declaring which state keys a generator's prompt consumes (e.g. `add_node(..., consumes=["docs", "history"])`), exposed on `NodeContext`/ledger. Enables exact "template referenced an empty state key" detection (PrismShine `MISSING_STATE_KEY`).

### 6. Public read path for warm-index chunk vectors + partition version  (required — confirm or expose)

PrismShine's evidence adapter must populate `preload[].vector` with the **raw 384-d chunk vectors** already held by the warm chunk index (ADR-005 `index(partition, version)`) and record the partition version per chunk (for `STALE_CACHE_REUSE`). If a public/supported read API for `(chunk_id → raw vector, partition, version)` already exists, add tests + docs making it a stable contract; if the raw vectors are internal-only, expose a read accessor. This is the linchpin of PrismShine's zero-re-embed guarantee inside ChorusGraph — without it the adapter would have to re-encode preload chunks, which the design forbids.

### 7. Encoder artifact id on warm-index entries  (nice-to-have)

Stamp the embedding model's artifact id (from `prismlang.encoder.model_id()`, landing in prismlang 0.1.2 — see `handoff-prismlang.md`) on warm-index entries or partitions, and surface it on retrieval results / ledger steps. Enables PrismShine's `ENCODER_VERSION_MISMATCH` detection to be exact (chunk vectors embedded under model A vs answer verified under model B). Fallback without it: PrismShine assumes all runtime vectors used the currently loaded model — true in steady state, blind across model upgrades with a warm index that survived them.

### 8. Entry `created_at` in cache-decision ledger detail  (nice-to-have)

Cache-gate hit decisions (`HIT_REUSE`/`HIT_REVALIDATE`/`HIT_AS_CONTEXT`) recorded in the ledger should include the hit entry's creation timestamp (and tags, if the sidecar ever carries them). Enables PrismShine's `CACHE_PREDATES_FACT_UPDATE` signature on cache-gate hits (compare entry age vs fact-correction time) — the prismlib handoffs cover this for PrismCache hits, but gate hits are only visible through the ledger. Fallback: the signature degrades to warning-severity on undated gate hits.

## Coordination with prismlib-plus 0.8.0

ChorusGraph hard-depends on `prismlib-plus` and uses PrismCache via `prism.cache`. In the same handoff wave, prismlib-plus 0.8.0 adds `invalidate_where(vector, threshold)`, tagged entries + `invalidate_tags(tags)`, and hit metadata (`created_at`, tags, hit score) — see `handoff-prismlib-plus.md`. Implications here:

- **No code change required for compatibility** — the PrismCache additions are inert/additive; existing ChorusGraph call sites are untouched.
- **Item 2 (cache-gate revalidation) may want to build on them** — e.g. entry `created_at`/tags from the new metadata surface instead of a parallel bookkeeping layer. Implementer's choice; if used, bump the dependency floor to `prismlib-plus>=0.8.0` in 1.3.0.
- **Item 8 (entry `created_at` in cache-decision ledger detail)** becomes nearly free once the 0.8.0 metadata surface exists — read it on hit and copy it into the ledger step.
- If both handoffs are done by the same person, land prismlib-plus 0.8.0 first.
- **Fix the `cortex` extra's package collision:** it currently resolves to `prismcortex[prism,...]`, whose `prism` extra hard-lists `prismlib` — while ChorusGraph core requires `prismlib-plus`. Both install the same `prism` import, so `chorusgraph[cortex]` co-installs both and install order decides which cache implementation wins. Once PrismCortex 0.3.0 ships its new `prism-plus` extra (see `handoff-prismcortex.md` item 3), change the extra to `cortex = ["prismcortex[prism-plus,gemini]>=0.3.0"]`.

## Constraints

- No breaking changes; all additions must be inert when unused.
- Deterministic-tier tests for every new API (repo convention: ~393 tests, coverage gate 71) + docs/ADR entries per repo convention (this repo keeps ADRs — add one for the interceptor decision).
- Version bump to 1.3.0.

## Future (NOT in 1.3.0 — awareness only)

PrismShine's post-v0 streaming track (verify sentence-by-sentence during streamed generation, retract/correct on failure — DESIGN.md §12.4) will need transport-level involvement from ChorusGraph. No action now; flagging it so 1.3.0 API choices (especially the interceptor) don't preclude a streaming hook later.

## Report back (for verification when you return)

- Which option was chosen for item 1 (interceptor vs alternative) + the new API signatures (exact)
- New APIs for items 2–4 and 6 (exact signatures; for 6, state whether the read path already existed)
- Test count before/after, pass status, coverage
- Any deviation from spec and why
