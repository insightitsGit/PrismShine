# Handoff: prismlib 0.4.0 → 0.5.0 — PrismCache API for PrismShine coupling

> **STATUS: SHIPPED & VERIFIED (Jul 18, 2026)** — `invalidate_where`/`invalidate_tags`/tagged entries/`HitMeta`+`last_hit_meta`+`on_hit` all landed incl. SQLite persistence; version synced in both files; tests green (203 passed); on PyPI. Verification record in `docs/UPSTREAM.md`.

**Repo:** `C:\code\PrismLib` (github.com/insightitsGit/prismlib)
**Requested by:** PrismShine (anti-hallucination verdict engine, design at `C:\code\PrismShine\docs\`)
**Priority:** HIGH — completes the prevention rail of PrismShine's consistency contract (DESIGN.md §6.1)

## Context (why)

PrismShine detects when a semantic cache serves an answer generated from a fact that has since been corrected (e.g. user said "Person A is my brother", later corrected to "sister" — a paraphrased query can hit the stale cached answer, bypassing recall and generation entirely). PrismShine needs PrismCache to support **selective invalidation** and **entry metadata exposure**. Today only `invalidate_all()` exists, which destroys the tenant's hit rate on every correction.

## Requested changes

### 1. `invalidate_where(vector, threshold) -> int`  (required)

Evict all entries whose stored (projected) query vector has cosine similarity ≥ `threshold` to the given vector. Vector arrives in the same tenant-projected space as stored entries (caller projects via the existing `TenantSpace`). Return count evicted. Must work on the in-memory store and the persisted store (`persist_path`).

### 2. Tagged entries + `invalidate_tags(tags) -> int`  (required)

- `get_or_call(..., tags: list[str] | None = None)` — optional subject/entity tags stored with the entry (e.g. `["person_a", "family"]`).
- `invalidate_tags(tags: list[str]) -> int` — evict entries having ANY of the tags.
- Tags must survive persistence round-trip.

### 3. Expose entry metadata on hits  (required)

On a cache hit, the caller must be able to read: `created_at` (epoch float), `tags`, `llm_model`, similarity score of the hit. Suggested: a `last_hit_meta` accessor or an optional richer return via a new method (do NOT change the `get_or_call` return type — backward compatibility).

### 4. `on_hit` callback  (nice-to-have)

`PrismCache.build(..., on_hit: Callable[[HitMeta], None] | None = None)` — invoked synchronously on every hit with the metadata from (3). Lets PrismShine feed cache decisions into its evidence bundle outside ChorusGraph.

## Constraints

- **No breaking changes** to existing public API (`build`, `get_or_call`, `aget_or_call`, `invalidate_all`, `purge_expired`, `get_metrics`).
- Also fix while in the area: `prism/__init__.py` still says `__version__ = "0.1.0"` while pyproject says 0.4.0 — sync both to 0.5.0.
- Follow existing test conventions (`tests/test_cache*.py`); add tests for: vector eviction hit/miss boundary at threshold, tag eviction, persistence round-trip of tags/created_at, callback firing, and thread-safety of eviction during concurrent `get_or_call`.
- Update README cache section with the new APIs (and make sure examples import-run — README drift is a known family problem).

## Report back (for verification when you return)

- New/changed public signatures (exact)
- Test count before/after and pass status
- Any deviation from this spec and why
- Confirm version bumped to 0.5.0 in BOTH pyproject.toml and `prism/__init__.py`
