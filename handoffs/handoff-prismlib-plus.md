# Handoff: prismlib-plus 0.7.0 → 0.8.0 — PrismCache API parity for PrismShine coupling

> **STATUS: SHIPPED & VERIFIED (Jul 18, 2026)** — full parity with prismlib 0.5.0 signatures + `evicted_by_vector`/`evicted_by_tags` metrics + Prometheus counters + README fix; tests green (225 passed); on PyPI. Verification record in `docs/UPSTREAM.md`.

**Repo:** `C:\code\PrismLabPlusAPI` (github.com/insightitsGit/prismlibplusapi)
**Requested by:** PrismShine (design at `C:\code\PrismShine\docs\`)
**Priority:** HIGH — prismlib-plus is the version ChorusGraph actually depends on (`prismlib-plus>=0.7.0`), so this is the one PrismShine will hit in production.

## Context (why)

Same requirement as the prismlib handoff (`handoff-prismlib.md` in this folder — read it first): PrismShine needs selective cache invalidation and hit metadata to purge/flag cached answers generated from since-corrected facts. prismlib-plus ships its own copy of `prism.cache`, so the changes must land here too (implement the identical spec).

## Requested changes

1–4. **Identical spec to `handoff-prismlib.md`**: `invalidate_where(vector, threshold)`, tagged entries + `invalidate_tags(tags)`, entry metadata on hits (`created_at`, `tags`, model, hit score), optional `on_hit` callback. Same constraints (no breaking changes; persistence round-trip; thread safety).

Plus, specific to this repo:

5. **Metrics surface** (nice-to-have): expose eviction counts (`evicted_by_vector`, `evicted_by_tags`) in `CacheMetrics` and in the Prometheus observability layer (`prism.observability`), so operators can see correction-driven eviction activity.

6. **Docs fix while in the area**: README says `cache.metrics()` but the code is `get_metrics()` — fix the README (known drift, noted in the PrismShine KB audit).

## Constraints

- No breaking changes to existing public API.
- Keep the implementation consistent with whatever lands in prismlib 0.5.0 (same method names/semantics), since both packages install as `prism` and developers switch between them.
- Tests per the prismlib handoff, added under this repo's `tests/` conventions (~204 existing test functions — follow their style).
- Version bump to 0.8.0 (pyproject + `prism/__init__.py` `__version__ = "0.8.0"`).

## Report back (for verification when you return)

- New/changed public signatures (exact) and confirmation they match prismlib 0.5.0's
- Test count before/after and pass status
- Metrics fields added (names)
- Any deviation from spec and why
