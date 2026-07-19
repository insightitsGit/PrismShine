# Handoff: PrismCortex 0.2.1 → 0.3.0 — correction events for PrismShine coupling

> **STATUS: SHIPPED & VERIFIED (Jul 18, 2026)** — `Memory.on_event` + `MemoryEvent` (with MeshBroadcast fan-out), superseded-fact provenance, and the `prism-plus` extra all landed; tests green (44 passed); on PyPI. Verification record in `docs/UPSTREAM.md`.

**Repo:** `C:\code\PrismCortex` (github.com/insightitsGit/PrismCortex)
**Requested by:** PrismShine (design at `C:\code\PrismShine\docs\` — see `INTEGRATION.md` §4 and §6)
**Priority:** MEDIUM — PrismShine has a polling fallback; events make the prevention rail push-based.

## Context (why)

When a user corrects a fact ("Person A is my brother" → "sister"), Cortex handles the graph via `ACCOMMODATE` (close old fact's `valid_to`, insert new). But semantic caches elsewhere in the stack (PrismCache, ChorusGraph cache gate) may still hold answers generated from the old fact. PrismShine subscribes to correction events and evicts/revalidates those entries at correction time. Without events it must poll `conflicts()` and diff graph versions per run — correct but late and wasteful.

## Requested changes

### 1. Memory event hook  (required)

```python
mem.on_event(callback: Callable[[MemoryEvent], None]) -> Unsubscribe
```

`MemoryEvent` (pydantic): `kind` (`"accommodate" | "conflict_opened" | "conflict_resolved" | "forget"`), `subject`, `relation`, `old_value | None`, `new_value | None`, `valid_from`, `source_event_id`, `tenant_id | None`. Synchronous dispatch, exceptions in callbacks logged and swallowed (memory operations must never fail because a subscriber did). Consider routing through the existing `MeshBroadcast` port so remote subscribers work too — implementer's choice, document it.

### 2. Correction metadata in recall provenance  (nice-to-have)

On recalled facts, expose whether the fact superseded another and the superseding `valid_from`, so PrismShine can compare cache-entry `created_at` against correction time without extra bitemporal queries.

### 3. Split the `prism` extra to stop the prismlib / prismlib-plus co-install  (required, packaging only)

The `prism` extra currently hard-lists `prismlib>=0.4.0`. ChorusGraph core depends on `prismlib-plus`, and **both packages install the same `prism` import** — so `pip install chorusgraph[cortex]` co-installs both into one `prism` directory and install order decides which cache implementation wins. Fix in 0.3.0:

```toml
[project.optional-dependencies]
prism      = ["prismlang>=0.1.1", "prismlib>=0.5.0",      "prismrag-patch>=0.2.1", "prismresonance>=0.3.0"]
prism-plus = ["prismlang>=0.1.1", "prismlib-plus>=0.8.0", "prismrag-patch>=0.2.1", "prismresonance>=0.3.0"]
```

Keep `prism` as-is for standalone users (floor bumped to the release with the new cache APIs); add `prism-plus` so hosts already on prismlib-plus (ChorusGraph — its 1.3.0 handoff switches the `cortex` extra to `prism-plus`) never pull `prismlib`. Document in the README that the two extras are mutually exclusive.

## Constraints

- No breaking changes; hooks inert when unused.
- Determinism guarantees unaffected (events are observability, not state).
- Tests: accommodate fires event with correct old/new values; conflict open/resolve fire; unsubscribe works; a throwing callback doesn't break `digest`. Follow existing test style (~44 tests across 7 files).
- Version bump to 0.3.0.

## Report back (for verification when you return)

- Exact `MemoryEvent` schema and `on_event` signature as implemented
- Whether MeshBroadcast routing was included
- Test count before/after and pass status
- Any deviation from spec and why
