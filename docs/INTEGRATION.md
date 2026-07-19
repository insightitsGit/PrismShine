# PrismShine — Ecosystem Integration

How PrismShine attaches to each sibling library. Verified against the KB (`kb/libraries/*.md`) — integration points below reference real APIs of the current versions.

---

## 1. ChorusGraph (primary host) — `prismshine[chorusgraph]`

### Attach points

**a) Post-generation node (guaranteed path, v0):**

```python
from prismshine import ShineGate
from prismshine.integrations.chorusgraph import shine_node

gate = ShineGate.build(profile="finance")
g.add_node("shine", shine_node(gate, answer_key="reply"))
g.add_edge("generate", "shine")
g.add_edge("shine", END)
```

The node builds an `EvidenceBundle` from the run's Route Ledger steps + current node state, verifies, writes the verdict into state (`state["shine_verdict"]`) and as a ledger step, and routes: `pass/flag` → continue, `block` → interrupt/replace answer, `regenerate` → loop back to generator with unsupported spans appended as repair feedback (bounded retries, respecting ChorusGraph anti-thrash conventions).

**b) Interceptor (pre-generation mode — SHIPPED in ChorusGraph 1.3.0, ADR-008):**

```python
compiled.register_interceptor(
    before_llm=shine_before_hook(gate),   # Tier-0 preload verdict → InterceptDecision.proceed/halt/reroute
    after_llm=shine_after_hook(gate),     # full verify, Tier-0 reused via evidence hash
)
```

Before the call: a fatal signature (e.g. `EMPTY_RETRIEVAL`) halts (`NodeInterrupt` with fallback) or reroutes to a repair hop before tokens are spent. After the call: full verify; `halt(fallback=...)` replaces the answer. Same pipeline, both moments.

**Contract caveat (matters for graph authors):** the hooks fire at the provider boundary inside `NodeContext.call_llm` (or an `Agent` built with the wrapped model). Nodes that call a raw provider SDK directly bypass the interceptor — for those graphs, use the `shine_node` post-generation path (a), which always works. PrismShine's ChorusGraph adapter should document "use `ctx.call_llm` to get pre-generation halting."

### Evidence mapping (ledger → bundle)

| ChorusGraph artifact | EvidenceBundle field |
|---|---|
| `RouteLedger` / `LedgerStep` (hop, scores, durations, rule_chain, cache `Decision.kind`; stable third-party `kind`/`detail` fields in 1.3.0) | `trace[]` (normalized `TraceStep`) |
| `ChorusStack.get_chunk_vectors(chunk_ids, partition=...)` → `ChunkVectorRecord` (raw 384-d vector, partition, version, `encoder_artifact_id`) — 1.3.0 public read path | `preload[]` with **reused raw vectors** + partition version (`STALE_CACHE_REUSE`) + encoder id (`ENCODER_VERSION_MISMATCH`) |
| graph state snapshot | `node_state` |
| cache gate decisions (`HIT_REUSE`/`HIT_REVALIDATE`/`HIT_AS_CONTEXT`/`MISS`, coarse/verify scores, entry `created_at` in 1.3.0) | `trace[]` cache-kind steps (entry age feeds `CACHE_PREDATES_FACT_UPDATE`) |
| `add_node(..., consumes=["docs", ...])` metadata (1.3.0) on `NodeContext`/ledger | exact `MISSING_STATE_KEY` detection |
| tenant / section config | `tenant_id`, `declared_sections` |

Verdicts write back as ledger steps so `chorusgraph-audit` and future trace consoles see Shine decisions inline with hops.

## 2. prismlang — `prismshine[coverage]`

- `prismshine.encoder.SharedEncoder` wraps `prismlang.encoder` (ONNX all-MiniLM-L6-v2, 384-d) via the public `get_session()` (0.1.2): if the host process (ChorusGraph) already initialized the encoder, the same session is provably reused — zero additional model load or memory. `encoder.model_id()` (0.1.2) supplies the artifact id for `ENCODER_VERSION_MISMATCH` and the verdict cache key.
- Raw 384-d space is the verification space (not 64-d JL — lossy, tenant-seeded; see DECISIONS ADR-3).
- No prismlang installed → Tier 2 degrades gracefully: hash-based lexical coverage with a `LOW_FIDELITY_SPACE`-style signal, Tiers 0–1 unaffected.

## 3. PrismGuard (peer — input + output symmetry) — `prismshine[guard]`

- `prismshine.integrations.prismguard` exposes the gate as an output-side check usable in PrismGuard-style pipelines, and consumes PrismGuard's input verdict: a gray-zone input (`GUARD_GRAY_INPUT` signature) raises output scrutiny for that run.
- Shared vocabulary: both libraries emit `decision + resolution_gate + fused_score`, so one audit trail covers inbound (Guard) and outbound (Shine).
- Combined recipe: `make_guard_handler(checker)` at graph entry (exists in PrismGuard today) + `shine_node(gate)` at graph exit.

## 4. PrismCortex — signal source

- Forensics detectors call `Memory.conflicts()` and inspect recall provenance to fire `MEMORY_CONFLICT_SERVED` / `STAGED_FACT_SERVED` / `EXPIRED_FACT_SERVED` / `CONFLICTING_PRELOAD_FACTS`.
- Verdict determinism mirrors Cortex: content-addressed `evidence_hash`, replayable verdicts. A future joint "replay certificate" can bind Cortex's fact provenance and Shine's grounding verdict for a single answer.

### Division of responsibility for contradictory user facts

Worked example — session 1: "Person A is my brother"; session 2: "Person A is my sister".

| Concern | Owner | Mechanism |
|---|---|---|
| Store both statements without losing history | PrismCortex | bitemporal edges; nothing is deleted |
| Recognize an explicit correction and swap the fact | PrismCortex | salience fast-track + `ACCOMMODATE` (close old `valid_to`, insert new) |
| Hold a *silent* contradiction open instead of guessing | PrismCortex | delta calc → conflict/staging; `conflicts()`, `resolve_conflict()` |
| Refuse to let an answer silently rely on the conflicted fact | **PrismShine** | `MEMORY_CONFLICT_SERVED` (recall path) / `CONFLICTING_PRELOAD_FACTS` (raw-history path), pre-generation capable |
| Purge/flag cached answers generated from the pre-correction fact | **PrismShine** + PrismCache | `on_fact_corrected` eviction hook + `CACHE_PREDATES_FACT_UPDATE` detection (§6) — cache hits bypass recall AND generation, so they need their own guard |
| Resolve | the agent + user | PrismShine advice → agent asks a clarifying question; user's reply becomes a Cortex correction |

Key point: PrismShine never adjudicates which user statement is true — it guarantees the contradiction is surfaced (to the agent, and via advice, to the user) instead of being papered over by a fluent answer.

## 5. PrismResonance — optional Tier-2 scoring mode

When preload chunks carry phase metadata (frequency families), Tier-2 support uses interference scoring instead of plain cosine, so an ARCHIVE-phase chunk cannot "support" an ACTIVE-context answer. Uses the same math as the in-process engines (`Re⟨q,p⟩ − λ·|Im⟨q,p⟩|`); no separate install needed when running inside ChorusGraph (resonance is already present).

## 6. PrismCache / prismlib-plus — verdict reuse AND correction-driven invalidation

- Exact reuse: PrismShine's own content-addressed cache (`prismshine.cache`, memory + SQLite).
- Paraphrase-level reuse (optional): route verdict lookups through PrismCache keyed by answer vector + preload hash, for high-traffic deployments where near-identical answers recur across sessions.

### Correction-driven cache invalidation (the "stale answer bypass" fix)

A semantic cache hit can serve an answer generated from a fact that has since been corrected — **bypassing Cortex recall and generation entirely**, so no amount of preload verification catches it unless the cache itself is checked. Two mechanisms, defense in depth:

1. **Detection (always on):** the `CACHE_PREDATES_FACT_UPDATE` handbook signature (HANDBOOK.md, cache family) fires on any cache hit whose entry predates a correction event for a related subject. Relatedness is computed as cosine between the corrected fact's embedding and the cached query vector — both already exist, zero extra cost. Fires even when invalidation (below) was missed or is unavailable.
2. **Invalidation hook (`prismshine.integrations` — now push-based, all native APIs):** subscribe via `Memory.on_event(callback)` (Cortex 0.3.0 `MemoryEvent`: accommodate / conflict_opened / conflict_resolved / forget) and on correction:
   - evict PrismCache entries with `invalidate_where(vector, τ_evict)` (0.5.0/0.8.0; default 0.55 — deliberately looser than hit thresholds: over-evicting costs one LLM call, under-evicting serves wrong facts) and/or `invalidate_tags(subjects)` where entries are tagged;
   - mark ChorusGraph cache-gate entries with `mark_revalidate(sidecar, query_vector=..., threshold=...)` (1.3.0) so paraphrase hits re-verify (`HIT_REVALIDATE`) instead of reusing — ChorusGraph 1.3.0 even ships `CortexMemoryService.bind_cache_revalidate(sidecar)` wiring these two natively;
   - bump the memory partition version with `ChorusStack.bump_partition_version(partition)` (1.3.0) so `STALE_CACHE_REUSE` catches any survivor.

**Upstream limitation — RESOLVED (Jul 2026):** PrismCache 0.5.0 (prismlib) / 0.8.0 (prismlib-plus) shipped the full selective-invalidation surface with identical signatures: `invalidate_where(vector, threshold) -> int`, tagged entries (`get_or_call(..., tags=[...])`) + `invalidate_tags(tags) -> int`, `HitMeta` (created_at, tags, model, similarity) via `last_hit_meta` / `on_hit` callback, and eviction metrics (`evicted_by_vector`/`evicted_by_tags`, Prometheus counters in plus). The `invalidate_all()` fallback remains only for third-party users pinned below these versions; the `CACHE_PREDATES_FACT_UPDATE` detection signature stays on as the dual-rail backstop per §6.1 of DESIGN — detection is never removed just because prevention improved.

## 7. prismrag-patch — signal source

Retrieval steps that flow through prismrag adapters carry `rule_chain` / category info; the `CATEGORY_MISMATCH` detector compares chunk categories against the query's inferred category. No direct dependency — data arrives via the ledger/adapters.

## 8. Standalone mode (no Insight runtime)

```python
from prismshine import ShineGate
from prismshine.evidence.adapters.generic import bundle_from_dict

bundle = bundle_from_dict({
    "question": q,
    "answer": a,
    "preload": [{"chunk_id": "c1", "text": t1, "vector": v1}, ...],  # vectors optional
    "trace": [...],                                                   # optional
})
verdict = ShineGate.build(profile="default").verify(bundle)
```

Without a trace, Tier 0 runs only the trace-independent detectors (context budget, duplication); Tiers 1–4 work fully. This keeps PrismShine adoptable by non-Insight stacks (LangGraph adapter provided) while the full cause-side value lights up inside ChorusGraph.
