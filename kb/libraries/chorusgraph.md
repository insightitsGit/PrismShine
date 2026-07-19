# ChorusGraph

> Native agent graph runtime (BSP scheduler — not a LangGraph wrapper) that integrates semantic cache, retrieval, Cortex memory, Route Ledger, checkpoints, and observability into one stack. The most integrated consumer of the whole Prism ecosystem.

| Field | Value |
|---|---|
| PyPI | `chorusgraph` |
| Version | 1.3.0 (Production/Stable) — PrismShine hooks: ADR-008 LLM interceptors (`register_interceptor(before_llm=, after_llm=)` + `InterceptDecision.proceed/halt/reroute`, firing in `NodeContext.call_llm`), `mark_revalidate()` (cache gate), `ChorusStack.bump_partition_version()` + `get_chunk_vectors() -> ChunkVectorRecord` (raw 384-d + partition version + encoder artifact id), stable third-party `LedgerStep.kind/detail`, `add_node(..., consumes=[...])`, cache-decision `created_at`, `CortexMemoryService.on_event`/`bind_cache_revalidate` |
| License | Apache-2.0 (open-core; Postgres/enterprise persistence license-gated) |
| Python | >= 3.11 |
| Local path | `C:\code\ChorusGraph` |
| GitHub | https://github.com/insightitsGit/ChorusGraph |

## Purpose

Solves the "glue six systems together" problem for production LLM agents. A native graph runtime with BSP (bulk-synchronous-parallel) scheduling over a Resonance bus + PrismLang envelopes, a two-stage semantic cache gate, swappable ports for cache/memory/tools/retrieval/persistence via `ChorusStack`, per-hop Route Ledger auditing, HITL interrupts, and checkpointing. LangGraph appears only as a benchmark baseline, not on the product path.

## Architecture

| Module | Role |
|---|---|
| `chorusgraph.core` | `Graph`, BSP `CompiledGraph` scheduler, channels, Resonance bus, cache interceptor, transport router |
| `chorusgraph.compose` | `ChorusStack` + ports: Cache / Memory / Tools / Retrieval / Persistence |
| `chorusgraph.cache_gate` | Two-stage semantic cache gate + sidecar + single-flight |
| `chorusgraph.ledger` | Route Ledger sinks (SQLite/Postgres) |
| `chorusgraph.memory` | `CortexMemoryService` (PrismCortex) |
| `chorusgraph.agents` | Unified `Agent` + ReAct / Plan-Solve / Reflection strategies |
| `chorusgraph.nodes` | Tool registry, retrieve helpers, roles |
| `chorusgraph.transport` | CHORUS Fabric / PrismAPI modes |
| `chorusgraph.checkpoint` / `persistence` | Checkpoints + SqliteGraphStore; Postgres license-gated |
| `chorusgraph.licensing` | Offline Ed25519 enterprise features |
| `chorusgraph.security` / `tenant` / `resilience` / `observability` | Allowlists, isolation, circuit breakers, health/metrics |
| `chorusgraph.audit` / `shadow` | Cold-query cache savings CLI, shadow harness |

## Public API

```python
from chorusgraph import (
    Graph, START, END, CompiledGraph, NodeContext, NodeFn,
    ChorusStack, RedisCacheBackend,
    gate, seed_cache_entry, Decision, DecisionKind, SidecarStore,
    CortexMemoryService, get_cortex_service,
    RouteLedger, LedgerSink, LedgerStep, SqliteLedgerSink, get_run, list_runs,
    PrismCheckpointer, create_checkpointer, sqlite_checkpointer,
    CachePolicy, Section, wrap, RunnableWithLedger,
    TransportMode, publish_hop, run_shadow_measurement,
)
from chorusgraph.compose import (
    ChorusStack, PrismCacheBackend, PrismRAGRetrievalBackend,
    KeywordRetrievalBackend, CortexMemoryBackend,
    PostgresPersistenceBackend, SqlitePersistenceBackend,
)
from chorusgraph.agents import Agent, run_react, run_plan_solve, run_reflection

# gate(query, section, cache, sidecar=None, *, coarse_threshold=0.88, verify_threshold=0.95, ...) -> Decision
# Graph(...).add_node(...).add_edge(...).compile(stack=...).invoke(state)
```

## Core algorithms

1. **BSP scheduler** (`CompiledGraph`): super-steps over Resonance bus + PrismLang envelopes; recursion limits, joins, HITL `GraphInterrupt`.
2. **Two-stage cache gate**: coarse = 64-d projected recall (`constructive_score` ≥ 0.88 default) → verify = cosine on raw 384-d (≥ 0.95 default; ≥ 0.97 for high-risk clinical). Policies: `HIT_REUSE` / `HIT_REVALIDATE` / `HIT_AS_CONTEXT` / `MISS`; exact/fingerprint direct keys.
3. **L1 single-flight** (ADR-006, opt-in): coalesce concurrent exact/fingerprint misses.
4. **ReAct anti-thrash** (ADR-007): `stop_on_repeated_action=True` by default.
5. **Warm chunk vectors** (ADR-005): `index(partition, version)` + `warm_retrieval` for query-only L2.
6. **Route Ledger**: per-hop audit (`rule_chain`, scores, durations).
7. **Send dedup** in scheduler (`COARSE_DEDUP_THRESHOLD`).

## Dependencies — the ecosystem hub

Hard pip deps on siblings: **`prismlang>=0.1.2`**, **`prismlib-plus>=0.8.0`**, **`prismresonance>=0.3.0`** (+ `cryptography`, `httpx`, `numpy`, `pydantic`). Extra `cortex` = `prismcortex[prism-plus,gemini]>=0.3.0` (prism-plus avoids the `prism` import collision).

| Sibling | Role in ChorusGraph |
|---|---|
| `prismlang` | `PrismProjector`, ONNX embedder path |
| `prismlib-plus` | PrismCache via `prism.cache` |
| `prismresonance` | Resonance bus |
| `prismcortex` (extra `cortex`) | L3 memory / GraphStore |
| `prismrag-patch` (retrieval) | Vector RAG + taxonomy remap (`PRISMRAG_LICENSE_KEY`) |
| PrismGuard | Recommended guard node (helper lives in PrismGuard repo) |

## Config / CLI

Env: `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `GEMINI_MODEL`, `CHORUSGRAPH_LICENSE_FILE`, `CHORUSGRAPH_PG_DSN`, `CHORUSGRAPH_REDIS_URL`, `CHORUSGRAPH_DETERMINISTIC`, `CHORUSGRAPH_LIVE`, `CHORUSGRAPH_CHORUS_CIPHER`, `CHORUSGRAPH_TLS_OFF`, `CHORUSGRAPH_MTLS`, `CHORUSGRAPH_ALLOW_HASH_EMBEDDER`, `CHORUSGRAPH_PENDING_WRITES_ROOT`, `PRISMRAG_LICENSE_KEY`.

CLI scripts: `chorusgraph-demo`, `chorusgraph-finance`, `chorusgraph-finance-memory`, `chorusgraph-finance-patterns`, `chorusgraph-use-cases`, `chorusgraph-shadow`, `chorusgraph-replay`, `chorusgraph-audit`, `chorusgraph-benchmark`, `chorusgraph-benchmark-volume`.

## Usage example

```python
from chorusgraph import Graph, START, END, ChorusStack
from chorusgraph.core.node import dict_node_adapter

stack = ChorusStack.defaults(tenant_id="demo")
g = Graph(tenant_id="demo", graph_id="hello")
g.add_node("echo",
           dict_node_adapter(lambda s: {"reply": f"Hello, {s.get('name', 'world')}"}, hop="echo"))
g.add_edge(START, "echo")
g.add_edge("echo", END)
out = g.compile(stack=stack).invoke({"name": "ChorusGraph"})
# {'reply': 'Hello, ChorusGraph'}
```

## Tests / benchmarks

- ~83 test files, ~393 test functions (README CI: 329+ deterministic-tier); coverage gate `fail_under=71`
- 8 benchmark scenarios: finance `fl1/fl2/fc1/fc2`, healthcare `hl1/hl2/hc1/hc2` + load + Azure results

## Gotchas

- Open-core: SQLite persistence free; Postgres/enterprise persistence gated by `CHORUSGRAPH_LICENSE_FILE`.
- Product path has NO LangGraph dependency — comparison only.
- Phase 2 open items: Postgres-native Cortex GraphStore, ledger token-cost fields, CHORUS cipher external audit, pen-test certification.
