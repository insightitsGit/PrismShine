# PrismCortex

> Compliance-grade agent memory: bitemporal knowledge graph + content-addressed render cache. Deterministic replay, time-travel recall, full audit trail. Top orchestration layer of the ecosystem.

| Field | Value |
|---|---|
| PyPI | `prismcortex` |
| Version | 0.3.0 — adds `Memory.on_event(callback) -> Unsubscribe` with `MemoryEvent` (accommodate/conflict_opened/conflict_resolved/forget; optional MeshBroadcast fan-out), superseded-fact provenance, and the `[prism-plus]` extra (mutually exclusive with `[prism]`; for prismlib-plus hosts) |
| License | MIT (open-core; commercial governance modules gated by Ed25519 license) |
| Python | >= 3.10 |
| Local path | `C:\code\PrismCortex` |
| GitHub | https://github.com/insightitsGit/PrismCortex |
| Install | `pip install prismcortex` (extras: `[prism]`, `[gemini]`, `[server]`) |

## Purpose

Agent memory that survives sessions and stands up to auditors. Digests each conversation turn into a bitemporal knowledge graph (facts have `valid_from`/`valid_to`; corrections close the old fact instead of deleting it), consolidates uncertain facts via `sleep()`, and recalls with a content-addressed render cache so the same query + same memory state replays byte-identically. Supports time-travel queries (`recall_at`), conflict inspection, and replay certificates.

## Architecture

| Module | Role |
|---|---|
| `prismcortex.engine.Memory` | Front door: `digest` / `recall` / `sleep` + enterprise APIs |
| `prismcortex.factory` | `reference_memory()` wiring |
| `prismcortex.models` | Pydantic graph/result types (`Node`, `Edge`, `Band`, ...) |
| `prismcortex.ports` | Protocols: `GistProjector`, `EntityExtractor`, `Renderer`, `GraphStore`, `ResonanceEngine`, `ResponseCache`, `MeshBroadcast`, `StagingBuffer` |
| `prismcortex.salience` | Cheap pre-LLM band classifier |
| `prismcortex.determinism` | Content-address / memo keys |
| `prismcortex.labels` | Alias / relation normalization / coref helpers |
| `prismcortex.adapters.reference` | In-memory reference adapters |
| `prismcortex.adapters.ann` | `AnnGraphStore` — IVF ANN at scale |
| `prismcortex.adapters.prism` | Production wrappers for Insight packages |
| `prismcortex.llm.gemini` | Gemini extract + render |
| `prismcortex.server` | FastAPI HTTP service (`uvicorn prismcortex.server:app`) |
| `auth` / `tenant` / `policy` / `licensing` / `tracing` | Multi-tenant RBAC, legal hold, Ed25519 license, tracing |

## Public API

```python
from prismcortex import (Memory, reference_memory, Band, DigestOutcome, DigestResult,
                         RecallResult, Node, Edge, Subgraph, StateDelta, GraphVersion)

mem = reference_memory(model=None, cache_path=None, embedding_dim=384, k=8,
                       max_facts=None, llm=None)

mem.digest(text, *, source_id=None, agent_id=None) -> DigestResult
mem.recall(query) -> RecallResult
mem.sleep() -> int                       # consolidate staged facts
mem.forget(source_id) -> dict
mem.conflicts() -> list[dict]
mem.explain(query) -> Explanation
mem.subgraph_at(query, at) -> Subgraph
mem.recall_at(query, at=None) -> RecallResult      # time travel
mem.replay_certificate(query) -> dict
mem.resolve_conflict(subject, relation, chosen_value) -> GraphVersion
```

## Core algorithms

1. **Salience gate**: keyword heuristics → `Band` (EMERGENCY/ALERT/NORMAL/NEUTRAL/ARCHIVE); skips the LLM on low-value turns; corrections/urgency fast-track.
2. **Bitemporal graph**: edges carry `valid_from`/`valid_to`; corrections use `Operation.ACCOMMODATE` (invalidate old + add new); history retained for time-travel.
3. **Delta calc**: subject coreference (alias → label → token overlap → embedding at `resolve_threshold=0.88`); values are exact-match only; conflicting values → stage or accommodate.
4. **Two-speed memory**: certain/urgent facts commit immediately; uncertain facts go to `StagingBuffer` → consolidated on `sleep()` (+ `resonance.consolidate()`).
5. **Replay determinism**: cache key = SHA-256 of `(query, canonical subgraph, template_id, model_id)`; rendered prose frozen; extraction memoized by input hash. Honest claim: replay after first render, not "temperature 0 = identical LLM".
6. **Confidence**: `1 − 0.5^weight` from reinforcement.
7. **IVF ANN** (`AnnGraphStore`): Lloyd-lite centroids + inverted lists once node count ≥ `PRISMCORTEX_ANN_THRESHOLD`.

## Dependencies

- Core: `pydantic>=2.5`, `numpy>=1.24`, `cryptography>=42`
- **`[prism]` extra — the one place the full Insight stack is pip-integrated**: `prismlang>=0.1.1`, `prismlib>=0.5.0`, `prismrag-patch>=0.2.1`, `prismresonance>=0.3.0`; **`[prism-plus]`** swaps `prismlib` for `prismlib-plus>=0.8.0` (mutually exclusive — both install the `prism` import)
- `[gemini]` → `google-genai`; `[server]` → FastAPI/uvicorn; `[bench]`/`[competitive]` → mem0/zep tooling

## Config / env

`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `PRISMCORTEX_API_KEY(S)`, `PRISMCORTEX_LICENSE_KEY`/`_PUBKEY`, `PRISMCORTEX_DATA`, `PRISMCORTEX_BACKEND` (`lite`|`prism`), `PRISMCORTEX_USE_ANN`, `PRISMCORTEX_ANN_THRESHOLD`, `PRISMCORTEX_MODEL`, `PRISMCORTEX_REGION`, `PRISMCORTEX_RATE_LIMIT_RPM`, `PRISMCORTEX_MAX_CONCURRENT_DIGEST`, `PRISMCORTEX_READ_POOL`, `PRISMCORTEX_STAGING_WARN`, `PRISMCORTEX_RETENTION_DAYS`, `PRISMCORTEX_TRACE`. No console script; server via uvicorn.

## Usage example

```python
from prismcortex import reference_memory

mem = reference_memory(cache_path=".prismcortex_cache/quickstart.json")
mem.digest("My name is Amin and my production deploy budget is $40,000.")
first = mem.recall("What is my deploy budget?")
mem.digest("Correction: my deploy budget is now $55,000.")
after = mem.recall("What is my deploy budget?")   # new fact; old one closed, not deleted
```

## Tests / benchmarks

- ~44 test functions across 7 files (graph engine, enterprise, server security, licensing, prism adapters, gemini e2e, scale bench)
- Substantial benchmark suite: scale/adversarial/messy/correctness benches, competitive runs vs mem0 and zep, Azure driver; results in `benchmarks/RESULTS.md`

## Gotchas

- Open-core: MIT core (digest/recall/graph/cache); console/governance modules gated by offline Ed25519 license.
- Roadmap gaps: adversarial 4/4, stress recall @ c=50, pen-test, messy-data validation, full LoCoMo run.
- `InProcessMesh` is a stand-in until Chorus/prismlib mesh is wired via the `MeshBroadcast` port.
