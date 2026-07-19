# PrismLang

> Deterministic 64-d vector language protocol for LangGraph multi-agent graphs — replaces growing text state with compact, tenant-isolated, auditable envelopes.

| Field | Value |
|---|---|
| PyPI | `prismlang` |
| Version | 0.1.2 (Alpha) — adds `encoder.model_id()` + `encoder.get_session()` (public exports; PrismShine shared-session + encoder-drift detection) |
| License | Apache-2.0 |
| Python | >= 3.10 |
| Local path | `C:\code\PrismLang` |
| GitHub | https://github.com/insightitsGit/prismlang |
| Install | `pip install prismlang` |

## Purpose

In LangGraph pipelines, every hop normally re-pays token tax on growing text history. PrismLang projects each agent's output into a compact 64-d envelope (with category slug + auditable `rule_chain`), so vectors flow through graph state instead of text. Tenant isolation is geometric: JL projections seeded by `SHA-256(tenant_id)` — different tenants live in mathematically different spaces (not an ACL). Benchmarks claim −57–62% prompt tokens vs text-state LangGraph.

## Architecture

| Module | Role |
|---|---|
| `prismlang.encoder` | ONNX `all-MiniLM-L6-v2` → 384-d unit vectors |
| `prismlang.taxonomy` | `Category`, `TaxonomyConfig`, keyword inference + direction vectors |
| `prismlang.projector` | `PrismProjector` — spherical blend + JL |
| `prismlang.middleware` | `@prism_node`, `@async_prism_node` decorators |
| `prismlang.envelope` / `state` | `PrismEnvelope`, `PrismState` (append-only `prism_sequence`) |
| `prismlang.translator` | `BoundaryTranslator` — exit-node structural/LLM report |
| `prismlang.checkpointer` | `JsonFileCheckpointer`, `PostgresCheckpointer` (+ async variants) |
| `prismlang.config` | `EMBED_DIM=384`, `DEFAULT_K=64`, `DEFAULT_ALPHA=0.3` |

## Public API

```python
from prismlang import (
    Category, TaxonomyConfig, PrismProjector,
    PrismState, PrismEnvelope,
    prism_node, async_prism_node,
    BoundaryTranslator,
    JsonFileCheckpointer, PostgresCheckpointer,
    AsyncJsonFileCheckpointer, AsyncPostgresCheckpointer,
    DEFAULT_ALPHA, DEFAULT_K, EMBED_DIM,
)

Category(slug, label, keywords: list[str])
TaxonomyConfig(categories, alpha=0.3)
    .infer_category(text) -> str
    .direction_vector(slug) -> np.ndarray

PrismProjector(taxonomy, tenant_id, k=64, alpha=None)
    .project(text) -> (slug, vector, rule_chain)
    .project_batch(texts)
    .matrix_fingerprint() -> str

prism_node(agent_id, projector)          # decorator
BoundaryTranslator(llm_fn=None).translate(state) -> str; .as_langgraph_node()
JsonFileCheckpointer(root=".prismlang_checkpoints")
PostgresCheckpointer(dsn=None)           # or DATABASE_URL
```

Encoder helpers (module-level): `encode`, `encode_batch`, `async_encode`, `async_encode_batch`.

## Core math

1. **Embed**: mean-pool ONNX MiniLM → L2-normalize → 384-d.
2. **Category**: keyword hit counts; no match → first category (deterministic fallback).
3. **Direction `eᵢ`**: mean of keyword embeddings, normalized (cached).
4. **Spherical blend**: `v' = normalize((1−α)·v + α·‖v‖·eᵢ)` (α default 0.3).
5. **JL reduction**: Gaussian `P ∈ ℝ^{k×384}` seeded by `int(sha256(tenant_id),16) % 2^32`, row-normalized; `p = normalize(P @ v')`, default k=64.
6. **LangGraph wiring**: `@prism_node` projects `raw_output` into a `PrismEnvelope` appended via `operator.add` on `prism_sequence`.

## Dependencies

- Required: `langgraph`, `onnxruntime`, `numpy`, `huggingface-hub`, `tokenizers`, `typing_extensions`
- Optional: `psycopg2-binary`, `asyncpg`, `aiofiles`
- No sibling Insight install deps. ChorusGraph depends on prismlang (>= 0.1.1); prismresonance has an optional `[prismlang]` extra.

## Config

| Var | Purpose |
|---|---|
| `DATABASE_URL` | Postgres checkpointer DSN |
| `PRISMLANG_USE_REAL_ENCODER` | `"1"` = real ONNX encoder in tests |
| `GEMINI_API_KEY` / `PRISMRAG_GEMINI_KEY` | Benchmarks only |

Model cache: `~/.prismlang/models` (`sentence-transformers/all-MiniLM-L6-v2`, downloaded on first use).

## Usage example

```python
from prismlang import Category, TaxonomyConfig, PrismProjector, PrismState, prism_node

taxonomy = TaxonomyConfig(categories=[
    Category("risk", "Market Risk", ["risk", "exposure", "volatility"]),
    Category("market", "Market Data", ["price", "equity", "bond"]),
])
projector = PrismProjector(taxonomy, tenant_id="acme-finance-prod", k=64)

@prism_node(agent_id="analyst", projector=projector)
def analyst(state: PrismState) -> dict:
    return {"raw_output": "Credit risk exposure elevated in EM bonds."}
# wire into StateGraph(PrismState) + BoundaryTranslator + checkpointer
```

## Tests / benchmarks

- 34 tests (encoder 7, projector 12, isolation 8, state 7); coverage gate `fail_under=80`
- `benchmarks/` — healthcare / finance / trade_market domains, `python -m benchmarks.run_all`

## Gotchas

- Isolation is geometric (JL), not access control — see `docs/SECURITY.md`.
- Alpha status; encoder downloads on first use.
- Shares math (spherical blend + tenant-seeded JL, 64-d) with PrismLib's `prism.lib.lang`, but implemented independently.
