# PrismLabPlusAPI (prismlib-plus)

> The "Plus" stack above base prismlib: everything in PrismLib plus PrismAPI (zero-re-embed vector protocol), enterprise HTTP/auth/observability, security, and MCP.

| Field | Value |
|---|---|
| PyPI | `prismlib-plus` (import root is `prism`, same as prismlib!) |
| Version | 0.8.0 |
| License | Apache-2.0 declared (no LICENSE file in tree) |
| Python | >= 3.11 |
| Local path | `C:\code\PrismLabPlusAPI` |
| GitHub | https://github.com/insightitsGit/prismlibplusapi |
| Install | `pip install prismlib-plus` |
| CLI | `prism-wrapper` → `prism.wrapper.main:cli_main` |

## Purpose

In-process intelligence data plane for LLM apps. Superset of PrismLib 0.4.0: semantic cache (PrismCache), WAL-streamed local vector replica (PrismDriver + wrapper), cluster mesh, and **PrismAPI** — a provider/consumer protocol where the provider embeds/projects content once at index time and consumers receive float32 vectors over CHORUS-style frames (zero re-embedding of results: `top_k` results = `top_k` embedding calls eliminated per query). Adds enterprise layers: auth, TLS, rate limiting, audit, Prometheus/OTel, FastAPI app factory, MCP tool server.

## Architecture

| Package | Role |
|---|---|
| `prism.lib` | Vendored core math: `lang` (JL + blend), `resonance` (wave cache), `fabric` (TensorCipher + CHORUSFabric) |
| `prism.cache` | `PrismCache` semantic LLM cache |
| `prism.ffi` | `PrismDriver` / `DriverConfig` / `LocalIndex` (async; multi-language binding stubs) |
| `prism.wrapper` | DB-node daemon: WAL intercept → vectorize → gRPC publish |
| `prism.cluster` | `PrismNode`, `ClusterCache`, `ContextCompressor`, `AlertManager`, `HealthMonitor` |
| `prism.api` | PrismAPI provider/consumer, auth, LangGraph nodes, MCP |
| `prism.bridge` | Vector-store patch adapters (pgvector/Chroma/Qdrant) + in-repo `PrismRAGPatch` |
| `prism.security` | TLS, rate limit, audit |
| `prism.observability` | Prometheus + optional OTel |
| `prism.enterprise` | `create_enterprise_app` FastAPI helper |

Like base prismlib, `prismresonance` and `chorus-fabric` are **vendored** under `prism.lib.*`, not pip deps.

## Public API (representative)

```python
from prism.lib import (PrismProjector, ProjectionConfig, TenantSpace,
                       PrismResonance, WavePacket, PhaseState,
                       CHORUSFabric, FabricConfig, TensorCipher)

from prism.cache import PrismCache
PrismCache.build(tenant_id, *, similarity_threshold=0.92, ttl_seconds=3600, llm_model="unknown")
cache.get_or_call(query, call_fn) / cache.aget_or_call(...) / cache.get_metrics()

from prism.ffi import PrismDriver, DriverConfig
DriverConfig.from_env()
async with PrismDriver(config) as driver:
    await driver.query(embedding=..., top_k=..., threshold=...)
    await driver.write(row_id=..., data=...)

from prism.api import PrismAPIProvider, PrismAPIClient, AuthConfig, generate_api_key
provider = PrismAPIProvider(projector, embedder, semantic_fields, id_field="id")
@provider.expose
def search(...): ...
client = PrismAPIClient(projector, embedder, host=..., port=9100, api_key=...)
resp = client.query("query text", top_k=5)   # resp.vectors, resp.sidecars

from prism.cluster import ClusterCache, AlertManager, PrismNode, ContextCompressor
from prism.enterprise import create_enterprise_app
```

## Core math

Same vendored core as PrismLib (see `kb/libraries/prismlib.md`): tenant JL (`TenantSpace`), Slerp/Nlerp/anchor blend modes, wave interference score `Re⟨q,p⟩ − λ·|Im⟨q,p⟩|`, TensorCipher (QR-orthogonal K + HMAC watermark). Plus:

- **PrismCache pipeline**: embed → JL project → resonance query → hit/miss → optional `call_fn`.
- **PrismDriver**: subscribe WAL/CHORUS stream → decrypt → `LocalIndex`/resonance; local ANN instead of DB round-trip.
- **PrismAPI**: provider projects once per doc into tenant space; consumer sends one query vector, receives `{v_proj, sidecar}` per result. Same `tenant_id` → same JL seed → multi-provider results directly comparable by cosine.

## Dependencies

- Core: `numpy`, `onnxruntime`, `onnx`
- Extras: `[cache]` sentence-transformers; `[fabric]`/`[wrapper]` grpcio/protobuf + asyncpg/aiomysql; `[vector]` chromadb/qdrant; `[enterprise]` fastapi/uvicorn; `[otel]` OpenTelemetry; alert backends
- No pip dep on sibling Insight libs (vendored)

## Config / env

| Prefix / var | Area |
|---|---|
| `PRISM_WRAPPER_*` | DB DSN, flavor, gRPC/TLS, tenant, tables, dim |
| `PRISM_DRIVER_*`, `PRISM_TENANT_ID` | Driver |
| `PRISM_API_KEYS`, `PRISM_API_BEARER_TOKENS`, `PRISM_API_REQUIRE_AUTH` | Auth (`AuthConfig.from_env`) |
| `PRISM_HTTP_HOST/PORT` | Enterprise app |
| `PRISM_MCP_API_KEY` | MCP tool server |
| `CHORUS_KEY`, `SMTP_*`, `NODE_ID`, `NODE_ROLE`, `PEERS` | Cluster |

## Usage example

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="my-app", llm_model="gpt-4o")

def ask(question: str) -> str:
    return cache.get_or_call(
        query=question,
        call_fn=lambda: openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        ).choices[0].message.content,
    )
```

Also see `examples/prismapi_quickstart.py`, `examples/enterprise_server.py`.

## Tests / benchmarks

- ~204 test functions (cache, resonance, lang, fabric, ffi, wrapper, bridge, API, security, enterprise, cluster)
- Rich benchmarks: `benchmark/load`, `benchmark/cluster`, `benchmark/api`, Azure scripts. Claims: ~91–96% cache hit rate, ~439× driver read speedup, ~76% cluster token savings (`BENCHMARK_RESULTS.md`)

## Gotchas

- **Import collision**: both `prismlib` and `prismlib-plus` install as `prism` — do not install both in one environment (PrismCortex 0.3.0's `[prism]` vs `[prism-plus]` extras exist for exactly this).
- ~~Docs drift: README shows `cache.metrics()`~~ fixed in 0.8.0.
- **0.8.0 (PrismShine coupling, parity with prismlib 0.5.0):** `invalidate_where`, tagged entries + `invalidate_tags`, `HitMeta`/`last_hit_meta`/`on_hit` — identical signatures to prismlib — plus `evicted_by_vector`/`evicted_by_tags` in `CacheMetrics` and Prometheus counters (`prism_cache_evicted_by_vector_total`, `prism_cache_evicted_by_tags_total`).
- pyproject repo URL still points at `insightitsGit/prismlib` instead of `prismlibplusapi`.
- No top-level LICENSE file despite Apache metadata.
- ChorusMesh paid tiers (Developer/Team/Business/Enterprise) are a separate commercial layer on the free cluster code.
