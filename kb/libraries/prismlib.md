# PrismLib

> Semantic LLM cache + WAL-streaming vector driver + cluster intelligence — the main app-facing middle-layer product.

| Field | Value |
|---|---|
| PyPI | `prismlib` (import as `prism`) |
| Version | 0.5.0 (pyproject + `__init__` in sync since 0.5.0) |
| License | Apache-2.0 |
| Python | >= 3.11 |
| Local path | `C:\code\PrismLib` |
| GitHub | https://github.com/insightitsGit/prismlib |
| Install | `pip install "prismlib[cache]"` or `pip install "prismlib[fabric]"` |

## Purpose

In-process intelligence stack targeting three problems without mandatory external infra (no Redis, no Pinecone, no Prometheus):

1. **LLM cost** — PrismCache: semantic cache with tenant-isolated JL projection (91–96% hit rates on paraphrased queries).
2. **DB read latency** — PrismDriver + Server Wrapper: WAL/CDC changes are vectorized on the DB node and streamed to a warm local index on app nodes (~2ms in-process dot product vs ~142ms network round-trip).
3. **Multi-node cache sharing** — PrismLib Micro: Blue/Green/Orange cluster with `ClusterCache` (TOKEN_SYNC broadcast), alerting, ~3s failover.

## Architecture

| Module | Role |
|---|---|
| `prism.lib.lang` | Tenant JL projection (`TenantSpace`, `PrismProjector`), spherical blend, `PayloadEnvelope` |
| `prism.lib.resonance` | In-process wave cache (`PrismResonance`, `WavePacket`, ONNX MatMul interference) |
| `prism.lib.fabric` | CHORUS-style transport (`CHORUSFabric`, `TensorCipher`, typed `CHORUSFrame`s) |
| `prism.cache` | End-user semantic LLM cache over lang + resonance |
| `prism.wrapper` | DB-node daemon: WAL/CDC → vectorize → publish frames |
| `prism.ffi` | App-node `PrismDriver` (C++ DLL or Python fallback + local index) |
| `prism.cluster` | Micro cluster: `PrismNode`, `ClusterCache`, `AlertManager`, `ContextCompressor` |
| `prism.bridge` | Vector-store adapters + `PrismRAGPatch` taxonomy patch |
| `prism.api` | Vector-native API provider/consumer (CHORUS frames, LangGraph helpers) |

**Important:** PrismLib *vendors* its own fabric/resonance implementations under `prism.lib.*`. It does NOT pip-depend on `chorus-fabric` or `prismresonance` — same brand/ideas, different code and APIs.

## Public API

```python
# Cache
from prism.cache import PrismCache, PrismCacheConfig
PrismCache.build(tenant_id, *, similarity_threshold=0.92, ttl_seconds=3600,
                 llm_model="unknown", embedder=None, store=None, persist_path=None)
cache.get_or_call(query, call_fn, *, metadata=None, tokens_in_response=None)
cache.aget_or_call(...)   # async
cache.invalidate_all(); cache.purge_expired(); cache.get_metrics()

# Lang / resonance / fabric
TenantSpace(tenant_id, input_dim, target_dim=64).project(v)
PrismProjector(ProjectionConfig).project(embedding, *, anchor_label=None, blend_weight=None)
PrismResonance(dim=64, lambda_destructive=0.3, ...).query(query_packet, *, top_k=10)
TensorCipher(dim, ttl_seconds=300).encrypt(vectors, sequence_number=0)  # -> (V_enc, watermark)
CHORUSFabric(FabricConfig).send / emit_health / emit_signal / receive / serve

# Driver / wrapper
PrismDriver(DriverConfig).connect(); .query(table, query_vector, top_k=...); .write(...)
WrapperDaemon(WrapperConfig).run()   # CLI: prism-wrapper

# Cluster
PrismNode(node_id, role: NodeRole, peers, ...)
ClusterCache(node_id, fabric, ...).get_or_call(query, query_vector, call_fn,
                                               context_chunks=..., chunk_vectors=...)
ContextCompressor(top_k=5, dim=64).compress(query_vector, context_chunks, chunk_vectors)
AlertManager(fabric, mail_config=SMTPConfig(...), ...)

# API
PrismAPIProvider(projector, embedder, semantic_fields=..., id_field=...).expose(fn)
PrismAPIClient(projector, embedder, ...).query(text, top_k=...) / .query_vector(...)
```

## Core math

1. **JL tenant isolation** (`TenantSpace`): `seed = SHA-256(tenant_id)[:4]`; projection matrix `P ~ N(0,1)/√k` with `k=64`; `z = v @ P`. Different tenants get independent Gaussian bases — wrong-basis reads look like isotropic noise. Isolation is mathematical, not a filter.
2. **Spherical blend** (`PrismProjector`): unit-normalize → optional SLERP/linear/anchor-only blend toward category anchors → JL → L2-normalize → `PayloadEnvelope` with `rule_chain`.
3. **Wave packets**: `z = A·e^(iφ)·v_unit` stored as real/imag float32; `PhaseState`: ACTIVE=0, EMERGENCY=π/6, ALERT=π/2, ARCHIVE=−π/2.
4. **Interference score**: `constructive = Re⟨q,p⟩`, `destructive = |Im⟨q,p⟩|`, `score = constructive − λ·destructive` (λ default 0.3). Runs as ONNX MatMul with NumPy fallback.
5. **TensorCipher**: QR-decomposed orthogonal `K`; `V_enc = V @ K`, decrypt `V @ Kᵀ`; watermark `HMAC-SHA256(stream_secret, key_id ‖ seq ‖ vector_bytes)`.
6. **ContextCompressor**: cosine top-K + hot-chunk boost from METRIC frames; token estimate ~1.3 × words.
7. **Failover**: GREEN active / BLUE warm / ORANGE syncing; heartbeat watch promotes BLUE on silence (~3s).

## Dependencies

- Core: `numpy`, `onnxruntime`, `onnx`
- Extras: `cache` → sentence-transformers; `cache-openai` / `cache-anthropic` / `cache-ollama`; `fabric`/`wrapper` → grpcio, protobuf; `vector` → asyncpg, chromadb, qdrant-client; alert backends (sendgrid/boto3/resend/psutil)
- No pip dependency on sibling Insight libs (fabric/resonance are vendored)

## Config / CLI

- CLI: `prism-wrapper` → `prism.wrapper.daemon:cli_main` (`--config`, `--dsn`, `--flavor`, `--port`)
- Wrapper env: `PRISM_WRAPPER_DB_DSN`, `_DB_FLAVOR`, `_DB_SLOT`, `_DB_TABLES`, `_GRPC_HOST/PORT`, `_TLS_CERT/KEY`, `_TARGET_DIM`, `_LOG_LEVEL`
- Driver env: `PRISM_WRAPPER_HOST/PORT/URL`, `PRISM_TENANT_ID`, `PRISM_TARGET_DIM`, `PRISM_DRIVER_PATH`, `PRISM_THRESHOLD`, `PRISM_TTL`
- Cluster: `NODE_ID`, `NODE_ROLE`, `PEERS`, `TOKEN_BUDGET`, `SMTP_*`
- Embedders: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## Usage example

```python
from prism.cache import PrismCache
from openai import OpenAI

cache = PrismCache.build(tenant_id="my-company", llm_model="gpt-4o",
                         similarity_threshold=0.92, ttl_seconds=3600)
client = OpenAI()

def ask(question: str) -> str:
    return cache.get_or_call(
        query=question,
        call_fn=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        ).choices[0].message.content,
        tokens_in_response=300,
    )
```

## Tests / benchmarks

- ~158 test methods across 12 files in `tests/` (cache, fabric, resonance, lang, wrapper, ffi, bridge, api)
- Large benchmark suite in `benchmark/`: Locust load tests (cache + driver), Azure cluster runs, PrismAPI, wrapper_sim; results in `BENCHMARK_RESULTS.md`

## Gotchas

- README snippets like `from prismresonance import PrismProjector, WaveIndex` / `from chorus_fabric import CHORUSPublisher` do NOT match those packages' real APIs — those symbols live in PrismLib's own modules.
- ~~Version string mismatch~~ fixed in 0.5.0 (both say 0.5.0).
- **0.5.0 (PrismShine coupling):** `PrismCache.invalidate_where(vector, threshold) -> int`, tagged entries (`get_or_call(..., tags=[...])`) + `invalidate_tags(tags) -> int` (persist across SQLite round-trip), `HitMeta` (created_at, tags, model, similarity) via thread-local `last_hit_meta` property and optional `on_hit` callback on `build()`.
- ChorusMesh and prismlib-plus are the paid extensions layered above this OSS core.
