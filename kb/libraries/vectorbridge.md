# VectorBridge

> Vector DB migration/sync middleware: move embeddings between Chroma, Qdrant, Weaviate, Pinecone, pgvector, and FAISS over a binary CHORUS-style cipher, with metric preflight, watermark integrity, and semantic post-validation.

| Field | Value |
|---|---|
| PyPI | `insight-vector-bridge` (import as `vectorbridge`) |
| Version | 0.1.0 |
| License | MIT (patent note on CHORUS wire format: USPTO provisional 64/096,156) |
| Python | >= 3.10 |
| Local path | `C:\code\VectorBridge` |
| GitHub | https://github.com/insightitsGit/vectorbridge |
| Install | `pip install insight-vector-bridge` |
| CLI | `vectorbridge run|init|status` |

## Purpose

Migration tool (NOT a real-time sync layer) for moving vector databases: batched, checkpointed/resumable transfer via an encrypted binary wire format (~5.55× bandwidth savings vs REST JSON in benchmarks), distance-metric mismatch guard, HMAC/watermark integrity on every batch, and post-migration semantic neighbor-overlap validation (pass if mean top-K overlap ≥ 0.95).

## Architecture

| Module | Role |
|---|---|
| `vectorbridge.bridge.Bridge` | Public entry: config / license / programmatic run |
| `vectorbridge.orchestrator.MigrationJob` | Batch loop, checkpoint, verify |
| `vectorbridge.transport` | QR orthogonal cipher, watermark, pack/unpack, `DirectTransport` |
| `vectorbridge.verify` | Metric guard + `SemanticValidator` |
| `vectorbridge.integrity.IntegrityReport` | Signed migration artifact |
| `vectorbridge.checkpoint` | Resume state under `.vectorbridge/{job_id}.json` |
| `vectorbridge.connectors.*` | Per-DB adapters (`PgvectorConnector`, etc.) |
| `vectorbridge.cli` | Click CLI |
| `vectorbridge.agent` | Docker/K8s agent loop |
| `vectorbridge.license` | Remote license/DWV metering API client |

## Public API

```python
from vectorbridge import (Bridge, VectorRecord, ConnectorConfig, IntegrityReport,
                          SemanticValidator, SemanticVerifyReport,
                          validate_metrics, MetricMismatchError)

Bridge(
    source_type: str, source_config: dict,
    target_type: str, target_config: dict,
    mode="full", batch_size=256, job_id=None,
    license_key=None, resume=True, metric_override=False,
    semantic_verify=True, semantic_probes=100, semantic_top_k=5,
)
Bridge.from_config(path) -> Bridge
Bridge.from_license(license_key, job_id=None) -> Bridge
bridge.run(verbose=True) -> IntegrityReport

validate_metrics(source_metric, target_metric, override=False)
SemanticValidator(source, target, n_probes=100, top_k=5).run(job_id="") -> SemanticVerifyReport
# transport: generate_key_pair, encrypt/decrypt, pack_batch/unpack_batch, DirectTransport.transfer(records)
```

## Core algorithms

1. **Tensor cipher**: QR-decomposed orthogonal `K`; `V_enc = V_raw @ K`, `V_dec = V_enc @ K.T`.
2. **Rolling watermark**: SHA-256-seeded unit vector added at strength 0.01; verified via cosine threshold (audit/provenance, not crypto-hard tamper proof).
3. **Wire frame**: magic `CH0R` + count/dim/seq header + HMAC-SHA256 + encrypted float32 + id/meta lengths.
4. **Metric guard**: normalizes aliases (`cosine`/`l2`/`dot`); raises `MetricMismatchError` unless overridden.
5. **Semantic validation**: random probe vectors; compare top-K neighbor ID overlap between source and target; pass if mean ≥ 0.95.
6. **Checkpointing**: resumable full-mode migration; `incremental` if connector supports it; `live` raises "not yet supported".

## Dependencies

- Core: `numpy`, `tqdm`, `click`, `rich`
- Connector extras: `psycopg2-binary`, `pinecone-client`, `chromadb`, `weaviate-client`, `qdrant-client`, `faiss-cpu`
- **No Insight pip deps** — CHORUS cipher is reimplemented locally in `vectorbridge/transport.py`

## Config

Env: `VB_LICENSE_KEY`, `VB_JOB_ID`, `VB_POLL_SECONDS`, `VECTORBRIDGE_API` (default `https://insightits.co/api/vectorbridge/v1`).

## Usage example

```python
from vectorbridge import Bridge

bridge = Bridge(
    source_type="chromadb",
    source_config={"chroma_path": "./chroma_data", "collection_name": "docs"},
    target_type="qdrant",
    target_config={"qdrant_url": "http://localhost:6333", "collection": "docs"},
)
report = bridge.run()
print(report.summary())
```

Or: `vectorbridge run --config vb_config.json`

## Tests / benchmarks

- ~42 test functions in 4 files (migration, transport, checkpoint, verify)
- Azure cross-DC / wire-format benchmark suite under `benchmark/` claiming ~5.55× bandwidth vs REST JSON

## Gotchas

- **README is out of sync with the code**: the `source={...}` dict constructor form and the `vectorbridge migrate` CLI command in README do NOT exist. Trust `Bridge(source_type=..., ...)` and `run`/`init`/`status`.
- `live` mode is not implemented — this is a migration tool, not real-time replication.
- MIT code, but commercial DWV metering via remote API in license/agent mode (offline grace if server unreachable).
