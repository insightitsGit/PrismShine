# PrismResonance

> Wave-memory re-ranking sidecar: static embeddings become complex wavepackets with mutable phase (operational context), retrieved via interference scoring. Foundation-layer library.

| Field | Value |
|---|---|
| PyPI | `prismresonance` |
| Version | 0.3.0 |
| License | MIT (core; Enterprise tier is commercial) |
| Python | >= 3.10 |
| Local path | `C:\code\PrismResonate` |
| GitHub | https://github.com/insightitsGit/prismresonance |
| Install | `pip install prismresonance` |

## Purpose

Dynamic memory sidecar for RAG / multi-agent systems. Source embeddings stay immutable (amplitude A); a mutable phase φ encodes operational context (EMERGENCY, ALERT, NORMAL, RECOVERY, ARCHIVE...). Retrieval uses wave interference — context-matching vectors constructively interfere (score rises), mismatched ones destructively interfere (score falls) — naturally demoting stale/wrong-context results without touching the source vector DB. Non-destructive: source DB is read-only; all mutable state lives in the sidecar registry.

## Architecture

| Module | Role |
|---|---|
| `prism.py` | Facade `PrismResonance` |
| `frequency.py` | `FrequencyFamily` phase bands |
| `compiler.py` | ONNX graph compile; `wavepack_encode(_batch)` |
| `engine.py` | Two-level ONNX retrieval, coherence shield, Hebbian nudge |
| `memory_map.py` | `ResonanceRegistry` RAM sidecar + SQLite persistence |
| `sleep.py` | Consolidation: decay, synaptic align, groups, emergent bands, flush |
| `broadcast.py` | Multi-agent frequency bus (in-process / Redis) |
| `stream.py` | Buffered `StreamIngester` |
| `adapters/` | `SourceAdapter`, `PrismLangAdapter` (pgvector/Chroma documented in README) |

## Public API

```python
from prismresonance import (
    PrismResonance, ResonanceRegistry, ChunkMeta, GroupMeta,
    ResonanceEngine, ResonanceResult,
    SleepManager, SleepConfig, EmergentBand,
    compile_resonance_graph, wavepack_encode, wavepack_encode_batch,
    FrequencyFamily, NEUTRAL, NORMAL, ALERT, EMERGENCY, RECOVERY, ARCHIVE,
    FrequencyBroadcast, InProcessBroadcast, RedisBroadcast,
    StreamIngester,
)

prism = PrismResonance.create(
    embedding_dim=384, onnx_path=..., state_path="resonance_state.db",
    coherence_threshold=0.5, top_k_groups=3, hebbian_rate=0.02,
    auto_sleep=False, sleep_interval=300.0, recompile=False, load_state=True,
)

prism.ingest(chunk_id, amplitude, frequency=NEUTRAL, parent_id=None, group_id=None)
prism.ingest_batch(chunk_ids, amplitudes, frequency=...)
prism.query(query_amplitude, query_phase, candidate_ids, candidate_amplitudes,
            flat=False)  # -> list[ResonanceResult(score, chunk_id, group_id, metadata)]
prism.delete_chunk / update_chunk / set_frequency / record_co_activation
prism.most_recent(n=10); prism.most_related(chunk_id, n=10)
prism.sleep(); prism.start_background_sleep(); prism.save(); prism.shutdown(); prism.status()
```

## Core math

1. **Wavepacket**: `z = A·e^(iφ)`; wire format is 2N float32: `[A cos φ | A sin φ]`.
2. **ONNX interference score**: `Signal = Σ (Real_Q·Real_DB + Imag_Q·Imag_DB)` ≡ amplitude product × `cos(φ_Q − φ_V)` coupling. Implemented as Slice + MatMul + Add ONNX graph.
3. **Frequency families** (π/6 spacing): NEUTRAL 0, NORMAL π/6, ALERT π/3, EMERGENCY π/2, RECOVERY 2π/3, ARCHIVE π.
4. **Two-level retrieval**: group centroid scan → top_k groups → chunk scan; `flat=True` bypasses grouping.
5. **Coherence shield**: hard drop of results below `coherence_threshold` (default 0.5).
6. **Hebbian nudge**: high-scoring chunks drift toward the query phase: `φ ← (1−η)φ + η·φ_query` (η = `hebbian_rate`, default 0.02).
7. **Sleep cycle**: temporal decay toward neutral phase (`decay_lambda = 1/3600` per idle hour), synaptic phase alignment on co-activation, group recompute/merge, emergent histogram bands, SQLite checkpoint.

## Dependencies

- Core: `numpy`, `onnxruntime`, `onnx` (no Torch)
- Optional: `prismlang>=0.1.1` extra (`[prismlang]`); Redis for `RedisBroadcast`
- Sibling relationships: optional PrismLang adapter. PrismLib embeds a *related but different* in-process resonance engine (`prism.lib.resonance`) — same ideas, different API. If you use PrismLib you don't need this package separately.

## Config / CLI

No console scripts, no env-var layer. Everything is tuned via `PrismResonance.create()` kwargs and `SleepConfig`. Persistence via `state_path` (SQLite) and `onnx_path`.

## Usage example

```python
import numpy as np
from prismresonance import PrismResonance
from prismresonance.frequency import FrequencyFamily as FF

prism = PrismResonance.create(embedding_dim=384,
                              onnx_path="resonance_engine.onnx",
                              state_path="resonance_state.db")
amp = np.random.randn(384).astype(np.float32)
amp /= np.linalg.norm(amp)
prism.ingest("chunk_001", amp, FF.EMERGENCY)
results = prism.query(query_amplitude=amp, query_phase=FF.EMERGENCY,
                      candidate_ids=["chunk_001"],
                      candidate_amplitudes=np.stack([amp]), flat=True)
prism.sleep(); prism.save(); prism.shutdown()
```

## Tests / benchmarks

- ~40 test functions in `tests/test_resonance_pipeline.py` + `tests/test_integration_local.py` (ingestion, phase gating, sleep, Hebbian, emergent bands, stream, broadcast, concurrency, persistence)
- One benchmark script: `benchmarks/benchmark_query.py`

## Gotchas

- Does NOT store documents or replace a vector DB — it's a re-ranking/memory sidecar only.
- MIT core vs commercial Enterprise tier (Redis hot layer, multi-tenant, dashboard, ~$349/mo) per DESIGN.md.
- Open roadmap items in DESIGN.md: group split, CHORUS phase sync, Redis hot layer, fuller benchmarks.
- Distinct from PrismLib's `prism.lib.resonance.PrismResonance` (which adds a destructive-interference penalty λ and has a different API).
