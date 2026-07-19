# CHORUS Fabric

> Binary gRPC wire protocol for float32 tensors — encryption, watermarking, and bandwidth compression for agent-to-agent vector transport. Foundation-layer library.

| Field | Value |
|---|---|
| PyPI | `chorus-fabric` (import as `chorus_fabric`) |
| Version | 0.1.0 |
| License | MIT (patent-pending core: US Provisional 64/096,156) |
| Python | >= 3.10 |
| Local path | `C:\code\CHORUS` (package at `chorus_fabric/`) |
| GitHub | https://github.com/insightitsGit/chorus-fabric |
| Install | `pip install chorus-fabric` |

## Purpose

Replaces HTTP/JSON embedding round-trips between AI agents with bidirectional gRPC streams of raw float32 tensors, protected by a matrix-multiply cipher (zero-overhead — it's just a matmul) and a rolling neural watermark. Agents exchange hidden states/embeddings with ~4.45× less bandwidth and no tokenization. Optimized for float32 tensors specifically — not for text/JSON/records.

## Architecture

| Component | Role |
|---|---|
| `crypto_engine.py` | Keys, encrypt/decrypt, orthogonal projections, watermark, wire pack |
| `client.py` | `ChorusClient` — handshake + send modes |
| `control_plane.py` | Session keys / TTL / projection bundles (port 50051) |
| `relay_node.py` | Amplify + forward ciphertext without decrypting; SHA-256 audit (port 50052) |
| `server.py` | Target pod: decrypt, verify watermark (port 50053) |
| `servers.py` | Launcher: `python -m chorus_fabric.servers {control_plane|relay|target}` |
| `fabric_pb2*.py` | Generated gRPC stubs |

Repo root also contains parallel standalone demo/benchmark scripts and Docker Compose topology — noise for consumers; prefer the packaged `chorus_fabric/` API.

## Public API

```python
from chorus_fabric import (
    ChorusClient,
    generate_key_pair, encrypt, decrypt,
    inject_watermark, verify_watermark,
    generate_orthogonal_projections, superpose_signals,
    DEFAULT_DIM,   # 128
)

client = ChorusClient(pod_id="pod-A", control_plane="localhost:50051",
                      relay="localhost:50052", target=None, dim=DEFAULT_DIM)
client.handshake(isolation_mode=False)          # -> session_id
client.send_direct(tensor, seq_start=0)          # -> list[dict] acks
client.send_isolation(tensor_a, tensor_b)        # needs isolation_mode=True
client.send_superposition(tensor_a, tensor_b)
client.send_stream(payloads)

generate_key_pair(dim=128)                       # -> (K, K_inv)
encrypt(v_raw, K); decrypt(v_enc, K_inv)
inject_watermark(v, seed, seq_num); verify_watermark(v, seed, seq_num, threshold=0.95)
generate_orthogonal_projections(dim)             # -> (W_A, W_B)
superpose_signals(v_A, v_B)
```

## Core math

1. **Cipher**: QR decomposition → sign-corrected diagonal → scale ∈ [0.5, 2.5]; `V_enc = V_raw @ K`, `V_raw = V_enc @ K_inv`. Float64 internal, float32 on wire. Since K is orthogonal-based, decryption is essentially `@ Kᵀ` — same cost as a neural net layer.
2. **Mode A (isolation)**: orthonormal basis split into halves; projectors `W_A`, `W_B` with `W_A @ W_B ≈ 0`; tunnel carries `W_A·V_A + W_B·V_B`; each party recovers its own half.
3. **Mode B (superposition)**: `V_collective = V_A + V_B`.
4. **Watermark**: 10% of dims (`WATERMARK_RATIO=0.10`); `SHA-256(seed ‖ seq)` seeds an RNG → unit vector overwrites first slice; verification = cosine ≥ 0.95.
5. **Relay**: amplifies ciphertext (`V_amp = factor × V_enc`) without ever holding `K_inv`.

## Dependencies

- `grpcio`, `grpcio-tools`, `protobuf`, `numpy>=1.24,<2`, `torch>=2.0`
- No dependencies on sibling Insight libs. (PrismLib reimplements a compatible cipher/frame layer rather than importing this package.)

## Config

No console scripts; run servers via `python -m chorus_fabric.servers control_plane|relay|target`.

| Env | Default | Meaning |
|---|---|---|
| `CHORUS_DIM` | 128 | Embedding dim |
| `CONTROL_PLANE_HOST/PORT` | control-plane / 50051 | Control plane address |
| `RELAY_PORT` | 50052 | Relay |
| `CHORUS_TARGET_HOST/PORT` | target-pod / 50053 | Target |
| `CHORUS_SESSION_TTL` | 3600 | Session key TTL (s) |
| `CHORUS_AMPLIFY_FACTOR` | 1.0 | Relay gain |
| `CHORUS_USE_RELAY` | true | Root client path |

## Usage example

```python
import torch
from chorus_fabric import ChorusClient

client = ChorusClient(pod_id="my-agent",
                      control_plane="localhost:50051",
                      relay="localhost:50052")
client.handshake()
signal = torch.randn(128)
acks = client.send_direct(signal)
# [{'seq': 0, 'forwarded': True, 'status': 'ok'}]
```

## Tests / benchmarks

- Root `test_suite.py`: 5 classes (`TestOrthogonalIsolation`, `TestHolographicSuperposition`, `TestWatermarkIntegrity`, `TestKeyManagement`, `TestRelayAmplification`), ~20 test methods, in-process (no network)
- `benchmark_client.py` + `results/CHORUS_BENCHMARK_REPORT.txt`; transatlantic benchmark: 179 ms p50, 7,766 watermarks verified; Docker Compose stack for full topology

## Gotchas

- Root README quickstart shows `await client.handshake()` but the packaged `ChorusClient` is sync — prefer the packaged API.
- pyproject GitHub URL still points at `aminparva84/chorus-fabric` (mirror of `insightitsGit/chorus-fabric`).
- When NOT to use: text/JSON/non-vector data.
