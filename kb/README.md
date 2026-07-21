# Insight Ecosystem Knowledge Base

Technical KB of the 12 Insight IT Solutions libraries, built from a deep read of the local source trees on `C:\code` (Jul 2026). Use this when designing anything in PrismShine that touches the ecosystem. One file per library in `kb/libraries/`.

Visual map: `kb/insight-ecosystem-map.png` (regenerate with `python kb/diagrams/make_ecosystem_map.py`). Includes the identified stack gaps (candidate PrismShine scope).

## The 12 libraries at a glance

| Library | PyPI / package | Ver | License | Local path | One-liner |
|---|---|---|---|---|---|
| [PrismLib](libraries/prismlib.md) | `prismlib` (imports as `prism`) | 0.5.0 | Apache-2.0 | `C:\code\PrismLib` | Semantic LLM cache + WAL vector driver + cluster mesh |
| [CHORUS Fabric](libraries/chorus-fabric.md) | `chorus-fabric` | 0.1.0 | MIT (patent-pending) | `C:\code\CHORUS` | Encrypted binary gRPC transport for float32 tensors |
| [PrismResonance](libraries/prismresonance.md) | `prismresonance` | 0.3.0 | MIT (open-core) | `C:\code\PrismResonate` | Wave-memory re-ranking sidecar (amplitude + phase) |
| [PrismRAG](libraries/prismrag-patch.md) | `prismrag-patch` | 1.0.0 local / 0.2.1 PyPI | Commercial | `C:\code\PrismRagLib` | Deterministic taxonomy remap before vector DB upsert/query |
| [PrismLang](libraries/prismlang.md) | `prismlang` | 0.1.2 | Apache-2.0 | `C:\code\PrismLang` | 64-d vector envelopes for LangGraph state (tenant-seeded JL) |
| [PrismLabPlusAPI](libraries/prismlib-plus.md) | `prismlib-plus` (imports as `prism`) | 0.8.0 | Apache-2.0 | `C:\code\PrismLabPlusAPI` | PrismLib superset + PrismAPI zero-re-embed protocol + enterprise |
| [PrismCortex](libraries/prismcortex.md) | `prismcortex` | 0.3.0 | MIT (open-core) | `C:\code\PrismCortex` | Bitemporal agent memory graph + deterministic replay |
| [ChorusMesh](libraries/chorusmesh.md) | `chorusmesh` | 0.1.0 | Commercial | `C:\code\ChorusMesh` | Paid alerts (Slack/PagerDuty) + Kafka/NATS transport over prismlib |
| [VectorBridge](libraries/vectorbridge.md) | `insight-vector-bridge` (imports as `vectorbridge`) | 0.1.0 | MIT | `C:\code\VectorBridge` | Vector DB migration with cipher transport + semantic validation |
| [PrismGuard](libraries/prismguard.md) | `prismguard` | 0.1.9 | Apache-2.0 (open-core) | `C:\code\PrismGaurd` | Prompt-injection firewall with auditable resolution gates |
| [ChorusGraph](libraries/chorusgraph.md) | `chorusgraph` | 1.3.0 | Apache-2.0 (open-core) | `C:\code\ChorusGraph` | Native BSP agent graph runtime — the ecosystem hub |
| [InsightPlugIn](libraries/insightplugin.md) | `insight-plugin` (VSIX) | 0.1.0 | MIT | `C:\code\InsightPlugIn` | Cursor/VS Code SMS remote control extension (TypeScript) |

All on GitHub under `insightitsGit/` (public). A marketing-level overview also exists at `C:\code\INSIGHT_ECOSYSTEM.md`.

## Real dependency graph (as implemented, not as marketed)

Verified from `pyproject.toml` files — this differs from the conceptual diagram in `INSIGHT_ECOSYSTEM.md`:

```
ChorusGraph 1.3.0  ──hard──►  prismlang>=0.1.2, prismlib-plus>=0.8.0, prismresonance>=0.3.0
                   ──extra──► prismcortex [cortex], prismrag-patch (retrieval)
                   ──soft───► PrismGuard (guard node helper lives in PrismGuard repo)

PrismCortex [prism] extra ──► prismlang, prismlib, prismrag-patch, prismresonance
PrismGuard  [prism] extra ──► prismlib, prismrag-patch(<1.0), prismcortex
ChorusMesh  ──hard────────►  prismlib>=0.4.0   (the only always-on sibling dep)
prismresonance [prismlang] extra ──► prismlang

Standalone (zero Insight pip deps):
  chorus-fabric, prismlib, prismlib-plus, prismrag-patch, prismlang,
  vectorbridge, insight-plugin
```

**Key architectural fact — vendoring, not importing:** `prismlib` and `prismlib-plus` each contain their *own* copies of the fabric/resonance/lang math under `prism.lib.*`. They do NOT pip-depend on `chorus-fabric` or `prismresonance`. VectorBridge likewise reimplements the CHORUS cipher locally. Same brand and math, different code and public APIs. Never assume `from chorus_fabric import ...` works against PrismLib's fabric layer or vice versa.

## Shared math primitives (appear across multiple libs)

- **Tenant-seeded JL projection**: Gaussian matrix seeded by `SHA-256(tenant_id)`, project to 64-d, L2-normalize. Isolation is geometric, not access control. (prismlang, prism.lib.lang in both prismlibs, PrismAPI)
- **Wave interference scoring**: embeddings as `z = A·e^(iφ)`; score from real/imag dot products; phase encodes operational context (EMERGENCY/ALERT/NORMAL/ARCHIVE...). (prismresonance, prism.lib.resonance, ChorusGraph bus, PrismCortex bands)
- **TensorCipher**: QR-decomposed orthogonal `K`; encrypt `V @ K`, decrypt `V @ Kᵀ`; HMAC-SHA256 or cosine watermark. Patent: USPTO provisional 64/096,156. (chorus-fabric, prism.lib.fabric, vectorbridge.transport)
- **Keyword→category blend**: assign category by keyword rules, blend vector toward category direction `(1−α)·v + α·dir`, normalize. (prismrag-patch α=0.25, prismlang α=0.3, prism.bridge)
- **Offline Ed25519/RS256 license gating** for commercial features. (prismcortex, prismguard, chorusgraph, chorusmesh; prismrag-patch uses online call-home instead)

## Choosing a library

| Problem | Use |
|---|---|
| LLM API cost / repeated paraphrased queries | PrismLib `PrismCache` |
| DB read latency / local semantic search | PrismLib `PrismDriver` + `prism-wrapper` |
| Wrong-category RAG results (have taxonomy) | prismrag-patch |
| Context-sensitive re-ranking | prismresonance |
| LangGraph token tax between agents | prismlang |
| Serving/consuming vectors across services (zero re-embed) | prismlib-plus PrismAPI |
| Agent memory across sessions, auditable | prismcortex |
| Prompt-injection filtering | prismguard |
| Full agent graph runtime (cache+memory+retrieval+audit in one) | chorusgraph |
| Vector DB migration (one-shot, not sync) | vectorbridge |
| Durable queues + Slack/PagerDuty for a CHORUS cluster | chorusmesh (paid) |
| Agent-to-agent tensor transport | chorus-fabric |

## Cross-cutting gotchas (verified against source)

1. **Import collision**: `prismlib` and `prismlib-plus` BOTH install the `prism` package. Never install both in one environment; prefer `prismlib-plus` (superset, 0.8.0). PrismCortex 0.3.0's `[prism]` / `[prism-plus]` extras exist to pick the right one per host.
2. **README drift is common** — trust the source over READMEs:
   - VectorBridge README's `Bridge(source={...})` form and `vectorbridge migrate` CLI don't exist (use `Bridge(source_type=..., source_config=...)` and `run|init|status`).
   - PrismLib README's `from prismresonance import PrismProjector` / `from chorus_fabric import CHORUSPublisher` snippets don't match those packages' real APIs.
   - prismlib-plus README `cache.metrics()` → code is `get_metrics()`.
   - chorus-fabric root README shows async `handshake()`; packaged client is sync.
3. **Version-string drift**: prismguard 0.1.7 vs 0.1.6 (prismlib's drift was fixed in 0.5.0).
4. **Marketing ahead of code**: ChorusMesh advertises Raft/multi-region but ships only alerts + Kafka/NATS + licensing (and has zero tests). VectorBridge `live` mode not implemented.
5. **Commercial boundaries**: prismrag-patch (online license, keys `prlib_*`), chorusmesh (JWT license), and the enterprise gates in prismcortex/prismguard/chorusgraph (offline Ed25519 via env/license files).
6. **PrismGuard pins `prismrag-patch<1.0.0`** — the published OSS line, not the local 1.0.0 commercial cut in `C:\code\PrismRagLib`.
7. **Folder-name traps**: `C:\code\PrismGaurd` (misspelled) = PrismGuard; `C:\code\CHORUS` = chorus-fabric; `C:\code\PrismResonate` = prismresonance; `C:\code\PrismRagLib` = prismrag-patch (no git remote configured); `C:\code\PrismLabPlusAPI` = prismlib-plus.

## Layering (for system design)

```
Top:        PrismCortex (memory) · ChorusGraph (runtime) — orchestration
Middle:     PrismLib / prismlib-plus · prismrag-patch · prismlang · PrismGuard
Foundation: chorus-fabric (transport) · prismresonance (wave memory)
Horizontal: VectorBridge (migration) · ChorusMesh (paid ops) · InsightPlugIn (IDE tooling)
```

Proven production combo (per `INSIGHT_ECOSYSTEM.md` Playbook C): CHORUS transport + PrismCache + PrismRAG taxonomy + PrismLang routing + PrismCortex memory, as run by InsightitsAIAgent across 8 vertical agents.
