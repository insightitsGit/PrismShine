# Stack suite — ACI run4 (full Guard + ChorusGraph)

> **INTERNAL** — not a PrismShine public claim. For Shine product numbers see
> [`../2026-07-20_run4_onnx/`](../2026-07-20_run4_onnx/README.md).

**Date:** 2026-07-20  
**Image:** `bench/insight-stack:v4`  
**Why:** Docs under-wiring fixed — `law_pilot` + ONNX + `prismrag` taxonomy + feedback + real ChorusGraph graph (`make_guard_handler` → shine pre/post).

Handoff for Guard docs: `handoffs/handoff-prismguard-docs-features.md`

## Health caps (truth)

```
guard: prismguard:law_pilot:onnx_ready:prismrag:feedback+chorusgraph
guard_caps: onnx_ready, prismrag_taxonomy, feedback_persist, llm_judge, law_overlay
chorus_caps: require_shine, make_guard_handler, register_interceptor, on_fact_corrected,
             bump_partition_version, SqliteLedgerSink, PrismCacheBackend
```

## Scoreboard

| system | S1 F1 | H1 F1 | R1 catch | R1 FA | P1 p50 |
|---|---:|---:|---:|---:|---:|
| **insight-stack** | 0.75 | **0.883** | **1.00** | 0.0 | ~478 ms |
| oss-llmguard | **1.00** | 0.429 | 0.00 | 0.0 | ~474 ms |
| oss-langgraph-hhem | 0.667 | 0.795 | 0.00 | 0.0 | ~144 ms |

Receipts: `bench/stack/results/aci_run4/`

## vs run3

| | run3 (ONNX only) | run4 (full stack) |
|---|---|---|
| S1 F1 | 0.75 | 0.75 |
| H1 F1 | 0.883 | 0.883 |
| R1 catch | 1.0 | 1.0 |
| P1 p50 | ~368 ms | ~478 ms |

Quality unchanged on this fixed set; **~110 ms** added from ChorusGraph envelope path (expected). Capability truth is now honest (`/health` caps).

## Containers

All stack ACI groups **Stopped** after the run.
