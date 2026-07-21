# Runtime suite scoreboard (ChorusGraph + PrismShine vs competitors)

Systems: **chorus-shine** (wired ledger) vs **oss-langgraph-hhem** (HHEM) vs **oss-langgraph-minilm** (DIY MiniLM) vs **oss-langgraph-lettuce** (LettuceDetect spans). No Guard / no S1.

R1 is **evidence-aware**: only systems that inspect the runtime ledger can catch empty retrieval / swallowed tool errors / stale cache. Content-only graphs report `saw_evidence=false` and always label R1 as `runtime_ok`.

| system | H1 hallucination F1 | R1 catch rate | R1 false alarm | saw_evidence | P1 p50 ms | P1 p95 ms | LLM calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| chorus-shine | 0.8947 | 1.0 | 0.0 | 1.0 | 1.73 | 3.51 | 0 |
| oss-langgraph-hhem | 0.7059 | 0.0 | 0.0 | 0.0 | 4446.52 | 16202.21 | 0 |
| oss-langgraph-minilm | 0.4878 | 0.0 | 0.0 | 0.0 | 1452.48 | 3696.09 | 0 |
| oss-langgraph-lettuce | 0.6875 | 0.0 | 0.0 | 0.0 | 7348.48 | 20796.06 | 0 |

P1 is derived from all H1+R1 requests. Latency is shim-internal (network excluded).
Public Shine-only vs HHEM (content track) remains in `docs/BENCHMARKS.md` / `bench/`.
Guard+ChorusGraph+Shine package QA remains INTERNAL in `bench/stack/`.