# Stack suite scoreboard

R1 is an **evidence-aware** runtime track and is intentionally separate from S1/H1.

| system | S1 attack F1 | H1 hallucination F1 | R1 catch rate (evidence-aware) | R1 false alarm | P1 p50 ms | P1 p95 ms | LLM calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| oss-llmguard | 1.0 | 0.4286 | 0.0 | 0.0 | 562.25 | 1093.26 | 0 |
| insight-stack | 0.3333 | 0.8831 | 1.0 | 0.0 | 1.82 | 4.08 | 0 |
| oss-langgraph-hhem | 0.6667 | 0.7945 | 0.0 | 0.0 | 164.81 | 454.98 | 0 |

P1 is derived from all evaluated S1/H1/R1 requests. Latency is shim-internal.