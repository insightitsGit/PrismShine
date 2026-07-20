# Stack suite scoreboard

R1 is an **evidence-aware** runtime track and is intentionally separate from S1/H1.

| system | S1 attack F1 | H1 hallucination F1 | R1 catch rate (evidence-aware) | R1 false alarm | P1 p50 ms | P1 p95 ms | LLM calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| oss-llmguard | 1.0 | 0.4286 | 0.0 | 0.0 | 473.7 | 1056.78 | 0 |
| insight-stack | 0.75 | 0.8831 | 1.0 | 0.0 | 477.59 | 695.41 | 0 |
| oss-langgraph-hhem | 0.6667 | 0.7945 | 0.0 | 0.0 | 144.35 | 321.5 | 0 |

P1 is derived from all evaluated S1/H1/R1 requests. Latency is shim-internal.