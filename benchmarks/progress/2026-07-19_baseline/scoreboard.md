# Comparative benchmark scoreboard

Created: 2026-07-19T10:06:33Z  |  B1 n=200  B2 n=2

| system | track | n | F1 | precision | recall | AUROC | p50 ms | p95 ms | LLM calls | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| hhem | B1 | 200 | 0.7458 | 0.8571 | 0.66 | 0.7934 | 163.88 | 227.49 | 0 | 0 |
| hhem | B2 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 104.58 | 104.73 | 0 | 0 |
| hhem | Bsum | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| prismshine-fast | B1 | 200 | 0.5306 | 0.4483 | 0.65 | 0.4275 | 27.42 | 84.71 | 0 | 0 |
| prismshine-fast | B2 | 2 | 0.6667 | 0.5 | 1.0 | 0.5 | 6.9 | 13.56 | 0 | 0 |
| prismshine-fast | Bsum | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |

Latency is shim-internal (network excluded). RAGAS runs a reduced subset (pinned local judge is slow); its row is comparable on quality, not throughput.
Fairness rules: docs/BENCHMARKS.md § Comparative suite.