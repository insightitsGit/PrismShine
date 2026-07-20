# Comparative benchmark scoreboard

Created: 2026-07-19T10:37:28Z  |  B1 n=200  B2 n=50

| system | track | n | F1 | precision | recall | AUROC | p50 ms | p95 ms | LLM calls | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| prismshine-fast | B1 | 200 | 0.7186 | 0.8955 | 0.6 | 0.838 | 29.88 | 67.81 | 0 | 0 |
| prismshine-fast | B2 | 50 | 1.0 | 1.0 | 1.0 | 1.0 | 13.59 | 43.34 | 0 | 0 |
| prismshine-fast | Bsum | 50 | 0.5652 | 0.619 | 0.52 | 0.5848 | 133.62 | 692.96 | 0 | 0 |
| hhem | B1 | 200 | 0.7458 | 0.8571 | 0.66 | 0.7934 | 169.02 | 237.31 | 0 | 0 |
| hhem | B2 | 50 | 0.9259 | 0.8621 | 1.0 | 1.0 | 134.78 | 204.84 | 0 | 0 |
| hhem | Bsum | 50 | 0.4737 | 0.6923 | 0.36 | 0.616 | 2101.0 | 5025.29 | 0 | 0 |

Latency is shim-internal (network excluded). RAGAS runs a reduced subset (pinned local judge is slow); its row is comparable on quality, not throughput.
Fairness rules: docs/BENCHMARKS.md § Comparative suite.