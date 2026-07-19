# Comparative benchmark scoreboard

Created: 2026-07-19T10:39:27Z  |  B1 n=200  B2 n=2

| system | track | n | F1 | precision | recall | AUROC | p50 ms | p95 ms | LLM calls | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| prismshine-fast | B1 | 200 | 0.7186 | 0.8955 | 0.6 | 0.838 | 0.23 | 0.45 | 0 | 0 |
| prismshine-fast | B2 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.55 | 0.89 | 0 | 0 |
| prismshine-fast | Bsum | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| hhem | B1 | 200 | 0.7458 | 0.8571 | 0.66 | 0.7934 | 279.86 | 417.02 | 0 | 0 |
| hhem | B2 | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 173.06 | 176.13 | 0 | 0 |
| hhem | Bsum | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |

Latency is shim-internal (network excluded). RAGAS runs a reduced subset (pinned local judge is slow); its row is comparable on quality, not throughput.
Fairness rules: docs/BENCHMARKS.md § Comparative suite.