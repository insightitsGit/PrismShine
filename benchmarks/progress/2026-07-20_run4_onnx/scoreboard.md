# Comparative benchmark scoreboard

Created: 2026-07-20T17:06:44Z  |  B1 n=200  B2 n=50

| system | track | n | F1 | precision | recall | AUROC | p50 ms | p95 ms | LLM calls | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| prismshine-fast | B1 | 200 | 0.8306 | 0.9157 | 0.76 | 0.8427 | 90.36 | 498.93 | 0 | 0 |
| prismshine-fast | B2 | 50 | 1.0 | 1.0 | 1.0 | 1.0 | 19.57 | 158.84 | 0 | 0 |
| prismshine-fast | Bsum | 50 | 0.6 | 0.6 | 0.6 | 0.5624 | 1398.05 | 9311.06 | 0 | 0 |
| hhem | B1 | 200 | 0.7458 | 0.8571 | 0.66 | 0.7934 | 215.86 | 316.65 | 0 | 0 |
| hhem | B2 | 50 | 0.9259 | 0.8621 | 1.0 | 1.0 | 165.56 | 255.51 | 0 | 0 |
| hhem | Bsum | 50 | 0.4737 | 0.6923 | 0.36 | 0.616 | 1898.79 | 3809.89 | 0 | 0 |

Latency is shim-internal (network excluded). RAGAS runs a reduced subset (pinned local judge is slow); its row is comparable on quality, not throughput.
Fairness rules: docs/BENCHMARKS.md § Comparative suite.