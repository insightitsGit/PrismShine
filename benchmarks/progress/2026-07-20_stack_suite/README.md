# Stack suite ACI run1 — 2026-07-20

Receipts: `bench/stack/results/aci_run1/` (copied scoreboard below).

Containers stopped after run. Note: `shine-bench` / `ragas-bench` ACI groups were
**deleted** (not just stopped) to free eastus core quota; images remain in ACR
`prismshinebenchbx7k2m` and can be recreated.

## Scoreboard

| system | S1 attack F1 | H1 hallucination F1 | R1 catch (evidence-aware) | R1 false alarm | P1 p50 ms |
|---|---:|---:|---:|---:|---:|
| **insight-stack** | 0.333 | **0.883** | **1.000** | 0.0 | **1.8** |
| oss-llmguard | **1.000** | 0.429 | 0.000 | 0.0 | 562 |
| oss-langgraph-hhem | 0.667 | 0.795 | 0.000 | 0.0 | 165 |

## Reading

- **R1** is the package moat: Insight catches all injected runtime failures; both OSS stacks score **0** (they ignore ledger evidence).
- **H1**: Insight leads vs HHEM-in-stack and vs MiniLM cosine.
- **S1**: `prismguard:rules_only` recall is weak on this jailbreak set (4/20); llm-guard wins injection. Follow-up: enable PrismGuard ONNX / authored seed profile for S1 parity.
- **P1**: Insight p50 ~2 ms (hash embedder + rules); competitors pay model latency.

## Commands used

ACR tags: `bench/insight-stack:v1`, `bench/oss-llmguard:v1`, `bench/oss-langgraph-hhem:v1`.
Runner: `python bench/stack/run_stack_bench.py --targets bench/stack/targets.aci.json --n-h1 40`.
