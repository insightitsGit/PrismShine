# Runtime suite — Docker receipt (2026-07-20)

ChorusGraph + PrismShine vs LangGraph-shaped HHEM / MiniLM / LettuceDetect.

- Data: HaluEval H1 n=40 (20 source rows × 2), R1 n=12
- Topology: `bench/compose.runtime.yaml` local Docker Desktop
- Images: `prismshine-chorus-shine`, `oss-langgraph-hhem`, `oss-langgraph-minilm`, `oss-langgraph-lettuce`

## Scoreboard

| system | H1 F1 | R1 catch | R1 FA | P1 p50 ms | LLM |
|---|---:|---:|---:|---:|---:|
| **chorus-shine** | **0.895** | **1.0** | 0.0 | **1.7** | 0 |
| oss-langgraph-hhem | 0.706 | 0.0 | 0.0 | 4447 | 0 |
| oss-langgraph-minilm | 0.488 | 0.0 | 0.0 | 1452 | 0 |
| oss-langgraph-lettuce | 0.688 | 0.0 | 0.0 | 7348 | 0 |

**Takeaway:** wired ChorusGraph+Shine owns R1 (competitors ignore ledger evidence) and leads H1 F1 at ~2500× lower p50 than HHEM in this containerized run.

## Full report

See **[FULL_REPORT.md](./FULL_REPORT.md)** for method, caveats, in-process gates, Azure teardown, and reproduce steps.

Artifacts: `summary.json`, `scoreboard.md`, `raw_*.jsonl`, `inprocess_bench_report.md` (if present).
