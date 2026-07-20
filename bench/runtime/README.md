# Runtime suite — ChorusGraph + PrismShine vs competitors

**What this proves:** the wired-runtime moat (ledger-aware cause-side halt +
grounding) without PrismGuard. Complements the public content-only Shine vs HHEM
suite (`bench/`, `docs/BENCHMARKS.md`) and stays separate from the INTERNAL
Guard+ChorusGraph+Shine package QA (`bench/stack/`, now pinned to
[`prismguard==0.1.8`](https://pypi.org/project/prismguard/0.1.8/)).

| System | Role | Sees ledger `evidence`? |
|--------|------|-------------------------|
| **chorus-shine** | ChorusGraph graph + PrismShine (`require_shine` / shine pre+post) | **Yes** |
| **oss-langgraph-hhem** | Sequential LangGraph-shaped nodes + Vectara HHEM-2.1-Open | No |
| **oss-langgraph-minilm** | Sequential LangGraph-shaped nodes + MiniLM cosine faithfulness (DIY) | No |
| **oss-langgraph-lettuce** | Sequential LangGraph-shaped nodes + [LettuceDetect](https://pypi.org/project/lettucedetect/) spans (closest open peer to Shine Tier-3) | No |

## Tracks

| Track | What | Why |
|-------|------|-----|
| **H1** | HaluEval hallucination F1 | Effect-side parity vs encoder / DIY / LettuceDetect |
| **R1** | Injected runtime failures (empty retrieval, tool error, stale cache, …) | Only evidence-aware systems can catch these |
| **P1** | Latency / LLM calls (derived) | Cost / speed story |

No **S1** (prompt injection) — that is Guard’s lane / `bench/stack/` (use **PrismGuard 0.1.8+**).

## Local

```powershell
docker compose -f bench/compose.runtime.yaml up --build -d
python -m pip install httpx
python -m bench.runtime.run_runtime_bench `
  --targets bench/runtime/targets.example.json `
  --n-h1 40 `
  --out bench/runtime/results/local
docker compose -f bench/compose.runtime.yaml down
```

## Azure ACI (same pattern as stack suite)

1. Build/push the four images to ACR.
2. Start ACI groups (identical CPU/RAM), write a targets JSON with public URLs.
3. Run `python -m bench.runtime.run_runtime_bench --targets … --out bench/runtime/results/aci`.
4. **Stop** ACI groups after the receipt (billing continues while Running).

## Smoke (no Docker)

```powershell
python -m bench.runtime.smoke_local --n-h1 4 --out bench/runtime/results/smoke
```

Exercises `chorus-shine` in-process plus lightweight competitor stubs (same
contracts as the containers) so CI stays green without baking HHEM / MiniLM /
LettuceDetect images.
