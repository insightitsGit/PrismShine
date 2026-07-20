# bench/ — containerized comparative benchmark

Implements `docs/BENCHMARKS.md` § Comparative suite: PrismShine vs **HHEM-2.1-Open**
(encoder SotA) and **RAGAS faithfulness** (LLM-judge path, pinned local Ollama judge),
each in its own container behind one common `POST /evaluate` contract.

## Layout

```
bench/
  shims/prismshine/   FastAPI shim + Dockerfile (T0–T3 fast path, MiniLM embedder baked in)
  shims/hhem/         HHEM-2.1-Open shim (weights baked in)
  shims/ragas/        RAGAS faithfulness shim (judge via Ollama)
  shims/ollama/       Ollama with llama3.2:3b-instruct-q4_K_M baked in
  runner/run_bench.py Orchestrator: HaluEval B1 + numbers-slice B2, F1/AUROC/latency scoreboard
  compose.yaml        Local run (docker compose)
  azure/deploy.ps1    Azure run (az CLI: RG + ACR builds + 3 ACI groups)
```

## Azure (az CLI)

```powershell
pwsh bench/azure/deploy.ps1          # from repo root; prints endpoints, writes targets.json
python -m pip install -r bench/runner/requirements.txt
python bench/runner/run_bench.py --targets bench/runner/targets.json `
  --n 100 --b2 25 --ragas-limit 30 --out bench/runner/results/run1
az group delete -n prismshine-bench-rg --yes --no-wait   # teardown
```

## Local

```powershell
docker compose -f bench/compose.yaml up --build -d
# targets: prismshine http://localhost:8001, hhem :8002, ragas :8003
```

## Notes

- Latency is measured **inside each shim** (network excluded); first 5 samples excluded (warm-up).
- RAGAS runs a reduced subset (`--ragas-limit`) — a pinned 3B CPU judge is slow; its row
  is comparable on quality, not throughput. Optional OpenAI mode = best-case competitor.
- Content-only track: shims receive only (question, context, answer). PrismShine's
  ledger-aware B3 track stays in the in-process suite (`prismshine bench --suite cause`).

## Stack suite

`bench/stack/` is **internal ecosystem QA** (not a PrismShine public scoreboard).
See `bench/stack/README.md`. Public claims use this comparative suite + `docs/BENCHMARKS.md`.
