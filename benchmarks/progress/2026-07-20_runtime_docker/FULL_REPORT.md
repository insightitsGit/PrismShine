# Full report — PrismShine runtime suite + in-process gates (2026-07-20)

**Date:** 2026-07-20  
**Repo:** PrismShine `main`  
**Azure teardown:** `az group delete -n prismshine-bench-rg --yes` queued (ACI + ACR `prismshinebenchbx7k2m`)

---

## 1. Executive verdict

**Good product receipt for the wired-runtime claim.**

| Claim | Result | Use in marketing? |
|---|---|---|
| Cause-side / ledger moat (R1) | chorus-shine catch **1.0**, competitors **0.0** | Yes — with “evidence-aware” label |
| Effect-side H1 vs open peers | Shine **0.895** > HHEM **0.706** > Lettuce **0.688** > MiniLM **0.488** | Yes as *runtime-suite* receipt; keep ACI run4 as public Shine-only headline |
| Latency / cost | Shine p50 **1.7 ms**, 0 LLM; peers seconds on CPU containers | Yes with “containerized CPU, cold-ish peers” caveat |
| In-process gates | cause / grounding / latency / consistency all **PASS** | Yes |

---

## 2. Runtime suite (Docker Desktop) — primary comparative receipt

**Path:** `bench/runtime/results/docker_2026-07-20/`  
**Progress copy:** this folder  

### Topology

```
docker compose -f bench/compose.runtime.yaml --project-directory . up -d
python -m bench.runtime.run_runtime_bench \
  --targets bench/runtime/targets.example.json \
  --n-h1 20 \
  --out bench/runtime/results/docker_2026-07-20
```

| System | Role | Sees ledger? |
|---|---|---|
| **chorus-shine** | ChorusGraph + PrismShine (`require_shine`, shine pre/post) | Yes |
| oss-langgraph-hhem | LangGraph-shaped + Vectara HHEM-2.1-Open | No |
| oss-langgraph-minilm | LangGraph-shaped + MiniLM cosine DIY | No |
| oss-langgraph-lettuce | LangGraph-shaped + LettuceDetect spans | No |

### Data

| Track | n | Description |
|---|---:|---|
| H1 | 40 | HaluEval QA (20 source rows × grounded/hallucinated) |
| R1 | 12 | Injected runtime failures + clean controls |
| P1 | 52 | Derived from all H1+R1 requests (shim-internal latency) |

### Scoreboard

| system | H1 F1 | H1 P | H1 R | R1 catch | R1 FA | saw_evidence | P1 p50 ms | P1 p95 ms | LLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **chorus-shine** | **0.8947** | 0.9444 | 0.85 | **1.0** | 0.0 | 1.0 | **1.73** | 3.51 | 0 |
| oss-langgraph-hhem | 0.7059 | 0.8571 | 0.60 | 0.0 | 0.0 | 0.0 | 4446.52 | 16202.21 | 0 |
| oss-langgraph-minilm | 0.4878 | 0.4762 | 0.50 | 0.0 | 0.0 | 0.0 | 1452.48 | 3696.09 | 0 |
| oss-langgraph-lettuce | 0.6875 | 0.9167 | 0.55 | 0.0 | 0.0 | 0.0 | 7348.48 | 20796.06 | 0 |

### Chorus-shine detail (H1)

- tp=17, fp=1, fn=3 → F1 **0.8947**
- R1: 6/6 injected failures caught, 0/6 clean false alarms

### How to read R1

Competitors are **content-only** — they receive `evidence` but ignore it (`saw_evidence=false`) and always label R1 as `runtime_ok`. R1 catch **0.0** is expected, not a broken competitor. The gap is the product story.

### Latency caveat

Peer p50 in the multi-second range includes first-load / CPU container variance (especially LettuceDetect). Do not claim “2500× faster than HHEM” as a hard SLO without a warm multi-run ACI median. The **order of magnitude** and **0 LLM** claims are solid.

### Artifacts

- `summary.json` — machine scoreboard  
- `scoreboard.md` — human table  
- `raw_*.jsonl` — per-sample receipts (in `bench/runtime/results/docker_2026-07-20/`)

---

## 3. In-process Shine suites (same day)

**Path:** `benchmarks/reports/local_2026-07-20/`  
**Command:** `prismshine bench --suite all --report benchmarks/reports/local_2026-07-20`  
**Overall:** **PASS**

| Suite | Result | Highlights |
|---|---|---|
| cause_side | PASS | catch_rate 1.0 (9/9), pre_gen_model_calls 0 |
| grounding | PASS | synthetic F1 1.0; span baseline lexical |
| latency_cost | PASS | fast p50 0.63 ms; judge escalation 0.0 |
| consistency | PASS | stale-cache detection 1.0 when prevention off |

---

## 4. Related public headline (unchanged)

Shine-only vs HHEM on ACI (ONNX Tier-3):  
`benchmarks/progress/2026-07-20_run4_onnx/`  

| track | Shine F1 | HHEM F1 |
|---|---:|---:|
| B1 QA | 0.831 | 0.746 |
| B2 numbers | 1.000 | 0.926 |

Use **run4** for public “beats HHEM” grounding claims; use **this runtime report** for “wired ChorusGraph+Shine vs LangGraph stacks.”

---

## 5. Azure / cost teardown

| Resource | Action |
|---|---|
| ACI `hhem-bench`, `insight-stack`, `oss-llmguard`, `oss-langgraph-hhem` | Deleted with RG |
| ACR `prismshinebenchbx7k2m` | Deleted with RG |
| RG `prismshine-bench-rg` | **Deleted** — `ResourceGroupNotFound` confirmed 2026-07-20 |
| Local `compose.runtime.yaml` | `docker compose … down` |

**Not deleted (other products / not PrismShine bench):** `rg-insightits-dev`, `rg-insightits-prod`, VS Online RG, NetworkWatcherRG.

---

## 6. Reproduce

```powershell
# Local Docker (no Azure cost)
docker compose -f bench/compose.runtime.yaml --project-directory . up --build -d
python -m bench.runtime.run_runtime_bench `
  --targets bench/runtime/targets.example.json --n-h1 20 `
  --out bench/runtime/results/rerun
docker compose -f bench/compose.runtime.yaml --project-directory . down

# In-process gates
prismshine bench --suite all --report benchmarks/reports/rerun
```
