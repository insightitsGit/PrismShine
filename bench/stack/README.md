# Stack suite (package vs package) — INTERNAL

**Not a PrismShine product benchmark.** Do not cite these scoreboards in README,
`docs/POSITIONING.md`, or public marketing. Public Shine receipts live in
`docs/BENCHMARKS.md` (vs HHEM / in-process cause·grounding·latency·consistency).

This folder is ecosystem integration QA only.

- `insight-stack`: PrismGuard (`make_guard_handler`) → ChorusGraph → PrismShine
  (`require_shine` / shine pre+post) with ledger evidence.
- `oss-llmguard`: LLM Guard `PromptInjection` (structural-regex fallback) and MiniLM
  cosine grounding.
- `oss-langgraph-hhem`: a light regex guard followed by Vectara HHEM.

S1 measures prompt-injection detection, H1 measures HaluEval hallucination F1, R1
measures injected runtime-failure catch rate and false alarms, and P1 derives
latency/call counters. R1 is explicitly evidence-aware: the two OSS paths ignore
`evidence` and report `saw_evidence=false`.

## PrismGuard wiring pitfall (read before changing the shim)

**Our run1 S1 miss was mostly a config mistake, amplified by Guard’s “quick start” defaults.**

| Path | What you get | When to use |
|---|---|---|
| `create_checker_rules_only()` / `create_checker_for_app("web_chat")` | Rules-first, **no ONNX** unless you opt in | Fast local smoke, FAQ hubs |
| `create_checker_for_app("law_pilot", use_onnx=True)` + `prismguard-model download` + `PRISMGUARD_USE_ONNX=1` | Seed + **ONNX** (matches Guard’s published law scorecard) | Security / injection benches |
| Same + `prismrag-patch` + `PRISMGUARD_FEEDBACK_PERSIST=1` (+ ChorusGraph handler) | Law overlay **words** + prismrag taxonomy + feedback queue | Full “learn from seed / traffic” path |

Guard’s README headline example is `web_chat` (“rules-first, no surprise ONNX”). Its **scorecard** numbers use the law + ONNX path. Developers who copy the quick start and then expect scorecard-level S1 will under-perform — same mistake we made with `rules_only` in run1.

`insight-stack` is pinned to **`law_pilot` + ONNX + `prismrag-patch` + feedback persist + ChorusGraph** (not `security_bench`: that factory forces HashEmbedder / `skip_taxonomy`). Hard-block only on **S1**; H1/R1 always reach Shine so Guard FPs cannot zero the runtime track (run2 bug).

Deps install `prismrag-patch` beside `prismguard[guard-model]>=0.1.8` and **`prismlib-plus` via ChorusGraph** — never `prismguard[prism]` (pulls bare `prismlib` and collides with plus).

**Pin:** use [`prismguard 0.1.8`](https://pypi.org/project/prismguard/0.1.8/) (or newer) for stack ACI rebuilds — DX / docs fixes from the Guard handoffs land there.

### Learn-from-DB / customer words (env)

| Env | Effect |
|---|---|
| `PRISMGUARD_FEEDBACK_PERSIST=1` | Queue blocks / near-miss allows for later `prismguard feedback export` → train |
| `PRISMGUARD_STORAGE_BACKEND` / `_DSN` | Default `memory`. Persistent pgvector/chroma/… needs Guard Team+ license |
| `PRISMGUARD_EXTRA_SEED_PATH` | Extra YAML of *your* attack/benign phrases imported at boot |
| `PRISMGUARD_TENANT_LEXICON_PATH` | Tenant entity lexicon (severity / force-classifier on override) |
| `GET /health` → `guard_caps` / `chorus_caps` | Truth for Guard + ChorusGraph wiring |

### ChorusGraph wiring (required for this container)

`START → guard → {s1_done | shine_pre (R1) | shine_post (H1)} → END` using:

- `prismguard.integrations.chorusgraph.make_guard_handler` + `route_after_guard`
- `prismshine.integrations.chorusgraph.require_shine` + `shine_node` / pre-gen adapter
- `ChorusStack` ledger + sidecar + `on_fact_corrected` consistency surface

See: `handoffs/handoff-prismguard-docs-features.md` (docs gap that caused the under-wire).

## Local

```powershell
docker compose -f bench/compose.stack.yaml up --build -d
python -m pip install httpx
python bench/stack/run_stack_bench.py `
  --targets bench/stack/targets.example.json --n-h1 40 --out bench/stack/results/local
docker compose -f bench/compose.stack.yaml down
```

The HHEM and MiniLM images download/bake model weights during build; allow enough
disk, RAM, and network access before timing the suite. The runner writes raw JSONL,
`summary.json`, and `scoreboard.md` under `--out`.

## Azure Container Instances

1. Build and push all three images to an ACR. `insight-stack` must be built with the
   repository root as context because it installs the local `prismshine` package.
2. Start one ACI group per image with externally reachable port 8000, identical CPU
   and memory limits, and record the three FQDNs:

```powershell
az container create -g <resource-group> -n insight-stack `
  --image <acr>.azurecr.io/insight-stack:<tag> --ports 8000 --ip-address Public `
  --cpu 2 --memory 4 --registry-login-server <acr>.azurecr.io
# Repeat for oss-llmguard and oss-langgraph-hhem with their own images.
```

3. Create a targets file using `http://<aci-fqdn>:8000`, run the benchmark, and save
   the receipt:

```powershell
python bench/stack/run_stack_bench.py --targets stack-targets.json --out bench/stack/results/aci
```

4. **Stop** every ACI group after the receipt is written (preferred over delete while iterating):

```powershell
az container stop -g <resource-group> -n insight-stack
az container stop -g <resource-group> -n oss-llmguard
az container stop -g <resource-group> -n oss-langgraph-hhem
```

ACI continues to incur compute charges while Running; stopping is mandatory after each benchmark. Delete the resource group only when tearing down for good.
