# Stack suite (package vs package)

This suite is separate from `bench/runner/` content-only HHEM benches. It compares:

- `insight-stack`: PrismGuard first, then PrismShine with wiring/ledger evidence.
- `oss-llmguard`: LLM Guard `PromptInjection` (structural-regex fallback) and MiniLM
  cosine grounding.
- `oss-langgraph-hhem`: a light regex guard followed by Vectara HHEM.

S1 measures prompt-injection detection, H1 measures HaluEval hallucination F1, R1
measures injected runtime-failure catch rate and false alarms, and P1 derives
latency/call counters. R1 is explicitly evidence-aware: the two OSS paths ignore
`evidence` and report `saw_evidence=false`.

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
