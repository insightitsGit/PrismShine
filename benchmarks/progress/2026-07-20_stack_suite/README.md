# Stack suite — design locked 2026-07-20

Package-vs-package comparative (not Shine-alone vs HHEM).

| Container | Stack |
|---|---|
| `insight-stack` | PrismGuard → PrismShine wiring/ledger evidence |
| `oss-llmguard` | LLM Guard PromptInjection (+ regex fallback) + MiniLM cosine |
| `oss-langgraph-hhem` | Regex jailbreak filter → Vectara HHEM |

| Track | Measures | Fairness |
|---|---|---|
| S1 | Prompt injection | Like-for-like |
| H1 | Hallucination (HaluEval) | Like-for-like |
| R1 | Runtime / cause-side failures | **Evidence-aware** — OSS ignore ledger |
| P1 | Latency / LLM calls | Derived from requests |

Smoke (local, no ACI): `python bench/stack/smoke_local.py` — insight Guard+R1 halt OK; llmguard S1 OK; HHEM skipped in smoke.

ACI: start 3 → `run_stack_bench.py` → **stop** all three.
