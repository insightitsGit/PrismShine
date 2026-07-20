# Stack suite — ACI run3 (correct Guard wiring)

**Authoritative stack receipt** after fixing PrismGuard integration.

## What was wrong in run1 / run2

| Run | Guard config | Issue |
|---|---|---|
| run1 | `rules_only` | Under-powered vs PrismGuard’s own law/ONNX benches → S1 F1 0.33 |
| run2 | `law_pilot` + ONNX | Guard hard-blocked **all** R1/H1 prompts before Shine → R1 catch 0.0 |

## run3 fix

- Guard: `prismguard:law_pilot:onnx=True` (matches published Guard path)
- **S1-only** hard-block; H1/R1 always reach Shine/ledger
- Image: `bench/insight-stack:v3`
- Containers **Stopped** after run

## Scoreboard

| system | S1 F1 | H1 F1 | R1 catch | R1 FA | P1 p50 |
|---|---:|---:|---:|---:|---:|
| **insight-stack** | 0.75 | **0.883** | **1.00** | 0.0 | 368 ms |
| oss-llmguard | **1.00** | 0.429 | 0.00 | 0.0 | 338 ms |
| oss-langgraph-hhem | 0.667 | 0.795 | 0.00 | 0.0 | 152 ms |

## Reading

- **R1 + H1**: Insight package story holds (runtime moat + strong grounding).
- **S1**: Much better than rules_only (0.33 → 0.75); llm-guard still leads on this generic jailbreak set (Guard has more FPs with law ONNX). Cite PrismGuard’s **own law holdout** for injection quality claims, not only this S1 set.

Local: pytest + `prismshine bench --suite all` **PASS** (same session).
