# Handoff — PrismGuard DX: quick-start vs scorecard path

**To:** PrismGuard maintainers  
**From:** PrismShine stack-suite integration (2026-07-20)  
**Why:** Integrators copied the README quick start, then compared S1 results to the law scorecard and concluded “Guard is weak.” That was a **config mismatch**, but it is a predictable developer mistake.

## What happened

1. PrismShine `insight-stack` used `create_checker_rules_only()` (same family as README `web_chat`).
2. Stack S1 attack F1 was **0.33**.
3. Switching to `law_pilot` + `PRISMGUARD_USE_ONNX=1` + `prismguard-model download` raised S1 to **0.75** on the same set.
4. Guard’s own law holdout still claims **100%** block — different dataset + fuller bench harness.

The library behavior was consistent with docs; the **docs hierarchy** still leads people to the light path while marketing numbers come from the heavy path.

## Ask (docs / API clarity)

1. **Above the fold in README:** a two-row table:

   | Goal | Call |
   |---|---|
   | Hub / FAQ, low false positives | `create_checker_for_app("web_chat")` |
   | Match published law injection scorecard | `create_checker_for_app("law_pilot", use_onnx=True)` after `pip install prismguard[guard-model]` + `prismguard-model download` + `PRISMGUARD_USE_ONNX=1` |

2. Next to the scorecard / COMPARISON_REPORT link: **“Do not expect these rates from `web_chat` / `rules_only`.”**

3. Optional: `create_checker_for_app("security_bench")` alias that fails loudly if ONNX weights are missing (instead of silently rules-degrading).

## Acceptance

- A new integrator reading only the first screen of the README can tell which factory matches the scorecard.
- No change required to scorecard methodology — only clearer path labeling.

## PrismShine side (already done / in flight)

- See also **`handoffs/handoff-prismguard-docs-features.md`** (full learn-from-seed / `[prism]` / feedback / ChorusGraph docs gap).
- Stack shim: `law_pilot` + ONNX + `prismrag-patch` + feedback + ChorusGraph `make_guard_handler` + Shine `require_shine`.
- `GET /health` → `guard_caps` / `chorus_caps`.
- `bench/stack/README.md` documents the pitfall.
- Stack S1 is not cited as a Guard quality claim without noting profile + artifact.
