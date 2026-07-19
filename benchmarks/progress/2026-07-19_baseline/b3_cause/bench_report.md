# PrismShine benchmark receipt

- Created: `2026-07-19T10:06:37.975646+00:00`
- Overall: **PASS**

## cause_side

- Suite: **PASS**
- Gates:
  - `catch_rate_min`: `0.9`
  - `catch_rate`: `1.0`
  - `false_alarm_on_clean`: `False`
  - `pre_gen_model_calls`: `0`
  - `tokens_avoided_cases`: `1`
- Metrics:
  - `n_injected`: `9`
  - `n_caught`: `9`
  - `catch_rate`: `1.0`
  - `clean_decision`: `pass`
- Notes:
  - Competitors that only see (context, question, answer) score N/A on this suite.
  - POSITIONING gate: >=90% injected runtime failures caught by Tier-0.
  - Pre-gen halt proves tokens avoided (model never called).
- Competitor baseline: `literature / not run`

Claims without a green receipt are banned - see `docs/POSITIONING.md`.
