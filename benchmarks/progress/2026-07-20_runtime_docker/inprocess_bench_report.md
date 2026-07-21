# PrismShine benchmark receipt

- Created: `2026-07-20T23:41:05.042906+00:00`
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

## grounding

- Suite: **PASS**
- Gates:
  - `synthetic_f1_min`: `0.85`
  - `synthetic_f1`: `1.0`
  - `within_5pts_of_span_baseline`: `True`
  - `f1_delta_vs_span`: `0.1`
  - `hard_negation_caught`: `True`
  - `calibrate_lift_met`: `True`
- Metrics:
  - `synthetic`: `{'tp': 11, 'fp': 0, 'tn': 8, 'fn': 0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'accuracy': 1.0}`
  - `auroc`: `1.0`
  - `n_pairs`: `19`
  - `span_baseline`: `{'tp': 9, 'fp': 0, 'tn': 8, 'fn': 2, 'precision': 1.0, 'recall': 0.8182, 'f1': 0.9, 'accuracy': 0.8947, 'backend': 'lexical', 'artifact_id': 'lettucedetect-onnx-v1+lexical', 'note': 'In-process SpanClassifier; pin via PRISMSHINE_SPAN_ONNX for onnx'}`
  - `ragtruth`: `{'status': 'ran', 'tp': 23, 'fp': 68, 'tn': 5, 'fn': 4, 'precision': 0.2527, 'recall': 0.8519, 'f1': 0.3898, 'accuracy': 0.28, 'n': 100, 'auroc': 0.5560629122272958}`
  - `domain_calibrate`: `{'profile': 'clinical', 'pre_calibrate_f1': 1.0, 'calibrated_f1': 1.0, 'pre_calibrate_auroc': 1.0, 'calibrated_auroc': 1.0, 'f1_lift': 0.0, 'lift_gate_min': 0.1, 'lift_met': True, 'threshold_status': 'validated-labeled', 'bands_after': [0.05, 0.25, 0.45], 'version': 'cal-clinical-0.1', 'notes': ['F1 lift measured on synthetic negatives (band fit); labeled packs preferred for claims.', 'Gate: +0.10 decision-F1 or calibrated F1 >= 0.90.']}`
- Notes:
  - Includes hard_effect_pairs (negation/entity/finance/legal) offline.
  - Set PRISMSHINE_BENCH_FULL=1 + datasets for real RAGTruth subset.
  - Pin ONNX: PRISMSHINE_SPAN_ONNX (+ optional PRISMSHINE_SPAN_TOKENIZER).
  - Domain calibrate lift is a separate receipt row under domain_calibrate.
- Competitor baseline: `in-process span baseline only`

## latency_cost

- Suite: **PASS**
- Gates:
  - `tier0_p50_ms_soft_max`: `50`
  - `tier0_p50_ms`: `0.024`
  - `fast_p50_ms_soft_max`: `100`
  - `fast_p50_ms`: `0.626`
  - `positioning_fast_p50_target_ms`: `25`
  - `judge_escalation_rate_max`: `0.1`
  - `judge_escalation_rate`: `0.0`
  - `judge_escalation_soft_ci_max`: `0.25`
- Metrics:
  - `tier0_p50_ms`: `0.024`
  - `tier0_p95_ms`: `0.079`
  - `fast_p50_ms`: `0.626`
  - `fast_p95_ms`: `0.84`
  - `iterations`: `30`
  - `cost`: `{'n_checks': 1000, 'judge_usd_per_1k_all_traffic': 2.0, 'shine_usd_per_1k': 0.0, 'judge_escalation_rate': 0.0, 'savings_vs_judge_all_usd_per_1k': 2.0, 'notes': ['Judge cost is a configurable proxy (default $0.002/call).', 'Shine fast path assumes $0 CPU marginal; only escalations bill.', "Competitor cells without a local run stay 'literature / not run'."]}`
- Notes:
  - CI soft budgets are looser than POSITIONING local targets (CPU variance).
  - Judge escalation is a proxy without a live Tier-4 judge (no API spend).
  - POSITIONING: fast path p50 < 25ms local; judge <=10% on default profile.
- Competitor baseline: `literature / not run`

## consistency

- Suite: **PASS**
- Gates:
  - `detection_catch_rate_when_prevention_off`: `1.0`
  - `detection_catch_rate_min`: `1.0`
  - `fresh_cache_false_positive`: `False`
  - `prevention_hook_callable`: `True`
- Metrics:
  - `n_stale_scenarios`: `2`
  - `n_caught`: `2`
- Notes:
  - POSITIONING: zero stale-cache serves after correction — dual-rail.
  - Prevention off (no cache object) must not disable CACHE_PREDATES_FACT_UPDATE.
  - Competitors cannot see cache-gate ledger detail; suite is Shine-only.
- Competitor baseline: `literature / not run`

Claims without a green receipt are banned - see `docs/POSITIONING.md`.
