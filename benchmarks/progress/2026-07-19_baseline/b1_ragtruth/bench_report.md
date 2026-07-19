# PrismShine benchmark receipt

- Created: `2026-07-19T10:08:30.291715+00:00`
- Overall: **PASS**

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
  - `ragtruth`: `{'status': 'ran', 'tp': 23, 'fp': 69, 'tn': 4, 'fn': 4, 'precision': 0.25, 'recall': 0.8519, 'f1': 0.3866, 'accuracy': 0.27, 'n': 100, 'auroc': 0.5286656519533232}`
  - `domain_calibrate`: `{'profile': 'clinical', 'pre_calibrate_f1': 1.0, 'calibrated_f1': 1.0, 'pre_calibrate_auroc': 1.0, 'calibrated_auroc': 1.0, 'f1_lift': 0.0, 'lift_gate_min': 0.1, 'lift_met': True, 'threshold_status': 'validated-labeled', 'bands_after': [0.05, 0.25, 0.45], 'version': 'cal-clinical-0.1', 'notes': ['F1 lift measured on synthetic negatives (band fit); labeled packs preferred for claims.', 'Gate: +0.10 decision-F1 or calibrated F1 >= 0.90.']}`
- Notes:
  - Includes hard_effect_pairs (negation/entity/finance/legal) offline.
  - Set PRISMSHINE_BENCH_FULL=1 + datasets for real RAGTruth subset.
  - Pin ONNX: PRISMSHINE_SPAN_ONNX (+ optional PRISMSHINE_SPAN_TOKENIZER).
  - Domain calibrate lift is a separate receipt row under domain_calibrate.
- Competitor baseline: `in-process span baseline only`

Claims without a green receipt are banned - see `docs/POSITIONING.md`.
