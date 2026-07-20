# PrismShine changelog

## 0.2.0 — enterprise hardening (2026-07-20)

- P0–P2 bug-fix handoff (`handoffs/handoff-fix1.md`): overrides persist, traffic-based Tier-4 budget, numpy warm-index vectors, Tier-0 LRU, ONNX load safety, monotone judge fusion, contradiction word boundaries, WeakKey wiring registries, encoder/judge LRUs, copycheck/judge/spans polish.
- `python -m prismshine.tools.ensure_span_onnx` — resolve or export Tier-3 ONNX for pip installs (weights not in the wheel).
- `python -m prismshine.bench.calibrate_minilm --embedder hash|minilm` — crash-safe calibration overlays.
- Comparative runner `--runs N` median scoreboard (enterprise multi-seed gate).
- Examples: `examples/enterprise_wiring_demo.py`, `examples/tier4_judge_demo.py`.
- Comparative receipt: HaluEval vs HHEM run4 ONNX (B1 F1 0.831 > HHEM 0.746; B2 1.0 / 0 FP).
- Public README rewritten for PyPI (install paths, wiring recipes, Shine-only benchmark headline).

## 0.1.0

- Initial unified pipeline: handbook forensics + tiered grounding, wiring helpers, bench suites.
