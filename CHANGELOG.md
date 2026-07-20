# PrismShine changelog

## 0.2.1 — packaging + install honesty (2026-07-20)

- **Sdist trim:** hatch `only-include` for `prismshine` / `tests` / `examples` / docs text — excludes `bench/`, `benchmarks/`, `kb/`, doc images, and baked tokenizer/ONNX so from-source installs stay lean (0.2.0 sdist was ~20 MB).
- **Docs:** README install section now calls out bare `pip install prismshine` vs enterprise `prismshine[spans]` + `python -m prismshine.tools.ensure_span_onnx` (Tier-3 / run4-parity path; bare install degrades via `MISSING_CAPABILITY_FLAG`).
- **CLI harden:** `verify` / `feedback` accept UTF-8 BOM JSON; missing/invalid inputs return exit 2 with a clear stderr message (no traceback); `prismshine verify --demo` uses a **packaged** sample (works after pip/git install); README no longer points at a fake `path/to/bundle.json`.
- **Runtime suite:** `bench/runtime/` — ChorusGraph+PrismShine vs LangGraph+HHEM / MiniLM / **LettuceDetect** (H1/R1/P1, no Guard). Smoke: `python -m bench.runtime.smoke_local`.
- **PrismGuard pin:** stack / `[guard]` extra → `prismguard>=0.1.8` ([PyPI](https://pypi.org/project/prismguard/0.1.8/)).

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
