# Market readiness snapshot — 2026-07-20 (post v0.2.0 enterprise hardening)

**Public PrismShine headline** (vs HHEM only). Sibling stack suites are out of scope here.

Saved scoreboard: [`scoreboard.md`](scoreboard.md) · [`summary.json`](summary.json)

## Are the results good?

**Yes, for the lane we measured.** Against Vectara HHEM-2.1-Open on the same ACI contract:

| Claim | Result |
|---|---|
| Beat / match encoder SotA on QA (B1) | **Win** — F1 0.831 vs 0.746 |
| Hard-number floor (B2) | **Win** — F1 1.0, **0 FP** (gate pass) |
| Faster default path | **Win** — B1 p50 90 ms vs 216 ms |
| Zero LLM spend on default | **Win** — 0 calls |
| Summarization (Bsum) | **Win vs HHEM** but absolute F1 0.60 is only okay |

## Enterprise readiness after v0.2.0

| Gap | Status |
|---|---|
| Handoff FIX-1…14 (decision bugs / hardening) | **Done** + `tests/fixes/` |
| Real calibration overlay (not identity placeholder) | **Done** — `cal-halueval-hash-0.1` (MiniLM path available; hash is CI-safe) |
| ONNX ensure / export for pip installs | **Done** — `python -m prismshine.tools.ensure_span_onnx` |
| Wiring demo (halt + verify + consistency) | **Done** — `examples/enterprise_wiring_demo.py` |
| Tier-4 judge example | **Done** — `examples/tier4_judge_demo.py` |
| Multi-run median comparative runner | **Done** — `run_bench.py --runs 3` |
| Cause / consistency / latency in-process receipts | Regenerate under `benchmarks/reports/` |
| Fair RAGAS/GPU judge row on ACI | **Still open** (cost/time) |
| PyPI publish + signed release | **Process** (code ready at 0.2.0) |
| 3-run median ACI re-bench with calibrated+ONNX image | **Still open** (start ACI → bench → stop) |

## Can we compete in the market now?

| Mode | Ready? |
|---|---|
| Open-source / self-hosted fast grounding checker | **Yes** |
| Compete with encoder tools (HHEM-class) | **Yes** on HaluEval receipt |
| Compete with LLM-judge stacks on accuracy | **Not proven** until GPU/API judge row |
| Enterprise GA as category-creator (wired moat) | **Code-ready**; need customer wiring + published `benchmarks/reports/*` + optional 3-run ACI |

### One-line answer

**Ship 0.2.0 as enterprise-ready open source** with honest receipts. Re-run ACI `--runs 3` when you want the median marketing table; stop containers after.
