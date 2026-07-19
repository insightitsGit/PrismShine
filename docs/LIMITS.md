# PrismShine — Scope Boundaries & Honest Limits

Read this before marketing, demos, or regulated deployments. Claims without receipts are banned (see `POSITIONING.md`).

## 1. PASS means grounded in the preload — not world-true

`decision="pass"` means: given the evidence in this `EvidenceBundle`, PrismShine did not find a cause-side failure or an ungrounded answer **relative to that preload**.

It does **not** mean:

- the preload facts are correct in the real world
- retrieved documents were not poisoned or outdated
- the model’s answer is safe, legal, or clinically appropriate

If the preload itself is wrong, a faithful answer correctly PASSes. Pair with:

| Concern | Owner |
|---|---|
| Prompt injection / adversarial input | PrismGuard |
| Source integrity / watermarks | VectorBridge / CHORUS |
| World-knowledge backstop | Tier-4 judge (opt-in) |

## 2. Streaming answers are not verified mid-token

v0 ships **buffered mode** only: verify the completed answer, then display (or replace on `block`).

A streamed hallucination can already be on screen before `verify()` runs. Sentence-by-sentence streaming verification is post-v0 and needs runtime transport support.

`ShineGate.build(..., buffered_display=True)` (default) records this in `capabilities()`. Do not demo PrismShine as live streaming safety.

## 3. The moat requires runtime wiring

| Install | What you get |
|---|---|
| `pip install prismshine` + plain dicts | Strong grounding checker (Tiers 1–3); forensics only on the `trace` you supply |
| + rich `trace` / cache / memory in the bundle | Full Tier-0 handbook |
| + ChorusGraph interceptors + `shine_node` | Pre-generation halt, ledger write-back, regenerate loop |
| + PrismCortex + PrismCache hooks | Consistency contract (invalidate + detect stale reuse) |

Without ledger/cache/memory evidence, you have **not** bought the category-defining product — you have a competitive grounding verifier. Adapters that omit `source="history"|"memory"` chunks will false-positive on correctly conversational answers.

### Minimum wiring checklist (ChorusGraph)

1. Prefer `ctx.call_llm` so `register_interceptor(before_llm=, after_llm=)` can halt pre-generation.
2. Always attach `shine_node` as the guaranteed post-generation path.
3. Populate docs **and** history/memory into preload (adapter does this when state keys exist).
4. Wire `bind_memory_invalidation` / `on_fact_corrected` for the consistency contract.
5. Keep `CACHE_PREDATES_FACT_UPDATE` enabled even if prevention hooks fail (dual-rail).
6. Call `require_shine(compiled, gate)` at compile time — fails fast if Shine is not wired.
7. Map LLM provider failures into `TraceStep(kind="llm", status=error|empty|timeout)` (or pass `llm_error=` to `shine_after_hook`).
8. Set `consumes=` / `expect_trace_kinds` so `TRACE_INCOMPLETE` fires instead of silent dormancy.
9. For multi-hop graphs set `node_state.answer_source_hop` to avoid `PARALLEL_PRELOAD_AMBIGUITY`.

### RuntimeAdapter (any orchestrator)

Implement `extract_bundle` / `enforce` / `pre_llm_hook` / `post_llm_hook` (`prismshine.runtime.RuntimeAdapter`).
Shipped: `ChorusGraphAdapter`, `LangGraphAdapter`, `DictStateAdapter` (`make_dict_adapter`).
Conformance: `tests/test_runtime_conformance.py`. BYO proof (no ChorusGraph): `tests/test_byo_runtime.py`.

Generic wiring (preferred for LangGraph / custom stacks): `prismshine.wiring` —
`wrap_llm`, `pre_llm_check` / `post_llm_check`, `shine_verify_node`, `record_*` trace helpers,
`require_shine_wiring`, `ShineDecision`. Feature parity table: `docs/INTEGRATION.md` §8.

## 4. Thresholds and competitive claims need receipts

The default matrix in DESIGN §5.5 ships as **v0 proposals**. Use `prismshine calibrate` (synthetic or labeled) and keep the overlay’s `calibration_version` in the verdict cache key. Public accuracy / moat / cost claims require green receipts from `prismshine bench` (`docs/BENCHMARKS.md`) and the gates in `POSITIONING.md`.

## 5. Contradiction and structured-output residual risk

- Cue-less contradictions still need Tier 3/4; clinical/finance profiles force escalation on contradiction cues.
- JSON/table answers use field-level Tier-1 copy-check; sentence coverage alone is insufficient for structured outputs.
- English-centric encoder at v0; multilingual is post-v0.

## 6. Tier-3 backend honesty

`gate.capabilities()` reports `span_backend`: `onnx` | `lexical` | `unavailable`.

- `onnx` — LettuceDetect-class model loaded
- `lexical` — deterministic token-absence detector (degraded; gray unresolved still cannot PASS)
- `unavailable` — `[spans]` missing; gray → flag (ADR-11)
