# Handoff — PrismGuard docs: scorecard / learn-from-seed features invisible to integrators

**To:** PrismGuard maintainers  
**From:** PrismShine (`insight-stack` package-vs-package suite)  
**Date:** 2026-07-20  
**Related:** `handoffs/handoff-prismguard-dx.md` (factory vs scorecard labeling)  
**Severity:** High — docs caused a real under-configured production-shaped integration

## Problem

PrismShine built an “Insight stack” shim meant to showcase **PrismGuard → ChorusGraph-shaped Shine**. Integrators (us) followed the README / quick-start surface and shipped a path that looked intentional but **omitted the features Guard’s own law scorecard and “learn from our words / DB” story depend on**.

That is a documentation hierarchy bug, not a library bug. The APIs exist; the docs do not make the full path discoverable or mandatory for scorecard-like claims.

## What we shipped first (wrong relative to Guard’s full path)

| Layer | What we used | What we missed |
|---|---|---|
| Factory | `create_checker_rules_only` / later `law_pilot`+ONNX | Still incomplete vs CPL / learn path |
| Extras | `prismguard[guard-model]` only | **`[prism]`** (prismrag taxonomy / word-graph on seed) |
| Seed | Assumed “factory seeds” | Never documented that **taxonomy is skipped** without `[prism]` / when `force_hash_embedder` |
| Feedback | unset | **`PRISMGUARD_FEEDBACK_PERSIST=1`** (queue → export → train) |
| Storage | default memory | How to attach **pgvector/chroma** (`PRISMGUARD_STORAGE_*`) for persistent seed/feedback (Team+) |
| Tenant words | unset | **`PRISMGUARD_TENANT_LEXICON_PATH`** + extra seed YAML import |
| ChorusGraph | not wired | **`make_guard_handler` / `route_after_guard`** as the documented graph entry (lives in Guard repo) |
| Scorecard alias | soft `law_pilot` | `security_bench` vs `law_pilot` tradeoff (**`security_bench` forces HashEmbedder / skip_taxonomy**) |

**Evidence:** Stack ACI run1 S1 F1 **0.33** on rules_only; run3 S1 F1 **0.75** after ONNX+law_pilot — still without `[prism]` / feedback / ChorusGraph node helpers. Latency looked “wrong” vs CPL because docs never explained selective escalation vs `classifier_mode: first` + always-on ONNX.

## What “full OSS Guard” actually requires (verified against source)

From `prismguard.runtime.factory` + `docs/integration-guide.md` + CPL `PrismGuardGate`:

1. **Profile:** `law_pilot` + `PRISMGUARD_USE_ONNX=1` + `prismguard-model download` for law scorecard-class injection.  
   Prefer **`law_pilot` over `security_bench` when seed taxonomy matters** — `security_bench` sets `force_hash_embedder=True` → `skip_taxonomy=True`.
2. **Extras:** `pip install "prismguard[guard-model,prism]"` so `has_prismrag()` is true and seed import builds taxonomy.
3. **Domain words:** `PRISMGUARD_DOMAIN=law` (or `law_pilot`) imports `domains/law/overlay.yaml`.
4. **Learn-from-traffic:** `PRISMGUARD_FEEDBACK_PERSIST=1` → `FeedbackReviewService` → `prismguard feedback export` → `prismguard-model train`.
5. **Learn-from-DB (Team+):** `PRISMGUARD_STORAGE_BACKEND=pgvector` + `PRISMGUARD_STORAGE_DSN` + license — **must be labeled Team+** next to any “learns from your DB” marketing.
6. **Tenant lexicon:** `PRISMGUARD_TENANT_LEXICON_PATH` for customer entity words / severity boosts.
7. **ChorusGraph:** `prismguard.integrations.chorusgraph.make_guard_handler` + `route_after_guard` before cache/RAG hops (example exists but is easy to miss; README still pushes `web_chat`).

## Ask (docs / DX — concrete)

### A. Above-the-fold README: three rows, not two

| Goal | Install | Call |
|---|---|---|
| Hub / FAQ, low FP | `pip install prismguard` | `create_checker_for_app("web_chat")` |
| Match published **law injection scorecard** | `pip install "prismguard[guard-model]"` + `prismguard-model download` + `PRISMGUARD_USE_ONNX=1` | `create_checker_for_app("law_pilot", use_onnx=True)` **or** `security_bench` |
| **Full learn-from-seed / word-graph / feedback** (what marketing implies) | `pip install "prismguard[guard-model,prism]"` + same ONNX steps + `PRISMGUARD_FEEDBACK_PERSIST=1` | `create_checker_for_app("law_pilot", use_onnx=True)` — **not** `security_bench` if you need taxonomy |

Call out explicitly:

> Without `[prism]`, seed overlay text is stored but **taxonomy/word-graph is skipped**. Do not claim “learns from your corpus” on that path.

### B. Scorecard / COMPARISON_REPORT page

- Banner: **“Do not expect these rates from `web_chat` / `rules_only` / `[guard-model]` alone.”**
- Checklist of env + extras used for the cited CPL row (domain, seed profile, ONNX artifact, feedback, storage backend, corpus_path).
- One sentence on **latency**: blended CPL ~200 ms assumes selective escalation; `classifier_mode: first` + ONNX on nearly every request will look ~350 ms — that is expected, not a broken install.

### C. “Learn from DB / our words” section (missing today as a single recipe)

Document the closed loop in one place:

```text
seed YAML / domain overlay  →  storage (memory or Team+ DB)
tenant lexicon (optional)   →  severity / force-classifier
feedback persist            →  export JSONL → corpus-plan → train → artifact → PRISMGUARD_GUARD_MODEL_PATH
```

Mark which steps are OSS vs Team+/Business.

### D. ChorusGraph section: scorecard path, not only hub fail-open

Current sketch uses `web_chat` + gray-continues. Add a second snippet:

```python
from prismguard.integrations.chorusgraph import (
    create_checker_for_app, make_guard_handler, route_after_guard,
)
checker = create_checker_for_app("law_pilot", use_onnx=True)
guard = make_guard_handler(checker, block_on=frozenset({"block", "gray"}))
# START → guard → [end | retrieve…]  BEFORE cache hops
```

### E. Factory docstring / `security_bench` warning

In README and `create_checker_for_app` docs: **`security_bench` fails loud on missing ONNX but disables transformer taxonomy.** Integrators matching “learn from seed words” should use `law_pilot` + opt-in ONNX + `[prism]`.

### F. Health / readiness helper (optional code, big DX win)

A one-liner or CLI that prints capability truth:

`onnx_ready`, `prismrag_taxonomy`, `feedback_persist`, `storage_backend`, `domain_overlay`, `tenant_lexicon`

So integrators cannot ship a silent half-stack and claim scorecard parity.

## Acceptance

- [ ] New integrator reading only README first screen can distinguish hub vs scorecard vs **learn-from-seed** installs.
- [ ] Scorecard page lists the exact extras/env of the cited run.
- [ ] “Learns from your DB/words” never appears without Team+ storage and/or seed+`[prism]`+feedback recipe.
- [ ] ChorusGraph example includes a law/security wiring, not only `web_chat`.
- [ ] No change required to scorecard methodology — labeling and discoverability only (unless you choose to add the readiness helper).

## PrismShine remediation (in progress)

- Handoff DX-1 already covered factory labeling; Guard handback added `security_bench`.
- This handoff covers the **second** miss: `[prism]`, feedback, storage, lexicon, ChorusGraph helpers.
- `insight-stack` is being rebuilt to:
  - `prismguard[guard-model,prism]` + feedback persist + law overlay
  - `make_guard_handler` + Shine `require_shine` / `shine_node` on a real ChorusGraph
  - `GET /health` → `guard_caps` / `chorus_caps` so missing features are visible

## Contact

PrismShine repo: `C:\code\PrismShine` — stack shim `bench/shims/insight-stack/`, suite docs `bench/stack/README.md`.
