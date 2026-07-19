# Handoff: prismlang 0.1.1 → 0.1.2 — encoder artifact id + session sharing for PrismShine

> **STATUS: SHIPPED & VERIFIED (Jul 18, 2026)** — `model_id()` + `get_session()` implemented, exported, tests green (42 passed), on PyPI. Verification record in `docs/UPSTREAM.md`.

**Repo:** `C:\code\PrismLang` (github.com/insightitsGit/prismlang)
**Requested by:** PrismShine (design at `C:\code\PrismShine\docs\` — see `DESIGN.md` §6 and the `ENCODER_VERSION_MISMATCH` signature in `HANDBOOK.md`)
**Priority:** LOW — small patch; PrismShine can hash the model file as a fallback. Note: `model_id()` is also consumed by ChorusGraph 1.3.0 item 7 (`handoff-chorusgraph.md` — stamping encoder ids on warm-index entries) and participates in PrismShine's verdict cache key, so if you do both handoffs, land this one first.

## Context (why)

PrismShine reuses prismlang's ONNX MiniLM encoder session to encode answer sentences (its Tier-2 verification), and compares those vectors against preload chunk vectors produced earlier. If the encoder model artifact changed between when chunks were embedded and when the answer is verified, cosine across the two vector sets is meaningless. PrismShine detects this via model artifact ids — prismlang should expose one authoritatively.

## Requested changes

### 1. Encoder artifact id  (required)

`prismlang.encoder.model_id() -> str` — stable identifier of the loaded model artifact (suggested: `"{hf_repo}@{revision}:{sha256(model.onnx)[:12]}"`). Computed once at session init, cached.

### 2. Shared session accessor  (required)

Public, documented way to obtain the initialized encoder session/singleton (e.g. `prismlang.encoder.get_session()` or a documented guarantee that module-level `encode`/`encode_batch` share one process-wide session). Goal: a host process (ChorusGraph + PrismShine) provably runs ONE ONNX session, and PrismShine can attach without triggering a second model load.

### 3. Expose both in `__init__` exports  (required)

Add to the public API surface + README (with import-tested examples).

## Constraints

- No breaking changes; no new dependencies.
- Tests: `model_id()` stability across calls, changes when model file changes (fixture), session identity (two encode paths, one session). Repo has 34 tests / coverage gate 80 — keep the gate green.
- Version bump to 0.1.2.

## Report back (for verification when you return)

- Exact signatures added
- Test count before/after and pass status
- Any deviation from spec and why
