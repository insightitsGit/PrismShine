# Handoff Fix1 — PrismShine v0.1.x bug-fix & hardening work order

**Scope:** fixes only — no new features. Found by a full code review at commit `e87c224`
(all 134 tests green; these are latent bugs the suite does not yet cover).
Every item lists file/line, evidence, the fix, and an acceptance test. Work top-down: P0 → P3.

Reference baseline: `git log` → `e87c224` "Enhance documentation and integration features".

---

## P0 — decision-affecting bugs (fix before any benchmark run)

### FIX-1: user threshold overrides silently dropped at verify time
- **Where:** `prismshine/gate.py` (~line 298, the `resolve_policy(...)` call inside `verify()`).
- **Evidence (reproduced):**
  ```python
  g = ShineGate.build(embedder=emb, overrides={"tau_sent": 0.99, "numeric_tolerance": 0.5})
  g.policy.tau_sent            # 0.99  (build-time: correct)
  # inside verify(): resolve_policy(profile=..., overrides=self.policy.extras or None)
  # -> tau_sent 0.62, numeric_tolerance 0.005  (defaults; user overrides GONE)
  ```
  `resolve_policy` applies known overrides as **attributes** on the policy and only puts
  *unknown* keys into `extras`. `verify()` re-resolves per-request policy (for the dynamic
  strictness bump) passing only `self.policy.extras` — so every recognized override
  (`tau_sent`, `numeric_tolerance`, `bands`, `tau_floor`, …) is lost on every single verify.
  Only `weights` survives (copied manually one line later).
- **Fix:** store the original overrides dict on the gate (`self._overrides = overrides or {}`
  in `__init__`/`build`) and pass it to the per-request `resolve_policy(...)` call
  (merged with `extras`). Remove the manual `policy.weights = dict(self.policy.weights)` special case.
- **Acceptance:** new test — `ShineGate.build(overrides={"numeric_tolerance": 0.5})`;
  answer says "$1400", preload says "$1000" (40% off) → must be **matched** (within tolerance),
  no `unmatched_currency` span. Second test: `overrides={"bands": (0.01, 0.02, 0.03)}`
  must produce `block` on a mildly-dirty bundle that passes with defaults.

### FIX-2: Tier-4 escalation budget denominates on the wrong population
- **Where:** `prismshine/grounding/judge.py` (`EscalationBudget.allow`) + `gate.py` (~line 533).
- **Evidence (reproduced):** `allow()` is only called when a judge is *needed*, so `_total`
  counts judge-needed calls, not traffic. Pattern over 30 gray-band verifies with budget=0.10:
  `YYYYYYYYYYnnnnnnnnnnnnnnnnnnnn` — first 10 allowed, then ~everything denied until the
  ratio decays. Design target (POSITIONING: "judge escalation ≤ 10% **of traffic**")
  is silently converted to "≤ ~10% of gray traffic" and behaves as a burst-then-starve pattern.
- **Fix:** add `EscalationBudget.observe()` incrementing `_total`; call it once at the top of
  every `ShineGate.verify()` (cache hits excluded or included — pick one and document).
  `allow()` then only checks `escalated/total` and increments `_escalated`.
  Keep thread-safety. Consider a sliding window (deque of last N) so a burst of dirty
  traffic doesn't starve judges for the rest of the process lifetime.
- **Acceptance:** simulate 100 verifies where 20 are gray: exactly ≤ 10 judge calls, and
  the *last* gray verify can still escalate if the rate allows (no permanent starvation).

### FIX-3: warm-index vectors silently dropped when records hold numpy arrays
- **Where:** `prismshine/evidence/adapters/chorusgraph.py` line ~113:
  ```python
  vec = getattr(rec, "vector_384", None) or getattr(rec, "vector", None)
  ```
- **Evidence:** `ChunkVectorRecord.vector_384` from ChorusGraph 1.3.0 is a numpy array.
  `bool(ndarray)` with >1 element raises `ValueError: truth value ... is ambiguous`;
  the surrounding blanket `except Exception: pass` then discards **all** warm-index vectors
  for the run. Coverage silently degrades to re-encode (extra latency) or lexical —
  exactly the zero-re-embed win the design promises, lost without a trace.
- **Fix:**
  ```python
  vec = getattr(rec, "vector_384", None)
  if vec is None:
      vec = getattr(rec, "vector", None)
  ```
  Also narrow the blanket `except Exception` around the whole injection loop to per-record
  handling with a `logger.debug`, so one bad record doesn't discard the rest.
- **Acceptance:** unit test with a stub record whose `vector_384` is `np.ones(384)` —
  bundle chunk must carry the vector and `vector_space == "raw-384@<artifact>"`.

---

## P1 — correctness / robustness

### FIX-4: Tier-0 cache never saves work and grows without bound
- **Where:** `gate.py` `verify()` (~lines 293–329) and `self._tier0_cache`.
- **Evidence:** `pre_forensics = run_forensics(bundle, self.handbook)` runs unconditionally
  *before* the `_tier0_cache` lookup — the cache changes nothing except pinning old results.
  `_tier0_cache` is a plain dict with no eviction: long-running services leak one
  `ForensicsResult` per unique preload for the process lifetime.
- **Fix:** compute `preload_key` first; on hit, reuse the cached result for the dynamic-bump
  check too (it reads `GUARD_GRAY_INPUT` from the same result). Make it an LRU
  (`OrderedDict`, maxsize ≈ 256, same pattern as `MemoryVerdictStore`).
- **Acceptance:** counter-instrumented test proving `run_forensics` executes once for two
  verifies over the same preload/trace with different answers; cache length bounded.

### FIX-5: pinned-ONNX partial load leaves the SpanClassifier half-initialized
- **Where:** `prismshine/grounding/spans.py`, pinned-artifact block in `_ensure_loaded()`.
- **Evidence:** the pinned path assigns `self._session = ort.InferenceSession(...)` *before*
  checking `tokenizer.json` exists. If the tokenizer is missing: this call falls through to
  hub download (fine online), but in the exact environment pinning targets (air-gapped CI)
  the hub fails, and if `allow_lexical_fallback=False` the method returns False while
  `self._session` stays set → the **next** `_ensure_loaded()` short-circuits `True` at the
  top with `self._tokenizer is None` → `classify()` crashes with `AttributeError`.
- **Fix:** load session + tokenizer into locals; only assign `self._session/_tokenizer/
  _backend/artifact_id` when both succeeded.
- **Acceptance:** test with `PRISMSHINE_SPAN_ONNX` pointing at a real dummy .onnx and no
  tokenizer.json beside it, `allow_lexical_fallback=False` → `available` is False on
  repeated calls, no exception from `classify()`.

### FIX-6: judge presence discards deterministic Tier-2/3 evidence in fusion
- **Where:** `prismshine/fusion.py` lines ~96–105.
- **Evidence:** when `judge_present`, the `else` branch skips t2 coverage, contradiction-cue
  and t3 span contributions entirely — a judge returning `risk=0.0` can wash a strong
  deterministic cue out of the fused score. The post-fusion `has_cues and decision=="pass"`
  guard in `gate.py` catches the *decision* on forced-judge profiles, but `fused_score`,
  band, gate name and calibration data all misreport.
- **Fix (decide + document as an ADR note):** judge replaces only the *gray residual* —
  e.g. `contrib += w["t4"] * judge + w["contradiction"] * cue` (cue always counted), or
  `fused = max(fused_deterministic, fused_with_judge)`. Deterministic evidence must be
  monotone: adding a judge may raise, never lower, the risk contribution of a confirmed cue.
- **Acceptance:** bundle with a negation-asymmetry cue + judge stub returning 0.0 →
  fused score ≥ the no-judge fused score; decision not `pass` on `finance`/`clinical`.

### FIX-7: opposite-pair contradiction matching has no word boundaries
- **Where:** `prismshine/grounding/contradiction.py` (~lines 84–87), pairs like `pass/fail`.
- **Evidence:** plain `in` substring checks — "the **pass**enger list" vs a chunk containing
  "**fail**ure analysis" produces `opposite:pass/fail`, a false contradiction cue that forces
  Tier-3 and can flag a clean answer (and, on finance/clinical, force a judge call = cost).
- **Fix:** precompile `\b{word}\b` regexes for each pair member; also delete the dead,
  unused `_opposite_hit()` helper.
- **Acceptance:** "The passenger completed the trip." vs chunk "failure rates were low" →
  no cue; "Revenue increased" vs "revenue decreased in Q2" → cue still fires.

### FIX-8: id()-keyed wiring registries leak and can false-positive
- **Where:** `_SHINE_ATTACHED` in `prismshine/integrations/chorusgraph.py` and `_WIRED`
  in `prismshine/wiring.py`; also `shine_node`'s `_mark_attached(compiled or state, node=True)`.
- **Evidence:** (a) marking is done **per request-state dict** inside `_node` — one registry
  entry per request, never evicted → unbounded growth; (b) CPython recycles `id()` values,
  so a dead graph's id can match a brand-new unwired graph → `is_shine_wired()` returns a
  false True and `require_shine` skips wiring — the exact failure the P0 wiring check exists
  to prevent.
- **Fix:** prefer the `_prismshine_attached` attribute as the source of truth; keep the dict
  only as a fallback for objects rejecting attributes, implemented as a
  `weakref.WeakKeyDictionary` (or an LRU of bounded size for unhashable/dict targets).
  Remove the per-request `_mark_attached(... or state ...)` call from `_node` entirely —
  wiring is a graph property, not a state property.
- **Acceptance:** loop 10k node calls → registry size stays O(1); test that a fresh object
  allocated after another is GC'd never reports wired.

### FIX-9: unbounded in-process caches (encoder memo, judge cache)
- **Where:** `SharedEncoder._memo` (`encoder.py`), `CachedJudge._cache` (`judge.py`).
- **Evidence:** plain dicts, no eviction; sentence-level memo grows with every unique
  sentence in every answer — real leak on long-running gateways. Additionally, if the
  prismlang encode fails mid-run the mode flips to lexical while `_memo` still holds
  raw-384 vectors — mixed-space cache entries.
- **Fix:** LRU with maxsize (memo ≈ 10k entries, judge ≈ 1k); clear `_memo` on mode flip.
- **Acceptance:** encode 20k unique sentences → memo length ≤ maxsize; mode-flip test
  asserts memo cleared.

---

## P2 — polish (safe, small)

### FIX-10: fusion dead code
`fusion.py` line ~115: `decision = "flag" if gray_unresolved else "flag"` — both branches
identical; keep `"flag"` and drop the conditional. Line ~50: `"REGENERATE" in early_gate or
early_gate.endswith(":REGENERATE")` — second clause is unreachable; keep one.

### FIX-11: arithmetic closure accepts dimensionally-meaningless combos
`copycheck.py` `_arithmetic_closure`: candidates include `a*b` and `a/b` for **currency**
values ($1000 × $1200 = 1,200,000 "derivable") — widens the accidental-legitimization
surface for fabricated figures. Restrict currency/percent facts to `+`, `-`, percent-change;
keep `*`, `/` for unitless numbers only. Also: `fact in unmatched` (line ~344) relies on
dataclass value-equality — duplicate facts smear match state; compare by identity
(`id()` set) or index. Delete the dead `if lexicon ...: pass` branch in `extract_facts` (~line 131).

### FIX-12: ONNX span classifier ignores model max sequence length
`spans.py` `_onnx_unsupported`: `context[:6000]` chars + answer can exceed the model's
token limit → runtime exception propagates out of `classify()` (only *load* is guarded).
Truncate by tokens (keep answer intact, trim context head), and wrap `classify()` in
try/except that degrades to the lexical backend with a logged warning.

### FIX-13: judge JSON robustness
`judge.py`: both judges `json.loads` the raw completion — models routinely wrap JSON in
markdown fences → silent fallback to `risk=0.5`. Strip fences before parsing; for OpenAI
pass `response_format={"type": "json_object"}`.

### FIX-14: misc
- `audit.py` `metrics()` reads counters without the lock — take `self._lock`.
- `SqliteVerdictStore`: no TTL/size cap — add optional `max_rows` pruning (delete oldest by `created_at`).
- Duplicate `on_fact_corrected` in `wiring.py` and `integrations/chorusgraph.py` —
  keep the wiring one, re-export from the integration.
- `encoder.py` `ensure_chunk_vectors`: user embedders are labeled `raw-384@user-embedder`
  regardless of true dimension — label as `user@{dim}d` so jl/raw heuristics don't misfire.
- Remove `_review_diff.txt` from the repo if it resurfaces (scratch file, already untracked).

---

## Explicitly verified OK (no action)
- Pre-gen halt fallback bug (empty advice → `[]`) — fixed in `e87c224`.
- Verdict cache key now includes `trace_hash`/`state_hash` — cause-side collision closed.
- Hard-fact decision floor + regression tests — in place.
- `enforce_verdict` regen bounding + degrade-to-flag — correct in all three paths
  (chorusgraph node, langgraph node, wiring node).
- Action matrix (`tests/test_action_matrix.py`, 13 scenarios) — all green at HEAD.

## Acceptance for the whole handoff
1. All existing tests stay green (`python -m pytest tests/ -q`).
2. One new test per FIX item above (subfolder `tests/fixes/` or extend existing files).
3. No public API changes except additive (`EscalationBudget.observe`).
4. `docs/DECISIONS.md` gains a short ADR note for FIX-6 (judge fusion semantics).
