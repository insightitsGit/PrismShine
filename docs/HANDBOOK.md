# PrismShine Handbook — Failure-Signature Taxonomy

The Handbook is PrismShine's "internal dictionary": a versioned, declarative catalog of deterministic failure signatures that the Tier-0 forensics engine evaluates against every `EvidenceBundle`. It is the cause-side counterpart to PrismGuard's structural rulebook.

Design intent: **a hallucination precursor should be a named, versioned, testable rule — not tribal knowledge.** When a verdict says `resolution_gate: HANDBOOK:EMPTY_RETRIEVAL`, an auditor can open the handbook and read exactly what fired and why.

---

## 1. Signature schema

Handbook files are YAML, merged in order (builtin → domain pack → tenant overrides). Schema (`prismshine.handbook.schema`):

```yaml
handbook_version: "0.1.0"
signatures:
  - id: EMPTY_RETRIEVAL              # stable, SCREAMING_SNAKE, never reused
    title: "Retrieval hop returned zero chunks"
    severity: fatal                  # fatal | error | warning | info
    scope: preload                   # preload | answer | run
    detector: retrieval.empty        # dotted ref to a built-in detector function
    params:                          # detector-specific parameters
      min_chunks: 1
      applies_to_sections: must_ground   # from bundle.declared_sections
    signal_value: 1.0                # contribution when it fires (0..1)
    advice: "Retrieval hop '{hop}' returned {n} chunks for a must-ground section. Halt or repair retrieval before generation."
    references: []                   # optional docs/ticket links
```

Rules:

- **Detectors are code, signatures are data.** A detector (e.g. `retrieval.empty`) is a pure function `detect(bundle, params) -> list[SignatureHit]` implemented in `prismshine.forensics.detectors`. The handbook binds detectors to ids, params, severities, and advice text. New thresholds/domain packs need no code change; genuinely new detection logic does.
- `fatal` severity + gate policy `halt_on_fatal` short-circuits the pipeline (§3 of DESIGN).
- Every hit carries an **evidence pointer** (trace step index / state key / chunk id) so audits can jump from verdict to raw evidence.
- Handbook version participates in the verdict cache key — bumping the handbook invalidates cached verdicts.

## 2. Initial catalog (v0.1 core handbook)

### Retrieval family

| id | severity | fires when |
|---|---|---|
| `EMPTY_RETRIEVAL` | fatal | retrieval-kind trace step with `status=empty` or 0 chunks feeding a must-ground section |
| `LOW_RELEVANCE_RETRIEVAL` | error | all retrieval scores below profile floor (e.g. max `constructive_score` < 0.55) — chunks exist but none are on-topic |
| `RETRIEVAL_ERROR` | fatal | retrieval-kind step `status in {error, timeout}` |
| `CATEGORY_MISMATCH` | warning | prismrag `rule_chain` category of retrieved chunks disagrees with query-inferred category |
| `PARTIAL_RETRIEVAL` | warning | fewer chunks than `top_k` requested AND below `min_chunks_expected` |

### Tool / API family

| id | severity | fires when |
|---|---|---|
| `TOOL_ERROR_SWALLOWED` | fatal | tool-kind step `status=error` but downstream state contains no error marker and generation proceeded |
| `TOOL_EMPTY_RESULT` | error | tool step ok but produced empty/null payload consumed by a must-ground section |
| `TOOL_TIMEOUT` | error | tool step `status=timeout` |
| `TOOL_SCHEMA_DRIFT` | warning | tool payload missing keys the consuming node read (state key accessed but absent/None) |

### Context assembly family

| id | severity | fires when |
|---|---|---|
| `CONTEXT_TRUNCATED` | error | assembled preload tokens > budget and tail was cut (from `context_budget`) |
| `MISSING_STATE_KEY` | error | generator prompt template referenced a state key that was empty/absent at generation time |
| `PRELOAD_DUPLICATION` | info | near-duplicate chunks consumed budget (wasted context; indirect hallucination risk) |
| `LOW_FIDELITY_SPACE` | info | coverage had to run in 64-d JL space for some chunks (lossy; stricter threshold applied) |
| `ENCODER_VERSION_MISMATCH` | error | preload chunk vectors and answer-sentence vectors were produced by different encoder model artifacts (ids carried in `vector_space` metadata) — cosine across mismatched spaces is meaningless; affected chunks are re-encoded once (write-back) or coverage is degraded with this signal |

### Cache family

| id | severity | fires when |
|---|---|---|
| `STALE_CACHE_REUSE` | error | cache decision `HIT_REUSE` whose entry predates a newer version of its source partition (warm-index version mismatch) |
| `CACHE_PREDATES_FACT_UPDATE` | error | a cache hit (`HIT_REUSE`/`HIT_AS_CONTEXT`) whose entry was created **before** the latest correction/`ACCOMMODATE` event for a subject related to the cached query — the cached answer was generated from a fact that has since changed (e.g. answer cached while "Person A is my brother" was current, served after the correction to "sister"). Detection: entry `created_at` vs the `valid_from` of correction events; relatedness via embedding similarity between the corrected fact's text and the cached query vector (both vectors already exist — zero extra embedding cost), plus subject tags in the entry sidecar when available |
| `MARGINAL_CACHE_HIT` | warning | `HIT_REUSE` with verify score within ε of threshold (e.g. < threshold + 0.01) |
| `CACHE_CONTEXT_MISUSE` | warning | `HIT_AS_CONTEXT` entry contributed the *only* support for a Tier-1 fact |

### Memory family (PrismCortex / user facts)

| id | severity | fires when |
|---|---|---|
| `MEMORY_CONFLICT_SERVED` | error | recall subgraph contained a fact listed in `Memory.conflicts()` |
| `STAGED_FACT_SERVED` | warning | answer support traces to a staged (unconsolidated) fact |
| `EXPIRED_FACT_SERVED` | error | supporting fact's `valid_to` is closed at query time (time-travel misuse) |
| `CONFLICTING_PRELOAD_FACTS` | error | two preload items (chat-history or memory chunks) assert incompatible values for the same subject + relation — e.g. "Person A is my brother" and "Person A is my sister" both in the preload. An answer citing *either* would look grounded to copy-check and coverage, so the conflict must be flagged at the preload level, pre-generation. Detection (v0, honest scope): (a) PrismCortex `conflicts()` when Cortex is in the loop; (b) without Cortex, an **exclusive-relation lexicon** — relation families where values are mutually exclusive (kinship terms, marital status, alive/deceased, employed-at, binary approvals) checked over subject-matched preload sentences. Open-domain conflict extraction is explicitly NOT claimed. |

**Recommended policy for `CONFLICTING_PRELOAD_FACTS` / `MEMORY_CONFLICT_SERVED`:** verdict `flag` (or `block` in strict profiles) with advice instructing the agent to ask the user a clarifying question naming both conflicting values and their sources — resolution by asking beats resolution by guessing, and it costs no LLM verification call. Once the user answers, the correction flows through PrismCortex `ACCOMMODATE` (old fact closed with `valid_to`, new fact inserted, history preserved) and the conflict disappears from future preloads.

### Guard / run family

| id | severity | fires when |
|---|---|---|
| `GUARD_GRAY_INPUT` | warning | PrismGuard passed the input in the gray zone — heightens output scrutiny (raises coverage thresholds for this run) |
| `HOP_BUDGET_EXHAUSTED` | warning | run hit recursion/hop limit before completing planned retrieval hops |
| `ANTI_THRASH_TRIGGERED` | info | ReAct stop-on-repeated-action fired — the agent may have answered without the data it was looping for |

### LLM hop family

| id | severity | fires when |
|---|---|---|
| `LLM_ERROR` | fatal | llm-kind step `status in {error, timeout}` (provider/auth/rate-limit mapped into the ledger) |
| `LLM_EMPTY_COMPLETION` | error | llm step empty / zero tokens / blank answer after an llm hop |
| `LLM_REFUSAL` | warning | finish_reason / refusal / content_filter on an llm hop |

### Trace completeness / cache-skip / parallel attribution

| id | severity | fires when |
|---|---|---|
| `TRACE_INCOMPLETE` | error | `consumes` / `expect_trace_kinds` declared but required ledger kinds missing (bans silent dormancy) |
| `RETRIEVAL_SKIPPED_AFTER_CACHE_MISS` | error | cache `MISS` with no subsequent retrieval before an llm hop |
| `HIT_REVALIDATE_IGNORED` | error | entry marked `must_revalidate` but decision was still `HIT_REUSE` |
| `PARALLEL_PRELOAD_AMBIGUITY` | warning | multiple retrieval hops without `node_state.answer_source_hop` |

## 3. Domain packs (licensed tier, schema identical)

- `clinical.yaml` — stricter floors (coverage τ aligned with ChorusGraph's 0.97 clinical verify), dosage/unit copy-check mandatory, `STAGED_FACT_SERVED` promoted to error.
- `finance.yaml` — numeric tolerance zero, currency normalization mandatory, `MARGINAL_CACHE_HIT` promoted to error during volatility phase (PrismResonance ALERT).
- `legal.yaml` — citation-format facts (case ids, statute refs) as a first-class copy-check type; `CATEGORY_MISMATCH` promoted to error.

## 4. Authoring & testing rules

1. Every signature ships with at least two fixture bundles: one that fires, one adjacent case that must NOT fire (false-positive guard).
2. `advice` strings must be actionable and reference concrete evidence fields (`{hop}`, `{n}`, `{key}`), never generic ("something went wrong" is banned).
3. Severity promotions/demotions live in domain packs or tenant overrides — never fork a signature id to change severity.
4. Ids are permanent: deprecate with `deprecated: true` + `replaced_by`, never delete or reuse.
