"""Detection → action matrix: every major error case, verified end-to-end.

Each scenario runs through the real gate (and shine_node where the action is
integration-applied) and asserts BOTH what fires and what happens next.
Run `python tests/test_action_matrix.py` directly to print the matrix.
"""

from __future__ import annotations

import hashlib

import numpy as np

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.integrations.chorusgraph import shine_node


def fake_embedder(texts: list[str]) -> np.ndarray:
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % dim
            out[i, h] += 1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


def _gate(**kwargs) -> ShineGate:
    return ShineGate.build(embedder=fake_embedder, **kwargs)


GOOD_TRACE = [
    {
        "hop": "retrieve",
        "kind": "retrieval",
        "status": "ok",
        "scores": {"constructive_score": 0.92},
        "detail": {"n_chunks": 3, "top_k": 3},
    }
]
GOOD_CHUNK = {
    "chunk_id": "c1",
    "text": "Revenue was $1000 in Q1 for Acme Corp.",
    "source": "retrieval",
}


def _bundle(**over):
    base = {
        "run_id": "matrix",
        "question": "What was revenue?",
        "answer": "Revenue was $1000 in Q1 for Acme Corp.",
        "preload": [GOOD_CHUNK],
        "trace": GOOD_TRACE,
        "declared_sections": ["must_ground"],
    }
    base.update(over)
    b, _ = bundle_from_dict(base)
    return b


# ---------------------------------------------------------------------------
# 1. Clean grounded answer → PASS, delivered untouched
# ---------------------------------------------------------------------------
def test_clean_pass():
    v = _gate().verify(_bundle())
    assert v.decision == "pass"
    assert v.resolution_gate == "CLEAN_FAST_PATH"


# ---------------------------------------------------------------------------
# 2. Pre-generation EMPTY_RETRIEVAL → BLOCK before any token is spent
# ---------------------------------------------------------------------------
def test_pregen_empty_retrieval_blocks_before_llm():
    v = _gate().verify(
        _bundle(
            answer=None,
            trace=[
                {
                    "hop": "retrieve",
                    "kind": "retrieval",
                    "status": "empty",
                    "detail": {"n_chunks": 0},
                }
            ],
        )
    )
    assert v.decision == "block"
    assert v.resolution_gate == "HANDBOOK:EMPTY_RETRIEVAL"
    assert v.tier_reached == 0 and v.coverage_mode == "skipped"
    assert any("retrieve" in a for a in v.advice)  # advice names the failing hop


# ---------------------------------------------------------------------------
# 3. Pre-generation fatal with halt_on_fatal=False → REGENERATE (bounded repair)
# ---------------------------------------------------------------------------
def test_pregen_fatal_regenerate_when_not_halting():
    from prismshine.config import ShineConfig

    g = ShineGate.build(embedder=fake_embedder, config=ShineConfig(halt_on_fatal=False))
    v = g.verify(
        _bundle(
            answer=None,
            trace=[
                {
                    "hop": "retrieve",
                    "kind": "retrieval",
                    "status": "empty",
                    "detail": {"n_chunks": 0},
                }
            ],
        )
    )
    assert v.decision == "regenerate"
    assert v.signatures[0].evidence.get("hop") == "retrieve"  # reroute target


# ---------------------------------------------------------------------------
# 4. Post-generation EMPTY_RETRIEVAL → BLOCK; shine_node swaps in fallback answer
# ---------------------------------------------------------------------------
def test_postgen_block_replaces_answer():
    node = shine_node(_gate())
    out = node(
        {
            "question": "What was revenue?",
            "reply": "Revenue was definitely $5 million.",
            "docs": [],
            "ledger_steps": [
                {
                    "hop": "retrieve",
                    "kind": "retrieval",
                    "status": "empty",
                    "detail": {"n_chunks": 0},
                }
            ],
            "declared_sections": ["must_ground"],
        }
    )
    assert out["shine_route"] == "block"
    assert "don't have" in out["reply"]  # honest fallback, never silent empty
    assert out["_ledger_append"]["kind"] == "shine.verdict"
    assert "EMPTY_RETRIEVAL" in out["_ledger_append"]["detail"]["signatures"]


# ---------------------------------------------------------------------------
# 5. RETRIEVAL_ERROR (timeout) → fatal → BLOCK, grounding tiers skipped
# ---------------------------------------------------------------------------
def test_retrieval_timeout_fatal():
    v = _gate().verify(
        _bundle(
            trace=[
                {"hop": "retrieve", "kind": "retrieval", "status": "timeout", "detail": {}}
            ]
        )
    )
    assert v.decision == "block"
    assert v.resolution_gate == "HANDBOOK:RETRIEVAL_ERROR"
    assert v.tier_reached == 0


# ---------------------------------------------------------------------------
# 6. TOOL_ERROR_SWALLOWED → fatal → BLOCK (generation proceeded past a dead tool)
# ---------------------------------------------------------------------------
def test_tool_error_swallowed_fatal():
    v = _gate().verify(
        _bundle(
            trace=GOOD_TRACE
            + [{"hop": "fx_rates", "kind": "tool", "status": "error", "detail": {}}]
        )
    )
    assert v.decision == "block"
    assert v.resolution_gate == "HANDBOOK:TOOL_ERROR_SWALLOWED"


# ---------------------------------------------------------------------------
# 7. LOW_RELEVANCE_RETRIEVAL (error severity) → heavy fusion pressure → not pass
# ---------------------------------------------------------------------------
def test_low_relevance_flags():
    v = _gate().verify(
        _bundle(
            trace=[
                {
                    "hop": "retrieve",
                    "kind": "retrieval",
                    "status": "ok",
                    "scores": {"constructive_score": 0.20},
                    "detail": {"n_chunks": 3, "top_k": 3},
                }
            ]
        )
    )
    assert any(s.id == "LOW_RELEVANCE_RETRIEVAL" for s in v.signatures)
    assert v.decision != "pass"


# ---------------------------------------------------------------------------
# 8. TOOL_EMPTY_RESULT (error) feeding a must-ground answer → not pass
# ---------------------------------------------------------------------------
def test_tool_empty_result_flags():
    v = _gate().verify(
        _bundle(
            trace=GOOD_TRACE
            + [{"hop": "fx_rates", "kind": "tool", "status": "ok", "detail": {"payload": []}}]
        )
    )
    assert any(s.id == "TOOL_EMPTY_RESULT" for s in v.signatures)
    assert v.decision != "pass"


# ---------------------------------------------------------------------------
# 9. STALE_CACHE_REUSE (cache entry predates partition version bump) → not pass
# ---------------------------------------------------------------------------
def test_stale_cache_reuse_flags():
    v = _gate().verify(
        _bundle(
            trace=GOOD_TRACE
            + [
                {
                    "hop": "cache_gate",
                    "kind": "cache",
                    "status": "ok",
                    "detail": {
                        "decision": "HIT_REUSE",
                        "entry_partition_version": 3,
                        "current_partition_version": 5,
                    },
                }
            ]
        )
    )
    assert any(s.id == "STALE_CACHE_REUSE" for s in v.signatures)
    assert v.decision != "pass"


# ---------------------------------------------------------------------------
# 10. CONFLICTING_PRELOAD_FACTS (brother vs sister in history) → clarify, not guess
# ---------------------------------------------------------------------------
def test_conflicting_preload_facts_flags_with_clarify_advice():
    v = _gate().verify(
        _bundle(
            question="Who is Person A?",
            answer="Person A is your brother.",
            preload=[
                GOOD_CHUNK,
                {"chunk_id": "h1", "text": "Person A is my brother.", "source": "history"},
                {"chunk_id": "h2", "text": "Person A is my sister.", "source": "history"},
            ],
        )
    )
    assert any(s.id == "CONFLICTING_PRELOAD_FACTS" for s in v.signatures)
    assert v.decision != "pass"
    hit = next(s for s in v.signatures if s.id == "CONFLICTING_PRELOAD_FACTS")
    assert hit.evidence["value_a"] != hit.evidence["value_b"]


# ---------------------------------------------------------------------------
# 11. Fabricated figure → hard-fact floor → FLAG even with good-looking coverage
# ---------------------------------------------------------------------------
def test_fabricated_number_floored():
    v = _gate().verify(_bundle(answer="Revenue was $9000 in Q1 for Acme Corp."))
    assert v.decision != "pass"
    assert any(sp.reason == "unmatched_currency" for sp in v.spans)


# ---------------------------------------------------------------------------
# 12. Answer ignores a healthy preload → T2_COVERAGE_COLLAPSE
# ---------------------------------------------------------------------------
def test_coverage_collapse_on_healthy_preload():
    v = _gate().verify(
        _bundle(
            answer=(
                "Wombats dig extensive burrow systems with their rodent-like teeth "
                "and powerful claws under moonlight."
            )
        )
    )
    assert v.resolution_gate in {"T2_COVERAGE_COLLAPSE", "T1_UNMATCHED_HARD_FACT"}
    assert v.decision in {"flag", "block"}


# ---------------------------------------------------------------------------
# 13. GUARD_GRAY_INPUT → warning + dynamic strictness bump recorded
# ---------------------------------------------------------------------------
def test_guard_gray_input_bumps_strictness():
    v = _gate().verify(
        _bundle(
            trace=GOOD_TRACE
            + [
                {
                    "hop": "guard",
                    "kind": "guard",
                    "status": "ok",
                    "detail": {"zone": "gray", "decision": "flag"},
                }
            ]
        )
    )
    assert any(s.id == "GUARD_GRAY_INPUT" for s in v.signatures)
    assert v.strictness_effective != "standard"  # bumped one level for this run


if __name__ == "__main__":
    import inspect
    import sys

    mod = sys.modules[__name__]
    tests = [f for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    print(f"{'scenario':52} result")
    print("-" * 62)
    for t in tests:
        try:
            t()
            print(f"{t.__name__:52} OK")
        except AssertionError as exc:
            print(f"{t.__name__:52} FAIL  {exc}")
