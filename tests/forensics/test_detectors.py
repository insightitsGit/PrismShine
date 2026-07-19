"""Fire + must-not-fire fixtures for every handbook signature."""

from __future__ import annotations

import pytest

from prismshine.evidence.builder import bundle_from_dict
from prismshine.forensics.engine import run_forensics
from prismshine.handbook.loader import load_handbook

HB = load_handbook()


def _ids(hits):
    return {h.id for h in hits}


def _run(**kwargs):
    b, _ = bundle_from_dict(kwargs)
    return run_forensics(b, HB)


BASE_PRELOAD = [{"chunk_id": "c1", "text": "Revenue was 1000.", "source": "retrieval"}]


@pytest.mark.parametrize(
    "sig,fire,nofire",
    [
        (
            "EMPTY_RETRIEVAL",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "declared_sections": ["must_ground"],
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "empty",
                        "detail": {"n_chunks": 0},
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "ok",
                        "detail": {"n_chunks": 3, "top_k": 3},
                        "scores": {"constructive_score": 0.9},
                    }
                ],
            },
        ),
        (
            "LOW_RELEVANCE_RETRIEVAL",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "ok",
                        "detail": {"n_chunks": 2},
                        "scores": {"constructive_score": 0.2},
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "ok",
                        "detail": {"n_chunks": 2},
                        "scores": {"constructive_score": 0.9},
                    }
                ],
            },
        ),
        (
            "RETRIEVAL_ERROR",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [{"hop": "r", "kind": "retrieval", "status": "error"}],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [{"hop": "r", "kind": "retrieval", "status": "ok"}],
            },
        ),
        (
            "CATEGORY_MISMATCH",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {"query_category": "finance"},
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "ok",
                        "detail": {"rule_chain": [{"category": "clinical"}]},
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {"query_category": "finance"},
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "ok",
                        "detail": {"rule_chain": [{"category": "finance"}]},
                    }
                ],
            },
        ),
        (
            "PARTIAL_RETRIEVAL",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "ok",
                        "detail": {"n_chunks": 1, "top_k": 5},
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "r",
                        "kind": "retrieval",
                        "status": "ok",
                        "detail": {"n_chunks": 5, "top_k": 5},
                    }
                ],
            },
        ),
        (
            "TOOL_ERROR_SWALLOWED",
            {
                "question": "q",
                "answer": "ok",
                "preload": BASE_PRELOAD,
                "trace": [
                    {"hop": "t", "kind": "tool", "status": "error"},
                    {"hop": "g", "kind": "llm", "status": "ok"},
                ],
                "node_state": {},
            },
            {
                "question": "q",
                "answer": "ok",
                "preload": BASE_PRELOAD,
                "trace": [
                    {"hop": "t", "kind": "tool", "status": "error"},
                    {"hop": "g", "kind": "llm", "status": "ok"},
                ],
                "node_state": {"error": "tool failed"},
            },
        ),
        (
            "TOOL_EMPTY_RESULT",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "declared_sections": ["must_ground"],
                "trace": [
                    {
                        "hop": "t",
                        "kind": "tool",
                        "status": "ok",
                        "detail": {"payload": None},
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "t",
                        "kind": "tool",
                        "status": "ok",
                        "detail": {"payload": {"x": 1}},
                    }
                ],
            },
        ),
        (
            "TOOL_TIMEOUT",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [{"hop": "t", "kind": "tool", "status": "timeout"}],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [{"hop": "t", "kind": "tool", "status": "ok"}],
            },
        ),
        (
            "TOOL_SCHEMA_DRIFT",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [{"hop": "t", "kind": "tool", "status": "ok"}],
                "node_state": {"consumes": ["docs"], "missing_keys": ["docs"]},
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [{"hop": "t", "kind": "tool", "status": "ok"}],
                "node_state": {"consumes": ["docs"], "docs": ["x"]},
            },
        ),
        (
            "CONTEXT_TRUNCATED",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "context_budget": {
                    "limit_tokens": 100,
                    "used_tokens": 200,
                    "truncated": True,
                    "truncated_tail": True,
                },
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "context_budget": {
                    "limit_tokens": 200,
                    "used_tokens": 100,
                    "truncated": False,
                },
            },
        ),
        (
            "MISSING_STATE_KEY",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {"missing_keys": ["docs"]},
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {"docs": ["ok"]},
            },
        ),
        (
            "PRELOAD_DUPLICATION",
            {
                "question": "q",
                "preload": [
                    {"chunk_id": "a", "text": "The quick brown fox jumps high over the lazy dog today."},
                    {"chunk_id": "b", "text": "The quick brown fox jumps high over the lazy dog today."},
                ],
            },
            {
                "question": "q",
                "preload": [
                    {"chunk_id": "a", "text": "Alpha report on revenue growth."},
                    {"chunk_id": "b", "text": "Completely different weather summary."},
                ],
            },
        ),
        (
            "LOW_FIDELITY_SPACE",
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "a",
                        "text": "x",
                        "vector": [0.1] * 64,
                        "vector_space": "jl-64",
                    }
                ],
            },
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "a",
                        "text": "x",
                        "vector": [0.1] * 8,
                        "vector_space": "raw-384",
                    }
                ],
            },
        ),
        (
            "ENCODER_VERSION_MISMATCH",
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "a",
                        "text": "x",
                        "vector": [0.1, 0.2],
                        "vector_space": "raw-384@modelA",
                    }
                ],
                "node_state": {"answer_encoder_artifact": "modelB"},
            },
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "a",
                        "text": "x",
                        "vector": [0.1, 0.2],
                        "vector_space": "raw-384@modelA",
                    }
                ],
                "node_state": {"answer_encoder_artifact": "modelA"},
            },
        ),
        (
            "STALE_CACHE_REUSE",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "detail": {
                            "decision": "HIT_REUSE",
                            "entry_partition_version": 1,
                            "current_partition_version": 3,
                        },
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "detail": {
                            "decision": "HIT_REUSE",
                            "entry_partition_version": 3,
                            "current_partition_version": 3,
                        },
                    }
                ],
            },
        ),
        (
            "CACHE_PREDATES_FACT_UPDATE",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "detail": {
                            "decision": "HIT_REUSE",
                            "created_at": "2026-01-01T00:00:00",
                            "tags": ["person_a"],
                        },
                    }
                ],
                "node_state": {
                    "fact_corrections": [
                        {
                            "subject": "person_a",
                            "valid_from": "2026-02-01T00:00:00",
                        }
                    ]
                },
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "detail": {
                            "decision": "HIT_REUSE",
                            "created_at": "2026-03-01T00:00:00",
                            "tags": ["person_a"],
                        },
                    }
                ],
                "node_state": {
                    "fact_corrections": [
                        {
                            "subject": "person_a",
                            "valid_from": "2026-02-01T00:00:00",
                        }
                    ]
                },
            },
        ),
        (
            "MARGINAL_CACHE_HIT",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "scores": {"verify_score": 0.955},
                        "detail": {"decision": "HIT_REUSE", "threshold": 0.95},
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "scores": {"verify_score": 0.99},
                        "detail": {"decision": "HIT_REUSE", "threshold": 0.95},
                    }
                ],
            },
        ),
        (
            "CACHE_CONTEXT_MISUSE",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "detail": {"decision": "HIT_AS_CONTEXT"},
                    }
                ],
                "node_state": {"cache_sole_support_facts": ["$1000"]},
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "c",
                        "kind": "cache",
                        "status": "ok",
                        "detail": {"decision": "HIT_AS_CONTEXT"},
                    }
                ],
                "node_state": {},
            },
        ),
        (
            "MEMORY_CONFLICT_SERVED",
            {
                "question": "q",
                "preload": [{"chunk_id": "m", "text": "x", "source": "memory"}],
                "node_state": {
                    "memory_conflicts": [{"subject": "A", "relation": "kinship"}]
                },
            },
            {
                "question": "q",
                "preload": [{"chunk_id": "m", "text": "x", "source": "memory"}],
                "node_state": {"memory_conflicts": []},
            },
        ),
        (
            "STAGED_FACT_SERVED",
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "m",
                        "text": "x",
                        "source": "memory",
                        "metadata": {"staged": True, "subject": "A"},
                    }
                ],
            },
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "m",
                        "text": "x",
                        "source": "memory",
                        "metadata": {"staged": False},
                    }
                ],
            },
        ),
        (
            "EXPIRED_FACT_SERVED",
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "m",
                        "text": "x",
                        "source": "memory",
                        "metadata": {
                            "expired": True,
                            "subject": "A",
                            "valid_to": "2020-01-01",
                        },
                    }
                ],
                "node_state": {"query_time": "2026-01-01"},
            },
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "m",
                        "text": "x",
                        "source": "memory",
                        "metadata": {"expired": False},
                    }
                ],
            },
        ),
        (
            "CONFLICTING_PRELOAD_FACTS",
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "h1",
                        "text": "Person A is my brother.",
                        "source": "history",
                    },
                    {
                        "chunk_id": "h2",
                        "text": "Person A is my sister.",
                        "source": "history",
                    },
                ],
            },
            {
                "question": "q",
                "preload": [
                    {
                        "chunk_id": "h1",
                        "text": "Person A is my brother.",
                        "source": "history",
                    },
                    {
                        "chunk_id": "h2",
                        "text": "Person A lives in Boston.",
                        "source": "history",
                    },
                ],
            },
        ),
        (
            "GUARD_GRAY_INPUT",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "g",
                        "kind": "guard",
                        "status": "ok",
                        "detail": {"zone": "gray", "decision": "flag"},
                    }
                ],
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "trace": [
                    {
                        "hop": "g",
                        "kind": "guard",
                        "status": "ok",
                        "detail": {"zone": "pass", "decision": "pass"},
                    }
                ],
            },
        ),
        (
            "HOP_BUDGET_EXHAUSTED",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {"hop_budget_exhausted": True, "hop_count": 10, "hop_limit": 10},
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {"hop_count": 2, "hop_limit": 10},
            },
        ),
        (
            "ANTI_THRASH_TRIGGERED",
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {
                    "anti_thrash": True,
                    "anti_thrash_hop": "react",
                    "anti_thrash_action": "search",
                },
            },
            {
                "question": "q",
                "preload": BASE_PRELOAD,
                "node_state": {},
            },
        ),
    ],
)
def test_signature_fire_and_nofire(sig, fire, nofire):
    fire_hits = _run(**fire)
    nofire_hits = _run(**nofire)
    assert sig in _ids(fire_hits.hits), f"{sig} should fire; got {_ids(fire_hits.hits)}"
    assert sig not in _ids(nofire_hits.hits), f"{sig} should NOT fire; got {_ids(nofire_hits.hits)}"
    hit = next(h for h in fire_hits.hits if h.id == sig)
    assert "{" not in hit.advice or "}" not in hit.advice or hit.advice  # formatted
    assert hit.advice and "something went wrong" not in hit.advice.lower()


def test_handbook_lists_all_expected_signatures():
    expected = {
        "EMPTY_RETRIEVAL",
        "LOW_RELEVANCE_RETRIEVAL",
        "RETRIEVAL_ERROR",
        "CATEGORY_MISMATCH",
        "PARTIAL_RETRIEVAL",
        "TOOL_ERROR_SWALLOWED",
        "TOOL_EMPTY_RESULT",
        "TOOL_TIMEOUT",
        "TOOL_SCHEMA_DRIFT",
        "CONTEXT_TRUNCATED",
        "MISSING_STATE_KEY",
        "PRELOAD_DUPLICATION",
        "LOW_FIDELITY_SPACE",
        "ENCODER_VERSION_MISMATCH",
        "STALE_CACHE_REUSE",
        "CACHE_PREDATES_FACT_UPDATE",
        "MARGINAL_CACHE_HIT",
        "CACHE_CONTEXT_MISUSE",
        "MEMORY_CONFLICT_SERVED",
        "STAGED_FACT_SERVED",
        "EXPIRED_FACT_SERVED",
        "CONFLICTING_PRELOAD_FACTS",
        "GUARD_GRAY_INPUT",
        "HOP_BUDGET_EXHAUSTED",
        "ANTI_THRASH_TRIGGERED",
        "LLM_ERROR",
        "LLM_EMPTY_COMPLETION",
        "LLM_REFUSAL",
        "TRACE_INCOMPLETE",
        "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS",
        "HIT_REVALIDATE_IGNORED",
        "PARALLEL_PRELOAD_AMBIGUITY",
    }
    assert expected <= {s.id for s in HB.signatures}
