"""B1 — Cause-side catch rate (Shine-only differentiator)."""

from __future__ import annotations

from typing import Any

from prismshine.bench.embed import hash_embedder
from prismshine.bench.report import SuiteResult
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.wiring import (
    pre_llm_check,
    record_cache,
    record_llm_error,
    record_retrieval,
    wrap_llm,
)


def _gate() -> ShineGate:
    return ShineGate.build(embedder=hash_embedder)


def _cases() -> list[dict[str, Any]]:
    """Injected failure matrix — competitors cannot see these without a ledger."""
    return [
        {
            "name": "EMPTY_RETRIEVAL",
            "expect": "EMPTY_RETRIEVAL",
            "pre_gen": True,
            "state": {
                "question": "What was revenue?",
                "docs": [],
                "declared_sections": ["must_ground"],
                "trace": [record_retrieval("retrieve", n_chunks=0, top_k=3)],
            },
        },
        {
            "name": "TOOL_ERROR_SWALLOWED",
            "expect": "TOOL_ERROR_SWALLOWED",
            "pre_gen": False,
            "bundle": {
                "question": "q",
                "answer": "I invent an answer anyway.",
                "preload": [{"chunk_id": "1", "text": "tool failed"}],
                "declared_sections": ["must_ground"],
                "trace": [
                    {
                        "hop": "tool",
                        "kind": "tool",
                        "status": "error",
                        "detail": {"error": "500"},
                    },
                    {"hop": "g", "kind": "llm", "status": "ok", "detail": {}},
                ],
            },
        },
        {
            "name": "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS",
            "expect": "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS",
            "pre_gen": False,
            "bundle": {
                "question": "q",
                "answer": "guessed without retrieval",
                "preload": [{"chunk_id": "1", "text": "stale"}],
                "declared_sections": ["must_ground"],
                "trace": [
                    record_cache("cache", "MISS"),
                    {"hop": "g", "kind": "llm", "status": "ok", "detail": {}},
                ],
            },
        },
        {
            "name": "HIT_REVALIDATE_IGNORED",
            "expect": "HIT_REVALIDATE_IGNORED",
            "pre_gen": False,
            "bundle": {
                "question": "q",
                "answer": "x",
                "preload": [{"chunk_id": "1", "text": "x"}],
                "trace": [record_cache("c", "HIT_REUSE", must_revalidate=True)],
            },
        },
        {
            "name": "LLM_ERROR",
            "expect": "LLM_ERROR",
            "pre_gen": False,
            "bundle": {
                "question": "q",
                "answer": "partial",
                "preload": [{"chunk_id": "1", "text": "x"}],
                "trace": [record_llm_error("gen", error="429 rate limit")],
            },
        },
        {
            "name": "TRACE_INCOMPLETE",
            "expect": "TRACE_INCOMPLETE",
            "pre_gen": False,
            "bundle": {
                "question": "q",
                "answer": "hello world today",
                "preload": [{"chunk_id": "1", "text": "hello world today"}],
                "trace": [],
                "node_state": {
                    "consumes": ["docs"],
                    "expect_trace_kinds": ["retrieval"],
                },
            },
        },
        {
            "name": "CACHE_PREDATES_FACT_UPDATE",
            "expect": "CACHE_PREDATES_FACT_UPDATE",
            "pre_gen": False,
            "bundle": {
                "question": "Who is A?",
                "answer": "Person A is my brother.",
                "preload": [
                    {
                        "chunk_id": "h1",
                        "text": "Person A is my sister.",
                        "source": "history",
                    }
                ],
                "trace": [
                    {
                        "hop": "cache",
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
        },
        {
            "name": "PARALLEL_PRELOAD_AMBIGUITY",
            "expect": "PARALLEL_PRELOAD_AMBIGUITY",
            "pre_gen": False,
            "bundle": {
                "question": "q",
                "answer": "a",
                "preload": [
                    {"chunk_id": "1", "text": "a"},
                    {"chunk_id": "2", "text": "b"},
                ],
                "node_state": {"parallel_hops": True},
                "trace": [
                    record_retrieval("r1", n_chunks=1),
                    record_retrieval("r2", n_chunks=1),
                    {"hop": "g", "kind": "llm", "status": "ok", "detail": {}},
                ],
            },
        },
        {
            "name": "CONFLICTING_PRELOAD_FACTS",
            "expect": "CONFLICTING_PRELOAD_FACTS",
            "pre_gen": False,
            "bundle": {
                "question": "Who is A?",
                "answer": "Person A is my brother.",
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
                "trace": [record_retrieval("r", n_chunks=2)],
            },
        },
    ]


def _clean_case() -> dict[str, Any]:
    text = "Revenue was $1000 in Q1 for Acme Corp."
    return {
        "question": "What was revenue?",
        "answer": text,
        "preload": [{"chunk_id": "c1", "text": text, "source": "retrieval"}],
        "declared_sections": ["must_ground"],
        "trace": [
            {
                "hop": "r",
                "kind": "retrieval",
                "status": "ok",
                "scores": {"constructive_score": 0.95},
                "detail": {"n_chunks": 1, "top_k": 1},
            }
        ],
    }


def run_cause_suite(*, gate: ShineGate | None = None) -> SuiteResult:
    gate = gate or _gate()
    cases_out: list[dict[str, Any]] = []
    caught = 0
    tokens_avoided = 0
    model_calls_on_halt_path = 0

    for case in _cases():
        expect = case["expect"]
        if case.get("pre_gen"):
            decision = pre_llm_check(gate, case["state"])
            ids = {s.id for s in (decision.verdict.signatures if decision.verdict else [])}
            ok = expect in ids and decision.should_halt
            # wrap_llm must not call the model
            called = {"n": 0}

            def model(_s: str, _u: str, _c=called) -> str:
                _c["n"] += 1
                return "should-not-run"

            wrapped = wrap_llm(
                model, gate, state_factory=lambda c=case: dict(c["state"])
            )
            _ = wrapped("sys", case["state"]["question"])
            model_calls_on_halt_path += called["n"]
            if called["n"] == 0 and decision.should_halt:
                tokens_avoided += 1
        else:
            b, _ = bundle_from_dict(case["bundle"])
            v = gate.verify(b)
            ids = {s.id for s in v.signatures}
            ok = expect in ids
        if ok:
            caught += 1
        cases_out.append(
            {
                "name": case["name"],
                "expect": expect,
                "caught": ok,
                "signatures": sorted(ids),
            }
        )

    # False-alarm check on clean traffic
    clean_b, _ = bundle_from_dict(_clean_case())
    clean_v = gate.verify(clean_b)
    cause_sigs = {
        "EMPTY_RETRIEVAL",
        "TOOL_ERROR_SWALLOWED",
        "RETRIEVAL_SKIPPED_AFTER_CACHE_MISS",
        "HIT_REVALIDATE_IGNORED",
        "LLM_ERROR",
        "TRACE_INCOMPLETE",
        "CACHE_PREDATES_FACT_UPDATE",
        "PARALLEL_PRELOAD_AMBIGUITY",
    }
    clean_ids = {s.id for s in clean_v.signatures}
    false_alarm = bool(clean_ids & cause_sigs)

    n = len(_cases())
    catch_rate = caught / max(n, 1)
    gate_ok = catch_rate >= 0.90 and not false_alarm and model_calls_on_halt_path == 0

    return SuiteResult(
        name="cause_side",
        passed=gate_ok,
        gates={
            "catch_rate_min": 0.90,
            "catch_rate": round(catch_rate, 4),
            "false_alarm_on_clean": false_alarm,
            "pre_gen_model_calls": model_calls_on_halt_path,
            "tokens_avoided_cases": tokens_avoided,
        },
        metrics={
            "n_injected": n,
            "n_caught": caught,
            "catch_rate": round(catch_rate, 4),
            "clean_decision": clean_v.decision,
        },
        cases=cases_out,
        notes=[
            "Competitors that only see (context, question, answer) score N/A on this suite.",
            "POSITIONING gate: >=90% injected runtime failures caught by Tier-0.",
            "Pre-gen halt proves tokens avoided (model never called).",
        ],
        competitor_baseline={
            "status": "literature / not run",
            "detail": (
                "Encoder classifiers and LLM judges cannot observe ledger failures; "
                "this suite is Shine-only by construction."
            ),
        },
    )
