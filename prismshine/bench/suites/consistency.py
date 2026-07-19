"""B4 — Consistency dual-rail (prevention off → detection must catch 100%)."""

from __future__ import annotations

from typing import Any

from prismshine.bench.embed import hash_embedder
from prismshine.bench.report import SuiteResult
from prismshine.evidence.builder import bundle_from_dict
from prismshine.forensics.engine import run_forensics
from prismshine.gate import ShineGate
from prismshine.handbook.loader import load_handbook
from prismshine.wiring import on_fact_corrected


def _stale_cache_bundle() -> Any:
    b, _ = bundle_from_dict(
        {
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
        }
    )
    return b


def run_consistency_suite(*, gate: ShineGate | None = None) -> SuiteResult:
    gate = gate or ShineGate.build(embedder=hash_embedder)
    hb = load_handbook()
    cases: list[dict[str, Any]] = []

    # Rail 1: prevention no-ops (no cache/sidecar) — detection must still fire
    b = _stale_cache_bundle()
    on_fact_corrected(cache=None, sidecar=None, stack=None, subjects=["person_a"])
    hits = run_forensics(b, hb)
    det_ok = "CACHE_PREDATES_FACT_UPDATE" in {h.id for h in hits.hits}
    cases.append(
        {
            "name": "detection_without_prevention",
            "caught": det_ok,
            "signatures": sorted({h.id for h in hits.hits}),
        }
    )

    # Rail 2: full gate.verify path (same evidence)
    v = gate.verify(b)
    verify_ok = "CACHE_PREDATES_FACT_UPDATE" in {s.id for s in v.signatures}
    cases.append(
        {
            "name": "gate_verify_stale_cache",
            "caught": verify_ok,
            "decision": v.decision,
            "resolution_gate": v.resolution_gate,
        }
    )

    # Fresh cache hit after correction timestamp — must NOT fire predates
    fresh, _ = bundle_from_dict(
        {
            "question": "Who is A?",
            "answer": "Person A is my sister.",
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
        }
    )
    fresh_hits = run_forensics(fresh, hb)
    fresh_ok = "CACHE_PREDATES_FACT_UPDATE" not in {h.id for h in fresh_hits.hits}
    cases.append(
        {
            "name": "fresh_cache_no_false_positive",
            "caught": fresh_ok,
            "signatures": sorted({h.id for h in fresh_hits.hits}),
        }
    )

    # Prevention duck-type: fake cache records invalidate call
    class _FakeCache:
        def __init__(self) -> None:
            self.calls = 0

        def invalidate_where(self, vector, tau_evict=0.55):  # noqa: ANN001
            self.calls += 1
            return 1

    cache = _FakeCache()
    on_fact_corrected(
        cache=cache,
        query_vector=[0.1] * 8,
        subjects=["person_a"],
        threshold=0.55,
    )
    prevention_invoked = cache.calls >= 1
    cases.append(
        {
            "name": "prevention_hook_invokes_invalidate",
            "caught": prevention_invoked,
            "invalidate_calls": cache.calls,
        }
    )

    # Dual-rail gate: when prevention is off, detection catch rate must be 100%
    detection_cases = [det_ok, verify_ok]
    detection_rate = sum(1 for x in detection_cases if x) / len(detection_cases)
    passed = detection_rate == 1.0 and fresh_ok

    return SuiteResult(
        name="consistency",
        passed=passed,
        gates={
            "detection_catch_rate_when_prevention_off": detection_rate,
            "detection_catch_rate_min": 1.0,
            "fresh_cache_false_positive": not fresh_ok,
            "prevention_hook_callable": prevention_invoked,
        },
        metrics={
            "n_stale_scenarios": len(detection_cases),
            "n_caught": sum(1 for x in detection_cases if x),
        },
        cases=cases,
        notes=[
            "POSITIONING: zero stale-cache serves after correction — dual-rail.",
            "Prevention off (no cache object) must not disable CACHE_PREDATES_FACT_UPDATE.",
            "Competitors cannot see cache-gate ledger detail; suite is Shine-only.",
        ],
        competitor_baseline={
            "status": "literature / not run",
            "detail": "Encoder/judge tools never observe cache HIT_REUSE created_at vs corrections.",
        },
    )
