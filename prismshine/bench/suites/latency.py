"""B3 — Latency p50/p95 + cost model + judge escalation rate."""

from __future__ import annotations

import time
from typing import Any

from prismshine.bench.embed import hash_embedder
from prismshine.bench.metrics import CostModel, percentile
from prismshine.bench.report import SuiteResult
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate


def _gate(**kwargs) -> ShineGate:
    return ShineGate.build(embedder=hash_embedder, **kwargs)


def _pre_bundle():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": None,
            "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "scores": {"constructive_score": 0.95},
                    "detail": {"n_chunks": 3, "top_k": 3},
                }
            ],
        }
    )
    return b


def _full_bundle(i: int = 0):
    text = f"Revenue was ${1000 + i} in Q1 for Acme Corp."
    b, _ = bundle_from_dict(
        {
            "run_id": f"lat_{i}",
            "question": "What was revenue?",
            "answer": text,
            "preload": [{"chunk_id": "c1", "text": text, "source": "retrieval"}],
            "trace": [
                {
                    "hop": "r",
                    "kind": "retrieval",
                    "status": "ok",
                    "scores": {"constructive_score": 0.95},
                    "detail": {"n_chunks": 3, "top_k": 3},
                }
            ],
        }
    )
    return b


def _mixed_corpus(n: int = 40) -> list[Any]:
    out = []
    for i in range(n):
        if i % 5 == 0:
            # hallucinated — may escalate
            b, _ = bundle_from_dict(
                {
                    "run_id": f"mix_bad_{i}",
                    "question": "What was revenue?",
                    "answer": f"Revenue was ${9000 + i} on Mars for Quokka.",
                    "preload": [
                        {
                            "chunk_id": "c1",
                            "text": f"Revenue was ${1000 + i} in Q1 for Acme Corp.",
                            "source": "retrieval",
                        }
                    ],
                    "trace": [
                        {
                            "hop": "r",
                            "kind": "retrieval",
                            "status": "ok",
                            "detail": {"n_chunks": 1},
                        }
                    ],
                }
            )
        else:
            b = _full_bundle(i)
        out.append(b)
    return out


def run_latency_suite(
    *,
    gate: ShineGate | None = None,
    iterations: int = 30,
    judge_usd_per_call: float = 0.002,
) -> SuiteResult:
    gate = gate or _gate()
    pre = _pre_bundle()
    # warm
    gate.verify(pre)
    full = _full_bundle(0)
    gate.verify(full)

    tier0_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        gate.verify(pre)
        tier0_ms.append((time.perf_counter() - t0) * 1000)

    fast_ms: list[float] = []
    for i in range(iterations):
        b = _full_bundle(i)  # unique content avoids verdict-cache hit skewing latency
        t0 = time.perf_counter()
        gate.verify(b)
        fast_ms.append((time.perf_counter() - t0) * 1000)

    # Judge escalation rate on mixed corpus (no real judge — count tier_reached / gray)
    escalations = 0
    corpus = _mixed_corpus(40)
    for b in corpus:
        v = gate.verify(b)
        # "Would escalate" proxy: gray/regenerate/block from grounding OR contradiction cue
        if v.decision in {"regenerate", "block"} or (
            v.decision == "flag" and v.tier_reached >= 3
        ):
            escalations += 1
        elif any(
            s.name.endswith("contradiction") or "contradiction" in s.name
            for s in v.signals
        ):
            escalations += 1
    esc_rate = escalations / max(len(corpus), 1)

    cost = CostModel(judge_usd_per_call=judge_usd_per_call).compare(
        n_checks=1000, judge_escalation_rate=esc_rate
    )

    t0_p50 = percentile(tier0_ms, 50)
    fast_p50 = percentile(fast_ms, 50)
    # Soft CI budgets (machines vary); POSITIONING target is local <25ms fast path
    passed = t0_p50 < 50 and fast_p50 < 100 and esc_rate <= 0.25

    return SuiteResult(
        name="latency_cost",
        passed=passed,
        gates={
            "tier0_p50_ms_soft_max": 50,
            "tier0_p50_ms": round(t0_p50, 3),
            "fast_p50_ms_soft_max": 100,
            "fast_p50_ms": round(fast_p50, 3),
            "positioning_fast_p50_target_ms": 25,
            "judge_escalation_rate_max": 0.10,
            "judge_escalation_rate": round(esc_rate, 4),
            "judge_escalation_soft_ci_max": 0.25,
        },
        metrics={
            "tier0_p50_ms": round(t0_p50, 3),
            "tier0_p95_ms": round(percentile(tier0_ms, 95), 3),
            "fast_p50_ms": round(fast_p50, 3),
            "fast_p95_ms": round(percentile(fast_ms, 95), 3),
            "iterations": iterations,
            "cost": cost,
        },
        notes=[
            "CI soft budgets are looser than POSITIONING local targets (CPU variance).",
            "Judge escalation is a proxy without a live Tier-4 judge (no API spend).",
            "POSITIONING: fast path p50 < 25ms local; judge <=10% on default profile.",
        ],
        competitor_baseline={
            "status": "literature / not run",
            "detail": (
                f"LLM-as-judge ≈ ${judge_usd_per_call}/check everywhere; "
                "Shine bills only escalations (see cost metrics)."
            ),
        },
    )
