"""Scoreboard helpers for the ChorusGraph+PrismShine runtime suite."""

from __future__ import annotations

from bench.runtime.run_runtime_bench import score, scoreboard


def test_runtime_score_r1_evidence_gap():
    results = {
        "H1": [
            {"gold": "hallucinated", "label": "hallucinated", "latency_ms": 10, "llm_calls": 0, "cost_usd": 0},
            {"gold": "grounded", "label": "grounded", "latency_ms": 12, "llm_calls": 0, "cost_usd": 0},
        ],
        "R1": [
            {"gold": "runtime_fail", "label": "runtime_fail", "saw_evidence": True, "latency_ms": 5, "llm_calls": 0, "cost_usd": 0},
            {"gold": "runtime_fail", "label": "runtime_ok", "saw_evidence": False, "latency_ms": 5, "llm_calls": 0, "cost_usd": 0},
            {"gold": "runtime_ok", "label": "runtime_ok", "saw_evidence": True, "latency_ms": 5, "llm_calls": 0, "cost_usd": 0},
        ],
    }
    s = score(results)
    assert s["H1"]["f1"] == 1.0
    assert s["R1_evidence_aware"]["catch_rate"] == 0.5
    assert s["R1_evidence_aware"]["false_alarm"] == 0.0
    assert s["P1"]["n"] == 5


def test_runtime_scoreboard_mentions_systems():
    summary = {
        "systems": {
            "chorus-shine": {
                "H1": {"f1": 0.8},
                "R1_evidence_aware": {
                    "catch_rate": 1.0,
                    "false_alarm": 0.0,
                    "saw_evidence_rate": 1.0,
                },
                "P1": {"p50_ms": 40, "p95_ms": 90, "llm_calls_total": 0},
            },
            "oss-langgraph-hhem": {
                "H1": {"f1": 0.7},
                "R1_evidence_aware": {
                    "catch_rate": 0.0,
                    "false_alarm": 0.0,
                    "saw_evidence_rate": 0.0,
                },
                "P1": {"p50_ms": 200, "p95_ms": 400, "llm_calls_total": 0},
            },
        }
    }
    md = scoreboard(summary)
    assert "chorus-shine" in md
    assert "R1 catch rate" in md
    assert "0.0" in md  # competitor catch rate
