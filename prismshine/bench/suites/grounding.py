"""B2 — Effect-side grounding quality (synthetic + optional RAGTruth)."""

from __future__ import annotations

import os
from typing import Any

from prismshine.bench.embed import hash_embedder
from prismshine.bench.metrics import Confusion, auroc
from prismshine.bench.report import SuiteResult
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.grounding.spans import SpanClassifier


def _gate() -> ShineGate:
    return ShineGate.build(embedder=hash_embedder)


def _good_trace() -> list[dict[str, Any]]:
    return [
        {
            "hop": "r",
            "kind": "retrieval",
            "status": "ok",
            "scores": {"constructive_score": 0.95},
            "detail": {"n_chunks": 3, "top_k": 3},
        }
    ]


def synthetic_pairs() -> list[tuple[dict[str, Any], bool]]:
    """(bundle_dict, is_hallucination) hard cases."""
    pairs: list[tuple[dict[str, Any], bool]] = []

    # Numbers
    for i in range(5):
        pairs.append(
            (
                {
                    "run_id": f"num_good_{i}",
                    "question": "What was revenue?",
                    "answer": f"Revenue was ${1000 + i} in Q1 for Acme Corp.",
                    "preload": [
                        {
                            "chunk_id": "c1",
                            "text": f"Revenue was ${1000 + i} in Q1 for Acme Corp.",
                            "source": "retrieval",
                        }
                    ],
                    "trace": _good_trace(),
                },
                False,
            )
        )
        pairs.append(
            (
                {
                    "run_id": f"num_bad_{i}",
                    "question": "What was revenue?",
                    "answer": f"Revenue was ${9000 + i} in Q1 for Zephyr Quokka.",
                    "preload": [
                        {
                            "chunk_id": "c1",
                            "text": f"Revenue was ${1000 + i} in Q1 for Acme Corp.",
                            "source": "retrieval",
                        }
                    ],
                    "trace": _good_trace(),
                },
                True,
            )
        )

    # Entities / dates
    pairs.append(
        (
            {
                "run_id": "ent_good",
                "question": "Who signed the deal?",
                "answer": "Alice signed the deal on 2024-03-15.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "Alice signed the deal on 2024-03-15.",
                        "source": "retrieval",
                    }
                ],
                "trace": _good_trace(),
            },
            False,
        )
    )
    pairs.append(
        (
            {
                "run_id": "ent_bad",
                "question": "Who signed the deal?",
                "answer": "Bob signed the deal on 2025-12-01.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "Alice signed the deal on 2024-03-15.",
                        "source": "retrieval",
                    }
                ],
                "trace": _good_trace(),
            },
            True,
        )
    )

    # Negation / contradiction cue
    pairs.append(
        (
            {
                "run_id": "neg_bad",
                "question": "Is the drug safe for children?",
                "answer": "The drug is safe for children.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "The drug is not safe for children.",
                        "source": "retrieval",
                    }
                ],
                "trace": _good_trace(),
                "declared_sections": ["must_ground"],
            },
            True,
        )
    )

    # Structured JSON
    pairs.append(
        (
            {
                "run_id": "json_good",
                "question": "revenue?",
                "answer": '{"revenue": 1000, "currency": "USD"}',
                "preload": [
                    {
                        "chunk_id": "1",
                        "text": '{"revenue": 1000, "currency": "USD"}',
                    }
                ],
                "trace": _good_trace(),
            },
            False,
        )
    )
    pairs.append(
        (
            {
                "run_id": "json_bad",
                "question": "revenue?",
                "answer": '{"revenue": 9999, "currency": "USD"}',
                "preload": [
                    {
                        "chunk_id": "1",
                        "text": '{"revenue": 1000, "currency": "USD"}',
                    }
                ],
                "trace": _good_trace(),
            },
            True,
        )
    )

    return pairs


def _pred_halluc(verdict) -> bool:
    return verdict.decision in {"flag", "block", "regenerate"} or verdict.fused_score >= 0.25


def _span_baseline_pred(bundle, clf: SpanClassifier) -> bool:
    """In-process LettuceDetect-class / lexical proxy — not an external product."""
    result = clf.classify(bundle)
    return result.unsupported_span_ratio >= 0.15 or bool(result.spans)


def _try_ragtruth(limit: int = 50) -> list[tuple[dict[str, Any], bool]] | None:
    if os.environ.get("PRISMSHINE_BENCH_FULL") != "1":
        return None
    import importlib

    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except Exception:  # noqa: BLE001
        return None
    try:
        ds = load_dataset("wandb/RAGTruth-test", split="test")
    except Exception:  # noqa: BLE001
        try:
            ds = load_dataset("RAGTruth/RAGTruth", split="test")
        except Exception:  # noqa: BLE001
            return None
    out: list[tuple[dict[str, Any], bool]] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        q = str(row.get("question") or row.get("query") or "")
        a = str(row.get("answer") or row.get("response") or "")
        ctx = row.get("context") or row.get("passage") or row.get("source") or ""
        if isinstance(ctx, list):
            texts = [str(c) for c in ctx]
        else:
            texts = [str(ctx)]
        label = row.get("hallucination") or row.get("label") or row.get("is_hallucination")
        if label is None and "labels" in row:
            label = bool(row["labels"])
        is_h = bool(label) if not isinstance(label, str) else label.lower() in {
            "1",
            "true",
            "hallucination",
            "yes",
        }
        out.append(
            (
                {
                    "run_id": f"ragtruth_{i}",
                    "question": q or "q",
                    "answer": a or "",
                    "preload": [
                        {"chunk_id": f"c{j}", "text": t, "source": "retrieval"}
                        for j, t in enumerate(texts)
                        if t.strip()
                    ]
                    or [{"chunk_id": "empty", "text": "(no context)", "source": "system"}],
                    "trace": _good_trace(),
                },
                is_h,
            )
        )
    return out or None


def run_grounding_suite(*, gate: ShineGate | None = None) -> SuiteResult:
    gate = gate or _gate()
    conf = Confusion()
    scores: list[float] = []
    labels: list[bool] = []
    cases: list[dict[str, Any]] = []

    for data, is_h in synthetic_pairs():
        b, _ = bundle_from_dict(data)
        v = gate.verify(b)
        pred = _pred_halluc(v)
        conf.update(y_true=is_h, y_pred=pred)
        scores.append(float(v.fused_score))
        labels.append(is_h)
        cases.append(
            {
                "run_id": data["run_id"],
                "is_hallucination": is_h,
                "predicted": pred,
                "decision": v.decision,
                "fused_score": v.fused_score,
            }
        )

    # Span-backend baseline on same pairs (in-process)
    clf = SpanClassifier(allow_lexical_fallback=True)
    span_conf = Confusion()
    for data, is_h in synthetic_pairs():
        b, _ = bundle_from_dict(data)
        span_conf.update(y_true=is_h, y_pred=_span_baseline_pred(b, clf))

    roc = auroc(scores, labels)
    f1_gate = conf.f1 >= 0.85

    rag = _try_ragtruth()
    rag_metrics: dict[str, Any] = {"status": "skipped"}
    if rag:
        rconf = Confusion()
        for data, is_h in rag:
            b, _ = bundle_from_dict(data)
            rconf.update(y_true=is_h, y_pred=_pred_halluc(gate.verify(b)))
        rag_metrics = {"status": "ran", **rconf.as_dict(), "n": len(rag)}

    delta = conf.f1 - span_conf.f1
    # Within 5 F1 pts of in-process span baseline (or better)
    vs_sota = abs(delta) <= 0.05 or conf.f1 >= span_conf.f1

    return SuiteResult(
        name="grounding",
        passed=f1_gate and vs_sota,
        gates={
            "synthetic_f1_min": 0.85,
            "synthetic_f1": round(conf.f1, 4),
            "within_5pts_of_span_baseline": vs_sota,
            "f1_delta_vs_span": round(delta, 4),
        },
        metrics={
            "synthetic": conf.as_dict(),
            "auroc": None if roc is None else round(roc, 4),
            "span_baseline": {
                **span_conf.as_dict(),
                "backend": clf.backend,
                "note": "In-process SpanClassifier (lexical/onnx), not external LettuceDetect API",
            },
            "ragtruth": rag_metrics,
        },
        cases=cases,
        notes=[
            "Synthetic hard cases: numbers, entities/dates, negation, structured JSON.",
            "Set PRISMSHINE_BENCH_FULL=1 + datasets to load RAGTruth subset.",
            "POSITIONING: example-level within 5 F1 of encoder SotA (span baseline proxy).",
        ],
        competitor_baseline={
            "status": "in-process span baseline only",
            "span_f1": round(span_conf.f1, 4),
            "shine_f1": round(conf.f1, 4),
            "detail": "External RAGAS/DeepEval/LettuceDetect Hub runs are not bundled.",
        },
    )
