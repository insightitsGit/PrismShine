"""B2 — Effect-side grounding quality (hard corpus + optional RAGTruth + calibrate lift)."""

from __future__ import annotations

from typing import Any

from prismshine.bench.embed import hash_embedder
from prismshine.bench.metrics import Confusion, auroc
from prismshine.bench.ragtruth import hard_effect_pairs, try_load_ragtruth
from prismshine.bench.report import SuiteResult
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.grounding.spans import SpanClassifier


def _gate(profile: str = "default") -> ShineGate:
    return ShineGate.build(embedder=hash_embedder, profile=profile)


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
    """Legacy synthetic numbers/entities + hard effect corpus."""
    pairs: list[tuple[dict[str, Any], bool]] = []
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
    pairs.append(
        (
            {
                "run_id": "json_bad",
                "question": "revenue?",
                "answer": '{"revenue": 9999, "currency": "USD"}',
                "preload": [
                    {"chunk_id": "1", "text": '{"revenue": 1000, "currency": "USD"}'}
                ],
                "trace": _good_trace(),
            },
            True,
        )
    )
    pairs.extend(hard_effect_pairs())
    return pairs


def _pred_halluc(verdict) -> bool:
    return verdict.decision in {"flag", "block", "regenerate"} or verdict.fused_score >= 0.25


def _span_baseline_pred(bundle, clf: SpanClassifier) -> bool:
    result = clf.classify(bundle)
    return result.unsupported_span_ratio >= 0.15 or bool(result.spans)


def _eval_pairs(
    gate: ShineGate, pairs: list[tuple[dict[str, Any], bool]]
) -> tuple[Confusion, list[float], list[bool], list[dict[str, Any]]]:
    conf = Confusion()
    scores: list[float] = []
    labels: list[bool] = []
    cases: list[dict[str, Any]] = []
    for data, is_h in pairs:
        b, _ = bundle_from_dict(data)
        v = gate.verify(b)
        pred = _pred_halluc(v)
        conf.update(y_true=is_h, y_pred=pred)
        scores.append(float(v.fused_score))
        labels.append(is_h)
        cases.append(
            {
                "run_id": data.get("run_id"),
                "is_hallucination": is_h,
                "predicted": pred,
                "decision": v.decision,
                "fused_score": v.fused_score,
                "gate": v.resolution_gate,
            }
        )
    return conf, scores, labels, cases


def _domain_calibrate_lift() -> dict[str, Any]:
    """Decision-F1 before vs after synthetic calibrate on a small grounded set."""
    from prismshine.calibrate import domain_calibrate_lift

    grounded = []
    for i in range(8):
        b, _ = bundle_from_dict(
            {
                "run_id": f"cal_g_{i}",
                "question": "What was revenue?",
                "answer": f"Revenue was ${1000 + i} in Q1 for Acme Corp.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": f"Revenue was ${1000 + i} in Q1 for Acme Corp.",
                    }
                ],
                "trace": _good_trace(),
            }
        )
        grounded.append(b)

    out = domain_calibrate_lift(
        grounded, profile="clinical", seed=7, embedder=hash_embedder
    )
    out["notes"] = [
        "F1 lift measured on synthetic negatives (band fit); labeled packs preferred for claims.",
        "Gate: +0.10 decision-F1 or calibrated F1 >= 0.90.",
    ]
    return out


def run_grounding_suite(*, gate: ShineGate | None = None) -> SuiteResult:
    gate = gate or _gate()
    pairs = synthetic_pairs()
    conf, scores, labels, cases = _eval_pairs(gate, pairs)

    clf = SpanClassifier(allow_lexical_fallback=True)
    span_conf = Confusion()
    for data, is_h in pairs:
        b, _ = bundle_from_dict(data)
        span_conf.update(y_true=is_h, y_pred=_span_baseline_pred(b, clf))

    roc = auroc(scores, labels)
    f1_gate = conf.f1 >= 0.85
    delta = conf.f1 - span_conf.f1
    vs_sota = abs(delta) <= 0.05 or conf.f1 >= span_conf.f1

    rag = try_load_ragtruth()
    rag_metrics: dict[str, Any] = {"status": "skipped", "hard_effect_offline": True}
    if rag:
        rconf, rscores, rlabels, _ = _eval_pairs(gate, rag)
        rag_metrics = {
            "status": "ran",
            **rconf.as_dict(),
            "n": len(rag),
            "auroc": auroc(rscores, rlabels),
        }

    cal = _domain_calibrate_lift()

    # Hard negation subset must not PASS as grounded
    neg_cases = [c for c in cases if c.get("run_id") == "hard_neg_bad"]
    neg_ok = all(c["predicted"] for c in neg_cases) if neg_cases else True

    passed = f1_gate and vs_sota and neg_ok

    return SuiteResult(
        name="grounding",
        passed=passed,
        gates={
            "synthetic_f1_min": 0.85,
            "synthetic_f1": round(conf.f1, 4),
            "within_5pts_of_span_baseline": vs_sota,
            "f1_delta_vs_span": round(delta, 4),
            "hard_negation_caught": neg_ok,
            "calibrate_lift_met": cal.get("lift_met"),
        },
        metrics={
            "synthetic": conf.as_dict(),
            "auroc": None if roc is None else round(roc, 4),
            "n_pairs": len(pairs),
            "span_baseline": {
                **span_conf.as_dict(),
                "backend": clf.backend,
                "artifact_id": clf.artifact_id,
                "note": "In-process SpanClassifier; pin via PRISMSHINE_SPAN_ONNX for onnx",
            },
            "ragtruth": rag_metrics,
            "domain_calibrate": cal,
        },
        cases=cases,
        notes=[
            "Includes hard_effect_pairs (negation/entity/finance/legal) offline.",
            "Set PRISMSHINE_BENCH_FULL=1 + datasets for real RAGTruth subset.",
            "Pin ONNX: PRISMSHINE_SPAN_ONNX (+ optional PRISMSHINE_SPAN_TOKENIZER).",
            "Domain calibrate lift is a separate receipt row under domain_calibrate.",
        ],
        competitor_baseline={
            "status": "in-process span baseline only",
            "span_f1": round(span_conf.f1, 4),
            "shine_f1": round(conf.f1, 4),
            "span_backend": clf.backend,
            "detail": "External RAGAS/DeepEval/LettuceDetect Hub runs are not bundled.",
        },
    )
