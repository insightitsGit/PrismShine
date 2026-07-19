"""RAGTruth / hard-effect corpus loaders for grounding receipts."""

from __future__ import annotations

import importlib
import os
from typing import Any


def hard_effect_pairs() -> list[tuple[dict[str, Any], bool]]:
    """RAGTruth-like hard cases that run offline (always available).

    Covers unsupported claims, entity swaps, negation, numbers, and cue-less
    polarity — the classes encoder tools compete on.
    """
    trace = [
        {
            "hop": "r",
            "kind": "retrieval",
            "status": "ok",
            "scores": {"constructive_score": 0.95},
            "detail": {"n_chunks": 2, "top_k": 2},
        }
    ]
    cases: list[tuple[dict[str, Any], bool]] = []

    # Negation / polarity (effect-side week point)
    cases.append(
        (
            {
                "run_id": "hard_neg_bad",
                "question": "Is the drug safe for children?",
                "answer": "The drug is safe for children.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "The drug is not safe for children and is contraindicated.",
                        "source": "retrieval",
                    }
                ],
                "trace": trace,
                "declared_sections": ["must_ground"],
            },
            True,
        )
    )
    cases.append(
        (
            {
                "run_id": "hard_neg_good",
                "question": "Is the drug safe for children?",
                "answer": "The drug is not safe for children.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "The drug is not safe for children and is contraindicated.",
                        "source": "retrieval",
                    }
                ],
                "trace": trace,
            },
            False,
        )
    )

    # Unsupported claim (hallucinated detail)
    cases.append(
        (
            {
                "run_id": "hard_uns_bad",
                "question": "What did the study find?",
                "answer": (
                    "The study found a 40% improvement and was published in Nature in 2019."
                ),
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "The study found modest improvement in symptoms among adults.",
                        "source": "retrieval",
                    }
                ],
                "trace": trace,
            },
            True,
        )
    )

    # Entity / date swap
    cases.append(
        (
            {
                "run_id": "hard_ent_bad",
                "question": "Who acquired Contoso?",
                "answer": "Fabrikam acquired Contoso on 12 March 2022 for $2.4 billion.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "Northwind acquired Contoso on 12 March 2021 for $1.1 billion.",
                        "source": "retrieval",
                    }
                ],
                "trace": trace,
            },
            True,
        )
    )
    cases.append(
        (
            {
                "run_id": "hard_ent_good",
                "question": "Who acquired Contoso?",
                "answer": "Northwind acquired Contoso on 12 March 2021 for $1.1 billion.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "Northwind acquired Contoso on 12 March 2021 for $1.1 billion.",
                        "source": "retrieval",
                    }
                ],
                "trace": trace,
            },
            False,
        )
    )

    # Finance polarity
    cases.append(
        (
            {
                "run_id": "hard_fin_bad",
                "question": "Did Acme report a profit?",
                "answer": "Acme reported a profit of $50 million in Q2.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "Acme reported a loss of $50 million in Q2.",
                        "source": "retrieval",
                    }
                ],
                "trace": trace,
            },
            True,
        )
    )

    # Legal polarity
    cases.append(
        (
            {
                "run_id": "hard_leg_bad",
                "question": "Was the defendant found guilty?",
                "answer": "The defendant was found guilty on all counts.",
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "The defendant was found innocent on all counts.",
                        "source": "retrieval",
                    }
                ],
                "trace": trace,
            },
            True,
        )
    )

    # Grounded multi-sentence
    cases.append(
        (
            {
                "run_id": "hard_ok_multi",
                "question": "Summarize the filing.",
                "answer": (
                    "Revenue was $1000 in Q1 for Acme Corp. "
                    "Alice signed the deal on 2024-03-15."
                ),
                "preload": [
                    {
                        "chunk_id": "c1",
                        "text": "Revenue was $1000 in Q1 for Acme Corp.",
                        "source": "retrieval",
                    },
                    {
                        "chunk_id": "c2",
                        "text": "Alice signed the deal on 2024-03-15.",
                        "source": "retrieval",
                    },
                ],
                "trace": trace,
            },
            False,
        )
    )

    return cases


def try_load_ragtruth(limit: int = 100) -> list[tuple[dict[str, Any], bool]] | None:
    """Load a public RAGTruth-style split when PRISMSHINE_BENCH_FULL=1."""
    if os.environ.get("PRISMSHINE_BENCH_FULL") != "1":
        return None
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except Exception:  # noqa: BLE001
        return None

    candidates = (
        ("wandb/RAGTruth-processed", "test"),
        ("flowaicom/RAGTruth_test", "test"),
        ("nimitkalra/RAGTruth", "test"),
    )
    ds = None
    loaded_name = ""
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            loaded_name = name
            break
        except Exception:  # noqa: BLE001
            continue
    if ds is None:
        return None

    out: list[tuple[dict[str, Any], bool]] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        q = str(row.get("question") or row.get("query") or "")
        a = str(row.get("answer") or row.get("response") or row.get("output") or "")
        ctx = row.get("context") or row.get("passage") or row.get("source") or ""
        if isinstance(ctx, list):
            texts = [str(c.get("text") if isinstance(c, dict) else c) for c in ctx]
        else:
            texts = [str(ctx)]
        # Prefer processed labels from wandb/RAGTruth-processed
        processed = row.get("hallucination_labels_processed")
        if isinstance(processed, dict):
            is_h = bool(
                int(processed.get("evident_conflict") or 0)
                + int(processed.get("baseless_info") or 0)
            )
        else:
            label = (
                row.get("hallucination")
                or row.get("hallucination_labels")
                or row.get("label")
                or row.get("is_hallucination")
                or row.get("hallucinated")
            )
            if isinstance(label, dict):
                is_h = any(bool(v) for v in label.values())
            elif isinstance(label, list):
                is_h = len(label) > 0
            elif isinstance(label, str):
                s = label.strip()
                low = s.lower()
                if low in {"", "[]", "0", "false", "no", "none"}:
                    is_h = False
                elif low in {"1", "true", "hallucination", "yes", "positive"}:
                    is_h = True
                elif s.startswith("["):
                    is_h = s not in {"[]", "[ ]"}
                else:
                    is_h = bool(s)
            else:
                is_h = bool(label)
        # wandb context sometimes appends a trailing "output:" prompt artifact
        texts = [t.replace("\n\noutput:", "").strip() for t in texts]
        preload = [
            {"chunk_id": f"c{j}", "text": t, "source": "retrieval"}
            for j, t in enumerate(texts)
            if t and t.strip()
        ] or [{"chunk_id": "empty", "text": "(no context)", "source": "system"}]
        out.append(
            (
                {
                    "run_id": f"ragtruth_{i}",
                    "question": q or "q",
                    "answer": a or "",
                    "preload": preload,
                    "trace": [
                        {
                            "hop": "r",
                            "kind": "retrieval",
                            "status": "ok",
                            "detail": {
                                "n_chunks": len(preload),
                                "dataset": loaded_name,
                            },
                        }
                    ],
                },
                is_h,
            )
        )
    return out or None
