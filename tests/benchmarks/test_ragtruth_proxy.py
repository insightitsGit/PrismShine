"""Proxy quality harness (synthetic RAGTruth-like labels; records F1)."""

from __future__ import annotations

import numpy as np

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate


def _embed(texts):
    dim = 32
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, int.from_bytes(__import__("hashlib").md5(tok.encode()).digest()[:4], "little") % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_synthetic_detection_f1():
    gate = ShineGate.build(embedder=_embed)
    pairs = []
    for i in range(10):
        good, _ = bundle_from_dict(
            {
                "run_id": f"good{i}",
                "question": "What was revenue?",
                "answer": f"Revenue was ${1000 + i} in Q1 for Acme Corp.",
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
                        "scores": {"constructive_score": 0.95},
                        "detail": {"n_chunks": 3, "top_k": 3},
                    }
                ],
            }
        )
        bad, _ = bundle_from_dict(
            {
                "run_id": f"bad{i}",
                "question": "What was revenue?",
                "answer": (
                    f"Revenue was ${9000 + i} in Q1 for Zephyr Quokka Industries "
                    "on the lunar colony."
                ),
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
                        "scores": {"constructive_score": 0.95},
                        "detail": {"n_chunks": 3, "top_k": 3},
                    }
                ],
            }
        )
        pairs.append((good, False))
        pairs.append((bad, True))

    tp = fp = tn = fn = 0
    for bundle, is_halluc in pairs:
        v = gate.verify(bundle)
        pred = v.decision in {"flag", "block", "regenerate"} or v.fused_score >= 0.25
        if is_halluc and pred:
            tp += 1
        elif is_halluc and not pred:
            fn += 1
        elif (not is_halluc) and pred:
            fp += 1
        else:
            tn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    assert f1 >= 0.5, (
        f"Tier2+ proxy F1={f1:.3f} prec={prec:.3f} rec={rec:.3f} "
        f"tp={tp} fp={fp} fn={fn} tn={tn}"
    )
