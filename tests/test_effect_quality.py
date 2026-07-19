"""Effect-side quality improvements: contradiction, feedback, hard corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prismshine.bench.ragtruth import hard_effect_pairs
from prismshine.bench.suites.grounding import run_grounding_suite
from prismshine.evidence.builder import bundle_from_dict
from prismshine.feedback import load_feedback_pairs, record_feedback
from prismshine.gate import ShineGate
from prismshine.grounding.contradiction import screen_contradictions
from prismshine.grounding.spans import PINNED_TOKENIZER_ENV, SpanClassifier


def _embed(texts):
    dim = 16
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_expanded_polarity_and_clinical_pairs():
    cues = screen_contradictions(
        ["The drug is safe for children."],
        [("c1", "The drug is not safe for children.")],
    )
    assert cues
    cues2 = screen_contradictions(
        ["Acme reported a profit of $50 million."],
        [("c1", "Acme reported a loss of $50 million.")],
    )
    assert cues2
    cues3 = screen_contradictions(
        ["Treatment is indicated for adults."],
        [("c1", "Treatment is contraindicated for adults.")],
    )
    assert cues3


def test_default_profile_negation_cannot_pass():
    gate = ShineGate.build(embedder=_embed, profile="default", judge=None)
    b, _ = bundle_from_dict(
        {
            "question": "Is the drug safe?",
            "answer": "The drug is safe for children.",
            "preload": [
                {
                    "chunk_id": "c1",
                    "text": "The drug is not safe for children.",
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
    v = gate.verify(b)
    assert v.decision != "pass"
    assert "CONTRADICTION" in v.resolution_gate or v.decision in {
        "flag",
        "block",
        "regenerate",
    }


def test_lexical_marks_contradiction_candidates():
    clf = SpanClassifier(allow_lexical_fallback=True)
    from prismshine.models import EvidenceBundle, PreloadChunk, Span

    b = EvidenceBundle(
        run_id="t",
        question="q",
        answer="The drug is safe for children.",
        preload=[
            PreloadChunk(
                chunk_id="c1",
                text="The drug is not safe for children.",
                source="retrieval",
            )
        ],
    )
    cand = [
        Span(
            start=0,
            end=len(b.answer or ""),
            text=b.answer or "",
            reason="contradiction_cue:negation_asymmetry",
            tier=2,
        )
    ]
    r = clf.classify(b, candidate_spans=cand)
    assert r.spans
    assert any("contradiction" in s.reason for s in r.spans)


def test_feedback_roundtrip(tmp_path: Path):
    gate = ShineGate.build(embedder=_embed)
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $9999.",
            "preload": [{"chunk_id": "1", "text": "Revenue was $1000."}],
        }
    )
    v = gate.verify(b)
    path = tmp_path / "feedback.jsonl"
    record_feedback(path, bundle=b, is_hallucination=True, verdict=v)
    pairs = load_feedback_pairs(path)
    assert len(pairs) == 1
    assert pairs[0][1] is True


def test_hard_effect_and_grounding_suite():
    assert len(hard_effect_pairs()) >= 6
    result = run_grounding_suite()
    assert result.gates["synthetic_f1"] >= 0.85
    assert result.gates["hard_negation_caught"] is True


def test_span_tokenizer_env_recognized(monkeypatch):
    monkeypatch.setenv(PINNED_TOKENIZER_ENV, "C:\\nonexistent\\tokenizer.json")
    clf = SpanClassifier(allow_lexical_fallback=True)
    assert clf._pinned_tokenizer.endswith("tokenizer.json")
