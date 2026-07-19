"""Conformance suite: any RuntimeAdapter must expose the four capabilities."""

from __future__ import annotations

import numpy as np

from prismshine.gate import ShineGate
from prismshine.integrations.chorusgraph import ChorusGraphAdapter
from prismshine.integrations.langgraph import LangGraphAdapter
from prismshine.runtime import REQUIRED_CAPABILITIES, assert_adapter, check_adapter


def _embed(texts):
    dim = 8
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_required_capabilities_constant():
    assert set(REQUIRED_CAPABILITIES) == {
        "extract_bundle",
        "enforce",
        "pre_llm_hook",
        "post_llm_hook",
    }


def test_incomplete_adapter_detected():
    class Partial:
        def extract_bundle(self, run):
            return None

    assert "enforce" in check_adapter(Partial())


def test_shipped_adapters_conform():
    gate = ShineGate.build(embedder=_embed)
    for adapter in (ChorusGraphAdapter(gate), LangGraphAdapter(gate)):
        assert_adapter(adapter)
        # Round-trip smoke: empty-ish state still returns a verdict type
        state = {
            "question": "q",
            "answer": "hello there friend",
            "reply": "hello there friend",
            "docs": ["hello there friend"],
        }
        v = adapter.post_llm_hook(state)
        assert v.decision in {"pass", "flag", "block", "regenerate"}
        out = adapter.enforce(v, state)
        assert "shine_route" in out
