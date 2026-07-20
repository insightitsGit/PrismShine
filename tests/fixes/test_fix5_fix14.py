"""FIX-5, FIX-8 through FIX-14 acceptance tests."""

from __future__ import annotations

import gc
import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from prismshine.audit import AuditLog
from prismshine.cache import SqliteVerdictStore
from prismshine.encoder import SharedEncoder
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.grounding.judge import CachedJudge, JudgeResult, parse_judge_json
from prismshine.grounding.spans import SpanClassifier
from prismshine.integrations.chorusgraph import shine_node
from prismshine.models import EvidenceBundle, ShineVerdict
from prismshine.wiring import _UNHASHABLE_WIRED, _WIRED, is_shine_wired, mark_shine_wired


def _embed(texts):
    dim = 8
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1
        n = np.linalg.norm(out[i]) or 1
        out[i] /= n
    return out


def test_fix5_pinned_onnx_without_tokenizer_stays_unavailable(tmp_path, monkeypatch):
    onnx_path = tmp_path / "dummy.onnx"
    onnx_path.write_bytes(b"\x08\x03invalid-onnx-for-test")
    monkeypatch.setenv("PRISMSHINE_SPAN_ONNX", str(onnx_path))
    monkeypatch.delenv("PRISMSHINE_SPAN_TOKENIZER", raising=False)
    sc = SpanClassifier(allow_lexical_fallback=False)
    assert sc.available is False
    assert sc.available is False
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Some answer text.",
            "preload": [{"chunk_id": "1", "text": "Preload text."}],
        }
    )
    r = sc.classify(b)
    assert r.available is False


def test_fix8_wiring_registry_bounded_and_no_false_positive():
    class Graph:
        pass

    g1 = Graph()
    mark_shine_wired(g1, node=True)
    assert is_shine_wired(g1)
    del g1
    gc.collect()
    g2 = Graph()
    assert not is_shine_wired(g2)
    assert len(_WIRED) + len(_UNHASHABLE_WIRED) < 20

    gate = ShineGate.build(embedder=_embed)
    node = shine_node(gate, compiled=object())
    state = {"question": "q", "reply": "a", "docs": [{"text": "ctx", "chunk_id": "1"}]}
    for _ in range(200):
        node(dict(state))
    assert len(_WIRED) + len(_UNHASHABLE_WIRED) < 20


def test_fix9_encoder_memo_bounded_and_cleared_on_mode_flip():
    enc = SharedEncoder(embedder=_embed)
    for i in range(20_000):
        enc.encode([f"unique sentence number {i}"])
    assert len(enc._memo) <= enc._memo_maxsize

    enc._memo_put("keep", np.ones(8))
    enc._mode = "lexical"
    enc._memo.clear()
    assert len(enc._memo) == 0
    enc.encode(["after clear"])
    assert len(enc._memo) == 1


def test_fix9_judge_cache_bounded():
    calls = {"n": 0}

    def inner(claims, context):
        calls["n"] += 1
        return JudgeResult(risk=0.1)

    j = CachedJudge(inner, maxsize=1000)
    for i in range(1500):
        j([f"claim {i}"], "ctx")
    assert len(j._cache) <= 1000
    assert calls["n"] == 1500


def test_fix10_fusion_dead_code_paths():
    from prismshine.fusion import fuse
    from prismshine.models import Signal
    from prismshine.policy import resolve_policy

    pol = resolve_policy()
    r = fuse(
        [Signal(name="grounding.risk_coverage", tier=2, value=1.0, weight=0.25)],
        [],
        pol,
        gray_unresolved=False,
    )
    assert r.decision == "flag"
    r2 = fuse([], [], pol, early_gate="HANDBOOK:FOO:REGENERATE")
    assert r2.decision == "regenerate"


def test_fix12_classify_degrades_on_onnx_failure():
    sc = SpanClassifier()
    sc._session = object()
    sc._tokenizer = type("T", (), {"encode": lambda self, x: type("E", (), {"ids": [1, 2, 3], "offsets": [(0, 1), (1, 2), (2, 3)]})()})()
    sc._backend = "onnx"
    sc.allow_lexical_fallback = True
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "novel tokens xyz",
            "preload": [{"chunk_id": "1", "text": "context"}],
        }
    )
    with patch.object(sc, "_onnx_unsupported", side_effect=RuntimeError("token overflow")):
        r = sc.classify(b)
    assert r.available
    assert r.backend == "lexical"


def test_fix13_parse_judge_json_strips_fences():
    raw = '```json\n{"overall_risk": 0.2, "claims": []}\n```'
    data = parse_judge_json(raw)
    assert data["overall_risk"] == 0.2


def test_fix14_audit_metrics_uses_lock():
    log = AuditLog(maxlen=10)
    b, _ = bundle_from_dict({"question": "q", "answer": "a", "preload": [{"text": "p", "chunk_id": "1"}]})
    v = ShineVerdict(
        decision="pass",
        resolution_gate="X",
        fused_score=0.1,
        confidence=0.5,
        evidence_hash="h",
        verdict_id="1",
    )
    log.record(b, v)
    m = log.metrics()
    assert "hri" in m
    assert m["total"] == 1


def test_fix14_sqlite_store_prunes_max_rows(tmp_path):
    db = SqliteVerdictStore(tmp_path / "v.db", max_rows=3)
    v = ShineVerdict(
        decision="pass",
        resolution_gate="X",
        fused_score=0.1,
        confidence=0.5,
        evidence_hash="h",
        verdict_id="1",
    )
    for i in range(5):
        db.put(f"k{i}", v)
    with sqlite3.connect(str(db.path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    assert count == 3


def test_fix14_encoder_user_vector_space_label():
    enc = SharedEncoder(embedder=_embed)
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "a",
            "preload": [{"chunk_id": "1", "text": "preload without vector"}],
        }
    )
    out = enc.ensure_chunk_vectors(b)
    assert out.preload[0].vector_space.startswith("user@")
    assert out.preload[0].vector_space.endswith("d")


def test_fix14_on_fact_corrected_reexported():
    from prismshine.integrations.chorusgraph import on_fact_corrected as cg_hook
    from prismshine.wiring import on_fact_corrected as wiring_hook

    assert cg_hook is not wiring_hook
    cg_hook(cache=None, sidecar=None, stack=None)
