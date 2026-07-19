from __future__ import annotations

import json
from pathlib import Path

from prismshine.audit import AuditLog
from prismshine.cache import MemoryVerdictStore, SqliteVerdictStore, verdict_cache_key
from prismshine.cli import main
from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
import numpy as np

from prismshine.models import ShineVerdict


def fake_embedder(texts: list[str]) -> np.ndarray:
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, int.from_bytes(__import__("hashlib").md5(tok.encode()).digest()[:4], "little") % dim] += 1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


def test_verdict_cache_roundtrip(tmp_path: Path):
    store = SqliteVerdictStore(tmp_path / "v.db")
    key = verdict_cache_key(
        preload_ids=["a"],
        preload_hash="h",
        answer_norm="ans",
        profile_id="default",
        handbook_version="0.1.0",
        calibration_version="identity-0",
        model_artifact_ids=[],
    )
    v = ShineVerdict(
        decision="pass",
        resolution_gate="CLEAN_FAST_PATH",
        fused_score=0.1,
        confidence=0.9,
        evidence_hash="abc",
        verdict_id="1",
    )
    store.put(key, v)
    assert store.get(key).decision == "pass"
    mem = MemoryVerdictStore(maxsize=2)
    mem.put("k1", v)
    assert mem.get("k1") is not None


def test_audit_hri():
    log = AuditLog()
    b, _ = bundle_from_dict(
        {"question": "q", "preload": [{"text": "t", "chunk_id": "1"}], "answer": "t"}
    )
    gate = ShineGate.build(embedder=fake_embedder)
    v = gate.verify(b)
    log.record(b, v)
    assert 0 <= log.hri() <= 100


def test_cli_capabilities():
    assert main(["capabilities"]) == 0


def test_cli_verify(tmp_path: Path):
    p = tmp_path / "b.json"
    p.write_text(
        json.dumps(
            {
                "question": "q",
                "answer": "Revenue was $1000.",
                "preload": [{"text": "Revenue was $1000.", "chunk_id": "1"}],
            }
        ),
        encoding="utf-8",
    )
    # may return 0 or 1 depending on decision
    assert main(["verify", str(p)]) in {0, 1}
