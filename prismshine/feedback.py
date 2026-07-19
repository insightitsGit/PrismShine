"""False-positive / false-negative feedback → calibrate datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from prismshine.evidence.builder import bundle_from_dict
from prismshine.models import EvidenceBundle, ShineVerdict


def record_feedback(
    path: str | Path,
    *,
    bundle: EvidenceBundle | dict[str, Any],
    is_hallucination: bool,
    verdict: ShineVerdict | dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Append one labeled example to a JSONL feedback file for later calibrate."""
    root = Path(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(bundle, EvidenceBundle):
        bundle_data = bundle.model_dump(mode="json")
    else:
        bundle_data = dict(bundle)
    row: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "is_hallucination": bool(is_hallucination),
        "bundle": bundle_data,
    }
    if verdict is not None:
        row["verdict"] = (
            verdict.model_dump(mode="json")
            if isinstance(verdict, ShineVerdict)
            else dict(verdict)
        )
        pred = row["verdict"].get("decision") in {"flag", "block", "regenerate"}
        row["error_type"] = (
            None
            if pred == bool(is_hallucination)
            else ("fp" if pred and not is_hallucination else "fn")
        )
    if note:
        row["note"] = note
    with root.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def load_feedback_pairs(
    path: str | Path,
) -> list[tuple[EvidenceBundle, bool]]:
    """Load JSONL feedback into (bundle, is_hallucination) pairs for calibrate_labeled."""
    root = Path(path)
    if not root.is_file():
        raise FileNotFoundError(root)
    pairs: list[tuple[EvidenceBundle, bool]] = []
    for line in root.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        b, _ = bundle_from_dict(data["bundle"])
        pairs.append((b, bool(data["is_hallucination"])))
    return pairs


def feedback_summary(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if not root.is_file():
        return {"n": 0, "fp": 0, "fn": 0, "ok": 0}
    fp = fn = ok = n = 0
    for line in root.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        data = json.loads(line)
        et = data.get("error_type")
        if et == "fp":
            fp += 1
        elif et == "fn":
            fn += 1
        else:
            ok += 1
    return {"n": n, "fp": fp, "fn": fn, "ok": ok, "path": str(root)}


def export_feedback_dir(
    jsonl_path: str | Path,
    out_dir: str | Path,
) -> int:
    """Write one labeled JSON file per feedback row (for ``prismshine calibrate --mode labeled``)."""
    pairs = load_feedback_pairs(jsonl_path)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for i, (bundle, is_h) in enumerate(pairs):
        payload = {
            "is_hallucination": is_h,
            "bundle": bundle.model_dump(mode="json"),
        }
        (root / f"fb_{i:04d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return len(pairs)
