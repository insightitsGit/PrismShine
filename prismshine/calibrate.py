"""Calibration harness: labeled + synthetic-perturbation modes."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prismshine.evidence.builder import bundle_from_dict
from prismshine.gate import ShineGate
from prismshine.models import EvidenceBundle


@dataclass
class CalibrationReport:
    mode: str
    n_samples: int
    auroc: float | None
    precision_at_bands: dict[str, float]
    recall_at_bands: dict[str, float]
    thresholds: dict[str, float]
    curves: dict[str, list[tuple[float, float]]]
    version: str
    notes: list[str] = field(default_factory=list)

    def to_yaml_overlay(self) -> dict[str, Any]:
        return {
            "calibration_version": self.version,
            "thresholds": self.thresholds,
            "curves": {k: [list(p) for p in v] for k, v in self.curves.items()},
        }


def _swap_number(answer: str, preload: str) -> str:
    nums = re.findall(r"\d+(?:\.\d+)?", answer)
    if not nums:
        return answer + " QUANTUM_VALUE_999"
    for n in nums:
        candidate = str(int(float(n)) + 17)
        if candidate not in preload:
            return answer.replace(n, candidate, 1)
    return answer.replace(nums[0], nums[0] + "9", 1)


def _swap_entity(answer: str, preload: str) -> str:
    ents = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", answer)
    for e in ents:
        fake = "Zephyr Quokka"
        if fake.lower() not in preload.lower():
            return answer.replace(e, fake, 1)
    return answer + " According to Zephyr Quokka."


def synthetic_negatives(
    grounded: list[EvidenceBundle], seed: int = 0
) -> list[tuple[EvidenceBundle, bool]]:
    rng = random.Random(seed)
    out: list[tuple[EvidenceBundle, bool]] = []
    for b in grounded:
        out.append((b, False))
        if not b.answer:
            continue
        preload = "\n".join(c.text for c in b.preload)
        mode = rng.choice(["number", "entity", "drop", "splice"])
        data = b.model_dump(mode="json")
        if mode == "number":
            data["answer"] = _swap_number(b.answer, preload)
        elif mode == "entity":
            data["answer"] = _swap_entity(b.answer, preload)
        elif mode == "drop" and data["preload"]:
            data["preload"] = data["preload"][: max(1, len(data["preload"]) // 2)]
        else:
            # splice unrelated
            data["answer"] = (b.answer or "") + " The moon is made of cheese."
            if data["preload"]:
                data["preload"].append(
                    {
                        "chunk_id": "splice",
                        "text": "Unrelated weather bulletin.",
                        "source": "system",
                    }
                )
        neg, _ = bundle_from_dict(data)
        out.append((neg, True))
    return out


def _scores(gate: ShineGate, pairs: list[tuple[EvidenceBundle, bool]]) -> list[tuple[float, bool]]:
    rows: list[tuple[float, bool]] = []
    for bundle, is_halluc in pairs:
        v = gate.verify(bundle)
        rows.append((v.fused_score, is_halluc))
    return rows


def _auroc(rows: list[tuple[float, bool]]) -> float | None:
    pos = [s for s, y in rows if y]
    neg = [s for s, y in rows if not y]
    if not pos or not neg:
        return None
    total = 0
    correct = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                correct += 1
            elif p == n:
                correct += 0.5
    return correct / total if total else None


def calibrate_labeled(
    pairs: list[tuple[EvidenceBundle, bool]],
    gate: ShineGate | None = None,
    version: str = "cal-labeled-0.1",
) -> CalibrationReport:
    gate = gate or ShineGate.build()
    rows = _scores(gate, pairs)
    auroc = _auroc(rows)
    # Fit simple thresholds: maximize Youden on fused score for flag band
    best_t, best_j = 0.55, -1.0
    for t in [i / 100 for i in range(5, 95)]:
        tp = sum(1 for s, y in rows if y and s >= t)
        fn = sum(1 for s, y in rows if y and s < t)
        fp = sum(1 for s, y in rows if (not y) and s >= t)
        tn = sum(1 for s, y in rows if (not y) and s < t)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    bands = gate.policy.bands
    prec: dict[str, float] = {}
    rec: dict[str, float] = {}
    for name, thr in (("pass", bands[0]), ("gray", bands[1]), ("act", bands[2])):
        pred = [s >= thr for s, _ in rows]
        tp = sum(1 for p, (_, y) in zip(pred, rows, strict=True) if p and y)
        fp = sum(1 for p, (_, y) in zip(pred, rows, strict=True) if p and not y)
        fn = sum(1 for p, (_, y) in zip(pred, rows, strict=True) if (not p) and y)
        prec[name] = tp / max(tp + fp, 1)
        rec[name] = tp / max(tp + fn, 1)
    return CalibrationReport(
        mode="labeled",
        n_samples=len(pairs),
        auroc=auroc,
        precision_at_bands=prec,
        recall_at_bands=rec,
        thresholds={"fused_flag": best_t, "tau_sent": gate.policy.tau_sent},
        curves={"grounding.risk_coverage": [(0.0, 0.0), (1.0, 1.0)]},
        version=version,
    )


def calibrate_synthetic(
    grounded: list[EvidenceBundle],
    gate: ShineGate | None = None,
    seed: int = 0,
    version: str = "cal-synth-0.1",
) -> CalibrationReport:
    pairs = synthetic_negatives(grounded, seed=seed)
    report = calibrate_labeled(pairs, gate=gate, version=version)
    report.mode = "synthetic"
    report.notes.append("Deterministic perturbations; zero LLM calls.")
    return report


def calibrate_dir(
    directory: str | Path,
    mode: str = "synthetic",
    gate: ShineGate | None = None,
) -> CalibrationReport:
    root = Path(directory)
    bundles: list[EvidenceBundle] = []
    labeled: list[tuple[EvidenceBundle, bool]] = []
    for path in sorted(root.glob("**/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "bundle" in data:
            b, _ = bundle_from_dict(data["bundle"])
            if "is_hallucination" in data:
                labeled.append((b, bool(data["is_hallucination"])))
            else:
                bundles.append(b)
        else:
            b, _ = bundle_from_dict(data)
            bundles.append(b)
    if mode == "labeled":
        if not labeled:
            raise ValueError("labeled mode requires JSON with is_hallucination")
        return calibrate_labeled(labeled, gate=gate)
    if not bundles:
        raise ValueError("no bundles found for synthetic calibration")
    return calibrate_synthetic(bundles, gate=gate)
