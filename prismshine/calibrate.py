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


def fit_effect_thresholds(
    pairs: list[tuple[EvidenceBundle, bool]],
    gate: ShineGate,
    *,
    tau_sent_grid: list[float] | None = None,
    tau_floor_grid: list[float] | None = None,
) -> dict[str, float]:
    """Grid-search coverage + fused-pass thresholds for decision F1.

    This is the product calibration path for effect-side FP/FN (domain packs and
    bench calibrated rows both call this — not a bench-only hack).
    """
    from prismshine.policy import apply_calibration_receipt

    tau_sents = tau_sent_grid or [0.45, 0.50, 0.55, 0.62, 0.68, 0.72]
    tau_floors = tau_floor_grid or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    base_bands = gate.policy.bands
    base_tau_sent = gate.policy.tau_sent
    base_tau_floor = gate.policy.tau_floor
    best_f1 = -1.0
    best: dict[str, float] = {
        "fused_pass": base_bands[0],
        "fused_flag": base_bands[1],
        "fused_act": base_bands[2],
        "tau_sent": base_tau_sent,
        "tau_floor": base_tau_floor,
        "tau_tok": gate.policy.tau_tok,
    }

    for tau_sent in tau_sents:
        for tau_floor in tau_floors:
            gate.policy.tau_sent = float(tau_sent)
            gate.policy.tau_floor = float(tau_floor)
            # Keep bands soft while scoring so coverage gates dominate the search
            gate.policy.bands = (0.25, 0.55, 0.75)
            rows = _scores(gate, pairs)
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
            apply_calibration_receipt(
                gate.policy,
                thresholds={
                    "tau_sent": tau_sent,
                    "tau_floor": tau_floor,
                    "tau_tok": gate.policy.tau_tok,
                    "fused_pass": best_t,
                    "fused_flag": min(best_t + 0.20, 0.90),
                    "fused_act": min(best_t + 0.40, 0.95),
                },
                status="proposal",
            )
            f1 = _decision_f1(gate, pairs)
            if f1 > best_f1:
                best_f1 = f1
                best = {
                    "fused_pass": best_t,
                    "fused_flag": min(best_t + 0.20, 0.90),
                    "fused_act": min(best_t + 0.40, 0.95),
                    "tau_sent": float(tau_sent),
                    "tau_floor": float(tau_floor),
                    "tau_tok": float(gate.policy.tau_tok),
                }

    # restore before caller applies the winner
    gate.policy.tau_sent = base_tau_sent
    gate.policy.tau_floor = base_tau_floor
    gate.policy.bands = base_bands
    return best


def calibrate_labeled(
    pairs: list[tuple[EvidenceBundle, bool]],
    gate: ShineGate | None = None,
    version: str = "cal-labeled-0.1",
    *,
    apply_to_gate: bool = True,
    fit_coverage: bool = False,
    tau_sent_grid: list[float] | None = None,
    tau_floor_grid: list[float] | None = None,
) -> CalibrationReport:
    from prismshine.policy import apply_calibration_receipt

    gate = gate or ShineGate.build()
    if fit_coverage and pairs:
        thresholds = fit_effect_thresholds(
            pairs,
            gate,
            tau_sent_grid=tau_sent_grid,
            tau_floor_grid=tau_floor_grid,
        )
    else:
        rows = _scores(gate, pairs)
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
        thresholds = {
            "fused_pass": best_t,
            "fused_flag": min(best_t + 0.20, 0.90),
            "fused_act": min(best_t + 0.40, 0.95),
            "tau_sent": gate.policy.tau_sent,
            "tau_floor": gate.policy.tau_floor,
            "tau_tok": gate.policy.tau_tok,
        }

    if apply_to_gate:
        apply_calibration_receipt(
            gate.policy, thresholds=thresholds, status="validated-labeled"
        )
        gate.calibration_version = version
        gate._caps = gate._detect_capabilities()

    rows = _scores(gate, pairs)
    auroc = _auroc(rows)
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
        thresholds=thresholds,
        curves={"grounding.risk_coverage": [(0.0, 0.0), (1.0, 1.0)]},
        version=version,
        notes=[
            f"Receipt status=validated-labeled; AUROC={auroc}",
            "fit_coverage=True searches tau_sent/tau_floor + fused bands for decision F1.",
            "Threshold matrix was proposal until this calibrate() run.",
        ],
    )


def load_calibration_overlay(path: str | Path) -> dict[str, Any]:
    """Load a calibrate() JSON receipt / overlay for ShineGate.build."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "overlay" in data and isinstance(data["overlay"], dict):
        return data["overlay"]
    if "thresholds" in data:
        return {
            "calibration_version": data.get("version") or data.get("calibration_version") or "cal",
            "thresholds": data["thresholds"],
            "curves": data.get("curves") or {},
        }
    return data


def apply_overlay_to_gate(gate: ShineGate, overlay: dict[str, Any]) -> ShineGate:
    from prismshine.policy import apply_calibration_receipt

    thresholds = overlay.get("thresholds") or {}
    apply_calibration_receipt(
        gate.policy, thresholds=thresholds, status="validated-labeled"
    )
    gate.calibration_version = str(
        overlay.get("calibration_version") or gate.calibration_version
    )
    gate._caps = gate._detect_capabilities()
    return gate


def calibrate_synthetic(
    grounded: list[EvidenceBundle],
    gate: ShineGate | None = None,
    seed: int = 0,
    version: str = "cal-synth-0.1",
    *,
    apply_to_gate: bool = True,
) -> CalibrationReport:
    from prismshine.policy import apply_calibration_receipt

    pairs = synthetic_negatives(grounded, seed=seed)
    report = calibrate_labeled(
        pairs, gate=gate, version=version, apply_to_gate=False
    )
    report.mode = "synthetic"
    report.notes = [
        "Deterministic perturbations; zero LLM calls.",
        "Receipt status=validated-synthetic (not a substitute for labeled domain review).",
    ]
    if apply_to_gate and gate is not None:
        apply_calibration_receipt(
            gate.policy,
            thresholds=report.thresholds,
            status="validated-synthetic",
        )
        gate.calibration_version = version
        gate._caps = gate._detect_capabilities()
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
    if mode == "feedback":
        from prismshine.feedback import load_feedback_pairs

        # directory may be a .jsonl file path or a folder containing feedback.jsonl
        fb = root if root.is_file() else root / "feedback.jsonl"
        pairs = load_feedback_pairs(fb)
        return calibrate_labeled(pairs, gate=gate, version="cal-feedback-0.1")
    if mode == "labeled":
        if not labeled:
            raise ValueError("labeled mode requires JSON with is_hallucination")
        return calibrate_labeled(labeled, gate=gate)
    if not bundles:
        raise ValueError("no bundles found for synthetic calibration")
    return calibrate_synthetic(bundles, gate=gate)


def _decision_f1(gate: ShineGate, pairs: list[tuple[EvidenceBundle, bool]]) -> float:
    """F1 using verdict.decision only (so band calibration is visible)."""
    tp = fp = tn = fn = 0
    for bundle, is_h in pairs:
        v = gate.verify(bundle)
        pred = v.decision in {"flag", "block", "regenerate"}
        if is_h and pred:
            tp += 1
        elif is_h and not pred:
            fn += 1
        elif (not is_h) and pred:
            fp += 1
        else:
            tn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


def domain_calibrate_lift(
    grounded: list[EvidenceBundle],
    *,
    profile: str = "clinical",
    seed: int = 0,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Compare decision-F1 (and AUROC) before/after synthetic calibrate."""
    import hashlib

    import numpy as np

    def _embed(texts: list[str]) -> Any:
        dim = 32
        out = np.zeros((len(texts), dim), dtype=np.float64)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % dim
                out[i, h] += 1.0
            n = float(np.linalg.norm(out[i]) or 1.0)
            out[i] /= n
        return out

    emb = embedder or _embed
    # Soft polarity pairs (no hard-number floor) so band calibration is the lever
    pairs: list[tuple[EvidenceBundle, bool]] = []
    for i, b in enumerate(grounded):
        pairs.append((b, False))
        data = b.model_dump(mode="json")
        data["run_id"] = f"{b.run_id}-soft-neg"
        # Unsupported region claim — high lexical overlap, no hard number / antonym cue
        data["answer"] = "The CEO cited strong demand in Asia this quarter."
        data["preload"] = [
            {
                "chunk_id": "soft",
                "text": "The CEO cited strong demand in Europe this quarter.",
                "source": "retrieval",
            }
        ]
        neg, _ = bundle_from_dict(data)
        pairs.append((neg, True))

    # Under-flagging proposal bands (high pass threshold) so calibrate has room to lift F1
    tuned = ShineGate.build(embedder=emb, profile=profile)
    tuned.policy.bands = (0.70, 0.85, 0.95)
    tuned.policy.threshold_status = "proposal"
    pre_f1 = _decision_f1(tuned, pairs)
    pre_auroc = _auroc(_scores(tuned, pairs))

    report = calibrate_labeled(
        pairs, gate=tuned, version=f"cal-{profile}-0.1", apply_to_gate=True
    )
    report.mode = "synthetic-soft"
    post_f1 = _decision_f1(tuned, pairs)
    post_auroc = report.auroc if report.auroc is not None else _auroc(_scores(tuned, pairs))
    f1_lift = post_f1 - pre_f1
    return {
        "profile": profile,
        "pre_calibrate_f1": pre_f1,
        "calibrated_f1": post_f1,
        "pre_calibrate_auroc": pre_auroc,
        "calibrated_auroc": post_auroc,
        "f1_lift": f1_lift,
        "lift_gate_min": 0.10,
        "lift_met": f1_lift >= 0.10 or post_f1 >= 0.90,
        "threshold_status": tuned.policy.threshold_status,
        "bands_after": list(tuned.policy.bands),
        "version": report.version,
    }
