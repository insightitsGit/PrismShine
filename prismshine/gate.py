"""ShineGate: pipeline orchestration, early exits, capability detection."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prismshine.audit import AuditLog, new_verdict_id
from prismshine.cache import (
    MemoryVerdictStore,
    SqliteVerdictStore,
    TieredVerdictStore,
    VerdictStore,
    verdict_cache_key,
)
from prismshine.config import ShineConfig, get_config
from prismshine.encoder import Embedder, SharedEncoder
from prismshine.forensics.engine import run_forensics
from prismshine.fusion import fuse
from prismshine.grounding.copycheck import copycheck
from prismshine.grounding.coverage import coverage_check
from prismshine.grounding.judge import EscalationBudget, Judge, build_judge
from prismshine.grounding.spans import SpanClassifier
from prismshine.grounding.splitter import split_sentences
from prismshine.handbook.loader import load_handbook
from prismshine.handbook.schema import Handbook
from prismshine.hashing import content_hash, evidence_hash
from prismshine.models import EvidenceBundle, ShineVerdict, Signal, Strictness
from prismshine.policy import EffectivePolicy, resolve_policy

logger = logging.getLogger(__name__)


@dataclass
class Capabilities:
    tiers: dict[str, bool]
    coverage_mode: str
    encoder_model_id: str | None
    span_classifier: bool
    judge: bool
    handbook_version: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tiers": self.tiers,
            "coverage_mode": self.coverage_mode,
            "encoder_model_id": self.encoder_model_id,
            "span_classifier": self.span_classifier,
            "judge": self.judge,
            "handbook_version": self.handbook_version,
            "notes": self.notes,
        }


class ShineGate:
    def __init__(
        self,
        handbook: Handbook,
        policy: EffectivePolicy,
        encoder: SharedEncoder,
        *,
        judge: Judge | None = None,
        span_classifier: SpanClassifier | None = None,
        verdict_store: VerdictStore | None = None,
        config: ShineConfig | None = None,
        calibration_version: str = "identity-0",
        calibration_curves: dict[str, list[tuple[float, float]]] | None = None,
    ) -> None:
        self.handbook = handbook
        self.policy = policy
        self.encoder = encoder
        self.judge = judge
        self.span_classifier = span_classifier
        self.verdict_store = verdict_store or MemoryVerdictStore()
        self.config = config or get_config()
        self.calibration_version = calibration_version
        self.calibration_curves = calibration_curves or {}
        self.audit = AuditLog()
        self.budget = EscalationBudget(policy.tier4_budget)
        self._tier0_cache: dict[str, Any] = {}
        self._caps = self._detect_capabilities()
        logger.info("ShineGate capabilities: %s", self._caps.as_dict())

    @classmethod
    def build(
        cls,
        profile: str = "default",
        handbook: str | Path | list[str | Path] | None = "builtin",
        judge: Judge | str | None = None,
        embedder: Embedder | None = None,
        verdict_store: VerdictStore | None = None,
        strictness: Strictness = "standard",
        overrides: dict[str, Any] | None = None,
        config: ShineConfig | None = None,
    ) -> ShineGate:
        cfg = (config or get_config()).merge(
            {
                "profile": profile,
                "strictness": strictness,
            }
        )
        if handbook is None or handbook == "builtin":
            hb = load_handbook()
        elif isinstance(handbook, (str, Path)):
            if str(handbook) == "builtin":
                hb = load_handbook()
            else:
                hb = load_handbook(handbook)
        else:
            hb = load_handbook(*handbook)

        if cfg.handbook_path:
            hb = load_handbook(cfg.handbook_path)

        pol = resolve_policy(
            profile=cfg.profile,
            strictness=cfg.strictness,
            overrides=overrides,
            halt_on_fatal=cfg.halt_on_fatal,
        )
        encoder = SharedEncoder(embedder=embedder)
        judge_impl: Judge | None
        if callable(judge) and not isinstance(judge, str):
            judge_impl = judge  # type: ignore[assignment]
        elif isinstance(judge, str):
            judge_impl = build_judge(judge)
        else:
            judge_impl = build_judge(cfg.judge_provider, cfg.judge_model)

        span = None if cfg.disable_tier3 else SpanClassifier(tau_tok=pol.tau_tok)

        store = verdict_store
        if store is None and cfg.verdict_db:
            store = TieredVerdictStore(sqlite=SqliteVerdictStore(cfg.verdict_db))

        return cls(
            handbook=hb,
            policy=pol,
            encoder=encoder,
            judge=judge_impl,
            span_classifier=span,
            verdict_store=store,
            config=cfg,
        )

    def _detect_capabilities(self) -> Capabilities:
        notes: list[str] = []
        span_ok = bool(self.span_classifier and self.span_classifier.available)
        if self.span_classifier and not span_ok:
            notes.append("Tier 3 unavailable -> unresolved gray resolves to flag")
        if self.encoder.mode == "lexical":
            notes.append("coverage_mode=lexical (no prismlang/user embedder)")
            notes.append("JL fallback threshold 0.80 needs calibration")
        if self.judge is None:
            notes.append("no judge configured -> gray cannot escalate to Tier 4")
        return Capabilities(
            tiers={
                "0": True,
                "1": True,
                "2": True,
                "3": span_ok,
                "4": self.judge is not None,
            },
            coverage_mode=self.encoder.mode,
            encoder_model_id=self.encoder.model_id,
            span_classifier=span_ok,
            judge=self.judge is not None,
            handbook_version=self.handbook.handbook_version,
            notes=notes,
        )

    def capabilities(self) -> dict[str, Any]:
        return self._caps.as_dict()

    def _cache_key(self, bundle: EvidenceBundle) -> str:
        preload_hash = content_hash(
            [{"id": c.chunk_id, "text": c.text} for c in bundle.preload]
        )
        artifacts = [
            a
            for a in [
                self.encoder.model_id,
                self.span_classifier.artifact_id if self.span_classifier else None,
            ]
            if a
        ]
        return verdict_cache_key(
            preload_ids=[c.chunk_id for c in bundle.preload],
            preload_hash=preload_hash,
            answer_norm=(bundle.answer or "").strip(),
            profile_id=self.policy.profile,
            handbook_version=self.handbook.handbook_version,
            calibration_version=self.calibration_version,
            model_artifact_ids=artifacts,
        )

    def verify(self, bundle: EvidenceBundle) -> ShineVerdict:
        key = self._cache_key(bundle)
        cached = self.verdict_store.get(key)
        if cached is not None:
            return cached

        ehash = evidence_hash(bundle)
        # Dynamic strictness bump
        dynamic = 0
        pre_forensics = run_forensics(bundle, self.handbook)
        if any(h.id == "GUARD_GRAY_INPUT" for h in pre_forensics.hits):
            dynamic += 1
        phase = bundle.node_state.get("resonance_phase") or bundle.node_state.get("phase")
        if str(phase).upper() in {"EMERGENCY", "ALERT"}:
            dynamic += 1
        policy = resolve_policy(
            profile=self.policy.profile,
            strictness=self.policy.strictness,
            overrides=self.policy.extras or None,
            dynamic_bump=dynamic,
            halt_on_fatal=self.policy.halt_on_fatal,
        )
        # carry weights
        policy.weights = dict(self.policy.weights)

        signals: list[Signal] = []
        spans = []
        tier_reached = 0
        coverage_mode = "skipped"

        # --- Tier 0 ---
        # Reuse cached tier0 by evidence hash of preload-only view
        preload_key = content_hash(
            {
                "q": bundle.question,
                "preload": [c.model_dump(mode="json") for c in bundle.preload],
                "trace": [t.model_dump(mode="json") for t in bundle.trace],
                "state": bundle.node_state,
            }
        )
        if preload_key in self._tier0_cache:
            forensics = self._tier0_cache[preload_key]
        else:
            forensics = pre_forensics
            self._tier0_cache[preload_key] = forensics
        signals.extend(forensics.signals)
        tier_reached = 0

        # Pre-generation mode
        if bundle.answer is None:
            verdict = self._preload_verdict(bundle, forensics, policy, ehash, tier_reached)
            self.verdict_store.put(key, verdict)
            self.audit.record(bundle, verdict)
            return verdict

        # Fatal early exit
        if forensics.fatal and policy.halt_on_fatal:
            fatal = next(h for h in forensics.hits if h.severity == "fatal")
            fused = fuse(
                signals,
                forensics.hits,
                policy,
                has_fatal_halt=True,
                early_gate=f"HANDBOOK:{fatal.id}",
            )
            verdict = ShineVerdict(
                decision=fused.decision,
                resolution_gate=fused.resolution_gate,
                fused_score=fused.fused_score,
                confidence=fused.confidence,
                signatures=forensics.hits,
                spans=[],
                tier_reached=0,
                coverage_mode="skipped",
                strictness_effective=policy.strictness_effective,
                dormant_families=forensics.dormant_families,
                evidence_hash=ehash,
                verdict_id=new_verdict_id(),
                signals=signals,
                advice=[h.advice for h in forensics.hits if h.advice],
            )
            self.verdict_store.put(key, verdict)
            self.audit.record(bundle, verdict)
            return verdict

        # --- Tier 1 ---
        tier_reached = 1
        cc = copycheck(
            bundle,
            numeric_tolerance=policy.numeric_tolerance,
            escalate_derived=policy.escalate_derived,
        )
        signals.extend(cc.signals)
        spans.extend(cc.spans)

        # --- Tier 2 ---
        if not policy.jl_allowed and any(
            c.vector_space.startswith("jl-64") for c in bundle.preload
        ):
            # clinical: refuse JL → escalate path via lexical + flag pressure
            coverage_mode = "lexical"
        cov = coverage_check(
            bundle,
            self.encoder,
            tau_sent=policy.tau_sent,
            tau_sent_jl=policy.tau_sent_jl,
        )
        tier_reached = 2
        coverage_mode = cov.coverage_mode
        signals.extend(cov.signals)
        spans.extend(cov.uncovered_spans)
        spans.extend([c.span for c in cov.contradiction_cues])

        # Fast path: Tier0 clean + Tier1 zero unmatched + high coverage, no cues
        tier0_clean = not any(h.severity in {"fatal", "error"} for h in forensics.hits)
        coverage_pass_threshold = max(0.75, 1.0 - policy.bands[0])
        if (
            tier0_clean
            and cc.unmatched_ratio == 0.0
            and not cov.contradiction_cues
            and cov.coverage >= coverage_pass_threshold
        ):
            verdict = ShineVerdict(
                decision="pass",
                resolution_gate="CLEAN_FAST_PATH",
                fused_score=max(0.0, cov.risk_coverage * 0.25),
                confidence=0.9,
                signatures=forensics.hits,
                spans=spans,
                tier_reached=2,
                coverage_mode=coverage_mode,
                strictness_effective=policy.strictness_effective,
                dormant_families=forensics.dormant_families,
                evidence_hash=ehash,
                verdict_id=new_verdict_id(),
                signals=signals,
                advice=[h.advice for h in forensics.hits if h.advice],
            )
            self.verdict_store.put(key, verdict)
            self.audit.record(bundle, verdict)
            return verdict

        # Coverage collapse
        if cov.coverage < policy.tau_floor:
            if tier0_clean:
                fused = fuse(signals, forensics.hits, policy)
                verdict = ShineVerdict(
                    decision="block" if fused.fused_score >= policy.bands[2] else "flag",
                    resolution_gate="T2_COVERAGE_COLLAPSE",
                    fused_score=max(fused.fused_score, 0.75),
                    confidence=0.85,
                    signatures=forensics.hits,
                    spans=spans,
                    tier_reached=2,
                    coverage_mode=coverage_mode,
                    strictness_effective=policy.strictness_effective,
                    dormant_families=forensics.dormant_families,
                    evidence_hash=ehash,
                    verdict_id=new_verdict_id(),
                    signals=signals,
                    advice=[h.advice for h in forensics.hits if h.advice]
                    + ["Answer coverage collapsed against a healthy preload."],
                )
                self.verdict_store.put(key, verdict)
                self.audit.record(bundle, verdict)
                return verdict
            # unhealthy tier0 — attribute to forensic gate
            top = next(
                (h for h in forensics.hits if h.severity in {"fatal", "error"}),
                forensics.hits[0] if forensics.hits else None,
            )
            gate = f"HANDBOOK:{top.id}" if top else "T0_UNHEALTHY"
            verdict = ShineVerdict(
                decision="block" if top and top.severity == "fatal" else "flag",
                resolution_gate=gate,
                fused_score=0.8,
                confidence=0.8,
                signatures=forensics.hits,
                spans=spans,
                tier_reached=2,
                coverage_mode=coverage_mode,
                strictness_effective=policy.strictness_effective,
                dormant_families=forensics.dormant_families,
                evidence_hash=ehash,
                verdict_id=new_verdict_id(),
                signals=signals,
                advice=[h.advice for h in forensics.hits if h.advice],
            )
            self.verdict_store.put(key, verdict)
            self.audit.record(bundle, verdict)
            return verdict

        # Fusion probe for gray
        probe = fuse(
            signals,
            forensics.hits,
            policy,
            calibration=self.calibration_curves,
        )
        need_t3 = (
            probe.band == "gray"
            or policy.mandatory_tier3
            or bool(cov.mandatory_tier3)
            or bool(cov.contradiction_cues)
        )

        # --- Tier 3 ---
        gray_unresolved = False
        if need_t3:
            if self.span_classifier and self.span_classifier.available:
                tier_reached = 3
                sr = self.span_classifier.classify(
                    bundle,
                    candidate_spans=cov.mandatory_tier3 + cov.uncovered_spans + cc.spans,
                    tau_tok=policy.tau_tok,
                )
                signals.extend(sr.signals)
                spans.extend(sr.spans)
            else:
                gray_unresolved = True

        # Re-probe
        probe = fuse(
            signals,
            forensics.hits,
            policy,
            calibration=self.calibration_curves,
            gray_unresolved=gray_unresolved and self.judge is None,
        )

        # --- Tier 4 ---
        judge_present = False
        if probe.band == "gray" and self.judge is not None and self.budget.allow():
            tier_reached = 4
            judge_present = True
            claims = split_sentences(bundle.answer)
            context = "\n".join(c.text for c in bundle.preload)
            result = self.judge(claims, context)
            signals.append(
                Signal(
                    name="grounding.judge_risk",
                    tier=4,
                    value=result.risk,
                    weight=0.45,
                    detail={"claims": result.claim_support},
                )
            )
        elif probe.band == "gray" and self.judge is None:
            gray_unresolved = True

        fused = fuse(
            signals,
            forensics.hits,
            policy,
            calibration=self.calibration_curves,
            gray_unresolved=gray_unresolved,
            judge_present=judge_present,
        )

        # ADR-11: missing capability must not pass
        decision = fused.decision
        gate = fused.resolution_gate
        if gray_unresolved and decision == "pass":
            decision = "flag"
            gate = "MISSING_CAPABILITY_FLAG"

        advice = [h.advice for h in forensics.hits if h.advice]
        if decision == "regenerate":
            advice.append(
                "Regenerate with feedback: unsupported spans + signature advice (bounded 1 attempt)."
            )

        verdict = ShineVerdict(
            decision=decision,
            resolution_gate=gate,
            fused_score=fused.fused_score,
            confidence=fused.confidence,
            signatures=forensics.hits,
            spans=spans,
            tier_reached=tier_reached,
            coverage_mode=coverage_mode,
            strictness_effective=policy.strictness_effective,
            dormant_families=forensics.dormant_families,
            evidence_hash=ehash,
            verdict_id=new_verdict_id(),
            signals=signals,
            advice=advice,
        )
        self.verdict_store.put(key, verdict)
        self.audit.record(bundle, verdict)
        return verdict

    def _preload_verdict(
        self,
        bundle: EvidenceBundle,
        forensics: Any,
        policy: EffectivePolicy,
        ehash: str,
        tier_reached: int,
    ) -> ShineVerdict:
        fatal = [h for h in forensics.hits if h.severity == "fatal"]
        errors = [h for h in forensics.hits if h.severity == "error"]
        if fatal and policy.halt_on_fatal:
            decision = "block"
            gate = f"HANDBOOK:{fatal[0].id}"
            score = 1.0
        elif fatal:
            decision = "regenerate"
            gate = f"HANDBOOK:{fatal[0].id}"
            score = 0.9
        elif errors:
            decision = "flag"
            gate = f"HANDBOOK:{errors[0].id}"
            score = 0.6
        elif forensics.hits:
            decision = "flag" if any(h.severity == "warning" for h in forensics.hits) else "pass"
            gate = f"HANDBOOK:{forensics.hits[0].id}"
            score = 0.3 if decision == "flag" else 0.05
        else:
            decision = "pass"
            gate = "PRELOAD_CLEAN"
            score = 0.0
        return ShineVerdict(
            decision=decision,  # type: ignore[arg-type]
            resolution_gate=gate,
            fused_score=score,
            confidence=0.9,
            signatures=forensics.hits,
            spans=[],
            tier_reached=tier_reached,
            coverage_mode="skipped",
            strictness_effective=policy.strictness_effective,
            dormant_families=forensics.dormant_families,
            evidence_hash=ehash,
            verdict_id=new_verdict_id(),
            signals=forensics.signals,
            advice=[h.advice for h in forensics.hits if h.advice],
        )

    async def averify(self, bundle: EvidenceBundle) -> ShineVerdict:
        """Async twin — OFFNX work offloaded to a thread."""
        return await asyncio.to_thread(self.verify, bundle)
