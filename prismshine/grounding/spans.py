"""Tier 3: ONNX token-classification span detector (LettuceDetect-class)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prismshine.models import EvidenceBundle, Signal, Span

logger = logging.getLogger(__name__)

# Candidate hubs (first successful ONNX+tokenizer wins). LettuceDetect-class.
DEFAULT_MODEL_CANDIDATES = (
    "Kriso/lettuce-detect-base",
    "lettucedetect/lettucedetect-base-modernbert",
)
DEFAULT_ARTIFACT = "lettucedetect-onnx-v1"


@dataclass
class SpanResult:
    unsupported_span_ratio: float
    spans: list[Span] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    artifact_id: str | None = None
    available: bool = False
    backend: str = "unavailable"  # onnx | lexical | unavailable


class SpanClassifier:
    """Lazy-loaded ONNX span classifier with honest backend reporting."""

    def __init__(
        self,
        model_id: str | None = None,
        tau_tok: float = 0.5,
        cache_dir: str | Path | None = None,
        allow_lexical_fallback: bool = True,
    ) -> None:
        self.model_id = model_id or DEFAULT_MODEL_CANDIDATES[0]
        self.model_candidates = (
            (model_id,) if model_id else DEFAULT_MODEL_CANDIDATES
        )
        self.tau_tok = tau_tok
        self.cache_dir = Path(cache_dir or Path.home() / ".prismshine" / "models")
        self.allow_lexical_fallback = allow_lexical_fallback
        self._session: Any = None
        self._tokenizer: Any = None
        self.artifact_id: str | None = None
        self._load_error: str | None = None
        self._backend: str = "unavailable"

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    @property
    def backend(self) -> str:
        self._ensure_loaded()
        return self._backend

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self._load_error and not self.allow_lexical_fallback:
            return False
        try:
            import onnxruntime as ort  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from tokenizers import Tokenizer  # type: ignore
        except ImportError as exc:
            self._load_error = f"spans extra missing: {exc}"
            self._backend = "unavailable"
            logger.debug(self._load_error)
            return False

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            for repo in self.model_candidates:
                onnx_path = None
                for candidate in (
                    "model.onnx",
                    "onnx/model.onnx",
                    "lettucedetect.onnx",
                    "model_quantized.onnx",
                ):
                    try:
                        onnx_path = hf_hub_download(
                            repo_id=repo,
                            filename=candidate,
                            cache_dir=str(self.cache_dir),
                        )
                        break
                    except Exception:  # noqa: BLE001
                        continue
                tok_path = None
                for candidate in ("tokenizer.json",):
                    try:
                        tok_path = hf_hub_download(
                            repo_id=repo,
                            filename=candidate,
                            cache_dir=str(self.cache_dir),
                        )
                        break
                    except Exception:  # noqa: BLE001
                        continue
                if onnx_path and tok_path:
                    self._session = ort.InferenceSession(
                        onnx_path, providers=["CPUExecutionProvider"]
                    )
                    self._tokenizer = Tokenizer.from_file(tok_path)
                    self.model_id = repo
                    self.artifact_id = f"{repo}@{Path(onnx_path).name}"
                    self._backend = "onnx"
                    return True

            if self.allow_lexical_fallback:
                # Honest degradation: deterministic lexical unsupported detector.
                # capabilities() must report span_backend=lexical (not onnx).
                self._session = "lexical"
                self.artifact_id = DEFAULT_ARTIFACT + "+lexical"
                self._backend = "lexical"
                self._load_error = (
                    "no ONNX LettuceDetect artifact on hub; using lexical Tier-3 backend"
                )
                logger.warning(self._load_error)
                return True

            self._load_error = "no ONNX span model available"
            self._backend = "unavailable"
            return False
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            self._backend = "unavailable"
            logger.debug("span classifier load failed: %s", exc)
            return False

    def classify(
        self,
        bundle: EvidenceBundle,
        *,
        candidate_spans: list[Span] | None = None,
        tau_tok: float | None = None,
    ) -> SpanResult:
        if not bundle.answer:
            return SpanResult(
                unsupported_span_ratio=0.0, available=False, backend="unavailable"
            )
        if not self._ensure_loaded():
            return SpanResult(
                unsupported_span_ratio=0.0, available=False, backend="unavailable"
            )

        thr = self.tau_tok if tau_tok is None else tau_tok
        preload = "\n".join(c.text for c in bundle.preload)
        answer = bundle.answer

        if self._session == "lexical":
            spans = self._lexical_unsupported(answer, preload, candidate_spans)
        else:
            spans = self._onnx_unsupported(answer, preload, thr, candidate_spans)

        unsupported_chars = sum(max(0, s.end - s.start) for s in spans)
        ratio = unsupported_chars / max(len(answer), 1)
        signals = [
            Signal(
                name="grounding.unsupported_span_ratio",
                tier=3,
                value=ratio,
                weight=0.35,
                spans=spans,
                detail={
                    "artifact_id": self.artifact_id,
                    "tau_tok": thr,
                    "backend": self._backend,
                },
            )
        ]
        return SpanResult(
            unsupported_span_ratio=ratio,
            spans=spans,
            signals=signals,
            artifact_id=self.artifact_id,
            available=True,
            backend=self._backend,
        )

    def _lexical_unsupported(
        self,
        answer: str,
        preload: str,
        candidates: list[Span] | None,
    ) -> list[Span]:
        """Token-level unsupported: answer tokens absent from preload (for candidates)."""
        preload_l = preload.lower()
        spans: list[Span] = []
        if candidates:
            for cand in candidates:
                tokens = cand.text.split()
                missing = [t for t in tokens if len(t) > 3 and t.lower() not in preload_l]
                if missing:
                    spans.append(
                        Span(
                            start=cand.start,
                            end=cand.end,
                            text=cand.text,
                            reason="unsupported_span",
                            tier=3,
                        )
                    )
        else:
            # whole-answer scan for content words missing from preload
            import re

            for m in re.finditer(r"[A-Za-z]{4,}", answer):
                tok = m.group(0)
                if tok.lower() not in preload_l:
                    spans.append(
                        Span(
                            start=m.start(),
                            end=m.end(),
                            text=tok,
                            reason="unsupported_span",
                            tier=3,
                        )
                    )
        return spans

    def _onnx_unsupported(
        self,
        answer: str,
        preload: str,
        thr: float,
        candidates: list[Span] | None,
    ) -> list[Span]:
        import numpy as np

        # Chunk preload to fit context window
        max_chars = 6000
        context = preload[:max_chars]
        text = f"{context}\n\n{answer}"
        encoded = self._tokenizer.encode(text)
        ids = encoded.ids
        # Run model — output shape assumed [seq, 2] or [1, seq, 2] logits/probs
        input_name = self._session.get_inputs()[0].name
        arr = np.asarray([ids], dtype=np.int64)
        feeds = {input_name: arr}
        # optional attention mask
        for inp in self._session.get_inputs()[1:]:
            if "mask" in inp.name.lower():
                feeds[inp.name] = np.ones_like(arr)
            elif "type" in inp.name.lower():
                feeds[inp.name] = np.zeros_like(arr)
        outs = self._session.run(None, feeds)
        logits = np.asarray(outs[0])
        if logits.ndim == 3:
            logits = logits[0]
        if logits.shape[-1] >= 2:
            # softmax over last dim, take unsupported class (index 1)
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = e / e.sum(axis=-1, keepdims=True)
            uns = probs[:, 1]
        else:
            uns = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))

        # Map tokens back to answer char spans — approximate via offsets
        offsets = encoded.offsets if hasattr(encoded, "offsets") else [(0, 0)] * len(ids)
        # Find answer region start in text
        ans_start_in_text = text.rfind(answer)
        spans: list[Span] = []
        in_span = False
        start_char = 0
        for i, p in enumerate(uns):
            if i >= len(offsets):
                break
            off = offsets[i]
            # only consider tokens inside answer
            if off[0] < ans_start_in_text:
                continue
            rel_start = off[0] - ans_start_in_text
            if p >= thr:
                if not in_span:
                    in_span = True
                    start_char = max(0, rel_start)
            elif in_span:
                in_span = False
                end_char = max(start_char, rel_start)
                frag = answer[start_char:end_char]
                if frag.strip():
                    spans.append(
                        Span(
                            start=start_char,
                            end=end_char,
                            text=frag,
                            reason="unsupported_span",
                            tier=3,
                        )
                    )
        if in_span:
            frag = answer[start_char:]
            if frag.strip():
                spans.append(
                    Span(
                        start=start_char,
                        end=len(answer),
                        text=frag,
                        reason="unsupported_span",
                        tier=3,
                    )
                )

        if candidates:
            # keep spans overlapping candidates
            filtered: list[Span] = []
            for s in spans:
                for c in candidates:
                    if s.start < c.end and s.end > c.start:
                        filtered.append(s)
                        break
            spans = filtered or spans
        return spans
