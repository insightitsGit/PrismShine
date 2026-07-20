"""Tier 3: ONNX token-classification span detector (LettuceDetect-class)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prismshine.models import EvidenceBundle, Signal, Span

logger = logging.getLogger(__name__)

import os

# Candidate hubs (first successful ONNX+tokenizer wins). LettuceDetect-class.
# Pin via PRISMSHINE_SPAN_MODEL / PRISMSHINE_SPAN_ONNX for CI reproducibility (P3).
DEFAULT_MODEL_CANDIDATES = (
    "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    "Kriso/lettuce-detect-base",
    "lettucedetect/lettucedetect-base-modernbert",
)
DEFAULT_ARTIFACT = "lettucedetect-onnx-v1"
PINNED_ARTIFACT_ENV = "PRISMSHINE_SPAN_ONNX"
PINNED_MODEL_ENV = "PRISMSHINE_SPAN_MODEL"
PINNED_TOKENIZER_ENV = "PRISMSHINE_SPAN_TOKENIZER"


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
        env_model = os.environ.get(PINNED_MODEL_ENV)
        self.model_id = model_id or env_model or DEFAULT_MODEL_CANDIDATES[0]
        self.model_candidates = (
            (model_id,)
            if model_id
            else ((env_model,) if env_model else DEFAULT_MODEL_CANDIDATES)
        )
        self._pinned_onnx = os.environ.get(PINNED_ARTIFACT_ENV)
        self._pinned_tokenizer = os.environ.get(PINNED_TOKENIZER_ENV)
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
        if self._backend == "onnx" and self._session is not None and self._tokenizer is not None:
            return True
        if self._backend == "lexical" and self._session == "lexical":
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
            # Pinned local ONNX path (CI / air-gapped)
            if self._pinned_onnx and Path(self._pinned_onnx).is_file():
                try:
                    session = ort.InferenceSession(
                        self._pinned_onnx, providers=["CPUExecutionProvider"]
                    )
                    tokenizer = None
                    tok_path = None
                    for candidate in (
                        self._pinned_tokenizer,
                        str(Path(self._pinned_onnx).with_name("tokenizer.json")),
                        str(Path(self._pinned_onnx).parent / "tokenizer.json"),
                    ):
                        if candidate and Path(candidate).is_file():
                            tokenizer = Tokenizer.from_file(str(candidate))
                            tok_path = str(candidate)
                            break
                    if tokenizer is None:
                        for repo in self.model_candidates:
                            try:
                                tok_path = hf_hub_download(
                                    repo_id=repo,
                                    filename="tokenizer.json",
                                    cache_dir=str(self.cache_dir),
                                )
                                tokenizer = Tokenizer.from_file(tok_path)
                                break
                            except Exception:  # noqa: BLE001
                                continue
                    if session is not None and tokenizer is not None:
                        self._session = session
                        self._tokenizer = tokenizer
                        self.artifact_id = f"pinned@{Path(self._pinned_onnx).name}"
                        self._backend = "onnx"
                        return True
                    logger.warning(
                        "pinned ONNX at %s loaded but no tokenizer; "
                        "set PRISMSHINE_SPAN_TOKENIZER or place tokenizer.json beside the onnx",
                        self._pinned_onnx,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("pinned ONNX load failed: %s", exc)

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
                    session = ort.InferenceSession(
                        onnx_path, providers=["CPUExecutionProvider"]
                    )
                    tokenizer = Tokenizer.from_file(tok_path)
                    self._session = session
                    self._tokenizer = tokenizer
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
            try:
                spans = self._onnx_unsupported(answer, preload, thr, candidate_spans)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ONNX classify failed (%s); degrading to lexical", exc)
                if not self.allow_lexical_fallback:
                    raise
                spans = self._lexical_unsupported(answer, preload, candidate_spans)
                self._backend = "lexical"

        # Contradiction-cue candidates are mandatory Tier-3 evidence — never drop them
        # when the ONNX path returns empty / non-overlapping spans.
        if candidate_spans:
            have = {(s.start, s.end, s.text) for s in spans}
            for cand in candidate_spans:
                if "contradiction" not in (cand.reason or ""):
                    continue
                key = (cand.start, cand.end, cand.text)
                if key in have:
                    continue
                spans.append(
                    Span(
                        start=cand.start,
                        end=cand.end,
                        text=cand.text,
                        reason="unsupported_span:contradiction_cue",
                        tier=3,
                    )
                )

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
        """Token-level unsupported + treat contradiction-cue candidates as unsupported."""
        preload_l = preload.lower()
        spans: list[Span] = []
        if candidates:
            for cand in candidates:
                # Contradiction cues are mandatory Tier-3: never clear them lexically
                if "contradiction" in (cand.reason or ""):
                    spans.append(
                        Span(
                            start=cand.start,
                            end=cand.end,
                            text=cand.text,
                            reason="unsupported_span:contradiction_cue",
                            tier=3,
                        )
                    )
                    continue
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

    def _max_seq_tokens(self) -> int:
        if self._tokenizer is not None and hasattr(self._tokenizer, "get_tokenizer"):
            inner = self._tokenizer.get_tokenizer()
            if hasattr(inner, "model_max_length") and inner.model_max_length:
                return int(inner.model_max_length)
        return 512

    def _onnx_unsupported(
        self,
        answer: str,
        preload: str,
        thr: float,
        candidates: list[Span] | None,
    ) -> list[Span]:
        import numpy as np

        max_tokens = self._max_seq_tokens()
        sep = "\n\n"
        sep_ids = self._tokenizer.encode(sep).ids
        answer_ids = self._tokenizer.encode(answer).ids
        budget = max_tokens - len(answer_ids) - len(sep_ids) - 2
        if budget < 32:
            budget = max(32, max_tokens // 2)
            answer_ids = answer_ids[: max(1, max_tokens // 4)]
            answer = self._tokenizer.decode(answer_ids)
        ctx_ids = self._tokenizer.encode(preload).ids
        if len(ctx_ids) > budget:
            ctx_ids = ctx_ids[-budget:]
        context = self._tokenizer.decode(ctx_ids) if ctx_ids else ""
        text = f"{context}{sep}{answer}"
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
