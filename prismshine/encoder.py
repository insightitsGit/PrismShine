"""SharedEncoder: user embedder → prismlang session → lexical fallback."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from prismshine.models import EvidenceBundle, PreloadChunk

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], np.ndarray]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def _hash_embed(texts: list[str], dim: int = 64) -> np.ndarray:
    """Deterministic lexical pseudo-embeddings (no network, no model)."""
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        tokens = text.lower().split()
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            out[i, idx] += sign
    return _l2_normalize(out)


class SharedEncoder:
    def __init__(
        self,
        embedder: Embedder | None = None,
        prefer_prismlang: bool = True,
    ) -> None:
        self._user_embedder = embedder
        self._memo: dict[str, np.ndarray] = {}
        self._mode = "lexical"
        self._model_id: str | None = None
        self._session: Any = None

        if embedder is not None:
            self._mode = "user-embedder"
            self._model_id = "user-embedder"
            return

        if prefer_prismlang:
            try:
                from prismlang import encoder as pl_encoder  # type: ignore

                self._session = pl_encoder.get_session()
                self._model_id = (
                    pl_encoder.model_id()
                    if hasattr(pl_encoder, "model_id")
                    else "prismlang-minilm"
                )
                self._mode = "raw-384"
            except Exception as exc:  # noqa: BLE001
                logger.debug("prismlang encoder unavailable: %s", exc)
                self._mode = "lexical"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def _sentence_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)

        results: list[np.ndarray | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            key = self._sentence_key(t)
            if key in self._memo:
                results[i] = self._memo[key]
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            encoded = self._encode_batch(missing_texts)
            for j, idx in enumerate(missing_idx):
                vec = encoded[j]
                self._memo[self._sentence_key(texts[idx])] = vec
                results[idx] = vec

        return np.stack([r for r in results if r is not None], axis=0)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        if self._user_embedder is not None:
            arr = np.asarray(self._user_embedder(texts), dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return _l2_normalize(arr)

        if self._session is not None:
            try:
                from prismlang import encoder as pl_encoder  # type: ignore

                if hasattr(pl_encoder, "encode_batch"):
                    arr = np.asarray(pl_encoder.encode_batch(texts), dtype=np.float64)
                else:
                    arr = np.asarray(
                        [pl_encoder.encode(t) for t in texts], dtype=np.float64
                    )
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return _l2_normalize(arr)
            except Exception as exc:  # noqa: BLE001
                logger.debug("prismlang encode failed, lexical fallback: %s", exc)
                self._mode = "lexical"

        return _hash_embed(texts)

    def ensure_chunk_vectors(self, bundle: EvidenceBundle) -> EvidenceBundle:
        """Encode chunks missing vectors; write back into the bundle."""
        need: list[tuple[int, PreloadChunk]] = [
            (i, c) for i, c in enumerate(bundle.preload) if c.vector is None
        ]
        if not need:
            return bundle
        texts = [c.text for _, c in need]
        vectors = self.encode(texts)
        for (i, chunk), vec in zip(need, vectors, strict=True):
            space = self._mode if self._mode != "lexical" else "none"
            if self._model_id and space == "raw-384":
                space = f"raw-384@{self._model_id}"
            elif self._mode == "user-embedder":
                space = "raw-384@user-embedder"
            bundle.preload[i] = chunk.model_copy(
                update={
                    "vector": vec.astype(float).tolist(),
                    "vector_space": space if space != "none" else chunk.vector_space,
                }
            )
        return bundle
