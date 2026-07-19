"""Tier 2: vector/lexical coverage with composite support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from prismshine.encoder import SharedEncoder
from prismshine.grounding.contradiction import ContradictionCue, screen_contradictions
from prismshine.grounding.copycheck import extract_facts
from prismshine.grounding.splitter import is_boilerplate, is_composite, split_sentences
from prismshine.models import EvidenceBundle, Signal, Span

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]{1,}", re.I)
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "with",
        "as",
        "by",
        "from",
        "it",
        "its",
    }
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _token_overlap(a: str, b: str) -> float:
    ta = {t for t in a.lower().split() if len(t) > 2}
    tb = {t for t in b.lower().split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _norm_text(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _content_tokens(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s) if t.lower() not in _STOP]


def containment_support(sentence: str, chunk_texts: list[str]) -> float:
    """Lexical / substring support for extractive answers (HaluEval-style spans).

    Cosine on 1–4 word answers vs long context sentences is near-zero even when
    the answer is copied from preload. Credit exact / near-exact containment
    before applying τ_sent.
    """
    if not sentence or not chunk_texts:
        return 0.0
    sent = _norm_text(sentence)
    if not sent:
        return 0.0
    norms = [_norm_text(t) for t in chunk_texts]
    corpus = " ".join(norms)
    corpus_toks = set()
    for n in norms:
        corpus_toks.update(_content_tokens(n))

    # Full-span extractive hit (phrase appears as contiguous text)
    if len(sent) >= 2 and sent in corpus:
        return 1.0

    toks = _content_tokens(sentence)
    if not toks:
        return 0.0

    # Short answers: every content token must appear as a whole token in preload
    if len(toks) <= 6 or len(sent) <= 48:
        if all(tok in corpus_toks for tok in toks):
            return 1.0
        return 0.0

    # Longer answers: high token recall against a single chunk
    best = 0.0
    for chunk in chunk_texts:
        ctoks = set(_content_tokens(chunk))
        if not ctoks:
            continue
        hit = sum(1 for t in toks if t in ctoks)
        best = max(best, hit / len(toks))
    if best >= 0.85:
        return 1.0
    if best >= 0.70:
        return float(best)
    return 0.0


@dataclass
class CoverageResult:
    coverage: float
    risk_coverage: float
    coverage_mode: str
    uncovered_spans: list[Span] = field(default_factory=list)
    contradiction_cues: list[ContradictionCue] = field(default_factory=list)
    mandatory_tier3: list[Span] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    sentence_support: list[float] = field(default_factory=list)


def _chunk_matrix(
    bundle: EvidenceBundle, encoder: SharedEncoder
) -> tuple[np.ndarray, list[str], list[str], bool]:
    """Returns vectors, texts, ids, used_jl_only."""
    encoder.ensure_chunk_vectors(bundle)
    vecs: list[np.ndarray] = []
    texts: list[str] = []
    ids: list[str] = []
    jl_only = True
    for c in bundle.preload:
        if c.vector is None:
            continue
        v = np.asarray(c.vector, dtype=np.float64)
        vecs.append(v)
        texts.append(c.text)
        ids.append(c.chunk_id)
        if c.vector_space.startswith("raw-384") or (
            len(v) >= 300 and not c.vector_space.startswith("jl-64")
        ):
            jl_only = False
    if not vecs:
        return np.zeros((0, 1)), [], [], False
    # pad/truncate to common dim
    dim = max(v.shape[0] for v in vecs)
    mat = np.zeros((len(vecs), dim), dtype=np.float64)
    for i, v in enumerate(vecs):
        mat[i, : min(dim, v.shape[0])] = v[:dim]
    return mat, texts, ids, jl_only and any(
        c.vector_space.startswith("jl-64") for c in bundle.preload
    )


def _resonance_score(q: np.ndarray, p: np.ndarray, lam: float = 0.5) -> float:
    """Re⟨q,p⟩ − λ·|Im⟨q,p⟩| when phase metadata packs real/imag halves."""
    if q.shape[0] % 2 != 0 or p.shape[0] % 2 != 0:
        return _cosine(q, p)
    n = q.shape[0] // 2
    qr, qi = q[:n], q[n:]
    pr, pi = p[:n], p[n:]
    real = float(np.dot(qr, pr) + np.dot(qi, pi))
    imag = float(np.dot(qr, pi) - np.dot(qi, pr))
    return real - lam * abs(imag)


def coverage_check(
    bundle: EvidenceBundle,
    encoder: SharedEncoder,
    *,
    tau_sent: float = 0.62,
    tau_sent_jl: float = 0.80,
    top_k: int = 3,
    use_resonance: bool = False,
) -> CoverageResult:
    if not bundle.answer:
        return CoverageResult(coverage=1.0, risk_coverage=0.0, coverage_mode="skipped")

    sentences = split_sentences(bundle.answer)
    if not sentences:
        return CoverageResult(coverage=1.0, risk_coverage=0.0, coverage_mode=encoder.mode)

    # offsets
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for s in sentences:
        idx = bundle.answer.find(s, cursor)
        if idx < 0:
            idx = cursor
        offsets.append((idx, idx + len(s)))
        cursor = idx + len(s)

    mode = encoder.mode
    chunk_mat, chunk_texts, chunk_ids, jl_only = _chunk_matrix(bundle, encoder)
    # Always keep raw preload texts for containment (even if vectors missing)
    if not chunk_texts:
        chunk_texts = [c.text for c in bundle.preload if c.text]
        chunk_ids = [c.chunk_id for c in bundle.preload if c.text]
    thr = tau_sent_jl if jl_only else tau_sent
    containment_hits = 0
    if mode == "lexical" or chunk_mat.shape[0] == 0:
        mode = "lexical"
        supports: list[float] = []
        best: list[tuple[str, str]] = []
        for s in sentences:
            if not chunk_texts:
                supports.append(0.0)
                best.append(("", ""))
                continue
            scores = [_token_overlap(s, t) for t in chunk_texts]
            j = int(np.argmax(scores))
            support = scores[j]
            cont = containment_support(s, chunk_texts)
            if cont > support:
                containment_hits += 1
                support = cont
            supports.append(support)
            best.append((chunk_ids[j] if chunk_ids else "", chunk_texts[j]))
        # lexical uses stricter promotion: thr higher
        thr = max(thr, 0.35)
    else:
        sent_vecs = encoder.encode(sentences)
        # align dims
        dim = min(sent_vecs.shape[1], chunk_mat.shape[1])
        sv = sent_vecs[:, :dim]
        cm = chunk_mat[:, :dim]
        supports = []
        best = []
        for i, s in enumerate(sentences):
            if use_resonance and any(
                c.metadata.get("phase") is not None for c in bundle.preload
            ):
                scores = [_resonance_score(sv[i], cm[j]) for j in range(cm.shape[0])]
                if mode != "lexical":
                    mode = "resonance"
            else:
                scores = [_cosine(sv[i], cm[j]) for j in range(cm.shape[0])]
            j = int(np.argmax(scores))
            support = scores[j]
            if is_composite(s) and cm.shape[0] > 1:
                order = np.argsort(scores)[::-1][:top_k]
                combo = cm[order].sum(axis=0)
                norm = np.linalg.norm(combo)
                if norm > 1e-12:
                    combo = combo / norm
                    support = max(support, _cosine(sv[i], combo))
            cont = containment_support(s, chunk_texts)
            if cont > support:
                containment_hits += 1
                support = cont
            supports.append(support)
            best.append((chunk_ids[j], chunk_texts[j]))

    # weights
    answer_facts = extract_facts(bundle.answer)
    weights: list[float] = []
    for s in sentences:
        if is_boilerplate(s):
            weights.append(0.25)
        elif any(f.raw in s for f in answer_facts):
            weights.append(2.0)
        else:
            weights.append(1.0)

    # contradiction screen on would-pass sentences
    pass_idx = [i for i, sup in enumerate(supports) if sup >= thr]
    cue_best = [best[i] if i in pass_idx else ("", "") for i in range(len(sentences))]
    # only screen those that would pass — set others empty
    for i in range(len(sentences)):
        if i not in pass_idx:
            cue_best[i] = ("", "")
    cues = screen_contradictions(sentences, cue_best, sentence_offsets=offsets)
    cue_indices = {c.sentence_index for c in cues}

    supported_flags = []
    uncovered: list[Span] = []
    mandatory: list[Span] = []
    for i, (s, sup, w) in enumerate(zip(sentences, supports, weights, strict=True)):
        ok = sup >= thr and i not in cue_indices
        supported_flags.append(ok)
        if not ok:
            span = Span(
                start=offsets[i][0],
                end=offsets[i][1],
                text=s,
                reason="uncovered" if i not in cue_indices else "contradiction_cue",
                tier=2,
            )
            if i in cue_indices:
                mandatory.append(span)
            elif w >= 1.0:
                uncovered.append(span)
                if mode == "lexical":
                    mandatory.append(span)

    w_sum = sum(weights) or 1.0
    coverage = sum(w for w, ok in zip(weights, supported_flags, strict=True) if ok) / w_sum
    risk = 1.0 - coverage

    signals = [
        Signal(
            name="grounding.risk_coverage",
            tier=2,
            value=risk,
            weight=0.25,
            spans=uncovered + mandatory,
            detail={
                "coverage": coverage,
                "tau_sent": thr,
                "jl_only": jl_only,
                "mode": mode,
                "containment_hits": containment_hits,
            },
        )
    ]
    if cues:
        signals.append(
            Signal(
                name="grounding.contradiction_cue",
                tier=2,
                value=min(1.0, len(cues) / max(len(sentences), 1)),
                weight=0.30,
                spans=[c.span for c in cues],
                detail={"count": len(cues)},
            )
        )
    if jl_only:
        signals.append(
            Signal(
                name="grounding.low_fidelity_space",
                tier=2,
                value=0.2,
                weight=0.1,
                detail={"tau_sent_jl": tau_sent_jl},
            )
        )

    return CoverageResult(
        coverage=coverage,
        risk_coverage=risk,
        coverage_mode=mode if mode != "raw-384" else ("raw-384" if not jl_only else "lexical"),
        uncovered_spans=uncovered,
        contradiction_cues=cues,
        mandatory_tier3=mandatory,
        signals=signals,
        sentence_support=supports,
    )
