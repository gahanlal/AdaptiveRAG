"""
Adaptive Chunker — RecursiveSplitter with adaptive strategy selection.

Based on: "Adaptive Chunking: Optimizing Chunking-Method Selection for RAG"
(Ekimetrics, LREC 2026)  https://github.com/ekimetrics/adaptive-chunking

Algorithm per structural segment
─────────────────────────────────
1. Run three RecursiveSplitter strategies (different chunk sizes / merge modes)
2. Score each candidate set using three intrinsic quality metrics:
   • SC  — Size Compliance: fraction of chunks within [min_tokens, max_tokens]
   • ICC — Intra-Chunk Cohesion: mean cosine-sim of chunk sentences to chunk embedding
   • DCC — Contextual Coherence: mean cosine-sim of chunk to sliding context window
3. Combined score = mean(SC, ICC, DCC); highest score wins
4. Tag each AdaptiveChunk with the winning strategy name

No spaCy, no scikit-learn required — uses tiktoken + sentence-transformers + numpy.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from backend.rag.parser import StructuralSegment
from backend.models import AdaptiveChunk

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")
_embedder: Optional[SentenceTransformer] = None

MIN_TOKENS = 40
MAX_TOKENS = 512
ICC_WINDOW = 3   # adjacent chunks used for DCC window


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _extract_entities(text: str) -> list[str]:
    """Regex-based proper noun extraction (no external NLP required)."""
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    return list(dict.fromkeys(matches))[:20]   # unique, capped


def _split_sentences(text: str) -> list[str]:
    """Simple regex sentence splitter — no NLTK download required."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"\'])|(?<=\n)\n+", text)
    return [p.strip() for p in parts if p.strip() and _count_tokens(p.strip()) >= 3]


# ---------------------------------------------------------------------------
# RecursiveSplitter — faithful port of the Ekimetrics core algorithm
# ---------------------------------------------------------------------------
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveSplitter:
    """
    Split text recursively using a priority-ordered list of separators.
    After splitting, merge results in one of two modes:
      • "to_chunk_size" — greedily merge until ~chunk_size tokens (adds overlap)
      • "small_only"    — only merge chunks below min_chunk_tokens with a neighbour
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_tokens: int = 50,
        merging: str = "to_chunk_size",
        separators: Optional[list[str]] = None,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must not exceed chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.merging = merging
        self.separators = separators if separators is not None else list(_DEFAULT_SEPARATORS)

    # --- internal helpers --------------------------------------------------

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if _count_tokens(text) <= self.chunk_size:
            return [text]

        sep, remaining = separators[0], separators[1:]

        if sep == "":
            return self._hard_split(text)

        parts = re.split(re.escape(sep), text)
        parts = [p for p in parts if p]

        result: list[str] = []
        for part in parts:
            if _count_tokens(part) > self.chunk_size:
                sub = self._recursive_split(part, remaining if remaining else [""])
                result.extend(sub)
            else:
                result.append(part)
        return result

    def _hard_split(self, text: str) -> list[str]:
        """Token-level hard split when no separator works."""
        tokens = _enc.encode(text)
        chunks: list[str] = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i: i + self.chunk_size]
            try:
                chunks.append(_enc.decode(chunk_tokens))
            except Exception:
                chunks.append(text[i: i + self.chunk_size * 4])  # char fallback
            i += self.chunk_size
        return chunks

    def _merge_to_size(self, splits: list[str]) -> list[str]:
        """Greedy merge with overlap — fills chunks up to ~chunk_size."""
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for split in splits:
            split_tokens = _count_tokens(split)
            if current_tokens + split_tokens > self.chunk_size and current_parts:
                chunks.append("".join(current_parts))
                # Build overlap from the tail of current_parts
                overlap_parts: list[str] = []
                overlap_tokens = 0
                for part in reversed(current_parts):
                    pt = _count_tokens(part)
                    if overlap_tokens + pt > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_tokens += pt
                current_parts = overlap_parts + [split]
                current_tokens = _count_tokens("".join(current_parts))
            else:
                current_parts.append(split)
                current_tokens += split_tokens

        if current_parts:
            chunks.append("".join(current_parts))
        return chunks

    def _merge_small_only(self, splits: list[str]) -> list[str]:
        """Merge only chunks below min_chunk_tokens with an adjacent neighbour."""
        result = list(splits)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(result):
                if _count_tokens(result[i]) < self.min_chunk_tokens:
                    if i < len(result) - 1 and (
                        _count_tokens(result[i]) + _count_tokens(result[i + 1])
                        <= self.chunk_size
                    ):
                        result[i + 1] = result[i] + result[i + 1]
                        result.pop(i)
                        changed = True
                        continue
                    elif i > 0 and (
                        _count_tokens(result[i - 1]) + _count_tokens(result[i])
                        <= self.chunk_size
                    ):
                        result[i - 1] = result[i - 1] + result[i]
                        result.pop(i)
                        changed = True
                        continue
                i += 1
        return result

    # --- public API --------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        initial = self._recursive_split(text, self.separators)
        initial = [c for c in initial if c.strip()]
        if self.merging == "to_chunk_size":
            return self._merge_to_size(initial)
        elif self.merging == "small_only":
            return self._merge_small_only(initial)
        return initial


# ---------------------------------------------------------------------------
# Intrinsic quality metrics (simplified — no external NLP)
# ---------------------------------------------------------------------------

def _score_size_compliance(chunks: list[str]) -> float:
    """Fraction of chunks whose token count is in [MIN_TOKENS, MAX_TOKENS]."""
    if not chunks:
        return 0.0
    in_range = sum(1 for c in chunks if MIN_TOKENS <= _count_tokens(c) <= MAX_TOKENS)
    return in_range / len(chunks)


def _score_icc(chunks: list[str], embedder: SentenceTransformer) -> float:
    """
    Intra-Chunk Cohesion — mean cosine similarity between each sentence and
    its parent-chunk embedding.  Higher = sentences stay on-topic within chunk.
    """
    if not chunks:
        return 0.0

    chunk_texts = [c for c in chunks if c.strip()]
    if not chunk_texts:
        return 0.0

    chunk_embeddings = embedder.encode(chunk_texts, normalize_embeddings=True)
    scores: list[float] = []

    for chunk_text, chunk_emb in zip(chunk_texts, chunk_embeddings):
        sents = _split_sentences(chunk_text)
        if len(sents) < 2:
            scores.append(1.0)   # single sentence → perfect cohesion
            continue
        sent_embeddings = embedder.encode(sents, normalize_embeddings=True)
        sims = np.dot(sent_embeddings, chunk_emb)
        scores.append(float(np.mean(sims)))

    return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0


def _score_dcc(chunks: list[str], embedder: SentenceTransformer, window: int = ICC_WINDOW) -> float:
    """
    Contextual Coherence — cosine similarity of each chunk to its surrounding
    context window (window adjacent chunks).  Higher = chunk fits its context.
    """
    if len(chunks) < 2:
        return 1.0

    chunk_texts = [c for c in chunks if c.strip()]
    if len(chunk_texts) < 2:
        return 1.0

    embs = embedder.encode(chunk_texts, normalize_embeddings=True)
    scores: list[float] = []

    for i, emb in enumerate(embs):
        lo = max(0, i - window)
        hi = min(len(embs), i + window + 1)
        window_indices = [j for j in range(lo, hi) if j != i]
        if not window_indices:
            continue
        window_emb = np.mean(embs[window_indices], axis=0)
        norm = np.linalg.norm(window_emb)
        if norm > 1e-9:
            window_emb /= norm
        scores.append(float(np.dot(emb, window_emb)))

    return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0


# ---------------------------------------------------------------------------
# Candidate strategies — mirrors the ekimetrics paper's strategy pool
# ---------------------------------------------------------------------------
_STRATEGIES: list[tuple[str, RecursiveSplitter]] = [
    (
        "recursive_512",
        RecursiveSplitter(chunk_size=512, chunk_overlap=50, min_chunk_tokens=60, merging="to_chunk_size"),
    ),
    (
        "recursive_256_small_only",
        RecursiveSplitter(chunk_size=256, chunk_overlap=25, min_chunk_tokens=40, merging="small_only"),
    ),
    (
        "recursive_400_para",
        RecursiveSplitter(
            chunk_size=400, chunk_overlap=40, min_chunk_tokens=60,
            merging="to_chunk_size", separators=["\n\n", "\n", ". ", ""],
        ),
    ),
]


# ---------------------------------------------------------------------------
# Per-segment adaptive chunking
# ---------------------------------------------------------------------------

def _select_best_strategy(
    text: str, embedder: SentenceTransformer
) -> tuple[str, list[str], dict[str, dict[str, float]]]:
    """
    Try all strategies on *text*; score each with SC + ICC + DCC.
    Returns (winning_strategy_name, winning_chunks, all_scores_dict).
    """
    scores_map: dict[str, dict[str, float]] = {}
    best_name = ""
    best_chunks: list[str] = []
    best_combined = -1.0

    for name, splitter in _STRATEGIES:
        raw = splitter.split_text(text)
        raw = [c for c in raw if c.strip()]
        if not raw:
            continue

        sc = _score_size_compliance(raw)
        icc = _score_icc(raw, embedder)
        dcc = _score_dcc(raw, embedder)
        combined = (sc + icc + dcc) / 3.0

        scores_map[name] = {
            "sc": round(sc, 4),
            "icc": round(icc, 4),
            "dcc": round(dcc, 4),
            "combined": round(combined, 4),
            "n_chunks": len(raw),
        }

        if combined > best_combined:
            best_combined = combined
            best_name = name
            best_chunks = raw

    if not best_name:
        # Last-resort fallback: hard token split
        best_name = "fallback_hard_split"
        best_chunks = RecursiveSplitter(512, 0, 40, "to_chunk_size").split_text(text)
        best_chunks = [c for c in best_chunks if c.strip()] or [text]
        scores_map[best_name] = {"sc": 0.0, "icc": 0.0, "dcc": 0.0, "combined": 0.0, "n_chunks": len(best_chunks)}

    return best_name, best_chunks, scores_map


def chunk_segment(
    segment: StructuralSegment,
    doc_id: str,
    embedder: Optional[SentenceTransformer] = None,
) -> tuple[list[AdaptiveChunk], dict[str, dict[str, float]]]:
    """
    Produce AdaptiveChunks from a single structural segment.
    Returns (chunks, strategy_scores).
    """
    if not segment.text.strip():
        return [], {}

    emb = embedder if embedder is not None else _get_embedder()
    best_name, raw_chunks, scores = _select_best_strategy(segment.text, emb)

    chunks: list[AdaptiveChunk] = []
    for i, text in enumerate(raw_chunks):
        chunks.append(
            AdaptiveChunk(
                chunk_id=f"ada_{uuid.uuid4().hex[:8]}",
                index=i,
                text=text,
                token_count=_count_tokens(text),
                section_title=segment.title or "Document",
                hierarchy_level=segment.level,
                parent_section=segment.parent_title or "",
                boundary_reason=best_name,   # winning strategy name
                entities=_extract_entities(text),
            )
        )
    return chunks, scores


def adaptive_chunk_all(
    segments: list[StructuralSegment],
    doc_id: str,
) -> tuple[list[AdaptiveChunk], dict[str, object]]:
    """
    Chunk all segments; select the best strategy per segment.

    Returns
    -------
    (all_chunks, aggregated_strategy_report)
        all_chunks            — flat list, globally re-indexed
        aggregated_strategy_report — {
            "segment_scores": {seg_title: {strategy: metric_dict}},
            "winner_counts":  {strategy_name: count}
        }
    """
    emb = _get_embedder()
    all_chunks: list[AdaptiveChunk] = []
    segment_scores: dict[str, dict] = {}
    winner_counts: dict[str, int] = {}

    for seg in segments:
        chunks, scores = chunk_segment(seg, doc_id, embedder=emb)
        all_chunks.extend(chunks)

        if scores:
            label = seg.title or f"seg_{len(segment_scores)}"
            segment_scores[label] = scores
            winner = max(scores, key=lambda k: scores[k]["combined"])
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

    # Re-index globally
    for i, c in enumerate(all_chunks):
        c.index = i

    report: dict[str, object] = {
        "segment_scores": segment_scores,
        "winner_counts": winner_counts,
    }
    return all_chunks, report
