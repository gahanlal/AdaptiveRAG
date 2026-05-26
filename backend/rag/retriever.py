"""Retrieval router — intent-based single retrieval or parallel multi-retrieval."""
from __future__ import annotations

import numpy as np
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from sentence_transformers import SentenceTransformer

from backend.models import AdaptiveChunk, RetrievedDoc
from backend.rag.indexer import MultiIndex

_enc = tiktoken.get_encoding("cl100k_base")

TOP_K = 5

_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------------------
# Individual retrievers
# ---------------------------------------------------------------------------

def _retrieve_vector(
    index: MultiIndex, query: str, top_k: int = TOP_K
) -> list[RetrievedDoc]:
    embedder = _get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q_emb = q_emb.astype(np.float32)

    results = index.vector.search(q_emb, k=top_k)
    retrieved = []
    for chunk_id, text, meta, score in results:
        retrieved.append(
            RetrievedDoc(
                chunk_id=chunk_id,
                text=text,
                section_title=str(meta.get("section_title", "—")),
                score=round(score, 4),
                source_index="vector",
            )
        )
    return retrieved


def _retrieve_structural(
    index: MultiIndex, query: str, top_k: int = TOP_K
) -> list[RetrievedDoc]:
    """Match query keywords against section titles."""
    query_lower = query.lower()
    scored: list[tuple[float, AdaptiveChunk]] = []

    for section_key, chunks in index.structural.items():
        # Score = fraction of query words found in section title
        query_words = set(query_lower.split())
        title_words = set(section_key.split())
        overlap = len(query_words & title_words)
        score = overlap / max(len(query_words), 1)
        if score > 0 or len(index.structural) == 1:
            for c in chunks:
                scored.append((score, c))

    # If no overlap, return all chunks with neutral score
    if not scored:
        for chunks in index.structural.values():
            for c in chunks:
                scored.append((0.1, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        RetrievedDoc(
            chunk_id=c.chunk_id,
            text=c.text,
            section_title=c.section_title,
            score=round(s, 4),
            source_index="structural",
        )
        for s, c in scored[:top_k]
    ]


def _retrieve_metadata(
    index: MultiIndex, query: str, top_k: int = TOP_K
) -> list[RetrievedDoc]:
    """Return chunks from level_1 sections (top-level headings)."""
    results: list[AdaptiveChunk] = []
    for level_key in ["level_1", "level_2", "level_3"]:
        results.extend(index.metadata.get(level_key, []))
        if len(results) >= top_k:
            break
    return [
        RetrievedDoc(
            chunk_id=c.chunk_id,
            text=c.text,
            section_title=c.section_title,
            score=0.5,
            source_index="metadata",
        )
        for c in results[:top_k]
    ]


def _retrieve_graph(
    index: MultiIndex, query: str, entities: list[str], top_k: int = TOP_K
) -> list[RetrievedDoc]:
    """Find chunks containing query entities and their graph neighbors."""
    matched_chunk_ids: set[str] = set()

    # Direct entity matches
    for ent in entities:
        ent_lower = ent.lower()
        if index.graph.has_node(ent_lower):
            cids = index.graph.nodes[ent_lower].get("chunk_ids", [])
            matched_chunk_ids.update(cids)
            # Neighbor entities
            for neighbor in index.graph.neighbors(ent_lower):
                n_cids = index.graph.nodes[neighbor].get("chunk_ids", [])
                matched_chunk_ids.update(n_cids)

    # Fallback: return top structural if no entities matched
    if not matched_chunk_ids:
        return _retrieve_structural(index, query, top_k)

    retrieved = []
    for cid in list(matched_chunk_ids)[:top_k]:
        chunk = index.chunk_by_id.get(cid)
        if chunk:
            retrieved.append(
                RetrievedDoc(
                    chunk_id=cid,
                    text=chunk.text,
                    section_title=chunk.section_title,
                    score=0.7,
                    source_index="graph",
                )
            )
    return retrieved


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def retrieve_single(
    index: MultiIndex,
    query: str,
    intent: str,
    entities: list[str],
    top_k: int = TOP_K,
) -> list[RetrievedDoc]:
    """Route to ONE retriever based on intent."""
    if intent == "structural":
        return _retrieve_structural(index, query, top_k)
    elif intent == "metadata":
        return _retrieve_metadata(index, query, top_k)
    elif intent == "graph":
        return _retrieve_graph(index, query, entities, top_k)
    else:  # semantic (default)
        return _retrieve_vector(index, query, top_k)


def retrieve_custom(
    index: MultiIndex,
    query: str,
    entities: list[str],
    selected: list[str],
    top_k: int = TOP_K,
) -> list[RetrievedDoc]:
    """Run only the specified retrievers in parallel, deduplicate, return top_k.

    ``selected`` is a list of index names from: "vector", "structural", "metadata", "graph".
    """
    _fn_map: dict[str, Callable[[], list[RetrievedDoc]]] = {
        "vector":     lambda: _retrieve_vector(index, query, top_k),
        "structural": lambda: _retrieve_structural(index, query, top_k),
        "metadata":   lambda: _retrieve_metadata(index, query, top_k),
        "graph":      lambda: _retrieve_graph(index, query, entities, top_k),
    }
    active = [_fn_map[k] for k in selected if k in _fn_map] or [_fn_map["vector"]]

    all_results: list[RetrievedDoc] = []
    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = [executor.submit(fn) for fn in active]
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception:
                pass

    seen: dict[str, RetrievedDoc] = {}
    for doc in all_results:
        if doc.chunk_id not in seen or doc.score > seen[doc.chunk_id].score:
            seen[doc.chunk_id] = doc

    return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]


def retrieve_multi(
    index: MultiIndex,
    query: str,
    entities: list[str],
    top_k: int = TOP_K,
) -> list[RetrievedDoc]:
    """Run ALL retrievers in parallel, deduplicate by chunk_id, return top_k."""
    retrievers: list[Callable[[], list[RetrievedDoc]]] = [
        lambda: _retrieve_vector(index, query, top_k),
        lambda: _retrieve_structural(index, query, top_k),
        lambda: _retrieve_metadata(index, query, top_k),
        lambda: _retrieve_graph(index, query, entities, top_k),
    ]

    all_results: list[RetrievedDoc] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fn) for fn in retrievers]
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception:
                pass

    # Deduplicate: keep highest score per chunk_id
    seen: dict[str, RetrievedDoc] = {}
    for doc in all_results:
        if doc.chunk_id not in seen or doc.score > seen[doc.chunk_id].score:
            seen[doc.chunk_id] = doc

    merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return merged[:top_k]


# ---------------------------------------------------------------------------
# Context expansion — parent-child / neighbor stitching
# ---------------------------------------------------------------------------

def expand_with_neighbors(
    index: MultiIndex,
    docs: list[RetrievedDoc],
    token_budget: int = 600,
) -> list[RetrievedDoc]:
    """
    Expand each retrieved chunk by prepending/appending its immediate
    section-neighbors (prev + next chunk within the same section).

    Budget is per-doc: the original chunk always fits; neighbors are added
    only if they stay within *token_budget* tokens total.
    This implements the "dynamic context expansion" / "parent-child retrieval"
    pattern without changing the chunk storage structure.
    """
    expanded: list[RetrievedDoc] = []

    for doc in docs:
        prev_id, next_id = index.chunk_neighbors.get(doc.chunk_id, (None, None))

        base_tokens = len(_enc.encode(doc.text))
        parts: list[str] = [doc.text]
        used = base_tokens

        # Prepend previous chunk if it fits
        if prev_id:
            prev_chunk = index.chunk_by_id.get(prev_id)
            if prev_chunk:
                prev_tokens = len(_enc.encode(prev_chunk.text))
                if used + prev_tokens <= token_budget:
                    parts.insert(0, prev_chunk.text)
                    used += prev_tokens

        # Append next chunk if it fits
        if next_id:
            next_chunk = index.chunk_by_id.get(next_id)
            if next_chunk:
                next_tokens = len(_enc.encode(next_chunk.text))
                if used + next_tokens <= token_budget:
                    parts.append(next_chunk.text)
                    used += next_tokens

        expanded_text = "\n\n".join(parts) if len(parts) > 1 else doc.text

        expanded.append(
            RetrievedDoc(
                chunk_id=doc.chunk_id,
                text=expanded_text,
                section_title=doc.section_title,
                score=doc.score,
                source_index=doc.source_index,
            )
        )

    return expanded
