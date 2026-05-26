"""Multi-index builder — FAISS vector + Structural + Metadata + Graph indexes.

Vector store: faiss-cpu (IndexFlatIP — inner product = cosine for L2-normalised vecs)
Embeddings:   sentence-transformers/all-MiniLM-L6-v2 (free, local)
No ChromaDB.  Python 3.9 compatible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.models import AdaptiveChunk

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------------------
# Index container
# ---------------------------------------------------------------------------

@dataclass
class VectorStore:
    """Thin wrapper around a FAISS index, keeping parallel text/metadata lists."""
    index: faiss.IndexFlatIP
    texts: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)

    def search(self, query_emb: np.ndarray, k: int) -> list[tuple[str, str, dict, float]]:
        """Returns list of (chunk_id, text, metadata, score)."""
        k = min(k, len(self.texts))
        if k == 0:
            return []
        scores, indices = self.index.search(query_emb.astype(np.float32), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((self.chunk_ids[idx], self.texts[idx], self.metadatas[idx], float(score)))
        return results


@dataclass
class MultiIndex:
    session_id: str
    # VectorIndex: FAISS + parallel lists
    vector: VectorStore
    # StructuralIndex: section_title (lower) → list[AdaptiveChunk]
    structural: dict[str, list[AdaptiveChunk]] = field(default_factory=dict)
    # MetadataIndex: "level_N" → list[AdaptiveChunk]
    metadata: dict[str, list[AdaptiveChunk]] = field(default_factory=dict)
    # GraphIndex: NetworkX entity co-occurrence graph
    graph: nx.Graph = field(default_factory=nx.Graph)
    # Quick lookup by chunk_id
    chunk_by_id: dict[str, AdaptiveChunk] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
_sessions: dict[str, MultiIndex] = {}


def build_indexes(session_id: str, chunks: list[AdaptiveChunk]) -> MultiIndex:
    """Build all four indexes from a list of AdaptiveChunks."""
    _sessions.pop(session_id, None)   # evict old session

    # --- VectorIndex ---
    embedder = _get_embedder()
    texts = [c.text for c in chunks]
    if texts:
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
    else:
        embeddings = np.empty((0, 384), dtype=np.float32)
        dim = 384
    faiss_index = faiss.IndexFlatIP(dim)
    if embeddings.shape[0] > 0:
        faiss_index.add(embeddings)

    chunk_ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "section_title": c.section_title,
            "hierarchy_level": c.hierarchy_level,
            "parent_section": c.parent_section,
            "token_count": c.token_count,
            "boundary_reason": c.boundary_reason,
        }
        for c in chunks
    ]
    vector_store = VectorStore(
        index=faiss_index,
        texts=texts,
        chunk_ids=chunk_ids,
        metadatas=metadatas,
    )

    # --- StructuralIndex ---
    structural: dict[str, list[AdaptiveChunk]] = {}
    for c in chunks:
        structural.setdefault((c.section_title or "").lower(), []).append(c)

    # --- MetadataIndex ---
    metadata: dict[str, list[AdaptiveChunk]] = {}
    for c in chunks:
        key = f"level_{c.hierarchy_level}"
        metadata.setdefault(key, []).append(c)

    # --- GraphIndex ---
    graph: nx.Graph = nx.Graph()
    for c in chunks:
        for ent in c.entities:
            if not graph.has_node(ent):
                graph.add_node(ent, chunk_ids=[])
            graph.nodes[ent]["chunk_ids"].append(c.chunk_id)
        for i, e1 in enumerate(c.entities):
            for e2 in c.entities[i + 1:]:
                if graph.has_edge(e1, e2):
                    graph[e1][e2]["weight"] += 1
                else:
                    graph.add_edge(e1, e2, weight=1)

    chunk_by_id = {c.chunk_id: c for c in chunks}

    mi = MultiIndex(
        session_id=session_id,
        vector=vector_store,
        structural=structural,
        metadata=metadata,
        graph=graph,
        chunk_by_id=chunk_by_id,
    )
    _sessions[session_id] = mi
    return mi


def get_index(session_id: str) -> Optional[MultiIndex]:
    return _sessions.get(session_id)
