"""
Stress tests for backend/rag/indexer.py
Tests: VectorStore search, build_indexes, get_index, graph construction.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backend.rag.indexer as indexer_module
from backend.rag.indexer import VectorStore, build_indexes, get_index
from backend.models import AdaptiveChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(idx: int, text: str, section: str = "Section", entities=None) -> AdaptiveChunk:
    return AdaptiveChunk(
        chunk_id=f"chunk_{idx}",
        index=idx,
        text=text,
        token_count=len(text.split()),
        section_title=section,
        hierarchy_level=1,
        parent_section="",
        boundary_reason="recursive_512",
        entities=entities or [],
    )


CHUNKS = [
    _make_chunk(0, "Retrieval-Augmented Generation combines retrieval with language models.", "Intro"),
    _make_chunk(1, "The embedding model converts text into dense vector representations.", "Methods"),
    _make_chunk(2, "FAISS enables fast approximate nearest neighbor search.", "Methods"),
    _make_chunk(3, "GPT-4o-mini generates fluent natural language responses.", "Results"),
    _make_chunk(4, "Simple RAG uses fixed-size chunking without semantic boundaries.", "Intro"),
    _make_chunk(5, "Adaptive RAG selects the best chunking strategy per segment.", "Discussion"),
    _make_chunk(6, "Precision and recall are common evaluation metrics.", "Evaluation"),
    _make_chunk(7, "The retriever finds the top-k most relevant chunks.", "Methods"),
]


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def _build_store(self, dim=32, n=8):
        import faiss
        vecs = np.random.randn(n, dim).astype("float32")
        # L2-normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)
        texts = [f"text {i}" for i in range(n)]
        chunk_ids = [f"cid_{i}" for i in range(n)]
        metas = [{"section": "S"} for _ in range(n)]
        return VectorStore(index=idx, texts=texts, chunk_ids=chunk_ids, metadatas=metas), vecs

    def test_search_returns_k_results(self):
        store, vecs = self._build_store()
        q = vecs[0:1]
        results = store.search(q, k=3)
        assert len(results) == 3

    def test_search_returns_correct_types(self):
        store, vecs = self._build_store()
        results = store.search(vecs[0:1], k=2)
        for cid, text, meta, score in results:
            assert isinstance(cid, str)
            assert isinstance(text, str)
            assert isinstance(meta, dict)
            assert isinstance(score, (float, np.floating))

    def test_search_scores_in_range(self):
        store, vecs = self._build_store()
        results = store.search(vecs[0:1], k=4)
        for _, _, _, score in results:
            assert -1.0 <= float(score) <= 1.0 + 1e-5

    def test_search_self_is_most_similar(self):
        store, vecs = self._build_store()
        results = store.search(vecs[0:1], k=1)
        assert results[0][0] == "cid_0"

    def test_search_k_larger_than_n(self):
        store, vecs = self._build_store(n=5)
        results = store.search(vecs[0:1], k=100)
        assert len(results) == 5  # capped at n

    def test_search_k_zero(self):
        store, vecs = self._build_store()
        results = store.search(vecs[0:1], k=0)
        assert results == []


# ---------------------------------------------------------------------------
# build_indexes tests
# ---------------------------------------------------------------------------

class TestBuildIndexes:
    def setup_method(self):
        indexer_module._sessions.clear()

    def test_build_indexes_stores_session(self):
        build_indexes("sess_build", CHUNKS)
        assert "sess_build" in indexer_module._sessions

    def test_build_indexes_creates_vector_store(self):
        import faiss
        build_indexes("sess_vec", CHUNKS)
        mi = indexer_module._sessions["sess_vec"]
        assert isinstance(mi.vector, VectorStore)
        assert isinstance(mi.vector.index, faiss.IndexFlatIP)

    def test_build_indexes_vector_count_matches_chunks(self):
        build_indexes("sess_count", CHUNKS)
        mi = indexer_module._sessions["sess_count"]
        assert mi.vector.index.ntotal == len(CHUNKS)

    def test_build_indexes_graph_created(self):
        import networkx as nx
        build_indexes("sess_graph", CHUNKS)
        mi = indexer_module._sessions["sess_graph"]
        assert isinstance(mi.graph, nx.Graph)

    def test_build_indexes_chunk_by_id_lookup(self):
        build_indexes("sess_cbi", CHUNKS)
        mi = indexer_module._sessions["sess_cbi"]
        for c in CHUNKS:
            assert c.chunk_id in mi.chunk_by_id

    def test_build_indexes_overwrites_old_session(self):
        build_indexes("sess_ow", CHUNKS)
        new_chunks = [_make_chunk(0, "Only one chunk.", "Only")]
        build_indexes("sess_ow", new_chunks)
        mi = indexer_module._sessions["sess_ow"]
        assert mi.vector.index.ntotal == 1

    def test_build_indexes_empty_chunks(self):
        build_indexes("sess_empty", [])
        mi = indexer_module._sessions.get("sess_empty")
        if mi:
            assert mi.vector.index.ntotal == 0


# ---------------------------------------------------------------------------
# get_index tests
# ---------------------------------------------------------------------------

class TestGetIndex:
    def setup_method(self):
        indexer_module._sessions.clear()

    def test_get_index_returns_multiindex(self):
        from backend.rag.indexer import MultiIndex
        build_indexes("gi_sess", CHUNKS)
        mi = get_index("gi_sess")
        assert isinstance(mi, MultiIndex)

    def test_get_index_none_for_missing(self):
        result = get_index("nonexistent_session_xyz")
        assert result is None

    def test_get_index_session_id_matches(self):
        build_indexes("gi_sess2", CHUNKS)
        mi = get_index("gi_sess2")
        assert mi.session_id == "gi_sess2"


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestIndexerStress:
    def setup_method(self):
        indexer_module._sessions.clear()

    def test_many_sessions(self):
        for i in range(30):
            build_indexes(f"stress_{i}", CHUNKS[:3])
        assert len(indexer_module._sessions) == 30

    def test_large_chunk_set(self):
        large_chunks = [
            _make_chunk(i, f"This is chunk number {i}. It contains some meaningful text.", "Section")
            for i in range(200)
        ]
        build_indexes("large_sess", large_chunks)
        mi = get_index("large_sess")
        assert mi.vector.index.ntotal == 200

    def test_chunks_with_entities(self):
        ent_chunks = [
            _make_chunk(0, "Apple Inc acquired Microsoft Corporation.", "Biz", ["Apple Inc", "Microsoft Corporation"]),
            _make_chunk(1, "Google LLC and Meta Inc compete.", "Tech", ["Google LLC", "Meta Inc"]),
        ]
        build_indexes("ent_sess", ent_chunks)
        mi = get_index("ent_sess")
        assert mi.graph.number_of_nodes() >= 2

    def test_repeated_build_same_session(self):
        for _ in range(5):
            build_indexes("repeated_sess", CHUNKS)
        mi = get_index("repeated_sess")
        assert mi.vector.index.ntotal == len(CHUNKS)

    def test_search_after_build(self):
        build_indexes("search_test", CHUNKS)
        mi = get_index("search_test")
        # Run a real search on the built index
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        q_vec = emb.encode(["What is RAG?"], normalize_embeddings=True)
        results = mi.vector.search(q_vec, k=3)
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
