"""
Stress tests for backend/rag/retriever.py
Tests: all four retrieval intents (vector/semantic, structural, metadata, graph),
       retrieve_single, retrieve_multi. Uses the actual MultiIndex API.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backend.rag.indexer as indexer_module
from backend.rag.indexer import build_indexes, get_index
from backend.rag.retriever import retrieve_single, retrieve_multi
from backend.models import AdaptiveChunk, RetrievedDoc


# ---------------------------------------------------------------------------
# Fixtures
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


TEST_CHUNKS = [
    _make_chunk(0, "RAG stands for Retrieval-Augmented Generation.", "Intro"),
    _make_chunk(1, "Dense embeddings are produced by sentence transformers.", "Methods"),
    _make_chunk(2, "FAISS provides efficient vector similarity search.", "Methods"),
    _make_chunk(3, "GPT-4o-mini is used for response generation.", "Results"),
    _make_chunk(4, "Chunking splits documents into manageable pieces.", "Intro", ["OpenAI Inc"]),
    _make_chunk(5, "Adaptive chunking selects the best strategy.", "Discussion", ["Google Research"]),
    _make_chunk(6, "Evaluation uses precision, recall, and F1.", "Evaluation"),
    _make_chunk(7, "The pipeline consists of retrieval and generation.", "Methods"),
]


def _get_test_index(session_id: str = "ret_sess"):
    """Build a fresh index for tests."""
    indexer_module._sessions.clear()
    build_indexes(session_id, TEST_CHUNKS)
    return get_index(session_id)


class TestRetriever:
    def setup_method(self):
        self.index = _get_test_index("ret_sess")

    # ----- retrieve_single (semantic/vector) -----

    def test_retrieve_single_vector_returns_list(self):
        results = retrieve_single(self.index, "What is RAG?", intent="semantic", entities=[])
        assert isinstance(results, list)

    def test_retrieve_single_vector_non_empty(self):
        results = retrieve_single(self.index, "What is RAG?", intent="semantic", entities=[])
        assert len(results) > 0

    def test_retrieve_single_vector_result_types(self):
        results = retrieve_single(self.index, "What is RAG?", intent="semantic", entities=[])
        for doc in results:
            assert isinstance(doc, RetrievedDoc)

    def test_retrieve_single_vector_scores_in_range(self):
        results = retrieve_single(self.index, "embedding vectors", intent="semantic", entities=[])
        for doc in results:
            assert -1.0 <= doc.score <= 1.0 + 1e-5

    # ----- retrieve_single (structural) -----

    def test_retrieve_single_structural_returns_list(self):
        results = retrieve_single(self.index, "methods section", intent="structural", entities=[])
        assert isinstance(results, list)

    # ----- retrieve_single (metadata) -----

    def test_retrieve_single_metadata_returns_list(self):
        results = retrieve_single(self.index, "methods", intent="metadata", entities=[])
        assert isinstance(results, list)

    # ----- retrieve_single (graph) -----

    def test_retrieve_single_graph_returns_list(self):
        results = retrieve_single(self.index, "OpenAI Inc", intent="graph", entities=["OpenAI Inc"])
        assert isinstance(results, list)

    def test_retrieve_single_graph_fallback_when_no_entity(self):
        results = retrieve_single(self.index, "nothing matches", intent="graph", entities=[])
        assert isinstance(results, list)

    # ----- custom k -----

    def test_retrieve_single_custom_k(self):
        results = retrieve_single(self.index, "RAG system", intent="semantic", entities=[], top_k=2)
        assert len(results) <= 2

    def test_retrieve_single_k_larger_than_n(self):
        results = retrieve_single(self.index, "RAG", intent="semantic", entities=[], top_k=100)
        assert len(results) <= len(TEST_CHUNKS)

    # ----- retrieve_multi -----

    def test_retrieve_multi_returns_list(self):
        results = retrieve_multi(self.index, "What is RAG?", entities=[])
        assert isinstance(results, list)

    def test_retrieve_multi_deduplicates(self):
        results = retrieve_multi(self.index, "embedding FAISS", entities=[])
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks in multi-mode results"

    def test_retrieve_multi_result_types(self):
        results = retrieve_multi(self.index, "RAG", entities=[])
        for doc in results:
            assert isinstance(doc, RetrievedDoc)

    def test_retrieve_multi_with_entities(self):
        results = retrieve_multi(self.index, "OpenAI research", entities=["OpenAI Inc"])
        assert isinstance(results, list)

    def test_retrieve_multi_empty_query(self):
        try:
            results = retrieve_multi(self.index, "", entities=[])
            assert isinstance(results, list)
        except Exception:
            pass  # acceptable


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestRetrieverStress:
    def setup_method(self):
        self.index = _get_test_index("stress_sess")

    def test_many_queries_vector(self):
        queries = [
            "What is retrieval?", "How does FAISS work?", "What are embeddings?",
            "Explain RAG pipeline", "What is chunking?", "How is text split?",
            "What models are used?", "Describe evaluation metrics",
        ]
        for q in queries:
            results = retrieve_single(self.index, q, intent="semantic", entities=[])
            assert isinstance(results, list)

    def test_many_queries_multi(self):
        queries = ["RAG", "FAISS", "embeddings", "chunking", "generation"]
        for q in queries:
            results = retrieve_multi(self.index, q, entities=[])
            assert isinstance(results, list)

    def test_special_characters_in_query(self):
        results = retrieve_single(self.index, "RAG? (what is it!)", intent="semantic", entities=[])
        assert isinstance(results, list)

    def test_unicode_query(self):
        results = retrieve_single(self.index, "検索拡張生成", intent="semantic", entities=[])
        assert isinstance(results, list)

    def test_long_query(self):
        long_q = "retrieval augmented generation system " * 20
        results = retrieve_single(self.index, long_q, intent="semantic", entities=[])
        assert isinstance(results, list)

    def test_all_intents_no_crash(self):
        for intent in ["semantic", "structural", "metadata", "graph"]:
            results = retrieve_single(self.index, "RAG pipeline", intent=intent, entities=["OpenAI Inc"])
            assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
