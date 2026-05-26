"""
Stress tests for backend/rag/simple_rag.py
Tests: chunk_text fixed-size chunking, ingest, query flow (mocked OpenAI).
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backend.rag.simple_rag as simple_rag
from backend.rag.simple_rag import chunk_text, ingest, query


SAMPLE_TEXT = (
    "Retrieval-Augmented Generation (RAG) combines retrieval systems with language models. "
    "The model first retrieves relevant passages from a knowledge base. "
    "Then it uses those passages to generate a grounded response. "
    "This approach reduces hallucinations compared to vanilla language models. "
    "Simple RAG uses fixed-size chunking without semantic awareness. "
    "Each chunk contains a fixed number of tokens with optional overlap. "
    "The overlap helps preserve context across chunk boundaries. "
    "Documents are embedded using sentence transformers. "
    "Embeddings are stored in a FAISS index for fast similarity search. "
    "At query time, the top-k nearest chunks are retrieved. "
) * 10


SMALL_TEXT = "Hello world. This is a test document. It has three sentences."


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_basic_chunking(self):
        chunks = chunk_text("test_id", SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_empty_text(self):
        chunks = chunk_text("test_id", "")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_text("test_id", "   \n\t ")
        # whitespace may tokenize to a chunk; we just require all chunks contain no meaningful text
        for c in chunks:
            assert c.text.strip() == "" or len(chunks) == 0

    def test_small_text_single_chunk(self):
        chunks = chunk_text("test_id", SMALL_TEXT)
        assert len(chunks) >= 1
        # All text should be in the chunks
        combined = " ".join(c.text for c in chunks)
        for word in ["Hello", "world", "test"]:
            assert word in combined

    def test_chunk_ids_unique(self):
        chunks = chunk_text("doc1", SAMPLE_TEXT)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_sequential(self):
        chunks = chunk_text("doc1", SAMPLE_TEXT)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_char_offsets(self):
        chunks = chunk_text("doc1", SAMPLE_TEXT)
        for c in chunks:
            assert c.char_start >= 0
            assert c.char_end > c.char_start
            assert c.char_end <= len(SAMPLE_TEXT) + 1

    def test_token_count_positive(self):
        chunks = chunk_text("doc1", SAMPLE_TEXT)
        for c in chunks:
            assert c.token_count > 0

    def test_token_count_not_exceeds_limit(self):
        chunks = chunk_text("doc1", SAMPLE_TEXT)
        # Default window 512 + 50 overlap. Allow small buffer.
        for c in chunks:
            assert c.token_count <= 580

    def test_chunk_text_not_empty(self):
        chunks = chunk_text("doc1", SAMPLE_TEXT)
        for c in chunks:
            assert c.text.strip()

    def test_deterministic(self):
        c1 = chunk_text("doc1", SAMPLE_TEXT)
        c2 = chunk_text("doc1", SAMPLE_TEXT)
        assert [c.text for c in c1] == [c.text for c in c2]


# ---------------------------------------------------------------------------
# ingest tests
# ---------------------------------------------------------------------------

class TestIngest:
    def setup_method(self):
        # Clear session cache between tests
        simple_rag._sessions.clear()

    def test_ingest_stores_session(self):
        ingest("sess1", SAMPLE_TEXT)
        assert "sess1" in simple_rag._sessions

    def test_ingest_creates_faiss_index(self):
        import faiss
        ingest("sess2", SAMPLE_TEXT)
        sess = simple_rag._sessions["sess2"]
        assert isinstance(sess.index, faiss.IndexFlatIP)

    def test_ingest_index_has_vectors(self):
        ingest("sess3", SAMPLE_TEXT)
        sess = simple_rag._sessions["sess3"]
        assert sess.index.ntotal > 0

    def test_ingest_overwrites_session(self):
        ingest("sess4", SAMPLE_TEXT)
        n1 = simple_rag._sessions["sess4"].index.ntotal
        ingest("sess4", SMALL_TEXT)
        n2 = simple_rag._sessions["sess4"].index.ntotal
        assert n2 <= n1  # Small text should have fewer or equal chunks

    def test_ingest_empty_text_creates_empty_index(self):
        ingest("sess5", "")
        if "sess5" in simple_rag._sessions:
            assert simple_rag._sessions["sess5"].index.ntotal == 0

    def test_ingest_multiple_sessions_isolated(self):
        ingest("sess_a", SAMPLE_TEXT)
        ingest("sess_b", SMALL_TEXT)
        assert simple_rag._sessions["sess_a"].index.ntotal != simple_rag._sessions["sess_b"].index.ntotal


# ---------------------------------------------------------------------------
# query tests (mocked OpenAI)
# ---------------------------------------------------------------------------

class TestQuery:
    def setup_method(self):
        simple_rag._sessions.clear()
        simple_rag._oai_client = None
        ingest("qsess", SAMPLE_TEXT)

    def _make_mock_oai(self, content="Test answer"):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.return_value = mock_response
        return mock_oai

    @patch("backend.rag.simple_rag._get_oai")
    def test_query_returns_dict(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai("RAG combines retrieval and generation.")
        result = query("qsess", "What is RAG?")
        assert isinstance(result, dict)

    @patch("backend.rag.simple_rag._get_oai")
    def test_query_has_required_keys(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai()
        result = query("qsess", "What is RAG?")
        for key in ("response", "citations", "path_taken", "execution_time_ms", "retrieval_count"):
            assert key in result, f"Missing key: {key}"

    @patch("backend.rag.simple_rag._get_oai")
    def test_query_execution_time_positive(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai()
        result = query("qsess", "What is RAG?")
        assert result["execution_time_ms"] >= 0

    @patch("backend.rag.simple_rag._get_oai")
    def test_query_retrieval_count_positive(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai()
        result = query("qsess", "What is RAG?")
        assert result["retrieval_count"] > 0

    def test_query_unknown_session_returns_error(self):
        # query() returns graceful error response when session not found
        result = query("nonexistent_session_xyz", "test query")
        assert isinstance(result, dict)
        assert "intent" in result
        assert result["intent"] == "error"

    @patch("backend.rag.simple_rag._get_oai")
    def test_query_path_taken_is_list(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai()
        result = query("qsess", "What is RAG?")
        assert isinstance(result["path_taken"], list)
        assert len(result["path_taken"]) > 0

    @patch("backend.rag.simple_rag._get_oai")
    def test_query_citations_is_list(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai()
        result = query("qsess", "What is RAG?")
        assert isinstance(result["citations"], list)


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestSimpleRagStress:
    def setup_method(self):
        simple_rag._sessions.clear()

    def test_ingest_many_sessions(self):
        for i in range(20):
            ingest(f"stress_sess_{i}", SAMPLE_TEXT)
        assert len(simple_rag._sessions) == 20

    def test_large_document(self):
        large_doc = SAMPLE_TEXT * 5
        ingest("large_sess", large_doc)
        sess = simple_rag._sessions["large_sess"]
        assert sess.index.ntotal > 0

    def test_chunk_text_with_only_whitespace_blocks(self):
        text = "\n\n".join(["  " for _ in range(50)])
        chunks = chunk_text("ws_doc", text)
        assert isinstance(chunks, list)

    def test_chunk_text_very_short_repeated(self):
        # Many short segments
        text = "x. " * 1000
        chunks = chunk_text("short_rep", text)
        assert len(chunks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
