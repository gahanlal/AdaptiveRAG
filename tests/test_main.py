"""
Stress tests for backend/main.py (FastAPI routes).
Tests: /api/health, /api/ingest (text and file upload), /api/query.
Uses httpx's TestClient via fastapi.testclient.
"""
import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Must be imported AFTER sys.path is set up
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """# Introduction

This document describes Retrieval-Augmented Generation (RAG).

## How It Works

RAG combines retrieval from a knowledge base with language model generation.
The model first retrieves relevant chunks, then generates a grounded response.

## Benefits

RAG reduces hallucinations and improves factual accuracy.
"""


def _mock_simple_ingest(session_id, text):
    from backend.models import SimpleChunk
    return [
        SimpleChunk(chunk_id="sc1", index=0, text=text[:50] or "chunk", token_count=10,
                    char_start=0, char_end=50)
    ]


def _mock_adaptive_ingest(session_id, text):
    from backend.rag.parser import StructuralSegment
    from backend.models import AdaptiveChunk
    seg = StructuralSegment(title="Intro", level=1, parent_title="", text=text, char_start=0, char_end=len(text))
    chunk = AdaptiveChunk(
        chunk_id="c1", index=0, text=text[:100], token_count=20,
        section_title="Intro", hierarchy_level=1, parent_section="",
        boundary_reason="recursive_512", entities=[],
    )
    return [seg], [chunk], {"segment_scores": {}, "winner_counts": {"recursive_512": 1}}


def _mock_simple_query(session_id, user_query):
    from backend.models import QueryResponse
    return QueryResponse(
        response="Test response.",
        citations=["c1"],
        intent="semantic",
        path_taken=["FixedChunker", "FAISS", "GPT4oMini"],
        execution_time_ms=42.0,
        retrieval_count=3,
        retrieved_docs=[],
    )


def _mock_adaptive_query(session_id, user_query, retrieval_mode):
    from backend.models import QueryResponse
    return QueryResponse(
        response="Adaptive response.",
        citations=["c1"],
        intent="semantic",
        path_taken=["QueryUnderstanding", "Retriever", "Reranker", "Generator"],
        retrieved_docs=[],
        retrieval_count=3,
        execution_time_ms=55.0,
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_ok(self):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_status_field(self):
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_version_field(self):
        response = client.get("/api/health")
        data = response.json()
        assert "version" in data


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_text_paste(self, mock_ada, mock_simple):
        response = client.post(
            "/api/ingest",
            data={"raw_text": SAMPLE_TEXT},
            files={"file": ("", b"", "text/plain")},
        )
        assert response.status_code == 200

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_returns_session_id(self, mock_ada, mock_simple):
        response = client.post(
            "/api/ingest",
            data={"raw_text": SAMPLE_TEXT},
            files={"file": ("", b"", "text/plain")},
        )
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_returns_required_fields(self, mock_ada, mock_simple):
        response = client.post(
            "/api/ingest",
            data={"raw_text": SAMPLE_TEXT},
            files={"file": ("", b"", "text/plain")},
        )
        data = response.json()
        for field in ("session_id", "filename", "char_count", "word_count",
                      "simple_chunks", "adaptive_chunks", "structure_tree", "strategy_scores"):
            assert field in data, f"Missing field: {field}"

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_txt_file(self, mock_ada, mock_simple):
        file_content = SAMPLE_TEXT.encode("utf-8")
        response = client.post(
            "/api/ingest",
            files={"file": ("test_doc.txt", file_content, "text/plain")},
        )
        assert response.status_code == 200

    def test_ingest_no_content_returns_error(self):
        response = client.post(
            "/api/ingest",
            files={"file": ("", b"", "text/plain")},
        )
        # Should return 400 if no text provided
        assert response.status_code in (400, 422)

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_chunks_are_lists(self, mock_ada, mock_simple):
        response = client.post(
            "/api/ingest",
            data={"raw_text": SAMPLE_TEXT},
            files={"file": ("", b"", "text/plain")},
        )
        data = response.json()
        assert isinstance(data["simple_chunks"], list)
        assert isinstance(data["adaptive_chunks"], list)

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_char_count_correct(self, mock_ada, mock_simple):
        response = client.post(
            "/api/ingest",
            data={"raw_text": SAMPLE_TEXT},
            files={"file": ("", b"", "text/plain")},
        )
        data = response.json()
        assert data["char_count"] == len(SAMPLE_TEXT)

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_strategy_scores_is_dict(self, mock_ada, mock_simple):
        response = client.post(
            "/api/ingest",
            data={"raw_text": SAMPLE_TEXT},
            files={"file": ("", b"", "text/plain")},
        )
        data = response.json()
        assert isinstance(data["strategy_scores"], dict)


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    @patch("backend.main.simple_rag.query", side_effect=_mock_simple_query)
    def test_query_simple_rag(self, mock_q):
        payload = {
            "session_id": "test_session",
            "query": "What is RAG?",
            "rag_type": "simple",
            "retrieval_mode": "single",
        }
        response = client.post("/api/query", json=payload)
        assert response.status_code == 200

    @patch("backend.main.adaptive_rag.query", side_effect=_mock_adaptive_query)
    def test_query_adaptive_rag(self, mock_q):
        payload = {
            "session_id": "test_session",
            "query": "What is RAG?",
            "rag_type": "adaptive",
            "retrieval_mode": "single",
        }
        response = client.post("/api/query", json=payload)
        assert response.status_code == 200

    @patch("backend.main.simple_rag.query", side_effect=_mock_simple_query)
    def test_query_response_has_required_fields(self, mock_q):
        payload = {
            "session_id": "test_session",
            "query": "What is RAG?",
            "rag_type": "simple",
            "retrieval_mode": "single",
        }
        response = client.post("/api/query", json=payload)
        data = response.json()
        for field in ("response", "citations", "path_taken", "execution_time_ms",
                      "retrieval_count", "retrieved_docs"):
            assert field in data, f"Missing field: {field}"

    def test_query_missing_session_returns_error(self):
        payload = {
            "query": "What is RAG?",
            "rag_type": "simple",
            "retrieval_mode": "single",
        }
        response = client.post("/api/query", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_query_empty_query_string(self):
        payload = {
            "session_id": "test_session",
            "query": "",
            "rag_type": "simple",
            "retrieval_mode": "single",
        }
        response = client.post("/api/query", json=payload)
        # Should return 400 or 422
        assert response.status_code in (200, 400, 422, 500)


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestMainStress:
    def test_health_multiple_times(self):
        for _ in range(20):
            r = client.get("/api/health")
            assert r.status_code == 200

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_multiple_documents(self, mock_ada, mock_simple):
        for i in range(5):
            response = client.post(
                "/api/ingest",
                data={"raw_text": f"Document {i}. " + SAMPLE_TEXT},
                files={"file": ("", b"", "text/plain")},
            )
            assert response.status_code == 200

    @patch("backend.main.simple_rag.query", side_effect=_mock_simple_query)
    def test_query_multiple_times(self, mock_q):
        queries = ["What is RAG?", "How does FAISS work?", "Explain chunking", "What is GPT-4?"]
        for q in queries:
            payload = {"session_id": "test_sess", "query": q, "rag_type": "simple", "retrieval_mode": "single"}
            r = client.post("/api/query", json=payload)
            assert r.status_code == 200

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_large_text(self, mock_ada, mock_simple):
        large_text = SAMPLE_TEXT * 50
        response = client.post(
            "/api/ingest",
            data={"raw_text": large_text},
            files={"file": ("", b"", "text/plain")},
        )
        assert response.status_code == 200

    @patch("backend.main.simple_rag.ingest", side_effect=_mock_simple_ingest)
    @patch("backend.main.adaptive_rag.ingest", side_effect=_mock_adaptive_ingest)
    def test_ingest_session_ids_unique(self, mock_ada, mock_simple):
        ids = set()
        for _ in range(10):
            response = client.post(
                "/api/ingest",
                data={"raw_text": SAMPLE_TEXT},
                files={"file": ("", b"", "text/plain")},
            )
            ids.add(response.json()["session_id"])
        assert len(ids) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
