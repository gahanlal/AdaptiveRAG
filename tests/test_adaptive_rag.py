"""
Stress tests for backend/rag/adaptive_rag.py
Tests: ingest (3-tuple return), query graph (mocked OpenAI), strategy report structure.
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backend.rag.indexer as indexer_module
import backend.rag.adaptive_rag as adaptive_rag_module
from backend.rag.adaptive_rag import ingest, query
from backend.models import QueryResponse


SAMPLE_DOC = """# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with language model generation. It was introduced to reduce hallucinations in LLMs.

## How RAG Works

The system first retrieves relevant documents from a knowledge base using dense embeddings.
Then it passes those documents as context to a language model for answer generation.

## Adaptive Chunking

Adaptive RAG uses multiple chunking strategies and selects the best one per segment.
The selection is based on quality metrics: size compliance, intra-chunk coherence,
and diversity between consecutive chunks.

# Results

Experiments show that adaptive chunking improves retrieval precision by 15% on average.
The recursive-512 strategy wins for long documents, while paragraph-based strategies
work better for short structured documents.
"""


def _mock_openai_response(content="Test answer."):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


# ---------------------------------------------------------------------------
# ingest tests
# ---------------------------------------------------------------------------

class TestAdaptiveRagIngest:
    def setup_method(self):
        indexer_module._sessions.clear()

    def test_ingest_returns_three_tuple(self):
        result = ingest("ada_sess1", SAMPLE_DOC)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_ingest_returns_segments(self):
        segments, chunks, report = ingest("ada_sess2", SAMPLE_DOC)
        assert isinstance(segments, list)
        assert len(segments) > 0

    def test_ingest_returns_chunks(self):
        segments, chunks, report = ingest("ada_sess3", SAMPLE_DOC)
        from backend.models import AdaptiveChunk
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, AdaptiveChunk)

    def test_ingest_returns_strategy_report(self):
        _, _, report = ingest("ada_sess4", SAMPLE_DOC)
        assert isinstance(report, dict)
        assert "segment_scores" in report
        assert "winner_counts" in report

    def test_ingest_builds_index(self):
        ingest("ada_sess5", SAMPLE_DOC)
        assert "ada_sess5" in indexer_module._sessions

    def test_ingest_chunk_ids_unique(self):
        _, chunks, _ = ingest("ada_sess6", SAMPLE_DOC)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_ingest_chunk_indexes_sequential(self):
        _, chunks, _ = ingest("ada_sess7", SAMPLE_DOC)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_ingest_empty_document(self):
        _, chunks, report = ingest("ada_sess_empty", "")
        assert isinstance(chunks, list)
        assert isinstance(report, dict)

    def test_ingest_winner_counts_are_ints(self):
        _, _, report = ingest("ada_sess8", SAMPLE_DOC)
        for strat, count in report["winner_counts"].items():
            assert isinstance(count, int)
            assert count >= 0

    def test_ingest_boundary_reason_is_strategy_name(self):
        _, chunks, _ = ingest("ada_sess9", SAMPLE_DOC)
        valid_strategies = {"recursive_512", "recursive_256_small_only", "recursive_400_para", "fallback_hard_split"}
        for c in chunks:
            if c.boundary_reason:
                assert c.boundary_reason in valid_strategies, f"Unknown strategy: {c.boundary_reason}"


# ---------------------------------------------------------------------------
# query tests (mocked OpenAI)
# ---------------------------------------------------------------------------

class TestAdaptiveRagQuery:
    @classmethod
    def setup_class(cls):
        import backend.rag.generator as gen_mod
        import backend.rag.reranker as reranker_mod
        gen_mod._oai_client = None
        reranker_mod._oai_client = None
        adaptive_rag_module._ada_oai_client = None
        indexer_module._sessions.clear()
        ingest("qsess_ada", SAMPLE_DOC)

    def setup_method(self):
        import backend.rag.generator as gen_mod
        import backend.rag.reranker as reranker_mod
        gen_mod._oai_client = None
        reranker_mod._oai_client = None
        adaptive_rag_module._ada_oai_client = None

    def _make_mock_oai(self, content="Answer."):
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.return_value = _mock_openai_response(content)
        return mock_oai

    @patch("backend.rag.adaptive_rag._get_oai_client")
    @patch("backend.rag.generator._get_oai")
    @patch("backend.rag.reranker._get_oai")
    def test_query_returns_query_response(self, mock_reranker, mock_gen, mock_ada):
        from backend.models import QueryResponse
        mock_ada.return_value = self._make_mock_oai('{"intent": "semantic", "entities": []}')
        mock_gen.return_value = self._make_mock_oai("Answer.")
        mock_reranker.return_value = self._make_mock_oai('{"scores": [0.9, 0.8, 0.7]}')
        result = query("qsess_ada", "What is RAG?", retrieval_mode="single")
        assert isinstance(result, QueryResponse)

    @patch("backend.rag.adaptive_rag._get_oai_client")
    @patch("backend.rag.generator._get_oai")
    @patch("backend.rag.reranker._get_oai")
    def test_query_has_required_fields(self, mock_reranker, mock_gen, mock_ada):
        mock_ada.return_value = self._make_mock_oai('{"intent": "semantic", "entities": []}')
        mock_gen.return_value = self._make_mock_oai("Answer.")
        mock_reranker.return_value = self._make_mock_oai('{"scores": [0.9]}')
        result = query("qsess_ada", "Explain adaptive chunking", retrieval_mode="single")
        assert hasattr(result, "response")
        assert hasattr(result, "citations")
        assert hasattr(result, "intent")
        assert hasattr(result, "path_taken")
        assert hasattr(result, "retrieved_docs")
        assert hasattr(result, "retrieval_count")
        assert hasattr(result, "execution_time_ms")

    @patch("backend.rag.adaptive_rag._get_oai_client")
    @patch("backend.rag.generator._get_oai")
    @patch("backend.rag.reranker._get_oai")
    def test_query_path_taken_is_list(self, mock_reranker, mock_gen, mock_ada):
        mock_ada.return_value = self._make_mock_oai('{"intent": "semantic", "entities": []}')
        mock_gen.return_value = self._make_mock_oai("Answer.")
        mock_reranker.return_value = self._make_mock_oai('{"scores": [0.9]}')
        result = query("qsess_ada", "RAG pipeline?", retrieval_mode="single")
        assert isinstance(result.path_taken, list)

    @patch("backend.rag.adaptive_rag._get_oai_client")
    @patch("backend.rag.generator._get_oai")
    @patch("backend.rag.reranker._get_oai")
    def test_query_execution_time_positive(self, mock_reranker, mock_gen, mock_ada):
        mock_ada.return_value = self._make_mock_oai('{"intent": "semantic", "entities": []}')
        mock_gen.return_value = self._make_mock_oai("Answer.")
        mock_reranker.return_value = self._make_mock_oai('{"scores": [0.9]}')
        result = query("qsess_ada", "RAG?", retrieval_mode="single")
        assert result.execution_time_ms >= 0

    def test_query_unknown_session_returns_empty(self):
        # adaptive_rag.query returns a graceful response with no retrieved docs when session is absent
        result = query("no_such_session_xyz_abc", "test", retrieval_mode="single")
        assert isinstance(result, QueryResponse)
        assert result.retrieval_count == 0


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestAdaptiveRagStress:
    def setup_method(self):
        indexer_module._sessions.clear()

    def test_ingest_many_docs(self):
        for i in range(5):
            doc = f"# Section {i}\n\nThis is content for section {i}. More words here.\n\n## Sub {i}\n\nSubsection content."
            ingest(f"stress_{i}", doc)
        assert len(indexer_module._sessions) == 5

    def test_ingest_large_document(self):
        large_doc = SAMPLE_DOC * 5
        _, chunks, report = ingest("large_ada", large_doc)
        assert len(chunks) > 0
        assert "segment_scores" in report

    def test_ingest_unicode_document(self):
        uni_doc = "# 日本語セクション\n\nこれはテストです。\n\n## 中文标题\n\n这是测试内容。\n"
        _, chunks, report = ingest("uni_ada", uni_doc)
        assert isinstance(chunks, list)

    def test_strategy_report_completeness(self):
        _, _, report = ingest("full_report", SAMPLE_DOC)
        seg_scores = report["segment_scores"]
        assert isinstance(seg_scores, dict)
        for seg_key, strat_scores in seg_scores.items():
            assert isinstance(strat_scores, dict)
            for strat_name, metrics in strat_scores.items():
                assert "combined" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
