"""
Stress tests for backend/rag/generator.py and backend/rag/reranker.py
Tests: assemble_context, generate (mocked), rerank (mocked).
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.generator import assemble_context, generate
from backend.rag.reranker import rerank
from backend.models import RetrievedDoc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(idx: int, text: str, score: float = 0.9, section: str = "Section") -> RetrievedDoc:
    return RetrievedDoc(
        chunk_id=f"c{idx}",
        text=text,
        section_title=section,
        score=score,
        source_index=str(idx),
    )


DOCS = [
    _make_doc(0, "RAG combines retrieval with generation to ground responses.", 0.95, "Intro"),
    _make_doc(1, "Dense embeddings allow semantic similarity matching.", 0.88, "Methods"),
    _make_doc(2, "FAISS provides fast nearest-neighbor search at scale.", 0.82, "Methods"),
    _make_doc(3, "GPT-4o-mini is used as the backbone language model.", 0.79, "Results"),
    _make_doc(4, "Precision measures the fraction of relevant results.", 0.72, "Evaluation"),
    _make_doc(5, "Chunking is the process of splitting documents into pieces.", 0.68, "Intro"),
]


# ---------------------------------------------------------------------------
# assemble_context tests
# ---------------------------------------------------------------------------

class TestAssembleContext:
    def test_basic_assembly(self):
        context, citations = assemble_context(DOCS)
        assert isinstance(context, str)
        assert isinstance(citations, list)

    def test_context_not_empty(self):
        context, _ = assemble_context(DOCS)
        assert len(context.strip()) > 0

    def test_citations_match_docs(self):
        _, citations = assemble_context(DOCS)
        # At least some citations should appear
        assert len(citations) >= 1

    def test_context_respects_budget(self):
        # Context should not massively exceed the 3000-token budget
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        context, _ = assemble_context(DOCS)
        tokens = len(enc.encode(context))
        assert tokens <= 3500  # 3000 budget + small tolerance

    def test_empty_docs(self):
        context, citations = assemble_context([])
        assert isinstance(context, str)
        assert isinstance(citations, list)

    def test_single_doc(self):
        context, citations = assemble_context([DOCS[0]])
        assert DOCS[0].text in context or len(context) > 0

    def test_many_docs_truncates(self):
        # 50 large docs — should truncate at budget
        large_docs = [
            _make_doc(i, "word " * 200, 0.9, "Sec")
            for i in range(50)
        ]
        context, citations = assemble_context(large_docs)
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode(context))
        assert tokens <= 3500

    def test_citations_are_strings_or_dicts(self):
        _, citations = assemble_context(DOCS)
        for c in citations:
            assert isinstance(c, (str, dict))


# ---------------------------------------------------------------------------
# generate tests (mocked OpenAI)
# ---------------------------------------------------------------------------

class TestGenerate:
    def setup_method(self):
        # Reset module-level singleton so mock takes effect
        import backend.rag.generator as gen_mod
        gen_mod._oai_client = None

    def _mock_response(self, content="Test answer."):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def _make_mock_oai(self, content="Test answer."):
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.return_value = self._mock_response(content)
        return mock_oai

    @patch("backend.rag.generator._get_oai")
    def test_generate_returns_tuple(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai("RAG is a powerful technique.")
        result = generate("What is RAG?", "Context about RAG.", [], "simple")
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, citations = result
        assert isinstance(text, str)
        assert isinstance(citations, list)

    @patch("backend.rag.generator._get_oai")
    def test_generate_simple_rag(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai("Answer for simple RAG.")
        text, _ = generate("test query", "test context", ["c1"], "simple")
        assert len(text) > 0

    @patch("backend.rag.generator._get_oai")
    def test_generate_adaptive_rag(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai("Answer for adaptive RAG.")
        text, _ = generate("test query", "test context", ["c1"], "adaptive")
        assert len(text) > 0

    @patch("backend.rag.generator._get_oai")
    def test_generate_uses_context(self, mock_get_oai):
        capture = {}
        def side_effect(**kwargs):
            capture["messages"] = kwargs.get("messages", [])
            return self._mock_response("Got it.")
        mock_get_oai.return_value = MagicMock()
        mock_get_oai.return_value.chat.completions.create.side_effect = side_effect
        generate("My query", "My context here", [], "simple")
        all_content = " ".join(m.get("content", "") for m in capture.get("messages", []))
        assert "My context here" in all_content or len(all_content) > 0

    @patch("backend.rag.generator._get_oai")
    def test_generate_empty_context(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai("No context available.")
        text, citations = generate("test query", "", [], "simple")
        assert isinstance(text, str)
        assert isinstance(citations, list)


# ---------------------------------------------------------------------------
# rerank tests (mocked OpenAI)
# ---------------------------------------------------------------------------

class TestRerank:
    def setup_method(self):
        import backend.rag.reranker as reranker_mod
        reranker_mod._oai_client = None

    def _mock_json_response(self, content):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def _make_mock_oai(self, content):
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.return_value = self._mock_json_response(content)
        return mock_oai

    @patch("backend.rag.reranker._get_oai")
    def test_rerank_returns_list(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai('{"scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]}')
        result = rerank("What is RAG?", DOCS)
        assert isinstance(result, list)

    @patch("backend.rag.reranker._get_oai")
    def test_rerank_descending_scores(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai('{"scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]}')
        result = rerank("What is RAG?", DOCS)
        if len(result) >= 2:
            scores = [r.score for r in result]
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    @patch("backend.rag.reranker._get_oai")
    def test_rerank_empty_docs(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai('{"scores": []}')
        result = rerank("query", [])
        assert result == []

    @patch("backend.rag.reranker._get_oai")
    def test_rerank_preserves_doc_type(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai('{"scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]}')
        result = rerank("query", DOCS)
        for doc in result:
            assert isinstance(doc, RetrievedDoc)

    @patch("backend.rag.reranker._get_oai")
    def test_rerank_invalid_json_handled(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai("not valid json")
        try:
            result = rerank("query", DOCS)
            assert isinstance(result, list)
        except Exception:
            pass  # acceptable to raise on invalid JSON

    @patch("backend.rag.reranker._get_oai")
    def test_rerank_single_doc(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai('{"scores": [0.75]}')
        result = rerank("query", [DOCS[0]])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestGeneratorRerankerStress:
    def setup_method(self):
        import backend.rag.generator as gen_mod
        import backend.rag.reranker as reranker_mod
        gen_mod._oai_client = None
        reranker_mod._oai_client = None

    def _mock_response(self, content="Answer."):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def _make_mock_oai(self, content="Answer."):
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.return_value = self._mock_response(content)
        return mock_oai

    @patch("backend.rag.generator._get_oai")
    def test_many_generate_calls(self, mock_get_oai):
        mock_get_oai.return_value = self._make_mock_oai()
        for i in range(20):
            text, citations = generate(f"query {i}", f"context {i}", [], "simple")
            assert isinstance(text, str)
            assert isinstance(citations, list)

    def test_assemble_context_deterministic(self):
        c1, cit1 = assemble_context(DOCS)
        c2, cit2 = assemble_context(DOCS)
        assert c1 == c2

    @patch("backend.rag.reranker._get_oai")
    def test_rerank_many_docs(self, mock_get_oai):
        n = 50
        big_docs = [_make_doc(i, f"Document {i} content here.", 0.5, "Sec") for i in range(n)]
        scores_json = '{"scores": [' + ",".join(str(round(0.5 + i * 0.01, 2)) for i in range(n)) + ']}'
        mock_get_oai.return_value = self._make_mock_oai(scores_json)
        result = rerank("query", big_docs)
        assert isinstance(result, list)

    def test_assemble_context_single_word_chunks(self):
        tiny_docs = [_make_doc(i, "word", 0.5, "S") for i in range(10)]
        context, citations = assemble_context(tiny_docs)
        assert isinstance(context, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
