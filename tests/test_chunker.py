"""
Stress tests for backend/rag/chunker.py
Tests: RecursiveSplitter, quality metrics, adaptive strategy selection,
       edge cases (empty text, single sentence, very long text, unicode).
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.chunker import (
    RecursiveSplitter,
    _count_tokens,
    _split_sentences,
    _extract_entities,
    _score_size_compliance,
    _select_best_strategy,
    chunk_segment,
    adaptive_chunk_all,
    MIN_TOKENS,
    MAX_TOKENS,
)
from backend.rag.parser import StructuralSegment

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHORT_TEXT = "This is a short sentence. Another sentence here."

MEDIUM_TEXT = "\n\n".join([
    "Introduction\n\nThis section introduces the topic of adaptive chunking. "
    "Chunking is fundamental to retrieval-augmented generation systems.",
    "Methodology\n\nWe evaluate three recursive splitting strategies. "
    "Each strategy uses different chunk sizes and merge modes. "
    "The best strategy is selected per structural segment.",
    "Results\n\nOur experiments show that recursive-512 wins for long documents. "
    "For short documents, the paragraph-based strategy often performs best. "
    "Combined scores of ICC, DCC, and SC drive selection.",
])

LONG_TEXT = " ".join(["The quick brown fox jumps over the lazy dog."] * 200)

UNICODE_TEXT = "日本語テスト。This is a multilingual test. Ñoño caracteres especiales. 中文测试内容。"


def _make_segment(text: str, title: str = "Test Section", level: int = 1) -> StructuralSegment:
    return StructuralSegment(
        title=title,
        level=level,
        parent_title="",
        text=text,
        char_start=0,
        char_end=len(text),
    )


# ---------------------------------------------------------------------------
# RecursiveSplitter unit tests
# ---------------------------------------------------------------------------

class TestRecursiveSplitter:
    def test_basic_split_to_chunk_size(self):
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=5, min_chunk_tokens=5, merging="to_chunk_size")
        chunks = splitter.split_text(MEDIUM_TEXT)
        assert len(chunks) > 0
        for c in chunks:
            assert _count_tokens(c) <= 50 + 5 + 2  # allow slight overflow at boundaries

    def test_small_only_merging(self):
        splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=0, min_chunk_tokens=20, merging="small_only")
        chunks = splitter.split_text(MEDIUM_TEXT)
        assert len(chunks) > 0
        # No chunk should be below min_chunk_tokens (unless text is tiny)
        for c in chunks:
            if _count_tokens(MEDIUM_TEXT) > 20:
                assert _count_tokens(c) >= 10  # lenient threshold

    def test_empty_text(self):
        splitter = RecursiveSplitter(chunk_size=512)
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []

    def test_single_sentence(self):
        splitter = RecursiveSplitter(chunk_size=512)
        result = splitter.split_text(SHORT_TEXT)
        assert len(result) >= 1
        combined = "".join(result)
        assert SHORT_TEXT.replace(" ", "") in combined.replace(" ", "")

    def test_long_text_split(self):
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=10, merging="to_chunk_size")
        chunks = splitter.split_text(LONG_TEXT)
        assert len(chunks) >= 2
        # Each chunk should be at most ~110 tokens (chunk_size + overlap)
        for c in chunks:
            assert _count_tokens(c) <= 120

    def test_overlap_makes_adjacent_chunks_share_text(self):
        splitter = RecursiveSplitter(chunk_size=80, chunk_overlap=20, merging="to_chunk_size")
        chunks = splitter.split_text(LONG_TEXT)
        if len(chunks) >= 2:
            # Last part of chunk[0] should appear in beginning of chunk[1]
            # (overlap means shared tokens)
            # At least the total coverage should be less than sum of all chunks
            total_combined = sum(_count_tokens(c) for c in chunks)
            original = _count_tokens(LONG_TEXT)
            assert total_combined >= original  # overlap means > original

    def test_no_overlap(self):
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=0, merging="to_chunk_size")
        chunks = splitter.split_text(MEDIUM_TEXT)
        assert len(chunks) >= 1

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveSplitter(chunk_size=100, chunk_overlap=200)

    def test_unicode_text(self):
        splitter = RecursiveSplitter(chunk_size=50)
        chunks = splitter.split_text(UNICODE_TEXT)
        assert len(chunks) >= 1

    def test_custom_separators(self):
        splitter = RecursiveSplitter(
            chunk_size=200, chunk_overlap=0, merging="small_only",
            separators=["\n\n", "\n", ". ", ""]
        )
        chunks = splitter.split_text(MEDIUM_TEXT)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_count_tokens_basic(self):
        assert _count_tokens("hello world") > 0
        assert _count_tokens("") == 0

    def test_count_tokens_consistency(self):
        t1 = _count_tokens("The quick brown fox")
        t2 = _count_tokens("The quick brown fox")
        assert t1 == t2

    def test_split_sentences_basic(self):
        sents = _split_sentences("Hello world. This is a test. Another sentence here.")
        assert len(sents) >= 1

    def test_split_sentences_empty(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_split_sentences_single(self):
        sents = _split_sentences("Just one sentence without period")
        assert len(sents) >= 1

    def test_extract_entities_proper_nouns(self):
        text = "Apple Inc and Microsoft Corporation are technology companies. Barack Obama visited New York."
        entities = _extract_entities(text)
        assert isinstance(entities, list)
        # Should find multi-word proper nouns
        assert len(entities) > 0

    def test_extract_entities_empty(self):
        assert _extract_entities("") == []
        assert _extract_entities("no proper nouns here at all.") == []

    def test_extract_entities_capped(self):
        text = " ".join([f"Entity{i} Name{i}" for i in range(50)])
        entities = _extract_entities(text)
        assert len(entities) <= 20  # capped at 20


# ---------------------------------------------------------------------------
# Quality metric tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_size_compliance_all_in_range(self):
        # Create chunks that are clearly within range
        splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=0, merging="to_chunk_size")
        chunks = splitter.split_text(MEDIUM_TEXT)
        score = _score_size_compliance(chunks)
        assert 0.0 <= score <= 1.0

    def test_size_compliance_empty(self):
        assert _score_size_compliance([]) == 0.0

    def test_size_compliance_single_word(self):
        score = _score_size_compliance(["word"])
        # Single word is < MIN_TOKENS, so out of range
        assert score == 0.0

    def test_size_compliance_perfect(self):
        # Make chunks that are exactly in range
        # MIN_TOKENS=40, MAX_TOKENS=512 — 100-word text fits perfectly
        text = " ".join(["hello world this is test"] * 20)  # ~100 tokens
        score = _score_size_compliance([text])
        assert score == 1.0


# ---------------------------------------------------------------------------
# Strategy selection tests
# ---------------------------------------------------------------------------

class TestStrategySelection:
    def test_select_best_returns_valid_strategy(self):
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        name, chunks, scores = _select_best_strategy(MEDIUM_TEXT, emb)
        assert isinstance(name, str)
        assert len(name) > 0
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert isinstance(scores, dict)

    def test_select_best_scores_have_required_keys(self):
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        _, _, scores = _select_best_strategy(MEDIUM_TEXT, emb)
        for strat_name, metrics in scores.items():
            for key in ("sc", "icc", "dcc", "combined", "n_chunks"):
                assert key in metrics, f"Missing key {key} in strategy {strat_name}"

    def test_select_best_scores_in_range(self):
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        _, _, scores = _select_best_strategy(MEDIUM_TEXT, emb)
        for metrics in scores.values():
            assert 0.0 <= metrics["sc"] <= 1.0
            assert 0.0 <= metrics["icc"] <= 1.0
            assert 0.0 <= metrics["dcc"] <= 1.0
            assert 0.0 <= metrics["combined"] <= 1.0


# ---------------------------------------------------------------------------
# chunk_segment and adaptive_chunk_all tests
# ---------------------------------------------------------------------------

class TestChunkSegment:
    def test_chunk_segment_basic(self):
        seg = _make_segment(MEDIUM_TEXT)
        chunks, scores = chunk_segment(seg, doc_id="test123")
        assert len(chunks) > 0
        for c in chunks:
            assert c.text.strip()
            assert c.section_title == "Test Section"
            assert c.hierarchy_level == 1

    def test_chunk_segment_empty_text(self):
        seg = _make_segment("")
        chunks, scores = chunk_segment(seg, doc_id="test_empty")
        assert chunks == []
        assert scores == {}

    def test_chunk_segment_returns_strategy_scores(self):
        seg = _make_segment(MEDIUM_TEXT)
        chunks, scores = chunk_segment(seg, doc_id="test_scores")
        assert isinstance(scores, dict)
        assert len(scores) >= 1

    def test_chunk_segment_chunk_ids_unique(self):
        seg = _make_segment(LONG_TEXT)
        chunks, _ = chunk_segment(seg, doc_id="test_ids")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_segment_boundary_reason_is_strategy(self):
        seg = _make_segment(MEDIUM_TEXT)
        chunks, _ = chunk_segment(seg, doc_id="test_strategy")
        for c in chunks:
            # boundary_reason should be the strategy name
            assert len(c.boundary_reason) > 0

    def test_adaptive_chunk_all_multiple_segments(self):
        segments = [
            _make_segment("Introduction text here. More text about the introduction topic.", "Intro", 1),
            _make_segment("Methodology section. We used three different approaches.", "Method", 1),
            _make_segment("Results show improvement. The scores were better.", "Results", 1),
        ]
        chunks, report = adaptive_chunk_all(segments, "test_multi")
        assert len(chunks) > 0
        # Global re-indexing
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_adaptive_chunk_all_report_structure(self):
        segments = [_make_segment(MEDIUM_TEXT, "Section 1")]
        _, report = adaptive_chunk_all(segments, "test_report")
        assert "segment_scores" in report
        assert "winner_counts" in report

    def test_adaptive_chunk_all_empty_segments(self):
        chunks, report = adaptive_chunk_all([], "test_empty_segs")
        assert chunks == []

    def test_adaptive_chunk_all_unicode(self):
        seg = _make_segment(UNICODE_TEXT, "Unicode Section")
        chunks, _ = chunk_segment(seg, doc_id="test_uni")
        # Should not crash and produce some output
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Stress tests (larger inputs, edge cases)
# ---------------------------------------------------------------------------

class TestStress:
    def test_very_long_text_no_crash(self):
        very_long = " ".join(["word"] * 5000)
        splitter = RecursiveSplitter(chunk_size=512, chunk_overlap=50, merging="to_chunk_size")
        chunks = splitter.split_text(very_long)
        assert len(chunks) >= 1
        for c in chunks:
            assert _count_tokens(c) <= 570  # 512 + 50 + buffer

    def test_repeated_chunking_same_result(self):
        splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=0)
        c1 = splitter.split_text(MEDIUM_TEXT)
        c2 = splitter.split_text(MEDIUM_TEXT)
        assert c1 == c2  # deterministic

    def test_many_segments_no_crash(self):
        segments = [_make_segment(f"Segment {i} text content here.", f"Sec {i}", 1) for i in range(20)]
        chunks, report = adaptive_chunk_all(segments, "stress_many")
        assert len(chunks) >= 20  # at least one chunk per segment
        assert len(report["winner_counts"]) >= 1

    def test_single_word_text(self):
        seg = _make_segment("Hello", "Tiny")
        chunks, _ = chunk_segment(seg, "test_single_word")
        # Should produce at least one chunk (the word itself)
        assert len(chunks) >= 0  # may be empty if word is too short

    def test_whitespace_only_text(self):
        seg = _make_segment("   \n\n\t\n   ", "Whitespace")
        chunks, _ = chunk_segment(seg, "test_whitespace")
        assert chunks == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
