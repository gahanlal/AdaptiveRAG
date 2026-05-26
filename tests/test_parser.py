"""
Stress tests for backend/rag/parser.py
Tests: parse_structure with markdown headings, ALLCAPS, numbered headings,
       title-case, mixed, empty text, edge cases.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.parser import parse_structure, StructuralSegment

# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

MARKDOWN_DOC = """# Introduction

This is the introduction paragraph. It contains some text about the topic.

## Background

Some background information here. This explains the context.

## Related Work

Prior work on this topic includes several approaches.

# Methodology

We describe our approach in detail.

## Data Collection

Data was collected from multiple sources.

## Processing

Processing involved several steps.

# Results

We present our findings here.
"""

ALLCAPS_DOC = """INTRODUCTION

This section provides an overview of the system.

METHODOLOGY

We describe the methods used in detail.

RESULTS AND DISCUSSION

The results are presented in this section.
"""

NUMBERED_DOC = """1. Introduction

This is section one.

1.1 Background

This is subsection 1.1 content.

1.2 Related Work

This is subsection 1.2.

2. Methods

Section two content.

2.1 Data

Subsection 2.1.

3. Conclusion

Final section.
"""

MIXED_DOC = """# Title Section

Introduction text here.

UPPERCASE HEADING

Some content under uppercase.

## Subsection One

Content of subsection.

1.1 Numbered subsection

Content here.
"""

PLAIN_DOC = """This is just plain text with no headings at all.
It goes on for multiple lines.
There are no structural elements.
"""


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestParseStructure:
    def test_markdown_headings(self):
        segments = parse_structure(MARKDOWN_DOC)
        assert len(segments) > 0
        titles = [s.title for s in segments]
        # Should detect Introduction, Background etc.
        assert any("Introduction" in t for t in titles)

    def test_markdown_levels(self):
        segments = parse_structure(MARKDOWN_DOC)
        levels = {s.level for s in segments}
        # Should detect at least 2 levels (H1 and H2)
        assert len(levels) >= 2

    def test_allcaps_headings(self):
        segments = parse_structure(ALLCAPS_DOC)
        assert len(segments) > 0
        # METHODOLOGY should be detected
        titles = [s.title.upper() for s in segments]
        assert any("METHOD" in t for t in titles)

    def test_numbered_headings(self):
        segments = parse_structure(NUMBERED_DOC)
        assert len(segments) > 0
        titles = [s.title for s in segments]
        assert any("Introduction" in t or "1." in t for t in titles)

    def test_plain_doc_returns_segments(self):
        segments = parse_structure(PLAIN_DOC)
        # Even without headings, parser should return at least one segment (full text)
        assert isinstance(segments, list)
        assert len(segments) >= 1

    def test_empty_string(self):
        segments = parse_structure("")
        assert isinstance(segments, list)

    def test_single_heading_no_body(self):
        segments = parse_structure("# Heading Only\n")
        assert isinstance(segments, list)

    def test_segment_text_not_empty(self):
        segments = parse_structure(MARKDOWN_DOC)
        # At least some segments should have text
        texts = [s.text.strip() for s in segments if s.text.strip()]
        assert len(texts) > 0

    def test_char_offsets_non_negative(self):
        segments = parse_structure(MARKDOWN_DOC)
        for s in segments:
            assert s.char_start >= 0
            assert s.char_end >= s.char_start

    def test_char_offsets_within_doc(self):
        segments = parse_structure(MARKDOWN_DOC)
        doc_len = len(MARKDOWN_DOC)
        for s in segments:
            assert s.char_start <= doc_len
            assert s.char_end <= doc_len + 1  # +1 for end boundary tolerance

    def test_segments_are_dataclasses(self):
        segments = parse_structure(MARKDOWN_DOC)
        for s in segments:
            assert isinstance(s, StructuralSegment)
            assert hasattr(s, "title")
            assert hasattr(s, "level")
            assert hasattr(s, "text")
            assert hasattr(s, "parent_title")
            assert hasattr(s, "char_start")
            assert hasattr(s, "char_end")

    def test_mixed_headings(self):
        segments = parse_structure(MIXED_DOC)
        assert len(segments) > 0

    def test_parent_title_propagated(self):
        segments = parse_structure(MARKDOWN_DOC)
        # Subsections should have parent = the H1 above them
        h2_segs = [s for s in segments if s.level == 2]
        if h2_segs:
            for s in h2_segs:
                assert isinstance(s.parent_title, str)


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestParseStructureStress:
    def test_very_long_document(self):
        big_doc = MARKDOWN_DOC * 50
        segments = parse_structure(big_doc)
        assert len(segments) >= 5

    def test_many_headings(self):
        lines = []
        for i in range(100):
            lines.append(f"## Section {i}\n\nContent for section {i}. More text here.\n")
        doc = "\n".join(lines)
        segments = parse_structure(doc)
        assert len(segments) >= 10

    def test_nested_headings_correct_levels(self):
        doc = "# H1\n\nText.\n\n## H2\n\nText.\n\n### H3\n\nText.\n\n## Another H2\n\nText.\n"
        segments = parse_structure(doc)
        assert any(s.level == 1 for s in segments)
        assert any(s.level == 2 for s in segments)

    def test_unicode_headings(self):
        doc = "# 日本語タイトル\n\nContent.\n\n## 中文标题\n\nMore content.\n"
        segments = parse_structure(doc)
        assert isinstance(segments, list)

    def test_only_whitespace(self):
        segments = parse_structure("   \n\n   \n\t\n  ")
        assert isinstance(segments, list)

    def test_heading_at_end(self):
        doc = "# Intro\n\nSome text.\n\n# Trailing"
        segments = parse_structure(doc)
        assert len(segments) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
