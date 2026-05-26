"""Structural parser — extract document sections/headings from raw text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class StructuralSegment:
    title: str
    level: int             # 1 = top, 2 = sub, 3 = subsub
    parent_title: str
    text: str              # raw text of this segment (may span many lines)
    char_start: int
    char_end: int


# Heading patterns (ordered by specificity)
_HEADING_PATTERNS = [
    # Markdown headings
    (re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE), "markdown"),
    # ALL-CAPS line (≥3 words, ≤80 chars) typical of PDF sections
    (re.compile(r"^([A-Z][A-Z0-9 \-:]{2,78})$", re.MULTILINE), "allcaps"),
    # Numbered headings: "1.", "1.1", "1.1.1" at line start
    (re.compile(r"^(\d+(?:\.\d+){0,2})\s{1,4}([A-Z].{0,80})$", re.MULTILINE), "numbered"),
    # Title-case line ≤ 80 chars followed by blank line
    (re.compile(r"^([A-Z][a-zA-Z0-9 \-:]{2,78})\n\n", re.MULTILINE), "titlecase"),
]


def _detect_headings(text: str) -> list[tuple[int, int, str, int]]:
    """Return list of (char_start, char_end, heading_text, level) sorted by position."""
    found: dict[int, tuple[int, int, str, int]] = {}

    for pattern, kind in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            if pos in found:
                continue  # first match wins
            if kind == "markdown":
                level = len(m.group(1))
                title = m.group(2).strip()
            elif kind == "numbered":
                dots = m.group(1).count(".")
                level = dots + 1
                title = m.group(2).strip()
            elif kind == "allcaps":
                level = 1
                title = m.group(0).strip()
            else:
                level = 2
                title = m.group(1).strip()

            found[pos] = (pos, m.end(), title, min(level, 3))

    return sorted(found.values(), key=lambda x: x[0])


def parse_structure(text: str) -> list[StructuralSegment]:
    """Split document into structural segments using detected headings."""
    headings = _detect_headings(text)

    if not headings:
        # No detectable structure — treat whole document as one segment
        return [
            StructuralSegment(
                title="Document",
                level=1,
                parent_title="",
                text=text,
                char_start=0,
                char_end=len(text),
            )
        ]

    segments: list[StructuralSegment] = []
    # Text before first heading (preamble)
    first_h_start = headings[0][0]
    if first_h_start > 0:
        preamble = text[:first_h_start].strip()
        if preamble:
            segments.append(
                StructuralSegment(
                    title="Introduction",
                    level=1,
                    parent_title="",
                    text=preamble,
                    char_start=0,
                    char_end=first_h_start,
                )
            )

    parent_stack: list[tuple[str, int]] = []  # (title, level)

    for i, (h_start, h_end, h_title, h_level) in enumerate(headings):
        # Text of this section = from end of this heading to start of next heading
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        section_text = text[h_end:next_start].strip()

        # Resolve parent
        while parent_stack and parent_stack[-1][1] >= h_level:
            parent_stack.pop()
        parent_title = parent_stack[-1][0] if parent_stack else ""
        parent_stack.append((h_title, h_level))

        segments.append(
            StructuralSegment(
                title=h_title,
                level=h_level,
                parent_title=parent_title,
                text=section_text,
                char_start=h_start,
                char_end=next_start,
            )
        )

    return segments
