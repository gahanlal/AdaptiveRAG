"""Context assembler + grounded generation with inline citations."""
from __future__ import annotations

import os
import tiktoken

from backend.models import RetrievedDoc
from backend.rag.llm_client import get_client

_oai_client = None
_enc = tiktoken.get_encoding("cl100k_base")
CONTEXT_TOKEN_BUDGET = 3000


def _get_oai():
    global _oai_client
    if _oai_client is None:
        _oai_client, _ = get_client()
    return _oai_client


def _get_model() -> str:
    _, model = get_client()
    return model


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(docs: list[RetrievedDoc]) -> tuple[str, list[str]]:
    """
    Build context string within CONTEXT_TOKEN_BUDGET tokens.
    Returns (context_string, list_of_citation_labels).
    """
    parts: list[str] = []
    citations: list[str] = []
    used_tokens = 0

    for doc in docs:
        label = f"[{doc.section_title} | {doc.chunk_id}]"
        block = f"{label}\n{doc.text}"
        block_tokens = _token_count(block)

        if used_tokens + block_tokens > CONTEXT_TOKEN_BUDGET:
            # Try to fit a truncated version
            remaining = CONTEXT_TOKEN_BUDGET - used_tokens
            if remaining < 50:
                break
            tokens = _enc.encode(doc.text)[:remaining - 20]
            truncated_text = _enc.decode(tokens) + "…"
            block = f"{label}\n{truncated_text}"

        parts.append(block)
        citations.append(label)
        used_tokens += _token_count(block)

        if used_tokens >= CONTEXT_TOKEN_BUDGET:
            break

    return "\n\n---\n\n".join(parts), citations


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    query: str,
    context: str,
    citations: list[str],
    rag_type: str = "adaptive",
) -> tuple[str, list[str]]:
    """
    Generate a grounded answer. Returns (response_text, used_citations).
    Citations are in format [Section Title | chunk_id].
    """
    oai = _get_oai()

    system_prompt = (
        "You are a precise, grounded question-answering assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "When you use information from a context block, cite it inline using "
        "the label exactly as it appears in brackets at the start of each block, "
        "e.g. [Introduction | ada_abc123ef].\n"
        "If the context does not contain enough information to answer, say: "
        "'The provided document does not contain sufficient information to answer this question.'\n"
        "Be concise and factual."
    )

    completion = oai.chat.completions.create(
        model=_get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.2,
    )

    response_text = completion.choices[0].message.content or ""

    # Extract which citations were actually referenced in the response
    used_citations = [c for c in citations if c in response_text]
    # Always include at least the top citation if model didn't inline any
    if not used_citations and citations:
        used_citations = citations[:1]

    return response_text, used_citations
