"""LLM-based reranker — scores each retrieved chunk 0–1 for relevance."""
from __future__ import annotations

import os
import json

from backend.models import RetrievedDoc
from backend.rag.llm_client import chat_with_fallback

TOP_K_RERANKED = 5

def rerank(query: str, docs: list[RetrievedDoc], top_k: int = TOP_K_RERANKED) -> list[RetrievedDoc]:
    """Score each doc 0.0–1.0 for relevance to query. Return top_k sorted descending."""
    if not docs:
        return []

    # Build a compact scoring prompt
    doc_list = "\n".join(
        f"{i}. [{d.section_title}] {d.text[:300]}"
        for i, d in enumerate(docs)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Documents:\n{doc_list}\n\n"
        "For each document (0-indexed), output a JSON object with key=index (int) "
        "and value=score (float 0.0–1.0) based on relevance to the query. "
        "Return ONLY valid JSON like: {\"0\": 0.9, \"1\": 0.3, ...}"
    )

    try:
        completion = chat_with_fallback(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise relevance scorer. "
                        "Return only the JSON scores with no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        scores: dict[str, float] = json.loads(raw)
    except Exception:
        # Fallback: keep original retrieval scores
        return sorted(docs, key=lambda d: d.score, reverse=True)[:top_k]

    # Apply LLM scores, fall back to retrieval score on missing keys
    scored_docs: list[RetrievedDoc] = []
    for i, doc in enumerate(docs):
        llm_score = float(scores.get(str(i), doc.score))
        scored_docs.append(
            RetrievedDoc(
                chunk_id=doc.chunk_id,
                text=doc.text,
                section_title=doc.section_title,
                score=round(llm_score, 4),
                source_index=doc.source_index,
            )
        )

    scored_docs.sort(key=lambda d: d.score, reverse=True)
    return scored_docs[:top_k]
