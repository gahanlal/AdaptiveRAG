"""Simple RAG pipeline — fixed-size chunking + FAISS + GPT-4o-mini.

Vector store: faiss-cpu (IndexFlatIP — inner product = cosine for L2-normalised vecs)
Embeddings:   sentence-transformers/all-MiniLM-L6-v2 (free, local, no API key)
No ChromaDB, no spaCy, Python 3.9 compatible.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from backend.models import QueryResponse, RetrievedDoc, SimpleChunk
from backend.rag.llm_client import get_client

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_embedder: Optional[SentenceTransformer] = None

_oai_client = None
_enc = tiktoken.get_encoding("cl100k_base")

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50
TOP_K = 5
EMBED_DIM = 384   # all-MiniLM-L6-v2 output dimension


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_oai():
    global _oai_client
    if _oai_client is None:
        _oai_client, _ = get_client()
    return _oai_client


def _get_model() -> str:
    _, model = get_client()
    return model


# ---------------------------------------------------------------------------
# Per-session FAISS store
# ---------------------------------------------------------------------------

@dataclass
class _SimpleSession:
    chunks: list[SimpleChunk] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    index: Optional[faiss.IndexFlatIP] = None     # inner product for cosine sim


_sessions: dict[str, _SimpleSession] = {}


# ---------------------------------------------------------------------------
# Chunking (fixed-size with token overlap)
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def chunk_text(doc_id: str, text: str) -> list[SimpleChunk]:
    """Split text into fixed-size token chunks with overlap."""
    tokens = _enc.encode(text)
    chunks: list[SimpleChunk] = []
    step = max(1, CHUNK_TOKENS - OVERLAP_TOKENS)
    idx = 0

    for i in range(0, len(tokens), step):
        window = tokens[i: i + CHUNK_TOKENS]
        if not window:
            break
        try:
            chunk_text_str = _enc.decode(window)
        except Exception:
            chunk_text_str = ""

        # Compute char positions from actual token prefix lengths (accurate even with overlap)
        char_start = len(_enc.decode(tokens[:i]))
        char_end = len(_enc.decode(tokens[:i + len(window)]))

        chunks.append(
            SimpleChunk(
                chunk_id=f"{doc_id}_simple_{idx}_{uuid.uuid4().hex[:6]}" if doc_id else f"simple_{idx}_{uuid.uuid4().hex[:6]}",
                index=idx,
                text=chunk_text_str,
                token_count=len(window),
                char_start=char_start,
                char_end=char_end,
            )
        )
        idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Ingest — build FAISS index for the session
# ---------------------------------------------------------------------------

def ingest(session_id: str, text: str) -> list[SimpleChunk]:
    """Chunk text, embed with all-MiniLM-L6-v2, store in FAISS. Returns chunks."""
    chunks = chunk_text(session_id, text)
    embedder = _get_embedder()

    texts = [c.text for c in chunks]
    if not texts:
        idx = faiss.IndexFlatIP(EMBED_DIM)
        _sessions[session_id] = _SimpleSession(chunks=chunks, texts=texts, index=idx)
        return chunks

    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)

    _sessions[session_id] = _SimpleSession(chunks=chunks, texts=texts, index=idx)
    return chunks


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(session_id: str, user_query: str) -> QueryResponse:
    """Retrieve top-k chunks and generate a response via GPT-4o-mini."""
    t0 = time.perf_counter()

    session = _sessions.get(session_id)
    if session is None or session.index is None:
        return QueryResponse(
            response="Session not found. Please ingest a document first.",
            citations=[],
            intent="error",
            path_taken=[],
            retrieved_docs=[],
            retrieval_count=0,
            execution_time_ms=0.0,
        ).model_dump()

    embedder = _get_embedder()
    q_emb = embedder.encode([user_query], normalize_embeddings=True, show_progress_bar=False)
    q_emb = q_emb.astype(np.float32)

    k = min(TOP_K, len(session.texts))
    scores_arr, indices_arr = session.index.search(q_emb, k)

    retrieved: list[RetrievedDoc] = []
    for score, idx_val in zip(scores_arr[0], indices_arr[0]):
        if idx_val < 0 or idx_val >= len(session.texts):
            continue
        chunk = session.chunks[idx_val]
        retrieved.append(
            RetrievedDoc(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                section_title="—",
                score=round(float(score), 4),
                source_index="vector",
            )
        )

    retrieved.sort(key=lambda x: x.score, reverse=True)

    context = "\n\n---\n\n".join(
        f"[Chunk {r.chunk_id}]\n{r.text}" for r in retrieved
    )

    oai = _get_oai()
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the provided context. Cite sources as [Chunk <chunk_id>]. "
        "If the context does not contain enough information, say so."
    )
    completion = oai.chat.completions.create(
        model=_get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
        ],
        temperature=0.2,
    )
    response_text = completion.choices[0].message.content or ""
    citations = [r.chunk_id for r in retrieved if r.chunk_id in response_text]

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    result = QueryResponse(
        response=response_text,
        citations=citations,
        intent="semantic",
        path_taken=["FixedChunker(512tok)", "FAISS_IP_Index", "GPT4oMini_Generator"],
        retrieved_docs=retrieved,
        retrieval_count=len(retrieved),
        execution_time_ms=elapsed_ms,
    )
    return result.model_dump()
