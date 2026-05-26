"""Pydantic request/response models shared between FastAPI routes and frontend."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared chunk representation
# ---------------------------------------------------------------------------

class SimpleChunk(BaseModel):
    chunk_id: str
    index: int
    text: str
    token_count: int
    char_start: int
    char_end: int


class AdaptiveChunk(BaseModel):
    chunk_id: str
    index: int
    text: str
    token_count: int
    section_title: str
    hierarchy_level: int
    parent_section: str
    boundary_reason: str   # winning strategy name, e.g. "recursive_512"
    entities: list[str]


class StructureNode(BaseModel):
    title: str
    level: int
    chunk_ids: list[str] = Field(default_factory=list)
    children: list["StructureNode"] = Field(default_factory=list)


StructureNode.model_rebuild()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    session_id: str
    filename: str
    char_count: int
    word_count: int
    # Simple RAG view
    simple_chunks: list[SimpleChunk]
    # Adaptive RAG view
    adaptive_chunks: list[AdaptiveChunk]
    structure_tree: list[StructureNode]
    # Strategy selection report from adaptive chunker
    # {"segment_scores": {seg: {strategy: {sc, icc, dcc, combined, n_chunks}}},
    #  "winner_counts":  {strategy_name: count}}
    strategy_scores: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    session_id: str
    query: str
    rag_type: str          # "simple" | "adaptive"
    retrieval_mode: str    # "single" | "multi"  (only used for adaptive)


class RetrievedDoc(BaseModel):
    chunk_id: str
    text: str
    section_title: str
    score: float
    source_index: str      # "vector" | "structural" | "metadata" | "graph"


class QueryResponse(BaseModel):
    response: str
    citations: list[str]
    intent: str            # only set for adaptive; "semantic" | "structural" | "metadata" | "graph"
    path_taken: list[str]  # ordered list of pipeline nodes that ran
    retrieved_docs: list[RetrievedDoc]
    retrieval_count: int
    execution_time_ms: float


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
