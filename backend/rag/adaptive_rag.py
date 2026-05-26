"""
Adaptive RAG — LangGraph StateGraph

Node order (mirrors the architecture from the design post):
  structural_parser
    → adaptive_chunker
      → multi_index_builder
        → query_understanding
          → retrieval_router  (conditional: single vs multi)
            → reranker_node
              → context_assembler
                → generator_node

Structure ALWAYS comes before chunking.
Chunking ALWAYS operates within structural boundaries.
"""
from __future__ import annotations

import os
import time
import uuid
import json
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

from backend.models import AdaptiveChunk, RetrievedDoc, QueryResponse, StructureNode
from backend.rag.parser import StructuralSegment, parse_structure
from backend.rag.chunker import adaptive_chunk_all
from backend.rag import indexer as idx_module
from backend.rag.retriever import retrieve_single, retrieve_multi, retrieve_custom
from backend.rag.reranker import rerank
from backend.rag.generator import assemble_context, generate

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    # Inputs
    session_id: str
    query: str
    rag_type: str            # always "adaptive" for this graph
    retrieval_mode: str      # "single" | "multi"
    raw_text: str            # populated during ingest phase

    # Structural parsing output
    segments: list[StructuralSegment]

    # Chunking output
    chunks: list[AdaptiveChunk]

    # Query understanding output
    intent: str              # "semantic" | "structural" | "metadata" | "graph"
    entities: list[str]
    filters: dict

    # Retrieval output
    retrieved_docs: list[RetrievedDoc]
    reranked_docs: list[RetrievedDoc]

    # Generation output
    context: str
    response: str
    citations: list[str]

    # Observability
    pipeline_steps: list[str]
    execution_time_ms: float
    t_start: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ada_oai_client = None


def _get_oai_client():
    global _ada_oai_client
    if _ada_oai_client is None:
        from backend.rag.llm_client import get_client
        _ada_oai_client, _ = get_client()
    return _ada_oai_client


def _get_model() -> str:
    from backend.rag.llm_client import get_client
    _, model = get_client()
    return model


def _append_step(state: RAGState, step: str) -> list[str]:
    return list(state.get("pipeline_steps", [])) + [step]


# ---------------------------------------------------------------------------
# Node 1: Structural Parser
# Structure FIRST — understand document organisation before any chunking.
# ---------------------------------------------------------------------------

def structural_parser_node(state: RAGState) -> dict:
    text = state["raw_text"]
    segments = parse_structure(text)
    return {
        "segments": segments,
        "pipeline_steps": _append_step(state, "StructuralParser"),
    }


# ---------------------------------------------------------------------------
# Node 2: Adaptive Chunker
# Chunking happens WITHIN structural boundaries.
# Boundary signals: cosine drop, entity shift, token cap, section boundary.
# ---------------------------------------------------------------------------

def adaptive_chunker_node(state: RAGState) -> dict:
    doc_id = state["session_id"]
    chunks, report = adaptive_chunk_all(state["segments"], doc_id)
    winner_counts = report.get("winner_counts", {})
    top_winner = max(winner_counts, key=winner_counts.get) if winner_counts else "unknown"
    return {
        "chunks": chunks,
        "pipeline_steps": _append_step(state, f"AdaptiveChunker[selected={top_winner},n={len(chunks)}]"),
    }


# ---------------------------------------------------------------------------
# Node 3: Multi-Index Builder
# Builds: VectorIndex + StructuralIndex (PageIndex) + MetadataIndex + GraphIndex
# ---------------------------------------------------------------------------

def multi_index_builder_node(state: RAGState) -> dict:
    idx_module.build_indexes(state["session_id"], state["chunks"])
    return {
        "pipeline_steps": _append_step(state, "MultiIndexBuilder"),
    }


# ---------------------------------------------------------------------------
# Node 4: Query Understanding
# Classify intent and extract entities.
# Intent classes: semantic | structural | metadata | graph
# ---------------------------------------------------------------------------

def query_understanding_node(state: RAGState) -> dict:
    intent = "semantic"
    entities: list[str] = []
    try:
        oai = _get_oai_client()
        prompt = (
            f"Classify this query into ONE intent category:\n"
            f"- semantic: general factual / conceptual question\n"
            f"- structural: asking about a specific section, chapter, or topic location\n"
            f"- metadata: asking about document-level info (author, date, title, level)\n"
            f"- graph: asking about relationships between entities or people\n\n"
            f"Also extract up to 5 named entities from the query.\n\n"
            f"Query: {state['query']}\n\n"
            f"Return JSON: {{\"intent\": \"...\", \"entities\": [...]}}"
        )
        resp = oai.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        intent = data.get("intent", "semantic")
        entities = data.get("entities", [])
    except Exception:
        pass

    return {
        "intent": intent,
        "entities": entities,
        "filters": {},
        "pipeline_steps": _append_step(state, f"QueryUnderstanding[intent={intent}]"),
    }


# ---------------------------------------------------------------------------
# Node 5a: Single Retriever — routes to ONE index based on intent
# ---------------------------------------------------------------------------

def single_retrieval_node(state: RAGState) -> dict:
    multi_index = idx_module.get_index(state["session_id"])
    if multi_index is None:
        return {"retrieved_docs": [], "pipeline_steps": _append_step(state, "SingleRetriever[no_index]")}

    docs = retrieve_single(
        index=multi_index,
        query=state["query"],
        intent=state["intent"],
        entities=state["entities"],
    )
    index_used = state["intent"] if state["intent"] in ("structural", "metadata", "graph") else "vector"
    return {
        "retrieved_docs": docs,
        "pipeline_steps": _append_step(state, f"SingleRetriever[{index_used}]"),
    }


# ---------------------------------------------------------------------------
# Node 5b: Multi Retriever — runs ALL indexes in parallel, deduplicates
# ---------------------------------------------------------------------------

def multi_retrieval_node(state: RAGState) -> dict:
    multi_index = idx_module.get_index(state["session_id"])
    if multi_index is None:
        return {"retrieved_docs": [], "pipeline_steps": _append_step(state, "MultiRetriever[no_index]")}

    docs = retrieve_multi(
        index=multi_index,
        query=state["query"],
        entities=state["entities"],
    )
    return {
        "retrieved_docs": docs,
        "pipeline_steps": _append_step(state, "MultiRetriever[vector+structural+metadata+graph]"),
    }


# ---------------------------------------------------------------------------
# Node 5c: Custom Retriever — runs user-selected indexes in parallel
# ---------------------------------------------------------------------------

def custom_retrieval_node(state: RAGState) -> dict:
    multi_index = idx_module.get_index(state["session_id"])
    if multi_index is None:
        return {"retrieved_docs": [], "pipeline_steps": _append_step(state, "CustomRetriever[no_index]")}

    selected = [s.strip() for s in state.get("retrieval_mode", "vector").split(",") if s.strip()]
    docs = retrieve_custom(
        index=multi_index,
        query=state["query"],
        entities=state["entities"],
        selected=selected,
    )
    label = "+".join(selected) if selected else "vector"
    return {
        "retrieved_docs": docs,
        "pipeline_steps": _append_step(state, f"CustomRetriever[{label}]"),
    }


# ---------------------------------------------------------------------------
# Node 5c: Custom Retriever — runs user-selected indexes in parallel
# ---------------------------------------------------------------------------

def custom_retrieval_node(state: RAGState) -> dict:
    multi_index = idx_module.get_index(state["session_id"])
    if multi_index is None:
        return {"retrieved_docs": [], "pipeline_steps": _append_step(state, "CustomRetriever[no_index]")}

    selected = [s.strip() for s in state.get("retrieval_mode", "vector").split(",") if s.strip()]
    docs = retrieve_custom(
        index=multi_index,
        query=state["query"],
        entities=state["entities"],
        selected=selected,
    )
    label = "+".join(selected) if selected else "vector"
    return {
        "retrieved_docs": docs,
        "pipeline_steps": _append_step(state, f"CustomRetriever[{label}]"),
    }


# ---------------------------------------------------------------------------
# Node 6: Reranker
# LLM scores each chunk 0–1 for relevance; returns top-k
# ---------------------------------------------------------------------------

def reranker_node(state: RAGState) -> dict:
    try:
        reranked = rerank(state["query"], state["retrieved_docs"])
    except Exception:
        reranked = state["retrieved_docs"]
    return {
        "reranked_docs": reranked,
        "pipeline_steps": _append_step(state, f"LLMReranker[top_{len(reranked)}]"),
    }


# ---------------------------------------------------------------------------
# Node 7: Context Assembler
# Builds context string within 3000 token budget; preserves hierarchy labels
# ---------------------------------------------------------------------------

def context_assembler_node(state: RAGState) -> dict:
    context, citations = assemble_context(state["reranked_docs"])
    return {
        "context": context,
        "citations": citations,
        "pipeline_steps": _append_step(state, "ContextAssembler"),
    }


# ---------------------------------------------------------------------------
# Node 8: Generator
# Grounded response with inline [Section | chunk_id] citations
# ---------------------------------------------------------------------------

def generator_node(state: RAGState) -> dict:
    try:
        response_text, used_citations = generate(
            query=state["query"],
            context=state["context"],
            citations=state["citations"],
            rag_type="adaptive",
        )
    except Exception:
        response_text = "Unable to generate a response at this time."
        used_citations = []
    elapsed = round((time.perf_counter() - state.get("t_start", time.perf_counter())) * 1000, 1)
    return {
        "response": response_text,
        "citations": used_citations,
        "execution_time_ms": elapsed,
        "pipeline_steps": _append_step(state, "Generator"),
    }


# ---------------------------------------------------------------------------
# Conditional edge: single vs multi retrieval mode
# ---------------------------------------------------------------------------

def _route_retrieval(state: RAGState) -> str:
    mode = state.get("retrieval_mode", "single")
    if mode == "single":
        return "single"
    if mode == "multi":
        return "multi"
    return "custom"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    builder = StateGraph(RAGState)

    builder.add_node("structural_parser", structural_parser_node)
    builder.add_node("adaptive_chunker", adaptive_chunker_node)
    builder.add_node("multi_index_builder", multi_index_builder_node)
    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("single_retriever", single_retrieval_node)
    builder.add_node("multi_retriever", multi_retrieval_node)
    builder.add_node("custom_retriever", custom_retrieval_node)
    builder.add_node("reranker", reranker_node)
    builder.add_node("context_assembler", context_assembler_node)
    builder.add_node("generator", generator_node)

    # Linear flow up to retrieval router
    builder.add_edge(START, "structural_parser")
    builder.add_edge("structural_parser", "adaptive_chunker")
    builder.add_edge("adaptive_chunker", "multi_index_builder")
    builder.add_edge("multi_index_builder", "query_understanding")

    # Conditional branch: single vs multi vs custom retrieval
    builder.add_conditional_edges(
        "query_understanding",
        _route_retrieval,
        {"single": "single_retriever", "multi": "multi_retriever", "custom": "custom_retriever"},
    )

    # All branches converge at reranker
    builder.add_edge("single_retriever", "reranker")
    builder.add_edge("multi_retriever", "reranker")
    builder.add_edge("custom_retriever", "reranker")

    # Final linear flow
    builder.add_edge("reranker", "context_assembler")
    builder.add_edge("context_assembler", "generator")
    builder.add_edge("generator", END)

    return builder


_compiled_graph = None
_query_graph = None   # cached query-phase graph


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph().compile()
    return _compiled_graph


def _get_query_graph():
    """Build and cache the query-phase LangGraph (no ingest nodes)."""
    global _query_graph
    if _query_graph is not None:
        return _query_graph

    builder = StateGraph(RAGState)
    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("single_retriever", single_retrieval_node)
    builder.add_node("multi_retriever", multi_retrieval_node)
    builder.add_node("custom_retriever", custom_retrieval_node)
    builder.add_node("reranker", reranker_node)
    builder.add_node("context_assembler", context_assembler_node)
    builder.add_node("generator", generator_node)

    builder.add_edge(START, "query_understanding")
    builder.add_conditional_edges(
        "query_understanding",
        _route_retrieval,
        {"single": "single_retriever", "multi": "multi_retriever", "custom": "custom_retriever"},
    )
    builder.add_edge("single_retriever", "reranker")
    builder.add_edge("multi_retriever", "reranker")
    builder.add_edge("custom_retriever", "reranker")
    builder.add_edge("reranker", "context_assembler")
    builder.add_edge("context_assembler", "generator")
    builder.add_edge("generator", END)

    _query_graph = builder.compile()
    return _query_graph


# ---------------------------------------------------------------------------
# Public API: ingest (build indexes) + query
# ---------------------------------------------------------------------------

def ingest(
    session_id: str,
    text: str,
) -> tuple[list[StructuralSegment], list[AdaptiveChunk], dict]:
    """
    Run structural parsing + adaptive chunking + index building.
    Returns (segments, chunks, strategy_report) for the preview UI.
    strategy_report contains {"segment_scores": {...}, "winner_counts": {...}}
    """
    segments = parse_structure(text)
    chunks, strategy_report = adaptive_chunk_all(segments, session_id)
    idx_module.build_indexes(session_id, chunks)
    return segments, chunks, strategy_report


def query(
    session_id: str,
    user_query: str,
    retrieval_mode: str = "single",
) -> QueryResponse:
    """
    Run query-time nodes (query_understanding → retrieve → rerank → generate).
    Assumes ingest() has already been called for this session_id.
    """
    t0 = time.perf_counter()

    graph = _get_query_graph()

    initial_state: RAGState = {
        "session_id": session_id,
        "query": user_query,
        "rag_type": "adaptive",
        "retrieval_mode": retrieval_mode,
        "raw_text": "",
        "segments": [],
        "chunks": [],
        "intent": "",
        "entities": [],
        "filters": {},
        "retrieved_docs": [],
        "reranked_docs": [],
        "context": "",
        "response": "",
        "citations": [],
        "pipeline_steps": [],
        "execution_time_ms": 0.0,
        "t_start": t0,
    }

    final_state: RAGState = graph.invoke(initial_state)

    return QueryResponse(
        response=final_state.get("response", ""),
        citations=final_state.get("citations", []),
        intent=final_state.get("intent", "semantic"),
        path_taken=final_state.get("pipeline_steps", []),
        retrieved_docs=final_state.get("reranked_docs", []),
        retrieval_count=len(final_state.get("retrieved_docs", [])),
        execution_time_ms=final_state.get("execution_time_ms", 0.0),
    )
