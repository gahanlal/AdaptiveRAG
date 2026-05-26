"""FastAPI backend — /api/ingest, /api/query, /api/health"""
from __future__ import annotations

import io
import uuid
import chardet
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

import backend.rag.simple_rag as simple_rag
import backend.rag.adaptive_rag as adaptive_rag
from backend.models import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SimpleChunk,
    AdaptiveChunk,
    StructureNode,
)

app = FastAPI(title="Adaptive RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Document text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _extract_text_from_docx(data: bytes) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(io.BytesIO(data))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_text(filename: str, data: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    if ext == "pdf":
        return _extract_text_from_pdf(data)
    elif ext == "docx":
        return _extract_text_from_docx(data)
    else:
        # TXT or plain bytes — detect encoding
        detected = chardet.detect(data)
        encoding = detected.get("encoding") or "utf-8"
        return data.decode(encoding, errors="replace")


# ---------------------------------------------------------------------------
# Build StructureNode tree from segments
# ---------------------------------------------------------------------------

def _build_structure_tree(
    segments, chunks: list[AdaptiveChunk]
) -> list[StructureNode]:
    """Convert flat segments + chunks into a nested StructureNode tree."""
    # Map section_title → chunk_ids
    section_to_chunks: dict[str, list[str]] = {}
    for c in chunks:
        section_to_chunks.setdefault(c.section_title, []).append(c.chunk_id)

    # Build flat nodes
    nodes: list[StructureNode] = []
    for seg in segments:
        node = StructureNode(
            title=seg.title,
            level=seg.level,
            chunk_ids=section_to_chunks.get(seg.title, []),
            children=[],
        )
        nodes.append(node)

    # Build hierarchy using parent_title references
    title_to_node: dict[str, StructureNode] = {n.title: n for n in nodes}
    roots: list[StructureNode] = []

    for seg, node in zip(segments, nodes):
        if seg.parent_title and seg.parent_title in title_to_node:
            title_to_node[seg.parent_title].children.append(node)
        else:
            roots.append(node)

    return roots if roots else nodes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    file: Optional[UploadFile] = File(default=None),
    raw_text: Optional[str] = Form(default=None),
    filename: str = Form(default="document.txt"),
):
    """
    Accept a file upload OR raw pasted text.
    Returns side-by-side Simple and Adaptive chunk previews.
    """
    # --- Extract text ---
    if file is not None and getattr(file, 'filename', None):
        data = await file.read()
        fname = file.filename
        text = _extract_text(fname, data)
    elif raw_text:
        fname = filename
        text = raw_text
    else:
        raise HTTPException(status_code=422, detail="Provide either a file or raw_text.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Document appears to be empty.")

    session_id = uuid.uuid4().hex

    # --- Simple RAG: fixed chunking ---
    simple_chunks: list[SimpleChunk] = simple_rag.ingest(session_id, text)

    # --- Adaptive RAG: structure → chunk → index ---
    segments, adaptive_chunks, strategy_report = adaptive_rag.ingest(session_id, text)

    # --- Build structural tree for UI ---
    structure_tree = _build_structure_tree(segments, adaptive_chunks)

    return IngestResponse(
        session_id=session_id,
        filename=fname,
        char_count=len(text),
        word_count=len(text.split()),
        simple_chunks=simple_chunks,
        adaptive_chunks=adaptive_chunks,
        structure_tree=structure_tree,
        strategy_scores=strategy_report,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run the selected pipeline for a query."""
    if req.rag_type == "simple":
        result = simple_rag.query(req.session_id, req.query)
    elif req.rag_type == "adaptive":
        result = adaptive_rag.query(
            session_id=req.session_id,
            user_query=req.query,
            retrieval_mode=req.retrieval_mode,
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unknown rag_type: {req.rag_type!r}")

    return result
