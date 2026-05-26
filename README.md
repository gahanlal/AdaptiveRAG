# Adaptive RAG Configurator

A FastAPI + Streamlit web app that lets you upload a document and compare **Simple RAG** vs **Adaptive RAG** side by side — then ask queries and see the full pipeline trace.

## Architecture

```
Document
  → Structural Parsing          (headings, sections, hierarchy)
  → Adaptive Semantic Chunking  (within structural boundaries)
  → Multi-Index Creation
      ├── Vector Index   (ChromaDB + all-MiniLM-L6-v2)
      ├── Structural/Page Index  (section hierarchy dict)
      ├── Metadata Index (level-based filtering)
      └── Graph Index    (NetworkX entity co-occurrence)
  → Query Understanding         (intent classification via GPT-4o-mini)
  → Hybrid Retrieval            (single-intent-routed OR parallel multi-index)
  → LLM Reranking               (GPT-4o-mini 0–1 scoring)
  → Context Assembly            (3 000-token budget)
  → Grounded Generation         (inline citations)
```

## Project Structure

```
AdaptiveRAG/
├── run.py                    ← starts both services
├── backend/
│   ├── main.py               ← FastAPI routes
│   ├── models.py             ← Pydantic models
│   └── rag/
│       ├── parser.py         ← structural parser
│       ├── chunker.py        ← adaptive semantic chunker
│       ├── indexer.py        ← multi-index builder
│       ├── retriever.py      ← single + multi retrieval
│       ├── reranker.py       ← LLM reranker
│       ├── generator.py      ← context assembly + generation
│       ├── simple_rag.py     ← simple fixed-chunk pipeline
│       └── adaptive_rag.py   ← LangGraph StateGraph pipeline
├── frontend/
│   └── app.py                ← Streamlit UI
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone and create virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 4. Configure API key
```bash
cp .env.example .env
# Edit .env and set your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 5. Run
```bash
python run.py
```

| Service   | URL |
|-----------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI   | http://localhost:8000 |
| API docs  | http://localhost:8000/docs |

## Usage

1. **Document tab** — upload a PDF / DOCX / TXT or paste text, click **Process Document**
2. Compare the two panels:
   - **Simple RAG** — flat numbered list of fixed 512-token chunks (no structure)
   - **Adaptive RAG** — semantic chunks nested under detected sections; toggle between Structural Tree view and Flat Chunk List
3. **Query tab** — type a question and click **Run Pipeline**
4. View: detected intent, pipeline trace, retrieval count, execution time (ms), response with citations, and per-chunk scores

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. Your OpenAI API key. |
