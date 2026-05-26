# Adaptive RAG Configurator

A FastAPI + Streamlit app that lets you upload a document and compare **Simple RAG** vs **Adaptive RAG** side by side — then ask queries and watch the full pipeline trace.

Live demo → [adaptiverag.streamlit.app](https://adaptiverag.streamlit.app) *(Streamlit Cloud)*

---

## What makes it truly adaptive

| Feature | Description |
|---|---|
| **Structural parsing** | Detects headings, sections, and hierarchy before any chunking |
| **Density-aware chunking** | Measures semantic variance per segment; boosts smaller chunks for diverse text, larger for coherent text |
| **Parent-child retrieval** | Every chunk knows its neighbours; retrieved context is expanded left/right within a token budget |
| **Recursive refinement** | If top reranked score < 0.3, the query is enriched with extracted entities and retrieval is retried once |
| **Multi-index** | Vector · Structural · Metadata · Graph — routed by intent or run in parallel |
| **LLM reranker** | Scores every candidate 0–1 before assembly |
| **Groq-first LLM** | Uses `llama-3.1-8b-instant` (free, fast); falls back to OpenAI automatically on any error |

---

## Pipeline

```
Document
  → StructuralParser          (headings, sections, hierarchy)
  → DensityAwareChunker       (semantic variance → chunk size bias)
      └─ chunk_neighbors map  (prev/next chunk IDs per section)
  → MultiIndexBuilder
      ├── Vector Index        (FAISS + all-MiniLM-L6-v2, free local)
      ├── Structural Index    (section hierarchy)
      ├── Metadata Index      (level / author / date filtering)
      └── Graph Index         (NetworkX entity co-occurrence)

Query
  → QueryUnderstanding        (intent + entity extraction)
  → [single | multi | custom] Retriever
  → ContextExpander           (neighbour expansion, 600-token budget)
  → Reranker                  (LLM 0–1 scoring, top-5)
  → [_should_refine?]
      └─ RetrievalRefiner     (entity-enriched re-query, max 1×)
  → ContextAssembler          (3 000-token budget, citation labels)
  → Generator                 (grounded response with inline citations)
```

---

## Project Structure

```
AdaptiveRAG/
├── run.py                    ← local launcher + Streamlit Cloud entry point
├── backend/
│   ├── main.py               ← FastAPI routes (/ingest, /query, /health)
│   ├── models.py             ← Pydantic models
│   └── rag/
│       ├── llm_client.py     ← multi-provider LLM factory (Groq / OpenAI / Ollama / custom)
│       ├── parser.py         ← structural parser
│       ├── chunker.py        ← density-aware adaptive chunker
│       ├── indexer.py        ← multi-index builder + chunk_neighbors map
│       ├── retriever.py      ← single / multi / custom retrieval + neighbour expansion
│       ├── reranker.py       ← LLM reranker
│       ├── generator.py      ← context assembly + grounded generation
│       ├── simple_rag.py     ← simple fixed-chunk pipeline (for comparison)
│       └── adaptive_rag.py   ← LangGraph StateGraph orchestration
├── frontend/
│   └── app.py                ← Streamlit UI
├── .streamlit/
│   ├── config.toml           ← disables file watcher (needed on Cloud)
│   └── secrets.toml.example  ← secrets template (never commit secrets.toml)
├── requirements.txt
└── tests/
```

---

## Local Setup

### 1. Clone & create virtualenv
```bash
git clone https://github.com/gahanlal/AdaptiveRAG.git
cd AdaptiveRAG
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure secrets
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and fill in your keys
```

**Minimum required** (Groq is free — get a key at [console.groq.com](https://console.groq.com)):
```toml
LLM_PROVIDER = "groq-fallback"
GROQ_API_KEY  = "gsk_..."
GROQ_MODEL    = "llama-3.1-8b-instant"
OPENAI_API_KEY = "sk-..."   # fallback — optional if you only want Groq
```

### 4. Run
```bash
python run.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

---

## Streamlit Cloud Deployment

1. Fork / push repo to GitHub
2. Create a new app on [share.streamlit.io](https://share.streamlit.io)
3. Set **Main file path** → `run.py`
4. In **App Settings → Secrets**, paste:
```toml
LLM_PROVIDER  = "groq-fallback"
GROQ_API_KEY  = "gsk_..."
GROQ_MODEL    = "llama-3.1-8b-instant"
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL  = "gpt-4o-mini"
```
`run.py` auto-detects the cloud environment, starts FastAPI as a background thread, and renders the Streamlit UI — no extra config needed.

---

## LLM Providers

| Provider | `LLM_PROVIDER` value | Notes |
|---|---|---|
| **Groq** (default) | `groq` | Free tier, very fast |
| **Groq → OpenAI fallback** | `groq-fallback` | Tries Groq first, falls back silently |
| **OpenAI** | `openai` | Requires `OPENAI_API_KEY` |
| **Ollama** (local) | `ollama` | Requires local Ollama server |
| **Custom endpoint** | `custom` | Any OpenAI-compatible API |

---

## Usage

1. **Document tab** — upload a PDF / DOCX / TXT or paste text, click **Process Document**
2. Compare the two panels:
   - **Simple RAG** — flat fixed 512-token chunks (no structure awareness)
   - **Adaptive RAG** — semantic chunks nested under detected sections, with density-aware sizing
3. **Query tab** — type a question, choose pipeline mode (single / multi / custom indexes), click **Run Pipeline**
4. View: detected intent · pipeline trace · retrieval count · execution time · response with inline citations · per-chunk scores

