"""
Adaptive RAG Configurator — Streamlit frontend

Layout:
  Sidebar  : RAG mode selector + retrieval mode
  Tab 1    : Document upload / paste → Process → side-by-side chunk comparison
  Tab 2    : Query → results (response, citations, pipeline trace, latency)
"""
from __future__ import annotations

import json
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive RAG Configurator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean card-like expanders */
    .streamlit-expanderHeader { font-weight: 600; }

    /* Section tree indentation */
    .tree-node { margin-left: 1.2rem; border-left: 2px solid #e0e0e0; padding-left: 0.8rem; }

    /* Pipeline step badges */
    .step-badge {
        display: inline-block;
        background: #f0f4ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-family: monospace;
        margin: 2px 3px;
    }
    .step-arrow { color: #94a3b8; font-size: 1rem; margin: 0 2px; }

    /* Metric card */
    div[data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
    }

    /* Chunk cards */
    .chunk-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background: #fafafa;
    }
    .chunk-meta { font-size: 0.75rem; color: #64748b; margin-bottom: 4px; }
    .strategy-tag {
        display: inline-block;
        font-size: 0.7rem;
        border-radius: 4px;
        padding: 1px 7px;
        margin-left: 6px;
        font-weight: 600;
    }
    .recursive_512              { background:#dbeafe; color:#1e40af; }
    .recursive_256_small_only   { background:#ede9fe; color:#5b21b6; }
    .recursive_400_para         { background:#dcfce7; color:#166534; }
    .fallback_hard_split        { background:#fee2e2; color:#991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "session_id": None,
    "filename": "",
    "ingest_result": None,
    "query_result": None,
    "rag_type": "simple",
    "retrieval_mode": "single",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar — branding + health check only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 RAG Configurator")
    st.markdown("---")
    st.caption("FastAPI backend on :8000 · Streamlit on :8501")

    if st.button("Check API health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            if r.ok:
                st.success(f"API online — v{r.json().get('version', '?')}")
            else:
                st.error(f"API returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")


# ---------------------------------------------------------------------------
# Inline config — Pipeline mode + Retrieval indexes (on the main page)
# ---------------------------------------------------------------------------
_INDEX_OPTIONS = {
    "vector":     "🔷 Vector — Semantic similarity",
    "structural": "📂 Structural — Section-title match",
    "metadata":   "🏷️ Metadata — Hierarchy level (H1/H2/H3)",
    "graph":      "🕸️ Graph — Entity co-occurrence",
}

cfg_left, cfg_right = st.columns([1, 2], gap="large")

with cfg_left:
    st.markdown("##### Pipeline Mode")
    rag_type = st.radio(
        "rag_type",
        options=["simple", "adaptive"],
        format_func=lambda x: "⚡ Simple RAG" if x == "simple" else "🧠 Adaptive RAG",
        index=0 if st.session_state.rag_type == "simple" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.rag_type = rag_type
    if rag_type == "simple":
        st.caption("Fixed 512-token chunks → FAISS → Generate")
    else:
        st.caption("Structure → Chunk → Multi-Index → Route → Rerank → Generate")

with cfg_right:
    if rag_type == "adaptive":
        st.markdown("##### Retrieval Indexes")
        idx_col, toggle_col = st.columns([3, 2], gap="medium")

        with idx_col:
            _prev = st.session_state.get("_selected_indexes", ["vector", "structural"])
            selected_indexes = st.multiselect(
                "indexes",
                options=list(_INDEX_OPTIONS.keys()),
                default=[k for k in _prev if k in _INDEX_OPTIONS],
                format_func=lambda k: _INDEX_OPTIONS[k],
                label_visibility="collapsed",
            )
            st.session_state["_selected_indexes"] = selected_indexes

        with toggle_col:
            st.markdown("<div style='padding-top:6px'></div>", unsafe_allow_html=True)
            intent_routed = st.toggle(
                "🎯 LLM intent-routed",
                value=st.session_state.get("_intent_routed", False),
                help="LLM detects the query intent and picks the best single index automatically.",
            )
            st.session_state["_intent_routed"] = intent_routed

        if intent_routed:
            retrieval_mode = "single"
            st.caption("LLM will route to: Vector · Structural · Metadata · or Graph")
        elif not selected_indexes:
            retrieval_mode = "vector"
            st.warning("No indexes selected — defaulting to Vector.", icon="⚠️")
        elif set(selected_indexes) == set(_INDEX_OPTIONS.keys()):
            retrieval_mode = "multi"
        else:
            retrieval_mode = ",".join(selected_indexes)

        st.session_state.retrieval_mode = retrieval_mode

        # Guide: when to use each index
        with st.expander("💡 When to use each index", expanded=False):
            st.markdown(
                """
| Index | Best for | Example questions |
|---|---|---|
| 🔷 **Vector** | General factual & conceptual questions — finds semantically similar passages | *"What is the main argument?"* · *"Explain the methodology"* |
| 📂 **Structural** | Questions about a specific section, chapter, or topic by name | *"What does section 3 say?"* · *"Summarise the introduction"* |
| 🏷️ **Metadata** | High-level document overview, authorship, titles, dates, hierarchy | *"What is this document about?"* · *"Who wrote this?"* |
| 🕸️ **Graph** | Relationships, comparisons, or connections between people / concepts / entities | *"How are X and Y related?"* · *"What entities appear together?"* |
| 🎯 **LLM intent-routed** | When unsure — the LLM reads your query and picks the best single index | Any query |

**Combining indexes** (e.g. Vector + Structural) runs both in parallel, merges results, and reranks — useful when your question spans topics and structure.
                """,
                unsafe_allow_html=False,
            )
    else:
        st.session_state.retrieval_mode = "single"

st.markdown("---")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strategy_tag(strategy: str) -> str:
    label_map = {
        "recursive_512": "recursive-512",
        "recursive_256_small_only": "recursive-256-small",
        "recursive_400_para": "recursive-400-para",
        "fallback_hard_split": "hard-split",
    }
    label = label_map.get(strategy, strategy)
    css_class = strategy.replace("-", "_")
    return f'<span class="strategy-tag {css_class}">{label}</span>'


def _boundary_tag(reason: str) -> str:
    if not reason:
        return ""
    return (
        f'<span style="display:inline-block;font-size:0.7rem;background:#f0fdf4;'
        f'color:#166534;border:1px solid #bbf7d0;border-radius:4px;'
        f'padding:1px 7px;margin-left:6px">{reason}</span>'
    )


def _render_pipeline_trace(steps: list[str]) -> None:
    if not steps:
        return
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="step-badge">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="step-arrow">→</span>')
    st.markdown("**Pipeline trace:**<br>" + " ".join(parts), unsafe_allow_html=True)


def _render_structure_tree(nodes, chunks_by_id: dict, depth: int = 0) -> None:
    """Recursively render the structural tree with chunks nested inside."""
    for node in nodes:
        indent = "　" * depth
        icon = "📂" if node.get("children") else "📄"
        level_label = f"L{node['level']}"
        chunk_count = len(node.get("chunk_ids", []))

        with st.expander(
            f"{indent}{icon} **{node['title']}**  `{level_label}`  · {chunk_count} chunk(s)",
            expanded=(depth == 0),
        ):
            # Show chunks belonging to this section
            for cid in node.get("chunk_ids", []):
                c = chunks_by_id.get(cid)
                if c:
                    reason_html = _boundary_tag(c.get("boundary_reason", ""))
                    ents = ", ".join(c.get("entities", [])[:5])
                    st.markdown(
                        f"""<div class="chunk-card">
                        <div class="chunk-meta">
                            🆔 {cid} &nbsp;|&nbsp; 🪙 {c['token_count']} tokens
                            {reason_html}
                            {"&nbsp;|&nbsp; 🏷️ " + ents if ents else ""}
                        </div>
                        <div style="font-size:0.87rem">{c['text'][:400]}{"…" if len(c['text']) > 400 else ""}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # Recurse
            if node.get("children"):
                _render_structure_tree(node["children"], chunks_by_id, depth + 1)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_doc, tab_query = st.tabs(["📄 Document", "💬 Query"])


# ============================================================
# TAB 1 — Document upload + side-by-side comparison
# ============================================================
with tab_doc:
    st.header("Upload or Paste a Document")
    st.caption(
        "Both pipelines process your document. "
        "Compare fixed chunking (Simple) vs. structure-aware chunking (Adaptive) side by side."
    )

    col_up, col_paste = st.columns([1, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )

    with col_paste:
        pasted_text = st.text_area(
            "…or paste raw text here",
            height=140,
            placeholder="Paste document text here (optional, used if no file uploaded)…",
        )

    process_btn = st.button("⚙️ Process Document", type="primary", width="stretch")

    if process_btn:
        if uploaded_file is None and not pasted_text.strip():
            st.warning("Please upload a file or paste some text first.")
        else:
            with st.spinner("Processing… (structural parsing + chunking + indexing)"):
                try:
                    if uploaded_file is not None:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
                        data = {}
                    else:
                        # Empty placeholder forces multipart — FastAPI sees file.filename=="" → falls through to raw_text
                        files = {"file": ("", b"", "text/plain")}
                        data = {"raw_text": pasted_text, "filename": "pasted_text.txt"}

                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        files=files,
                        data=data,
                        timeout=120,
                    )
                    if resp.ok:
                        result = resp.json()
                        st.session_state.ingest_result = result
                        st.session_state.session_id = result["session_id"]
                        st.session_state.filename = result["filename"]
                        st.session_state.query_result = None  # reset previous query
                        st.success(
                            f"Processed **{result['filename']}** — "
                            f"{result['word_count']:,} words · "
                            f"{result['char_count']:,} chars"
                        )
                    else:
                        st.error(f"Ingest failed ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    # --- Display side-by-side comparison ---
    if st.session_state.ingest_result:
        result = st.session_state.ingest_result
        simple_chunks = result["simple_chunks"]
        adaptive_chunks = result["adaptive_chunks"]
        structure_tree = result["structure_tree"]

        # Adaptive chunks dict for tree lookup
        ada_chunks_by_id = {c["chunk_id"]: c for c in adaptive_chunks}

        st.markdown("---")
        st.subheader("Document Processing Comparison")

        # Summary row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simple chunks", len(simple_chunks))
        m2.metric("Adaptive chunks", len(adaptive_chunks))
        m3.metric("Detected sections", len(structure_tree))
        m4.metric("Total words", f"{result['word_count']:,}")

        st.markdown("")

        # Two-column layout
        col_simple, col_adaptive = st.columns(2, gap="large")

        # ---- Simple RAG Column ----
        with col_simple:
            st.markdown(
                "### ⚡ Simple RAG  \n"
                "Fixed 512-token chunks, 50-token overlap.  \n"
                "*No structure awareness.*"
            )
            st.caption(f"{len(simple_chunks)} chunks total")

            for chunk in simple_chunks:
                with st.expander(
                    f"Chunk #{chunk['index'] + 1}  ·  🪙 {chunk['token_count']} tokens",
                    expanded=False,
                ):
                    st.markdown(
                        f"<div class='chunk-meta'>ID: {chunk['chunk_id']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))

        # ---- Adaptive RAG Column ----
        with col_adaptive:
            st.markdown(
                "### 🧠 Adaptive RAG  \n"
                "RecursiveSplitter — multiple strategies compete, best wins per segment.  \n"
                "*Structure-aware · Strategy-selected · Entity-tagged.*"
            )

            # Strategy selection summary
            strategy_scores = result.get("strategy_scores", {})
            winner_counts = strategy_scores.get("winner_counts", {})
            if winner_counts:
                st.markdown("**Strategy competition results:**")
                cols = st.columns(len(winner_counts))
                for i, (strat, count) in enumerate(sorted(winner_counts.items(), key=lambda x: -x[1])):
                    short = strat.replace("recursive_", "r").replace("_small_only", "-sm")
                    cols[i].metric(f"`{short}`", f"won {count}×", help=f"Strategy '{strat}' won for {count} segment(s)")

                # Per-segment detail table
                seg_scores = strategy_scores.get("segment_scores", {})
                if seg_scores:
                    with st.expander("Per-segment strategy scores", expanded=False):
                        for seg_title, strat_map in seg_scores.items():
                            st.markdown(f"**{seg_title}**")
                            rows = []
                            for sname, metrics in strat_map.items():
                                rows.append({
                                    "Strategy": sname,
                                    "SC": round(metrics.get("sc", 0), 3),
                                    "ICC": round(metrics.get("icc", 0), 3),
                                    "DCC": round(metrics.get("dcc", 0), 3),
                                    "Combined": round(metrics.get("combined", 0), 3),
                                    "Chunks": metrics.get("n_chunks", 0),
                                })
                            rows.sort(key=lambda r: -r["Combined"])
                            st.dataframe(rows, width="stretch", hide_index=True)

            view_toggle = st.radio(
                "View as",
                options=["Structural Tree", "Flat Chunk List"],
                horizontal=True,
                key="ada_view_toggle",
            )

            if view_toggle == "Flat Chunk List":
                st.caption(f"{len(adaptive_chunks)} chunks total")
                for chunk in adaptive_chunks:
                    strategy_html = _strategy_tag(chunk.get("boundary_reason", ""))
                    ents = ", ".join(chunk.get("entities", [])[:5])
                    with st.expander(
                        f"Chunk #{chunk['index'] + 1}  ·  [{chunk['section_title']}]  ·  🪙 {chunk['token_count']} tokens",
                        expanded=False,
                    ):
                        st.markdown(
                            f"<div class='chunk-meta'>"
                            f"ID: {chunk['chunk_id']} &nbsp;|&nbsp; "
                            f"Level: {chunk['hierarchy_level']} &nbsp;|&nbsp; "
                            f"Parent: {chunk['parent_section'] or '—'} "
                            f"{strategy_html}"
                            f"{'&nbsp;|&nbsp; Entities: ' + ents if ents else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
            else:
                # Structural tree view
                st.caption(
                    f"{len(adaptive_chunks)} chunks across "
                    f"{len(structure_tree)} top-level sections"
                )
                _render_structure_tree(structure_tree, ada_chunks_by_id, depth=0)


# ============================================================
# TAB 2 — Query + Results
# ============================================================
with tab_query:
    st.header("Ask a Question")

    if not st.session_state.session_id:
        st.info("Process a document in the **Document** tab first.")
    else:
        st.caption(
            f"Document: **{st.session_state.filename}** · "
            f"Mode: **{st.session_state.rag_type.upper()}**"
            + (
                f" · Retrieval: **{st.session_state.retrieval_mode}**"
                if st.session_state.rag_type == "adaptive"
                else ""
            )
        )

        query_text = st.text_input(
            "Your question",
            placeholder="e.g. What are the main findings in section 3?",
        )

        run_btn = st.button("🚀 Run Pipeline", type="primary", disabled=not query_text.strip())

        if run_btn and query_text.strip():
            with st.spinner("Running pipeline…"):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "query": query_text.strip(),
                        "rag_type": st.session_state.rag_type,
                        "retrieval_mode": st.session_state.retrieval_mode,
                    }
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json=payload,
                        timeout=120,
                    )
                    if resp.ok:
                        st.session_state.query_result = resp.json()
                    else:
                        st.error(f"Query failed ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

        # --- Show results ---
        if st.session_state.query_result:
            qr = st.session_state.query_result

            st.markdown("---")

            # Metrics row
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("⏱️ Execution time", f"{qr['execution_time_ms']:.0f} ms")
            mc2.metric("📦 Chunks retrieved", qr["retrieval_count"])
            mc3.metric("📌 Chunks reranked", len(qr["retrieved_docs"]))

            # Intent (adaptive only)
            if st.session_state.rag_type == "adaptive" and qr.get("intent"):
                intent_colors = {
                    "semantic": "#3b82f6",
                    "structural": "#8b5cf6",
                    "metadata": "#f59e0b",
                    "graph": "#10b981",
                }
                color = intent_colors.get(qr["intent"], "#64748b")
                st.markdown(
                    f"**Detected intent:** "
                    f'<span style="background:{color};color:white;padding:2px 10px;'
                    f'border-radius:6px;font-size:0.85rem">{qr["intent"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

            # Pipeline trace
            _render_pipeline_trace(qr.get("path_taken", []))
            st.markdown("")

            # Response
            st.subheader("Response")
            st.markdown(qr["response"])

            # Citations
            if qr.get("citations"):
                st.markdown("**Citations used:**")
                for cite in qr["citations"]:
                    st.markdown(f"- `{cite}`")

            # Retrieved chunks detail
            with st.expander(
                f"Retrieved chunks ({len(qr['retrieved_docs'])} shown after reranking)",
                expanded=False,
            ):
                for i, doc in enumerate(qr["retrieved_docs"]):
                    badge_color = {
                        "vector": "#3b82f6",
                        "structural": "#8b5cf6",
                        "metadata": "#f59e0b",
                        "graph": "#10b981",
                    }.get(doc["source_index"], "#64748b")

                    st.markdown(
                        f"<div class='chunk-card'>"
                        f"<div class='chunk-meta'>"
                        f"#{i+1} &nbsp;|&nbsp; "
                        f"<strong>{doc['section_title']}</strong> &nbsp;|&nbsp; "
                        f"Score: <strong>{doc['score']:.3f}</strong> &nbsp;|&nbsp; "
                        f"<span style='background:{badge_color};color:white;"
                        f"border-radius:4px;padding:1px 7px;font-size:0.72rem'>"
                        f"{doc['source_index']}</span>"
                        f"</div>"
                        f"<div style='font-size:0.87rem'>"
                        f"{doc['text'][:400]}{'…' if len(doc['text']) > 400 else ''}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
