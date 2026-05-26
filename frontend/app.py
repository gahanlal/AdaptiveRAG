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
    /* ══ GLOBALS ══════════════════════════════════════════════════════════ */
    [data-testid="stAppViewContainer"] > .main { background: #f8fafc; }
    /* Force dark text everywhere so the light background is always readable */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] td,
    [data-testid="stAppViewContainer"] th,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
        color: #0f172a !important;
    }
    /* Keep sidebar text using its own colours (overridden below) */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * { color: inherit; }
    hr { border-color: #f1f5f9 !important; margin: 1.4rem 0 !important; }
    [data-testid="stAlert"] { border-radius: 10px !important; }

    /* ══ SIDEBAR ══════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small { color: #94a3b8 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] hr { border-color: #1e293b !important; }
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
        background: #334155 !important;
        color: #f1f5f9 !important;
    }

    /* ══ TABS ═════════════════════════════════════════════════════════════ */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 22px !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #f0f4ff !important;
        color: #1e3a8a !important;
        box-shadow: 0 1px 5px rgba(0,0,0,0.1) !important;
    }

    /* ══ METRICS ══════════════════════════════════════════════════════════ */
    div[data-testid="metric-container"] {
        background: #f0f4ff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #64748b !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* ══ BUTTONS ══════════════════════════════════════════════════════════ */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 10px rgba(37,99,235,0.3) !important;
        transition: box-shadow 0.15s, transform 0.15s !important;
    }
    [data-testid="baseButton-primary"]:hover {
        box-shadow: 0 4px 18px rgba(37,99,235,0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* ══ EXPANDERS ════════════════════════════════════════════════════════ */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background: #f8fafc !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stExpander"] summary {
        background: #f8fafc !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
    }
    [data-testid="stExpander"] summary:hover { background: #f1f5f9 !important; }

    /* ══ UPLOAD ZONE ══════════════════════════════════════════════════════ */
    [data-testid="stFileUploader"] > div {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 14px !important;
        background: #f8fafc !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: #2563eb !important;
        background: #eff6ff !important;
    }

    /* ══ TEXT INPUT ═══════════════════════════════════════════════════════ */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        border-radius: 10px !important;
        border: 1.5px solid #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 10px 14px !important;
        background: #f8fafc !important;
        transition: border-color 0.15s, box-shadow 0.15s !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
        background: #f1f5f9 !important;
    }

    /* ══ BORDERED CONTAINERS (st.container border=True) ══════════════════ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        background: #f8fafc !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
        padding: 4px 8px !important;
    }

    /* ══ PIPELINE BADGES ══════════════════════════════════════════════════ */
    .step-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1.5px solid #bfdbfe;
        border-radius: 20px;
        padding: 3px 11px;
        font-size: 0.73rem;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        margin: 2px 1px;
        letter-spacing: 0.01em;
        box-shadow: 0 1px 2px rgba(37,99,235,0.07);
    }
    .step-arrow { color: #94a3b8; font-size: 0.8rem; margin: 0 1px; vertical-align: middle; }

    /* ══ CHUNK CARDS ══════════════════════════════════════════════════════ */
    .chunk-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        background: #f8fafc;
        transition: border-color 0.15s, box-shadow 0.15s;
    }
    .chunk-card:hover {
        border-color: #bfdbfe;
        box-shadow: 0 2px 10px rgba(37,99,235,0.07);
    }
    .chunk-meta { font-size: 0.73rem; color: #64748b; margin-bottom: 6px; font-weight: 500; }

    /* ══ SECTION TREE ═════════════════════════════════════════════════════ */
    .tree-node { margin-left: 1.2rem; border-left: 2px solid #e2e8f0; padding-left: 0.8rem; }

    /* ══ STRATEGY TAGS ════════════════════════════════════════════════════ */
    .strategy-tag {
        display: inline-block;
        font-size: 0.67rem;
        border-radius: 20px;
        padding: 2px 9px;
        margin-left: 6px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .recursive_512            { background: #dbeafe; color: #1e40af; }
    .recursive_256_small_only { background: #ede9fe; color: #5b21b6; }
    .recursive_400_para       { background: #dcfce7; color: #166534; }
    .fallback_hard_split      { background: #fee2e2; color: #991b1b; }
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

    if st.button("Check API health", use_container_width=True):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            if r.ok:
                st.success(f"API online — v{r.json().get('version', '?')}")
            else:
                st.warning("Backend is still starting up.")
        except Exception:
            st.warning("Backend is still starting up.")

    st.markdown("---")

    # Live session status card
    if st.session_state.session_id:
        mode_color = "#3b82f6" if st.session_state.rag_type == "simple" else "#8b5cf6"
        mode_label = "⚡ Simple RAG" if st.session_state.rag_type == "simple" else "🧠 Adaptive RAG"
        fname = st.session_state.filename or "document"
        fname_short = fname[:26] + "…" if len(fname) > 26 else fname
        st.markdown(
            f'<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px 16px">'
            f'<div style="font-size:0.65rem;font-weight:700;color:#475569;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:8px">Active Session</div>'
            f'<div style="font-size:0.83rem;font-weight:600;color:#e2e8f0;margin-bottom:8px">'
            f'📄 {fname_short}</div>'
            f'<span style="background:{mode_color}25;color:{mode_color};font-size:0.72rem;'
            f'font-weight:700;padding:3px 10px;border-radius:20px;border:1px solid {mode_color}40">'
            f'{mode_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#1e293b;border:1px solid #1e293b;border-radius:12px;'
            'padding:14px 16px;text-align:center">'
            '<div style="font-size:1.5rem;margin-bottom:6px">📭</div>'
            '<div style="font-size:0.82rem;color:#64748b;font-weight:600">No document loaded</div>'
            '<div style="font-size:0.74rem;color:#475569;margin-top:4px">'
            'Upload a file in the Document tab</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("FastAPI :8000 · Streamlit :8501")


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


def _structure_sunburst(tree: list, doc_name: str, chunks: list | None = None) -> "object | None":
    """Return a plotly Sunburst chart of the document's section hierarchy.
    Falls back to a flat grouping by chunk section-title when no structural tree exists.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    ids:     list[str] = ["root"]
    labels:  list[str] = [doc_name[:40]]
    parents: list[str] = [""]
    values:  list[int] = [0]
    colors:  list[str] = ["#3b82f6"]
    level_colors = {1: "#6366f1", 2: "#8b5cf6", 3: "#06b6d4", 4: "#10b981"}

    def walk(nodes: list, parent_id: str) -> None:
        for n in nodes:
            nid = (parent_id + "/" + n["title"])[:80]
            ids.append(nid)
            labels.append(n["title"][:40])
            parents.append(parent_id)
            values.append(max(len(n.get("chunk_ids", [])), 1))
            colors.append(level_colors.get(n.get("level", 1), "#64748b"))
            if n.get("children"):
                walk(n["children"], nid)

    walk(tree, "root")

    # Fallback: tree was empty — build a flat chart grouped by chunk section title
    if len(ids) < 2 and chunks:
        sec_counts: dict[str, int] = {}
        for c in chunks:
            sec = (c.get("section_title") or "Untitled")[:40]
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
        ids     = ["root"] + [f"root/{s}" for s in sec_counts]
        labels  = [doc_name[:40]] + list(sec_counts.keys())
        parents = [""] + ["root"] * len(sec_counts)
        values  = [0] + list(sec_counts.values())
        colors  = ["#3b82f6"] + ["#6366f1"] * len(sec_counts)

    if len(ids) < 2:
        return None

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color="white", width=1)),
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Chunks: %{value}<extra></extra>",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _make_pipeline_figure(rag_type: str, steps: list) -> "object | None":
    """
    Return a plotly Figure showing the full pipeline graph.
    Nodes whose name appears in *steps* are highlighted in blue; others greyed out.

    Simple RAG  — horizontal left-to-right linear flow.
    Adaptive RAG — top-down branching graph with optional refinement loop.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    visited = " ".join(s.lower() for s in steps)

    def hit(*kws: str) -> bool:
        return any(k.lower() in visited for k in kws)

    # Colour palette
    A_FILL, A_LINE, A_TEXT = "#dbeafe", "#1d4ed8", "#1e3a8a"
    I_FILL, I_LINE, I_TEXT = "#f8fafc",  "#e2e8f0", "#94a3b8"
    E_COLOR = "#cbd5e1"
    W, H = 0.9, 0.32   # half-width / half-height of each box

    if rag_type == "simple":
        # (id, x, y, display-label, match-keywords)
        node_defs = [
            ("in",  0.0, 0, "📄 Input",              []),
            ("ch",  1.8, 0, "Fixed Chunker\n512 tok", ["fixedchunker"]),
            ("fa",  3.6, 0, "FAISS\nVector Index",    ["faiss"]),
            ("ge",  5.4, 0, "Generator",              ["generator"]),
            ("ou",  7.2, 0, "💬 Response",            ["generator"]),  # lit when generator ran
        ]
        edge_defs = [("in","ch",""), ("ch","fa",""), ("fa","ge",""), ("ge","ou","")]
        xrng, yrng, fig_h = [-0.7, 7.9], [-0.7, 0.7], 160
    else:
        node_defs = [
            ("qu",  3.0, 6.2, "Query\nUnderstanding",  ["queryunderstanding"]),
            ("sr",  0.6, 5.0, "Single\nRetriever",     ["singleretriever"]),
            ("mr",  3.0, 5.0, "Multi\nRetriever",      ["multiretriever"]),
            ("cr",  5.4, 5.0, "Custom\nRetriever",     ["customretriever"]),
            ("ce",  3.0, 3.8, "Context\nExpander",     ["contextexpansion"]),
            ("rk",  3.0, 2.6, "Reranker",              ["llmreranker"]),
            ("rf",  0.8, 1.4, "Retrieval\nRefiner",    ["retrievalrefinement"]),
            ("ca",  5.2, 1.4, "Context\nAssembler",    ["contextassembler"]),
            ("ge",  5.2, 0.2, "Generator",             ["generator"]),
            ("ou",  5.2,-1.0, "💬 Response",           ["generator"]),
        ]
        edge_defs = [
            ("qu","sr",""), ("qu","mr",""), ("qu","cr",""),
            ("sr","ce",""), ("mr","ce",""), ("cr","ce",""),
            ("ce","rk",""),
            ("rk","rf","low score"), ("rk","ca","confident"),
            ("rf","rk","retry ↑"),
            ("ca","ge",""), ("ge","ou",""),
        ]
        xrng, yrng, fig_h = [-0.3, 7.1], [-1.6, 7.1], 540

    pos = {nid: (x, y) for nid, x, y, *_ in node_defs}
    fig = go.Figure()

    # ── Arrow-head edges ─────────────────────────────────────────────────
    for src_id, dst_id, label in edge_defs:
        sx, sy = pos[src_id]
        dx, dy = pos[dst_id]
        going_up = dy > sy
        ay_start = sy + H if going_up else sy - H
        ay_end   = dy - H if going_up else dy + H
        fig.add_annotation(
            x=dx, y=ay_end,
            ax=sx, ay=ay_start,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=0.9, arrowwidth=1.5,
            arrowcolor=E_COLOR,
            showarrow=True,
            text=label,
            font=dict(size=8, color="#94a3b8"),
        )

    # ── Node rectangles + labels ─────────────────────────────────────────
    for nid, x, y, label, kws in node_defs:
        active = hit(*kws)
        fig.add_shape(
            type="rect",
            x0=x-W, y0=y-H, x1=x+W, y1=y+H,
            fillcolor=A_FILL if active else I_FILL,
            line=dict(color=A_LINE if active else I_LINE, width=2.5 if active else 1),
        )
        fig.add_annotation(
            x=x, y=y,
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(
                size=9.5,
                color=A_TEXT if active else I_TEXT,
                family="'Courier New', monospace",
            ),
            xanchor="center", yanchor="middle",
        )

    fig.update_layout(
        xaxis=dict(range=xrng, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=yrng, showgrid=False, zeroline=False, showticklabels=False),
        height=fig_h,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


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

    with st.expander("ℹ️ Simple RAG vs Adaptive RAG — differences & how to test", expanded=False):
        st.markdown("""
#### At a glance

| | ⚡ Simple RAG | 🧠 Adaptive RAG |
|---|---|---|
| **Chunking** | Fixed 512-token windows, 50-token overlap | Structure-aware — sections detected via heading hierarchy |
| **Strategies** | One fixed strategy | Multiple strategies compete per section; best score wins |
| **Retrieval** | Single FAISS vector search | Up to 4 parallel indexes (vector · structural · metadata · graph) |
| **Reranking** | None | LLM reranker removes low-quality chunks |
| **Context window** | Raw chunk text only | Parent + neighbour chunks expanded for richer context |
| **Routing** | Always vector | LLM detects query intent → picks the best index automatically |
| **Refinement** | None | Low-confidence retrieval loops back for a targeted second pass |

#### Why Adaptive RAG gives better answers

- Chunks follow **natural section boundaries**, so answers never split mid-thought
- Multiple indexes mean structural questions (*"what does section 3 say?"*) and conceptual
  questions (*"explain the methodology"*) both get the right chunks
- LLM reranking removes noise — only high-signal passages reach the generator
- The refinement loop catches cases where the first retrieval pass was too shallow

#### How to test the difference

1. **Upload a structured document** — a research paper, report, or manual with clear headings works best
2. After processing, inspect the **Document Structure Map** that appears immediately — Adaptive
   detects named sections; Simple has no concept of structure
3. Compare **chunk counts** in the side-by-side view — Adaptive typically produces *fewer but more
   complete* chunks that stay inside one section
4. Go to the **Query** tab and try these question types:
   - *Broad/conceptual* — **"Summarise the main contributions"** — both should respond; compare depth
   - *Section-specific* — **"What does the introduction say?"** — Adaptive retrieves the correct
     section; Simple may mix in unrelated content
   - *Cross-section* — **"Compare the approach in section 2 and section 5"** — Adaptive's multi-index
     finds both sections; Simple returns scattered results
5. After each query, expand the **Pipeline Flow diagram** to see which nodes were activated —
   Adaptive shows context expansion, reranking, and the chosen retriever
6. Toggle **LLM intent-routed** on the main page, repeat the same query, and watch whether a
   different retriever is chosen
        """)

    col_up, col_paste = st.columns([1, 1])

    with col_up:
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT — multiple files allowed",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
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
        if not uploaded_files and not pasted_text.strip():
            st.warning("Please upload a file or paste some text first.")
        else:
            with st.spinner("Processing… (structural parsing + chunking + indexing)"):
                try:
                    if uploaded_files:
                        request_files = [
                            ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                            for f in uploaded_files
                        ]
                        data = {}
                    else:
                        # Empty placeholder so FastAPI receives multipart; falls through to raw_text
                        request_files = [("files", ("", b"", "text/plain"))]
                        data = {"raw_text": pasted_text, "filename": "pasted_text.txt"}

                    resp = requests.post(
                        f"{API_BASE}/ingest",
                        files=request_files,
                        data=data,
                        timeout=120,
                    )
                    if resp.ok:
                        result = resp.json()
                        st.session_state.ingest_result = result
                        st.session_state.session_id = result["session_id"]
                        st.session_state.filename = result["filename"]
                        st.session_state.query_result = None  # reset previous query
                        n_files = len(uploaded_files)
                        label = (
                            f"**{result['filename']}**"
                            if n_files <= 1
                            else f"**{n_files} files** ({result['filename']})"
                        )
                        st.success(
                            f"Processed {label} — "
                            f"{result['word_count']:,} words · "
                            f"{result['char_count']:,} chars"
                        )
                    else:
                        st.error("Processing failed. Please try again or use a different document.")
                except Exception:
                    st.error("Could not reach the backend. The service may still be starting — please wait a few seconds and try again.")

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

        # ---- Document Structure Map (top of results) ----
        fig_sun = _structure_sunburst(structure_tree, result["filename"], adaptive_chunks)
        if fig_sun is not None:
            st.markdown("")
            st.subheader("📊 Document Structure Map")
            st.caption(
                "Sunburst of the detected section hierarchy. "
                "Each sector = one section; size = adaptive chunk count. Hover for details."
            )
            st.plotly_chart(fig_sun, use_container_width=True, key="structure_map")

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

        # ---- Document Structure Map (plotly sunburst) ----
        fig_sun = _structure_sunburst(structure_tree, result["filename"])
        if fig_sun is not None:
            st.markdown("---")
            st.subheader("📊 Document Structure Map")
            st.caption(
                "Each sector represents a section; size = number of adaptive chunks. "
                "Hover for details."
            )
            st.plotly_chart(fig_sun, use_container_width=True)


# ============================================================
# TAB 2 — Query + Results
# ============================================================
with tab_query:
    st.header("Ask a Question")

    if not st.session_state.session_id:
        st.markdown(
            '<div style="text-align:center;padding:56px 24px">'
            '<div style="font-size:3rem;margin-bottom:14px">📭</div>'
            '<div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:8px">'
            'No document loaded yet</div>'
            '<div style="color:#94a3b8;font-size:0.9rem;max-width:360px;margin:0 auto">'
            'Head to the <strong>📄 Document</strong> tab, upload a file or paste text, '
            'then return here to query it.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Session info pill
        mode_icon = "⚡" if st.session_state.rag_type == "simple" else "🧠"
        mode_name = "Simple RAG" if st.session_state.rag_type == "simple" else "Adaptive RAG"
        mode_bg   = "#eff6ff" if st.session_state.rag_type == "simple" else "#f5f3ff"
        mode_cl   = "#1d4ed8" if st.session_state.rag_type == "simple" else "#6d28d9"
        retrieval_span = (
            f'<span style="margin-left:auto;color:#94a3b8;font-size:0.78rem">'
            f'Retrieval: <code style="background:#334155;color:#e2e8f0;padding:1px 6px;border-radius:4px">'
            f'{st.session_state.retrieval_mode}</code></span>'
            if st.session_state.rag_type == "adaptive" else ""
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:11px 18px;'
            f'background:#1e293b;border:1px solid #334155;border-radius:12px;margin-bottom:18px;">'
            f'<span style="background:{mode_cl}30;color:{mode_cl};font-size:0.76rem;font-weight:700;'
            f'padding:3px 12px;border-radius:20px;border:1px solid {mode_cl}50;white-space:nowrap">'
            f'{mode_icon} {mode_name}</span>'
            f'<span style="color:#e2e8f0;font-size:0.85rem;font-weight:500">'
            f'📄 {st.session_state.filename}</span>'
            f'{retrieval_span}'
            f'</div>',
            unsafe_allow_html=True,
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
                        st.error("Query failed. Please try rephrasing your question or re-uploading the document.")
                except Exception:
                    st.error("Could not reach the backend. The service may still be starting — please wait a few seconds and try again.")

        # --- Show results ---
        if st.session_state.query_result:
            qr = st.session_state.query_result

            st.markdown("")
            st.markdown(
                '<div style="display:flex;align-items:center;gap:12px;margin:8px 0">'
                '<div style="height:1px;flex:1;background:#e2e8f0"></div>'
                '<span style="font-size:0.72rem;font-weight:700;color:#94a3b8;'
                'text-transform:uppercase;letter-spacing:0.06em">Results</span>'
                '<div style="height:1px;flex:1;background:#e2e8f0"></div></div>',
                unsafe_allow_html=True,
            )

            # Metrics row
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("⏱️ Execution time", f"{qr['execution_time_ms']:.0f} ms")
            mc2.metric("📦 Chunks retrieved", qr["retrieval_count"])
            mc3.metric("📌 Chunks reranked", len(qr["retrieved_docs"]))

            steps = qr.get("path_taken", [])
            fig_flow = _make_pipeline_figure(st.session_state.rag_type, steps)

            # ── Adaptive RAG: flow diagram side-by-side with the response ──────
            if st.session_state.rag_type == "adaptive" and fig_flow is not None:
                col_flow, col_resp = st.columns([5, 6], gap="large")

                with col_flow:
                    st.markdown("#### 🔄 Pipeline Flow")
                    # Intent badge
                    if qr.get("intent"):
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
                            f'border-radius:6px;font-size:0.82rem">{qr["intent"]}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")
                    _render_pipeline_trace(steps)
                    st.markdown("")
                    st.plotly_chart(fig_flow, use_container_width=True, key="pipeline_flow")

                with col_resp:
                    st.markdown("#### 💬 Response")
                    with st.container(border=True):
                        st.markdown(qr["response"])
                    if qr.get("citations"):
                        st.markdown("")
                        st.markdown(
                            '<span style="font-size:0.8rem;font-weight:700;color:#64748b;'
                            'text-transform:uppercase;letter-spacing:0.05em">Citations</span>',
                            unsafe_allow_html=True,
                        )
                        for cite in qr["citations"]:
                            st.markdown(
                                f'<code style="background:#f1f5f9;color:#374151;padding:2px 8px;'
                                f'border-radius:5px;font-size:0.8rem">{cite}</code>',
                                unsafe_allow_html=True,
                            )

            # ── Simple RAG (or adaptive with no figure): linear layout ──────────
            else:
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

                _render_pipeline_trace(steps)
                st.markdown("")

                if fig_flow is not None:
                    with st.expander("🔄 Pipeline Flow  *(blue = executed nodes)*", expanded=True):
                        st.plotly_chart(fig_flow, use_container_width=True, key="pipeline_flow")
                st.markdown("")

                st.subheader("Response")
                with st.container(border=True):
                    st.markdown(qr["response"])

                if qr.get("citations"):
                    st.markdown(
                        '<span style="font-size:0.8rem;font-weight:700;color:#64748b;'
                        'text-transform:uppercase;letter-spacing:0.05em">Citations</span>',
                        unsafe_allow_html=True,
                    )
                    for cite in qr["citations"]:
                        st.markdown(
                            f'<code style="background:#f1f5f9;color:#374151;padding:2px 8px;'
                            f'border-radius:5px;font-size:0.8rem">{cite}</code>',
                            unsafe_allow_html=True,
                        )

            # Retrieved chunks detail
            with st.expander(
                f"🗂️ Retrieved chunks · {len(qr['retrieved_docs'])} shown after reranking",
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
