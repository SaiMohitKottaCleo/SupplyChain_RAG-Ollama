"""
Cleo Supply Chain Intelligence — Streamlit UI
Run: streamlit run app/main.py
"""
import tempfile
import os
from pathlib import Path

import streamlit as st

from app.generation.generator import AnswerGenerator
from app.ingestion.pipeline import IngestionPipeline
from app.database.chroma import ChromaClient
from app.config import COLLECTIONS

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Cleo Supply Chain Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached singletons — load once, reuse on every rerun ──────────────────────
# @st.cache_resource creates a singleton that survives across Streamlit reruns.
# Without this, AnswerGenerator() would reload the sentence-transformer model
# (~400MB) on every user message — catastrophic latency.

@st.cache_resource
def get_generator() -> AnswerGenerator:
    return AnswerGenerator()


@st.cache_resource
def get_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


# ── Session state init ────────────────────────────────────────────────────────
# chat_history: list of {"role": "user"|"assistant", "content": str, "meta": dict}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── Helper: confidence badge ──────────────────────────────────────────────────
def confidence_badge(score: float, confident: bool) -> str:
    if not confident:
        return "🔴 No match"
    if score >= 0.7:
        return f"🟢 {score:.0%}"
    if score >= 0.5:
        return f"🟡 {score:.0%}"
    return f"🟠 {score:.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ CSCI Settings")
    st.divider()

    # ── Ollama status ─────────────────────────────────────────────────────────
    st.subheader("Model Status")
    gen = get_generator()
    if gen.llm.is_available():
        st.success("phi3:mini  ✓  Online", icon="🟢")
    else:
        st.error("Ollama offline — run `ollama serve`", icon="🔴")

    st.divider()

    # ── Document upload ───────────────────────────────────────────────────────
    st.subheader("📂 Upload Document")

    collection_choice = st.selectbox(
        "Target collection",
        options=list(COLLECTIONS.keys()),
        index=list(COLLECTIONS.keys()).index("uploaded_documents"),
        help="Which knowledge base should this document be added to?",
    )

    uploaded_file = st.file_uploader(
        "Drop a file here",
        type=["pdf", "docx", "txt", "md", "xlsx", "xls", "csv"],
        help="PDF, DOCX, TXT, Markdown, Excel, or CSV",
    )

    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True, type="primary"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                # Save bytes to a temp file — our loader needs a real file path
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                pipeline = get_pipeline()
                result = pipeline.ingest(
                    source=tmp_path,
                    collection_name=collection_choice,
                    extra_metadata={
                        "source":      uploaded_file.name,
                        "source_type": suffix.lstrip("."),
                    },
                )
                os.unlink(tmp_path)

            if result["status"] == "ok":
                st.success(
                    f"✓ Ingested {result['chunks_stored']} chunks "
                    f"from **{uploaded_file.name}**"
                )
                # Bust the collection stats cache
                st.cache_data.clear()
            else:
                st.error(f"Ingestion failed: {result.get('error')}")

    st.divider()

    # ── Collection stats ──────────────────────────────────────────────────────
    st.subheader("📊 Knowledge Base")

    @st.cache_data(ttl=30)
    def get_stats():
        return ChromaClient.get_collection_stats()

    stats = get_stats()
    total = sum(stats.values())
    st.metric("Total chunks indexed", total)

    # Only show non-empty collections
    non_empty = {k: v for k, v in stats.items() if v > 0}
    if non_empty:
        for name, count in sorted(non_empty.items(), key=lambda x: -x[1]):
            label = name.replace("_", " ").title()
            st.progress(count / max(non_empty.values()), text=f"{label}: {count}")
    else:
        st.caption("No documents ingested yet.")

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_result  = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚢 Cleo Supply Chain Intelligence")
st.caption(
    "Ask anything about EDI standards, supply chain processes, "
    "integrations, troubleshooting, or compliance. "
    "Powered by phi3:mini + ChromaDB."
)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns([1, 2, 2])
            cols[0].caption(f"Confidence: {confidence_badge(meta['confidence'], meta['confident'])}")
            if meta.get("sources"):
                cols[1].caption(f"Sources: {', '.join(meta['sources'])}")
            if meta.get("collections_searched"):
                cols[2].caption(f"Collections: {', '.join(meta['collections_searched'])}")

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a supply chain or EDI question...")

if query:
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": query, "meta": {}})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            result = get_generator().answer(query)

        answer   = result["answer"]
        confident = result["confident"]
        confidence = result["confidence"]

        st.markdown(answer)

        # Confidence + source row
        cols = st.columns([1, 2, 2])
        cols[0].caption(f"Confidence: {confidence_badge(confidence, confident)}")
        if result.get("sources"):
            cols[1].caption(f"Sources: {', '.join(result['sources'])}")
        if result.get("collections_searched"):
            cols[2].caption(f"Collections: {', '.join(result['collections_searched'])}")

        # Debug panel (collapsed by default)
        if result.get("chunks"):
            with st.expander(
                f"🔍 Retrieved chunks ({result['chunks_used']} used)", expanded=False
            ):
                for i, chunk in enumerate(result["chunks"], 1):
                    meta  = chunk.get("metadata", {})
                    score = chunk.get("rrf_score", chunk.get("score", 0))
                    source = meta.get("source", "unknown")
                    col_name = meta.get("collection", "")
                    st.markdown(
                        f"**Chunk {i}** — `{source}` | "
                        f"Collection: `{col_name}` | "
                        f"RRF score: `{score:.4f}`"
                    )
                    st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                    if i < len(result["chunks"]):
                        st.divider()

    # Persist to session state
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "meta": {
            "confidence":          confidence,
            "confident":           confident,
            "sources":             result.get("sources", []),
            "collections_searched": result.get("collections_searched", []),
        },
    })
    st.session_state.last_result = result
