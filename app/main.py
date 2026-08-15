"""YT CORTEX v1.0 — Home."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.session import get_manager, init_session, set_active_kb
from src import config

st.set_page_config(
    page_title="YT CORTEX v1.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .cortex-title { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em; }
    .cortex-tagline { color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem; }
    .upload-box {
        border: 2px dashed #d1d5db; border-radius: 12px;
        padding: 2.5rem; text-align: center; background: #f9fafb;
    }
    .kb-card {
        border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 1rem; margin-bottom: 0.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

init_session()
manager = get_manager()

st.sidebar.markdown("# YT CORTEX v1.0")
st.sidebar.markdown("*Document Intelligence Platform*")
st.sidebar.divider()
st.sidebar.caption(f"Embedding: {config.EMBEDDING_MODEL}")
if config.OPENAI_API_KEY:
    st.sidebar.success("LLM: OpenAI connected")
else:
    st.sidebar.info("LLM: Extractive mode (set OPENAI_API_KEY for full RAG)")

st.markdown('<p class="cortex-title">YT CORTEX v1.0</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="cortex-tagline">Turn your documents into an intelligent knowledge base.</p>',
    unsafe_allow_html=True,
)

# Create knowledge base
with st.expander("Create Knowledge Base", expanded=not manager.list_knowledge_bases()):
    col1, col2 = st.columns([3, 1])
    with col1:
        kb_name = st.text_input("Knowledge base name", placeholder="College Notes")
    with col2:
        if st.button("Create", type="primary", use_container_width=True):
            if kb_name.strip():
                kb = manager.create_knowledge_base(kb_name.strip())
                set_active_kb(kb.id)
                st.success(f"Created knowledge base: {kb.name}")
                st.rerun()
            else:
                st.error("Enter a name.")

# Select active KB
kbs = manager.list_knowledge_bases()
if kbs:
    kb_options = {f"{kb.name} ({len(kb.documents)} docs)": kb.id for kb in kbs}
    selected = st.selectbox(
        "Active Knowledge Base",
        options=list(kb_options.keys()),
        index=0,
    )
    set_active_kb(kb_options[selected])
    active_kb = manager.get_knowledge_base(kb_options[selected])

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("#### DROP YOUR DOCUMENT HERE")
    st.caption("PDF · DOCX · TXT · MD")
    uploaded = st.file_uploader(
        "Browse files",
        type=["pdf", "docx", "txt", "md", "markdown"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if uploaded and st.button("Upload & Index Document", type="primary", use_container_width=True):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            progress_bar = st.progress(0)
            status = st.empty()

            def on_progress(p):
                step_pct = len(p.steps_completed) / 6
                progress_bar.progress(min(step_pct, 1.0))
                status.text(p.message)

            try:
                doc, prog = manager.ingest_document(
                    active_kb.id, tmp_path, on_progress=on_progress
                )
                progress_bar.progress(1.0)
                status.empty()
                msg = f"✓ {doc.name} indexed — {doc.chunk_count:,} chunks"
                if doc.is_scanned_warning:
                    msg += "\n\n⚠ This document appears to contain scanned pages. OCR may be required."
                st.success(msg)
                Path(tmp_path).unlink(missing_ok=True)
                st.rerun()
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                Path(tmp_path).unlink(missing_ok=True)

    with col_b:
        if st.button("Load Sample Documents", use_container_width=True):
            examples = [
                ROOT / "examples" / "machine_learning_notes.md",
                ROOT / "examples" / "university_regulations.md",
            ]
            for ex in examples:
                if ex.exists():
                    manager.ingest_document(active_kb.id, ex)
            st.success("Sample documents indexed.")
            st.rerun()

    st.subheader("Recent Knowledge Bases")
    for kb in kbs:
        st.markdown(
            f'<div class="kb-card"><strong>{kb.name}</strong> — '
            f'{len(kb.documents)} documents · {kb.total_chunks:,} chunks</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Create a knowledge base to get started.")
