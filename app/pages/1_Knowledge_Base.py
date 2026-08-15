"""Knowledge Base page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.document_card import render_document_card
from app.components.session import get_active_kb_id, get_manager, init_session

init_session()
manager = get_manager()
kb_id = get_active_kb_id()

st.markdown("## Knowledge Base")

if not kb_id:
    st.warning("Select or create a knowledge base on the Home page.")
    st.stop()

kb = manager.get_knowledge_base(kb_id)
if not kb:
    st.error("Knowledge base not found.")
    st.stop()

status = "Ready" if kb.is_ready else "Indexing..."
st.metric("Documents", len(kb.documents))
col1, col2, col3 = st.columns(3)
col1.metric("Chunks", f"{kb.total_chunks:,}")
col2.metric("Status", status)
col3.metric("Name", kb.name)

st.divider()

if not kb.documents:
    st.info("No documents yet. Upload documents from the Home page.")
else:
    for doc in kb.documents:
        def _delete(doc_id, _kb=kb_id):
            manager.delete_document(_kb, doc_id)
            st.rerun()
        render_document_card(doc, kb_id, _delete)

if st.button("Delete Knowledge Base", type="secondary"):
    manager.delete_knowledge_base(kb_id)
    st.session_state.active_kb_id = None
    st.rerun()
