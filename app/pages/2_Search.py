"""Semantic Search page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.session import get_active_kb_id, get_manager, init_session
from app.components.source_card import render_search_result
from src.retrieval.semantic_search import semantic_search

init_session()
manager = get_manager()
kb_id = get_active_kb_id()

st.markdown("## Search Your Knowledge")

if not kb_id:
    st.warning("Select or create a knowledge base on the Home page.")
    st.stop()

kb = manager.get_knowledge_base(kb_id)
if not kb or not kb.documents:
    st.info("Upload documents before searching.")
    st.stop()

query = st.text_input(
    "Search query",
    placeholder="random forest overfitting",
    label_visibility="collapsed",
)

if st.button("Search", type="primary") or query:
    if not query.strip():
        st.stop()

    with st.spinner("Searching..."):
        store = manager.get_vector_store(kb_id)
        results = semantic_search(store, query.strip())
        st.session_state.last_search_results = results

    if not results:
        st.warning("No results found.")
    else:
        st.caption("Relevance scores are retrieval scores, not probabilities.")
        for i, result in enumerate(results, start=1):
            render_search_result(result, i)
            st.divider()
