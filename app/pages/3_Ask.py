"""Ask CORTEX — RAG page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.chat_message import render_answer
from app.components.session import get_active_kb_id, get_manager, init_session
from src.generation.generator import generate_answer
from src.retrieval.semantic_search import semantic_search

init_session()
manager = get_manager()
kb_id = get_active_kb_id()

st.markdown("## Ask CORTEX")

if not kb_id:
    st.warning("Select or create a knowledge base on the Home page.")
    st.stop()

kb = manager.get_knowledge_base(kb_id)
if not kb or not kb.documents:
    st.info("Upload documents before asking questions.")
    st.stop()

question = st.text_area(
    "Your question",
    placeholder="What is Random Forest and why does it reduce overfitting?",
    height=100,
)

if st.button("Ask CORTEX", type="primary"):
    if not question.strip():
        st.error("Enter a question.")
        st.stop()

    with st.spinner("Retrieving and generating..."):
        store = manager.get_vector_store(kb_id)
        results = semantic_search(store, question.strip())
        answer = generate_answer(question.strip(), results)
        st.session_state.last_answer = answer

if st.session_state.get("last_answer"):
    render_answer(st.session_state.last_answer)

    with st.expander("Why this answer? — Retrieval Debug"):
        debug = st.session_state.last_answer.retrieval_debug
        st.markdown("**Vector Retrieval**")
        for item in debug.get("vector_retrieval", []):
            st.markdown(
                f"{item['rank']}. {item['document']} — p.{item['page']} — "
                f"Score: {item['score']:.2f}"
            )
        st.markdown("**Selected Context**")
        for item in debug.get("selected_context", []):
            st.markdown(
                f"{item['rank']}. {item['document']} — p.{item['page']} — "
                f"Score: {item['score']:.2f}"
            )
