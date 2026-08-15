"""Shared session state helpers."""

from __future__ import annotations

import streamlit as st

from src.knowledge.knowledge_base import KnowledgeBaseManager


def get_manager() -> KnowledgeBaseManager:
    if "kb_manager" not in st.session_state:
        st.session_state.kb_manager = KnowledgeBaseManager()
    return st.session_state.kb_manager


def get_active_kb_id() -> str | None:
    return st.session_state.get("active_kb_id")


def set_active_kb(kb_id: str) -> None:
    st.session_state.active_kb_id = kb_id


def init_session() -> None:
    defaults = {
        "active_kb_id": None,
        "last_search_results": None,
        "last_answer": None,
        "selected_citation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    get_manager()
