"""Source card and citation panel."""

from __future__ import annotations

import streamlit as st

from src.citations.citation_engine import format_citation_line
from src.models.schemas import Citation, RetrievalResult


def render_search_result(result: RetrievalResult, index: int) -> None:
    st.markdown(f"**{index:02d} — {result.document_name} — Page {result.page_number}**")
    st.markdown(f"*Relevance: {result.score:.2f}*")
    excerpt = result.text[:400] + ("..." if len(result.text) > 400 else "")
    st.markdown(f"> {excerpt}")
    if st.button("Open Source", key=f"src_{result.chunk_id}_{index}"):
        st.session_state.selected_citation = {
            "document_name": result.document_name,
            "page_number": result.page_number,
            "text": result.text,
            "score": result.score,
        }


def render_citation_panel(citation: Citation) -> None:
    st.markdown("### SOURCE")
    st.markdown(f"**{citation.document_name}**")
    if citation.page_number:
        st.markdown(f"Page {citation.page_number}")
    st.divider()
    st.markdown(f'"{citation.excerpt}"')


def render_citations_list(citations: list[Citation]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for c in citations:
        if st.button(format_citation_line(c), key=f"cite_{c.index}_{c.chunk_id}"):
            st.session_state.selected_citation = {
                "document_name": c.document_name,
                "page_number": c.page_number,
                "text": c.excerpt,
                "score": None,
            }
