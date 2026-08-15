"""Chat message rendering."""

from __future__ import annotations

import streamlit as st

from src.models.schemas import Answer


def render_answer(answer: Answer) -> None:
    if answer.abstained:
        st.warning(answer.response)
    else:
        st.markdown(answer.response)

    if answer.citations:
        st.divider()
        from app.components.source_card import render_citations_list
        render_citations_list(answer.citations)

    if st.session_state.get("selected_citation"):
        cite = st.session_state.selected_citation
        st.divider()
        st.markdown("### SOURCE PANEL")
        st.markdown(f"**{cite['document_name']}** — Page {cite.get('page_number', 'N/A')}")
        if cite.get("score") is not None:
            st.caption(f"Retrieval score: {cite['score']:.2f}")
        st.markdown(f'"{cite["text"]}"')
