"""Document card component."""

from __future__ import annotations

import streamlit as st

from src.models.schemas import Document, DocumentStatus


def render_document_card(doc: Document, kb_id: str, on_delete: callable) -> None:
    status_icon = "✓" if doc.status == DocumentStatus.INDEXED else "⏳"
    warning = " ⚠ Scanned pages" if doc.is_scanned_warning else ""

    col1, col2, col3 = st.columns([4, 2, 1])
    with col1:
        st.markdown(f"**{doc.name}**{warning}")
        st.caption(f"{doc.chunk_count:,} chunks · {doc.page_count} pages")
    with col2:
        st.markdown(f"{status_icon} {doc.status.value.title()}")
    with col3:
        if st.button("Delete", key=f"del_{kb_id}_{doc.id}"):
            on_delete(doc.id)
