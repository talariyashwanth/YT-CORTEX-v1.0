"""Citation formatting and source tracking."""

from __future__ import annotations

from src.models.schemas import Citation, RetrievalResult


def build_citations(results: list[RetrievalResult]) -> list[Citation]:
    """Build citation objects from retrieval results."""
    citations: list[Citation] = []
    for i, r in enumerate(results, start=1):
        excerpt = r.text[:300].strip()
        if len(r.text) > 300:
            excerpt += "..."
        citations.append(
            Citation(
                index=i,
                document_name=r.document_name,
                page_number=r.page_number,
                chunk_id=r.chunk_id,
                excerpt=excerpt,
            )
        )
    return citations


def format_citation_line(citation: Citation) -> str:
    page = f" — p.{citation.page_number}" if citation.page_number else ""
    return f"[{citation.index}] {citation.document_name}{page}"


def format_sources_block(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["Sources:", "─" * 30]
    for c in citations:
        lines.append(format_citation_line(c))
    return "\n".join(lines)
