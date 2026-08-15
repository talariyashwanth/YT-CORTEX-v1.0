"""Tests for citation engine."""

from src.citations.citation_engine import build_citations, format_citation_line
from src.models.schemas import RetrievalResult


def test_build_citations():
    results = [
        RetrievalResult(
            chunk_id="c1", score=0.9, rank=1,
            document_id="d1", document_name="Regulations.pdf",
            page_number=14, section_title="Attendance",
            text="Students must maintain 75% attendance.",
        )
    ]
    citations = build_citations(results)
    assert len(citations) == 1
    assert citations[0].index == 1
    assert citations[0].page_number == 14
    assert "75%" in citations[0].excerpt


def test_format_citation_line():
    from src.models.schemas import Citation
    c = Citation(index=1, document_name="Regulations.pdf", page_number=14,
                 chunk_id="c1", excerpt="test")
    line = format_citation_line(c)
    assert "[1]" in line
    assert "Regulations.pdf" in line
    assert "p.14" in line
