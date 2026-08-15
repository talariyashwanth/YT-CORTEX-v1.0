"""Tests for document loaders."""

from pathlib import Path

import pytest

from src.ingestion.loaders import IngestionError, load_document, load_markdown

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_load_markdown():
    path = EXAMPLES / "machine_learning_notes.md"
    doc = load_document(path)
    assert doc.document_name == "machine_learning_notes.md"
    assert len(doc.pages) > 0
    # The markdown loader splits each "#" heading into its own page, so the
    # title page holds only the H1 and the body content lives in later pages.
    full_text = " ".join(p.text for p in doc.pages)
    assert "Random Forest" in full_text
    assert doc.pages[0].section_title == "Machine Learning Notes"


def test_load_university_regulations():
    doc = load_document(EXAMPLES / "university_regulations.md")
    full_text = " ".join(p.text for p in doc.pages)
    assert "75%" in full_text
    assert "attendance" in full_text.lower()


def test_reject_missing_file():
    with pytest.raises(IngestionError):
        load_document("nonexistent.pdf")


def test_reject_unsupported_format(tmp_path):
    bad = tmp_path / "file.xyz"
    bad.write_text("data")
    with pytest.raises(IngestionError):
        load_document(bad)
