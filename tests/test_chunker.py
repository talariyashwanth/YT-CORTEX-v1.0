"""Tests for text chunking."""

from pathlib import Path

from src.chunking.chunker import chunk_document
from src.ingestion.loaders import load_document
from src.ingestion.parser import parse_document

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_chunk_document_preserves_metadata():
    extracted = parse_document(load_document(EXAMPLES / "machine_learning_notes.md"))
    chunks = chunk_document(extracted)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_name == "machine_learning_notes.md"
        assert chunk.document_id == extracted.document_id
        assert chunk.page_number >= 1
        assert len(chunk.text) > 0


def test_chunks_have_unique_ids():
    extracted = parse_document(load_document(EXAMPLES / "machine_learning_notes.md"))
    chunks = chunk_document(extracted)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
