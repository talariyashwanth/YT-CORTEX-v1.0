"""Tests for knowledge base pipeline."""

import shutil
from pathlib import Path

import pytest

from src import config
from src.knowledge.knowledge_base import KnowledgeBaseManager
from src.retrieval.semantic_search import semantic_search

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DOCUMENTS_DIR", tmp_path / "data" / "documents")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "data" / "chroma")
    monkeypatch.setattr(config, "KB_REGISTRY_PATH", tmp_path / "data" / "knowledge_bases.json")
    return KnowledgeBaseManager()


def test_create_and_ingest(manager):
    kb = manager.create_knowledge_base("Test KB")
    doc, progress = manager.ingest_document(kb.id, EXAMPLES / "machine_learning_notes.md")
    assert doc.chunk_count > 0
    assert len(progress.steps_completed) == 6

    kb_updated = manager.get_knowledge_base(kb.id)
    assert len(kb_updated.documents) == 1


def test_semantic_search_after_ingest(manager):
    kb = manager.create_knowledge_base("Search KB")
    manager.ingest_document(kb.id, EXAMPLES / "machine_learning_notes.md")
    store = manager.get_vector_store(kb.id)
    results = semantic_search(store, "random forest overfitting")
    assert len(results) > 0
    assert results[0].score > 0
    assert "Random Forest" in results[0].text or "overfitting" in results[0].text.lower()


def test_delete_document(manager):
    kb = manager.create_knowledge_base("Delete KB")
    doc, _ = manager.ingest_document(kb.id, EXAMPLES / "machine_learning_notes.md")
    assert manager.delete_document(kb.id, doc.id)
    kb_updated = manager.get_knowledge_base(kb.id)
    assert len(kb_updated.documents) == 0
