"""Tests for embeddings."""

from src.embeddings.embedder import embed_query, embed_texts


def test_embed_texts_shape():
    texts = ["Random Forest reduces overfitting.", "Decision trees split on features."]
    embeddings = embed_texts(texts)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_embed_query():
    vec = embed_query("What is Random Forest?")
    assert len(vec) > 0
