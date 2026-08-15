"""Embedding generation using Sentence Transformers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src import config


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    if not texts:
        return np.array([])
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def embed_query(query: str) -> np.ndarray:
    """Generate embedding for a single query."""
    return embed_texts([query])[0]
