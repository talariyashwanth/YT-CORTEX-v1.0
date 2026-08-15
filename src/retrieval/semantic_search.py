"""Semantic search over the vector index."""

from __future__ import annotations

from src import config
from src.embeddings.embedder import embed_query
from src.models.schemas import RetrievalResult
from src.retrieval.vector_store import VectorStore


def semantic_search(
    store: VectorStore,
    query: str,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[RetrievalResult]:
    """Perform semantic search and return ranked results."""
    embedding = embed_query(query).tolist()
    return store.search(
        query_embedding=embedding,
        top_k=top_k or config.TOP_K_RETRIEVAL,
        document_ids=document_ids,
    )
