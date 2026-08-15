"""Chroma vector store wrapper."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from src import config
from src.models.schemas import Chunk, RetrievalResult


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, collection_name: str):
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
                "section_title": c.section_title or "",
            }
            for c in chunks
        ]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def delete_by_document(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass

    def search(
        self,
        query_embedding: list[float],
        top_k: int = config.TOP_K_RETRIEVAL,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        where: dict[str, Any] | None = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: list[RetrievalResult] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieval_results

        for rank, (chunk_id, doc, meta, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            # Chroma cosine distance: 0 = identical; convert to similarity
            score = round(1.0 - distance, 4)
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    rank=rank,
                    document_id=meta.get("document_id", ""),
                    document_name=meta.get("document_name", ""),
                    page_number=int(meta.get("page_number", 0)),
                    section_title=meta.get("section_title", ""),
                    text=doc or "",
                )
            )
        return retrieval_results

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
