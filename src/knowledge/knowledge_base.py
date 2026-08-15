"""Knowledge base management."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.chunking.chunker import chunk_document
from src.embeddings.embedder import embed_texts
from src.ingestion.loaders import load_document
from src.ingestion.parser import parse_document
from src.models.schemas import (
    Document,
    DocumentStatus,
    ExtractedDocument,
    KnowledgeBase,
    ProcessingProgress,
    ProcessingStep,
)
from src.retrieval.vector_store import VectorStore


class KnowledgeBaseManager:
    """Manages knowledge bases, document ingestion, and indexing."""

    def __init__(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        if config.KB_REGISTRY_PATH.exists():
            return json.loads(config.KB_REGISTRY_PATH.read_text(encoding="utf-8"))
        return {"knowledge_bases": {}}

    def _save_registry(self) -> None:
        config.KB_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.KB_REGISTRY_PATH.write_text(
            json.dumps(self._registry, indent=2), encoding="utf-8"
        )

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        kbs = []
        for kb_id, data in self._registry.get("knowledge_bases", {}).items():
            docs = []
            for d in data.get("documents", []):
                d = dict(d)
                if isinstance(d.get("status"), str):
                    d["status"] = DocumentStatus(d["status"])
                docs.append(Document(**d))
            kbs.append(
                KnowledgeBase(
                    id=kb_id,
                    name=data["name"],
                    description=data.get("description", ""),
                    created_at=data.get("created_at", ""),
                    documents=docs,
                )
            )
        return kbs

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase | None:
        for kb in self.list_knowledge_bases():
            if kb.id == kb_id:
                return kb
        return None

    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBase:
        kb_id = str(uuid.uuid4())[:8]
        kb_data = {
            "name": name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": [],
        }
        self._registry.setdefault("knowledge_bases", {})[kb_id] = kb_data
        self._save_registry()
        return KnowledgeBase(
            id=kb_id,
            name=name,
            description=description,
            created_at=kb_data["created_at"],
        )

    def _get_store(self, kb_id: str) -> VectorStore:
        return VectorStore(collection_name=f"kb_{kb_id}")

    def _kb_doc_dir(self, kb_id: str) -> Path:
        path = config.DOCUMENTS_DIR / kb_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ingest_document(
        self,
        kb_id: str,
        file_path: Path | str,
        on_progress: callable | None = None,
    ) -> tuple[Document, ProcessingProgress]:
        """Full pipeline: load → parse → chunk → embed → index."""
        kb = self.get_knowledge_base(kb_id)
        if kb is None:
            raise ValueError(f"Knowledge base '{kb_id}' not found.")

        path = Path(file_path)
        doc_id = str(uuid.uuid4())[:12]
        progress = ProcessingProgress(document_name=path.name)

        def _step(step: ProcessingStep, message: str = "") -> None:
            progress.steps_completed.append(step)
            progress.message = message
            if on_progress:
                on_progress(progress)

        _step(ProcessingStep.VALIDATED, "File validated")

        dest = self._kb_doc_dir(kb_id) / f"{doc_id}_{path.name}"
        shutil.copy2(path, dest)

        extracted: ExtractedDocument = load_document(dest, document_id=doc_id)
        _step(ProcessingStep.EXTRACTED, "Text extracted")
        _step(ProcessingStep.PAGES_DETECTED, f"{len(extracted.pages)} pages detected")

        parsed = parse_document(extracted)
        chunks = chunk_document(parsed)
        progress.chunk_count = len(chunks)
        _step(ProcessingStep.CHUNKED, f"{len(chunks)} chunks created")

        if not chunks:
            raise ValueError("No chunks produced from document.")

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts).tolist()
        _step(ProcessingStep.EMBEDDED, "Embeddings generated")

        store = self._get_store(kb_id)
        store.add_chunks(chunks, embeddings)
        _step(ProcessingStep.INDEXED, "Knowledge index updated")

        doc = Document(
            id=doc_id,
            name=path.name,
            file_type=path.suffix.lower(),
            upload_time=datetime.now(timezone.utc).isoformat(),
            page_count=len(parsed.pages),
            chunk_count=len(chunks),
            status=DocumentStatus.INDEXED,
            file_path=str(dest),
            is_scanned_warning=extracted.is_scanned_warning,
        )

        doc_dict = {**doc.__dict__, "status": doc.status.value}
        kb_data = self._registry["knowledge_bases"][kb_id]
        kb_data["documents"].append(doc_dict)
        self._save_registry()

        return doc, progress

    def delete_document(self, kb_id: str, document_id: str) -> bool:
        kb_data = self._registry.get("knowledge_bases", {}).get(kb_id)
        if not kb_data:
            return False

        docs = kb_data.get("documents", [])
        doc = next((d for d in docs if d["id"] == document_id), None)
        if not doc:
            return False

        file_path = doc.get("file_path")
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()

        store = self._get_store(kb_id)
        store.delete_by_document(document_id)

        kb_data["documents"] = [d for d in docs if d["id"] != document_id]
        self._save_registry()
        return True

    def delete_knowledge_base(self, kb_id: str) -> bool:
        if kb_id not in self._registry.get("knowledge_bases", {}):
            return False
        store = self._get_store(kb_id)
        store.delete_collection()
        doc_dir = config.DOCUMENTS_DIR / kb_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        del self._registry["knowledge_bases"][kb_id]
        self._save_registry()
        return True

    def get_vector_store(self, kb_id: str) -> VectorStore:
        return self._get_store(kb_id)
