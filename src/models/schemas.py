"""Core data models for YT CORTEX."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class ProcessingStep(str, Enum):
    VALIDATED = "validated"
    EXTRACTED = "extracted"
    PAGES_DETECTED = "pages_detected"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"


@dataclass
class PageContent:
    page_number: int
    text: str
    section_title: str = ""


@dataclass
class ExtractedDocument:
    document_id: str
    document_name: str
    file_type: str
    pages: list[PageContent]
    metadata: dict[str, Any] = field(default_factory=dict)
    is_scanned_warning: bool = False


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    section_title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    id: str
    name: str
    file_type: str
    upload_time: str
    page_count: int
    chunk_count: int
    status: DocumentStatus
    file_path: str = ""
    is_scanned_warning: bool = False


@dataclass
class KnowledgeBase:
    id: str
    name: str
    description: str = ""
    created_at: str = ""
    documents: list[Document] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(d.chunk_count for d in self.documents)

    @property
    def is_ready(self) -> bool:
        return bool(self.documents) and all(
            d.status == DocumentStatus.INDEXED for d in self.documents
        )


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    rank: int
    document_id: str
    document_name: str
    page_number: int
    section_title: str
    text: str


@dataclass
class Citation:
    index: int
    document_name: str
    page_number: int
    chunk_id: str
    excerpt: str


@dataclass
class Answer:
    question: str
    response: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]
    abstained: bool = False
    retrieval_debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingProgress:
    document_name: str
    steps_completed: list[ProcessingStep] = field(default_factory=list)
    chunk_count: int = 0
    message: str = ""
