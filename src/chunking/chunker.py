"""Text chunking with overlap."""

from __future__ import annotations

import uuid

from src import config
from src.ingestion.parser import split_into_paragraphs
from src.models.schemas import Chunk, ExtractedDocument


def _token_to_chars(tokens: int) -> int:
    return tokens * config.CHARS_PER_TOKEN


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            break_point = chunk.rfind(". ")
            if break_point > chunk_size // 2:
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def chunk_document(
    extracted: ExtractedDocument,
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Chunk an extracted document preserving page and section metadata."""
    chunk_size = _token_to_chars(chunk_size_tokens or config.CHUNK_SIZE_TOKENS)
    overlap = _token_to_chars(overlap_tokens or config.CHUNK_OVERLAP_TOKENS)

    chunks: list[Chunk] = []
    chunk_index = 0

    for page in extracted.pages:
        paragraphs = split_into_paragraphs(page.text)
        buffer = ""
        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    for piece in _split_text(buffer, chunk_size, overlap):
                        chunk_index += 1
                        chunks.append(
                            Chunk(
                                id=f"{extracted.document_id}_{chunk_index}",
                                document_id=extracted.document_id,
                                document_name=extracted.document_name,
                                page_number=page.page_number,
                                chunk_id=f"chunk_{chunk_index}",
                                section_title=page.section_title,
                                text=piece,
                                metadata={"source_text": piece},
                            )
                        )
                for piece in _split_text(para, chunk_size, overlap):
                    chunk_index += 1
                    chunks.append(
                        Chunk(
                            id=f"{extracted.document_id}_{chunk_index}",
                            document_id=extracted.document_id,
                            document_name=extracted.document_name,
                            page_number=page.page_number,
                            chunk_id=f"chunk_{chunk_index}",
                            section_title=page.section_title,
                            text=piece,
                            metadata={"source_text": piece},
                        )
                    )
                buffer = ""

        if buffer:
            for piece in _split_text(buffer, chunk_size, overlap):
                chunk_index += 1
                chunks.append(
                    Chunk(
                        id=f"{extracted.document_id}_{chunk_index}",
                        document_id=extracted.document_id,
                        document_name=extracted.document_name,
                        page_number=page.page_number,
                        chunk_id=f"chunk_{chunk_index}",
                        section_title=page.section_title,
                        text=piece,
                        metadata={"source_text": piece},
                    )
                )

    return chunks
