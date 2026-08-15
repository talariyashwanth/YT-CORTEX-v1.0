"""Document parsing utilities."""

from __future__ import annotations

import re

from src.models.schemas import ExtractedDocument, PageContent


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_document(extracted: ExtractedDocument) -> ExtractedDocument:
    """Clean and normalize extracted document content."""
    cleaned_pages: list[PageContent] = []
    for page in extracted.pages:
        cleaned = clean_text(page.text)
        if cleaned:
            cleaned_pages.append(
                PageContent(
                    page_number=page.page_number,
                    text=cleaned,
                    section_title=page.section_title,
                )
            )
    extracted.pages = cleaned_pages
    extracted.metadata["page_count"] = len(cleaned_pages)
    return extracted


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]
