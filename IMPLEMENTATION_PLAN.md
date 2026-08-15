# YT CORTEX v1.0 — Implementation Plan

## Product Definition

YT CORTEX is a **document intelligence and RAG platform** — NOT an ML Debugger.

**Core workflow:** Upload → Extract → Chunk → Embed → Index → Search / Ask → Retrieve → Generate → Cite

## Architecture

```text
Documents (PDF/DOCX/TXT/MD)
        ↓
   Ingestion (PyMuPDF, python-docx)
        ↓
   Parsing & Cleaning
        ↓
   Chunking (700 tokens, 80 overlap)
        ↓
   Embeddings (Sentence Transformers)
        ↓
   Vector Store (ChromaDB)
        ↓
   Semantic Search / RAG
        ↓
   Grounded Answer + Citations
```

## MVP Phases (Implemented)

| Phase | Components | Status |
|-------|-----------|--------|
| 1 — Ingestion | `loaders.py`, `parser.py` | Done |
| 2 — Chunking & Embeddings | `chunker.py`, `embedder.py` | Done |
| 3 — Vector Index | `vector_store.py`, `semantic_search.py` | Done |
| 4 — RAG & Citations | `generator.py`, `citation_engine.py` | Done |
| 5 — Streamlit UI | Home, KB, Search, Ask | Done |

## V1.1 (Future)

- Retrieval Debug dedicated page
- Conversation memory / query rewriting
- Summarize, Explain, Analyze modes
- Better source panel UX

## V2.0 (Future)

- Hybrid search (BM25 + semantic)
- Cross-encoder reranking (stub exists)
- Evaluation dashboard
- Configurable retrieval parameters

## Testing Strategy

- Unit tests: loaders, chunker, embeddings, citations, generator abstention
- Integration tests: full KB ingest → search pipeline
- Run: `pytest tests/ -v`

## Acceptance Criteria (PRD §39)

- [x] Upload supported documents
- [x] Text extraction with page metadata
- [x] Chunking with overlap
- [x] Embeddings generated and stored
- [x] Semantic search works
- [x] Ask mode with grounded answers
- [x] Citations and source inspection
- [x] Abstention on insufficient evidence
- [x] No ML Debugger functionality
- [x] Core pipeline tests
- [x] Runs from requirements.txt
