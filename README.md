# YT CORTEX v1.0

**Turn your documents into an intelligent knowledge base.**

YT CORTEX is an AI-powered document intelligence platform. Upload PDFs, DOCX, TXT, or Markdown files, and CORTEX extracts, chunks, embeds, and indexes them so you can **search semantically**, **ask questions**, and get **grounded answers with citations**.

> This is a **document intelligence / RAG platform** — not an ML Debugger.

## Features (MVP)

- Upload PDF, DOCX, TXT, Markdown
- Text extraction with page/section metadata
- Configurable chunking with overlap
- Sentence Transformer embeddings
- Chroma vector index
- Semantic search (no LLM required)
- Ask mode with grounded answers and citations
- Abstention when evidence is insufficient
- Retrieval debug panel on Ask page
- Optional OpenAI LLM (set `OPENAI_API_KEY`)

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

pytest tests/ -v
streamlit run app/main.py
```

## Usage

1. **Home** — Create a knowledge base, upload documents (or load samples)
2. **Knowledge Base** — View indexed documents, delete/re-index
3. **Search** — Semantic search across your documents
4. **Ask** — Ask questions, get grounded answers with source citations

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Enable LLM-powered RAG answers |
| `CORTEX_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `CORTEX_LLM_MODEL` | `gpt-4o-mini` | OpenAI model for generation |
| `CORTEX_ENABLE_RERANKING` | `false` | Enable cross-encoder reranking |

Without `OPENAI_API_KEY`, Ask mode uses extractive grounded answers from retrieved passages.

## Project Structure

```
yt-cortex-v1/
├── app/
│   ├── main.py                 # Home page
│   ├── pages/                  # Knowledge Base, Search, Ask
│   └── components/
├── src/
│   ├── ingestion/              # Document loaders & parser
│   ├── chunking/               # Text chunking
│   ├── embeddings/             # Sentence Transformers
│   ├── retrieval/              # Chroma + semantic search
│   ├── generation/             # RAG prompts & generator
│   ├── citations/              # Citation engine
│   ├── knowledge/              # Knowledge base manager
│   └── evaluation/             # Retrieval metrics
├── examples/                   # Sample documents
├── tests/
└── data/                       # Runtime storage (gitignored)
```

## Sample Documents

- `examples/machine_learning_notes.md` — Random Forest, overfitting, ensembles
- `examples/university_regulations.md` — Attendance, scholarships, exams

## License

MIT
