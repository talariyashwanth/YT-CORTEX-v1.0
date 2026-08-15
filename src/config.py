"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"
KB_REGISTRY_PATH = DATA_DIR / "knowledge_bases.json"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
MAX_FILE_SIZE_MB = 50

# Chunking (token approx: 1 token ≈ 4 chars)
CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 80
CHARS_PER_TOKEN = 4

# Embeddings
EMBEDDING_MODEL = os.getenv("CORTEX_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval
TOP_K_RETRIEVAL = 10
TOP_K_CONTEXT = 5
MIN_RELEVANCE_SCORE = 0.35

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("CORTEX_LLM_MODEL", "gpt-4o-mini")

# Reranking (optional)
ENABLE_RERANKING = os.getenv("CORTEX_ENABLE_RERANKING", "false").lower() == "true"
RERANKER_MODEL = os.getenv("CORTEX_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
