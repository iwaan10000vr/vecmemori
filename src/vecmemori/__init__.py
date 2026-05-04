"""vecmemori — Hybrid memory engine for AI agents.

A local SQLite-backed fact store with:
- FTS5 full-text search (with optional fugashi Japanese tokenization)
- Optional neural embedding (sentence-transformers) for semantic search

Core modules (zero external dependencies except numpy):
    MemoryStore    — SQLite fact store with entity resolution and trust scoring
    FactRetriever  — 2-strategy hybrid search (FTS5 + neural embeddings)
    embedder       — SentenceTransformer embedding model wrapper
"""

from .store import MemoryStore
from .retrieval import FactRetriever

__all__ = ["MemoryStore", "FactRetriever"]
