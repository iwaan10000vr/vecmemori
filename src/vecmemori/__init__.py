"""vecmemori — Hybrid vector + symbolic memory engine for AI agents.

A local SQLite-backed fact store with:
- FTS5 full-text search
- HRR (Holographic Reduced Representation) symbolic algebra
- Optional neural embedding (sentence-transformers) for semantic search

Core modules (zero external dependencies except numpy):
    MemoryStore    — SQLite fact store with entity resolution and trust scoring
    FactRetriever  — 4-strategy hybrid search (FTS5 + Jaccard + HRR + embeddings)
    hrr            — HRR vector symbolic algebra (phase vectors)
    embedder       — SentenceTransformer embedding model wrapper
"""

from .store import MemoryStore
from .retrieval import FactRetriever

__all__ = ["MemoryStore", "FactRetriever"]
