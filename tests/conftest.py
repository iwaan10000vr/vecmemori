"""pytest fixtures for vecmemori tests."""

import tempfile
import os
import pytest

from vecmemori import MemoryStore, FactRetriever


@pytest.fixture
def db_path():
    """Temporary SQLite database path."""
    path = tempfile.mktemp(suffix=".db")
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def store(db_path):
    """MemoryStore instance backed by a temp DB."""
    s = MemoryStore(db_path=db_path, default_trust=0.5, require_embeddings=False)
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with a handful of sample facts."""
    facts = [
        ("User prefers dark mode in all applications", "user_pref", "preference,ui"),
        ("Project uses pytest for testing", "project", "testing,dev"),
        ("GPU is NVIDIA RTX 5060 Ti 16GB", "tool", "hardware,gpu"),
        ("Working directory is WSL/Ubuntu", "tool", "environment"),
        ("User wants to avoid cloud dependencies", "user_pref", "privacy"),
    ]
    for content, category, tags in facts:
        store.add_fact(content, category=category, tags=tags)
    return store


@pytest.fixture
def retriever(store):
    """FactRetriever with default weights."""
    return FactRetriever(store=store, require_embeddings=False)


@pytest.fixture
def populated_retriever(populated_store):
    """FactRetriever on a populated store."""
    return FactRetriever(store=populated_store, require_embeddings=False)
