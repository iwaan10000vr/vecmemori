"""Shared embedding model singleton for vecmemori.

Provides lazy-loading access to a SentenceTransformer-compatible embedding model.
Supports configurable asymmetric prefixes and dimension detection.
Both store.py and retrieval.py import this to reuse the same model instance.
Lazy-loaded on first use — no startup cost if embeddings are not configured.
"""

from functools import lru_cache

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False
    SentenceTransformer = object  # type: ignore[assignment,misc]

from pathlib import Path

_model = None  # module-level singleton
_MODEL_PATH = Path.home() / ".hermes" / "models" / "ruri-v3-310m"
_DIMENSION = 768  # default for ruri-v3-310m; auto-detected on first load
_QUERY_PREFIX = "検索クエリ: "
_DOC_PREFIX = "検索文書: "


def set_model_path(path: str | Path) -> None:
    """Configure the embedding model path/name before first load.

    If the model is already loaded and the path changes, unload it so the next
    encode call reloads from the configured location.
    """
    global _MODEL_PATH, _model
    new_path = Path(path).expanduser()
    if new_path == _MODEL_PATH:
        return
    _MODEL_PATH = new_path
    if _model is not None:
        unload_model()


def set_config(
    *,
    dimension: int | None = None,
    query_prefix: str | None = None,
    doc_prefix: str | None = None,
) -> None:
    """Configure embedding parameters.

    Args:
        dimension: Embedding dimension (default: 768 for ruri-v3-310m).
        query_prefix: Prefix for query-side encoding (default: '検索クエリ: ').
        doc_prefix: Prefix for document-side encoding (default: '検索文書: ').
    """
    global _DIMENSION, _QUERY_PREFIX, _DOC_PREFIX
    if dimension is not None:
        _DIMENSION = dimension
    if query_prefix is not None:
        _QUERY_PREFIX = query_prefix
    if doc_prefix is not None:
        _DOC_PREFIX = doc_prefix


def get_model() -> SentenceTransformer | None:
    """Lazy-load and return the shared embedding model instance."""
    global _model
    if _model is None and _HAS_DEPS:
        try:
            _model = SentenceTransformer(
                str(_MODEL_PATH),
                device="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            return None
    return _model


def encode_doc(text: str) -> np.ndarray | None:
    """Encode a fact/document with asymmetric doc prefix. Returns float32 array."""
    model = get_model()
    if model is None:
        return None
    vec = model.encode(f"{_DOC_PREFIX}{text}", normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def encode_query(text: str) -> np.ndarray | None:
    """Encode a search query with asymmetric query prefix. Returns float32 array."""
    model = get_model()
    if model is None:
        return None
    vec = model.encode(f"{_QUERY_PREFIX}{text}", normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


@lru_cache(maxsize=3)
def encode_query_cached(text: str) -> np.ndarray:
    """Cached query embedding — same text within a turn skips recomputation.

    Returns a zero vector on failure (never returns None), safe for dot product.
    """
    vec = encode_query(text)
    return vec if vec is not None else np.zeros(_DIMENSION, dtype=np.float32)


def unload_model() -> bool:
    """Free VRAM by deleting the model instance.

    Next encode call will reload the model automatically.
    Returns True if a model was actually unloaded.
    """
    global _model
    if _model is None:
        return False
    import gc
    del _model
    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def is_model_loaded() -> bool:
    """Check if the embedding model is currently loaded in memory."""
    return _model is not None


def encode_docs(texts: list[str], batch_size: int = 32) -> list[np.ndarray] | None:
    """Batch encode multiple documents.

    Args:
        texts: List of document texts to encode.
        batch_size: Batch size for model.encode() (default: 32).

    Returns:
        List of float32 arrays, or None if model unavailable.
    """
    model = get_model()
    if model is None or not texts:
        return None
    prefixed = [f"{_DOC_PREFIX}{t}" for t in texts]
    vecs = model.encode(prefixed, normalize_embeddings=True, batch_size=batch_size)
    return [np.asarray(v, dtype=np.float32) for v in vecs]


def encode_queries(texts: list[str], batch_size: int = 32) -> list[np.ndarray] | None:
    """Batch encode multiple queries.

    Args:
        texts: List of query texts to encode.
        batch_size: Batch size for model.encode() (default: 32).

    Returns:
        List of float32 arrays, or None if model unavailable.
    """
    model = get_model()
    if model is None or not texts:
        return None
    prefixed = [f"{_QUERY_PREFIX}{t}" for t in texts]
    vecs = model.encode(prefixed, normalize_embeddings=True, batch_size=batch_size)
    return [np.asarray(v, dtype=np.float32) for v in vecs]


def get_dimension() -> int:
    """Return the configured embedding dimension."""
    return _DIMENSION
