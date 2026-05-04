"""Hybrid keyword/neural retrieval for the memory store.

Combines FTS5 full-text search with neural embedding similarity.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

logger = logging.getLogger(__name__)

from ._tokenizer import tokenize_query

try:
    import numpy as np
    from ._embedder import encode_query_cached, is_available as embedding_is_available
    _HAS_EMBEDDER_MODULE = True
except ImportError:
    _HAS_EMBEDDER_MODULE = False


class FactRetriever:
    """FTS5 + neural embedding hybrid fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.40,
        ruri_weight: float = 0.60,
        ruri_keep_alive: int = -1,  # -1=always loaded, 0=unload after search, N=keep N seconds
        require_embeddings: bool = True,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self._ruri_keep_alive = ruri_keep_alive
        self.require_embeddings = require_embeddings

        if ruri_weight > 0 and require_embeddings and not self._embedding_available():
            raise RuntimeError(
                "Embedding model is required for vecmemori retrieval. Install vecmemori "
                "and download/configure a local SentenceTransformer model."
            )

        # Explicit FTS-only mode is available for tests and diagnostics only.
        if ruri_weight > 0 and not require_embeddings and not self._embedding_available():
            fts_weight += ruri_weight
            ruri_weight = 0.0

        self.fts_weight = fts_weight
        self.ruri_weight = ruri_weight
        self._query_embedding = None

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → neural rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Neural similarity: neural embedding cosine similarity
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        self._query_embedding = None
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            if self.ruri_weight > 0 and self._embedding_available():
                candidates = self._all_facts(category, min_trust, limit * 3)
            if not candidates:
                return []

        query_tokens_set = set(query.lower().split()) if query else set()
        scored = []

        for fact in candidates:
            fts_score = fact.get("fts_rank", 0.0)

            # Neural cosine similarity. Missing embeddings are not treated as
            # neutral relevance; they are ignored for the neural component.
            embedding_sim = 0.0
            if self.ruri_weight > 0 and fact.get("ruri_embedding"):
                query_emb = self._get_query_embedding(query)
                fact_emb = np.frombuffer(fact["ruri_embedding"], dtype=np.float32)
                if query_emb.shape == fact_emb.shape:
                    embedding_sim = float(np.dot(query_emb, fact_emb))

            relevance = self.fts_weight * fts_score + self.ruri_weight * embedding_sim
            score = relevance * fact["trust_score"]

            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "search q=%r fact#%d: fts=%.3f embedding=%.3f trust=%.2f → %.4f",
                    query[:40], fact["fact_id"],
                    fts_score, embedding_sim,
                    fact["trust_score"], score,
                )
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        for fact in results:
            fact.pop("ruri_embedding", None)

        if self._ruri_keep_alive == 0 and _HAS_EMBEDDER_MODULE:
            from ._embedder import unload_model
            unload_model()
        return results

    def _embedding_available(self) -> bool:
        return _HAS_EMBEDDER_MODULE and embedding_is_available()

    def _get_query_embedding(self, query: str) -> "np.ndarray":
        """Compute and cache query embedding for the current search."""
        if not _HAS_EMBEDDER_MODULE:
            raise RuntimeError("Embedding dependencies are not installed")
        if self._query_embedding is None:
            vec = encode_query_cached(query)
            if vec is None:
                raise RuntimeError("Embedding model failed to encode query")
            self._query_embedding = vec
        return self._query_embedding

    def _all_facts(
        self,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Fallback: get all facts when FTS5 returns nothing.

        Uses a generous multiplier to ensure neural search has enough candidates.
        """
        conn = self.store._conn
        where = ["f.trust_score >= ?"]
        params: list = [min_trust]
        if category:
            where.append("f.category = ?")
            params.append(category)
        effective_limit = max(limit, 200)
        params.append(effective_limit)
        rows = conn.execute(
            f"SELECT f.*, 0.5 as fts_rank FROM facts f WHERE {' AND '.join(where)} ORDER BY f.fact_id DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        fts_query = tokenize_query(query)
        params.append(fts_query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []

        if not rows:
            return []

        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank
            results.append(fact)

        return results

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
