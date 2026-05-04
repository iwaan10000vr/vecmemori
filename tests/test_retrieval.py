"""Tests for the FactRetriever (hybrid search pipeline).

Note: These tests run without ruri-v3 embeddings (not installed in test venv).
The retriever auto-redistributes ruri_weight to FTS5 when ruri
is unavailable, so search quality is maintained.
"""

import pytest
from vecmemori import FactRetriever


class TestFactRetrieverBasic:
    def test_search_returns_results(self, populated_retriever):
        results = populated_retriever.search("dark mode")
        assert len(results) >= 1
        assert "score" in results[0]

    def test_search_empty_query(self, retriever):
        results = retriever.search("")
        assert len(results) == 0

    def test_search_returns_scored(self, populated_retriever):
        results = populated_retriever.search("gpu")
        for r in results:
            assert isinstance(r["score"], float)
            assert r["score"] >= 0

    def test_search_respects_limit(self, populated_retriever):
        results = populated_retriever.search("pytest", limit=1)
        assert len(results) <= 1

    def test_search_respects_category(self, populated_retriever):
        results = populated_retriever.search("preference", category="user_pref")
        for r in results:
            assert r["category"] == "user_pref"

    def test_search_respects_min_trust(self, store, retriever):
        store.add_fact("Very low trust fact", category="general")
        results_low = retriever.search("low trust", min_trust=0.0)
        results_high = retriever.search("low trust", min_trust=0.9)
        assert len(results_low) >= 1
        assert len(results_high) == 0

    def test_search_returns_sorted(self, populated_retriever):
        results = populated_retriever.search("test")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_search_strips_binary(self, populated_retriever):
        """ruri_embedding must be stripped from output."""
        results = populated_retriever.search("dark")
        for r in results:
            assert "ruri_embedding" not in r, "ruri_embedding leaked"

    def test_search_cjk_fallback(self, populated_retriever):
        """CJK query that doesn't match FTS5 should still work via fallback."""
        results = populated_retriever.search("日本語のテスト")
        assert isinstance(results, list)

    def test_multiple_searches_different_queries(self, populated_retriever):
        """Query embedding cache must reset between searches."""
        r1 = populated_retriever.search("dark mode")
        r2 = populated_retriever.search("gpu")
        r3 = populated_retriever.search("pytest")
        contents_1 = {item["content"] for item in r1}
        contents_2 = {item["content"] for item in r2}
        assert contents_1 != contents_2 or len(r1) == 0 or len(r2) == 0

    def test_search_with_japanese_query(self, populated_retriever):
        """Japanese query should not crash even without fugashi in test env."""
        results = populated_retriever.search("日本語")
        assert isinstance(results, list)

    def test_temporal_decay(self, populated_retriever):
        """Temporal decay should not crash when enabled."""
        retriever = FactRetriever(store=populated_retriever.store, temporal_decay_half_life=30)
        results = retriever.search("dark mode")
        assert isinstance(results, list)
