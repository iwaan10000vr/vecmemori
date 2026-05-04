"""Tests for the MemoryStore (SQLite-backed fact store)."""

import pytest
from vecmemori import MemoryStore


class TestMemoryStoreBasic:
    def test_add_fact(self, store):
        fid = store.add_fact("Test fact", category="general", tags="test")
        assert isinstance(fid, int)
        assert fid > 0

    def test_add_fact_empty_raises(self, store):
        with pytest.raises(ValueError, match="content must not be empty"):
            store.add_fact("")

    def test_add_fact_deduplicate(self, store):
        fid1 = store.add_fact("Unique fact", category="general")
        fid2 = store.add_fact("Unique fact", category="general")
        assert fid1 == fid2  # same content → same id

    def test_search_facts(self, populated_store):
        results = populated_store.search_facts("dark mode")
        assert len(results) >= 1
        assert "dark mode" in results[0]["content"].lower()

    def test_search_facts_no_match(self, populated_store):
        results = populated_store.search_facts("zzzznonexistent")
        assert len(results) == 0

    def test_search_facts_category_filter(self, populated_store):
        results = populated_store.search_facts("test", category="project")
        for r in results:
            assert r["category"] == "project"

    def test_search_facts_trust_threshold(self, store):
        store.add_fact("Low trust fact", category="general")
        # Should be findable at min_trust=0
        results = store.search_facts("Low trust", min_trust=0.0)
        assert len(results) >= 1

    def test_update_fact_content(self, store):
        fid = store.add_fact("Old content", category="general")
        updated = store.update_fact(fid, content="New content")
        assert updated is True
        results = store.search_facts("New content")
        assert len(results) >= 1

    def test_update_fact_trust(self, store):
        fid = store.add_fact("Trust test fact", category="general")
        updated = store.update_fact(fid, trust_delta=0.2)
        assert updated is True

    def test_update_fact_nonexistent(self, store):
        updated = store.update_fact(99999, content="ghost")
        assert updated is False

    def test_remove_fact(self, store):
        fid = store.add_fact("To be removed", category="general")
        removed = store.remove_fact(fid)
        assert removed is True
        results = store.search_facts("To be removed")
        assert len(results) == 0

    def test_remove_fact_nonexistent(self, store):
        removed = store.remove_fact(99999)
        assert removed is False

    def test_list_facts(self, populated_store):
        facts = populated_store.list_facts()
        assert len(facts) >= 5

    def test_list_facts_category(self, populated_store):
        facts = populated_store.list_facts(category="tool")
        assert all(f["category"] == "tool" for f in facts)

    def test_list_facts_trust_filter(self, store):
        store.add_fact("High trust", category="general")
        facts = store.list_facts(min_trust=0.0)
        assert len(facts) >= 1


class TestMemoryStoreFeedback:
    def test_record_feedback_helpful(self, store):
        fid = store.add_fact("Feedback test", category="general")
        result = store.record_feedback(fid, helpful=True)
        assert result["fact_id"] == fid
        assert result["new_trust"] > result["old_trust"]

    def test_record_feedback_unhelpful(self, store):
        fid = store.add_fact("Feedback test 2", category="general")
        before = store._conn.execute(
            "SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()[0]
        result = store.record_feedback(fid, helpful=False)
        assert result["new_trust"] < result["old_trust"]

    def test_record_feedback_nonexistent(self, store):
        with pytest.raises(KeyError):
            store.record_feedback(99999, helpful=True)


class TestMemoryStoreEntities:
    def test_entity_extraction_capitalized(self, store):
        store.add_fact("John Doe prefers Vim", category="general")
        # Entity "John Doe" should be extracted
        row = store._conn.execute(
            "SELECT name FROM entities WHERE name LIKE ?", ("John Doe",)
        ).fetchone()
        assert row is not None, "Capitalized entity not extracted"

    def test_entity_extraction_quoted(self, store):
        store.add_fact('The "QuickBrownFox" is fast', category="general")
        row = store._conn.execute(
            "SELECT name FROM entities WHERE name LIKE ?", ("QuickBrownFox",)
        ).fetchone()
        assert row is not None, "Quoted entity not extracted"

    def test_entity_linking(self, store):
        fid = store.add_fact("Alice uses Arch Linux", category="tool")
        rows = store._conn.execute(
            """SELECT e.name FROM entities e
               JOIN fact_entities fe ON fe.entity_id = e.entity_id
               WHERE fe.fact_id = ?""",
            (fid,),
        ).fetchall()
        names = {r["name"].lower() for r in rows}
        assert "alice" in names or "arch linux" in names


def test_store_requires_embeddings_by_default(monkeypatch, db_path):
    """Default store mode should fail closed when embeddings are unavailable."""
    import vecmemori.store as store_module

    monkeypatch.setattr(store_module, "_HAS_EMBEDDER_MODULE", False)
    s = MemoryStore(db_path=db_path)
    try:
        with pytest.raises(RuntimeError, match="Embedding dependencies are required"):
            s.add_fact("Embedding required")
    finally:
        s.close()
