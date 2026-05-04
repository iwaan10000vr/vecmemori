<h1 align="center">vecmemori</h1>
<p align="center">
  <em>Hybrid memory engine for AI agents — SQLite + FTS5 + HRR algebra + neural embeddings.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#model-swap">Model Swap</a> •
  <a href="#why-vecmemori">Why vecmemori?</a> •
  <a href="#faq">FAQ</a>
</p>

---

**vecmemori** is a local, private, persistent fact memory for AI agents. It stores short facts (preferences, decisions, project details) in SQLite and retrieves them using a **4-strategy hybrid search** — combining full-text search, token overlap, symbolic vector algebra, and optional neural embeddings.

Unlike cloud-based memory services, vecmemori runs entirely on your machine. No data ever leaves your device (unless you choose to use a remote embedding model).

## Quick Start

```python
from vecmemori import MemoryStore, FactRetriever

# Create a local fact store
store = MemoryStore(db_path="memory.db")

# Save facts
store.add_fact("User prefers dark mode in all applications", category="user_pref")
store.add_fact("Project uses pytest for testing", category="project")

# Search with hybrid retrieval
retriever = FactRetriever(store=store)
results = retriever.search("testing preferences", limit=5)
for r in results:
    print(f"[{r['trust_score']:.2f}] {r['content']}")
```

## Features

- **4-strategy hybrid search** — FTS5 full-text + Jaccard token overlap + HRR symbolic vectors + neural embedding cosine similarity (weighted: 0.30 / 0.10 / 0.20 / 0.40)
- **Entity resolution** — auto-extracts entities from fact content, links them for relational queries
- **Trust scoring** — asymmetric feedback (helpful: +0.05, unhelpful: -0.10), facts float up or sink down
- **Algebraic retrieval** — probe (entity-centric), reason (multi-entity AND), related (structural adjacency), contradict (automated hygiene)
- **Temporal decay** — optionally decay older facts by half-life in days
- **Graceful degradation** — works with just numpy; neural embeddings are optional
- **Configurable embeddings** — swap in any SentenceTransformer-compatible model

## Installation

```bash
# Core (FTS5 + Jaccard + HRR — works with numpy only)
pip install vecmemori

# With neural embeddings
pip install vecmemori[embed]

# With Hermes Agent plugin support
pip install vecmemori[hermes]
```

### Model Download

vecmemori works without any embedding model (HRR + FTS5 only). For semantic search, download ruri-v3-310m:

```bash
# huggingface-cli (recommended)
pip install huggingface-hub
bash scripts/download_model.sh

# Or git-lfs
apt install git-lfs
bash scripts/download_model.sh
```

### Hermes Agent Plugin

After installing `vecmemori[hermes]`:

```bash
# Symlink the adapter
ln -s $(python -c "import vecmemori; print(vecmemori.__path__[0])")/hermes \
       ~/.hermes/plugins/vecmemori

# Configure
hermes config set memory.provider vecmemori
hermes memory setup
```

## Usage

### Core Library

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore()
retriever = FactRetriever(store=store)

# Add facts
store.add_fact("GPU: RTX 5060 Ti 16GB", category="tool", tags="hardware,gpu")
store.add_fact("User prefers concise CLI output", category="user_pref")

# Hybrid search
results = retriever.search("gpu memory", limit=5)

# Entity probe — finds facts about a specific entity
results = retriever.probe("RTX 5060")

# Multi-entity reasoning — finds facts connected to ALL entities simultaneously
results = retriever.reason(["gpu", "deep learning"])

# Structural adjacency
results = retriever.related("NVIDIA")

# Contradiction detection
contradictions = retriever.contradict()

# Feedback loop (trains trust scores)
feedback = store.record_feedback(fact_id=42, helpful=True)
```

## Configuration

### MemoryStore

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `memory.db` | Path to SQLite database |
| `default_trust` | `0.5` | Initial trust score for new facts [0.0–1.0] |
| `hrr_dim` | `1024` | HRR vector dimension (higher = more capacity) |

### FactRetriever Weights

The four strategies combine into a single relevance score:

```
score = (0.30 × fts_score
         + 0.10 × jaccard_score
         + 0.20 × hrr_similarity
         + 0.40 × cosine_similarity)
        × trust_score
        × temporal_decay
```

| Parameter | Default | Signal |
|-----------|---------|--------|
| `fts_weight` | 0.30 | Keyword precision (BM25) |
| `jaccard_weight` | 0.10 | Lexical diversity (token overlap) |
| `hrr_weight` | 0.20 | Symbolic structure (phase vectors) |
| `ruri_weight` | 0.40 | Semantic proximity (neural cosine) |

If numpy is unavailable, weights auto-redistribute. If embeddings are unavailable, `ruri_weight` is redistributed to FTS5 and Jaccard.

### Hermes Plugin (config.yaml)

```yaml
memory:
  provider: vecmemori

plugins:
  vecmemori:
    auto_extract: true              # Auto-extract facts at session end
    default_trust: 0.5
    embedding_weight: 0.40           # Neural embedding weight
    embedding_model: ~/.hermes/models/ruri-v3-310m
    retrieval_planner: false         # LLM-driven multi-query search
    prefetch_limit: 5                # Facts injected each turn
```

## Model Swap

vecmemori ships with ruri-v3-310m (Japanese, 768-dim) as default. Any SentenceTransformer-compatible model works:

```python
from vecmemori._embedder import set_config, set_model_path

# English model (no asymmetric prefix)
set_config(dimension=384, query_prefix="", doc_prefix="")
set_model_path("/path/to/all-MiniLM-L6-v2")

# BGE with custom prefix
set_config(dimension=1024, query_prefix="Represent this sentence: ", doc_prefix="")
set_model_path("/path/to/bge-large-en-v1.5")
```

When swapping models, existing embeddings in the database become incompatible.
To regenerate all embeddings:

```python
from vecmemori import MemoryStore
store = MemoryStore()
count = store.rebuild_all_vectors()
```

### Recommended Models

| Model | Lang | Dim | Size |
|-------|------|-----|------|
| cl-nagoya/ruri-v3-310m | 🇯🇵 | 768 | 1.2 GB |
| all-MiniLM-L6-v2 | 🇬🇧 | 384 | 80 MB |
| BAAI/bge-large-en-v1.5 | 🇬🇧 | 1024 | 1.3 GB |
| intfloat/multilingual-e5-large | 🌐 | 1024 | 2.1 GB |

## Why vecmemori?

| Feature | vecmemori | Cloud memory services |
|---------|-----------|----------------------|
| Data stays local | ✅ Always | ❌ Uploads to API |
| Cost | $0 | Per-token pricing |
| Offline | ✅ Full offline | ❌ Needs internet |
| Embedding model | Swap any | Vendor-locked |
| Symbolic reasoning | ✅ HRR algebra | ❌ Dense vectors only |
| Auto-extraction | ✅ LLM on session end | ✅ |
| Contradiction detection | ✅ Automated | ❌ |

## Architecture

vecmemori combines **four orthogonal similarity signals** into a single relevance score:

```
User query
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 (SQLite BM25)    │  weight=0.30  — keyword precision
│ 2. Jaccard token overlap │  weight=0.10  — lexical diversity
│ 3. HRR phase vectors     │  weight=0.20  — symbolic structure
│ 4. Neural cosine sim.    │  weight=0.40  — semantic similarity
└──────────┬───────────────┘
           ▼
    relevance × trust_score × temporal_decay
           ▼
    sorted results (top-N)
```

### Project Structure

```
vecmemori/
└── src/vecmemori/
    ├── store.py         SQLite fact store (CRUD, FTS5, entity resolution)
    ├── retrieval.py     4-strategy hybrid search pipeline
    ├── hrr.py           HRR vector symbolic algebra (Plate 1995)
    ├── _embedder.py     Embedding model wrapper
    └── hermes/          Optional Hermes Agent adapter
```

## FAQ

**Q: Do I need a GPU?**  
No. vecmemori works on CPU. Neural embeddings benefit from CUDA but are optional.

**Q: Can I use it without an embedding model?**  
Yes. Core functionality (FTS5 + Jaccard + HRR) works with numpy only.

**Q: How do I migrate from the old Hermes bundled holographic plugin?**  
The SQLite schema is identical — your existing `memory_store.db` works directly with vecmemori. No data transformation is needed.

**Q: How much memory does it use?**  
Without embeddings: negligible (~50 MB RSS). With ruri-v3-310m: ~1.5 GB VRAM/RAM.

**Q: Can I use it with agents other than Hermes?**  
Yes. vecmemori is a standalone Python library. Only the `hermes/` adapter requires Hermes.

## License

MIT — see [LICENSE](LICENSE).

The HRR implementation (`hrr.py`) implements the algorithm described in Plate (1995) — *Holographic Reduced Representations*. The code is original, MIT-licensed, and does not incorporate any third-party HRR library.
