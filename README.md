<h1 align="center">vecmemori</h1>
<p align="center">
  <em>Local fact memory for AI agents — SQLite + FTS5 + neural embeddings</em>
  <br>
  <a href="README.ja.md">日本語 README</a>
</p>

---

**vecmemori** is a local, persistent fact memory engine for AI agents. It stores short facts such as user preferences, project decisions, and environment details in SQLite, then retrieves relevant facts with a dual strategy:

- **FTS5 full-text search** for keyword precision
- **Neural embeddings** for semantic similarity

Embeddings are a core requirement. vecmemori is not intended to be only a keyword-search database.

## Privacy model

The core library stores data locally in SQLite and uses a local SentenceTransformer-compatible embedding model by default. Data does not leave your device when you use the core library with a local model.

Some integrations can intentionally send text to configured LLM providers:

- Hermes Agent `auto_extract` may send recent conversation messages to an LLM to extract durable facts.
- Hermes Agent `retrieval_planner` / planner injection may send conversation context to an LLM to generate better retrieval queries.
- A remote embedding provider, if you explicitly configure one outside the default local model flow, may receive text for embedding.

Do not enable those options unless the configured provider is acceptable for your privacy requirements.

## Installation

```bash
pip install vecmemori
pip install vecmemori[ja]       # Japanese FTS5 tokenization with fugashi + unidic-lite
pip install vecmemori[hermes]   # adapter dependencies for Hermes Agent environments
pip install vecmemori[all]      # Japanese + Hermes adapter dependencies
```

Install a local embedding model before writing/searching facts:

```bash
bash scripts/download_model.sh
```

Default model path:

```text
~/.cache/vecmemori/models/ruri-v3-310m
```

The default model is [`cl-nagoya/ruri-v3-310m`](https://huggingface.co/cl-nagoya/ruri-v3-310m), which is Apache-2.0 licensed and works well for Japanese.

## Quick Start

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
store.add_fact("User prefers dark mode in all applications", category="user_pref")
store.add_fact("Project uses pytest for testing", category="project")

retriever = FactRetriever(store=store)
results = retriever.search("testing preferences", limit=5)

for r in results:
    print(f"[{r['trust_score']:.2f}] {r['content']}")
```

For explicit tests or diagnostics without embeddings, pass `require_embeddings=False` to both `MemoryStore` and `FactRetriever`. This is not the recommended production mode.

## Japanese FTS5 support

SQLite's default `unicode61` tokenizer does not segment Japanese text well. Install the Japanese extra to pre-tokenize text with [fugashi](https://github.com/polm/fugashi) / MeCab:

```bash
pip install vecmemori[ja]
python -c "from vecmemori._tokenizer import has_tokenizer; print('Japanese FTS5:', has_tokenizer())"
```

Neural embedding search is still the main semantic recall path. Japanese FTS5 is an additional keyword-precision signal.

## Features

- **Dual-strategy search** — FTS5 + neural embeddings, weighted by default at 0.40 / 0.60
- **Required semantic embeddings** — local SentenceTransformer-compatible model
- **Japanese support** — ruri-v3 embeddings plus optional fugashi tokenization for FTS5
- **Entity extraction** — lightweight entity extraction from fact content
- **Trust scoring** — helpful/unhelpful feedback adjusts fact reliability
- **Temporal decay** — optional recency weighting
- **Model swapping** — configurable embedding dimension and query/document prefixes
- **Hermes Agent integration** — memory provider adapter, `fact_store` tool, prefetch, optional LLM planner/extraction

## Python API

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
fact_id = store.add_fact("GPU: RTX 5060 Ti 16GB", category="tool", tags="hardware,gpu")
store.record_feedback(fact_id=fact_id, helpful=True)

retriever = FactRetriever(store=store)
results = retriever.search("gpu memory", limit=5)

store.rebuild_all_embeddings()  # after changing embedding model/config
store.close()
```

`probe`, `reason`, and `contradict` are Hermes `fact_store` tool actions, not methods on the standalone `FactRetriever` Python API.

## Hermes Agent integration

`vecmemori[hermes]` installs the adapter dependencies (`PyYAML`, `httpx`) needed by the Hermes plugin module. It does **not** install Hermes Agent itself. Use this extra inside an environment where Hermes Agent is already installed/configured.

Lifecycle when used as a Hermes memory provider:

1. **Every user message:** prefetch searches facts and injects top matches into context.
2. **Tool call:** `fact_store` can add/search/update/remove facts explicitly.
3. **Built-in memory mirroring:** Hermes `memory` writes can be mirrored into vecmemori.
4. **Session end:** optional `auto_extract` can ask a configured LLM to extract durable facts.
5. **Planner:** optional `retrieval_planner` can ask a configured LLM to generate multiple retrieval queries.

Hermes-only tool actions:

- `add`, `search`, `probe`, `related`, `reason`, `contradict`, `update`, `remove`, `list`

The `reason`/`contradict` actions are pragmatic semantic recall helpers, not symbolic algebraic proof or guaranteed contradiction detection.

## Configuration

Common options:

- `db_path`: SQLite database path, default `memory.db` in standalone use
- `default_trust`: initial trust score, default `0.5`
- `fts_weight`: keyword score weight, default `0.40`
- `ruri_weight`: semantic embedding score weight, default `0.60` (legacy config key name)
- `prefetch_limit`: Hermes facts injected per turn, default `5`
- `auto_extract`: Hermes LLM extraction on session end, default depends on plugin config
- `retrieval_planner`: Hermes LLM multi-query retrieval, default `false`
- `embedding_model`: local embedding model path
- `embedding_trust_remote_code`: allow Hugging Face custom model code, default `false`

## Model Swap

```python
from vecmemori._embedder import set_config, set_model_path

set_config(dimension=384, query_prefix="", doc_prefix="")
set_model_path("/path/to/all-MiniLM-L6-v2")

from vecmemori import MemoryStore
store = MemoryStore("memory.db")
store.rebuild_all_embeddings()
```

Recommended examples:

- `cl-nagoya/ruri-v3-310m` — Japanese, 768 dim, ~1.2 GB
- `sentence-transformers/all-MiniLM-L6-v2` — English, 384 dim, small
- `BAAI/bge-large-en-v1.5` — English, 1024 dim
- `intfloat/multilingual-e5-large` — multilingual, 1024 dim

## Architecture

```text
User Query
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 / BM25           │  weight=0.40 — keyword match
│ 2. Neural cosine sim     │  weight=0.60 — semantic similarity
└──────────┬───────────────┘
           ▼
    Relevance × Trust × Decay
           ▼
    Sorted results
```

## Development

```bash
pip install -e ".[dev,ja]"
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
check-manifest
```

## License

MIT — see [LICENSE](LICENSE).

Third-party license information is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Attribution for the Hermes Agent / Holographic memory plugin origin is included in [NOTICE](NOTICE) and the third-party notices.

## Acknowledgments

vecmemori began as a fork/rewrite of the MIT-licensed `holographic` memory plugin bundled with [Hermes Agent](https://github.com/nousresearch/hermes-agent) by Nous Research. Since then it has been extracted into a standalone package and simplified around FTS5 + neural embedding retrieval.

Thanks to the maintainers of Hermes Agent, SQLite, NumPy, sentence-transformers, PyTorch, fugashi/MeCab, UniDic, and cl-nagoya's ruri models.
