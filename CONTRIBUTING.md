# Contributing to vecmemori

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/iwaan10000vr/vecmemori
cd vecmemori
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ja]"
bash scripts/download_model.sh
```

## Running Tests

```bash
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
check-manifest
```

## Code Structure

- `src/vecmemori/` — Core library (Python + numpy + SentenceTransformer-compatible embeddings)
  - `store.py` — SQLite fact store
  - `retrieval.py` — Dual-strategy search pipeline (FTS5 + neural embeddings)
  - `_embedder.py` — Embedding model wrapper
- `src/vecmemori/hermes/` — Optional Hermes Agent adapter

## Pull Request Guidelines

1. Keep Hermes-specific code inside `src/vecmemori/hermes/`
2. Add tests for new functionality
3. Update both `README.md` and `README.ja.md` for user-facing changes
4. Update `THIRD_PARTY_NOTICES.md` when adding dependencies
5. Update the model-swap guide if you change the embedding interface

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
