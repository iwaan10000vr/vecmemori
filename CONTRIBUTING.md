# Contributing to vecmemori

Thanks for your interest! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/iwaan/vecmemori
cd vecmemori
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Structure

- `src/vecmemori/` — Core library (pure Python + numpy)
  - `store.py` — SQLite fact store
  - `retrieval.py` — Dual-strategy search pipeline (FTS5 + neural embeddings)
  - `_embedder.py` — Embedding model wrapper
- `src/vecmemori/hermes/` — Optional Hermes Agent adapter

## Pull Request Guidelines

1. Keep the core library free of framework dependencies
2. Add tests for new functionality
3. Update the model-swap guide if you change the embedding interface

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
