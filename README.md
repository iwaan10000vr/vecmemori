<h1 align="center">vecmemori</h1>
<p align="center">
  <em>Local, private fact memory for AI agents — SQLite + FTS5 + neural embeddings</em>
  <br>
  <em>AIエージェントのためのローカル記憶エンジン</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
  <a href="#usage">Usage</a> •
  <a href="#model-swap">Model Swap</a> •
  <a href="#configuration">Configuration</a>
  <br>
  <a href="#日本語">日本語</a>
</p>

---

**vecmemori** is a local, private, persistent fact memory for AI agents. It stores short facts (preferences, decisions, project details) in SQLite and retrieves them using a **dual-strategy hybrid search** — combining full-text search (FTS5) with neural embedding similarity.

Unlike cloud-based memory services, vecmemori runs entirely on your machine. No data ever leaves your device (unless you choose to use a remote embedding model).

---

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

## Installation

```bash
pip install vecmemori                        # core (numpy only)
pip install vecmemori[embed]                 # with neural embeddings
pip install vecmemori[ja]                    # with Japanese FTS5 (fugashi + unidic-lite)
pip install vecmemori[all]                   # everything
pip install vecmemori[hermes]                # with Hermes Agent plugin (includes embed)
```

### Japanese FTS5 support

vecmemori uses [fugashi](https://github.com/polm/fugashi) (MeCab) to tokenize Japanese text before indexing it in FTS5. This enables proper keyword search for Japanese queries — searching `"ダークモード"` finds facts containing `"ダーク"` or `"モード"` individually.

Without `[ja]`, FTS5 falls back to SQLite's built-in unicode61 tokenizer, which does not split Japanese text. Neural embedding search still works for semantic matching.

```bash
# Verify Japanese tokenizer is active
python -c "from vecmemori._tokenizer import has_tokenizer; print('Japanese FTS5:', has_tokenizer())"
```

## Features

- **Dual-strategy hybrid search** — FTS5 + neural embeddings (weighted: 0.40 / 0.60)
- **🇯🇵 Japanese FTS5** — fugashi (MeCab) tokenizer for Japanese full-text search (install with `vecmemori[ja]`)
- **Entity resolution** — auto-extracts entities from fact content
- **Trust scoring** — asymmetric feedback (helpful: +0.05, unhelpful: -0.10)
- **Temporal decay** — optionally decay older facts
- **Graceful degradation** — works with numpy only; embeddings and tokenizer optional
- **Configurable embeddings** — swap any SentenceTransformer model
- **Retrieval Planner** — optional LLM-driven multi-query search for deeper recall
- **Algebraic retrieval** — probe (entity), reason (multi-entity AND), contradict (conflict detection)
- **Hermes Agent integration** — plug-and-play memory provider with auto-prefetch per turn
- **Auto-extraction** — LLM extracts durable facts from conversation on session end

## Usage

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore()
retriever = FactRetriever(store=store)

store.add_fact("GPU: RTX 5060 Ti 16GB", category="tool", tags="hardware,gpu")
store.add_fact("User prefers concise CLI output", category="user_pref")

# Hybrid search (FTS5 + neural embeddings)
results = retriever.search("gpu memory", limit=5)

# Entity probe
results = retriever.probe("RTX 5060")

# Multi-entity reasoning
results = retriever.reason(["gpu", "deep learning"])

# Contradiction detection
results = retriever.contradict()

# Feedback
store.record_feedback(fact_id=42, helpful=True)
```

## How It Works with Hermes Agent

When used as a Hermes Agent memory provider (`vecmemori[hermes]`), vecmemori integrates into every stage of the conversation lifecycle:

### Memory Recall (Prefetch) — Every Message

On every user message, vecmemori automatically searches the fact store and injects relevant facts into the system prompt:

```
User sends a message
    │
    ▼
vecmemori.prefetch(message)
    │
    ├─► FTS5 + neural embedding search
    ├─► Top-N facts (default: 5) selected
    ├─► Injected as "## Vecmemori Memory" section
    └─► Model sees relevant background facts
        before generating a response
```

This happens silently — no tool call is needed. The model simply "knows" relevant facts from previous sessions.

### Memory Storage (fact_store) — On Demand

The model can explicitly save facts using the `fact_store` tool:

```python
fact_store(action="add", content="User prefers Rust for systems programming")
```

Key tool actions:
- `add` — save a new fact (auto-deduplicates by content)
- `search` — keyword/semantic search
- `probe` — entity-centric recall
- `reason` — find facts connected to multiple entities
- `contradict` — find contradictory facts
- `update` / `remove` / `list` — CRUD operations

### Memory Tool Mirroring

When the model uses Hermes' built-in `memory` tool, vecmemori automatically mirrors the write as a structured fact:

```
memory(action="add", target="memory", content="...")
    │
    ├─► Built-in: saved to MEMORY.md (always active)
    └─► vecmemori mirror: saved as a fact (category: user_pref or general)
```

Facts accumulate even without explicit `fact_store` calls.

### Auto-Extraction — On Session End

When a session ends (CLI exit, `/reset`, timeout), vecmemori sends the last ~40 messages to an LLM which extracts durable facts:

```
Session ends
    │
    ▼
vecmemori.on_session_end(messages)
    │
    ├─► LLM receives conversation + extraction prompt
    ├─► LLM returns JSON: [{content, category, tags}, ...]
    └─► Each fact is saved via MemoryStore.add_fact()
```

The extraction prompt is in Japanese (optimized for Japanese-speaking environments) and targets:
- User preferences and habits
- Decisions made
- Project requirements and progress
- Tool and configuration choices

Enable with `auto_extract: true` in config (default: true).

### Retrieval Planner — Optional Enhancement

When enabled (`retrieval_planner: true`), vecmemori goes beyond single-query search. On each turn, it uses an LLM to generate multiple search questions from the conversation context, fans out retrieval across all of them, and merges results:

```
User message
    │
    ▼
LLM generates 3-6 search questions
    │
    ├─► "What does the user think about X?"
    ├─► "What constraints were set about Y?"
    └─► "What past decisions relate to Z?"
           │
           ▼
    Each question → separate search query
           │
           ▼
    Results merged, deduplicated, scored
           │
           ▼
    Top candidates injected into context
```

This helps discover facts that the current message doesn't directly mention.

### Summary Table

| Trigger | What happens | Config |
|---------|-------------|--------|
| Every user message | Search facts → inject top N | `prefetch_limit` (default: 5) |
| Model calls fact_store | Save/update/delete facts | Always available |
| Model calls memory tool | Auto-mirror as fact | Always active |
| Session ends | LLM extracts facts | `auto_extract: true` |
| Each turn (planner) | Multi-query LLM search | `retrieval_planner: false` |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fts_weight` | 0.40 | Keyword precision (BM25) |
| `ruri_weight` | 0.60 | Semantic similarity (neural embedding) |
| `db_path` | `memory.db` | SQLite path |
| `default_trust` | 0.5 | Initial trust score for new facts |
| `prefetch_limit` | 5 | Facts injected per turn |
| `auto_extract` | true | Auto-extract on session end |
| `retrieval_planner` | false | LLM-driven multi-query search |

When numpy or embedding model is unavailable, weights are redistributed automatically (fallback to single-strategy search).

## Model Swap

```python
from vecmemori._embedder import set_config, set_model_path

# English model (MiniLM)
set_config(dimension=384, query_prefix="", doc_prefix="")
set_model_path("/path/to/all-MiniLM-L6-v2")
```

When swapping models, regenerate existing embeddings:
```python
from vecmemori import MemoryStore
store = MemoryStore()
store.rebuild_all_vectors()
```

### Recommended Models

| Model | Language | Dim | Size |
|-------|----------|-----|------|
| cl-nagoya/ruri-v3-310m | 🇯🇵 Japanese | 768 | 1.2 GB |
| all-MiniLM-L6-v2 | 🇬🇧 English | 384 | 80 MB |
| BAAI/bge-large-en-v1.5 | 🇬🇧 English | 1024 | 1.3 GB |
| intfloat/multilingual-e5-large | 🌐 Multilingual | 1024 | 2.1 GB |

## Architecture

vecmemori combines **2 orthogonal similarity signals** into a final score:

```
User Query
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 (SQLite BM25)    │  weight=0.40 — keyword match
│ 2. Neural cosine sim     │  weight=0.60 — semantic similarity
└──────────┬───────────────┘
           ▼
    Relevance × Trust × Decay
           ▼
    Sorted results (top N)
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments / 謝辞

vecmemori stands on the shoulders of several open source projects, research works, and a prior plugin. We acknowledge and thank their creators.

### Hermes Agent

vecmemori was originally designed as a memory provider for **[Hermes Agent](https://github.com/nousresearch/hermes-agent)** by **Nous Research**. The `vecmemori/hermes/` adapter module is a thin compatibility layer that allows vecmemori to plug into Hermes Agent's memory provider system. We thank Nous Research for creating an extensible agent platform.

### The Original Holographic Plugin

vecmemori began as a fork and rewrite of the `holographic` memory plugin bundled with [Hermes Agent](https://github.com/nousresearch/hermes-agent). The original plugin provided hybrid FTS5 + HRR + Jaccard search with entity resolution and trust scoring.

Since the fork, vecmemori has evolved significantly:
- HRR (Holographic Reduced Representation) and Jaccard similarity have been removed — the system now focuses on a streamlined FTS5 + neural embedding pipeline
- fugashi (MeCab) Japanese tokenization has been added for proper CJK full-text search
- The codebase has been extracted into a standalone pip-installable package
- A new public API (MemoryStore, FactRetriever) replaces the original plugin interface
- Robust model-agnostic embedding system with configurable dimension and prefixes

The original `holographic` plugin remains available in Hermes Agent under the MIT license. We thank its authors for the foundation.

### Japanese Tokenization (fugashi + MeCab)

Japanese text tokenization is powered by **[fugashi](https://github.com/polm/fugashi)** (MIT license), a Python wrapper for **[MeCab](https://taku910.github.io/mecab/)** (BSD license) by Taku Kudo and Nippon Telegraph and Telephone Corporation. The UniDic dictionary is distributed with permission from the National Institute for Japanese Language and Linguistics (NINJAL).

### Core Dependencies

| Library | License |
|---------|---------|
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [SQLite](https://sqlite.org/) | Public Domain |
| [sentence-transformers](https://www.sbert.net/) (optional) | Apache 2.0 |
| [fugashi](https://github.com/polm/fugashi) (optional) | MIT |
| [unidic-lite](https://github.com/polm/unidic-lite) (optional) | Public Domain |
| [ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m) (optional) | Apache 2.0 |

vecmemori itself is released under the **MIT License** — you are free to use, modify, and distribute it in any context, commercial or otherwise, with attribution appreciated but not required.

---

# 日本語

**vecmemori** は AI エージェントのためのローカル・プライベートな事実記憶エンジンです。ユーザーの好み、決定事項、プロジェクト情報などの短い事実を SQLite に保存し、**2系統のハイブリッド検索**（全文検索 + ニューラル埋め込み）で関連情報を取得します。

クラウド型記憶サービスと異なり、vecmemori は完全にあなたのマシン上で動作します。データが外部に送信されることはありません（リモートの埋め込みモデルを使う場合を除く）。

## インストール

```bash
pip install vecmemori                        # コア（numpyのみ）
pip install vecmemori[embed]                 # ニューラル埋め込み込み
pip install vecmemori[ja]                    # 日本語FTS5対応（fugashi + unidic-lite）
pip install vecmemori[all]                   # 全部入り
pip install vecmemori[hermes]                # Hermes Agent プラグイン込み（embed含む）
```

### 日本語FTS5対応

vecmemori は [fugashi](https://github.com/polm/fugashi)（MeCab）を使って日本語テキストを形態素解析し、FTS5 にインデックスします。`"ダークモード"` で検索すると `"ダーク"` や `"モード"` に分割されてヒットします。

`[ja]` なしの場合は SQLite 標準の unicode61 トークナイザーにフォールバックします（日本語は分割されません）。意味検索（ruri-v3）は別系統として独立して動作します。

## クイックスタート

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
store.add_fact("ユーザーはダークモードを好む", category="user_pref")
store.add_fact("プロジェクトは pytest を使っている", category="project")

retriever = FactRetriever(store=store)
results = retriever.search("テーマ設定", limit=5)
for r in results:
    print(f"[{r['trust_score']:.2f}] {r['content']}")
```

## 特徴

| 機能 | 説明 |
|------|------|
| **2系統ハイブリッド検索** | FTS5 + ニューラル埋め込み（重み: 0.40 / 0.60） |
| **🇯🇵 日本語FTS5** | fugashi（MeCab）による形態素解析で日本語全文検索を実現 |
| **エンティティ解決** | 事実から自動的に固有名詞を抽出・リンク |
| **信頼度スコア** | 非対称フィードバック（参考になった: +0.05、参考にならない: -0.10） |
| **時間減衰** | 古い事実を減衰させるオプション |
| **グレースフルデグラデーション** | numpy だけでも動作。埋め込みもトークナイザーもオプション |
| **モデル差し替え** | 任意の SentenceTransformer モデルに対応 |
| **Retrieval Planner** | LLM 駆動の複数クエリ検索で隠れた記憶も発見 |
| **代数検索** | エンティティ検索 / 複数エンティティAND検索 / 矛盾検出 |
| **Hermes Agent 連携** | プラグアンドプレイで毎ターン自動プレフェッチ |
| **自動抽出** | セッション終了時に LLM が会話から事実を抽出 |

## 設定

| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `fts_weight` | 0.40 | キーワード精度（BM25） |
| `ruri_weight` | 0.60 | 意味的類似度（ニューラル埋め込み） |
| `db_path` | `memory.db` | SQLite データベースパス |
| `default_trust` | 0.5 | 新規事実の初期信頼度 |
| `prefetch_limit` | 5 | 毎ターン注入する事実数 |
| `auto_extract` | true | セッション終了時の自動抽出 |
| `retrieval_planner` | false | LLM 駆動マルチクエリ検索 |

numpy や埋め込みモデルが利用できない場合、重みは自動的に再配分されます。

## 埋め込みモデルの差し替え

```python
from vecmemori._embedder import set_config, set_model_path

# 英語モデル（MiniLM）
set_config(dimension=384, query_prefix="", doc_prefix="")
set_model_path("/path/to/all-MiniLM-L6-v2")
```

モデル差し替え後、既存の埋め込みを再生成:
```python
from vecmemori import MemoryStore
store = MemoryStore()
store.rebuild_all_vectors()
```

### 推奨モデル

| モデル | 言語 | 次元 | サイズ |
|--------|------|------|--------|
| cl-nagoya/ruri-v3-310m | 🇯🇵 日本語 | 768 | 1.2 GB |
| all-MiniLM-L6-v2 | 🇬🇧 英語 | 384 | 80 MB |
| BAAI/bge-large-en-v1.5 | 🇬🇧 英語 | 1024 | 1.3 GB |
| intfloat/multilingual-e5-large | 🌐 多言語 | 1024 | 2.1 GB |

## アーキテクチャ

vecmemori は **2つの直交する類似度信号** を組み合わせて最終スコアを計算します:

```
ユーザークエリ
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 (SQLite BM25)    │  重み=0.40 — キーワード一致
│ 2. ニューラルcos類似度   │  重み=0.60 — 意味的近さ
└──────────┬───────────────┘
           ▼
    関連度 × 信頼度 × 時間減衰
           ▼
    ソート結果 (上位N件)
```

## ライセンス

MIT — [LICENSE](LICENSE) 参照。

## 謝辞 / Acknowledgments

vecmemori は以下のプロジェクトや研究の上に成り立っています。

### Hermes Agent

vecmemori は元々 **[Hermes Agent](https://github.com/nousresearch/hermes-agent)**（**Nous Research** 開発）のメモリプロバイダーとして設計されました。`vecmemori/hermes/` アダプターモジュールは、vecmemori を Hermes Agent のメモリプロバイダーシステムに接続するための薄い互換レイヤーです。拡張可能なエージェントプラットフォームを提供してくださった Nous Research に感謝します。

### オリジナル Holographic プラグイン

vecmemori は Hermes Agent にバンドルされていた `holographic` メモリプラグインのフォークとして始まり、その後大幅に書き換えられました。オリジナルのプラグインは FTS5 + HRR + Jaccard のハイブリッド検索、エンティティ解決、信頼度スコアを提供していました。

フォーク以降の主な変更:
- HRR（Holographic Reduced Representation）と Jaccard 類似度を削除 — FTS5 + ニューラル埋め込みに特化
- fugashi（MeCab）による日本語トークナイゼーションを追加
- スタンドアロンの pip インストール可能なパッケージとして独立
- 新しい公開 API（MemoryStore, FactRetriever）を提供
- 次元・プレフィックス設定可能なモデル非依存の埋め込みシステム

オリジナルの `holographic` プラグインは Hermes Agent で MIT ライセンスのまま利用可能です。

### 日本語トークナイゼーション（fugashi + MeCab）

日本語テキストのトークナイズは **[fugashi](https://github.com/polm/fugashi)**（MIT ライセンス）を使用しています。fugashi は **[MeCab](https://taku910.github.io/mecab/)**（BSD ライセンス、Taku Kudo および日本電信電話株式会社）の Python ラッパーです。UniDic 辞書は国立国語研究所の許可を得て配布されています。

### コア依存ライブラリ

| ライブラリ | ライセンス |
|-----------|-----------|
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [SQLite](https://sqlite.org/) | Public Domain |
| [sentence-transformers](https://www.sbert.net/)（オプション） | Apache 2.0 |
| [fugashi](https://github.com/polm/fugashi)（オプション） | MIT |
| [unidic-lite](https://github.com/polm/unidic-lite)（オプション） | Public Domain |
| [ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)（オプション） | Apache 2.0 |

vecmemori 自体は **MIT ライセンス** でリリースされています。商用・非商用を問わず、自由に使用・改変・再配布できます。 attribution は歓迎しますが必須ではありません。
