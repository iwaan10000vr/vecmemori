<h1 align="center">vecmemori</h1>
<p align="center">
  <em>Hybrid memory engine for AI agents — SQLite + FTS5 + HRR algebra + neural embeddings</em>
  <br>
  <em>AIエージェントのためのハイブリッド記憶エンジン</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#model-swap">Model Swap</a>
  <br>
  <a href="#日本語">日本語</a>
</p>

---

**vecmemori** is a local, private, persistent fact memory for AI agents. It stores short facts (preferences, decisions, project details) in SQLite and retrieves them using a **4-strategy hybrid search** — combining full-text search, token overlap, symbolic vector algebra, and optional neural embeddings.

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

## Features

- **4-strategy hybrid search** — FTS5 + Jaccard + HRR + neural embeddings (weighted: 0.30 / 0.10 / 0.20 / 0.40)
- **Entity resolution** — auto-extracts entities from fact content
- **Trust scoring** — asymmetric feedback (helpful: +0.05, unhelpful: -0.10)
- **Algebraic retrieval** — probe, reason, related, contradict
- **Temporal decay** — optionally decay older facts
- **Graceful degradation** — works with numpy only; embeddings optional
- **Configurable embeddings** — swap any SentenceTransformer model

## Installation

```bash
pip install vecmemori            # core (numpy only)
pip install vecmemori[embed]     # with neural embeddings
pip install vecmemori[hermes]    # with Hermes Agent plugin
```

## Usage

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore()
retriever = FactRetriever(store=store)

store.add_fact("GPU: RTX 5060 Ti 16GB", category="tool", tags="hardware,gpu")
store.add_fact("User prefers concise CLI output", category="user_pref")

# Hybrid search
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

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fts_weight` | 0.30 | Keyword precision (BM25) |
| `jaccard_weight` | 0.10 | Lexical diversity |
| `hrr_weight` | 0.20 | Symbolic structure |
| `ruri_weight` | 0.40 | Semantic similarity |
| `db_path` | `memory.db` | SQLite path |
| `default_trust` | 0.5 | Initial trust score |
| `hrr_dim` | 1024 | HRR vector dimension |

Weights auto-redistribute when numpy or embeddings are unavailable.

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

## License

MIT — see [LICENSE](LICENSE).

---

# 日本語

**vecmemori** は AI エージェントのためのローカル・プライベートな事実記憶エンジンです。ユーザーの好み、決定事項、プロジェクト情報などの短い事実を SQLite に保存し、**4系統のハイブリッド検索**（全文検索 + トークン重複 + 記号ベクトル代数 + ニューラル埋め込み）で関連情報を取得します。

クラウド型記憶サービスと異なり、vecmemori は完全にあなたのマシン上で動作します。データが外部に送信されることはありません（リモートの埋め込みモデルを使う場合を除く）。

## インストール

```bash
pip install vecmemori            # コア（numpyのみ）
pip install vecmemori[embed]     # ニューラル埋め込み込み
pip install vecmemori[hermes]    # Hermes Agent プラグイン込み
```

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
| **4系統ハイブリッド検索** | FTS5 + Jaccard + HRR + 埋め込み（重み: 0.30 / 0.10 / 0.20 / 0.40） |
| **エンティティ解決** | 事実から自動的に固有名詞を抽出・リンク |
| **信頼度スコア** | 非対称フィードバック（参考になった: +0.05、参考にならない: -0.10） |
| **代数検索** | エンティティ検索 / 複数エンティティAND検索 / 構造的隣接発見 / 矛盾検出 |
| **グレースフルデグラデーション** | numpy だけでも動作。埋め込みはオプション |
| **モデル差し替え** | 任意の SentenceTransformer モデルに対応 |

## 主なユースケース

vecmemori は以下のようなシナリオで特に威力を発揮します:

- **AI アシスタントの長期記憶** — ユーザーの好みや設定を永続的に保存し、セッションを越えて参照
- **プロジェクト知識の蓄積** — 技術選定や決定事項を事実として記録
- **プライバシー重視の環境** — すべてローカルで完結。外部API不要
- **日本語特化** — ruri-v3 による高精度な日本語意味検索に対応

## 設定

| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `fts_weight` | 0.30 | キーワード精度（BM25） |
| `jaccard_weight` | 0.10 | 語彙の多様性 |
| `hrr_weight` | 0.20 | 記号的構造 |
| `ruri_weight` | 0.40 | 意味的類似度 |
| `db_path` | `memory.db` | SQLite データベースパス |
| `default_trust` | 0.5 | 新規事実の初期信頼度 |
| `hrr_dim` | 1024 | HRR ベクトル次元数 |

numpy や埋め込みモデルが利用できない場合、重みは自動的に再配分されます。

## Hermes Agent プラグイン

vecmemori は Hermes Agent のメモリプロバイダーとして動作します:

```bash
pip install vecmemori[hermes]

# アダプターをプラグインディレクトリに設置
ln -s $(python -c "import vecmemori; print(vecmemori.__path__[0])")/hermes \
       ~/.hermes/plugins/vecmemori

# 設定
hermes config set memory.provider vecmemori
hermes memory setup
```

### 旧 holographic プラグインからの移行

`~/.hermes/user-docs/vecmemori/migration.md` に完全な移行手順があります。SQLite スキーマは同一のため、データ変換は不要です。

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

vecmemori は **4つの直交する類似度信号** を組み合わせて最終スコアを計算します:

```
ユーザークエリ
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 (SQLite BM25)    │  重み=0.30 — キーワード一致
│ 2. Jaccard 係数          │  重み=0.10 — 語彙の重なり
│ 3. HRR 位相ベクトル      │  重み=0.20 — 記号的構造
│ 4. ニューラルcos類似度   │  重み=0.40 — 意味的近さ
└──────────┬───────────────┘
           ▼
    関連度 × 信頼度 × 時間減衰
           ▼
    ソート結果 (上位N件)
```

## ライセンス

MIT — [LICENSE](LICENSE) 参照。

HRR 実装（`hrr.py`）は Plate (1995) — *Holographic Reduced Representations* のアルゴリズムを実装したものです。コードはオリジナル、MITライセンスで提供され、第三者の HRR ライブラリは含みません。
