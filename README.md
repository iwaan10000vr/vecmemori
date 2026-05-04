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
- **🇯🇵 Japanese FTS5** — fugashi (MeCab) tokenizer for Japanese full-text search (install with `vecmemori[ja]`)
- **Entity resolution** — auto-extracts entities from fact content
- **Trust scoring** — asymmetric feedback (helpful: +0.05, unhelpful: -0.10)
- **Algebraic retrieval** — probe, reason, related, contradict
- **Temporal decay** — optionally decay older facts
- **Graceful degradation** — works with numpy only; embeddings and tokenizer optional
- **Configurable embeddings** — swap any SentenceTransformer model

## Installation

```bash
pip install vecmemori               # core (numpy only)
pip install vecmemori[embed]        # with neural embeddings
pip install vecmemori[ja]           # with Japanese FTS5 (fugashi + unidic-lite)
pip install vecmemori[all]          # everything
pip install vecmemori[hermes]       # with Hermes Agent plugin (includes embed)
```

### Japanese FTS5 support

vecmemori uses [fugashi](https://github.com/polm/fugashi) (MeCab) to tokenize Japanese text before indexing it in FTS5. This enables proper keyword search for Japanese queries — searching `"ダークモード"` finds facts containing `"ダーク"` or `"モード"` individually.

Without `[ja]`, FTS5 falls back to SQLite's built-in unicode61 tokenizer, which does not split Japanese text. Neural embedding search (ruri-v3) still works for semantic matching.

```bash
# Verify Japanese tokenizer is active
python -c "from vecmemori._tokenizer import has_tokenizer; print('Japanese FTS5:', has_tokenizer())"
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
    ├─► FTS5 + Jaccard + HRR + neural search
    ├─► Top-N facts (default: 5) selected
    ├─► Injected as "## Vecmemori Memory" section
    └─► Model sees relevant background facts
        before generating a response
```

This happens silently — no tool call is needed. The model simply "knows" relevant facts from previous sessions. The `prefetch_limit` config option controls how many facts are injected per turn (default: 5).

### Memory Storage (fact_store) — On Demand

The model can explicitly save facts using the `fact_store` tool:

```python
# Called by the model automatically when it decides
# something is worth remembering
fact_store(action="add", content="User prefers Rust for systems programming")
```

Key tool actions:
- `add` — save a new fact (auto-deduplicates by content)
- `search` — keyword/ semantic search
- `probe` — entity-centric recall
- `reason` — find facts connected to multiple entities
- `contradict` — find contradictory facts
- `update` / `remove` / `list` — CRUD

### Memory Tool Mirroring — On Built-in Memory Write

When the model uses Hermes' built-in `memory` tool (which writes to MEMORY.md / USER.md), vecmemori automatically mirrors the write as a structured fact:

```
memory(action="add", target="memory", content="...")
    │
    ├─► Built-in: saved to MEMORY.md (always active)
    └─► vecmemori mirror: saved as a fact (category: user_pref or general)
```

This means facts accumulate even without explicit `fact_store` calls.

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

The extraction prompt is in Japanese (optimized for the user's environment) and targets:
- User preferences and habits
- Decisions made
- Project requirements and progress
- Tool and configuration choices

Enable with `auto_extract: true` in config (default: true).

### Retrieval Planner — Optional Enhancement

When enabled (`retrieval_planner: true`), vecmemori goes beyond single-query search. On each turn, it uses an LLM to generate multiple search queries from the conversation context, fans out retrieval across all of them, and merges results:

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
| `fts_weight` | 0.30 | Keyword precision (BM25) |
| `jaccard_weight` | 0.10 | Lexical diversity |
| `hrr_weight` | 0.20 | Symbolic structure |
| `ruri_weight` | 0.40 | Semantic similarity |
| `db_path` | `memory.db` | SQLite path |
| `default_trust` | 0.5 | Initial trust score |
| `hrr_dim` | 1024 | HRR vector dimension |
| `prefetch_limit` | 5 | Facts injected per turn |
| `auto_extract` | true | Auto-extract on session end |
| `retrieval_planner` | false | LLM-driven multi-query search |

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
pip install vecmemori               # コア（numpyのみ）
pip install vecmemori[embed]        # ニューラル埋め込み込み
pip install vecmemori[ja]           # 日本語FTS5対応（fugashi + unidic-lite）
pip install vecmemori[all]          # 全部入り
pip install vecmemori[hermes]       # Hermes Agent プラグイン込み（embed含む）
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

SQLite スキーマは同一です。既存の `memory_store.db` をそのまま使うだけで、初回起動時に自動でスキーマ移行（`fts_text` カラム追加・FTS5再構築）が実行されます。

念のため移行前にバックアップを推奨します:
```bash
cp ~/.hermes/memory_store.db ~/.hermes/memory_store.db.backup
```

## Hermes Agent 上の動作

vecmemori は Hermes Agent のメモリプロバイダーとして、会話のあらゆる段階で動作します:

### メモリ読み出し（prefetch）— 毎メッセージ

ユーザーがメッセージを送信するたびに、vecmemori が自動的に事実ストアを検索し、関連する事実をシステムプロンプトに注入します:

```
ユーザーメッセージ
    │
    ▼
vecmemori.prefetch(message)
    │
    ├─► FTS5 + Jaccard + HRR + ニューラル検索
    ├─► 上位N件（デフォルト: 5件）を選択
    ├─► 「## Vecmemori Memory」として注入
    └─► モデルが応答生成前に関連背景情報を参照
```

これはツールコールなしでサイレントに実行されます。モデルは前のセッションの関連知識を「知っている」状態で応答を生成できます。注入件数は `prefetch_limit` で設定可能（デフォルト: 5）。

### 記憶化（fact_store）— 適宜呼び出し

モデルが「これは覚えておくべき」と判断したときに、`fact_store` ツールで明示的に事実を保存します:

```python
# モデルが自動的に呼び出す
fact_store(action="add", content="ユーザーはRustでのシステムプログラミングを好む")
```

主なツールアクション:
- `add` — 新規事実を保存（内容で自動重複排除）
- `search` — キーワード/意味検索
- `probe` — エンティティ中心の検索
- `reason` — 複数エンティティに同時に関連する事実を検索
- `contradict` — 矛盾する事実を検出
- `update` / `remove` / `list` — CRUD操作

### memory ツールのミラーリング — 内蔵メモリ書き込み時

モデルが Hermes の内蔵 `memory` ツール（MEMORY.md / USER.md への書き込み）を使うと、vecmemori が自動的に同じ内容を構造化事実としてミラーリングします:

```
memory(action="add", target="memory", content="...")
    │
    ├─► 内蔵: MEMORY.md に保存（常時有効）
    └─► vecmemori ミラー: 事実として保存（カテゴリ: user_pref / general）
```

これにより、`fact_store` を明示的に呼ばなくても事実が自動的に蓄積されます。

### 自動抽出（auto-extraction）— セッション終了時

セッションが終了すると（CLI終了、/reset、タイムアウト）、vecmemori は直近約40メッセージを LLM に送信し、耐久性のある事実を抽出します:

```
セッション終了
    │
    ▼
vecmemori.on_session_end(messages)
    │
    ├─► LLM が会話 + 抽出プロンプトを受信
    ├─► LLM が JSON を返却: [{content, category, tags}, ...]
    └─► 各事実を MemoryStore.add_fact() で保存
```

抽出プロンプトは日本語で、以下の情報を対象としています:
- ユーザーの好み・習慣
- 決定事項
- プロジェクト要件や進捗
- ツール・設定に関する選択

設定: `auto_extract: true`（デフォルト: true）。

### Retrieval Planner — オプション機能

`retrieval_planner: true` を有効にすると、毎ターン LLM が会話文脈から複数の検索クエリを生成し、ファンアウト検索して結果を統合します:

```
ユーザーメッセージ
    │
    ▼
LLM が 3-6 個の検索質問を生成
    │
    ├─► 「ユーザーはXについてどう思っていたか？」
    ├─► 「Yに関する制約は何があったか？」
    └─► 「Zに関連する過去の決定は？」
           │
           ▼
    各質問 → 個別の検索クエリ
           │
           ▼
    結果を統合・重複排除・スコアリング
           │
           ▼
    上位候補をコンテキストに注入
```

これにより、現在のメッセージに直接言及されていない事実も発見できます。

### 動作サマリー

| 契機 | 動作 | 設定 |
|------|------|------|
| 毎ユーザーメッセージ | 事実検索 → 上位N件注入 | `prefetch_limit`（デフォルト: 5） |
| モデルが fact_store 呼び出し | 事実の保存/更新/削除 | 常時利用可能 |
| モデルが memory ツール呼び出し | 自動ミラーリング | 常時有効 |
| セッション終了 | LLM が事実抽出 | `auto_extract: true` |
| 毎ターン（planner） | 複数クエリLLM検索 | `retrieval_planner: false` |

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
