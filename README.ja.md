<h1 align="center">vecmemori</h1>
<p align="center">
  <em>AIエージェントのためのローカル事実記憶 — SQLite + FTS5 + ニューラル埋め込み</em>
  <br>
  <a href="README.md">English README</a>
</p>

---

**vecmemori** は、AIエージェント向けのローカル・永続的な事実記憶エンジンです。ユーザーの好み、プロジェクトの決定事項、環境情報などの短い事実を SQLite に保存し、次の2系統で関連する記憶を検索します。

- **FTS5全文検索** — キーワード一致の精度を担当
- **ニューラル埋め込み** — 意味的な近さを担当

埋め込みは中核機能です。vecmemori は「単なる全文検索DB」ではなく、意味検索を前提としたメモリシステムです。

## プライバシーモデル

コアライブラリは SQLite とローカルの SentenceTransformer 互換モデルを使います。この構成ではデータは端末内に留まります。

ただし、統合機能によっては設定済みの LLM プロバイダーにテキストを送る場合があります。

- Hermes Agent の `auto_extract` は、会話履歴を LLM に送り、保存すべき事実を抽出することがあります。
- Hermes Agent の `retrieval_planner` / planner injection は、検索クエリ生成のために会話文脈を LLM に送ることがあります。
- デフォルト外のリモート埋め込みプロバイダーを明示的に設定した場合、埋め込み対象テキストが外部に送られる可能性があります。

利用する LLM/埋め込みプロバイダーを信頼できる場合のみ、それらの機能を有効にしてください。

## インストール

```bash
pip install vecmemori
pip install vecmemori[ja]       # fugashi + unidic-lite による日本語FTS5対応
pip install vecmemori[hermes]   # Hermes Agent 環境向けアダプター依存
pip install vecmemori[all]      # 日本語 + Hermes アダプター依存
```

事実の保存・検索の前に、ローカル埋め込みモデルを用意します。

```bash
bash scripts/download_model.sh
```

デフォルトのモデル配置先:

```text
~/.cache/vecmemori/models/ruri-v3-310m
```

デフォルトモデルは [`cl-nagoya/ruri-v3-310m`](https://huggingface.co/cl-nagoya/ruri-v3-310m) です。Apache-2.0 ライセンスで、日本語に強いモデルです。

## クイックスタート

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
store.add_fact("ユーザーはダークモードを好む", category="user_pref")
store.add_fact("プロジェクトでは pytest を使う", category="project")

retriever = FactRetriever(store=store)
results = retriever.search("テスト環境と好み", limit=5)

for r in results:
    print(f"[{r['trust_score']:.2f}] {r['content']}")
```

埋め込みなしの明示的なテスト・診断だけは、`MemoryStore` と `FactRetriever` の両方に `require_embeddings=False` を渡すことで実行できます。本番利用では推奨しません。

## 日本語FTS5対応

SQLite 標準の `unicode61` tokenizer は日本語をうまく分割できません。`vecmemori[ja]` を入れると [fugashi](https://github.com/polm/fugashi) / MeCab で事前分かち書きしてから FTS5 に登録します。

```bash
pip install vecmemori[ja]
python -c "from vecmemori._tokenizer import has_tokenizer; print('Japanese FTS5:', has_tokenizer())"
```

意味検索の主軸はニューラル埋め込みです。日本語FTS5は、キーワード一致の精度を補強する役割です。

## 特徴

- **2系統検索** — FTS5 + ニューラル埋め込み。デフォルト重みは 0.40 / 0.60
- **埋め込み前提** — ローカル SentenceTransformer 互換モデルを使う
- **日本語対応** — ruri-v3 埋め込み + fugashi による任意のFTS5形態素解析
- **エンティティ抽出** — 事実文から軽量に固有表現を抽出
- **信頼度スコア** — helpful/unhelpful フィードバックで信頼度を調整
- **時間減衰** — 古い事実を弱めるオプション
- **モデル差し替え** — 埋め込み次元、query/doc prefix を設定可能
- **Hermes Agent連携** — memory provider、`fact_store` tool、prefetch、任意のLLM planner/extraction

## Python API

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
fact_id = store.add_fact("GPU: RTX 5060 Ti 16GB", category="tool", tags="hardware,gpu")
store.record_feedback(fact_id=fact_id, helpful=True)

retriever = FactRetriever(store=store)
results = retriever.search("GPUメモリ", limit=5)

store.rebuild_all_embeddings()  # 埋め込みモデル/設定変更後に再生成
store.close()
```

`probe`, `reason`, `contradict` は standalone の `FactRetriever` メソッドではなく、Hermes の `fact_store` tool action です。

## Hermes Agent連携

`vecmemori[hermes]` は Hermes plugin module に必要な `PyYAML` / `httpx` などを入れます。Hermes Agent本体はインストールしません。既に Hermes Agent が入っている環境に追加するための extra です。

Hermes memory provider として使う場合の流れ:

1. **毎ユーザーメッセージ:** prefetch が関連factを検索し、上位候補を文脈に注入
2. **tool call:** `fact_store` で fact を明示的に追加・検索・更新・削除
3. **memory mirroring:** Hermes標準の `memory` tool 書き込みを vecmemori にも反映
4. **session end:** 任意で `auto_extract` が LLM に durable fact 抽出を依頼
5. **planner:** 任意で `retrieval_planner` が LLM に複数検索クエリ生成を依頼

Hermes専用 tool action:

- `add`, `search`, `probe`, `related`, `reason`, `contradict`, `update`, `remove`, `list`

`reason` / `contradict` は実用的な意味検索補助であり、数学的な代数推論や完全な矛盾検出を保証するものではありません。

## 設定

主な設定:

- `db_path`: SQLite DB パス。standalone ではデフォルト `memory.db`
- `default_trust`: 新規factの初期信頼度。デフォルト `0.5`
- `fts_weight`: キーワード検索スコアの重み。デフォルト `0.40`
- `ruri_weight`: 意味埋め込みスコアの重み。デフォルト `0.60`（互換性のため旧キー名を維持）
- `prefetch_limit`: Hermesで毎ターン注入するfact数。デフォルト `5`
- `auto_extract`: Hermesでセッション終了時にLLM抽出を行うか
- `retrieval_planner`: HermesでLLMによるマルチクエリ検索を使うか
- `embedding_model`: ローカル埋め込みモデルパス
- `embedding_trust_remote_code`: Hugging Face custom model code を許可するか。デフォルト `false`

## 埋め込みモデルの差し替え

```python
from vecmemori._embedder import set_config, set_model_path

set_config(dimension=384, query_prefix="", doc_prefix="")
set_model_path("/path/to/all-MiniLM-L6-v2")

from vecmemori import MemoryStore
store = MemoryStore("memory.db")
store.rebuild_all_embeddings()
```

推奨例:

- `cl-nagoya/ruri-v3-310m` — 日本語、768次元、約1.2GB
- `sentence-transformers/all-MiniLM-L6-v2` — 英語、384次元、小型
- `BAAI/bge-large-en-v1.5` — 英語、1024次元
- `intfloat/multilingual-e5-large` — 多言語、1024次元

## アーキテクチャ

```text
ユーザークエリ
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 / BM25           │  weight=0.40 — キーワード一致
│ 2. ニューラルcos類似度   │  weight=0.60 — 意味的近さ
└──────────┬───────────────┘
           ▼
    関連度 × 信頼度 × 時間減衰
           ▼
    ソート結果
```

## 開発

```bash
pip install -e ".[dev,ja]"
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
check-manifest
```

## ライセンス

MIT — [LICENSE](LICENSE) を参照。

第三者ライセンス情報は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にまとめています。Hermes Agent / Holographic memory plugin 由来部分の attribution は [NOTICE](NOTICE) と third-party notices に記載しています。

## 謝辞

vecmemori は、Nous Research の [Hermes Agent](https://github.com/nousresearch/hermes-agent) に含まれていた MIT ライセンスの `holographic` memory plugin のフォーク/再実装として始まりました。その後、standalone package として独立し、FTS5 + ニューラル埋め込み検索を中心に整理されています。

Hermes Agent、SQLite、NumPy、sentence-transformers、PyTorch、fugashi/MeCab、UniDic、cl-nagoya の ruri モデルの開発者・メンテナーに感謝します。
