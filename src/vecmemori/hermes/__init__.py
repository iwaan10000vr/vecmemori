"""vecmemori[hermes] — Hermes Agent memory provider plugin.

Provides the MemoryProvider interface that plugs into Hermes Agent's
memory system. This is a thin adapter — all core logic lives in vecmemori.

Setup:
    hermes config set memory.provider vecmemori

Config (under plugins.vecmemori in config.yaml):
    auto_extract           Auto-extract facts at session end (default: true)
    auto_extract_model     Model for LLM extraction (empty = main model)
    default_trust          Default trust score for new facts (default: 0.5)
    fact_storage_language  Language used to write fact content (unset = refuse writes)
    retrieval_planner      Use LLM question-driven multi-query search (default: false)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from hermes_cli.config import cfg_get

from vecmemori import MemoryStore, FactRetriever

logger = logging.getLogger(__name__)

FACT_LANGUAGE_UNSET_ERROR = (
    "vecmemori fact_storage_language is not set. Ask the user which language "
    "facts should be stored in, then save it under plugins.vecmemori."
)


def _normalize_fact_language(value: Any) -> str | None:
    """Return a normalized fact storage language code, or None if unset."""
    if value is None:
        return None
    lang = str(value).strip()
    if not lang or lang.lower() in {"null", "none", "unset", "auto"}:
        return None
    return lang


def _fact_language_instruction(language: str) -> str:
    return (
        f"All fact content MUST be written in this storage language: {language}. "
        "Do not choose another language based on the conversation language. "
        "Keep proper nouns, project names, file paths, code identifiers, product names, "
        "technical terms, and user-preferred original spellings as-is when that is natural "
        "for the target language. Return only the fact text; do not add translation notes."
    )


def _looks_like_english_sentence(text: str) -> bool:
    """Heuristic guard for the common failure mode: full English facts in Japanese mode."""
    stripped = text.strip()
    if not stripped:
        return False
    letters = sum(ch.isalpha() for ch in stripped)
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in stripped)
    cjk = sum("\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" for ch in stripped)
    return letters >= 12 and ascii_letters / max(letters, 1) > 0.85 and cjk == 0


def _language_guard_error(language: str, content: str) -> str | None:
    """Return an error when content obviously violates the configured storage language."""
    if language.lower() in {"ja", "japanese", "日本語"} and _looks_like_english_sentence(content):
        return (
            "fact_storage_language is set to Japanese, but the fact content looks like a "
            "full English sentence. Rewrite the fact in Japanese and call fact_store again. "
            "Keep proper nouns and technical terms in their original spelling when natural."
        )
    return None

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Structured semantic fact memory. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for explicit fact storage and retrieval.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — Entity-adjacent recall using semantic search.\n"
        "• reason — Search for facts related to multiple supplied entities.\n"
        "• contradict — List candidate facts for manual conflict inspection.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first.\n"
        "IMPORTANT: add/update writes are allowed only after plugins.vecmemori.fact_storage_language is configured; "
        "fact content must be written in that configured language while preserving proper nouns, code, paths, and natural technical terms."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "vecmemori", default={}) or {}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class VecmemoriMemoryProvider(MemoryProvider):
    """vecmemori-backed memory provider for Hermes Agent."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))
        self._fact_storage_language = _normalize_fact_language(self._config.get("fact_storage_language"))

    @property
    def name(self) -> str:
        return "vecmemori"

    def is_available(self) -> bool:
        return True  # SQLite always available

    def save_config(self, values, hermes_home):
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["vecmemori"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        _default_embedder = f"{display_hermes_home()}/models/ruri-v3-310m"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "true", "choices": ["true", "false"]},
        {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
        {"key": "embedding_weight", "description": "Neural embedding weight in hybrid search (0.0=disabled)", "default": "0.60"},
            {"key": "embedding_model", "description": "Local embedding model path/name", "default": _default_embedder},
            {"key": "embedding_keep_alive", "description": "Model keep-alive: -1=always, 0=unload after search, N=keep N sec", "default": "-1"},
            {"key": "embedding_trust_remote_code", "description": "Allow custom HuggingFace model code execution", "default": "false", "choices": ["true", "false"]},
            {"key": "auto_extract_llm", "description": "Use LLM for auto-extraction instead of regex", "default": "true", "choices": ["true", "false"]},
            {"key": "auto_extract_model", "description": "Model for LLM auto-extraction (empty = use main model)", "default": ""},
            {"key": "fact_storage_language", "description": "Language for stored fact content (e.g. ja, en). Empty/null refuses fact writes until the user chooses.", "default": ""},
            {"key": "retrieval_planner", "description": "Use LLM question-driven planning for multi-query search", "default": "false", "choices": ["true", "false"]},
            {"key": "retrieval_planner_model", "description": "Model for retrieval planning (empty = main model)", "default": ""},
            {"key": "planner_max_questions", "description": "Maximum unresolved questions per planner run", "default": "6"},
            {"key": "planner_max_queries", "description": "Maximum total planner search queries per turn", "default": "6"},
            {"key": "planner_per_query_limit", "description": "Facts retrieved per planner query", "default": "4"},
            {"key": "planner_max_candidate_facts", "description": "Maximum deduplicated candidates before injection", "default": "24"},
            {"key": "planner_max_injected_facts", "description": "Maximum facts injected into the current turn", "default": "10"},
            {"key": "planner_llm_injection", "description": "Use LLM to compact/rerank candidate facts before injection", "default": "false", "choices": ["true", "false"]},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)

        default_trust = float(self._config.get("default_trust", 0.5))
        embedding_weight = float(self._config.get("embedding_weight", 0.60))
        embedding_keep_alive = int(self._config.get("embedding_keep_alive", -1))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        embedding_model = self._config.get("embedding_model",
                          self._config.get("ruri_model", _hermes_home + "/models/ruri-v3-310m"))
        if isinstance(embedding_model, str):
            embedding_model = embedding_model.replace("$HERMES_HOME", _hermes_home)
            embedding_model = embedding_model.replace("${HERMES_HOME}", _hermes_home)
            try:
                from vecmemori._embedder import set_model_path, set_trust_remote_code
                set_model_path(embedding_model)
                set_trust_remote_code(self._as_bool(self._config.get("embedding_trust_remote_code", False)))
            except Exception as e:
                logger.debug("Failed to configure embedding model path %r: %s", embedding_model, e)

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, require_embeddings=True)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            ruri_weight=embedding_weight,
            ruri_keep_alive=embedding_keep_alive,
            require_embeddings=True,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        language_line = (
            f"Fact storage language: {self._fact_storage_language}. "
            f"When adding/updating facts, write fact content in {self._fact_storage_language}; "
            "keep proper nouns, code, paths, project names, and natural technical terms as-is.\n"
            if self._fact_storage_language
            else "Fact storage language is NOT configured. Do not add/update facts; ask the user which language to store facts in first.\n"
        )
        if total == 0:
            return (
                "# Vecmemori Memory\n"
                "Active. Empty fact store — proactively add facts only when the storage language is configured.\n"
                + language_line +
                "Use fact_store to store structured facts about people, projects, preferences.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Vecmemori Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            + language_line +
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            raw_limit = int(self._config.get("prefetch_limit", 5))
            results = self._retriever.search(query, min_trust=self._min_trust, limit=raw_limit)

            if self._as_bool(self._config.get("retrieval_planner", False)):
                results = self._planner_augmented_results(query, results)

            if not results:
                return ""
            return self._format_prefetch_results(results, query)
        except Exception as e:
            logger.debug("vecmemori prefetch failed: %s", e)
            return ""

    def _planner_augmented_results(self, query: str, base_results: list[dict]) -> list[dict]:
        plan = self._llm_plan_retrieval(query)
        questions = plan.get("questions", []) if isinstance(plan, dict) else []
        if not isinstance(questions, list) or not questions:
            return base_results

        max_queries = int(self._config.get("planner_max_queries", 6))
        per_query_limit = int(self._config.get("planner_per_query_limit", 4))
        max_candidates = int(self._config.get("planner_max_candidate_facts", 24))

        merged: dict[int, dict] = {}
        for result in base_results or []:
            fid = result.get("fact_id")
            if fid is None:
                continue
            result = dict(result)
            result.setdefault("retrieval_reason", "raw user/context query")
            result.setdefault("planner_score", float(result.get("score", 0.0)))
            merged[int(fid)] = result

        query_count = 0
        for q in questions:
            if not isinstance(q, dict):
                continue
            if q.get("already_answered_by_context") is True:
                continue
            reason = str(q.get("question") or q.get("why_it_matters") or "planner query")[:240]
            search_queries = q.get("search_queries") or q.get("queries") or []
            if isinstance(search_queries, str):
                search_queries = [search_queries]
            for search_query in search_queries:
                search_query = str(search_query or "").strip()
                if not search_query:
                    continue
                query_count += 1
                if query_count > max_queries:
                    break
                for result in self._retriever.search(
                    search_query,
                    min_trust=self._min_trust,
                    limit=per_query_limit,
                ):
                    fid = result.get("fact_id")
                    if fid is None:
                        continue
                    result = dict(result)
                    planner_score = float(result.get("score", 0.0)) + 0.05
                    if int(fid) in merged:
                        if planner_score > float(merged[int(fid)].get("planner_score", 0.0)):
                            merged[int(fid)].update(result)
                        merged[int(fid)]["planner_score"] = max(
                            float(merged[int(fid)].get("planner_score", 0.0)), planner_score
                        )
                        merged[int(fid)]["retrieval_reason"] = (
                            str(merged[int(fid)].get("retrieval_reason", "")) + "; " + reason
                        )[:500]
                    else:
                        result["planner_score"] = planner_score
                        result["retrieval_reason"] = reason
                        merged[int(fid)] = result
                if len(merged) >= max_candidates:
                    break
            if query_count >= max_queries or len(merged) >= max_candidates:
                break

        ranked = sorted(
            merged.values(),
            key=lambda r: (float(r.get("planner_score", r.get("score", 0.0))), float(r.get("trust_score", 0.0))),
            reverse=True,
        )
        return ranked[:max_candidates]

    def _format_prefetch_results(self, results: list[dict], query: str = "") -> str:
        if self._as_bool(self._config.get("planner_llm_injection", False)):
            try:
                compact = self._llm_compact_injection(query, results)
                if compact.strip():
                    return "## Vecmemori Memory\n" + compact.strip()
            except Exception as e:
                logger.debug("vecmemori injection compaction failed: %s", e)

        max_injected = int(self._config.get("planner_max_injected_facts", self._config.get("prefetch_limit", 5)))
        compact = self._as_bool(self._config.get("planner_compact_injection", True))
        lines = []
        for r in results[:max_injected]:
            trust = r.get("trust_score", r.get("trust", 0))
            score = r.get("planner_score", r.get("score"))
            content = r.get("content", "")
            if compact:
                lines.append(f"- trust={trust:.2f}, relevance={float(score or 0):.3f}: {content}")
            else:
                reason = r.get("retrieval_reason")
                suffix = f" — why: {reason}" if reason else ""
                lines.append(f"- [{trust:.1f}] {content}{suffix}")
        if compact:
            return (
                "## Vecmemori Memory\n"
                "Potentially relevant background for this turn.\n"
                + "\n".join(lines)
            )
        return "## Vecmemori Memory\n" + "\n".join(lines)

    def _llm_compact_injection(self, query: str, results: list[dict]) -> str:
        import httpx
        max_candidates = int(self._config.get("planner_injection_candidate_limit", 16))
        api_conf = self._get_extract_api_config()
        endpoint = api_conf["base_url"].rstrip("/") + "/chat/completions"
        candidates = []
        for r in results[:max_candidates]:
            candidates.append({
                "fact_id": r.get("fact_id"),
                "trust": r.get("trust_score", r.get("trust")),
                "relevance": r.get("planner_score", r.get("score")),
                "content": r.get("content", ""),
                "why_retrieved": r.get("retrieval_reason", ""),
            })
        system_prompt = (
            "You compress memory retrieval candidates for an assistant.\n"
            "Return short background notes, not an explanation of all candidates.\n"
            "Hard-exclude truly irrelevant candidates. Keep uncertain but plausible signals as gentle, non-committal background.\n"
            "Do NOT enumerate negative alternatives.\n"
            "Prefer 2-6 bullets. Include a caution if links are speculative.\n"
            "Output plain text only, starting with 'Potentially relevant background:'"
        )
        user_prompt = (
            "Current conversation/latest message context:\n"
            f"{query[:4000]}\n\n"
            "Scored candidate facts as JSON:\n"
            f"{json.dumps(candidates, ensure_ascii=False)[:6000]}"
        )
        resp = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_conf['api_key']}"},
            json={
                "model": self._config.get("retrieval_planner_model") or api_conf["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": int(self._config.get("planner_injection_max_tokens", 2500)),
            },
            timeout=float(self._config.get("planner_injection_timeout", 90)),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content", "")

    def _llm_plan_retrieval(self, query: str) -> dict:
        import json as json_mod
        import httpx
        max_questions = int(self._config.get("planner_max_questions", 6))
        api_conf = self._get_extract_api_config()
        endpoint = api_conf["base_url"].rstrip("/") + "/chat/completions"
        system_prompt = (
            "You are a long-term MEMORY retrieval planner for a personal AI assistant.\n"
            "Read the current conversation context and latest user message.\n"
            "Generate unresolved questions whose answers may be in the user's saved memory/facts.\n"
            "Search queries are for a private memory database, NOT the public web.\n"
            "If the current context already answers a question, mark already_answered_by_context=true.\n"
            "Return compact JSON only.\n"
            'Schema: {"questions":[{"question":str,"why_it_matters":str,"already_answered_by_context":bool,"search_queries":[str],"priority":"high|medium|low"}]}'
        )
        user_prompt = (
            f"Hard limits: max_questions={max_questions}, max_search_queries_total="
            f"{int(self._config.get('planner_max_queries', 6))}.\n\n"
            "Conversation/latest message context:\n"
            f"{query[:6000]}"
        )
        resp = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_conf['api_key']}"},
            json={
                "model": self._config.get("retrieval_planner_model") or api_conf["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": int(self._config.get("retrieval_planner_max_tokens", 4000)),
            },
            timeout=float(self._config.get("retrieval_planner_timeout", 90)),
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"].get("content", "")
        code = text.strip()
        if not code:
            return {"questions": []}
        if "```json" in code:
            code = code.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in code:
            code = code.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            plan = json_mod.loads(code)
        except json_mod.JSONDecodeError:
            start = code.find("{")
            end = code.rfind("}")
            if start >= 0 and end > start:
                plan = json_mod.loads(code[start:end + 1])
            else:
                return {"questions": []}
        if not isinstance(plan, dict):
            return {"questions": []}
        questions = plan.get("questions", [])
        if isinstance(questions, list) and len(questions) > max_questions:
            plan["questions"] = questions[:max_questions]
        return plan

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.get("auto_extract", True):
            return
        if not self._store or not messages:
            return
        self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action == "add" and self._store and content:
            try:
                validation_error = self._validate_fact_write(content)
                if validation_error:
                    logger.warning("vecmemori memory_write mirror skipped: %s", validation_error)
                    return
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("vecmemori memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None
        self._retriever = None

    def _require_fact_storage_language(self) -> str | None:
        self._fact_storage_language = _normalize_fact_language(
            self._config.get("fact_storage_language")
        )
        return self._fact_storage_language

    def _validate_fact_write(self, content: str) -> str | None:
        language = self._require_fact_storage_language()
        if not language:
            return FACT_LANGUAGE_UNSET_ERROR
        return _language_guard_error(language, content)

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                content = args["content"]
                validation_error = self._validate_fact_write(content)
                if validation_error:
                    return tool_error(validation_error)
                fact_id = store.add_fact(
                    content,
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                # probe is an alias for entity-focused search
                entity = args.get("entity", "")
                if entity:
                    results = retriever.search(
                        entity,
                        category=args.get("category"),
                        min_trust=float(args.get("min_trust", self._min_trust)),
                        limit=int(args.get("limit", 10)),
                    )
                else:
                    results = []
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                entity = args.get("entity", "")
                if entity:
                    results = retriever.search(
                        entity,
                        category=args.get("category"),
                        min_trust=float(args.get("min_trust", self._min_trust)),
                        limit=int(args.get("limit", 10)),
                    )
                else:
                    results = []
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                # Search for each entity and union results. This is not symbolic
                # reasoning; it is a pragmatic multi-query semantic recall.
                seen: set[int] = set()
                merged: list[dict] = []
                for entity in entities:
                    for result in retriever.search(
                        entity,
                        category=args.get("category"),
                        min_trust=float(args.get("min_trust", self._min_trust)),
                        limit=int(args.get("limit", 10)),
                    ):
                        fid = result.get("fact_id")
                        if fid not in seen:
                            seen.add(fid)
                            merged.append(result)
                return json.dumps({"results": merged, "count": len(merged)})

            elif action == "contradict":
                # Contradiction detection requires entity overlap analysis;
                # fall back to listing recent facts for manual inspection
                results = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results), "note": "Automatic contradiction detection is not available in this version; returned candidates for manual inspection."})

            elif action == "update":
                if "content" in args and args.get("content") is not None:
                    validation_error = self._validate_fact_write(args.get("content") or "")
                    if validation_error:
                        return tool_error(validation_error)
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction -----------------------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        if not self._store or not messages:
            return
        if not self._require_fact_storage_language():
            logger.warning("vecmemori auto-extract skipped: %s", FACT_LANGUAGE_UNSET_ERROR)
            return
        try:
            self._llm_extract_facts(messages)
        except Exception as e:
            logger.debug("LLM auto-extract failed (non-critical): %s", e)

    def _llm_extract_facts(self, messages: list) -> None:
        import json as json_mod
        lines = []
        user_msg_count = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "user":
                user_msg_count += 1
            text = content[:600] if len(content) > 600 else content
            lines.append(f"[{role}] {text}")

        if not lines or user_msg_count == 0:
            return

        conversation = "\n".join(lines[-40:])
        fact_language = self._require_fact_storage_language()
        if not fact_language:
            return
        language_instruction = _fact_language_instruction(fact_language)

        system_prompt = (
            "あなたは会話から覚えておくべき事実を抽出するアシスタントです。\n"
            "以下の会話を読み、将来の会話で参照すべき情報のみをJSON配列で返してください。\n"
            f"保存言語ルール: {language_instruction}\n\n"
            "抽出対象:\n"
            "- ユーザーの好み・習慣（〜したい、〜が好き、〜を使う、〜は嫌）\n"
            "- 決定事項（〜に決めた、〜にする、やめる、契約した）\n"
            "- プロジェクト・タスクの要件や進捗\n"
            "- ツール・設定に関する決定\n"
            "- 人間関係や役割（○○が△△を担当）\n\n"
            "除外:\n"
            "- 単なる雑談や感情表現\n"
            "- 一般的な知識（LLMが元から知っていること）\n"
            "- あいまいすぎる発言\n\n"
            "カテゴリ:\n"
            "- user_pref: 好み・習慣・価値観\n"
            "- project: プロジェクト・タスク情報\n"
            "- tool: ツール・設定\n"
            "- general: その他重要情報\n\n"
            "JSON配列のみを返してください:\n"
            '[{"content": "事実の内容（簡潔に1文）", "category": "user_pref|project|tool|general", "tags": "タグ1,タグ2"}, ...]\n'
            "該当なし → []"
        )

        api_conf = self._get_extract_api_config()
        endpoint = api_conf["base_url"].rstrip("/") + "/chat/completions"

        import httpx
        timeout_val = self._config.get("auto_extract_timeout", 300)
        try:
            resp = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_conf['api_key']}"},
                json={
                    "model": api_conf["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"会話:\n{conversation}"},
                    ],
                    "temperature": 0.1,
                    "max_tokens": self._config.get("auto_extract_max_tokens", 4000),
                },
                timeout=timeout_val,
            )
            resp.raise_for_status()
        except httpx.ReadTimeout:
            logger.warning("LLM extract timed out (%ss) — skipping", timeout_val)
            return
        except httpx.HTTPStatusError as e:
            logger.warning("LLM extract API error: %s (status=%s)", e.response.status_code, e.response.status_code)
            return
        text = resp.json()["choices"][0]["message"]["content"]

        code = text.strip()
        if "```json" in code:
            code = code.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in code:
            code = code.split("```", 1)[1].split("```", 1)[0].strip()

        facts = json_mod.loads(code)
        if not isinstance(facts, list):
            return

        count = 0
        for fact in facts:
            content = (fact.get("content") or "").strip()
            if len(content) < 5:
                continue
            category = fact.get("category", "general")
            if category not in ("user_pref", "project", "tool", "general"):
                category = "general"
            tags = fact.get("tags", "")
            validation_error = self._validate_fact_write(content)
            if validation_error:
                logger.warning("Skipping extracted fact that violates fact_storage_language: %s", validation_error)
                continue
            try:
                self._store.add_fact(content, category=category, tags=tags)
                count += 1
            except Exception as e:
                logger.debug("Failed to add extracted fact: %s", e)

        if count:
            logger.info("LLM extracted %d facts from conversation", count)

    def _get_extract_api_config(self) -> dict:
        from pathlib import Path
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        model = self._config.get("auto_extract_model") or ""
        if not model:
            model = cfg.get("model", {}).get("default", "deepseek-v4-flash")

        provider = self._config.get("auto_extract_provider") or ""
        if not provider:
            provider = cfg.get("model", {}).get("provider", "opencode-go")

        base_url = self._config.get("auto_extract_base_url") or ""
        if not base_url:
            base_url = cfg.get("model", {}).get("base_url", "https://opencode.ai/zen/go/v1")

        api_key = cfg.get("model", {}).get("api_key", "")

        import os as _os
        _CREDENTIAL_ENV_MAP = {
            "opencode-go": "OPENCODE_GO_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "OPENCLAW_ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        if api_key in ("local", "") and provider in _CREDENTIAL_ENV_MAP:
            env_key = _CREDENTIAL_ENV_MAP[provider]
            api_key = _os.environ.get(env_key, api_key)
        if api_key in ("local", ""):
            api_key = _os.environ.get("OPENCODE_GO_API_KEY", api_key)

        # Fallback: directly read .env if os.environ still doesn't have the key
        if api_key in ("local", ""):
            try:
                from dotenv import dotenv_values
                env_path = get_hermes_home() / ".env"
                if env_path.exists():
                    env_vals = dotenv_values(str(env_path))
                    if provider in _CREDENTIAL_ENV_MAP:
                        env_key = _CREDENTIAL_ENV_MAP[provider]
                        api_key = env_vals.get(env_key, api_key)
                    if api_key in ("local", ""):
                        api_key = env_vals.get("OPENCODE_GO_API_KEY", api_key)
            except ImportError:
                pass

        return {"model": model, "base_url": base_url, "api_key": api_key, "provider": provider}


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the vecmemori memory provider with Hermes Agent's plugin system."""
    config = _load_plugin_config()
    provider = VecmemoriMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
