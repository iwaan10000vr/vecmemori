"""Tests for Hermes adapter fact storage language policy."""

import importlib
import json
import sys
import types

import pytest

from vecmemori import MemoryStore


def _load_hermes_module():
    """Import vecmemori.hermes with lightweight Hermes dependency stubs."""
    agent_mod = types.ModuleType("agent")
    memory_provider_mod = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider_mod.MemoryProvider = MemoryProvider
    agent_mod.memory_provider = memory_provider_mod
    sys.modules.setdefault("agent", agent_mod)
    sys.modules.setdefault("agent.memory_provider", memory_provider_mod)

    tools_mod = types.ModuleType("tools")
    registry_mod = types.ModuleType("tools.registry")

    def tool_error(message):
        return json.dumps({"error": message})

    registry_mod.tool_error = tool_error
    tools_mod.registry = registry_mod
    sys.modules.setdefault("tools", tools_mod)
    sys.modules.setdefault("tools.registry", registry_mod)

    hermes_cli_mod = types.ModuleType("hermes_cli")
    config_mod = types.ModuleType("hermes_cli.config")

    def cfg_get(config, *keys, default=None):
        current = config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    config_mod.cfg_get = cfg_get
    hermes_cli_mod.config = config_mod
    sys.modules.setdefault("hermes_cli", hermes_cli_mod)
    sys.modules.setdefault("hermes_cli.config", config_mod)

    return importlib.import_module("vecmemori.hermes")


@pytest.fixture
def hermes_module():
    return _load_hermes_module()


@pytest.fixture
def provider_factory(hermes_module, db_path):
    providers = []

    def make(config):
        provider = hermes_module.VecmemoriMemoryProvider(config=config)
        provider._store = MemoryStore(db_path=db_path, default_trust=0.5, require_embeddings=False)
        provider._retriever = None
        providers.append(provider)
        return provider

    yield make

    for provider in providers:
        if provider._store:
            provider._store.close()


def test_fact_store_add_refuses_when_storage_language_unset(provider_factory):
    provider = provider_factory({"fact_storage_language": None})

    result = json.loads(provider._handle_fact_store({"action": "add", "content": "ユーザーは日本語保存を望む"}))

    assert "error" in result
    assert "fact_storage_language is not set" in result["error"]
    assert provider._store.list_facts() == []


def test_fact_store_add_accepts_configured_japanese_fact(provider_factory):
    provider = provider_factory({"fact_storage_language": "ja"})

    result = json.loads(provider._handle_fact_store({"action": "add", "content": "ユーザーはtokenmaxxingという語を原語で保持したい", "category": "user_pref"}))

    assert result["status"] == "added"
    facts = provider._store.list_facts()
    assert facts[0]["content"] == "ユーザーはtokenmaxxingという語を原語で保持したい"


def test_fact_store_add_rejects_full_english_sentence_in_japanese_mode(provider_factory):
    provider = provider_factory({"fact_storage_language": "ja"})

    result = json.loads(provider._handle_fact_store({"action": "add", "content": "User wants all future facts to be stored in Japanese."}))

    assert "error" in result
    assert "Japanese" in result["error"]
    assert provider._store.list_facts() == []


def test_system_prompt_tells_agent_to_ask_when_language_unset(provider_factory):
    provider = provider_factory({"fact_storage_language": ""})

    prompt = provider.system_prompt_block()

    assert "Fact storage language is NOT configured" in prompt
    assert "ask the user which language" in prompt


def test_auto_extract_prompt_contains_configured_language_rule(hermes_module):
    instruction = hermes_module._fact_language_instruction("en")

    assert "en" in instruction
    assert "MUST" in instruction
    assert "proper nouns" in instruction
