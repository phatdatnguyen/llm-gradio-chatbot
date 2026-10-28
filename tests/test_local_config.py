"""Offline discovery, per-session routing, and revocation regressions."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace
import os
import threading

import pytest

from chatbot import config, tokens
from chatbot.providers.base import ProviderError


@pytest.fixture
def metadata(monkeypatch):
    for name, value in (("MODEL_CHOICES", []), ("MODEL_TOKEN_LIMITS", {}),
                        ("MODEL_MAX_OUTPUT_TOKENS", {}), ("_MODEL_INFO", {}),
                        ("_EMBEDDING_INFO", {}), ("DEFAULT_MODEL", ""),
                        ("DEFAULT_EMBEDDING_MODEL", ""), ("_CONNECTIONS", {}),
                        ("_EMBEDDING_DIMENSIONS", {}), ("_CONNECTION_RESOLVER", None)):
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, "_CURRENT_CONNECTION", ContextVar("test_connection", default=config._UNBOUND))
    monkeypatch.setattr(config, "DEFAULT_CONTEXT_TOKENS", 4096)
    monkeypatch.setattr(config, "OUTPUT_TOKEN_LIMIT", 0)
    for name in tuple(os.environ):
        if name.startswith(("LMSTUDIO_", "OLLAMA_", "TEXTGEN_", "KOBOLDCPP_")):
            monkeypatch.delenv(name)
    records = {
        "lmstudio": [
            {"id": "local-chat", "type": "llm", "loaded": True, "context_length": 8192, "vision": True},
            {"id": "not-loaded", "type": "llm", "loaded": False, "context_length": None},
            {"id": "nomic-embed-text", "type": "embedding", "loaded": False,
             "context_length": None, "max_context_length": 2048, "identity": "nomic-q4"},
        ],
        "ollama": [
            {"id": "local-chat", "type": "llm", "available": True, "loaded": False,
             "context_length": None, "max_context_length": 131072, "vision": False},
        ],
        "textgen": [{"id": "local-chat", "type": "llm", "loaded": True,
                     "context_length": None, "vision": False, "structured_output": True}],
        "koboldcpp": [{"id": "local-chat", "type": "llm", "loaded": True,
                       "context_length": 131072, "vision": False, "structured_output": True}],
    }
    calls, providers = [], {}
    for name in records:
        providers[name] = SimpleNamespace(
            discover_models=lambda name=name: calls.append((name, "discover")) or deepcopy(records[name]),
            embeddings=lambda model, texts, name=name: calls.append((name, "embed", model, texts)) or [[1., 2., 3.]],
        )
    monkeypatch.setattr(config, "_create_provider", lambda name, settings: providers[name])
    return SimpleNamespace(records=records, providers=providers, calls=calls)


def test_disconnected_reads_and_selector_do_not_contact_a_server(metadata, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_MODEL", "configured-but-not-connected")
    assert [option["name"] for option in config.provider_options()] == ["lmstudio", "ollama", "textgen", "koboldcpp"]
    assert config.get_endpoint("Ollama") == config.OLLAMA_BASE_URL
    assert config.get_model_choices() == [] and config.get_default_model() == ""
    assert config.get_embedding_settings()["model"] == ""
    assert config.get_max_context_tokens("unknown") == 4096
    assert not config.supports_vision("unknown") and not config.supports_structured_output("unknown")
    with pytest.raises(RuntimeError, match="Connect"):
        config.get_provider()
    with pytest.raises(RuntimeError, match="Connect"):
        config.resolve_model()
    assert metadata.calls == []


def test_textgen_connect_is_chat_only_unless_embeddings_are_explicit(metadata, monkeypatch):
    monkeypatch.setenv("TEXTGEN_CONTEXT_LENGTH", "16384")
    monkeypatch.setenv("TEXTGEN_MAX_OUTPUT_TOKENS", "0")
    with config.use_connection(config.connect("textgen")):
        assert config.resolve_model() == "local-chat"
        assert config.get_model_context_info("local-chat") == {
            "loaded": None, "maximum": None, "effective": 16384, "configured": 16384}
        assert config.supports_structured_output("local-chat") and not config.supports_vision("local-chat")
        assert config.get_embedding_settings()["model"] == "" and config.max_output_tokens("local-chat") is None
        with pytest.raises(RuntimeError, match="embedding provider"):
            config.get_embedding_provider()
    assert metadata.calls == [("textgen", "discover")]
    metadata.records["textgen"][0]["structured_output"] = False
    with config.use_connection(config.connect("textgen")):
        assert not config.supports_structured_output("local-chat")


@pytest.mark.parametrize("provider_name", ["textgen", "koboldcpp"])
def test_explicit_embeddings_keep_identity_and_are_revoked_together(metadata, provider_name):
    from chatbot import api
    from rag_chatbot.config import Settings
    with config.use_connection(config.connect("lmstudio")):
        previous_fingerprint = Settings().fingerprint()
    metadata.calls.clear()
    connection = config.connect(provider_name, embedding_connector="lmstudio")
    with config.use_connection(connection):
        assert Settings().fingerprint() == previous_fingerprint
        assert config.get_provider() is metadata.providers[provider_name]
        assert config.get_embedding_provider() is metadata.providers["lmstudio"]
        assert api.embed_texts(["query"], dimensions=3) == [[1.0, 2.0, 3.0]]
        assert config.refresh_models().embedding_provider_name == "lmstudio"
        config.release_connection(connection)
        with pytest.raises(RuntimeError, match="connection changed"):
            api.embed_texts(["query"], dimensions=3)
    assert not any(call[:2] == (provider_name, "embed") for call in metadata.calls)


@pytest.mark.parametrize("provider_name", ["textgen", "koboldcpp"])
def test_separate_embedding_failure_keeps_chat_available_without_fallback(metadata, provider_name):
    metadata.providers["ollama"].discover_models = lambda: (_ for _ in ()).throw(ProviderError("Ollama stopped"))
    connection = config.connect(provider_name, embedding_connector="ollama")
    with config.use_connection(connection):
        assert config.get_provider() is metadata.providers[provider_name]
        assert config.resolve_model() == "local-chat"
        assert connection.embedding_error == "Ollama stopped"
        assert connection.embedding_provider_name == "ollama"
        assert config.get_embedding_settings()["dimensions"] == 0
    assert metadata.calls == [(provider_name, "discover")]


def test_koboldcpp_loaded_context_and_caps_are_session_bound_without_native_embeddings(metadata, monkeypatch):
    original = config.connect("KoboldCpp")
    with config.use_connection(original):
        assert config.get_model_context_info("local-chat") == {
            "loaded": 131072, "maximum": None, "effective": 131072, "configured": None}
        assert config.max_output_tokens("local-chat") is None
        assert config.supports_structured_output("local-chat")
        with pytest.raises(RuntimeError, match="embedding provider"):
            config.get_embedding_provider()
    monkeypatch.setenv("KOBOLDCPP_CONTEXT_LENGTH", "16384")
    monkeypatch.setenv("KOBOLDCPP_MAX_OUTPUT_TOKENS", "6000")
    with config.use_connection(config.connect("koboldcpp")):
        assert config.get_max_context_tokens("local-chat") == 16384
        assert config.max_output_tokens("local-chat") == 6000
    monkeypatch.setenv("KOBOLDCPP_CONTEXT_LENGTH", "262144")
    with config.use_connection(config.connect("koboldcpp")):
        assert config.get_max_context_tokens("local-chat") == 131072
    with config.use_connection(original):
        assert config.get_max_context_tokens("local-chat") == 131072
    assert all(call == ("koboldcpp", "discover") for call in metadata.calls)


def test_textgen_context_override_is_session_bound_and_checked_before_discovery(metadata):
    first = config.connect("textgen", context_length=131072.0)
    second = config.connect("textgen", context_length=16384)
    with config.use_connection(first):
        assert config.get_max_context_tokens("local-chat") == 131072
        assert config.refresh_models().configured_context == 131072
    with config.use_connection(second):
        assert config.get_max_context_tokens("local-chat") == 16384
    calls = list(metadata.calls)
    for invalid in (True, 8192.5, float("nan"), 0, 1048577):
        with pytest.raises(ValueError, match="context length"):
            config.connect("textgen", context_length=invalid)
    assert metadata.calls == calls


def test_connect_discovers_loaded_lmstudio_chat_and_probes_separate_embedding(metadata):
    connection = config.connect("LM Studio")
    assert config.current_connection() is None and config.get_model_choices() == []
    with config.use_connection(connection.key):
        assert config.get_provider() is metadata.providers["lmstudio"]
        assert config.get_model_choices() == ["local-chat"]
        assert config.resolve_model() == "local-chat"
        assert config.get_max_context_tokens("local-chat") == 8192
        assert config.supports_vision("local-chat")
        embedding = config.get_embedding_settings()
        assert embedding["model"] == "nomic-embed-text" and embedding["dimensions"] == 3
        assert embedding["context_length"] == 2048
        assert embedding["document_prefix"] == "search_document: "
        assert embedding["query_prefix"] == "search_query: "
        embedding["dimensions"] = 999
        assert config.get_embedding_settings()["dimensions"] == 3
        with pytest.raises(TypeError):
            connection.model_info["local-chat"]["context_length"] = 999
    assert [call[1] for call in metadata.calls] == ["discover", "embed"]


def test_ollama_lists_installed_models_without_loading_or_guessing_maximum_context(metadata):
    connection = config.connect("ollama")
    with config.use_connection(connection):
        assert config.get_model_choices() == ["local-chat"]
        assert config.get_model_context_info("local-chat") == {
            "loaded": None, "maximum": 131072, "effective": 4096, "configured": None,
        }
        assert config.get_embedding_settings()["model"] == ""
    assert metadata.calls == [("ollama", "discover")]


def test_simultaneous_sessions_keep_same_model_name_on_different_providers(metadata):
    connections = [config.connect(name, check_embeddings=False) for name in ("lmstudio", "ollama")]
    barrier = threading.Barrier(2)

    def read(connection):
        with config.use_connection(connection):
            barrier.wait(timeout=5)
            return config.get_provider(), config.get_max_context_tokens("local-chat"), config.supports_vision("local-chat")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(read, connections))
    assert results == [(metadata.providers["lmstudio"], 8192, True),
                       (metadata.providers["ollama"], 4096, False)]
    assert config.current_connection() is None


def test_session_resolver_is_overridden_by_explicit_connection_or_disconnected_binding(metadata):
    lmstudio = config.connect("lmstudio", check_embeddings=False)
    ollama = config.connect("ollama", check_embeddings=False)
    config.set_connection_resolver(lambda: lmstudio)
    assert config.current_connection() is lmstudio
    with config.use_connection(ollama):
        assert config.get_connection_key() == ollama.key
        with config.use_connection(None):
            assert config.current_connection() is None
            with pytest.raises(RuntimeError, match="Connect"):
                config.get_provider()
        assert config.current_connection() is ollama
    assert config.current_connection() is lmstudio


def test_reconnect_revokes_old_work_without_rerouting_it(metadata):
    old = config.connect("lmstudio", check_embeddings=False)
    new = config.connect("ollama", check_embeddings=False)
    with config.use_connection(old):
        config.release_connection(old.key)
        assert config.current_connection() is old
        assert config.get_max_context_tokens("local-chat") == 8192
        with pytest.raises(RuntimeError, match="connection changed"):
            config.get_provider()
        with pytest.raises(RuntimeError, match="connection changed"):
            config.resolve_model("local-chat")
    with pytest.raises(RuntimeError, match="no longer active"):
        config.get_connection(old.key)
    with config.use_connection(new):
        assert config.get_provider() is metadata.providers["ollama"]


def test_failed_connection_preserves_existing_snapshot_and_has_no_global_side_effect(metadata):
    connection = config.connect("lmstudio", check_embeddings=False)
    metadata.providers["ollama"].discover_models = lambda: (_ for _ in ()).throw(ProviderError("offline"))
    with config.use_connection(connection):
        with pytest.raises(ProviderError, match="offline"):
            config.connect("ollama")
        assert config.get_provider() is metadata.providers["lmstudio"]
    assert not connection.revoked.is_set()
    assert list(config._CONNECTIONS) == [connection.key]


def test_multiple_embedding_candidates_require_explicit_choice_without_probe(metadata, monkeypatch):
    metadata.records["lmstudio"].append({"id": "other-embed", "type": "embedding"})
    connection = config.connect("lmstudio")
    assert connection.embedding_settings["model"] == ""
    assert [call[1] for call in metadata.calls] == ["discover"]
    monkeypatch.setenv("LMSTUDIO_EMBEDDING_MODEL", "other-embed")
    selected = config.connect("lmstudio")
    assert selected.embedding_settings["model"] == "other-embed"
    assert selected.embedding_settings["dimensions"] == 3


def test_unknown_embedding_override_is_not_sent_to_server(metadata, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_EMBEDDING_MODEL", "uninstalled")
    connection = config.connect("lmstudio")
    assert "installed embedding model" in connection.embedding_error
    assert connection.embedding_settings["model"] == ""
    assert [call[1] for call in metadata.calls] == ["discover"]


@pytest.mark.parametrize("provider_name", ["lmstudio", "ollama"])
def test_only_lmstudio_fallback_accepts_explicit_discovered_unknown_embedding(metadata, monkeypatch, provider_name):
    metadata.records[provider_name] = [
        {"id": "local-chat", "type": "unknown", "loaded": None},
        {"id": "local-embedding", "type": "unknown", "loaded": None, "identity": "local-embedding"},
    ]
    monkeypatch.setenv(provider_name.upper() + "_EMBEDDING_MODEL", "local-embedding")
    connection = config.connect(provider_name)
    if provider_name == "lmstudio":
        assert connection.embedding_error is None
        assert connection.embedding_settings["model"] == "local-embedding"
        assert connection.embedding_settings["dimensions"] == 3
        assert connection.model_choices == ("local-chat",)
        assert [call[1] for call in metadata.calls] == ["discover", "embed"]
    else:
        assert "installed embedding model" in connection.embedding_error
        assert connection.embedding_settings["model"] == ""
        assert [call[1] for call in metadata.calls] == ["discover"]


def test_session_connection_keeps_prior_lmstudio_local_v2_fingerprint(metadata):
    from rag_chatbot.config import Settings

    # These are the pre-connector LM Studio identity and preprocessing fields.
    legacy = Settings(embedding_model="nomic-embed-text", embedding_dimensions=3,
        embedding_identity="lmstudio:" + config.BASE_URL + ":nomic-q4",
        embedding_context_length=2048, document_prefix="search_document: ", query_prefix="search_query: ")
    connection = config.connect("lmstudio")
    with config.use_connection(connection):
        current = Settings()
        assert current.fingerprint() == legacy.fingerprint()
        assert connection.key not in current.embedding_identity


def test_embedding_probe_failure_leaves_chat_connected_without_usable_vectors(metadata, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_EMBEDDING_DIMENSIONS", "2")
    connection = config.connect("lmstudio")
    assert "does not match" in connection.embedding_error
    assert connection.embedding_settings["dimensions"] == 0
    with config.use_connection(connection):
        assert config.resolve_model() == "local-chat"


def test_refresh_context_preserves_embedding_fingerprint_but_not_old_budget(metadata):
    metadata.records["lmstudio"][0]["max_context_length"] = 131072
    old = config.connect("lmstudio")
    metadata.records["lmstudio"][0]["context_length"] = 131072
    with config.use_connection(old):
        new = config.refresh_models()
        assert config.get_max_context_tokens("local-chat") == 8192
    with config.use_connection(new):
        assert config.get_model_context_info("local-chat")["loaded"] == 131072
        prompt = [{"role": "user", "content": "Hello"}]
        assert tokens.reply_budget(prompt, "local-chat") == 131072 - tokens.count_tokens(prompt, "local-chat")
    assert new.embedding_settings == old.embedding_settings
    assert len([call for call in metadata.calls if call[1] == "embed"]) == 1


def test_embedding_identity_changes_invalidate_dimensions_and_do_not_cross_providers(metadata):
    first = config.connect("lmstudio")
    metadata.records["lmstudio"][-1]["identity"] = "new-revision"
    updated = config.connect("lmstudio", check_embeddings=False)
    assert updated.embedding_settings["dimensions"] == 0
    metadata.records["ollama"].append(deepcopy(metadata.records["lmstudio"][-1]))
    other = config.connect("ollama", check_embeddings=False)
    assert other.embedding_settings["identity"] != updated.embedding_settings["identity"]
    assert other.embedding_settings["dimensions"] == 0
    assert first.embedding_settings["dimensions"] == 3


@pytest.mark.parametrize("limit,expected", [(0, None), (6000, 6000), (100000, 8192)])
def test_output_limit_is_auto_or_explicit_cap(metadata, monkeypatch, limit, expected):
    monkeypatch.setattr(config, "OUTPUT_TOKEN_LIMIT", limit)
    with config.use_connection(config.connect("lmstudio", check_embeddings=False)):
        assert config.max_output_tokens("local-chat") == expected


def test_ollama_explicit_request_can_expand_loaded_context_up_to_model_maximum(metadata, monkeypatch):
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "16384")
    monkeypatch.setenv("OLLAMA_MAX_OUTPUT_TOKENS", "5000")
    metadata.records["ollama"][0].update(loaded=True, context_length=8192, max_context_length=12288)
    with config.use_connection(config.connect("ollama", check_embeddings=False)):
        assert config.get_model_context_info("local-chat") == {
            "loaded": 8192, "maximum": 12288, "effective": 12288, "configured": 16384,
        }
        assert config.max_output_tokens("local-chat") == 5000
    with config.use_connection(config.connect("lmstudio", check_embeddings=False)):
        assert config.max_output_tokens("local-chat") is None


def test_unloaded_context_fallback_cannot_exceed_small_advertised_maximum(metadata):
    metadata.records["ollama"][0]["max_context_length"] = 1024
    with config.use_connection(config.connect("ollama", check_embeddings=False)):
        assert config.get_max_context_tokens("local-chat") == 1024


def test_context_cap_is_distinct_from_loaded_window_and_model_maximum(metadata, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_CONTEXT_LENGTH", "16384")
    monkeypatch.setattr(config, "DEFAULT_CONTEXT_TOKENS", 16384)
    metadata.records["lmstudio"][0].update(context_length=131072, max_context_length=131072)
    with config.use_connection(config.connect("lmstudio", check_embeddings=False)):
        assert config.get_model_context_info("local-chat") == {
            "loaded": 131072, "maximum": 131072, "effective": 16384, "configured": 16384,
        }


def test_unknown_configured_chat_is_not_added_to_discovered_choices(metadata, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_MODEL", "not-loaded")
    with pytest.raises(ValueError, match="not available"):
        config.connect("lmstudio")
    assert config._CONNECTIONS == {}


def test_metadata_failure_does_not_register_an_incomplete_connection(metadata):
    metadata.records["lmstudio"].append(deepcopy(metadata.records["lmstudio"][0]))
    with pytest.raises(ProviderError, match="duplicate"):
        config.connect("lmstudio")
    assert config._CONNECTIONS == {}
