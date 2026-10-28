"""Explicit connection actions, per-session routing, and disconnected startup."""
from threading import Event, Thread
from types import SimpleNamespace

import gradio as gr
from gradio.context import LocalContext
import pytest

from chatbot import api, config, connections
from chatbot.providers.base import ProviderError


@pytest.fixture
def providers(monkeypatch):
    calls = []
    def provider(name):
        def discover():
            calls.append((name, "discover"))
            return [{"id": model, "type": "llm", "loaded": True,
                     "context_length": None if name == "textgen" else 16384,
                     "max_context_length": None if name == "textgen" else 32768,
                     "vision": name not in ("textgen", "koboldcpp"), "structured_output": True,
                     "identity": name + model} for model in ("shared", "second")]
        return SimpleNamespace(name=name, discover_models=discover)
    found = {name: provider(name) for name in ("lmstudio", "ollama", "textgen", "koboldcpp")}
    monkeypatch.setattr(config, "_create_provider", lambda name, options: found[name])
    monkeypatch.setattr(config, "_CONNECTIONS", {})
    monkeypatch.setattr(config, "_CONNECTION_RESOLVER", connections.request_connection)
    monkeypatch.setattr(connections, "_SESSIONS", {})
    monkeypatch.setattr(config, "MODEL_CHOICES", [])
    monkeypatch.setattr(config, "DEFAULT_MODEL", "")
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("TEXTGEN_MODEL", raising=False)
    monkeypatch.delenv("KOBOLDCPP_MODEL", raising=False)
    request_token = LocalContext.request.set(None)
    binding = config._CURRENT_CONNECTION.set(config._UNBOUND)
    yield found, calls
    config._CURRENT_CONNECTION.reset(binding)
    LocalContext.request.reset(request_token)


def connect(name, session="first", **kwargs):
    return connections.connect_connection(name, request=gr.Request(session_hash=session), **kwargs)


def test_startup_and_radio_selection_do_not_contact_either_server(providers, monkeypatch):
    import webui
    app = webui.create_app()
    tabs = [item["props"]["label"] for item in app.config["components"] if item["type"] == "tabitem"]
    assert tabs[0] == "Connections"
    assert app.connections_tab.connect.value == "Connect"
    assert app.chat_tab.llm_model.value is None and app.rag_tab.model.value is None
    assert app.chat_tab.llm_model.choices == app.rag_tab.model.choices == []
    assert all(block.fn is not connections.connect_connection for block in app.fns.values()
               if any(target[1] == "load" for target in block.targets))
    assert "11434" in connections.select_connector("ollama")
    assert "5000" in connections.select_connector("textgen")
    assert "5001" in connections.select_connector("koboldcpp")
    assert connections.show_embedding_connector("textgen")["visible"]
    assert connections.show_embedding_connector("koboldcpp")["visible"]
    assert not connections.show_textgen_context("koboldcpp")["visible"]
    assert connections.show_textgen_context("textgen")["visible"]
    assert app.connections_tab.embedding_connector.value == "none"
    assert providers[1] == []
    with pytest.raises(RuntimeError, match="Connect"):
        config.get_provider()


def test_connect_contacts_only_selected_server_and_updates_both_models(providers):
    state = {"answer_revision": 2, "tool_job": {"id": "old", "cancelled": False}}
    status, chat, rag = connect("ollama", operation_state=state)
    assert providers[1] == [("ollama", "discover")]
    assert "Connected to **Ollama**" in status and "embedding model" in status
    assert chat["choices"] == rag["choices"] == ["shared", "second"]
    assert chat["value"] == rag["value"] == "shared"
    assert state["answer_revision"] == 3 and state["tool_job"]["cancelled"]


@pytest.mark.parametrize("provider_name", ["textgen", "koboldcpp"])
def test_connect_uses_only_explicit_embedding_selection_and_appropriate_context(providers, provider_name):
    status, _, _ = connect(provider_name, textgen_context=131072)
    assert providers[1] == [(provider_name, "discover")]
    assert "RAG embeddings are disabled" in status
    assert ("131,072" if provider_name == "textgen" else "16,384") in status
    providers[1].clear()
    connect(provider_name, embedding_connector="ollama", textgen_context=131072)
    assert providers[1] == [(provider_name, "discover"), ("ollama", "discover")]
    connection = connections._SESSIONS["first"]["connection"]
    assert connection.embedding_provider_name == "ollama"
    assert connection.configured_context == (131072 if provider_name == "textgen" else None)


def test_sessions_with_identical_model_names_keep_independent_connections(providers):
    connect("lmstudio", "first")
    connect("ollama", "second")
    connect("textgen", "third")
    connect("koboldcpp", "fourth")
    for session, expected in (("first", "lmstudio"), ("second", "ollama"), ("third", "textgen"), ("fourth", "koboldcpp")):
        token = LocalContext.request.set(gr.Request(session_hash=session))
        try:
            assert config.resolve_model("shared") == "shared"
            assert config.get_provider() is providers[0][expected]
        finally:
            LocalContext.request.reset(token)
    assert config.get_model_choices() == []


def test_disconnected_event_does_not_become_connected_midflight(providers):
    token = LocalContext.request.set(gr.Request(session_hash="first"))
    try:
        assert config.current_connection() is None
        connect("ollama")
        with pytest.raises(RuntimeError, match="Connect"):
            config.get_provider()
    finally:
        LocalContext.request.reset(token)


def test_reconnect_preserves_models_but_connector_change_uses_new_default(providers):
    connect("lmstudio")
    _, chat, rag = connect("lmstudio", chat_model="second", rag_model="shared")
    assert (chat["value"], rag["value"]) == ("second", "shared")
    _, chat, rag = connect("ollama", chat_model="second", rag_model="second")
    assert chat["value"] == rag["value"] == "shared"


def test_failed_connect_is_disconnected_and_does_not_fall_back(providers):
    connect("lmstudio")
    old = connections._SESSIONS["first"]["connection"]
    providers[0]["ollama"].discover_models = lambda: (_ for _ in ()).throw(ProviderError("Server stopped"))
    status, chat, rag = connect("ollama")
    assert "Disconnected" in status and "Server stopped" in status
    assert chat["choices"] == rag["choices"] == []
    assert old.revoked.is_set() and connections._SESSIONS["first"]["connection"] is None


def test_late_connect_cannot_replace_a_newer_connection(providers):
    entered, finish = Event(), Event()
    discover = providers[0]["lmstudio"].discover_models
    def slow():
        entered.set()
        assert finish.wait(5)
        return discover()
    providers[0]["lmstudio"].discover_models = slow
    results = []
    worker = Thread(target=lambda: results.append(connect("lmstudio")))
    worker.start()
    try:
        assert entered.wait(5)
        connect("ollama")
    finally:
        finish.set()
        worker.join(5)
    assert results == [(gr.skip(),) * 3]
    assert connections._SESSIONS["first"]["connection"].provider_name == "ollama"
    assert len(config._CONNECTIONS) == 1


def test_reconnect_closes_old_stream_without_rerouting(providers):
    closed = []
    def stream(payload):
        try:
            yield {"choices": [{"delta": {"content": "First"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"content": "Old"}, "finish_reason": "stop"}]}
        finally:
            closed.append(True)
    providers[0]["lmstudio"].stream_chat = stream
    providers[0]["ollama"].stream_chat = lambda payload: pytest.fail("Old requests cannot switch providers")
    connect("lmstudio")
    request = gr.Request(session_hash="first")
    token = LocalContext.request.set(request)
    try:
        response = api.stream_chat("shared", [{"role": "user", "content": "Hello"}])
        assert next(response) == "First"
        connect("ollama")
        with pytest.raises(RuntimeError, match="changed|Connect|reconnect"):
            next(response)
        assert closed == [True]
    finally:
        LocalContext.request.reset(token)


def test_unload_releases_only_its_session(providers):
    connect("lmstudio", "first")
    connect("ollama", "second")
    first = connections._SESSIONS["first"]["connection"]
    connections.release_session(gr.Request(session_hash="first"))
    assert first.revoked.is_set()
    assert connections._SESSIONS["second"]["connection"].provider_name == "ollama"
