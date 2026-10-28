"""Runtime connector changes keep embedding profiles, jobs, and vector owners isolated."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chatbot import config
from rag_chatbot import service, workspaces
from rag_chatbot.config import Settings


class Provider:
    def __init__(self, name):
        self.name = name
        self.embedded = []
        self.chats = []
        self.after_embedding = None

    def embeddings(self, model, texts):
        self.embedded.append((model, list(texts)))
        if self.after_embedding:
            self.after_embedding()
        return [[1.0, 0.2, 0.1] for _ in texts]

    def chat(self, payload):
        self.chats.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": "Measured value 10 nm [1]."},
                             "finish_reason": "stop"}]}


def connection(name, key):
    provider = Provider(name)
    return config.Connection(key=key, provider_name=name, base_url=f"http://127.0.0.1/{name}",
        provider=provider, records=(), model_choices=("same-chat-name",), default_model="same-chat-name",
        model_info={"same-chat-name": {"vision": True, "identity": name + "-chat-digest"}},
        context_limits={"same-chat-name": 8192}, output_limits={"same-chat-name": None},
        embedding_settings={"model": "same-embedding-name", "dimensions": 3, "context_length": 2048,
                            "identity": name + ":endpoint:model-digest", "document_prefix": name + "-doc: ",
                            "query_prefix": name + "-query: "},
        embedding_error=None, default_context=8192, configured_context=None, output_limit=0, image_budget=256)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "libraries"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_LOCAL_PATH", raising=False)
    monkeypatch.setattr(service, "_SERVICES", {})
    monkeypatch.setattr(service, "_INDEX_LOCKS", {})
    monkeypatch.setattr(service, "_VECTOR_RESOURCES", {})
    with config.use_connection(None):
        workspaces.create_workspace("Research")
    yield "Research"
    stores = {id(item.store): item.store for item in service._SERVICES.values()}
    for store in stores.values():
        store.close()
    assert service._VECTOR_RESOURCES == {}


def bound_service(workspace, snapshot):
    with config.use_connection(snapshot):
        return service.get_service(workspace)


def test_settings_read_embedding_metadata_once_and_disconnected_browsing_stays_local(workspace, monkeypatch):
    calls = []

    def metadata():
        calls.append(True)
        return {"model": "embed", "dimensions": 3, "identity": "one-snapshot",
                "context_length": 1024, "document_prefix": "document: ", "query_prefix": "query: "}

    monkeypatch.setattr(config, "get_embedding_settings", metadata)
    settings = Settings()
    assert len(calls) == 1
    assert (settings.embedding_model, settings.embedding_dimensions, settings.embedding_identity,
            settings.embedding_context_length, settings.document_prefix, settings.query_prefix) == (
                "embed", 3, "one-snapshot", 1024, "document: ", "query: ")
    with config.use_connection(None):
        local = service.get_service(workspace)
        assert local.store.list_documents() == []
        assert local.store._vectors is None
        assert service._VECTOR_RESOURCES == {}


def test_connector_profiles_reindex_without_restart_or_cross_provider_vectors(workspace, tmp_path):
    lm, ollama = connection("lmstudio", "lm-session"), connection("ollama", "ollama-session")
    first, second = bound_service(workspace, lm), bound_service(workspace, ollama)
    assert first.settings.fingerprint() != second.settings.fingerprint()
    assert first.store is not second.store
    source = tmp_path / "measurement.txt"
    source.write_text("Measured value 10 nm.", encoding="utf-8")
    # Calling an old service while another connection is bound must still use
    # its own provider, prefixes, metadata, and vector fingerprint.
    with config.use_connection(ollama):
        assert "1 indexed" in list(first.ingest([source], "science"))[-1]
    original = first.store.list_documents()[0]
    preserved_path = Path(original["path"])
    with pytest.raises(ValueError, match="not indexed"):
        second.store.snapshot_sources([original["id"]], "science")
    with config.use_connection(lm):
        assert "1 indexed" in list(second.ingest([source], "science"))[-1]
    current = next(row for row in second.store.list_documents()
                   if row["fingerprint"] == second.settings.fingerprint())
    assert current["id"] != original["id"]
    assert preserved_path.read_bytes() == source.read_bytes()
    assert first.store.settings is first.settings and second.store.settings is second.settings
    assert lm.provider.embedded[0] == ("same-embedding-name", ["lmstudio-doc: Measured value 10 nm."])
    assert ollama.provider.embedded[0] == ("same-embedding-name", ["ollama-doc: Measured value 10 nm."])
    assert first.store.vector_client.resource is second.store.vector_client.resource
    assert first.store.ensure_index() != second.store.ensure_index()
    with config.use_connection(ollama):
        result = list(first.answer("What was measured?", [], "science", [original["id"]],
                                   stream_output=False))[-1]
    assert result[0][-1]["sources"][0]["document_id"] == original["id"]
    assert len(lm.provider.chats) == 1 and not ollama.provider.chats
    # Closing one profile releases only its lease on the shared embedded DB.
    first.store.close()
    assert second.store.search("Measured", [1.0, 0.2, 0.1], "science", [current["id"]])


def test_compatible_reconnect_reuses_profile_without_mutating_previous_service(workspace):
    first_connection = connection("lmstudio", "first-session")
    next_connection = replace(first_connection, key="next-session", provider=Provider("new-client"))
    first = bound_service(workspace, first_connection)
    second = bound_service(workspace, next_connection)
    assert first is not second and first.store is second.store
    assert first.connection is first_connection and second.connection is next_connection
    assert first.settings.fingerprint() == second.settings.fingerprint()
    second._vectors(["Test"])
    assert not first_connection.provider.embedded
    assert next_connection.provider.embedded == [("same-embedding-name", ["lmstudio-doc: Test"])]


def test_workspace_index_lock_and_revocation_prevent_stale_publication(workspace, tmp_path):
    old_connection, new_connection = connection("lmstudio", "old"), connection("ollama", "new")
    old, new = bound_service(workspace, old_connection), bound_service(workspace, new_connection)
    source = tmp_path / "data.txt"
    source.write_text("Measured value 10 nm.", encoding="utf-8")
    pending = old.ingest([source], "science")
    assert next(pending).startswith("Preparing")
    try:
        with pytest.raises(RuntimeError, match="already running"):
            list(new.ingest([source], "science"))
        new.cancel_ingestion()
        assert not old._cancel_indexing.is_set()
        old_connection.provider.after_embedding = old_connection.revoked.set
        with pytest.raises(RuntimeError, match="connection changed"):
            list(pending)
    finally:
        pending.close()
    assert not old.is_indexing and not new.is_indexing
    assert all(row["status"] != "ready" for row in old.store.list_documents())
    assert len(old_connection.provider.embedded) == 1 and not new_connection.provider.embedded
    assert "1 indexed" in list(new.ingest([source], "science"))[-1]
    with config.use_connection(new_connection), pytest.raises(RuntimeError, match="connection changed"):
        old._vectors(["Must not be rerouted"])
    assert len(old_connection.provider.embedded) == len(new_connection.provider.embedded) == 1


def test_stream_steps_restore_worker_context_and_close_on_revocation(workspace, tmp_path):
    selected, other = connection("lmstudio", "stream-owner"), connection("ollama", "other-session")
    rag = bound_service(workspace, selected)
    source = tmp_path / "measurement.txt"
    source.write_text("Measured value 10 nm.", encoding="utf-8")
    list(rag.ingest([source], "science"))
    document = rag.store.list_documents()[0]
    events = []

    def stream(*args, **kwargs):
        try:
            events.append(("started", config.get_connection_key()))
            yield "Measured value"
            pytest.fail("A revoked stream must not request another delta")
        finally:
            events.append(("closed", config.get_connection_key()))

    rag.streamer = stream
    iterator = rag.answer("What was measured?", [], "science", [document["id"]])

    def advance():
        with config.use_connection(other):
            try:
                return next(iterator)
            finally:
                assert config.current_connection() is other

    # Gradio can resume the same generator on a different worker thread.
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(advance).result()[0][-1]["content"] == "Measured value"
    selected.revoked.set()
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(RuntimeError, match="connection changed"):
            worker.submit(advance).result()
    assert events == [("started", selected.key), ("closed", selected.key)]
