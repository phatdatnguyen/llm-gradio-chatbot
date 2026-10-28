"""Local embedding identities, bounded requests, and offline Unicode handling."""

from dataclasses import replace
import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from chatbot import config, tokens
from rag_chatbot.config import Settings
from rag_chatbot.service import RAGService
from rag_chatbot.store import LibraryStore
from rag_chatbot.tools.evidence import bounded_batches, processing_budget
from rag_chatbot.tools import evidence


def test_embedded_qdrant_is_default_and_server_override_stays_local(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_LOCAL_PATH", raising=False)
    assert Settings().qdrant_local_path == str(tmp_path / "data" / ".cache" / "qdrant")
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    assert Settings().qdrant_local_path is None
    monkeypatch.setenv("QDRANT_URL", "https://example.org")
    with pytest.raises(ValueError, match="loopback"):
        Settings()


def test_custom_catalogs_derive_isolated_vector_paths_and_honor_explicit_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "unrelated-default"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_LOCAL_PATH", raising=False)
    first = Settings(data_dir=tmp_path / "first")
    second = Settings(data_dir=tmp_path / "second")
    assert first.qdrant_local_path == str(tmp_path / "first" / ".cache" / "qdrant")
    assert second.qdrant_local_path == str(tmp_path / "second" / ".cache" / "qdrant")
    assert first.qdrant_local_path != second.qdrant_local_path
    assert Settings(data_dir=tmp_path, qdrant_local_path=None).qdrant_local_path is None
    monkeypatch.setenv("QDRANT_LOCAL_PATH", str(tmp_path / "explicit-vectors"))
    assert Settings(data_dir=tmp_path).qdrant_local_path == str(tmp_path / "explicit-vectors")


@pytest.mark.parametrize("legacy", [False, True])
def test_reindex_uses_new_chunk_cache_directory_without_changing_prior_original(tmp_path, legacy):
    settings = Settings(data_dir=tmp_path / "workspace", embedding_model="local-embed",
                        embedding_dimensions=3, chunk_size=80, chunk_overlap=8)
    source = tmp_path / "evidence.txt"
    source.write_text("Evidence about measurements and context. " * 20, encoding="utf-8")
    original = source.read_bytes()
    store = LibraryStore(settings, vector_client=QdrantClient(location=":memory:"))
    initial = RAGService(settings, store=store, embedder=lambda texts: [[1.0, 0.2, 0.3] for _ in texts])
    try:
        assert "1 indexed" in list(initial.ingest([source], "library"))[-1]
        first = store.list_documents()[0]
        prior_path = Path(first["path"])
        cache_files = list(prior_path.parent.glob("chunks-*.json"))
        prior_caches = {path: path.read_bytes() for path in cache_files}
        assert prior_caches
        if legacy:
            # A pre-migration catalog row has the old cloud index fingerprint.
            with store._db:
                store._db.execute("UPDATE documents SET fingerprint=? WHERE id=?",
                                  ("v1:text-embedding-3-small:1536:600:80", first["id"]))
    finally:
        store.close()

    changed = settings if legacy else replace(settings, chunk_size=32, chunk_overlap=4)
    calls = []
    replacement = LibraryStore(changed, vector_client=QdrantClient(location=":memory:"))
    service = RAGService(changed, store=replacement,
                         embedder=lambda texts: calls.extend(texts) or [[1.0, 0.2, 0.3] for _ in texts])
    try:
        assert "1 indexed" in list(service.ingest([source], "library"))[-1]
        current = next(row for row in replacement.list_documents()
                       if row["fingerprint"] == changed.fingerprint())
        assert current["id"] != first["id"]
        assert Path(current["path"]).parent != prior_path.parent
        assert calls and all(len(text.encode("utf-8")) <= changed.chunk_size for text in calls)
        assert prior_path.read_bytes() == source.read_bytes() == original
        assert all(path.read_bytes() == content for path, content in prior_caches.items())
    finally:
        replacement.close()


def test_embedding_fingerprint_separates_models_prefixes_and_old_cloud_vectors(tmp_path):
    settings = Settings(data_dir=tmp_path, embedding_model="local-embed", embedding_dimensions=3,
                        embedding_identity="lmstudio:local-embed:revision-1")
    assert settings.fingerprint().startswith("local-v2:")
    variants = [replace(settings, embedding_identity="lmstudio:local-embed:revision-2"),
                replace(settings, embedding_dimensions=4),
                replace(settings, document_prefix="search_document: "),
                replace(settings, query_prefix="search_query: "),
                replace(settings, embedding_context_length=4096),
                replace(settings, chunk_size=300)]
    assert len({settings.fingerprint(), *(item.fingerprint() for item in variants)}) == len(variants) + 1


def test_missing_embedding_configuration_fails_before_staging(tmp_path):
    settings = Settings(data_dir=tmp_path / "workspace", embedding_model="", embedding_dimensions=0)
    service = RAGService(settings)
    source = tmp_path / "source.txt"
    source.write_text("Local document", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="Configure a local embedding model"):
            list(service.ingest([source], "library"))
        assert service.store.list_documents() == []
        assert service.store._vectors is None
    finally:
        service.store.close()


def test_document_and_query_prefixes_are_applied_once(tmp_path):
    calls = []
    settings = Settings(data_dir=tmp_path, embedding_model="local-embed", embedding_dimensions=3,
                        document_prefix="search_document: ", query_prefix="search_query: ")
    service = RAGService(settings, embedder=lambda texts: calls.append(texts) or [[1.0, 0.2, 0.3]])
    try:
        service._vectors(["Evidence"])
        service._vectors(["Question"], query=True)
        assert calls == [["search_document: Evidence"], ["search_query: Question"]]
    finally:
        service.store.close()


def test_followup_query_preserves_unicode_and_fits_embedding_context(tmp_path):
    settings = Settings(data_dir=tmp_path, embedding_context_length=128, query_prefix="search_query: ")
    service = RAGService(settings, embedder=lambda texts: [])
    try:
        result = service._query("Why?", [{"role": "user", "content": "Việt Nam 😀 " * 100}])
        assert result.startswith("Current question: Why?")
        assert "\ufffd" not in result
        assert tokens.count_text_tokens(settings.query_prefix + result, "local") <= 96
    finally:
        service.store.close()


def test_small_context_batches_preserve_every_item(monkeypatch):
    monkeypatch.setattr(config, "get_max_context_tokens", lambda model: 4096)
    monkeypatch.setattr(tokens, "reserve_for_output", lambda model: 1024)
    budget, output = processing_budget("local", overhead=2048)
    assert (budget, output) == (1024, 1024)
    items = [{"source_id": str(index), "text": "Evidence " * 45} for index in range(10)]
    batches = bounded_batches(items, "local", budget=budget)
    assert len(batches) > 1
    assert [item for batch in batches for item in batch] == items


def test_tool_output_budget_is_independent_of_ordinary_chat_cap(monkeypatch):
    monkeypatch.setattr(config, "get_max_context_tokens", lambda model: 131072)
    monkeypatch.setattr(config, "max_output_tokens", lambda model: 64)
    _, output = processing_budget("local")
    assert output == 32768


def test_large_context_uses_remaining_capacity_without_a_six_thousand_token_ceiling(monkeypatch):
    monkeypatch.setattr(config, "get_max_context_tokens", lambda model: 131072)
    monkeypatch.setattr(evidence, "MAX_INPUT_TOKENS", 0)
    available, output = processing_budget("local", overhead=4096)
    assert output == 32768
    assert available == 131072 - 32768 - 4096
    items = [{"source_id": str(index), "text": "Measured evidence. " * 50} for index in range(40)]
    batches = bounded_batches(items, "local", budget=available)
    assert batches == [items]


def test_explicit_input_cap_can_lower_automatic_context_budget(monkeypatch):
    monkeypatch.setattr(config, "get_max_context_tokens", lambda model: 131072)
    monkeypatch.setattr(evidence, "MAX_INPUT_TOKENS", 6000)
    available, output = processing_budget("local", overhead=4096)
    assert available == 6000
    assert output == 32768


def test_impossible_tool_context_fails_before_generation(monkeypatch):
    monkeypatch.setattr(config, "get_max_context_tokens", lambda model: 1024)
    monkeypatch.setattr(tokens, "reserve_for_output", lambda model: 256)
    with pytest.raises(ValueError, match="context is too small"):
        processing_budget("local", overhead=800)


def test_physical_storage_pool_respects_platform_path_case(tmp_path):
    from rag_chatbot import service

    first = Settings(data_dir=tmp_path / "Library", qdrant_local_path=str(tmp_path / "Vectors"))
    second = replace(first, data_dir=tmp_path / "library", qdrant_local_path=str(tmp_path / "vectors"))
    first_handle = service._shared_vector_client(first)
    second_handle = service._shared_vector_client(second)
    try:
        same_location = os.name == "nt"
        assert (first_handle.resource is second_handle.resource) is same_location
        assert (service._store_identity(first) == service._store_identity(second)) is same_location
        # Relative traversal to the same physical location still shares a lease.
        alias = replace(first, qdrant_local_path=str(tmp_path / "unused" / ".." / "Vectors"))
        alias_handle = service._shared_vector_client(alias)
        try:
            assert alias_handle.resource is first_handle.resource
        finally:
            alias_handle.close()
    finally:
        first_handle.close()
        second_handle.close()
