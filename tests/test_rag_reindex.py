"""Reindex retained originals without losing source identity or crossing profiles."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from chatbot import config
from rag_chatbot.config import Settings
from rag_chatbot.service import RAGService
from rag_chatbot.store import LibraryStore


@pytest.fixture
def library(tmp_path):
    stores, calls = [], []
    client = QdrantClient(path=str(tmp_path / "vectors"))
    base = Settings(data_dir=tmp_path / "library", embedding_model="old-embedding",
                    embedding_dimensions=3, embedding_identity="old-profile",
                    chunk_size=200, chunk_overlap=8)

    def build(settings):
        store = LibraryStore(settings, vector_client=client)
        stores.append(store)

        def embed(texts):
            calls.append((settings.embedding_identity, list(texts)))
            return [[1.0] + [0.2] * (settings.embedding_dimensions - 1) for _ in texts]

        with config.use_connection(None):
            return RAGService(settings, store=store, embedder=embed)

    old = build(base)
    first = tmp_path / "report.txt"
    first.write_text("Cedar measured value 10 nm. " * 18, encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("Unselected source remains unchanged.", encoding="utf-8")
    assert "2 indexed" in list(old.ingest([first, second], "Science", display_names=["Report.txt", None]))[-1]
    documents = {row["name"]: row for row in old.store.list_documents("Science")}
    current = build(replace(base, embedding_model="new-embedding", embedding_dimensions=4,
                            embedding_identity="new-profile", chunk_size=64))
    calls.clear()
    yield old, current, documents["Report.txt"], documents["second.txt"], calls, first
    for store in stores:
        store.close()


def test_reindex_changes_only_selected_profile_and_reuses_compatible_caches(library):
    old, current, selected, untouched, calls, _ = library
    original = Path(selected["path"])
    saved_bytes = original.read_bytes()
    before_caches = set(original.parent.glob("chunks-*.json"))
    assert "1 indexed" in list(current.reindex([selected["id"]], "Science"))[-1]
    updated = current.store.get_document(selected["id"])
    for field in ("id", "name", "path", "collection", "content_hash"):
        assert updated[field] == selected[field]
    assert original.read_bytes() == saved_bytes
    assert len(current.store.list_documents()) == 2
    assert current.store.get_document(untouched["id"]) == untouched
    assert updated["status"] == "ready" and updated["fingerprint"] == current.settings.fingerprint()
    assert updated["dimensions"] == 4 and updated["chunk_count"] > selected["chunk_count"]
    assert calls and {profile for profile, _ in calls} == {"new-profile"}
    assert not old.store.search("Cedar", None, "Science", [selected["id"]])
    assert current.store.search("Cedar", [1.0, 0.2, 0.2, 0.2], "Science", [selected["id"]])
    after_caches = set(original.parent.glob("chunks-*.json"))
    assert before_caches < after_caches
    calls.clear()
    assert "1 indexed" in list(current.reindex([selected["id"]], "Science"))[-1]
    assert calls == []
    assert current.store.get_document(selected["id"])["generation"] != updated["generation"]


@pytest.mark.parametrize("invalid", ["unknown", "other_collection", "missing", "changed", "catalog"])
def test_complete_reindex_selection_is_checked_before_model_work(library, invalid):
    old, current, valid, bad, calls, _ = library
    selected = [valid["id"], bad["id"]]
    if invalid == "unknown":
        selected[-1] = "not-in-this-workspace"
    elif invalid == "other_collection":
        with old.store._db:
            old.store._db.execute("UPDATE documents SET collection='Other' WHERE id=?", (bad["id"],))
    elif invalid == "missing":
        Path(bad["path"]).unlink()
    elif invalid == "changed":
        Path(bad["path"]).write_text("Changed original", encoding="utf-8")
    else:
        with old.store._db:
            old.store._db.execute("UPDATE documents SET path='library.sqlite3' WHERE id=?", (bad["id"],))
    with pytest.raises(ValueError):
        list(current.reindex(selected, "Science"))
    assert calls == []
    assert current.store.get_document(valid["id"]) == valid


def test_reindex_rejects_symlink_escape_before_model_work(library, tmp_path):
    _, current, valid, bad, calls, _ = library
    source = Path(bad["path"])
    external = tmp_path / "outside.txt"
    external.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(external)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="sources directory|data directory"):
        list(current.reindex([valid["id"], bad["id"]], "Science"))
    assert calls == []
    assert current.store.get_document(valid["id"]) == valid


def test_reindex_collision_preserves_both_existing_catalog_entries(library):
    _, current, original, _, calls, upload = library
    assert "1 indexed" in list(current.ingest([upload], "Science"))[-1]
    before = current.store.list_documents()
    calls.clear()
    with pytest.raises(ValueError, match="another copy.*current embedding"):
        list(current.reindex([original["id"]], "Science"))
    assert current.store.list_documents() == before
    assert all(Path(document["path"]).is_file() for document in before)
    assert calls == []


def test_cancelled_reindex_stays_unpublished_and_resumes_cached_batches(library):
    old, current, document, _, calls, _ = library
    progress = current.reindex([document["id"]], "Science")
    assert next(progress).startswith("Preparing")
    assert not old.store.search("Cedar", None, "Science", [document["id"]])
    assert next(progress).startswith("Embedding")
    current.cancel_ingestion()
    assert "Indexing stopped" in list(progress)[-1]
    failed = current.store.get_document(document["id"])
    assert failed["status"] == "failed" and not failed["generation"]
    assert not current.store.search("Cedar", None, "Science", [document["id"]])
    calls.clear()
    assert "1 indexed" in list(current.reindex([document["id"]], "Science"))[-1]
    assert not current.is_indexing and calls == []


def test_deleted_snapshot_cannot_be_reindexed_or_resurrected(library):
    _, current, document, _, calls, _ = library
    snapshot = current.store.prepare_reindex([document["id"]], "Science")[0]
    current.store.delete_document(document["id"])
    with pytest.raises(ValueError, match="changed before reindexing"):
        current.store.stage_reindex(snapshot)
    assert current.store.get_document(document["id"])["status"] == "deleted"
    assert Path(document["path"]).is_file() and not calls


def test_stop_during_original_preflight_does_not_stage_or_embed(library, monkeypatch):
    _, current, document, _, calls, _ = library
    verify = current.store.verify_reindex_original

    def stop_after_hash(snapshot):
        source = verify(snapshot)
        current.cancel_ingestion()
        return source

    monkeypatch.setattr(current.store, "verify_reindex_original", stop_after_hash)
    with pytest.raises(RuntimeError, match="Indexing stopped"):
        list(current.reindex([document["id"]], "Science"))
    assert current.store.get_document(document["id"]) == document
    assert not calls and not current.is_indexing


def _restore_legacy_cache(document):
    directory = Path(document["path"]).parent
    marker = directory / "indexed-profile.json"
    profile = json.loads(marker.read_text(encoding="utf-8"))
    current_key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:20]
    profile.pop("fingerprint")
    profile["version"] = 3
    legacy_key = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:20]
    (directory / f"chunks-{current_key}.json").replace(directory / f"chunks-{legacy_key}.json")
    marker.write_text(json.dumps(profile), encoding="utf-8")
    (directory / "legacy-chunk-fingerprint.json").unlink()


def test_compatible_legacy_vision_profile_and_chunks_remain_reusable(library, monkeypatch):
    old, _, document, _, calls, upload = library
    old._injected_vision = True
    assert "1 indexed" in list(old.ingest([upload], "Science", use_vision=True))[-1]
    _restore_legacy_cache(old.store.get_document(document["id"]))
    monkeypatch.setattr("rag_chatbot.service.extract_elements",
                        lambda *args, **kwargs: pytest.fail("Compatible legacy chunks should be reused"))
    calls.clear()
    assert "1 unchanged" in list(old.ingest([upload], "Science", use_vision=True))[-1]
    assert "1 indexed" in list(old.reindex([document["id"]], "Science", use_vision=True))[-1]
    assert not calls


def test_interrupted_profile_change_never_adopts_legacy_chunks_under_new_fingerprint(library):
    _, current, document, _, calls, _ = library
    _restore_legacy_cache(document)
    pending = current.reindex([document["id"]], "Science")
    assert next(pending).startswith("Preparing")
    pending.close()
    assert current.store.get_document(document["id"])["fingerprint"] == current.settings.fingerprint()
    assert current.store.get_document(document["id"])["status"] == "failed"
    assert "1 indexed" in list(current.reindex([document["id"]], "Science"))[-1]
    assert current.store.get_document(document["id"])["chunk_count"] > document["chunk_count"]
    assert calls and all(len(text.encode("utf-8")) <= current.settings.chunk_size
                         for _, batch in calls for text in batch)
