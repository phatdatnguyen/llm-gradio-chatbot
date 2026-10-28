"""Local embedding and vision API boundaries."""
from types import SimpleNamespace

import pytest

from chatbot import api


def test_embedding_arguments_preserve_text_and_expect_dimensions(monkeypatch):
    calls = []
    def embeddings(model, texts):
        calls.append((model, texts))
        return [[1., 0.], [0., 1.]]
    monkeypatch.setattr(api.config, "get_provider", lambda: SimpleNamespace(embeddings=embeddings))
    assert api.embed_texts(["first", "second"], "local-embedding", 2) == [[1., 0.], [0., 1.]]
    assert calls == [("local-embedding", ["first", "second"])]


@pytest.mark.parametrize("vectors", [[], [[1.]], [[float("nan"), 1.]], [[0., 0.]], [[True, 1.]]])
def test_bad_embedding_responses_are_rejected(monkeypatch, vectors):
    monkeypatch.setattr(api.config, "get_provider", lambda: SimpleNamespace(embeddings=lambda *args: vectors))
    with pytest.raises(RuntimeError):
        api.embed_texts(["one"], "local-embedding", dimensions=2)


def test_embedding_dimensions_can_be_discovered(monkeypatch):
    monkeypatch.setattr(api.config, "get_provider", lambda: SimpleNamespace(embeddings=lambda *args: [[1., 2., 3.]]))
    assert api.embed_texts(["one"], "local-embedding") == [[1., 2., 3.]]


@pytest.mark.parametrize("texts", [[], [""], [" "], [None]])
def test_empty_embedding_input_does_not_call_provider(monkeypatch, texts):
    monkeypatch.setattr(api.config, "get_provider", lambda: pytest.fail("No provider access"))
    with pytest.raises(ValueError):
        api.embed_texts(texts)


def test_missing_embedding_configuration_is_actionable(monkeypatch):
    monkeypatch.setattr(api.config, "DEFAULT_EMBEDDING_MODEL", "")
    monkeypatch.setattr(api.config, "get_provider", lambda: pytest.fail("No provider access"))
    with pytest.raises(ValueError, match="Select a local embedding model"):
        api.embed_texts(["one"])


def test_vision_uses_shared_local_chat_and_local_image(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "image_file_to_data_url", lambda path: "data:image/png;base64,AA==")
    monkeypatch.setattr(api, "send_chat", lambda *args, **kwargs: calls.append((args, kwargs)) or "Table: 42")
    assert api.describe_document_image("page.png", model="local-vision") == "Table: 42"
    args, kwargs = calls[0]
    assert args[0] == "local-vision"
    assert args[1][0]["image_url"].startswith("data:image/")
    assert "Do not invent" in args[1][0]["content"]
    assert kwargs["system_prompt"]
    assert kwargs["allow_partial"] is False


def test_vision_length_is_not_accepted_as_indexable_transcription(monkeypatch):
    calls = []
    def chat(payload):
        calls.append(payload)
        return {"choices": [{"finish_reason": "length", "message": {"content": "Only the first page section"}}]}
    monkeypatch.setattr(api, "image_file_to_data_url", lambda path: "data:image/png;base64,AA==")
    monkeypatch.setattr(api.config, "resolve_model", lambda value: "local-vision")
    monkeypatch.setattr(api.config, "supports_vision", lambda value: True)
    monkeypatch.setattr(api.config, "get_max_context_tokens", lambda *args: 16384)
    monkeypatch.setattr(api.config, "get_provider", lambda: SimpleNamespace(chat=chat))
    with pytest.raises(RuntimeError, match="incomplete"):
        api.describe_document_image("page.png", model="local-vision")
    assert len(calls) == 1
