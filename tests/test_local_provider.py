"""Wire-level local adapter tests run without a server or network connection."""
import json

import httpx
import pytest

from chatbot.providers import LMStudioProvider, ProviderError
from chatbot.providers import lmstudio
from chatbot.providers.lmstudio import validate_base_url


def provider(monkeypatch, handler):
    backend = LMStudioProvider("http://localhost:1234/v1")
    monkeypatch.setattr(backend, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False))
    return backend


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:1234/v1/", "http://127.0.0.1:1234"),
    ("http://127.0.0.1:1234", "http://127.0.0.1:1234"),
    ("https://192.168.1.5:1234/api/v1", "https://192.168.1.5:1234"),
    ("http://[::1]:1234", "http://[::1]:1234"),
])
def test_only_explicit_local_endpoints_are_normalized(url, expected):
    assert validate_base_url(url) == expected


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1", "http://8.8.8.8:1234", "http://model.example.test",
    "http://0.0.0.0:1234", "file:///model", "http://user:password@localhost:1234",
    "http://localhost:1234?url=https://example.org", "http://localhost:1234/other/path",
    "http://127.0.0.1:invalid", "http://169.254.169.254", "http://[::ffff:8.8.8.8]",
])
def test_public_or_ambiguous_endpoints_are_rejected(url):
    with pytest.raises(ValueError):
        LMStudioProvider(url)


def test_native_discovery_uses_actual_loaded_context(monkeypatch):
    loaded_context = 4096
    calls = []
    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"models": [
            {"key": "local-chat", "type": "llm", "max_context_length": 131072,
             "loaded_instances": [{"id": "loaded-chat", "config": {"context_length": loaded_context}}],
             "capabilities": {"vision": True}, "selected_variant": "local-chat@q4"},
            {"key": "local-embedding", "type": "embedding", "max_context_length": 2048,
             "loaded_instances": [], "quantization": {"name": "F16"}},
        ]})
    rows = provider(monkeypatch, handle).discover_models()
    assert rows[0] == {"id": "loaded-chat", "type": "llm", "context_length": loaded_context,
                       "vision": True, "loaded": True, "max_context_length": 131072,
                       "identity": "local-chat@q4"}
    assert rows[1]["context_length"] is None and rows[1]["loaded"] is False
    assert rows[1]["identity"] == "local-embedding@F16"
    assert calls == ["http://127.0.0.1:1234/api/v1/models"]


def test_old_server_fallback_does_not_invent_capabilities(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request.url.path)
        return (httpx.Response(404) if request.url.path == "/api/v1/models" else
                httpx.Response(200, json={"data": [{"id": "some-model"}]}))
    row = provider(monkeypatch, handle).discover_models()[0]
    assert row["type"] == "unknown" and row["context_length"] is None and not row["vision"]
    assert calls == ["/api/v1/models", "/v1/models"]


@pytest.mark.parametrize("status", [401, 500])
def test_model_list_auth_or_server_errors_are_not_hidden_by_fallback(monkeypatch, status):
    calls = []
    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(status, json={"error": {"message": "failure"}})
    with pytest.raises(ProviderError, match=f"HTTP {status}"):
        provider(monkeypatch, handle).discover_models()
    assert len(calls) == 1


def test_redirects_are_never_followed(monkeypatch):
    calls = []
    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://example.org"})
    with pytest.raises(ProviderError, match="redirects"):
        provider(monkeypatch, handle).chat({"model": "local", "messages": []})
    assert len(calls) == 1


def test_chat_serializes_local_wire_format(monkeypatch):
    calls = []
    result = {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
    def handle(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=result)
    assert provider(monkeypatch, handle).chat({"model": "local", "messages": [], "max_tokens": 100}) == result
    assert calls == [("/v1/chat/completions", {"model": "local", "messages": [], "max_tokens": 100, "stream": False})]


def test_stream_decodes_sse_and_closes_on_cancellation(monkeypatch):
    closed = []
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"one"},"finish_reason":null}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        def close(self):
            closed.append(True)
    backend = provider(monkeypatch, lambda request: httpx.Response(200, stream=Body()))
    stream = backend.stream_chat({"model": "local", "messages": []})
    assert next(stream)["choices"][0]["delta"]["content"] == "one"
    stream.close()
    assert closed == [True]


def test_stream_handles_comments_and_done(monkeypatch):
    wire = b': keepalive\n\ndata: {"choices":[]}\n\ndata: [DONE]\n\n'
    events = list(provider(monkeypatch, lambda request: httpx.Response(200, content=wire)).stream_chat({}))
    assert events == [{"choices": []}]


def test_bad_stream_event_fails_without_retry(monkeypatch):
    calls = []
    def handle(request):
        calls.append(True)
        return httpx.Response(200, content=b'data: not-json\n\n')
    with pytest.raises(ProviderError, match="invalid stream"):
        list(provider(monkeypatch, handle).stream_chat({}))
    assert calls == [True]


def test_stream_bounds_unterminated_events_and_total_bytes_before_reading_more(monkeypatch):
    monkeypatch.setattr(lmstudio, "MAX_EVENT_BYTES", 128)
    monkeypatch.setattr(lmstudio, "MAX_RESPONSE_BYTES", 240)
    for chunks, message in (([b"data: " + b"x" * 200], "event exceeded"),
                            ([b'data: {"choices":[]}\n\n'] * 12, "stream exceeded")):
        closed = []
        class Body(httpx.SyncByteStream):
            def __iter__(self):
                yield from chunks
                pytest.fail("The byte bound must stop the stream before another read")
            def close(self):
                closed.append(True)
        backend = provider(monkeypatch, lambda request: httpx.Response(200, stream=Body()))
        with pytest.raises(ProviderError, match=message):
            list(backend.stream_chat({}))
        assert closed == [True]


def test_embeddings_reorder_rows_without_requesting_cloud_dimensions(monkeypatch):
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0., 1.]}, {"index": 0, "embedding": [1., 0.]}]})
    assert provider(monkeypatch, handle).embeddings("local-embedding", ["a", "b"]) == [[1., 0.], [0., 1.]]
    assert calls == [{"model": "local-embedding", "input": ["a", "b"]}]


@pytest.mark.parametrize("rows", [
    [{"index": 1, "embedding": [1., 0.]}],
    [{"index": True, "embedding": [1., 0.]}],
    [{"index": 0, "embedding": [0., 0.]}],
    [{"index": 0, "embedding": [True, 1.]}],
    [{"index": 0, "embedding": "bad"}],
])
def test_bad_embedding_wire_data_fails(monkeypatch, rows):
    backend = provider(monkeypatch, lambda request: httpx.Response(200, json={"data": rows}))
    with pytest.raises(ProviderError):
        backend.embeddings("local-embedding", ["a"])


def test_client_disables_environment_proxies_and_redirects(monkeypatch):
    captured = []
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: captured.append(kwargs))
    LMStudioProvider("http://localhost:1234", api_key="local-secret")._client()
    assert captured[0]["trust_env"] is False
    assert captured[0]["follow_redirects"] is False
    assert captured[0]["headers"] == {"Authorization": "Bearer local-secret"}
    with pytest.raises(ValueError, match="control characters"):
        LMStudioProvider("http://localhost:1234", api_key="local-token\r\nX-Header: value")
