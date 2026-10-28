"""Native Ollama contract and local-only transport tests, without live inference."""

import copy
import json

import httpx
import pytest

from chatbot.providers.base import ProviderError
from chatbot.providers.ollama import OllamaProvider
from chatbot.providers import transport


MODEL = "installed-chat:latest"
DETAILS = {"capabilities": ["completion", "vision", "thinking"],
           "model_info": {"general.architecture": "synthetic", "synthetic.context_length": 32768}}
PAYLOAD = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
           "max_tokens": 7000, "context_length": 8192, "temperature": 0.4, "top_p": 0.8}


def backend(monkeypatch, inference=None, *, tags=None, details=None, running=None):
    calls = []

    def handle(request):
        path = request.url.path
        payload = json.loads(request.content) if request.content else None
        calls.append((request.method, path, payload))
        if path == "/api/tags":
            return httpx.Response(200, json={"models": tags if tags is not None else [
                {"name": MODEL, "digest": "local-digest"}]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": running or []})
        if path == "/api/show":
            return httpx.Response(200, json=details if details is not None else DETAILS)
        if path in {"/api/chat", "/api/embed"} and inference is not None:
            return inference(request, payload)
        pytest.fail(f"Unexpected request: {request.method} {path}")

    provider = OllamaProvider("http://localhost:11434")
    monkeypatch.setattr(provider, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(handle), trust_env=False, follow_redirects=False))
    return provider, calls


def reply(content="Answer", *, done=True, reason="stop", **extra):
    return {"message": {"role": "assistant", "content": content, **extra},
            "done": done, "done_reason": reason}


def test_discovery_separates_loaded_context_from_advertised_and_keeps_installed_models(monkeypatch):
    provider, calls = backend(monkeypatch, tags=[
        {"name": MODEL, "digest": "first-digest"},
        {"name": "unloaded:latest", "digest": "second-digest"},
        {"name": "proxy:latest", "remote_host": "https://ollama.com", "digest": "remote"},
    ], running=[{"name": MODEL, "context_length": 4096}])
    rows = provider.discover_models()
    assert rows == [
        {"id": MODEL, "type": "llm", "context_length": 4096, "max_context_length": 32768,
         "vision": True, "loaded": True, "available": True, "identity": "first-digest"},
        {"id": "unloaded:latest", "type": "llm", "context_length": None, "max_context_length": 32768,
         "vision": True, "loaded": False, "available": True, "identity": "second-digest"},
    ]
    assert [(method, path) for method, path, _ in calls] == [
        ("GET", "/api/tags"), ("GET", "/api/ps"), ("POST", "/api/show"), ("POST", "/api/show")]
    assert all(payload is None or payload["model"] != "proxy:latest" for _, _, payload in calls)


def test_discovery_classifies_embedding_from_capabilities_without_model_name_guessing(monkeypatch):
    provider, _ = backend(monkeypatch, details={"capabilities": ["embedding"],
        "model_info": {"general.architecture": "encoder", "encoder.context_length": 2048}})
    assert provider.discover_models()[0]["type"] == "embedding"
    provider, _ = backend(monkeypatch, details={"capabilities": ["image"], "model_info": {}})
    assert provider.discover_models() == []


@pytest.mark.parametrize("stage", ["tags", "show"])
def test_remote_metadata_is_rejected_before_sending_chat_content(monkeypatch, stage):
    tags = [{"name": MODEL, "digest": "digest", **({"remote_model": "upstream"} if stage == "tags" else {})}]
    details = {**DETAILS, **({"remote_host": "https://ollama.com"} if stage == "show" else {})}
    provider, calls = backend(monkeypatch, tags=tags, details=details)
    with pytest.raises(ProviderError, match="remote-proxy"):
        provider.chat(PAYLOAD)
    assert not any(path in {"/api/chat", "/api/embed"} for _, path, _ in calls)


@pytest.mark.parametrize("model", ["model:cloud", "model:8b-cloud"])
def test_explicit_cloud_routing_suffix_is_rejected_before_any_request(monkeypatch, model):
    provider, calls = backend(monkeypatch)
    with pytest.raises(ProviderError, match="remote-proxy"):
        provider.chat({**PAYLOAD, "model": model})
    assert calls == []


def test_model_removed_since_discovery_is_rejected_without_pulling_or_generating(monkeypatch):
    provider, calls = backend(monkeypatch, tags=[])
    with pytest.raises(ProviderError, match="not installed locally"):
        provider.chat(PAYLOAD)
    assert [path for _, path, _ in calls] == ["/api/tags"]


def test_prepare_and_chat_preserve_exact_schema_images_and_context_without_mutation(monkeypatch):
    response = reply(thinking="hidden reasoning")
    response.update(prompt_eval_count=100, eval_count=75)
    provider, calls = backend(monkeypatch, lambda request, payload: httpx.Response(200, json=response))
    original = {**PAYLOAD, "model": "installed-chat", "messages": [
        {"role": "system", "content": "Use sources only."},
        {"role": "user", "content": [{"type": "text", "text": "Read the image"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "result", "strict": True, "schema": {"$defs": {"id": {"enum": ["s1"]}},
            "type": "object", "properties": {"source": {"$ref": "#/$defs/id"}}}}}}
    before = copy.deepcopy(original)
    prepared = provider.prepare_chat_payload(original, context_length=16384)
    result = provider.chat(prepared)
    wire = calls[-1][2]
    assert wire == {"model": MODEL + ":local", "stream": False, "truncate": False, "shift": False,
        "messages": [{"role": "system", "content": "Use sources only."},
                     {"role": "user", "content": "Read the image", "images": ["AA=="]}],
        "options": {"num_ctx": 16384, "num_predict": 7000, "temperature": 0.4, "top_p": 0.8},
        "format": original["response_format"]["json_schema"]["schema"]}
    assert original == before and prepared["context_length"] == 16384
    assert result["choices"] == [{"finish_reason": "stop", "message": {
        "content": "Answer", "reasoning_content": "hidden reasoning"}}]
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 75, "total_tokens": 175}


@pytest.mark.parametrize("url", ["https://example.org/photo.png", "file:///private.png", "data:image/png;base64,not-base64"])
def test_invalid_images_fail_before_metadata_or_inference(monkeypatch, url):
    provider, calls = backend(monkeypatch)
    payload = {**PAYLOAD, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}]}]}
    with pytest.raises(ValueError, match="image"):
        provider.chat(payload)
    assert calls == []


def test_context_above_advertised_maximum_is_rejected_without_inference(monkeypatch):
    provider, calls = backend(monkeypatch)
    with pytest.raises(ProviderError, match="advertised maximum"):
        provider.chat({**PAYLOAD, "context_length": 65536})
    assert calls[-1][1] == "/api/show"


@pytest.mark.parametrize("overrides", [{"context_length": None}, {"max_tokens": -1}, {"max_tokens": 9000}])
def test_missing_or_inconsistent_budgets_fail_before_any_request(monkeypatch, overrides):
    provider, calls = backend(monkeypatch)
    with pytest.raises(ValueError, match="budgets"):
        provider.chat({**PAYLOAD, **overrides})
    assert calls == []


def test_length_stop_and_missing_finish_reason_are_not_mislabeled_as_success(monkeypatch):
    provider, _ = backend(monkeypatch, lambda request, payload: httpx.Response(200, json=reply(reason="length")))
    assert provider.chat(PAYLOAD)["choices"][0]["finish_reason"] == "length"
    provider, _ = backend(monkeypatch, lambda request, payload: httpx.Response(200, json=reply(reason=None)))
    assert provider.chat(PAYLOAD)["choices"][0]["finish_reason"] is None


def test_stream_normalizes_thinking_final_text_and_closes_on_cancellation(monkeypatch):
    closed = []
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield (json.dumps(reply("", done=False, reason=None, thinking="private")) + "\n").encode()
            yield (json.dumps(reply("Last delta", reason="length")) + "\n").encode()
        def close(self):
            closed.append(True)
    provider, calls = backend(monkeypatch, lambda request, payload: httpx.Response(200, stream=Body()))
    output = provider.stream_chat(PAYLOAD)
    assert next(output)["choices"][0]["delta"] == {"content": "", "reasoning_content": "private"}
    output.close()
    assert closed == [True]
    events = list(provider.stream_chat(PAYLOAD))
    assert events[-1]["choices"][0] == {"delta": {"content": "Last delta"}, "finish_reason": "length"}
    assert closed == [True, True]
    assert calls[-1][2]["stream"] is True


@pytest.mark.parametrize("wire, message", [
    (b'{"message":{"content":"partial"},"done":false}\n', "before the model finished"),
    (b'{"error":"model failed"}\n', "stream reported an error"),
    (b'not json\n', "invalid stream event"),
])
def test_stream_failures_close_the_response_without_retry(monkeypatch, wire, message):
    responses = []
    def generate(request, payload):
        response = httpx.Response(200, content=wire)
        responses.append(response)
        return response
    provider, calls = backend(monkeypatch, generate)
    with pytest.raises(ProviderError, match=message):
        list(provider.stream_chat(PAYLOAD))
    assert len(responses) == 1 and responses[0].is_closed
    assert len([path for _, path, _ in calls if path == "/api/chat"]) == 1


def test_embeddings_use_native_batch_order_and_disable_truncation(monkeypatch):
    provider, calls = backend(monkeypatch, lambda request, payload: httpx.Response(200,
        json={"embeddings": [[1., 0.], [0., 1.]]}), details={"capabilities": ["embedding"]})
    assert provider.embeddings(MODEL, ["first", "second"]) == [[1., 0.], [0., 1.]]
    assert calls[-1] == ("POST", "/api/embed", {"model": MODEL + ":local", "input": ["first", "second"], "truncate": False})


@pytest.mark.parametrize("vectors", [[], [[0, 0]], [[True, 1]], [[float("nan"), 1]], [[1, 0], [1]]])
def test_invalid_embedding_batches_are_rejected(monkeypatch, vectors):
    provider, _ = backend(monkeypatch, lambda request, payload: httpx.Response(200,
        content=json.dumps({"embeddings": vectors})), details={"capabilities": ["embedding"]})
    inputs = ["a", "b"] if len(vectors) == 2 else ["a"]
    with pytest.raises(ProviderError, match="embedding|dimensions"):
        provider.embeddings(MODEL, inputs)


def test_wrong_embedding_capability_fails_before_inference(monkeypatch):
    provider, calls = backend(monkeypatch)
    with pytest.raises(ProviderError, match="does not support embedding"):
        provider.embeddings(MODEL, ["document"])
    assert calls[-1][1] == "/api/show"


@pytest.mark.parametrize("url", ["https://ollama.com", "http://8.8.8.8:11434", "http://169.254.169.254",
                                  "http://user:secret@localhost:11434", "http://localhost:11434?remote=true"])
def test_nonlocal_or_ambiguous_server_urls_are_rejected(url):
    with pytest.raises(ValueError):
        OllamaProvider(url)


def test_transport_pins_localhost_and_disables_proxies_redirects_with_bounded_timeout(monkeypatch):
    captured = []
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: captured.append(kwargs))
    provider = OllamaProvider("http://localhost:11434/api", api_key="local-token", timeout=900)
    provider._client()
    assert provider.base_url == "http://127.0.0.1:11434"
    assert captured[0]["trust_env"] is False and captured[0]["follow_redirects"] is False
    assert captured[0]["headers"] == {"Authorization": "Bearer local-token"}
    assert captured[0]["timeout"].connect == 5 and captured[0]["timeout"].read == 900
    assert OllamaProvider("http://[::1]:11434").base_url == "http://[::1]:11434"
    assert OllamaProvider("https://192.168.1.5:11434").base_url == "https://192.168.1.5:11434"


def test_redirect_errors_and_response_limits_do_not_retry(monkeypatch):
    provider = OllamaProvider("http://localhost:11434")
    calls = []
    def redirect(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "https://ollama.com"})
    monkeypatch.setattr(provider, "_client", lambda: httpx.Client(transport=httpx.MockTransport(redirect)))
    with pytest.raises(ProviderError, match="redirects"):
        provider.discover_models()
    assert len(calls) == 1
    monkeypatch.setattr(transport, "MAX_RESPONSE_BYTES", 16)
    monkeypatch.setattr(provider, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 17))))
    with pytest.raises(ProviderError, match="size limit"):
        provider.discover_models()


def test_unterminated_stream_lines_are_bounded(monkeypatch):
    monkeypatch.setattr(transport, "MAX_EVENT_BYTES", 16)
    provider, _ = backend(monkeypatch, lambda request, payload: httpx.Response(200, content=b"x" * 17))
    with pytest.raises(ProviderError, match="event exceeded"):
        list(provider.stream_chat(PAYLOAD))


def test_discovered_embedding_digest_is_pinned_until_reconnect(monkeypatch):
    tags = [{"name": MODEL, "digest": "original-weights"}]
    provider, calls = backend(monkeypatch, tags=tags, details={"capabilities": ["embedding"]})
    provider.discover_models()
    tags[0]["digest"] = "replacement-weights"
    with pytest.raises(ProviderError, match="changed after connecting"):
        provider.embeddings(MODEL, ["document"])
    assert calls[-1][1] == "/api/tags"
    assert not any(path == "/api/embed" for _, path, _ in calls)


def test_server_without_local_routing_fails_explicitly_without_fallback(monkeypatch):
    provider, calls = backend(monkeypatch, lambda request, payload: httpx.Response(404,
        json={"error": "model not found"}))
    with pytest.raises(ProviderError, match=":local routing"):
        provider.chat(PAYLOAD)
    assert len([path for _, path, _ in calls if path == "/api/chat"]) == 1
