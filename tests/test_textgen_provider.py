"""Textgen's ignored model field, native prompt trimming, and transport boundary."""

import copy
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from chatbot.providers.base import ProviderError
from chatbot.providers.textgen import TextGenProvider
from chatbot.providers import textgen


MODEL = "loaded.gguf"
PAYLOAD = {"model": MODEL, "messages": [{"role": "system", "content": "Use evidence."},
            {"role": "user", "content": "A complete source passage."}],
           "max_tokens": 3000, "context_length": 4096, "temperature": 0.4, "top_p": 0.8}


def reply(content="Answer", *, model=MODEL, reason="stop", usage=None):
    return {"model": model, "choices": [{"finish_reason": reason, "message": {
        "role": "assistant", "content": content, "reasoning_content": "private"}}],
        "usage": usage or {"prompt_tokens": 50, "completion_tokens": 5}}


def chunk(content="Text", *, reason=None, model=MODEL, usage=None):
    value = {"model": model, "choices": [{"finish_reason": reason, "delta": {"content": content}}]}
    if usage is not None:
        value["usage"] = usage
    return b"data: " + json.dumps(value).encode() + b"\r\n\r\n"


def backend(monkeypatch, inference=None, *, metadata=None, prompt=None, token_count=50, properties=None):
    calls = []
    info = metadata if metadata is not None else {"model_name": MODEL, "loader": "llama.cpp", "lora_names": []}
    spec = {"paths": {path: {method: {}} for path, method in textgen._REQUIRED_PATHS.items()},
            "components": {"schemas": {"ChatCompletionRequest": {"properties": properties if properties is not None else
                {"grammar_string": {}, "truncation_length": {}, "auto_max_new_tokens": {}}}}}}

    def handle(request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, path, body))
        if path == "/openapi.json":
            return httpx.Response(200, json=spec)
        if path == "/v1/internal/model/info":
            return httpx.Response(200, json=info)
        if path == "/v1/internal/chat-prompt":
            rendered = prompt if prompt is not None else "\n".join(message["content"] for message in body["messages"])
            return httpx.Response(200, json={"prompt": rendered})
        if path == "/v1/internal/token-count":
            return httpx.Response(200, json={"length": token_count})
        if path == "/v1/chat/completions" and inference is not None:
            return inference(request, body)
        pytest.fail(f"Unexpected request: {request.method} {path}")

    provider = TextGenProvider("http://localhost:5000/v1")
    monkeypatch.setattr(provider, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(handle), trust_env=False, follow_redirects=False))
    return provider, calls, info


def test_discovery_is_explicit_and_never_probes_or_downloads_embeddings(monkeypatch):
    provider, calls, _ = backend(monkeypatch)
    assert calls == [] and provider.base_url == "http://127.0.0.1:5000"
    record = provider.discover_models()[0]
    assert record["id"] == MODEL and record["loaded"] and record["structured_output"]
    assert record["context_length"] is None and record["max_context_length"] is None
    assert record["vision"] is False
    with pytest.raises(ProviderError, match="separate installed local embedding"):
        provider.embeddings("anything", ["source"])
    assert [path for _, path, _ in calls] == ["/openapi.json", "/v1/internal/model/info"]


@pytest.mark.parametrize("change", [{"model_name": "other.gguf"}, {"lora_names": ["new-adapter"]}])
def test_changed_loaded_model_or_loras_rejects_before_any_prompt_is_sent(monkeypatch, change):
    provider, calls, info = backend(monkeypatch)
    provider.discover_models()
    info.update(change)
    with pytest.raises(ProviderError, match="changed"):
        provider.chat(PAYLOAD)
    assert all(method == "GET" for method, _, _ in calls)


def test_unloaded_server_has_no_model_and_unlisted_ids_do_not_generate(monkeypatch):
    provider, calls, _ = backend(monkeypatch, metadata={"model_name": "None", "loader": None, "lora_names": []})
    assert provider.discover_models() == []
    with pytest.raises(ProviderError, match="not the connected"):
        provider.chat(PAYLOAD)
    assert all(method == "GET" for method, _, _ in calls)


def test_chat_keeps_complete_messages_sampling_and_bound_context_without_mutation(monkeypatch):
    provider, calls, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, json=reply()))
    provider.discover_models()
    original = copy.deepcopy(PAYLOAD)
    prepared = provider.prepare_chat_payload({key: value for key, value in PAYLOAD.items() if key != "context_length"},
                                             context_length=4096)
    assert provider.chat(prepared)["choices"][0]["message"]["content"] == "Answer"
    wire = next(body for _, path, body in calls if path == "/v1/chat/completions")
    assert wire == {"model": MODEL, "messages": original["messages"], "stream": False,
                    "max_tokens": 3000, "truncation_length": 4096, "auto_max_new_tokens": False,
                    "mode": "instruct", "tool_choice": "none", "temperature": 0.4, "top_p": 0.8}
    probe = next(body for _, path, body in calls if path == "/v1/internal/chat-prompt")
    assert probe["truncation_length"] > 4096 and probe["messages"] == wire["messages"]
    assert PAYLOAD == original


def test_exact_scoped_schema_is_compiled_to_native_grammar_without_weak_fallback(monkeypatch):
    schemas = []
    def compile_schema(schema):
        schemas.append(copy.deepcopy(schema))
        return 'root ::= "strict-result"'
    monkeypatch.setitem(sys.modules, "chatbot.providers.schema_grammar", SimpleNamespace(schema_to_grammar=compile_schema))
    provider, calls, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, json=reply('{}')))
    provider.discover_models()
    schema = {"type": "object", "properties": {"source_id": {"$ref": "#/$defs/source_identifier"}},
              "$defs": {"source_identifier": {"type": "string", "enum": ["only-source"]}}, "required": ["source_id"]}
    provider.chat({**PAYLOAD, "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}})
    assert schemas == [schema]
    wire = next(body for _, path, body in calls if path == "/v1/chat/completions")
    assert wire["grammar_string"] == 'root ::= "strict-result"' and "response_format" not in wire
    assert len([path for _, path, _ in calls if path == "/v1/chat/completions"]) == 1


def test_unsupported_loader_is_not_advertised_for_structured_output(monkeypatch):
    provider, calls, _ = backend(monkeypatch, metadata={"model_name": MODEL, "loader": "Transformers", "lora_names": []})
    assert provider.discover_models()[0]["structured_output"] is False
    before = len(calls)
    with pytest.raises(ProviderError, match="schema-constrained"):
        provider.chat({**PAYLOAD, "response_format": {"type": "json_schema", "json_schema": {"schema": {}}}})
    assert len(calls) == before


@pytest.mark.parametrize("prompt,count,match", [("A shortened passage.", 5, "omitted or changed"),
                                              (None, 4097, "exceed the app context")])
def test_complete_prompt_must_survive_template_and_fit_before_generation(monkeypatch, prompt, count, match):
    provider, calls, _ = backend(monkeypatch, prompt=prompt, token_count=count)
    provider.discover_models()
    with pytest.raises(ProviderError, match=match):
        provider.chat(PAYLOAD)
    assert not any(path == "/v1/chat/completions" for _, path, _ in calls)


def test_native_template_overhead_adjusts_only_ordinary_output_not_input(monkeypatch):
    provider, calls, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, json=reply()), token_count=1200)
    provider.discover_models()
    provider.chat(PAYLOAD)
    wire = next(body for _, path, body in calls if path == "/v1/chat/completions")
    assert wire["max_tokens"] == 4096 - 1200 and wire["messages"] == PAYLOAD["messages"]
    monkeypatch.setitem(sys.modules, "chatbot.providers.schema_grammar",
                        SimpleNamespace(schema_to_grammar=lambda schema: 'root ::= "result"'))
    with pytest.raises(ProviderError, match="reply reserve exceed"):
        provider.chat({**PAYLOAD, "response_format": {"type": "json_schema", "json_schema": {"schema": {}}}})
    assert sum(path == "/v1/chat/completions" for _, path, _ in calls) == 1


@pytest.mark.parametrize("overrides", [{"max_tokens": 4096}, {"max_tokens": -1}, {"context_length": None},
                                      {"messages": [{"role": "user", "content": [{"type": "image_url",
                                        "image_url": {"url": "https://example.org/image.png"}}]}]}])
def test_invalid_budgets_and_images_fail_before_any_server_request(monkeypatch, overrides):
    provider, calls, _ = backend(monkeypatch)
    with pytest.raises(ValueError):
        provider.chat({**PAYLOAD, **overrides})
    assert calls == []


def test_actual_response_model_and_post_generation_loras_are_checked(monkeypatch):
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, json=reply(model="other.gguf")))
    provider.discover_models()
    with pytest.raises(ProviderError, match="different model"):
        provider.chat(PAYLOAD)
    def changed(request, body):
        info["lora_names"] = ["new"]
        return httpx.Response(200, json=reply())
    provider, _, info = backend(monkeypatch, changed)
    provider.discover_models()
    with pytest.raises(ProviderError, match="changed"):
        provider.chat(PAYLOAD)


def test_usage_at_output_limit_cannot_be_mislabeled_as_complete(monkeypatch):
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, json=reply(
        usage={"prompt_tokens": 50, "completion_tokens": 3000})))
    provider.discover_models()
    assert provider.chat(PAYLOAD)["choices"][0]["finish_reason"] == "length"


def test_stream_yields_immediately_and_closes_response_when_stopped(monkeypatch):
    closed = []
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield chunk("First")
            pytest.fail("Stopping after the first delta must not read ahead")
        def close(self):
            closed.append(True)
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, stream=Body()))
    provider.discover_models()
    stream = provider.stream_chat(PAYLOAD)
    assert next(stream)["choices"][0]["delta"]["content"] == "First"
    stream.close()
    assert closed == [True]


@pytest.mark.parametrize("wire,match", [(chunk("Partial"), "before the model finished"),
                                       (chunk("Partial") + b"data: [DONE]\n\n", "without a completion status"),
                                       (b'data: {"error":"failed"}\n\n', "stream reported an error")])
def test_truncated_or_error_stream_is_not_success(monkeypatch, wire, match):
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, content=wire))
    provider.discover_models()
    with pytest.raises(ProviderError, match=match):
        list(provider.stream_chat(PAYLOAD))


def test_complete_stream_preserves_terminal_status_and_checks_model(monkeypatch):
    wire = chunk("Answer") + chunk("", reason="length") + b"data: [DONE]\n\n"
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, content=wire))
    provider.discover_models()
    assert list(provider.stream_chat(PAYLOAD))[-1]["choices"][0]["finish_reason"] == "length"
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, content=chunk(model="other.gguf")))
    provider.discover_models()
    with pytest.raises(ProviderError, match="different model"):
        list(provider.stream_chat(PAYLOAD))


def test_stream_line_is_bounded_before_newline_or_json_decode(monkeypatch):
    monkeypatch.setattr(textgen, "MAX_EVENT_BYTES", 128)
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, content=b"data: " + b"x" * 200))
    provider.discover_models()
    with pytest.raises(ProviderError, match="event exceeded"):
        list(provider.stream_chat(PAYLOAD))


def test_discovery_without_context_controls_fails_before_generation(monkeypatch):
    provider, calls, _ = backend(monkeypatch, properties={"grammar_string": {}})
    with pytest.raises(ProviderError, match="context controls"):
        provider.discover_models()
    assert len(calls) == 1
