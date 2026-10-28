"""KoboldCpp prompt preservation, native constraints, and response isolation."""

import copy
import json

import httpx
import pytest

from chatbot.providers.base import ProviderError
from chatbot.providers import koboldcpp
from chatbot.providers.koboldcpp import KoboldCppProvider


MODEL = "koboldcpp/local.gguf"
PAYLOAD = {"model": MODEL, "messages": [{"role": "system", "content": "Use the evidence."},
            {"role": "user", "content": "The complete source passage: {{[INPUT]}}."}],
           "context_length": 4096, "max_tokens": 3000, "temperature": 0.4, "top_p": 0.8}
SCHEMA = {"type": "object", "properties": {"source_id": {"$ref": "#/$defs/source_identifier"}},
          "required": ["source_id"], "additionalProperties": False,
          "$defs": {"source_identifier": {"type": "string", "enum": ["source-A"]}}}


def structured(**changes):
    return {**PAYLOAD, "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}, **changes}


def chunk(text="Answer", *, reason=None, model=MODEL, delta=None):
    data = {"model": model, "choices": [{"index": 0, "finish_reason": reason,
            "delta": {"content": text} if delta is None else delta}]}
    return b"data: " + json.dumps(data).encode() + b"\r\n\r\n"


def usage(*, prompt=50, completion=5, model=MODEL):
    data = {"model": model, "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
    return b"data: " + json.dumps(data).encode() + b"\n\n"


def completed(text="Answer", *, reason="stop", prompt=50, completion=5):
    return chunk(text) + chunk("", reason=reason) + usage(prompt=prompt, completion=completion) + b"data: [DONE]\n\n"


def backend(monkeypatch, inference=None, *, prompt=None, token_count=50):
    calls = []
    state = {"model": MODEL, "context": 4096, "capabilities": {
        "result": "KoboldCpp", "version": "1.123.1", "llm": True, "router": False,
        "defaults_modified": False, "jinja": False, "mcp": False}}

    def handle(request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, path, body))
        if path == "/api/extra/version":
            return httpx.Response(200, json=state["capabilities"])
        if path == "/api/v1/model":
            return httpx.Response(200, json={"result": state["model"]})
        if path == "/api/extra/true_max_context_length":
            return httpx.Response(200, json={"value": state["context"]})
        if path == "/api/extra/tokencount":
            rendered = prompt if prompt is not None else "\n".join(item["content"] for item in body["messages"])
            return httpx.Response(200, json={"prompt": rendered, "value": token_count})
        if path == "/v1/chat/completions":
            return inference(request, body) if inference else httpx.Response(200, content=completed(prompt=token_count))
        pytest.fail(f"Unexpected request: {request.method} {path}")

    provider = KoboldCppProvider("http://localhost:5001/v1")
    monkeypatch.setattr(provider, "_client", lambda: httpx.Client(
        transport=httpx.MockTransport(handle), trust_env=False, follow_redirects=False))
    return provider, calls, state


def test_discovery_only_reads_loaded_model_and_never_uses_native_embeddings(monkeypatch):
    provider, calls, state = backend(monkeypatch)
    assert calls == [] and provider.base_url == "http://127.0.0.1:5001"
    record = provider.discover_models()[0]
    assert record["id"] == MODEL and record["loaded"] and record["structured_output"]
    assert record["context_length"] == 4096 and record["max_context_length"] is None
    assert record["vision"] is False
    with pytest.raises(ProviderError, match="separate installed local embedding"):
        provider.embeddings("anything", ["source"])
    assert [path for _, path, _ in calls] == ["/api/extra/version", "/api/v1/model", "/api/extra/true_max_context_length"]
    state["capabilities"]["llm"] = False
    assert provider.discover_models() == []


@pytest.mark.parametrize("change,match", [({"version": "1.122.0"}, "1.123.1"),
    ({"result": "Unverified backend"}, "version metadata"), ({"router": True}, "Router Mode"),
    ({"defaults_modified": True}, "Generation Defaults"), ({"llm": "true"}, "capabilities")])
def test_unsupported_or_overriding_server_settings_fail_before_prompt(monkeypatch, change, match):
    provider, calls, state = backend(monkeypatch)
    state["capabilities"].update(change)
    with pytest.raises(ProviderError, match=match):
        provider.discover_models()
    assert all(method == "GET" for method, _, _ in calls)


def test_protected_model_placeholder_is_not_a_successful_connection(monkeypatch):
    provider, calls, state = backend(monkeypatch)
    state["model"] = "koboldcpp/protected-model"
    with pytest.raises(ProviderError, match="KOBOLDCPP_API_KEY"):
        provider.discover_models()
    assert all(method == "GET" for method, _, _ in calls)


@pytest.mark.parametrize("field,value", [("model", "different.gguf"), ("context", 8192)])
def test_replaced_model_and_context_reject_before_prompt(monkeypatch, field, value):
    provider, calls, state = backend(monkeypatch)
    provider.discover_models()
    state[field] = value
    with pytest.raises(ProviderError, match="changed"):
        provider.chat(PAYLOAD)
    assert all(method == "GET" for method, _, _ in calls)


def test_blocking_chat_uses_one_sse_request_and_preserves_messages_and_sampling(monkeypatch):
    provider, calls, _ = backend(monkeypatch)
    provider.discover_models()
    original = copy.deepcopy(PAYLOAD)
    prepared = provider.prepare_chat_payload({key: value for key, value in PAYLOAD.items() if key != "context_length"},
                                             context_length=4096)
    assert provider.chat(prepared)["choices"][0]["message"]["content"] == "Answer"
    wire = next(body for _, path, body in calls if path == "/v1/chat/completions")
    assert wire["stream"] and wire["stream_options"] == {"include_usage": True}
    assert wire["max_tokens"] == 3000 and wire["max_context_length"] == 4096
    assert wire["messages"] == original["messages"]
    assert wire["temperature"] == 0.4 and wire["top_p"] == 0.8
    assert wire["encapsulate_thinking"] and not wire["grammar_retain_state"]
    assert not wire["replace_instruct_placeholders"] and wire["tool_choice"] == "none" and wire["tools"] == []
    assert PAYLOAD == original and sum(path == "/v1/chat/completions" for _, path, _ in calls) == 1


def test_scoped_schema_is_native_grammar_and_visible_in_budgeted_prompt(monkeypatch):
    provider, calls, _ = backend(monkeypatch)
    provider.discover_models()
    original = copy.deepcopy(SCHEMA)
    provider.chat(structured())
    wire = next(body for _, path, body in calls if path == "/v1/chat/completions")
    probe = next(body for _, path, body in calls if path == "/api/extra/tokencount")
    assert "root ::=" in wire["grammar"] and "source-A" in wire["grammar"]
    assert "response_format" not in wire and "grammar" not in probe
    assert json.dumps(SCHEMA, ensure_ascii=False) in wire["messages"][0]["content"]
    assert probe["messages"] == wire["messages"] and SCHEMA == original
    bad = copy.deepcopy(structured())
    bad["response_format"]["json_schema"]["schema"]["not"] = {"type": "null"}
    before = len(calls)
    with pytest.raises(ValueError, match="unsupported"):
        provider.chat(bad)
    assert len(calls) == before


@pytest.mark.parametrize("overrides", [{"max_tokens": 4096}, {"max_tokens": -1}, {"context_length": None},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.org/image"}}]}]}])
def test_invalid_budgets_and_media_fail_without_server_calls(monkeypatch, overrides):
    provider, calls, _ = backend(monkeypatch)
    with pytest.raises(ValueError):
        provider.chat({**PAYLOAD, **overrides})
    assert calls == []


def test_native_output_cap_adjusts_only_output_and_keeps_structured_reserve_strict(monkeypatch):
    provider, calls, _ = backend(monkeypatch, token_count=1200)
    provider.discover_models()
    provider.chat(PAYLOAD)
    wire = next(body for _, path, body in calls if path == "/v1/chat/completions")
    assert wire["max_tokens"] == 4096 - int(4096 * 0.35)
    assert wire["messages"] == PAYLOAD["messages"]
    with pytest.raises(ProviderError, match="reply reserve"):
        provider.chat(structured())
    assert sum(path == "/v1/chat/completions" for _, path, _ in calls) == 1


def test_missing_or_oversized_prompt_rejects_without_generation(monkeypatch):
    for prompt, count in (("The source was shortened.", 50), (None, 4090)):
        provider, calls, _ = backend(monkeypatch, prompt=prompt, token_count=count)
        provider.discover_models()
        with pytest.raises(ProviderError, match="omitted or changed|reply reserve"):
            provider.chat(PAYLOAD)
        assert not any(path == "/v1/chat/completions" for _, path, _ in calls)


def test_blocking_and_streaming_keep_reasoning_separate_including_thinking_only_length(monkeypatch):
    wire = chunk(delta={"reasoning_content": "private unfinished thoughts"}) + chunk("", reason="length")
    wire += usage(completion=3000) + b"data: [DONE]\n\n"
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, content=wire))
    provider.discover_models()
    result = provider.chat(PAYLOAD)
    assert result["choices"][0]["message"] == {"role": "assistant", "content": ""}
    assert result["choices"][0]["finish_reason"] == "length"
    assert "private" not in json.dumps(list(provider.stream_chat(PAYLOAD)))


def test_stream_yields_immediately_and_closes_response_on_cancellation(monkeypatch):
    closed = []
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            yield chunk("First")
            pytest.fail("Stopping after one delta must not read ahead")
        def close(self):
            closed.append(True)
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, stream=Body()))
    provider.discover_models()
    stream = provider.stream_chat(PAYLOAD)
    assert next(stream)["choices"][0]["delta"]["content"] == "First"
    stream.close()
    assert closed == [True]


@pytest.mark.parametrize("wire,match", [(chunk("Partial"), "before the model finished"),
    (chunk("", reason="stop") + b"data: [DONE]\n\n", "token usage"),
    (chunk(model="other.gguf"), "different model"), (b'data: {"error":"failed"}\n\n', "stream reported an error"),
    (completed(prompt=10), "truncated the checked input"), (completed(prompt=99), "changed the checked prompt framing")])
def test_incomplete_mismatched_or_trimmed_response_is_not_accepted(monkeypatch, wire, match):
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, content=wire))
    provider.discover_models()
    with pytest.raises(ProviderError, match=match):
        provider.chat(PAYLOAD)


def test_terminal_usage_normalizes_limit_and_checks_model_after_generation(monkeypatch):
    provider, _, state = backend(monkeypatch, lambda request, body: httpx.Response(200, content=completed(completion=3000)))
    provider.discover_models()
    assert provider.chat(PAYLOAD)["choices"][0]["finish_reason"] == "length"
    def changed(request, body):
        state["context"] = 8192
        return httpx.Response(200, content=completed())
    provider, _, state = backend(monkeypatch, changed)
    provider.discover_models()
    with pytest.raises(ProviderError, match="changed"):
        provider.chat(PAYLOAD)


def test_http_errors_do_not_retry_and_unterminated_stream_events_are_bounded(monkeypatch):
    provider, calls, _ = backend(monkeypatch, lambda request, body: httpx.Response(503, json={"error": "Busy"}))
    provider.discover_models()
    with pytest.raises(ProviderError, match="HTTP 503"):
        provider.chat(PAYLOAD)
    assert sum(path == "/v1/chat/completions" for _, path, _ in calls) == 1
    monkeypatch.setattr(koboldcpp, "MAX_EVENT_BYTES", 128)
    provider, _, _ = backend(monkeypatch, lambda request, body: httpx.Response(200, content=b"data: " + b"x" * 200))
    provider.discover_models()
    with pytest.raises(ProviderError, match="event exceeded"):
        provider.chat(PAYLOAD)
