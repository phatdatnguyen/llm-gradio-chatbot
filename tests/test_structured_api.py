"""Strict local structured generation with independent local schema validation."""
from types import SimpleNamespace

import pytest

from chatbot import api, config


SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}},
          "required": ["title"], "additionalProperties": False}
HISTORY = [{"role": "user", "content": "Summarize this evidence."}]


@pytest.fixture(autouse=True)
def local_model(monkeypatch):
    monkeypatch.setattr(config, "resolve_model", lambda value: value or "local-model")
    monkeypatch.setattr(config, "get_max_context_tokens", lambda *args: 16384)
    monkeypatch.setattr(config, "supports_structured_output", lambda model: model == "local-model")


def install(monkeypatch, text='{"title":"Result"}', *, refusal=False, finish="stop"):
    calls = []
    def chat(payload):
        calls.append(payload)
        return {"choices": [{"finish_reason": finish, "message": {
            "content": text, "refusal": "Declined" if refusal else None}}]}
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(chat=chat))
    return calls


def test_schema_and_local_limits_sent_once(monkeypatch):
    calls = install(monkeypatch)
    assert api.generate_structured("local-model", HISTORY, SCHEMA, system_prompt="Sources only",
                                   schema_name="summary", max_output_tokens=1000) == {"title": "Result"}
    assert len(calls) == 1
    request = calls[0]
    assert request["messages"] == [{"role": "system", "content": "Sources only"}, *HISTORY]
    assert request["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "summary", "schema": SCHEMA, "strict": True}}
    assert request["max_tokens"] == 1000
    assert not {"tools", "max_completion_tokens", "reasoning_effort"} & request.keys()


def test_refusal_is_never_a_tool_result(monkeypatch):
    calls = install(monkeypatch, refusal=True)
    with pytest.raises(RuntimeError, match="declined"):
        api.generate_structured("local-model", HISTORY, SCHEMA)
    assert len(calls) == 1


@pytest.mark.parametrize("finish", ["length", None])
def test_truncation_is_failure_even_for_parseable_json(monkeypatch, finish):
    calls = install(monkeypatch, finish=finish)
    with pytest.raises(RuntimeError, match="incomplete"):
        api.generate_structured("local-model", HISTORY, SCHEMA)
    assert len(calls) == 1


@pytest.mark.parametrize("text", ['```json\n{"title":"x"}\n```',
    '{"title":"x","title":"y"}', '{"title":NaN}', '{"title":1e400}', '[]'])
def test_invalid_json_rejected_without_retry(monkeypatch, text):
    calls = install(monkeypatch, text)
    with pytest.raises(RuntimeError, match="JSON|object"):
        api.generate_structured("local-model", HISTORY, SCHEMA)
    assert len(calls) == 1


@pytest.mark.parametrize("text", ['{}', '{"title":7}', '{"title":"ok","extra":"bad"}'])
def test_server_ignoring_schema_cannot_publish_invalid_result(monkeypatch, text):
    calls = install(monkeypatch, text)
    with pytest.raises(RuntimeError, match="schema"):
        api.generate_structured("local-model", HISTORY, SCHEMA)
    assert len(calls) == 1


def test_unsupported_model_fails_before_provider(monkeypatch):
    monkeypatch.setattr(config, "get_provider", lambda: pytest.fail("No provider access"))
    with pytest.raises(ValueError, match="not enabled"):
        api.generate_structured("unsupported", HISTORY, SCHEMA)


def test_full_source_input_cannot_be_silently_trimmed(monkeypatch):
    calls = install(monkeypatch)
    monkeypatch.setattr(config, "get_max_context_tokens", lambda *args: 1000)
    with pytest.raises(api.InputTooLargeError):
        api.generate_structured("local-model", [{"role": "user", "content": "Source " * 1500}, *HISTORY], SCHEMA)
    assert calls == []


def test_explicit_tool_budget_preserves_full_input_when_chat_reserve_would_trim(monkeypatch):
    calls = install(monkeypatch)
    source_history = [{"role": "user", "content": "x" * 13000}, *HISTORY]
    api.generate_structured("local-model", source_history, SCHEMA, max_output_tokens=1000)
    assert calls[0]["messages"] == source_history
    assert calls[0]["max_tokens"] == 1000


def test_invalid_citation_schema_reports_field_without_echoing_model_text(monkeypatch):
    from rag_chatbot.tools.citations import scope_citation_schema
    schema = scope_citation_schema({"type": "object", "properties": {
        "source_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["source_ids"], "additionalProperties": False}, ["s1", "s2"])
    calls = install(monkeypatch, text='{"source_ids":["PRIVATE_MODEL_TEXT"]}')
    with pytest.raises(RuntimeError, match="source_ids.0: a value is outside") as error:
        api.generate_structured("local-model", HISTORY, schema)
    assert "PRIVATE_MODEL_TEXT" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("kwargs", [{"schema_name": "bad name"}, {"max_output_tokens": 0},
                                    {"max_output_tokens": True}, {"max_output_tokens": 50000}])
def test_invalid_options_fail_without_request(monkeypatch, kwargs):
    calls = install(monkeypatch)
    with pytest.raises(ValueError):
        api.generate_structured("local-model", HISTORY, SCHEMA, **kwargs)
    assert not calls
