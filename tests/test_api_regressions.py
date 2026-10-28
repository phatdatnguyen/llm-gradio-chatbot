"""Local chat behavior, including cancellation and incomplete replies."""
from types import SimpleNamespace

import pytest

from chatbot import api, config


HISTORY = [{"role": "user", "content": "Hello"}]


@pytest.fixture(autouse=True)
def local_model(monkeypatch):
    monkeypatch.setattr(config, "resolve_model", lambda value: value or "local-model")
    monkeypatch.setattr(config, "get_max_context_tokens", lambda *args: 8192)
    monkeypatch.setattr(config, "supports_vision", lambda value: False)


def response(text="Answer", finish="stop", refusal=None):
    return {"choices": [{"finish_reason": finish, "message": {"content": text, "refusal": refusal}}]}


def chunk(text=None, finish=None, **extra):
    return {"choices": [{"finish_reason": finish, "delta": {"content": text, **extra}}]}


def install(monkeypatch, result):
    calls = []
    def chat(payload):
        calls.append(payload)
        return result
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(chat=chat))
    return calls


def test_local_chat_preserves_system_and_sampling_without_cloud_options(monkeypatch):
    calls = install(monkeypatch, response())
    assert api.send_chat("local-model", HISTORY, temperature=0.4, top_p=0.8,
                         system_prompt="Be brief") == "Answer"
    assert calls[0]["messages"] == [{"role": "system", "content": "Be brief"}, *HISTORY]
    assert calls[0]["temperature"] == 0.4 and calls[0]["top_p"] == 0.8
    assert calls[0]["max_tokens"] > 4096
    assert not {"reasoning", "reasoning_effort", "tools", "input"} & calls[0].keys()


@pytest.mark.parametrize("finish", [None, "content_filter"])
def test_incomplete_blocking_response_is_not_success(monkeypatch, finish):
    calls = install(monkeypatch, response(finish=finish))
    with pytest.raises(RuntimeError, match="incomplete"):
        api.send_chat("local-model", HISTORY)
    assert len(calls) == 1


def test_reply_limit_preserves_blocking_answer_and_citations_without_retry(monkeypatch):
    calls = install(monkeypatch, response("Measured value is 7 [1].", finish="length"))
    result = api.send_chat("local-model", HISTORY)
    assert result.startswith("Measured value is 7 [1].")
    assert result.count(api.REPLY_LIMIT_NOTICE) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("text, thinking", [(None, None), ("  ", "internal reasoning must not become an answer")])
def test_length_without_answer_is_actionable_and_does_not_publish_thinking(monkeypatch, text, thinking):
    result = response(text, finish="length")
    result["choices"][0]["message"]["reasoning_content"] = thinking
    calls = install(monkeypatch, result)
    with pytest.raises(RuntimeError, match="before producing an answer") as error:
        api.send_chat("local-model", HISTORY)
    assert "internal reasoning" not in str(error.value)
    assert len(calls) == 1


def test_explicit_strict_completion_still_rejects_length(monkeypatch):
    calls = install(monkeypatch, response("Partial transcription", finish="length"))
    with pytest.raises(RuntimeError, match="incomplete"):
        api.send_chat("local-model", HISTORY, allow_partial=False)
    assert len(calls) == 1


@pytest.mark.parametrize("text", [None, "  ", {"text": "wrong shape"}])
def test_empty_or_malformed_reply_is_reported(monkeypatch, text):
    install(monkeypatch, response(text))
    with pytest.raises(RuntimeError, match="empty|invalid"):
        api.send_chat("local-model", HISTORY)


def test_refusal_remains_displayable_in_normal_chat(monkeypatch):
    install(monkeypatch, response(None, refusal="Declined"))
    assert api.send_chat("local-model", HISTORY) == "Declined"


def test_switching_to_text_model_detects_images_in_older_turns():
    with pytest.raises(ValueError, match="cannot read images"):
        api._build_call("local-model", [{"role": "user", "content": "See", "image_url": "data:image/png;base64,AA=="}, *HISTORY],
                        1, 1, "")


@pytest.mark.parametrize("url", ["https://example.org/image.png", "http://127.0.0.1/image.png",
                                  "file:///tmp/image.png", "assets/missing.png"])
def test_old_remote_images_are_not_forwarded_or_rewritten(monkeypatch, url):
    monkeypatch.setattr(config, "supports_vision", lambda model: True)
    monkeypatch.setattr(config, "get_provider", lambda: pytest.fail("Must reject before inference"))
    history = [{"role": "user", "content": "Describe", "image_url": url}]
    with pytest.raises(ValueError, match="upload a local copy"):
        api.send_chat("local-model", history)
    assert history == [{"role": "user", "content": "Describe", "image_url": url}]


def test_local_image_data_can_still_be_sent(monkeypatch):
    monkeypatch.setattr(config, "supports_vision", lambda model: True)
    calls = install(monkeypatch, response())
    url = "data:image/png;base64,AA=="
    assert api.send_chat("local-model", [{"role": "user", "content": "Describe", "image_url": url}]) == "Answer"
    assert calls[0]["messages"][0]["content"][-1]["image_url"]["url"] == url


def test_stream_keeps_text_and_ignores_thinking_metadata(monkeypatch):
    events = [chunk(reasoning_content="private reasoning"), chunk("Answer"), chunk(finish="stop"), {"choices": []}]
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(stream_chat=lambda payload: iter(events)))
    final = []
    assert list(api.stream_chat("local-model", HISTORY, final_text_callback=final.append)) == ["Answer"]
    assert final == ["Answer"]


def test_stream_cleanup_when_user_stops(monkeypatch):
    closed = []
    def stream(payload):
        try:
            yield chunk("First")
            yield chunk("Second", finish="stop")
        finally:
            closed.append(True)
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(stream_chat=stream))
    output = api.stream_chat("local-model", HISTORY)
    assert next(output) == "First"
    output.close()
    assert closed == [True]


@pytest.mark.parametrize("events", [
    [chunk("Measured value "), chunk("is 7 [1].", finish="length")],
    [chunk("Measured value is 7 [1]."), chunk(finish="length"), {"choices": []}],
])
def test_stream_reply_limit_keeps_last_delta_and_annotates_final_callback(monkeypatch, events):
    calls, closed, final = [], [], []
    def stream(payload):
        calls.append(payload)
        try:
            yield from events
        finally:
            closed.append(True)
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(stream_chat=stream))
    text = "".join(api.stream_chat("local-model", HISTORY, final_text_callback=final.append))
    assert text.startswith("Measured value is 7 [1].")
    assert text.count("is 7 [1].") == 1
    assert text.count(api.REPLY_LIMIT_NOTICE) == 1
    assert final == [text.strip()]
    assert closed == [True] and len(calls) == 1


@pytest.mark.parametrize("events", [
    [chunk(finish="length")],
    [chunk("  "), chunk(finish="length")],
    [chunk(reasoning_content="internal reasoning"), chunk(finish="length")],
    [chunk(finish="length", reasoning_content="internal reasoning")],
])
def test_stream_length_without_answer_fails_without_retry_or_callback(monkeypatch, events):
    calls, closed, final, pieces = [], [], [], []
    def stream(payload):
        calls.append(payload)
        try:
            yield from events
        finally:
            closed.append(True)
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(stream_chat=stream))
    with pytest.raises(RuntimeError, match="before producing an answer") as error:
        for delta in api.stream_chat("local-model", HISTORY, final_text_callback=final.append):
            pieces.append(delta)
    assert "internal reasoning" not in "".join(pieces) + str(error.value)
    assert api.REPLY_LIMIT_NOTICE not in "".join(pieces)
    assert final == [] and closed == [True] and len(calls) == 1


def test_stopping_after_final_text_does_not_complete_callback(monkeypatch):
    closed, final = [], []
    def stream(payload):
        try:
            yield chunk("Useful partial [1].", finish="length")
        finally:
            closed.append(True)
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(stream_chat=stream))
    output = api.stream_chat("local-model", HISTORY, final_text_callback=final.append)
    assert next(output) == "Useful partial [1]."
    output.close()
    assert closed == [True] and final == []


@pytest.mark.parametrize("events,match", [
    ([chunk("partial")], "before the model finished"),
    ([chunk("partial", finish="content_filter")], "incomplete"),
    ([chunk(finish="stop")], "empty"),
    ([{"error": "failed"}], "error"),
    ([{"choices": "wrong"}], "invalid"),
])
def test_failed_stream_does_not_publish_final(monkeypatch, events, match):
    monkeypatch.setattr(config, "get_provider", lambda: SimpleNamespace(stream_chat=lambda payload: iter(events)))
    final = []
    with pytest.raises(RuntimeError, match=match):
        list(api.stream_chat("local-model", HISTORY, final_text_callback=final.append))
    assert final == []


@pytest.mark.parametrize("kwargs", [{"temperature": float("nan")}, {"top_p": True}, {"temperature": -1}])
def test_invalid_sampling_fails_before_request(monkeypatch, kwargs):
    monkeypatch.setattr(config, "get_provider", lambda: pytest.fail("No provider access"))
    with pytest.raises(ValueError):
        api.send_chat("local-model", HISTORY, **kwargs)
