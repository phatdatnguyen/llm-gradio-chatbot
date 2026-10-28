"""Offline regression tests for UI state and event wiring."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import api
from chatbot import history_store


@pytest.fixture
def ui(monkeypatch, local_models):
    # Build the UI without inspecting the user's saved conversations or sending
    # Gradio analytics. Local model calls are stubbed individually below.
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(history_store, "get_history_file_list", lambda: [])
    module = importlib.import_module("chatbot.chatbot")
    with module.gr.Blocks() as demo:
        components = module.build_tab()
    return SimpleNamespace(**vars(module), **vars(components), demo=demo)


def _send(ui, history, **overrides):
    values = {
        "llm_model": "local-chat", "temperature": 1,
        "top_p": 1, "text": "Please help with this draft", "image": None,
        "document": None, "history": history,
        "system_prompt": "", "stream_output": False,
    }
    values.update(overrides)
    return ui.on_user_input(**values)


@pytest.mark.parametrize("stream_output", [False, True])
def test_failed_send_preserves_prompt_for_copying_without_resending(ui, monkeypatch,
                                                                  stream_output):
    previous = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    warnings = []
    monkeypatch.setattr(ui.gr, "Warning", warnings.append)

    def fail(*args, **kwargs):
        raise RuntimeError("Test request failed")

    def fail_mid_stream(*args, **kwargs):
        yield "Partial response"
        fail()

    monkeypatch.setattr(api, "send_chat", fail)
    monkeypatch.setattr(api, "stream_chat", fail_mid_stream)

    final = list(_send(ui, previous, stream_output=stream_output))[-1]
    state, display, *inputs = final
    assert state == previous
    assert len(previous) == 2
    assert display[:-2] == previous
    assert display[-2] == {"role": "user", "content": "Please help with this draft"}
    assert "Request failed" in display[-1]["content"]
    assert "not added to the conversation" in display[-1]["content"]
    assert all(value == ui.gr.skip() for value in inputs)
    assert warnings == ["Test request failed"]


def test_failed_load_keeps_active_chat(ui, monkeypatch):
    def fail(name):
        raise FileNotFoundError("Saved file is missing")

    monkeypatch.setattr(history_store, "load_history", fail)
    state, display, status = ui.on_load_history("missing")
    assert state == ui.gr.skip()
    assert display == ui.gr.skip()
    assert "Saved file is missing" in status


def test_successful_load_replaces_active_chat(ui, monkeypatch):
    saved = [{"role": "user", "content": "Saved question"}]
    monkeypatch.setattr(history_store, "load_history", lambda name: (saved, "Loaded"))
    assert ui.on_load_history("saved") == (saved, saved, "Loaded")


def test_send_triggers_share_pending_event_and_chat_changes_cancel_it(ui):
    dependencies = ui.demo.config["dependencies"]
    send = next(item for item in dependencies if item["id"] == ui.send_event["id"])
    assert set(map(tuple, send["targets"])) == {
        (ui.text_input._id, "submit"), (ui.send_button._id, "click"),
    }
    assert send["trigger_mode"] == "once"
    for button in (ui.stop_button, ui.new_chat_button, ui.load_button):
        assert any(
            (button._id, "click") in map(tuple, item["targets"])
            and send["id"] in item["cancels"]
            for item in dependencies
        )


def test_closing_chat_event_closes_the_underlying_response_stream(ui, monkeypatch):
    closed = []

    def respond():
        try:
            yield "First token"
            yield "Unconsumed token"
        finally:
            closed.append(True)

    # Retain a reference so closing is tested independently of CPython garbage
    # collection. Stop/New/Load cancellation must release the response directly.
    response = respond()
    monkeypatch.setattr(api, "stream_chat", lambda *args, **kwargs: response)
    event = _send(ui, [], stream_output=True)
    next(event)
    assert next(event)[0][-1]["content"] == "First token"
    assert not closed
    event.close()
    assert closed == [True]


@pytest.mark.parametrize("stream_output", [False, True])
def test_local_attachments_and_settings_reach_only_chat_generation(ui, monkeypatch, stream_output):
    user = {"role": "user", "content": "Attached document", "image_url": "data:image/png;base64,AA=="}
    built, requests = [], []
    def build(text, **kwargs):
        built.append((text, kwargs))
        return user
    def send(model, history, **kwargs):
        requests.append((model, history, kwargs))
        return "Answer"
    def stream(model, history, **kwargs):
        yield send(model, history, **kwargs)
    monkeypatch.setattr(api, "build_user_message", build)
    monkeypatch.setattr(api, "send_chat", send)
    monkeypatch.setattr(api, "stream_chat", stream)
    monkeypatch.setattr(ui.config, "supports_vision", lambda model: True)
    updates = list(_send(ui, [], stream_output=stream_output, image="image.png", document="notes.txt",
                         temperature=0.4, top_p=0.8, system_prompt="Be concise"))
    assert built == [("Please help with this draft", {"image_path": "image.png", "document_path": "notes.txt"})]
    assert len(requests) == 1
    assert requests[0][0] == "local-chat"
    assert requests[0][1][0] == user
    assert requests[0][2] == {"temperature": 0.4, "top_p": 0.8, "system_prompt": "Be concise"}
    assert all(len(update) == 5 for update in updates)
    assert updates[-1][0] == [user, {"role": "assistant", "content": "Answer"}]


def test_text_model_rejects_local_image_without_dropping_attachment(ui, monkeypatch):
    warnings = []
    monkeypatch.setattr(ui.gr, "Warning", warnings.append)
    monkeypatch.setattr(ui.config, "supports_vision", lambda model: False)
    monkeypatch.setattr(api, "build_user_message", lambda *args, **kwargs: pytest.fail("Unsupported image must not be read"))
    previous = [{"role": "user", "content": "Earlier"}]
    result = list(_send(ui, previous, image="image.png"))
    assert len(result) == 1 and result[0][0] == previous
    assert result[0][2:] == (ui.gr.skip(),) * 3
    assert "vision-capable model" in warnings[0]
