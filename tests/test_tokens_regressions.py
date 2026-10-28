"""Offline context-budget and complete-turn trimming regression coverage."""

import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import messages, tokens


MODEL = "local-model.gguf"


@pytest.mark.parametrize("text", ["Xin ch\u00e0o Vi\u1ec7t Nam \u6f22\u5b57 \U0001f600 <|endoftext|> <|fim_prefix|>", ""])
def test_estimator_is_offline_unicode_safe_and_counts_every_byte(monkeypatch, text):
    def deny_network(*args, **kwargs):
        raise AssertionError("Token budgeting must not use the network")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    encoding = tokens.get_encoding(MODEL)
    encoded = encoding.encode(text, disallowed_special=())
    assert encoded == text.encode("utf-8")
    assert encoding.decode(encoded) == text
    assert encoding.decode(list(encoded)) == text
    assert tokens.count_text_tokens(text, MODEL) == len(encoded)
    assert encoding.name == "utf8-byte-estimate"


def test_byte_decoder_rejects_partial_unicode_instead_of_corrupting_it():
    encoding = tokens.get_encoding(MODEL)
    encoded = encoding.encode("😀")
    with pytest.raises(UnicodeDecodeError):
        encoding.decode(encoded[:2])


def test_structured_content_and_schema_framing_are_charged():
    plain = {"role": "user", "content": "question"}
    structured = {"role": "user", "content": [{"type": "text", "text": "question"}]}
    with_schema = {**plain, "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}}
    assert tokens.count_message_tokens(structured, MODEL) > tokens.count_message_tokens(plain, MODEL)
    schema_bytes = len(messages.serialize_message_value(with_schema["schema"]).encode("utf-8"))
    assert tokens.count_message_tokens(with_schema, MODEL) - tokens.count_message_tokens(plain, MODEL) == schema_bytes
    assert tokens.count_tokens([plain], MODEL) > tokens.count_text_tokens("question", MODEL)


@pytest.mark.parametrize("context", [128, 32768])
def test_reply_reserve_scales_down_to_small_contexts(monkeypatch, context):
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)
    monkeypatch.setattr(tokens.config, "max_output_tokens", lambda model: 4096)
    reserve = tokens.reserve_for_output(MODEL)
    assert reserve == min(4096, context // 4)
    history = [{"role": "user", "content": "Hi"}]
    assert tokens.trim_history(history, MODEL) == history


def test_auto_output_uses_all_remaining_context_above_old_ceiling(monkeypatch):
    context = 131072
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda model: context)
    monkeypatch.setattr(tokens.config, "max_output_tokens", lambda model: None)
    history = [{"role": "system", "content": "Answer fully."}, {"role": "user", "content": "Hi"}]
    budget = tokens.reply_budget(history, MODEL)
    assert budget == context - tokens.count_tokens(history, MODEL)
    assert budget > 4096
    assert budget > tokens.reserve_for_output(MODEL)


@pytest.mark.parametrize("cap", [1024, 100000])
def test_explicit_cap_never_exceeds_available_context(monkeypatch, cap):
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda model: 8192)
    monkeypatch.setattr(tokens.config, "max_output_tokens", lambda model: cap)
    history = [{"role": "user", "content": "x" * 3000}]
    assert tokens.reply_budget(history, MODEL) == min(cap, 8192 - tokens.count_tokens(history, MODEL))


def test_auto_reply_budget_charges_images_and_rejects_exhausted_context(monkeypatch):
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda model: 8192)
    monkeypatch.setattr(tokens.config, "max_output_tokens", lambda model: None)
    monkeypatch.setattr(tokens.config, "image_token_budget", lambda *args: 4096)
    history = messages.prepare_chat_messages([
        {"role": "user", "content": "Describe", "image_url": "data:image/png;base64,AAAA"}])
    assert tokens.reply_budget(history, MODEL) < 4096
    history.append({"role": "user", "content": "x" * 8192})
    with pytest.raises(tokens.InputTooLargeError):
        tokens.reply_budget(history, MODEL)


def test_trim_history_drops_answer_whose_question_no_longer_fits(monkeypatch):
    history = [
        {"role": "user", "content": "old question " * 100},
        {"role": "assistant", "content": "An old answer."},
        {"role": "user", "content": "A new question."},
    ]
    context = tokens.count_tokens(history[1:], MODEL)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)
    assert tokens.trim_history(history, MODEL, reserved_tokens=0) == history[2:]
    assert len(history) == 3


def test_trim_history_keeps_complete_recent_exchange(monkeypatch):
    history = [
        {"role": "user", "content": "old question " * 100},
        {"role": "assistant", "content": "An old answer."},
        {"role": "user", "content": "A recent question."},
        {"role": "assistant", "content": "A recent answer."},
        {"role": "user", "content": "A follow-up question."},
    ]
    context = tokens.count_tokens(history[1:], MODEL)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)
    assert tokens.trim_history(history, MODEL, reserved_tokens=0) == history[2:]


def test_newest_assistant_reply_is_never_detached_from_its_question(monkeypatch):
    history = [
        {"role": "user", "content": "large question " * 100},
        {"role": "assistant", "content": "Short answer."},
    ]
    context = tokens.count_tokens(history[1:], MODEL)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)
    with pytest.raises(tokens.InputTooLargeError):
        tokens.trim_history(history, MODEL, reserved_tokens=0)


def test_trim_history_keeps_entire_history_at_exact_budget(monkeypatch):
    history = [
        {"role": "user", "content": "A question."},
        {"role": "assistant", "content": "An answer."},
        {"role": "user", "content": "A follow-up question."},
    ]
    context = tokens.count_tokens(history, MODEL)
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)
    assert tokens.trim_history(history, MODEL, reserved_tokens=0) == history


def test_system_prompt_and_reply_reserve_are_charged(monkeypatch):
    history = [{"role": "user", "content": "old"}, {"role": "user", "content": "new"}]
    prompt = "Follow these instructions."
    prompt_cost = tokens.count_message_tokens({"role": "system", "content": prompt}, MODEL)
    context = tokens.count_tokens(history, MODEL) + prompt_cost + 16
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: context)
    assert tokens.trim_history(history, MODEL, system_prompt=prompt, reserved_tokens=16) == history
    assert tokens.trim_history(history, MODEL, system_prompt=prompt, reserved_tokens=17) == history[-1:]


def test_oversized_newest_input_raises_without_empty_payload(monkeypatch):
    history = [{"role": "user", "content": "x" * 500}]
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: 256)
    with pytest.raises(tokens.InputTooLargeError, match="estimated tokens"):
        tokens.trim_history(history, MODEL, reserved_tokens=0)
    assert history[0]["content"] == "x" * 500


@pytest.mark.parametrize("reserve", [-1, True, 1.5])
def test_invalid_reply_reserve_cannot_bypass_budget(reserve):
    with pytest.raises(ValueError, match="nonnegative integer"):
        tokens.trim_history([{"role": "user", "content": "hi"}], MODEL, reserved_tokens=reserve)


def test_prompt_or_reply_exhausting_context_is_rejected(monkeypatch):
    monkeypatch.setattr(tokens.config, "get_max_context_tokens", lambda *args: 128)
    with pytest.raises(tokens.InputTooLargeError):
        tokens.trim_history([{"role": "user", "content": "hi"}], MODEL, reserved_tokens=128)
    with pytest.raises(tokens.InputTooLargeError):
        tokens.trim_history([{"role": "user", "content": "hi"}], MODEL,
                            reserved_tokens=0, system_prompt="x" * 128)


@pytest.mark.parametrize("prepare", [lambda history: history, messages.prepare_chat_messages])
def test_image_bytes_are_not_counted_as_text(monkeypatch, prepare):
    monkeypatch.setattr(tokens.config, "image_token_budget", lambda *args: 256)
    small = prepare([{"role": "user", "content": "Describe it.", "image_url": "data:image/png;base64," + "A" * 100}])
    large = prepare([{"role": "user", "content": "Describe it.", "image_url": "data:image/png;base64," + "A" * 2_000_000}])
    assert tokens.count_tokens(small, MODEL) == tokens.count_tokens(large, MODEL)


@pytest.mark.parametrize("invalid", [None, 0, True, 1.5])
def test_unbudgeted_image_attachments_are_rejected(monkeypatch, invalid):
    monkeypatch.setattr(tokens.config, "image_token_budget", lambda *args: invalid)
    history = [{"role": "user", "content": "Describe it.", "image_url": "data:image/png;base64,AAAA"}]
    with pytest.raises(tokens.ImageTokenBudgetError, match="positive image token reserve"):
        tokens.count_tokens(history, MODEL)


def test_each_image_uses_its_configured_reserve_and_detail(monkeypatch):
    monkeypatch.setattr(tokens.config, "image_token_budget", lambda model, detail: 128 if detail == "low" else 256)
    low = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}
    high = {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"}
    single = {"role": "user", "content": [low]}
    double = {"role": "user", "content": [low, high]}
    assert tokens.count_message_tokens(double, MODEL) - tokens.count_message_tokens(single, MODEL) == 256


def test_base64_inside_plain_user_text_remains_in_text_budget():
    text = "data:image/png;base64," + "A" * 100
    with_text = {"role": "user", "content": text}
    without_text = {"role": "user", "content": ""}
    assert tokens.count_message_tokens(with_text, MODEL) - tokens.count_message_tokens(without_text, MODEL) == len(text)
