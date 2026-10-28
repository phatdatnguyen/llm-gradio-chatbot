"""Offline context estimates and complete-turn history trimming for local models.

Local servers can load arbitrary tokenizers and chat templates. Until a provider
supplies an accurate tokenizer, one UTF-8 byte is charged as one estimated token,
with additional framing reserves. This deliberately conservative estimate is not
the model's actual token count. It needs no downloaded assets or network access.
The server's configured context limit, rather than a model-name table, governs
trimming. Image token reserves must be configured separately.
"""

from functools import lru_cache

from . import config
from .log import logger
from .messages import serialize_message_value

# Allow space for local chat-template role markers and separators. These are
# conservative estimates, not a claim about any particular model's tokenizer.
TOKENS_PER_MESSAGE = 32
TOKENS_PER_REQUEST = 32


class InputTooLargeError(Exception):
    """The newest complete turn cannot fit without silently discarding input."""

    def __init__(self, model, needed_tokens, budget_tokens):
        self.model = model
        self.needed_tokens = needed_tokens
        self.budget_tokens = budget_tokens
        super().__init__(
            f"This input needs about {needed_tokens:,} estimated tokens, which "
            f"does not fit {model}'s usable context of {budget_tokens:,} tokens. "
            "Try a shorter document, reduce the reply limit, or configure a "
            "larger context in the local model server."
        )


class ImageTokenBudgetError(ValueError):
    """An image cannot safely be budgeted for the selected local model."""


class Utf8BudgetEncoding:
    """A byte-budget codec, not a model tokenizer.

    ``decode`` is strict by default: callers splitting encoded text must use
    UTF-8 boundaries rather than losing or replacing partial Unicode characters.
    Special-token spellings are ordinary text and receive no special handling.
    """

    name = "utf8-byte-estimate"

    def encode(self, text, *, disallowed_special=(), **kwargs):
        return text.encode("utf-8")

    def encode_ordinary(self, text):
        return self.encode(text)

    def decode(self, values, errors="strict"):
        return bytes(values).decode("utf-8", errors=errors)


@lru_cache(maxsize=128)
def get_encoding(model):
    """Return an entirely local byte estimator for any server model identifier."""
    return Utf8BudgetEncoding()


def _count_image_tokens(image_url, model, detail="auto"):
    """Reserve configured vision tokens without fetching or decoding an image."""
    budget = config.image_token_budget(model, detail)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ImageTokenBudgetError(
            "Image context use is unknown for this local model. Configure a "
            "positive image token reserve before sending image attachments."
        )
    return budget


def _count_content_tokens(content, model):
    """Count text and structured framing; charge typed images separately."""
    if not isinstance(content, list):
        return count_text_tokens(serialize_message_value(content), model)

    total = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
            image_url = part.get("image_url")
            detail = (image_url.get("detail", "auto") if isinstance(image_url, dict)
                      else part.get("detail", "auto"))
            total += _count_image_tokens(image_url, model, detail)
        else:
            total += count_text_tokens(serialize_message_value(part), model)
    return total


def count_message_tokens(message, model):
    total = TOKENS_PER_MESSAGE
    for key, value in message.items():
        if key == "content":
            total += _count_content_tokens(value, model)
        elif key == "image_url" and value:
            detail = value.get("detail", "auto") if isinstance(value, dict) else "auto"
            total += _count_image_tokens(value, model, detail)
        else:
            total += count_text_tokens(serialize_message_value(value), model)
    return total


def count_tokens(messages, model):
    return TOKENS_PER_REQUEST + sum(count_message_tokens(message, model) for message in messages)


def count_text_tokens(text, model):
    """Return conservative estimated tokens, including all Unicode bytes."""
    if not text:
        return 0
    return len(get_encoding(model).encode(text, disallowed_special=()))


def reserve_for_output(model):
    """Keep a quarter-context planning reserve while fitting input/evidence.

    This is a minimum allowance, not the request's generation limit. Shorter
    prompts leave more room, all of which is made available by reply_budget().
    An explicit user cap can reduce this reserve, but cannot consume input space.
    """
    limit = config.max_output_tokens(model)
    context = config.get_max_context_tokens(model)
    return max(0, min(limit, context // 4) if limit is not None else context // 4)


def reply_budget(messages, model):
    """Use remaining context after the complete wire prompt, subject to user cap.

    count_tokens includes conservative text, image and chat-template estimates.
    Never request more output than fits alongside the input we actually send.
    """
    context = config.get_max_context_tokens(model)
    needed = count_tokens(messages, model)
    available = context - needed
    if available <= 0:
        raise InputTooLargeError(model, needed, max(context - 1, 0))
    limit = config.max_output_tokens(model)
    return min(limit, available) if limit is not None else available


def _conversation_turns(messages):
    """Group each user prompt with all replies belonging to that prompt."""
    turns = []
    for message in messages:
        if not turns or message.get("role") == "user":
            turns.append([])
        turns[-1].append(message)
    return turns


def trim_history(messages, model, system_prompt="", reserved_tokens=None):
    """Drop complete oldest turns while preserving the newest turn and prompt.

    The external system prompt, message framing and reply reserve all count
    toward the loaded context limit. An oversized newest turn raises instead of
    returning an empty payload or an assistant reply without its user prompt.
    """
    if not messages:
        return []
    if reserved_tokens is None:
        reserved_tokens = reserve_for_output(model)
    if isinstance(reserved_tokens, bool) or not isinstance(reserved_tokens, int) or reserved_tokens < 0:
        raise ValueError("The reply token reserve must be a nonnegative integer.")

    max_context = config.get_max_context_tokens(model)
    system_tokens = count_message_tokens(
        {"role": "system", "content": system_prompt}, model
    ) if system_prompt else 0
    budget = max_context - reserved_tokens - system_tokens - TOKENS_PER_REQUEST
    if budget <= 0:
        raise InputTooLargeError(model, system_tokens + TOKENS_PER_REQUEST,
                                 max(max_context - reserved_tokens, 0))

    kept_turns = []
    total = 0
    for turn in reversed(_conversation_turns(messages)):
        turn_tokens = sum(count_message_tokens(message, model) for message in turn)
        if total + turn_tokens > budget:
            if not kept_turns:
                raise InputTooLargeError(model, turn_tokens, budget)
            break
        kept_turns.append(turn)
        total += turn_tokens
    trimmed = [message for turn in reversed(kept_turns) for message in turn]
    dropped = len(messages) - len(trimmed)
    if dropped:
        logger.info(
            "Trimmed %d old message(s) to fit %s (%d/%d estimated tokens used).",
            dropped, model, total, budget,
        )
    return trimmed
