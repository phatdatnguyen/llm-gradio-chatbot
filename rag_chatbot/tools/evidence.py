"""Bounded whole-source evidence preparation for Studio tools."""

import json
import os

from chatbot import config, tokens


def _setting(name, default, minimum, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


# Zero uses all input space available in the loaded context. Positive values are
# optional user caps, not fixed app ceilings. Limits are read at process start.
MAX_INPUT_TOKENS = _setting("STUDIO_MAX_INPUT_TOKENS", 0, 0, 1048576)
MAX_CONSOLIDATION_ROUNDS = _setting("STUDIO_MAX_CONSOLIDATION_ROUNDS", 20, 1, 100)
MAX_MODEL_CALLS = _setting("STUDIO_MAX_MODEL_CALLS", 256, 1, 10000)
MAX_TOOL_OUTPUT_TOKENS = 32768
MAX_DOCUMENTS = 40
MAX_CHUNKS = 512
MAX_SOURCE_CHARACTERS = 500_000


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def token_size(value, model):
    return tokens.count_text_tokens(json_text(value), model)


def processing_budget(model, *, overhead=1024, maximum_output=MAX_TOOL_OUTPUT_TOKENS):
    """Reserve output and prompt framing inside the server's effective context."""
    context = config.get_max_context_tokens(model)
    # Studio has its own bounded output budget, independent of the chat cap.
    output = min(maximum_output, max(1, context // 4))
    available = context - output - overhead
    if MAX_INPUT_TOKENS:
        available = min(MAX_INPUT_TOKENS, available)
    if available < 256:
        raise ValueError("The local model context is too small for these tool instructions and output schema. "
                         "Use shorter instructions or reload the model with a larger context.")
    return available, output


def bounded_batches(items, model, *, budget=None):
    """Preserve every item and its ordering; reject oversized individual items."""
    if budget is None:
        budget, _ = processing_budget(model)
    batches, batch = [], []
    # Counting each serialized item plus framing is conservative for a JSON list.
    count = 2
    for item in items:
        size = token_size(item, model) + 2
        if size + 2 > budget:
            raise ValueError("An evidence section is too large for a tool batch. "
                             "Reindex with smaller chunks or select smaller sources.")
        if batch and count + size > budget:
            batches.append(batch)
            batch, count = [], 2
        batch.append(item)
        count += size
    if batch:
        batches.append(batch)
    return batches


def selected_source_snapshot(store, selected_ids, collection):
    """Read the complete bounded snapshot and assign stable run-local aliases."""
    snapshot = store.snapshot_sources(
        selected_ids, collection, max_documents=MAX_DOCUMENTS, max_chunks=MAX_CHUNKS,
        max_characters=MAX_SOURCE_CHARACTERS,
    )
    # Short, run-local identifiers reduce prompt overhead; persisted references
    # always contain the original stable chunk IDs.
    aliases = {f"s{index}": source for index, source in enumerate(snapshot["sources"], 1)}
    return snapshot, aliases


def prepare_snapshot(store, selected_ids, collection, model, *, budget=None):
    snapshot, aliases = selected_source_snapshot(store, selected_ids, collection)
    document_inputs = {document["id"]: [] for document in snapshot["documents"]}
    for alias, source in aliases.items():
        document_inputs[source["document_id"]].append({
            "source_id": alias, "filename": source["filename"],
            "locator": source["locator"], "text": source["text"],
        })
    # Preflight every batch before any model request. A too-large late document
    # must not cost the user calls for the earlier documents.
    batches = {identifier: bounded_batches(items, model, budget=budget)
               for identifier, items in document_inputs.items()}
    return snapshot, aliases, batches


def validate_source_ids(value, allowed, *, allow_empty=False):
    prefix = "The tool returned missing, duplicated, or unknown source citations. "
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(prefix + "Citations must be a list of exact source IDs.")
    if not value and not allow_empty:
        raise ValueError(prefix + "At least one supporting source ID is required.")
    if len(value) != len(set(value)):
        raise ValueError(prefix + "Each source ID must appear only once in a citation list.")
    if any(item not in allowed for item in value):
        raise ValueError(prefix + "A citation refers to a source outside this request's evidence.")
    return value


def require_object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("The tool returned an invalid output structure.")
    return value


def require_text(value, *, limit=6000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("The tool returned empty or oversized text.")
    return value
