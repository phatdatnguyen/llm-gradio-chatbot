"""A context refresh during batch planning must never discard source passages."""

from types import SimpleNamespace

from chatbot import api, config
from rag_chatbot.tools import evidence, service


def test_context_increase_during_singleton_fit_keeps_evidence(monkeypatch):
    model = "local-refresh-test"
    context = {"loaded": 131072}
    source = {"id": "chunk-1", "document_id": "document-1", "filename": "report.txt",
              "locator": "paragraph 1", "text": "The launch date is June 12."}
    document = {"id": "document-1", "name": "report.txt"}
    guards = []
    store = SimpleNamespace(
        snapshot_sources=lambda *args, **kwargs: {"documents": [document], "sources": [source]},
        validate_source_snapshot=lambda documents: guards.append(list(documents)),
    )
    monkeypatch.setattr(service, "get_service", lambda workspace: SimpleNamespace(store=store))
    monkeypatch.setattr(config, "supports_structured_output", lambda selected: selected == model)
    monkeypatch.setattr(config, "get_max_context_tokens", lambda selected: context["loaded"])
    monkeypatch.setattr(service, "MAX_INPUT_TOKENS", 0)
    monkeypatch.setattr(evidence, "MAX_INPUT_TOKENS", 0)

    original_size = api.structured_request_tokens
    original_check = service._check_request_budget
    refresh = {"measured": False, "pending": False, "applied": False}

    def measure_with_smaller_loaded_window(*args, **kwargs):
        needed = original_size(*args, **kwargs)
        if not refresh["measured"]:
            # The first complete prompt measurement occurs in fitting_batches.
            # Force its single item just outside the current loaded window.
            context["loaded"] = needed + evidence.MAX_TOOL_OUTPUT_TOKENS - 1
            refresh.update(measured=True, pending=True)
        return needed

    def refresh_before_recheck(*args, **kwargs):
        if refresh["pending"]:
            # A Connections refresh publishes a larger window after the failed
            # fit condition, but before its detailed budget check runs.
            assert not calls
            context["loaded"] = 131072
            refresh.update(pending=False, applied=True)
        return original_check(*args, **kwargs)

    monkeypatch.setattr(api, "structured_request_tokens", measure_with_smaller_loaded_window)
    monkeypatch.setattr(service, "_check_request_budget", refresh_before_recheck)
    calls = []

    def generate(selected, history, schema, **options):
        import json

        prompt = json.loads(history[-1]["content"])
        calls.append((options["schema_name"], prompt))
        if options["schema_name"] == "tool_evidence_notes":
            return {"notes": [{"source_id": row["source_id"], "text": row["text"]}
                              for row in prompt["evidence"]]}
        assert options["schema_name"] == "tool_summary"
        return {"title": "Launch", "sections": [{"heading": "Date", "paragraphs": [
            {"text": source["text"], "source_ids": ["s1"]},
        ]}]}

    monkeypatch.setattr(api, "generate_structured", generate)
    result = list(service.generate_tool("workspace", "summary", [document["id"]], model=model))[-1]["result"]

    assert refresh["applied"]
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_summary"]
    assert calls[0][1]["evidence"][0]["text"] == source["text"]
    assert calls[1][1]["evidence"]["notes"] == [{"text": source["text"], "source_ids": ["s1"]}]
    assert result["coverage"]["processed_chunks"] == 1
    assert result["sources"] == [source]
    assert guards
