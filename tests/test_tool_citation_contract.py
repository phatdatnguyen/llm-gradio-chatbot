"""Local schemas must describe the exact citation contract the validators enforce."""

from copy import deepcopy
import json
from types import SimpleNamespace

import jsonschema
import pytest

from chatbot import config
from rag_chatbot.tools import service
from rag_chatbot.tools import evidence
from rag_chatbot.tools.citations import scope_citation_schema


def _valid(schema, value):
    return jsonschema.Draft202012Validator(schema).is_valid(value)


@pytest.mark.parametrize("citations", [[], ["s2", "s2"], ["s1"], ["unknown"], ["2"], ["[s2]"], [None]])
def test_consolidation_schema_rejects_invalid_ids_within_exact_batch(citations):
    schema = scope_citation_schema(service.POINTS_SCHEMA, ["s2", "s3"])
    assert not _valid(schema, {"points": [{"text": "Evidence", "source_ids": citations}]})
    assert _valid(schema, {"points": [{"text": "s1 is allowed as ordinary prose", "source_ids": ["s2"]}]})


def test_notes_schema_constrains_identifiers_without_disabling_missing_passage_recovery():
    schema = scope_citation_schema(service.NOTES_SCHEMA, ["s7", "s8"])
    assert _valid(schema, {"notes": [{"source_id": "s7", "text": "An evidence note"}]})
    assert _valid(schema, {"notes": []})
    assert not _valid(schema, {"notes": [{"source_id": "s1", "text": "A wrong batch's identifier"}]})
    assert not _valid(schema, {"notes": [{"source_id": 7, "text": "A numeric identifier"}]})


def test_scoping_returns_independent_schemas_without_changing_shared_templates():
    original = deepcopy(service.SUMMARY_SCHEMA)
    first = scope_citation_schema(service.SUMMARY_SCHEMA, ["s1"])
    second = scope_citation_schema(service.SUMMARY_SCHEMA, ["s9"])
    value = {"title": "A title", "sections": [{"heading": "A heading", "paragraphs": [
        {"text": "A finding", "source_ids": ["s1"]}]}]}
    assert _valid(first, value)
    assert not _valid(second, value)
    assert service.SUMMARY_SCHEMA == original
    assert first["$defs"]["source_identifier"]["enum"] == ["s1"]
    assert second["$defs"]["source_identifier"]["enum"] == ["s9"]


def test_optional_citation_fields_stay_optional_but_synthesis_requires_evidence():
    cells = scope_citation_schema(service.CELLS_SCHEMA, ["s2"],
                                  allow_empty_paths=(("cells", "items", "source_ids"),))
    assert _valid(cells, {"cells": [{"column": "Observation", "value": "Not found", "source_ids": []}]})
    assert not _valid(cells, {"cells": [{"column": "Observation", "value": "Found", "source_ids": ["s1"]}]})
    compare = scope_citation_schema(service.COMPARE_SCHEMA, ["s1", "s2"],
        allow_empty_paths=(("aspects", "items", "findings", "items", "source_ids"),))
    value = {"title": "Comparison", "aspects": [{"aspect": "Method", "findings": [
        {"document_id": "a", "status": "found", "text": "Found", "source_ids": ["s1"]},
        {"document_id": "b", "status": "not_found", "text": "Not found", "source_ids": []}],
        "synthesis": {"text": "Only A reports it", "source_ids": ["s1"]}}]}
    assert _valid(compare, value)
    value["aspects"][0]["synthesis"]["source_ids"] = []
    assert not _valid(compare, value)


def test_optional_mind_map_root_does_not_relax_nonroot_semantic_validation():
    schema = scope_citation_schema(service.MIND_MAP_SCHEMA, ["s1"],
                                   allow_empty_paths=(("nodes", "items", "source_ids"),))
    value = {"title": "Map", "nodes": [
        {"id": "r", "parent_id": "", "label": "Root", "description": "", "source_ids": []},
        {"id": "c", "parent_id": "r", "label": "Finding", "description": "Evidence", "source_ids": []}]}
    assert _valid(schema, value)
    with pytest.raises(ValueError, match="source citation"):
        service._mind_map(value, {"s1": {"id": "chunk-1", "document_id": "a"}}, [{"id": "a"}])


class SnapshotStore:
    def __init__(self, *, dense=False):
        self.documents = [{"id": name, "name": name + ".txt", "content_hash": name,
                           "fingerprint": "local-test", "generation": "ready"}
                          for name in ("a", "b")]
        size = 6 if dense else 2
        self.sources = [{"id": f"{document['id']}-chunk-{index}", "document_id": document["id"],
                         "filename": document["name"], "locator": f"Section {index}",
                         "text": ("Evidence measurements and qualifications. " * 16 if dense else "Measured evidence.")
                                 + str(index),
                         "source_path": f"sources/{document['id']}.txt", "kind": "text"}
                        for document in self.documents for index in range(size)]

    def snapshot_sources(self, selected_ids, collection, **bounds):
        documents = [document for identifier in selected_ids for document in self.documents
                     if document["id"] == identifier]
        sources = [source for document in documents for source in self.sources
                   if source["document_id"] == document["id"]]
        return {"documents": deepcopy(documents), "sources": deepcopy(sources)}

    def validate_source_snapshot(self, documents):
        assert documents


@pytest.fixture
def local_model(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_MODEL", "citation-contract-model")
    monkeypatch.setattr(config, "MODEL_CHOICES", ["citation-contract-model"])
    monkeypatch.setitem(config.MODEL_TOKEN_LIMITS, "citation-contract-model", 65536)


@pytest.fixture
def small_evidence_budget(monkeypatch):
    monkeypatch.setattr(service, "MAX_INPUT_TOKENS", 6000)
    monkeypatch.setattr(evidence, "MAX_INPUT_TOKENS", 6000)


def _fake_model(calls, *, invalid_consolidation=None):
    def generate(model, history, schema, **options):
        prompt = json.loads(history[-1]["content"])
        name = options["schema_name"]
        calls.append({"name": name, "prompt": deepcopy(prompt), "schema": deepcopy(schema),
                      "model": model, "history": deepcopy(history), "options": deepcopy(options)})
        evidence = prompt["evidence"]
        if name == "tool_evidence_notes":
            result = {"notes": [{"source_id": item["source_id"], "text": item["text"]} for item in evidence]}
        elif name == "tool_consolidated_notes":
            ids = list(dict.fromkeys(identifier for point in evidence for identifier in point["source_ids"]))
            if invalid_consolidation is not None:
                # Bypass transport validation deliberately: downstream validators
                # must still reject bad data from an injected or faulty backend.
                return {"points": [{"text": "A generated claim", "source_ids": invalid_consolidation}]}
            result = {"points": [{"text": "Consolidated evidence and its qualifications.", "source_ids": ids}]}
        elif name == "tool_presentation":
            result = {"title": "Evidence deck", "slides": [
                {"title": "Document " + str(index), "bullets": ["Supported finding"], "notes": "",
                 "source_ids": document["source_ids"]}
                for index, document in enumerate(evidence["documents"], 1)]}
        elif name == "tool_summary":
            result = {"title": "Evidence summary", "sections": [{"heading": "Findings", "paragraphs": [
                {"text": point["text"], "source_ids": point["source_ids"]} for point in evidence["notes"]]}]}
        elif name == "tool_spreadsheet_row":
            absent = evidence["filename"] == "b.txt"
            result = {"cells": [{"column": "Observation", "value": "Not found" if absent else "Measured",
                                  "source_ids": [] if absent else evidence["notes"][0]["source_ids"]}]}
        else:
            raise AssertionError("Unexpected model request: " + name)
        jsonschema.Draft202012Validator(schema).validate(result)
        return result
    return generate


def _setup(monkeypatch, *, dense=False, invalid_consolidation=None):
    store = SnapshotStore(dense=dense)
    calls = []
    monkeypatch.setattr(service, "get_service", lambda workspace: SimpleNamespace(store=store))
    monkeypatch.setattr(service.api, "generate_structured", _fake_model(calls, invalid_consolidation=invalid_consolidation))
    return calls


@pytest.mark.parametrize("tool_id", ["summary", "presentation"])
def test_requests_send_explicit_aliases_and_required_citation_schema(local_model, monkeypatch, tool_id):
    calls = _setup(monkeypatch)
    original = deepcopy(service.PRESENTATION_SCHEMA if tool_id == "presentation" else service.SUMMARY_SCHEMA)
    result = list(service.generate_tool("workspace", tool_id, ["a", "b"], slide_count=2))[-1]["result"]
    assert result["coverage"]["processed_chunks"] == 4
    notes = [call for call in calls if call["name"] == "tool_evidence_notes"]
    assert len(notes) == 2
    for call in notes:
        ids = [item["source_id"] for item in call["prompt"]["evidence"]]
        assert call["prompt"]["allowed_source_ids"] == ids
        assert call["schema"]["$defs"]["source_identifier"]["enum"] == ids
    final = calls[-1]
    assert set(final["prompt"]["allowed_source_ids"]) == {"s1", "s2", "s3", "s4"}
    sample = ({"title": "Deck", "slides": [{"title": "Slide", "bullets": ["Finding"], "notes": "", "source_ids": []}]}
              if tool_id == "presentation" else {"title": "Summary", "sections": [{"heading": "Finding",
                  "paragraphs": [{"text": "Finding", "source_ids": []}]}]})
    assert not _valid(final["schema"], sample)
    assert (service.PRESENTATION_SCHEMA if tool_id == "presentation" else service.SUMMARY_SCHEMA) == original


def test_spreadsheet_request_aliases_are_scoped_to_each_document(local_model, monkeypatch):
    calls = _setup(monkeypatch)
    result = list(service.generate_tool("workspace", "spreadsheet", ["a", "b"], columns=["Observation"]))[-1]["result"]
    rows = [call for call in calls if call["name"] == "tool_spreadsheet_row"]
    assert [set(row["prompt"]["allowed_source_ids"]) for row in rows] == [{"s1", "s2"}, {"s3", "s4"}]
    assert result["content"]["rows"][1]["cells"][0]["source_ids"] == []
    assert not _valid(rows[1]["schema"], {"cells": [{"column": "Observation", "value": "Finding", "source_ids": ["s1"]}]})


def test_presentation_consolidation_scope_contains_only_current_batch(local_model, small_evidence_budget, monkeypatch):
    calls = _setup(monkeypatch, dense=True)
    result = list(service.generate_tool("workspace", "presentation", ["a", "b"], slide_count=2))[-1]["result"]
    assert result["coverage"]["processed_chunks"] == 12
    consolidations = [call for call in calls if call["name"] == "tool_consolidated_notes"]
    assert len(consolidations) >= 2
    all_ids = {f"s{index}" for index in range(1, 13)}
    for call in consolidations:
        expected = {identifier for point in call["prompt"]["evidence"] for identifier in point["source_ids"]}
        assert set(call["prompt"]["allowed_source_ids"]) == expected
        assert set(call["schema"]["$defs"]["source_identifier"]["enum"]) == expected
        outside = all_ids - expected
        if outside:
            assert not _valid(call["schema"], {"points": [{"text": "Wrong batch", "source_ids": [next(iter(outside))]}]})


@pytest.mark.parametrize("bad_ids", [[], ["unknown"], ["s1", "s1"]])
def test_invalid_presentation_consolidation_remains_strict_with_stage_diagnostic(
        local_model, small_evidence_budget, monkeypatch, bad_ids):
    calls = _setup(monkeypatch, dense=True, invalid_consolidation=bad_ids)
    with pytest.raises(ValueError) as error:
        list(service.generate_tool("workspace", "presentation", ["a", "b"], slide_count=2))
    message = str(error.value).lower()
    assert "presentation" in message and "consolidat" in message
    assert "missing, duplicated, or unknown source citations" in message
    assert len([call for call in calls if call["name"] == "tool_consolidated_notes"]) == 1
    assert not any(call["name"] == "tool_presentation" for call in calls)


@pytest.mark.parametrize("context,consolidates", [(131072, False), (8192, True)])
def test_presentation_uses_loaded_context_and_budgets_complete_requests(
        local_model, monkeypatch, context, consolidates):
    monkeypatch.setitem(config.MODEL_TOKEN_LIMITS, config.DEFAULT_MODEL, context)
    monkeypatch.setattr(service, "MAX_INPUT_TOKENS", 0)
    monkeypatch.setattr(evidence, "MAX_INPUT_TOKENS", 0)
    calls = _setup(monkeypatch, dense=True)
    result = list(service.generate_tool("workspace", "presentation", ["a", "b"],
        instructions="Preserve measurements and qualifications. " * 20, slide_count=2))[-1]["result"]

    assert result["coverage"]["processed_chunks"] == 12
    expected_sources = {source["id"] for source in SnapshotStore(dense=True).sources}
    assert {identifier for slide in result["content"]["slides"] for identifier in slide["source_ids"]} == expected_sources
    assert any(call["name"] == "tool_consolidated_notes" for call in calls) is consolidates
    if not consolidates:
        assert [call["name"] for call in calls] == ["tool_evidence_notes", "tool_evidence_notes", "tool_presentation"]
        # This evidence would have hit the old fixed 6,000-token ceiling.
        assert service.token_size(calls[-1]["prompt"]["evidence"]["notes"], config.DEFAULT_MODEL) > 6000
    all_ids = {f"s{index}" for index in range(1, 13)}
    assert set(calls[-1]["prompt"]["allowed_source_ids"]) == all_ids
    for call in calls:
        input_tokens = service.api.structured_request_tokens(
            call["model"], call["history"], call["schema"], call["options"]["system_prompt"])
        output_tokens = call["options"]["max_output_tokens"]
        assert output_tokens == min(32768, context // 4)
        assert input_tokens + output_tokens <= context
