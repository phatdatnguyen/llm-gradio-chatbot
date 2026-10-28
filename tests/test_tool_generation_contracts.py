"""Request schemas constrain generation without weakening artifact validation."""

from copy import deepcopy
import json
from types import SimpleNamespace

import jsonschema
import pytest

from chatbot import config
from rag_chatbot.tools import contracts, learning, presentation, review, service
from rag_chatbot.tools.citations import scope_citation_schema


DOCUMENTS = [{"id": "doc-a", "name": "First"}, {"id": "doc-b", "name": "Second"}]
SOURCES = [{"id": "s1", "document_id": "doc-a"}, {"id": "s2", "document_id": "doc-b"}]
SCOPES = {"doc-a": ["s1"], "doc-b": ["s2"]}


def study_set(tool_id, count):
    if tool_id == "flashcards":
        return {"title": "Study", "cards": [
            {"front": f"Question {index}", "back": "Supported answer", "source_ids": ["s1"]}
            for index in range(count)]}
    return {"title": "Study", "questions": [
        {"question": f"Question {index}", "options": ["A", "B", "C", "D"],
         "correct_option": 0, "explanation": "Supported answer", "source_ids": ["s1"]}
        for index in range(count)]}


def study_schema(tool_id, count):
    base = service.FLASHCARDS_SCHEMA if tool_id == "flashcards" else service.QUIZZES_SCHEMA
    return contracts.learning_contract(scope_citation_schema(base, ["s1"]), tool_id, count)


@pytest.mark.parametrize("tool_id,count", [
    ("flashcards", 1), ("flashcards", learning.DEFAULT_LEARNING_COUNT), ("flashcards", learning.MAX_FLASHCARDS),
    ("quizzes", 1), ("quizzes", learning.DEFAULT_LEARNING_COUNT), ("quizzes", learning.MAX_QUIZ_QUESTIONS),
])
def test_learning_generation_requires_exact_count_at_minimum_default_and_maximum(tool_id, count):
    validator = jsonschema.Draft202012Validator(study_schema(tool_id, count))
    validator.validate(study_set(tool_id, count))
    for wrong_count in (count - 1, count + 1):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(study_set(tool_id, wrong_count))


@pytest.mark.parametrize("tool_id,key", [("flashcards", "cards"), ("quizzes", "questions")])
def test_learning_generation_preserves_legacy_sets_and_citation_scope(tool_id, key):
    base = service.FLASHCARDS_SCHEMA if tool_id == "flashcards" else service.QUIZZES_SCHEMA
    original = deepcopy(base)
    schema = study_schema(tool_id, learning.DEFAULT_LEARNING_COUNT)
    jsonschema.Draft202012Validator.check_schema(schema)
    # Stored sets retain their original count; only new requests require the slider's count.
    saved = study_set(tool_id, 1)
    learning.validate_learning_content(tool_id, {key: saved[key]}, allowed_source_ids={"s1"})
    generated = study_set(tool_id, learning.DEFAULT_LEARNING_COUNT)
    generated[key][0]["source_ids"] = ["unselected"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(generated, schema)
    assert base == original


@pytest.mark.parametrize("tool_id,field,maximum", [
    ("flashcards", "front", learning.MAX_PROMPT_CHARACTERS),
    ("flashcards", "back", learning.MAX_ANSWER_CHARACTERS),
    ("quizzes", "question", learning.MAX_PROMPT_CHARACTERS),
    ("quizzes", "explanation", learning.MAX_ANSWER_CHARACTERS),
])
def test_learning_generation_text_bounds_match_validators(tool_id, field, maximum):
    key = "cards" if tool_id == "flashcards" else "questions"
    value = study_set(tool_id, 1)
    validator = jsonschema.Draft202012Validator(study_schema(tool_id, 1))
    value[key][0][field] = "x" * maximum
    validator.validate(value)
    learning.validate_learning_content(tool_id, {key: value[key]})
    for invalid in ("", "x" * (maximum + 1)):
        value[key][0][field] = invalid
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(value)


def test_quiz_generation_constrains_options_and_answer_index():
    validator = jsonschema.Draft202012Validator(study_schema("quizzes", 1))
    value = study_set("quizzes", 1)
    question = value["questions"][0]
    question["options"][0] = "x" * learning.MAX_OPTION_CHARACTERS
    question["correct_option"] = 3
    validator.validate(value)
    for field, invalid in (
        ("options", ["A", "B", "C"]), ("options", ["A", "B", "C", "D", "E"]),
        ("options", ["A", "B", "C", "A"]), ("options", ["A", "B", "C", ""]),
        ("options", ["A", "B", "C", "x" * (learning.MAX_OPTION_CHARACTERS + 1)]),
        ("correct_option", -1), ("correct_option", 4), ("correct_option", True),
    ):
        broken = study_set("quizzes", 1)
        broken["questions"][0][field] = invalid
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(broken)


@pytest.mark.parametrize("tool_id,key,count", [("flashcards", "cards", 3), ("quizzes", "questions", 2)])
def test_learning_pipeline_budgets_and_sends_the_requested_count(monkeypatch, tool_id, key, count):
    model = "learning-contract-model"
    monkeypatch.setattr(config, "DEFAULT_MODEL", model)
    monkeypatch.setattr(config, "MODEL_CHOICES", [model])
    monkeypatch.setitem(config.MODEL_TOKEN_LIMITS, model, 65536)
    snapshot = {
        "documents": [{"id": "doc-a", "name": "Source.txt", "content_hash": "hash",
                       "fingerprint": "local-test", "generation": "ready"}],
        "sources": [{"id": "chunk-a", "document_id": "doc-a", "filename": "Source.txt",
                     "locator": "Section 1", "text": "Supported answer.",
                     "source_path": "sources/Source.txt", "kind": "text"}],
    }
    store = SimpleNamespace(snapshot_sources=lambda *args, **kwargs: deepcopy(snapshot),
                            validate_source_snapshot=lambda documents: None)
    monkeypatch.setattr(service, "get_service", lambda workspace: SimpleNamespace(store=store))
    requests, budgeted = [], []
    original_estimator = service.api.structured_request_tokens

    def estimate(model, history, schema, system_prompt=""):
        if key in schema["properties"]:
            assert schema["properties"][key]["minItems"] == schema["properties"][key]["maxItems"] == count
            budgeted.append(deepcopy(schema))
        return original_estimator(model, history, schema, system_prompt)

    def generate(model, history, schema, **options):
        prompt = json.loads(history[-1]["content"])
        requests.append(options["schema_name"])
        if options["schema_name"] == "tool_evidence_notes":
            value = {"notes": [{"source_id": row["source_id"], "text": row["text"]}
                               for row in prompt["evidence"]]}
        else:
            assert options["schema_name"] == f"tool_{tool_id}"
            assert prompt["requested_count"] == count
            assert budgeted and schema == budgeted[-1]
            value = study_set(tool_id, count)
        jsonschema.validate(value, schema)
        return value

    monkeypatch.setattr(service.api, "structured_request_tokens", estimate)
    monkeypatch.setattr(service.api, "generate_structured", generate)
    result = list(service.generate_tool("workspace", tool_id, ["doc-a"], item_count=count))[-1]["result"]
    assert len(result["content"][key]) == result["settings"]["item_count"] == count
    assert result["content"][key][0]["source_ids"] == ["chunk-a"]
    assert requests == ["tool_evidence_notes", f"tool_{tool_id}"]


def presentation_schema(count=1):
    scoped = scope_citation_schema(service.PRESENTATION_SCHEMA, ["s1", "s2"])
    return contracts.presentation_contract(scoped, count)


def compare_schema():
    scoped = scope_citation_schema(service.COMPARE_SCHEMA, ["s1", "s2"],
        allow_empty_paths=(("aspects", "items", "findings", "items", "source_ids"),))
    return contracts.compare_contract(scoped, DOCUMENTS, SCOPES)


def deck(*, count=1, bullets=None):
    return {"title": "Deck", "slides": [
        {"title": f"Slide {number}", "bullets": list(bullets or ["A fact"]),
         "notes": "", "source_ids": ["s1"]}
        for number in range(count)
    ]}


def comparison():
    return {"title": "Comparison", "aspects": [{
        "aspect": "Methods", "findings": [
            {"document_id": "doc-a", "status": "found", "text": "First method", "source_ids": ["s1"]},
            {"document_id": "doc-b", "status": "found", "text": "Second method", "source_ids": ["s2"]},
        ], "synthesis": {"text": "The methods differ.", "finding_document_ids": ["doc-a", "doc-b"]},
    }]}


@pytest.mark.parametrize("count", [3, 4, 5])
def test_presentation_bullet_branches_guarantee_total_without_truncation(count):
    maximum = min(presentation.MAX_BULLET_CHARACTERS, presentation.MAX_BULLET_TOTAL_CHARACTERS // count)
    value = deck(bullets=["x" * maximum] * count)
    original = deepcopy(value)
    jsonschema.validate(value, presentation_schema())
    presentation.validate_presentation({"slides": value["slides"]}, allowed_source_ids={"s1"})
    assert value == original
    value["slides"][0]["bullets"][0] += "x"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, presentation_schema())


@pytest.mark.parametrize("field,value", [
    ("title", "x" * (presentation.MAX_TITLE_CHARACTERS + 1)),
    ("title", ""),
    ("notes", "x" * (presentation.MAX_NOTES_CHARACTERS + 1)),
    ("bullets", []),
    ("bullets", ["x"] * (presentation.MAX_BULLETS + 1)),
    ("bullets", [""]),
    ("source_ids", ["unselected-source"]),
])
def test_presentation_generation_rejects_oversize_or_invalid_slide_fields(field, value):
    content = deck()
    content["slides"][0][field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(content, presentation_schema())


@pytest.mark.parametrize("actual", [1, 3])
def test_presentation_generation_requires_exact_requested_slide_count(actual):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(deck(count=actual), presentation_schema(2))
    jsonschema.validate(deck(count=2), presentation_schema(2))


def test_existing_decks_with_unequal_bullet_lengths_keep_original_validation():
    value = deck(bullets=["x" * 180, "x" * 180, "x" * 180, "x" * 30, "x" * 30])
    original = deepcopy(value)
    assert presentation.validate_presentation({"slides": value["slides"]}) == {"slides": value["slides"]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, presentation_schema())
    assert value == original


@pytest.mark.parametrize("count", [0, review.MAX_COMPARE_ASPECTS + 1])
def test_compare_contract_rejects_invalid_aspect_count(count):
    value = comparison()
    value["aspects"] = value["aspects"] * count
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, compare_schema())


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_id", "unknown_id", "swapped", "cross_citation"])
def test_compare_contract_requires_exact_canonical_findings_and_own_sources(mutation):
    value = comparison()
    findings = value["aspects"][0]["findings"]
    if mutation == "missing":
        findings.pop()
    elif mutation == "extra":
        findings.append(deepcopy(findings[0]))
    elif mutation == "duplicate_id":
        findings[1]["document_id"] = "doc-a"
    elif mutation == "unknown_id":
        findings[0]["document_id"] = "other-document"
    elif mutation == "swapped":
        findings.reverse()
    else:
        findings[0]["source_ids"] = ["s2"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, compare_schema())


def test_compare_found_and_not_found_citation_branches_remain_strict():
    value = comparison()
    second = value["aspects"][0]["findings"][1]
    second.update(status="not_found", source_ids=[])
    jsonschema.validate(value, compare_schema())
    resolved = contracts.resolve_compare_synthesis(value, DOCUMENTS)
    assert resolved["aspects"][0]["synthesis"]["source_ids"] == ["s1"]
    review.validate_compare({"aspects": resolved["aspects"]}, documents=DOCUMENTS, sources=SOURCES)
    second["source_ids"] = ["s2"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, compare_schema())
    second.update(status="found", source_ids=[])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, compare_schema())


@pytest.mark.parametrize("field", ["aspect", "finding", "synthesis"])
def test_compare_contract_uses_shared_text_limits(field):
    value = comparison()
    aspect = value["aspects"][0]
    if field == "aspect":
        aspect["aspect"] = "x" * (review.MAX_COMPARE_ASPECT_CHARACTERS + 1)
    elif field == "finding":
        aspect["findings"][0]["text"] = "x" * (review.MAX_COMPARE_FINDING_CHARACTERS + 1)
    else:
        aspect["synthesis"]["text"] = "x" * (review.MAX_COMPARE_SYNTHESIS_CHARACTERS + 1)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, compare_schema())


def test_compare_semantic_validation_still_checks_synthesis_against_found_sources():
    value = contracts.resolve_compare_synthesis(comparison(), DOCUMENTS)
    value["aspects"][0]["findings"][1].update(status="not_found", source_ids=[])
    with pytest.raises(ValueError, match="outside this request"):
        review.validate_compare({"aspects": value["aspects"]}, documents=DOCUMENTS, sources=SOURCES)


def test_saved_compare_validator_still_accepts_and_normalizes_findings_order():
    value = contracts.resolve_compare_synthesis(comparison(), DOCUMENTS)
    value["aspects"][0]["findings"].reverse()
    validated = review.validate_compare({"aspects": value["aspects"]}, documents=DOCUMENTS, sources=SOURCES)
    assert [row["document_id"] for row in validated["aspects"][0]["findings"]] == ["doc-a", "doc-b"]


@pytest.mark.parametrize("scope", [
    {"doc-a": ["s1"]},
    {"doc-a": ["s1"], "doc-b": []},
    {"doc-a": ["s1"], "doc-b": ["s1"]},
    {"doc-a": ["s1", "s1"], "doc-b": ["s2"]},
    {"doc-a": ["s1"], "doc-b": [""]},
])
def test_compare_rejects_incomplete_or_ambiguous_document_scopes(scope):
    with pytest.raises(ValueError, match="source scope"):
        contracts.compare_contract(service.COMPARE_SCHEMA, DOCUMENTS, scope)


def test_contracts_are_json_serializable_independent_copies_with_no_source_text():
    original_presentation = deepcopy(service.PRESENTATION_SCHEMA)
    original_compare = deepcopy(service.COMPARE_SCHEMA)
    documents = deepcopy(DOCUMENTS)
    documents[0]["name"] = "private source title never belongs in the schema"
    schemas = [presentation_schema(2), contracts.compare_contract(
        scope_citation_schema(service.COMPARE_SCHEMA, ["s1", "s2"]), documents, SCOPES)]
    for schema in schemas:
        encoded = json.dumps(schema, ensure_ascii=False)
        jsonschema.Draft202012Validator.check_schema(json.loads(encoded))
        assert documents[0]["name"] not in encoded
    assert service.PRESENTATION_SCHEMA == original_presentation
    assert service.COMPARE_SCHEMA == original_compare


def test_compare_synthesis_resolves_only_explicit_findings_without_rewriting_input():
    value = comparison()
    value["aspects"][0]["synthesis"] = {
        "text": "The first method has this limitation.\nKeep this exact wording.",
        "finding_document_ids": ["doc-a"],
    }
    original = deepcopy(value)
    jsonschema.validate(value, compare_schema())
    resolved = contracts.resolve_compare_synthesis(value, DOCUMENTS)
    assert value == original
    assert resolved["aspects"][0]["findings"] == original["aspects"][0]["findings"]
    assert resolved["aspects"][0]["synthesis"] == {
        "text": original["aspects"][0]["synthesis"]["text"], "source_ids": ["s1"],
    }
    review.validate_compare({"aspects": resolved["aspects"]}, documents=DOCUMENTS, sources=SOURCES)


def test_compare_synthesis_preserves_explicit_finding_selection_order():
    value = comparison()
    value["aspects"][0]["synthesis"]["finding_document_ids"] = ["doc-b", "doc-a"]
    resolved = contracts.resolve_compare_synthesis(value, DOCUMENTS)
    assert resolved["aspects"][0]["synthesis"]["source_ids"] == ["s2", "s1"]


@pytest.mark.parametrize("references", [[], ["unknown"], ["s1"], ["doc-a", "doc-a"], [True], "doc-a"])
def test_compare_synthesis_rejects_invalid_or_ambiguous_finding_references(references):
    value = comparison()
    value["aspects"][0]["synthesis"]["finding_document_ids"] = references
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, compare_schema())
    with pytest.raises(ValueError, match="unique, known finding"):
        contracts.resolve_compare_synthesis(value, DOCUMENTS)


def test_compare_synthesis_does_not_invent_citations_for_only_not_found_selection():
    value = comparison()
    value["aspects"][0]["findings"][1].update(status="not_found", source_ids=[])
    value["aspects"][0]["synthesis"]["finding_document_ids"] = ["doc-b"]
    jsonschema.validate(value, compare_schema())
    with pytest.raises(ValueError, match="at least one found finding"):
        contracts.resolve_compare_synthesis(value, DOCUMENTS)


def test_compare_generation_has_no_canonical_wire_fallback():
    value = comparison()
    value["aspects"][0]["synthesis"] = {"text": "The methods differ.", "source_ids": ["s1", "s2"]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, compare_schema())
    with pytest.raises(ValueError, match="invalid output structure"):
        contracts.resolve_compare_synthesis(value, DOCUMENTS)
    # Previously saved canonical results still use the unchanged reader contract.
    review.validate_compare({"aspects": value["aspects"]}, documents=DOCUMENTS, sources=SOURCES)


@pytest.mark.parametrize("mutation", ["duplicate_document", "missing_document", "unknown_document",
                                     "duplicate_citation", "not_found_with_citation", "found_without_citation"])
def test_compare_resolver_never_repairs_invalid_findings(mutation):
    value = comparison()
    findings = value["aspects"][0]["findings"]
    if mutation == "duplicate_document":
        findings[1]["document_id"] = "doc-a"
    elif mutation == "missing_document":
        findings.pop()
    elif mutation == "unknown_document":
        findings[0]["document_id"] = "unknown"
    elif mutation == "duplicate_citation":
        findings[0]["source_ids"] = ["s1", "s1"]
    elif mutation == "not_found_with_citation":
        findings[0]["status"] = "not_found"
    else:
        findings[0]["source_ids"] = []
    original = deepcopy(value)
    with pytest.raises(ValueError):
        contracts.resolve_compare_synthesis(value, DOCUMENTS)
    assert value == original


def test_compare_resolved_findings_still_require_document_ownership_validation():
    value = comparison()
    value["aspects"][0]["findings"][0]["source_ids"] = ["s2"]
    resolved = contracts.resolve_compare_synthesis(value, DOCUMENTS)
    with pytest.raises(ValueError, match="only its own document"):
        review.validate_compare({"aspects": resolved["aspects"]}, documents=DOCUMENTS, sources=SOURCES)


@pytest.mark.parametrize("character", ["\x00", "\b", "\f", "\x7f"])
def test_compare_domain_validator_rejects_json_decoded_control_characters(character):
    value = comparison()
    value["aspects"][0]["findings"][0]["text"] = "Before" + character + "after"
    # JSON decoding converts accidental LaTeX escapes such as \f into controls.
    wire_value = json.loads(json.dumps(value))
    jsonschema.validate(wire_value, compare_schema())
    resolved = contracts.resolve_compare_synthesis(wire_value, DOCUMENTS)
    with pytest.raises(ValueError, match="control characters"):
        review.validate_compare({"aspects": resolved["aspects"]}, documents=DOCUMENTS, sources=SOURCES)


@pytest.mark.parametrize("text", [
    "Tiếng Việt — 漢字 😀", r"\(\frac{a}{b}\) and \[x^2\]", "First\nSecond\r\nThird\tcolumn",
])
def test_generated_text_contract_accepts_unicode_math_and_allowed_whitespace(text):
    value = comparison()
    value["aspects"][0]["findings"][0]["text"] = text
    wire_value = json.loads(json.dumps(value, ensure_ascii=False))
    jsonschema.validate(wire_value, compare_schema())
    resolved = contracts.resolve_compare_synthesis(wire_value, DOCUMENTS)
    review.validate_compare({"aspects": resolved["aspects"]}, documents=DOCUMENTS, sources=SOURCES)
    assert resolved["aspects"][0]["findings"][0]["text"] == text


