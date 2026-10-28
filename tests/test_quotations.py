"""Source-anchored Evidence Check catalogs, schemas, and model selection parsing."""

from copy import deepcopy
import json

import jsonschema
import pytest

from rag_chatbot.tools import quotations, review


CLAIMS = [{"id": "c1", "text": "The measurement increased."},
          {"id": "c2", "text": "The result was independently replicated."}]
ALIASES = {"s1": {"text": "Measured 10 nm.\n  The first trial increased."},
           "s2": {"text": "The second trial did not increase.\tNo replication was reported."}}


def _assessment():
    return {"title": "Evidence review", "claims": [
        {"id": "c1", "verdict": "mixed", "rationale": "The trials differ.", "evidence": [
            {"quote_id": "q1", "stance": "supports"},
            {"quote_id": "q2", "stance": "contradicts"}]},
        {"id": "c2", "verdict": "not_found", "rationale": "No selected support.", "evidence": []}]}


def _selection():
    return {"claims": [{"id": "c1", "evidence": [
        {"quote_id": "q2", "stance": "contradicts"},
        {"quote_id": "q1", "stance": "supports"}]},
        {"id": "c2", "evidence": [{"quote_id": "q1", "stance": "context"}]}]}


def test_catalog_preserves_exact_whitespace_and_source_order_without_mutation():
    original = deepcopy(ALIASES)
    catalog = quotations.build_quote_catalog(ALIASES)
    assert catalog == [{"quote_id": "q1", "source_id": "s1", "quote": ALIASES["s1"]["text"]},
                       {"quote_id": "q2", "source_id": "s2", "quote": ALIASES["s2"]["text"]}]
    assert ALIASES == original


@pytest.mark.parametrize('length', [1200, 1201, 2200, 4807])
def test_long_catalog_has_complete_exact_overlapping_unicode_coverage(length):
    text = ("\u0394\u7cbe\U0001f642 abc\n" * length)[:length]
    catalog = quotations.build_quote_catalog({"s1": {"text": text}})
    overlap = quotations.QUOTE_OVERLAP_CHARACTERS
    step = review.MAX_QUOTE_CHARACTERS - overlap
    for index, row in enumerate(catalog):
        assert row["quote"] == text[index * step:index * step + review.MAX_QUOTE_CHARACTERS]
        assert len(row["quote"]) <= review.MAX_QUOTE_CHARACTERS
        assert row["quote"].encode("utf-8").decode("utf-8") == row["quote"]
    reconstructed = catalog[0]["quote"] + "".join(row["quote"][overlap:] for row in catalog[1:])
    assert reconstructed == text


def test_schemas_bound_ids_claim_count_and_quotes_and_leave_inputs_unchanged():
    catalog = quotations.build_quote_catalog(ALIASES)
    original = deepcopy((catalog, CLAIMS))
    schema = quotations.assessment_schema(catalog, CLAIMS)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(_assessment(), schema)
    assert schema["$defs"]["quote_identifier"]["enum"] == ["q1", "q2"]
    assert schema["properties"]["claims"]["minItems"] == 2
    assert schema["properties"]["claims"]["maxItems"] == 2
    jsonschema.validate(_selection(), quotations.selection_schema(catalog, CLAIMS))
    assert (catalog, CLAIMS) == original


@pytest.mark.parametrize("damage", ["unknown_quote", "unknown_claim", "missing_claim", "extra_claim",
                                    "model_quote", "long_rationale", "unknown_stance"])
def test_schema_rejects_invalid_wire_contract(damage):
    value = _assessment()
    row = value["claims"][0]
    if damage == "unknown_quote":
        row["evidence"][0]["quote_id"] = "invented"
    elif damage == "unknown_claim":
        row["id"] = "c99"
    elif damage == "missing_claim":
        value["claims"].pop()
    elif damage == "extra_claim":
        value["claims"].append(deepcopy(row))
    elif damage == "model_quote":
        row["evidence"][0]["quote"] = "Invented text."
    elif damage == "long_rationale":
        row["rationale"] = "x" * 4001
    else:
        row["evidence"][0]["stance"] = "neutral"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, quotations.assessment_schema(quotations.build_quote_catalog(ALIASES), CLAIMS))


def test_resolution_copies_originals_and_retains_existing_domain_validation():
    catalog = quotations.build_quote_catalog(ALIASES)
    value = _assessment()
    value["claims"].reverse()
    original = deepcopy(value)
    resolved = quotations.resolve_assessment(value, catalog, CLAIMS)
    assert [row["id"] for row in resolved["claims"]] == ["c1", "c2"]
    for item, alias in zip(resolved["claims"][0]["evidence"], ALIASES):
        assert item["quote"] == ALIASES[alias]["text"]
        assert item["source_id"] == alias
    assert all("text" not in row for row in resolved["claims"])
    content = {"claims": [{**row, "text": claim["text"]}
                          for row, claim in zip(resolved["claims"], CLAIMS)]}
    review.validate_evidence_check(content, claims=CLAIMS,
                                  sources=[{"id": key, **source} for key, source in ALIASES.items()])
    assert value == original


@pytest.mark.parametrize("damage", ["unknown_quote", "duplicate_quote", "duplicate_claim", "missing_claim",
                                    "unknown_claim", "unknown_verdict", "stance_mismatch", "invalid_stance",
                                    "model_quote", "empty_rationale", "invalid_title", "too_many_quotes"])
def test_resolution_rejects_invalid_output_without_schema_gate(damage):
    value = _assessment()
    row = value["claims"][0]
    if damage == "unknown_quote":
        row["evidence"][0]["quote_id"] = "q99"
    elif damage == "duplicate_quote":
        row["evidence"][1]["quote_id"] = "q1"
    elif damage == "duplicate_claim":
        value["claims"][1]["id"] = "c1"
    elif damage == "missing_claim":
        value["claims"].pop()
    elif damage == "unknown_claim":
        row["id"] = "other"
    elif damage == "unknown_verdict":
        row["verdict"] = "true"
    elif damage == "stance_mismatch":
        row["verdict"] = "supported"
    elif damage == "invalid_stance":
        row["evidence"][0]["stance"] = []
    elif damage == "model_quote":
        row["evidence"][0]["quote"] = "Not in the source."
    elif damage == "empty_rationale":
        row["rationale"] = " "
    elif damage == "invalid_title":
        value["title"] = "bad\x00title"
    else:
        row["evidence"] = [{"quote_id": "q1", "stance": "supports"}] * 9
    with pytest.raises(ValueError):
        quotations.resolve_assessment(value, quotations.build_quote_catalog(ALIASES), CLAIMS)


def test_selection_keeps_all_candidates_in_source_order_and_permits_cross_claim_reuse():
    catalog = quotations.build_quote_catalog(ALIASES)
    value = _selection()
    selected = quotations.selected_quotes(value, catalog, CLAIMS)
    assert selected == catalog
    assert selected[0] is not catalog[0]
    assert value == _selection()


def test_selection_rejects_same_quote_with_conflicting_stances_in_one_claim():
    value = _selection()
    value["claims"][0]["evidence"][0]["quote_id"] = "q1"
    with pytest.raises(ValueError, match="repeat"):
        quotations.selected_quotes(value, quotations.build_quote_catalog(ALIASES), CLAIMS)


def test_identical_excerpt_aliases_cannot_bypass_duplicate_quote_check():
    catalog = [{"quote_id": "q1", "source_id": "s1", "quote": "One  passage."},
               {"quote_id": "q2", "source_id": "s1", "quote": "One\npassage."}]
    with pytest.raises(ValueError, match="same cited excerpt"):
        quotations.resolve_assessment(_assessment(), catalog, CLAIMS)


def test_empty_catalog_allows_only_no_evidence_and_not_found_assessments():
    value = _assessment()
    value["claims"][0].update(verdict="not_found", evidence=[])
    schema = quotations.assessment_schema([], CLAIMS)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(value, schema)
    assert quotations.resolve_assessment(value, [], CLAIMS) == value
    selection = {"claims": [{"id": claim["id"], "evidence": []} for claim in CLAIMS]}
    jsonschema.validate(selection, quotations.selection_schema([], CLAIMS))
    assert quotations.selected_quotes(selection, [], CLAIMS) == []


@pytest.mark.parametrize("verdict,stances", [
    ("supported", ["supports"]),
    ("supported", ["supports", "context"]),
    ("contradicted", ["contradicts"]),
    ("contradicted", ["contradicts", "context"]),
    ("mixed", ["supports", "contradicts"]),
    ("not_found", []),
    ("not_found", ["context"]),
])
def test_assessment_schema_enforces_consistent_verdict_and_decisive_prefix(verdict, stances):
    catalog = quotations.build_quote_catalog(ALIASES)
    value = _assessment()
    value["claims"][0].update(verdict=verdict, evidence=[
        {"quote_id": f"q{index}", "stance": stance} for index, stance in enumerate(stances, 1)])
    jsonschema.validate(value, quotations.assessment_schema(catalog, CLAIMS))
    quotations.resolve_assessment(value, catalog, CLAIMS)


@pytest.mark.parametrize("verdict,stances", [
    ("supported", []),
    ("supported", ["context"]),
    ("supported", ["contradicts"]),
    ("supported", ["supports", "contradicts"]),
    ("contradicted", []),
    ("contradicted", ["context"]),
    ("contradicted", ["supports"]),
    ("mixed", []),
    ("mixed", ["supports"]),
    ("mixed", ["supports", "context"]),
    ("mixed", ["contradicts", "context"]),
    ("not_found", ["supports"]),
    ("not_found", ["contradicts"]),
])
def test_schema_rejects_verdict_stance_mismatch_before_resolution(verdict, stances):
    value = _assessment()
    value["claims"][0].update(verdict=verdict, evidence=[
        {"quote_id": f"q{index}", "stance": stance} for index, stance in enumerate(stances, 1)])
    schema = quotations.assessment_schema(quotations.build_quote_catalog(ALIASES), CLAIMS)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def test_schema_constrains_order_without_rejecting_legacy_valid_order_in_resolver():
    catalog = quotations.build_quote_catalog(ALIASES)
    value = _assessment()
    value["claims"][0]["evidence"].reverse()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, quotations.assessment_schema(catalog, CLAIMS))
    quotations.resolve_assessment(value, catalog, CLAIMS)


@pytest.mark.parametrize("assessment", [False, True])
def test_schema_requires_each_exact_claim_once_in_canonical_order(assessment):
    catalog = quotations.build_quote_catalog(ALIASES)
    value = _assessment() if assessment else _selection()
    schema = (quotations.assessment_schema if assessment else quotations.selection_schema)(catalog, CLAIMS)
    for changed in [list(reversed(value["claims"])), [value["claims"][0], value["claims"][0]]]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**value, "claims": changed}, schema)


def test_shared_stance_definitions_keep_many_claims_and_empty_catalog_bounded():
    claims = [{"id": f"c{index}", "text": f"Claim {index}."} for index in range(1, 21)]
    catalog = quotations.build_quote_catalog({f"s{index}": {"text": f"Passage {index}."}
                                             for index in range(1, 84)})
    schema = quotations.assessment_schema(catalog, claims)
    # Reuse the expensive stance/quote alternatives across all twenty claim slots.
    encoded = json.dumps(schema, separators=(",", ":"))
    assert encoded.count('"enum":["q1","q2"') == 1
    assert len(encoded) < 40000
    empty = quotations.assessment_schema([], claims)
    assert len(json.dumps(empty, separators=(",", ":"))) < 7000
    value = {"title": "No evidence", "claims": [
        {"id": claim["id"], "verdict": "not_found", "rationale": "No evidence selected.", "evidence": []}
        for claim in claims]}
    jsonschema.validate(value, empty)
    value["claims"][0]["verdict"] = "supported"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, empty)


@pytest.mark.parametrize('field,control', [('title', '\x00'), ('rationale', '\x0c'), ('title', '\x1f'), ('rationale', '\x7f')])
def test_resolver_rejects_disallowed_controls_in_model_text(field, control):
    value = _assessment()
    target = value if field == "title" else value["claims"][0]
    target[field] = "Before" + control + "after"
    catalog = quotations.build_quote_catalog(ALIASES)
    # JSON permits escaped controls, so domain validation remains authoritative.
    jsonschema.validate(value, quotations.assessment_schema(catalog, CLAIMS))
    with pytest.raises(ValueError, match="nonempty bounded text"):
        quotations.resolve_assessment(value, catalog, CLAIMS)


def test_generated_text_uses_normal_json_strings_with_bounds_and_literal_latex():
    catalog = quotations.build_quote_catalog(ALIASES)
    value = _assessment()
    value["title"] = r"Ratio \frac{a}{b}"
    value["claims"][0]["rationale"] = "First line.\nSecond\tline.\r" + r"\frac{a}{b}"
    schema = quotations.assessment_schema(catalog, CLAIMS)
    assert schema["properties"]["title"] == {"type": "string", "minLength": 1, "maxLength": 200}
    assert schema["$defs"]["rationale"] == {"type": "string", "minLength": 1, "maxLength": 4000}
    jsonschema.validate(value, schema)
    resolved = quotations.resolve_assessment(value, catalog, CLAIMS)
    assert resolved["claims"][0]["evidence"][0]["quote"] == ALIASES["s1"]["text"]


@pytest.mark.parametrize('aliases', [{}, {'s1': {'text': ' '}}, {'': {'text': 'Text'}}])
def test_catalog_rejects_invalid_sources(aliases):
    with pytest.raises(ValueError):
        quotations.build_quote_catalog(aliases)


@pytest.mark.parametrize("damage", ["duplicate_id", "extra_field", "oversized_quote", "empty_source"])
def test_schema_and_resolution_reject_malformed_app_catalog(damage):
    catalog = quotations.build_quote_catalog(ALIASES)
    if damage == "duplicate_id":
        catalog[1]["quote_id"] = "q1"
    elif damage == "extra_field":
        catalog[0]["extra"] = True
    elif damage == "oversized_quote":
        catalog[0]["quote"] = "x" * 1201
    else:
        catalog[0]["source_id"] = ""
    with pytest.raises(ValueError):
        quotations.assessment_schema(catalog, CLAIMS)
    with pytest.raises(ValueError):
        quotations.resolve_assessment(_assessment(), catalog, CLAIMS)
