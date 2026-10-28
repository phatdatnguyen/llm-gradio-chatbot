"""Schema-specific grammar acceptance, source scoping, and bounded compilation."""
from copy import deepcopy
import json
import re

import jsonschema
import pytest

from chatbot.providers import schema_grammar
from rag_chatbot.tools import citations, contracts, quotations, service, tabular_plot


def _accepts(grammar, value):
    """Read the finite GBNF subset as regex, independently of its compiler."""
    rules = dict(line.split(" ::= ", 1) for line in grammar.splitlines())
    token = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\\\]])*\]|[A-Za-z0-9-]+|\{\d+(?:,\d*)?\}|[()|*+?]')
    expanded = {}
    def expand(name):
        if name in expanded:
            return expanded[name]
        result, end = [], 0
        for match in token.finditer(rules[name]):
            assert not rules[name][end:match.start()].strip()
            text = match.group()
            if text.startswith('"'):
                result.append(re.escape(json.loads(text)))
            elif re.fullmatch(r"[A-Za-z0-9-]+", text):
                result.append("(?:" + expand(text) + ")")
            else:
                result.append(text)
            end = match.end()
        assert not rules[name][end:].strip()
        expanded[name] = "".join(result)
        return expanded[name]
    # The upstream literal renderer spells non-ASCII enum values as JSON \u
    # escapes. Both spellings decode to the same source identifier.
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return re.fullmatch(expand("root"), text) is not None


def test_all_current_tool_contracts_compile_without_mutating_schemas():
    sources = ["s1", "s2"]
    schemas = [citations.scope_citation_schema(value, sources)
               for name, value in vars(service).items() if name.endswith("_SCHEMA")]
    schemas.extend([
        contracts.learning_contract(citations.scope_citation_schema(service.FLASHCARDS_SCHEMA, sources), "flashcards", 60),
        contracts.learning_contract(citations.scope_citation_schema(service.QUIZZES_SCHEMA, sources), "quizzes", 40),
        contracts.presentation_contract(citations.scope_citation_schema(service.PRESENTATION_SCHEMA, sources), 20),
        contracts.compare_contract(citations.scope_citation_schema(service.COMPARE_SCHEMA, sources),
            [{"id": "a", "name": "First"}, {"id": "b", "name": "Second"}], {"a": ["s1"], "b": ["s2"]}),
        tabular_plot.MAPPING_SCHEMA,
    ])
    claims = [{"id": "c1", "text": "A claim"}, {"id": "c2", "text": "Another claim"}]
    catalog = quotations.build_quote_catalog({"s1": {"text": "First quote."}, "s2": {"text": "Second quote."}})
    schemas.extend([quotations.assessment_schema(catalog, claims), quotations.selection_schema(catalog, claims),
                    quotations.assessment_schema([], claims)])
    originals = deepcopy(schemas)
    for schema in schemas:
        grammar = schema_grammar.schema_to_grammar(schema)
        assert "root ::=" in grammar and len(grammar.encode()) <= schema_grammar.MAX_GRAMMAR_BYTES
    assert schemas == originals


def test_grammar_restricts_citation_enum_and_preserves_escaped_literals():
    identifiers = ['s1', 'a"b\\c', "Unicode-Δ"]
    schema = citations.scope_citation_schema(service.POINTS_SCHEMA, identifiers)
    grammar = schema_grammar.schema_to_grammar(schema)
    valid = {"points": [{"text": "Evidence", "source_ids": identifiers}]}
    assert _accepts(grammar, valid)
    for identifiers in ([], ["unknown"]):
        invalid = {"points": [{"text": "Evidence", "source_ids": identifiers}]}
        assert not _accepts(grammar, invalid)
    assert not _accepts(grammar, {"points": [{"text": "Evidence", "source_ids": ["s1"], "extra": "x"}]})
    assert not _accepts(grammar, {"points": [{"text": "Evidence"}]})


def test_quiz_and_presentation_counts_text_bounds_and_integer_range_are_in_grammar():
    schema = contracts.learning_contract(citations.scope_citation_schema(service.QUIZZES_SCHEMA, ["s1"]), "quizzes", 1)
    grammar = schema_grammar.schema_to_grammar(schema)
    valid = {"title": "Quiz", "questions": [{"question": "Which?", "options": ["A", "B", "C", "D"],
             "correct_option": 3, "explanation": "Evidence", "source_ids": ["s1"]}]}
    assert _accepts(grammar, valid)
    for key, value in (("correct_option", -1), ("correct_option", 4), ("correct_option", True),
                       ("options", ["A", "B", "C"]), ("question", "")):
        invalid = deepcopy(valid)
        invalid["questions"][0][key] = value
        assert not _accepts(grammar, invalid)
    assert not _accepts(grammar, {"title": "Quiz", "questions": valid["questions"] * 2})
    presentation = contracts.presentation_contract(citations.scope_citation_schema(service.PRESENTATION_SCHEMA, ["s1"]), 1)
    grammar = schema_grammar.schema_to_grammar(presentation)
    deck = {"title": "Deck", "slides": [{"title": "Slide", "bullets": ["x" * 120] * 5,
                                        "notes": "", "source_ids": ["s1"]}]}
    assert _accepts(grammar, deck)
    deck["slides"][0]["bullets"][0] += "x"
    assert not _accepts(grammar, deck)


def test_tuple_positions_and_separate_references_keep_their_own_scope():
    schema = {"type": "array", "prefixItems": [{"$ref": "#/$defs/a_b"}, {"$ref": "#/$defs/a-b"}],
              "minItems": 2, "maxItems": 2, "$defs": {
                  "a_b": {"type": "string", "enum": ["first"]},
                  "a-b": {"type": "string", "enum": ["second"]}}}
    grammar = schema_grammar.schema_to_grammar(schema)
    assert _accepts(grammar, ["first", "second"])
    assert not _accepts(grammar, ["first", "first"])
    assert not _accepts(grammar, ["first"])
    assert not _accepts(grammar, ["first", "second", "second"])


def test_non_context_free_uniqueness_still_requires_the_complete_local_schema():
    schema = {"type": "array", "items": {"type": "string", "enum": ["s1", "s2"]},
              "minItems": 1, "maxItems": 2, "uniqueItems": True}
    # The grammar constrains membership/count; equality across arbitrary values
    # belongs to the mandatory shared JSON Schema validator, as in llama.cpp.
    assert _accepts(schema_grammar.schema_to_grammar(schema), ["s1", "s1"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(["s1", "s1"], schema)


def test_large_bounds_preserve_limits_without_native_repetition_expansion():
    # Native llama.cpp rejects large {m,n} expansion (typically >=2,000).
    # Quiz explanations allow 3,000 characters and direct plots 10,000 items.
    schemas = [
        {"type": "string", "minLength": 2500, "maxLength": 3000},
        {"type": "array", "items": {"type": "null"}, "minItems": 3000, "maxItems": 3000},
    ]
    for schema, make_value, valid, invalid in (
        (schemas[0], lambda count: "x" * count, (2500, 2751, 3000), (2499, 3001)),
        (schemas[1], lambda count: [None] * count, (3000,), (2999, 3001)),
    ):
        grammar = schema_grammar.schema_to_grammar(schema)
        bounds = [int(number) for expression in re.findall(r"\{([\d,]+)\}", grammar)
                  for number in expression.split(",") if number]
        assert max(bounds) <= 32
        assert len(grammar.splitlines()) < 64
        assert all(_accepts(grammar, make_value(count)) for count in valid)
        assert not any(_accepts(grammar, make_value(count)) for count in invalid)


@pytest.mark.parametrize("schema", [
    {"type": "string", "pattern": "^.*$"},
    {"type": "number", "minimum": 0},
    {"type": "array", "items": {"type": "string"}, "contains": {"type": "string"}},
    {"type": "object", "additionalProperties": True},
    {"$ref": "https://example.invalid/schema"},
    {"$ref": "#/$defs/loop", "$defs": {"loop": {"$ref": "#/$defs/loop"}}},
    {"type": "array", "items": {"type": "string"}, "maxItems": 100000000},
    {"type": "string", "maxLength": 1, "enum": ["too long"]},
])
def test_unsupported_or_unsafe_constraints_fail_without_generic_json_fallback(schema):
    with pytest.raises(ValueError, match="cannot be constrained"):
        schema_grammar.schema_to_grammar(schema)


def test_schema_and_grammar_size_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(schema_grammar, "MAX_SCHEMA_BYTES", 30)
    with pytest.raises(ValueError, match="size limit"):
        schema_grammar.schema_to_grammar({"type": "string", "enum": ["long value"]})
    monkeypatch.setattr(schema_grammar, "MAX_SCHEMA_BYTES", 10000)
    monkeypatch.setattr(schema_grammar, "MAX_GRAMMAR_BYTES", 20)
    with pytest.raises(ValueError, match="size limit"):
        schema_grammar.schema_to_grammar({"type": "string"})
