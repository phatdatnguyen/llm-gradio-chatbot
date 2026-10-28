"""Generation-only schemas that match the trusted tool validators.

Schema helpers narrow requests without changing saved artifacts. Apply them after
citation scoping so document-specific Compare rules remain intact. The Compare
resolver copies explicit finding references into canonical citations without
rewriting model text. Domain validators still check relationships and coverage.
"""

from copy import deepcopy

from . import learning, presentation, review
from .evidence import require_object


def _text(maximum, *, allow_empty=False):
    minimum = 0 if allow_empty else 1
    # Use the backend's normal JSON-string grammar. A negated control-character
    # regex can bypass JSON escaping in llama.cpp's raw pattern converter.
    # Decoded control characters remain rejected by the domain validators.
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def learning_contract(schema, tool_id, item_count):
    """Constrain this study request while retaining the saved-output format.

    Count and field bounds are part of generation, not only post-validation.
    Prompt/option uniqueness after whitespace normalization and document
    coverage remain the responsibility of the trusted learning validators.
    """
    item_count = learning.validate_item_count(tool_id, item_count)
    result = deepcopy(schema)
    result["properties"]["title"] = _text(200)
    key = "cards" if tool_id == "flashcards" else "questions"
    items = result["properties"][key]
    items.update(minItems=item_count, maxItems=item_count)
    fields = items["items"]["properties"]
    if tool_id == "flashcards":
        fields["front"] = _text(learning.MAX_PROMPT_CHARACTERS)
        fields["back"] = _text(learning.MAX_ANSWER_CHARACTERS)
    else:
        fields["question"] = _text(learning.MAX_PROMPT_CHARACTERS)
        fields["explanation"] = _text(learning.MAX_ANSWER_CHARACTERS)
        fields["options"] = {
            "type": "array", "items": _text(learning.MAX_OPTION_CHARACTERS),
            "minItems": learning.QUIZ_OPTION_COUNT, "maxItems": learning.QUIZ_OPTION_COUNT,
            "uniqueItems": True,
        }
        fields["correct_option"] = {
            "type": "integer", "minimum": 0, "maximum": learning.QUIZ_OPTION_COUNT - 1,
        }
    return result


def presentation_contract(schema, slide_count):
    """Constrain exact slide count and readable text without truncating output.

    JSON Schema cannot sum string lengths. A count-specific per-bullet allowance
    guarantees the existing total bound: up to three bullets can use 180 chars
    each; four get 150 each; five get 120 each. Previously saved decks retain the
    full validator policy, including unequal bullet lengths totaling at most 600.
    """
    slide_count = presentation.validate_slide_count(slide_count)
    result = deepcopy(schema)
    result["properties"]["title"] = _text(200)
    slides = result["properties"]["slides"]
    slides.update(minItems=slide_count, maxItems=slide_count)
    properties = slides["items"]["properties"]
    properties["title"] = _text(presentation.MAX_TITLE_CHARACTERS)
    properties["notes"] = _text(presentation.MAX_NOTES_CHARACTERS, allow_empty=True)
    # Consecutive counts with the same allowance share one branch. In
    # particular, 1-3 bullets all allow 180 characters; repeating their text
    # schema needlessly consumes small models' context windows.
    branches = []
    for count in range(1, presentation.MAX_BULLETS + 1):
        maximum = min(presentation.MAX_BULLET_CHARACTERS,
                      presentation.MAX_BULLET_TOTAL_CHARACTERS // count)
        if branches and branches[-1]["items"]["maxLength"] == maximum:
            branches[-1]["maxItems"] = count
        else:
            branches.append({"type": "array", "minItems": count, "maxItems": count,
                             "items": _text(maximum)})
    properties["bullets"] = {"anyOf": branches}
    return result


def compare_contract(schema, documents, source_ids_by_document):
    """Bind every finding position to one canonical document and its sources.

    ``prefixItems`` plus an exact array length keeps document IDs unique and in
    selection order. Omitting ``items`` also works with tuple grammar converters
    that prefer that field over ``prefixItems`` when both are present.
    """
    document_ids = review.validate_compare_documents(documents)
    if not isinstance(source_ids_by_document, dict) or set(source_ids_by_document) != set(document_ids):
        raise ValueError("Compare needs a source scope for every selected document.")
    seen = set()
    for identifier in document_ids:
        ids = source_ids_by_document[identifier]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("Compare document source scopes must be nonempty and distinct.")
        seen.update(ids)

    result = deepcopy(schema)
    result["properties"]["title"] = _text(200)
    aspects = result["properties"]["aspects"]
    aspects.update(minItems=1, maxItems=review.MAX_COMPARE_ASPECTS)
    properties = aspects["items"]["properties"]
    properties["aspect"] = _text(review.MAX_COMPARE_ASPECT_CHARACTERS)
    properties["synthesis"] = {
        "type": "object",
        "properties": {
            "text": _text(review.MAX_COMPARE_SYNTHESIS_CHARACTERS),
            "finding_document_ids": {"type": "array",
                                     "items": {"type": "string", "enum": list(document_ids)},
                                     "minItems": 1, "maxItems": len(document_ids), "uniqueItems": True},
        },
        "required": ["text", "finding_document_ids"], "additionalProperties": False,
    }
    finding = properties["findings"]["items"]
    positions = []
    for identifier in document_ids:
        alternatives = []
        for status in ("found", "not_found"):
            branch = deepcopy(finding)
            fields = branch["properties"]
            fields["document_id"] = {"type": "string", "enum": [identifier]}
            fields["status"] = {"type": "string", "enum": [status]}
            fields["text"] = _text(review.MAX_COMPARE_FINDING_CHARACTERS)
            fields["source_ids"] = {
                "type": "array", "items": {"type": "string", "enum": list(source_ids_by_document[identifier])},
                "minItems": 1 if status == "found" else 0,
                "maxItems": len(source_ids_by_document[identifier]) if status == "found" else 0,
                "uniqueItems": True,
            }
            alternatives.append(branch)
        positions.append({"anyOf": alternatives})
    properties["findings"] = {"type": "array", "prefixItems": positions,
                               "minItems": len(document_ids), "maxItems": len(document_ids)}
    return result


def resolve_compare_synthesis(value, documents):
    """Resolve only the findings explicitly selected for each synthesis.

    A not_found finding may describe a limitation alongside a selected found
    finding; it contributes no citations. Referencing only not_found findings
    cannot produce a cited synthesis. Old canonical output is deliberately not
    accepted as a generation fallback; saved artifact readers stay unchanged.
    """
    document_ids = review.validate_compare_documents(documents)
    require_object(value, {"title", "aspects"})
    aspects = value["aspects"]
    if not isinstance(aspects, list) or not 1 <= len(aspects) <= review.MAX_COMPARE_ASPECTS:
        raise ValueError("Compare requires between 1 and 12 comparison aspects.")
    result = deepcopy(value)
    for aspect in result["aspects"]:
        require_object(aspect, {"aspect", "findings", "synthesis"})
        findings = aspect["findings"]
        if not isinstance(findings, list) or len(findings) != len(document_ids):
            raise ValueError("Each comparison aspect must contain one finding for every selected document.")
        by_document = {}
        for finding in findings:
            require_object(finding, {"document_id", "status", "text", "source_ids"})
            identifier = finding["document_id"]
            if not isinstance(identifier, str) or identifier not in document_ids or identifier in by_document:
                raise ValueError("Compare returned an unknown or duplicate document finding.")
            status, citations = finding["status"], finding["source_ids"]
            if (status not in ("found", "not_found") or not isinstance(citations, list)
                    or any(not isinstance(item, str) or not item for item in citations)
                    or len(citations) != len(set(citations))
                    or (status == "found" and not citations)
                    or (status == "not_found" and citations)):
                raise ValueError("Compare findings must retain valid citations for their found/not_found status.")
            by_document[identifier] = citations
        synthesis = aspect["synthesis"]
        require_object(synthesis, {"text", "finding_document_ids"})
        selected = synthesis["finding_document_ids"]
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(identifier, str) or identifier not in by_document for identifier in selected)
                or len(selected) != len(set(selected))):
            raise ValueError("Compare synthesis must select unique, known finding document IDs.")
        citations = list(dict.fromkeys(source_id for identifier in selected for source_id in by_document[identifier]))
        if not citations:
            raise ValueError("Compare synthesis must reference at least one found finding with evidence.")
        aspect["synthesis"] = {"text": synthesis["text"], "source_ids": citations}
    return result
