"""Exact source excerpts selected by app-owned IDs, never transcribed by a model.

Quote catalogs retain source order and cover long passages with overlapping exact
slices. Model responses choose IDs and stances; resolution copies the original
strings without fuzzy matching, normalization, or additional model requests.
"""

from .evidence import require_object
from .review import (
    MAX_EVIDENCE_PER_CLAIM, MAX_QUOTE_CHARACTERS, STANCES, VERDICTS,
    validate_claims,
)


QUOTE_OVERLAP_CHARACTERS = 200


def _text(value, limit, label):
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any((ord(char) < 32 and char not in "\n\r\t") or ord(char) == 127
                   for char in value)):
        raise ValueError(f"Evidence Check {label} must be nonempty bounded text.")
    return value


def build_quote_catalog(aliases):
    """Copy all selected passages into ordered, overlapping bounded excerpts.

    Unicode slicing uses Python character boundaries. No stripping, sentence
    rewriting, or whitespace conversion changes the quoted source text.
    """
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError("Evidence Check needs a nonempty source snapshot.")
    catalog = []
    step = MAX_QUOTE_CHARACTERS - min(QUOTE_OVERLAP_CHARACTERS, MAX_QUOTE_CHARACTERS - 1)
    for source_id, source in aliases.items():
        if not isinstance(source_id, str) or not source_id.strip() or not isinstance(source, dict):
            raise ValueError("Evidence Check has an invalid source snapshot.")
        text = source.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Evidence Check source passages must contain text.")
        for start in range(0, len(text), step):
            quote = text[start:start + MAX_QUOTE_CHARACTERS]
            _text(quote, MAX_QUOTE_CHARACTERS, "source excerpt")
            catalog.append({"quote_id": f"q{len(catalog) + 1}",
                            "source_id": source_id, "quote": quote})
            if start + MAX_QUOTE_CHARACTERS >= len(text):
                break
    return catalog


def _catalog(catalog):
    if not isinstance(catalog, list):
        raise ValueError("Evidence Check needs an app-owned quote catalog.")
    found = {}
    for row in catalog:
        require_object(row, {"quote_id", "source_id", "quote"})
        identifier = row["quote_id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in found:
            raise ValueError("Evidence Check quote identifiers must be unique and nonempty.")
        if not isinstance(row["source_id"], str) or not row["source_id"].strip():
            raise ValueError("Evidence Check quote sources must have exact identifiers.")
        _text(row["quote"], MAX_QUOTE_CHARACTERS, "source excerpt")
        found[identifier] = row
    return found


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _generated_text(maximum):
    # Use the decoder's normal JSON-string grammar. Regex patterns can bypass
    # its quote/backslash escaping; decoded control characters are checked by
    # _text() instead. Keep original catalog text untouched.
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _schema(catalog, claims, *, assessment):
    known = _catalog(catalog)
    claims = validate_claims(claims)
    quote_type = {"$ref": "#/$defs/quote_identifier"} if known else {"type": "string"}
    maximum = min(MAX_EVIDENCE_PER_CLAIM, len(known))
    definitions = {}
    if known:
        definitions["quote_identifier"] = {"type": "string", "enum": list(known)}

    def reference(name):
        return {"$ref": f"#/$defs/{name}"}

    def excerpt(stances):
        return _object({"quote_id": dict(quote_type),
                        "stance": {"type": "string", "enum": list(stances)}})

    def ordered_evidence(first, later):
        # A finite tuple alternative enforces the decisive prefix without
        # contains/conditionals or a prefixItems+items tail. Some grammar
        # converters ignore prefixItems when items is also present.
        return {"anyOf": [
            {"type": "array", "prefixItems": [reference(name) for name in
                [*first, *([later] * (count - len(first)))]],
             "minItems": count, "maxItems": count, "uniqueItems": True}
            for count in range(len(first), maximum + 1)
        ]}

    definitions["evidence_not_found"] = (
        {"type": "array", "minItems": 0, "maxItems": maximum, "uniqueItems": True,
         "items": excerpt(["context"])} if known else
        {"type": "array", "maxItems": 0}
    )
    positions = []
    if assessment:
        definitions["rationale"] = _generated_text(4000)
        verdicts = ["not_found"]
        if known:
            definitions.update({
                "support": excerpt(["supports"]),
                "contradiction": excerpt(["contradicts"]),
                "support_context": excerpt(["supports", "context"]),
                "contradiction_context": excerpt(["contradicts", "context"]),
                "evidence_supported": ordered_evidence(["support"], "support_context"),
                "evidence_contradicted": ordered_evidence(["contradiction"], "contradiction_context"),
            })
            verdicts = ["supported", "contradicted", "not_found"]
        if maximum >= 2:
            definitions["any_stance"] = excerpt(STANCES)
            definitions["evidence_mixed"] = ordered_evidence(["support", "contradiction"], "any_stance")
            verdicts.insert(2, "mixed")
        for claim in claims:
            # Each alternative is a complete object: do not mix properties with
            # anyOf on the same node, which is not portable across decoders.
            positions.append({"anyOf": [_object({
                "id": {"type": "string", "enum": [claim["id"]]},
                "verdict": {"type": "string", "enum": [verdict]},
                "rationale": reference("rationale"),
                "evidence": reference(f"evidence_{verdict}"),
            }) for verdict in verdicts]})
    else:
        definitions["selection"] = {
            "type": "array", "minItems": 0, "maxItems": maximum, "uniqueItems": True,
            "items": excerpt(STANCES),
        }
        positions = [_object({"id": {"type": "string", "enum": [claim["id"]]},
                              "evidence": reference("selection")}) for claim in claims]
    properties = {}
    if assessment:
        properties["title"] = _generated_text(200)
    properties["claims"] = {"type": "array", "prefixItems": positions,
                            "minItems": len(claims), "maxItems": len(claims)}
    schema = _object(properties)
    schema["$defs"] = definitions
    return schema


def assessment_schema(catalog, claims):
    """Constrain final verdicts to the provided claims and exact quote choices."""
    return _schema(catalog, claims, assessment=True)


def selection_schema(catalog, claims):
    """Select relevant excerpts while allowing a claim to have no evidence."""
    return _schema(catalog, claims, assessment=False)


def _claim_rows(value, claims, *, assessment):
    require_object(value, {"title", "claims"} if assessment else {"claims"})
    if assessment:
        _text(value["title"], 200, "title")
    canonical = validate_claims(claims)
    rows = value["claims"]
    if not isinstance(rows, list) or len(rows) != len(canonical):
        raise ValueError("Evidence Check must assess every requested claim exactly once.")
    allowed = {claim["id"] for claim in canonical}
    found = {}
    for row in rows:
        require_object(row, {"id", "verdict", "rationale", "evidence"}
                       if assessment else {"id", "evidence"})
        identifier = row["id"]
        if not isinstance(identifier, str) or identifier not in allowed or identifier in found:
            raise ValueError("Evidence Check returned an unknown or duplicate claim identifier.")
        found[identifier] = row
    return [found[claim["id"]] for claim in canonical]


def _resolve_evidence(evidence, known):
    if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE_PER_CLAIM:
        raise ValueError("Each Evidence Check claim permits at most 8 evidence excerpts.")
    resolved, identifiers, excerpts = [], set(), set()
    for item in evidence:
        require_object(item, {"quote_id", "stance"})
        identifier = item["quote_id"]
        if not isinstance(identifier, str) or identifier not in known:
            raise ValueError("Evidence Check returned an unknown quote identifier.")
        if identifier in identifiers:
            raise ValueError("Evidence Check must not repeat a quote for one claim.")
        identifiers.add(identifier)
        stance = item["stance"]
        if not isinstance(stance, str) or stance not in STANCES:
            raise ValueError("Evidence Check returned an invalid evidence stance.")
        row = known[identifier]
        # Match the existing saved-output rule, without altering the copied quote.
        key = (row["source_id"], " ".join(row["quote"].split()))
        if key in excerpts:
            raise ValueError("Evidence Check must not repeat the same cited excerpt for one claim.")
        excerpts.add(key)
        resolved.append({"source_id": row["source_id"], "quote": row["quote"], "stance": stance})
    return resolved


def resolve_assessment(value, catalog, claims):
    """Convert quote-ID selections into the existing exact-quotation wire shape."""
    known = _catalog(catalog)
    rows = _claim_rows(value, claims, assessment=True)
    resolved = []
    expected = {"supported": {"supports"}, "contradicted": {"contradicts"},
                "mixed": {"supports", "contradicts"}, "not_found": set()}
    for row in rows:
        verdict = row["verdict"]
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise ValueError("Evidence Check returned an invalid verdict.")
        rationale = _text(row["rationale"], 4000, "rationale")
        evidence = _resolve_evidence(row["evidence"], known)
        decisive = {item["stance"] for item in evidence} - {"context"}
        if decisive != expected[verdict]:
            raise ValueError("Evidence Check verdict must match the supporting and contradicting evidence stances.")
        resolved.append({"id": row["id"], "verdict": verdict,
                         "rationale": rationale, "evidence": evidence})
    return {"title": value["title"], "claims": resolved}


def selected_quotes(value, catalog, claims):
    """Return all chosen original excerpts in catalog order, without truncation.

    A quote may support different claims, but may appear only once within a
    claim's selection. Selection stances are validated; callers retaining the
    intermediate classifications must carry those selections separately.
    """
    known = _catalog(catalog)
    rows = _claim_rows(value, claims, assessment=False)
    selected = set()
    for row in rows:
        _resolve_evidence(row["evidence"], known)
        selected.update(item["quote_id"] for item in row["evidence"])
    return [dict(row) for identifier, row in known.items() if identifier in selected]
