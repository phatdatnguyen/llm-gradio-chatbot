"""Request-scoped citation contracts for local structured generation."""

from copy import deepcopy


def evidence_source_ids(evidence):
    """Collect IDs from app-owned evidence fields, never from passage text."""
    found = set()

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key == "source_id":
                    if not isinstance(item, str) or not item:
                        raise ValueError("Invalid source identifier in tool evidence.")
                    found.add(item)
                elif key == "source_ids":
                    if not isinstance(item, list) or any(not isinstance(identifier, str) or not identifier for identifier in item):
                        raise ValueError("Invalid source identifiers in tool evidence.")
                    found.update(item)
                else:
                    visit(item)

    visit(evidence)
    return found


def scope_citation_schema(schema, allowed_ids, *, allow_empty_paths=()):
    """Bind citation fields to this request without mutating shared schemas.

    One shared enum keeps large snapshots from duplicating IDs at every field.
    Local validators still enforce uniqueness and document/claim ownership even
    when a server's grammar engine ignores a JSON Schema constraint.
    """
    identifiers = list(allowed_ids)
    if (not identifiers or any(not isinstance(item, str) or not item for item in identifiers)
            or len(identifiers) != len(set(identifiers))):
        raise ValueError("Tool citation scope needs unique, nonempty source identifiers.")
    scoped = deepcopy(schema)
    definition = "source_identifier"
    definitions = scoped.setdefault("$defs", {})
    if definition in definitions:
        raise ValueError("The tool schema already defines its citation scope.")
    definitions[definition] = {"type": "string", "enum": identifiers}
    reference = {"$ref": f"#/$defs/{definition}"}
    empty_paths = set(allow_empty_paths)

    def visit(node, path=()):
        if not isinstance(node, dict):
            return
        for name, child in node.get("properties", {}).items():
            child_path = (*path, name)
            if name == "source_id":
                node["properties"][name] = dict(reference)
            elif name == "source_ids":
                node["properties"][name] = {
                    "type": "array", "items": dict(reference),
                    "minItems": 0 if child_path in empty_paths else 1,
                    "maxItems": len(identifiers), "uniqueItems": True,
                }
            else:
                visit(child, child_path)
        if "items" in node:
            visit(node["items"], (*path, "items"))
        for keyword in ("anyOf", "oneOf", "allOf"):
            for child in node.get(keyword, []):
                visit(child, path)

    visit(scoped)
    return scoped
