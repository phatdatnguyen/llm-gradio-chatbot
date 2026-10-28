"""Bounded schema-specific GBNF for local llama.cpp-compatible servers.

The pinned llama.cpp converter supplies the grammar syntax and range algorithms.
This boundary rejects unsupported constraints instead of silently making a generic
JSON grammar, prevents remote/recursive refs, and fixes literal/reference escaping.
Large bounded repetitions use shared rules within llama.cpp's parser limits.
GBNF cannot enforce arbitrary ``uniqueItems`` or exclusive ``oneOf`` semantics;
the shared facade always enforces those with the complete original JSON Schema.
No schema is mutated and no network request or generated code is executed here.
"""

from copy import deepcopy
import json

import jsonschema

from ._llama_schema import PRIMITIVE_RULES, RESERVED_NAMES, SchemaConverter


MAX_SCHEMA_BYTES = 1024 * 1024
MAX_GRAMMAR_BYTES = 1024 * 1024
MAX_SCHEMA_DEPTH = 48
MAX_SCHEMA_NODES = 20000
MAX_RULES = 10000
MAX_REPEAT = 10000
_ANNOTATIONS = {"title", "description", "$comment", "$schema", "default", "examples", "$defs", "definitions"}
_COMMON = _ANNOTATIONS | {"type", "enum", "const"}
_KEYWORDS = {
    "object": {"properties", "required", "additionalProperties"},
    "array": {"items", "prefixItems", "minItems", "maxItems", "uniqueItems"},
    "string": {"minLength", "maxLength"},
    "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"},
    "number": set(), "boolean": set(), "null": set(),
}


def _fail(detail):
    raise ValueError("This schema cannot be constrained by the local GBNF grammar: " + detail)


def _reference(root, reference):
    if not isinstance(reference, str) or not reference.startswith("#/"):
        _fail("only local JSON-pointer references are supported.")
    target = root
    try:
        for part in reference[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            target = target[int(key)] if isinstance(target, list) else target[key]
    except (KeyError, IndexError, ValueError, TypeError):
        _fail("a local reference does not exist.")
    if not isinstance(target, dict):
        _fail("a local reference must identify a schema object.")
    return target


def _bound(node, minimum, maximum):
    low, high = node.get(minimum, 0), node.get(maximum)
    for value in (low, high):
        if value is not None and (type(value) is not int or not 0 <= value <= MAX_REPEAT):
            _fail("a length or item count is outside the supported range.")
    if high is not None and low > high:
        _fail("a minimum exceeds its maximum.")


def _validate_schema(root):
    count = 0

    def visit(node, depth):
        nonlocal count
        count += 1
        if count > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
            _fail("the schema exceeds the size or nesting limit.")
        if not isinstance(node, dict):
            _fail("boolean or unconstrained subschemas are unsupported.")
        for name in ("$defs", "definitions"):
            if name in node:
                if not isinstance(node[name], dict):
                    _fail("definitions must be objects.")
                for value in node[name].values():
                    visit(value, depth + 1)
        if "$ref" in node:
            if set(node) - (_ANNOTATIONS | {"$ref"}):
                _fail("reference siblings are unsupported.")
            _reference(root, node["$ref"])
            return
        unions = [name for name in ("anyOf", "oneOf") if name in node]
        if unions:
            name = unions[0]
            if len(unions) != 1 or set(node) - (_ANNOTATIONS | {name}):
                _fail("union siblings are unsupported; use complete alternatives.")
            if not isinstance(node[name], list) or not node[name]:
                _fail("unions require nonempty alternatives.")
            for value in node[name]:
                visit(value, depth + 1)
            return
        types = node.get("type")
        types = types if isinstance(types, list) else [types]
        if not types or any(not isinstance(kind, str) or kind not in _KEYWORDS for kind in types):
            _fail("an explicit supported type is required.")
        if len(set(types)) != len(types):
            _fail("duplicate union types are unsupported.")
        allowed = _COMMON | set().union(*(_KEYWORDS[kind] for kind in types))
        if set(node) - allowed:
            _fail("an unsupported validation keyword was supplied.")
        if "enum" in node:
            if not isinstance(node["enum"], list) or not node["enum"]:
                _fail("enum must contain at least one value.")
        if "enum" in node or "const" in node:
            validator = jsonschema.validators.validator_for(root)(root)
            values = node.get("enum", [node.get("const")])
            if any(not validator.evolve(schema=node).is_valid(value) for value in values):
                _fail("a constant or enum value conflicts with its constraints.")
        if "object" in types:
            properties = node.get("properties", {})
            required = node.get("required", [])
            if (not isinstance(properties, dict) or len(properties) > 128
                    or node.get("additionalProperties") is not False
                    or not isinstance(required, list)
                    or any(not isinstance(value, str) or value not in properties for value in required)
                    or len(required) != len(set(required))):
                _fail("objects need bounded properties, valid required names, and additionalProperties:false.")
            # Avoid the upstream generic-object branch for the empty object.
            node.setdefault("properties", {})
            for value in properties.values():
                visit(value, depth + 1)
        if "array" in types:
            _bound(node, "minItems", "maxItems")
            if "uniqueItems" in node and type(node["uniqueItems"]) is not bool:
                _fail("uniqueItems must be boolean.")
            if "prefixItems" in node:
                items = node["prefixItems"]
                if (not isinstance(items, list) or node.get("minItems") != len(items)
                        or node.get("maxItems") != len(items) or node.get("items", False) is not False):
                    _fail("tuple arrays require an exact length and no additional items.")
                node.pop("items", None)
                for value in items:
                    visit(value, depth + 1)
            elif node.get("maxItems") == 0 and "items" not in node:
                node["items"] = {"type": "null"}
            elif isinstance(node.get("items"), dict):
                visit(node["items"], depth + 1)
            else:
                _fail("array items must have a constrained schema.")
        if "string" in types:
            _bound(node, "minLength", "maxLength")
        if "integer" in types:
            keys = _KEYWORDS["integer"]
            if any(type(node[key]) is not int or abs(node[key]) >= 10**15 for key in keys & set(node)):
                _fail("integer bounds must be integers with at most 15 digits.")
            lower = max(node.get("minimum", -10**15), node.get("exclusiveMinimum", -10**15 - 1) + 1)
            upper = min(node.get("maximum", 10**15), node.get("exclusiveMaximum", 10**15 + 1) - 1)
            if lower > upper:
                _fail("integer bounds have no valid values.")
            # The upstream converter chooses inclusive over exclusive bounds;
            # normalize their intersection instead of discarding either one.
            if "minimum" in node or "exclusiveMinimum" in node:
                node["minimum"] = lower
            if "maximum" in node or "exclusiveMaximum" in node:
                node["maximum"] = upper
            node.pop("exclusiveMinimum", None)
            node.pop("exclusiveMaximum", None)

    visit(root, 0)


class _BoundedConverter(SchemaConverter):
    def __init__(self, schema):
        super().__init__(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
        self.schema = schema
        self._resolved = {}
        self._resolving = set()
        self._depth = 0
        self._visits = 0
        self._exact_repetitions = {}
        self._bounded_repetitions = {}
        self._grammar_bytes = sum(len((name + " ::= " + rule + "\n").encode("utf-8"))
                                  for name, rule in self._rules.items())

    def _format_literal(self, literal):
        # GBNF quotes the JSON spelling, so backslashes need another escaping
        # layer. This also prevents property/enum strings from injecting rules.
        return json.dumps(literal, ensure_ascii=False)

    def _resolve_ref(self, reference):
        if reference in self._resolving:
            _fail("recursive references are unsupported.")
        if reference not in self._resolved:
            self._resolving.add(reference)
            target = _reference(self.schema, reference)
            name = f"definition-{len(self._resolved)}-{len(self._resolving)}"
            self._resolved[reference] = self.visit(target, name)
            self._resolving.remove(reference)
        return self._resolved[reference]

    def _add_rule(self, name, rule):
        previous_count = len(self._rules)
        result = super()._add_rule(name, rule)
        if len(self._rules) != previous_count:
            self._grammar_bytes += len((result + " ::= " + rule + "\n").encode("utf-8"))
        if len(self._rules) > MAX_RULES or self._grammar_bytes > MAX_GRAMMAR_BYTES:
            _fail("the compiled grammar exceeds the size limit.")
        return result

    def _exact(self, atom, count):
        if count == 0:
            return ""
        if count == 1:
            return atom
        if count <= 32:
            return f"{atom}{{{count}}}"
        key = atom, count
        if key not in self._exact_repetitions:
            half = self._exact(atom, count // 2)
            remainder = f" {atom}" if count % 2 else ""
            rule = self._add_rule(f"repeat-exact-{len(self._exact_repetitions)}",
                                  f"{half} {half}{remainder}")
            self._exact_repetitions[key] = rule
        return self._exact_repetitions[key]

    def _up_to(self, atom, count):
        if count == 0:
            return ""
        if count <= 32:
            return f"{atom}{{0,{count}}}"
        key = atom, count
        if key not in self._bounded_repetitions:
            # Disjoint length ranges avoid the ambiguous sequence of optional
            # chunks that can make native GBNF sampling exponentially slow.
            power = 1 << (count.bit_length() - 1)
            shorter = self._up_to(atom, power - 1)
            longer = self._exact(atom, power) + " " + self._up_to(atom, count - power)
            rule = self._add_rule(f"repeat-up-to-{len(self._bounded_repetitions)}",
                                  f"({shorter} | {longer})")
            self._bounded_repetitions[key] = rule
        return self._bounded_repetitions[key]

    def _repeat(self, atom, minimum, maximum):
        # llama.cpp expands {m,n} at parsing time and has a 2,000-element
        # repetition guard. Large string/item bounds therefore need shared
        # logarithmic rules, preserving every allowed count and the exact cap.
        mandatory = self._exact(atom, minimum)
        optional = f"{atom}*" if maximum is None else self._up_to(atom, maximum - minimum)
        return " ".join(part for part in (mandatory, optional) if part)

    def _visit_bounded(self, schema, name):
        if any(keyword in schema for keyword in ("$ref", "enum", "const", "anyOf", "oneOf")):
            return None
        rule_name = name + "-" if name in RESERVED_NAMES else name or "root"
        if schema.get("type") == "string" and ("minLength" in schema or "maxLength" in schema):
            char = self._add_primitive("char", PRIMITIVE_RULES["char"])
            repeated = self._repeat(char, schema.get("minLength", 0), schema.get("maxLength"))
            return self._add_rule(rule_name, r'"\"" ' + repeated + r' "\"" space')
        if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
            item = self.visit(schema["items"], f'{name}{"-" if name else ""}item')
            minimum, maximum = schema.get("minItems", 0), schema.get("maxItems")
            if maximum == 0:
                content = ""
            else:
                tail = self._repeat(f'("," space {item})', max(0, minimum - 1),
                                    None if maximum is None else maximum - 1)
                content = item + " " + tail
                if minimum == 0:
                    content = "(" + content + ")?"
            return self._add_rule(rule_name, '"[" space ' + content + ' "]" space')
        return None

    def visit(self, schema, name):
        self._visits += 1
        self._depth += 1
        try:
            if self._depth > MAX_SCHEMA_DEPTH or self._visits > MAX_SCHEMA_NODES:
                _fail("reference expansion exceeds the size or nesting limit.")
            bounded = self._visit_bounded(schema, name)
            if bounded is not None:
                return bounded
            return super().visit(schema, name)
        finally:
            self._depth -= 1


def schema_to_grammar(schema):
    """Return schema-specific GBNF, or fail before any inference can be sent."""
    if not isinstance(schema, dict):
        _fail("the root must be a schema object.")
    try:
        encoded = json.dumps(schema, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_SCHEMA_BYTES:
            _fail("the schema exceeds the size limit.")
        normalized = deepcopy(schema)
        _validate_schema(normalized)
        converter = _BoundedConverter(normalized)
        converter.visit(normalized, "")
        grammar = converter.format_grammar()
        if len(grammar.encode("utf-8")) > MAX_GRAMMAR_BYTES:
            _fail("the compiled grammar exceeds the size limit.")
        return grammar
    except (AssertionError, KeyError, RecursionError, UnicodeError, TypeError) as exc:
        raise ValueError("This schema cannot be constrained by the local GBNF grammar.") from exc
