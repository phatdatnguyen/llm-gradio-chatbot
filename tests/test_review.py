"""Compare and Evidence Check validators and whole-source mocked generation."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import jsonschema
from chatbot import config

pytestmark = pytest.mark.usefixtures("local_models")

from rag_chatbot.config import Settings
from rag_chatbot.store import LibraryStore
from rag_chatbot.tools import review, service


DOCUMENTS = [{"id": "doc-1", "name": "First"}, {"id": "doc-2", "name": "Second"}]
SOURCES = [{"id": "source-1", "document_id": "doc-1", "text": "The first study reported 10 nm."},
           {"id": "source-2", "document_id": "doc-2", "text": "The second study reported 20 nm."}]
CLAIMS = [{"id": "c1", "text": "The first study reported 10 nm."}]


def comparison():
    return {"aspects": [{"aspect": "Reported measurement", "findings": [
        {"document_id": "doc-1", "status": "found", "text": "10 nm", "source_ids": ["source-1"]},
        {"document_id": "doc-2", "status": "found", "text": "20 nm", "source_ids": ["source-2"]}],
        "synthesis": {"text": "The reported measurements differ.", "source_ids": ["source-1", "source-2"]}}]}


def assessment(verdict="supported"):
    stances = {"supported": ["supports"], "contradicted": ["contradicts"],
               "mixed": ["supports", "contradicts"], "not_found": []}[verdict]
    return {"claims": [{**CLAIMS[0], "verdict": verdict, "rationale": "Review these source passages.",
                        "evidence": [{"source_id": SOURCES[index]["id"], "quote": SOURCES[index]["text"],
                                      "stance": stance} for index, stance in enumerate(stances)]}]}


def test_claim_parser_assigns_ids_and_keeps_user_wording():
    assert review.parse_claims("  Claim 1!\n\n  Claim 2? \r\n") == [
        {"id": "c1", "text": "Claim 1!"}, {"id": "c2", "text": "Claim 2?"}]
    assert len(review.parse_claims("\n".join(f"Claim {index}" for index in range(20)))) == 20


@pytest.mark.parametrize("claims", [None, [], True, "", " \n ", "x" * 1001,
                                    "\n".join("Claim" for _ in range(21)), "bad\x00claim"])
def test_invalid_claims_fail_before_sources_or_model_work(monkeypatch, claims):
    monkeypatch.setattr(service, "get_service", lambda *_: pytest.fail("source access"))
    monkeypatch.setattr(service.api, "generate_structured", lambda *a, **k: pytest.fail("model request"))
    with pytest.raises(ValueError):
        list(service.generate_tool("work", "evidence_check", ["doc"], claims_text=claims))


@pytest.mark.parametrize("damage", ["id", "text", "extra", "multiline", "leading_space"])
def test_saved_claims_are_canonical(damage):
    claims = deepcopy(CLAIMS)
    if damage == "id":
        claims[0]["id"] = "c2"
    elif damage == "text":
        claims[0]["text"] = ""
    elif damage == "extra":
        claims[0]["extra"] = True
    elif damage == "multiline":
        claims[0]["text"] = "One\nTwo"
    else:
        claims[0]["text"] = " One"
    with pytest.raises(ValueError):
        review.validate_claims(claims)


def test_compare_normalizes_document_order_without_mutating_input():
    content = comparison()
    content["aspects"][0]["findings"].reverse()
    before = deepcopy(content)
    actual = review.validate_compare(content, documents=DOCUMENTS, sources=SOURCES)
    assert content == before
    assert [item["document_id"] for item in actual["aspects"][0]["findings"]] == ["doc-1", "doc-2"]


def test_compare_missing_finding_is_explicit_and_uncited():
    content = comparison()
    content["aspects"][0]["findings"][1].update(status="not_found", text="Not found in the selected evidence.", source_ids=[])
    content["aspects"][0]["synthesis"]["source_ids"] = ["source-1"]
    assert review.validate_compare(content, documents=DOCUMENTS, sources=SOURCES) == content


@pytest.mark.parametrize("damage", ["cross_document", "unknown_source", "empty_citation", "duplicate_citation",
    "cited_not_found", "all_missing", "missing_document", "duplicate_document", "unknown_document", "unknown_status",
    "no_aspects", "too_many_aspects", "duplicate_aspect", "synthesis_unrelated", "synthesis_empty", "extra", "long_text"])
def test_compare_rejects_invalid_or_misattributed_findings(damage):
    content = comparison()
    aspect = content["aspects"][0]
    finding = aspect["findings"][0]
    sources = deepcopy(SOURCES)
    if damage == "cross_document":
        finding["source_ids"] = ["source-2"]
    elif damage == "unknown_source":
        finding["source_ids"] = ["missing"]
    elif damage == "empty_citation":
        finding["source_ids"] = []
    elif damage == "duplicate_citation":
        finding["source_ids"] *= 2
    elif damage == "cited_not_found":
        finding["status"] = "not_found"
    elif damage == "all_missing":
        for row in aspect["findings"]:
            row.update(status="not_found", source_ids=[])
    elif damage == "missing_document":
        aspect["findings"].pop()
    elif damage == "duplicate_document":
        aspect["findings"][1] = deepcopy(finding)
    elif damage == "unknown_document":
        finding["document_id"] = "missing"
    elif damage == "unknown_status":
        finding["status"] = "maybe"
    elif damage == "no_aspects":
        content["aspects"] = []
    elif damage == "too_many_aspects":
        content["aspects"] = [{**deepcopy(aspect), "aspect": f"Aspect {i}"} for i in range(13)]
    elif damage == "duplicate_aspect":
        content["aspects"].append({**deepcopy(aspect), "aspect": " REPORTED  measurement "})
    elif damage == "synthesis_unrelated":
        sources.append({**SOURCES[0], "id": "unrelated"})
        aspect["synthesis"]["source_ids"] = ["unrelated"]
    elif damage == "synthesis_empty":
        aspect["synthesis"]["source_ids"] = []
    elif damage == "long_text":
        finding["text"] = "x" * 3001
    else:
        aspect["extra"] = "untrusted"
    with pytest.raises(ValueError):
        review.validate_compare(content, documents=DOCUMENTS, sources=sources)


@pytest.mark.parametrize("verdict", review.VERDICTS)
def test_evidence_verdicts_and_quote_occurrence_are_validated_without_mutation(verdict):
    content = assessment(verdict)
    before = deepcopy(content)
    assert review.validate_evidence_check(content, claims=CLAIMS, sources=SOURCES) == before
    assert content == before


def test_not_found_may_include_context_without_decisive_evidence():
    content = assessment()
    content["claims"][0]["verdict"] = "not_found"
    content["claims"][0]["evidence"][0]["stance"] = "context"
    assert review.validate_evidence_check(content, claims=CLAIMS, sources=SOURCES) == content


def test_quote_whitespace_only_normalization():
    content = assessment()
    content["claims"][0]["evidence"][0]["quote"] = "The first\nstudy reported\t10 nm."
    review.validate_evidence_check(content, claims=CLAIMS, sources=SOURCES)


@pytest.mark.parametrize("damage", ["altered_text", "unknown_id", "duplicate_claim", "missing_claim", "unknown_verdict",
    "no_support", "context_only", "mixed_without_contradiction", "contradicted_with_support", "not_found_with_support",
    "unknown_source", "wrong_source", "paraphrased_quote", "case_changed", "number_changed", "empty_quote",
    "oversized_quote", "duplicate_excerpt", "unknown_stance", "too_many_quotes", "extra_field", "empty_rationale"])
def test_evidence_rejects_malformed_or_unverifiable_assessment(damage):
    content = assessment()
    claim = content["claims"][0]
    quote = claim["evidence"][0]
    if damage == "altered_text":
        claim["text"] += "?"
    elif damage == "unknown_id":
        claim["id"] = "c2"
    elif damage == "duplicate_claim":
        content["claims"].append(deepcopy(claim))
    elif damage == "missing_claim":
        content["claims"] = []
    elif damage == "unknown_verdict":
        claim["verdict"] = "true"
    elif damage == "no_support":
        claim["evidence"] = []
    elif damage == "context_only":
        quote["stance"] = "context"
    elif damage == "mixed_without_contradiction":
        claim["verdict"] = "mixed"
    elif damage == "contradicted_with_support":
        claim["verdict"] = "contradicted"
    elif damage == "not_found_with_support":
        claim["verdict"] = "not_found"
    elif damage == "unknown_source":
        quote["source_id"] = "unknown"
    elif damage == "wrong_source":
        quote["source_id"] = "source-2"
    elif damage == "paraphrased_quote":
        quote["quote"] = "10 nm was reported by the first study."
    elif damage == "case_changed":
        quote["quote"] = quote["quote"].upper()
    elif damage == "number_changed":
        quote["quote"] = quote["quote"].replace("10", "100")
    elif damage == "empty_quote":
        quote["quote"] = " "
    elif damage == "oversized_quote":
        quote["quote"] = "x" * 1201
    elif damage == "duplicate_excerpt":
        claim["evidence"].append({**quote, "stance": "context"})
    elif damage == "unknown_stance":
        quote["stance"] = "neutral"
    elif damage == "too_many_quotes":
        claim["evidence"] *= 9
    elif damage == "extra_field":
        quote["url"] = "https://untrusted.invalid"
    else:
        claim["rationale"] = ""
    with pytest.raises(ValueError):
        review.validate_evidence_check(content, claims=CLAIMS, sources=SOURCES)


@pytest.fixture
def catalog(tmp_path):
    store = LibraryStore(Settings(data_dir=tmp_path / "catalog"))
    documents = []
    for index, texts in enumerate([["The first study reported 10 nm.", "Its scope is limited to this sample."],
                                   ["The second study reported 20 nm."]], 1):
        path = tmp_path / f"source-{index}.txt"
        path.write_text("\n".join(texts), encoding="utf-8")
        document = store.stage_document(path, "science", store.settings.fingerprint())
        with store._db:
            for ordinal, text in enumerate(texts):
                store._db.execute("INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
                    (f"doc{index}-chunk{ordinal}", document["id"], text, f"page {ordinal + 1}", "text", ordinal))
            store._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=? WHERE id=?",
                              (len(texts), document["id"]))
        documents.append(document["id"])
    yield store, documents
    store.close()


def setup_generation(monkeypatch, catalog, *, transform=None, after_call=None):
    store, documents = catalog
    calls = []
    monkeypatch.setattr(service, "get_service", lambda _: SimpleNamespace(store=store))

    def generate(model, history, schema, **options):
        prompt = json.loads(history[-1]["content"])
        name = options["schema_name"]
        calls.append((name, prompt))
        if name == "tool_evidence_notes":
            result = {"notes": [{"source_id": row["source_id"], "text": row["text"]}
                                for row in prompt["evidence"]]}
        elif name == "tool_compare":
            result = {"title": "Comparison", "aspects": [{"aspect": "Measurement", "findings": [
                {"document_id": doc["id"], "status": "found", "text": "Reported measurement.",
                 "source_ids": doc["source_ids"][:1]} for doc in reversed(prompt["evidence"]["documents"])],
                "synthesis": {"text": "Different measurements.", "finding_document_ids": [
                    doc["id"] for doc in prompt["evidence"]["documents"]]}}]}
        elif name == "tool_evidence_check":
            quote = next(item for item in prompt["quotes"] if "The first study reported 10 nm." in item["quote"])
            result = {"title": "Evidence Check", "claims": [
                {"id": claim["id"], "verdict": "supported" if index == 0 else "not_found",
                 "rationale": "A source-scoped assessment.", "evidence": [
                     {"quote_id": quote["quote_id"], "stance": "supports"}]
                 if index == 0 else []} for index, claim in enumerate(prompt["requested_claims"])]}
        elif name == "tool_quote_selection":
            matching = [item for item in prompt["quotes"] if "The first study reported 10 nm." in item["quote"]]
            result = {"claims": [{"id": claim["id"], "evidence": [
                {"quote_id": item["quote_id"], "stance": "supports"} for item in matching] if index == 0 else []}
                for index, claim in enumerate(prompt["requested_claims"])]}
        else:
            pytest.fail(f"Unexpected model request {name}")
        if after_call:
            after_call(name)
        return transform(name, result) if transform else result

    monkeypatch.setattr(service.api, "generate_structured", generate)
    return calls, documents


@pytest.mark.parametrize("tool_id", ["compare", "evidence_check"])
def test_generation_reads_every_chunk_maps_citations_and_keeps_claims(catalog, monkeypatch, tool_id):
    calls, documents = setup_generation(monkeypatch, catalog)
    text = "The first study reported 10 nm.\nThe sample was independently replicated."
    result = list(service.generate_tool("work", tool_id, documents, claims_text=text))[-1]["result"]
    assert result["coverage"] == {"documents": 2, "chunks": 3, "processed_chunks": 3}
    if tool_id == "compare":
        assert result["settings"]["model_requests"] == 3
        assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", "tool_compare"]
        aspect = result["content"]["aspects"][0]
        assert [finding["document_id"] for finding in aspect["findings"]] == documents
        assert aspect["synthesis"]["source_ids"] == ["doc1-chunk0", "doc2-chunk0"]
    else:
        assert result["settings"]["model_requests"] == 1
        assert [name for name, _ in calls] == ["tool_evidence_check"]
        assert result["settings"]["claims"] == review.parse_claims(text)
        assert result["content"]["claims"][0]["text"] == CLAIMS[0]["text"]
        assert result["content"]["claims"][0]["evidence"][0]["source_id"] == "doc1-chunk0"
        assert result["content"]["claims"][1]["evidence"] == []
        assert all(prompt["requested_claims"] == review.parse_claims(text) for _, prompt in calls)
        assert {item["source_id"] for item in calls[0][1]["quotes"]} == {"s1", "s2", "s3"}
        assert result["content"]["claims"][0]["evidence"][0]["quote"] == SOURCES[0]["text"]


@pytest.mark.parametrize("count", [0, 1, 9])
def test_compare_document_limit_precedes_model_requests(monkeypatch, count):
    documents = [{"id": f"doc-{index}"} for index in range(count)]
    monkeypatch.setattr(service, "get_service", lambda _: SimpleNamespace(store=object()))
    monkeypatch.setattr(service, "prepare_snapshot", lambda *a, **k: ({"documents": documents}, {}, {}))
    monkeypatch.setattr(service.api, "generate_structured", lambda *a, **k: pytest.fail("model request"))
    with pytest.raises(ValueError, match="2 and 8"):
        list(service.generate_tool("work", "compare", [document["id"] for document in documents]))


@pytest.mark.parametrize("tool_id,change_at", [("compare", "notes"), ("compare", "final"),
                                               ("evidence_check", "final")])
@pytest.mark.parametrize("change", ["cancel", "snapshot"])
def test_stale_or_cancelled_results_never_publish(catalog, monkeypatch, tool_id, change_at, change):
    stopped = False

    def invalidate(name):
        nonlocal stopped
        target = "tool_evidence_notes" if change_at == "notes" else f"tool_{tool_id}"
        if name != target:
            return
        if change == "cancel":
            stopped = True
        else:
            with catalog[0]._db:
                catalog[0]._db.execute("UPDATE documents SET generation='new'")

    calls, documents = setup_generation(monkeypatch, catalog, after_call=invalidate)
    events = []
    with pytest.raises((service.ToolCancelled, ValueError)):
        for event in service.generate_tool("work", tool_id, documents,
                                           claims_text=CLAIMS[0]["text"], is_cancelled=lambda: stopped):
            events.append(event)
    assert all(event["result"] is None for event in events)
    assert len(calls) == (3 if tool_id == "compare" and change_at == "final" else 1)


@pytest.mark.parametrize("tool_id", ["compare", "evidence_check"])
def test_invalid_final_result_fails_without_retry(catalog, monkeypatch, tool_id):
    def damage(name, result):
        if name == "tool_compare":
            result["aspects"][0]["findings"][0]["source_ids"] = ["s1"]  # belongs to other document
        elif name == "tool_evidence_check":
            result["claims"][0]["evidence"][0]["quote_id"] = "invented-quote"
        return result

    calls, documents = setup_generation(monkeypatch, catalog, transform=damage)
    with pytest.raises(ValueError):
        list(service.generate_tool("work", tool_id, documents, claims_text=CLAIMS[0]["text"]))
    assert len(calls) == (3 if tool_id == "compare" else 1)


def test_compare_synthesis_copies_citations_only_from_explicitly_selected_findings(catalog, monkeypatch):
    def select_second_document(name, result):
        if name == "tool_compare":
            aspect = result["aspects"][0]
            # The fake deliberately returns document findings in reverse order.
            aspect["synthesis"]["finding_document_ids"] = [aspect["findings"][0]["document_id"]]
        return result

    calls, documents = setup_generation(monkeypatch, catalog, transform=select_second_document)
    result = list(service.generate_tool("work", "compare", documents))[-1]["result"]
    assert result["content"]["aspects"][0]["synthesis"] == {
        "text": "Different measurements.", "source_ids": ["doc2-chunk0"]}
    assert len(calls) == 3


@pytest.mark.parametrize("damage", ["unknown", "duplicate", "old_wire"])
def test_invalid_compare_finding_references_are_not_repaired_or_retried(catalog, monkeypatch, damage):
    def invalidate(name, result):
        if name == "tool_compare":
            synthesis = result["aspects"][0]["synthesis"]
            if damage == "unknown":
                synthesis["finding_document_ids"] = ["unknown-document"]
            elif damage == "duplicate":
                synthesis["finding_document_ids"] *= 2
            else:
                del synthesis["finding_document_ids"]
                synthesis["source_ids"] = ["s1"]
        return result

    calls, documents = setup_generation(monkeypatch, catalog, transform=invalidate)
    with pytest.raises(ValueError):
        list(service.generate_tool("work", "compare", documents))
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", "tool_compare"]


@pytest.mark.parametrize("tool_id", ["compare", "evidence_check"])
def test_budget_failure_precedes_model_requests(catalog, monkeypatch, tool_id):
    calls, documents = setup_generation(monkeypatch, catalog)
    monkeypatch.setattr(service, "MAX_MODEL_CALLS", 2 if tool_id == "compare" else 0)
    with pytest.raises(ValueError, match="too many|request limit"):
        list(service.generate_tool("work", tool_id, documents, claims_text=CLAIMS[0]["text"]))
    assert calls == []


@pytest.fixture
def large_quote_catalog(catalog, tmp_path, monkeypatch):
    store, documents = catalog
    texts = [("This separate background measurement has limited scope and does not report the first study. " * 10)
             + f"Observation {index}." for index in range(12)]
    path = tmp_path / "background.txt"
    path.write_text("\n".join(texts), encoding="utf-8")
    document = store.stage_document(path, "science", store.settings.fingerprint())
    with store._db:
        store._db.executemany("INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
            [(f"background-{index}", document["id"], text, f"section {index}", "text", index)
             for index, text in enumerate(texts)])
        store._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=? WHERE id=?",
                          (len(texts), document["id"]))
    monkeypatch.setitem(config.MODEL_TOKEN_LIMITS, config.DEFAULT_MODEL, 8192)
    return store, [*documents, document["id"]]


def test_evidence_scans_every_original_quote_before_selecting_candidates(large_quote_catalog, monkeypatch):
    calls, documents = setup_generation(monkeypatch, large_quote_catalog)
    result = list(service.generate_tool("work", "evidence_check", documents,
                                        claims_text=CLAIMS[0]["text"]))[-1]["result"]
    scans = [prompt for name, prompt in calls if name == "tool_quote_selection"]
    assert len(scans) > 1
    assert {name for name, _ in calls} == {"tool_quote_selection", "tool_evidence_check"}
    assert calls[-1][0] == "tool_evidence_check"
    assert result["settings"]["model_requests"] == len(scans) + 1
    assert result["coverage"] == {"documents": 3, "chunks": 15, "processed_chunks": 15}
    quotes = [item for prompt in scans for item in prompt["quotes"]]
    assert len({item["quote_id"] for item in quotes}) == len(quotes)
    assert {item["source_id"] for item in quotes} == {f"s{index}" for index in range(1, 16)}
    assert all(prompt["requested_claims"] == CLAIMS for _, prompt in calls)
    originals = {f"s{index}": source["text"] for index, source in enumerate(result["sources"], 1)}
    assert all(item["quote"] in originals[item["source_id"]] for item in quotes)
    assert len(calls[-1][1]["quotes"]) == 1
    assert result["content"]["claims"][0]["evidence"][0]["quote"] == SOURCES[0]["text"]


@pytest.mark.parametrize("change", ["cancel", "snapshot"])
def test_quote_selection_obeys_cancellation_and_snapshot_guards(large_quote_catalog, monkeypatch, change):
    stopped = False

    def invalidate(name):
        nonlocal stopped
        if name == "tool_quote_selection":
            if change == "cancel":
                stopped = True
            else:
                with large_quote_catalog[0]._db:
                    large_quote_catalog[0]._db.execute("UPDATE documents SET generation='new'")

    calls, documents = setup_generation(monkeypatch, large_quote_catalog, after_call=invalidate)
    events = []
    with pytest.raises((service.ToolCancelled, ValueError)):
        for event in service.generate_tool("work", "evidence_check", documents,
                claims_text=CLAIMS[0]["text"], is_cancelled=lambda: stopped):
            events.append(event)
    assert all(event["result"] is None for event in events)
    assert [name for name, _ in calls] == ["tool_quote_selection"]


def test_unknown_selected_quote_fails_without_final_request_or_repair(large_quote_catalog, monkeypatch):
    def damage(name, result):
        if name == "tool_quote_selection":
            result["claims"][0]["evidence"] = [{"quote_id": "invented-quote", "stance": "supports"}]
        return result

    calls, documents = setup_generation(monkeypatch, large_quote_catalog, transform=damage)
    with pytest.raises(ValueError):
        list(service.generate_tool("work", "evidence_check", documents, claims_text=CLAIMS[0]["text"]))
    assert [name for name, _ in calls] == ["tool_quote_selection"]


def test_selected_quote_pool_exceeding_context_fails_without_dropping_candidates(large_quote_catalog, monkeypatch):
    def select_all(name, result):
        assert name == "tool_quote_selection", "An oversized candidate pool must fail before final generation"
        rows = calls[-1][1]["quotes"]
        assert len(rows) <= review.MAX_EVIDENCE_PER_CLAIM
        result["claims"][0]["evidence"] = [
            {"quote_id": row["quote_id"], "stance": "context"} for row in rows]
        return result

    calls, documents = setup_generation(monkeypatch, large_quote_catalog, transform=select_all)
    with pytest.raises(ValueError, match="No quotations were dropped"):
        list(service.generate_tool("work", "evidence_check", documents, claims_text=CLAIMS[0]["text"]))
    assert len(calls) > 1
    assert all(name == "tool_quote_selection" for name, _ in calls)
    assert {quote["source_id"] for _, prompt in calls for quote in prompt["quotes"]} == {
        f"s{index}" for index in range(1, 16)}


def test_quote_scan_request_limit_is_checked_before_any_model_call(large_quote_catalog, monkeypatch):
    calls, documents = setup_generation(monkeypatch, large_quote_catalog)
    monkeypatch.setattr(service, "MAX_MODEL_CALLS", 2)
    with pytest.raises(ValueError, match="too many model requests"):
        list(service.generate_tool("work", "evidence_check", documents, claims_text=CLAIMS[0]["text"]))
    assert calls == []


def test_large_original_chunk_is_sliced_before_evidence_batching(catalog, tmp_path, monkeypatch):
    store, _ = catalog
    original = CLAIMS[0]["text"] + "\n" + "\n".join(
        f"Background observation {index:04d}: retain its scope, qualifications, and original wording."
        for index in range(300))
    path = tmp_path / "long-original.txt"
    path.write_text(original, encoding="utf-8")
    document = store.stage_document(path, "science", store.settings.fingerprint())
    with store._db:
        store._db.execute("INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
                          ("long-original-chunk", document["id"], original, "section 1", "text", 0))
        store._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=1 WHERE id=?",
                          (document["id"],))
    monkeypatch.setitem(config.MODEL_TOKEN_LIMITS, config.DEFAULT_MODEL, 16384)
    # This chunk cannot enter the generic notes pipeline. Evidence Check must
    # obtain the complete snapshot first and slice original quotations itself.
    with pytest.raises(ValueError, match="too large for a tool batch"):
        service.prepare_snapshot(store, [document["id"]], None, config.DEFAULT_MODEL)
    calls, documents = setup_generation(monkeypatch, (store, [document["id"]]))
    result = list(service.generate_tool("work", "evidence_check", documents,
                                        claims_text=CLAIMS[0]["text"]))[-1]["result"]
    scans = [prompt for name, prompt in calls if name == "tool_quote_selection"]
    assert len(scans) > 1
    quotes = [quote for prompt in scans for quote in prompt["quotes"]]
    assert {quote["source_id"] for quote in quotes} == {"s1"}
    covered = [False] * len(original)
    for quote in quotes:
        start = original.index(quote["quote"])
        covered[start:start + len(quote["quote"])] = [True] * len(quote["quote"])
    assert all(covered), "Exact slices must cover every character of the original chunk"
    assert result["coverage"] == {"documents": 1, "chunks": 1, "processed_chunks": 1}
    assert result["content"]["claims"][0]["evidence"] == [
        {"source_id": "long-original-chunk", "quote": quotes[0]["quote"], "stance": "supports"}]


def test_no_selected_quote_candidates_produce_valid_not_found(large_quote_catalog, monkeypatch):
    store, documents = large_quote_catalog
    monkeypatch.setattr(service, "get_service", lambda _: SimpleNamespace(store=store))
    calls = []

    def no_evidence(model, history, schema, **options):
        prompt = json.loads(history[-1]["content"])
        name = options["schema_name"]
        calls.append((name, prompt))
        if name == "tool_quote_selection":
            result = {"claims": [{"id": claim["id"], "evidence": []}
                                  for claim in prompt["requested_claims"]]}
        else:
            assert name == "tool_evidence_check"
            assert prompt["quotes"] == []
            result = {"title": "No relevant evidence", "claims": [
                {"id": claim["id"], "verdict": "not_found", "rationale": "No relevant evidence was selected.",
                 "evidence": []} for claim in prompt["requested_claims"]]}
        jsonschema.Draft202012Validator(schema).validate(result)
        return result

    monkeypatch.setattr(service.api, "generate_structured", no_evidence)
    result = list(service.generate_tool("work", "evidence_check", documents,
                                        claims_text=CLAIMS[0]["text"]))[-1]["result"]
    assert sum(name == "tool_quote_selection" for name, _ in calls) > 1
    assert calls[-1][0] == "tool_evidence_check"
    assert result["content"]["claims"] == [{**CLAIMS[0], "verdict": "not_found",
        "rationale": "No relevant evidence was selected.", "evidence": []}]
    assert result["coverage"]["processed_chunks"] == 15


def test_single_quote_that_cannot_fit_complete_request_fails_before_model_calls(catalog, monkeypatch):
    monkeypatch.setitem(config.MODEL_TOKEN_LIMITS, config.DEFAULT_MODEL, 8192)
    calls, documents = setup_generation(monkeypatch, catalog)
    actual_size = service.api.structured_request_tokens
    sized_requests = []

    def record_size(model, history, schema, system_prompt=""):
        size = actual_size(model, history, schema, system_prompt)
        sized_requests.append((json.loads(history[-1]["content"]), size))
        return size

    monkeypatch.setattr(service.api, "structured_request_tokens", record_size)
    instructions = "Review only selected source excerpts. " * 175
    with pytest.raises(ValueError, match="one original excerpt exceed"):
        list(service.generate_tool("work", "evidence_check", documents, instructions,
                                   claims_text=CLAIMS[0]["text"]))
    assert calls == []
    single_requests = [(prompt, size) for prompt, size in sized_requests if len(prompt["quotes"]) == 1]
    assert single_requests
    assert all(size + 8192 // 4 > 8192 for _, size in single_requests)
    assert all(service.token_size(prompt["quotes"], config.DEFAULT_MODEL) < 5120
               for prompt, _ in single_requests)
