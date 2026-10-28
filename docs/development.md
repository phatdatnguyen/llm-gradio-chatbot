# Development and maintenance

[Back to README](../Readme.md)

## Project layout

```text
webui.py                 Connections, Chat, RAG Chat, and Plugins shell
plugins/                 Bundled tool catalog, enable/disable settings, manager UI
chatbot/
  chatbot.py             Chat layout and callbacks
  api.py, config.py       Shared local calls, discovery, and capability settings
  providers/             LM Studio/Ollama/textgen/KoboldCpp adapters, schema grammar, bounded transport, provider errors
  connections.py         Manual Connect and per-session connection lifecycle
  messages.py, tokens.py  Message conversion and context budgeting
  readers.py             Chat attachment readers
  history_store.py        Chat persistence
  log.py                 Shared logging
  history/               Existing chats and image assets (ignored)
rag_chatbot/
  chatbot.py             RAG layout and callbacks
  service.py             Indexing, retrieval, evidence assembly
  ingestion.py           Document/image extraction and chunking
  viewer.py              On-demand PDF, image, Office, and spreadsheet previews
  office_preview.py      Local Office-to-PDF conversion and preview cache
  store.py               SQLite catalog and Qdrant hybrid retrieval
  history.py             RAG conversation persistence
  workspaces.py          Working-directory creation, validation, and settings
  config.py, schemas.py   Settings and data types
  tools/                 Source snapshots, citation schemas, generation, jobs/UI, exports
  data/                  Working directories (ignored)
    Default/
      sources/           Managed originals, derived images, extraction caches
      chats/             Saved RAG conversations and citations
      artifacts/         Saved tool outputs, provenance, and exports
      library.sqlite3    This directory's document catalog and text index
docs/                    RAG, Studio, and development guides
compose.yaml             Local Qdrant server
start_webui.bat          Windows launcher
start_webui.sh           macOS/Linux launcher
requirements.txt        Shared runtime dependencies with Windows-only markers
requirements-dev.txt    Test dependencies
tests/                   Offline unit and integration tests
```

Existing chat JSON and relative image references need no conversion when moving
`history/` to `chatbot/history/`. The loader also recognizes the legacy root directory
when the new one does not exist. Back up `chatbot/history/`, the entire RAG data
directory, and the Qdrant embedded cache or Docker volume together. Stop indexing/app writes while
creating a consistent backup.

## Running the tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary libraries, local Qdrant, fake provider responses, and HTTP
mock transports. They do not need local inference servers or send inference requests. Token and
chunk budgets use a conservative UTF-8 byte estimate and need no downloaded assets.
Wire tests cover discovery, SSE/NDJSON cancellation, schema validation, embedding ordering,
local endpoint restrictions, and errors without cloud fallback.

Keep the suite focused on observable behavior and past bugs. Use representative
valid, invalid, and boundary cases for shared validators rather than repeating
every malformed value across every tool and format. Retain coverage of citation
integrity, workspace isolation, cancellation, atomic writes, path safety, and
model request contracts. Avoid tests that only repeat constants, snapshot cosmetic
UI details, or duplicate another test of the same behavior.

Run `python tests/manual_rag_smoke.py` for an optional localhost Gradio HTTP/RPC
check of upload, indexing, answers, citation previews, and saved conversations.
It uses temporary storage and fake models, so it makes no inference-server requests.
Run `python tests/manual_tools_smoke.py` for the same kind of check of tool
generation, edits, source references, saved outputs, exports, and Plugins controls,
including slide navigation and PPTX export, flashcard reveal/navigation, and local
quiz scoring/restart after workspace switches.

To run the optional real converter check on macOS/Linux with LibreOffice installed:

```sh
RUN_LIBREOFFICE_SMOKE=1 python -m pytest -q tests/test_libreoffice_integration.py
```

For an explicit test against your running LM Studio server, use:

```sh
python tests/manual_lmstudio_smoke.py
```

This checks chat, streaming, embeddings, RAG citations, and Summary save/restore/export
using synthetic text in disposable storage. It requires a loaded chat model and an
installed embedding model, sends local inference requests, and does not read your
saved chats or source library. It does not download models.

For Ollama with installed chat and embedding models, run
`python tests/manual_ollama_smoke.py`. It checks the same flow, plus vision when
the selected model advertises that capability. Its request context defaults to
16,384 for the test process unless `OLLAMA_CONTEXT_LENGTH` is already set.

For textgen with a loaded supported llama.cpp model and an already installed Ollama
embedding model, run `python tests/manual_textgen_smoke.py`. It explicitly selects
Ollama embeddings and checks chat, streaming, RAG, Summary persistence/export,
and exact quiz/slide counts in temporary storage.
It leaves `TEXTGEN_CONTEXT_LENGTH` unchanged; the normal 4,096 fallback applies
unless that variable was explicitly configured to fit the loaded server context.
Set it to the actual loaded window before running this check; the small fallback
may not fit the Studio schemas and their output reserve.

For KoboldCpp 1.123.1 or newer with a loaded model and an installed Ollama embedding model,
run `python tests/manual_koboldcpp_smoke.py`. It selects Ollama embeddings explicitly,
uses the discovered loaded context, and checks chat, streaming, RAG citations,
Summary persistence/export, and exact quiz/slide counts in disposable storage.

Local verification does not establish that every platform has passed. The optional
LibreOffice and filesystem-link tests skip when their prerequisites are unavailable.

## Local provider boundaries

`chatbot.api` remains the shared facade for chat, streaming, embeddings, vision,
and structured output. LM Studio, Ollama, textgen, and KoboldCpp adapters use `httpx` directly. There is
no OpenAI SDK, key-file import, cloud fallback, web search, URL fetcher, or image
generation.

The UI starts disconnected. Only Connect creates a provider, discovers models, and
checks embeddings. `config.Connection` freezes provider settings, metadata, budgets,
and embedding identity. Each browser session owns a connection, and callbacks bind
it for every generator step. Reconnect revokes previous work; selector changes alone
make no requests. Another session's connection must never replace that binding.

Adapters accept loopback/private LAN IPs, disable proxies/redirects, and never retry.
LM Studio discovery uses loaded-instance context, with conservative legacy fallback.
Ollama discovers installed files using tags/show and loaded windows using ps; no
model is loaded during discovery. Ollama capability metadata determines usable
models. `prepare_chat_payload` binds the effective context for native `num_ctx`;
`max_tokens` becomes `num_predict`, schema becomes `format`, and images become local
base64. Disable native truncation/context shifting. Embeddings use `truncate:false`.
Ollama calls force `:local` routing, filter cloud/proxy models before sending content,
and reject changed digests until reconnect. No model installation endpoint is called
by app code. See [local connections](local-connections.md) for settings.

Textgen reads `/openapi.json` and `/v1/internal/model/info` to expose only the loaded
model and supported API controls. Its request `model` field is not a model switch: guard loaded model, loader,
and LoRA metadata before inference and reject stale selections. Before chat, request
the server-rendered prompt through `/v1/internal/chat-prompt` and check that every
complete message survives; textgen otherwise silently trims history or user input.
Render with a large non-generation window, then use native token counting to check
the complete prompt against the effective context. Inference receives that context
as `truncation_length`; only an ordinary answer's reserve may decrease if native
framing needs more room. A structured tool's explicit reserve must fit unchanged.
Loaded context is unknown unless independently supplied by metadata. The textgen-only
**Textgen context length** Number field initially uses `TEXTGEN_CONTEXT_LENGTH`
or 4,096. Connect passes its validated integer value as `context_length` into the
immutable session snapshot; edits alone make no requests or alter an active
connection. CLI callers can pass the same override to `config.connect()`, and
`refresh_models()` preserves it. Require 512–1,048,576 tokens and reject invalid
values before discovery. This request budget must fit the actual server window;
it does not reload or resize the model. Do not infer vision from the model name or
loader; this connector currently leaves it unavailable.

The textgen llama.cpp path uses app-owned schema-to-GBNF conversion and native
`grammar_string`, not an ignored `response_format` field. Preserve request-scoped
source-ID enums, reject unsupported schema features before generation, and keep local
schema/domain validation. No generic JSON grammar fallback is allowed.
Native textgen embeddings ignore the requested model and can download on first use,
so never call that endpoint. The embedding selector for textgen and KoboldCpp defaults to None;
an explicit LM Studio/Ollama choice creates and binds that embedding connection as
part of the session snapshot. Retain its fingerprint unchanged so compatible vectors
do not depend on which chat server is selected.

KoboldCpp reads `/api/extra/version`, `/api/v1/model`, and
`/api/extra/true_max_context_length` to discover its currently loaded model and
allocated window. The advertised model maximum remains unknown. Pin and recheck
metadata before inference; an explicit `KOBOLDCPP_CONTEXT_LENGTH` only lowers the
loaded window. Reject server routing and generation overrides that could replace
prepared fields or invalidate preflight. No model-management endpoint is called.
Before inference, `/api/extra/tokencount` renders and counts the complete messages;
verify their ordered text and native count against the immutable request budget.
Convert supported schemas to native `grammar`, retaining citation enums and local
schema/domain validation. Use `/v1/chat/completions` SSE for both public modes,
collecting that stream for blocking replies so reasoning remains separate from
answer content. Do not infer completion from a closed connection without a valid
terminal event. Native output limits are 80% of context when the prompt uses less
than 25% of context, and 65% otherwise. Apply these limits to ordinary replies;
reject a structured reserve that cannot fit unchanged. No fixed app output ceiling
or evidence truncation is introduced. This adapter currently accepts text only and
uses the explicitly selected embedding connection for RAG.

Connections reports loaded context, advertised maximum, and effective request budget
separately. LM Studio's explicit context setting can only lower the loaded window;
Ollama's can request a different native window up to the model maximum. Unloaded
Ollama models use a conservative fallback unless explicitly configured. Never label
an advertised maximum as a loaded context. Local schema validation
and domain/citation validators remain mandatory even when the server claims JSON-schema support. Completion truncation
must not publish a partial structured artifact or index partial vision extraction.
Ordinary chat accepts an output-limit stop only when visible answer text exists,
appending an app-owned notice. RAG then validates citations and retains the answer
and source snapshot normally. No-answer length stops remain actionable failures;
reasoning metadata is not displayed as an answer. No automatic retry occurs.
Closing a streamed generator closes its HTTP response. All UI failures use the
shared provider exception contract. Network I/O waits default to 900 seconds
(`LMSTUDIO_TIMEOUT` / `OLLAMA_TIMEOUT` / `TEXTGEN_TIMEOUT` / `KOBOLDCPP_TIMEOUT`); connection attempts are capped at 5 seconds. This is an
I/O timeout rather than a total generation deadline; active streaming can run longer.

Ordinary Chat/RAG output limits are computed per request from the remaining
effective context after message, system-prompt, attachment, and framing estimates.
There is no fixed 2,048/4,096 output ceiling. `LMSTUDIO_MAX_OUTPUT_TOKENS`, `OLLAMA_MAX_OUTPUT_TOKENS`, `TEXTGEN_MAX_OUTPUT_TOKENS`, or `KOBOLDCPP_MAX_OUTPUT_TOKENS`, when
positive, is an optional upper bound; unset or `0` means automatic. Preparation
reserves at least a quarter of the effective context for an answer, subject to that
optional cap. This planning reserve is not a generation ceiling: the final request
receives all remaining estimated context. Keep trimming and request budgets
consistent. Studio schema generation budgets evidence against the remaining effective
context rather than a fixed 6,000-token input ceiling. Its output reserve is the
smaller of 32,768 tokens and a quarter of context, independent of the ordinary-chat
cap. Include actual task text, scoped schema, citation/document metadata, message
framing, and output space when computing both evidence batches and the final
consolidation target. Vision extraction uses the
chat request path but requires complete output; truncation must fail indexing.

For source-ID generation, `rag_chatbot/tools/citations.py` deep-copies request
schemas and shares the
allowed identifier enum through `$defs/source_identifier`. Collect identifiers
only from app-owned fields in that request's evidence; include the exact strings
in `allowed_source_ids` in the prompt. Require nonempty citations where the domain
requires them, and preserve valid empty-citation exceptions. Local schema and
domain validation still reject duplicates, unknown IDs, and incorrect ownership;
a server grammar may ignore `uniqueItems`. Budget the expanded schema before
generation. Errors identify the tool and extraction, consolidation, or final-output
stage, with consolidation round/batch numbers, without exposing raw model passages.
Do not retry or weaken validation after a failure.

`tools/contracts.py` mirrors Flashcards, Quizzes, Presentation, and Compare validation bounds in
generation schemas without changing persisted formats. Presentation constrains
exact slide count and count-dependent bullet lengths within the existing total.
Learning schemas require the selected item count; quizzes additionally require
four options and an integer answer index from 0 to 3. Saved older study sets do not
need to match the current slider, and generation never pads or truncates a result.
Compare constrains aspect count, document findings, status, and per-document source
ownership. Synthesis selects `finding_document_ids`; the app resolves those to the
selected findings' validated citation lists before applying canonical validation.

Evidence Check uses `tools/quote_review.py` and `tools/quotations.py` instead of
lossy evidence notes. Build an ordered catalog of exact original substrings, with
app-owned quote IDs and overlapping bounded spans for long passages. Send the full
catalog when it fits; otherwise scan every source batch and pool every selected
candidate, including supporting, contradicting, and context excerpts. Final
assessment chooses quote IDs and stances; the app copies the original quote/source
strings, then validates canonical claims, source occurrence, and verdict/stance
consistency. Reject unknown or duplicate IDs. No model-written quotations, fuzzy
matching, hidden repair calls, or silent candidate trimming. Preflight all batches
and the final candidate pool, retain snapshot/cancellation guards, and leave saved
artifact and export formats unchanged.

Direct-table Plot represents Auto separately from an explicit chart type. The
model chooses a supported type from the instructions and may create a neutral
title when none is supplied; coordinates still come directly from the local table.

`STUDIO_MAX_INPUT_TOKENS` defaults to `0` (automatic); positive values up to
1,048,576 impose a lower evidence-input cap. Consolidation defaults to 20 rounds
(`STUDIO_MAX_CONSOLIDATION_ROUNDS`, range 1-100), with at most 256 model calls
(`STUDIO_MAX_MODEL_CALLS`, range 1-10,000). Settings are read at process startup.
Larger limits still require measurable reduction each round, complete citation
coverage, current source snapshots, and cancellation checks. Never drop evidence
or continue a nonprogressing loop to fit a request.

Both chat components support `$...$`, `$$...$$`, `\(...\)`, and `\[...\]` math
delimiters. Preserve the existing export behavior for equation text.

A finite model context still limits output, and thinking uses the generation
allowance before visible text. Refresh model metadata after changing loaded context;
restart for environment changes. Never use automatic continuation to disguise
truncation or publish an incomplete structured result.

The default database is embedded Qdrant; `QDRANT_URL` selects an explicit loopback
server unless `QDRANT_LOCAL_PATH` overrides it. Local fingerprints use `local-v2`;
never reuse old cloud vectors. Back up original data and reindex it locally.
Embedding identity, prefixes, dimensions, context and chunk/preprocessing settings
belong in compatibility checks. Preserve source/history/artifact files, including
old Graphics outputs, when removing unused generation controls.

The vendored [llama.cpp converter](../chatbot/providers/_llama_schema.py) records
its upstream tag, source hash, modifications, and complete MIT notice. Preserve
that notice separately from the [project license](../LICENSE.txt).

Implementation references: [Ollama chat](https://docs.ollama.com/api/chat),
[Ollama embeddings](https://docs.ollama.com/api/embed),
[Ollama model metadata](https://docs.ollama.com/api-reference/show-model-details),
[LM Studio model discovery](https://lmstudio.ai/docs/developer/rest/list),
[LM Studio structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output),
[LM Studio embeddings](https://lmstudio.ai/docs/developer/openai-compat/embeddings),
[KoboldCpp API implementation](https://github.com/LostRuins/koboldcpp/blob/concedo/koboldcpp.py),
[KoboldCpp native generation](https://github.com/LostRuins/koboldcpp/blob/concedo/kcpp_src/gpttype_adapter.cpp),
[llama.cpp schema constraints](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md),
[Qdrant local setup](https://qdrant.tech/documentation/quickstart/),
[LangChain recursive splitting](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter),
[python-pptx speaker notes](https://python-pptx.readthedocs.io/en/latest/user/notes.html).
