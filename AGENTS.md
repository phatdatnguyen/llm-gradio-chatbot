# AGENTS.md

This file provides guidance when working with code in this repository.

The active implementation supports local LM Studio, Ollama, textgen, and KoboldCpp. This guidance supersedes the
former OpenAI routing, cloud model-name tables, web-search, and Graphics generation
requirements. Preserve all source, history, citation, artifact, cancellation, and
workspace isolation invariants below.

## Setup and Running

Use Python 3.10+ on Windows, macOS, or Linux.

```sh
# Create the environment (use python3 on macOS/Linux before activation)
python -m venv chatbot-env
```

Activate it with `.\chatbot-env\Scripts\Activate.ps1` in Windows PowerShell,
`chatbot-env\Scripts\activate.bat` in Command Prompt, or
`. chatbot-env/bin/activate` on macOS/Linux. Activation is optional if commands use
the environment's Python executable directly.

```sh
# Install dependencies
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip check

# Run the app
python webui.py

# Run the tests
python -m pytest -q
```

Start LM Studio's local server with a loaded chat model, or start Ollama with an
installed local chat model. Defaults are `http://127.0.0.1:1234/v1` and
`http://127.0.0.1:11434`; override with `LMSTUDIO_BASE_URL` or `OLLAMA_BASE_URL`.
Textgen requires a loaded model and its API, default `http://127.0.0.1:5000/v1`,
overridden with `TEXTGEN_BASE_URL`. KoboldCpp uses its loaded model at
`http://127.0.0.1:5001/v1`, overridden with `KOBOLDCPP_BASE_URL`.
Optional local authentication uses the matching `*_API_KEY`. The app does not use
the OpenAI SDK or read `OPENAI_API_KEY` or Python key files. Preserve ignored legacy
credential files; never commit them.

Launch with `start_webui.bat` on Windows or `sh start_webui.sh` on macOS/Linux.
Both use the local `chatbot-env` interpreter and resolve the project directory.
The UI starts disconnected. Opening it or selecting a connector must not construct
a provider, discover models, or make inference requests. Only **Connect** contacts
the selected server and probes an installed embedding candidate when configured. The probe may
load the installed model but must never download it. A sole embedding candidate
is selected automatically; multiple candidates need explicit
`LMSTUDIO_EMBEDDING_MODEL` or `OLLAMA_EMBEDDING_MODEL`.
For textgen and KoboldCpp, RAG embeddings are explicitly selected separately as None, Ollama, or
LM Studio; default None. Never call textgen's native embedding endpoint: it ignores
the requested model and can download weights on first use. Contact an embedding
server only after that explicit selection and Connect. Preserve that embedding
server's identity/fingerprint instead of tying vectors to the chat connector.
KoboldCpp embeddings are also unavailable through this connector; use the explicit
Ollama or LM Studio selection for RAG.

Keep immutable connection snapshots per browser session. Bind the same snapshot
for every callback/generator step, including thread changes; never let reconnecting
redirect an already started request. Reject revoked, stale, and disconnected work.
Changing the connector radio alone must not switch active connections. A failed
Connect leaves that session disconnected, without affecting another session.
Endpoint/environment settings belong to the process and require restart to change.
The textgen-only **Textgen context length** field is a session override, initially
`TEXTGEN_CONTEXT_LENGTH` or 4,096. Editing it must make no requests or change an
active snapshot; Connect applies a validated whole-number budget of 512–1,048,576.
CLI `connect(..., context_length=...)` uses the same override, and refresh preserves
it. It does not resize the loaded model and must fit the server's actual window.
See `docs/local-connections.md` for setup and provider-specific settings.

Embedded Qdrant is the default unless `QDRANT_URL` explicitly selects a loopback
server; `QDRANT_LOCAL_PATH` overrides the embedded cache location. Never run multiple
app processes against the same embedded path. For server mode, start Docker and
run `docker compose up -d`, then set `QDRANT_URL=http://127.0.0.1:6333`.
RAG indexing checks Qdrant before local embeddings. Tests inject providers and local
Qdrant and never require running inference servers.

Gradio is pinned at 6.30.0; older 6.13.0 freezes on Dataframe updates. Keep runtime
requirements platform-neutral and guard `pywin32` with its Windows marker. Native
document readers do not require an Office installation. Optional Office PDF previews
use Word/PowerPoint on Windows and LibreOffice on macOS/Linux; the latter can be
configured with `LIBREOFFICE_PATH`. All converter imports/processes stay lazy.
See Readme.md for installation and fallback behavior.

Keep Python 3.10+ and Windows/macOS/Linux compatibility. Run
`RUN_LIBREOFFICE_SMOKE=1 python -m pytest -q tests/test_libreoffice_integration.py`
on macOS/Linux to opt into the real converter check; ordinary tests mock converters.

## Architecture

| Module | Role |
|---|---|
| `webui.py` | Root Connections, Chat, RAG Chat, and Plugins Gradio shell. |
| `chatbot/chatbot.py` | Chat layout, callbacks, and `build_tab()`. |
| `chatbot/config.py` | Immutable connection snapshots, callback binding, discovered models, budgets, and provider factory. |
| `chatbot/api.py` | Shared local chat/streaming/embeddings/vision/structured facade; strict response validation. |
| `chatbot/providers/` | LM Studio, native Ollama, textgen, and KoboldCpp adapters, bounded transport, schema grammar, and `ProviderError`; no SDK or cloud fallback. |
| `chatbot/connections.py` | Manual Connect, session-owned connection lifecycle, embedding check, and status UI. |
| `chatbot/messages.py` | History ↔ API payload conversion, image encoding, display formatting. |
| `chatbot/readers.py` | Chat attachment text extraction from PDF/DOCX/XLSX/PPTX/HTML/TXT. |
| `chatbot/tokens.py` | Conservative offline UTF-8 budgeting and complete-turn `trim_history()`. |
| `chatbot/history_store.py` | Save/load/delete Chat history, with image externalization. |
| `chatbot/log.py` | Shared `logger`. Use this, not `print()`. |
| `rag_chatbot/chatbot.py` | Independent RAG tab, library and citation UI. |
| `rag_chatbot/service.py` | Indexing, cached extraction/embedding batches, retrieval and evidence assembly. |
| `rag_chatbot/ingestion.py` | Source-aware readers, opt-in vision callbacks, token chunks; no API calls. |
| `rag_chatbot/viewer.py` | Bounded on-demand document previews, scoped to workspace sources; no API calls. |
| `rag_chatbot/office_preview.py` | Optional local Office-to-PDF conversion and cached previews; originals stay unchanged. |
| `rag_chatbot/store.py` | SQLite catalog/FTS5, Qdrant vectors, scoped hybrid search. |
| `rag_chatbot/history.py` | Separate versioned RAG histories with citation snapshots. |
| `rag_chatbot/workspaces.py` | Safe named working directories under RAG_DATA_DIR, per-directory settings and vector namespaces. |
| `rag_chatbot/config.py` | RAG settings and index fingerprint. |
| `rag_chatbot/tools/` | Selected-source snapshots, bounded Studio jobs including Compare/Evidence Check, session UI, artifacts and exports. |
| `plugins/` | Trusted bundled tool catalog, local enable/disable settings, dependency status and Plugins tab. |
| `tests/` | Pure logic, mocked API, local storage, UI wiring, and optional conversion/RPC tests. |

Both tabs obtain their connected default through `chatbot.config.get_default_model()`,
selected from that connection's discovered models or provider-specific `*_MODEL`.
Before Connect, model choices remain empty; never use another session's defaults. RAG selected
references remain directly below Open reference in the latest-answer accordion.

### Model discovery and local routing

Keep model discovery and capability metadata centralized. For LM Studio, prefer
native `/api/v1/models` loaded-instance metadata, with `/v1/models` fallback when
the native endpoint is absent. Do not infer capabilities from cloud model-name prefixes. Use
loaded context rather than advertised maximum context. Connections must distinguish
loaded, advertised maximum, and effective app context, including any environment
cap. Changing LM Studio's default requires reloading an existing model and then
refreshing discovery. Never hardcode a particular model's context. Fallbacks must remain
conservative and allow explicit configuration. Refreshing one session must not
silently change another session's model. Reject stale embedding probe results and
stale model selections. Metadata readers and refreshes must use the shared lock.

`config.get_provider()` returns the active callback's immutable connection provider
and fails while disconnected. LM Studio generation uses `/v1/chat/completions` and
embeddings use `/v1/embeddings`. Ollama uses `/api/tags`, `/api/ps`, and `/api/show`
for read-only discovery; completion/vision/embedding capabilities come from metadata,
never model-name prefixes. Installed local completion models remain selectable when
unloaded. Report their loaded context as unknown and use a conservative request
budget unless explicitly configured. `OLLAMA_CONTEXT_LENGTH` sets native `num_ctx`
per request, bounded by the advertised maximum; LM Studio's explicit context setting
remains a lower cap on its loaded window.

Textgen discovers its API controls through `/openapi.json` and only its currently
loaded model through `/v1/internal/model/info`. The chat request's model field does not switch models;
check the loaded model, loader, and LoRAs before requests and reject stale selections.
Never call model-load/unload or global stop endpoints. Use `/v1/chat/completions`
with bounded SSE and the shared response contract. Its API does not expose reliable
loaded-context or vision metadata: report context unknown, use the session's
**Textgen context length** override or `TEXTGEN_CONTEXT_LENGTH`/4,096 default, and
leave vision unavailable. The explicit context is a request budget, not a model
reload or advertised capacity.
Preflight the complete server-rendered prompt through `/v1/internal/chat-prompt`
and reject omitted/truncated message text before generation. This endpoint must
not perform generation. Preserve the app's conservative budgeting and output limits.
Textgen's `response_format` is not a schema guarantee. For the supported llama.cpp
loader, translate supported JSON schemas to bounded app-owned GBNF for its native
`grammar_string`; preserve citation enums and keep all local schema/domain validators.
Reject unsupported schema features/loaders before generation, with no generic-JSON
or weaker-format fallback. Keep all textgen wire handling within its provider.

Require KoboldCpp 1.123.1 or newer. It discovers its current model through `/api/v1/model`, server capabilities
through `/api/extra/version`, and allocated context through
`/api/extra/true_max_context_length`. Keep advertised maximum unknown; do not label
the loaded allocation as the model maximum. `KOBOLDCPP_CONTEXT_LENGTH` is an
optional lower cap. Recheck the bound model/context before inference; never load,
unload, switch, or download a model. Reject server routing or generation overrides
that could invalidate the local request or its preflight.
Preflight the complete ordered messages with `/api/extra/tokencount`, checking
both returned prompt and native token count. Send `/v1/chat/completions` with
schema-specific native `grammar`, preserving source enums and all local validators.
Use bounded SSE for streaming and collect the same stream for blocking replies,
keeping reasoning separate from visible answers and preserving finish validation.
KoboldCpp has a native output maximum based on the context and prompt size; respect
that limit without truncating input or adding a fixed app output ceiling. A
structured output reserve must fit unchanged. The initial adapter is text-only;
use the explicitly selected Ollama/LM Studio connection for RAG embeddings.

The shared facade calls `prepare_chat_payload` when the provider implements it.
Ollama maps the bound context/output budget to native `num_ctx`/`num_predict`, schema
to `format`, and local image data to base64 `images`. Disable native truncation and
context shifting. Native `/api/embed` must use `truncate:false`; validate dimensions,
finite nonzero vectors, and complete batch length. Normalize native NDJSON into the
shared response/delta contract without exposing thinking as answer text.

Force Ollama's `:local` source selector for show/chat/embed while keeping canonical
model IDs in UI/config. Reject remote metadata and explicit cloud-routing suffixes
before metadata or inference requests. Require a compatible Ollama version; never
retry without the local-routing guard. Pin discovered model digests to the connection
and reject tag replacement until reconnecting, protecting embedding fingerprints.
The adapters accept only loopback or explicit private LAN IPs, disable environment
proxies and redirects, bound responses/events, and never silently retry requests.
Keep provider-specific wire formats out of UI, ingestion, history, and tool modules.

Web search, external URL fetching, cloud Responses calls, and image generation/editing
are unavailable. Configure model-native thinking in the inference server; do not transmit old
cloud reasoning-effort values. Preserve ordinary sampling controls. Normal chat may
trim complete old exchanges; structured tools must preflight the entire source,
system/schema overhead, and output reserve without silently trimming evidence.
Validate JSON locally against its schema and retain domain/citation validators.
Refusals, invalid JSON, duplicate keys, nonfinite numbers, and incomplete structured
or vision-extraction replies are failures, with no weaker-format retry. Ordinary
Chat/RAG replies that reach the output limit with visible answer text retain that
partial text with an app-owned truncation notice; finish citation validation and
preserve the resulting history/source snapshots. A length stop without visible
answer text is an actionable failure, never a reason to expose thinking metadata.
Do not retry or continue automatically. Closing a stream must close its HTTP
response. Do not introduce model-download or installation side effects.

Ordinary Chat/RAG replies use the remaining effective context after the prepared
request, system prompt, attachments, and framing are budgeted. Do not impose a
fixed 2,048/4,096 output ceiling. `LMSTUDIO_MAX_OUTPUT_TOKENS` or
`OLLAMA_MAX_OUTPUT_TOKENS` / `TEXTGEN_MAX_OUTPUT_TOKENS` / `KOBOLDCPP_MAX_OUTPUT_TOKENS` is an optional user cap for its provider; unset or `0` means automatic maximum. The quarter-context planning reserve
is a minimum for preparing history/evidence, subject to that cap, not a generation
ceiling. The final request uses all remaining estimated context within native provider limits. Keep trimming and
request output limits consistent. Studio structured generation retains explicit
per-tool processing/output budgets independently of the ordinary-chat cap. Both
structured generation and vision extraction must finish completely. Thinking
consumes the same finite generation allowance; when
no answer appears, suggest reducing thinking/input or increasing effective context.
Connect again after changing model context. Environment changes need an app
restart. No automatic continuation or hidden retries.

Token and chunk budgets use a conservative UTF-8 byte estimate with framing/image
reserves, not the model's exact tokenizer. No tokenizer assets are downloaded.
Embedding identity, dimensions, prefixes, context, chunk settings, and preprocessing
are part of the `local-v2` fingerprint; old OpenAI vectors require reindexing even
when their dimensions match. Keep compatible local caches reusable.

### Message / history flow

1. Gradio hands `on_user_input` the text plus any local image/document.
2. `api.build_user_message()` combines them into one user turn: documents use
   `<<<DOCUMENT_CONTENT>>>` blocks and images use a data URL on `image_url`. Legacy
   saved link blocks remain readable, but new URL fetching is rejected.
3. `tokens.trim_history()` drops the oldest turns to fit the model's context, charging the
   system prompt and reserving room for output. It raises `InputTooLargeError` rather than
   sending an empty payload.
4. `api.stream_chat()` (or `send_chat()`) runs the request.
5. `messages.replace_history_content()` builds the display copy, collapsing the bulky
   inlined blocks back to `[File: …]` / `[URL: …]` labels.

Both Chat and RAG Chat support `$...$` and `\(...\)` inline math, plus `$$...$$`
and `\[...\]` display math. Keep the delimiters aligned between the tabs.

`on_user_input` is a generator, so streamed replies render token by token. On failure it
rolls the pending user turn back out of history so it is not silently resent.

### Chat history on disk

Saved to `chatbot/history/*.json` (gitignored). Images are written to
`chatbot/history/assets/<name>/` and referenced by relative path to keep JSON small.
Loading still accepts inline base64, so pre-existing saved chats need no conversion.

RAG working directories live under `rag_chatbot/data/<name>/` by default (override
the parent with `RAG_DATA_DIR`). Each has `sources/`, `chats/`, and `library.sqlite3`.
Every RAG UI callback must receive the active working directory explicitly; never
store a process-global active directory. Services are cached per directory, while
conversation, reference, and active-directory state belongs to the Gradio session.
Opening a different directory cancels pending work and resets session content. Late
results from old jobs must not overwrite the new directory's UI. Qdrant namespaces
and embedded database paths must also be separate for each working directory. The
SQLite catalog is authoritative: only `ready` documents with the current fingerprint
and published vector generation are searchable. Failed uploads and deleted documents
must never appear in retrieval, even if vector cleanup fails. Originals are retained
for saved citations. Do not mix conversation state or persistence between tabs.
Identical reuploads repair missing/corrupt managed originals atomically after a hash
check; preserve valid vectors and extraction/embedding caches. A ready catalog row
without a published generation is not searchable and must be republished on retry.
Changing embedding/chunk settings requires reindexing; never reuse incompatible vectors.
Reindex selected uses the checked catalog entries' hash-verified saved originals,
keeping document IDs, names, collections, original paths, and saved citations intact.
Preflight the entire selection before model work; reject stale/deleted selections,
unsafe or damaged originals, and conflicting current-profile copies. Reindexing
shares ingestion, cancellation, and publication guards. Scope chunk caches by index
fingerprint while retaining compatible legacy caches. Stop indexing stops both upload
and reindex jobs; late results must not overwrite another workspace or connection.
The Workspace dropdown opens on user `input`, not programmatic `change`, to avoid
reopen loops. Disable selection during the switch; stale opens must not publish
state or finish a newer pending switch.
Gradio dropdown `input` also fires on unchanged blur. Dynamic document, sheet,
history, and reference selections can arrive after their choices have changed.
Primitive-only read callbacks bypass component preprocessing and validate against
current session/catalog membership before reading files; File/Dataframe uploads
and model/export controls retain normal preprocessing. Avoid redundant dropdown
inputs in file callbacks when canonical session state already owns the selection.
Unchanged viewer selections must preserve the current page/sheet. Saved citation
downloads must resolve inside the workspace's sources directory, never its catalog,
chats, or artifacts. History lists must exclude atomic temporary files and links.

RAG chat shares local streaming and sampling with the API layer. Empty source
selection makes no answer request. Preserve the legacy web-search provenance flag
in saved histories and exclude web-derived assistant context from local document
answers. Nonstreaming requests use the blocking shared sender. Model thinking is
configured in the inference server, without a cloud reasoning-effort dropdown.

Document preview selection is separate from the retrieval filter. Load previews only
on demand, reset them when the working directory changes, and discard late results.
The viewer and ingestion must share `ingestion.PDFIUM_LOCK` for all PDFium operations.
Office previews run locally with a bounded conversion worker; never close a user's
existing Office application or execute document macros to produce a preview.
LibreOffice must use a disposable profile with macros/active content disabled; only
terminate processes owned by that conversion. Keep preview and indexing support
distinct: legacy DOC/PPT previews do not imply support for indexing those formats.

Source checkboxes define the shared selection for chat and future tools. An empty
selection means no sources; never convert it into the service's `None` (all sources).
Keep selections keyed by document ID, preserve unchecked sources across refresh/indexing,
and reject stale updates after a workspace switch. Editable upload filenames are validated
library metadata; never rename the user's upload or move stable managed originals.

### Studio tools

Summary, Spreadsheet, Presentation, Plot, Mind Map, Flashcards, Quizzes, Compare,
and Evidence Check are implemented. Graphics generation and Chat image generation
are removed. Audio/video are outside this release. Tools share source checkboxes
and the local Prompt model, but never reuse or mutate chat history. All text
generation goes through `chatbot.api.generate_structured()` and the shared
`config.supports_structured_output()` predicate. Keep strict schema validation and
source budgets adapted to the loaded local context. Treat refusals, malformed JSON,
and incomplete replies as failures; never retry using a weaker output format.

For source-ID generation, `tools/citations.py` must deep-copy shared schemas and constrain
`source_id`/`source_ids` through a shared `$defs/source_identifier` enum containing
only that request's evidence IDs. Keep exact allowed strings in the prompt, and
never derive IDs from passage text. Enforce nonempty citations where required;
preserve explicit empty-citation exceptions. Server grammar constraints supplement
local uniqueness, scope, source-ownership, and domain validators. Budget the scoped
schema before inference. Report tool/stage and consolidation round/batch on failure
without logging raw model passages. Keep Studio processing/output budgets independent
of the ordinary Chat/RAG output cap. Studio evidence input uses available effective
context; do not restore a fixed 6,000-token ceiling. Reserve actual task, schema,
citation/document metadata, framing, and output overhead for each stage. Default
output is at most 32,768 tokens and at most a quarter of loaded context.
`STUDIO_MAX_INPUT_TOKENS` is an optional lower input cap (`0` means automatic,
positive values up to 1,048,576). `STUDIO_MAX_CONSOLIDATION_ROUNDS` defaults to 20
(range 1-100); `STUDIO_MAX_MODEL_CALLS` defaults to 256 (range 1-10,000). Raising
these limits never permits omitted evidence, invalid citations, ignored cancellation,
or nonprogressing consolidation. No automatic repair calls or weaker validation.

Whole-document tools use ordered catalog snapshots of all selected ready chunks,
including content hashes, fingerprints and published generations. Do not implement
them by asking `RAGService.answer()` to summarize its top-k retrieval window. Validate
citations against the snapshot; check cancellation and snapshot validity before each
model call and before publishing the result. Exceeding an explicit processing budget
must fail before generation rather than silently omit evidence. These checks establish
source membership, not that the model's claim is factually supported.
When initial evidence notes omit source identifiers, carry forward each missing
source's full original passage with only its own ID, preserving source order.
Generated malformed, unknown, or duplicate identifiers must still fail validation.
When consolidated notes omit source identifiers, carry forward the original notes
covering those identifiers with their complete text and citations. Never attach
omitted IDs to unrelated generated text. Keep malformed/unknown citations strict,
preserve nonshrinking batches, and require progress across each consolidation round
under the existing token, round, and request limits.

Spreadsheet keeps one current result under `artifacts/spreadsheet/current.json`,
saved automatically after generation and restored on workspace open. Its explicit
Save button below the editable dataframe updates that same result, including cell
edits and original references; there is no saved-output dropdown, Load button, or
version-creation button. Retain older JSON under `artifacts/<uuid>/` and adopt the
latest only when no current Spreadsheet exists. Keep atomic replacement
and cancellation guards so stale jobs cannot overwrite current outputs.
Tool outputs stay separate from `chats/`. Summary keeps a single current result under
`artifacts/summary/current.json`, replaced atomically after generation or Edit/Save
and restored automatically on workspace open. Summary export uses the chosen portable
filename and downloads directly without creating another history entry. Older Summary
artifacts are retained, with the latest adopted only when no current result exists.
Spreadsheet's direct Export includes current cell edits and writes the chosen portable
filename under `artifacts/spreadsheet/exports/`, without adding a history version.
Both tool layouts put the preview/table on the left and bounded reference excerpts
and vertically stacked filename/format/export controls on the right.
Presentation (`presentation`) appears before Plot and Mind Map in tool tabs and the Plugins
catalog. It uses the whole-source evidence pipeline with strict cited slide data:
title, bullets, notes, and source_ids. `tools/presentation.py` owns the bounds and
validation. Mirror those bounds in `tools/contracts.py` generation schemas,
including count-dependent bullet limits that obey the total-character bound. Do
not truncate generated bullets or relax existing saved-deck validation. The
generation schema permits 1-5 bullets: 1-3 have at most 180 characters each,
four have at most 150 each, and five have at most 120 each, guaranteeing the
600-character total. Slide titles allow 100 characters and notes 1,500; empty
notes remain valid. Derive schema and prompt bounds from shared validation
constants, and keep request schemas independent copies of the base definitions.
The slide-count slider defaults to 10, range 1–20; generation must return
exactly that count, including any cover, and cover every selected document. Keep one
current deck at `artifacts/presentation/current.json`, replaced atomically with
cancellation guards and restored on workspace open. Slide navigation is session-only
and must reject stale tokens/workspace revisions, including after workspace restore.
The left preview/notes and right References/export controls follow other tools.
PPTX is the default export; Markdown and JSON preserve source/provenance snapshots.
Export with lazy `python-pptx`, native editable text, escaped plain data, bounded
text layout, and per-slide references/source details in notes. Do not add unrequested
cover/reference slides or run generated markup/code. No Office installation or remote
renderer is needed. Missing PPTX dependencies must not break startup or text exports.
Preserve legacy Graphics manifests, raster assets, and history images on disk.
Only inert manifest compatibility needed for reading old artifact metadata remains;
there is no Graphics generation, UI, plugin, or direct exporter. Removing code must
never delete or rewrite user artifacts, source libraries, credentials, or histories.
Plot tab selection reloads only that tab's current result to refresh its preview
following hidden-tab mounting. Skip active/stopping jobs and deselection; reject
reads superseded by workspace switches, generation, or newer opens. Restore from
disk without model requests and preserve prior output on restore errors.

Plot (`plot`) appears immediately before Mind Map in the tool tabs and Plugins catalog.
Use the bounded whole-source pipeline to extract cited numeric data; chart type is
Auto (default), Bar, Line, or Scatter. Auto chooses a supported type from the
instructions; an explicitly selected type must be honored. A user-provided title
and grouping column are optional in direct-table mapping. Encode Auto as an
automatic selection with no explicit requested type, and allow its schema to
choose Bar, Line, or Scatter. A requested chart type in the instructions is valid
under Auto. Derive a neutral title from selected columns when none was provided;
do not reject a valid mapping for missing title/grouping or treat Auto as an
unsupported fourth chart type.
Validate content centrally in `tools/plot.py`: at most six unique series and 10,000
total points (also the per-series limit); finite bounded numeric values (never booleans),
categorical bars versus numeric line/scatter x values, and valid source identifiers.
Each point retains a short verbatim evidence excerpt from a cited source and its
explicit numerical values must occur in that excerpt. These checks establish
occurrence, not correct semantic pairing or units; show the read-only data table
and retain references for review. Do not invent missing values, interpolate,
convert units, or digitize images in this stage. Each selected document must have
plotted data or a cited limitation; no usable values is an actionable failure.
Keep one current plot under `artifacts/plot/current.json`, replaced atomically with
cancellation guards and restored on workspace open. Session tokens and workspace
revisions guard references and exports. Follow the standard left preview/table and
right References/vertical export layout. PNG is the default export; SVG/PDF render
the same data, CSV/JSON retain values and source/provenance snapshots. Escape
untrusted CSV text while preserving validated numeric fields as numbers.
Matplotlib is isolated in `plugins/plot/requirements.txt`, included by the main
requirements file. Import it lazily; missing dependencies must leave startup,
data viewing, and CSV/JSON exports working. Use trusted Figure/Agg rendering under
a lock without pyplot, GUI windows, external LaTeX, or model-produced code/markup.
Render text literally, including dollar signs and TeX-like strings. Do not pad
missing categories with zeros or silently drop invalid points. Rendering and
exports must not make model calls.
For one selected CSV/TSV, `tools/tabular_plot.py` validates a complete hash-checked
local table and requests only column/group selection through generate_structured.
Copy all row coordinates locally, preserving duplicates; never transcribe thousands
of points through model notes. Use separate bounds of 10,000 rows/chunks, 128 columns,
8 MiB original/source text, and 16 MiB saved output. Require explicit indexed numeric
passages, validate the mapping, and recheck source hashes/generations/cancellation
before and after generation. Unsupported filtering/aggregation/calculation requests
fail explicitly. Keep the UI table bounded to the first 200 rows with a clear label;
the rendered chart and CSV/JSON exports retain every point. Dense rendering reduces
tick labels/markers without sampling points or creating unbounded figure dimensions.
Mind Map (`mind_maps`) uses the same two-column layout with a bounded map viewport
and export controls; it intentionally has no References panel. Keep one current map
per workspace under `artifacts/mind_maps/current.json`. Validate a connected tree
with one root, bounded nodes/depth, and source IDs from the selected snapshot.
At the generation boundary, assign app-owned node IDs and remap parent links using
exact original IDs. Model ID formatting (numeric, Unicode, punctuation, or length)
must not reject an otherwise valid tree. Still reject blank/duplicate IDs, unknown
parents, cycles, invalid citations, and excessive size/depth; keep persisted/rendered
IDs strict, and never rewrite source IDs as node IDs.
Generate SVG only from trusted rendering code and escaped data; no model-produced
SVG, HTML, scripts, or remote renderer. Preserve source/provenance data in saved
maps and Markdown/JSON exports. Exclude standalone References and Translation tools.
Flashcards (`flashcards`) and Quizzes (`quizzes`) share the whole-source evidence
pipeline, with cited answers and coverage of every selected document. Validate their
content centrally in `tools/learning.py`: bounded, nonempty cards/questions, unique
prompts, four distinct quiz options, an integer answer index 0–3, and known citations.
Each learning tool has an integer count slider to the right of its instructions,
default 20, bounded to 1–60 Flashcards or 1–40 quiz questions. Validate the count
before model work, pass it explicitly to generation, and set the generation array's
minItems and maxItems to that count through `tools/contracts.py`. Validate the exact
count again without retries, padding, or truncation. Quiz generation must constrain
four options and an integer answer index 0–3. Record `settings.item_count`.
Keep count validation at generation time so older saved sets remain loadable.
Keep one current output per tool under `artifacts/<tool_id>/current.json`, replaced
atomically after generation and restored on workspace open. Study navigation, revealed
answers, and quiz attempts are session state only. Check answers and calculate scores
locally; count a question once until Restart quiz, and never make another API request
for study interactions. Reset study state on workspace changes or regeneration and
reject stale interactions. Markdown/CSV/JSON exports include complete content and
answer keys, source snapshots, and provenance; escape formula-like CSV values.
Quiz options display A./B./C./D. on full-width rows; their callback values remain
0-3. Quizzes defaults to Aiken (`aiken`, `.txt`) export and also supports Moodle XML
(`moodle_xml`, `.xml`). These formats are quiz-only and require no new dependencies.
Aiken contains only single-line questions/options and `ANSWER: X`, separated by
blank lines; protect parser-ambiguous question prefixes. It cannot retain explanations
or references. Moodle XML uses escaped `plain_text`, single-answer multichoice,
ABCD numbering, and explanation/cited evidence in general feedback. Keep the saved
payload and complete JSON source/provenance snapshot unchanged during exports.
Compare (`compare`) and Evidence Check (`evidence_check`) complete the current rollout
and follow Quizzes in the tool tabs and Plugins catalog. Both process all selected
indexed sources with the shared local Prompt model without modifying chat history. Validate
their structured content centrally in `tools/review.py`. Compare requires 2–8
documents and 1–12 unique aspects, with exactly one finding per selected document
per aspect. Found findings cite only their own document; not_found findings have
no citations. Mirror these bounds and document-specific scopes in generation
contracts. Each aspect has found evidence and a synthesis citing its findings.
Synthesis generation selects `finding_document_ids`; copy the selected findings'
validated citation lists locally before canonical validation. Never add a source
that the selected findings did not cite. Persist the existing `source_ids` format.
Reject unknown or duplicate finding references and old canonical fields in new
generation responses; do not add a fallback generation format. A selected
not_found finding can describe a limitation but contributes no citations, so a
synthesis must also explicitly select at least one found finding. Preserve model
text and the selected findings' citation order without mutating the response.
Evidence Check accepts 1–20 nonempty claims, one per line, at most 1,000 characters
each. Assign app-owned claim IDs and retain exact input text in settings and results;
never let the model silently rewrite or omit a claim. Classify selected-source
support as supported, contradicted, mixed, or not_found. Validate each evidence
quotation against its cited original passage and validate stance/verdict consistency.
Not_found is insufficient selected-source evidence, never proof that a claim is false.
Evidence Check uses `tools/quotations.py` and `tools/quote_review.py`, not model
notes/consolidation. Build ordered app-owned quote IDs over exact original
substrings, using overlapping bounded spans for long passages. Assess the full
catalog directly when it fits; otherwise scan every source batch and pool all
selected candidates and their assessments. Preserve supporting, contradicting,
and context candidates without silently trimming the pool. The final model selects
quote IDs and stances; the app copies exact original quotes/source IDs and runs
existing claim, occurrence, duplicate, and verdict/stance validators. Reject
unknown IDs, model-written quote fields, and candidate pools that cannot fit. Never
use fuzzy matching or hidden repair calls. Keep snapshot/cancellation guards and
canonical artifact/export formats. Explain that model verdicts are not independent
fact verification. Do not force citations to irrelevant documents to satisfy coverage.
Evidence Check must snapshot sources without applying the other tools' note-batch
limits before quote slicing. Its direct, scan, and final requests independently
budget complete instructions, claims, document metadata, scoped schemas, excerpt
text, earlier selections, and output. Preflight all scan batches and reserve the
final request before generation. Empty candidate selections may produce a strict
not_found assessment; never manufacture quotations to fill that absence.
Both tools keep one atomic current output at artifacts/<tool_id>/current.json,
restore it on workspace open, and retain the prior output on failure/cancellation.
Use read-only reports/tables with bounded previews, References, and vertical
filename/format/Export controls. Markdown/CSV/JSON retain source/provenance details;
CSV cells must be formula-safe. No additional dependencies or browser installers.
Guard references and exports with session tokens and workspace revisions.

Preserve source snapshots and mark user edits. Scope all paths to the workspace and
reject traversal/symlink escapes. Render
data using trusted templates; never execute generated Python, JavaScript, SVG scripts,
spreadsheet formulas, or shell commands. Escape untrusted spreadsheet values on export.
Every direct exporter must check its cancellation predicate before work and before
atomic replacement. A stale export must preserve any existing download, including
when a workspace, output token, filename, or format changes during serialization.

The Plugins tab manages only the trusted bundled catalog in this stage. Persist app-wide
enable/disable settings under ignored `.cache/plugins.json`; dependency checks and
exporters stay lazy. Optional export failures must not break UI startup or basic text
exports. There is no browser-driven package installation or arbitrary plugin loader.
Only the plugin table's Enabled boolean column is editable. Apply checkbox changes
immediately, validate against trusted catalog rows, and reject attempts to enable
planned tools or modify metadata. Update changed settings only so stale views cannot
overwrite unrelated choices from another session.

## Conventions

- Log through `from chatbot.log import logger` (or `.log` inside chatbot); no `print()` in app code.
- Catch specific exceptions. Bare `except:` previously hid a real tokenizer bug.
- Keep tests focused on behavior, representative boundaries, and past regressions.
  Avoid redundant parameter cross-products, cosmetic UI snapshots, constant-only
  checks, and tests that mirror the implementation. Preserve citation/data integrity,
  workspace isolation, cancellation, atomic persistence, path safety, and API contracts.
- Prefer adding to a `config.py` table or predicate over an inline model-name check.
- Use pathlib and platform guards for native integrations. Physical filenames must
  fit UTF-8 byte limits as well as character limits; keep Windows-invalid names
  rejected on every platform so data directories remain portable.
- Clean generated artifacts without deleting API keys, histories, sources, vectors,
  or user work. Keep local data and credentials ignored by Git.
