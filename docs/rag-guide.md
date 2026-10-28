# RAG Chat guide

[Back to README](../Readme.md) | [Studio tools](studio-tools.md)

## Supported sources

- PDF page references, DOCX sections/tables, XLSX/XLS sheet/cell locations, PPTX
  slides/tables/notes, CSV/TSV rows, HTML, Markdown, and plain text.
- Optional local vision extraction for scanned PDF pages, image files, and embedded
  raster pictures in DOCX/PPTX. Text-bearing PDF pages currently use native text;
  figures on those pages are not automatically interpreted. Legacy `.doc` and `.ppt`
  files must first be converted to `.docx` and `.pptx`.
- Conservatively budgeted chunks using LangChain text splitters, local LM Studio/Ollama
  embeddings, Qdrant semantic search, and SQLite FTS5 keyword
  search combined with reciprocal rank fusion.
- Separate working directories and document filters, content-hash deduplication, persistent originals,
  resumable extraction/embedding caches, and independent saved RAG conversations.
- Context budgeting, document-grounded prompting, and validation that generated numbered
  references belong to retrieved sources. These checks do not prove a claim is correct;
  use the source preview to verify important answers.

Indexing sends extracted text to the connected local embedding model.
Answering sends the question, recent conversation context, and retrieved passages
to the selected local chat model. Optional vision sends image data to a local
vision-capable model. Library originals, catalogs, caches, vectors, and saved chats
stay in the local workspace/database. A configured private LAN inference endpoint
receives the request data on that selected machine.

Start LM Studio/textgen/KoboldCpp with a loaded chat model, or Ollama with an installed local
chat model. In **Connections**, choose the server and press **Connect**. The app starts
disconnected and does not contact a server merely because its radio button is selected.
A sole installed embedding candidate is checked automatically; the check can load it
but never downloads a model. With multiple candidates, set the matching
`LMSTUDIO_EMBEDDING_MODEL` or `OLLAMA_EMBEDDING_MODEL` and restart. See the
[local connections guide](local-connections.md).
When using textgen or KoboldCpp, explicitly choose Ollama or LM Studio under
**RAG embeddings** before connecting. Both default to chat only; their native
embedding endpoints are not called. The selected embedding server supplies
indexing/search, and the selected chat server supplies the answer. Keeping the
same embedding server and model keeps compatible indexes reusable. Vision
extraction is unavailable through the textgen and KoboldCpp connectors.

## Workspaces and controls

Each working directory is a subfolder of `rag_chatbot/data`, containing its own
`sources/`, `chats/`, `artifacts/` (after saving a tool result), and `library.sqlite3`. The active directory controls indexing,
retrieval, chat saving/loading, and source previews. Qdrant indexes are also separated
by working directory. Selecting a directory in **Workspace** opens it immediately.
To create one, type a new name and press Enter; **Create / Open** remains available.

The RAG tab has one outer row. Its left column (one third) contains the working
directory, Sources, and Document viewer. The right column (two thirds) contains
two equal columns: the chatbot with Prompt controls below, and saved chat history
with the **References from the latest answer** accordion below. The selected
reference's download, image, and excerpt appear directly below **Open reference**
inside that accordion. The Tools area spans the bottom of
the right column. **Toggle chat history** shows or hides the saved-chat controls
while keeping references available.

Prompt includes the local **Model**, **Stream responses**, **Temperature**, and
**Top-p** controls. Turning off streaming waits for the complete answer. Configure
model-native thinking in the inference server. Web search and external URL fetching are
unavailable; an empty source selection makes no document-answer request.
Studio tools use the selected model with their own structured-generation settings.

Chat replies use the remaining effective context after the prepared request is
accounted for, without a fixed output-token ceiling. The matching
`LMSTUDIO_MAX_OUTPUT_TOKENS`, `OLLAMA_MAX_OUTPUT_TOKENS`, `TEXTGEN_MAX_OUTPUT_TOKENS`, or `KOBOLDCPP_MAX_OUTPUT_TOKENS` is an optional lower cap; leave it unset or `0` for the automatic maximum. Context and
image accounting are conservative estimates, and thinking consumes the same
finite generation allowance. For more room, reload the model with a larger context
in LM Studio, or set `OLLAMA_CONTEXT_LENGTH` and restart for Ollama, then Connect
again. Connections shows the loaded window, model maximum, and effective app budget
separately. Changing an LM Studio default alone does not resize a loaded model.
For textgen, enter the server's loaded context in **Textgen context length** under
Connections, then Connect again. The field sets this session's app budget without
reloading the model or restarting the app. Its initial value comes from
`TEXTGEN_CONTEXT_LENGTH`, otherwise 4,096. Textgen reports no loaded window, so
the UI labels it unknown; enter no more than the actual server context.
KoboldCpp reports its allocated context automatically. Reload in KoboldCpp and
Connect again to change it; `KOBOLDCPP_CONTEXT_LENGTH` is an optional lower cap.

If a limit is reached after answer text appears, that text stays visible with a
notice and its references. You can save it or ask to continue; the app never
continues automatically. If thinking uses the allowance before an answer appears,
the app reports the missing answer. Reduce thinking or shorten the input in that
case. Studio tools keep their own explicit processing/output budgets, independent
of the ordinary-chat cap. Structured results and vision extraction reject incomplete
output.

## Document previews

The document viewer loads one page, frame, slide, or table segment at a time:

- PDFs have rendered pages with page navigation; images have a local image preview
  and frame navigation for supported multipage images.
- XLS/XLSX workbooks have a sheet selector and a paginated grid with row numbers
  and column letters. XLSX formulas are shown as formulas rather than recalculated.
  Previews are limited to 10,000 rows and 50 columns per sheet.
- Word and PowerPoint use locally installed Microsoft Office on Windows, or
  optional LibreOffice on macOS/Linux, to create cached PDF previews. Originals
  are unchanged. If conversion is unavailable,
  DOCX/PPTX use a content preview with text, tables, and embedded pictures; this
  fallback does not reproduce the original page/slide layout. Legacy DOC/PPT files
  require a local converter or downloading the original to open them.
- Text, HTML, and CSV/TSV sources also have bounded local previews.

Previewing does not call an inference server or send documents to an online viewer. To preserve
open presentations, a new PowerPoint conversion requires the existing PowerPoint
application to be closed; cached previews remain usable. Preview support is independent
of indexing support: legacy DOC/PPT sources still need conversion to DOCX/PPTX or PDF
before they can be indexed.

## Start the vector database

**Embedded Qdrant is the default** when neither `QDRANT_URL` nor
`QDRANT_LOCAL_PATH` is set. No Docker installation is needed. Each workspace uses
its own database under `RAG_DATA_DIR/.cache/qdrant/` (by default,
`rag_chatbot/data/.cache/qdrant/`). Run only one app process against
an embedded database. Set `QDRANT_LOCAL_PATH` to choose a different cache base;
workspace databases remain separate below it.

For a larger library, start Docker Desktop (Windows/macOS) or Docker Engine with
the Compose plugin (Linux), then run:

```sh
docker compose up -d
```

Enable the local Qdrant server explicitly before starting the app:

```powershell
# Windows PowerShell
$env:QDRANT_URL = "http://127.0.0.1:6333"
python webui.py
```

```sh
# macOS/Linux
QDRANT_URL="http://127.0.0.1:6333" python webui.py
```

Clear `QDRANT_LOCAL_PATH` first if it was set, because it overrides server mode.
The compose file pins Qdrant `v1.19.2`, binds it to loopback, and persists vectors
in a Docker named volume. `docker compose stop` preserves data. Do not use
`docker compose down -v` unless you intend to delete those vectors. `QDRANT_URL`
must use a loopback HTTP(S) address; `QDRANT_API_KEY` is optional for an authenticated
local server. The Chat tab still works when the vector database is unavailable.
Indexing checks the database before requesting local embeddings.

Switching database backends does not migrate vectors. Keep the old data backed up
and use a separate `RAG_DATA_DIR` for a trial, or reindex originals into the selected
backend. `RAG_DATA_DIR` defaults to `rag_chatbot/data` and contains the workspace
catalogs, source files, chats, and tool artifacts. Include the embedded cache or
server volume in backups.

Embedding, vision, chunking, and retrieval settings live in `rag_chatbot/config.py`
and the provider-specific `LMSTUDIO_*` / `OLLAMA_*` / `TEXTGEN_*` / `KOBOLDCPP_*` settings in the connection guide. Local index fingerprints
start with `local-v2` and include the embedding identity, dimensions, prefixes,
context, chunking, and preprocessing version. Changing these settings requires
reindexing affected documents. Existing OpenAI vectors are never reused as local
vectors, even when dimensions match. After connecting to the chosen embedding
model, check files marked **reindex needed** in **Sources for chat and tools** and
click **Reindex selected**. It reads the originals already saved in that workspace;
saved chats, citation snapshots, and artifacts are retained. Missing or damaged
originals require reuploading the source file.

Context checks use conservative UTF-8 byte estimates without tokenizer downloads.
They can consume budget faster than a model's real tokenizer, especially for long
or multilingual sources. Reduce source selection or load a larger context in
your inference server and reconnect; the app does not silently omit evidence to make a tool fit.

## Use RAG Chat

1. Open **RAG Chat** and select a working directory to open it automatically, or
   type a new name and press Enter. An empty **Default** directory is available on first use.
2. Upload files in **Sources**. The filename textbox starts with the uploaded name;
   edit it before indexing if you want a different name in the library and citations.
   For multiple uploads, choose each file in the upload selector to edit its name.
   Keep the original file extension (it is added if omitted). Enable vision if you
   want scanned pages/images read, then click **Index documents**.
3. Check document statuses. Select files marked **reindex needed**, failed, or stopped
   in **Sources for chat and tools**, then click **Reindex selected** to use their
   saved originals. Compatible extraction and embedding batches are reused.
   Use the vision checkbox to include scanned pages/images. If the saved original
   is missing or damaged, upload the same file again instead.
   Reimporting an already indexed file with vision enabled upgrades its extraction
   without adding a duplicate. Stop prevents further OCR/embedding calls after the
   current request finishes; use Refresh library to see the final status.
4. Check the sources to use for chat, then ask a question using the Prompt controls.
   Ready sources start checked; **Select all** checks ready sources and **Clear**
   selects no documents and prevents a document-answer request.
   Your choices survive refresh and indexing. Newly indexed sources
   start checked, and failed sources are never searched. Changing the checks stops
   a reply in progress; the next question uses the new selection.
5. Select a document in **Document viewer** to open its preview automatically. This
   selection is independent of the checked sources used for chat retrieval.
6. Select a reference in the history/reference column to inspect its passage and
   original file. Save and load conversations from that directory's history controls.

Save a conversation before switching working directories. Opening another directory
clears the active conversation, uploads, filters, document preview, and references, and stops work from
the previous directory. Files and saved chats in other directories remain available
when you open those directories again.

Source names are library metadata: the file on your computer and the managed original's
stable path are unchanged. Renaming an identical reimport updates its library name
without creating a duplicate or recomputing embeddings. Existing saved chats retain
their citation snapshots. **Remove checked sources** removes the checked documents
from retrieval; failed documents can also be checked for removal.
Previous replies remain visible when the selection changes. If earlier replies cite
unchecked or unavailable sources, their generated text is omitted from the model's
conversation context; previous user questions remain available for follow-ups.
Legacy saved replies that used web search are also excluded from subsequent document-only context,
including after saving and reloading, so web-derived text is not reused as local evidence.

Removing a document excludes it from retrieval. Its managed original and caches are
retained so saved citations remain usable. A changed file is a new library version;
remove the old version explicitly when it should no longer be searched. The initial
version supports files up to 100 MiB and bounded page/image/cell counts, not an
unlimited background ingestion service. Native Office formulas are represented by
cached values where available; the app does not recalculate workbooks. OCR quality,
retrieval quality, and model abstention should be evaluated on your own documents.

This implementation uses `langchain-text-splitters` selectively. The full `langchain`
and `langchain-community` packages are unnecessary for the current readers and direct
HTTP provider integration. Future work can add reranking, full-page figure understanding,
folder watching, and evaluation against a representative research library.

## Troubleshooting

| Problem | What to try |
|---|---|
| Indexing reports that Qdrant is unavailable | Check the embedded directory is writable and no second app owns it; for explicit server mode, start Docker and Qdrant. |
| No local model is available | Start the selected server, make a local chat model available, and press Connections > Connect. |
| Embedding configuration is missing or dimensions differ | Choose an installed embedding model, run the connection check, then reindex with matching settings. |
| Input or a Studio task exceeds context | Reduce source selection or load a larger context in your inference server and reconnect; the app uses conservative estimates. |
| Scanned pages or images have no searchable text | Load a local vision-capable model and reupload with vision extraction enabled; extraction makes additional local model requests. |
| An answer has insufficient evidence | Check that the intended sources are ready and checked; try a more specific question. |
| A saved original is missing or damaged | Reupload the same original file. The managed copy is repaired while a valid index is reused. |
| Office layout preview is unavailable | Use the content preview, download the original, or install the local converter described below. |
| The browser shows old controls after an update | Restart the app, then hard-refresh with Ctrl+Shift+R (Cmd+Shift+R on macOS). |

Use **Refresh library** after a stopped import to see its final status. Retrying the
same file reuses completed extraction and embedding batches where possible. Keep
only one app process open when using embedded Qdrant.

## Optional Office layout previews

| Platform | Local converter | Without the converter |
|---|---|---|
| Windows | Installed Microsoft Word/PowerPoint and platform-marked `pywin32` | DOCX/PPTX content preview |
| macOS | LibreOffice application | DOCX/PPTX content preview |
| Linux | LibreOffice Writer and Impress | DOCX/PPTX content preview |

Install [LibreOffice](https://www.libreoffice.org/download/download-libreoffice/)
if you want page/slide layout previews on macOS/Linux. The app finds `soffice` or
`libreoffice` on `PATH`, or the standard macOS application location. For a custom
installation, set `LIBREOFFICE_PATH` to its executable, for example:

```sh
export LIBREOFFICE_PATH="/Applications/LibreOffice.app/Contents/MacOS/soffice"
```

Conversions use a temporary profile with macros disabled, have a time limit, and
cache the resulting PDF beside the managed original. Your normal LibreOffice
profile is not used. Fonts and converter versions can affect pagination and layout.
Minimal Linux installations also need fonts, such as the `fonts-dejavu-core` package.
DOC/PPT files can be previewed with a converter but must still be converted to a
supported format before indexing. PDF/image/spreadsheet previews and DOCX/PPTX
text extraction need neither Microsoft Office nor LibreOffice.
