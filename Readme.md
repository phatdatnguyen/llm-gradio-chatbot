# Local Chat and RAG Chat

A Gradio app for chatting with models served by **LM Studio, Ollama, textgen, or KoboldCpp** and working with your
own documents. Runs on Windows, macOS, and Linux with Python 3.10+.

- **Chat:** streaming conversations, local document/image attachments, LaTeX, and saved history.
- **RAG Chat:** document workspaces, local embeddings, cited answers, previews, and nine Studio tools.
- **Connections:** choose a local server and connect manually; discover models and check embeddings.
- **Plugins:** enable or disable the bundled tools and check export dependencies.

This release implements **LM Studio, Ollama, text-generation-webui (textgen/oobabooga), and KoboldCpp**. There is no OpenAI SDK,
OpenAI API-key requirement, cloud fallback, web search, URL fetching, or image
generation. Existing chat histories and saved tool artifacts remain on disk.

Both chat tabs render inline math with `$...$` or `\(...\)` and display math
with `$$...$$` or `\[...\]`.

## Installation

From the project directory, create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv chatbot-env
.\chatbot-env\Scripts\Activate.ps1
```

If activation is restricted, use `chatbot-env\Scripts\python.exe` instead of
`python` below. Command Prompt can use `chatbot-env\Scripts\activate.bat`.

**macOS/Linux:**

```sh
python3 -m venv chatbot-env
. chatbot-env/bin/activate
```

Install dependencies:

```sh
python -m pip install -r requirements.txt
python -m pip check
```

Install [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/),
[textgen](https://github.com/oobabooga/textgen), or
[KoboldCpp](https://github.com/LostRuins/koboldcpp) separately
and install your models there. Initial setup normally needs an internet connection;
this chatbot never downloads models.

## Start the app and connect

1. Start your chosen local server. In **LM Studio**, load a chat model and start
   the Developer server (`http://127.0.0.1:1234/v1`). In **Ollama**, install a local
   chat model and start Ollama (`http://127.0.0.1:11434`); installed models load when used.
   In **textgen**, load a model and enable its API (`http://127.0.0.1:5000/v1`).
   In **KoboldCpp**, load a model and start its server (`http://127.0.0.1:5001/v1`).
2. For RAG, install an embedding model in LM Studio or Ollama. A chat model alone
   cannot create a vector index. For Ollama, `ollama pull nomic-embed-text` is an
   optional manual setup command.
3. Run `python webui.py`, or use `start_webui.bat` / `sh start_webui.sh`.
4. Open the printed local URL, normally `http://127.0.0.1:7860`. The app starts
   **disconnected**. In **Connections**, choose **LM Studio**, **Ollama**, **textgen**, or **KoboldCpp**.
   For textgen, set **Textgen context length** to the context used to load its model.
   For textgen or KoboldCpp RAG, explicitly choose Ollama or LM Studio under
   **RAG embeddings**; its default is **None (chat only)**. Press **Connect**. Editing these
   controls alone does not connect or switch the active connection.
5. Choose a discovered chat model. Both chat tabs start with the connected session's
   default model. Press **Connect** again after changing models or context settings.

Connect discovers chat models and, when using LM Studio/Ollama embeddings, checks
the sole installed embedding candidate. This
check can load that embedding model, but never downloads one. With several candidates,
set `LMSTUDIO_EMBEDDING_MODEL` or `OLLAMA_EMBEDDING_MODEL` before starting the app.
An embedding error leaves ordinary chat available. Each browser session owns its
connection; one session cannot redirect another session's requests. Reconnecting
cancels pending work on the previous connection rather than moving it to another server.

## Connection settings

Environment settings are read by the app process; restart after changing them.
Use the matching `LMSTUDIO_`, `OLLAMA_`, `TEXTGEN_`, or `KOBOLDCPP_` prefix. The detailed
[local connections guide](docs/local-connections.md) covers all settings and setup.

| Setting | LM Studio | Ollama |
|---|---|---|
| Server URL | `LMSTUDIO_BASE_URL`, default `http://127.0.0.1:1234/v1` | `OLLAMA_BASE_URL`, default `http://127.0.0.1:11434` |
| Chat model | `LMSTUDIO_MODEL` | `OLLAMA_MODEL` |
| Embedding model | `LMSTUDIO_EMBEDDING_MODEL` | `OLLAMA_EMBEDDING_MODEL` |
| Context | `LMSTUDIO_CONTEXT_LENGTH`: fallback 4,096; explicit values cap the loaded window | `OLLAMA_CONTEXT_LENGTH`: explicit native request window, bounded by the model maximum; otherwise loaded context or fallback 4,096 |
| Ordinary reply cap | `LMSTUDIO_MAX_OUTPUT_TOKENS` | `OLLAMA_MAX_OUTPUT_TOKENS` |

Textgen uses `TEXTGEN_BASE_URL` (default `http://127.0.0.1:5000/v1`),
`TEXTGEN_MODEL`, `TEXTGEN_CONTEXT_LENGTH`, and `TEXTGEN_MAX_OUTPUT_TOKENS`.
Only its currently loaded model is selectable. Its API does not report loaded
context. **Textgen context length** in Connections sets this session's request
budget when you press **Connect**; its initial value is `TEXTGEN_CONTEXT_LENGTH`
or 4,096. Enter the model's loaded window, for example 131072 for a model actually
loaded with that context. This does not resize the model. Studio generation is
supported with the `llama.cpp` loader; textgen image inputs/vision extraction are unavailable. Its
native embedding endpoint is not called. The explicitly chosen Ollama/LM Studio
embedding connector keeps its own settings and index identity, so changing only
the chat server does not require reindexing.

KoboldCpp requires version **1.123.1 or newer** and uses `KOBOLDCPP_BASE_URL` (default `http://127.0.0.1:5001/v1`),
`KOBOLDCPP_MODEL`, `KOBOLDCPP_CONTEXT_LENGTH`, and `KOBOLDCPP_MAX_OUTPUT_TOKENS`.
Connect discovers its currently loaded model and allocated context automatically.
An explicit context setting can lower that budget; it does not resize the model.
Chat, streaming, and Studio generation use the local loaded model. Image inputs
and vision extraction are unavailable through this connector. RAG uses the
explicitly selected Ollama/LM Studio embedding connection, with the same index
identity as that connection used directly. KoboldCpp's native output maximum also
applies, even when the app's optional reply cap is unset.

Leave the reply cap unset or `0` to use the remaining effective context after the
prompt, history, attachments, and framing. There is no fixed 2,048/4,096 output
ceiling. Studio tools retain their own processing/output budgets. Connections shows
loaded context, advertised model maximum, and the app's effective request budget
separately. Changing LM Studio's default requires reloading the existing model
before reconnecting. Ollama receives its effective context explicitly on each request.

The connectors accept loopback or an explicitly configured private LAN IP, reject
public destinations and redirects, and ignore environment HTTP proxies. With a LAN
endpoint, request data goes to that selected machine. Ollama cloud/proxy models are
excluded; a current server with explicit local-routing support is required.

Thinking uses the same finite generation allowance as answer text. Configure
model-native thinking in your inference server; the app does not send cloud reasoning
options. A reply that reaches its limit keeps visible text with a notice, without
automatic continuation. Structured tools and vision extraction require complete
output. Conservative UTF-8 estimates avoid tokenizer downloads, but may provide less
usable input space than the model's exact tokenizer.

## RAG Chat

Embedded Qdrant is the default, so Docker is optional. Keep only one app process
running against a given embedded database. For larger libraries, see the
[local server database setup](docs/rag-guide.md#start-the-vector-database).

1. Select a **Workspace**, or type a new name and press Enter.
2. Upload files under **Sources**, adjust their display filenames if needed, and
   click **Index documents**. Enable vision only with a local vision-capable model.
3. Check the documents to use, then ask a question. Empty selection means no sources.
4. Open **References** to inspect supporting passages, or choose a source in
   **Document viewer** to preview it.
5. Save conversations before switching workspaces. Each retained Studio tool saves
   its latest successful result and restores it when you return.

For files marked **reindex needed**, check them in **Sources for chat and tools**
and click **Reindex selected**. This uses the originals saved in the workspace
with the connected embedding model; no upload is needed. The same button retries
failed or stopped indexing when the saved original is still available.

Supported sources include PDF, DOCX, XLS/XLSX, PPTX, CSV/TSV, HTML, Markdown, text,
and images. Convert legacy DOC/PPT files before indexing. Office is not required
for native extraction; optional local converters provide page/slide layout previews.

Workspaces live in `rag_chatbot/data/` by default. Extracted text, questions, passages,
and optional image inputs go to your configured local inference server. Review source
references when accuracy matters.

**Existing OpenAI-based indexes must be reindexed.** Local embeddings have different
vector spaces even when dimensions match. The `local-v2` fingerprint keeps incompatible
vectors out of search. Back up your data, reconnect to the local server, then check
the affected files and click **Reindex selected**. Saved chats, originals, and old
artifacts are retained. Reupload only if a saved original is missing or damaged.

See the [RAG Chat guide](docs/rag-guide.md) for source management, database options,
previews, storage, and troubleshooting.

## Studio tools

| Tool | What it does |
|---|---|
| Summary | Creates a cited summary that you can edit and export. |
| Spreadsheet | Extracts fields into an editable table, with one row per document. |
| Presentation | Creates slides with notes and editable PowerPoint export. |
| Plot | Makes bar, line, or scatter charts with up to 10,000 points. |
| Mind Map | Organizes document topics into a visual tree. |
| Flashcards | Creates cards for local study and review. |
| Quizzes | Creates and scores questions; exports Aiken or Moodle XML. |
| Compare | Compares 2-8 selected documents. |
| Evidence Check | Reviews claims against selected sources with quotations. |

Tools share the checked sources and Prompt model. They request schema-constrained output
from the connected local server, constrain citations to the sources in each request, and validate
the returned structure and citations locally. Invalid,
refused, or incomplete output fails without replacing the previous result. Local model
quality and loaded context determine which tasks fit; no weaker-format retry is made.
Evidence batches use the available effective context rather than a fixed 6,000-token
ceiling. Larger runs have configurable round/request limits; see
[Studio processing settings](docs/studio-tools.md#processing-limits-and-cancellation).
Image generation is not included. Existing history images and older Graphics
artifacts are retained on disk; there is no Graphics tab or exporter.

Presentation requests exactly the chosen slide count. Slides have 1-5 bullets:
up to 180 characters each for 1-3 bullets, 150 each for four, or 120 each for five.
Slide titles are limited to 100 characters and notes to 1,500. These generation
constraints keep slides readable without truncating the returned text; existing
saved decks remain readable under their original validation rules.

Compare generates 1-12 aspects with one finding for each selected document.
For each synthesis, the model explicitly selects the findings it discusses;
the app copies citations from those findings into the saved result. It never
substitutes unrelated sources to repair a citation.

Evidence Check selects app-assigned IDs for original source excerpts, and the
app copies the exact quotations locally. If all excerpts fit, assessment takes
one request. Otherwise, every source batch is reviewed before the selected
candidates are assessed together. An oversized candidate pool fails explicitly
instead of dropping quotations. Verdicts remain model assessments for review.

For an Iris scatter plot, select only the CSV and ask for sepal length on the x-axis,
sepal width on the y-axis, colored by species. Coordinates are copied locally after
one model request selects columns. The data table shows up to 200 points; charts and
CSV/JSON exports retain all points. **Auto** permits the chart type requested in
your instructions; a custom title and grouping column are optional. Without a
title, the app requests a neutral label based on the selected columns.

See the [Studio tools guide](docs/studio-tools.md) and
[development, testing, and backup guide](docs/development.md).

## License and third-party code

The project license text is in [LICENSE.txt](LICENSE.txt). The vendored
[llama.cpp schema converter](chatbot/providers/_llama_schema.py) retains its
upstream MIT notice, source version, and modification notes. Preserve those
notices when redistributing the code.
