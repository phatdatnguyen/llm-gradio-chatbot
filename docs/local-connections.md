# Local connections

[Back to README](../Readme.md)

The app starts disconnected. Choose **LM Studio**, **Ollama**, **textgen**, or **KoboldCpp** in **Connections**
and press **Connect**. Opening the app or changing the connector radio button makes
no server requests. The connection belongs to the browser session; reconnecting
invalidates its earlier work and does not change other sessions' models or servers.

## Server setup

**LM Studio:** install models in LM Studio, load a chat model, and start its Developer
server. The default endpoint is `http://127.0.0.1:1234/v1`. Connect reads the actual
loaded context, not just the model's advertised maximum. Reload an existing model
after changing LM Studio's default context, then Connect again.

**Ollama:** start Ollama and install local models yourself. Its default endpoint is
`http://127.0.0.1:11434`. Connect lists installed completion models without loading
them; inference loads the selected model when needed. For document retrieval,
install an embedding model, for example:

```sh
ollama pull nomic-embed-text
```

That is a manual installation command. The app has no pull/download path. A sole
embedding candidate is selected and checked on Connect; multiple candidates require
an explicit embedding-model setting. The check sends a short synthetic text and may
load the installed embedding model. Missing embeddings do not prevent ordinary chat.

Ollama requests force `:local` routing and reject cloud/proxy metadata and explicit
cloud selectors. Use a current Ollama version supporting this syntax. If an older
server rejects it, update Ollama; the app does not fall back to remote routing.
Replacing model files under an existing tag requires reconnecting. Model digests
are part of embedding identity, so changed models require reindexing.

**textgen (text-generation-webui / oobabooga):** load a model in textgen and enable
its API at `http://127.0.0.1:5000/v1`. The chatbot selects the currently loaded model;
it never loads or switches textgen models. Connect again after changing the model
or LoRAs. Chat and streaming are supported. Studio generation uses schema-constrained
GBNF with the supported `llama.cpp` loader. Other loaders are not enabled for Studio
generation. Image inputs and vision extraction are unavailable through this connector
because the API does not advertise reliable loaded vision capabilities.

Set **Textgen context length** to the context used to load the model in textgen,
then press **Connect**. For example, enter `131072` if the server loaded the model
with a 131,072-token window. This sets the app's request budget; it does not reload
or resize the server model. The API leaves the loaded window unknown. The field
initially uses `TEXTGEN_CONTEXT_LENGTH`, or 4,096 when that variable is unset.
Changing the field takes effect only on the next Connect and needs no app restart.

For RAG with textgen, choose **Ollama** or **LM Studio** under **RAG embeddings**, then press **Connect**. The default is **None (chat only)**. This explicit
choice contacts textgen for generation and the chosen local embedding server for
discovery and an embedding check. It uses that server's existing embedding settings;
for example, an installed Ollama `nomic-embed-text` can serve embeddings while textgen
answers questions. Changing either selector or the context field alone makes no requests.

The native textgen embedding endpoint is not used: it ignores the requested model
and may download a model on first use. There is no automatic fallback to another
embedding server. See the upstream [embedding implementation](https://github.com/oobabooga/textgen/blob/main/modules/api/embeddings.py)
and [request definitions](https://github.com/oobabooga/textgen/blob/main/modules/api/typing.py).

**KoboldCpp:** use version **1.123.1 or newer**, load a model, and start its local server at `http://127.0.0.1:5001/v1`.
Connect discovers the currently loaded model and allocated context automatically.
It does not load or switch models. Connect again after changing the loaded model
or context. An optional `KOBOLDCPP_CONTEXT_LENGTH` can lower the app budget, but
cannot enlarge the loaded window. Chat and streaming are supported; Studio uses
schema-constrained native grammar with local validation. Image inputs and vision
extraction are unavailable through this connector. The native server limits output
to 80% of context for a prompt below 25% of context, or 65% otherwise; the app
respects that limit as well as the remaining space. Server routing and generation
overrides that can change the prepared request are rejected.

For RAG with KoboldCpp, explicitly choose **Ollama** or **LM Studio** under
**RAG embeddings**, then press **Connect**. The default is **None (chat only)**.
The app uses the selected embedding server's installed model, settings, and index
identity; it does not call KoboldCpp embeddings or download models. An embedding
failure leaves ordinary chat available, without switching to another server.

## Environment settings

Set these before launching. Replace `PREFIX` with `LMSTUDIO`, `OLLAMA`, `TEXTGEN`, or `KOBOLDCPP` for the
server you use. The app does not automatically read a `.env` file.

| Variable | Meaning |
|---|---|
| `PREFIX_BASE_URL` | Local server endpoint; defaults are shown above. |
| `PREFIX_API_KEY` | Optional local-server authentication token. |
| `PREFIX_MODEL` | Exact discovered chat model ID; otherwise the first available chat model. |
| `PREFIX_EMBEDDING_MODEL` | LM Studio/Ollama only: exact installed embedding model ID; otherwise the sole candidate. |
| `PREFIX_MAX_OUTPUT_TOKENS` | Optional ordinary Chat/RAG output cap; unset or `0` uses remaining effective context. Studio limits are separate. |
| `PREFIX_TIMEOUT` | Network I/O timeout, default 900 seconds; connection attempts are capped at 5 seconds. |
| `PREFIX_VISION_MODEL` | LM Studio/Ollama only: optional document-extraction vision model; otherwise the connected default chat model. |
| `PREFIX_IMAGE_TOKEN_BUDGET` | Conservative reserve per image, default 4,096. |
| `PREFIX_EMBEDDING_DIMENSIONS` | LM Studio/Ollama only: optional dimension check; prefer dimensions detected by Connect. |
| `PREFIX_EMBEDDING_CONTEXT_LENGTH` | LM Studio/Ollama only: embedding input budget; defaults to discovered context or a conservative fallback. |
| `PREFIX_EMBEDDING_DOCUMENT_PREFIX` | LM Studio/Ollama only: optional retrieval document prefix. |
| `PREFIX_EMBEDDING_QUERY_PREFIX` | LM Studio/Ollama only: optional retrieval query prefix. |

Nomic embedding models receive `search_document: ` and `search_query: ` prefixes
automatically. Changing embedding identity, dimensions, prefixes, or chunk settings
requires reindexing. Switching between LM Studio and Ollama also changes identity,
even if both expose a model with the same name and dimensions.
Textgen or KoboldCpp with the same Ollama/LM Studio embedding connection keeps that embedding
identity, so changing only the chat server does not require reindexing.

Context has provider-specific behavior:

- `LMSTUDIO_CONTEXT_LENGTH` supplies a fallback (4,096 by default). When explicitly
  set, it can lower the effective context below the loaded window.
- `OLLAMA_CONTEXT_LENGTH` explicitly requests a native `num_ctx` window, bounded by
  the model's advertised maximum. Without it, the app uses a reported loaded window
  or the conservative 4,096 fallback. A large requested window uses more memory.
- `TEXTGEN_CONTEXT_LENGTH` sets the initial **Textgen context length** field and
  the default request budget for CLI callers, otherwise 4,096. The UI field can
  override it per connection; enter no more than the model's actual loaded window,
  then Connect. CLI callers can use `config.connect("textgen", context_length=131072)`
  for a model loaded with that window. These values do not resize the server model.
  Before generation, the connector checks textgen's rendered prompt and refuses a
  request that would silently lose message text.

- `KOBOLDCPP_CONTEXT_LENGTH` optionally caps the discovered allocated window.
  Without it, the app uses the loaded context reported by KoboldCpp; reconnect
  after loading with a different context. This setting never resizes the model.

For example, on Windows PowerShell:

```powershell
$env:OLLAMA_CONTEXT_LENGTH = "32768"
$env:OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
python webui.py
```

On macOS/Linux:

```sh
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest python webui.py
```

Press **Connect** after opening the app. Restart the app after changing environment
settings. Leaving the ordinary output cap unset does not remove the model's physical
context limit or the separate Studio processing limits.
