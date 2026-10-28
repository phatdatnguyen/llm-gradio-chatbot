"""Explicit, session-scoped local connections. Startup performs no discovery."""

from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr

from . import config
from .log import logger
from .providers.base import ProviderError


_SESSIONS = {}
_SESSION_LOCK = RLock()
_REQUEST_SNAPSHOT = "_local_chat_connection_snapshot"


def request_connection():
    """Freeze the session connection on Gradio's per-event request object.

    Gradio reinstates this same request before each generator step, including
    steps executed by different worker threads. An event that starts disconnected
    stays disconnected; reconnecting cannot redirect an existing event.
    """
    from gradio.context import LocalContext
    request = LocalContext.request.get(None)
    if request is None:
        return None
    with _SESSION_LOCK:
        if _REQUEST_SNAPSHOT not in request.__dict__:
            entry = _SESSIONS.get(request.session_hash, {})
            request.__dict__[_REQUEST_SNAPSHOT] = entry.get("connection")
        return request.__dict__[_REQUEST_SNAPSHOT]


def _model_status(model):
    context = config.get_model_context_info(model)
    details = [f"{context['effective']:,} context budget"]
    details.append(f"loaded at {context['loaded']:,}" if context["loaded"] is not None else
                   "not loaded or loaded context unavailable; using configured request budget")
    if context["maximum"] is not None:
        details.append(f"model maximum {context['maximum']:,}")
    limit = config.max_output_tokens(model)
    details.append(f"up to {limit:,} reply tokens, limited by remaining context" if limit is not None else
                   "automatic reply budget: all remaining context")
    return f"`{model}` (" + "; ".join(details) + ")"


def _connection_status(connection):
    with config.use_connection(connection):
        choices = config.get_model_choices()
        lines = [f"Connected to **{connection.provider_label}** at `{connection.base_url}`."]
        if choices:
            lines.append("Available chat models: " + ", ".join(_model_status(model) for model in choices) + ".")
        else:
            lines.append("No usable local chat models were found. Load or install a chat model in the selected server, then Connect again.")
        embedding = config.get_embedding_settings()
        if connection.embedding_error:
            lines.append("Chat is connected. RAG embedding check failed: " + connection.embedding_error)
        elif embedding["model"] and embedding["dimensions"]:
            source = ""
            if connection.provider_name in config.SEPARATE_EMBEDDING_PROVIDERS:
                label = dict((name, label) for label, name in config.PROVIDER_CHOICES)[connection.embedding_provider_name]
                source = f" via **{label}** at `{connection.embedding_base_url}`"
            lines.append(f"RAG embeddings: `{embedding['model']}` ({embedding['dimensions']} dimensions){source}.")
        elif connection.provider_name in config.SEPARATE_EMBEDDING_PROVIDERS and not connection.embedding_provider_name:
            lines.append("RAG embeddings are disabled. Choose Ollama or LM Studio under **RAG embeddings** "
                         "and Connect to use an installed embedding model.")
        else:
            prefix = (connection.embedding_provider_name or connection.provider_name).upper()
            lines.append("RAG indexing and search need a local embedding model. Install one yourself; "
                         f"if several are available, set `{prefix}_EMBEDDING_MODEL` and Connect again. "
                         "Chat is available without embeddings.")
        if connection.provider_name == "lmstudio":
            lines.append("Changing LM Studio's Default Context Length requires reloading the model and connecting again.")
        elif connection.provider_name == "ollama":
            lines.append("Ollama loads installed models when used. Set `OLLAMA_CONTEXT_LENGTH` before "
                         "starting the app to request a different context size.")
        elif connection.provider_name == "textgen":
            lines.append("Textgen uses its currently loaded model. Set **Textgen context length** to match its "
                         "loaded context; the API may not report that size. Connect again after "
                         "changing its model, loader, or adapters.")
            if choices and not all(config.supports_structured_output(model) for model in choices):
                lines.append("Studio generation needs a supported grammar-capable textgen loader. Load with llama.cpp and Connect again.")
            lines.append("Image inputs are unavailable when textgen does not expose verified vision support.")
        else:
            lines.append("KoboldCpp uses its currently loaded model and reports its loaded context automatically. "
                         "Set `KOBOLDCPP_CONTEXT_LENGTH` before starting the app for a lower context budget. "
                         "Connect again after changing the model or its loaded context.")
            lines.append("This connector supports text inputs and uses native grammar constraints for Studio tools. "
                         "RAG embeddings come from the separately selected server.")
            lines.append("Replies use the remaining context within KoboldCpp's native output limit "
                         "(up to 80% of the context for short prompts, otherwise 65%).")
        lines.append("Changing embedding connectors or models requires **Reindex selected** in RAG Chat. "
                     "Changing only the chat connector keeps compatible indexes usable.")
        return "\n\n".join(lines)


def select_connector(connector):
    """Changing a radio selection only updates the displayed endpoint."""
    return config.get_endpoint(connector)


def show_embedding_connector(connector):
    return gr.update(visible=connector in config.SEPARATE_EMBEDDING_PROVIDERS)


def show_textgen_context(connector):
    return gr.update(visible=connector == "textgen")


def connect_connection(connector, chat_model=None, rag_model=None, operation_state=None,
                       embedding_connector="none", textgen_context=None, request: gr.Request = None):
    """Adopt discovery only for the session and Connect action that requested it."""
    session = request.session_hash if request else None
    if not session:
        return "Open the app in a browser session before connecting.", gr.skip(), gr.skip()
    token = uuid4().hex
    with _SESSION_LOCK:
        old = _SESSIONS.get(session, {}).get("connection")
        _SESSIONS[session] = {"pending": token, "connection": None}
        if old is not None:
            config.release_connection(old.key)
    if operation_state is not None:
        from rag_chatbot.tools.ui import cancel_tool_job
        operation_state["connecting"] = True
        operation_state["answer_revision"] = operation_state.get("answer_revision", 0) + 1
        cancel_tool_job(operation_state)
    connection = None
    try:
        connection = config.connect(connector, check_embeddings=True,
                                    embedding_connector=embedding_connector if connector in config.SEPARATE_EMBEDDING_PROVIDERS else None,
                                    context_length=textgen_context if connector == "textgen" else None)
        status = _connection_status(connection)
        choices = list(connection.model_choices)
        # An identical model ID on another connector is a different selection.
        same_provider = old is not None and old.provider_name == connection.provider_name
        result = (status,
                  gr.update(choices=choices, value=chat_model if same_provider and chat_model in choices else connection.default_model or None),
                  gr.update(choices=choices, value=rag_model if same_provider and rag_model in choices else connection.default_model or None))
        with _SESSION_LOCK:
            if _SESSIONS.get(session, {}).get("pending") != token:
                config.release_connection(connection.key)
                return (gr.skip(),) * 3
            _SESSIONS[session] = {"connection": connection}
            if operation_state is not None:
                operation_state["connecting"] = False
            return result
    except (ProviderError, ValueError, RuntimeError) as exc:
        if connection is not None:
            config.release_connection(connection.key)
        with _SESSION_LOCK:
            if _SESSIONS.get(session, {}).get("pending") != token:
                return (gr.skip(),) * 3
            _SESSIONS[session] = {"connection": None}
            if operation_state is not None:
                operation_state["connecting"] = False
        logger.warning("Local connection failed: %s", exc)
        return f"Disconnected. Connection failed: {exc}", gr.update(choices=[], value=None), gr.update(choices=[], value=None)


def release_session(request: gr.Request):
    with _SESSION_LOCK:
        entry = _SESSIONS.pop(request.session_hash, {}) if request else {}
    if entry.get("connection") is not None:
        config.release_connection(entry["connection"].key)


def refresh_connected_library(workspace, collection, selected_ids, review_id, operation_state):
    """Refresh index compatibility after a connection change without inference."""
    from rag_chatbot import chatbot as rag_ui
    skipped = (gr.skip(),) * (4 + rag_ui.REVIEW_OUTPUT_COUNT)
    if not workspace or operation_state.get("switching"):
        return skipped
    connection = config.current_connection()
    if connection is None:
        return skipped
    result = rag_ui.on_refresh(workspace, collection, selected_ids, review_id, operation_state)
    try:
        config.ensure_connection_current()
    except RuntimeError:
        return skipped
    return result


def build_tab():
    gr.Markdown("Choose your local server, then press **Connect**. The app starts disconnected.")
    connector = gr.Radio(choices=list(config.PROVIDER_CHOICES), value="lmstudio", label="Connector")
    endpoint = gr.Textbox(label="Server URL", value=config.get_endpoint("lmstudio"), interactive=False)
    gr.Markdown("Selecting a radio button does not connect or switch the active connection. "
                "Use `LMSTUDIO_BASE_URL`, `OLLAMA_BASE_URL`, `TEXTGEN_BASE_URL`, or `KOBOLDCPP_BASE_URL` before starting the app for a different local address. "
                "Connect discovers models and checks an installed embedding model; it never downloads models.")
    embedding_connector = gr.Dropdown(label="RAG embeddings", value="none", visible=False,
        choices=[("None (chat only)", "none"), ("Ollama", "ollama"), ("LM Studio", "lmstudio")],
        info="Optional: Connect will also contact this explicitly selected server to check its installed embedding model. "
             "Ollama uses OLLAMA_BASE_URL; LM Studio uses LMSTUDIO_BASE_URL.")
    textgen_context = gr.Number(label="Textgen context length", value=config.get_context_default("textgen"),
        minimum=512, maximum=1048576, precision=0, visible=False,
        info="Enter the context length (n_ctx) used to load the model in textgen, then Connect. "
             "This sets the app's budget; it does not reload or resize the server model. "
             "TEXTGEN_CONTEXT_LENGTH sets the initial value.")
    connect = gr.Button("Connect", variant="primary")
    status = gr.Markdown("Disconnected. Choose a connector and press Connect.")
    connector.input(select_connector, connector, endpoint, queue=False)
    connector.input(show_embedding_connector, connector, embedding_connector, queue=False)
    connector.input(show_textgen_context, connector, textgen_context, queue=False, api_name=False)
    return SimpleNamespace(connector=connector, endpoint=endpoint, connect=connect, status=status,
                           embedding_connector=embedding_connector, textgen_context=textgen_context)
