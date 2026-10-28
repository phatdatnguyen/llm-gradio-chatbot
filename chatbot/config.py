"""Explicit local connections with immutable, callback-bound model metadata.

Importing configuration, reading a selector, and building the UI never construct
providers or contact servers. Only connect discovers and probes models. A session
keeps its returned key and binds that snapshot while running callbacks.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import os
import threading
from types import MappingProxyType
from uuid import uuid4


def _integer(name, default, minimum=1, maximum=1048576):
    try:
        result = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a whole number.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return result


PROVIDER_CHOICES = (("LM Studio", "lmstudio"), ("Ollama", "ollama"), ("textgen", "textgen"),
                    ("KoboldCpp", "koboldcpp"))
SEPARATE_EMBEDDING_PROVIDERS = frozenset({"textgen", "koboldcpp"})
BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
TEXTGEN_BASE_URL = os.getenv("TEXTGEN_BASE_URL", "http://127.0.0.1:5000/v1").rstrip("/")
KOBOLDCPP_BASE_URL = os.getenv("KOBOLDCPP_BASE_URL", "http://127.0.0.1:5001/v1").rstrip("/")
DEFAULT_CONTEXT_TOKENS = _integer("LMSTUDIO_CONTEXT_LENGTH", 4096, 512)
OUTPUT_TOKEN_LIMIT = _integer("LMSTUDIO_MAX_OUTPUT_TOKENS", 0, 0)
DEFAULT_IMAGE_TOKEN_BUDGET = _integer("LMSTUDIO_IMAGE_TOKEN_BUDGET", 4096, 256)
# Disconnected metadata also provides explicit, offline injection seams for tests.
# Environment-selected models apply only after successful discovery.
DEFAULT_MODEL = ""
DEFAULT_EMBEDDING_MODEL = ""
MODEL_CHOICES = []
MODEL_TOKEN_LIMITS = {}
MODEL_MAX_OUTPUT_TOKENS = {}
_MODEL_INFO = {}
_EMBEDDING_INFO = {}
_LOCK = threading.RLock()
_CONNECTIONS = {}
_EMBEDDING_DIMENSIONS = {}
_UNBOUND = object()
_CURRENT_CONNECTION = ContextVar("local_inference_connection", default=_UNBOUND)
_CONNECTION_RESOLVER = None


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Connection:
    key: str
    provider_name: str
    base_url: str
    provider: object = field(repr=False, compare=False)
    records: tuple
    model_choices: tuple
    default_model: str
    model_info: object
    context_limits: object
    output_limits: object
    embedding_settings: object
    embedding_error: str | None
    default_context: int
    configured_context: int | None
    output_limit: int
    image_budget: int
    revoked: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    embedding_provider: object = field(default=None, repr=False, compare=False)
    embedding_provider_name: str = ""
    embedding_base_url: str = ""

    @property
    def provider_label(self):
        return dict((name, label) for label, name in PROVIDER_CHOICES)[self.provider_name]


def _provider_name(name):
    value = str(name or "").strip().lower().replace(" ", "")
    if value not in {identifier for _, identifier in PROVIDER_CHOICES}:
        raise ValueError("Choose LM Studio, Ollama, textgen, or KoboldCpp before connecting.")
    return value


def get_endpoint(provider_name):
    return {"lmstudio": BASE_URL, "ollama": OLLAMA_BASE_URL, "textgen": TEXTGEN_BASE_URL,
            "koboldcpp": KOBOLDCPP_BASE_URL}[_provider_name(provider_name)]


def provider_options():
    """Return display metadata without constructing a client or contacting it."""
    return [{"name": name, "label": label, "base_url": get_endpoint(name)}
            for label, name in PROVIDER_CHOICES]


def get_context_default(provider_name):
    provider_name = _provider_name(provider_name)
    return (DEFAULT_CONTEXT_TOKENS if provider_name == "lmstudio" else
            _integer(provider_name.upper() + "_CONTEXT_LENGTH", 4096, 512))


def _settings(provider_name):
    prefix = provider_name.upper()
    context = get_context_default(provider_name)
    return {
        "prefix": prefix, "base_url": get_endpoint(provider_name),
        "context": context, "context_cap": context if prefix + "_CONTEXT_LENGTH" in os.environ else None,
        "output_limit": OUTPUT_TOKEN_LIMIT if provider_name == "lmstudio" else
                        _integer(prefix + "_MAX_OUTPUT_TOKENS", 0, 0),
        "image_budget": DEFAULT_IMAGE_TOKEN_BUDGET if provider_name == "lmstudio" else
                        _integer(prefix + "_IMAGE_TOKEN_BUDGET", 4096, 256),
        "model": os.getenv(prefix + "_MODEL", "").strip(),
        "embedding_model": os.getenv(prefix + "_EMBEDDING_MODEL", "").strip(),
        "embedding_dimensions": os.getenv(prefix + "_EMBEDDING_DIMENSIONS"),
        "document_prefix": os.getenv(prefix + "_EMBEDDING_DOCUMENT_PREFIX"),
        "query_prefix": os.getenv(prefix + "_EMBEDDING_QUERY_PREFIX"),
        "timeout": _integer(prefix + "_TIMEOUT", 900, 1, 3600),
    }


def _create_provider(provider_name, settings):
    if provider_name == "lmstudio":
        from .providers.lmstudio import LMStudioProvider
        provider_class = LMStudioProvider
    elif provider_name == "ollama":
        from .providers.ollama import OllamaProvider
        provider_class = OllamaProvider
    elif provider_name == "textgen":
        from .providers.textgen import TextGenProvider
        provider_class = TextGenProvider
    else:
        from .providers.koboldcpp import KoboldCppProvider
        provider_class = KoboldCppProvider
    return provider_class(settings["base_url"],
                          api_key=os.getenv(settings["prefix"] + "_API_KEY") or None,
                          timeout=settings["timeout"])


def set_connection_resolver(resolver):
    """Install the UI's session resolver without importing its framework here."""
    if resolver is not None and not callable(resolver):
        raise TypeError("The connection resolver must be callable.")
    global _CONNECTION_RESOLVER
    with _LOCK:
        _CONNECTION_RESOLVER = resolver


def current_connection():
    connection = _CURRENT_CONNECTION.get()
    if connection is not _UNBOUND:
        return connection
    with _LOCK:
        resolver = _CONNECTION_RESOLVER
    return resolver() if resolver is not None else None


def get_connection_key():
    connection = current_connection()
    return connection.key if connection is not None else ""


def get_connection(key):
    with _LOCK:
        connection = _CONNECTIONS.get(key)
    if connection is None or connection.revoked.is_set():
        raise RuntimeError("This connection is no longer active. Click Connect again.")
    return connection


@contextmanager
def use_connection(connection):
    """Bind one callback step, restoring the previous binding even on failure.

    Empty keys deliberately bind disconnected state so history and preview remain
    available. Inference enforces connection validity separately.
    """
    if connection and not isinstance(connection, Connection):
        connection = get_connection(connection)
    token = _CURRENT_CONNECTION.set(connection or None)
    try:
        yield connection
    finally:
        _CURRENT_CONNECTION.reset(token)


def release_connection(key):
    """Revoke an old session key without rerouting its active callbacks."""
    if isinstance(key, Connection):
        key = key.key
    with _LOCK:
        connection = _CONNECTIONS.pop(key, None)
        if connection is not None:
            connection.revoked.set()


def ensure_connection_current():
    connection = current_connection()
    if connection is not None and connection.revoked.is_set():
        raise RuntimeError("The connection changed while this operation was running. Start it again.")
    return connection


def get_provider():
    connection = ensure_connection_current()
    if connection is None:
        raise RuntimeError("Choose a local server in Connections and click Connect before using a model.")
    return connection.provider


def get_embedding_provider():
    """Return the explicitly selected embedding backend from the same snapshot."""
    connection = ensure_connection_current()
    if connection is None:
        return get_provider()
    if connection.embedding_provider is not None:
        return connection.embedding_provider
    if connection.provider_name in SEPARATE_EMBEDDING_PROVIDERS:
        raise RuntimeError(f"Choose a RAG embedding provider for {connection.provider_label} in Connections, then click Connect.")
    return connection.provider


def _embedding_settings(provider_name, options, records):
    embeddings = [item for item in records if item.get("type") == "embedding"]
    if provider_name == "lmstudio" and options["embedding_model"]:
        # The compatible /v1/models fallback has no model-type metadata. Keep
        # explicit embedding configuration usable with a discovered ID; the
        # connection probe verifies it. Never infer a model's role from its name.
        embeddings.extend(item for item in records
                          if item["id"] == options["embedding_model"] and item.get("type") == "unknown")
    selected = options["embedding_model"] or (embeddings[0]["id"] if len(embeddings) == 1 else "")
    record = next((item for item in embeddings if item["id"] == selected), {})
    error = None
    if selected and not record:
        error = f"{options['prefix']}_EMBEDDING_MODEL must name an installed embedding model."
        selected = ""
    identity = provider_name + ":" + options["base_url"] + ":" + str(record.get("identity") or selected)
    with _LOCK:
        dimensions = _EMBEDDING_DIMENSIONS.get(identity, 0) if selected else 0
    context = record.get("context_length") or record.get("max_context_length") or 2048
    if type(context) is not int or context < 128:
        context = 2048
    nomic = "nomic-embed-text" in selected
    return {
        "model": selected,
        "dimensions": _integer(options["prefix"] + "_EMBEDDING_DIMENSIONS", dimensions, 0, 65536),
        "context_length": _integer(options["prefix"] + "_EMBEDDING_CONTEXT_LENGTH", context, 128),
        "identity": identity,
        "document_prefix": (options["document_prefix"] if options["document_prefix"] is not None else
                            "search_document: " if nomic else ""),
        "query_prefix": (options["query_prefix"] if options["query_prefix"] is not None else
                         "search_query: " if nomic else ""),
    }, error


def _probe_embedding(provider, settings, options):
    from .providers.base import ProviderError
    vectors = provider.embeddings(settings["model"], [settings["query_prefix"] + "Local connection check."])
    if (not isinstance(vectors, list) or len(vectors) != 1 or not isinstance(vectors[0], list)
            or not vectors[0] or len(vectors[0]) > 65536
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vectors[0])
            or not any(vectors[0])):
        raise ProviderError("The embedding connection check returned an invalid vector.")
    dimensions = len(vectors[0])
    if options["embedding_dimensions"] is not None and settings["dimensions"] != dimensions:
        raise ValueError(options["prefix"] + "_EMBEDDING_DIMENSIONS does not match the selected embedding model.")
    settings["dimensions"] = dimensions
    with _LOCK:
        _EMBEDDING_DIMENSIONS[settings["identity"]] = dimensions


def _discover(provider):
    from .providers.base import ProviderError
    records = provider.discover_models()
    if (not isinstance(records, list) or any(not isinstance(item, dict)
            or not isinstance(item.get("id"), str) or not item["id"].strip() for item in records)):
        raise ProviderError("The local server returned invalid model metadata.")
    if len({item["id"] for item in records}) != len(records):
        raise ProviderError("The local server returned duplicate model identifiers.")
    return records


def connect(provider_name, *, check_embeddings=True, embedding_connector=None, context_length=None):
    """Discover one selected server and return a new independent connection.

    No global default is published. The caller adopts this key only if its own
    session still wants the result. An embedding failure leaves chat available
    while reporting the error and withholding a usable embedding dimension.
    """
    from .providers.base import ProviderError
    provider_name = _provider_name(provider_name)
    if embedding_connector not in (None, "none", "lmstudio", "ollama"):
        raise ValueError("Choose None, Ollama, or LM Studio for RAG embeddings.")
    if provider_name not in SEPARATE_EMBEDDING_PROVIDERS and embedding_connector not in (None, "none"):
        raise ValueError("A separate embedding connector is available only with textgen or KoboldCpp.")
    options = _settings(provider_name)
    if context_length is not None:
        if (provider_name != "textgen" or type(context_length) not in (int, float)
                or not 512 <= context_length <= 1048576
                or int(context_length) != context_length):
            raise ValueError("Textgen context length must be a whole number between 512 and 1048576.")
        options["context"] = options["context_cap"] = int(context_length)
    provider = _create_provider(provider_name, options)
    records = _discover(provider)
    embedding_provider, embedding_name, embedding_options = provider, provider_name, options
    if provider_name in SEPARATE_EMBEDDING_PROVIDERS:
        embedding_provider, embedding_name = None, ""
        embedding = {"model": "", "dimensions": 0, "context_length": 2048,
                     "identity": "", "document_prefix": "", "query_prefix": ""}
        embedding_error = None
        if embedding_connector in ("lmstudio", "ollama"):
            embedding_name = embedding_connector
            embedding_options = _settings(embedding_name)
            try:
                embedding_provider = _create_provider(embedding_name, embedding_options)
                embedding_records = _discover(embedding_provider)
                embedding, embedding_error = _embedding_settings(embedding_name, embedding_options, embedding_records)
            except (ProviderError, ValueError, RuntimeError) as exc:
                embedding_error = str(exc)
                embedding_provider = None
    else:
        embedding, embedding_error = _embedding_settings(provider_name, options, records)
    chats = [item for item in records if item.get("type") not in ("embedding", "decision")
             and item.get("available", item.get("loaded", True)) is not False
             and (provider_name in SEPARATE_EMBEDDING_PROVIDERS or item["id"] != embedding["model"])]
    model_info = {item["id"]: dict(item) for item in chats}
    choices = tuple(model_info)
    explicit = options["model"]
    if explicit and explicit not in choices:
        raise ValueError(f"{options['prefix']}_MODEL is not available from this server. Check it and reconnect.")
    context_limits, output_limits = {}, {}
    for model in choices:
        context = model_info[model].get("context_length")
        context = context if type(context) is int and context >= 512 else options["context"]
        if provider_name == "ollama" and options["context_cap"] is not None:
            # Ollama accepts num_ctx per request, so an explicit request may
            # exceed a previous loaded window without changing LM Studio's cap.
            context = options["context_cap"]
        elif options["context_cap"] is not None:
            context = min(context, options["context_cap"])
        maximum = model_info[model].get("max_context_length")
        if type(maximum) is int and maximum >= 512:
            context = min(context, maximum)
        context_limits[model] = context
        output_limits[model] = min(options["output_limit"], context) if options["output_limit"] else None
    if check_embeddings and embedding["model"]:
        try:
            _probe_embedding(embedding_provider, embedding, embedding_options)
        except (ProviderError, ValueError, RuntimeError) as exc:
            embedding["dimensions"] = 0
            embedding_error = str(exc)
    connection = Connection(
        key=uuid4().hex, provider_name=provider_name, base_url=options["base_url"], provider=provider,
        records=tuple(_freeze(item) for item in records), model_choices=choices,
        default_model=explicit or (choices[0] if choices else ""), model_info=_freeze(model_info),
        context_limits=_freeze(context_limits), output_limits=_freeze(output_limits),
        embedding_settings=_freeze(embedding), embedding_error=embedding_error,
        default_context=options["context"], configured_context=options["context_cap"],
        output_limit=options["output_limit"], image_budget=options["image_budget"],
        embedding_provider=embedding_provider, embedding_provider_name=embedding_name,
        embedding_base_url=embedding_options["base_url"] if embedding_name else "",
    )
    with _LOCK:
        _CONNECTIONS[connection.key] = connection
    return connection


def refresh_models(*, check_embeddings=False):
    """Explicit CLI compatibility: return a fresh snapshot for the caller to bind."""
    connection = current_connection()
    return connect(connection.provider_name if connection is not None else "lmstudio",
                   check_embeddings=check_embeddings,
                   embedding_connector=(connection.embedding_provider_name or None)
                   if connection is not None and connection.provider_name in SEPARATE_EMBEDDING_PROVIDERS else None,
                   context_length=connection.configured_context
                   if connection is not None and connection.provider_name == "textgen" else None)


def get_model_choices():
    connection = current_connection()
    with _LOCK:
        return list(connection.model_choices if connection is not None else MODEL_CHOICES)


def get_default_model():
    connection = current_connection()
    with _LOCK:
        return connection.default_model if connection is not None else DEFAULT_MODEL


def resolve_model(model=None):
    ensure_connection_current()
    selected = str(model or get_default_model() or "").strip()
    if not selected:
        raise RuntimeError("Choose a local server in Connections and click Connect, then select a chat model.")
    if selected not in get_model_choices():
        raise RuntimeError("That model is not available on this connection. Select a listed model or reconnect.")
    return selected


def get_embedding_settings():
    """Read bound immutable settings; disconnected browsing needs no server."""
    connection = current_connection()
    if connection is not None:
        return dict(connection.embedding_settings)
    with _LOCK:
        return {"model": DEFAULT_EMBEDDING_MODEL, "dimensions": _EMBEDDING_INFO.get("dimensions", 0),
                "context_length": 2048, "identity": "", "document_prefix": "", "query_prefix": ""}


def get_max_context_tokens(model_name):
    connection = current_connection()
    with _LOCK:
        return (connection.context_limits.get(model_name, connection.default_context) if connection is not None
                else MODEL_TOKEN_LIMITS.get(model_name, DEFAULT_CONTEXT_TOKENS))


def get_model_context_info(model_name):
    """Loaded and advertised windows differ; never assume the maximum is loaded."""
    connection = current_connection()
    with _LOCK:
        info = (connection.model_info if connection is not None else _MODEL_INFO).get(model_name, {})
        loaded, maximum = info.get("context_length"), info.get("max_context_length")
    return {
        "loaded": loaded if type(loaded) is int and loaded >= 512 else None,
        "maximum": maximum if type(maximum) is int and maximum > 0 else None,
        "effective": get_max_context_tokens(model_name),
        "configured": connection.configured_context if connection is not None else
                      DEFAULT_CONTEXT_TOKENS if "LMSTUDIO_CONTEXT_LENGTH" in os.environ else None,
    }


def max_output_tokens(model_name):
    connection = current_connection()
    with _LOCK:
        return (connection.output_limits.get(model_name, connection.output_limit or None) if connection is not None
                else MODEL_MAX_OUTPUT_TOKENS.get(model_name, OUTPUT_TOKEN_LIMIT or None))


def supports_structured_output(model_name):
    if not model_name or model_name not in get_model_choices():
        return False
    connection = current_connection()
    with _LOCK:
        info = (connection.model_info if connection is not None else _MODEL_INFO).get(model_name, {})
        return info.get("structured_output", connection is None or
                        connection.provider_name not in SEPARATE_EMBEDDING_PROVIDERS) is True


def supports_vision(model_name):
    connection = current_connection()
    with _LOCK:
        return bool((connection.model_info if connection is not None else _MODEL_INFO)
                    .get(model_name, {}).get("vision", False))


def image_token_budget(model_name, detail="auto", dimensions=None):
    connection = current_connection()
    return connection.image_budget if connection is not None else DEFAULT_IMAGE_TOKEN_BUDGET


def supports_streaming(model_name):
    return True


def check_model_tables():
    choices = get_model_choices()
    if len(choices) != len(set(choices)):
        raise ValueError("The local model list contains duplicate identifiers.")
    return True
