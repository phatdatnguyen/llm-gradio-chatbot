"""RAG settings; model generation capabilities remain in chatbot.config."""

import os
import hashlib
import json
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from chatbot import config as model_config


_DEFAULT_QDRANT_PATH = object()
_DEFAULT_EMBEDDING = object()


def _vision_model():
    connection = model_config.current_connection()
    provider = connection.provider_name.upper() if connection is not None else "LMSTUDIO"
    return os.getenv(f"{provider}_VISION_MODEL", model_config.get_default_model())


def _qdrant_url():
    value = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    parsed = urlsplit(value)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        local = False
    if (not local or parsed.scheme not in {"http", "https"} or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("QDRANT_URL must point to a loopback HTTP(S) server for this local-only app.")
    return value


def _qdrant_local_path():
    if os.getenv("QDRANT_LOCAL_PATH"):
        return os.environ["QDRANT_LOCAL_PATH"]
    if os.getenv("QDRANT_URL"):
        return None
    return _DEFAULT_QDRANT_PATH


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(
        os.getenv("RAG_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
    ).resolve())
    qdrant_url: str = field(default_factory=_qdrant_url)
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_local_path: str | None = field(default_factory=_qdrant_local_path)
    embedding_model: str = field(default=_DEFAULT_EMBEDDING)
    embedding_dimensions: int = field(default=_DEFAULT_EMBEDDING)
    vision_model: str = field(default_factory=_vision_model)
    chunk_size: int = 600
    chunk_overlap: int = 80
    candidate_count: int = 20
    evidence_count: int = 8
    evidence_token_budget: int = 6000
    # Workspace identity isolates Qdrant collections without changing extraction
    # fingerprints. Empty keeps direct, standalone LibraryStore use compatible.
    namespace: str = ""
    embedding_identity: str = field(default=_DEFAULT_EMBEDDING)
    embedding_context_length: int = field(default=_DEFAULT_EMBEDDING)
    document_prefix: str = field(default=_DEFAULT_EMBEDDING)
    query_prefix: str = field(default=_DEFAULT_EMBEDDING)

    def __post_init__(self):
        fields = {"embedding_model": ("model", ""), "embedding_dimensions": ("dimensions", 0),
                  "embedding_identity": ("identity", "disconnected"),
                  "embedding_context_length": ("context_length", 2048),
                  "document_prefix": ("document_prefix", ""), "query_prefix": ("query_prefix", "")}
        if any(getattr(self, name) is _DEFAULT_EMBEDDING for name in fields):
            # Read one connection snapshot, never mix discovery/probe revisions.
            embedding = model_config.get_embedding_settings()
            for name, (option, default) in fields.items():
                if getattr(self, name) is _DEFAULT_EMBEDDING:
                    object.__setattr__(self, name, embedding.get(option, default))
        if self.qdrant_local_path is _DEFAULT_QDRANT_PATH:
            # Honor an explicit data_dir just as we honor RAG_DATA_DIR. Keeping
            # this under the catalog root also isolates sibling standalone stores.
            object.__setattr__(self, "qdrant_local_path", str(Path(self.data_dir).resolve() / ".cache" / "qdrant"))

    def fingerprint(self):
        # Never reuse cloud embeddings, or vectors made with different local
        # model/prefix/chunking settings, even when vector dimensions coincide.
        identity = {
            "provider": self.embedding_identity, "model": self.embedding_model,
            "dimensions": self.embedding_dimensions, "context": self.embedding_context_length,
            "document_prefix": self.document_prefix, "query_prefix": self.query_prefix,
            "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
            "tokenizer": "utf8-conservative-v1", "preprocessing": "local-prefix-v1",
        }
        return "local-v2:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
