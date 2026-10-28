"""LM Studio discovery and wire formats using the shared bounded local transport."""

import math

import httpx

from .base import ProviderError
from .transport import (LocalHTTPTransport, MAX_EVENT_BYTES, MAX_RESPONSE_BYTES,
                        validate_base_url as _validate_base_url)


def validate_base_url(base_url):
    """Retain the public LM Studio URL normalizer for existing callers."""
    return _validate_base_url(base_url, paths=LMStudioProvider.url_paths)


class LMStudioProvider(LocalHTTPTransport):
    """One immutable connection configuration; clients are scoped to each request."""

    server_name = "LM Studio"
    url_paths = ("", "/v1", "/api/v1")

    def discover_models(self):
        """Prefer native loaded-context metadata; fall back only for absent endpoints."""
        try:
            result = self._request_json("GET", "/api/v1/models")
        except ProviderError as exc:
            if exc.status_code not in {404, 405, 501}:
                raise
            result = self._request_json("GET", "/v1/models")
            rows = result.get("data")
            if not isinstance(rows, list):
                raise ProviderError("The local model list is malformed.")
            models = []
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                    raise ProviderError("The local model list contains an invalid identifier.")
                models.append({"id": row["id"], "type": "unknown", "context_length": None,
                               "vision": False, "loaded": None, "max_context_length": None,
                               "identity": row["id"]})
            return models
        rows = result.get("models")
        if not isinstance(rows, list):
            raise ProviderError("The native local model list is malformed.")
        models = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("key"), str) or not row["key"].strip():
                raise ProviderError("The native model list contains an invalid identifier.")
            capabilities = row.get("capabilities") or {}
            if not isinstance(capabilities, dict):
                raise ProviderError("The local model capabilities are malformed.")
            instances = row.get("loaded_instances", [])
            if not isinstance(instances, list):
                raise ProviderError("The local model instances are malformed.")
            for instance in instances or [{}]:
                if not isinstance(instance, dict):
                    raise ProviderError("The local model instance is malformed.")
                identifier = instance.get("id", row["key"])
                settings = instance.get("config") or {}
                if not isinstance(identifier, str) or not identifier.strip() or not isinstance(settings, dict):
                    raise ProviderError("The local model instance metadata is malformed.")
                context = settings.get("context_length")
                if type(context) is not int or context <= 0:
                    context = None
                quantization = row.get("quantization") or {}
                quantization_name = quantization.get("name", "") if isinstance(quantization, dict) else ""
                identity = row.get("selected_variant") or f"{row['key']}@{quantization_name}"
                maximum = row.get("max_context_length")
                if type(maximum) is not int or maximum <= 0:
                    maximum = None
                models.append({"id": identifier, "type": row.get("type", "unknown"),
                               "context_length": context, "vision": capabilities.get("vision") is True,
                               "loaded": bool(instance), "max_context_length": maximum,
                               "identity": identity})
        return models

    def chat(self, payload):
        return self._request_json("POST", "/v1/chat/completions", {**payload, "stream": False})

    def stream_chat(self, payload):
        """Yield decoded SSE objects; closing the generator closes its HTTP stream."""
        try:
            with self._client() as client:
                with client.stream("POST", self.base_url + "/v1/chat/completions",
                                   json={**payload, "stream": True}) as response:
                    if not response.is_success:
                        self._check_status(response, self._read_bounded(response))
                    for event in self._iter_sse(response, max_event_bytes=MAX_EVENT_BYTES,
                                                max_response_bytes=MAX_RESPONSE_BYTES):
                        if event is None:
                            return
                        yield event
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

    def embeddings(self, model, texts):
        result = self._request_json("POST", "/v1/embeddings", {"model": model, "input": texts})
        rows = result.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ProviderError("The embedding response did not match the input batch.")
        if any(not isinstance(row, dict) or type(row.get("index")) is not int for row in rows):
            raise ProviderError("The embedding response contains invalid indexes.")
        rows = sorted(rows, key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ProviderError("The embedding response did not match the input batch.")
        vectors = [row.get("embedding") for row in rows]
        dimensions = None
        for vector in vectors:
            if (not isinstance(vector, list) or not vector
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                    or not any(vector)):
                raise ProviderError("The local embedding response contained invalid vectors.")
            if dimensions is not None and len(vector) != dimensions:
                raise ProviderError("The local embedding response changed vector dimensions.")
            dimensions = len(vector)
        return vectors
