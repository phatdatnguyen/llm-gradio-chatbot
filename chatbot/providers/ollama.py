"""Native Ollama adapter; installed local models only, with no download requests.

The facade binds its effective context with ``prepare_chat_payload``. Native
``num_ctx`` and ``num_predict`` then use that same budget. Metadata requests do
not load models; inference may load an already installed model. Ollama response
fields are normalized at this boundary so callers retain their strict completion
and structured-output checks.
"""

import base64
import binascii
import copy
import math
import threading

import httpx

from .base import ProviderError
from .transport import LocalHTTPTransport


def _positive_int(value):
    return value if type(value) is int and value > 0 else None


def _remote(metadata):
    return bool(metadata.get("remote_model") or metadata.get("remote_host"))


def _cloud_reference(name):
    # Ollama treats these source suffixes as explicit upstream routing even for
    # /api/show. They must be rejected before making that metadata request.
    tail = name.rsplit("/", 1)[-1]
    tag = tail.rsplit(":", 1)[-1].strip().lower() if ":" in tail else ""
    return tag == "cloud" or tag.endswith("-cloud")


class OllamaProvider(LocalHTTPTransport):
    server_name = "Ollama"
    url_paths = ("", "/api", "/v1")

    def __init__(self, base_url, api_key=None, timeout=900):
        super().__init__(base_url, api_key=api_key, timeout=timeout)
        self._metadata_lock = threading.RLock()
        self._discovered_identities = None

    def _local_request(self, path, payload):
        try:
            return self._request_json("POST", path, payload)
        except ProviderError as exc:
            message = str(exc).lower()
            if (exc.status_code == 404 or (exc.status_code == 400 and "model" in message
                                          and ("invalid" in message or "not found" in message))):
                raise ProviderError(
                    f"{exc}. Reconnect to check the installed model; "
                    "an older Ollama server may need an update for :local routing. No fallback was attempted.",
                    status_code=exc.status_code,
                ) from exc
            raise

    def _installed_models(self):
        rows = self._request_json("GET", "/api/tags").get("models")
        if not isinstance(rows, list):
            raise ProviderError("Ollama returned a malformed installed model list.")
        installed = {}
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                    or not row["name"].strip() or row["name"] in installed):
                raise ProviderError("Ollama returned an invalid or duplicate model identifier.")
            installed[row["name"]] = row
        return installed

    def _show_local(self, name, installed):
        if _remote(installed) or _cloud_reference(name):
            raise ProviderError("Cloud-backed and remote-proxy Ollama models are unavailable in this local app.")
        result = self._local_request("/api/show", {"model": name + ":local"})
        if _remote(result):
            raise ProviderError("Cloud-backed and remote-proxy Ollama models are unavailable in this local app.")
        capabilities = result.get("capabilities")
        if (not isinstance(capabilities, list)
                or any(not isinstance(value, str) for value in capabilities)):
            raise ProviderError("Ollama did not provide model capabilities. Update Ollama and refresh models.")
        return result

    @staticmethod
    def _maximum_context(details):
        info = details.get("model_info") or {}
        if not isinstance(info, dict):
            raise ProviderError("Ollama returned malformed model context metadata.")
        architecture = info.get("general.architecture")
        if isinstance(architecture, str):
            maximum = _positive_int(info.get(f"{architecture}.context_length"))
            if maximum is not None:
                return maximum
        candidates = [_positive_int(value) for key, value in info.items()
                      if isinstance(key, str) and key.endswith(".context_length")]
        return min((value for value in candidates if value is not None), default=None)

    def discover_models(self):
        """Inspect installed files and running contexts without loading any model."""
        installed = self._installed_models()
        running = self._request_json("GET", "/api/ps").get("models")
        if not isinstance(running, list) or any(not isinstance(row, dict) for row in running):
            raise ProviderError("Ollama returned a malformed running model list.")
        loaded = {row.get("name", row.get("model")): row for row in running
                  if isinstance(row.get("name", row.get("model")), str)}
        records = []
        for name, row in installed.items():
            # Remote stubs are filtered before /api/show as well as before inference.
            if _remote(row) or _cloud_reference(name):
                continue
            details = self._show_local(name, row)
            capabilities = details["capabilities"]
            if "completion" not in capabilities and "embedding" not in capabilities:
                continue
            digest = row.get("digest")
            if not isinstance(digest, str) or not digest.strip():
                raise ProviderError("Ollama did not provide the installed model digest. Refresh models.")
            current = loaded.get(name)
            records.append({
                "id": name, "type": "llm" if "completion" in capabilities else "embedding",
                "context_length": _positive_int(current.get("context_length")) if current else None,
                "max_context_length": self._maximum_context(details),
                "vision": "vision" in capabilities, "loaded": current is not None,
                "available": True, "identity": digest,
            })
        with self._metadata_lock:
            self._discovered_identities = {row["id"]: row["identity"] for row in records}
        return records

    def _require_local_model(self, model, capability):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Select an installed local Ollama model.")
        if _cloud_reference(model):
            raise ProviderError("Cloud-backed and remote-proxy Ollama models are unavailable in this local app.")
        installed = self._installed_models()
        name = model
        if name not in installed and ":" not in name.rsplit("/", 1)[-1]:
            name += ":latest"
        if name not in installed:
            raise ProviderError("That model is not installed locally in Ollama. Install it yourself and refresh models.")
        with self._metadata_lock:
            identities = self._discovered_identities
            if identities is not None and identities.get(name) != installed[name].get("digest"):
                raise ProviderError("The installed Ollama model changed after connecting. Reconnect before using it.")
        details = self._show_local(name, installed[name])
        if capability not in details["capabilities"]:
            raise ProviderError(f"The selected Ollama model does not support {capability}.")
        return name, details

    @staticmethod
    def prepare_chat_payload(payload, *, context_length):
        """Bind the app's already-preflighted context; never mutate caller data."""
        if _positive_int(context_length) is None:
            raise ValueError("The Ollama request requires a positive context length.")
        return {**payload, "context_length": context_length}

    @staticmethod
    def _messages(messages):
        if not isinstance(messages, list) or not messages:
            raise ValueError("Enter a message before requesting an Ollama response.")
        result = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Ollama chat contains an unsupported message role.")
            content = message.get("content")
            native = {"role": message["role"]}
            if isinstance(content, str):
                native["content"] = content
            elif isinstance(content, list):
                texts, images = [], []
                for part in content:
                    if not isinstance(part, dict):
                        raise ValueError("Ollama chat contains an invalid content part.")
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                    elif part.get("type") == "image_url":
                        image = part.get("image_url")
                        url = image.get("url") if isinstance(image, dict) else image
                        if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                            raise ValueError("Ollama images must be uploaded local image data, never URLs or file paths.")
                        encoded = url.split(";base64,", 1)[1]
                        try:
                            decoded = base64.b64decode(encoded, validate=True)
                        except (ValueError, binascii.Error) as exc:
                            raise ValueError("Ollama received invalid local image data.") from exc
                        if not decoded:
                            raise ValueError("Ollama received empty local image data.")
                        images.append(encoded)
                    else:
                        raise ValueError("Ollama chat contains an unsupported content part.")
                native["content"] = "\n".join(texts)
                if images:
                    native["images"] = images
            else:
                raise ValueError("Ollama chat contains invalid message text.")
            result.append(native)
        return result

    def _chat_request(self, payload, *, stream):
        context = payload.get("context_length")
        output = payload.get("max_tokens")
        if _positive_int(context) is None or _positive_int(output) is None or output > context:
            raise ValueError("Ollama requires explicit positive context and output budgets that fit together.")
        messages = self._messages(payload.get("messages"))
        options = {"num_ctx": context, "num_predict": output}
        for name, upper in (("temperature", 2), ("top_p", 1)):
            if name in payload:
                value = payload[name]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
                    raise ValueError(f"Ollama {name} must be between 0 and {upper}.")
                options[name] = value
        request = {"messages": messages, "options": options, "stream": stream,
                   "truncate": False, "shift": False}
        if "response_format" in payload:
            response_format = payload["response_format"]
            schema = response_format.get("json_schema") if isinstance(response_format, dict) else None
            if (not isinstance(response_format, dict) or response_format.get("type") != "json_schema"
                    or not isinstance(schema, dict) or not isinstance(schema.get("schema"), dict)):
                raise ValueError("Ollama structured generation requires a JSON schema.")
            request["format"] = copy.deepcopy(schema["schema"])
        name, details = self._require_local_model(payload.get("model"), "completion")
        maximum = self._maximum_context(details)
        if maximum is not None and context > maximum:
            raise ProviderError("The requested context exceeds the Ollama model's advertised maximum. Refresh models.")
        if any(message.get("images") for message in messages) and "vision" not in details["capabilities"]:
            raise ProviderError("The selected Ollama model cannot read images.")
        request["model"] = name + ":local"
        return request

    @staticmethod
    def _normalize(result, *, streaming):
        if _remote(result):
            raise ProviderError("Ollama returned a remote-model response; local inference is required.")
        if type(result.get("done")) is not bool or not isinstance(result.get("message"), dict):
            raise ProviderError("Ollama returned a malformed chat response.")
        source = result["message"]
        message = {}
        for original, normalized in (("content", "content"), ("thinking", "reasoning_content")):
            if original in source:
                if not isinstance(source[original], str):
                    raise ProviderError("Ollama returned invalid message text.")
                message[normalized] = source[original]
        reason = result.get("done_reason") if result["done"] else None
        if reason is not None and not isinstance(reason, str):
            raise ProviderError("Ollama returned an invalid completion status.")
        if source.get("tool_calls"):
            reason = "tool_calls"
        normalized = {"choices": [{"finish_reason": reason,
                                    "delta" if streaming else "message": message}]}
        if result["done"] and all(type(result.get(key)) is int and result[key] >= 0
                                  for key in ("prompt_eval_count", "eval_count")):
            normalized["usage"] = {"prompt_tokens": result["prompt_eval_count"],
                                   "completion_tokens": result["eval_count"],
                                   "total_tokens": result["prompt_eval_count"] + result["eval_count"]}
        return normalized

    def chat(self, payload):
        request = self._chat_request(payload, stream=False)
        result = self._local_request("/api/chat", request)
        return self._normalize(result, streaming=False)

    def stream_chat(self, payload):
        """Normalize native NDJSON and close the response on completion or Stop."""
        request = self._chat_request(payload, stream=True)
        try:
            with self._client() as client:
                with client.stream("POST", self.base_url + "/api/chat", json=request) as response:
                    if not response.is_success:
                        self._check_status(response, self._read_bounded(response))
                    for event in self._iter_ndjson(response):
                        yield self._normalize(event, streaming=True)
                        if event["done"]:
                            return
                    raise ProviderError("The Ollama response stream ended before the model finished.")
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

    def embeddings(self, model, texts):
        if (not isinstance(texts, list) or not texts
                or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError("Ollama embeddings require a nonempty list of text inputs.")
        name, _ = self._require_local_model(model, "embedding")
        result = self._local_request("/api/embed", {"model": name + ":local", "input": texts, "truncate": False})
        if _remote(result):
            raise ProviderError("Ollama returned remote embeddings; local inference is required.")
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ProviderError("The embedding response did not match the input batch.")
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
