"""Text-generation-webui chat adapter with explicit discovery and prompt preflight.

The server ignores the request's model name and can silently shorten chat history.
Pin its loaded model/LoRAs, and check the full rendered prompt before generation.
The embeddings route is deliberately unused: its lazy loader can download models.
"""

import copy
import json
import math
import threading

import httpx

from .base import ProviderError
from .transport import LocalHTTPTransport, MAX_EVENT_BYTES, MAX_RESPONSE_BYTES


_PROMPT_RENDER_WINDOW = 2**31 - 1
_REQUIRED_PATHS = {"/v1/chat/completions": "post", "/v1/internal/chat-prompt": "post",
                   "/v1/internal/token-count": "post", "/v1/internal/model/info": "get"}


def _positive_int(value):
    return value if type(value) is int and value > 0 else None


class TextGenProvider(LocalHTTPTransport):
    server_name = "text-generation-webui"
    url_paths = ("", "/v1")

    def __init__(self, base_url, api_key=None, timeout=900):
        super().__init__(base_url, api_key=api_key, timeout=timeout)
        self._metadata_lock = threading.RLock()
        self._discovered = None
        self._structured = False

    @staticmethod
    def _model_metadata(info):
        name, loader, loras = info.get("model_name"), info.get("loader"), info.get("lora_names")
        if (name is not None and not isinstance(name, str)
                or not isinstance(loader, (str, type(None)))
                or not isinstance(loras, list) or any(not isinstance(value, str) or not value.strip() for value in loras)
                or len(set(loras)) != len(loras)):
            raise ProviderError("Textgen returned malformed loaded-model metadata.")
        name = name.strip() if isinstance(name, str) else ""
        return {"model_name": name if name != "None" else "", "loader": loader or "",
                "lora_names": tuple(loras), "context_length": _positive_int(info.get("context_length")),
                "max_context_length": _positive_int(info.get("max_context_length"))}

    def _read_model(self):
        return self._model_metadata(self._request_json("GET", "/v1/internal/model/info"))

    def discover_models(self):
        """Read API/loaded-model metadata without loading models or embeddings."""
        spec = self._request_json("GET", "/openapi.json")
        paths = spec.get("paths")
        if (not isinstance(paths, dict) or any(not isinstance(paths.get(path), dict)
                or method not in paths[path] for path, method in _REQUIRED_PATHS.items())):
            raise ProviderError("Textgen must expose its chat, model-info, chat-prompt, and token-count API endpoints.")
        components = spec.get("components")
        schemas = components.get("schemas") if isinstance(components, dict) else None
        chat_schema = schemas.get("ChatCompletionRequest") if isinstance(schemas, dict) else None
        properties = chat_schema.get("properties", {}) if isinstance(chat_schema, dict) else {}
        if not isinstance(properties, dict) or not {"truncation_length", "auto_max_new_tokens"}.issubset(properties):
            raise ProviderError("Textgen does not advertise the required explicit context controls. Update the server.")
        info = self._read_model()
        with self._metadata_lock:
            self._discovered = info
            self._structured = info["loader"] == "llama.cpp" and "grammar_string" in properties
        if not info["model_name"]:
            return []
        return [{"id": info["model_name"], "type": "llm", "loaded": True, "available": True,
                 "context_length": info["context_length"], "max_context_length": info["max_context_length"],
                 "vision": False, "structured_output": self._structured,
                 "identity": json.dumps(info, sort_keys=True, separators=(",", ":"))}]

    def _require_current(self, model):
        with self._metadata_lock:
            expected = self._discovered
        if expected is None:
            raise ProviderError("Connect to Textgen before using its loaded model.")
        if not model or model != expected["model_name"]:
            raise ProviderError("That model is not the connected Textgen model. Load it in Textgen and Connect again.")
        if self._read_model() != expected:
            raise ProviderError("The loaded Textgen model, LoRAs, or context changed. Connect again before generating.")
        return expected

    @staticmethod
    def prepare_chat_payload(payload, *, context_length):
        if _positive_int(context_length) is None:
            raise ValueError("The Textgen request requires a positive context length.")
        return {**payload, "context_length": context_length}

    @staticmethod
    def _messages(messages):
        if not isinstance(messages, list) or not messages:
            raise ValueError("Enter a message before requesting a Textgen response.")
        result = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Textgen chat contains an unsupported message role.")
            content = message.get("content")
            if isinstance(content, list):
                if any(not isinstance(part, dict) or part.get("type") != "text"
                       or not isinstance(part.get("text"), str) for part in content):
                    raise ValueError("Textgen does not expose verified vision capability. Use text or a connected vision model.")
                content = "\n".join(part["text"] for part in content)
            if not isinstance(content, str):
                raise ValueError("Textgen chat contains invalid message text.")
            result.append({"role": message["role"], "content": content})
        return result

    def _chat_request(self, payload, *, stream):
        context, output = payload.get("context_length"), payload.get("max_tokens")
        if _positive_int(context) is None or _positive_int(output) is None or output >= context:
            raise ValueError("Textgen requires explicit positive context and output budgets that fit together.")
        messages = self._messages(payload.get("messages"))
        request = {"model": payload.get("model"), "messages": messages, "stream": stream,
                   "max_tokens": output, "truncation_length": context, "auto_max_new_tokens": False,
                   "mode": "instruct", "tool_choice": "none"}
        for name, upper in (("temperature", 2), ("top_p", 1)):
            if name in payload:
                value = payload[name]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
                    raise ValueError(f"Textgen {name} must be between 0 and {upper}.")
                request[name] = value
        if "response_format" in payload:
            with self._metadata_lock:
                supported = self._structured
            if not supported:
                raise ProviderError("This Textgen loader does not expose supported schema-constrained generation.")
            response_format = payload["response_format"]
            contract = response_format.get("json_schema") if isinstance(response_format, dict) else None
            if (not isinstance(response_format, dict) or response_format.get("type") != "json_schema"
                    or not isinstance(contract, dict) or not isinstance(contract.get("schema"), dict)):
                raise ValueError("Textgen structured generation requires a JSON schema.")
            from .schema_grammar import schema_to_grammar
            request["grammar_string"] = schema_to_grammar(contract["schema"])
        info = self._require_current(request["model"])
        for maximum in (info["context_length"], info["max_context_length"]):
            if maximum is not None and context > maximum:
                raise ProviderError("The app context exceeds the reported Textgen context. Connect again or lower TEXTGEN_CONTEXT_LENGTH.")
        self._preflight_prompt(request)
        # Neither a template probe nor another session may redirect this request.
        self._require_current(request["model"])
        return request

    def _preflight_prompt(self, request):
        # This endpoint only renders/tokenizes; the large window prevents its
        # usual history truncation. Inference still receives the bound app window.
        probe = {**request, "stream": False, "truncation_length": _PROMPT_RENDER_WINDOW}
        probe.pop("grammar_string", None)
        prompt = self._request_json("POST", "/v1/internal/chat-prompt", probe).get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ProviderError("Textgen returned an invalid rendered chat prompt.")
        position = 0
        for message in request["messages"]:
            text = message["content"].strip()
            if text:
                found = prompt.find(text, position)
                if found < 0:
                    raise ProviderError("The Textgen chat template omitted or changed message text. Choose a template that preserves the complete input.")
                position = found + len(text)
        count = self._request_json("POST", "/v1/internal/token-count", {"text": prompt}).get("length")
        if type(count) is not int or count < 0:
            raise ProviderError("Textgen returned an invalid prompt token count.")
        remaining = request["truncation_length"] - count
        if remaining < request["max_tokens"]:
            if remaining <= 0 or "grammar_string" in request:
                raise ProviderError("The complete Textgen prompt and reply reserve exceed the app context. "
                                    "Reduce the input or set TEXTGEN_CONTEXT_LENGTH to the model's loaded window, then reconnect.")
            # Ordinary answers can use a smaller remaining allowance when the
            # native template adds more framing than the offline estimate. Never
            # trim evidence or lower a structured tool's explicit output reserve.
            request["max_tokens"] = remaining

    @staticmethod
    def _checked_result(result, request):
        if result.get("model") != request["model"]:
            raise ProviderError("Textgen answered with a different model. Connect again; the reply was not accepted.")
        choices = result.get("choices")
        if not isinstance(choices, list):
            raise ProviderError("Textgen returned malformed response choices.")
        result = copy.deepcopy(result)
        usage = result.get("usage")
        if isinstance(usage, dict):
            completed, prompt = usage.get("completion_tokens"), usage.get("prompt_tokens")
            limited = (type(completed) is int and completed >= request["max_tokens"]
                       or type(completed) is int and type(prompt) is int
                       and prompt + completed >= request["truncation_length"])
            if limited:
                for choice in result["choices"]:
                    if isinstance(choice, dict) and choice.get("finish_reason") == "stop":
                        choice["finish_reason"] = "length"
        return result

    def chat(self, payload):
        request = self._chat_request(payload, stream=False)
        result = self._request_json("POST", "/v1/chat/completions", request)
        self._require_current(request["model"])
        return self._checked_result(result, request)

    @staticmethod
    def _sse_events(response):
        yield from LocalHTTPTransport._iter_sse(response, max_event_bytes=MAX_EVENT_BYTES,
                                               max_response_bytes=MAX_RESPONSE_BYTES)

    def stream_chat(self, payload):
        request = self._chat_request(payload, stream=True)
        finished = False
        try:
            with self._client() as client:
                with client.stream("POST", self.base_url + "/v1/chat/completions", json=request) as response:
                    if not response.is_success:
                        self._check_status(response, self._read_bounded(response))
                    for raw in self._sse_events(response):
                        if raw is None:
                            if not finished:
                                raise ProviderError("Textgen ended its stream without a completion status.")
                            return
                        result = self._checked_result(raw, request)
                        terminal = any(isinstance(choice, dict) and choice.get("finish_reason") is not None
                                       for choice in result["choices"])
                        if terminal:
                            self._require_current(request["model"])
                            finished = True
                        yield result
                    if not finished:
                        raise ProviderError("The Textgen response stream ended before the model finished.")
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

    def embeddings(self, model, texts):
        raise ProviderError("Textgen's native embedding loader may download models and cannot be verified through its API. "
                            "Choose a separate installed local embedding connector for RAG.")
