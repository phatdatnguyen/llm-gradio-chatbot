"""KoboldCpp's loaded-model chat API, with native prompt and grammar checks.

Only metadata, tokenization, and generation endpoints are used. Both public chat
interfaces use one native SSE request: KoboldCpp's nonstreaming parser can put
unfinished thinking in content, whereas its stream separates reasoning deltas.
"""

import json
import math
import re
import threading

import httpx

from .base import ProviderError
from .transport import LocalHTTPTransport, MAX_EVENT_BYTES, MAX_RESPONSE_BYTES


_MIN_VERSION = (1, 123, 1)
# The native backend can add model-specific framing after the Python tokenizer
# endpoint has rendered the chat template. Reserve room without dropping input.
_NATIVE_FRAMING_RESERVE = 32


def _positive_int(value):
    return value if type(value) is int and value > 0 else None


class KoboldCppProvider(LocalHTTPTransport):
    server_name = "KoboldCpp"
    url_paths = ("", "/v1")

    def __init__(self, base_url, api_key=None, timeout=900):
        super().__init__(base_url, api_key=api_key, timeout=timeout)
        self._metadata_lock = threading.RLock()
        self._discovered = None

    def _read_model(self):
        capabilities = self._request_json("GET", "/api/extra/version")
        version = capabilities.get("version")
        if (capabilities.get("result") != "KoboldCpp" or not isinstance(version, str)
                or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version)):
            raise ProviderError("The server did not return valid KoboldCpp version metadata.")
        parsed = tuple(int(part) for part in version.split("."))
        if (parsed + (0,) * (3 - len(parsed))) < _MIN_VERSION:
            raise ProviderError("This connector requires KoboldCpp 1.123.1 or newer. Update the local server.")
        for flag in ("llm", "router", "defaults_modified", "jinja"):
            if type(capabilities.get(flag)) is not bool:
                raise ProviderError("KoboldCpp did not report its required generation capabilities.")
        if capabilities["router"]:
            raise ProviderError("Use a direct KoboldCpp model server with Router Mode disabled, then Connect again.")
        if capabilities["defaults_modified"]:
            raise ProviderError("Clear KoboldCpp's custom Generation Defaults, then Connect again. "
                                "Those defaults can override the complete prompt, context budget, or required grammar.")
        model = self._request_json("GET", "/api/v1/model").get("result")
        if model == "koboldcpp/protected-model":
            raise ProviderError("KoboldCpp requires authentication. Set KOBOLDCPP_API_KEY and restart the app.")
        context = self._request_json("GET", "/api/extra/true_max_context_length").get("value")
        if not isinstance(model, str) or not model.strip():
            raise ProviderError("KoboldCpp returned invalid loaded-model metadata.")
        if type(context) is not int or not 512 <= context <= 1048576:
            raise ProviderError("KoboldCpp did not report a valid loaded context length.")
        return {"model": model, "context_length": context, "version": version,
                "loaded": capabilities["llm"] and model not in ("inactive", "koboldcpp/inactive"),
                "jinja": capabilities["jinja"]}

    def discover_models(self):
        """Discover the already loaded model without making inference requests."""
        info = self._read_model()
        with self._metadata_lock:
            self._discovered = info
        if not info["loaded"]:
            return []
        return [{"id": info["model"], "type": "llm", "loaded": True, "available": True,
                 "context_length": info["context_length"], "max_context_length": None,
                 "vision": False, "structured_output": True,
                 "identity": json.dumps(info, sort_keys=True, separators=(",", ":"))}]

    def _require_current(self, model):
        with self._metadata_lock:
            expected = self._discovered
        if expected is None:
            raise ProviderError("Connect to KoboldCpp before using its loaded model.")
        if not expected["loaded"] or not model or model != expected["model"]:
            raise ProviderError("That model is not the connected KoboldCpp model. Load it in KoboldCpp and Connect again.")
        if self._read_model() != expected:
            raise ProviderError("The loaded KoboldCpp model, context, or server settings changed. Connect again before generating.")
        return expected

    @staticmethod
    def prepare_chat_payload(payload, *, context_length):
        if _positive_int(context_length) is None:
            raise ValueError("The KoboldCpp request requires a positive context length.")
        return {**payload, "context_length": context_length}

    @staticmethod
    def _messages(messages):
        if not isinstance(messages, list) or not messages:
            raise ValueError("Enter a message before requesting a KoboldCpp response.")
        result = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("KoboldCpp chat contains an unsupported message role.")
            content = message.get("content")
            if isinstance(content, list):
                if any(not isinstance(part, dict) or part.get("type") != "text"
                       or not isinstance(part.get("text"), str) for part in content):
                    raise ValueError("KoboldCpp vision is unavailable in this connector. Use text or another connected vision model.")
                content = "\n".join(part["text"] for part in content)
            if not isinstance(content, str):
                raise ValueError("KoboldCpp chat contains invalid message text.")
            result.append({"role": message["role"], "content": content})
        return result

    def _chat_request(self, payload):
        context, output = payload.get("context_length"), payload.get("max_tokens")
        if _positive_int(context) is None or _positive_int(output) is None or output >= context:
            raise ValueError("KoboldCpp requires explicit positive context and output budgets that fit together.")
        request = {"model": payload.get("model"), "messages": self._messages(payload.get("messages")),
                   "stream": True, "stream_options": {"include_usage": True},
                   "max_tokens": output, "max_context_length": context,
                   "tools": [], "tool_choice": "none", "encapsulate_thinking": True,
                   "grammar_retain_state": False, "replace_instruct_placeholders": False}
        for name, upper in (("temperature", 2), ("top_p", 1)):
            if name in payload:
                value = payload[name]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
                    raise ValueError(f"KoboldCpp {name} must be between 0 and {upper}.")
                request[name] = value
        if "response_format" in payload:
            response_format = payload["response_format"]
            contract = response_format.get("json_schema") if isinstance(response_format, dict) else None
            if (not isinstance(response_format, dict) or response_format.get("type") != "json_schema"
                    or not isinstance(contract, dict) or not isinstance(contract.get("schema"), dict)):
                raise ValueError("KoboldCpp structured generation requires a JSON schema.")
            from .schema_grammar import schema_to_grammar
            request["grammar"] = schema_to_grammar(contract["schema"])
            instructions = ("Return only one JSON object matching this schema. Follow its field descriptions "
                            "and exact allowed values. Do not include Markdown or additional commentary.\n"
                            + json.dumps(contract["schema"], ensure_ascii=False, allow_nan=False))
            if request["messages"][0]["role"] == "system":
                request["messages"][0]["content"] += "\n\n" + instructions
            else:
                request["messages"].insert(0, {"role": "system", "content": instructions})
        info = self._require_current(request["model"])
        if context > info["context_length"]:
            raise ProviderError("The app context exceeds KoboldCpp's loaded window. Connect again or lower KOBOLDCPP_CONTEXT_LENGTH.")
        count = self._preflight_prompt(request)
        self._require_current(request["model"])
        return request, count

    def _preflight_prompt(self, request):
        probe = {key: value for key, value in request.items()
                 if key not in {"grammar", "grammar_retain_state", "stream_options"}}
        probe.update(stream=False, special=True)
        result = self._request_json("POST", "/api/extra/tokencount", probe)
        prompt, count = result.get("prompt"), result.get("value")
        if not isinstance(prompt, str) or not prompt or type(count) is not int or count < 0:
            raise ProviderError("KoboldCpp returned an invalid rendered prompt or token count.")
        position = 0
        for message in request["messages"]:
            text = message["content"].strip()
            if text:
                found = prompt.find(text, position)
                if found < 0:
                    raise ProviderError("The KoboldCpp chat template omitted or changed message text. "
                                        "Choose a template that preserves the complete input.")
                position = found + len(text)
        context = request["max_context_length"]
        remaining = context - count - _NATIVE_FRAMING_RESERVE
        # Match the server's own output ceiling, not an arbitrary app ceiling.
        # Native framing near the 25% threshold uses the conservative branch.
        ratio = 0.2 if count + _NATIVE_FRAMING_RESERVE < context * 0.25 else 0.35
        server_limit = context - max(min(context - 4, 16), int(context * ratio))
        allowance = min(remaining, server_limit)
        if allowance < request["max_tokens"]:
            if allowance <= 0 or "grammar" in request:
                raise ProviderError("The complete KoboldCpp prompt and reply reserve exceed the available context or native output limit. "
                                    "Reduce the input or load a larger context, then Connect again.")
            request["max_tokens"] = allowance
        return count

    @staticmethod
    def _sse_events(response):
        yield from LocalHTTPTransport._iter_sse(response, max_event_bytes=MAX_EVENT_BYTES,
                                               max_response_bytes=MAX_RESPONSE_BYTES)

    @staticmethod
    def _checked_event(result, request):
        if result.get("model") != request["model"]:
            raise ProviderError("KoboldCpp answered with a different model. Connect again; the reply was not accepted.")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ProviderError("KoboldCpp returned malformed response choices.")
        if not choices:
            return {"model": request["model"], "choices": [], "usage": result.get("usage")}
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise ProviderError("KoboldCpp returned an invalid response choice.")
        delta, reason = choice.get("delta"), choice.get("finish_reason")
        if not isinstance(delta, dict) or reason not in (None, "stop", "length"):
            raise ProviderError("KoboldCpp returned an invalid delta or unsuccessful completion status.")
        if delta.get("tool_calls") or delta.get("function_call"):
            raise ProviderError("KoboldCpp returned an unsupported tool call.")
        if delta.get("refusal"):
            raise ProviderError("The local model refused the request.")
        text = delta.get("content")
        if text is not None and not isinstance(text, str):
            raise ProviderError("KoboldCpp returned invalid answer text.")
        # Only visible answer text belongs in the shared response contract.
        return {"model": request["model"], "choices": [{"index": 0, "finish_reason": reason,
                  "delta": {"content": text} if text is not None else {}}]}

    @staticmethod
    def _checked_usage(usage, request, input_count):
        if (not isinstance(usage, dict) or any(type(usage.get(key)) is not int
                or usage[key] < 0 for key in ("prompt_tokens", "completion_tokens"))):
            raise ProviderError("KoboldCpp did not report valid completed-request token usage.")
        # Native BOS handling can differ by one; larger losses mean some of the
        # checked prompt was dropped. Never publish such a result as complete.
        if usage["prompt_tokens"] < max(0, input_count - 1):
            raise ProviderError("KoboldCpp truncated the checked input. Disable Smart Context or increase its loaded context, then reconnect.")
        if usage["prompt_tokens"] > input_count + _NATIVE_FRAMING_RESERVE:
            raise ProviderError("KoboldCpp changed the checked prompt framing; the reply was not accepted. Check the chat template and reconnect.")
        if usage["prompt_tokens"] + usage["completion_tokens"] > request["max_context_length"]:
            raise ProviderError("KoboldCpp exceeded the bound request context; the reply was not accepted.")
        return {key: usage[key] for key in ("prompt_tokens", "completion_tokens")}

    def stream_chat(self, payload):
        request, input_count = self._chat_request(payload)
        terminal, usage = None, None
        try:
            with self._client() as client:
                with client.stream("POST", self.base_url + "/v1/chat/completions", json=request) as response:
                    if not response.is_success:
                        self._check_status(response, self._read_bounded(response))
                    for raw in self._sse_events(response):
                        if raw is None:
                            if terminal is None or usage is None:
                                raise ProviderError("KoboldCpp ended its stream without a completion status and token usage.")
                            self._require_current(request["model"])
                            if (usage["completion_tokens"] >= request["max_tokens"]
                                    or usage["prompt_tokens"] + usage["completion_tokens"] >= request["max_context_length"]):
                                terminal["choices"][0]["finish_reason"] = "length"
                            terminal["usage"] = usage
                            yield terminal
                            return
                        event = self._checked_event(raw, request)
                        if not event["choices"]:
                            if terminal is None or usage is not None:
                                raise ProviderError("KoboldCpp sent token usage out of sequence.")
                            usage = self._checked_usage(event["usage"], request, input_count)
                            continue
                        if terminal is not None:
                            raise ProviderError("KoboldCpp sent answer data after its completion status.")
                        if event["choices"][0]["finish_reason"] is not None:
                            terminal = event
                        else:
                            yield event
                    raise ProviderError("The KoboldCpp response stream ended before the model finished.")
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc

    def chat(self, payload):
        pieces, terminal = [], None
        stream = self.stream_chat(payload)
        try:
            for event in stream:
                choice = event["choices"][0]
                text = choice["delta"].get("content")
                if text:
                    pieces.append(text)
                if choice["finish_reason"] is not None:
                    terminal = event
        finally:
            stream.close()
        if terminal is None:
            raise ProviderError("KoboldCpp did not complete the response.")
        return {"model": terminal["model"], "usage": terminal["usage"], "choices": [{"index": 0,
                "finish_reason": terminal["choices"][0]["finish_reason"],
                "message": {"role": "assistant", "content": "".join(pieces)}}]}

    def embeddings(self, model, texts):
        raise ProviderError("Choose a separate installed local embedding connector for KoboldCpp RAG.")
