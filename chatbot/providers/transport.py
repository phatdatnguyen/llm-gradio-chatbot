"""Bounded HTTP transport for explicitly configured local inference servers."""

import ipaddress
import json
import math
from urllib.parse import urlsplit, urlunsplit

import httpx

from .base import ProviderError


MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024
_LAN_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7",
))


def validate_base_url(value, *, paths=("",)):
    """Pin localhost and reject public hosts, credentials, and arbitrary paths."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter the local inference server URL.")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise ValueError("The local server URL is invalid.") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("Use an http(s) local server URL without credentials, query or fragment.")
    host = parsed.hostname.lower()
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Use localhost, a loopback address, or an explicit private LAN IP.") from exc
    if not (address.is_loopback or any(address in network for network in _LAN_NETWORKS)):
        raise ValueError("Only loopback and private LAN inference servers are allowed.")
    if parsed.path.rstrip("/") not in paths:
        raise ValueError("Use the local inference server root URL.")
    authority = f"[{host}]" if address.version == 6 else host
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parsed.scheme, authority, "", "", ""))


class LocalHTTPTransport:
    """Request-owned clients, with no proxy inheritance, redirects, or retries."""

    server_name = "Local inference server"
    url_paths = ("",)

    def __init__(self, base_url, api_key=None, timeout=900):
        self.base_url = validate_base_url(base_url, paths=self.url_paths)
        if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                or not 0 < timeout <= 3600):
            raise ValueError("Local request timeout must be between 0 and 3600 seconds.")
        if api_key is not None and (not isinstance(api_key, str)
                                   or any(ord(char) < 32 or ord(char) == 127 for char in api_key)):
            raise ValueError("The local server token must be text without control characters.")
        self.timeout = timeout
        self._api_key = api_key.strip() if api_key else None

    def _client(self):
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        return httpx.Client(timeout=httpx.Timeout(self.timeout, connect=min(5, self.timeout)),
                            headers=headers, follow_redirects=False, trust_env=False)

    def _transport_error(self, exc):
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError("The local model request timed out. Check the model load and request timeout.")
        return ProviderError(f"Cannot reach {self.server_name} at {self.base_url}. "
                             "Start its local server and check the address.")

    @staticmethod
    def _read_bounded(response):
        body = bytearray()
        for chunk in response.iter_bytes(chunk_size=65536):
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ProviderError("The local server response exceeded the size limit.")
        return bytes(body)

    def _check_status(self, response, body=b""):
        if response.is_redirect:
            raise ProviderError("Local inference server redirects are not allowed.",
                                status_code=response.status_code)
        if not response.is_success:
            detail = ""
            try:
                payload = json.loads(body)
                error = payload.get("error") if isinstance(payload, dict) else None
                detail = error.get("message", "") if isinstance(error, dict) else error or ""
            except (ValueError, UnicodeDecodeError):
                pass
            suffix = f": {str(detail)[:500]}" if detail else ""
            raise ProviderError(f"{self.server_name} returned HTTP {response.status_code}{suffix}",
                                status_code=response.status_code)

    def _request_json(self, method, path, payload=None):
        try:
            with self._client() as client:
                with client.stream(method, self.base_url + path, json=payload) as response:
                    body = self._read_bounded(response)
                    self._check_status(response, body)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        try:
            result = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProviderError("The local server returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise ProviderError("The local server returned an invalid response object.")
        if result.get("error"):
            raise ProviderError("The local server reported a model error.")
        return result

    @staticmethod
    def _iter_sse(response, *, max_event_bytes=MAX_EVENT_BYTES, max_response_bytes=MAX_RESPONSE_BYTES):
        """Decode bounded SSE, yielding None for DONE without buffering future deltas."""
        pending, data = bytearray(), []
        total = event_size = 0

        def decode(raw):
            return None if raw.strip() == b"[DONE]" else LocalHTTPTransport._decode_event(raw)

        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > max_response_bytes:
                raise ProviderError("The local response stream exceeded the size limit.")
            pending.extend(chunk)
            while b"\n" in pending:
                raw, _, remaining = pending.partition(b"\n")
                pending = bytearray(remaining)
                event_size += len(raw) + 1
                if event_size > max_event_bytes:
                    raise ProviderError("The local response stream event exceeded the size limit.")
                line = bytes(raw).rstrip(b"\r")
                if not line:
                    if data:
                        yield decode(b"\n".join(data))
                    data, event_size = [], 0
                elif line.startswith(b"data:"):
                    data.append(line[5:].lstrip(b" "))
            if len(pending) + event_size > max_event_bytes:
                raise ProviderError("The local response stream event exceeded the size limit.")
        if pending:
            line = bytes(pending).rstrip(b"\r")
            if line.startswith(b"data:"):
                data.append(line[5:].lstrip(b" "))
        if data:
            yield decode(b"\n".join(data))

    @staticmethod
    def _iter_ndjson(response):
        """Bound each unfinished line before decoding and the complete response."""
        pending = bytearray()
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise ProviderError("The local response stream exceeded the size limit.")
            pending.extend(chunk)
            while b"\n" in pending:
                raw, _, remaining = pending.partition(b"\n")
                pending = bytearray(remaining)
                if len(raw) > MAX_EVENT_BYTES:
                    raise ProviderError("The local response stream event exceeded the size limit.")
                if raw.strip():
                    yield LocalHTTPTransport._decode_event(raw)
            if len(pending) > MAX_EVENT_BYTES:
                raise ProviderError("The local response stream event exceeded the size limit.")
        if pending.strip():
            yield LocalHTTPTransport._decode_event(pending)

    @staticmethod
    def _decode_event(raw):
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProviderError("The local server returned an invalid stream event.") from exc
        if not isinstance(value, dict) or value.get("error"):
            raise ProviderError("The local response stream reported an error.")
        return value
