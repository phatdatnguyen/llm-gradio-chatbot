"""Explicit fake local model metadata for UI tests; no model server required."""
import pytest


@pytest.fixture
def local_models(monkeypatch):
    from chatbot import config
    monkeypatch.setattr(config, "DEFAULT_MODEL", "local-chat")
    monkeypatch.setattr(config, "MODEL_CHOICES", ["local-chat"])
    monkeypatch.setattr(config, "MODEL_TOKEN_LIMITS", {"local-chat": 32768})
    monkeypatch.setattr(config, "MODEL_MAX_OUTPUT_TOKENS", {"local-chat": 4096})
    monkeypatch.setattr(config, "_MODEL_INFO", {"local-chat": {"vision": True}})
