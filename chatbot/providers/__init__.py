"""Local inference backends; no cloud client or implicit remote fallback."""

from .base import ProviderError
from .lmstudio import LMStudioProvider

__all__ = ["LMStudioProvider", "ProviderError"]
