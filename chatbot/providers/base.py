"""Shared error contract for local inference backends."""


class ProviderError(RuntimeError):
    """A local server could not complete a validated inference request."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code
