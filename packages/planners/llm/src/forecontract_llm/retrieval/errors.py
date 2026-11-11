"""The closed, value-free error taxonomy of the controlled evidence retrieval.

A :class:`RetrievalError` is a facility failure, never a plan result. It carries a stable
``code`` from :data:`RETRIEVAL_ERROR_CODES`, a structural ``stage`` from
:data:`RETRIEVAL_ERROR_STAGES` and an optional ``request_id`` that names the planner's own
request. It never stores or returns raw evidence, a model response, a local path or a facility
exception string; ``str`` and ``repr`` are built only from the fixed vocabulary and a fixed
sentence chosen by this module.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "RETRIEVAL_ERROR_CODES",
    "RETRIEVAL_ERROR_STAGES",
    "RetrievalError",
]

#: The closed set of facility failure codes (C06).
RETRIEVAL_ERROR_CODES: Final[tuple[str, ...]] = (
    "invalid_input",
    "invalid_state",
    "binding_mismatch",
    "access_denied",
    "source_invalid",
    "selection_invalid",
    "duplicate_read",
    "budget_exceeded",
    "provider_refused",
    "provider_failed",
    "reader_failed",
    "audit_failed",
    "invalid_response",
    "timeout",
)

#: The closed set of stages a failure can be attributed to.
RETRIEVAL_ERROR_STAGES: Final[tuple[str, ...]] = (
    "lifecycle",
    "input",
    "context",
    "scope",
    "request",
    "audit",
    "provider",
    "response",
    "binding",
    "grant",
    "selection",
    "reader",
    "budget",
    "deadline",
    "cleanup",
    "source",
)

#: One fixed, value-free sentence per code. A caller never supplies the message.
_MESSAGES: Final[dict[str, str]] = {
    "invalid_input": "a supplied value does not match the declared retrieval protocol",
    "invalid_state": "the retrieval capability is not in a state that permits this operation",
    "binding_mismatch": "the supplied identifiers do not belong to this plan",
    "access_denied": "the requested evidence or view is not part of the frozen scope",
    "source_invalid": "the registered evidence snapshot is not the validated source",
    "selection_invalid": "the selector or text range is not a legal selection",
    "duplicate_read": "this plan already requested the same evidence view and range",
    "budget_exceeded": "a declared retrieval budget was exceeded",
    "provider_refused": "the provider refused the request",
    "provider_failed": "the provider call failed",
    "reader_failed": "the evidence reader failed",
    "audit_failed": "the audit record could not be persisted",
    "invalid_response": "the provider answer is not the declared retrieval message",
    "timeout": "the planning deadline expired",
}


def _bounded_request_id(value: object) -> str | None:
    """Accept only a bounded identifier; anything else is refused without echoing it."""
    if value is None:
        return None
    if type(value) is not str or not 1 <= len(value) <= 128:
        raise TypeError("a request identifier must be a string of at most 128 characters")
    return value


class RetrievalError(Exception):
    """A bounded facility failure of the controlled retrieval capability."""

    __slots__ = ("_code", "_message", "_request_id", "_stage")

    def __init__(self, code: str, stage: str, *, request_id: str | None = None) -> None:
        if code not in RETRIEVAL_ERROR_CODES:
            raise ValueError("unknown retrieval error code")
        if stage not in RETRIEVAL_ERROR_STAGES:
            raise ValueError("unknown retrieval error stage")
        self._code = code
        self._stage = stage
        self._request_id = _bounded_request_id(request_id)
        self._message = _MESSAGES[code]
        text = f"{code} at {stage}: {self._message}"
        if self._request_id is not None:
            text = f"{text} [request {self._request_id}]"
        super().__init__(text)

    @property
    def code(self) -> str:
        """The closed failure code."""
        return self._code

    @property
    def stage(self) -> str:
        """The structural stage of the failure."""
        return self._stage

    @property
    def request_id(self) -> str | None:
        """The planner request this failure belongs to, when one exists."""
        return self._request_id

    @property
    def message(self) -> str:
        """The fixed, value-free sentence for this code."""
        return self._message

    def __repr__(self) -> str:
        return (
            f"RetrievalError(code={self._code!r}, stage={self._stage!r}, "
            f"request_id={self._request_id!r})"
        )
