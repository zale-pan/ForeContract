"""Stable, bounded error types for the structured tracing store.

Four failure classes are kept apart on purpose, because a caller has to react differently to
each one:

* :class:`TraceValidationError` - the caller's record or its association with the facts
  already recorded is invalid. Nothing was written.
* :class:`TraceCorruptionError` - an existing file is not a valid trace. It names the failing
  line number and never skips, truncates or repairs the file.
* :class:`TraceStorageError` - the operating system refused an open, read, write, flush or
  close. The original exception stays available as ``__cause__`` inside this process.
* :class:`TraceNotFoundError` - the requested path, run or record does not exist.

Every public message is fixed and bounded. It never transcribes a rejected value, a raw
exception message or a full filesystem path, so a message is safe to log and to render. The
raw cause is deliberately *not* serialized anywhere: the tracing store records safe codes,
never transport-level error text.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "TRACE_ERROR_CODES",
    "TraceCorruptionError",
    "TraceError",
    "TraceNotFoundError",
    "TraceStorageError",
    "TraceValidationError",
]

#: The closed set of failure codes the tracing surface can raise.
TRACE_ERROR_CODES: Final[tuple[str, ...]] = (
    "TRACE_VALIDATION_INVALID",
    "TRACE_CORRUPT_FILE",
    "TRACE_STORAGE_FAILURE",
    "TRACE_NOT_FOUND",
)


def _cause_name(exc: BaseException | None) -> str:
    """Return only the exception class name, never its message or arguments."""
    if exc is None:
        return "unknown"
    return type(exc).__name__


class TraceError(Exception):
    """Base class for every tracing failure.

    ``code`` is one of :data:`TRACE_ERROR_CODES` and ``message`` is a fixed, bounded
    sentence that carries no rejected value, no raw exception text and no full path.
    """

    code: str = "TRACE_VALIDATION_INVALID"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        self.message = message
        self.cause_name = _cause_name(cause)
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class TraceValidationError(TraceError):
    """A supplied record, or its association with recorded facts, is invalid."""

    code = "TRACE_VALIDATION_INVALID"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)


class TraceCorruptionError(TraceError):
    """An existing trace file is not a valid trace.

    ``line_number`` is the 1-based physical line that failed, or ``None`` when the file as a
    whole could not be decoded. The message stays fixed; the line number is the only detail
    that is reported and it is not sensitive.
    """

    code = "TRACE_CORRUPT_FILE"

    def __init__(self, message: str, *, line_number: int | None = None) -> None:
        self.line_number = line_number
        detail = message if line_number is None else f"{message} at line {line_number}"
        super().__init__(detail)


class TraceStorageError(TraceError):
    """The operating system refused an open, read, write, flush or close.

    The class name of the original exception is recorded for local diagnostics; the exception
    itself is preserved as ``__cause__`` and is never serialized into a trace.
    """

    code = "TRACE_STORAGE_FAILURE"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)


class TraceNotFoundError(TraceError):
    """The requested trace path, run or record does not exist."""

    code = "TRACE_NOT_FOUND"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)
