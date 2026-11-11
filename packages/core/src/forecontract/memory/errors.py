"""Bounded, value-free error types for the Memory subpackage.

Five failure classes are kept apart, because a caller reacts differently to each one:

* :class:`MemoryValidationError` - the caller's value, record or query is invalid. Nothing was
  written and no stored state changed.
* :class:`MemoryNotFoundError` - a requested Run or Memory item does not exist.
* :class:`MemoryCorruptionError` - an existing Memory file is not a valid Memory. It names the
  failing 1-based line number and never skips, truncates or repairs the file.
* :class:`MemoryStorageError` - the operating system refused a Memory file operation. The
  original exception stays available as ``__cause__`` inside this process.
* :class:`MemorySourceError` - a real trace source is missing, corrupt, unreadable or does not
  match the request. The tracing cause is preserved separately, and a read that happened while
  replaying stored history also carries that Memory line number.

Every public message is fixed and bounded. It never transcribes a rejected key, a rejected
value, a filesystem path or a raw exception message, and no cause is ever serialized.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "MEMORY_ERROR_CODES",
    "MemoryCorruptionError",
    "MemoryError",
    "MemoryNotFoundError",
    "MemorySourceError",
    "MemoryStorageError",
    "MemoryValidationError",
]

#: The closed set of failure codes the Memory surface can raise.
MEMORY_ERROR_CODES: Final[tuple[str, ...]] = (
    "memory_validation",
    "memory_not_found",
    "memory_corruption",
    "memory_storage",
    "memory_source",
)


def _cause_name(exc: BaseException | None) -> str:
    """Return only the exception class name, never its message or arguments."""
    if exc is None:
        return "unknown"
    return type(exc).__name__


class MemoryError(Exception):
    """Base class for every Memory failure.

    ``code`` is one of :data:`MEMORY_ERROR_CODES` and ``message`` is a fixed, bounded sentence
    that carries no rejected value, no rejected key, no full path and no raw cause text.
    """

    code: str = "memory_validation"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        self.message = message
        self.cause_name = _cause_name(cause)
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class MemoryValidationError(MemoryError):
    """A supplied value, record, relationship or query is invalid."""

    code = "memory_validation"


class MemoryNotFoundError(MemoryError):
    """The requested Run state or Memory item does not exist."""

    code = "memory_not_found"


class MemoryCorruptionError(MemoryError):
    """An existing Memory file is not a valid Memory.

    ``line_number`` is the 1-based physical line that failed, or ``None`` when the file as a
    whole could not be decoded. Both are safe to report.
    """

    code = "memory_corruption"

    def __init__(self, message: str, *, line_number: int | None = None) -> None:
        self.line_number = line_number
        detail = message if line_number is None else f"{message} at line {line_number}"
        super().__init__(detail)


class MemoryStorageError(MemoryError):
    """The operating system refused a Memory file open, read, write, flush or close."""

    code = "memory_storage"


class MemorySourceError(MemoryError):
    """A real trace source is missing, corrupt, unreadable or does not match the request.

    ``line_number`` is set when the failure was found while replaying an existing Memory line,
    so an operator can locate the stored version whose source no longer resolves.
    """

    code = "memory_source"

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        line_number: int | None = None,
    ) -> None:
        self.line_number = line_number
        detail = message if line_number is None else f"{message} at line {line_number}"
        super().__init__(detail, cause=cause)
