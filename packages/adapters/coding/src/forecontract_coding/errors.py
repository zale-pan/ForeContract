"""Bounded, explicit public error codes for ``forecontract_coding``.

Every failure carries a fixed uppercase code, a bounded message without secrets or host stacks,
and a structural location tuple.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CODING_ACTION_INVALID",
    "CODING_ACTION_REPLAY",
    "CODING_FILE_CONFLICT",
    "CODING_FILE_NOT_FOUND",
    "CODING_INPUT_INVALID",
    "CODING_OBSERVATION_OVERFLOW",
    "CODING_PATH_CONFLICT",
    "CODING_PROCESS_CANCELLED",
    "CODING_PROCESS_CLEANUP_FAILED",
    "CODING_PROCESS_START_FAILED",
    "CODING_PROCESS_TIMEOUT",
    "CODING_SCOPE_VIOLATION",
    "CODING_VERIFICATION_INVALID",
    "CodingError",
]

CODING_INPUT_INVALID: Final[str] = "CODING_INPUT_INVALID"
CODING_SCOPE_VIOLATION: Final[str] = "CODING_SCOPE_VIOLATION"
CODING_PATH_CONFLICT: Final[str] = "CODING_PATH_CONFLICT"
CODING_FILE_CONFLICT: Final[str] = "CODING_FILE_CONFLICT"
CODING_FILE_NOT_FOUND: Final[str] = "CODING_FILE_NOT_FOUND"
CODING_PROCESS_START_FAILED: Final[str] = "CODING_PROCESS_START_FAILED"
CODING_PROCESS_TIMEOUT: Final[str] = "CODING_PROCESS_TIMEOUT"
CODING_PROCESS_CANCELLED: Final[str] = "CODING_PROCESS_CANCELLED"
CODING_PROCESS_CLEANUP_FAILED: Final[str] = "CODING_PROCESS_CLEANUP_FAILED"
CODING_ACTION_INVALID: Final[str] = "CODING_ACTION_INVALID"
CODING_ACTION_REPLAY: Final[str] = "CODING_ACTION_REPLAY"
CODING_VERIFICATION_INVALID: Final[str] = "CODING_VERIFICATION_INVALID"
CODING_OBSERVATION_OVERFLOW: Final[str] = "CODING_OBSERVATION_OVERFLOW"


class CodingError(Exception):
    """A bounded domain/execution error raised by ``forecontract_coding``."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        location: tuple[str | int, ...] = (),
    ) -> None:
        self.code = code
        self.message = message
        self.location = tuple(location)
        super().__init__(f"{code}: {message}")
