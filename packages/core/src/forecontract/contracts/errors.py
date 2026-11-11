"""Stable build-time rejection type for the Contract Builder.

A build rejection is a data decision, not a crash: every one carries exactly one of the
documented gate codes, a structural JSON-style location built from field names and list
indices, and a message that never echoes an arbitrary payload value copied out of the
rejected input. Unexpected dependency failures keep their original exception as ``__cause__``
so the operator can still see the real defect, while the public surface stays stable.
"""

from __future__ import annotations

from typing import Final

__all__ = ["BUILD_ERROR_CODES", "ContractBuildError"]

BUILD_ERROR_CODES: Final[tuple[str, ...]] = (
    "INPUT_INVALID",
    "ADAPTER_MISMATCH",
    "UNSUPPORTED_CAPABILITY",
    "INVALID_DOMAIN_PAYLOAD",
    "VERIFICATION_BINDING_INVALID",
    "EXPECTED_OUTCOME_REQUIRED",
    "BUDGET_INVALID",
    "PRECONDITION_UNSATISFIED",
    "DEPENDENCY_FAILURE",
)


def _format_location(location: tuple[str | int, ...]) -> str:
    """Render a structural location as ``field[0].nested`` without a payload value."""
    text = ""
    for part in location:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text or "<root>"


class ContractBuildError(ValueError):
    """A deterministic, located Contract Builder rejection.

    ``code`` is one of the documented gate codes and ``location`` locates the rejected
    structure. The class deliberately exposes the three fields separately and derives its
    message text from them, so a caller can branch on ``code`` without parsing a string.
    """

    def __init__(self, code: str, location: tuple[str | int, ...], message: str) -> None:
        if code not in BUILD_ERROR_CODES:
            raise ValueError(f"undocumented build error code: {code!r}")
        parts = tuple(location)
        for part in parts:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise TypeError(f"invalid location part: {part!r}")
        self.code = code
        self.location = parts
        self.message = message
        super().__init__(f"{code} at {_format_location(parts)}: {message}")
