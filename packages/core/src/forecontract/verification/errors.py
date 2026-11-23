"""The single public facility error of the verification package.

A verification failure is not a verification verdict. Damaged input, an unusable rule and an
unusable binding are all facility errors: they are raised, never returned as
:class:`~forecontract.models.outcomes.VerificationStatus` values, so a caller can never mistake
a malformed request for a negative result.

``code`` is one of exactly three documented values, ``location`` is a structural position built
from field names and list indices, and ``message`` is a fixed bounded sentence. Neither the
message nor the location ever echoes a supplied value, a queried path, an observation state, an
identifier or a raw dependency error. A diagnostic cause may be kept in-process on
``__cause__``; it is never placed in a verdict, in metadata or in any serialized form.
"""

from __future__ import annotations

from typing import Final

__all__ = ["VerificationError"]

#: The closed set of facility failure codes this package can raise.
VERIFICATION_ERROR_CODES: Final[tuple[str, ...]] = (
    "VERIFICATION_INPUT_INVALID",
    "VERIFICATION_RULE_INVALID",
    "VERIFICATION_BINDING_INVALID",
)


def _format_location(location: tuple[str | int, ...]) -> str:
    """Render a structural position as ``field[0].nested``, never a supplied value."""
    text = ""
    for part in location:
        if isinstance(part, bool):
            text += ".<index>"
        elif isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text or "<root>"


class VerificationError(Exception):
    """A malformed verification request or an unusable rule or binding.

    ``code`` is one of :data:`VERIFICATION_ERROR_CODES`, ``location`` locates the offending
    structure and ``message`` is a fixed bounded sentence that carries no supplied value.
    """

    def __init__(
        self,
        code: str,
        location: tuple[str | int, ...],
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if code not in VERIFICATION_ERROR_CODES:
            raise ValueError("an unknown verification error code is not allowed")
        self.code = code
        self.location = tuple(location)
        self.message = message
        super().__init__(f"{code} at {_format_location(self.location)}: {message}"[:240])
        if cause is not None:
            self.__cause__ = cause
