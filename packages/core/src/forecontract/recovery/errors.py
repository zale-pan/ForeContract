"""The single public facility error of the recovery package.

A recovery failure is not a recovery decision. Damaged input, an unusable committed policy and
unusable evidence are all facility errors: they are raised, never returned as a
:class:`~forecontract.recovery.models.RecoveryAssessment`, so a caller can never mistake a
malformed request for a business "do not recover" verdict.

``code`` is one of exactly three documented values, ``location`` is a structural position built
from field names and list indices, and ``message`` is a fixed bounded sentence. Neither the
message nor the location ever echoes a supplied value, a raw outcome, an observation state, an
identifier, an object repr or a raw dependency error. A diagnostic cause may be kept in-process
on ``__cause__``; it is never placed in a decision, in metadata or in any serialized form.
"""

from __future__ import annotations

from typing import Final

__all__ = ["RecoveryError"]

#: The closed set of facility failure codes this package can raise.
RECOVERY_ERROR_CODES: Final[tuple[str, ...]] = (
    "INPUT_INVALID",
    "POLICY_INVALID",
    "EVIDENCE_INVALID",
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


class RecoveryError(Exception):
    """A malformed recovery request, an unusable policy or unusable evidence.

    ``code`` is one of :data:`RECOVERY_ERROR_CODES`, ``location`` locates the offending structure
    and ``message`` is a fixed bounded sentence that carries no supplied value.
    """

    def __init__(
        self,
        code: str,
        location: tuple[str | int, ...],
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if code not in RECOVERY_ERROR_CODES:
            raise ValueError("an unknown recovery error code is not allowed")
        self.code = code
        self.location = tuple(location)
        self.message = message
        super().__init__(f"{code} at {_format_location(self.location)}: {message}"[:240])
        if cause is not None:
            self.__cause__ = cause
