"""The closed facility-error taxonomy of the offline LLM planner.

A :class:`PlannerError` is a facility failure, never a plan result. It carries a stable
``code`` from the closed :data:`PLANNER_ERROR_CODES` tuple, a structural ``location`` built
from predefined segments, and a fixed safe ``message``. It never stores or returns the
provider exception, the prompt, the model response or a credential, and neither ``str`` nor
``repr`` echoes a supplied value.
"""

from __future__ import annotations

from typing import Final

__all__ = ["PLANNER_ERROR_CODES", "PlannerError"]

#: The closed set of facility failure codes this package can raise.
PLANNER_ERROR_CODES: Final[tuple[str, ...]] = (
    "INPUT_INVALID",
    "CONTEXT_MISMATCH",
    "REQUEST_TOO_LARGE",
    "RESPONSE_INVALID",
    "RESPONSE_TOO_LARGE",
    "UNSUPPORTED_ACTION",
    "UNSUPPORTED_OUTCOME",
    "PROVIDER_REFUSED",
    "PROVIDER_FAILURE",
    "TIMEOUT",
    "BUDGET_EXCEEDED",
    "BUSY",
    "CLOSED",
    "POISONED",
)

#: The predefined location segments. A caller-supplied key is never copied into an error path.
LOCATION_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "action",
        "action_guidance",
        "action_type",
        "bytes",
        "calls",
        "capabilities",
        "cleanup",
        "config",
        "content",
        "context",
        "deadline",
        "environment_type",
        "forecast",
        "guidance",
        "json",
        "lifecycle",
        "metadata",
        "observation",
        "outcome_guidance",
        "outcome_type",
        "outcomes",
        "output_tokens",
        "provider",
        "record",
        "request",
        "request_id",
        "response",
        "response_schema",
        "schema",
        "schema_version",
        "usage",
    }
)


def _format_location(location: tuple[str | int, ...]) -> str:
    """Render a structural location as ``field[0].nested``, never a supplied value."""
    text = ""
    for part in location:
        if isinstance(part, int) and not isinstance(part, bool):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + part
    return text or "<root>"


class PlannerError(Exception):
    """A bounded facility failure of the offline LLM planner.

    ``code`` is one of :data:`PLANNER_ERROR_CODES`, ``location`` locates the rejected
    structure and ``message`` is a fixed sentence that carries no supplied value. All three
    are read-only properties, and this class deliberately keeps no original exception as a
    public ``cause`` or ``context``.
    """

    __slots__ = ("_code", "_location", "_message")

    def __init__(
        self,
        code: str,
        message: str,
        *,
        location: tuple[str | int, ...] = (),
    ) -> None:
        if code not in PLANNER_ERROR_CODES:
            raise ValueError("unknown planner error code")
        parts = tuple(location)
        for part in parts:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise TypeError("a location segment must be a field name or an index")
            if isinstance(part, str) and part not in LOCATION_SEGMENTS:
                raise ValueError("a location segment must be a predefined field name")
            if isinstance(part, int) and part < 0:
                raise ValueError("a location index must not be negative")
        self._code = code
        self._location = parts
        self._message = message
        super().__init__(f"{code} at {_format_location(parts)}: {message}")

    @property
    def code(self) -> str:
        """The closed failure code."""
        return self._code

    @property
    def location(self) -> tuple[str | int, ...]:
        """The structural location of the rejection."""
        return self._location

    @property
    def message(self) -> str:
        """The fixed, value-free diagnostic sentence."""
        return self._message

    def __repr__(self) -> str:
        return (
            f"PlannerError(code={self._code!r}, location={self._location!r}, "
            f"message={self._message!r})"
        )
