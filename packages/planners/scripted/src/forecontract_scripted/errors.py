"""Explicit facility errors for the deterministic ScriptedPlanner.

These are facility failures, not business outcomes and not plan results. A planner error never
consumes a scripted step and is never returned to a caller as data.

Every error carries a stable ``code`` and a structural ``path`` built from field names and list
indices. A message never transcribes a supplied observation state, a scripted forecast or a raw
dependency error: the diagnostic cause stays in-process as ``__cause__`` and is never
serialized.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "PLANNER_ERROR_CODES",
    "PlannerError",
    "PlannerInputError",
    "PlannerObservationMismatchError",
    "PlannerScriptError",
    "PlannerScriptExhaustedError",
]

#: The closed set of facility failure codes this package can raise.
PLANNER_ERROR_CODES: Final[tuple[str, ...]] = (
    "planner_input",
    "planner_script",
    "planner_observation_mismatch",
    "planner_script_exhausted",
)


def _format_path(path: tuple[str | int, ...]) -> str:
    """Render a structural location as ``field[0].nested``, never a supplied value."""
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text or "<root>"


class PlannerError(Exception):
    """Base class for every ScriptedPlanner facility failure.

    ``code`` is one of :data:`PLANNER_ERROR_CODES`, ``path`` locates the rejected structure and
    ``message`` is a fixed, bounded sentence that carries no supplied value.
    """

    code: str = "planner_input"

    def __init__(
        self,
        message: str,
        *,
        path: tuple[str | int, ...] = (),
        cause: BaseException | None = None,
    ) -> None:
        self.message = message
        self.path = tuple(path)
        super().__init__(f"{self.code} at {_format_path(self.path)}: {message}")
        if cause is not None:
            self.__cause__ = cause


class PlannerInputError(PlannerError):
    """A supplied observation or argument does not match the declared schema."""

    code = "planner_input"


class PlannerScriptError(PlannerError):
    """A supplied script is not a valid deterministic plan."""

    code = "planner_script"


class PlannerObservationMismatchError(PlannerError):
    """The observation does not match the current scripted step."""

    code = "planner_observation_mismatch"


class PlannerScriptExhaustedError(PlannerError):
    """Every scripted planning step has already been consumed."""

    code = "planner_script_exhausted"
