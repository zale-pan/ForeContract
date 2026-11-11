"""Explicit facility errors for the deterministic Fake environment adapter.

These are facility failures, not business outcomes: an exception never consumes a script step
and is never returned as an :class:`ActualOutcome`. Capability and payload rejections reuse the
existing Core exceptions unchanged, so a rejected action type and a rejected payload keep their
established meanings.

Every error carries a stable ``code`` and a structural ``path`` built from field names and list
indices. A message never transcribes an action payload, a state value, a script value or a raw
dependency error: the diagnostic cause stays in-process as ``__cause__`` and is never
serialized.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "FAKE_ERROR_CODES",
    "FakeError",
    "FakeInputError",
    "FakeLifecycleError",
    "FakeScenarioError",
    "FakeScriptExhaustedError",
    "FakeStepMismatchError",
]

#: The closed set of facility failure codes this package can raise.
FAKE_ERROR_CODES: Final[tuple[str, ...]] = (
    "fake_input",
    "fake_scenario",
    "fake_lifecycle",
    "fake_step_mismatch",
    "fake_script_exhausted",
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


class FakeError(Exception):
    """Base class for every Fake facility failure.

    ``code`` is one of :data:`FAKE_ERROR_CODES`, ``path`` locates the rejected structure and
    ``message`` is a fixed, bounded sentence that carries no supplied value.
    """

    code: str = "fake_input"

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


class FakeInputError(FakeError):
    """A supplied action, observation or argument does not match the declared schema."""

    code = "fake_input"


class FakeScenarioError(FakeError):
    """A supplied scenario is not a valid deterministic script."""

    code = "fake_scenario"


class FakeLifecycleError(FakeError):
    """The requested lifecycle transition is not allowed from the current state."""

    code = "fake_lifecycle"


class FakeStepMismatchError(FakeError):
    """The action does not match the current scripted step."""

    code = "fake_step_mismatch"


class FakeScriptExhaustedError(FakeError):
    """Every scripted step has already been executed."""

    code = "fake_script_exhausted"
