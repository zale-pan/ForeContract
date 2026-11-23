"""The deterministic ScriptedPlanner.

:class:`ScriptedPlanner` exposes exactly one planning method, ``async plan(observation)``. It
has no ``start``, ``stop`` or ``reset``: a replay is started by creating a new instance.

The order is fixed. The supplied observation is re-validated, then exhaustion is checked, then
the expected environment type and the complete expected state are matched, then the detached
forecast is prepared, and only then does the cursor advance exactly once. Any rejection leaves
the cursor untouched. A repeated successful call is not idempotent: each success consumes one
scripted step.

The planner never reads a Fake adapter, real environment state or a future script step, never
imports the Fake distribution, never touches Memory and never decides that a task succeeded. It
only returns the forecast the script declares for the observation it was given.
"""

from __future__ import annotations

from forecontract.models.actions import Forecast
from forecontract.models.observations import Observation

from forecontract_scripted.errors import (
    PlannerInputError,
    PlannerObservationMismatchError,
    PlannerScriptError,
    PlannerScriptExhaustedError,
)
from forecontract_scripted.models import (
    PlannerScript,
    validate_existing_instance,
    validated,
)

__all__ = ["ScriptedPlanner"]


def _json_equal(left: object, right: object) -> bool:
    """Type-aware recursive JSON equality.

    ``True`` is not ``1`` and ``1`` is not ``1.0``. Object key order is irrelevant and array
    order is significant. The two fixture distributions are independent, so this small
    comparison is deliberately local rather than imported from the Fake package.
    """
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, bool) and isinstance(right, bool):
        return left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, float) and isinstance(right, float):
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(item, other) for item, other in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_json_equal(left[key], right[key]) for key in left)
    return False


class ScriptedPlanner:
    """A deterministic replay of one :class:`PlannerScript`."""

    def __init__(self, script: PlannerScript) -> None:
        validate_existing_instance(script, PlannerScriptError)
        self._script = validated(script, PlannerScript, script=True)
        self._cursor = 0

    @property
    def script_id(self) -> str:
        """The stable identifier of the replayed script."""
        return self._script.script_id

    async def plan(self, observation: Observation) -> Forecast:
        """Return the scripted forecast for this observation and consume one step.

        The observation is re-validated before exhaustion is checked, so an invalid input is
        always an input failure rather than an exhaustion failure. Matching compares the
        environment type and the complete state; the identifier, timestamp, summary, artifacts
        and metadata of the observation are ignored. The returned forecast is a detached copy.
        A supplied existing observation instance is read back from its own storage first, so a
        copied observation cannot hide an invalid stored field or an undeclared key behind a
        shallow instance check (F04).
        """
        if isinstance(observation, Observation):
            validate_existing_instance(observation, PlannerInputError)
        candidate = validated(observation, Observation)
        if self._cursor >= len(self._script.steps):
            raise PlannerScriptExhaustedError("every scripted planning step has been consumed",
                                              path=("steps",))
        step = self._script.steps[self._cursor]
        if step.expected_environment_type != candidate.environment_type or not _json_equal(
            step.expected_state, candidate.state
        ):
            raise PlannerObservationMismatchError(
                "the observation does not match the current scripted step",
                path=("steps", self._cursor),
            )
        forecast = step.forecast.model_copy(deep=True)
        self._cursor += 1
        return forecast
