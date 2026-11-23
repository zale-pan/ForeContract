"""The deterministic Fake environment adapter.

:class:`FakeEnvironmentAdapter` implements the existing Core :class:`Adapter` protocol exactly -
``name``, synchronous ``capabilities()``/``validate_action()`` and asynchronous
``start()``/``stop()``/``observe()``/``execute()`` - with no context, deadline, retry or
cancellation parameter.

The adapter is a replay of one :class:`FakeScenario`. It keeps actual environment state, the
visible observation and the execution receipt as three separate things: the *actual* state of a
stage is fixture data that no public accessor returns, the *observation* is whatever the
scenario author exposed, and the *receipt* is a detached copy of the scripted outcome. Nothing
copies hidden state into an observation or into a receipt.

Execution order is fixed: lifecycle check, then ``validate_action``, then exhaustion, then the
current step match, then the receipt is prepared, and only then do the stage and cursor advance
together. Any rejection leaves the lifecycle state, the stage and the cursor untouched.
"""

from __future__ import annotations

from typing import Final

from forecontract.models.actions import ActionSpec, AdapterCapabilities
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome
from forecontract.protocols.adapter import (
    InvalidActionPayloadError,
    UnsupportedActionError,
)

from forecontract_fake.errors import (
    FakeInputError,
    FakeLifecycleError,
    FakeScenarioError,
    FakeScriptExhaustedError,
    FakeStepMismatchError,
)
from forecontract_fake.models import (
    FAKE_ENVIRONMENT_TYPE,
    FakeScenario,
    json_equal,
    validate_existing_instance,
    validated,
)

__all__ = ["FakeEnvironmentAdapter"]

#: Lifecycle states. ``new`` is the initial state and a restart is never allowed.
NEW: Final[str] = "new"
STARTED: Final[str] = "started"
STOPPED: Final[str] = "stopped"


class FakeEnvironmentAdapter:
    """A deterministic, fully scripted environment adapter."""

    def __init__(self, scenario: FakeScenario) -> None:
        # The instance gate reads the record's own storage before anything else, so a copied or
        # constructed scenario cannot smuggle an undeclared key, an invalid stored field or a
        # nested forgery past the public constructor (F04).
        validate_existing_instance(scenario, FakeScenarioError)
        self._scenario = validated(scenario, FakeScenario, scenario=True)
        self._lifecycle = NEW
        self._cursor = 0

    # ------------------------------------------------------------------ declaration

    @property
    def name(self) -> str:
        """The stable adapter name declared by the scenario."""
        return self._scenario.adapter_name

    def capabilities(self) -> AdapterCapabilities:
        """Return a detached declaration of what this scenario can do.

        The supported action types are the catalog types in first-appearance order. The
        condition and outcome capability lists are deliberately empty: this fixture declares no
        production domain-verification capability, and Builder interoperation in the tests uses
        an explicit test-local wrapper instead of widening this declaration.
        """
        action_types: list[str] = []
        for action in self._scenario.allowed_actions:
            if action.action_type not in action_types:
                action_types.append(action.action_type)
        return AdapterCapabilities(
            schema_version="0.1",
            adapter_name=self._scenario.adapter_name,
            environment_type=FAKE_ENVIRONMENT_TYPE,
            supported_actions=action_types,
            supported_observation_types=["fake.state"],
            supported_condition_types=[],
            supported_outcome_types=[],
        )

    def validate_action(self, action: ActionSpec) -> None:
        """Reject an unsupported action type or a payload outside the catalog.

        The method performs no I/O, changes no state and does not depend on the lifecycle or on
        the current step. An unsupported action type is reported before any payload matching, so
        it always wins over a domain payload rejection. A supplied existing instance is read back
        from its own storage before the payload comparison, so a copied action cannot hide an
        invalid field or an undeclared key behind a shallow instance check (F04).
        """
        if isinstance(action, ActionSpec):
            validate_existing_instance(action, FakeInputError)
        candidate = validated(action, ActionSpec)
        catalog = [item for item in self._scenario.allowed_actions
                   if item.action_type == candidate.action_type]
        if not catalog:
            raise UnsupportedActionError("the scenario does not declare this action type")
        if not any(json_equal(item.payload, candidate.payload) for item in catalog):
            raise InvalidActionPayloadError("the action payload is outside the declared catalog")

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Move ``new`` to ``started``; a repeated start on ``started`` is a no-op."""
        if self._lifecycle == STARTED:
            return
        if self._lifecycle == STOPPED:
            raise FakeLifecycleError("a stopped adapter cannot be restarted", path=("lifecycle",))
        self._lifecycle = STARTED

    async def stop(self) -> None:
        """Move any non-stopped state to ``stopped``; a repeated stop is a no-op."""
        if self._lifecycle == STOPPED:
            return
        self._lifecycle = STOPPED

    # ------------------------------------------------------------------ interaction

    async def observe(self) -> Observation:
        """Return a detached copy of the current stage observation without consuming a step.

        The final stage stays observable after the script is exhausted.
        """
        self._require_started(("observe",))
        return self._scenario.stages[self._cursor].observation.model_copy(deep=True)

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        """Replay one scripted step and return a detached copy of its outcome.

        The order is fixed and observable: lifecycle, then validation, then exhaustion, then the
        current step match. The receipt is prepared before anything advances, and the stage and
        cursor advance exactly once only after the receipt exists, so a rejection at any point
        leaves the adapter exactly as it was. A scripted ``failed`` or ``unknown`` outcome still
        consumes its step.
        """
        self._require_started(("execute",))
        self.validate_action(action)
        if self._cursor >= len(self._scenario.steps):
            raise FakeScriptExhaustedError("every scripted step has already been executed",
                                           path=("steps",))
        candidate = validated(action, ActionSpec)
        step = self._scenario.steps[self._cursor]
        if step.action.action_type != candidate.action_type or not json_equal(
            step.action.payload, candidate.payload
        ):
            raise FakeStepMismatchError(
                "the action does not match the current scripted step",
                path=("steps", self._cursor),
            )
        receipt = step.outcome.model_copy(deep=True)
        self._cursor += 1
        return receipt

    # ------------------------------------------------------------------ internals

    def _require_started(self, path: tuple[str | int, ...]) -> None:
        if self._lifecycle != STARTED:
            raise FakeLifecycleError("this operation requires a started adapter", path=path)
