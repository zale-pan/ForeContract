"""Acceptance tests for the deterministic Contract Builder.

The suite drives the real ``ContractBuilder`` from the package source tree through every
gate, using a test-local adapter and a test-local domain hook fixture that genuinely inspect
the payload shape they claim to validate. No browser, Runtime, Fake loop, network or live
environment is involved, and the adapter records every call so the absence of lifecycle
interaction is provable rather than assumed.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.contracts import (
    ActionContract,
    ContractBuilder,
    ContractBuildError,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
    Precondition,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.protocols.adapter import (
    InvalidActionPayloadError,
    UnsupportedActionError,
)

AWARE = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
ENVIRONMENT = "browser"


class RecordingAdapter:
    """A test-local adapter that records calls and can never run a lifecycle method."""

    def __init__(
        self,
        *,
        name: str = "fake",
        environment_type: str = ENVIRONMENT,
        supported_actions: tuple = ("click",),
        supported_condition_types: tuple = ("visible",),
        supported_outcome_types: tuple = ("dom_changed",),
        supported_observation_types: tuple = ("dom",),
        validate=None,
        capabilities=None,
    ) -> None:
        self._name = name
        self._environment_type = environment_type
        self._supported_actions = list(supported_actions)
        self._supported_condition_types = list(supported_condition_types)
        self._supported_outcome_types = list(supported_outcome_types)
        self._supported_observation_types = list(supported_observation_types)
        self._validate = validate
        self._capabilities = capabilities
        self.calls: list[str] = []
        self.lifecycle_calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self) -> AdapterCapabilities:
        self.calls.append("capabilities")
        if self._capabilities is not None:
            return self._capabilities
        return AdapterCapabilities(
            schema_version="0.1",
            adapter_name=self._name,
            environment_type=self._environment_type,
            supported_actions=list(self._supported_actions),
            supported_observation_types=list(self._supported_observation_types),
            supported_condition_types=list(self._supported_condition_types),
            supported_outcome_types=list(self._supported_outcome_types),
        )

    def validate_action(self, action: ActionSpec) -> None:
        self.calls.append("validate_action")
        if self._validate is not None:
            return self._validate(action)
        return None

    async def start(self) -> None:
        self.lifecycle_calls.append("start")
        raise AssertionError("start must never be called by the Builder")

    async def stop(self) -> None:
        self.lifecycle_calls.append("stop")
        raise AssertionError("stop must never be called by the Builder")

    async def observe(self) -> Observation:
        self.lifecycle_calls.append("observe")
        raise AssertionError("observe must never be called by the Builder")

    async def execute(self, action: ActionSpec):
        self.lifecycle_calls.append("execute")
        raise AssertionError("execute must never be called by the Builder")


class RecordingHooks:
    """Injected hooks that record the call order and can misbehave on demand."""

    def __init__(self, *, pre=None, out=None, bind=None, check=None) -> None:
        self.calls: list[tuple] = []
        self._pre, self._out, self._bind, self._check = pre, out, bind, check

    def validate_precondition(self, condition: Precondition):
        self.calls.append(("validate_precondition", condition.condition_type))
        if self._pre is not None:
            return self._pre(condition)
        return None

    def validate_expected_outcome(self, outcome: ExpectedOutcome):
        self.calls.append(("validate_expected_outcome", outcome.outcome_type))
        if self._out is not None:
            return self._out(outcome)
        return None

    def validate_verification_binding(self, binding: VerificationBinding, outcome: ExpectedOutcome):
        self.calls.append(("validate_verification_binding", binding.outcome_index))
        if self._bind is not None:
            return self._bind(binding, outcome)
        return None

    def check_precondition(self, condition: Precondition, observation: Observation):
        self.calls.append(("check_precondition", condition.condition_type))
        if self._check is not None:
            return self._check(condition, observation)
        return VerificationStatus.PASS


class DomainFixture:
    """A meaningful test-local domain checker, not an always-succeeding stub."""

    def validate_action(self, action: ActionSpec) -> None:
        selector = action.payload.get("selector")
        if not isinstance(selector, str) or not selector.strip():
            raise InvalidActionPayloadError("the action payload needs a non-blank selector")

    def validate_precondition(self, condition: Precondition) -> None:
        if condition.condition_type != "visible":
            raise InvalidActionPayloadError("only a visible precondition is understood")
        if not isinstance(condition.payload.get("selector"), str):
            raise InvalidActionPayloadError("the precondition needs a selector")

    def validate_expected_outcome(self, outcome: ExpectedOutcome) -> None:
        if not isinstance(outcome.payload.get("marker"), str):
            raise InvalidActionPayloadError("the outcome needs a marker")

    def validate_verification_binding(self, binding: VerificationBinding, outcome: ExpectedOutcome) -> None:
        if binding.verifier_type != "dom_verifier":
            raise InvalidActionPayloadError("the verifier is unsupported")
        if binding.config.get("expected") != outcome.payload.get("marker"):
            raise InvalidActionPayloadError("the verifier config does not match the outcome")

    def check_precondition(self, condition: Precondition, observation: Observation) -> VerificationStatus:
        if observation.state.get("has_selector") is True:
            return VerificationStatus.PASS
        if "has_selector" not in observation.state:
            return VerificationStatus.INCONCLUSIVE
        return VerificationStatus.FAIL


def forecast(*outcomes: ExpectedOutcome, action_type: str = "click", payload: dict | None = None) -> Forecast:
    return Forecast(
        schema_version="0.1",
        forecast_id="f-1",
        proposed_action=ActionSpec(
            schema_version="0.1",
            action_type=action_type,
            payload={"selector": "#go"} if payload is None else payload,
        ),
        expected_outcomes=list(outcomes) or [outcome()],
    )


def outcome(index: int = 0, required: bool = True, outcome_type: str = "dom_changed") -> ExpectedOutcome:
    return ExpectedOutcome(
        schema_version="0.1",
        outcome_type=outcome_type,
        payload={"marker": f"ready-{index}"},
        required=required,
    )


def observation(*, environment_type: str = ENVIRONMENT, state: dict | None = None) -> Observation:
    return Observation(
        schema_version="0.1",
        observation_id="o-1",
        environment_type=environment_type,
        state={"has_selector": True} if state is None else state,
        timestamp=AWARE,
    )


def precondition(*, required: bool = True, condition_type: str = "visible") -> Precondition:
    return Precondition(
        schema_version="0.1",
        condition_type=condition_type,
        payload={"selector": "#go"},
        required=required,
    )


def binding(index: int, *, verifier: str = "dom_verifier", expected: str | None = None) -> VerificationBinding:
    return VerificationBinding(
        schema_version="0.1",
        outcome_index=index,
        verifier_type=verifier,
        config={"expected": f"ready-{index}" if expected is None else expected},
    )


def policy(*bindings: VerificationBinding) -> VerificationPolicy:
    return VerificationPolicy(schema_version="0.1", bindings=list(bindings) or [binding(0)])


def builder_inputs(**overrides) -> dict:
    fields = {
        "forecast": forecast(),
        "observation": observation(),
        "adapter": RecordingAdapter(validate=DomainFixture().validate_action),
        "hooks": domain_hooks(),
        "preconditions": [precondition()],
        "verification_policy": policy(),
        "timeout_ms": 100,
        "remaining_budget_ms": 250,
        "contract_id": "c-1",
        "goal_id": "g-1",
        "created_at": AWARE,
        "metadata": {"run": 1},
    }
    fields.update(overrides)
    return fields


def domain_hooks() -> RecordingHooks:
    fixture = DomainFixture()
    return RecordingHooks(
        pre=fixture.validate_precondition,
        out=fixture.validate_expected_outcome,
        bind=fixture.validate_verification_binding,
        check=fixture.check_precondition,
    )


def build(**overrides) -> ActionContract:
    return ContractBuilder().build(**builder_inputs(**overrides))


def rejection(**overrides) -> ContractBuildError:
    with pytest.raises(ContractBuildError) as info:
        build(**overrides)
    return info.value


def snapshot(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return copy.deepcopy(value)


# --- legal commits ------------------------------------------------------------

def test_a_single_required_outcome_commits_the_accepted_candidate() -> None:
    built = build()
    assert isinstance(built, ActionContract)
    assert built.contract_id == "c-1" and built.goal_id == "g-1"
    assert built.forecast_id == "f-1" and built.observation_id == "o-1"
    assert built.adapter_name == "fake" and built.environment_type == ENVIRONMENT
    assert built.action.action_type == "click"
    assert built.action.payload == {"selector": "#go"}
    assert [item.outcome_type for item in built.expected_outcomes] == ["dom_changed"]
    assert built.timeout_ms == 100.0
    assert built.created_at == AWARE
    assert built.metadata == {"run": 1}


def test_multiple_outcomes_keep_their_supplied_order_and_bindings() -> None:
    built = build(
        forecast=forecast(outcome(0), outcome(1, outcome_type="second"), outcome(2, outcome_type="third")),
        verification_policy=policy(binding(1), binding(0), binding(2)),
        adapter=RecordingAdapter(supported_outcome_types=("dom_changed", "second", "third"),
                                 validate=DomainFixture().validate_action),
    )
    assert [item.outcome_type for item in built.expected_outcomes] == ["dom_changed", "second", "third"]
    assert [item.outcome_index for item in built.verification_policy.bindings] == [1, 0, 2]


def test_optional_outcomes_with_a_complete_binding_are_committed() -> None:
    built = build(
        forecast=forecast(outcome(0), outcome(1, required=False, outcome_type="advisory")),
        verification_policy=policy(binding(0), binding(1)),
        adapter=RecordingAdapter(supported_outcome_types=("dom_changed", "advisory"),
                                 validate=DomainFixture().validate_action),
    )
    assert [item.required for item in built.expected_outcomes] == [True, False]
    assert len(built.verification_policy.bindings) == 2


def test_an_empty_precondition_list_is_a_legal_commit() -> None:
    assert build(preconditions=[]).preconditions == ()


def test_metadata_defaults_to_an_empty_object() -> None:
    built = build(metadata=None)
    assert built.metadata == {}
    assert json.loads(built.model_dump_json())["metadata"] == {}


def test_the_recovery_policy_is_the_explicit_abort_policy() -> None:
    policy_value = build().recovery_policy
    assert isinstance(policy_value, RecoveryPolicy)
    assert policy_value.mode == "abort"
    assert policy_value.schema_version == "0.1"


def test_created_at_accepts_the_iso_representation() -> None:
    assert build(created_at="2026-09-30T12:00:00Z").created_at == AWARE


def test_the_same_inputs_produce_the_same_content() -> None:
    first, second = build(), build()
    assert first.model_dump_json() == second.model_dump_json()
    assert first == second


def test_a_built_contract_roundtrips_through_both_ingress_routes() -> None:
    built = build()
    text = built.model_dump_json()
    assert ActionContract.model_validate_json(text).model_dump_json() == text
    assert ActionContract.model_validate(built.model_dump(mode="json")).model_dump_json() == text


def test_a_built_contract_is_deeply_immutable() -> None:
    built = build()
    before = built.model_dump_json()
    for mutate in (
        lambda: built.metadata.__setitem__("run", 99),
        lambda: built.action.payload.__setitem__("selector", "#evil"),
        lambda: built.preconditions[0].payload.__setitem__("selector", "#evil"),
        lambda: built.expected_outcomes[0].payload.__setitem__("marker", "evil"),
        lambda: built.verification_policy.bindings[0].config.__setitem__("expected", "evil"),
        lambda: built.preconditions.__setitem__(0, None),
        lambda: setattr(built, "contract_id", "other"),
    ):
        with pytest.raises((AttributeError, TypeError, ValidationError)):
            mutate()
        assert built.model_dump_json() == before


# --- gate two: adapter declaration and capability support ---------------------

def test_an_unsupported_action_is_rejected_before_the_payload_is_validated() -> None:
    adapter = RecordingAdapter(supported_actions=("type_only",))
    error = rejection(adapter=adapter)
    assert error.code == "UNSUPPORTED_CAPABILITY"
    assert error.location == ("forecast", "proposed_action", "action_type")
    assert adapter.calls == ["capabilities"], adapter.calls


def test_an_unsupported_optional_condition_type_is_rejected() -> None:
    error = rejection(preconditions=[precondition(required=False, condition_type="unknown_condition")])
    assert error.code == "UNSUPPORTED_CAPABILITY"
    assert error.location == ("preconditions", 0, "condition_type")


def test_an_unsupported_optional_outcome_type_is_rejected() -> None:
    error = rejection(forecast=forecast(outcome(0, required=False, outcome_type="unknown_outcome")))
    assert error.code == "UNSUPPORTED_CAPABILITY"
    assert error.location == ("forecast", "expected_outcomes", 0, "outcome_type")


def test_an_empty_capability_list_means_none_supported() -> None:
    adapter = RecordingAdapter(supported_condition_types=())
    assert rejection(adapter=adapter).code == "UNSUPPORTED_CAPABILITY"
    assert build(adapter=adapter, preconditions=[]).preconditions == ()


def test_no_downstream_callback_runs_after_an_earlier_rejection() -> None:
    adapter = RecordingAdapter(validate=lambda action: (_ for _ in ()).throw(AssertionError("unreachable")))
    hooks = domain_hooks()
    error = rejection(adapter=adapter, hooks=hooks, forecast=forecast(outcome(0, outcome_type="unknown")))
    assert error.code == "UNSUPPORTED_CAPABILITY"
    assert hooks.calls == []
    assert "validate_action" not in adapter.calls


def test_an_adapter_name_mismatch_is_reported_as_an_adapter_mismatch() -> None:
    adapter = RecordingAdapter(
        capabilities=AdapterCapabilities(
            schema_version="0.1",
            adapter_name="a-different-name",
            environment_type=ENVIRONMENT,
            supported_actions=["click"],
            supported_condition_types=["visible"],
            supported_outcome_types=["dom_changed"],
        )
    )
    error = rejection(adapter=adapter)
    assert error.code == "ADAPTER_MISMATCH"
    assert error.location == ("adapter", "name")


def test_an_environment_type_mismatch_is_reported_as_an_adapter_mismatch() -> None:
    error = rejection(observation=observation(environment_type="desktop"))
    assert error.code == "ADAPTER_MISMATCH"
    assert error.location == ("observation", "environment_type")


def test_declared_observation_types_are_not_consulted() -> None:
    """Observation carries no observation_type, so no observation capability is checked."""
    adapter = RecordingAdapter(supported_observation_types=())
    observation_value = observation(state={"has_selector": True, "anything": [1, 2, 3]})
    assert build(adapter=adapter, observation=observation_value).observation_id == "o-1"


def test_malformed_dependency_capabilities_are_a_dependency_failure() -> None:
    adapter = RecordingAdapter(capabilities={"adapter_name": "fake"})
    error = rejection(adapter=adapter)
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location[:2] == ("adapter", "capabilities")


def test_a_missing_capabilities_method_is_a_dependency_failure() -> None:
    class Bare:
        name = "fake"

    assert rejection(adapter=Bare()).code == "DEPENDENCY_FAILURE"


# --- gate three: domain callbacks ---------------------------------------------

def test_a_domain_payload_rejection_is_reported_as_an_invalid_payload() -> None:
    adapter = RecordingAdapter(validate=DomainFixture().validate_action)
    error = rejection(adapter=adapter, forecast=forecast(payload={}))
    assert error.code == "INVALID_DOMAIN_PAYLOAD"
    assert error.location == ("forecast", "proposed_action", "payload")


def test_an_adapter_unsupported_error_is_translated_to_a_capability_rejection() -> None:
    def validate(action: ActionSpec) -> None:
        raise UnsupportedActionError("not really supported")

    error = rejection(adapter=RecordingAdapter(validate=validate))
    assert error.code == "UNSUPPORTED_CAPABILITY"
    assert error.__cause__ is not None


def test_an_unexpected_adapter_exception_is_a_dependency_failure_with_its_cause() -> None:
    def validate(action: ActionSpec) -> None:
        raise RuntimeError("dependency exploded")

    error = rejection(adapter=RecordingAdapter(validate=validate))
    assert error.code == "DEPENDENCY_FAILURE"
    assert isinstance(error.__cause__, RuntimeError)


def test_a_non_none_validate_action_return_is_a_dependency_failure() -> None:
    error = rejection(adapter=RecordingAdapter(validate=lambda action: True))
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("adapter", "validate_action")


def test_an_adapter_that_mutates_the_action_is_a_dependency_failure() -> None:
    def validate(action: ActionSpec) -> None:
        action.payload["selector"] = "#evil"

    spec = ActionSpec(schema_version="0.1", action_type="click", payload={"selector": "#go"})
    error = rejection(adapter=RecordingAdapter(validate=validate), forecast=forecast(payload=spec.payload))
    assert error.code == "DEPENDENCY_FAILURE"
    assert spec.payload == {"selector": "#go"}


def test_the_domain_hooks_run_in_the_supplied_order() -> None:
    hooks = domain_hooks()
    build(
        hooks=hooks,
        preconditions=[precondition(), precondition(required=False)],
        forecast=forecast(outcome(0), outcome(1, outcome_type="second")),
        verification_policy=policy(binding(0), binding(1)),
        adapter=RecordingAdapter(supported_outcome_types=("dom_changed", "second"),
                                 validate=DomainFixture().validate_action),
    )
    assert [call[0] for call in hooks.calls] == [
        "validate_precondition",
        "validate_precondition",
        "validate_expected_outcome",
        "validate_expected_outcome",
        "validate_verification_binding",
        "validate_verification_binding",
        "check_precondition",
        "check_precondition",
    ]


def test_a_condition_hook_rejection_stops_the_remaining_callbacks() -> None:
    fixture = DomainFixture()
    hooks = RecordingHooks(
        pre=lambda condition: (_ for _ in ()).throw(InvalidActionPayloadError("bad condition")),
        out=fixture.validate_expected_outcome,
        bind=fixture.validate_verification_binding,
        check=fixture.check_precondition,
    )
    error = rejection(hooks=hooks)
    assert error.code == "INVALID_DOMAIN_PAYLOAD"
    assert error.location == ("preconditions", 0)
    assert [call[0] for call in hooks.calls] == ["validate_precondition"]


def test_a_binding_hook_rejection_is_a_binding_rejection() -> None:
    def reject(binding: VerificationBinding, outcome: ExpectedOutcome) -> None:
        raise InvalidActionPayloadError("unsupported verifier")

    error = rejection(hooks=RecordingHooks(bind=reject))
    assert error.code == "VERIFICATION_BINDING_INVALID"
    assert error.location == ("verification_policy", "bindings", 0)


def test_an_out_of_range_binding_index_is_rejected_before_the_hook_is_called() -> None:
    hooks = domain_hooks()
    error = rejection(hooks=hooks, verification_policy=policy(binding(3)))
    assert error.code == "VERIFICATION_BINDING_INVALID"
    assert error.location == ("verification_policy", "bindings", 0, "outcome_index")
    assert ("validate_verification_binding", 3) not in hooks.calls


def test_a_missing_hook_method_is_a_dependency_failure() -> None:
    class Partial:
        def validate_precondition(self, condition): return None

    error = rejection(hooks=Partial())
    assert error.code == "DEPENDENCY_FAILURE"


def test_a_validation_hook_returning_a_value_is_a_dependency_failure() -> None:
    error = rejection(hooks=RecordingHooks(pre=lambda condition: "accepted"))
    assert error.code == "DEPENDENCY_FAILURE"


def test_a_hook_that_mutates_its_argument_is_a_dependency_failure() -> None:
    def mutate(condition: Precondition) -> None:
        condition.payload["selector"] = "#evil"

    condition = precondition()
    error = rejection(hooks=RecordingHooks(pre=mutate), preconditions=[condition])
    assert error.code == "DEPENDENCY_FAILURE"
    assert condition.payload == {"selector": "#go"}


def test_an_unexpected_hook_exception_is_a_dependency_failure() -> None:
    def explode(condition: Precondition) -> None:
        raise RuntimeError("hook exploded")

    error = rejection(hooks=RecordingHooks(pre=explode))
    assert error.code == "DEPENDENCY_FAILURE"
    assert isinstance(error.__cause__, RuntimeError)


# --- gate four: structural completeness ---------------------------------------

def test_an_empty_required_outcome_list_is_rejected() -> None:
    empty = Forecast(
        schema_version="0.1",
        forecast_id="f-1",
        proposed_action=ActionSpec(schema_version="0.1", action_type="click", payload={"selector": "#go"}),
        expected_outcomes=[],
    )
    error = rejection(
        forecast=empty,
        verification_policy=VerificationPolicy(schema_version="0.1", bindings=[]),
    )
    assert error.code == "EXPECTED_OUTCOME_REQUIRED"
    assert error.location == ("forecast", "expected_outcomes")


def test_an_optional_only_candidate_is_rejected() -> None:
    error = rejection(
        forecast=forecast(outcome(0, required=False)),
        verification_policy=policy(binding(0)),
    )
    assert error.code == "EXPECTED_OUTCOME_REQUIRED"
    assert error.location == ("forecast", "expected_outcomes")


def test_a_duplicate_binding_index_is_rejected() -> None:
    error = rejection(verification_policy=policy(binding(0), binding(0)))
    assert error.code == "VERIFICATION_BINDING_INVALID"


def test_a_missing_binding_for_an_outcome_is_rejected() -> None:
    error = rejection(
        forecast=forecast(outcome(0), outcome(1, outcome_type="dom_changed")),
        verification_policy=policy(binding(0)),
    )
    assert error.code == "VERIFICATION_BINDING_INVALID"


# --- gate five: the budget ----------------------------------------------------

@pytest.mark.parametrize("value", [True, False, "100", None, 0, -1, 0.0, float("nan"), float("inf"), object()])
def test_a_non_positive_or_non_finite_timeout_is_a_budget_error(value) -> None:
    error = rejection(timeout_ms=value)
    assert error.code == "BUDGET_INVALID"
    assert error.location == ("timeout_ms",)


@pytest.mark.parametrize("value", [True, "250", None, 0, -1, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_remaining_budget_is_a_budget_error(value) -> None:
    error = rejection(remaining_budget_ms=value)
    assert error.code == "BUDGET_INVALID"
    assert error.location == ("remaining_budget_ms",)


def test_a_timeout_above_the_remaining_budget_is_rejected() -> None:
    error = rejection(timeout_ms=300, remaining_budget_ms=250)
    assert error.code == "BUDGET_INVALID"
    assert error.location == ("timeout_ms",)


def test_a_timeout_equal_to_the_remaining_budget_is_allowed() -> None:
    assert build(timeout_ms=250, remaining_budget_ms=250).timeout_ms == 250.0


def test_an_integer_timeout_above_the_remaining_budget_is_still_rejected() -> None:
    assert rejection(timeout_ms=251, remaining_budget_ms=250.5).code == "BUDGET_INVALID"


# --- gate six: precondition verdicts ------------------------------------------

@pytest.mark.parametrize("verdict", [VerificationStatus.FAIL, VerificationStatus.INCONCLUSIVE])
def test_a_required_precondition_that_is_not_pass_rejects_the_commit(verdict) -> None:
    error = rejection(hooks=RecordingHooks(check=lambda condition, observation: verdict))
    assert error.code == "PRECONDITION_UNSATISFIED"
    assert error.location == ("preconditions", 0)


@pytest.mark.parametrize("verdict", [VerificationStatus.FAIL, VerificationStatus.INCONCLUSIVE])
def test_an_optional_precondition_may_fail_and_is_still_kept(verdict) -> None:
    built = build(
        hooks=RecordingHooks(check=lambda condition, observation: verdict),
        preconditions=[precondition(required=False), precondition(required=False)],
    )
    assert len(built.preconditions) == 2
    assert [item.required for item in built.preconditions] == [False, False]


def test_a_required_precondition_still_passes_when_it_returns_pass() -> None:
    built = build(
        hooks=RecordingHooks(check=lambda condition, observation: VerificationStatus.PASS),
        preconditions=[precondition(required=True), precondition(required=False)],
    )
    assert [item.required for item in built.preconditions] == [True, False]


@pytest.mark.parametrize("value", [None, "pass", True, 1, VerificationStatus])
def test_a_check_that_does_not_return_a_verification_status_is_a_dependency_failure(value) -> None:
    error = rejection(hooks=RecordingHooks(check=lambda condition, observation: value))
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("preconditions", 0)


# --- gate one: malformed ingress ---------------------------------------------

@pytest.mark.parametrize("value", ["", "   ", 1, True, None])
def test_a_malformed_contract_identifier_is_an_input_error(value) -> None:
    error = rejection(contract_id=value)
    assert error.code == "INPUT_INVALID"
    assert error.location == ("contract_id",)


@pytest.mark.parametrize("value", [
    datetime(2026, 9, 30, 12, 0),  # noqa: DTZ001 - a naive datetime is the rejected input here
    "2026-09-30T12:00:00",
    "not a time",
    12345,
])
def test_a_naive_or_malformed_created_at_is_an_input_error(value) -> None:
    error = rejection(created_at=value)
    assert error.code == "INPUT_INVALID"
    assert error.location == ("created_at",)


@pytest.mark.parametrize("value", ["object", [1, 2], 7, True])
def test_a_malformed_metadata_argument_is_an_input_error(value) -> None:
    error = rejection(metadata=value)
    assert error.code == "INPUT_INVALID"


def test_a_malformed_forecast_object_is_an_input_error() -> None:
    error = rejection(forecast="not a forecast")
    assert error.code == "INPUT_INVALID"
    assert error.location[0] == "forecast"


@pytest.mark.parametrize("value", [
    (),
    (precondition(),),
    frozenset(),
    {"condition"},
    {"condition": precondition()},
    "",
    b"",
])
def test_only_a_real_list_of_preconditions_is_accepted(value) -> None:
    """F02: a tuple, set, mapping or text is rejected at ingress, not just a string."""
    error = rejection(preconditions=value)
    assert error.code == "INPUT_INVALID"
    assert error.location == ("preconditions",)


def test_a_generator_of_preconditions_is_rejected() -> None:
    """F02: a generator is not a list either, and is rejected before it is consumed."""
    error = rejection(preconditions=(item for item in [precondition()]))
    assert error.code == "INPUT_INVALID"
    assert error.location == ("preconditions",)


def test_empty_and_nonempty_lists_of_preconditions_are_still_accepted() -> None:
    assert build(preconditions=[]).preconditions == ()
    assert len(build(preconditions=[precondition()]).preconditions) == 1
    assert len(build(preconditions=[precondition(), precondition(required=False)]).preconditions) == 2


@pytest.mark.parametrize("payload", [
    {"a": float("nan")},
    {"a": float("inf")},
    {"a": b"raw"},
    {1: "value"},
    {"nested": {"deep": b"raw"}},
])
def test_an_in_place_contaminated_forecast_payload_is_an_input_error(payload) -> None:
    forecast_value = forecast()
    forecast_value.proposed_action.payload.clear()
    forecast_value.proposed_action.payload.update(payload)
    error = rejection(forecast=forecast_value)
    assert error.code == "INPUT_INVALID"
    assert error.location[0] == "forecast"


@pytest.mark.parametrize("payload", [{"a": float("nan")}, {"a": b"raw"}, {1: "x"}])
def test_contaminated_metadata_is_an_input_error(payload) -> None:
    error = rejection(metadata=payload)
    assert error.code == "INPUT_INVALID"
    assert error.location[0] == "metadata"


def test_a_contaminated_observation_is_an_input_error() -> None:
    observation_value = observation()
    observation_value.state["deep"] = {"bad": float("nan")}
    assert rejection(observation=observation_value).code == "INPUT_INVALID"


# --- caller input safety and dependency isolation -----------------------------

def test_the_builder_never_mutates_its_caller_objects_on_success() -> None:
    inputs = builder_inputs()
    before = {name: snapshot(value) for name, value in inputs.items() if isinstance(value, BaseModel)}
    ContractBuilder().build(**inputs)
    for name, value in before.items():
        assert snapshot(inputs[name]) == value, name


def test_the_builder_never_mutates_its_caller_objects_on_failure() -> None:
    built_inputs = builder_inputs(hooks=RecordingHooks(pre=lambda condition: (_ for _ in ()).throw(RuntimeError("boom"))))
    before = {name: snapshot(value) for name, value in built_inputs.items() if isinstance(value, BaseModel)}
    with pytest.raises(ContractBuildError):
        ContractBuilder().build(**built_inputs)
    for name, value in before.items():
        assert snapshot(built_inputs[name]) == value, name


def test_the_builder_never_calls_a_lifecycle_method() -> None:
    adapter = RecordingAdapter(validate=DomainFixture().validate_action)
    ContractBuilder().build(**builder_inputs(adapter=adapter))
    assert adapter.lifecycle_calls == []
    assert adapter.calls == ["capabilities", "validate_action"]


def test_a_failed_build_never_calls_a_lifecycle_method() -> None:
    adapter = RecordingAdapter(validate=DomainFixture().validate_action)
    with pytest.raises(ContractBuildError):
        ContractBuilder().build(**builder_inputs(adapter=adapter, timeout_ms=10_000))
    assert adapter.lifecycle_calls == []


def test_the_test_local_domain_fixture_is_meaningful() -> None:
    """The fixture rejects a wrong payload and accepts a right one, so it cannot be a stub."""
    fixture = DomainFixture()
    with pytest.raises(InvalidActionPayloadError):
        fixture.validate_action(ActionSpec(schema_version="0.1", action_type="click", payload={}))
    fixture.validate_action(ActionSpec(schema_version="0.1", action_type="click", payload={"selector": "#go"}))
    with pytest.raises(InvalidActionPayloadError):
        fixture.validate_precondition(precondition(condition_type="other"))
    assert fixture.check_precondition(precondition(), observation(state={"has_selector": False})) is VerificationStatus.FAIL
    assert fixture.check_precondition(precondition(), observation(state={})) is VerificationStatus.INCONCLUSIVE
    assert fixture.check_precondition(precondition(), observation()) is VerificationStatus.PASS


def test_a_matching_domain_fixture_commits_a_full_candidate() -> None:
    fixture = DomainFixture()
    adapter = RecordingAdapter(validate=fixture.validate_action)
    hooks = RecordingHooks(
        pre=fixture.validate_precondition,
        out=fixture.validate_expected_outcome,
        bind=fixture.validate_verification_binding,
        check=fixture.check_precondition,
    )
    built = build(adapter=adapter, hooks=hooks)
    assert built.verification_policy.bindings[0].config == {"expected": "ready-0"}


def test_the_error_text_never_echoes_a_payload_value() -> None:
    secret = "super-secret-payload-value"
    spec = ActionSpec(schema_version="0.1", action_type="click", payload={"selector": "#go"})
    spec.payload["token"] = secret
    adapter = RecordingAdapter(validate=lambda action: (_ for _ in ()).throw(InvalidActionPayloadError(secret)))
    error = rejection(adapter=adapter, forecast=forecast(payload=spec.payload))
    assert secret not in str(error)
    assert secret not in error.message


# --- round-002 corrections: F03 JSON-typed mutation, F04 budgets --------------
# round-002 fixed scalar type erasure (True/1, 1/1.0). round-003 closes the remaining F03
# cases, added at the end of this module: mapping-key type and container type erasure.

def typed_forecast() -> Forecast:
    """A candidate whose JSON scalars carry both a boolean and an integer."""
    return Forecast(
        schema_version="0.1",
        forecast_id="f-1",
        proposed_action=ActionSpec(
            schema_version="0.1",
            action_type="click",
            payload={"selector": "#go", "flag": True, "count": 1},
        ),
        expected_outcomes=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="dom_changed",
                payload={"marker": "ready-0", "count": 1},
            )
        ],
    )


def typed_observation() -> Observation:
    return Observation(
        schema_version="0.1",
        observation_id="o-1",
        environment_type=ENVIRONMENT,
        state={"has_selector": True, "flag": True},
        timestamp=AWARE,
    )


def typed_precondition() -> Precondition:
    return Precondition(
        schema_version="0.1",
        condition_type="visible",
        payload={"selector": "#go", "flag": True},
        required=True,
    )


def test_a_bool_to_integer_change_in_the_action_is_a_dependency_failure() -> None:
    """F03: True and 1 are equal in Python but not in JSON, so the rewrite is a failure."""
    def validate(action: ActionSpec) -> None:
        action.payload["flag"] = 1

    error = rejection(adapter=RecordingAdapter(validate=validate), forecast=typed_forecast())
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("forecast", "proposed_action")


def test_an_integer_to_float_change_in_the_action_is_a_dependency_failure() -> None:
    """F03: the same holds for 1 and 1.0."""
    def validate(action: ActionSpec) -> None:
        action.payload["count"] = 1.0

    error = rejection(adapter=RecordingAdapter(validate=validate), forecast=typed_forecast())
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("forecast", "proposed_action")


@pytest.mark.parametrize("case,location", [
    ("condition", ("preconditions", 0)),
    ("outcome", ("forecast", "expected_outcomes", 0)),
    ("observation", ("observation",)),
])
def test_a_json_type_change_in_a_hook_argument_is_a_dependency_failure(case: str, location) -> None:
    """F03: every hook argument surface detects a JSON type-changing rewrite."""
    hooks = RecordingHooks()
    if case == "condition":
        def rewrite(condition: Precondition) -> None:
            condition.payload["flag"] = 1

        hooks = RecordingHooks(pre=rewrite)
    elif case == "outcome":
        def rewrite(outcome: ExpectedOutcome) -> None:
            outcome.payload["count"] = 1.0

        hooks = RecordingHooks(out=rewrite)
    else:
        def rewrite(condition: Precondition, observation: Observation) -> VerificationStatus:
            observation.state["flag"] = 1
            return VerificationStatus.PASS

        hooks = RecordingHooks(check=rewrite)
    error = rejection(hooks=hooks, forecast=typed_forecast(), observation=typed_observation(),
                      preconditions=[typed_precondition()])
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == location
    if case == "condition":
        assert ("validate_expected_outcome", "dom_changed") not in hooks.calls
    if case == "outcome":
        assert ("validate_verification_binding", 0) not in hooks.calls


def test_a_binding_argument_mutation_attempt_is_a_dependency_failure() -> None:
    """F03: a binding config refuses an in-place write, and that is still a failure."""
    def rewrite(binding: VerificationBinding, outcome: ExpectedOutcome) -> None:
        binding.config["audit_changed"] = 1

    error = rejection(hooks=RecordingHooks(bind=rewrite))
    assert error.code == "DEPENDENCY_FAILURE"


def test_an_unchanged_json_scalar_is_not_reported_as_a_mutation() -> None:
    """F03 control: rewriting a value with an equal value of the same JSON type is fine."""
    def rewrite(condition: Precondition) -> None:
        condition.payload["flag"] = True

    built = build(hooks=RecordingHooks(pre=rewrite), preconditions=[typed_precondition()])
    assert built.preconditions[0].payload["flag"] is True


def test_a_detected_mutation_preserves_the_exception_contract() -> None:
    """F03: the failure is a located ContractBuildError, not a raw Python error."""
    def explode(condition: Precondition) -> None:
        raise RuntimeError("dependency exploded")

    error = rejection(hooks=RecordingHooks(pre=explode))
    assert isinstance(error, ContractBuildError)
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("preconditions", 0)
    assert isinstance(error.__cause__, RuntimeError)


@pytest.mark.parametrize("value", [10**400, -(10**400), 10**309])
def test_a_huge_integer_timeout_is_a_budget_error_without_overflow(value) -> None:
    """F04: an unrepresentable integer is a range error at the budget gate."""
    error = rejection(timeout_ms=value)
    assert error.code == "BUDGET_INVALID"
    assert error.location == ("timeout_ms",)
    assert isinstance(error.__cause__, OverflowError)


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_a_huge_integer_remaining_budget_is_a_budget_error_without_overflow(value) -> None:
    error = rejection(remaining_budget_ms=value)
    assert error.code == "BUDGET_INVALID"
    assert error.location == ("remaining_budget_ms",)


def test_the_timeout_budget_is_reported_before_the_remaining_budget() -> None:
    error = rejection(timeout_ms=10**400, remaining_budget_ms=10**400)
    assert error.code == "BUDGET_INVALID"
    assert error.location == ("timeout_ms",)


def test_a_huge_budget_is_reported_before_any_precondition_is_evaluated() -> None:
    """F04: the budget gate keeps its position ahead of precondition evaluation."""
    hooks = RecordingHooks(check=lambda condition, observation: VerificationStatus.FAIL)
    error = rejection(hooks=hooks, timeout_ms=10**400)
    assert error.code == "BUDGET_INVALID"
    assert ("check_precondition", "visible") not in hooks.calls


def test_budget_equality_and_ordinary_ranges_still_work() -> None:
    assert build(timeout_ms=250, remaining_budget_ms=250).timeout_ms == 250.0
    assert build(timeout_ms=1, remaining_budget_ms=250).timeout_ms == 1.0
    assert build(timeout_ms=10**18, remaining_budget_ms=10**18).timeout_ms == float(10**18)


# --- round-003 correction: F03 key-type and container-type mutation -----------
# ``"1" == 1`` is false in Python, but the round-002 comparison first rewrote every key with
# ``str()``, so the string key "1" and the integer key 1 produced the same comparison input.
# ``[1] == (1,)`` is false too, but both were flattened to the same "array" shape. These tests
# pin the two refused substitutions, the surfaces they must be refused on and the ordinary
# same-type rewrites that must keep committing.

def keyed_forecast() -> Forecast:
    """A candidate whose action payload holds a numeric *string* key and a nested array."""
    return Forecast(
        schema_version="0.1",
        forecast_id="f-1",
        proposed_action=ActionSpec(
            schema_version="0.1",
            action_type="click",
            payload={"selector": "#go", "1": True, "nested": [1]},
        ),
        expected_outcomes=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="dom_changed",
                payload={"marker": "ready-0", "1": True, "nested": [1]},
            )
        ],
    )


def keyed_precondition() -> Precondition:
    return Precondition(
        schema_version="0.1",
        condition_type="visible",
        payload={"selector": "#go", "1": True, "nested": [1]},
        required=True,
    )


def keyed_observation() -> Observation:
    return Observation(
        schema_version="0.1",
        observation_id="o-1",
        environment_type=ENVIRONMENT,
        state={"has_selector": True, "1": True, "nested": [1]},
        timestamp=AWARE,
    )


def replace_string_key_with_integer_key(container) -> None:
    """Re-insert the value under the integer key 1: equal in Python, a different JSON key."""
    container.payload[1] = container.payload.pop("1")


def replace_dictionary_key_with_integer_key(value: dict) -> None:
    value[1] = value.pop("1")


def replace_list_with_equal_tuple(container) -> None:
    """Replace the nested array with a tuple of equal elements: equal to ``==``, not in JSON."""
    container.payload["nested"] = tuple(container.payload["nested"])


def _action_payload(container) -> dict:
    """The JSON object a callback surface exposes: the model payload or the observation state."""
    return getattr(container, "payload", container.state)


def test_a_string_key_replaced_by_an_integer_key_in_the_action_is_a_dependency_failure() -> None:
    """F03: an integer JSON object key is never the string key that was supplied."""
    def rewrite(action: ActionSpec) -> None:
        replace_string_key_with_integer_key(action)

    error = rejection(adapter=RecordingAdapter(validate=rewrite), forecast=keyed_forecast())
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("forecast", "proposed_action")


def test_a_string_key_replaced_by_an_integer_key_inside_metadata_is_a_dependency_failure() -> None:
    """F03: the top-level metadata surface is policed by the same comparison."""
    def rewrite(action: ActionSpec) -> None:
        replace_dictionary_key_with_integer_key(action.metadata)

    error = rejection(
        adapter=RecordingAdapter(validate=rewrite),
        forecast=keyed_forecast(),
        metadata={"1": True, "run": 1},
    )
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("adapter", "validate_action")


def test_a_list_replaced_by_an_equal_tuple_in_the_action_is_a_dependency_failure() -> None:
    """F03: a JSON array stays a list; a tuple substitution is a change, not an equal value."""
    def rewrite(action: ActionSpec) -> None:
        replace_list_with_equal_tuple(action)

    error = rejection(adapter=RecordingAdapter(validate=rewrite), forecast=keyed_forecast())
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == ("forecast", "proposed_action")


@pytest.mark.parametrize("mutation,surface,location,next_callback", [
    ("key", "condition", ("preconditions", 0), ("validate_expected_outcome", "dom_changed")),
    ("tuple", "condition", ("preconditions", 0), ("validate_expected_outcome", "dom_changed")),
    ("key", "outcome", ("forecast", "expected_outcomes", 0), ("validate_verification_binding", 0)),
    ("tuple", "outcome", ("forecast", "expected_outcomes", 0), ("validate_verification_binding", 0)),
    ("key", "binding", ("verification_policy", "bindings", 0), None),
    ("tuple", "binding", ("verification_policy", "bindings", 0), None),
    ("key", "observation", ("observation",), None),
    ("tuple", "observation", ("observation",), None),
])
def test_a_key_or_container_type_change_in_a_hook_argument_is_a_dependency_failure(
    mutation: str, surface: str, location, next_callback
) -> None:
    """F03: every hook argument surface refuses both substitutions, at the mutated argument."""
    def rewrite(value, _outcome=None) -> None:
        if mutation == "key":
            replace_dictionary_key_with_integer_key(_action_payload(value))
        else:
            _action_payload(value)["nested"] = tuple(_action_payload(value)["nested"])

    hooks = RecordingHooks()
    if surface == "condition":
        hooks = RecordingHooks(pre=lambda condition: rewrite(condition))
    elif surface == "outcome":
        hooks = RecordingHooks(out=lambda outcome: rewrite(outcome))
    elif surface == "binding":
        hooks = RecordingHooks(bind=lambda binding, outcome: replace_dictionary_key_with_integer_key(binding.config))
    else:
        def check(condition: Precondition, observation: Observation) -> VerificationStatus:
            rewrite(observation)
            return VerificationStatus.PASS

        hooks = RecordingHooks(check=check)
    error = rejection(hooks=hooks, forecast=keyed_forecast(), observation=keyed_observation(),
                      preconditions=[keyed_precondition()])
    assert error.code == "DEPENDENCY_FAILURE"
    assert error.location == location
    if next_callback is not None:
        assert next_callback not in hooks.calls


def test_an_unchanged_string_key_and_list_are_not_reported_as_a_mutation() -> None:
    """F03 control: rewriting the same key and the same array type with equal values is fine."""
    def rewrite(action: ActionSpec) -> None:
        action.payload["1"] = True
        action.payload["nested"] = [1]

    built = build(adapter=RecordingAdapter(validate=rewrite), forecast=keyed_forecast())
    assert built.action.payload["1"] is True
    assert built.action.payload["nested"] == [1]
    assert built.model_dump(mode="python")["action"]["payload"] == {
        "selector": "#go", "1": True, "nested": [1],
    }


def test_the_committed_content_keeps_string_keys_and_array_shape_after_a_rejected_key_change() -> None:
    """F03: the rejected substitution changes neither the caller's data nor the commit shape."""
    def rewrite(action: ActionSpec) -> None:
        replace_string_key_with_integer_key(action)

    candidate = keyed_forecast()
    error = rejection(adapter=RecordingAdapter(validate=rewrite), forecast=candidate)
    assert error.code == "DEPENDENCY_FAILURE"
    assert candidate.proposed_action.payload == {"selector": "#go", "1": True, "nested": [1]}
    built = build(forecast=keyed_forecast())
    dumped = built.model_dump(mode="python")
    assert dumped["action"]["payload"] == {"selector": "#go", "1": True, "nested": [1]}
    assert isinstance(dumped["action"]["payload"]["nested"], list)
    assert json.loads(built.model_dump_json())["action"]["payload"]["nested"] == [1]


# --- P2-T05 round-001: the optional committed recovery policy ------------------

def test_an_omitted_recovery_policy_keeps_the_original_abort_behaviour() -> None:
    """The Builder commits the same two-field abort policy whether or not one is supplied."""
    default_built = build()
    explicit_built = build(recovery_policy=RecoveryPolicy(schema_version="0.1", mode="abort"))
    assert default_built.recovery_policy == explicit_built.recovery_policy
    assert default_built.recovery_policy.model_dump() == {"schema_version": "0.1",
                                                          "mode": "abort"}
    assert json.loads(default_built.model_dump_json())["recovery_policy"] == {
        "schema_version": "0.1", "mode": "abort"}
    assert default_built.model_dump_json() == explicit_built.model_dump_json()


def test_a_supplied_bounded_policy_is_committed_immutably() -> None:
    policy = RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=2,
                            allow_replan=True)
    built = build(recovery_policy=policy)
    assert built.recovery_policy.model_dump() == {"schema_version": "0.1", "mode": "bounded",
                                                  "max_retries": 2, "allow_replan": True}
    dumped = json.loads(built.model_dump_json())["recovery_policy"]
    assert dumped == {"schema_version": "0.1", "mode": "bounded", "max_retries": 2,
                      "allow_replan": True}
    # The committed policy is a detached snapshot, not the caller's object.
    assert built.recovery_policy is not policy
    assert built.recovery_policy == policy
    # A committed contract stays deeply immutable: neither route can rewrite the policy.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        built.recovery_policy.max_retries = 5
    with pytest.raises(ValidationError):
        built.recovery_policy = policy


def test_a_rejected_recovery_policy_is_an_input_error() -> None:
    for bad in ({"mode": "bounded"},
                {"mode": "retry"},
                {"mode": "bounded", "max_retries": -1, "allow_replan": True},
                {"mode": "bounded", "max_retries": 0, "allow_replan": "yes"},
                {"mode": "abort", "max_retries": 0, "allow_replan": False},
                "abort",
                {"schema_version": "9.9", "mode": "abort"}):
        error = rejection(recovery_policy=bad)
        assert error.code == "INPUT_INVALID"
        assert error.location and error.location[0] == "recovery_policy"
        assert len(str(error)) < 240


def test_the_recovery_policy_does_not_change_the_other_builder_gates() -> None:
    """The new input changes no earlier gate: a rejected candidate still fails the same way."""
    bounded = RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=1,
                             allow_replan=True)
    mismatched = RecordingAdapter(capabilities=AdapterCapabilities(
        schema_version="0.1", adapter_name="a-different-name", environment_type=ENVIRONMENT,
        supported_actions=["click"], supported_condition_types=["visible"],
        supported_outcome_types=["dom_changed"]))
    for overrides in ({"adapter": RecordingAdapter(supported_condition_types=())},
                      {"adapter": mismatched},
                      {"timeout_ms": 500},
                      {"verification_policy": VerificationPolicy(
                          schema_version="0.1", bindings=[])}):
        with_policy = rejection(recovery_policy=bounded, **overrides)
        without_policy = rejection(**overrides)
        assert with_policy.code == without_policy.code
        assert with_policy.location == without_policy.location


def test_an_explicitly_null_bounded_control_is_refused_through_the_builder() -> None:
    """F02: the Builder refuses an abort policy that explicitly carries a bounded control."""
    for policy in ({"schema_version": "0.1", "mode": "abort", "max_retries": None},
                   {"schema_version": "0.1", "mode": "abort", "allow_replan": None}):
        error = rejection(recovery_policy=policy)
        assert error.code == "INPUT_INVALID"
        assert error.location and error.location[0] == "recovery_policy"


def test_a_legal_policy_roundtrips_through_the_committed_contract() -> None:
    """F03 control: both committed policy shapes survive serialization and reload unchanged."""
    for policy in (None,
                   RecoveryPolicy(schema_version="0.1", mode="abort"),
                   RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=0,
                                  allow_replan=True)):
        built = build(**({} if policy is None else {"recovery_policy": policy}))
        reloaded = ActionContract.model_validate_json(built.model_dump_json())
        assert reloaded.recovery_policy == built.recovery_policy
        assert reloaded.model_dump_json() == built.model_dump_json()
        assert json.loads(built.model_dump_json())["recovery_policy"]["mode"] in ("abort",
                                                                                 "bounded")
