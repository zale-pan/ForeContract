"""Acceptance tests for the independent task-completion oracle.

The oracle answers a different question from the step verifier, and these tests keep the two
separated: a passing step verdict must not make a task verdict pass, a failing step verdict must
not stop the oracle from passing on independent criteria, and a completion hint must not
influence anything. The joint cases drive the real Fake adapter and the real ScriptedPlanner
through test-local capability and hook wrappers.
"""

from __future__ import annotations

import asyncio
import datetime
import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
FAKE_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "fake" / "src"
SCRIPTED_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "scripted" / "src"
for candidate in (CORE_SRC, FAKE_SRC, SCRIPTED_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts import (
    ContractBuilder,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import (
    ActualOutcome,
    ErrorInfo,
    ExecutionStatus,
    VerificationStatus,
)
from forecontract.verification import (
    PathExistsRule,
    StepVerifier,
    TaskCompletionCriteria,
    TaskCompletionOracle,
    ValueEqualsRule,
    VerificationError,
    VerificationEvidence,
)

NOW = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
MARKER = "SYNTHETIC_PRIVATE_VALUE"


def observation(state: object | None = None, *, identifier: str = "obs-1") -> Observation:
    return Observation(schema_version="0.1", observation_id=identifier, environment_type="fake",
                       state=state if state is not None else {"orders": {"count": 2}}, timestamp=NOW)


def evidence(state: object | None = None, *, complete: bool = True,
             identifier: str = "obs-1") -> VerificationEvidence:
    return VerificationEvidence(schema_version="0.1",
                                observation=observation(state, identifier=identifier),
                                state_complete=complete)


def condition(rule_type: str, path: list, *, expected: object = None,
              required: bool = True) -> ExpectedOutcome:
    payload: dict = {"schema_version": "0.1", "rule_type": rule_type, "path": path}
    if rule_type == "value_equals":
        payload["expected"] = expected
    return ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=required,
                           payload=payload)


def criteria(conditions: list[ExpectedOutcome], *, goal_id: str = "g-1") -> TaskCompletionCriteria:
    return TaskCompletionCriteria(schema_version="0.1", goal_id=goal_id, conditions=conditions)


def evaluate(criteria_record: TaskCompletionCriteria, piece: VerificationEvidence):
    return TaskCompletionOracle().evaluate_completion(criteria=criteria_record, evidence=piece)


# --- A06/A07: the independent oracle ---

def test_the_oracle_uses_independently_supplied_criteria() -> None:
    record = criteria([condition("value_equals", ["orders", "count"], expected=2)])
    result = evaluate(record, evidence({"orders": {"count": 2}}))
    assert result.status is VerificationStatus.PASS
    assert result.reason == "REQUIRED_PASSED"
    assert result.verifier == "json_task"
    assert dict(result.metadata) == {"scope": "task", "goal_id": "g-1",
                                     "observation_id": "obs-1", "state_complete": True}
    assert [item.outcome_index for item in result.outcome_results] == [0]


def test_a_task_failure_is_not_a_permanent_verdict() -> None:
    unmet = criteria([condition("path_exists", ["not_there"])])
    assert evaluate(unmet, evidence({"a": 1})).status is VerificationStatus.FAIL
    satisfied = criteria([condition("path_exists", ["a"])])
    assert evaluate(satisfied, evidence({"a": 1})).status is VerificationStatus.PASS


@pytest.mark.parametrize("statuses,expected", [
    (["PASS", "PASS"], VerificationStatus.PASS),
    (["PASS", "FAIL"], VerificationStatus.FAIL),
    (["FAIL", "INCONCLUSIVE"], VerificationStatus.FAIL),
    (["PASS", "INCONCLUSIVE"], VerificationStatus.INCONCLUSIVE),
])
def test_the_task_verdict_uses_required_all_aggregation(statuses: list,
                                                       expected: VerificationStatus) -> None:
    conditions = []
    for status in statuses:
        if status == "PASS":
            conditions.append(condition("path_exists", ["present"]))
        elif status == "FAIL":
            conditions.append(condition("path_exists", ["missing"]))
        else:
            conditions.append(condition("value_equals", ["missing"], expected=1))
    result = evaluate(criteria(conditions), evidence({"present": 1}))
    assert result.status is expected
    assert result.reason in {"REQUIRED_PASSED", "REQUIRED_FAILED", "REQUIRED_INCONCLUSIVE"}


def test_an_optional_condition_never_decides_but_is_kept() -> None:
    record = criteria([
        condition("path_exists", ["present"]),
        condition("path_exists", ["missing"], required=False),
    ])
    result = evaluate(record, evidence({"present": 1}))
    assert result.status is VerificationStatus.PASS
    assert [item.status for item in result.outcome_results] == [
        VerificationStatus.PASS, VerificationStatus.FAIL
    ]


def test_at_least_one_required_condition_is_enforced() -> None:
    with pytest.raises(VerificationError) as info:
        criteria([condition("path_exists", ["a"], required=False)])
    assert info.value.code == "VERIFICATION_INPUT_INVALID"


def test_criteria_cannot_be_empty() -> None:
    with pytest.raises(VerificationError):
        TaskCompletionCriteria(schema_version="0.1", goal_id="g-1", conditions=[])


def test_the_goal_identifier_is_preserved_and_must_be_non_blank() -> None:
    record = criteria([condition("path_exists", ["a"])], goal_id="goal x")
    assert record.goal_id == "goal x"
    with pytest.raises(VerificationError):
        criteria([condition("path_exists", ["a"])], goal_id="  ")


def test_criteria_and_evidence_are_validated_before_any_evaluation() -> None:
    record = criteria([
        condition("path_exists", ["a"]),
        condition("unsupported", ["b"]),
    ])
    with pytest.raises(VerificationError) as info:
        evaluate(record, evidence({"a": 1}))
    assert info.value.code == "VERIFICATION_RULE_INVALID"
    assert info.value.location[0] == "criteria"
    with pytest.raises(VerificationError) as input_info:
        evaluate(record, {"state_complete": True})
    assert input_info.value.code == "VERIFICATION_INPUT_INVALID"


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"goal_id": 17},
    {"conditions": "not-a-list"},
])
def test_a_forged_copied_criteria_record_is_refused(mutation: dict) -> None:
    """F01/F02: an update that no longer matches the schema is refused at the copy itself."""
    from pydantic import ValidationError

    with pytest.raises((VerificationError, ValidationError)):
        criteria([condition("path_exists", ["a"])]).model_copy(update=mutation)


def test_a_forged_constructed_criteria_record_is_refused_at_public_ingress() -> None:
    forged = TaskCompletionCriteria.model_construct(
        schema_version="9.9", goal_id="g-1", conditions=[condition("path_exists", ["a"])])
    with pytest.raises(VerificationError):
        TaskCompletionCriteria.model_validate(forged)
    with pytest.raises(VerificationError):
        evaluate(forged, evidence({"a": 1}))


def test_an_unknown_self_key_is_refused_rather_than_raising_a_type_error() -> None:
    fields = {"schema_version": "0.1", "goal_id": "g-1",
              "conditions": [condition("path_exists", ["a"]).model_dump(mode="json")]}
    with pytest.raises(VerificationError) as info:
        TaskCompletionCriteria.model_validate({**fields, "self": MARKER})
    assert MARKER not in str(info.value)
    with pytest.raises(VerificationError):
        TaskCompletionCriteria(**{**fields, "self": MARKER})


def test_the_oracle_is_deterministic_and_isolated() -> None:
    record = criteria([condition("value_equals", ["a"], expected=1)])
    piece = evidence({"a": 1})
    first = evaluate(record, piece)
    second = evaluate(record, piece)
    assert first == second and first is not second
    piece.observation.state["a"] = 99
    assert evaluate(record, evidence({"a": 1})).status is VerificationStatus.PASS


def test_criteria_own_their_conditions_and_do_not_alias_the_caller() -> None:
    """F04: mutating the caller's outcome after construction cannot change the criteria."""
    caller_payload = {"b": [1]}
    caller_outcome = condition("path_exists", ["a"])
    caller_outcome = caller_outcome.model_copy(update={"payload": caller_payload})
    built = criteria([caller_outcome])
    caller_payload["b"].append(2)
    assert built.conditions[0].payload["b"] == [1]
    built.conditions[0].payload["b"].append(3)
    assert caller_payload["b"] == [1, 2]


def test_a_copied_invalid_nested_condition_is_refused() -> None:
    """F02: a nested condition that no longer matches its schema is refused at both ingresses."""
    forged = ExpectedOutcome.model_construct(
        schema_version="9.9", outcome_type="json_rule", required=True,
        payload={"schema_version": "0.1", "rule_type": "path_exists", "path": ["a"]})
    with pytest.raises(VerificationError):
        criteria([forged])
    with pytest.raises(VerificationError):
        TaskCompletionCriteria.model_validate({"schema_version": "0.1", "goal_id": "g-1",
                                               "conditions": [forged]})


def test_a_rule_error_in_a_later_condition_is_not_masked_by_an_earlier_failure() -> None:
    record = criteria([
        condition("path_exists", ["missing"]),
        condition("value_equals", ["a"]),
    ])
    malformed = TaskCompletionCriteria.model_construct(
        schema_version="0.1", goal_id="g-1",
        conditions=[record.conditions[0],
                    ExpectedOutcome.model_construct(schema_version="0.1",
                                                    outcome_type="json_rule", required=True,
                                                    payload={"schema_version": "0.1",
                                                             "rule_type": "unknown",
                                                             "path": ["a"]})])
    with pytest.raises(VerificationError) as info:
        evaluate(malformed, evidence({"a": 1}))
    assert info.value.code == "VERIFICATION_RULE_INVALID"


# --- A07/A10: step and task are separate, and hints do not decide ---

def _step_contract(*, expected: int, rule_type: str = "value_equals") -> object:
    outcome = condition(rule_type, ["orders", "count"], expected=expected) \
        if rule_type == "value_equals" else condition(rule_type, ["orders", "count"])
    return ContractBuilder().build(
        forecast=Forecast(schema_version="0.1", forecast_id="f-1", proposed_action=ActionSpec(
            schema_version="0.1", action_type="click"), expected_outcomes=[outcome]),
        observation=observation({"orders": {"count": 2}}),
        adapter=_DeclaringFakeAdapter(_fake_adapter()),
        hooks=_LocalHooks(),
        preconditions=[],
        verification_policy=VerificationPolicy(
            schema_version="0.1",
            bindings=[VerificationBinding(schema_version="0.1", outcome_index=0,
                                          verifier_type="json_rule")]),
        timeout_ms=100, remaining_budget_ms=250, contract_id="c-1", goal_id="g-1",
        created_at=NOW,
    )


class _DeclaringFakeAdapter:
    """A test-local capability wrapper; the production Fake still declares neither capability."""

    def __init__(self, adapter: object) -> None:
        self._adapter = adapter

    @property
    def name(self) -> str:
        return self._adapter.name

    def capabilities(self) -> AdapterCapabilities:
        return self._adapter.capabilities().model_copy(update={
            "supported_condition_types": ["json_rule"],
            "supported_outcome_types": ["json_rule"],
        })

    def validate_action(self, candidate: ActionSpec) -> None:
        self._adapter.validate_action(candidate)


class _LocalHooks:
    """A test-local hook set that accepts json_rule payloads and passes every precondition."""

    def validate_precondition(self, precondition: object) -> None:
        del precondition

    def validate_expected_outcome(self, outcome: object) -> None:
        if outcome.outcome_type != "json_rule":
            raise ValueError("only json_rule is supported here")

    def validate_verification_binding(self, binding: object, outcome: object) -> None:
        if binding.verifier_type != "json_rule":
            raise ValueError("only json_rule is supported here")
        del outcome

    def check_precondition(self, precondition: object, observation: object):
        return VerificationStatus.PASS


def _fake_adapter() -> object:
    from forecontract_fake import (
        FakeEnvironmentAdapter,
        FakeScenario,
        FakeStage,
        FakeStep,
    )

    built = FakeScenario(
        schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
        allowed_actions=[ActionSpec(schema_version="0.1", action_type="click")],
        stages=[
            FakeStage(schema_version="0.1", actual_state={"orders": {"count": 2}},
                      observation=observation({"orders": {"count": 2}})),
            FakeStage(schema_version="0.1", actual_state={"orders": {"count": 2}},
                      observation=observation({"orders": {"count": 2}}, identifier="obs-2")),
        ],
        steps=[FakeStep(schema_version="0.1", step_id="step-0",
                        action=ActionSpec(schema_version="0.1", action_type="click"),
                        outcome=ActualOutcome(schema_version="0.1", outcome_id="out-0",
                                              status="succeeded", changes={}))])
    return FakeEnvironmentAdapter(built)


def test_a_passing_step_does_not_make_the_task_pass() -> None:
    contract = _step_contract(expected=2)
    step = StepVerifier().verify_step(contract=contract,
                                     evidence=evidence({"orders": {"count": 2}}))
    assert step.status is VerificationStatus.PASS
    unmet = criteria([condition("path_exists", ["goal_achieved"])])
    task = evaluate(unmet, evidence({"orders": {"count": 2}}))
    assert task.status is VerificationStatus.FAIL
    missing = criteria([condition("value_equals", ["goal_achieved"], expected=True)])
    assert evaluate(missing, evidence({"orders": {"count": 2}})).status \
        is VerificationStatus.INCONCLUSIVE


def test_a_failing_step_does_not_stop_an_independent_task_pass() -> None:
    contract = _step_contract(expected=99)
    step = StepVerifier().verify_step(contract=contract,
                                     evidence=evidence({"orders": {"count": 2}}))
    assert step.status is VerificationStatus.FAIL
    satisfied = criteria([condition("path_exists", ["orders", "count"])])
    assert evaluate(satisfied, evidence({"orders": {"count": 2}})).status \
        is VerificationStatus.PASS


def test_a_completion_hint_has_no_influence_on_the_oracle() -> None:
    optimistic = Forecast(schema_version="0.1", forecast_id="f-1",
                          proposed_action=ActionSpec(schema_version="0.1", action_type="click"),
                          expected_outcomes=[condition("path_exists", ["orders"])],
                          task_completion_hint=True)
    assert optimistic.task_completion_hint is True
    unmet = criteria([condition("path_exists", ["goal_achieved"])])
    assert evaluate(unmet, evidence({"orders": {"count": 2}})).status is VerificationStatus.FAIL
    satisfied = criteria([condition("path_exists", ["orders"])])
    assert evaluate(satisfied, evidence({"orders": {"count": 2}})).status \
        is VerificationStatus.PASS


def test_the_oracle_never_reads_a_contract_or_a_hint() -> None:
    import inspect

    source = inspect.getsource(TaskCompletionOracle.evaluate_completion)
    assert "contract" not in source
    assert "hint" not in source
    assert "Forecast" not in source


# --- A10: joint scenarios across the real fixtures ---

def _joint(*, status: str, error_code: str | None, hint: bool) -> tuple[object, object]:
    from forecontract_fake import (
        FakeEnvironmentAdapter,
        FakeScenario,
        FakeStage,
        FakeStep,
    )
    from forecontract_scripted import PlannerScript, PlannerStep, ScriptedPlanner

    produced_action = ActionSpec(schema_version="0.1", action_type="click")
    error = None if error_code is None else ErrorInfo(schema_version="0.1", code=error_code,
                                                      message="scripted")
    outcome = ActualOutcome(schema_version="0.1", outcome_id="out-0", status=status,
                            changes={"orders": {"count": 2}}, error=error)
    truth = {"orders": {"count": 2}}
    scenario = FakeScenario(
        schema_version="0.1", scenario_id="joint", adapter_name="joint-fake",
        allowed_actions=[produced_action],
        stages=[
            FakeStage(schema_version="0.1", actual_state={"orders": {"count": 0}},
                      observation=observation({"orders": {"count": 0}})),
            FakeStage(schema_version="0.1", actual_state=truth,
                      observation=observation(truth, identifier="obs-after")),
        ],
        steps=[FakeStep(schema_version="0.1", step_id="joint-step", action=produced_action,
                        outcome=outcome)])
    script = PlannerScript(schema_version="0.1", script_id="joint-plan", steps=[
        PlannerStep(schema_version="0.1", step_id="ps-0", expected_environment_type="fake",
                    expected_state=truth,
                    forecast=Forecast(schema_version="0.1", forecast_id="f-joint",
                                      proposed_action=produced_action, expected_outcomes=[],
                                      task_completion_hint=hint))])
    return FakeEnvironmentAdapter(scenario), ScriptedPlanner(script)


def test_an_unknown_result_is_an_effect_the_oracle_can_still_verify() -> None:
    adapter, planner = _joint(status="unknown", error_code="RESPONSE_LOST", hint=True)
    produced_action = ActionSpec(schema_version="0.1", action_type="click")

    async def run() -> tuple[object, object, object]:
        await adapter.start()
        receipt = await adapter.execute(produced_action)
        after = await adapter.observe()
        planned = await planner.plan(after)
        await adapter.stop()
        return receipt, after, planned

    receipt, after, planned = asyncio.run(run())
    assert receipt.status is ExecutionStatus.UNKNOWN
    assert planned.task_completion_hint is True
    piece = VerificationEvidence(schema_version="0.1", observation=after, state_complete=True)
    # The hint is ignored: the independently supplied criteria decide.
    satisfied = criteria([condition("value_equals", ["orders", "count"], expected=2)])
    assert evaluate(satisfied, piece).status is VerificationStatus.PASS
    unmet = criteria([condition("path_exists", ["goal_achieved"])])
    assert evaluate(unmet, piece).status is VerificationStatus.FAIL


def test_a_succeeded_execution_against_a_wrong_expectation_fails_the_step() -> None:
    adapter, _ = _joint(status="succeeded", error_code=None, hint=False)
    produced_action = ActionSpec(schema_version="0.1", action_type="click")

    async def run() -> object:
        await adapter.start()
        await adapter.execute(produced_action)
        return await adapter.observe()

    after = asyncio.run(run())
    contract = _step_contract(expected=98)
    step = StepVerifier().verify_step(
        contract=contract,
        evidence=VerificationEvidence(schema_version="0.1", observation=after,
                                      state_complete=True))
    # The contract's expectation was fixed before the execution; the real state disagrees.
    assert step.status is VerificationStatus.FAIL
    assert step.outcome_results[0].reason == "VALUE_MISMATCH"


def test_the_production_fixtures_still_declare_no_json_rule_capability() -> None:
    adapter, _ = _joint(status="succeeded", error_code=None, hint=False)
    declared = adapter.capabilities()
    assert declared.supported_outcome_types == []
    assert declared.supported_condition_types == []
    assert _DeclaringFakeAdapter(adapter).capabilities().supported_outcome_types == ["json_rule"]


_ = (PathExistsRule, ValueEqualsRule)
