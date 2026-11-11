"""Acceptance tests for the deterministic ScriptedPlanner.

Every fixture is synthetic and embedded here: the tests build real :class:`PlannerScript`
records, drive the real async ``plan`` method and assert values, identities, order and
exhaustion rather than the mere absence of exceptions.

Coverage follows the approved A08/A09/A10/A11 rows for the planner side: complete
environment/state matching, non-consuming rejections, legitimately producible negative
candidates, input ownership and bounded facility errors.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
SCRIPTED_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "scripted" / "src"
FAKE_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "fake" / "src"
for candidate in (CORE_SRC, SCRIPTED_SRC, FAKE_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts import (
    ContractBuilder,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.contracts.errors import ContractBuildError
from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome, ErrorInfo, ExecutionStatus
from forecontract.protocols.adapter import UnsupportedActionError
from forecontract_scripted import (
    PlannerInputError,
    PlannerObservationMismatchError,
    PlannerScript,
    PlannerScriptError,
    PlannerScriptExhaustedError,
    PlannerStep,
    ScriptedPlanner,
)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
MARKER = "SYNTHETIC_PRIVATE_VALUE"


def action(action_type: str = "click", payload: dict | None = None) -> ActionSpec:
    return ActionSpec(schema_version="0.1", action_type=action_type,
                      payload=payload if payload is not None else {"selector": "#go"})


def forecast(forecast_id: str = "f-1", *, outcomes: list[ExpectedOutcome] | None = None,
             candidate: ActionSpec | None = None, hint: bool | None = None) -> Forecast:
    return Forecast(schema_version="0.1", forecast_id=forecast_id,
                    proposed_action=candidate if candidate is not None else action(),
                    expected_outcomes=outcomes if outcomes is not None else [],
                    task_completion_hint=hint)


def observation(index: int = 0, *, state: dict | None = None, environment_type: str = "fake",
                observation_id: str | None = None, summary: str | None = None) -> Observation:
    return Observation(schema_version="0.1", observation_id=observation_id or f"obs-{index}",
                       environment_type=environment_type,
                       state=state if state is not None else {"count": index},
                       timestamp=NOW, summary=summary)


def plan_step(index: int = 0, *, state: dict | None = None, environment_type: str = "fake",
              produced: Forecast | None = None, step_id: str | None = None) -> PlannerStep:
    return PlannerStep(schema_version="0.1", step_id=step_id or f"ps-{index}",
                       expected_environment_type=environment_type,
                       expected_state=state if state is not None else {"count": index},
                       forecast=produced if produced is not None else forecast(f"f-{index + 1}"))


def script(*steps: PlannerStep, script_id: str = "p-1") -> PlannerScript:
    return PlannerScript(schema_version="0.1", script_id=script_id,
                         steps=list(steps) if steps else [])


def drive(coroutine):
    return asyncio.run(coroutine)


# --- A08: determinism, matching and negative candidates ------------------------

def test_two_planners_replay_the_same_script_identically() -> None:
    built = script(plan_step(0), plan_step(1))
    first, second = ScriptedPlanner(built), ScriptedPlanner(built)
    assert drive(first.plan(observation(0))) == drive(second.plan(observation(0)))
    assert drive(first.plan(observation(1))) == drive(second.plan(observation(1)))
    assert drive(ScriptedPlanner(built).plan(observation(0))).forecast_id == "f-1"


def test_an_empty_script_is_immediately_exhausted() -> None:
    planner = ScriptedPlanner(script())
    with pytest.raises(PlannerScriptExhaustedError):
        drive(planner.plan(observation(0)))


def test_the_planner_exposes_only_the_declared_surface() -> None:
    planner = ScriptedPlanner(script())
    assert planner.script_id == "p-1"
    for absent in ("start", "stop", "reset", "name", "capabilities"):
        assert not hasattr(planner, absent)


def test_matching_uses_the_complete_state_and_the_environment_type() -> None:
    built = script(plan_step(0, state={"a": 1, "b": [1, 2]}))
    planner = ScriptedPlanner(built)
    assert drive(planner.plan(observation(0, state={"b": [1, 2], "a": 1}))).forecast_id == "f-1"


def test_observation_metadata_and_identity_are_ignored_for_matching() -> None:
    planner = ScriptedPlanner(script(plan_step(0)))
    assert drive(planner.plan(observation(0, observation_id="other", summary="different"))).forecast_id == "f-1"


@pytest.mark.parametrize("candidate_state,matches", [
    ({"count": 0}, True),
    ({"count": True}, False),
    ({"count": 0.0}, False),
    ({"count": 1}, False),
    ({}, False),
    ({"count": 0, "extra": 1}, False),
    ({"count": None}, False),
])
def test_state_matching_is_exact_and_type_aware(candidate_state: dict, matches: bool) -> None:
    planner = ScriptedPlanner(script(plan_step(0, state={"count": 0})))
    if matches:
        assert drive(planner.plan(observation(0, state=candidate_state))).forecast_id == "f-1"
    else:
        with pytest.raises(PlannerObservationMismatchError):
            drive(planner.plan(observation(0, state=candidate_state)))


def test_a_different_environment_type_does_not_match() -> None:
    planner = ScriptedPlanner(script(plan_step(0, environment_type="fake")))
    with pytest.raises(PlannerObservationMismatchError):
        drive(planner.plan(observation(0, environment_type="browser")))


def test_nested_array_order_matters_and_object_order_does_not() -> None:
    planner = ScriptedPlanner(script(plan_step(0, state={"items": [1, 2], "nested": {"a": 1, "b": 2}})))
    accepted = observation(0, state={"nested": {"b": 2, "a": 1}, "items": [1, 2]})
    assert drive(planner.plan(accepted)).forecast_id == "f-1"
    other = ScriptedPlanner(script(plan_step(0, state={"items": [1, 2]})))
    with pytest.raises(PlannerObservationMismatchError):
        drive(other.plan(observation(0, state={"items": [2, 1]})))


def test_a_mismatch_does_not_consume_a_step() -> None:
    planner = ScriptedPlanner(script(plan_step(0), plan_step(1)))
    with pytest.raises(PlannerObservationMismatchError):
        drive(planner.plan(observation(0, state={"count": 99})))
    assert drive(planner.plan(observation(0))).forecast_id == "f-1"
    assert drive(planner.plan(observation(1))).forecast_id == "f-2"


def test_each_successful_call_consumes_exactly_one_step() -> None:
    planner = ScriptedPlanner(script(plan_step(0), plan_step(1)))
    assert drive(planner.plan(observation(0))).forecast_id == "f-1"
    with pytest.raises(PlannerObservationMismatchError):
        drive(planner.plan(observation(0)))
    assert drive(planner.plan(observation(1))).forecast_id == "f-2"
    with pytest.raises(PlannerScriptExhaustedError):
        drive(planner.plan(observation(1)))


def test_legitimate_negative_candidates_can_be_produced() -> None:
    outcomes = [ExpectedOutcome(schema_version="0.1", outcome_type="dom_changed")]
    built = script(
        plan_step(0, produced=forecast("f-empty", outcomes=[])),
        plan_step(1, produced=forecast("f-unsupported", candidate=action("teleport"))),
        plan_step(2, produced=forecast("f-wrong", outcomes=outcomes)),
        plan_step(3, produced=forecast("f-false-completion", outcomes=[], hint=True)),
    )
    planner = ScriptedPlanner(built)
    empty = drive(planner.plan(observation(0)))
    assert empty.expected_outcomes == [] and empty.task_completion_hint is None
    unsupported = drive(planner.plan(observation(1)))
    assert unsupported.proposed_action.action_type == "teleport"
    wrong = drive(planner.plan(observation(2)))
    assert [item.outcome_type for item in wrong.expected_outcomes] == ["dom_changed"]
    false_completion = drive(planner.plan(observation(3)))
    assert false_completion.task_completion_hint is True


def test_the_planner_makes_no_commitment_decision() -> None:
    """A returned forecast is data: the planner never validates it against a Builder."""
    planner = ScriptedPlanner(script(plan_step(0, produced=forecast("f-1", candidate=action("nope")))))
    returned = drive(planner.plan(observation(0)))
    assert returned.proposed_action.action_type == "nope"
    assert not hasattr(planner, "build")


# --- A09/A10: ownership and bounded errors -------------------------------------

def test_an_invalid_observation_is_rejected_before_exhaustion_is_checked() -> None:
    planner = ScriptedPlanner(script())
    with pytest.raises(PlannerInputError) as info:
        drive(planner.plan({"state": MARKER}))
    assert MARKER not in str(info.value)
    assert info.value.code == "planner_input"


def test_a_copied_script_with_an_undeclared_key_is_rejected() -> None:
    forged = script(plan_step(0)).model_copy(update={MARKER: MARKER})
    with pytest.raises(PlannerScriptError) as info:
        ScriptedPlanner(forged)
    assert MARKER not in str(info.value)


def test_a_copied_script_with_an_invalid_stored_field_is_rejected() -> None:
    forged = script(plan_step(0)).model_copy(update={"schema_version": "9.9"})
    with pytest.raises(PlannerScriptError):
        ScriptedPlanner(forged)


def test_a_copied_step_with_an_unknown_key_is_rejected() -> None:
    original = script(plan_step(0))
    forged_step = original.steps[0].model_copy(update={MARKER: MARKER})
    forged = PlannerScript.model_construct(
        schema_version="0.1", script_id="p-1", steps=[forged_step]
    )
    assert MARKER in forged.steps[0].__dict__
    with pytest.raises(PlannerScriptError):
        ScriptedPlanner(forged)


def test_a_nested_illegal_state_mutation_is_rejected() -> None:
    """Either the in-place edit itself or the later revalidation must reject the content."""
    built = script(plan_step(0))
    with pytest.raises(Exception) as info:
        built.steps[0].expected_state[1] = "non-string-key"
        ScriptedPlanner(built)
    assert info.value is not None


def test_mutating_the_script_afterwards_does_not_change_the_planner() -> None:
    built = script(plan_step(0))
    planner = ScriptedPlanner(built)
    built.steps[0].expected_state["count"] = 99
    assert drive(planner.plan(observation(0))).forecast_id == "f-1"


def test_mutating_a_returned_forecast_does_not_change_the_script() -> None:
    built = script(plan_step(0, produced=forecast("f-1", candidate=action("click", {"a": 1}))))
    planner = ScriptedPlanner(built)
    returned = drive(planner.plan(observation(0)))
    returned.proposed_action.payload["a"] = 99
    other = ScriptedPlanner(built)
    assert drive(other.plan(observation(0))).proposed_action.payload == {"a": 1}


def test_a_snapshot_of_the_script_is_a_deep_copy() -> None:
    built = script(plan_step(0))
    duplicate = copy.deepcopy(built)
    duplicate.steps[0].expected_state["count"] = 42
    assert built.steps[0].expected_state["count"] == 0


def test_the_script_requires_unique_step_and_forecast_identifiers() -> None:
    with pytest.raises(PlannerScriptError):
        script(plan_step(0, step_id="same"), plan_step(1, step_id="same"))
    with pytest.raises(PlannerScriptError):
        script(plan_step(0, produced=forecast("f-same")), plan_step(1, produced=forecast("f-same")))


@pytest.mark.parametrize("field,value", [
    ("schema_version", "0.2"),
    ("script_id", 1),
    ("script_id", ""),
    ("steps", [{"step_id": "x"}]),
])
def test_a_malformed_script_field_is_rejected(field: str, value: object) -> None:
    fields = {"schema_version": "0.1", "script_id": "p-1", "steps": [plan_step(0)]}
    fields[field] = value
    with pytest.raises(PlannerScriptError):
        PlannerScript(**fields)


def test_a_malformed_planner_step_field_is_rejected() -> None:
    with pytest.raises(PlannerScriptError):
        PlannerStep(schema_version="0.1", step_id="ps-0", expected_environment_type="",
                    expected_state={}, forecast=forecast())
    with pytest.raises(PlannerScriptError):
        PlannerStep(schema_version="0.1", step_id="ps-0", expected_environment_type="fake",
                    expected_state={"bad": (1, 2)}, forecast=forecast())


def test_the_public_json_ingress_is_bounded_and_value_free() -> None:
    fields = script(plan_step(0)).model_dump(mode="json")
    fields[MARKER] = MARKER
    with pytest.raises(PlannerScriptError) as info:
        PlannerScript.model_validate_json(json.dumps(fields))
    assert MARKER not in str(info.value)
    with pytest.raises(PlannerScriptError):
        PlannerScript.model_validate_json("{" + MARKER)


def test_planner_errors_expose_a_stable_code_and_path() -> None:
    planner = ScriptedPlanner(script(plan_step(0)))
    drive(planner.plan(observation(0)))
    try:
        drive(planner.plan(observation(0)))
    except PlannerScriptExhaustedError as exc:
        assert exc.code == "planner_script_exhausted"
        assert exc.path == ("steps",)
    else:  # pragma: no cover - the second call must be exhausted
        raise AssertionError("the exhausted script must reject")
    mismatch = ScriptedPlanner(script(plan_step(0)))
    try:
        drive(mismatch.plan(observation(1)))
    except PlannerObservationMismatchError as exc:
        assert exc.code == "planner_observation_mismatch"
        assert exc.path == ("steps", 0)
    else:  # pragma: no cover - the observation must not match
        raise AssertionError("a mismatching observation must be rejected")


def test_an_error_never_echoes_a_state_value_or_a_raw_cause() -> None:
    planner = ScriptedPlanner(script(plan_step(0, state={"secret": MARKER})))
    with pytest.raises(PlannerObservationMismatchError) as info:
        drive(planner.plan(observation(0, state={"secret": MARKER + "-other"})))
    assert MARKER not in str(info.value)


def test_a_planner_input_error_keeps_the_cause_in_process_only() -> None:
    """An in-place nested contamination is caught at the call boundary, not laundered."""
    contaminated = observation(0)
    contaminated.state["bad"] = (1, 2)
    planner = ScriptedPlanner(script(plan_step(0)))
    with pytest.raises(PlannerInputError) as info:
        drive(planner.plan(contaminated))
    assert info.value.__cause__ is not None
    assert MARKER not in str(info.value)


def test_the_planner_never_imports_the_fake_package() -> None:
    """The two fixture distributions stay independent: neither source imports the other."""
    import ast

    repository = Path(__file__).resolve().parents[2]
    packages = {
        "forecontract_fake": repository / "packages" / "adapters" / "fake" / "src",
        "forecontract_scripted": repository / "packages" / "planners" / "scripted" / "src",
    }
    imported: dict[str, set[str]] = {name: set() for name in packages}
    for name, root in packages.items():
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported[name].update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported[name].add(node.module)
    assert not [item for item in imported["forecontract_scripted"] if item.startswith("forecontract_fake")]
    assert not [item for item in imported["forecontract_fake"] if item.startswith("forecontract_scripted")]
    assert any(item.startswith("forecontract") for item in imported["forecontract_fake"])
    assert any(item.startswith("forecontract") for item in imported["forecontract_scripted"])


# --- round-002 correction: F03 blanks, F04 depth, F05 surface, joint scenarios ---

@pytest.mark.parametrize("blank", [" ", "\t", "\n", " \t\n ", "\r\n"])
def test_a_whitespace_only_script_identifier_is_rejected(blank: str) -> None:
    """F03: a blank script identifier is refused and never normalized into a valid one."""
    with pytest.raises(PlannerScriptError) as info:
        PlannerScript(schema_version="0.1", script_id=blank, steps=[])
    assert "script_id" in str(info.value)


@pytest.mark.parametrize("blank", [" ", "\t", "\n", " \t\n "])
@pytest.mark.parametrize("field", ["step_id", "expected_environment_type"])
def test_a_whitespace_only_step_identifier_is_rejected(field: str, blank: str) -> None:
    """F03: the step identifier and the expected environment type share the same strictness."""
    fields = {"schema_version": "0.1", "step_id": "ps-0", "expected_environment_type": "fake",
              "expected_state": {}, "forecast": forecast("f-1")}
    fields[field] = blank
    with pytest.raises(PlannerScriptError) as info:
        PlannerStep(**fields)
    assert field in str(info.value)


def test_an_identifier_containing_whitespace_is_preserved_verbatim() -> None:
    """F03 control: only an all-whitespace value is refused; the value is never rewritten."""
    built = script(plan_step(0, produced=forecast("f-1")), script_id="p 1\tx")
    assert ScriptedPlanner(built).script_id == "p 1\tx"
    step = plan_step(0, environment_type="fake browser", produced=forecast("f-1"))
    assert step.expected_environment_type == "fake browser"


def _forged_script() -> PlannerScript:
    built = script(plan_step(0))
    built.steps[0].forecast.__dict__[MARKER] = MARKER
    return built


def test_an_unknown_key_on_a_deeply_nested_forecast_is_rejected() -> None:
    """F04: a Core record nested inside a step cannot hide a stored key."""
    with pytest.raises(PlannerScriptError) as info:
        ScriptedPlanner(_forged_script())
    assert MARKER not in str(info.value)


def test_an_unknown_key_on_a_nested_expected_outcome_is_rejected() -> None:
    """F04: a record inside a forecast's list field is inspected as well."""
    produced = forecast("f-1", outcomes=[ExpectedOutcome(schema_version="0.1",
                                                         outcome_type="dom_changed")])
    produced.expected_outcomes[0].__dict__[MARKER] = MARKER
    with pytest.raises(PlannerScriptError) as info:
        script(plan_step(0, produced=produced))
    assert MARKER not in str(info.value)


def test_an_unknown_key_on_a_nested_proposed_action_is_rejected() -> None:
    """F04: the proposed action inside a nested forecast is inspected."""
    produced = forecast("f-1")
    produced.proposed_action.__dict__[MARKER] = MARKER
    with pytest.raises(PlannerScriptError):
        ScriptedPlanner(script(plan_step(0, produced=produced)))


def test_an_invalid_nested_core_field_is_a_facility_error() -> None:
    """F04: a nested schema violation becomes the boundary's own bounded error."""
    produced = forecast("f-1").model_copy(update={"proposed_action": action().model_copy(
        update={"schema_version": "9.9"})})
    with pytest.raises(PlannerScriptError) as info:
        ScriptedPlanner(script(plan_step(0, produced=produced)))
    assert MARKER not in str(info.value)


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"script_id": 17},
    {"steps": "not-a-list"},
])
def test_public_instance_ingress_rejects_a_copied_invalid_script(mutation: dict) -> None:
    """F04: model_validate and the constructor both revalidate an existing instance."""
    forged = script(plan_step(0)).model_copy(update=mutation)
    with pytest.raises(PlannerScriptError):
        PlannerScript.model_validate(forged)
    with pytest.raises(PlannerScriptError):
        ScriptedPlanner(forged)


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"observation_id": 17},
    {"state": "not-an-object"},
])
def test_public_instance_ingress_rejects_a_copied_invalid_observation(mutation: dict) -> None:
    """F04: the planning call revalidates a supplied existing observation."""
    planner = ScriptedPlanner(script(plan_step(0)))
    with pytest.raises(PlannerInputError):
        drive(planner.plan(observation(0).model_copy(update=mutation)))


def test_public_instance_ingress_accepts_valid_instances() -> None:
    """F04 control: ordinary valid instances still pass both ingresses."""
    built = script(plan_step(0), plan_step(1))
    assert PlannerScript.model_validate(built) == built
    planner = ScriptedPlanner(built)
    assert drive(planner.plan(observation(0))).forecast_id == "f-1"
    assert drive(planner.plan(observation(1))).forecast_id == "f-2"


def test_the_python_and_json_roundtrip_still_works() -> None:
    """F04 control: no coercion was added; legal Python and JSON both load as before."""
    built = script(plan_step(0))
    fields = built.model_dump(mode="json")
    assert PlannerScript.model_validate(fields).model_dump(mode="json") == fields
    assert PlannerScript.model_validate_json(built.model_dump_json()) == built


def test_the_public_surface_is_closed_to_fixture_constants() -> None:
    """F05: the package exports components, models and error classes only."""
    import forecontract_scripted

    assert sorted(forecontract_scripted.__all__) == [
        "PlannerError",
        "PlannerInputError",
        "PlannerObservationMismatchError",
        "PlannerScript",
        "PlannerScriptError",
        "PlannerScriptExhaustedError",
        "PlannerStep",
        "ScriptedPlanner",
    ]
    assert [name for name in forecontract_scripted.__all__ if name.isupper()] == []
    from forecontract_scripted.errors import PLANNER_ERROR_CODES

    assert len(PLANNER_ERROR_CODES) == 4
    assert not hasattr(forecontract_scripted, "PLANNER_ERROR_CODES")


# --- A11 joint scenarios with the Fake fixture and a test-local Builder wrapper ---

class DeclaringFakeAdapter:
    """A test-local wrapper that adds one declared capability to the Fake fixture.

    The Fake adapter deliberately declares no production domain-verification capability, so a
    Builder round in these tests uses this explicit local wrapper. It is not a production hook,
    verifier or Runtime; it only widens the declaration for one test.
    """

    def __init__(self, adapter, outcome_types: tuple[str, ...] = ("dom_changed",)) -> None:
        self._adapter = adapter
        self._outcome_types = outcome_types

    @property
    def name(self) -> str:
        return self._adapter.name

    def capabilities(self) -> AdapterCapabilities:
        declared = self._adapter.capabilities()
        return declared.model_copy(update={"supported_outcome_types": list(self._outcome_types)})

    def validate_action(self, action: ActionSpec) -> None:
        self._adapter.validate_action(action)


class LocalExpectedOutcomeHooks:
    """A test-local domain hook set for one Builder round.

    The Fake adapter declares no production domain-verification capability, so the tests supply
    this explicit local hook instead of a production verifier. It records what it was asked to
    validate and accepts only the one outcome type the local wrapper declares.
    """

    def __init__(self, accepted: str = "dom_changed") -> None:
        self.accepted = accepted
        self.calls: list[tuple[str, str]] = []

    def validate_expected_outcome(self, outcome) -> None:
        self.calls.append(("validate_expected_outcome", outcome.outcome_type))
        if outcome.outcome_type != self.accepted:
            raise UnsupportedActionError("the local hook does not understand this outcome type")

    def validate_precondition(self, condition) -> None:
        self.calls.append(("validate_precondition", condition.condition_type))

    def validate_verification_binding(self, binding, outcome) -> None:
        self.calls.append(("validate_verification_binding", binding.verifier_type))
        if binding.outcome_index < 0:
            raise UnsupportedActionError("the binding index is outside the candidate")


def _single_binding_policy() -> VerificationPolicy:
    return VerificationPolicy(
        schema_version="0.1",
        bindings=[VerificationBinding(schema_version="0.1", outcome_index=0,
                                      verifier_type="dom_verifier")],
    )


def _joint_scenario(*, status: str, code: str | None, hint: bool) -> tuple[object, object]:
    """One Fake script whose actual outcome feeds the ScriptedPlanner as a real observation."""
    from forecontract_fake import (
        FakeEnvironmentAdapter,
        FakeScenario,
        FakeStage,
        FakeStep,
    )

    error = None if code is None else ErrorInfo(schema_version="0.1", code=code, message="scripted")
    produced = ActualOutcome(schema_version="0.1", outcome_id="out-0", status=status,
                             changes={"orders": 1}, error=error)
    truth = {"orders": 1}
    fake = FakeScenario(
        schema_version="0.1", scenario_id="joint", adapter_name="joint-fake",
        allowed_actions=[action()],
        stages=[
            FakeStage(schema_version="0.1", actual_state={"orders": 0},
                      observation=observation(0, state={"orders": 0})),
            FakeStage(schema_version="0.1", actual_state=truth,
                      observation=observation(1, state=truth)),
        ],
        steps=[FakeStep(schema_version="0.1", step_id="joint-step", action=action(),
                        outcome=produced)],
    )
    produced_forecast = forecast("f-joint", outcomes=[], hint=hint)
    planned = script(plan_step(0, state=truth, produced=produced_forecast), script_id="joint-plan")
    return FakeEnvironmentAdapter(fake), ScriptedPlanner(planned)


def test_a_joint_scenario_separates_execution_fact_from_plan_claim() -> None:
    """A11: the Fake reports what happened; the Planner separately claims completion."""
    adapter, planner = _joint_scenario(status="failed", code="EXECUTION_FAILED", hint=True)

    async def run() -> tuple[object, object]:
        await adapter.start()
        receipt = await adapter.execute(action())
        current = await adapter.observe()
        planned = await planner.plan(current)
        await adapter.stop()
        return receipt, planned

    receipt, planned = drive(run())
    assert receipt.status is ExecutionStatus.FAILED
    assert receipt.changes == {"orders": 1}
    assert planned.task_completion_hint is True
    assert planned.expected_outcomes == []
    # The claim and the fact are different objects with different content; neither is derived
    # from the other, and nothing copied the hidden actual state into either one.
    assert "actual_state" not in receipt.model_dump()


def test_a_joint_scenario_observes_an_effect_after_an_unknown_result() -> None:
    """A11: an unknown result still consumes the step, and the next observation shows truth."""
    adapter, planner = _joint_scenario(status="unknown", code="RESPONSE_LOST", hint=False)

    async def run() -> tuple[object, object, object]:
        await adapter.start()
        receipt = await adapter.execute(action())
        observed = await adapter.observe()
        planned = await planner.plan(observed)
        return receipt, observed, planned

    receipt, observed, planned = drive(run())
    assert receipt.status is ExecutionStatus.UNKNOWN
    assert receipt.error is not None and receipt.error.code == "RESPONSE_LOST"
    assert observed.state == {"orders": 1}
    assert planned.task_completion_hint is False


def test_a_joint_plan_claiming_success_while_execution_failed_is_still_producible() -> None:
    """A11: a false completion hint is a legitimate candidate, not an error to raise."""
    adapter, planner = _joint_scenario(status="failed", code="ENVIRONMENT_REJECTED", hint=True)

    async def run() -> object:
        await adapter.start()
        await adapter.execute(action())
        current = await adapter.observe()
        return await planner.plan(current)

    planned = drive(run())
    assert planned.task_completion_hint is True


def test_a_test_local_builder_wrapper_commits_the_scripted_candidate() -> None:
    """A11/A12: a real Builder round accepts the forecast once the capability is declared."""
    adapter, _ = _joint_scenario(status="succeeded", code=None, hint=True)

    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="dom_changed", required=True)
    candidate = Forecast(schema_version="0.1", forecast_id="f-build", proposed_action=action(),
                         expected_outcomes=[outcome], task_completion_hint=True)
    built = ContractBuilder().build(
        forecast=candidate,
        observation=observation(0, state={"orders": 0}),
        adapter=DeclaringFakeAdapter(adapter),
        hooks=LocalExpectedOutcomeHooks(),
        preconditions=[],
        verification_policy=_single_binding_policy(),
        timeout_ms=100,
        remaining_budget_ms=250,
        contract_id="c-joint",
        goal_id="g-joint",
        created_at=NOW,
    )
    assert built.contract_id == "c-joint"
    assert built.forecast_id == "f-build"
    assert built.action.action_type == "click"
    assert [item.outcome_type for item in built.expected_outcomes] == ["dom_changed"]
    assert built.adapter_name == adapter.name


def test_the_local_wrapper_is_what_makes_the_outcome_declared() -> None:
    """The wrapper is meaningful: the unwrapped Fake capability set still refuses the outcome."""
    adapter, _ = _joint_scenario(status="succeeded", code=None, hint=False)
    assert adapter.capabilities().supported_outcome_types == []
    assert DeclaringFakeAdapter(adapter).capabilities().supported_outcome_types == ["dom_changed"]

    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="dom_changed", required=True)
    candidate = Forecast(schema_version="0.1", forecast_id="f-build", proposed_action=action(),
                         expected_outcomes=[outcome])
    with pytest.raises(ContractBuildError) as info:
        ContractBuilder().build(
            forecast=candidate,
            observation=observation(0, state={"orders": 0}),
            adapter=adapter,
            hooks=LocalExpectedOutcomeHooks(),
            preconditions=[],
            verification_policy=_single_binding_policy(),
            timeout_ms=100,
            remaining_budget_ms=250,
            contract_id="c-joint",
            goal_id="g-joint",
            created_at=NOW,
        )
    assert info.value.code == "UNSUPPORTED_CAPABILITY"


def test_the_local_wrapper_delegates_action_validation() -> None:
    """The wrapper never weakens action validation: an unsupported action is still refused."""
    adapter, _ = _joint_scenario(status="succeeded", code=None, hint=False)
    wrapper = DeclaringFakeAdapter(adapter)
    wrapper.validate_action(action())
    with pytest.raises(UnsupportedActionError):
        wrapper.validate_action(action("teleport"))


# --- round-003 correction: raw nested structure is rejected before dependent validators ---

_DAMAGED_FORECAST_ID = ["SYNTHETIC_LIST_ID"]


def _script_mapping(*, step_id: object = "ps-0", produced: object = None) -> dict:
    """A raw Python mapping for one script, with the given step and forecast content."""
    candidate = produced if produced is not None else forecast("f-1")
    return {
        "schema_version": "0.1", "script_id": "p-1",
        "steps": [{"schema_version": "0.1", "step_id": step_id,
                   "expected_environment_type": "fake", "expected_state": {},
                   "forecast": candidate}],
    }


def _damaged_forecast() -> Forecast:
    """A real Core instance whose stored forecast identifier is not a string."""
    return forecast("f-1").model_copy(update={"forecast_id": _DAMAGED_FORECAST_ID})


@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_sequence_valued_forecast_identifier_is_a_bounded_facility_error(ingress: str) -> None:
    """F04: the uniqueness read never runs before the nested forecast is revalidated."""
    payload = _script_mapping(produced=_damaged_forecast())
    with pytest.raises(PlannerScriptError) as info:
        if ingress == "constructor":
            PlannerScript(**payload)
        else:
            PlannerScript.model_validate(payload)
    message = str(info.value)
    assert "SYNTHETIC" not in message
    assert info.value.code == "planner_script"
    assert len(message) < 240


@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_sequence_valued_step_identifier_is_a_bounded_rejection(ingress: str) -> None:
    """F04: a wrongly typed step identifier is refused as a facility error either way."""
    payload = _script_mapping(step_id=["a"])
    with pytest.raises(PlannerScriptError):
        if ingress == "constructor":
            PlannerScript(**payload)
        else:
            PlannerScript.model_validate(payload)


def test_no_public_ingress_raises_a_raw_python_error_for_damaged_nested_content() -> None:
    """F04: every planner ingress reports the boundary error, never AttributeError/TypeError."""
    attempts = [
        lambda: PlannerScript(**_script_mapping(produced=_damaged_forecast())),
        lambda: PlannerScript.model_validate(_script_mapping(produced=_damaged_forecast())),
        lambda: ScriptedPlanner(script(plan_step(0)).model_copy(update={
            "steps": [plan_step(0).model_copy(update={"forecast": _damaged_forecast()})]})),
    ]
    for attempt in attempts:
        with pytest.raises(PlannerScriptError):
            attempt()


def test_legal_nested_content_still_constructs_and_round_trips() -> None:
    """F04 control: the pre-check refuses only damaged structure, never legal content."""
    built = script(plan_step(0))
    assert PlannerScript.model_validate(built) == built
    assert PlannerScript.model_validate_json(built.model_dump_json()) == built
    assert PlannerScript(**built.model_dump(mode="python")) == built
    planner = ScriptedPlanner(PlannerScript(**_script_mapping(produced=forecast("f-1"))))
    assert planner.script_id == "p-1"


# --- round-004 correction: declared container shape first, bounded recursion ---

def _cyclic_list() -> list:
    container: list = []
    container.append(container)
    return container


def _mapping_with_cyclic_steps(container: object) -> dict:
    data = script(plan_step(0)).model_dump(mode="python")
    data["steps"] = container
    return data


@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_cyclic_steps_container_is_a_bounded_rejection(ingress: str) -> None:
    """F04: a cyclic declared list terminates as a facility error, never a RecursionError."""
    payload = _mapping_with_cyclic_steps(_cyclic_list())
    with pytest.raises(PlannerScriptError) as info:
        if ingress == "constructor":
            PlannerScript(**payload)
        else:
            PlannerScript.model_validate(payload)
    assert info.value.code == "planner_script"
    assert len(str(info.value)) < 240


@pytest.mark.parametrize("container", [_cyclic_list(), 7, "steps", {"a": 1}, ()])
def test_a_non_list_steps_container_is_a_bounded_rejection(container: object) -> None:
    """F04: the declared strict list field refuses any shape that is not a list."""
    with pytest.raises(PlannerScriptError):
        PlannerScript(**_mapping_with_cyclic_steps(container))


def test_a_deeply_nested_non_record_element_terminates_as_a_facility_error() -> None:
    """F04: recursion over list elements is bounded by the declared element shape."""
    deep: list = []
    cursor = deep
    for _ in range(60):
        nxt: list = []
        cursor.append(nxt)
        cursor = nxt
    with pytest.raises(PlannerScriptError):
        PlannerScript(**_mapping_with_cyclic_steps(deep))


def test_a_legal_shared_reference_and_nested_state_still_construct() -> None:
    """F04 control: a shared nested reference and a legal nested state stay usable."""
    shared_action = action()
    shared = PlannerScript(schema_version="0.1", script_id="p-1", steps=[
        PlannerStep(schema_version="0.1", step_id="ps-0", expected_environment_type="fake",
                    expected_state={"count": 0},
                    forecast=Forecast(schema_version="0.1", forecast_id="f-0",
                                      proposed_action=shared_action, expected_outcomes=[])),
        PlannerStep(schema_version="0.1", step_id="ps-1", expected_environment_type="fake",
                    expected_state={"count": 1},
                    forecast=Forecast(schema_version="0.1", forecast_id="f-1",
                                      proposed_action=shared_action, expected_outcomes=[]))])
    assert shared.steps[0].forecast.proposed_action is shared.steps[1].forecast.proposed_action
    assert PlannerScript.model_validate(shared.model_dump(mode="python")).steps[1].step_id == "ps-1"
    nested = {"a": [1, {"b": [2, [3, {"c": None}]]}], "d": {"e": [4]}}
    built = PlannerScript(schema_version="0.1", script_id="p-nested", steps=[
        PlannerStep(schema_version="0.1", step_id="ps-nested",
                    expected_environment_type="fake", expected_state=nested,
                    forecast=forecast("f-nested"))])
    planner = ScriptedPlanner(built)
    assert drive(planner.plan(observation(0, state=nested))).forecast_id == "f-nested"


# --- round-005 correction: declared element type before any nested rebuild ---

def _self_holding_script() -> PlannerScript:
    """A valid script that holds itself as one of its own steps."""
    built = script(plan_step(0))
    built.steps.append(built)
    return built


def _cause_depth(exc: BaseException) -> int:
    """The length of the cause chain, so a wrapped recursion can be told from a direct reject."""
    depth, cause = 0, exc.__cause__
    while cause is not None and depth < 500:
        depth += 1
        cause = cause.__cause__
    return depth


@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_wrong_class_cyclic_step_element_is_refused_directly(ingress: str) -> None:
    """F04: the declared element class is checked before anything is rebuilt, so no recursion."""
    payload = script(plan_step(0)).model_dump(mode="python")
    payload["steps"] = [_self_holding_script()]
    with pytest.raises(PlannerScriptError) as info:
        if ingress == "constructor":
            PlannerScript(**payload)
        else:
            PlannerScript.model_validate(payload)
    message = str(info.value)
    assert "steps" in message
    assert len(message) < 240
    assert _cause_depth(info.value) == 0


def test_a_wrong_class_element_does_not_recurse_or_wrap() -> None:
    """F04: the rejection is bounded even when the wrong element holds a whole script."""
    payload = script(plan_step(0)).model_dump(mode="python")
    payload["steps"] = [script(plan_step(0))]
    with pytest.raises(PlannerScriptError) as info:
        PlannerScript(**payload)
    assert _cause_depth(info.value) == 0
    assert len(str(info.value)) < 240


def test_legal_sharing_and_acyclic_structures_still_work() -> None:
    """F04 control: a shared nested object and a shared JSON state stay usable."""
    shared_action = action()
    shared = script(plan_step(0), plan_step(1))
    shared.steps[0] = shared.steps[0].model_copy(update={
        "forecast": forecast("f-0", candidate=shared_action)})
    shared.steps[1] = shared.steps[1].model_copy(update={
        "forecast": forecast("f-1", candidate=shared_action)})
    assert shared.steps[0].forecast.proposed_action is shared.steps[1].forecast.proposed_action
    nested = {"a": [1, {"b": [2, [3, {"c": None}]]}]}
    built = PlannerScript(schema_version="0.1", script_id="p-shared", steps=[
        PlannerStep(schema_version="0.1", step_id="ps-0", expected_environment_type="fake",
                    expected_state=nested, forecast=forecast("f-0")),
        PlannerStep(schema_version="0.1", step_id="ps-1", expected_environment_type="fake",
                    expected_state=nested, forecast=forecast("f-1"))])
    assert built.steps[0].expected_state == nested
    planner = ScriptedPlanner(built)
    assert drive(planner.plan(observation(0, state=nested))).forecast_id == "f-0"


# --- round-006 correction: the parser boundary is checked from the planner side too ---

_SELF_MARKER = "SYNTHETIC_SELF_KEY"


def _planner_public_records() -> list[tuple[str, type, object]]:
    """One valid record per planner model owned by this module."""
    return [
        ("PlannerScript", PlannerScript, script(plan_step(0))),
        ("PlannerStep", PlannerStep, plan_step(0)),
    ]


@pytest.mark.parametrize("form", ["mapping", "json"])
@pytest.mark.parametrize("name,model,record", _planner_public_records(),
                         ids=[entry[0] for entry in _planner_public_records()])
def test_an_unknown_self_key_is_a_bounded_facility_error(form: str, name: str, model: type,
                                                        record: object) -> None:
    """F06: an unknown key named 'self' is an ordinary unknown key, not a raw TypeError."""
    payload = {**record.model_dump(mode="python"), "self": _SELF_MARKER}
    with pytest.raises(PlannerScriptError) as info:
        if form == "mapping":
            model.model_validate(payload)
        else:
            model.model_validate_json(json.dumps(payload, default=str))
    message = str(info.value)
    assert message.startswith("planner_script")
    assert len(message) < 240
    assert _SELF_MARKER not in message


@pytest.mark.parametrize("form", ["mapping", "json"])
@pytest.mark.parametrize("name,model,record", _planner_public_records(),
                         ids=[entry[0] for entry in _planner_public_records()])
def test_an_ordinary_unknown_key_is_still_rejected(form: str, name: str, model: type,
                                                  record: object) -> None:
    """F06 control: the pre-existing extra=forbid behaviour is unchanged for any other key."""
    payload = {**record.model_dump(mode="python"), _SELF_MARKER: 1}
    with pytest.raises(PlannerScriptError):
        if form == "mapping":
            model.model_validate(payload)
        else:
            model.model_validate_json(json.dumps(payload, default=str))


@pytest.mark.parametrize("form", ["mapping", "json"])
@pytest.mark.parametrize("name,model,record", _planner_public_records(),
                         ids=[entry[0] for entry in _planner_public_records()])
def test_a_valid_record_still_parses_through_both_public_forms(form: str, name: str, model: type,
                                                              record: object) -> None:
    """F06 control: a legal mapping and a legal JSON document still parse to an equal record."""
    if form == "mapping":
        parsed = model.model_validate(record.model_dump(mode="json"))
    else:
        parsed = model.model_validate_json(record.model_dump_json())
    assert parsed == record
    assert type(parsed) is model


def test_shared_references_and_acyclic_nesting_survive_the_parser_boundary() -> None:
    """F06 control: sharing and acyclic depth behave identically through mapping and JSON."""
    shared_action = action()
    built = PlannerScript(schema_version="0.1", script_id="p-shared", steps=[
        PlannerStep(schema_version="0.1", step_id="ps-0", expected_environment_type="fake",
                    expected_state={"count": 0},
                    forecast=Forecast(schema_version="0.1", forecast_id="f-0",
                                      proposed_action=shared_action, expected_outcomes=[])),
        PlannerStep(schema_version="0.1", step_id="ps-1", expected_environment_type="fake",
                    expected_state={"count": 1},
                    forecast=Forecast(schema_version="0.1", forecast_id="f-1",
                                      proposed_action=shared_action, expected_outcomes=[]))])
    from_mapping = PlannerScript.model_validate(built.model_dump(mode="python"))
    from_json = PlannerScript.model_validate_json(built.model_dump_json())
    assert from_mapping == built == from_json
    # Parsing from data builds an equivalent record; the shared source object was serialised
    # twice, so equality is the contract here rather than object identity.
    assert (
        from_mapping.steps[0].forecast.proposed_action
        == from_mapping.steps[1].forecast.proposed_action
    )
    nested = {"a": [1, {"b": [2, [3, {"c": None}]]}]}
    deep = PlannerScript(schema_version="0.1", script_id="p-deep", steps=[
        PlannerStep(schema_version="0.1", step_id="ps-0", expected_environment_type="fake",
                    expected_state=nested, forecast=forecast("f-0"))])
    assert PlannerScript.model_validate(deep.model_dump(mode="python")) == deep
    assert PlannerScript.model_validate_json(deep.model_dump_json()) == deep
