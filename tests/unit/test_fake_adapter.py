"""Acceptance tests for the deterministic Fake environment adapter.

Every fixture is synthetic and embedded here: the tests build real :class:`FakeScenario`
records, drive the real async adapter and assert values, identities, order and state
transitions rather than the mere absence of exceptions.

Coverage follows the approved A01-A11 rows for the Fake side: replay determinism, strict
structure validation, type-aware matching, the full lifecycle, the fixed execution order,
hidden-truth separation, input ownership and bounded facility errors.
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
FAKE_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "fake" / "src"
SCRIPTED_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "scripted" / "src"
for candidate in (CORE_SRC, FAKE_SRC, SCRIPTED_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.models.actions import ActionSpec, AdapterCapabilities, Forecast
from forecontract.models.observations import ArtifactRef, Observation
from forecontract.models.outcomes import ActualOutcome, ErrorInfo, ExecutionStatus
from forecontract.protocols.adapter import (
    InvalidActionPayloadError,
    UnsupportedActionError,
)
from forecontract_fake import (
    FakeEnvironmentAdapter,
    FakeInputError,
    FakeLifecycleError,
    FakeScenario,
    FakeScenarioError,
    FakeScriptExhaustedError,
    FakeStage,
    FakeStep,
    FakeStepMismatchError,
)
from forecontract_scripted import PlannerScriptError

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
MARKER = "SYNTHETIC_PRIVATE_VALUE"


def action(action_type: str = "click", payload: dict | None = None,
           metadata: dict | None = None) -> ActionSpec:
    return ActionSpec(schema_version="0.1", action_type=action_type,
                      payload=payload if payload is not None else {"selector": "#go"},
                      metadata=metadata or {})


def observation(index: int, *, state: dict | None = None, environment_type: str = "fake",
                observation_id: str | None = None) -> Observation:
    return Observation(schema_version="0.1", observation_id=observation_id or f"obs-{index}",
                       environment_type=environment_type,
                       state=state if state is not None else {"visible": index},
                       timestamp=NOW)


def stage(index: int, *, actual_state: dict | None = None, visible: dict | None = None) -> FakeStage:
    return FakeStage(
        schema_version="0.1",
        actual_state=actual_state if actual_state is not None else {"count": index},
        observation=observation(index, state=visible if visible is not None else {"visible": index}),
    )


def outcome(index: int = 0, *, status: str = "succeeded", changes: dict | None = None,
            state_before: dict | None = None, state_after: dict | None = None,
            error: ErrorInfo | None = None) -> ActualOutcome:
    fields: dict = {
        "schema_version": "0.1",
        "outcome_id": f"out-{index}",
        "status": status,
        "changes": changes if changes is not None else {"count": index + 1},
    }
    if state_before is not None:
        fields["state_before"] = state_before
    if state_after is not None:
        fields["state_after"] = state_after
    if error is not None:
        fields["error"] = error
    return ActualOutcome(**fields)


def step(index: int, *, candidate: ActionSpec | None = None,
         produced: ActualOutcome | None = None) -> FakeStep:
    return FakeStep(schema_version="0.1", step_id=f"step-{index}",
                    action=candidate if candidate is not None else action(),
                    outcome=produced if produced is not None else outcome(index))


def scenario(*, steps: list[FakeStep] | None = None,
             catalog: list[ActionSpec] | None = None,
             stages: list[FakeStage] | None = None,
             adapter_name: str = "fake-1") -> FakeScenario:
    scripted = steps if steps is not None else [step(0)]
    allowed = catalog if catalog is not None else [action()]
    places = stages if stages is not None else [stage(index) for index in range(len(scripted) + 1)]
    return FakeScenario(schema_version="0.1", scenario_id="s-1", adapter_name=adapter_name,
                        allowed_actions=allowed, stages=places, steps=scripted)


def drive(coroutine):
    return asyncio.run(coroutine)


def started(instance: FakeEnvironmentAdapter) -> FakeEnvironmentAdapter:
    drive(instance.start())
    return instance


# --- A01: determinism, zero steps, no consumption on observe -------------------

def test_two_instances_replay_the_same_sequence_identically() -> None:
    built = scenario(steps=[step(0), step(1)], catalog=[action()],
                     stages=[stage(0), stage(1), stage(2)])
    first, second = started(FakeEnvironmentAdapter(built)), started(FakeEnvironmentAdapter(built))
    assert drive(first.observe()) == drive(second.observe())
    assert drive(first.execute(action())) == drive(second.execute(action()))
    assert drive(first.observe()) == drive(second.observe())
    assert drive(first.execute(action())) == drive(second.execute(action()))


def test_a_zero_step_scenario_replays_immediately() -> None:
    adapter = FakeEnvironmentAdapter(scenario(steps=[], catalog=[], stages=[stage(0)]))
    started(adapter)
    assert drive(adapter.observe()).observation_id == "obs-0"
    with pytest.raises(UnsupportedActionError):
        drive(adapter.execute(action()))


def test_a_zero_step_scenario_with_a_catalog_is_exhausted_immediately() -> None:
    adapter = FakeEnvironmentAdapter(scenario(steps=[], catalog=[action()], stages=[stage(0)]))
    started(adapter)
    assert drive(adapter.observe()).observation_id == "obs-0"
    with pytest.raises(FakeScriptExhaustedError):
        drive(adapter.execute(action()))


def test_an_empty_catalog_is_legal_and_declares_no_actions() -> None:
    adapter = FakeEnvironmentAdapter(scenario(steps=[], catalog=[], stages=[stage(0)]))
    assert adapter.capabilities().supported_actions == []
    assert adapter.capabilities().supported_outcome_types == []


def test_observe_does_not_consume_a_step() -> None:
    adapter = started(FakeEnvironmentAdapter(scenario(steps=[step(0), step(1)],
                                                      stages=[stage(0), stage(1), stage(2)])))
    for _ in range(3):
        assert drive(adapter.observe()).observation_id == "obs-0"
    drive(adapter.execute(action()))
    assert drive(adapter.observe()).observation_id == "obs-1"


def test_the_adapter_declares_the_agreed_capabilities() -> None:
    capabilities = FakeEnvironmentAdapter(scenario()).capabilities()
    assert isinstance(capabilities, AdapterCapabilities)
    assert capabilities.adapter_name == "fake-1"
    assert capabilities.environment_type == "fake"
    assert capabilities.supported_actions == ["click"]
    assert capabilities.supported_observation_types == ["fake.state"]
    assert capabilities.supported_condition_types == []
    assert capabilities.supported_outcome_types == []


def test_capability_action_types_keep_first_appearance_order_without_duplicates() -> None:
    catalog = [action("second", {"a": 1}), action("first", {"b": 2}), action("second", {"a": 3})]
    adapter = FakeEnvironmentAdapter(scenario(steps=[], catalog=catalog, stages=[stage(0)]))
    assert adapter.capabilities().supported_actions == ["second", "first"]


def test_capabilities_are_a_fresh_object_each_call() -> None:
    adapter = FakeEnvironmentAdapter(scenario())
    first, second = adapter.capabilities(), adapter.capabilities()
    assert first == second
    assert first is not second
    first.supported_actions.append("extra")
    assert adapter.capabilities().supported_actions == ["click"]


# --- A02: strict structure validation ------------------------------------------

def test_a_scenario_requires_one_more_stage_than_steps() -> None:
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0)], stages=[stage(0)])
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0)], stages=[stage(0), stage(1), stage(2)])


def test_step_outcome_and_observation_identifiers_must_be_unique() -> None:
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0), step(0)], stages=[stage(0), stage(1), stage(2)])
    twin = FakeStep(schema_version="0.1", step_id="step-1", action=action(), outcome=outcome(0))
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0), twin], stages=[stage(0), stage(1), stage(2)])
    with pytest.raises(FakeScenarioError):
        FakeScenario(schema_version="0.1", scenario_id="s", adapter_name="a", allowed_actions=[],
                     stages=[stage(0), _repeated_observation_stage(0)], steps=[])


def _repeated_observation_stage(index: int) -> FakeStage:
    """A stage whose observation reuses the first stage identifier."""
    return FakeStage(schema_version="0.1", actual_state={"count": index},
                     observation=observation(index, observation_id="obs-0"))


@pytest.mark.parametrize("field,value", [
    ("schema_version", "0.2"),
    ("scenario_id", 1),
    ("scenario_id", ""),
    ("adapter_name", None),
    ("allowed_actions", [{"action_type": "click"}]),
    ("stages", [{"actual_state": {}}]),
])
def test_a_malformed_scenario_field_is_rejected(field: str, value: object) -> None:
    fields = {
        "schema_version": "0.1", "scenario_id": "s-1", "adapter_name": "fake-1",
        "allowed_actions": [action()], "stages": [stage(0), stage(1)], "steps": [step(0)],
    }
    fields[field] = value
    with pytest.raises(FakeScenarioError):
        FakeScenario(**fields)


def test_an_unknown_field_is_rejected() -> None:
    """An undeclared field is refused, and the public JSON ingress never echoes it."""
    fields = scenario().model_dump(mode="json")
    fields[MARKER] = MARKER
    with pytest.raises(FakeScenarioError) as info:
        FakeScenario.model_validate(fields)
    assert MARKER not in str(info.value)
    with pytest.raises(FakeScenarioError) as json_info:
        FakeScenario.model_validate_json(json.dumps(fields))
    assert MARKER not in str(json_info.value)
    with pytest.raises(FakeScenarioError):
        FakeScenario.model_validate_json("{" + MARKER)


def test_a_non_fake_observation_environment_is_rejected() -> None:
    with pytest.raises(FakeScenarioError):
        FakeScenario(schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
                     allowed_actions=[], stages=[
                         FakeStage(schema_version="0.1", actual_state={},
                                   observation=observation(0, environment_type="browser"))],
                     steps=[])


def test_a_step_outside_the_catalog_is_rejected() -> None:
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0, candidate=action("other", {"z": 1}))],
                 catalog=[action()], stages=[stage(0), stage(1)])


def test_a_repeated_catalog_entry_is_rejected() -> None:
    with pytest.raises(FakeScenarioError):
        scenario(steps=[], catalog=[action(), action()], stages=[stage(0)])


def test_a_scenario_without_a_catalog_cannot_script_a_step() -> None:
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0)], catalog=[], stages=[stage(0), stage(1)])


def test_a_receipt_snapshot_must_match_the_stage_when_supplied() -> None:
    matching = step(0, produced=outcome(0, state_before={"count": 0}, state_after={"count": 1}))
    adapter = FakeEnvironmentAdapter(scenario(steps=[matching], stages=[stage(0), stage(1)]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.state_before == {"count": 0}
    assert receipt.state_after == {"count": 1}


def test_the_receipt_carries_no_hidden_actual_state() -> None:
    produced = outcome(0)
    adapter = FakeEnvironmentAdapter(scenario(steps=[step(0, produced=produced)]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.state_before is None and receipt.state_after is None
    assert receipt.changes == {"count": 1}
    assert "actual_state" not in receipt.model_dump()


# --- A03: validation order and no side effects ---------------------------------

def test_an_unsupported_action_type_precedes_payload_rejection() -> None:
    adapter = FakeEnvironmentAdapter(scenario())
    with pytest.raises(UnsupportedActionError):
        adapter.validate_action(action("other", {"unexpected": True}))
    with pytest.raises(InvalidActionPayloadError):
        adapter.validate_action(action("click", {"selector": "#other"}))


def test_validation_does_not_depend_on_the_lifecycle_or_the_current_step() -> None:
    adapter = FakeEnvironmentAdapter(scenario(steps=[step(0)], stages=[stage(0), stage(1)]))
    adapter.validate_action(action())
    started(adapter)
    drive(adapter.execute(action()))
    adapter.validate_action(action())


def test_validation_has_no_side_effects_on_state() -> None:
    adapter = FakeEnvironmentAdapter(scenario())
    before = adapter.capabilities()
    adapter.validate_action(action())
    assert adapter.capabilities() == before


def test_a_valid_but_non_current_action_is_refused_without_advancing() -> None:
    first, second = action("click", {"selector": "#a"}), action("click", {"selector": "#b"})
    adapter = FakeEnvironmentAdapter(scenario(
        steps=[step(0, candidate=first), step(1, candidate=second)],
        catalog=[first, second], stages=[stage(0), stage(1), stage(2)]))
    started(adapter)
    with pytest.raises(FakeStepMismatchError):
        drive(adapter.execute(second))
    assert drive(adapter.observe()).observation_id == "obs-0"
    drive(adapter.execute(first))
    assert drive(adapter.observe()).observation_id == "obs-1"


def test_metadata_never_affects_matching() -> None:
    adapter = FakeEnvironmentAdapter(scenario())
    started(adapter)
    adapter.validate_action(action(metadata={"note": "ignored"}))
    assert drive(adapter.execute(action(metadata={"other": 1}))).outcome_id == "out-0"


# --- A04: type-aware deep JSON matching ----------------------------------------

def payload_scenario(payload: dict) -> FakeEnvironmentAdapter:
    return FakeEnvironmentAdapter(scenario(catalog=[action("click", payload)], steps=[],
                                           stages=[stage(0)]))


@pytest.mark.parametrize("catalog_payload,candidate_payload,accepted", [
    ({"flag": True}, {"flag": True}, True),
    ({"flag": True}, {"flag": 1}, False),
    ({"flag": 1}, {"flag": True}, False),
    ({"count": 1}, {"count": 1.0}, False),
    ({"count": 1.0}, {"count": 1}, False),
    ({"count": 1.0}, {"count": 1.0}, True),
    ({"value": None}, {"value": None}, True),
    ({"value": None}, {"value": 0}, False),
    ({"items": [1, 2]}, {"items": [1, 2]}, True),
    ({"items": [1, 2]}, {"items": [2, 1]}, False),
    ({"items": [1, 2]}, {"items": [1, 2, 3]}, False),
    ({"nested": {"a": 1, "b": 2}}, {"nested": {"b": 2, "a": 1}}, True),
    ({"nested": {"a": 1}}, {"nested": {"a": 1, "b": 2}}, False),
    ({"text": "1"}, {"text": 1}, False),
])
def test_payload_matching_is_type_aware_and_recursive(
    catalog_payload: dict, candidate_payload: dict, accepted: bool,
) -> None:
    adapter = payload_scenario(catalog_payload)
    if accepted:
        adapter.validate_action(action("click", candidate_payload))
    else:
        with pytest.raises(InvalidActionPayloadError):
            adapter.validate_action(action("click", candidate_payload))


# --- A05/A07: execution outcomes, lifecycle and exhaustion ---------------------

@pytest.mark.parametrize("status,error_code", [
    ("succeeded", None),
    ("failed", "ENVIRONMENT_REJECTED"),
    ("failed", "EXECUTION_FAILED"),
    ("unknown", "RESPONSE_LOST"),
    ("unknown", "TIMEOUT"),
])
def test_every_scripted_outcome_status_is_returned_unchanged(
    status: str, error_code: str | None,
) -> None:
    info = None if error_code is None else ErrorInfo(schema_version="0.1", code=error_code,
                                                     message="scripted")
    produced = outcome(0, status=status, changes={}, error=info)
    adapter = FakeEnvironmentAdapter(scenario(steps=[step(0, produced=produced)]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.status is ExecutionStatus(status)
    if error_code is None:
        assert receipt.error is None
    else:
        assert receipt.error is not None
        assert receipt.error.code == error_code


def test_a_failed_outcome_may_keep_partial_effects_and_still_consumes_the_step() -> None:
    produced = outcome(0, status="failed", changes={"partial": True},
                       error=ErrorInfo(schema_version="0.1", code="EXECUTION_FAILED",
                                       message="scripted"))
    adapter = FakeEnvironmentAdapter(scenario(steps=[step(0, produced=produced)]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.status is ExecutionStatus.FAILED
    assert receipt.changes == {"partial": True}
    with pytest.raises(FakeScriptExhaustedError):
        drive(adapter.execute(action()))


def test_a_succeeded_outcome_cannot_carry_an_error() -> None:
    """The existing Core rule is preserved: the fixture adds no new enum or status."""
    with pytest.raises(Exception) as info:
        outcome(0, status="succeeded",
                error=ErrorInfo(schema_version="0.1", code="EXECUTION_FAILED", message="x"))
    assert "SUCCEEDED" in str(info.value)


def test_lifecycle_transitions_and_idempotence() -> None:
    adapter = FakeEnvironmentAdapter(scenario())
    with pytest.raises(FakeLifecycleError):
        drive(adapter.observe())
    with pytest.raises(FakeLifecycleError):
        drive(adapter.execute(action()))
    drive(adapter.start())
    drive(adapter.start())
    assert drive(adapter.observe()).observation_id == "obs-0"
    drive(adapter.stop())
    drive(adapter.stop())
    with pytest.raises(FakeLifecycleError):
        drive(adapter.observe())
    with pytest.raises(FakeLifecycleError):
        drive(adapter.start())


def test_observation_remains_available_after_exhaustion() -> None:
    adapter = started(FakeEnvironmentAdapter(scenario()))
    drive(adapter.execute(action()))
    assert drive(adapter.observe()).observation_id == "obs-1"
    with pytest.raises(FakeScriptExhaustedError):
        drive(adapter.execute(action()))
    assert drive(adapter.observe()).observation_id == "obs-1"


def test_a_rejected_execution_never_advances_the_cursor() -> None:
    adapter = started(FakeEnvironmentAdapter(
        scenario(steps=[step(0), step(1)], stages=[stage(0), stage(1), stage(2)])))
    with pytest.raises(InvalidActionPayloadError):
        drive(adapter.execute(action(payload={"selector": "#nope"})))
    assert drive(adapter.observe()).observation_id == "obs-0"
    drive(adapter.execute(action()))
    assert drive(adapter.observe()).observation_id == "obs-1"


def test_a_repeated_action_can_be_a_separate_later_step() -> None:
    adapter = started(FakeEnvironmentAdapter(
        scenario(steps=[step(0), step(1)], stages=[stage(0), stage(1), stage(2)])))
    assert drive(adapter.execute(action())).outcome_id == "out-0"
    assert drive(adapter.execute(action())).outcome_id == "out-1"


def test_observe_returns_the_scripted_visible_view_not_the_actual_state() -> None:
    built = scenario(steps=[], catalog=[],
                     stages=[stage(0, actual_state={"secret": MARKER}, visible={"shown": 1})])
    adapter = started(FakeEnvironmentAdapter(built))
    observation = drive(adapter.observe())
    assert observation.state == {"shown": 1}
    assert MARKER not in observation.model_dump_json()


# --- A09/A10: ownership, isolation and bounded errors --------------------------

def test_mutating_the_scenario_afterwards_does_not_change_the_adapter() -> None:
    built = scenario()
    adapter = FakeEnvironmentAdapter(built)
    built.adapter_name = "changed"
    built.stages[0].observation.state["visible"] = 99
    assert adapter.name == "fake-1"
    assert drive(_started_observe(adapter)).state == {"visible": 0}


async def _started_observe(adapter: FakeEnvironmentAdapter) -> Observation:
    await adapter.start()
    return await adapter.observe()


def test_mutating_a_returned_receipt_does_not_change_the_script() -> None:
    adapter = started(FakeEnvironmentAdapter(scenario()))
    receipt = drive(adapter.execute(action()))
    receipt.changes["tampered"] = True
    second = FakeEnvironmentAdapter(scenario())
    started(second)
    assert drive(second.execute(action())).changes == {"count": 1}


def test_mutating_a_returned_observation_does_not_change_the_stage() -> None:
    adapter = started(FakeEnvironmentAdapter(scenario()))
    observation = drive(adapter.observe())
    observation.state["tampered"] = True
    assert drive(adapter.observe()).state == {"visible": 0}


def test_a_copied_scenario_with_an_undeclared_key_is_rejected() -> None:
    forged = scenario().model_copy(update={MARKER: MARKER})
    with pytest.raises(FakeScenarioError) as info:
        FakeEnvironmentAdapter(forged)
    assert MARKER not in str(info.value)


def test_a_copied_scenario_with_an_invalid_stored_field_is_rejected() -> None:
    forged = scenario().model_copy(update={"schema_version": "9.9"})
    with pytest.raises(FakeScenarioError):
        FakeEnvironmentAdapter(forged)


def test_a_nested_illegal_mutation_is_rejected_without_json_laundering() -> None:
    """Either the in-place edit itself or the later revalidation must reject the content."""
    built = scenario()
    with pytest.raises(Exception) as info:
        built.stages[0].observation.state[1] = "non-string-key"
        FakeEnvironmentAdapter(built)
    assert info.value is not None


def test_a_nested_illegal_action_payload_change_is_rejected() -> None:
    built = scenario()
    with pytest.raises(Exception) as info:
        built.steps[0].action.payload["items"] = (1, 2)
        FakeEnvironmentAdapter(built)
    assert info.value is not None


def test_an_invalid_action_argument_is_a_bounded_input_error() -> None:
    adapter = FakeEnvironmentAdapter(scenario())
    with pytest.raises(FakeInputError) as info:
        adapter.validate_action({"action_type": MARKER})
    message = str(info.value)
    assert MARKER not in message
    assert info.value.code == "fake_input"


def test_an_action_with_a_nested_illegal_payload_is_rejected() -> None:
    adapter = FakeEnvironmentAdapter(scenario())
    forged = action().model_copy(update={"payload": {"nested": {1: "x"}}})
    with pytest.raises((InvalidActionPayloadError, FakeInputError)):
        adapter.validate_action(forged)


def test_facility_errors_expose_a_stable_code_and_path() -> None:
    adapter = started(FakeEnvironmentAdapter(scenario()))
    drive(adapter.execute(action()))
    try:
        drive(adapter.execute(action()))
    except FakeScriptExhaustedError as exc:
        assert exc.code == "fake_script_exhausted"
        assert exc.path == ("steps",)
    else:  # pragma: no cover - the second execute must be exhausted
        raise AssertionError("the exhausted script must reject")
    adapter2 = FakeEnvironmentAdapter(scenario())
    with pytest.raises(FakeLifecycleError) as lifecycle:
        drive(adapter2.execute(action()))
    assert lifecycle.value.code == "fake_lifecycle"
    assert lifecycle.value.path == ("execute",)


def test_a_mismatch_error_names_the_step_index_and_no_payload() -> None:
    second = action("click", {"selector": "#b"})
    adapter = FakeEnvironmentAdapter(scenario(
        steps=[step(0), step(1, candidate=action("click", {"selector": "#b"}))],
        catalog=[action(), second], stages=[stage(0), stage(1), stage(2)]))
    started(adapter)
    try:
        drive(adapter.execute(second))
    except FakeStepMismatchError as exc:
        assert exc.path == ("steps", 0)
        assert "#b" not in str(exc)
    else:  # pragma: no cover - the first step expects another action
        raise AssertionError("a non-current action must be rejected")


def test_no_public_error_message_echoes_the_scenario_or_a_raw_cause() -> None:
    built = scenario()
    built.stages[0].observation.state["marker"] = MARKER
    try:
        FakeEnvironmentAdapter(built.model_copy(update={"schema_version": MARKER}))
    except FakeScenarioError as exc:
        assert MARKER not in str(exc)
    else:  # pragma: no cover - the forged version must be rejected
        raise AssertionError("an invalid copied scenario must be rejected")


def test_a_snapshot_of_the_scenario_is_a_deep_copy() -> None:
    built = scenario()
    duplicate = copy.deepcopy(built)
    duplicate.stages[0].actual_state["count"] = 99
    assert built.stages[0].actual_state["count"] == 0


# --- round-002 correction: F01 snapshot truth, F02 fault mapping, F03 blanks, F04 depth ---

def _outcome_with_snapshots(*, before: dict | None, after: dict | None,
                            status: str = "succeeded") -> ActualOutcome:
    fields: dict = {"schema_version": "0.1", "outcome_id": "out-0", "status": status,
                    "changes": {}}
    if before is not None:
        fields["state_before"] = before
    if after is not None:
        fields["state_after"] = after
    return ActualOutcome(**fields)


def _fault_outcome(status: str, code: str | None) -> ActualOutcome:
    info = None if code is None else ErrorInfo(schema_version="0.1", code=code, message="scripted")
    return ActualOutcome(schema_version="0.1", outcome_id="out-0", status=status, changes={},
                         error=info)


@pytest.mark.parametrize("before,after,label", [
    ({"count": 99}, None, "a contradictory state_before"),
    (None, {"count": 99}, "a contradictory state_after"),
    ({"count": 0.0}, None, "an int/float-confused state_before"),
    (None, {"count": 1.0}, "an int/float-confused state_after"),
    ({"count": False}, None, "a bool/int-confused state_before"),
    (None, {"count": True}, "a bool/int-confused state_after"),
    ({"count": 0, "extra": 1}, None, "a superset state_before"),
])
def test_a_contradictory_snapshot_is_rejected(before: dict | None, after: dict | None,
                                              label: str) -> None:
    """F01: an explicit snapshot that contradicts stage truth is refused, not accepted."""
    produced = _outcome_with_snapshots(before=before, after=after)
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0, produced=produced)])


def test_a_matching_snapshot_pair_is_accepted() -> None:
    """F01 control: a snapshot equal to its stage truth is legal and returned unchanged."""
    produced = _outcome_with_snapshots(before={"count": 0}, after={"count": 1})
    adapter = FakeEnvironmentAdapter(scenario(steps=[step(0, produced=produced)]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.state_before == {"count": 0}
    assert receipt.state_after == {"count": 1}


def test_an_omitted_snapshot_stays_legal() -> None:
    """F01 control: omission is legal and the fixture never synthesises hidden state."""
    adapter = FakeEnvironmentAdapter(
        scenario(steps=[step(0, produced=_outcome_with_snapshots(before=None, after=None))]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.state_before is None and receipt.state_after is None
    assert "actual_state" not in receipt.model_dump()


def test_a_snapshot_forged_after_construction_is_still_rejected() -> None:
    """F01: the public constructor re-checks snapshots, not only the model validator."""
    built = scenario()
    forged = built.steps[0].outcome.model_copy(update={"state_before": {"count": 99}})
    built.steps[0] = built.steps[0].model_copy(update={"outcome": forged})
    with pytest.raises(FakeScenarioError):
        FakeEnvironmentAdapter(built)


@pytest.mark.parametrize("status,code", [
    ("failed", None),
    ("unknown", None),
    ("failed", "TIMEOUT"),
    ("failed", "RESPONSE_LOST"),
    ("unknown", "ENVIRONMENT_REJECTED"),
    ("unknown", "EXECUTION_FAILED"),
    ("failed", "OTHER"),
    ("unknown", "OTHER"),
])
def test_an_illegal_status_and_code_combination_is_rejected(status: str, code: str | None) -> None:
    """F02: a missing ErrorInfo or a code outside the status's closed set is refused."""
    produced = _fault_outcome(status, code)
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0, produced=produced)])


@pytest.mark.parametrize("status,code", [
    ("succeeded", None),
    ("failed", "ENVIRONMENT_REJECTED"),
    ("failed", "EXECUTION_FAILED"),
    ("unknown", "RESPONSE_LOST"),
    ("unknown", "TIMEOUT"),
])
def test_every_legal_status_and_code_combination_is_accepted(status: str,
                                                            code: str | None) -> None:
    """F02 control: the four agreed codes and the succeeded status all remain legal."""
    produced = _fault_outcome(status, code)
    adapter = FakeEnvironmentAdapter(scenario(steps=[step(0, produced=produced)]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.status is ExecutionStatus(status)
    if code is None:
        assert receipt.error is None
    else:
        assert receipt.error is not None and receipt.error.code == code


def test_a_failed_outcome_can_still_carry_partial_effects() -> None:
    """F02: the closed mapping constrains the error only; partial effects stay permitted."""
    produced = ActualOutcome(schema_version="0.1", outcome_id="out-0", status="failed",
                             changes={"partial": True},
                             error=ErrorInfo(schema_version="0.1", code="EXECUTION_FAILED",
                                             message="scripted"))
    adapter = FakeEnvironmentAdapter(scenario(steps=[step(0, produced=produced)]))
    started(adapter)
    receipt = drive(adapter.execute(action()))
    assert receipt.status is ExecutionStatus.FAILED
    assert receipt.changes == {"partial": True}


def test_an_outcome_failure_is_not_rewritten_into_a_core_wide_error() -> None:
    """F02: the Fake boundary closes the set; Core keeps its wider meaning untouched."""
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0, produced=_fault_outcome("failed", "SOMETHING_ELSE"))])
    # The Core model still accepts that value on its own, which is the preserved wider semantics.
    assert _fault_outcome("failed", "SOMETHING_ELSE").error is not None


@pytest.mark.parametrize("blank", [" ", "\t", "\n", " \t\n ", "\r\n"])
@pytest.mark.parametrize("field", ["scenario_id", "adapter_name"])
def test_a_whitespace_only_scenario_identifier_is_rejected(field: str, blank: str) -> None:
    """F03: a blank identifier is refused and never normalized into a valid one."""
    fields = {"schema_version": "0.1", "scenario_id": "s-1", "adapter_name": "fake-1",
              "allowed_actions": [action()], "stages": [stage(0), stage(1)], "steps": [step(0)]}
    fields[field] = blank
    with pytest.raises(FakeScenarioError) as info:
        FakeScenario(**fields)
    assert field in str(info.value)


@pytest.mark.parametrize("blank", [" ", "\t", "\n"])
def test_a_whitespace_only_step_identifier_is_rejected(blank: str) -> None:
    """F03: the step identifier is checked at the same strictness as the scenario."""
    with pytest.raises(FakeScenarioError):
        FakeStep(schema_version="0.1", step_id=blank, action=action(),
                 outcome=_outcome_with_snapshots(before=None, after=None))


def test_a_legitimate_identifier_is_not_normalized() -> None:
    """F03 control: an identifier that merely contains whitespace is preserved verbatim."""
    padded = FakeEnvironmentAdapter(scenario(adapter_name="a b\tc")).name
    assert padded == "a b\tc"


def _forged_scenario() -> FakeScenario:
    built = scenario()
    artifact = ArtifactRef(schema_version="0.1", artifact_id="artifact-0", artifact_type="file",
                           uri="synthetic://artifact")
    artifact.__dict__[MARKER] = MARKER
    built.stages[0].observation.artifacts = [artifact]
    return built


def test_an_unknown_key_on_a_deeply_nested_core_record_is_rejected() -> None:
    """F04: a nested Core record several levels down cannot hide a stored key."""
    with pytest.raises(FakeScenarioError) as info:
        FakeEnvironmentAdapter(_forged_scenario())
    assert MARKER not in str(info.value)


def test_an_unknown_key_on_a_nested_action_or_outcome_is_rejected() -> None:
    """F04: any reachable nested record is inspected, not only the first level."""
    forged_action = action().model_copy(update={MARKER: MARKER})
    built = scenario()
    built.steps[0] = built.steps[0].model_copy(update={"action": forged_action})
    with pytest.raises(FakeScenarioError):
        FakeEnvironmentAdapter(built)
    forged_outcome = _outcome_with_snapshots(before=None, after=None).model_copy(
        update={MARKER: MARKER})
    built = scenario()
    built.steps[0] = built.steps[0].model_copy(update={"outcome": forged_outcome})
    with pytest.raises(FakeScenarioError):
        FakeEnvironmentAdapter(built)


def test_an_unknown_key_on_a_nested_error_info_is_rejected() -> None:
    """F04: an ErrorInfo nested inside an outcome is inspected as well."""
    info = ErrorInfo(schema_version="0.1", code="EXECUTION_FAILED", message="scripted")
    produced = ActualOutcome(schema_version="0.1", outcome_id="out-0", status="failed",
                             changes={}, error=info.model_copy(update={MARKER: MARKER}))
    with pytest.raises(FakeScenarioError):
        scenario(steps=[step(0, produced=produced)])


def test_an_invalid_nested_core_field_is_a_facility_error_not_a_raw_one() -> None:
    """F04: a nested schema violation becomes the boundary's own bounded error."""
    info = ErrorInfo(schema_version="0.1", code="EXECUTION_FAILED", message="scripted")
    produced = ActualOutcome(schema_version="0.1", outcome_id="out-0", status="failed",
                             changes={}, error=info.model_copy(update={"schema_version": "9.9"}))
    with pytest.raises(FakeScenarioError) as caught:
        scenario(steps=[step(0, produced=produced)])
    assert MARKER not in str(caught.value)


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"scenario_id": 17},
    {"adapter_name": ""},
    {"stages": "not-a-list"},
])
def test_public_instance_ingress_rejects_a_copied_invalid_scenario(mutation: dict) -> None:
    """F04: model_validate and the constructor both revalidate an existing instance."""
    forged = scenario().model_copy(update=mutation)
    with pytest.raises(FakeScenarioError):
        FakeScenario.model_validate(forged)
    with pytest.raises(FakeScenarioError):
        FakeEnvironmentAdapter(forged)


def test_public_instance_ingress_accepts_a_valid_instance() -> None:
    """F04 control: an ordinary valid instance still passes both ingresses."""
    built = scenario()
    assert FakeScenario.model_validate(built) == built
    assert FakeScenario.model_validate(built.model_copy()) == built
    adapter = FakeEnvironmentAdapter(built)
    started(adapter)
    assert drive(adapter.observe()).observation_id == "obs-0"


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"action_type": 17},
    {"payload": "not-an-object"},
])
def test_a_copied_invalid_action_is_rejected_at_the_public_call(mutation: dict) -> None:
    """F04: the public call boundary revalidates an existing action instance."""
    adapter = FakeEnvironmentAdapter(scenario())
    with pytest.raises(FakeInputError):
        adapter.validate_action(action().model_copy(update=mutation))


def test_a_valid_action_instance_is_still_accepted_at_the_public_call() -> None:
    """F04 control: an ordinary action instance still validates and executes."""
    adapter = started(FakeEnvironmentAdapter(scenario()))
    adapter.validate_action(action())
    assert drive(adapter.execute(action())).outcome_id == "out-0"


def test_the_public_python_and_json_roundtrip_still_works() -> None:
    """F04 control: no coercion was added; legal Python and JSON both load as before."""
    built = scenario()
    fields = built.model_dump(mode="json")
    assert FakeScenario.model_validate(fields).model_dump(mode="json") == fields
    assert FakeScenario.model_validate_json(built.model_dump_json()) == built


def test_the_public_surface_is_closed_to_fixture_constants() -> None:
    """F05: the package exports components, models and error classes only."""
    import forecontract_fake

    assert sorted(forecontract_fake.__all__) == [
        "FakeEnvironmentAdapter",
        "FakeError",
        "FakeInputError",
        "FakeLifecycleError",
        "FakeScenario",
        "FakeScenarioError",
        "FakeScriptExhaustedError",
        "FakeStage",
        "FakeStep",
        "FakeStepMismatchError",
    ]
    assert [name for name in forecontract_fake.__all__ if name.isupper()] == []
    # The internal constants remain available to their own modules, just not as public exports.
    from forecontract_fake.models import FAKE_ENVIRONMENT_TYPE

    assert FAKE_ENVIRONMENT_TYPE == "fake"
    assert not hasattr(forecontract_fake, "FAKE_ENVIRONMENT_TYPE")
    assert not hasattr(forecontract_fake, "FAKE_ERROR_CODES")


# --- round-003 correction: raw nested structure is rejected before dependent validators ---

def _scenario_mapping(*, step_id: object = "step-0", outcome: object = None) -> dict:
    """A raw Python mapping for one scenario, with the given step and outcome content."""
    produced = outcome if outcome is not None else {
        "schema_version": "0.1", "outcome_id": "out-0", "status": "succeeded", "changes": {}}
    return {
        "schema_version": "0.1", "scenario_id": "s-1", "adapter_name": "fake-1",
        "allowed_actions": [action()], "stages": [stage(0), stage(1)],
        "steps": [{"schema_version": "0.1", "step_id": step_id, "action": action(),
                   "outcome": produced}],
    }


_DAMAGED_STATUS = "SYNTHETIC_NOT_A_STATUS"


def _damaged_outcome() -> ActualOutcome:
    """A real Core instance whose stored status is not one of the closed values."""
    return outcome(0).model_copy(update={"status": _DAMAGED_STATUS})


@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_damaged_nested_outcome_status_is_a_bounded_facility_error(ingress: str) -> None:
    """F04: the dependent status read never runs before the nested record is revalidated."""
    payload = _scenario_mapping(outcome=_damaged_outcome())
    with pytest.raises(FakeScenarioError) as info:
        if ingress == "constructor":
            FakeScenario(**payload)
        else:
            FakeScenario.model_validate(payload)
    message = str(info.value)
    assert _DAMAGED_STATUS not in message
    assert info.value.code == "fake_scenario"
    assert len(message) < 240


@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_damaged_nested_outcome_status_in_a_plain_mapping_is_rejected(ingress: str) -> None:
    """F04: a plain nested mapping that never produced a valid record is refused too."""
    payload = _scenario_mapping(outcome={
        "schema_version": "0.1", "outcome_id": "out-0", "status": _DAMAGED_STATUS,
        "changes": {}})
    with pytest.raises(FakeScenarioError) as info:
        if ingress == "constructor":
            FakeScenario(**payload)
        else:
            FakeScenario.model_validate(payload)
    assert _DAMAGED_STATUS not in str(info.value)


def test_a_damaged_outcome_status_on_the_step_record_is_a_facility_error() -> None:
    """F04: FakeStep's own constructor reports the boundary error, not an AttributeError."""
    with pytest.raises(FakeScenarioError) as info:
        FakeStep(schema_version="0.1", step_id="step-0", action=action(),
                 outcome=_damaged_outcome())
    assert _DAMAGED_STATUS not in str(info.value)
    assert "outcome" in str(info.value)


@pytest.mark.parametrize("field,value", [
    ("step_id", ["a"]),
    ("step_id", {"a": 1}),
    ("outcome_id", ["o"]),
    ("outcome_id", {"o": 1}),
])
@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_sequence_or_object_identifier_is_a_bounded_rejection(field: str, value: object,
                                                               ingress: str) -> None:
    """F04 control: a wrongly typed identifier is refused as a facility error either way."""
    if field == "step_id":
        payload = _scenario_mapping(step_id=value)
    else:
        payload = _scenario_mapping(outcome={
            "schema_version": "0.1", "outcome_id": value, "status": "succeeded", "changes": {}})
    with pytest.raises(FakeScenarioError):
        if ingress == "constructor":
            FakeScenario(**payload)
        else:
            FakeScenario.model_validate(payload)


def test_no_public_ingress_raises_a_raw_python_error_for_damaged_nested_content() -> None:
    """F04: every public ingress reports the boundary error, never AttributeError/TypeError."""
    damaged = _damaged_outcome()
    attempts = [
        lambda: FakeScenario(**_scenario_mapping(outcome=damaged)),
        lambda: FakeScenario.model_validate(_scenario_mapping(outcome=damaged)),
        lambda: FakeStep(schema_version="0.1", step_id="step-0", action=action(), outcome=damaged),
        lambda: FakeEnvironmentAdapter(scenario().model_copy(update={
            "steps": [{**scenario().steps[0].model_dump(mode="python"), "outcome": damaged}]})),
    ]
    for attempt in attempts:
        with pytest.raises(FakeScenarioError):
            attempt()


def test_legal_nested_content_still_constructs_and_round_trips() -> None:
    """F04 control: the pre-check refuses only damaged structure, never legal content."""
    built = scenario()
    assert FakeScenario.model_validate(built) == built
    assert FakeScenario.model_validate_json(built.model_dump_json()) == built
    assert FakeScenario(**scenario().model_dump(mode="python")) == built
    # A legal Core enum instance and a legal nested Core instance both still pass.
    assert FakeScenario(**_scenario_mapping(outcome=outcome(0))) == built


# --- round-004 correction: declared container shape first, bounded recursion ---

def _cyclic_list() -> list:
    container: list = []
    container.append(container)
    return container


def _mapping_with_cyclic_steps(container: object) -> dict:
    data = scenario().model_dump(mode="python")
    data["steps"] = container
    return data


@pytest.mark.parametrize("ingress", ["constructor", "model_validate"])
def test_a_cyclic_steps_container_is_a_bounded_rejection(ingress: str) -> None:
    """F04: a cyclic declared list terminates as a facility error, never a RecursionError."""
    payload = _mapping_with_cyclic_steps(_cyclic_list())
    with pytest.raises(FakeScenarioError) as info:
        if ingress == "constructor":
            FakeScenario(**payload)
        else:
            FakeScenario.model_validate(payload)
    assert info.value.code == "fake_scenario"
    assert len(str(info.value)) < 240


@pytest.mark.parametrize("container,label", [
    (_cyclic_list(), "a cyclic list"),
    (7, "a scalar"),
    ("steps", "a string"),
    ({"a": 1}, "a mapping"),
    ((), "a tuple"),
])
def test_a_non_list_steps_container_is_a_bounded_rejection(container: object,
                                                           label: str) -> None:
    """F04: the declared strict list field refuses any shape that is not a list."""
    with pytest.raises(FakeScenarioError):
        FakeScenario(**_mapping_with_cyclic_steps(container))


def test_a_deeply_nested_non_record_element_terminates_as_a_facility_error() -> None:
    """F04: recursion over list elements is bounded by the declared element shape."""
    deep: list = []
    cursor = deep
    for _ in range(60):
        nxt: list = []
        cursor.append(nxt)
        cursor = nxt
    with pytest.raises(FakeScenarioError):
        FakeScenario(**_mapping_with_cyclic_steps(deep))


def test_legal_shared_references_and_nested_json_still_construct() -> None:
    """F04 control: a shared nested reference and a legal nested JSON object stay usable."""
    shared_action = action()
    shared = FakeScenario(
        schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
        allowed_actions=[shared_action], stages=[stage(0), stage(1), stage(2)],
        steps=[FakeStep(schema_version="0.1", step_id="step-0", action=shared_action,
                        outcome=outcome(0)),
               FakeStep(schema_version="0.1", step_id="step-1", action=shared_action,
                        outcome=outcome(1))])
    assert shared.steps[0].action is shared.steps[1].action
    assert FakeScenario.model_validate(shared.model_dump(mode="python")).steps[1].step_id == "step-1"
    nested = {"a": [1, {"b": [2, [3, {"c": None}]]}], "d": {"e": [4]}}
    built = FakeScenario(**{**scenario().model_dump(mode="python"), "stages": [
        FakeStage(schema_version="0.1", actual_state=nested,
                  observation=observation(0, observation_id="obs-nested")),
        stage(1)]})
    assert built.stages[0].actual_state == nested
    adapter = FakeEnvironmentAdapter(built)
    started(adapter)
    assert drive(adapter.observe()).observation_id == "obs-nested"


# --- round-005 correction: declared element type before any nested rebuild ---

def _self_holding_scenario() -> FakeScenario:
    """A valid scenario that holds itself as one of its own steps."""
    built = scenario()
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
    payload = scenario().model_dump(mode="python")
    payload["steps"] = [_self_holding_scenario()]
    with pytest.raises(FakeScenarioError) as info:
        if ingress == "constructor":
            FakeScenario(**payload)
        else:
            FakeScenario.model_validate(payload)
    message = str(info.value)
    assert "steps" in message
    assert len(message) < 240
    # A direct rejection, not a RecursionError that was wrapped layer by layer.
    assert _cause_depth(info.value) == 0


@pytest.mark.parametrize("field,element", [
    ("steps", "scenario"),
    ("stages", "step"),
    ("allowed_actions", "stage"),
])
def test_a_wrong_class_list_element_is_refused(field: str, element: str) -> None:
    """F04: an element of the wrong model class is refused instead of being rebuilt."""
    payload = scenario().model_dump(mode="python")
    payload[field] = [_self_holding_scenario() if element == "scenario" else
                      scenario().steps[0] if element == "step" else stage(0)]
    with pytest.raises(FakeScenarioError) as info:
        FakeScenario(**payload)
    assert field in str(info.value)


def test_a_wrong_class_element_does_not_recurse_or_wrap() -> None:
    """F04: the rejection is bounded even when the wrong element holds a whole record."""
    payload = scenario().model_dump(mode="python")
    payload["steps"] = [scenario()]
    with pytest.raises(FakeScenarioError) as info:
        FakeScenario(**payload)
    assert _cause_depth(info.value) == 0
    assert len(str(info.value)) < 240


def test_legal_sharing_and_acyclic_structures_still_work() -> None:
    """F04 control: shared nested objects and acyclic depth stay usable after the gate."""
    shared_artifact = ArtifactRef(schema_version="0.1", artifact_id="artifact-shared",
                                  artifact_type="file", uri="synthetic://artifact")
    scenario_steps = [step(0)]
    built = FakeScenario(schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
                         allowed_actions=[action()], steps=scenario_steps,
                         stages=[
                             FakeStage(schema_version="0.1", actual_state={"count": 0},
                                       observation=Observation(
                                           schema_version="0.1", observation_id="obs-a",
                                           environment_type="fake", state={"visible": 0},
                                           timestamp=NOW, artifacts=[shared_artifact])),
                             FakeStage(schema_version="0.1", actual_state={"count": 1},
                                       observation=Observation(
                                           schema_version="0.1", observation_id="obs-b",
                                           environment_type="fake", state={"visible": 1},
                                           timestamp=NOW, artifacts=[shared_artifact]))])
    assert built.stages[0].observation.artifacts[0] is shared_artifact
    assert FakeScenario.model_validate(built.model_dump(mode="python")) == built
    # A legal nested JSON object is untouched, and the same object may be shared by two fields.
    nested = {"a": [1, {"b": [2, [3, {"c": None}]]}]}
    shared_state = FakeScenario(schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
                                allowed_actions=[], steps=[],
                                stages=[FakeStage(schema_version="0.1", actual_state=nested,
                                                  observation=observation(0, observation_id="o1"))])
    assert shared_state.stages[0].actual_state == nested


# --- round-006 correction: an unknown "self" key is an ordinary unknown key ---

def _public_records() -> list[tuple[str, type, object]]:
    """One valid record per public model, for the parser-boundary matrix."""
    from forecontract_scripted import PlannerScript, PlannerStep

    built_stage = stage(0)
    built_step = step(0)
    built_scenario = scenario()
    built_planner_step = PlannerStep(
        schema_version="0.1", step_id="ps-0", expected_environment_type="fake",
        expected_state={"count": 0},
        forecast=Forecast(schema_version="0.1", forecast_id="f-1", proposed_action=action(),
                          expected_outcomes=[]))
    built_script = PlannerScript(schema_version="0.1", script_id="p-1",
                                 steps=[built_planner_step])
    return [
        ("FakeScenario", FakeScenario, built_scenario),
        ("FakeStage", FakeStage, built_stage),
        ("FakeStep", FakeStep, built_step),
        ("PlannerScript", PlannerScript, built_script),
        ("PlannerStep", PlannerStep, built_planner_step),
    ]


def _marked(record: object, key: str) -> dict:
    """A raw Python projection of a valid record with one extra key added."""
    return {**record.model_dump(mode="python"), key: _SELF_MARKER}


_SELF_MARKER = "SYNTHETIC_SELF_KEY"


@pytest.mark.parametrize("form", ["mapping", "json"])
@pytest.mark.parametrize("name,model,record", _public_records(),
                         ids=[entry[0] for entry in _public_records()])
def test_an_unknown_self_key_is_a_bounded_facility_error(form: str, name: str, model: type,
                                                        record: object) -> None:
    """F06: an unknown key named 'self' is rejected like any other unknown key, not a TypeError."""
    payload = _marked(record, "self")
    error = FakeScenarioError if name.startswith("Fake") else None
    with pytest.raises((FakeScenarioError, PlannerScriptError)) as info:
        if form == "mapping":
            model.model_validate(payload)
        else:
            model.model_validate_json(json.dumps(payload, default=str))
    message = str(info.value)
    assert len(message) < 240
    assert _SELF_MARKER not in message
    if error is not None:
        assert isinstance(info.value, error)


@pytest.mark.parametrize("form", ["mapping", "json"])
@pytest.mark.parametrize("name,model,record", _public_records(),
                         ids=[entry[0] for entry in _public_records()])
def test_an_ordinary_unknown_key_is_still_rejected(form: str, name: str, model: type,
                                                  record: object) -> None:
    """F06 control: the pre-existing extra=forbid behaviour is unchanged for any other key."""
    payload = _marked(record, _SELF_MARKER)
    with pytest.raises((FakeScenarioError, PlannerScriptError)) as info:
        if form == "mapping":
            model.model_validate(payload)
        else:
            model.model_validate_json(json.dumps(payload, default=str))
    assert _SELF_MARKER not in str(info.value)


@pytest.mark.parametrize("form", ["mapping", "json"])
@pytest.mark.parametrize("name,model,record", _public_records(),
                         ids=[entry[0] for entry in _public_records()])
def test_a_valid_record_still_parses_through_both_public_form(form: str, name: str, model: type,
                                                             record: object) -> None:
    """F06 control: a legal mapping and a legal JSON document still parse to an equal record."""
    if form == "mapping":
        parsed = model.model_validate(record.model_dump(mode="json"))
    else:
        parsed = model.model_validate_json(record.model_dump_json())
    assert parsed == record
    assert type(parsed) is model


def test_the_supported_validation_parameters_are_still_forwarded() -> None:
    """F06 control: the wrapper still forwards the supported parser parameters."""
    built = stage(0)
    assert FakeStage.model_validate(built.model_dump(mode="python"), strict=True) == built
    assert FakeStage.model_validate(built.model_dump(mode="python"), by_name=True) == built
    assert FakeStage.model_validate(built.model_dump(mode="python"), context={"a": 1}) == built
    assert FakeStage.model_validate_json(built.model_dump_json(), strict=True) == built
    assert FakeStage.model_validate_json(built.model_dump_json(), context={"a": 1}) == built


def test_attribute_based_parsing_still_works() -> None:
    """F06 control: from_attributes parsing is unaffected by the corrected constructor."""

    class Duck:
        def __init__(self, **fields: object) -> None:
            for key, value in fields.items():
                setattr(self, key, value)

    for name, model, record in _public_records():
        duck = Duck(**record.model_dump(mode="python"))
        assert model.model_validate(duck, from_attributes=True) == record, name


def test_shared_references_and_acyclic_nesting_survive_the_parser_boundary() -> None:
    """F06 control: sharing and acyclic depth behave identically through mapping and JSON."""
    shared_action = action()
    built = FakeScenario(
        schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
        allowed_actions=[shared_action], stages=[stage(0), stage(1), stage(2)],
        steps=[FakeStep(schema_version="0.1", step_id="step-0", action=shared_action,
                        outcome=outcome(0)),
               FakeStep(schema_version="0.1", step_id="step-1", action=shared_action,
                        outcome=outcome(1))])
    from_mapping = FakeScenario.model_validate(built.model_dump(mode="python"))
    from_json = FakeScenario.model_validate_json(built.model_dump_json())
    assert from_mapping == built == from_json
    # Parsing from data builds an equivalent record; the shared source object was serialised
    # twice, so equality is the contract here rather than object identity.
    assert from_mapping.steps[0].action == from_mapping.steps[1].action
    assert from_json.steps[0].action == from_json.steps[1].action
    nested = {"a": [1, {"b": [2, [3, {"c": None}]]}], "d": {"e": [4]}}
    deep = FakeScenario(schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
                        allowed_actions=[], steps=[],
                        stages=[FakeStage(schema_version="0.1", actual_state=nested,
                                          observation=observation(0, observation_id="o-deep"))])
    assert FakeScenario.model_validate(deep.model_dump(mode="python")) == deep
    assert FakeScenario.model_validate_json(deep.model_dump_json()) == deep
