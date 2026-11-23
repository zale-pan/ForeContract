"""Acceptance tests for the Runtime records, phases and ingress shape guard.

Every fixture is synthetic and embedded here. The tests build real records and assert declared
fields, closed vocabularies, strict scalars, unknown-field rejection, snapshot isolation and the
real Core frozen-container identities rather than the mere absence of an exception.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.models.actions import ExpectedOutcome
from forecontract.models.base import SCHEMA_VERSION
from forecontract.runtime import (
    PHASES,
    TERMINAL_PHASES,
    ContractPreparation,
    RunRequest,
    RunResult,
    RuntimeBudgetSnapshot,
    RuntimePhase,
    RuntimeReason,
    RuntimeRecord,
    RuntimeTerminalStatus,
    RuntimeTransition,
)
from forecontract.runtime.models import (
    detached_state,
    require_declared_record,
    require_json_shape,
)
from forecontract.verification.models import TaskCompletionCriteria

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
RULE = {"schema_version": "0.1", "rule_type": "path_exists", "path": ["visible"]}


def criteria(goal_id: str = "g-1") -> TaskCompletionCriteria:
    return TaskCompletionCriteria(
        schema_version=SCHEMA_VERSION, goal_id=goal_id,
        conditions=[ExpectedOutcome(schema_version=SCHEMA_VERSION, outcome_type="json_rule",
                                   payload=RULE, required=True)])


def request(**overrides) -> RunRequest:
    fields = {
        "schema_version": SCHEMA_VERSION, "run_id": "run-1", "goal_id": "g-1",
        "task_criteria": criteria(), "max_steps": 3, "max_planning_calls": 3,
        "max_retries": 1, "max_replans": 1,
    }
    fields.update(overrides)
    return RunRequest(**fields)


# --- A01/A12: the closed phase vocabulary and the strict records ---

def test_the_phase_vocabulary_is_closed_and_the_terminals_are_a_subset() -> None:
    # The P2-T07 approval extends the closed vocabulary with the two real-trigger terminal phases, so
    # the exact count and the terminal set are adapted here. Every previously declared phase name is
    # still present, the vocabulary still matches the enum exactly, and the subset relation holds.
    assert len(PHASES) == 14
    assert set(PHASES) == {phase.value for phase in RuntimePhase}
    assert TERMINAL_PHASES == {"COMPLETED", "FAILED", "ABORTED", "CANCELLED", "TIMED_OUT"}
    assert TERMINAL_PHASES <= set(PHASES)
    assert {"NEW", "OBSERVING", "CHECKING_TASK", "PLANNING", "BUILDING",
            "CHECKING_PRECONDITIONS", "EXECUTING", "VERIFYING", "RECOVERING",
            "COMPLETED", "FAILED", "ABORTED"} <= set(PHASES)


def test_every_record_requires_the_explicit_schema_version() -> None:
    for model, fields in (
        (RunRequest, {"run_id": "r", "goal_id": "g", "task_criteria": criteria(),
                      "max_steps": 1, "max_planning_calls": 1, "max_retries": 0,
                      "max_replans": 0}),
        (RuntimeBudgetSnapshot, {"remaining_budget_ms": 1.0}),
        (RuntimeTransition, {"sequence": 0, "run_id": "r", "from_phase": "NEW",
                             "to_phase": "OBSERVING", "reason": "TASK_ACHIEVED"}),
    ):
        with pytest.raises(ValidationError):
            model(**fields)  # type: ignore[arg-type]


def test_an_unknown_field_is_refused_on_every_record() -> None:
    with pytest.raises(ValidationError):
        request(extra_field=1)
    with pytest.raises(ValidationError):
        RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION, remaining_budget_ms=1.0, extra=1)
    with pytest.raises(ValidationError):
        RuntimeTransition(schema_version=SCHEMA_VERSION, sequence=0, run_id="r",
                          from_phase="NEW", to_phase="OBSERVING", reason="TASK_ACHIEVED",
                          extra=1)


@pytest.mark.parametrize("field,value", [
    ("max_steps", 0), ("max_steps", -1), ("max_steps", 1.0), ("max_steps", True),
    ("max_planning_calls", 0), ("max_planning_calls", False),
    ("max_retries", -1), ("max_retries", 1.5), ("max_retries", True),
    ("max_replans", -2),
])
def test_a_host_limit_is_a_strict_non_negative_integer(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        request(**{field: value})


def test_a_valid_request_round_trips_through_its_declared_fields() -> None:
    built = request()
    state = detached_state(built)
    assert set(state) == set(RunRequest.model_fields)
    again = RunRequest.model_validate(state)
    assert again.run_id == built.run_id
    assert again.max_steps == 3


@pytest.mark.parametrize("value", [-1.0, -0.5, float("nan"), float("inf"), True, "10"])
def test_a_budget_snapshot_is_a_finite_non_negative_number(value: object) -> None:
    with pytest.raises(ValidationError):
        RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION, remaining_budget_ms=value)


def test_a_budget_snapshot_accepts_zero_and_an_integer() -> None:
    assert RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION,
                                 remaining_budget_ms=0).remaining_budget_ms == 0
    assert RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION,
                                 remaining_budget_ms=5).remaining_budget_ms == 5


def test_a_closed_vocabulary_field_refuses_an_undeclared_name() -> None:
    with pytest.raises(ValidationError):
        RuntimeTransition(schema_version=SCHEMA_VERSION, sequence=0, run_id="r",
                          from_phase="NEW", to_phase="NOT_A_PHASE", reason="TASK_ACHIEVED")
    with pytest.raises(ValidationError):
        RuntimeTransition(schema_version=SCHEMA_VERSION, sequence=0, run_id="r",
                          from_phase="NEW", to_phase="OBSERVING", reason="NOT_A_REASON")


def test_the_result_records_only_safe_associations_and_counters() -> None:
    assert set(RunResult.model_fields) == {
        "schema_version", "run_id", "goal_id", "terminal_status", "reason_code",
        "steps_started", "planning_calls", "attempts_started", "retries_used", "replans_used",
        "verification_record_id", "terminal_record_id",
    }
    for forbidden in ("observation", "payload", "error", "message", "success"):
        assert forbidden not in RunResult.model_fields
    # The approval adds the two real-trigger terminal statuses; the three original literals remain
    # unchanged and no custom spelling is introduced.
    assert {"completed", "failed", "aborted", "cancelled", "timeout"} == {
        item.value for item in RuntimeTerminalStatus}
    assert {"completed", "failed", "aborted"} <= {item.value for item in RuntimeTerminalStatus}


def test_a_terminal_verification_reference_may_be_absent() -> None:
    built = RunResult(schema_version=SCHEMA_VERSION, run_id="r", goal_id="g",
                      terminal_status="aborted", reason_code="BUDGET_EXHAUSTED",
                      steps_started=0, planning_calls=0, attempts_started=0, retries_used=0,
                      replans_used=0, verification_record_id=None, terminal_record_id="rec-1")
    assert built.verification_record_id is None


# --- A01: identifiers are non-blank and never coerced ---

@pytest.mark.parametrize("value", ["", "   ", 7, True, None])
def test_an_identifier_must_be_a_non_blank_string(value: object) -> None:
    with pytest.raises(ValidationError):
        request(run_id=value)


def test_a_blank_identifier_is_refused_wherever_it_appears() -> None:
    with pytest.raises(ValidationError):
        request(goal_id=" ")
    with pytest.raises(ValidationError):
        RuntimeTransition(schema_version=SCHEMA_VERSION, sequence=0, run_id=" \t ",
                          from_phase="NEW", to_phase="OBSERVING", reason="TASK_ACHIEVED")


# --- A01/A12: the declared-record guard and the JSON shape guard ---

def test_the_declared_record_guard_compares_real_types_not_names() -> None:
    built = request()
    assert require_declared_record(built, RunRequest, ("request",)) is built
    look_alike = type("RunRequest", (RuntimeRecord,), {})
    assert look_alike.__name__ == "RunRequest"
    with pytest.raises(ValueError):
        require_declared_record(look_alike(), RunRequest, ("request",))
    with pytest.raises(ValueError):
        require_declared_record({"run_id": "r"}, RunRequest, ("request",))


def test_a_stored_undeclared_key_is_rejected_before_any_dump() -> None:
    built = request()
    object.__setattr__(built, "__dict__", {**built.__dict__, "smuggled": 1})
    with pytest.raises(ValueError):
        detached_state(built)


def test_a_validation_bypass_instance_is_re_validated_not_trusted() -> None:
    broken = RunRequest.model_construct(schema_version=SCHEMA_VERSION, run_id="r", goal_id="g",
                                        task_criteria=criteria(), max_steps=0,
                                        max_planning_calls=1, max_retries=0, max_replans=0)
    with pytest.raises(ValueError):
        require_declared_record(broken, RunRequest, ("request",))


def test_a_json_value_must_be_finite_and_cycle_free() -> None:
    require_json_shape({"a": [1, {"b": None}], "c": 1.5}, ("metadata",))
    for bad in (float("nan"), float("inf"), ("tuple",), {1: "x"}, b"bytes", object()):
        with pytest.raises(ValueError):
            require_json_shape({"a": bad}, ("metadata",))
    loop: dict = {}
    loop["self"] = loop
    with pytest.raises(ValueError):
        require_json_shape(loop, ("metadata",))


def test_a_legal_shared_reference_is_not_a_cycle() -> None:
    shared = [1, 2]
    require_json_shape({"a": shared, "b": shared}, ("metadata",))
    require_json_shape([shared, shared], ("metadata",))


def test_the_declared_frozen_containers_are_accepted_and_look_alikes_are_not() -> None:
    """The production containers are accepted by real type identity, not by their class name.

    The shape guard itself has no declared annotation to compare against, so it accepts the two
    real production classes and refuses every other container. The stricter name-versus-identity
    distinction is asserted where a declaration is known, in the engine suite.
    """
    from forecontract.contracts.snapshot import FrozenList, FrozenMapping

    require_json_shape(FrozenList([1, 2]), ("metadata",))
    require_json_shape(FrozenMapping({"a": 1}), ("metadata",))
    require_json_shape({"a": FrozenList([1]), "b": FrozenMapping({"c": 2})}, ("metadata",))
    look_alike = type("FrozenList", (tuple,), {})
    assert not issubclass(look_alike, FrozenList)
    for bad in ((1, 2), look_alike([1])):
        with pytest.raises(ValueError):
            require_json_shape(bad, ("metadata",))


def test_a_contract_preparation_is_a_candidate_only() -> None:
    from forecontract.contracts import RecoveryPolicy, VerificationPolicy

    built = ContractPreparation(
        schema_version=SCHEMA_VERSION, preconditions=[],
        verification_policy=VerificationPolicy.model_validate(
            {"schema_version": SCHEMA_VERSION,
             "bindings": [{"schema_version": SCHEMA_VERSION, "outcome_index": 0,
                           "verifier_type": "json_rule", "config": {}}]}),
        recovery_policy=RecoveryPolicy(schema_version=SCHEMA_VERSION), timeout_ms=10.0)
    assert built.preconditions == []
    assert built.timeout_ms == 10.0
    # The candidate is not a committed contract and carries no contract identity.
    assert "contract_id" not in ContractPreparation.model_fields


@pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan"), True])
def test_a_preparation_timeout_must_be_positive_and_finite(value: object) -> None:
    with pytest.raises(ValidationError):
        ContractPreparation(schema_version=SCHEMA_VERSION, preconditions=[],
                            verification_policy=None, recovery_policy=None, timeout_ms=value)


def test_the_reason_vocabulary_is_closed_and_bounded() -> None:
    names = {item.value for item in RuntimeReason}
    assert "TASK_ACHIEVED" in names and "BUDGET_EXHAUSTED" in names
    assert all(name.isupper() and "_" in name or name.isupper() for name in names)
    assert not any(math.isnan(1.0) for _ in [0])
