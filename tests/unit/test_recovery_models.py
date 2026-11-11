"""Acceptance tests for the recovery records.

These tests own the ingress, ownership and closed-vocabulary properties of
``RecoveryContext``, ``ActionSafetyDeclaration``, ``RecoveryEvidence`` and
``RecoveryAssessment``: strict shapes at construction, mapping, existing-instance and JSON
ingress; bounded cycle refusal; detached ownership in both directions; transactional assignment;
and the closed kind, decision, reason and observation combinations.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.contracts import RecoveryPolicy
from forecontract.models.observations import Observation
from forecontract.models.outcomes import (
    ActualOutcome,
    ErrorInfo,
    VerificationResult,
    VerificationStatus,
)
from forecontract.recovery import (
    ActionSafetyDeclaration,
    RecoveryAssessment,
    RecoveryContext,
    RecoveryError,
    RecoveryEvidence,
)
from forecontract.verification import VerificationEvidence

NOW = datetime.datetime(2026, 10, 10, 12, 0, tzinfo=datetime.timezone.utc)
MARKER = "SYNTHETIC_PRIVATE_VALUE"


def observation(identifier: str = "obs-1") -> Observation:
    return Observation(schema_version="0.1", observation_id=identifier, environment_type="fake",
                       state={"a": 1}, timestamp=NOW)


def outcome(status: str = "failed", *, recoverable: object = True) -> ActualOutcome:
    error = None if status == "succeeded" else ErrorInfo(
        schema_version="0.1", code="E", message="m", recoverable=recoverable)
    return ActualOutcome(schema_version="0.1", outcome_id="o-1", status=status, error=error)


def context(**overrides: object) -> RecoveryContext:
    fields: dict = {
        "schema_version": "0.1", "run_id": "r-1", "step_id": "s-1", "contract_id": "c-1",
        "attempt_id": "a-1", "actual_outcome": outcome(), "remaining_budget_ms": 100.0,
        "remaining_retries": 1, "remaining_replans": 1, "contract_retry_count": 0,
        "stop_requested": False,
    }
    fields.update(overrides)
    return RecoveryContext(**fields)


def safety(**overrides: object) -> ActionSafetyDeclaration:
    fields: dict = {
        "schema_version": "0.1", "contract_id": "c-1", "adapter_name": "fake-1",
        "action_type": "click", "retry_safety": "idempotent",
    }
    fields.update(overrides)
    return ActionSafetyDeclaration(**fields)


def step_result(status: str = "fail") -> VerificationResult:
    return VerificationResult(
        schema_version="0.1", status=VerificationStatus(status), verifier="json_step",
        outcome_results=[], reason="REQUIRED_FAILED",
        metadata={"contract_id": "c-1", "goal_id": "g-1", "observation_id": "obs-1",
                  "scope": "step", "state_complete": True})


def evidence(**overrides: object) -> RecoveryEvidence:
    fields: dict = {
        "schema_version": "0.1", "run_id": "r-1", "step_id": "s-1", "contract_id": "c-1",
        "attempt_id": "a-1", "outcome_id": "o-1",
        "verification_evidence": VerificationEvidence(schema_version="0.1",
                                                      observation=observation(),
                                                      state_complete=True),
        "step_result": step_result(), "effect_state": "not_applied",
        "preconditions_status": VerificationStatus.PASS,
    }
    fields.update(overrides)
    return RecoveryEvidence(**fields)


def parse(model: type, fields: dict, mode: str) -> object:
    if mode == "constructor":
        return model(**fields)
    if mode == "mapping":
        return model.model_validate(fields)
    return model.model_validate_json(json.dumps(fields, default=str))


# --- A01: legal construction at every ingress ---

@pytest.mark.parametrize("mode", ["constructor", "mapping"])
def test_every_record_builds_from_a_legal_mapping(mode: str) -> None:
    assert parse(RecoveryContext, context().model_dump(mode="json"), mode).run_id == "r-1"
    assert parse(ActionSafetyDeclaration, safety().model_dump(mode="json"), mode).adapter_name \
        == "fake-1"
    assert parse(RecoveryAssessment, assessment().model_dump(mode="json"), mode).kind == "decision"


def assessment(**overrides: object) -> RecoveryAssessment:
    fields: dict = {
        "schema_version": "0.1", "kind": "decision", "decision": "abort",
        "reason": "POLICY_ABORT", "run_id": "r-1", "step_id": "s-1", "contract_id": "c-1",
        "attempt_id": "a-1", "outcome_id": "o-1", "observation_id": None,
    }
    fields.update(overrides)
    return RecoveryAssessment(**fields)


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"run_id": "  "},
    {"run_id": 17},
    {"remaining_budget_ms": -1},
    {"remaining_budget_ms": float("nan")},
    {"remaining_budget_ms": float("inf")},
    {"remaining_budget_ms": True},
    {"remaining_retries": -1},
    {"remaining_retries": True},
    {"remaining_retries": 1.5},
    {"contract_retry_count": "0"},
    {"stop_requested": "yes"},
    {"stop_requested": 0},
    {"unknown_field": 1},
])
def test_an_invalid_context_is_a_facility_error(mutation: dict) -> None:
    fields = context().model_dump(mode="json")
    fields.update(mutation)
    with pytest.raises(RecoveryError) as info:
        RecoveryContext(**fields)
    assert info.value.code == "INPUT_INVALID"
    assert len(str(info.value)) < 240
    assert MARKER not in str(info.value)


@pytest.mark.parametrize("bad", [b"idempotent", b"read_only", 1, None, ["idempotent"],
                                 "safe", "IDEMPOTENT"])
def test_the_repeat_safety_vocabulary_is_closed(bad: object) -> None:
    with pytest.raises(RecoveryError) as info:
        safety(retry_safety=bad)
    assert info.value.code == "INPUT_INVALID"
    for value in ("read_only", "idempotent", "non_idempotent", "unknown"):
        assert safety(retry_safety=value).retry_safety == value


@pytest.mark.parametrize("bad", [b"applied", 1, None, "APPLIED", "maybe"])
def test_the_effect_state_vocabulary_is_closed(bad: object) -> None:
    with pytest.raises(RecoveryError) as info:
        evidence(effect_state=bad)
    assert info.value.code == "EVIDENCE_INVALID"
    for value in ("not_applied", "applied", "unknown"):
        assert evidence(effect_state=value).effect_state == value


def test_a_context_must_carry_a_real_actual_outcome() -> None:
    with pytest.raises(RecoveryError):
        context(actual_outcome={"outcome_id": "o-1"})
    with pytest.raises(RecoveryError):
        context(actual_outcome=None)


def test_evidence_must_carry_a_real_verification_evidence_and_step_result() -> None:
    with pytest.raises(RecoveryError):
        evidence(verification_evidence={"observation": observation().model_dump(mode="json"),
                                        "state_complete": True})
    with pytest.raises(RecoveryError):
        evidence(step_result={"verifier": "json_step"})


# --- A10: the closed verdict vocabulary ---

@pytest.mark.parametrize("kind,decision,reason", [
    ("observation_required", None, "OBSERVATION_REQUIRED"),
    ("no_recovery", None, "STEP_VERIFIED"),
    ("no_recovery", None, "STEP_VERIFIED_AFTER_EXECUTION_ANOMALY"),
    ("decision", "retry", "RETRY_ALLOWED"),
    ("decision", "replan", "REPLAN_ALLOWED"),
    ("decision", "abort", "STOP_REQUESTED"),
    ("decision", "abort", "BUDGET_EXHAUSTED"),
    ("decision", "abort", "POLICY_ABORT"),
    ("decision", "abort", "VERIFICATION_INCONCLUSIVE"),
    ("decision", "abort", "EFFECT_UNRESOLVED"),
    ("decision", "abort", "RECOVERY_NOT_ALLOWED"),
])
def test_every_legal_verdict_line_is_accepted(kind: str, decision: str | None,
                                              reason: str) -> None:
    observation_id = None if kind == "observation_required" else "obs-1"
    built = assessment(kind=kind, decision=decision, reason=reason,
                       observation_id=observation_id)
    assert built.reason == reason


@pytest.mark.parametrize("kind,decision,reason", [
    ("decision", "abort", "RETRY_ALLOWED"),
    ("decision", "retry", "REPLAN_ALLOWED"),
    ("no_recovery", None, "POLICY_ABORT"),
    ("observation_required", None, "STEP_VERIFIED"),
    ("decision", "retry", "STEP_VERIFIED"),
    ("decision", None, "POLICY_ABORT"),
    ("decision", "abort", "OBSERVATION_REQUIRED"),
])
def test_a_self_contradictory_verdict_is_refused(kind: str, decision: str | None,
                                                 reason: str) -> None:
    with pytest.raises(RecoveryError):
        assessment(kind=kind, decision=decision, reason=reason, observation_id=None)


def test_an_observation_request_reports_no_observation_and_a_verified_step_does() -> None:
    with pytest.raises(RecoveryError):
        assessment(kind="observation_required", decision=None, reason="OBSERVATION_REQUIRED",
                   observation_id="obs-1")
    with pytest.raises(RecoveryError):
        assessment(kind="no_recovery", decision=None, reason="STEP_VERIFIED",
                   observation_id=None)
    assert assessment(kind="no_recovery", decision=None, reason="STEP_VERIFIED",
                      observation_id="obs-1").observation_id == "obs-1"


@pytest.mark.parametrize("reason", ["UNKNOWN_REASON", "retry_allowed", "", "STEP_VERIFIED "])
def test_an_unknown_reason_code_is_refused(reason: str) -> None:
    """The closed reason vocabulary is enforced at construction and at assignment."""
    from pydantic import ValidationError

    with pytest.raises(RecoveryError) as info:
        assessment(reason=reason)
    assert info.value.code == "INPUT_INVALID"
    record = assessment(reason="POLICY_ABORT")
    before = record.model_dump(mode="json")
    with pytest.raises((RecoveryError, ValidationError)):
        record.reason = reason
    assert record.model_dump(mode="json") == before
    assert record.reason == "POLICY_ABORT"


def test_the_public_surface_is_exactly_six_names() -> None:
    from forecontract import recovery

    assert sorted(recovery.__all__) == sorted([
        "ActionSafetyDeclaration", "RecoveryAssessment", "RecoveryContext", "RecoveryEngine",
        "RecoveryError", "RecoveryEvidence"])


# --- A01: nested shape, cycles and ownership ---

def test_a_copied_invalid_nested_record_is_refused() -> None:
    """A nested record that no longer matches its own schema is refused, however it was made."""
    for mutation in ({"schema_version": "9.9"}, {"outcome_id": b"o"}, {"status": 17}):
        stored = dict(outcome().__dict__)
        stored.update(mutation)
        forged = ActualOutcome.model_construct(**stored)
        assert forged.__dict__[next(iter(mutation))] != getattr(outcome(), next(iter(mutation)))
        with pytest.raises(RecoveryError):
            context(actual_outcome=forged)
    undeclared = outcome()
    undeclared.__dict__[MARKER] = 1
    with pytest.raises(RecoveryError) as info:
        context(actual_outcome=undeclared)
    assert MARKER not in str(info.value)


def test_a_copied_invalid_nested_step_result_is_refused() -> None:
    stored = dict(step_result().__dict__)
    stored["schema_version"] = "9.9"
    forged = VerificationResult.model_construct(**stored)
    with pytest.raises(RecoveryError) as info:
        evidence(step_result=forged)
    assert info.value.code == "EVIDENCE_INVALID"


def test_a_raw_container_cycle_is_a_bounded_error() -> None:
    cycle: list = []
    cycle.append(cycle)
    with pytest.raises(RecoveryError) as info:
        RecoveryContext.model_validate({"schema_version": "0.1", "run_id": "r-1",
                                        "step_id": "s-1", "contract_id": "c-1",
                                        "attempt_id": "a-1", "actual_outcome": cycle,
                                        "remaining_budget_ms": 1, "remaining_retries": 0,
                                        "remaining_replans": 0, "contract_retry_count": 0,
                                        "stop_requested": False})
    assert info.value.code == "INPUT_INVALID"
    assert len(str(info.value)) < 240


def test_a_legal_shared_reference_is_still_accepted() -> None:
    shared = {"k": [1, 2]}
    first = Observation(schema_version="0.1", observation_id="obs-1",
                        environment_type="fake", state={"a": shared, "b": shared},
                        timestamp=NOW)
    piece = VerificationEvidence(schema_version="0.1", observation=first, state_complete=True)
    record = evidence(verification_evidence=piece)
    assert record.verification_evidence.observation.state["a"] == shared
    assert record.verification_evidence.observation.state["b"] == shared


def test_a_record_owns_its_nested_state_and_does_not_alias_the_caller() -> None:
    caller_state = {"a": {"b": [1]}}
    caller_outcome = ActualOutcome(schema_version="0.1", outcome_id="o-1", status="failed",
                                  changes=caller_state)
    built = context(actual_outcome=caller_outcome)
    caller_state["a"]["b"].append(2)
    assert built.actual_outcome.changes["a"]["b"] == [1]
    built.actual_outcome.changes["a"]["b"].append(3)
    assert caller_state["a"]["b"] == [1, 2]


def test_existing_instance_ingress_returns_an_independent_record() -> None:
    for original in (context(), safety(), evidence(), assessment()):
        before = original.model_dump(mode="json")
        parsed = type(original).model_validate(original)
        assert parsed is not original
        assert parsed == original
        assert original.model_dump(mode="json") == before


def test_a_failed_assignment_preserves_the_previous_valid_record() -> None:
    from pydantic import ValidationError

    record = assessment()
    before = record.model_dump(mode="json")
    with pytest.raises((RecoveryError, ValidationError)):
        record.reason = "UNKNOWN_REASON"
    assert record.model_dump(mode="json") == before

    other = context()
    other_before = other.model_dump(mode="json")
    with pytest.raises((RecoveryError, ValidationError)):
        other.stop_requested = "yes"
    assert other.model_dump(mode="json") == other_before


def test_a_legal_assignment_still_takes_effect() -> None:
    record = context()
    record.stop_requested = True
    assert record.stop_requested is True
    record.remaining_budget_ms = 0
    assert record.remaining_budget_ms == 0


# --- A02: the committed policy shape ---

def test_an_abort_policy_keeps_exactly_its_two_original_fields() -> None:
    policy = RecoveryPolicy(schema_version="0.1", mode="abort")
    assert policy.model_dump() == {"schema_version": "0.1", "mode": "abort"}
    assert policy.model_dump(mode="json") == {"schema_version": "0.1", "mode": "abort"}
    assert json.loads(policy.model_dump_json()) == {"schema_version": "0.1", "mode": "abort"}
    assert RecoveryPolicy(schema_version="0.1").model_dump() == {"schema_version": "0.1",
                                                                 "mode": "abort"}


def test_a_bounded_policy_requires_explicit_controls() -> None:
    policy = RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=0,
                            allow_replan=False)
    assert policy.model_dump() == {"schema_version": "0.1", "mode": "bounded",
                                   "max_retries": 0, "allow_replan": False}
    assert policy.max_retries == 0


@pytest.mark.parametrize("kwargs", [
    {"mode": "bounded"},
    {"mode": "bounded", "max_retries": 1},
    {"mode": "bounded", "allow_replan": True},
    {"mode": "abort", "max_retries": 1},
    {"mode": "abort", "allow_replan": True},
    {"mode": "retry"},
    {"mode": "replan"},
    {"mode": "bounded", "max_retries": True, "allow_replan": True},
    {"mode": "bounded", "max_retries": 1.5, "allow_replan": True},
    {"mode": "bounded", "max_retries": "1", "allow_replan": True},
    {"mode": "bounded", "max_retries": -1, "allow_replan": True},
    {"mode": "bounded", "max_retries": 0, "allow_replan": "yes"},
    {"mode": "bounded", "max_retries": 0, "allow_replan": 0},
])
def test_an_invalid_policy_is_refused(kwargs: dict) -> None:
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        RecoveryPolicy(schema_version="0.1", **kwargs)


# --- round-002 corrections: F02 explicit bounded controls, F03 roundtrip, F04 strict enum ---

@pytest.mark.parametrize("field", ["max_retries", "allow_replan"])
@pytest.mark.parametrize("mode", ["constructor", "mapping", "json"])
def test_an_abort_policy_refuses_an_explicitly_supplied_null_control(field: str,
                                                                    mode: str) -> None:
    """F02: an abort policy refuses a bounded control that was supplied, even as null."""
    from pydantic import ValidationError

    fields = {"schema_version": "0.1", "mode": "abort", field: None}
    with pytest.raises((ValidationError, ValueError)):
        if mode == "constructor":
            RecoveryPolicy(**fields)
        elif mode == "mapping":
            RecoveryPolicy.model_validate(fields)
        else:
            RecoveryPolicy.model_validate_json(json.dumps(fields))
    # Omission stays legal, and the committed shape stays exactly two fields.
    omitted = RecoveryPolicy(schema_version="0.1", mode="abort")
    assert omitted.model_dump() == {"schema_version": "0.1", "mode": "abort"}
    assert json.loads(omitted.model_dump_json()) == {"schema_version": "0.1", "mode": "abort"}


def test_a_bounded_policy_still_requires_both_controls_explicitly() -> None:
    """F02 control: bounded keeps requiring its controls, including a legal explicit zero."""
    from pydantic import ValidationError

    for fields in ({"mode": "bounded"}, {"mode": "bounded", "max_retries": 0},
                   {"mode": "bounded", "allow_replan": True}):
        with pytest.raises((ValidationError, ValueError)):
            RecoveryPolicy(schema_version="0.1", **fields)
    legal = RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=0,
                           allow_replan=False)
    assert legal.model_dump() == {"schema_version": "0.1", "mode": "bounded", "max_retries": 0,
                                  "allow_replan": False}


def test_valid_evidence_roundtrips_through_mapping_and_json() -> None:
    """F03: a legal record survives its own serialized forms unchanged."""
    original = evidence()
    assert RecoveryEvidence.model_validate(original.model_dump(mode="json")) == original
    assert RecoveryEvidence.model_validate_json(original.model_dump_json()) == original
    assert RecoveryEvidence.model_validate(original.model_dump(mode="python")) == original
    assert RecoveryEvidence.model_validate(original) == original
    # The rebuilt record owns its nested observation rather than aliasing the source.
    rebuilt = RecoveryEvidence.model_validate(original.model_dump(mode="json"))
    assert rebuilt.verification_evidence is not original.verification_evidence


def test_a_damaged_serialized_evidence_record_is_still_refused() -> None:
    """F03 control: reconstruction validates the original structure instead of washing it."""
    document = json.loads(evidence().model_dump_json())
    document["verification_evidence"]["observation"]["schema_version"] = "9.9"
    with pytest.raises(RecoveryError):
        RecoveryEvidence.model_validate(document)
    broken = json.loads(evidence().model_dump_json())
    broken["verification_evidence"]["unknown_field"] = 1
    with pytest.raises(RecoveryError):
        RecoveryEvidence.model_validate(broken)
    missing = json.loads(evidence().model_dump_json())
    del missing["verification_evidence"]
    with pytest.raises(RecoveryError):
        RecoveryEvidence.model_validate(missing)


@pytest.mark.parametrize("route", ["constructor", "mapping", "instance", "json", "assignment"])
def test_bytes_can_never_become_a_preconditions_verdict(route: str) -> None:
    """F04: every local enum boundary refuses bytes instead of decoding them."""
    from pydantic import ValidationError

    record = evidence()
    if route == "constructor":
        with pytest.raises((RecoveryError, ValidationError)):
            RecoveryEvidence(**{**dict(record.__dict__), "preconditions_status": b"pass"})
    elif route == "mapping":
        with pytest.raises((RecoveryError, ValidationError)):
            RecoveryEvidence.model_validate({**record.model_dump(mode="python"),
                                             "preconditions_status": b"pass"})
    elif route == "instance":
        forged = RecoveryEvidence.model_construct(
            **{**dict(record.__dict__), "preconditions_status": b"pass"})
        with pytest.raises((RecoveryError, ValidationError)):
            RecoveryEvidence.model_validate(forged)
    elif route == "json":
        document = json.loads(record.model_dump_json())
        document["preconditions_status"] = "pass"
        assert RecoveryEvidence.model_validate_json(json.dumps(document)).preconditions_status \
            is VerificationStatus.PASS
    else:
        target = RecoveryEvidence.model_validate(record)
        before = target.model_dump(mode="json")
        with pytest.raises((RecoveryError, ValidationError)):
            target.preconditions_status = b"pass"
        assert target.model_dump(mode="json") == before
    if route != "json":
        for value, expected in (("pass", VerificationStatus.PASS),
                                ("fail", VerificationStatus.FAIL),
                                ("inconclusive", VerificationStatus.INCONCLUSIVE),
                                (VerificationStatus.PASS, VerificationStatus.PASS)):
            assert evidence(preconditions_status=value).preconditions_status is expected


def test_the_other_local_enums_also_refuse_bytes() -> None:
    """F04 control: the same names-only ingress covers the effect and safety vocabularies."""
    with pytest.raises(RecoveryError):
        evidence(effect_state=b"applied")
    with pytest.raises(RecoveryError):
        safety(retry_safety=b"idempotent")
    with pytest.raises(RecoveryError):
        assessment(kind=b"decision")


# --- round-003 corrections: F07 arbitrary-length cycles across models and containers ---

def _linked_outcomes(size: int) -> ActualOutcome:
    """A chain of ``size`` outcome records whose last one points back at the first."""
    nodes = [ActualOutcome.model_construct(**dict(outcome().__dict__)) for _ in range(size)]
    for index, node in enumerate(nodes):
        node.__dict__["changes"] = {"next": nodes[(index + 1) % size]}
    return nodes[0]


@pytest.mark.parametrize("size", [2, 3, 5, 8])
@pytest.mark.parametrize("route", ["constructor", "mapping", "instance", "assignment"])
def test_an_arbitrary_length_model_cycle_is_a_bounded_error(size: int, route: str) -> None:
    """F07: a cycle of any length across model boundaries is refused before it recurses."""
    from pydantic import ValidationError

    data = {**dict(context().__dict__), "actual_outcome": _linked_outcomes(size)}
    with pytest.raises((RecoveryError, ValidationError)) as info:
        if route == "constructor":
            RecoveryContext(**data)
        elif route == "mapping":
            RecoveryContext.model_validate(data)
        elif route == "instance":
            RecoveryContext.model_validate(RecoveryContext.model_construct(**data))
        else:
            target = context()
            target.actual_outcome = _linked_outcomes(size)
    message = str(info.value)
    assert "RecursionError" not in message
    assert "maximum recursion depth" not in message


@pytest.mark.parametrize("size", [2, 3, 5])
def test_a_mixed_model_and_container_cycle_is_a_bounded_error(size: int) -> None:
    """F07: the same bound holds when the cycle alternates models and raw containers."""
    first = ActualOutcome.model_construct(**dict(outcome().__dict__))
    second = ActualOutcome.model_construct(**dict(outcome().__dict__))
    first.__dict__["changes"] = {"a": [{"b": second}]}
    second.__dict__["changes"] = {"c": (first,)}
    with pytest.raises(RecoveryError) as info:
        context(actual_outcome=first)
    assert info.value.code == "INPUT_INVALID"
    assert len(str(info.value)) < 240
    assert size


def test_a_legal_shared_reference_is_still_accepted_after_the_cycle_rule() -> None:
    """F07 control: a JSON value shared between sibling fields is not a cycle."""
    shared = {"leaf": [1]}
    parent = ActualOutcome.model_construct(**{**dict(outcome().__dict__),
                                              "changes": {"a": shared, "b": shared}})
    rebuilt = context(actual_outcome=parent)
    assert rebuilt.actual_outcome.changes["a"] == {"leaf": [1]}
    assert rebuilt.actual_outcome.changes["b"] == {"leaf": [1]}
    shared["leaf"].append(2)
    assert rebuilt.actual_outcome.changes["a"] == {"leaf": [1]}


def test_a_deeply_shared_container_is_still_accepted() -> None:
    """F07 control: a JSON container referenced at two depths is copied per reference."""
    shared = {"leaf": [1]}
    parent = ActualOutcome.model_construct(**{**dict(outcome().__dict__),
                                              "changes": {"a": {"deep": [shared]},
                                                          "b": shared}})
    rebuilt = context(actual_outcome=parent)
    assert rebuilt.actual_outcome.changes["b"] == {"leaf": [1]}
    assert rebuilt.actual_outcome.changes["a"]["deep"] == [{"leaf": [1]}]
    shared["leaf"].append(2)
    assert rebuilt.actual_outcome.changes["b"] == {"leaf": [1]}
