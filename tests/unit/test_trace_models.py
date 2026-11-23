"""Acceptance tests for the versioned structured-tracing records.

The suite drives the real :mod:`forecontract.tracing` surface from the package source tree. It
asserts values, identities and order rather than the mere absence of exceptions: a rejected
record is checked through its error class, and an accepted record is checked through its fields
and its round-trip.

No Runtime, Memory, network, live environment or repository build output is involved. All
fixtures are synthetic and contain no real credential, path or governance content.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.models.outcomes import ExecutionStatus, VerificationStatus
from forecontract.tracing import (
    ATTEMPT_RESULT_LABELS,
    ATTEMPT_SOURCE_LEVEL,
    CLOSED_CODES,
    EVENT_KINDS,
    EVENT_MODELS,
    RUN_SOURCE_LEVEL,
    SOURCE_LEVELS,
    TERMINAL_RESULT_LABELS,
    TERMINAL_STATUSES,
    AttemptFinished,
    AttemptRef,
    AttemptStarted,
    ContractCommitted,
    RunFinished,
    RunStarted,
    StepStarted,
    TraceValidationError,
    VerificationRecorded,
    attempt_label,
    attempt_ref,
    parse_record,
    source_location,
    terminal_label,
)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
ISO = "2026-10-08T12:00:00Z"

ENVELOPE = {
    "schema_version": "0.1",
    "record_id": "record-1",
    "run_id": "run-1",
    "sequence": 0,
    "occurred_at": ISO,
}


def envelope(**overrides) -> dict:
    payload = dict(ENVELOPE)
    payload.update(overrides)
    return payload


def _payload(defaults: dict, overrides: dict) -> dict:
    payload = dict(ENVELOPE)
    payload.update(defaults)
    payload.update(overrides)
    return payload


def run_started(**overrides) -> RunStarted:
    return RunStarted(**_payload({"kind": "run_started"}, overrides))


def step_started(**overrides) -> StepStarted:
    return StepStarted(**_payload({"kind": "step_started", "step_id": "step-1"}, overrides))


def contract_committed(**overrides) -> ContractCommitted:
    return ContractCommitted(
        **_payload({"kind": "contract_committed", "step_id": "step-1", "contract_id": "contract-1"}, overrides)
    )


def attempt_started(**overrides) -> AttemptStarted:
    defaults = {
        "kind": "attempt_started", "step_id": "step-1", "contract_id": "contract-1",
        "attempt_id": "attempt-1",
    }
    return AttemptStarted(**_payload(defaults, overrides))


def attempt_finished(**overrides) -> AttemptFinished:
    defaults = {
        "kind": "attempt_finished", "step_id": "step-1", "contract_id": "contract-1",
        "attempt_id": "attempt-1", "execution_status": ExecutionStatus.SUCCEEDED,
    }
    return AttemptFinished(**_payload(defaults, overrides))


def verification_recorded(**overrides) -> VerificationRecorded:
    defaults = {
        "kind": "verification_recorded", "scope": "run",
        "verification_status": VerificationStatus.PASS,
    }
    return VerificationRecorded(**_payload(defaults, overrides))


def run_finished(**overrides) -> RunFinished:
    return RunFinished(**_payload({"kind": "run_finished", "terminal_status": "failure"}, overrides))


# --- closed variant set and envelope -----------------------------------------

def test_the_event_kind_set_is_closed_and_every_variant_has_a_model() -> None:
    assert list(EVENT_MODELS) == list(EVENT_KINDS)
    assert len(EVENT_KINDS) == 7
    for kind, model in EVENT_MODELS.items():
        assert model.model_fields["kind"].default == kind


def test_every_variant_round_trips_through_json_text() -> None:
    records = [
        run_started(),
        step_started(sequence=1),
        contract_committed(sequence=2),
        attempt_started(sequence=3),
        attempt_finished(sequence=4),
        verification_recorded(sequence=5, scope="attempt", step_id="step-1",
                              contract_id="contract-1", attempt_id="attempt-1"),
        run_finished(sequence=6),
    ]
    for record in records:
        text = record.model_dump_json()
        assert parse_record(json.loads(text)) == record
        assert type(parse_record(json.loads(text))) is type(record)


def test_a_record_from_json_text_keeps_its_source_identity() -> None:
    record = attempt_finished(record_id="attempt-fact", attempt_id="attempt-7", sequence=9)
    reloaded = parse_record(json.loads(record.model_dump_json()))
    assert reloaded.record_id == "attempt-fact"
    assert reloaded.attempt_id == "attempt-7"
    assert reloaded.sequence == 9
    assert reloaded.occurred_at == NOW
    assert isinstance(reloaded.occurred_at, datetime)
    assert reloaded.execution_status is ExecutionStatus.SUCCEEDED


def test_the_public_record_alias_covers_every_variant() -> None:
    samples = [
        run_started(sequence=0),
        step_started(sequence=0),
        contract_committed(sequence=0),
        attempt_started(sequence=0),
        attempt_finished(sequence=0),
        verification_recorded(sequence=0),
        run_finished(sequence=0, terminal_status="failure"),
    ]
    for sample in samples:
        assert isinstance(sample, BaseModel)
        assert type(sample) is EVENT_MODELS[sample.kind]
        assert parse_record(json.loads(sample.model_dump_json())) == sample


# --- strict typing and closed schema -----------------------------------------

@pytest.mark.parametrize("kind", ["run_ended", "RUN_STARTED", "attempt", ""])
def test_an_unknown_event_kind_is_rejected(kind: str) -> None:
    with pytest.raises(TraceValidationError):
        parse_record(envelope(kind=kind))


def test_an_extra_field_is_rejected_on_every_variant() -> None:
    for kind in EVENT_KINDS:
        payload = envelope(kind=kind)
        payload["unapproved_field"] = "must not be accepted"
        with pytest.raises(TraceValidationError):
            parse_record(payload)


def test_an_unknown_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunStarted(**envelope(kind="run_started", schema_version="0.2"))
    with pytest.raises(ValidationError):
        RunStarted(**envelope(kind="run_started", schema_version="1"))


@pytest.mark.parametrize("field,value", [
    ("record_id", 1),
    ("record_id", True),
    ("record_id", "   "),
    ("run_id", 7),
    ("run_id", ""),
    ("sequence", True),
    ("sequence", "0"),
    ("sequence", -1),
    ("sequence", 1.0),
    ("occurred_at", "not-a-time"),
    ("occurred_at", "2026-10-08T12:00:00"),
])
def test_a_malformed_envelope_field_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RunStarted(**envelope(kind="run_started", **{field: value}))


def test_a_tuple_is_not_accepted_where_a_json_value_is_expected() -> None:
    """A tuple is not a JSON array; the contract rejects it instead of normalising it."""
    with pytest.raises(ValidationError):
        RunStarted(**envelope(kind="run_started", record_id=("a",)))


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_a_non_finite_timestamp_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        RunStarted(**envelope(kind="run_started", occurred_at=float(value)))


def test_a_naive_timestamp_is_rejected() -> None:
    # A naive datetime is exactly what must be refused, so building one is the point here.
    naive = datetime(2026, 10, 8, 12, 0)  # noqa: DTZ001
    with pytest.raises(ValidationError):
        RunStarted(**envelope(kind="run_started", occurred_at=naive))


def test_records_are_frozen_and_reject_assignment() -> None:
    record = run_started()
    with pytest.raises(ValidationError):
        record.sequence = 4


# --- per-variant required fields ---------------------------------------------

@pytest.mark.parametrize("payload,missing", [
    ({"kind": "step_started"}, "step_id"),
    ({"kind": "contract_committed", "step_id": "step-1"}, "contract_id"),
    ({"kind": "attempt_started", "step_id": "step-1", "contract_id": "contract-1"}, "attempt_id"),
    ({"kind": "attempt_finished", "step_id": "s", "contract_id": "c", "attempt_id": "a"}, "execution_status"),
    ({"kind": "verification_recorded", "scope": "run"}, "verification_status"),
    ({"kind": "run_finished"}, "terminal_status"),
])
def test_a_missing_variant_field_is_rejected(payload: dict, missing: str) -> None:
    with pytest.raises(TraceValidationError):
        parse_record(envelope(**payload))


def test_an_unknown_execution_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        attempt_finished(execution_status="maybe")


def test_an_unknown_verification_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        verification_recorded(verification_status="probably")


@pytest.mark.parametrize("status", ["unknown", "running", "SUCCESS", ""])
def test_an_unknown_terminal_status_is_rejected(status: str) -> None:
    with pytest.raises(ValidationError):
        run_finished(terminal_status=status)


@pytest.mark.parametrize("scope", ["global", "ATTEMPT", ""])
def test_an_unknown_verification_scope_is_rejected(scope: str) -> None:
    with pytest.raises(ValidationError):
        verification_recorded(scope=scope)


# --- closed safe codes --------------------------------------------------------

def test_the_closed_code_set_is_the_approved_one() -> None:
    assert set(CLOSED_CODES) == {
        "execution_failed", "execution_unknown", "verification_failed",
        "verification_inconclusive", "user_cancelled", "deadline_exceeded",
        "policy_abort", "internal_error",
    }
    assert len(CLOSED_CODES) == len(set(CLOSED_CODES)) == 8


@pytest.mark.parametrize("code", sorted(CLOSED_CODES))
def test_only_closed_codes_are_accepted(code: str) -> None:
    finished = attempt_finished(error_code=code)
    assert finished.error_code == code
    assert run_finished(reason_code=code).reason_code == code


@pytest.mark.parametrize("code", ["boom", "executionfailed", "user cancelled", "EXECUTION_FAILED", "raw text"])
def test_a_free_text_code_is_rejected(code: str) -> None:
    with pytest.raises(ValidationError):
        attempt_finished(error_code=code)
    with pytest.raises(ValidationError):
        run_finished(reason_code=code)


def test_optional_codes_default_to_none_and_are_not_invented() -> None:
    assert attempt_finished().error_code is None
    assert run_finished().reason_code is None
    assert run_finished().verification_record_id is None


# --- attempt reference key and source levels ---------------------------------

def test_attempt_ref_is_the_four_part_experience_key() -> None:
    record = attempt_started(attempt_id="attempt-9")
    assert attempt_ref(record) == AttemptRef("run-1", "step-1", "contract-1", "attempt-9")
    assert attempt_ref(attempt_finished(attempt_id="attempt-9")) == AttemptRef(
        "run-1", "step-1", "contract-1", "attempt-9"
    )
    attempt_verification = verification_recorded(
        scope="attempt", step_id="step-1", contract_id="contract-1", attempt_id="attempt-9"
    )
    assert attempt_ref(attempt_verification) == AttemptRef("run-1", "step-1", "contract-1", "attempt-9")


@pytest.mark.parametrize("record", [run_started(), step_started(), run_finished()])
def test_a_run_level_record_has_no_attempt_key(record) -> None:
    with pytest.raises(ValueError):
        attempt_ref(record)


def test_a_run_scoped_verification_has_no_attempt_key() -> None:
    with pytest.raises(ValueError):
        attempt_ref(verification_recorded(scope="run"))


def test_the_two_source_levels_have_different_location_shapes() -> None:
    assert SOURCE_LEVELS == (ATTEMPT_SOURCE_LEVEL, RUN_SOURCE_LEVEL)
    attempt_location = source_location(attempt_started(attempt_id="attempt-3"))
    run_location = source_location(run_started(record_id="record-3"))
    assert attempt_location == ("run-1", "step-1", "contract-1", "attempt-3")
    assert run_location == ("run-1", "record-3")
    assert len(attempt_location) == 4 and len(run_location) == 2
    assert attempt_location != run_location


# --- explicit result mappings -------------------------------------------------

def test_attempt_status_mapping_is_the_approved_one() -> None:
    assert ATTEMPT_RESULT_LABELS == {"succeeded": "success", "failed": "failure", "unknown": "unknown"}
    assert attempt_label(ExecutionStatus.SUCCEEDED) == "success"
    assert attempt_label(ExecutionStatus.FAILED) == "failure"
    assert attempt_label(ExecutionStatus.UNKNOWN) == "unknown"


def test_terminal_status_mapping_is_one_to_one_and_closed() -> None:
    assert TERMINAL_STATUSES == ("success", "failure", "cancelled", "timeout", "aborted")
    assert TERMINAL_RESULT_LABELS == {
        "success": "success", "failure": "failure", "cancelled": "cancelled",
        "timeout": "timeout", "aborted": "aborted",
    }
    for status in TERMINAL_STATUSES:
        assert terminal_label(status) == status


def test_a_verification_record_is_never_an_attempt_result() -> None:
    """The verification status has its own namespace and cannot label an attempt outcome."""
    assert set(ATTEMPT_RESULT_LABELS.values()) == {"success", "failure", "unknown"}
    assert "pass" not in ATTEMPT_RESULT_LABELS
    assert "inconclusive" not in ATTEMPT_RESULT_LABELS


# --- scope-specific verification rules ---------------------------------------

def test_an_attempt_scoped_verification_requires_all_three_references() -> None:
    with pytest.raises(ValidationError):
        verification_recorded(scope="attempt")
    with pytest.raises(ValidationError):
        verification_recorded(scope="attempt", step_id="step-1", contract_id="contract-1")
    granted = verification_recorded(
        scope="attempt", step_id="step-1", contract_id="contract-1", attempt_id="attempt-1"
    )
    assert granted.scope == "attempt" and granted.attempt_id == "attempt-1"


@pytest.mark.parametrize("field", ["step_id", "contract_id", "attempt_id"])
def test_a_run_scoped_verification_must_not_carry_an_attempt_reference(field: str) -> None:
    with pytest.raises(ValidationError):
        verification_recorded(scope="run", **{field: "value-1"})


def test_a_successful_terminal_fact_requires_a_verification_reference() -> None:
    with pytest.raises(ValidationError):
        run_finished(terminal_status="success")
    granted = run_finished(terminal_status="success", verification_record_id="verification-1")
    assert granted.verification_record_id == "verification-1"


@pytest.mark.parametrize("status", ["failure", "cancelled", "timeout", "aborted"])
def test_a_non_success_terminal_fact_may_omit_the_reference(status: str) -> None:
    assert run_finished(terminal_status=status).terminal_status == status


# --- no raw payload surface ---------------------------------------------------

@pytest.mark.parametrize("field", [
    "metadata", "message", "error_message", "artifact_uri", "url", "observations",
    "actual_outcome", "state_after", "prompt", "reasoning", "password", "api_key",
])
def test_no_raw_payload_field_is_accepted(field: str) -> None:
    with pytest.raises(TraceValidationError):
        parse_record(envelope(kind="run_finished", terminal_status="failure", **{field: "value"}))


def test_a_rejected_record_error_never_transcribes_the_rejected_value() -> None:
    marker = "SYNTHETIC-MARKER-NEVER-ECHOED"
    try:
        parse_record(envelope(kind="run_finished", terminal_status="failure", metadata=marker))
    except TraceValidationError as exc:
        assert marker not in str(exc)
    else:  # pragma: no cover - the record must be rejected
        raise AssertionError("a raw metadata field must not be accepted")


# --- round-003 correction: F03 ingress revalidation, F04 bounded messages -----

def test_a_read_back_record_is_revalidated_from_its_detached_raw_fields() -> None:
    """F03: ingress must not trust class identity; the detached raw state decides."""
    state = run_started().model_dump(mode="python", warnings=False)
    assert state["schema_version"] == "0.1"
    for forged in (
        {**state, "sequence": True},
        {**state, "schema_version": "9.9"},
        {**state, "kind": "run_ended"},
        {**state, "record_id": 7},
    ):
        with pytest.raises(ValidationError):
            RunStarted.model_validate(forged)
        with pytest.raises(TraceValidationError):
            parse_record(forged)


def test_a_schema_rejection_keeps_a_safe_field_location_and_a_useful_category() -> None:
    """F04: a declared field location is still reported, with a bounded category."""
    with pytest.raises(ValidationError) as info:
        RunStarted(**envelope(kind="run_started", sequence=True))
    rendered = str(info.value)
    assert "sequence" in rendered, rendered
    assert "cross-field rule" in rendered or "integer" in rendered

    with pytest.raises(ValidationError) as kind_info:
        RunStarted(**envelope(kind="run_ended"))
    kind_rendered = str(kind_info.value)
    assert "kind" in kind_rendered, kind_rendered
    assert "run_ended" not in kind_rendered, kind_rendered


def test_a_declared_field_location_is_reported_by_parse_record() -> None:
    """F04: the bounded message keeps a declared location, so a caller can still locate it."""
    with pytest.raises(TraceValidationError) as info:
        parse_record(envelope(kind="run_started", sequence="not-an-integer"))
    message = str(info.value)
    assert message.startswith("invalid record at run_started.sequence: "), message


@pytest.mark.parametrize("field", [
    "SYNTHETIC_PRIVATE_FIELD", "metadata", "api_key", "prompt",
])
def test_an_attacker_controlled_field_name_is_never_echoed(field: str) -> None:
    """F04: an unknown key must not be transcribed by any public rejection."""
    marker_value = "SYNTHETIC_PRIVATE_VALUE"
    payload = envelope(kind="run_started", **{field: marker_value})
    with pytest.raises(TraceValidationError) as info:
        parse_record(payload)
    message = str(info.value)
    assert field not in message, message
    assert marker_value not in message, message
    assert "invalid record at " in message


def test_a_rejected_value_is_never_echoed_by_the_model_constructor() -> None:
    """F04: the public model rejection hides the rejected input value as well."""
    with pytest.raises(ValidationError) as info:
        RunStarted(**envelope(kind="run_started", sequence="SYNTHETIC_PRIVATE_VALUE"))
    rendered = str(info.value)
    assert "SYNTHETIC_PRIVATE_VALUE" not in rendered
    assert "SYNTHETIC_PRIVATE" not in rendered
    assert info.value.errors()[0]["loc"] == ("sequence",)


def test_a_declared_field_name_is_kept_while_an_unknown_one_is_replaced() -> None:
    """F04: a declared location stays useful, an undeclared key becomes a placeholder."""
    with pytest.raises(ValidationError) as declared:
        RunStarted(**envelope(kind="run_started", sequence="not-an-integer"))
    assert declared.value.errors()[0]["loc"] == ("sequence",)

    with pytest.raises(ValidationError) as undeclared:
        RunStarted(**envelope(kind="run_started", metadata="value-1"))
    assert undeclared.value.errors()[0]["loc"] == ("<field>",)


def test_a_rejected_rejection_carries_no_rejected_input_in_its_error_list() -> None:
    """F04: nothing in the structured error list holds the attacker-supplied value."""
    with pytest.raises(ValidationError) as info:
        RunStarted(**envelope(kind="run_started", metadata="SYNTHETIC_PRIVATE_VALUE"))
    for error in info.value.errors():
        assert error.get("input") in (None, ""), error
        assert "SYNTHETIC_PRIVATE" not in str(error)


def test_the_rebuilt_validation_error_hides_the_key_in_every_public_rendering() -> None:
    """F04: str, json and errors() of a rebuilt rejection all lose the attacker key."""
    marker = "SYNTHETIC_PRIVATE_FIELD"
    with pytest.raises(ValidationError) as info:
        RunStarted(**envelope(kind="run_started", **{marker: 1}))
    rendered = str(info.value)
    assert marker not in rendered, rendered
    assert marker not in info.value.json(), info.value.json()
    assert marker not in str(info.value.errors()), info.value.errors()
    assert info.value.errors()[0]["loc"] == ("<field>",)
