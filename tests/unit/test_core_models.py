"""Tests for the versioned Core data models.

The models are imported from the source tree, which is permitted for unit tests; it is not
installation evidence (the package-boundary suite owns build and install acceptance).

Round-002 corrections (F01-F03): JSON payload failures report the offending nested field or
index, model list fields reject non-list containers and enum fields reject unrelated input
types, and a rejected field assignment leaves the previous valid state intact.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.models import (
    ActionSpec,
    ActualOutcome,
    AdapterCapabilities,
    ArtifactRef,
    ErrorInfo,
    ExecutionStatus,
    ExpectedOutcome,
    Forecast,
    JsonObject,
    JsonValue,
    Observation,
    OutcomeVerification,
    Precondition,
    VerificationResult,
    VerificationStatus,
)

UTC = dt.timezone.utc
TIMESTAMP = dt.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def artifact(**overrides: object) -> ArtifactRef:
    data: dict[str, object] = {
        "schema_version": "0.1", "artifact_id": "artifact-1", "artifact_type": "file",
        "uri": "file:///tmp/artifact-1",
    }
    data.update(overrides)
    return ArtifactRef(**data)


def observation(**overrides: object) -> Observation:
    data: dict[str, object] = {
        "schema_version": "0.1", "observation_id": "observation-1", "environment_type": "local_test_site",
        "state": {"url": "https://example.invalid", "count": 2, "ratio": 0.5, "flag": True, "none": None},
        "timestamp": TIMESTAMP, "artifacts": [artifact()], "metadata": {"source": "unit-test"},
    }
    data.update(overrides)
    return Observation(**data)


def action_spec(**overrides: object) -> ActionSpec:
    data: dict[str, object] = {
        "schema_version": "0.1", "action_type": "click", "payload": {"selector": "#submit"},
    }
    data.update(overrides)
    return ActionSpec(**data)


def forecast(**overrides: object) -> Forecast:
    data: dict[str, object] = {
        "schema_version": "0.1", "forecast_id": "forecast-1", "proposed_action": action_spec(),
        "expected_outcomes": [ExpectedOutcome(schema_version="0.1", outcome_type="page_contains")],
        "intent": "submit the form", "confidence": 0.5, "task_completion_hint": False,
    }
    data.update(overrides)
    return Forecast(**data)


def actual_outcome(**overrides: object) -> ActualOutcome:
    data: dict[str, object] = {
        "schema_version": "0.1", "outcome_id": "outcome-1", "status": ExecutionStatus.SUCCEEDED,
        "changes": {"url": "https://example.invalid/done"}, "duration_ms": 12.5,
    }
    data.update(overrides)
    return ActualOutcome(**data)


def outcome_verification(**overrides: object) -> OutcomeVerification:
    data: dict[str, object] = {
        "schema_version": "0.1", "outcome_index": 0, "status": VerificationStatus.PASS,
        "has_actual_value": True, "actual_value": {"text": "done"},
    }
    data.update(overrides)
    return OutcomeVerification(**data)


def verification_result(**overrides: object) -> VerificationResult:
    data: dict[str, object] = {
        "schema_version": "0.1", "status": VerificationStatus.PASS, "verifier": "unit-test-verifier",
        "outcome_results": [outcome_verification()],
    }
    data.update(overrides)
    return VerificationResult(**data)


def capabilities(**overrides: object) -> AdapterCapabilities:
    data: dict[str, object] = {
        "schema_version": "0.1", "adapter_name": "local-test-adapter", "environment_type": "local_test_site",
        "supported_actions": ["observe", "click"], "supported_observation_types": ["page"],
        "supported_condition_types": ["element_visible"], "supported_outcome_types": ["page_contains"],
    }
    data.update(overrides)
    return AdapterCapabilities(**data)


PUBLIC_MODELS = (
    ArtifactRef, ErrorInfo, Observation, ActionSpec, Precondition, ExpectedOutcome, Forecast,
    ActualOutcome, OutcomeVerification, VerificationResult, AdapterCapabilities,
)
SAMPLES = {
    "ArtifactRef": artifact, "ErrorInfo": lambda: ErrorInfo(schema_version="0.1", code="E1", message="boom"),
    "Observation": observation, "ActionSpec": action_spec,
    "Precondition": lambda: Precondition(schema_version="0.1", condition_type="element_visible"),
    "ExpectedOutcome": lambda: ExpectedOutcome(schema_version="0.1", outcome_type="page_contains"),
    "Forecast": forecast, "ActualOutcome": actual_outcome, "OutcomeVerification": outcome_verification,
    "VerificationResult": verification_result, "AdapterCapabilities": capabilities,
}


def assert_roundtrips(model: object) -> None:
    model_type = type(model)
    assert model_type.model_validate_json(model.model_dump_json()) == model
    assert model_type.model_validate(model.model_dump(mode="json")) == model


# The JSON schema is a recursive tagged union, so an error location interleaves the branch
# labels with the offending keys and indices. ``_json_path`` drops the labels; the fixtures
# below deliberately avoid key names that collide with them.
JSON_BRANCH_LABELS = frozenset({"null", "bool", "int", "float", "str", "list", "dict", "invalid"})


def json_path(error: ValidationError) -> tuple:
    """The offending JSON field/index path reported for the first error."""
    return tuple(part for part in error.errors()[0]["loc"] if part not in JSON_BRANCH_LABELS)


# --- positive contract -------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_public_model_supports_both_roundtrips(name: str) -> None:
    assert_roundtrips(SAMPLES[name]())


@pytest.mark.parametrize("sample", sorted(SAMPLES))
def test_public_models_forbid_extra_fields_and_validate_assignment(sample: str) -> None:
    model_type = SAMPLES[sample]().__class__
    assert model_type.model_config["extra"] == "forbid"
    assert model_type.model_config["validate_assignment"] is True


def test_precondition_and_expected_outcome_defaults() -> None:
    precondition = Precondition(schema_version="0.1", condition_type="element_visible")
    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="page_contains")
    assert precondition.payload == {} and precondition.required is True
    assert outcome.payload == {} and outcome.required is True
    assert_roundtrips(precondition)
    assert_roundtrips(outcome)


def test_forecast_expected_outcomes_required_but_may_be_empty() -> None:
    empty = forecast(expected_outcomes=[])
    assert empty.expected_outcomes == []
    assert_roundtrips(empty)
    data = {"schema_version": "0.1", "forecast_id": "f", "proposed_action": action_spec()}
    with pytest.raises(ValidationError) as error:
        Forecast(**data)
    assert error.value.errors()[0]["loc"] == ("expected_outcomes",)


def test_capability_lists_may_be_empty_and_preserve_order() -> None:
    minimal = capabilities(supported_observation_types=[], supported_condition_types=[],
                           supported_outcome_types=[], supported_actions=[])
    assert minimal.supported_actions == []
    ordered = capabilities(supported_actions=["b", "a", "c"])
    assert ordered.supported_actions == ["b", "a", "c"]
    assert_roundtrips(ordered)


def test_independent_defaults_are_factories() -> None:
    first, second = observation(), observation()
    assert first.metadata is not second.metadata
    assert first.artifacts is not second.artifacts
    first.metadata["mutated"] = True
    assert second.metadata == {"source": "unit-test"}
    result_a, result_b = verification_result(), verification_result()
    assert result_a.outcome_results is not result_b.outcome_results
    actual_a, actual_b = actual_outcome(), actual_outcome()
    assert actual_a.changes is not actual_b.changes
    assert ErrorInfo(schema_version="0.1", code="E", message="m").details == {}
    assert action_spec().metadata == {}


# --- schema version ----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_schema_version_is_required_for_every_model(name: str) -> None:
    model_type = SAMPLES[name]().__class__
    payload = SAMPLES[name]().model_dump(mode="json")
    payload.pop("schema_version")
    with pytest.raises(ValidationError) as error:
        model_type.model_validate(payload)
    assert error.value.errors()[0]["loc"] == ("schema_version",)
    assert error.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize("version", ["0.2", "0.10", "1.0", "", None, 0.1, True, ["0.1"]])
def test_unknown_schema_version_is_rejected(version: object) -> None:
    with pytest.raises(ValidationError) as error:
        ArtifactRef.model_validate({
            "schema_version": version, "artifact_id": "a", "artifact_type": "file", "uri": "u",
        })
    assert error.value.errors()[0]["loc"] == ("schema_version",)


def test_nested_schema_version_is_required_and_located() -> None:
    with pytest.raises(ValidationError) as nested:
        observation(artifacts=[{"artifact_id": "a", "artifact_type": "file", "uri": "u"}])
    assert nested.value.errors()[0]["loc"] == ("artifacts", 0, "schema_version")
    with pytest.raises(ValidationError) as proposed:
        forecast(proposed_action={"action_type": "click"})
    assert proposed.value.errors()[0]["loc"] == ("proposed_action", "schema_version")
    with pytest.raises(ValidationError) as error_info:
        actual_outcome(status=ExecutionStatus.FAILED, error={"code": "E", "message": "m"})
    assert error_info.value.errors()[0]["loc"] == ("error", "schema_version")
    with pytest.raises(ValidationError) as results:
        verification_result(outcome_results=[{"outcome_index": 0, "status": "pass"}])
    assert results.value.errors()[0]["loc"] == ("outcome_results", 0, "schema_version")


def test_nested_unknown_schema_version_is_rejected() -> None:
    bad_artifact = {"schema_version": "0.2", "artifact_id": "a", "artifact_type": "file", "uri": "u"}
    with pytest.raises(ValidationError) as outer:
        observation(artifacts=[bad_artifact])
    assert outer.value.errors()[0]["loc"] == ("artifacts", 0, "schema_version")
    assert outer.value.errors()[0]["type"] == "literal_error"
    with pytest.raises(ValidationError) as direct:
        ArtifactRef.model_validate(bad_artifact)
    assert direct.value.errors()[0]["loc"] == ("schema_version",)


def test_extra_fields_are_forbidden_at_every_level() -> None:
    with pytest.raises(ValidationError) as top:
        observation(unexpected="x")
    assert top.value.errors()[0]["loc"] == ("unexpected",)
    assert top.value.errors()[0]["type"] == "extra_forbidden"
    nested = {**artifact().model_dump(mode="json"), "extra": "x"}
    with pytest.raises(ValidationError) as inner:
        observation(artifacts=[nested])
    assert inner.value.errors()[0]["loc"] == ("artifacts", 0, "extra")


# --- strict scalars, numbers and identifiers ---------------------------------

@pytest.mark.parametrize("field,value,expected", [
    ("observation_id", True, "string_type"),
    ("observation_id", 7, "string_type"),
    ("summary", 5, "string_type"),
    ("environment_type", 1.5, "string_type"),
])
def test_strings_never_accept_numbers_or_booleans(field: str, value: object, expected: str) -> None:
    with pytest.raises(ValidationError) as error:
        observation(**{field: value})
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == expected


@pytest.mark.parametrize("field,value", [
    ("has_actual_value", "true"), ("has_actual_value", 1), ("has_actual_value", 0),
])
def test_booleans_never_accept_strings_or_integers(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as error:
        outcome_verification(**{field: value})
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "bool_type"


def test_required_flag_is_a_strict_boolean() -> None:
    with pytest.raises(ValidationError) as error:
        Precondition(schema_version="0.1", condition_type="element_visible", required=1)
    assert error.value.errors()[0]["type"] == "bool_type"
    assert Precondition(schema_version="0.1", condition_type="c", required=False).required is False


def test_integers_reject_booleans_and_floats() -> None:
    with pytest.raises(ValidationError) as boolean:
        outcome_verification(outcome_index=True)
    assert boolean.value.errors()[0]["type"] == "int_type"
    with pytest.raises(ValidationError) as floating:
        outcome_verification(outcome_index=1.0)
    assert floating.value.errors()[0]["type"] == "int_type"


def test_numbers_reject_booleans_strings_and_non_finite_values() -> None:
    for value, expected in ((True, "float_type"), ("0.5", "float_type"), (float("nan"), "finite_number"),
                            (float("inf"), "finite_number"), (float("-inf"), "finite_number")):
        with pytest.raises(ValidationError) as error:
            forecast(confidence=value)
        assert error.value.errors()[0]["loc"] == ("confidence",), value
        assert error.value.errors()[0]["type"] == expected, value
    for value in (True, float("nan"), float("inf")):
        with pytest.raises(ValidationError) as error:
            actual_outcome(duration_ms=value)
        assert error.value.errors()[0]["loc"] == ("duration_ms",)
    assert forecast(confidence=1).confidence == 1


def test_ranged_numbers_are_enforced() -> None:
    with pytest.raises(ValidationError) as high:
        forecast(confidence=1.0001)
    assert high.value.errors()[0]["loc"] == ("confidence",)
    with pytest.raises(ValidationError) as low:
        forecast(confidence=-0.1)
    assert low.value.errors()[0]["loc"] == ("confidence",)
    with pytest.raises(ValidationError) as negative_index:
        outcome_verification(outcome_index=-1)
    assert negative_index.value.errors()[0]["type"] == "greater_than_equal"
    with pytest.raises(ValidationError) as negative_duration:
        actual_outcome(duration_ms=-0.5)
    assert negative_duration.value.errors()[0]["type"] == "greater_than_equal"
    assert forecast(confidence=0.0).confidence == 0.0
    assert forecast(confidence=1.0).confidence == 1.0


@pytest.mark.parametrize("factory,field", [
    (artifact, "artifact_id"), (artifact, "artifact_type"), (artifact, "uri"),
    (lambda **o: ErrorInfo(**{"schema_version": "0.1", "code": "E", "message": "m", **o}), "code"),
    (capabilities, "adapter_name"),
    (lambda **o: VerificationResult(**{"schema_version": "0.1", "status": "pass", "verifier": "v", **o}), "verifier"),
])
def test_identifiers_and_type_names_need_a_non_whitespace_character(factory, field: str) -> None:
    with pytest.raises(ValidationError) as error:
        factory(**{field: "   "})
    assert field in error.value.errors()[0]["loc"]


def test_capability_elements_must_be_unique_and_non_blank() -> None:
    with pytest.raises(ValidationError) as duplicate:
        capabilities(supported_actions=["click", "click"])
    assert duplicate.value.errors()[0]["loc"] == ("supported_actions",)
    with pytest.raises(ValidationError) as blank:
        capabilities(supported_actions=["click", "  "])
    assert blank.value.errors()[0]["loc"] == ("supported_actions", 1)
    with pytest.raises(ValidationError) as wrong_type:
        capabilities(supported_actions=[1])
    assert wrong_type.value.errors()[0]["loc"] == ("supported_actions", 0)


# --- JSON payload rules ------------------------------------------------------

def test_json_object_accepts_only_json_values() -> None:
    payload = {"text": "x", "count": 1, "ratio": 0.5, "flag": False, "none": None,
               "list": [1, "two", None, {"nested": True}], "nested": {"a": [1.5]}}
    model = observation(state=payload)
    assert model.state == payload
    assert_roundtrips(model)


@pytest.mark.parametrize("bad", [
    b"bytes", dt.datetime(2026, 1, 1, tzinfo=UTC), {1, 2}, (1, 2), object(), complex(1, 2),
])
def test_json_payloads_reject_non_json_values(bad: object) -> None:
    with pytest.raises(ValidationError) as error:
        observation(state={"bad": bad})
    assert json_path(error.value) == ("state", "bad")
    assert error.value.errors()[0]["loc"][-1] == "invalid"


@pytest.mark.parametrize("payload,expected", [
    ({"outer": [{"bad": b"bytes"}]}, ("state", "outer", 0, "bad")),
    ({"outer": [{"bad": {"deeper": float("inf")}}]}, ("state", "outer", 0, "bad", "deeper")),
    ({"items": [1, {1: "x"}]}, ("state", "items", 1, 1, "[key]")),
    ({"items": [1, (1, 2)]}, ("state", "items", 1)),
    ({"items": [1, {2, 3}]}, ("state", "items", 1)),
])
def test_nested_json_failures_report_the_offending_path(payload: dict, expected: tuple) -> None:
    """F01: the error location names the offending nested field or index, not just the field."""
    with pytest.raises(ValidationError) as error:
        observation(state=payload)
    assert json_path(error.value) == expected
    assert len(error.value.errors()[0]["loc"]) > 1


def test_json_payloads_reject_non_string_keys_and_non_mapping_objects() -> None:
    with pytest.raises(ValidationError) as keys:
        observation(state={1: "x"})
    assert json_path(keys.value) == ("state", 1, "[key]")
    with pytest.raises(ValidationError) as sequence:
        observation(state=[1, 2])
    assert sequence.value.errors()[0]["loc"] == ("state",)
    assert sequence.value.errors()[0]["type"] == "dict_type"


def test_json_payloads_reject_non_finite_numbers_at_any_depth() -> None:
    with pytest.raises(ValidationError) as flat:
        observation(state={"a": float("nan")})
    assert json_path(flat.value) == ("state", "a")
    with pytest.raises(ValidationError) as deep:
        observation(state={"a": [1, {"b": float("inf")}]})
    assert json_path(deep.value) == ("state", "a", 1, "b")


def test_json_value_field_accepts_scalars_lists_objects_and_null() -> None:
    for value in (None, True, 0, 1.5, "text", [1, "two", None], {"k": [1, {"n": None}]}):
        assert outcome_verification(actual_value=value, has_actual_value=True).actual_value == value
    with pytest.raises(ValidationError):
        outcome_verification(actual_value=b"bytes", has_actual_value=True)


def test_deeply_located_errors_identify_nested_fields_and_indices() -> None:
    contaminated = {**artifact().model_dump(mode="json"), "metadata": {"bad": b"bytes"}}
    with pytest.raises(ValidationError) as metadata:
        observation(artifacts=[artifact().model_dump(mode="json"), contaminated])
    assert json_path(metadata.value) == ("artifacts", 1, "metadata", "bad")
    broken_result = {**outcome_verification().model_dump(mode="json"), "outcome_index": 9, "status": "nope"}
    with pytest.raises(ValidationError) as index:
        verification_result(outcome_results=[outcome_verification().model_dump(mode="json"), broken_result])
    assert index.value.errors()[0]["loc"] == ("outcome_results", 1, "status")


# --- assignment validation and the documented mutation boundary --------------

def test_assignment_validation_covers_every_field() -> None:
    model = observation()
    with pytest.raises(ValidationError) as wrong_type:
        model.observation_id = 5
    assert wrong_type.value.errors()[0]["loc"] == ("observation_id",)
    with pytest.raises(ValidationError) as bad_payload:
        model.metadata = {"bad": float("inf")}
    assert json_path(bad_payload.value) == ("metadata", "bad")
    with pytest.raises(ValidationError) as nested_version:
        model.artifacts = [{"artifact_id": "a", "artifact_type": "t", "uri": "u"}]
    assert nested_version.value.errors()[0]["loc"] == ("artifacts", 0, "schema_version")
    model.summary = "updated"
    assert model.summary == "updated"


def test_in_place_nested_mutation_is_not_validated_and_ingress_must_revalidate() -> None:
    """Documents the REQ-02 boundary: Pydantic does not observe in-place nested edits."""
    model = observation()
    model.state["bytes"] = b"not-json"
    assert model.state["bytes"] == b"not-json"
    with pytest.raises(ValidationError):
        Observation.model_validate(model.model_dump())


# --- timestamps --------------------------------------------------------------

def test_timestamp_requires_timezone_awareness() -> None:
    # A naive datetime is constructed deliberately: it must be rejected.
    naive_value = dt.datetime(2026, 1, 1, 12, 0)  # noqa: DTZ001
    with pytest.raises(ValidationError) as naive:
        observation(timestamp=naive_value)
    assert naive.value.errors()[0]["loc"] == ("timestamp",)
    assert naive.value.errors()[0]["type"] == "timezone_aware"
    with pytest.raises(ValidationError) as epoch:
        observation(timestamp=1767225600)
    assert epoch.value.errors()[0]["loc"] == ("timestamp",)
    assert observation(timestamp=TIMESTAMP).timestamp.tzinfo is not None


def test_timestamp_accepts_the_iso_json_representation() -> None:
    as_string = observation(timestamp="2026-01-01T12:00:00+00:00")
    assert as_string.timestamp == TIMESTAMP
    zulu = observation(timestamp="2026-01-01T12:00:00Z")
    assert zulu.timestamp == TIMESTAMP
    assert_roundtrips(as_string)
    assert_roundtrips(zulu)


# --- enums -------------------------------------------------------------------

def test_status_enums_are_separate_and_use_their_json_values() -> None:
    assert ExecutionStatus is not VerificationStatus
    assert {member.value for member in ExecutionStatus} == {"succeeded", "failed", "unknown"}
    assert {member.value for member in VerificationStatus} == {"pass", "fail", "inconclusive"}
    assert actual_outcome(status="succeeded").status is ExecutionStatus.SUCCEEDED
    assert outcome_verification(status="inconclusive").status is VerificationStatus.INCONCLUSIVE
    assert actual_outcome().model_dump(mode="json")["status"] == "succeeded"
    assert outcome_verification().model_dump(mode="json")["status"] == "pass"


@pytest.mark.parametrize("value", ["SUCCEEDED", "ok", "", True, 1, None, ["succeeded"]])
def test_unknown_enum_values_are_rejected(value: object) -> None:
    with pytest.raises(ValidationError) as error:
        actual_outcome(status=value)
    assert error.value.errors()[0]["loc"] == ("status",)


# --- cross-field rules -------------------------------------------------------

def test_succeeded_execution_must_not_carry_a_failure_error() -> None:
    failure = ErrorInfo(schema_version="0.1", code="E1", message="boom")
    with pytest.raises(ValidationError, match="SUCCEEDED"):
        actual_outcome(status=ExecutionStatus.SUCCEEDED, error=failure)
    assert actual_outcome(status=ExecutionStatus.FAILED, error=failure).error == failure
    assert actual_outcome(status=ExecutionStatus.UNKNOWN, error=failure).error == failure
    assert actual_outcome(status=ExecutionStatus.SUCCEEDED).error is None
    assert_roundtrips(actual_outcome(status=ExecutionStatus.FAILED, error=failure))


def test_observed_null_is_distinct_from_an_absent_value() -> None:
    with pytest.raises(ValidationError, match="actual_value"):
        outcome_verification(has_actual_value=False, actual_value=1)
    absent = outcome_verification(has_actual_value=False, actual_value=None)
    observed_null = outcome_verification(has_actual_value=True, actual_value=None)
    observed_value = outcome_verification(has_actual_value=True, actual_value=0)
    assert absent.has_actual_value is False and absent.actual_value is None
    assert observed_null.has_actual_value is True and observed_null.actual_value is None
    assert observed_value.actual_value == 0
    for model in (absent, observed_null, observed_value):
        restored = OutcomeVerification.model_validate_json(model.model_dump_json())
        assert (restored.has_actual_value, restored.actual_value) == (model.has_actual_value, model.actual_value)
        assert_roundtrips(model)


def test_repeated_outcome_index_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate outcome_index"):
        verification_result(outcome_results=[
            outcome_verification(outcome_index=0),
            outcome_verification(outcome_index=0, status=VerificationStatus.FAIL),
        ])
    with pytest.raises(ValidationError, match="duplicate outcome_index"):
        verification_result(outcome_results=[outcome_verification(outcome_index=0)]).outcome_results = [
            outcome_verification(outcome_index=1), outcome_verification(outcome_index=1),
        ]
    assert verification_result(outcome_results=[
        outcome_verification(outcome_index=0), outcome_verification(outcome_index=1),
    ]).outcome_results[1].outcome_index == 1


def test_deferred_cross_object_rules_are_not_enforced_here() -> None:
    """Cross-object index bounds, coverage and aggregate status are P1-T03 concerns."""
    assert verification_result(outcome_results=[outcome_verification(outcome_index=99)]).outcome_results[0].outcome_index == 99
    assert forecast(expected_outcomes=[]).expected_outcomes == []
    assert VerificationResult.model_fields.keys().isdisjoint({"confidence"})


def test_error_info_introduces_no_runtime_stage_context() -> None:
    fields = set(ErrorInfo.model_fields)
    assert fields == {"schema_version", "code", "message", "recoverable", "details"}
    assert VerificationResult.model_fields.keys().isdisjoint({"run_id", "step_id", "stage"})


# --- F02: strict container and enum input types ------------------------------

@pytest.mark.parametrize("factory,field,value", [
    (capabilities, "supported_actions", ("click",)),
    (capabilities, "supported_observation_types", ("page",)),
    (observation, "artifacts", ()),
    (actual_outcome, "artifacts", ()),
    (forecast, "expected_outcomes", ()),
    (verification_result, "outcome_results", ()),
])
def test_model_list_fields_reject_non_list_containers(factory, field: str, value: object) -> None:
    """F02: tuples and other sequences are not coerced into a JSON list field."""
    with pytest.raises(ValidationError) as error:
        factory(**{field: value})
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "list_type"


@pytest.mark.parametrize("factory,field,value", [
    (actual_outcome, "status", b"succeeded"),
    (actual_outcome, "status", 1),
    (actual_outcome, "status", True),
    (outcome_verification, "status", b"pass"),
    (verification_result, "status", b"pass"),
])
def test_enum_fields_reject_unrelated_input_types(factory, field: str, value: object) -> None:
    """F02: bytes and unrelated scalars are never decoded into an enum value."""
    with pytest.raises(ValidationError) as error:
        factory(**{field: value})
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "value_error"


def test_enum_json_strings_remain_accepted_in_both_modes() -> None:
    outcome = actual_outcome(status="failed")
    assert outcome.status is ExecutionStatus.FAILED
    assert_roundtrips(outcome)
    result = verification_result(status="inconclusive", outcome_results=[outcome_verification(status="fail")])
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.outcome_results[0].status is VerificationStatus.FAIL
    assert_roundtrips(result)


# --- F03: rejected assignment preserves the previous valid state -------------

def test_rejected_assignment_preserves_the_previous_state() -> None:
    failure = ErrorInfo(schema_version="0.1", code="E1", message="boom")
    outcome = actual_outcome(status=ExecutionStatus.SUCCEEDED)
    with pytest.raises(ValidationError, match="SUCCEEDED"):
        outcome.error = failure
    assert outcome.error is None
    assert outcome.status is ExecutionStatus.SUCCEEDED
    failed = actual_outcome(status=ExecutionStatus.FAILED, error=failure)
    with pytest.raises(ValidationError, match="SUCCEEDED"):
        failed.status = ExecutionStatus.SUCCEEDED
    assert failed.status is ExecutionStatus.FAILED
    assert failed.error == failure
    assert_roundtrips(failed)


def test_rejected_observed_null_assignment_preserves_the_previous_state() -> None:
    observed_value = outcome_verification(has_actual_value=True, actual_value=0)
    with pytest.raises(ValidationError, match="actual_value"):
        observed_value.has_actual_value = False
    assert observed_value.has_actual_value is True
    assert observed_value.actual_value == 0
    # (False, None) is a consistent "nothing observed" state, so this assignment is legal.
    observed_null = outcome_verification(has_actual_value=True, actual_value=None)
    observed_null.has_actual_value = False
    assert observed_null.has_actual_value is False
    assert observed_null.actual_value is None
    absent = outcome_verification(has_actual_value=False, actual_value=None)
    with pytest.raises(ValidationError, match="actual_value"):
        absent.actual_value = 7
    assert absent.has_actual_value is False
    assert absent.actual_value is None
    assert_roundtrips(observed_null)


def test_rejected_list_and_payload_assignment_preserves_the_previous_state() -> None:
    result = verification_result(outcome_results=[outcome_verification(outcome_index=0)])
    with pytest.raises(ValidationError, match="duplicate outcome_index"):
        result.outcome_results = [
            outcome_verification(outcome_index=1), outcome_verification(outcome_index=1),
        ]
    assert [item.outcome_index for item in result.outcome_results] == [0]
    model = observation(metadata={"keep": 1})
    with pytest.raises(ValidationError):
        model.metadata = {"bad": b"bytes"}
    assert model.metadata == {"keep": 1}
    assert_roundtrips(result)


def test_construction_still_rejects_the_same_conflicts() -> None:
    failure = ErrorInfo(schema_version="0.1", code="E1", message="boom")
    with pytest.raises(ValidationError, match="SUCCEEDED"):
        actual_outcome(status=ExecutionStatus.SUCCEEDED, error=failure)
    with pytest.raises(ValidationError, match="actual_value"):
        outcome_verification(has_actual_value=False, actual_value=1)
    with pytest.raises(ValidationError, match="duplicate outcome_index"):
        verification_result(outcome_results=[
            outcome_verification(outcome_index=0), outcome_verification(outcome_index=0),
        ])


# --- public surface ----------------------------------------------------------

def test_public_exports_are_complete_and_root_exports_are_untouched() -> None:
    import forecontract
    from forecontract import models

    expected = {
        "ActionSpec", "ActualOutcome", "AdapterCapabilities", "ArtifactRef", "ErrorInfo", "ExecutionStatus",
        "ExpectedOutcome", "Forecast", "JsonObject", "JsonValue", "Observation", "OutcomeVerification",
        "Precondition", "VerificationResult", "VerificationStatus",
    }
    assert set(models.__all__) == expected
    assert all(hasattr(models, name) for name in expected)
    assert forecontract.__all__ == ["__version__"]
    assert not hasattr(forecontract, "Observation")


def test_json_types_are_usable_as_annotations() -> None:
    assert JsonValue is not None and JsonObject is not None
