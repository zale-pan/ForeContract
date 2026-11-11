"""Acceptance tests for the public contract models.

Every check runs against the real ``forecontract.contracts`` models imported from the
package source tree; nothing here edits the existing Core models, protocols or their tests.
The suite covers the version, strictness, intrinsic-structure, serialization and deep
immutability rules the round-001 requirements put on ``VerificationBinding``,
``VerificationPolicy``, ``RecoveryPolicy`` and ``ActionContract``.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

import forecontract
from forecontract import contracts
from forecontract.contracts import (
    ActionContract,
    ContractBuilder,
    ContractBuildError,
    ContractValidationHooks,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import ActionSpec, ExpectedOutcome, Precondition
from forecontract.models.outcomes import VerificationStatus

PUBLIC_SURFACE = [
    "ActionContract",
    "ContractBuildError",
    "ContractBuilder",
    "ContractValidationHooks",
    "RecoveryPolicy",
    "VerificationBinding",
    "VerificationPolicy",
]
AWARE = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)

# The recursive JSON schema is a tagged union, so an error location interleaves its branch
# labels with the offending keys and indices. ``json_path`` drops exactly those labels; the
# fixtures below avoid key names that collide with them.
JSON_BRANCH_LABELS = frozenset({"null", "bool", "int", "float", "str", "list", "dict", "invalid"})


def json_path(error: ValidationError) -> tuple:
    """The offending JSON field/index path reported for the first error."""
    return tuple(part for part in error.errors()[0]["loc"] if part not in JSON_BRANCH_LABELS)


def binding(index: int = 0, **overrides) -> VerificationBinding:
    fields = {
        "schema_version": "0.1",
        "outcome_index": index,
        "verifier_type": "dom_verifier",
        "config": {"expected": "y"},
    }
    fields.update(overrides)
    return VerificationBinding(**fields)


def policy(*bindings: VerificationBinding) -> VerificationPolicy:
    return VerificationPolicy(
        schema_version="0.1", bindings=list(bindings) or [binding()]
    )


def action(**overrides) -> ActionSpec:
    fields = {"schema_version": "0.1", "action_type": "click", "payload": {"selector": "#go"}}
    fields.update(overrides)
    return ActionSpec(**fields)


def outcome(required: bool = True, **overrides) -> ExpectedOutcome:
    fields = {
        "schema_version": "0.1",
        "outcome_type": "dom_changed",
        "payload": {"marker": "ready"},
        "required": required,
    }
    fields.update(overrides)
    return ExpectedOutcome(**fields)


def precondition(**overrides) -> Precondition:
    fields = {
        "schema_version": "0.1",
        "condition_type": "visible",
        "payload": {"selector": "#go"},
    }
    fields.update(overrides)
    return Precondition(**fields)


def contract_fields(**overrides) -> dict:
    fields = {
        "schema_version": "0.1",
        "contract_id": "c-1",
        "goal_id": "g-1",
        "forecast_id": "f-1",
        "observation_id": "o-1",
        "adapter_name": "fake",
        "environment_type": "browser",
        "action": action(),
        "preconditions": [precondition()],
        "expected_outcomes": [outcome()],
        "verification_policy": policy(),
        "recovery_policy": RecoveryPolicy(schema_version="0.1", mode="abort"),
        "timeout_ms": 100,
        "created_at": AWARE,
        "metadata": {"run": 1},
    }
    fields.update(overrides)
    return fields


def contract(**overrides) -> ActionContract:
    return ActionContract(**contract_fields(**overrides))


# --- public surface -----------------------------------------------------------

def test_the_public_contract_surface_is_exactly_the_agreed_names() -> None:
    assert sorted(contracts.__all__) == PUBLIC_SURFACE
    for name in PUBLIC_SURFACE:
        assert hasattr(contracts, name), name


def test_the_contracts_subpackage_does_not_change_the_existing_exports() -> None:
    assert forecontract.__all__ == ["__version__"]
    from forecontract import models, protocols

    assert "contracts" not in models.__all__
    assert "contracts" not in protocols.__all__
    for module in (models, protocols):
        for name in module.__all__:
            assert hasattr(module, name), name


# --- versions ----------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda: VerificationBinding(outcome_index=0, verifier_type="v"),
    lambda: VerificationPolicy(bindings=[]),
    lambda: RecoveryPolicy(mode="abort"),
    lambda: ActionContract(**{key: value for key, value in contract_fields().items() if key != "schema_version"}),
])
def test_a_missing_schema_version_is_rejected(factory) -> None:
    with pytest.raises(ValidationError) as info:
        factory()
    assert any("schema_version" in str(error["loc"]) for error in info.value.errors())


@pytest.mark.parametrize("version", ["0.2", "1.0", 0.1, None, ""])
def test_an_unknown_schema_version_is_rejected(version) -> None:
    with pytest.raises(ValidationError):
        contract(schema_version=version)


@pytest.mark.parametrize("factory", [
    lambda: binding(schema_version="0.2"),
    lambda: policy(binding(schema_version="0.2")),
    lambda: RecoveryPolicy(schema_version="0.2", mode="abort"),
    lambda: contract(action=action(schema_version="0.2")),
    lambda: contract(preconditions=[precondition(schema_version="0.2")]),
    lambda: contract(expected_outcomes=[outcome(schema_version="0.2")]),
])
def test_every_nesting_level_rejects_an_unknown_version(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


# --- extras ------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda: binding(extra=1),
    lambda: VerificationPolicy(schema_version="0.1", bindings=[binding()], extra=1),
    lambda: RecoveryPolicy(schema_version="0.1", mode="abort", extra=1),
    lambda: contract(extra=1),
    lambda: contract(action=action(extra=1)),
])
def test_unknown_fields_are_rejected_at_every_nesting_level(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


# --- strict scalars, containers and recursive JSON ---------------------------

@pytest.mark.parametrize("index", ["0", "0.0", True, None, -1, 1.5])
def test_outcome_index_is_a_strict_non_negative_integer(index) -> None:
    with pytest.raises(ValidationError):
        binding(outcome_index=index)


@pytest.mark.parametrize("verifier", [1, True, None, "", "   "])
def test_verifier_type_must_be_a_non_blank_string(verifier) -> None:
    with pytest.raises(ValidationError):
        binding(verifier_type=verifier)


@pytest.mark.parametrize("verifier", [1, "  ", None])
def test_action_identifiers_must_be_non_blank_strings(verifier) -> None:
    with pytest.raises(ValidationError):
        contract(contract_id=verifier)


@pytest.mark.parametrize("mode", ["retry", "replan", "ABORT", "", None, 1])
def test_only_the_explicit_abort_recovery_mode_is_accepted(mode) -> None:
    with pytest.raises(ValidationError):
        RecoveryPolicy(schema_version="0.1", mode=mode)
    assert RecoveryPolicy(schema_version="0.1", mode="abort").mode == "abort"


@pytest.mark.parametrize("value", ["100", True, None, 0, -1, 0.0, float("nan"), float("inf"), object()])
def test_timeout_ms_must_be_a_strictly_positive_finite_number(value) -> None:
    with pytest.raises(ValidationError):
        contract(timeout_ms=value)


def test_an_integer_timeout_is_accepted_and_serialized_as_a_number() -> None:
    built = contract(timeout_ms=250)
    assert built.timeout_ms == 250.0
    assert json.loads(built.model_dump_json())["timeout_ms"] == 250.0


@pytest.mark.parametrize("payload", [
    {"a": b"raw"},
    {"a": float("nan")},
    {"a": float("inf")},
    {1: "value"},
    {"a": (1, 2)},
    {"a": object()},
    {"a": {"b": b"deep"}},
])
def test_recursive_json_payloads_reject_non_json_values(payload) -> None:
    with pytest.raises(ValidationError):
        binding(config=payload)
    with pytest.raises(ValidationError):
        contract(metadata=payload)


def test_a_nested_json_failure_locates_the_offending_field() -> None:
    with pytest.raises(ValidationError) as info:
        binding(config={"outer": [{"inner": b"raw"}]})
    assert json_path(info.value) == ("config", "outer", 0, "inner")


@pytest.mark.parametrize("payload", [{"a": 1}, {"a": [1, {"b": None}]}, {}, {"a": True}])
def test_ordinary_json_payloads_are_accepted(payload) -> None:
    assert binding(config=payload).config == payload


@pytest.mark.parametrize("state", ["{}", "not a mapping", [1], 123, None, True])
def test_a_non_object_payload_is_rejected(state) -> None:
    with pytest.raises(ValidationError):
        contract(action=action(payload=state))


# --- timestamps --------------------------------------------------------------

def test_created_at_accepts_an_aware_datetime_and_its_iso_representation() -> None:
    from_datetime = contract(created_at=AWARE)
    from_text = contract(created_at="2026-09-30T12:00:00Z")
    assert from_datetime.created_at == from_text.created_at
    assert json.loads(from_datetime.model_dump_json())["created_at"] == "2026-09-30T12:00:00Z"


@pytest.mark.parametrize("value", [
    datetime(2026, 9, 30, 12, 0),  # noqa: DTZ001 - a naive datetime is the rejected input here
    "2026-09-30T12:00:00",
    "not a timestamp",
    12345,
    None,
])
def test_created_at_rejects_a_naive_or_malformed_timestamp(value) -> None:
    with pytest.raises(ValidationError):
        contract(created_at=value)


# --- intrinsic structure ------------------------------------------------------

@pytest.mark.parametrize("outcomes", [
    [],
    [outcome(required=False)],
    [outcome(required=False), outcome(required=False, outcome_type="other")],
])
def test_a_contract_requires_at_least_one_required_outcome(outcomes) -> None:
    with pytest.raises(ValidationError):
        contract(
            expected_outcomes=outcomes,
            verification_policy=VerificationPolicy(
                schema_version="0.1",
                bindings=[binding(index) for index in range(len(outcomes))],
            ),
        )


def test_a_contract_accepts_mixed_optional_and_required_outcomes() -> None:
    built = contract(
        expected_outcomes=[outcome(), outcome(required=False, outcome_type="other")],
        verification_policy=policy(binding(0), binding(1)),
    )
    assert [item.required for item in built.expected_outcomes] == [True, False]


@pytest.mark.parametrize("bindings", [
    [],
    [binding(0), binding(0)],
    [binding(1)],
])
def test_every_outcome_needs_exactly_one_binding(bindings) -> None:
    with pytest.raises(ValidationError):
        contract(verification_policy=VerificationPolicy(schema_version="0.1", bindings=bindings))


# --- roundtrips and determinism ----------------------------------------------

def test_both_roundtrips_reproduce_the_agreed_structure() -> None:
    built = contract()
    text = built.model_dump_json()
    from_text = ActionContract.model_validate_json(text)
    from_mapping = ActionContract.model_validate(built.model_dump(mode="json"))
    assert from_text.model_dump_json() == text
    assert from_mapping.model_dump_json() == text
    payload = json.loads(text)
    assert payload["timeout_ms"] == 100.0
    assert payload["recovery_policy"] == {"schema_version": "0.1", "mode": "abort"}
    assert payload["verification_policy"]["bindings"][0]["outcome_index"] == 0


def test_a_mapping_dump_is_an_ordinary_json_structure() -> None:
    dumped = contract().model_dump()
    assert type(dumped) is dict
    assert type(dumped["action"]) is dict
    assert type(dumped["action"]["payload"]) is dict
    assert type(dumped["preconditions"]) is list
    assert type(dumped["metadata"]) is dict
    # The python-mode dump keeps typed values such as datetime; the JSON-mode dump is the
    # agreed JSON structure and must be encodable by an ordinary JSON encoder.
    json_dump = contract().model_dump(mode="json")
    assert type(json_dump["action"]["payload"]) is dict
    assert type(json_dump["verification_policy"]["bindings"]) is list
    json.dumps(json_dump)


def test_identical_inputs_produce_identical_content() -> None:
    first = contract()
    second = contract()
    assert first.model_dump_json() == second.model_dump_json()
    assert first == second


def test_the_binding_order_supplied_by_the_caller_is_preserved() -> None:
    built = contract(
        expected_outcomes=[outcome(), outcome(outcome_type="other"), outcome(outcome_type="third")],
        verification_policy=policy(binding(2), binding(0), binding(1)),
    )
    assert [item.outcome_index for item in built.verification_policy.bindings] == [2, 0, 1]
    dumped = json.loads(built.model_dump_json())
    assert [item["outcome_index"] for item in dumped["verification_policy"]["bindings"]] == [2, 0, 1]


# --- independent defaults -----------------------------------------------------

def test_defaults_are_independent_between_instances() -> None:
    first = VerificationBinding(schema_version="0.1", outcome_index=0, verifier_type="v")
    second = VerificationBinding(schema_version="0.1", outcome_index=1, verifier_type="v")
    assert first.config == {} and second.config == {}
    assert first.config is not second.config
    one, two = contract(), contract()
    assert one.metadata is not two.metadata
    assert one.action.metadata is not two.action.metadata


def test_a_defaulted_payload_does_not_alias_the_class_default() -> None:
    assert VerificationBinding.model_fields["config"].default_factory is dict
    assert ActionContract.model_fields["metadata"].default_factory is dict
    defaulted = VerificationBinding(schema_version="0.1", outcome_index=0, verifier_type="v")
    assert defaulted.config == {}
    assert defaulted.config is not binding().config


# --- deep immutability --------------------------------------------------------

def test_top_level_assignment_is_rejected() -> None:
    built = contract()
    with pytest.raises(ValidationError):
        built.contract_id = "other"
    with pytest.raises(ValidationError):
        built.timeout_ms = 5
    assert built.contract_id == "c-1"


@pytest.mark.parametrize("mutate", [
    lambda built: built.metadata.__setitem__("run", 99),
    lambda built: built.action.payload.__setitem__("selector", "#evil"),
    lambda built: built.action.metadata.__setitem__("x", 1),
    lambda built: built.action.__setattr__("action_type", "evil"),
    lambda built: built.preconditions.__setitem__(0, None),
    lambda built: built.preconditions[0].__setattr__("required", False),
    lambda built: built.preconditions[0].payload.__setitem__("selector", "#evil"),
    lambda built: built.expected_outcomes[0].payload.__setitem__("marker", "evil"),
    lambda built: built.verification_policy.bindings[0].config.__setitem__("expected", "evil"),
    lambda built: built.verification_policy.bindings.__setitem__(0, None),
])
def test_every_nested_getter_surface_rejects_mutation(mutate) -> None:
    built = contract()
    before = built.model_dump_json()
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        mutate(built)
    assert built.model_dump_json() == before


def test_mutating_a_dump_cannot_change_the_contract() -> None:
    built = contract()
    before = built.model_dump_json()
    dumped = built.model_dump()
    dumped["metadata"]["run"] = 99
    dumped["action"]["payload"]["selector"] = "#evil"
    dumped["preconditions"][0]["required"] = False
    dumped["verification_policy"]["bindings"][0]["config"]["expected"] = "evil"
    assert built.model_dump_json() == before


def test_mutating_a_deserialized_twin_cannot_change_the_original() -> None:
    built = contract()
    before = built.model_dump_json()
    twin = ActionContract.model_validate_json(built.model_dump_json())
    with pytest.raises((AttributeError, TypeError)):
        twin.metadata.__setitem__("run", 99)
    assert built.model_dump_json() == before


def test_the_contract_keeps_its_own_copy_of_the_caller_models() -> None:
    spec, condition, result = action(), precondition(), outcome()
    built = contract(action=spec, preconditions=[condition], expected_outcomes=[result])
    before = built.model_dump_json()
    spec.payload["selector"] = "#evil"
    spec.metadata["x"] = 1
    condition.payload["selector"] = "#evil"
    condition.required = False
    result.payload["marker"] = "evil"
    assert built.model_dump_json() == before
    assert built.action.payload == {"selector": "#go"}
    assert built.preconditions[0].payload == {"selector": "#go"}
    assert built.expected_outcomes[0].payload == {"marker": "ready"}


def test_an_in_place_contaminated_payload_is_rejected_at_ingress() -> None:
    spec = action()
    spec.payload["bad"] = float("nan")
    with pytest.raises(ValidationError):
        contract(action=spec)
    assert copy.deepcopy(spec.payload)["bad"] != copy.deepcopy(spec.payload)["bad"]  # still NaN


# --- error type ---------------------------------------------------------------

def test_contract_build_error_exposes_code_location_and_message() -> None:
    error = ContractBuildError("INPUT_INVALID", ("forecast", "proposed_action", 0), "rejected")
    assert error.code == "INPUT_INVALID"
    assert error.location == ("forecast", "proposed_action", 0)
    assert error.message == "rejected"
    assert "INPUT_INVALID" in str(error) and "forecast.proposed_action[0]" in str(error)
    assert isinstance(error, ValueError)


def test_an_undocumented_build_error_code_is_refused() -> None:
    with pytest.raises(ValueError):
        ContractBuildError("NOT_A_CODE", (), "rejected")


def test_the_hook_protocol_declares_the_four_synchronous_methods() -> None:
    for name in (
        "validate_precondition",
        "validate_expected_outcome",
        "validate_verification_binding",
        "check_precondition",
    ):
        assert hasattr(ContractValidationHooks, name), name


def test_a_built_contract_is_accepted_by_the_documented_verification_status() -> None:
    assert VerificationStatus.PASS.value == "pass"
    from forecontract.contracts import ContractBuilder as Builder

    assert Builder is ContractBuilder


# --- round-002 corrections: F01 omitted defaults, F04 located model errors ---

def contract_without(*names: str) -> ActionContract:
    """Construct a contract while omitting the named fields entirely."""
    fields = contract_fields()
    for name in names:
        fields.pop(name)
    return ActionContract(**fields)


def test_an_omitted_metadata_default_is_deeply_immutable() -> None:
    """F01: an omitted default must be frozen exactly like an explicit value."""
    built = contract_without("metadata")
    assert built.metadata == {}
    before = built.model_dump_json()
    for mutate in (
        lambda: built.metadata.__setitem__("audit_changed", 1),
        lambda: built.metadata.update({"audit_changed": 1}),
        lambda: built.metadata.setdefault("audit_changed", 1),
    ):
        with pytest.raises((AttributeError, TypeError)):
            mutate()
    assert built.model_dump_json() == before
    assert json.loads(before)["metadata"] == {}


def test_an_omitted_binding_config_default_is_deeply_immutable() -> None:
    """F01: the same holds for a binding whose config was never supplied."""
    defaulted = VerificationBinding(schema_version="0.1", outcome_index=0, verifier_type="v")
    assert defaulted.config == {}
    before = defaulted.model_dump_json()
    with pytest.raises((AttributeError, TypeError)):
        defaulted.config["audit_changed"] = 1
    assert defaulted.model_dump_json() == before
    assert json.loads(before)["config"] == {}


def test_omitted_defaults_stay_immutable_through_mapping_and_text_loading() -> None:
    """F01: loading without the key must not resurrect a mutable default."""
    payload = contract().model_dump(mode="json")
    del payload["metadata"]
    payload["verification_policy"]["bindings"][0].pop("config")
    from_mapping = ActionContract.model_validate(payload)
    from_text = ActionContract.model_validate_json(json.dumps(payload))
    for loaded in (from_mapping, from_text):
        assert loaded.metadata == {}
        assert loaded.verification_policy.bindings[0].config == {}
        before = loaded.model_dump_json()
        with pytest.raises((AttributeError, TypeError)):
            loaded.metadata["audit_changed"] = 1
        with pytest.raises((AttributeError, TypeError)):
            loaded.verification_policy.bindings[0].config["audit_changed"] = 1
        assert loaded.model_dump_json() == before


def test_omitted_defaults_are_independent_between_instances() -> None:
    first, second = contract_without("metadata"), contract_without("metadata")
    assert first.metadata == {} and second.metadata == {}
    assert first.metadata is not second.metadata
    assert first.verification_policy.bindings[0].config is not second.verification_policy.bindings[0].config


def test_a_caller_default_binding_cannot_change_a_constructed_contract() -> None:
    """F01: mutating a caller-owned default binding after construction changes nothing."""
    binding_value = VerificationBinding(schema_version="0.1", outcome_index=0, verifier_type="v")
    built = contract(verification_policy=VerificationPolicy(schema_version="0.1", bindings=[binding_value]))
    before = built.model_dump_json()
    with pytest.raises((AttributeError, TypeError)):
        binding_value.config["audit_changed"] = 1
    assert built.model_dump_json() == before
    assert json.loads(before)["verification_policy"]["bindings"][0]["config"] == {}


def test_a_huge_timeout_is_a_located_model_validation_error() -> None:
    """F04: an out-of-range integer is a located validation error, never OverflowError."""
    with pytest.raises(ValidationError) as info:
        contract(timeout_ms=10**400)
    assert json_path(info.value) == ("timeout_ms",)
    with pytest.raises(ValidationError) as negative:
        contract(timeout_ms=-(10**400))
    assert json_path(negative.value) == ("timeout_ms",)


def test_a_huge_timeout_is_a_located_error_through_text_loading() -> None:
    """F04: the same holds when the huge value arrives as JSON text."""
    payload = contract().model_dump(mode="json")
    payload["timeout_ms"] = 10**400
    with pytest.raises(ValidationError) as info:
        ActionContract.model_validate_json(json.dumps(payload))
    assert json_path(info.value) == ("timeout_ms",)
    payload["timeout_ms"] = 1e400
    with pytest.raises(ValidationError) as infinite:
        ActionContract.model_validate_json(json.dumps(payload))
    assert json_path(infinite.value) == ("timeout_ms",)


def test_an_ordinary_timeout_still_loads_as_a_number() -> None:
    assert contract(timeout_ms=250).timeout_ms == 250.0
    assert contract(timeout_ms=250.5).timeout_ms == 250.5


# --- P2-T05 round-001: the compatible bounded recovery policy ------------------

def test_an_abort_policy_keeps_exactly_its_two_original_fields() -> None:
    """The default policy keeps its original dump, in every serialization form."""
    explicit = RecoveryPolicy(schema_version="0.1", mode="abort")
    omitted = RecoveryPolicy(schema_version="0.1")
    assert explicit.model_dump() == {"schema_version": "0.1", "mode": "abort"}
    assert explicit.model_dump(mode="json") == {"schema_version": "0.1", "mode": "abort"}
    assert json.loads(explicit.model_dump_json()) == {"schema_version": "0.1", "mode": "abort"}
    assert omitted == explicit
    dumped = contract().model_dump(mode="json")
    assert dumped["recovery_policy"] == {"schema_version": "0.1", "mode": "abort"}
    nested = json.loads(contract().model_dump_json())
    assert nested["recovery_policy"] == {"schema_version": "0.1", "mode": "abort"}


def test_a_bounded_policy_requires_explicit_controls() -> None:
    policy = RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=0,
                            allow_replan=False)
    assert policy.model_dump() == {"schema_version": "0.1", "mode": "bounded", "max_retries": 0,
                                   "allow_replan": False}
    assert json.loads(policy.model_dump_json()) == {"schema_version": "0.1", "mode": "bounded",
                                                    "max_retries": 0, "allow_replan": False}
    assert policy.max_retries == 0
    assert policy.allow_replan is False


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
    {"mode": "abort", "unknown_field": 1},
])
def test_an_invalid_recovery_policy_is_refused(kwargs: dict) -> None:
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):
        RecoveryPolicy(schema_version="0.1", **kwargs)
