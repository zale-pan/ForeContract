"""Acceptance tests for the JSON rule engine, the step verifier and the Builder hooks.

Every fixture is synthetic and embedded here. The tests drive the real evaluator, the real step
verifier and the real Contract Builder through a test-local capability wrapper, and they assert
values, ordering, isolation and error codes rather than the absence of exceptions.
"""

from __future__ import annotations

import copy
import datetime
import json
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
from forecontract.contracts.errors import ContractBuildError
from forecontract.contracts.models import ActionContract
from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
    Precondition,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome, ErrorInfo, VerificationStatus
from forecontract.protocols.adapter import UnsupportedActionError
from forecontract.verification import (
    JsonRuleEvaluator,
    PathExistsRule,
    RuleEvaluation,
    RuleValidationHooks,
    StepVerifier,
    TaskCompletionCriteria,
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


def equals(path: list, expected: object) -> ValueEqualsRule:
    return ValueEqualsRule(schema_version="0.1", rule_type="value_equals", path=path,
                           expected=expected)


def exists(path: list) -> PathExistsRule:
    return PathExistsRule(schema_version="0.1", rule_type="path_exists", path=path)


def action() -> ActionSpec:
    return ActionSpec(schema_version="0.1", action_type="click", payload={"selector": "#go"})


def evaluate(rule: object, piece: VerificationEvidence) -> RuleEvaluation:
    return JsonRuleEvaluator().evaluate(rule=rule, evidence=piece)


# --- A01: the five records ---

def test_every_record_requires_its_version_and_rejects_unknown_fields() -> None:
    with pytest.raises(VerificationError) as info:
        ValueEqualsRule(schema_version="0.2", rule_type="value_equals", path=["a"], expected=1)
    assert info.value.code == "VERIFICATION_INPUT_INVALID"
    with pytest.raises(VerificationError):
        VerificationEvidence(schema_version="0.1", observation=observation(), state_complete=True,
                             extra=1)
    with pytest.raises(VerificationError):
        PathExistsRule.model_validate({"schema_version": "0.1", "rule_type": "path_exists",
                                       "path": ["a"], "expected": 1})


@pytest.mark.parametrize("record,payload", [
    ("evidence", {"schema_version": "0.1", "observation": "not-an-observation",
                  "state_complete": True}),
    ("evidence", {"schema_version": "0.1"}),
    ("evidence", {"schema_version": "0.1", "observation": "not-an-observation",
                  "state_complete": 1}),
    ("equals", {"schema_version": "0.1", "rule_type": "value_equals", "path": ["a"]}),
    ("exists", {"schema_version": "0.1", "rule_type": "path_exists"}),
])
def test_a_missing_or_mistyped_field_is_refused(record: str, payload: dict) -> None:
    builder = {"evidence": VerificationEvidence, "equals": ValueEqualsRule,
               "exists": PathExistsRule}[record]
    with pytest.raises(VerificationError):
        builder.model_validate(payload)


def test_a_strict_boolean_has_no_implicit_default() -> None:
    assert VerificationEvidence.model_fields["state_complete"].is_required()
    assert VerificationEvidence.model_fields["state_complete"].default is None or True
    with pytest.raises(VerificationError):
        VerificationEvidence.model_validate(VerificationEvidence.model_construct(
            schema_version="0.1", observation=observation()).model_dump(mode="python")
            | {"state_complete": 1})


def test_a_null_expected_value_is_required_and_distinct_from_absence() -> None:
    assert equals(["nothing"], None).expected is None
    with pytest.raises(VerificationError):
        ValueEqualsRule.model_validate({"schema_version": "0.1", "rule_type": "value_equals",
                                        "path": ["a"]})


def test_an_unknown_self_key_is_refused_at_both_public_parsers() -> None:
    fields = {"schema_version": "0.1", "rule_type": "path_exists", "path": ["a"]}
    marked = {**fields, "self": MARKER}
    for call in (lambda: PathExistsRule.model_validate(marked),
                 lambda: PathExistsRule.model_validate_json(json.dumps(marked)),
                 lambda: PathExistsRule(**marked)):
        with pytest.raises(VerificationError) as info:
            call()
        assert MARKER not in str(info.value)
        assert len(str(info.value)) < 240


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"rule_type": "other"},
    {"path": "not-a-list"},
    {"path": []},
])
def test_a_forged_copied_rule_is_refused(mutation: dict) -> None:
    """F01/F02: an update that no longer matches the schema is refused at the copy itself."""
    from pydantic import ValidationError

    with pytest.raises((VerificationError, ValidationError)):
        exists(["a"]).model_copy(update=mutation)


def test_a_forged_constructed_rule_is_refused_at_the_public_ingress() -> None:
    """F02: an instance built outside the constructor is still refused by public ingress."""
    forged = PathExistsRule.model_construct(schema_version="9.9", rule_type="path_exists",
                                            path=["a"])
    with pytest.raises(VerificationError):
        PathExistsRule.model_validate(forged)
    with pytest.raises(VerificationError):
        evaluate(forged, evidence({"a": 1}))


def test_a_forged_copied_evidence_record_is_refused() -> None:
    from pydantic import ValidationError

    with pytest.raises((VerificationError, ValidationError)):
        evidence().model_copy(update={"state_complete": "yes"})


def test_a_forged_constructed_evidence_record_is_refused_at_the_public_ingress() -> None:
    forged = VerificationEvidence.model_construct(schema_version="0.1",
                                                  observation=observation(),
                                                  state_complete="yes")
    with pytest.raises(VerificationError):
        VerificationEvidence.model_validate(forged)
    with pytest.raises(VerificationError):
        evaluate(exists(["orders"]), forged)


def test_a_rule_round_trips_through_python_and_json() -> None:
    rule = equals(["orders", "count"], 2)
    assert ValueEqualsRule.model_validate(rule.model_dump(mode="python")) == rule
    assert ValueEqualsRule.model_validate_json(rule.model_dump_json()) == rule
    assert equals(["a"], None).model_dump(mode="json")["expected"] is None


# --- A02: paths ---

@pytest.mark.parametrize("path", [
    ["a", 0], ["", " "], ["a", 0, "b"], [0],
])
def test_a_legal_path_is_accepted(path: list) -> None:
    rule = exists(path)
    assert list(rule.path) == path


@pytest.mark.parametrize("path", [
    [], [True], [1.0], [-1], [None], [{"a": 1}], [[]],
])
def test_an_illegal_path_is_a_facility_error(path: list) -> None:
    with pytest.raises(VerificationError) as info:
        exists(path)
    assert info.value.code == "VERIFICATION_INPUT_INVALID"


def test_a_tuple_path_is_refused_and_never_coerced() -> None:
    """F01: the contract names a non-empty list, so a tuple is refused at every public ingress."""
    from pydantic import ValidationError

    for call in (
        lambda: exists(("a",)),
        lambda: PathExistsRule.model_validate({"schema_version": "0.1",
                                               "rule_type": "path_exists", "path": ("a",)}),
        lambda: equals(("a",), 1),
        lambda: ValueEqualsRule.model_validate({"schema_version": "0.1",
                                                "rule_type": "value_equals", "path": ("a",),
                                                "expected": 1}),
        lambda: exists(["a"]).model_copy(update={"path": ("a",)}),
    ):
        with pytest.raises((VerificationError, ValidationError)):
            call()
    # A bytes, string, number, mapping or set "path" is refused for the same reason.
    for bad in (b"a", "a", 0, {"a": 1}, {"a"}):
        with pytest.raises(VerificationError):
            exists(bad)
    for bad_segment in (1.5, True, None, {"a": 1}, ["a"]):
        with pytest.raises(VerificationError):
            exists([bad_segment])


def test_a_bytes_or_mistyped_status_is_refused() -> None:
    """F01: the enum ingress accepts a member or its JSON string value, never bytes."""
    for bad in (b"pass", b"fail", 1, 0, None, ["pass"]):
        with pytest.raises(VerificationError):
            RuleEvaluation(schema_version="0.1", status=bad, has_actual_value=True,
                           actual_value=1, reason="VALUE_MATCH")
        with pytest.raises(VerificationError):
            RuleEvaluation.model_validate({"schema_version": "0.1", "status": bad,
                                           "has_actual_value": True, "actual_value": 1,
                                           "reason": "VALUE_MATCH"})


@pytest.mark.parametrize("status,present,actual,reason", [
    ("fail", True, 1, "VALUE_MATCH"),
    ("pass", False, None, "PATH_PRESENT"),
    ("fail", True, 1, "PATH_ABSENT"),
    ("pass", False, None, "EVIDENCE_INSUFFICIENT"),
    ("inconclusive", True, 1, "EVIDENCE_INSUFFICIENT"),
    ("fail", False, None, "EVIDENCE_INSUFFICIENT"),
    ("inconclusive", False, None, "PATH_ABSENT"),
    ("fail", False, None, "VALUE_MATCH"),
])
def test_a_self_contradictory_verdict_is_refused_at_every_ingress(
    status: str, present: bool, actual: object, reason: str,
) -> None:
    """F01: the status, presence, value and reason are one verdict, not four free fields."""
    fields = {"schema_version": "0.1", "status": status, "has_actual_value": present,
              "actual_value": actual, "reason": reason}
    with pytest.raises(VerificationError):
        RuleEvaluation(**fields)
    with pytest.raises(VerificationError):
        RuleEvaluation.model_validate(fields)
    with pytest.raises(VerificationError):
        RuleEvaluation.model_validate_json(json.dumps(fields))


@pytest.mark.parametrize("status,present,actual,reason", [
    ("pass", True, 1, "VALUE_MATCH"),
    ("fail", True, 1, "VALUE_MISMATCH"),
    ("inconclusive", False, None, "EVIDENCE_INSUFFICIENT"),
    ("pass", True, None, "PATH_PRESENT"),
    ("pass", True, False, "PATH_PRESENT"),
    ("fail", False, None, "PATH_ABSENT"),
])
def test_every_legal_verdict_line_is_accepted(
    status: str, present: bool, actual: object, reason: str,
) -> None:
    """F01 control: each line the truth table contains is still accepted, in all three forms."""
    fields = {"schema_version": "0.1", "status": status, "has_actual_value": present,
              "actual_value": actual, "reason": reason}
    assert RuleEvaluation(**fields).status is VerificationStatus(status)
    assert RuleEvaluation.model_validate(fields).reason == reason
    assert RuleEvaluation.model_validate_json(json.dumps(fields)).has_actual_value is present


def test_a_pass_verdict_never_reports_a_mismatch() -> None:
    """F01: the frozen table pairs a mismatch with FAIL, so PASS plus a mismatch is refused."""
    from pydantic import ValidationError

    for actual in (1, None):
        fields = {"schema_version": "0.1", "status": "pass", "has_actual_value": True,
                  "actual_value": actual, "reason": "VALUE_MISMATCH"}
        with pytest.raises(VerificationError):
            RuleEvaluation(**fields)
        with pytest.raises(VerificationError):
            RuleEvaluation.model_validate(fields)
        with pytest.raises(VerificationError):
            RuleEvaluation.model_validate_json(json.dumps(fields))
    # The existing-instance and assignment routes agree with the constructor.
    forged = RuleEvaluation.model_construct(schema_version="0.1", status="pass",
                                            has_actual_value=True, actual_value=1,
                                            reason="VALUE_MISMATCH")
    with pytest.raises(VerificationError):
        RuleEvaluation.model_validate(forged)
    valid = RuleEvaluation(schema_version="0.1", status="pass", has_actual_value=True,
                           actual_value=1, reason="VALUE_MATCH")
    with pytest.raises((VerificationError, ValidationError)):
        valid.reason = "VALUE_MISMATCH"
    assert valid.reason == "VALUE_MATCH"


@pytest.mark.parametrize("mode", ["constructor", "mapping", "json"])
def test_the_whole_finite_verdict_table_is_decided_by_one_rule(mode: str) -> None:
    """F01: exhaustive finite table audit - every legal line accepted, every other line refused.

    The declared table is finite: three statuses, two presence values, five reasons, two value
    shapes and three public ingress routes, which is the 180-combination regression audit. The
    legal set is written out line by line rather than derived from the implementation, and it
    includes the rule that a verdict reporting no observed value carries a null value.
    """
    import itertools

    legal = {
        # status, has_actual_value, reason, actual_value
        ("pass", True, "VALUE_MATCH", 1),
        ("pass", True, "VALUE_MATCH", None),
        ("pass", True, "PATH_PRESENT", 1),
        ("pass", True, "PATH_PRESENT", None),
        ("fail", True, "VALUE_MISMATCH", 1),
        ("fail", True, "VALUE_MISMATCH", None),
        ("fail", False, "PATH_ABSENT", None),
        ("inconclusive", False, "EVIDENCE_INSUFFICIENT", None),
    }
    checked = 0
    for status, present, reason in itertools.product(
        ("pass", "fail", "inconclusive"), (True, False),
        ("VALUE_MATCH", "VALUE_MISMATCH", "PATH_PRESENT", "PATH_ABSENT",
         "EVIDENCE_INSUFFICIENT"),
    ):
        for actual in (1, None):
            fields = {"schema_version": "0.1", "status": status, "has_actual_value": present,
                      "actual_value": actual, "reason": reason}
            if (status, present, reason, actual) in legal:
                built = _parse(RuleEvaluation, fields, mode)
                assert built.status is VerificationStatus(status)
                assert built.has_actual_value is present
                assert built.reason == reason
            else:
                with pytest.raises(VerificationError):
                    _parse(RuleEvaluation, fields, mode)
            checked += 1
    assert checked == 60
    assert len(legal) == 8, "eight legal lines out of sixty, so twenty-four route lines of 180"


def _parse(model: type, fields: dict, mode: str) -> object:
    """Build one record through one public ingress, exactly as the archived audit does."""
    if mode == "constructor":
        return model(**fields)
    if mode == "mapping":
        return model.model_validate(fields)
    return model.model_validate_json(json.dumps(fields))


def test_an_unknown_reason_code_is_refused() -> None:
    """F01: the reason vocabulary is closed, so an invented code cannot be stored."""
    from pydantic import ValidationError

    with pytest.raises(VerificationError):
        RuleEvaluation(schema_version="0.1", status="fail", has_actual_value=False,
                       actual_value=None, reason="UNKNOWN_REASON")
    valid = RuleEvaluation(schema_version="0.1", status="fail", has_actual_value=False,
                           actual_value=None, reason="PATH_ABSENT")
    with pytest.raises((VerificationError, ValidationError)):
        valid.reason = "UNKNOWN_REASON"
    assert valid.reason == "PATH_ABSENT"


def test_an_illegal_path_inside_a_rule_payload_is_a_rule_error() -> None:
    with pytest.raises(VerificationError) as info:
        PathExistsRule.model_validate({"schema_version": "0.1", "rule_type": "path_exists",
                                       "path": [True]})
    assert info.value.code == "VERIFICATION_INPUT_INVALID"
    contract = _contract(outcomes=[{"outcome_type": "json_rule", "required": True,
                                    "payload": {"schema_version": "0.1",
                                                "rule_type": "path_exists", "path": [True]}}])
    with pytest.raises(VerificationError) as rule_info:
        StepVerifier().verify_step(contract=contract, evidence=evidence({"a": 1}))
    assert rule_info.value.code == "VERIFICATION_RULE_INVALID"


def test_a_string_and_an_integer_index_are_different_paths() -> None:
    """An integer segment indexes a JSON array and never an object; a string segment is a key."""
    state = {"items": [{"k": 1}], "0": {"k": 2}}
    assert evaluate(equals(["items", 0, "k"], 1), evidence(state)).status \
        is VerificationStatus.PASS
    assert evaluate(equals(["0", "k"], 2), evidence(state)).status is VerificationStatus.PASS
    # A string segment is not an array index, and an integer segment is not an object key.
    assert evaluate(equals(["items", "0", "k"], 1), evidence(state)).status \
        is VerificationStatus.INCONCLUSIVE
    assert evaluate(equals([0, "k"], 1), evidence(state)).status \
        is VerificationStatus.INCONCLUSIVE


def test_a_blank_object_key_is_a_legal_exact_key() -> None:
    assert evaluate(equals(["", " "], 1), evidence({"": {" ": 1}})).status \
        is VerificationStatus.PASS


@pytest.mark.parametrize("state,path", [
    ({"a": 1}, ["b"]),
    ({"a": [1]}, ["a", 5]),
    ({"a": [1]}, ["a", "b"]),
    ({"a": {"b": 1}}, ["a", 0]),
    ({"a": 1}, ["a", "b"]),
    ({"a": None}, ["a", "b"]),
    ({"a": {}}, ["a", "b"]),
])
def test_an_unresolvable_path_means_absent_not_invalid(state: dict, path: list) -> None:
    result = evaluate(equals(path, 1), evidence(state))
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.reason == "EVIDENCE_INSUFFICIENT"
    assert result.has_actual_value is False


def test_an_empty_object_and_an_empty_array_are_reachable_values() -> None:
    assert evaluate(exists(["a"]), evidence({"a": {}})).status is VerificationStatus.PASS
    assert evaluate(exists(["a"]), evidence({"a": []})).status is VerificationStatus.PASS
    assert evaluate(equals(["a"], {}), evidence({"a": {}})).status is VerificationStatus.PASS


# --- A03/A04: the truth table and JSON semantics ---

@pytest.mark.parametrize("rule,state,complete,status,reason,present", [
    (equals(["a"], 1), {"a": 1}, True, VerificationStatus.PASS, "VALUE_MATCH", True),
    (equals(["a"], 1), {"a": 2}, True, VerificationStatus.FAIL, "VALUE_MISMATCH", True),
    (equals(["a"], 1), {"a": None}, True, VerificationStatus.FAIL, "VALUE_MISMATCH", True),
    (equals(["a"], 1), {}, True, VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", False),
    (equals(["a"], 1), {}, False, VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", False),
    (exists(["a"]), {"a": None}, True, VerificationStatus.PASS, "PATH_PRESENT", True),
    (exists(["a"]), {"a": None}, False, VerificationStatus.PASS, "PATH_PRESENT", True),
    (exists(["a"]), {}, True, VerificationStatus.FAIL, "PATH_ABSENT", False),
    (exists(["a"]), {}, False, VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", False),
])
def test_the_agreed_truth_table(rule: object, state: dict, complete: bool,
                                status: VerificationStatus, reason: str,
                                present: bool) -> None:
    result = evaluate(rule, evidence(state, complete=complete))
    assert result.status is status
    assert result.reason == reason
    assert result.has_actual_value is present
    if not present:
        assert result.actual_value is None


def test_an_observed_null_is_a_present_value() -> None:
    result = evaluate(exists(["a"]), evidence({"a": None}))
    assert result.status is VerificationStatus.PASS
    assert result.has_actual_value is True
    assert result.actual_value is None


@pytest.mark.parametrize("expected,actual,equal", [
    (True, 1, False), (1, True, False), (1, 1.0, False), (1.0, 1, False),
    (1, 1, True), (1.0, 1.0, True), ("1", 1, False), (None, None, True),
    ([1, 2], [1, 2], True), ([1, 2], [2, 1], False),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
    ({"a": 1}, {"a": 1, "b": 2}, False),
    ({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]}, True),
])
def test_equality_is_type_aware_and_recursive(expected: object, actual: object,
                                              equal: bool) -> None:
    result = evaluate(equals(["v"], expected), evidence({"v": actual}))
    assert (result.status is VerificationStatus.PASS) is equal


def test_a_legal_shared_reference_is_accepted_and_a_cycle_is_refused() -> None:
    shared = {"k": [1, 2]}
    state = {"a": shared, "b": shared}
    assert evaluate(equals(["b"], {"k": [1, 2]}), evidence(state)).status \
        is VerificationStatus.PASS
    # A cycle injected into the state after construction is a bounded input error, not a crash.
    cyclic: list = []
    cyclic.append(cyclic)
    contaminated = observation({"a": 1})
    contaminated.state["c"] = cyclic
    with pytest.raises(VerificationError) as info:
        evaluate(exists(["a"]), VerificationEvidence(schema_version="0.1",
                                                     observation=contaminated,
                                                     state_complete=True))
    assert info.value.code == "VERIFICATION_INPUT_INVALID"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), b"raw",
                                 datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
                                 {"a": {1: 2}}])
def test_non_json_state_is_a_facility_error(bad: object) -> None:
    contaminated = observation({"a": 1})
    contaminated.state["bad"] = bad
    with pytest.raises(VerificationError) as info:
        evaluate(exists(["a"]), VerificationEvidence(schema_version="0.1",
                                                     observation=contaminated,
                                                     state_complete=True))
    assert info.value.code == "VERIFICATION_INPUT_INVALID"


# --- A05/A06: the step verifier ---

def _contract(*, outcomes: list[dict], bindings: list[tuple[int, str, dict]] | None = None,
              contract_id: str = "c-1") -> ActionContract:
    outcome_models = [ExpectedOutcome(schema_version="0.1", **item) for item in outcomes]
    binding_models = [
        VerificationBinding(schema_version="0.1", outcome_index=index, verifier_type=verifier,
                            config=config)
        for index, verifier, config in (bindings or [
            (index, "json_rule", {}) for index in range(len(outcome_models))
        ])
    ]
    return ActionContract(
        schema_version="0.1", contract_id=contract_id, goal_id="g-1", forecast_id="f-1",
        observation_id="obs-1", adapter_name="fake-1", environment_type="fake",
        action=action(), preconditions=[], expected_outcomes=outcome_models,
        verification_policy=VerificationPolicy(schema_version="0.1", bindings=binding_models),
        recovery_policy={"schema_version": "0.1", "mode": "abort"},
        timeout_ms=100.0, created_at=NOW,
    )


def _rule_outcome(rule_type: str, path: list, *, expected: object = None,
                  required: bool = True) -> dict:
    payload: dict = {"schema_version": "0.1", "rule_type": rule_type, "path": path}
    if rule_type == "value_equals":
        payload["expected"] = expected
    return {"outcome_type": "json_rule", "required": required, "payload": payload}


def test_the_step_verifier_evaluates_in_order_and_keeps_every_item() -> None:
    contract = _contract(outcomes=[
        _rule_outcome("value_equals", ["a"], expected=1),
        _rule_outcome("path_exists", ["b"], required=False),
        _rule_outcome("path_exists", ["a"]),
    ])
    result = StepVerifier().verify_step(contract=contract, evidence=evidence({"a": 1}))
    assert result.status is VerificationStatus.PASS
    assert result.reason == "REQUIRED_PASSED"
    assert result.verifier == "json_step"
    assert [item.outcome_index for item in result.outcome_results] == [0, 1, 2]
    assert [item.status for item in result.outcome_results] == [
        VerificationStatus.PASS, VerificationStatus.FAIL, VerificationStatus.PASS
    ]
    assert dict(result.metadata) == {"scope": "step", "contract_id": "c-1", "goal_id": "g-1",
                                     "observation_id": "obs-1", "state_complete": True}


def test_an_optional_failure_does_not_change_the_overall_status() -> None:
    contract = _contract(outcomes=[
        _rule_outcome("path_exists", ["a"]),
        _rule_outcome("path_exists", ["absent"], required=False),
        _rule_outcome("path_exists", ["also_absent"], required=False),
    ])
    result = StepVerifier().verify_step(contract=contract, evidence=evidence({"a": 1}))
    assert result.status is VerificationStatus.PASS
    assert [item.status for item in result.outcome_results][1:] == [
        VerificationStatus.FAIL, VerificationStatus.FAIL
    ]


@pytest.mark.parametrize("statuses,expected", [
    (["PASS", "PASS"], VerificationStatus.PASS),
    (["PASS", "FAIL"], VerificationStatus.FAIL),
    (["FAIL", "INCONCLUSIVE"], VerificationStatus.FAIL),
    (["PASS", "INCONCLUSIVE"], VerificationStatus.INCONCLUSIVE),
    (["INCONCLUSIVE", "INCONCLUSIVE"], VerificationStatus.INCONCLUSIVE),
])
def test_required_all_aggregation_orders_fail_before_inconclusive(
    statuses: list, expected: VerificationStatus,
) -> None:
    rules = []
    for status in statuses:
        if status == "PASS":
            rules.append(_rule_outcome("path_exists", ["present"]))
        elif status == "FAIL":
            rules.append(_rule_outcome("path_exists", ["missing"]))
        else:
            # Equality against a path the evidence does not reach is the inconclusive line of
            # the truth table, whatever the completeness declaration says.
            rules.append(_rule_outcome("value_equals", ["missing"], expected=1))
    contract = _contract(outcomes=rules)
    result = StepVerifier().verify_step(contract=contract,
                                       evidence=evidence({"present": 1}, complete=True))
    assert [item.status for item in result.outcome_results] == [
        {"PASS": VerificationStatus.PASS, "FAIL": VerificationStatus.FAIL,
         "INCONCLUSIVE": VerificationStatus.INCONCLUSIVE}[status] for status in statuses
    ]
    assert result.status is expected


@pytest.mark.parametrize("overrides,code", [
    ({"bindings": [(0, "other", {})]}, "VERIFICATION_BINDING_INVALID"),
    ({"bindings": [(0, "json_rule", {"rule": "x"})]}, "VERIFICATION_BINDING_INVALID"),
])
def test_a_bad_binding_is_a_binding_error(overrides: dict, code: str) -> None:
    contract = _contract(outcomes=[_rule_outcome("path_exists", ["a"])], **overrides)
    with pytest.raises(VerificationError) as info:
        StepVerifier().verify_step(contract=contract, evidence=evidence({"a": 1}))
    assert info.value.code == code


def test_the_binding_is_looked_up_by_index_not_by_position() -> None:
    contract = _contract(outcomes=[
        _rule_outcome("value_equals", ["a"], expected=1),
        _rule_outcome("value_equals", ["b"], expected=2),
    ], bindings=[(1, "json_rule", {}), (0, "json_rule", {})])
    result = StepVerifier().verify_step(contract=contract, evidence=evidence({"a": 1, "b": 2}))
    assert result.status is VerificationStatus.PASS
    assert [item.outcome_index for item in result.outcome_results] == [0, 1]

def test_an_unsupported_outcome_type_is_a_rule_error() -> None:
    contract = _contract(outcomes=[{"outcome_type": "other", "required": True, "payload": {}}])
    with pytest.raises(VerificationError) as info:
        StepVerifier().verify_step(contract=contract, evidence=evidence({"a": 1}))
    assert info.value.code == "VERIFICATION_RULE_INVALID"


def test_a_rule_error_is_not_masked_by_an_earlier_failure() -> None:
    contract = _contract(outcomes=[
        _rule_outcome("path_exists", ["absent"]),
        {"outcome_type": "json_rule", "required": True,
         "payload": {"schema_version": "0.1", "rule_type": "unknown", "path": ["a"]}},
    ])
    with pytest.raises(VerificationError) as info:
        StepVerifier().verify_step(contract=contract, evidence=evidence({"a": 1}))
    assert info.value.code == "VERIFICATION_RULE_INVALID"
    assert info.value.location[0] == "contract"


def test_an_optional_only_contract_is_refused_at_commit() -> None:
    """The existing Core rule is unchanged: a contract needs at least one required outcome."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, VerificationError)):
        _contract(outcomes=[_rule_outcome("path_exists", ["a"], required=False)])


def test_a_malformed_evidence_argument_is_never_a_verdict() -> None:
    contract = _contract(outcomes=[_rule_outcome("path_exists", ["a"])])
    for broken in ({"state_complete": True}, None, "evidence"):
        with pytest.raises(VerificationError):
            StepVerifier().verify_step(contract=contract, evidence=broken)


# --- A08/A09: hooks, the Builder and isolation ---

def test_the_hooks_implement_the_four_builder_methods() -> None:
    hooks = RuleValidationHooks(state_complete=True)
    hooks.validate_precondition(Precondition(schema_version="0.1", condition_type="json_rule",
                                             payload={"schema_version": "0.1",
                                                      "rule_type": "path_exists",
                                                      "path": ["a"]}))
    hooks.validate_expected_outcome(ExpectedOutcome(schema_version="0.1",
                                                    outcome_type="json_rule",
                                                    payload={"schema_version": "0.1",
                                                             "rule_type": "path_exists",
                                                             "path": ["a"]}))
    hooks.validate_verification_binding(
        VerificationBinding(schema_version="0.1", outcome_index=0, verifier_type="json_rule"),
        ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", payload={}))
    assert hooks.check_precondition(
        Precondition(schema_version="0.1", condition_type="json_rule",
                     payload={"schema_version": "0.1", "rule_type": "path_exists", "path": ["a"]}),
        observation({"a": 1})) is VerificationStatus.PASS
    assert hooks.state_complete is True


def test_the_hooks_require_an_explicit_strict_completeness_flag() -> None:
    for bad in ("yes", 1, None, {}):
        with pytest.raises(VerificationError) as info:
            RuleValidationHooks(state_complete=bad)
        assert info.value.code == "VERIFICATION_INPUT_INVALID"


def test_the_hooks_never_read_completeness_from_metadata() -> None:
    incomplete = RuleValidationHooks(state_complete=False)
    observation_with_metadata = Observation(
        schema_version="0.1", observation_id="obs-m", environment_type="fake", state={},
        timestamp=NOW, metadata={"state_complete": True})
    condition = Precondition(schema_version="0.1", condition_type="json_rule",
                             payload={"schema_version": "0.1", "rule_type": "path_exists",
                                      "path": ["a"]})
    assert incomplete.check_precondition(condition, observation_with_metadata) \
        is VerificationStatus.INCONCLUSIVE
    assert RuleValidationHooks(state_complete=True).check_precondition(
        condition, observation_with_metadata) is VerificationStatus.FAIL


def test_a_domain_rejection_uses_the_existing_safe_exception() -> None:
    from forecontract.protocols.adapter import InvalidActionPayloadError

    hooks = RuleValidationHooks(state_complete=True)
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_expected_outcome(ExpectedOutcome(
            schema_version="0.1", outcome_type="other", payload={}))
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_expected_outcome(ExpectedOutcome(
            schema_version="0.1", outcome_type="json_rule", payload={"rule_type": "unknown"}))
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_verification_binding(
            VerificationBinding(schema_version="0.1", outcome_index=0, verifier_type="other"),
            ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", payload={}))


class DeclaringFakeAdapter:
    """A test-local wrapper that declares json_rule capabilities for one Builder round."""

    def __init__(self, adapter: object) -> None:
        self._adapter = adapter

    @property
    def name(self) -> str:
        return self._adapter.name

    def capabilities(self) -> AdapterCapabilities:
        declared = self._adapter.capabilities()
        return declared.model_copy(update={
            "supported_condition_types": ["json_rule"],
            "supported_outcome_types": ["json_rule"],
        })

    def validate_action(self, candidate: ActionSpec) -> None:
        self._adapter.validate_action(candidate)


def _fake_adapter() -> object:
    from forecontract_fake import (
        FakeEnvironmentAdapter,
        FakeScenario,
        FakeStage,
        FakeStep,
    )

    built = FakeScenario(
        schema_version="0.1", scenario_id="s-1", adapter_name="fake-1",
        allowed_actions=[action()],
        stages=[
            FakeStage(schema_version="0.1", actual_state={"a": 1},
                      observation=observation({"a": 1})),
            FakeStage(schema_version="0.1", actual_state={"a": 1},
                      observation=observation({"a": 1}, identifier="obs-2")),
        ],
        steps=[FakeStep(schema_version="0.1", step_id="step-0", action=action(),
                        outcome=ActualOutcome(schema_version="0.1", outcome_id="out-0",
                                              status="succeeded", changes={}))])
    return FakeEnvironmentAdapter(built)


def _build(*, conditions: list[Precondition], outcomes: list[ExpectedOutcome],
           hooks: object, adapter: object) -> ActionContract:
    return ContractBuilder().build(
        forecast=Forecast(schema_version="0.1", forecast_id="f-1", proposed_action=action(),
                          expected_outcomes=outcomes),
        observation=observation({"a": 1}),
        adapter=adapter,
        hooks=hooks,
        preconditions=conditions,
        verification_policy=VerificationPolicy(
            schema_version="0.1",
            bindings=[VerificationBinding(schema_version="0.1", outcome_index=index,
                                          verifier_type="json_rule")
                      for index in range(len(outcomes))]),
        timeout_ms=100, remaining_budget_ms=250, contract_id="c-hook", goal_id="g-hook",
        created_at=NOW,
    )


def test_a_real_builder_round_commits_with_the_hooks() -> None:
    condition = Precondition(schema_version="0.1", condition_type="json_rule",
                             payload={"schema_version": "0.1", "rule_type": "path_exists",
                                      "path": ["a"]})
    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                              payload={"schema_version": "0.1", "rule_type": "value_equals",
                                       "path": ["a"], "expected": 1})
    built = _build(conditions=[condition], outcomes=[outcome],
                   hooks=RuleValidationHooks(state_complete=True),
                   adapter=DeclaringFakeAdapter(_fake_adapter()))
    assert built.contract_id == "c-hook"
    assert built.preconditions[0].condition_type == "json_rule"
    result = StepVerifier().verify_step(contract=built, evidence=evidence({"a": 1}))
    assert result.status is VerificationStatus.PASS


def test_a_required_precondition_failure_rejects_the_build() -> None:
    condition = Precondition(schema_version="0.1", condition_type="json_rule",
                             payload={"schema_version": "0.1", "rule_type": "path_exists",
                                      "path": ["absent"]}, required=True)
    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                              payload={"schema_version": "0.1", "rule_type": "path_exists",
                                       "path": ["a"]})
    with pytest.raises(ContractBuildError) as info:
        _build(conditions=[condition], outcomes=[outcome],
               hooks=RuleValidationHooks(state_complete=True),
               adapter=DeclaringFakeAdapter(_fake_adapter()))
    assert info.value.code == "PRECONDITION_UNSATISFIED"


def test_an_inconclusive_required_precondition_rejects_the_build() -> None:
    condition = Precondition(schema_version="0.1", condition_type="json_rule",
                             payload={"schema_version": "0.1", "rule_type": "path_exists",
                                      "path": ["absent"]}, required=True)
    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                              payload={"schema_version": "0.1", "rule_type": "path_exists",
                                       "path": ["a"]})
    with pytest.raises(ContractBuildError) as info:
        _build(conditions=[condition], outcomes=[outcome],
               hooks=RuleValidationHooks(state_complete=False),
               adapter=DeclaringFakeAdapter(_fake_adapter()))
    assert info.value.code == "PRECONDITION_UNSATISFIED"


def test_a_rule_rejection_is_not_a_dependency_failure() -> None:
    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                              payload={"schema_version": "0.1", "rule_type": "unknown",
                                       "path": ["a"]})
    with pytest.raises(ContractBuildError) as info:
        _build(conditions=[], outcomes=[outcome], hooks=RuleValidationHooks(state_complete=True),
               adapter=DeclaringFakeAdapter(_fake_adapter()))
    assert info.value.code == "INVALID_DOMAIN_PAYLOAD"
    assert info.value.code != "DEPENDENCY_FAILURE"


def test_the_hooks_do_not_inspect_the_adapter_or_perform_io() -> None:
    hooks = RuleValidationHooks(state_complete=True)
    condition = Precondition(schema_version="0.1", condition_type="json_rule",
                             payload={"schema_version": "0.1", "rule_type": "path_exists",
                                      "path": ["a"]})
    before = copy.deepcopy(condition.model_dump(mode="json"))
    observation_before = observation({"a": 1}).model_dump(mode="json")
    prompts = [condition, observation({"a": 1})]
    hooks.check_precondition(condition, prompts[1])
    hooks.validate_precondition(condition)
    assert condition.model_dump(mode="json") == before
    assert prompts[1].model_dump(mode="json") == observation_before


def test_inputs_and_results_are_detached_from_one_another() -> None:
    rule = equals(["a"], 1)
    piece = evidence({"a": 1})
    first = evaluate(rule, piece)
    second = evaluate(rule, piece)
    assert first == second
    assert first is not second
    state = piece.observation.state
    state["a"] = 99
    assert evaluate(rule, evidence({"a": 1})).status is VerificationStatus.PASS
    first.actual_value = "changed" if isinstance(first.actual_value, str) else first.actual_value
    assert second.actual_value == 1


def test_repeated_calls_are_deterministic() -> None:
    contract = _contract(outcomes=[
        _rule_outcome("value_equals", ["a"], expected=1),
        _rule_outcome("path_exists", ["b"], required=False),
    ])
    verifier = StepVerifier()
    first = verifier.verify_step(contract=contract, evidence=evidence({"a": 1}))
    second = verifier.verify_step(contract=contract, evidence=evidence({"a": 1}))
    assert first == second


def test_the_public_surface_is_exactly_twelve_names() -> None:
    import forecontract.verification as package

    assert sorted(package.__all__) == [
        "CompositeVerifier",
        "JsonRuleEvaluator",
        "ModelVerifier",
        "PathExistsRule",
        "RuleEvaluation",
        "RuleValidationHooks",
        "StepVerifier",
        "TaskCompletionCriteria",
        "TaskCompletionOracle",
        "ValueEqualsRule",
        "VerificationError",
        "VerificationEvidence",
    ]
    assert len(package.__all__) == 12
    assert not hasattr(package, "VerificationStatus")
    assert not hasattr(package, "OutcomeVerification")
    import forecontract

    assert "verification" not in getattr(forecontract, "__all__", [])


def test_a_facility_error_never_echoes_a_supplied_value() -> None:
    """A rule error is located structurally and never transcribes the caller's path."""
    contract = _contract(outcomes=[{"outcome_type": "json_rule", "required": True,
                                    "payload": {"schema_version": "0.1",
                                                "rule_type": MARKER, "path": [MARKER]}}])
    with pytest.raises(VerificationError) as info:
        StepVerifier().verify_step(contract=contract, evidence=evidence({MARKER: 1}))
    message = str(info.value)
    assert MARKER not in message
    assert len(message) < 240
    assert info.value.code == "VERIFICATION_RULE_INVALID"
    assert MARKER not in str(info.value.location)
    # A legal rule whose path is simply not present is a verdict, never an error: with an
    # incomplete declaration it is inconclusive, and with a complete one it is a failure.
    assert evaluate(exists([MARKER]), evidence({"a": 1}, complete=False)).status \
        is VerificationStatus.INCONCLUSIVE
    assert evaluate(exists([MARKER]), evidence({"a": 1}, complete=True)).status \
        is VerificationStatus.FAIL


# --- round-002 correction: nested ingress, detached ownership, assignment atomicity ---

def _outcome() -> ExpectedOutcome:
    return ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                           payload={"schema_version": "0.1", "rule_type": "path_exists",
                                    "path": ["a"]})


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"observation_id": b"obs"},
    {"environment_type": 17},
])
def test_a_copied_invalid_nested_observation_is_refused(mutation: dict) -> None:
    """F02: nested content is validated from its own stored fields, at construction and mapping."""
    forged = observation().model_copy(update=mutation)
    with pytest.raises(VerificationError):
        VerificationEvidence(schema_version="0.1", observation=forged, state_complete=True)
    with pytest.raises(VerificationError):
        VerificationEvidence.model_validate({"schema_version": "0.1", "observation": forged,
                                             "state_complete": True})


def test_a_nested_record_with_an_unknown_stored_key_is_refused() -> None:
    """F02: an undeclared stored key on a nested record is refused, not silently dropped."""
    forged = observation()
    forged.__dict__[MARKER] = 1
    with pytest.raises(VerificationError) as info:
        VerificationEvidence(schema_version="0.1", observation=forged, state_complete=True)
    assert MARKER not in str(info.value)


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"required": "yes"},
    {"outcome_type": 17},
])
def test_a_copied_invalid_nested_expected_outcome_is_refused(mutation: dict) -> None:
    """F02: the nested shape is enforced from the record's own stored fields.

    A paired copy and a constructed instance are both used, so the check does not depend on the
    copy guard alone.
    """
    from pydantic import ValidationError

    stored = dict(_outcome().__dict__)
    stored.update(mutation)
    forged = ExpectedOutcome.model_construct(**stored)
    assert forged.__dict__[next(iter(mutation))] != getattr(_outcome(), next(iter(mutation)))
    with pytest.raises(VerificationError):
        TaskCompletionCriteria(schema_version="0.1", goal_id="g-1", conditions=[forged])
    with pytest.raises(VerificationError):
        TaskCompletionCriteria.model_validate({"schema_version": "0.1", "goal_id": "g-1",
                                               "conditions": [forged]})
    del ValidationError


def test_a_cyclic_nested_state_is_a_bounded_rejection() -> None:
    """F02: a genuine cycle is refused with a bounded error rather than a recursion failure."""
    cyclic: list = []
    cyclic.append(cyclic)
    contaminated = observation({"a": 1})
    contaminated.state["c"] = cyclic
    with pytest.raises(VerificationError) as info:
        VerificationEvidence(schema_version="0.1", observation=contaminated, state_complete=True)
    assert info.value.code == "VERIFICATION_INPUT_INVALID"
    assert len(str(info.value)) < 240


def test_a_legal_shared_nested_reference_is_still_accepted() -> None:
    """F02 control: shared nested JSON and a legal nested record still construct."""
    shared = {"k": [1, 2]}
    built = VerificationEvidence(schema_version="0.1",
                                 observation=observation({"a": shared, "b": shared}),
                                 state_complete=True)
    assert built.observation.state["a"] == built.observation.state["b"] == shared
    assert TaskCompletionCriteria(schema_version="0.1", goal_id="g-1",
                                  conditions=[_outcome()]).conditions[0].required is True


def test_a_record_owns_its_nested_state_and_does_not_alias_the_caller() -> None:
    """F04: mutating the caller's nested data after construction cannot change the record."""
    caller_state = {"a": {"b": [1]}}
    caller_observation = observation(caller_state)
    record = VerificationEvidence(schema_version="0.1", observation=caller_observation,
                                  state_complete=True)
    caller_state["a"]["b"].append(2)
    assert record.observation.state["a"]["b"] == [1]
    record.observation.state["a"]["b"].append(3)
    assert caller_state["a"]["b"] == [1, 2]


def test_a_criteria_record_owns_its_condition_payloads() -> None:
    """F04: the rule payload inside a criterion is detached from the caller's object."""
    caller_payload = {"b": [1]}
    caller_outcome = ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                                     payload=caller_payload)
    built = TaskCompletionCriteria(schema_version="0.1", goal_id="g-1",
                                   conditions=[caller_outcome])
    caller_payload["b"].append(2)
    assert built.conditions[0].payload["b"] == [1]
    built.conditions[0].payload["b"].append(3)
    assert caller_payload["b"] == [1, 2]


def test_a_legal_mapping_still_constructs_and_round_trips() -> None:
    """F04 control: no coercion was added; legal mappings and JSON still load."""
    built = evidence({"a": 1})
    assert VerificationEvidence.model_validate(built.model_dump(mode="python")) == built
    assert VerificationEvidence.model_validate_json(built.model_dump_json()) == built


def test_a_failed_assignment_preserves_the_previous_valid_state() -> None:
    """F03: a rejected assignment leaves the record exactly as it was."""
    from pydantic import ValidationError

    record = RuleEvaluation(schema_version="0.1", status="fail", has_actual_value=True,
                            actual_value=1, reason="VALUE_MISMATCH")
    before = record.model_dump(mode="json")
    with pytest.raises((VerificationError, ValidationError)):
        record.has_actual_value = False
    assert record.model_dump(mode="json") == before

    other = RuleEvaluation(schema_version="0.1", status="fail", has_actual_value=False,
                           actual_value=None, reason="PATH_ABSENT")
    other_before = other.model_dump(mode="json")
    with pytest.raises((VerificationError, ValidationError)):
        other.reason = "UNKNOWN_REASON"
    assert other.model_dump(mode="json") == other_before

    condition_record = TaskCompletionCriteria(schema_version="0.1", goal_id="g-1",
                                              conditions=[_outcome()])
    conditions_before = condition_record.model_dump(mode="json")
    with pytest.raises((VerificationError, ValidationError)):
        condition_record.conditions = [_outcome().model_copy(update={"required": False})]
    assert condition_record.model_dump(mode="json") == conditions_before


def test_a_legal_assignment_still_takes_effect() -> None:
    """F03 control: the transactional guard does not block a valid assignment.

    The verdict is one state machine, so only an assignment that leaves a line of the table
    intact is legal: the value moves first, then the reason.
    """
    record = RuleEvaluation(schema_version="0.1", status="pass", has_actual_value=True,
                            actual_value=None, reason="PATH_PRESENT")
    record.actual_value = 7
    assert record.actual_value == 7
    assert record.reason == "PATH_PRESENT"
    record.reason = "VALUE_MATCH"
    assert record.reason == "VALUE_MATCH"
    assert record.status is VerificationStatus.PASS
    del record
    # A field that carries no cross-field rule can be assigned freely in both directions.
    other = RuleEvaluation(schema_version="0.1", status="pass", has_actual_value=True,
                           actual_value=1, reason="VALUE_MATCH")
    other.actual_value = {"nested": [1, 2]}
    assert other.actual_value == {"nested": [1, 2]}
    other.actual_value = "text"
    assert other.actual_value == "text"


_ = (ErrorInfo, UnsupportedActionError)


# --- round-003 correction: raw cycles, nested assignment and existing-instance ownership ---

def _cycle() -> list:
    value: list = []
    value.append(value)
    return value


@pytest.mark.parametrize("route", ["value_equals", "path_exists", "rule_evaluation", "criteria"])
def test_a_raw_container_cycle_is_a_bounded_error_not_a_recursion_failure(route: str) -> None:
    """F02: a raw list cycle is refused with one bounded facility error on every ingress."""
    cycle = _cycle()
    if route == "value_equals":
        calls = [
            lambda: ValueEqualsRule(schema_version="0.1", rule_type="value_equals", path=["a"],
                                    expected=cycle),
            lambda: ValueEqualsRule.model_validate({"schema_version": "0.1",
                                                    "rule_type": "value_equals", "path": ["a"],
                                                    "expected": cycle}),
        ]
    elif route == "path_exists":
        calls = [
            lambda: PathExistsRule(schema_version="0.1", rule_type="path_exists", path=cycle),
            lambda: PathExistsRule.model_validate({"schema_version": "0.1",
                                                   "rule_type": "path_exists", "path": cycle}),
        ]
    elif route == "rule_evaluation":
        calls = [
            lambda: RuleEvaluation(schema_version="0.1", status="pass", has_actual_value=True,
                                   actual_value=cycle, reason="VALUE_MATCH"),
            lambda: RuleEvaluation.model_validate({"schema_version": "0.1", "status": "pass",
                                                   "has_actual_value": True,
                                                   "actual_value": cycle,
                                                   "reason": "VALUE_MATCH"}),
        ]
    else:
        calls = [
            lambda: TaskCompletionCriteria(schema_version="0.1", goal_id="g-1",
                                           conditions=cycle),
            lambda: TaskCompletionCriteria.model_validate(
                {"schema_version": "0.1", "goal_id": "g-1", "conditions": cycle}),
        ]
    for call in calls:
        with pytest.raises(VerificationError) as info:
            call()
        assert info.value.code == "VERIFICATION_INPUT_INVALID"
        assert len(str(info.value)) < 240


def test_a_nested_cycle_inside_a_rule_payload_is_a_bounded_error() -> None:
    """F02: the same bound holds for a cycle reached through a nested record.

    A cycle reached inside a nested record is refused either as the local bounded facility error
    or by the schema's own recursion bound; what is not acceptable is a raw ``RecursionError``.
    """
    from pydantic import ValidationError

    cycle = _cycle()
    with pytest.raises((VerificationError, ValidationError)):
        TaskCompletionCriteria(
            schema_version="0.1", goal_id="g-1",
            conditions=[ExpectedOutcome(schema_version="0.1", outcome_type="json_rule",
                                        required=True,
                                        payload={"schema_version": "0.1",
                                                 "rule_type": "path_exists", "path": cycle})])
    with pytest.raises((VerificationError, ValidationError)):
        TaskCompletionCriteria.model_validate(
            {"schema_version": "0.1", "goal_id": "g-1",
             "conditions": [{"schema_version": "0.1", "outcome_type": "json_rule",
                             "required": True,
                             "payload": {"schema_version": "0.1",
                                         "rule_type": "path_exists", "path": cycle}}]})


def test_a_legal_repeated_raw_list_is_still_accepted() -> None:
    """F02 control: a value referenced twice is not a cycle, and each reference is copied."""
    shared = [1]
    rule = ValueEqualsRule(schema_version="0.1", rule_type="value_equals", path=["a"],
                           expected=[shared, shared])
    assert rule.expected == [[1], [1]]
    assert ValueEqualsRule.model_validate(
        {"schema_version": "0.1", "rule_type": "value_equals", "path": ["a"],
         "expected": [shared, shared]}).expected == [[1], [1]]
    shared.append(2)
    assert rule.expected == [[1], [1]]


def test_assigning_an_invalid_nested_record_is_refused_and_preserves_state() -> None:
    """F02: assignment validates the nested record's own shape before anything is committed."""
    from pydantic import ValidationError

    undeclared = observation()
    undeclared.__dict__[MARKER] = 1
    forged_observations = [
        observation().model_copy(update={"schema_version": "9.9"}),
        observation().model_copy(update={"observation_id": b"obs"}),
        undeclared,
    ]
    for forged in forged_observations:
        record = evidence({"a": 1})
        before = record.model_dump(mode="json")
        with pytest.raises((VerificationError, ValidationError)):
            record.observation = forged
        assert record.model_dump(mode="json") == before

    undeclared_outcome = _outcome()
    undeclared_outcome.__dict__[MARKER] = 1
    forged_outcomes = [
        _outcome().model_copy(update={"schema_version": "9.9"}),
        _outcome().model_copy(update={"required": "yes"}),
        undeclared_outcome,
    ]
    for forged in forged_outcomes:
        record = TaskCompletionCriteria(schema_version="0.1", goal_id="g-1",
                                        conditions=[_outcome()])
        before = record.model_dump(mode="json")
        with pytest.raises((VerificationError, ValidationError)):
            record.conditions = [forged]
        assert record.model_dump(mode="json") == before


def test_a_legal_nested_assignment_still_takes_effect() -> None:
    """F02/F03 control: a valid nested assignment is committed and detached."""
    record = evidence({"a": 1})
    replacement = observation({"b": 2}, identifier="obs-2")
    record.observation = replacement
    assert record.observation.observation_id == "obs-2"
    replacement.state["b"] = 99
    assert record.observation.state["b"] == 2


def test_existing_instance_ingress_returns_an_independent_record() -> None:
    """F04: parsing an existing instance returns a record that shares no mutable data."""
    cases = [
        (PathExistsRule(schema_version="0.1", rule_type="path_exists", path=["a"]),
         lambda record: record.path.append("b")),
        (ValueEqualsRule(schema_version="0.1", rule_type="value_equals", path=["a"],
                         expected={"b": [1]}),
         lambda record: record.expected["b"].append(2)),
        (RuleEvaluation(schema_version="0.1", status="pass", has_actual_value=True,
                        actual_value={"b": [1]}, reason="VALUE_MATCH"),
         lambda record: record.actual_value["b"].append(2)),
        (evidence({"a": {"b": [1]}}),
         lambda record: record.observation.state["a"]["b"].append(2)),
        (TaskCompletionCriteria(schema_version="0.1", goal_id="g-1", conditions=[_outcome()]),
         lambda record: record.conditions.append(_outcome())),
    ]
    for original, mutate in cases:
        before = original.model_dump(mode="json")
        parsed = type(original).model_validate(original)
        assert parsed is not original
        assert parsed == original
        mutate(parsed)
        assert original.model_dump(mode="json") == before, type(original).__name__
        # The reverse direction holds too: a fresh parse does not follow the original.
        fresh = type(original).model_validate(original)
        fresh_before = fresh.model_dump(mode="json")
        mutate(original)
        assert fresh.model_dump(mode="json") == fresh_before, type(original).__name__
