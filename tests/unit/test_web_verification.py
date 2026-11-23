"""Acceptance tests for the Web step verifier and the independent task oracle (W01/W06/W07).

The two entry points are kept separate on purpose: the step verifier reads only the committed
contract's expected outcomes, while the task oracle reads only independently supplied
completion criteria. Both share the five-rule truth table, evaluate every valid item in order
and aggregate required items only (``FAIL`` first, then ``INCONCLUSIVE``, otherwise ``PASS``).

Every expected outcome needs exactly one ``web_rule`` binding with an empty configuration. The
tests drive an unordered binding list as a positive control and missing, repeated, out-of-range,
wrong-verifier and non-empty-config bindings as facility failures. An unknown or malformed
optional item is never ignored, and a required failure can never be softened by an optional
pass. The measured Web values stay in the evidence: each item carries
``has_actual_value=False``/``actual_value=None`` and the result metadata names the rule type and
target position instead of copying the value into the result.
"""

from __future__ import annotations

import datetime
import inspect
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts.models import (
    ActionContract,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import ActionSpec, ExpectedOutcome
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract_web.errors import WebValidationError
from forecontract_web.models import TestIdTarget
from forecontract_web.rules import (
    WebRuleEvaluator,
    WebRuleInvalid,
    WebVerificationInputInvalid,
)
from forecontract_web.verification import (
    WebStepVerifier,
    WebTaskCompletionOracle,
)
from forecontract_web.verification import (
    WebVerificationInputInvalid as ImportedInputInvalid,
)

NOW = datetime.datetime(2026, 10, 29, 12, 0, tzinfo=datetime.timezone.utc)
PAGE = "http://127.0.0.1:8000/checkout"
MARKER = "SYNTHETIC_WEB_VERIFICATION_MARKER_5e2"
SCHEMA = "0.1"


def id_target(test_id: str) -> TestIdTarget:
    return TestIdTarget(strategy="test_id", test_id=test_id)


def record(target, *, count=1, visible=None, text=None, value=None) -> dict:
    return {
        "target": target.model_dump(),
        "status": "collected",
        "match_count": count,
        "visible": visible,
        "text": text,
        "value": value,
        "error_code": None,
    }


def evidence(*, complete: bool = True, identifier: str = "obs-1") -> VerificationEvidence:
    state = {
        "schema_version": SCHEMA,
        "url": PAGE,
        "targets": [
            record(id_target("ok"), visible=True, text="ready now", value="seed"),
            record(id_target("hidden"), visible=False, text="hidden text", value=""),
            record(id_target("zero"), count=0),
            record(id_target("many"), count=2),
        ],
    }
    return VerificationEvidence(
        schema_version=SCHEMA,
        observation=Observation(
            schema_version=SCHEMA,
            observation_id=identifier,
            environment_type="web",
            state=state,
            timestamp=NOW,
        ),
        state_complete=complete,
    )


def outcome(status: str, *, required: bool = True) -> ExpectedOutcome:
    """One expected outcome whose verdict is exactly the requested status."""
    if status == "PASS":
        rule_type, payload = "web.element_visible", {"target": id_target("ok").model_dump()}
    elif status == "FAIL":
        rule_type, payload = "web.element_visible", {"target": id_target("hidden").model_dump()}
    elif status == "INCONCLUSIVE":
        rule_type, payload = "web.element_visible", {"target": id_target("missing").model_dump()}
    elif status == "MISSING_ZERO":
        rule_type, payload = "web.element_hidden", {"target": id_target("zero").model_dump()}
    elif status == "TEXT_MATCH":
        rule_type, payload = "web.text_contains", {
            "target": id_target("ok").model_dump(), "text": "ready",
        }
    elif status == "VALUE_MATCH":
        rule_type, payload = "web.value_equals", {
            "target": id_target("ok").model_dump(), "value": "seed",
        }
    else:
        raise AssertionError(status)
    return ExpectedOutcome(
        schema_version=SCHEMA, outcome_type=rule_type, payload=payload, required=required
    )


def binding(index: int, *, verifier_type: str = "web_rule", config: dict | None = None):
    return VerificationBinding(
        schema_version=SCHEMA, outcome_index=index, verifier_type=verifier_type,
        config=config if config is not None else {},
    )


def contract(outcomes: list[ExpectedOutcome], *, bindings: list | None = None,
             metadata: dict | None = None, contract_id: str = "contract-1",
             goal_id: str = "goal-1") -> ActionContract:
    if bindings is None:
        bindings = [binding(index) for index in range(len(outcomes))]
    return ActionContract(
        schema_version=SCHEMA,
        contract_id=contract_id,
        goal_id=goal_id,
        forecast_id="forecast-1",
        observation_id="obs-1",
        adapter_name="playwright_web",
        environment_type="web",
        action=ActionSpec(
            schema_version=SCHEMA,
            action_type="web.click",
            payload={"target": id_target("ok").model_dump()},
        ),
        preconditions=[],
        expected_outcomes=outcomes,
        verification_policy=VerificationPolicy(schema_version=SCHEMA, bindings=bindings),
        recovery_policy=RecoveryPolicy(schema_version=SCHEMA),
        timeout_ms=1000.0,
        created_at=NOW,
        metadata=metadata if metadata is not None else {},
    )


def criteria(conditions: list[ExpectedOutcome], *, goal_id: str = "goal-1") -> TaskCompletionCriteria:
    return TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id=goal_id, conditions=conditions
    )


_UNSET = object()


def verify(recorded: ActionContract, piece: object = _UNSET):
    return WebStepVerifier().verify_step(
        contract=recorded, evidence=evidence() if piece is _UNSET else piece
    )


def complete(recorded: TaskCompletionCriteria, piece: object = _UNSET):
    return WebTaskCompletionOracle().evaluate_completion(
        criteria=recorded, evidence=evidence() if piece is _UNSET else piece
    )


# ------------------------------------------------------------------ signatures and purity (W01)


def test_the_three_entry_points_have_the_agreed_signatures() -> None:
    evaluate = inspect.signature(WebRuleEvaluator.evaluate)
    assert list(evaluate.parameters) == ["self", "rule_type", "payload", "evidence"]
    assert evaluate.parameters["rule_type"].kind is inspect.Parameter.KEYWORD_ONLY
    assert evaluate.parameters["payload"].kind is inspect.Parameter.KEYWORD_ONLY
    assert evaluate.parameters["evidence"].kind is inspect.Parameter.KEYWORD_ONLY
    step = inspect.signature(WebStepVerifier.verify_step)
    assert list(step.parameters) == ["self", "contract", "evidence"]
    assert step.parameters["contract"].kind is inspect.Parameter.KEYWORD_ONLY
    task = inspect.signature(WebTaskCompletionOracle.evaluate_completion)
    assert list(task.parameters) == ["self", "criteria", "evidence"]
    assert task.parameters["criteria"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_two_upper_verifiers_take_no_constructor_argument_and_hold_no_state() -> None:
    assert vars(WebStepVerifier()) == {}
    assert vars(WebTaskCompletionOracle()) == {}
    with pytest.raises(TypeError):
        WebStepVerifier("callback")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        WebTaskCompletionOracle("callback")  # type: ignore[call-arg]


# ------------------------------------------------------------------ step result shape (W06)


def test_the_step_result_is_ordered_and_refers_to_the_evidence_by_position() -> None:
    recorded = contract([
        outcome("PASS"), outcome("FAIL"), outcome("INCONCLUSIVE"),
    ])
    result = verify(recorded)
    assert result.verifier == "web_step_verifier"
    assert result.schema_version == SCHEMA
    assert result.status is VerificationStatus.FAIL
    assert result.reason == "REQUIRED_FAILED"
    assert [item.outcome_index for item in result.outcome_results] == [0, 1, 2]
    assert [item.status for item in result.outcome_results] == [
        VerificationStatus.PASS, VerificationStatus.FAIL, VerificationStatus.INCONCLUSIVE,
    ]
    assert [item.reason for item in result.outcome_results] == [
        "RULE_MATCH", "RULE_MISMATCH", "EVIDENCE_INSUFFICIENT",
    ]
    for item in result.outcome_results:
        assert item.has_actual_value is False
        assert item.actual_value is None
    assert dict(result.metadata) == {
        "scope": "step",
        "contract_id": "contract-1",
        "goal_id": "goal-1",
        "observation_id": "obs-1",
        "state_complete": True,
        "rule_evidence_refs": [
            {"rule_type": "web.element_visible", "target_index": 0},
            {"rule_type": "web.element_visible", "target_index": 1},
            {"rule_type": "web.element_visible", "target_index": None},
        ],
    }


def test_the_step_result_keeps_the_measured_values_out_of_the_diagnostic() -> None:
    conditions = [outcome("TEXT_MATCH"), outcome("VALUE_MATCH"), outcome("MISSING_ZERO")]
    recorded = contract(conditions)
    result = verify(recorded)
    assert result.status is VerificationStatus.PASS
    assert MARKER not in result.model_dump_json()
    assert "ready now" not in result.model_dump_json()
    assert "seed" not in result.model_dump_json()


@pytest.mark.parametrize(("statuses", "required", "expected", "reason"), [
    (["PASS", "PASS"], [True, True], VerificationStatus.PASS, "REQUIRED_PASSED"),
    (["PASS", "FAIL"], [True, True], VerificationStatus.FAIL, "REQUIRED_FAILED"),
    (["FAIL", "INCONCLUSIVE"], [True, True], VerificationStatus.FAIL, "REQUIRED_FAILED"),
    (["PASS", "INCONCLUSIVE"], [True, True], VerificationStatus.INCONCLUSIVE,
     "REQUIRED_INCONCLUSIVE"),
    (["INCONCLUSIVE", "PASS"], [True, True], VerificationStatus.INCONCLUSIVE,
     "REQUIRED_INCONCLUSIVE"),
    (["FAIL", "PASS"], [True, False], VerificationStatus.FAIL, "REQUIRED_FAILED"),
    (["INCONCLUSIVE", "PASS"], [True, False], VerificationStatus.INCONCLUSIVE,
     "REQUIRED_INCONCLUSIVE"),
    (["PASS", "FAIL"], [True, False], VerificationStatus.PASS, "REQUIRED_PASSED"),
    (["PASS", "INCONCLUSIVE"], [True, False], VerificationStatus.PASS, "REQUIRED_PASSED"),
])
def test_the_step_verdict_uses_required_all_aggregation(statuses: list, required: list,
                                                       expected, reason: str) -> None:
    outcomes = [outcome(status, required=is_required)
                for status, is_required in zip(statuses, required)]
    result = verify(contract(outcomes))
    assert (result.status, result.reason) == (expected, reason)
    assert [item.status for item in result.outcome_results] == [
        VerificationStatus(status.lower()) for status in statuses
    ]


def test_an_optional_pass_cannot_cancel_a_required_failure() -> None:
    recorded = contract([
        outcome("FAIL", required=True),
        outcome("PASS", required=False),
    ])
    result = verify(recorded)
    assert result.status is VerificationStatus.FAIL
    assert [item.status for item in result.outcome_results] == [
        VerificationStatus.FAIL, VerificationStatus.PASS,
    ]


# ------------------------------------------------------------------ step bindings (W06/W08)


def test_an_unordered_but_complete_binding_list_is_accepted() -> None:
    recorded = contract(
        [outcome("PASS"), outcome("PASS")],
        bindings=[binding(1), binding(0)],
    )
    result = verify(recorded)
    assert result.status is VerificationStatus.PASS
    assert [item.outcome_index for item in result.outcome_results] == [0, 1]


def _mutated_contract(raw: ActionContract, **updates):
    fields = dict(raw.__dict__)
    fields.update(updates)
    return ActionContract.model_construct(**fields)


def _policy(bindings: list):
    return VerificationPolicy.model_construct(schema_version=SCHEMA, bindings=bindings)


@pytest.mark.parametrize("bindings", [
    [],
    [binding(0)],
    [binding(0), binding(0)],
    [binding(0), binding(2)],
])
def test_a_missing_repeated_or_out_of_range_binding_is_a_fixed_input_error(bindings: list) -> None:
    base = contract([outcome("PASS"), outcome("PASS")])
    recorded = _mutated_contract(base, verification_policy=_policy(bindings))
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(recorded)
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_binding_for_another_verifier_is_a_fixed_input_error() -> None:
    base = contract([outcome("PASS")])
    recorded = _mutated_contract(
        base, verification_policy=_policy([binding(0, verifier_type="json_rule")])
    )
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(recorded)
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.path == (
        "contract", "verification_policy", "bindings", 0, "verifier_type",
    )


def test_a_non_empty_binding_config_is_a_fixed_input_error() -> None:
    base = contract([outcome("PASS")])
    recorded = _mutated_contract(
        base, verification_policy=_policy([binding(0, config={"text": MARKER})])
    )
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(recorded)
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.path == (
        "contract", "verification_policy", "bindings", 0, "config",
    )
    assert MARKER not in str(caught.value)


def test_every_outcome_type_must_be_one_of_the_five_web_rules() -> None:
    unknown = ExpectedOutcome.model_construct(
        schema_version=SCHEMA, outcome_type="web.url_equals", payload={}, required=False,
        metadata={},
    )
    base = contract([outcome("PASS"), outcome("PASS", required=False)])
    recorded = _mutated_contract(
        base,
        expected_outcomes=[base.expected_outcomes[0], unknown],
        verification_policy=_policy([binding(0), binding(1)]),
    )
    with pytest.raises(WebRuleInvalid) as caught:
        verify(recorded)
    assert caught.value.code == "WEB_RULE_INVALID"
    assert caught.value.path == ("contract", "expected_outcomes", 1, "outcome_type")


def test_a_malformed_optional_rule_is_never_ignored() -> None:
    bad = ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="web.text_contains", required=False,
        payload={"target": id_target("ok").model_dump(), "text": ""},
    )
    base = contract([outcome("FAIL"), bad])
    with pytest.raises(WebRuleInvalid) as caught:
        verify(base)
    assert caught.value.code == "WEB_RULE_INVALID"
    assert caught.value.path[:2] == ("contract", "expected_outcomes")


def test_an_unusable_contract_is_a_fixed_input_error() -> None:
    for candidate in (None, 1, {}, "contract"):
        with pytest.raises(WebVerificationInputInvalid) as caught:
            verify(candidate)  # type: ignore[arg-type]
        assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
        assert caught.value.path == ("contract",)


def test_a_contract_without_a_required_outcome_is_a_fixed_input_error() -> None:
    optional = outcome("PASS", required=False)
    base = contract([outcome("PASS")])
    recorded = _mutated_contract(
        base, expected_outcomes=[optional], verification_policy=_policy([binding(0)])
    )
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(recorded)
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"


def test_unusable_evidence_is_a_fixed_input_error_for_the_step_verifier() -> None:
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(contract([outcome("PASS")]), None)  # type: ignore[arg-type]
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.path == ("evidence",)


# ------------------------------------------------------------------ step independence


def test_the_step_verifier_does_not_read_an_execution_status_or_a_completion_hint() -> None:
    recorded = contract(
        [outcome("FAIL")],
        metadata={"execution_status": "succeeded", "task_completion_hint": True},
    )
    result = verify(recorded)
    assert result.status is VerificationStatus.FAIL
    dumped = result.model_dump()
    assert "execution_status" not in repr(dumped["metadata"])
    assert "task_completion_hint" not in repr(dumped["metadata"])


def test_the_step_verifier_does_not_modify_the_contract_or_the_evidence() -> None:
    recorded = contract([outcome("PASS"), outcome("FAIL")])
    piece = evidence()
    before = (recorded.model_dump(mode="json"), piece.model_dump(mode="json"))
    verify(recorded, piece)
    assert recorded.model_dump(mode="json") == before[0]
    assert piece.model_dump(mode="json") == before[1]
    assert next(iter(recorded.expected_outcomes)).payload == {
        "target": id_target("ok").model_dump()
    }


# ------------------------------------------------------------------ task oracle (W07)


def test_the_task_result_uses_only_the_independent_criteria() -> None:
    recorded = criteria([outcome("PASS")])
    result = complete(recorded)
    assert result.verifier == "web_task_completion_oracle"
    assert result.status is VerificationStatus.PASS
    assert result.reason == "REQUIRED_PASSED"
    assert dict(result.metadata) == {
        "scope": "task",
        "goal_id": "goal-1",
        "observation_id": "obs-1",
        "state_complete": True,
        "rule_evidence_refs": [
            {"rule_type": "web.element_visible", "target_index": 0},
        ],
    }
    assert "contract_id" not in result.metadata
    for item in result.outcome_results:
        assert item.has_actual_value is False
        assert item.actual_value is None


@pytest.mark.parametrize(("statuses", "required", "expected", "reason"), [
    (["PASS", "PASS"], [True, True], VerificationStatus.PASS, "REQUIRED_PASSED"),
    (["PASS", "FAIL"], [True, True], VerificationStatus.FAIL, "REQUIRED_FAILED"),
    (["FAIL", "INCONCLUSIVE"], [True, True], VerificationStatus.FAIL, "REQUIRED_FAILED"),
    (["PASS", "INCONCLUSIVE"], [True, True], VerificationStatus.INCONCLUSIVE,
     "REQUIRED_INCONCLUSIVE"),
    (["FAIL", "PASS"], [True, False], VerificationStatus.FAIL, "REQUIRED_FAILED"),
    (["PASS", "FAIL"], [True, False], VerificationStatus.PASS, "REQUIRED_PASSED"),
])
def test_the_task_verdict_uses_required_all_aggregation(statuses: list, required: list,
                                                       expected, reason: str) -> None:
    conditions = [outcome(status, required=is_required)
                  for status, is_required in zip(statuses, required)]
    result = complete(criteria(conditions))
    assert (result.status, result.reason) == (expected, reason)


def test_the_task_oracle_is_independent_of_the_step_verdict() -> None:
    failing_step = contract([outcome("FAIL")])
    passing_task = criteria([outcome("PASS")])
    assert verify(failing_step).status is VerificationStatus.FAIL
    assert complete(passing_task).status is VerificationStatus.PASS

    passing_step = contract([outcome("PASS")])
    failing_task = criteria([outcome("FAIL")])
    assert verify(passing_step).status is VerificationStatus.PASS
    assert complete(failing_task).status is VerificationStatus.FAIL


def test_an_empty_or_optional_only_criteria_set_is_a_fixed_input_error() -> None:
    empty = TaskCompletionCriteria.model_construct(
        schema_version=SCHEMA, goal_id="goal-1", conditions=[]
    )
    optional_only = TaskCompletionCriteria.model_construct(
        schema_version=SCHEMA, goal_id="goal-1",
        conditions=[outcome("PASS", required=False)],
    )
    for recorded in (empty, optional_only):
        with pytest.raises(WebVerificationInputInvalid) as caught:
            complete(recorded)
        assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
        assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_an_unknown_optional_condition_is_never_ignored() -> None:
    unknown = ExpectedOutcome.model_construct(
        schema_version=SCHEMA, outcome_type="web.does_not_exist", payload={}, required=False,
        metadata={},
    )
    recorded = criteria([outcome("FAIL"), unknown])
    with pytest.raises(WebRuleInvalid) as caught:
        complete(recorded)
    assert caught.value.code == "WEB_RULE_INVALID"
    assert caught.value.path == ("criteria", "conditions", 1, "outcome_type")


def test_an_unusable_criteria_record_is_a_fixed_input_error() -> None:
    for candidate in (None, 1, {}, "criteria"):
        with pytest.raises(WebVerificationInputInvalid) as caught:
            complete(candidate)  # type: ignore[arg-type]
        assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
        assert caught.value.path == ("criteria",)


def test_unusable_evidence_is_a_fixed_input_error_for_the_task_oracle() -> None:
    with pytest.raises(WebVerificationInputInvalid) as caught:
        complete(criteria([outcome("PASS")]), None)  # type: ignore[arg-type]
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.path == ("evidence",)


def test_the_state_complete_declaration_is_reported_not_rewritten() -> None:
    recorded = criteria([outcome("MISSING_ZERO")])
    result = complete(recorded, evidence(complete=False))
    assert result.status is VerificationStatus.PASS
    assert result.metadata["state_complete"] is False


def test_the_oracle_does_not_modify_the_criteria_or_the_evidence() -> None:
    recorded = criteria([outcome("PASS"), outcome("FAIL")])
    piece = evidence()
    before = (recorded.model_dump(mode="json"), piece.model_dump(mode="json"))
    complete(recorded, piece)
    assert recorded.model_dump(mode="json") == before[0]
    assert piece.model_dump(mode="json") == before[1]


# ------------------------------------------------------------------ diagnostics and errors


@pytest.mark.parametrize("factory", [WebStepVerifier, WebTaskCompletionOracle])
def test_a_verification_error_is_bounded_and_carries_no_supplied_value(factory) -> None:
    target = TestIdTarget(strategy="test_id", test_id=MARKER)
    bad = ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="web.text_contains",
        payload={"target": target.model_dump(), "text": ""},
    )
    with pytest.raises(WebRuleInvalid) as caught:
        if factory is WebStepVerifier:
            verify(contract([bad]))
        else:
            complete(criteria([bad]))
    error = caught.value
    assert isinstance(error, WebValidationError)
    assert error.code == "WEB_RULE_INVALID"
    assert MARKER not in str(error)
    assert MARKER not in repr(error)
    assert MARKER not in repr(error.args)
    assert error.__cause__ is None and error.__context__ is None
    for value in vars(error).values():
        assert MARKER not in repr(value)


def test_the_round_input_error_is_the_same_existing_web_error_class() -> None:
    assert ImportedInputInvalid is WebVerificationInputInvalid
    with pytest.raises(WebValidationError):
        verify(contract([outcome("PASS")]), None)  # type: ignore[arg-type]


def test_a_serialized_result_cannot_launder_an_unknown_rule() -> None:
    unknown = ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="web.unknown", payload={}, required=True
    )
    with pytest.raises(WebRuleInvalid):
        verify(contract([unknown]))
    with pytest.raises(ValidationError):
        ExpectedOutcome(schema_version=SCHEMA, outcome_type=" ", payload={}, required=True)


# ------------------------------------------------------------------ stored structure (F01)

def _url_outcome(expected_url: str, *, required: bool = True) -> ExpectedOutcome:
    return ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="web.url_match",
        payload={"url": expected_url}, required=required,
    )


def _url_evidence(recorded: type[VerificationEvidence], url: str) -> VerificationEvidence:
    state = {"schema_version": SCHEMA, "url": url, "targets": []}
    return recorded(
        schema_version=SCHEMA,
        observation=Observation(
            schema_version=SCHEMA,
            observation_id="obs-1",
            environment_type="web",
            state=state,
            timestamp=NOW,
        ),
        state_complete=True,
    )


class _RewritingEvidence(VerificationEvidence):
    """A valid evidence subclass whose serializer rewrites the actual stored URL."""

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["observation"]["state"]["url"] = PAGE
        return data


class _RewritingCriteria(TaskCompletionCriteria):
    """A valid criteria subclass whose serializer rewrites the actual stored condition."""

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["conditions"][0]["payload"]["url"] = PAGE
        return data


class _RewritingContract(ActionContract):
    """A valid contract subclass whose serializer rewrites the actual stored outcome."""

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["expected_outcomes"][0]["payload"]["url"] = PAGE
        return data


def test_an_overridden_evidence_dump_cannot_rewrite_a_step_verdict() -> None:
    actual = PAGE + "/actually-different"
    piece = _url_evidence(_RewritingEvidence, actual)
    result = verify(contract([_url_outcome(PAGE)]), piece)
    assert result.status is VerificationStatus.FAIL
    assert result.outcome_results[0].reason == "RULE_MISMATCH"


def test_an_overridden_criteria_dump_cannot_rewrite_a_goal() -> None:
    recorded = _RewritingCriteria(
        schema_version=SCHEMA,
        goal_id="goal-1",
        conditions=[_url_outcome(PAGE + "/actually-different")],
    )
    result = complete(recorded)
    assert result.status is VerificationStatus.FAIL
    assert result.outcome_results[0].reason == "RULE_MISMATCH"


def test_an_overridden_contract_dump_cannot_rewrite_a_step_verdict() -> None:
    base = contract([_url_outcome(PAGE + "/actually-different")])
    recorded = _RewritingContract.model_construct(**dict(base.__dict__))
    result = verify(recorded, evidence())
    assert result.status is VerificationStatus.FAIL
    assert result.outcome_results[0].reason == "RULE_MISMATCH"


def test_an_ordinary_criteria_subclass_still_evaluates() -> None:
    class _OrdinaryCriteria(TaskCompletionCriteria):
        pass

    recorded = _OrdinaryCriteria(
        schema_version=SCHEMA, goal_id="goal-1", conditions=[outcome("PASS")]
    )
    assert complete(recorded).status is VerificationStatus.PASS


def test_an_empty_criteria_extra_store_still_evaluates() -> None:
    absent = criteria([outcome("PASS")])
    object.__setattr__(absent, "__pydantic_extra__", None)
    empty = criteria([outcome("PASS")])
    object.__setattr__(empty, "__pydantic_extra__", {})
    for recorded in (absent, empty):
        assert complete(recorded).status is VerificationStatus.PASS


def test_a_non_empty_criteria_extra_store_is_a_fixed_input_error() -> None:
    recorded = criteria([outcome("PASS")])
    object.__setattr__(recorded, "__pydantic_extra__", {"hidden": MARKER})
    with pytest.raises(WebVerificationInputInvalid) as caught:
        complete(recorded)
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert MARKER not in str(error)
    assert error.__cause__ is None and error.__context__ is None


# ------------------------------------------------------------------ expected URL grammar (F02)

@pytest.mark.parametrize("expected", [
    pytest.param("", id="empty"),
    pytest.param("not a url", id="relative"),
    pytest.param("javascript:alert(1)", id="javascript"),
    pytest.param("https://user:pass@example.com/", id="userinfo"),
    pytest.param("http://example.com:bad", id="bad-port"),
    pytest.param("ftp://example.com/", id="other-scheme"),
    pytest.param(" http://example.com/", id="leading-space"),
])
def test_an_invalid_expected_url_in_an_outcome_is_a_fixed_rule_error(expected) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        verify(contract([_url_outcome(expected)]))
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert error.path[:3] == ("contract", "expected_outcomes", 0)
    assert error.__cause__ is None and error.__context__ is None


def test_an_invalid_expected_url_in_a_condition_is_a_fixed_rule_error() -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        complete(criteria([_url_outcome("not a url")]))
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert error.path[:3] == ("criteria", "conditions", 0)


# ------------------------------------------------------------------ result target index (F03)

def test_a_url_outcome_carries_a_null_target_index_in_the_step_result() -> None:
    result = verify(contract([_url_outcome(PAGE)]))
    assert result.metadata["rule_evidence_refs"] == [
        {"rule_type": "web.url_match", "target_index": None},
    ]
    assert result.outcome_results[0].has_actual_value is False
    assert result.outcome_results[0].actual_value is None


def test_a_url_condition_carries_a_null_target_index_in_the_task_result() -> None:
    result = complete(criteria([_url_outcome(PAGE)]))
    assert result.metadata["rule_evidence_refs"] == [
        {"rule_type": "web.url_match", "target_index": None},
    ]


# ------------------------------------------------------------------ remaining F01 (round-003)


def _step_contract_with_policy(policy: RecoveryPolicy) -> ActionContract:
    """Return a valid step contract whose nested recovery policy is the supplied record."""
    base = contract([outcome("PASS")])
    fields = dict(base.__dict__)
    fields["recovery_policy"] = policy
    return ActionContract.model_construct(**fields)


@pytest.mark.parametrize(("control", "supplied"), [
    pytest.param("max_retries", {"max_retries": 1}, id="abort-retries"),
    pytest.param("allow_replan", {"allow_replan": True}, id="abort-replan"),
])
def test_a_real_non_null_abort_control_is_a_fixed_input_error(
    control: str, supplied: dict
) -> None:
    """F01 step/recovery/abort-retries and step/recovery/abort-replan.

    The nested policy really stores the non-null control, and the ordinary unchanged Core
    constructor refuses the same shape, so the Web projection must not erase it.
    """
    policy = RecoveryPolicy.model_construct(
        schema_version=SCHEMA, mode="abort", **supplied
    )
    assert getattr(policy, control) is not None
    with pytest.raises(ValidationError):
        RecoveryPolicy(schema_version=SCHEMA, mode="abort", **supplied)
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(_step_contract_with_policy(policy))
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert error.path == ("contract", "recovery_policy", control)
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_an_ordinary_abort_recovery_policy_still_verifies() -> None:
    """F01 step/recovery/ordinary-abort-control: the default abort record stays accepted."""
    recorded = contract([outcome("PASS")])
    policy = recorded.recovery_policy
    assert (policy.mode, policy.max_retries, policy.allow_replan) == ("abort", None, None)
    assert verify(recorded).status is VerificationStatus.PASS


def test_an_ordinary_bounded_recovery_policy_still_verifies() -> None:
    """F01 step/recovery/ordinary-bounded-control: a legal bounded policy stays accepted."""
    policy = RecoveryPolicy(
        schema_version=SCHEMA, mode="bounded", max_retries=1, allow_replan=False
    )
    assert (policy.mode, policy.max_retries, policy.allow_replan) == ("bounded", 1, False)
    assert verify(_step_contract_with_policy(policy)).status is VerificationStatus.PASS


# ------------------------------------------------------------------ remaining F01 (round-004)


class _ForeignAttributeRecord:
    """A foreign record whose every attribute read raises the synthetic marker."""

    def __getattribute__(self, name: str) -> object:
        raise ValueError(MARKER)


class _ForeignClassRecord:
    """A Pydantic-lookalike foreign record whose ``__class__`` property raises."""

    __pydantic_decorators__ = True

    @property
    def __class__(self):
        raise ValueError(MARKER)


def _assert_foreign_fixed_error(failure: BaseException, path: tuple[str, ...]) -> None:
    """Assert one rejected foreign record became the fixed, cause-free input error."""
    assert isinstance(failure, WebVerificationInputInvalid)
    assert failure.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert failure.path == path
    assert MARKER not in str(failure) and MARKER not in repr(failure)
    assert failure.__cause__ is None and failure.__context__ is None


def test_a_foreign_contract_attribute_read_is_a_fixed_input_error() -> None:
    """F01 contract/foreign-attribute: the contract is classified by its real type only."""
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(_ForeignAttributeRecord())  # type: ignore[arg-type]
    _assert_foreign_fixed_error(caught.value, ("contract",))


def test_a_foreign_contract_class_property_is_a_fixed_input_error() -> None:
    """F01 contract/foreign-class: a foreign ``__class__`` is never called by classification."""
    with pytest.raises(WebVerificationInputInvalid) as caught:
        verify(_ForeignClassRecord())  # type: ignore[arg-type]
    _assert_foreign_fixed_error(caught.value, ("contract",))


def test_a_foreign_criteria_attribute_read_is_a_fixed_input_error() -> None:
    """F01 criteria/foreign-attribute: the criteria are classified by their real type only."""
    with pytest.raises(WebVerificationInputInvalid) as caught:
        complete(_ForeignAttributeRecord())  # type: ignore[arg-type]
    _assert_foreign_fixed_error(caught.value, ("criteria",))


def test_a_foreign_criteria_class_property_is_a_fixed_input_error() -> None:
    """F01 criteria/foreign-class: a foreign ``__class__`` is never called by classification."""
    with pytest.raises(WebVerificationInputInvalid) as caught:
        complete(_ForeignClassRecord())  # type: ignore[arg-type]
    _assert_foreign_fixed_error(caught.value, ("criteria",))


def test_the_canonical_step_required_all_positive_still_verifies() -> None:
    """Canonical step required-all positive: every required outcome passes."""
    recorded = contract([outcome("PASS"), outcome("TEXT_MATCH")])
    result = verify(recorded)
    assert (result.status, result.reason, result.verifier) == (
        VerificationStatus.PASS, "REQUIRED_PASSED", "web_step_verifier",
    )
    assert [item.status for item in result.outcome_results] == [
        VerificationStatus.PASS, VerificationStatus.PASS,
    ]


def test_the_canonical_task_required_all_positive_still_completes() -> None:
    """Canonical task required-all positive: every required condition passes."""
    recorded = criteria([outcome("PASS"), outcome("VALUE_MATCH")])
    result = complete(recorded)
    assert (result.status, result.reason, result.verifier) == (
        VerificationStatus.PASS, "REQUIRED_PASSED", "web_task_completion_oracle",
    )
    assert [item.status for item in result.outcome_results] == [
        VerificationStatus.PASS, VerificationStatus.PASS,
    ]
