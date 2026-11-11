"""Unit acceptance of the Web contract-validation hooks and their factory.

The tests use the real Core records, the real accepted Web rule evaluator and the public Web
observation grammar; the only synthetic part is the fixed evidence they build. No browser,
adapter, runtime or task oracle is involved.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts.models import VerificationBinding
from forecontract.models.actions import ExpectedOutcome, Precondition
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.protocols.adapter import InvalidActionPayloadError
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract_web.rules import WebVerificationInputInvalid
from forecontract_web.runtime_hooks import (
    WebContractHooksFactory,
    WebContractValidationHooks,
    WebTaskCriteriaValidator,
)

SCHEMA = "0.1"
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
LOGIN = "http://127.0.0.1:8123/login"


def target(name: str) -> dict:
    return {"strategy": "test_id", "test_id": name}


def record(name: str, count: int, *, visible: bool | None = None,
           text: str | None = None, value: str | None = None) -> dict:
    return {"target": target(name), "status": "collected", "match_count": count,
            "visible": visible, "text": text, "value": value, "error_code": None}


def observation(url: str | None = LOGIN, targets: list[dict] | None = None) -> Observation:
    return Observation(
        schema_version=SCHEMA, observation_id="obs-1", environment_type="web",
        timestamp=NOW,
        state={"schema_version": SCHEMA, "url": url,
               "targets": targets if targets is not None else [record("welcome", 1,
                                                                      visible=True,
                                                                      text="登录成功",
                                                                      value="登录成功")]},
    )


def evidence(complete: bool = True, *, obs: Observation | None = None) -> VerificationEvidence:
    return VerificationEvidence(schema_version=SCHEMA, observation=obs or observation(),
                                state_complete=complete)


def precondition(rule_type: str, payload: dict) -> Precondition:
    return Precondition(schema_version=SCHEMA, condition_type=rule_type, payload=payload,
                        required=True)


def outcome(rule_type: str, payload: dict) -> ExpectedOutcome:
    return ExpectedOutcome(schema_version=SCHEMA, outcome_type=rule_type, payload=payload,
                           required=True)


def binding(verifier_type: str = "web_rule", config: dict | None = None) -> VerificationBinding:
    return VerificationBinding(schema_version=SCHEMA, outcome_index=0,
                               verifier_type=verifier_type, config=config or {})


# --------------------------------------------------------------------------- factory


def test_the_factory_uses_the_evidence_completeness_strictly() -> None:
    factory = WebContractHooksFactory()
    assert factory.create(evidence=evidence(True)).state_complete is True
    assert factory.create(evidence=evidence(False)).state_complete is False


def test_the_factory_rejects_evidence_without_an_explicit_boolean() -> None:
    damaged = VerificationEvidence.model_construct(
        schema_version=SCHEMA, observation=observation(), state_complete=1)
    with pytest.raises(WebVerificationInputInvalid):
        WebContractHooksFactory().create(evidence=damaged)


# --------------------------------------------------------------------------- all five rules


@pytest.mark.parametrize("rule_type,payload,status", [
    ("web.url_match", {"url": LOGIN}, VerificationStatus.PASS),
    ("web.element_visible", {"target": target("welcome")}, VerificationStatus.PASS),
    ("web.element_hidden", {"target": target("welcome")}, VerificationStatus.FAIL),
    ("web.text_contains", {"target": target("welcome"), "text": "登录"}, VerificationStatus.PASS),
    ("web.value_equals", {"target": target("welcome"), "value": "登录成功"},
     VerificationStatus.PASS),
])
def test_every_declared_rule_validates_and_evaluates(rule_type: str, payload: dict,
                                                     status: VerificationStatus) -> None:
    hooks = WebContractValidationHooks(state_complete=True)
    condition = precondition(rule_type, payload)
    hooks.validate_precondition(condition)
    hooks.validate_expected_outcome(outcome(rule_type, payload))
    assert hooks.check_precondition(condition, observation()) is status


def test_each_rule_honours_the_truth_table_for_missing_and_mismatching_evidence() -> None:
    hooks = WebContractValidationHooks(state_complete=True)
    missing = observation(targets=[record("welcome", 0)])
    assert hooks.check_precondition(
        precondition("web.element_visible", {"target": target("welcome")}),
        missing) is VerificationStatus.FAIL
    assert hooks.check_precondition(
        precondition("web.element_hidden", {"target": target("welcome")}),
        missing) is VerificationStatus.PASS
    absent_record = observation(targets=[])
    assert hooks.check_precondition(
        precondition("web.value_equals", {"target": target("welcome"), "value": "x"}),
        absent_record) is VerificationStatus.INCONCLUSIVE
    assert hooks.check_precondition(
        precondition("web.url_match", {"url": LOGIN}), observation(url=None)
    ) is VerificationStatus.INCONCLUSIVE
    assert hooks.check_precondition(
        precondition("web.url_match", {"url": LOGIN + "?x=1"}), observation()
    ) is VerificationStatus.FAIL


def test_a_null_attribute_is_inconclusive_not_a_failure() -> None:
    hooks = WebContractValidationHooks(state_complete=True)
    obs = observation(targets=[record("welcome", 1)])
    assert hooks.check_precondition(
        precondition("web.element_visible", {"target": target("welcome")}), obs
    ) is VerificationStatus.INCONCLUSIVE


# --------------------------------------------------------------------------- invalid payloads


@pytest.mark.parametrize("rule_type,payload", [
    ("web.url_match", {}),
    ("web.url_match", {"url": "/relative"}),
    ("web.element_visible", {"target": {"strategy": "test_id"}}),
    ("web.text_contains", {"target": target("welcome"), "text": ""}),
    ("web.value_equals", {"target": target("welcome")}),
    ("web.unknown_rule", {"target": target("welcome")}),
    ("json_rule", {"path": ["x"]}),
])
def test_an_invalid_or_unknown_rule_is_a_domain_rejection(rule_type: str,
                                                          payload: dict) -> None:
    hooks = WebContractValidationHooks(state_complete=True)
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_precondition(precondition(rule_type, payload))
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_expected_outcome(outcome(rule_type, payload))


def test_the_binding_must_be_web_rule_with_an_empty_configuration() -> None:
    hooks = WebContractValidationHooks(state_complete=True)
    valid = outcome("web.url_match", {"url": LOGIN})
    hooks.validate_verification_binding(binding(), valid)
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_verification_binding(binding("json_rule"), valid)
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_verification_binding(binding("web_rule", {"x": 1}), valid)
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_verification_binding("not-a-binding", valid)


def test_an_injected_stored_key_is_refused_without_echoing_it() -> None:
    hooks = WebContractValidationHooks(state_complete=True)
    condition = precondition("web.url_match", {"url": LOGIN})
    condition.__dict__["injected-hidden-key"] = "marker-value"
    with pytest.raises(InvalidActionPayloadError) as raised:
        hooks.validate_precondition(condition)
    assert "marker-value" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_a_damaged_observation_is_a_domain_rejection_not_a_verdict() -> None:
    hooks = WebContractValidationHooks(state_complete=True)
    damaged = Observation(schema_version=SCHEMA, observation_id="obs-1",
                          environment_type="web", timestamp=NOW,
                          state={"schema_version": SCHEMA, "url": None,
                                 "targets": [{"target": target("welcome"),
                                              "status": "collected"}]})
    with pytest.raises(InvalidActionPayloadError) as raised:
        hooks.check_precondition(
            precondition("web.element_visible", {"target": target("welcome")}), damaged)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_an_unusable_evidence_record_is_refused_by_the_factory() -> None:
    with pytest.raises(WebVerificationInputInvalid):
        WebContractHooksFactory().create(evidence="not-evidence")  # type: ignore[arg-type]


# --- F05: the independent Web task-criteria prevalidator -----------------------------------


def criteria(*conditions: ExpectedOutcome) -> TaskCompletionCriteria:
    return TaskCompletionCriteria(schema_version=SCHEMA, goal_id="goal-1",
                                  conditions=list(conditions))


def test_the_web_task_criteria_validator_accepts_every_declared_rule() -> None:
    conditions = [
        outcome("web.url_match", {"url": LOGIN}),
        outcome("web.element_visible", {"target": target("welcome")}),
        outcome("web.text_contains", {"target": target("welcome"), "text": "登录"}),
        outcome("web.value_equals", {"target": target("welcome"), "value": "登录成功"}),
    ]
    assert WebTaskCriteriaValidator().validate(criteria=criteria(*conditions)) is None


def test_the_web_task_criteria_validator_checks_optional_conditions_too() -> None:
    optional = ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="web.unknown_rule",
        payload={"target": target("welcome")}, required=False)
    with pytest.raises(VerificationError) as raised:
        WebTaskCriteriaValidator().validate(
            criteria=criteria(outcome("web.url_match", {"url": LOGIN}), optional))
    assert raised.value.code == "VERIFICATION_INPUT_INVALID"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("bad", [
    outcome("web.url_match", {}),
    outcome("web.url_match", {"url": "/relative"}),
    outcome("web.element_visible", {"target": {"strategy": "test_id"}}),
    outcome("web.text_contains", {"target": target("welcome"), "text": ""}),
    outcome("json_rule", {"path": ["x"]}),
])
def test_the_web_task_criteria_validator_rejects_a_malformed_rule(
        bad: ExpectedOutcome) -> None:
    with pytest.raises(VerificationError) as raised:
        WebTaskCriteriaValidator().validate(criteria=criteria(bad))
    assert raised.value.code == "VERIFICATION_INPUT_INVALID"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_the_web_task_criteria_validator_rejects_empty_or_optional_only_criteria() -> None:
    with pytest.raises(VerificationError):
        WebTaskCriteriaValidator().validate(criteria=criteria())
    only_optional = ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="web.url_match",
        payload={"url": LOGIN}, required=False)
    with pytest.raises(VerificationError):
        WebTaskCriteriaValidator().validate(criteria=criteria(only_optional))


def test_the_web_task_criteria_validator_does_not_mutate_the_criteria() -> None:
    supplied = criteria(outcome("web.url_match", {"url": LOGIN}))
    before = supplied.model_dump()
    assert WebTaskCriteriaValidator().validate(criteria=supplied) is None
    assert supplied.model_dump() == before
