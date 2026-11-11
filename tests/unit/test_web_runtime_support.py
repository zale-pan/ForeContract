"""Unit acceptance of the strict Web ``RuntimeDomainSupport`` and its policies.

The tests exercise the real Core records, the real Contract Builder for a committed action and
the accepted Web rule evaluator. No browser, adapter lifecycle, site, runtime or task oracle is
involved.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts import (
    ContractBuilder,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import (
    ActionSpec,
    ExpectedOutcome,
    Forecast,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome, VerificationStatus
from forecontract.recovery import (
    ActionSafetyDeclaration,
    RecoveryContext,
    RecoveryEngine,
    RecoveryError,
    RecoveryEvidence,
)
from forecontract.verification.models import VerificationEvidence
from forecontract_web.models import (
    WEB_ACTION_TYPES,
    WEB_OBSERVATION_TYPES,
    TestIdTarget,
)
from forecontract_web.rule_capabilities import build_web_rule_capabilities
from forecontract_web.rule_models import WEB_RULE_TYPES
from forecontract_web.runtime_hooks import WebContractValidationHooks
from forecontract_web.runtime_support import (
    WebActionPolicy,
    WebRuntimeDomainSupport,
    WebRuntimeSupportError,
)
from forecontract_web.validation import validate_web_action
from forecontract_web.verification import WebStepVerifier

SCHEMA = "0.1"
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
LOGIN = "http://127.0.0.1:8123/login"
DASHBOARD = "http://127.0.0.1:8123/dashboard"
USERNAME = TestIdTarget(strategy="test_id", test_id="username")
PASSWORD = TestIdTarget(strategy="test_id", test_id="password")
SUBMIT = TestIdTarget(strategy="test_id", test_id="submit")
WELCOME = TestIdTarget(strategy="test_id", test_id="welcome")


def target_payload(target: TestIdTarget) -> dict:
    return target.model_dump()


def record(target: TestIdTarget, count: int, *, visible: bool | None = None,
           text: str | None = None, value: str | None = None) -> dict:
    return {"target": target_payload(target), "status": "collected",
            "match_count": count, "visible": visible, "text": text, "value": value,
            "error_code": None}


def observation(url: str, targets: list[dict]) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id="obs-1",
                       environment_type="web", timestamp=NOW,
                       state={"schema_version": SCHEMA, "url": url, "targets": targets})


def login_observation() -> Observation:
    return observation(LOGIN, [
        record(USERNAME, 1, visible=True, text="", value=""),
        record(PASSWORD, 1, visible=True, text="", value=""),
        record(SUBMIT, 1, visible=True, text="登录"),
        record(WELCOME, 0),
    ])


def dashboard_observation() -> Observation:
    return observation(DASHBOARD, [
        record(USERNAME, 0), record(PASSWORD, 0), record(SUBMIT, 0),
        record(WELCOME, 1, visible=True, text="登录成功", value=None),
    ])


def navigate_action(metadata: dict | None = None) -> ActionSpec:
    return ActionSpec(schema_version=SCHEMA, action_type="web.navigate",
                      payload={"url": LOGIN}, metadata=metadata or {})


def click_action() -> ActionSpec:
    return ActionSpec(schema_version=SCHEMA, action_type="web.click",
                      payload={"target": target_payload(SUBMIT)})


def precondition(rule_type: str, payload: dict) -> dict:
    return {"schema_version": SCHEMA, "condition_type": rule_type,
            "payload": payload, "required": True}


def outcome(rule_type: str, payload: dict, required: bool = True) -> ExpectedOutcome:
    return ExpectedOutcome(schema_version=SCHEMA, outcome_type=rule_type, payload=payload,
                           required=required)


def dashboard_outcomes() -> list[ExpectedOutcome]:
    return [outcome("web.url_match", {"url": DASHBOARD}),
            outcome("web.element_visible", {"target": target_payload(WELCOME)})]


def policy(*, action: ActionSpec | None = None,
           preconditions: list | None = None,
           retry_safety: str = "non_idempotent",
           applied_when: list[ExpectedOutcome] | None = None,
           not_applied_when: list[ExpectedOutcome] | None = None,
           timeout_ms: object = 5000.0) -> WebActionPolicy:
    return WebActionPolicy(
        action=action or click_action(),
        preconditions=preconditions or [],
        retry_safety=retry_safety,
        applied_when=applied_when if applied_when is not None else dashboard_outcomes(),
        not_applied_when=not_applied_when or [],
        recovery_policy=RecoveryPolicy(schema_version=SCHEMA, mode="bounded",
                                       max_retries=0, allow_replan=False),
        timeout_ms=timeout_ms,
    )


class StubAdapter:
    """A synchronous, no-I/O capability declaration used only to build a real contract."""

    @property
    def name(self) -> str:
        return "playwright_web"

    def capabilities(self):
        return build_web_rule_capabilities(
            adapter_name="playwright_web",
            supported_actions=list(WEB_ACTION_TYPES),
            supported_observation_types=list(WEB_OBSERVATION_TYPES),
            supported_condition_types=list(WEB_RULE_TYPES),
            supported_outcome_types=list(WEB_RULE_TYPES),
        )

    def validate_action(self, action: ActionSpec) -> None:
        validate_web_action(action)


def build_contract(action: ActionSpec, outcomes: list[ExpectedOutcome],
                   observation_value: Observation | None = None) -> object:
    forecast = Forecast(schema_version=SCHEMA, forecast_id="forecast-1",
                        proposed_action=action, expected_outcomes=outcomes)
    return ContractBuilder().build(
        forecast=forecast,
        observation=observation_value or login_observation(),
        adapter=StubAdapter(),
        hooks=WebContractValidationHooks(state_complete=True),
        preconditions=[],
        verification_policy=VerificationPolicy(
            schema_version=SCHEMA,
            bindings=[VerificationBinding(schema_version=SCHEMA, outcome_index=index,
                                          verifier_type="web_rule")
                      for index in range(len(outcomes))]),
        timeout_ms=5000.0,
        remaining_budget_ms=30000.0,
        contract_id="contract-1",
        goal_id="goal-1",
        created_at=NOW,
        recovery_policy=RecoveryPolicy(schema_version=SCHEMA, mode="bounded",
                                       max_retries=0, allow_replan=False),
    )


def support(*policies: WebActionPolicy, targets: list | None = None,
            require_url: bool = True) -> WebRuntimeDomainSupport:
    return WebRuntimeDomainSupport(
        adapter_name="playwright_web",
        required_targets=targets if targets is not None else [WELCOME],
        require_url=require_url,
        action_policies=list(policies) if policies else [policy()],
    )


def evidence(obs: Observation, complete: bool = True) -> VerificationEvidence:
    return VerificationEvidence(schema_version=SCHEMA, observation=obs,
                                state_complete=complete)


# --------------------------------------------------------------------------- policy strictness


def test_a_valid_policy_keeps_its_exact_declared_content() -> None:
    built = policy(timeout_ms=3000)
    assert built.timeout_ms == 3000.0
    assert built.action.action_type == "web.click"
    assert len(built.applied_when) == 2
    assert built.applied_when[0].required is True
    with pytest.raises(ValidationError):
        built.timeout_ms = 1.0  # frozen


@pytest.mark.parametrize("timeout", [True, False, "5000", 0, -1, math.nan, math.inf, None])
def test_the_policy_timeout_is_a_finite_positive_number(timeout: object) -> None:
    with pytest.raises(WebRuntimeSupportError) as raised:
        policy(timeout_ms=timeout)
    assert raised.value.code == "WEB_RUNTIME_SUPPORT_INVALID"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_a_policy_rejects_an_unknown_field() -> None:
    with pytest.raises(WebRuntimeSupportError):
        WebActionPolicy(
            action=click_action(), preconditions=[], retry_safety="non_idempotent",
            applied_when=[], not_applied_when=[], recovery_policy=RecoveryPolicy(
                schema_version=SCHEMA, mode="abort"), timeout_ms=5000.0,
            unexpected=True,
        )


def test_a_policy_rejects_an_invalid_web_action_payload() -> None:
    with pytest.raises(WebRuntimeSupportError):
        policy(action=ActionSpec(schema_version=SCHEMA, action_type="web.click",
                                 payload={"target": {"strategy": "test_id"}}))


def test_an_effect_group_must_be_all_required_web_rules() -> None:
    optional = outcome("web.url_match", {"url": DASHBOARD}, required=False)
    with pytest.raises(WebRuntimeSupportError):
        policy(applied_when=[optional])
    with pytest.raises(WebRuntimeSupportError):
        policy(not_applied_when=[outcome("web.unknown", {})])


def test_a_policy_revalidates_a_damaged_nested_record() -> None:
    damaged = outcome("web.url_match", {"url": DASHBOARD})
    damaged.__dict__["hidden"] = 1
    with pytest.raises(WebRuntimeSupportError):
        policy(applied_when=[damaged])


# --------------------------------------------------------------------------- support strictness


@pytest.mark.parametrize("kwargs", [
    {"adapter_name": ""},
    {"adapter_name": 5},
    {"require_url": 1},
    {"require_url": "true"},
    {"required_targets": "not-a-list"},
    {"required_targets": [USERNAME, USERNAME]},
    {"action_policies": []},
    {"action_policies": [policy(), policy()]},
])
def test_support_configuration_is_strict(kwargs: dict) -> None:
    base = {"adapter_name": "playwright_web", "required_targets": [WELCOME],
            "require_url": True, "action_policies": [policy()]}
    base.update(kwargs)
    with pytest.raises(WebRuntimeSupportError) as raised:
        WebRuntimeDomainSupport(**base)
    assert raised.value.code == "WEB_RUNTIME_SUPPORT_INVALID"


def test_support_rejects_an_undeclared_configuration_keyword() -> None:
    with pytest.raises(WebRuntimeSupportError):
        WebRuntimeDomainSupport(
            adapter_name="playwright_web", required_targets=[WELCOME], require_url=True,
            action_policies=[policy()], callback=lambda: None)


def test_support_accepts_a_plain_locator_mapping_and_is_detached() -> None:
    domain = support(targets=[target_payload(WELCOME)])
    assert domain.to_evidence(dashboard_observation()).state_complete is True


# --------------------------------------------------------------------------- to_evidence


def test_to_evidence_states_completeness_from_the_explicit_declarations() -> None:
    domain = support(targets=[USERNAME, WELCOME], require_url=True)
    complete = domain.to_evidence(login_observation())
    assert complete.state_complete is True
    assert complete.observation.observation_id == "obs-1"
    incomplete_url = domain.to_evidence(observation("http://127.0.0.1:8123/login", [
        record(USERNAME, 0), record(WELCOME, 0)]))
    assert incomplete_url.state_complete is True
    no_url = observation(None, [record(USERNAME, 1, visible=True),
                                record(WELCOME, 1, visible=True)])
    assert domain.to_evidence(no_url).state_complete is False
    missing_target = domain.to_evidence(observation(LOGIN, [record(USERNAME, 1,
                                                                  visible=True)]))
    assert missing_target.state_complete is False
    unavailable = domain.to_evidence(observation(LOGIN, [
        record(USERNAME, 1, visible=True),
        {"target": target_payload(WELCOME), "status": "unavailable",
         "match_count": None, "visible": None, "text": None, "value": None,
         "error_code": "TARGET_QUERY_FAILED"},
    ]))
    assert unavailable.state_complete is False


def test_null_attributes_do_not_make_an_otherwise_collected_target_incomplete() -> None:
    domain = support(targets=[WELCOME], require_url=False)
    result = domain.to_evidence(observation(LOGIN, [record(WELCOME, 1)]))
    assert result.state_complete is True


def test_a_structurally_damaged_snapshot_is_a_facility_error() -> None:
    domain = support()
    damaged = Observation(schema_version=SCHEMA, observation_id="obs-1",
                          environment_type="web", timestamp=NOW,
                          state={"schema_version": SCHEMA, "url": None,
                                 "targets": [{"target": target_payload(WELCOME),
                                              "status": "collected"}]})
    with pytest.raises(WebRuntimeSupportError) as raised:
        domain.to_evidence(damaged)
    assert raised.value.code == "WEB_RUNTIME_SUPPORT_INVALID"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


# --------------------------------------------------------------------------- prepare_contract


def test_prepare_contract_returns_the_matched_policy_with_one_web_rule_binding() -> None:
    preconditions = [precondition("web.url_match", {"url": LOGIN})]
    built = policy(preconditions=preconditions)
    domain = support(built)
    action = click_action()
    forecast = Forecast(
        schema_version=SCHEMA, forecast_id="forecast-1", proposed_action=action,
        expected_outcomes=[outcome("web.url_match", {"url": DASHBOARD}),
                           outcome("web.element_visible",
                                   {"target": target_payload(WELCOME)})])
    preparation = domain.prepare_contract(forecast=forecast,
                                          evidence=evidence(login_observation()))
    assert preparation.preconditions == built.preconditions
    assert preparation.recovery_policy == built.recovery_policy
    assert preparation.timeout_ms == built.timeout_ms
    assert [binding.outcome_index for binding in
            preparation.verification_policy.bindings] == [0, 1]
    assert all(binding.verifier_type == "web_rule" for binding in
               preparation.verification_policy.bindings)
    assert all(len(binding.config) == 0 for binding in
               preparation.verification_policy.bindings)
    # The forecast itself is never rewritten.
    assert forecast.proposed_action == action


def test_action_matching_is_type_sensitive_and_includes_metadata() -> None:
    domain = support(policy(action=navigate_action({"count": 1}), applied_when=[],
                            not_applied_when=[], retry_safety="read_only"))
    same = Forecast(schema_version=SCHEMA, forecast_id="f", proposed_action=navigate_action(
        {"count": 1}), expected_outcomes=dashboard_outcomes())
    assert domain.prepare_contract(forecast=same, evidence=evidence(login_observation()))

    for wrong_metadata in ({"count": True}, {"count": 1.0}, {"count": "1"}, {}):
        wrong = Forecast(schema_version=SCHEMA, forecast_id="f",
                         proposed_action=navigate_action(wrong_metadata),
                         expected_outcomes=dashboard_outcomes())
        with pytest.raises(WebRuntimeSupportError) as raised:
            domain.prepare_contract(forecast=wrong, evidence=evidence(login_observation()))
        assert raised.value.code == "WEB_RUNTIME_POLICY_NOT_FOUND"


def test_a_different_action_with_the_same_type_name_is_not_matched() -> None:
    domain = support(policy(action=click_action()))
    other = ActionSpec(schema_version=SCHEMA, action_type="web.click",
                       payload={"target": target_payload(USERNAME)})
    forecast = Forecast(schema_version=SCHEMA, forecast_id="f", proposed_action=other,
                        expected_outcomes=dashboard_outcomes())
    with pytest.raises(WebRuntimeSupportError) as raised:
        domain.prepare_contract(forecast=forecast, evidence=evidence(login_observation()))
    assert raised.value.code == "WEB_RUNTIME_POLICY_NOT_FOUND"


# --------------------------------------------------------------------------- safety and effect


def test_declare_action_safety_is_bound_to_the_real_committed_contract() -> None:
    action = click_action()
    outcomes = dashboard_outcomes()
    contract = build_contract(action, outcomes)
    domain = support(policy(action=action, retry_safety="non_idempotent"))
    declaration = domain.declare_action_safety(contract=contract)
    assert declaration.contract_id == "contract-1"
    assert declaration.adapter_name == "playwright_web"
    assert declaration.action_type == "web.click"
    assert declaration.retry_safety == "non_idempotent"


def test_declare_action_safety_refuses_an_unknown_action_or_foreign_adapter() -> None:
    domain = support(policy(action=click_action()))
    other = build_contract(navigate_action(), [outcome("web.url_match", {"url": LOGIN})])
    with pytest.raises(WebRuntimeSupportError) as raised:
        domain.declare_action_safety(contract=other)
    assert raised.value.code in ("WEB_RUNTIME_POLICY_NOT_FOUND",
                                 "WEB_RUNTIME_SUPPORT_INVALID")


def test_the_effect_table_is_exactly_the_declared_one() -> None:
    outcomes = dashboard_outcomes()
    action = click_action()
    contract = build_contract(action, outcomes)
    domain = support(policy(action=action, applied_when=outcomes, not_applied_when=[]))
    assert domain.assess_effect(
        contract=contract,
        actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                     status="unknown"),
        evidence=evidence(dashboard_observation())) == "applied"
    assert domain.assess_effect(
        contract=contract,
        actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                     status="succeeded"),
        evidence=evidence(login_observation())) == "unknown"


def test_conflicting_or_missing_effect_proof_is_unknown() -> None:
    action = click_action()
    outcomes = dashboard_outcomes()
    contract = build_contract(action, outcomes)
    both = support(policy(
        action=action,
        applied_when=[outcome("web.url_match", {"url": DASHBOARD})],
        not_applied_when=[outcome("web.element_visible",
                                  {"target": target_payload(WELCOME)})]))
    assert both.assess_effect(
        contract=contract,
        actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                     status="succeeded"),
        evidence=evidence(dashboard_observation())) == "unknown"

    negated = support(policy(
        action=action,
        applied_when=[outcome("web.url_match", {"url": DASHBOARD})],
        not_applied_when=[outcome("web.url_match", {"url": LOGIN})]))
    assert negated.assess_effect(
        contract=contract,
        actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                     status="failed"),
        evidence=evidence(login_observation())) == "not_applied"
    assert negated.assess_effect(
        contract=contract,
        actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                     status="succeeded"),
        evidence=evidence(dashboard_observation())) == "applied"


def test_no_effect_judgement_is_unknown_never_a_vacuous_pass() -> None:
    action = click_action()
    contract = build_contract(action, dashboard_outcomes())
    domain = support(policy(action=action, applied_when=[], not_applied_when=[]))
    assert domain.assess_effect(
        contract=contract,
        actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                     status="succeeded"),
        evidence=evidence(dashboard_observation())) == "unknown"


def test_empty_effect_groups_still_validate_a_damaged_evidence_record() -> None:
    """F03: an empty group means no judgement, never 'skip the evidence validation'."""
    action = click_action()
    contract = build_contract(action, dashboard_outcomes())
    domain = support(policy(action=action, applied_when=[], not_applied_when=[]))
    damaged = VerificationEvidence.model_construct(
        schema_version=SCHEMA, observation=object(), state_complete=True)
    with pytest.raises(WebRuntimeSupportError) as raised:
        domain.assess_effect(
            contract=contract,
            actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                         status="unknown"),
            evidence=damaged)
    assert raised.value.code == "WEB_RUNTIME_SUPPORT_INVALID"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    with pytest.raises(WebRuntimeSupportError):
        domain.assess_effect(
            contract=contract,
            actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                         status="unknown"),
            evidence=None)  # type: ignore[arg-type]


def test_empty_effect_groups_still_require_a_usable_web_observation() -> None:
    action = click_action()
    contract = build_contract(action, dashboard_outcomes())
    domain = support(policy(action=action, applied_when=[], not_applied_when=[]))
    malformed = VerificationEvidence(
        schema_version=SCHEMA,
        observation=Observation(schema_version=SCHEMA, observation_id="obs-bad",
                                environment_type="web", timestamp=NOW,
                                state={"schema_version": SCHEMA, "url": "not-a-url",
                                       "targets": {}}),
        state_complete=True)
    with pytest.raises(WebRuntimeSupportError) as raised:
        domain.assess_effect(
            contract=contract,
            actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                         status="unknown"),
            evidence=malformed)
    assert raised.value.code == "WEB_RUNTIME_SUPPORT_INVALID"


def test_legal_evidence_without_effect_conditions_is_the_documented_unknown() -> None:
    action = click_action()
    contract = build_contract(action, dashboard_outcomes())
    domain = support(policy(action=action, applied_when=[], not_applied_when=[]))
    assert domain.assess_effect(
        contract=contract,
        actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                     status="failed"),
        evidence=evidence(login_observation())) == "unknown"


def test_a_damaged_evidence_or_contract_is_a_facility_error_not_unknown() -> None:
    action = click_action()
    contract = build_contract(action, dashboard_outcomes())
    domain = support(policy(action=action))
    damaged = VerificationEvidence.model_construct(
        schema_version=SCHEMA, observation=object(), state_complete=True)
    with pytest.raises(WebRuntimeSupportError) as raised:
        domain.assess_effect(
            contract=contract,
            actual_outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="o-1",
                                         status="unknown"),
            evidence=damaged)
    assert raised.value.code == "WEB_RUNTIME_SUPPORT_INVALID"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_recovery_engine_refuses_a_real_web_step_verifier_as_post_attempt_evidence() -> None:
    """The historic node id is retained, but the correction removes the hardcoded label.

    A real ``WebStepVerifier`` result carries its true ``web_step_verifier`` label. The generic
    Core engine accepts it when the host declares that identity at construction, and the default
    JSON-configured engine still refuses it: the bound is the declared dependency identity, not a
    relabelled verdict and not a whitelist learned from the result.
    """
    action = click_action()
    contract = build_contract(action, dashboard_outcomes())
    verification_evidence = evidence(login_observation())
    step_result = WebStepVerifier().verify_step(contract=contract,
                                                evidence=verification_evidence)
    assert step_result.verifier == "web_step_verifier"
    assert step_result.status.value == "fail"
    outcome = ActualOutcome(schema_version=SCHEMA, outcome_id="o-1", status="unknown")
    context = RecoveryContext(
        schema_version=SCHEMA, run_id="run-1", step_id="step-1",
        contract_id=contract.contract_id, attempt_id="attempt-1",
        actual_outcome=outcome, remaining_budget_ms=30000.0,
        remaining_retries=0, remaining_replans=0, contract_retry_count=0,
        stop_requested=False)
    recovery_evidence = RecoveryEvidence(
        schema_version=SCHEMA, run_id="run-1", step_id="step-1",
        contract_id=contract.contract_id, attempt_id="attempt-1", outcome_id="o-1",
        verification_evidence=verification_evidence, step_result=step_result,
        effect_state="unknown", preconditions_status=VerificationStatus.PASS)
    safety = ActionSafetyDeclaration(
        schema_version=SCHEMA, contract_id=contract.contract_id,
        adapter_name="playwright_web", action_type="web.click",
        retry_safety="non_idempotent")
    engine = RecoveryEngine(step_verifier_name="web_step_verifier")
    assessment = engine.assess(contract=contract, context=context, safety=safety,
                               evidence=recovery_evidence)
    assert assessment.kind == "decision"
    assert assessment.decision == "abort"
    assert assessment.reason == "EFFECT_UNRESOLVED"
    with pytest.raises(RecoveryError) as raised:
        RecoveryEngine().assess(contract=contract, context=context, safety=safety,
                                evidence=recovery_evidence)
    assert raised.value.code == "EVIDENCE_INVALID"
    assert "step_result" in raised.value.location
    assert "verifier" in raised.value.location
