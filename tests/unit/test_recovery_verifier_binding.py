"""Acceptance of the generic Recovery step-verifier identity binding.

The Core recovery engine used to compare every post-attempt step result with one hardcoded
``json_step`` label. A real Web host produces ``web_step_verifier`` results, so that hardcode made
a genuine verdict unusable. The correction is a generic constructor-declared identity: the host
says which verifier it composed, the engine checks the real label against that declaration and
never learns a label from the verdict it is judging.

These tests use the real committed ``ActionContract``, the real Core ``StepVerifier`` and a real
``verification.VerificationResult`` carrying the Web label. The legacy JSON path is proved to stay
byte-for-byte compatible.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.contracts import (
    ContractBuilder,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import (
    ActualOutcome,
    ErrorInfo,
    OutcomeVerification,
    VerificationResult,
    VerificationStatus,
)
from forecontract.recovery import (
    ActionSafetyDeclaration,
    RecoveryContext,
    RecoveryEngine,
    RecoveryError,
    RecoveryEvidence,
)
from forecontract.verification import StepVerifier, VerificationEvidence

NOW = datetime.datetime(2026, 10, 10, 12, 0, tzinfo=datetime.timezone.utc)
JSON_STEP = "json_step"
WEB_STEP = "web_step_verifier"
PAYLOAD = {"schema_version": "0.1", "rule_type": "path_exists", "path": ["a"]}


class _Adapter:
    name = "fake-1"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(schema_version="0.1", adapter_name="fake-1",
                                   environment_type="fake", supported_actions=["click"],
                                   supported_condition_types=["json_rule"],
                                   supported_outcome_types=["json_rule"])

    def validate_action(self, candidate: ActionSpec) -> None:
        del candidate


class _Hooks:
    def validate_precondition(self, precondition: object) -> None:
        del precondition

    def validate_expected_outcome(self, outcome: object) -> None:
        del outcome

    def validate_verification_binding(self, binding: object, outcome: object) -> None:
        del binding, outcome

    def check_precondition(self, precondition: object, observed: Observation):
        del precondition, observed
        return VerificationStatus.PASS


def build(*, policy: RecoveryPolicy | None = None, contract_id: str = "c-1",
          goal_id: str = "g-1", observation_id: str = "obs-1") -> object:
    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                              payload=PAYLOAD)
    extra = {} if policy is None else {"recovery_policy": policy}
    return ContractBuilder().build(
        forecast=Forecast(schema_version="0.1", forecast_id="f-1", proposed_action=ActionSpec(
            schema_version="0.1", action_type="click", payload={}),
            expected_outcomes=[outcome]),
        observation=Observation(schema_version="0.1", observation_id=observation_id,
                                environment_type="fake", state={"a": 1}, timestamp=NOW),
        adapter=_Adapter(), hooks=_Hooks(), preconditions=[],
        verification_policy=VerificationPolicy(schema_version="0.1", bindings=[
            VerificationBinding(schema_version="0.1", outcome_index=0,
                                verifier_type="json_rule")]),
        timeout_ms=100, remaining_budget_ms=250, contract_id=contract_id, goal_id=goal_id,
        created_at=NOW, **extra,
    )


def bounded(max_retries: int = 0, allow_replan: bool = False) -> RecoveryPolicy:
    return RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=max_retries,
                          allow_replan=allow_replan)


def context(contract: object, *, status: str = "failed") -> RecoveryContext:
    error = None if status == "succeeded" else ErrorInfo(
        schema_version="0.1", code="E", message="m", recoverable=True)
    return RecoveryContext(
        schema_version="0.1", run_id="r-1", step_id="s-1", contract_id=contract.contract_id,
        attempt_id="a-1",
        actual_outcome=ActualOutcome(schema_version="0.1", outcome_id="o-1", status=status,
                                     error=error),
        remaining_budget_ms=100.0, remaining_retries=0, remaining_replans=0,
        contract_retry_count=0, stop_requested=False)


def safety(contract: object, retry_safety: str = "non_idempotent") -> ActionSafetyDeclaration:
    return ActionSafetyDeclaration(schema_version="0.1", contract_id=contract.contract_id,
                                   adapter_name=contract.adapter_name,
                                   action_type=contract.action.action_type,
                                   retry_safety=retry_safety)


def piece(contract: object, *, observation_id: str = "obs-1",
          complete: bool = True, state: dict | None = None) -> VerificationEvidence:
    observed = Observation(schema_version="0.1", observation_id=observation_id,
                           environment_type="fake",
                           state={"a": 1} if state is None else state, timestamp=NOW)
    return VerificationEvidence(schema_version="0.1", observation=observed,
                                state_complete=complete)


def web_result(contract: object, *, observation_id: str = "obs-1", complete: bool = True,
               status: str = "fail") -> VerificationResult:
    """Build a real-shaped Web step-verifier result with its true declared label."""
    reason = "REQUIRED_PASSED" if status == "pass" else "REQUIRED_FAILED"
    item = OutcomeVerification(schema_version="0.1", outcome_index=0,
                               status=VerificationStatus(status), reason=reason)
    return VerificationResult(
        schema_version="0.1", status=VerificationStatus(status), verifier=WEB_STEP,
        outcome_results=[item], reason=reason,
        metadata={"scope": "step", "contract_id": contract.contract_id,
                  "goal_id": contract.goal_id, "observation_id": observation_id,
                  "state_complete": complete},
    )


def recovery_evidence(contract: object, evidence_piece: VerificationEvidence,
                      result: VerificationResult, *, effect: str = "unknown",
                      observation_id: str = "obs-1") -> RecoveryEvidence:
    return RecoveryEvidence(
        schema_version="0.1", run_id="r-1", step_id="s-1",
        contract_id=contract.contract_id, attempt_id="a-1", outcome_id="o-1",
        verification_evidence=evidence_piece, step_result=result,
        effect_state=effect, preconditions_status=VerificationStatus.PASS)


def assess(engine: RecoveryEngine, contract: object, evidence_piece: RecoveryEvidence | None,
           *, ctx: RecoveryContext | None = None):
    return engine.assess(contract=contract, context=ctx or context(contract),
                         safety=safety(contract), evidence=evidence_piece)


class _NameWithSuffix(str):
    """A real ``str`` subclass, which the constructor must never silently accept."""


# --- the default and the declared identity -------------------------------------------------


def test_the_default_engine_binds_the_json_step_identity() -> None:
    assert RecoveryEngine()._step_verifier_name == JSON_STEP
    assert RecoveryEngine(step_verifier_name=JSON_STEP)._step_verifier_name == JSON_STEP


def test_a_real_web_step_verifier_is_accepted_when_declared() -> None:
    """F01/F06: the host-declared Web identity makes a real Web verdict usable."""
    contract = build(policy=bounded())
    ctx = context(contract)
    evidence_piece = piece(contract)
    result = web_result(contract, status="fail")
    engine = RecoveryEngine(step_verifier_name=WEB_STEP)
    assessment = assess(engine, contract,
                        recovery_evidence(contract, evidence_piece, result), ctx=ctx)
    assert assessment.kind == "decision"
    assert assessment.decision == "abort"
    assert assessment.reason == "EFFECT_UNRESOLVED"


def test_a_web_step_verifier_is_refused_by_the_default_engine() -> None:
    contract = build(policy=bounded())
    evidence_piece = piece(contract)
    result = web_result(contract, status="fail")
    with pytest.raises(RecoveryError) as raised:
        assess(RecoveryEngine(), contract,
               recovery_evidence(contract, evidence_piece, result))
    assert raised.value.code == "EVIDENCE_INVALID"
    assert raised.value.location == ("evidence", "step_result", "verifier")


def test_a_mismatched_declared_label_is_refused() -> None:
    """A host may not relabel a verdict, and a declaration does not authenticate a producer."""
    contract = build(policy=bounded())
    web_piece = piece(contract)
    with pytest.raises(RecoveryError) as raised:
        assess(RecoveryEngine(step_verifier_name="some_other_verifier"), contract,
               recovery_evidence(contract, web_piece, web_result(contract, status="fail")))
    assert raised.value.code == "EVIDENCE_INVALID"
    assert raised.value.location == ("evidence", "step_result", "verifier")

    json_piece = piece(contract)
    json_result = StepVerifier().verify_step(contract=contract, evidence=json_piece)
    with pytest.raises(RecoveryError) as raised:
        assess(RecoveryEngine(step_verifier_name=WEB_STEP), contract,
               recovery_evidence(contract, json_piece, json_result))
    assert raised.value.code == "EVIDENCE_INVALID"


def test_the_engine_never_learns_a_label_from_the_result() -> None:
    contract = build(policy=bounded())
    evidence_piece = piece(contract)
    web_piece = recovery_evidence(contract, evidence_piece, web_result(contract))
    engine = RecoveryEngine()
    for _ in range(2):
        with pytest.raises(RecoveryError):
            assess(engine, contract, web_piece)
        assert engine._step_verifier_name == JSON_STEP
    legacy = StepVerifier().verify_step(contract=contract, evidence=evidence_piece)
    assert assess(engine, contract,
                  recovery_evidence(contract, evidence_piece, legacy)).kind == "no_recovery"


# --- the configured name is validated and re-checked ---------------------------------------


@pytest.mark.parametrize("name", [None, "", "   ", 1, True, b"json_step", "x" * 129,
                                  _NameWithSuffix(JSON_STEP)])
def test_an_unusable_configured_name_is_refused(name: object) -> None:
    with pytest.raises(RecoveryError) as raised:
        RecoveryEngine(step_verifier_name=name)  # type: ignore[arg-type]
    assert raised.value.code == "POLICY_INVALID"
    assert raised.value.location == ("step_verifier_name",)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_the_constructor_refusal_does_not_carry_the_supplied_name() -> None:
    marker = "SYNTHETIC_STEP_VERIFIER_NAME"
    with pytest.raises(RecoveryError) as raised:
        RecoveryEngine(step_verifier_name=marker + "x" * 129)
    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)


def test_the_configured_identity_is_rechecked_on_every_assess() -> None:
    contract = build(policy=bounded())
    evidence_piece = piece(contract)
    result = StepVerifier().verify_step(contract=contract, evidence=evidence_piece)
    engine = RecoveryEngine()
    engine._step_verifier_name = 123  # simulate external damage to the stored configuration
    with pytest.raises(RecoveryError) as raised:
        assess(engine, contract, recovery_evidence(contract, evidence_piece, result))
    assert raised.value.code == "POLICY_INVALID"


# --- the legacy JSON path stays compatible -------------------------------------------------


def test_the_legacy_json_path_is_unchanged() -> None:
    contract = build(policy=bounded())
    json_piece = piece(contract)
    result = StepVerifier().verify_step(contract=contract, evidence=json_piece)
    assessment = assess(RecoveryEngine(), contract,
                        recovery_evidence(contract, json_piece, result, effect="not_applied"),
                        ctx=context(contract, status="succeeded"))
    assert assessment.kind == "no_recovery"
    assert assessment.reason == "STEP_VERIFIED"

    retry_contract = build(policy=bounded(max_retries=2, allow_replan=False))
    retry_piece = piece(retry_contract, state={})
    retry_result = StepVerifier().verify_step(contract=retry_contract, evidence=retry_piece)
    assert retry_result.status.value == "fail"
    retry = assess(RecoveryEngine(), retry_contract,
                   recovery_evidence(retry_contract, retry_piece, retry_result,
                                     effect="not_applied"),
                   ctx=RecoveryContext(
                       schema_version="0.1", run_id="r-1", step_id="s-1",
                       contract_id=retry_contract.contract_id, attempt_id="a-1",
                       actual_outcome=ActualOutcome(
                           schema_version="0.1", outcome_id="o-1", status="failed",
                           error=ErrorInfo(schema_version="0.1", code="E", message="m",
                                           recoverable=True)),
                       remaining_budget_ms=100.0, remaining_retries=1, remaining_replans=0,
                       contract_retry_count=0, stop_requested=False))
    assert retry.kind == "decision" and retry.decision == "retry"


# --- the declared identity does not relax the association checks ---------------------------


def test_the_declared_identity_preserves_the_contract_association_checks() -> None:
    contract = build(policy=bounded())
    evidence_piece = piece(contract)
    result = web_result(contract, status="fail")
    engine = RecoveryEngine(step_verifier_name=WEB_STEP)
    forged_metadata = result.model_dump()
    forged_metadata["metadata"]["contract_id"] = "other-contract"
    forged = VerificationResult.model_validate(forged_metadata)
    with pytest.raises(RecoveryError) as raised:
        assess(engine, contract,
               recovery_evidence(contract, evidence_piece, forged))
    assert raised.value.code == "EVIDENCE_INVALID"
    assert "contract_id" in raised.value.location


def test_the_declared_identity_preserves_the_observation_association_checks() -> None:
    contract = build(policy=bounded())
    evidence_piece = piece(contract)
    result = web_result(contract, status="fail")
    engine = RecoveryEngine(step_verifier_name=WEB_STEP)
    forged_metadata = result.model_dump()
    forged_metadata["metadata"]["observation_id"] = "another-observation"
    forged = VerificationResult.model_validate(forged_metadata)
    with pytest.raises(RecoveryError) as raised:
        assess(engine, contract,
               recovery_evidence(contract, evidence_piece, forged))
    assert raised.value.code == "EVIDENCE_INVALID"
    assert "observation_id" in raised.value.location


def test_the_declared_identity_preserves_the_outcome_coverage_checks() -> None:
    contract = build(policy=bounded())
    evidence_piece = piece(contract)
    result = web_result(contract, status="fail")
    engine = RecoveryEngine(step_verifier_name=WEB_STEP)
    forged_metadata = result.model_dump()
    forged_metadata["outcome_results"] = []
    forged_metadata["status"] = "fail"
    forged = VerificationResult.model_validate(forged_metadata)
    with pytest.raises(RecoveryError) as raised:
        assess(engine, contract,
               recovery_evidence(contract, evidence_piece, forged))
    assert raised.value.code == "EVIDENCE_INVALID"
    assert "outcome_results" in raised.value.location
