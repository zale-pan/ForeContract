"""Separate Web step verification and independent task-completion verdicts.

Two upper-level entry points sit above the same pure
:class:`~forecontract_web.rules.WebRuleEvaluator` but answer different questions:

* :class:`WebStepVerifier` verifies the committed action contract's own expected outcomes. It
  validates the whole contract, every expected outcome - including the optional ones - and the
  whole evidence before it produces any result, and it returns the Core
  :class:`~forecontract.models.outcomes.VerificationResult` with one ordered
  :class:`~forecontract.models.outcomes.OutcomeVerification` per expected outcome.
* :class:`WebTaskCompletionOracle` evaluates only independently supplied
  :class:`~forecontract.verification.models.TaskCompletionCriteria` conditions. It never reads a
  forecast, a step verdict, a planner hint or an execution status, so a passing step is not task
  completion and a failing step cannot suppress an independently passing task.

Both consume the same evidence truth table and aggregate only required items: ``FAIL`` first,
then ``INCONCLUSIVE``, and ``PASS`` only when every required item passed. An empty set or a set
with no required item is a facility error. Every valid item is evaluated, so an unknown or
malformed optional item is never ignored because an earlier item already failed, and an
optional pass can never cancel a required failure.

The measured Web values stay in the supplied observation. Each item therefore reports
``has_actual_value=False`` and ``actual_value=None`` and refers to its input through
``observation_id`` plus the ordered ``rule_evidence_refs`` metadata; no copied URL, text or
value appears in a result.

This module is not a Runtime integration, a `RuntimeDomainSupport`, a retry policy or a
full-site end-to-end claim. It performs no I/O, starts no browser and modifies no input.
"""

from __future__ import annotations

from typing import Final

from forecontract.contracts.models import ActionContract
from forecontract.models.actions import ExpectedOutcome
from forecontract.models.outcomes import (
    OutcomeVerification,
    VerificationResult,
    VerificationStatus,
)
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from pydantic import BaseModel

from forecontract_web.rule_models import WEB_RULE_TYPES, WebRuleEvaluation
from forecontract_web.rules import (
    WebRuleInvalid,
    WebVerificationInputInvalid,
    _evaluate_prepared,
    _prepare_rule,
    _revalidate_evidence,
    _revalidated_model,
)

__all__ = ["WebStepVerifier", "WebTaskCompletionOracle"]

#: The one verifier type a Web step binding may declare.
WEB_RULE_BINDING: Final[str] = "web_rule"
#: The two fixed verifier names of this round.
STEP_VERIFIER: Final[str] = "web_step_verifier"
TASK_VERIFIER: Final[str] = "web_task_completion_oracle"

#: The fixed overall reason codes of required-all aggregation, matching Core's wording.
REQUIRED_FAILED: Final[str] = "REQUIRED_FAILED"
REQUIRED_INCONCLUSIVE: Final[str] = "REQUIRED_INCONCLUSIVE"
REQUIRED_PASSED: Final[str] = "REQUIRED_PASSED"

_SCHEMA_VERSION: Final[str] = "0.1"
_CONTRACT_MESSAGE: Final[str] = "the action contract is not usable"
_CRITERIA_MESSAGE: Final[str] = "the task completion criteria are not usable"
_RESULT_MESSAGE: Final[str] = "the verification result does not match the declared schema"


def _known_rule_type(rule_type: object) -> bool:
    """Whether a value spells one of the five supported rule type names exactly."""
    if not isinstance(rule_type, str):
        return False
    return any(str.__eq__(rule_type, name) for name in WEB_RULE_TYPES)


def _prepare_outcome(outcome: object, location: tuple[str | int, ...]) -> tuple[str, BaseModel]:
    """Validate one expected outcome completely and return its prepared rule."""
    if not isinstance(outcome, ExpectedOutcome):
        raise WebVerificationInputInvalid(
            "a validated expected outcome is required", path=location
        )
    rule_type = outcome.outcome_type
    if not _known_rule_type(rule_type):
        raise WebRuleInvalid(
            "the web rule type is not supported", path=location + ("outcome_type",)
        )
    return _prepare_rule(rule_type, outcome.payload, location=location + ("payload",))


def _revalidate_contract(contract: object) -> ActionContract:
    """Revalidate the whole contract and its index bindings before any evaluation."""
    rebuilt = _revalidated_model(
        contract, ActionContract, _CONTRACT_MESSAGE,
        WebVerificationInputInvalid, ("contract",),
    )
    outcomes = list(rebuilt.expected_outcomes)
    if not outcomes:
        raise WebVerificationInputInvalid(
            "at least one expected outcome is required",
            path=("contract", "expected_outcomes"),
        )
    if not any(outcome.required for outcome in outcomes):
        raise WebVerificationInputInvalid(
            "at least one expected outcome must be required",
            path=("contract", "expected_outcomes"),
        )
    bindings = list(rebuilt.verification_policy.bindings)
    indices = [binding.outcome_index for binding in bindings]
    if len(set(indices)) != len(indices):
        raise WebVerificationInputInvalid(
            "verification bindings must not repeat an outcome index",
            path=("contract", "verification_policy", "bindings"),
        )
    if sorted(indices) != list(range(len(outcomes))):
        raise WebVerificationInputInvalid(
            "every expected outcome needs exactly one verification binding",
            path=("contract", "verification_policy", "bindings"),
        )
    for position, binding in enumerate(bindings):
        location = ("contract", "verification_policy", "bindings", position)
        if not (
            isinstance(binding.verifier_type, str)
            and str.__eq__(binding.verifier_type, WEB_RULE_BINDING)
        ):
            raise WebVerificationInputInvalid(
                "the verifier type is not supported", path=location + ("verifier_type",)
            )
        if len(binding.config) != 0:
            raise WebVerificationInputInvalid(
                "a web_rule binding must not carry a configuration",
                path=location + ("config",),
            )
    return rebuilt


def _revalidate_criteria(criteria: object) -> TaskCompletionCriteria:
    """Revalidate the whole criteria record before any condition is evaluated."""
    rebuilt = _revalidated_model(
        criteria, TaskCompletionCriteria, _CRITERIA_MESSAGE,
        WebVerificationInputInvalid, ("criteria",),
    )
    conditions = list(rebuilt.conditions)
    if not conditions:
        raise WebVerificationInputInvalid(
            "at least one completion condition is required",
            path=("criteria", "conditions"),
        )
    if not any(condition.required for condition in conditions):
        raise WebVerificationInputInvalid(
            "at least one completion condition must be required",
            path=("criteria", "conditions"),
        )
    return rebuilt


def _aggregate_status(evaluations: list[WebRuleEvaluation],
                      required: list[bool]) -> VerificationStatus:
    """Aggregate required items only: FAIL beats INCONCLUSIVE, otherwise PASS."""
    required_statuses = [
        evaluation.status
        for evaluation, is_required in zip(evaluations, required)
        if is_required
    ]
    if not required_statuses:
        raise WebVerificationInputInvalid(
            "at least one required item is required", path=("<set>",)
        )
    if VerificationStatus.FAIL in required_statuses:
        return VerificationStatus.FAIL
    if VerificationStatus.INCONCLUSIVE in required_statuses:
        return VerificationStatus.INCONCLUSIVE
    if all(status is VerificationStatus.PASS for status in required_statuses):
        return VerificationStatus.PASS
    raise WebVerificationInputInvalid(
        "the aggregated status is not defined", path=("<set>",)
    )


def _aggregate_reason(status: VerificationStatus) -> str:
    """The fixed overall reason code for one aggregated status."""
    if status is VerificationStatus.FAIL:
        return REQUIRED_FAILED
    if status is VerificationStatus.INCONCLUSIVE:
        return REQUIRED_INCONCLUSIVE
    return REQUIRED_PASSED


def _outcome_result(index: int, evaluation: WebRuleEvaluation) -> OutcomeVerification:
    """One ordered per-item result that keeps the measured value in the evidence."""
    try:
        return OutcomeVerification(
            schema_version=_SCHEMA_VERSION,
            outcome_index=index,
            status=evaluation.status,
            has_actual_value=False,
            actual_value=None,
            reason=evaluation.reason,
        )
    except Exception:  # noqa: BLE001 - a self-contradictory result is a bounded input error
        raise WebVerificationInputInvalid(
            "the evaluated item does not match the declared schema",
            path=("results", index),
        )


def _evidence_refs(evaluations: list[WebRuleEvaluation]) -> list[dict]:
    """The ordered per-item locator metadata of one result."""
    return [
        {"rule_type": evaluation.rule_type, "target_index": evaluation.target_index}
        for evaluation in evaluations
    ]


class WebStepVerifier:
    """Verify one committed contract's expected outcomes against one piece of evidence.

    The class holds no state and takes no constructor argument, so the same request always
    produces the same verdict and no callback can be injected.
    """

    def verify_step(self, *, contract: ActionContract,
                    evidence: VerificationEvidence) -> VerificationResult:
        """Return the ordered step verdict of the committed contract."""
        observation, state, state_complete = _revalidate_evidence(evidence)
        rebuilt = _revalidate_contract(contract)
        outcomes = list(rebuilt.expected_outcomes)
        prepared = [
            _prepare_outcome(outcome, ("contract", "expected_outcomes", index))
            for index, outcome in enumerate(outcomes)
        ]
        evaluations = [
            _evaluate_prepared(item, state, observation.observation_id)
            for item in prepared
        ]
        results = [
            _outcome_result(index, evaluation)
            for index, evaluation in enumerate(evaluations)
        ]
        status = _aggregate_status(
            evaluations, [outcome.required for outcome in outcomes]
        )
        metadata = {
            "scope": "step",
            "contract_id": rebuilt.contract_id,
            "goal_id": rebuilt.goal_id,
            "observation_id": observation.observation_id,
            "state_complete": state_complete,
            "rule_evidence_refs": _evidence_refs(evaluations),
        }
        try:
            return VerificationResult(
                schema_version=_SCHEMA_VERSION,
                status=status,
                verifier=STEP_VERIFIER,
                outcome_results=results,
                reason=_aggregate_reason(status),
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001 - a malformed aggregate is a bounded input error
            raise WebVerificationInputInvalid(_RESULT_MESSAGE, path=("result",))


class WebTaskCompletionOracle:
    """Evaluate independently supplied completion criteria against one piece of evidence.

    The class holds no state and takes no constructor argument, so the same request always
    produces the same verdict, and it cannot consult a forecast, a step result or an execution
    status even indirectly.
    """

    def evaluate_completion(self, *, criteria: TaskCompletionCriteria,
                            evidence: VerificationEvidence) -> VerificationResult:
        """Return the task-completion verdict for the independently supplied criteria."""
        observation, state, state_complete = _revalidate_evidence(evidence)
        rebuilt = _revalidate_criteria(criteria)
        conditions = list(rebuilt.conditions)
        prepared = [
            _prepare_outcome(condition, ("criteria", "conditions", index))
            for index, condition in enumerate(conditions)
        ]
        evaluations = [
            _evaluate_prepared(item, state, observation.observation_id)
            for item in prepared
        ]
        results = [
            _outcome_result(index, evaluation)
            for index, evaluation in enumerate(evaluations)
        ]
        status = _aggregate_status(
            evaluations, [condition.required for condition in conditions]
        )
        metadata = {
            "scope": "task",
            "goal_id": rebuilt.goal_id,
            "observation_id": observation.observation_id,
            "state_complete": state_complete,
            "rule_evidence_refs": _evidence_refs(evaluations),
        }
        try:
            return VerificationResult(
                schema_version=_SCHEMA_VERSION,
                status=status,
                verifier=TASK_VERIFIER,
                outcome_results=results,
                reason=_aggregate_reason(status),
                metadata=metadata,
            )
        except Exception:  # noqa: BLE001 - a malformed aggregate is a bounded input error
            raise WebVerificationInputInvalid(_RESULT_MESSAGE, path=("result",))
