"""The step verifier: evaluate a committed contract's expected outcomes in order.

:class:`StepVerifier` never decides whether a task is complete; it produces the ordered verdict
of the committed contract's own expected outcomes and aggregates them by the required-all rule.

The whole request is validated before any evaluation starts. A missing, repeated or
out-of-range binding, a binding that points at another verifier, a non-empty binding config, an
unsupported outcome type and an unusable rule payload are all facility errors, and every
expected outcome is evaluated in list order - including the optional ones - so a later illegal
item is never skipped because an earlier item already failed.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from forecontract.contracts.models import ActionContract
from forecontract.models.actions import ExpectedOutcome
from forecontract.models.outcomes import (
    OutcomeVerification,
    VerificationResult,
    VerificationStatus,
)
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import (
    PATH_EXISTS,
    VALUE_EQUALS,
    PathExistsRule,
    RuleEvaluation,
    ValueEqualsRule,
    VerificationEvidence,
    normalise_path_segments,
    validate_declared_instance,
)
from forecontract.verification.rules import JsonRuleEvaluator, revalidate_argument

__all__ = ["StepVerifier"]

#: The one outcome type, precondition type and verifier type this package understands.
JSON_RULE = "json_rule"

#: The fixed overall reason codes of required-all aggregation.
REQUIRED_FAILED = "REQUIRED_FAILED"
REQUIRED_INCONCLUSIVE = "REQUIRED_INCONCLUSIVE"
REQUIRED_PASSED = "REQUIRED_PASSED"

#: The fixed verifier name of a step verdict.
STEP_VERIFIER = "json_step"

RULE_MODELS = (ValueEqualsRule, PathExistsRule)


def parse_json_rule(payload: object, location: tuple[str | int, ...]) -> object:
    """Build the rule named by ``rule_type`` from one closed payload object.

    The rule type is decided here, inside the facility, so a caller never has to guess which
    record an arbitrary mapping is. A missing, unknown or malformed rule is a rule error located
    at the payload, never at the caller's queried path.
    """
    if not isinstance(payload, Mapping):
        raise VerificationError(
            "VERIFICATION_RULE_INVALID", location, "the rule payload must be an object"
        )
    # A committed contract holds its payload in a deeply immutable mapping; the rule records are
    # built from an ordinary dict with a plain list path, so the same content is read back in
    # plain form first.
    payload = normalise_path_segments({key: payload[key] for key in payload})
    rule_type = payload.get("rule_type")
    model: type | None = None
    if rule_type == VALUE_EQUALS:
        model = ValueEqualsRule
    elif rule_type == PATH_EXISTS:
        model = PathExistsRule
    if model is None:
        raise VerificationError(
            "VERIFICATION_RULE_INVALID",
            location + ("rule_type",),
            "the rule type is not supported",
        )
    try:
        return model.model_validate(payload)
    except VerificationError as exc:
        raise VerificationError(
            "VERIFICATION_RULE_INVALID", location, exc.message, cause=exc
        ) from exc


def prevalidate_outcome(outcome: object, location: tuple[str | int, ...]) -> object:
    """Validate one expected outcome as a json_rule condition and return its rule."""
    if not isinstance(outcome, ExpectedOutcome):
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID", location, "a validated expected outcome is required"
        )
    validate_declared_instance(outcome, tuple(ExpectedOutcome.model_fields))
    if outcome.outcome_type != JSON_RULE:
        raise VerificationError(
            "VERIFICATION_RULE_INVALID",
            location + ("outcome_type",),
            "the outcome type is not supported",
        )
    return parse_json_rule(outcome.payload, location + ("payload",))


def prevalidate_contract(contract: object) -> list[object]:
    """Validate the whole contract and return its rules in expected-outcome list order.

    Nothing is evaluated until this function has returned, so a request that cannot be
    evaluated in full never produces a partial verdict.
    """
    if not isinstance(contract, ActionContract):
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID", ("contract",), "a validated contract is required"
        )
    validate_declared_instance(contract, tuple(ActionContract.model_fields))
    outcomes = list(contract.expected_outcomes)
    if not outcomes:
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("contract", "expected_outcomes"),
            "at least one expected outcome is required",
        )
    if not any(outcome.required for outcome in outcomes):
        raise VerificationError(
            "VERIFICATION_BINDING_INVALID",
            ("contract", "expected_outcomes"),
            "at least one expected outcome must be required",
        )
    rules = [
        prevalidate_outcome(outcome, ("contract", "expected_outcomes", index))
        for index, outcome in enumerate(outcomes)
    ]
    bindings = list(contract.verification_policy.bindings)
    indices = [binding.outcome_index for binding in bindings]
    if len(set(indices)) != len(indices):
        raise VerificationError(
            "VERIFICATION_BINDING_INVALID",
            ("contract", "verification_policy", "bindings"),
            "verification bindings must not repeat an outcome index",
        )
    if sorted(indices) != list(range(len(outcomes))):
        raise VerificationError(
            "VERIFICATION_BINDING_INVALID",
            ("contract", "verification_policy", "bindings"),
            "every expected outcome needs exactly one verification binding",
        )
    for position, binding in enumerate(bindings):
        location = ("contract", "verification_policy", "bindings", position)
        validate_declared_instance(binding, tuple(type(binding).model_fields))
        if binding.verifier_type != JSON_RULE:
            raise VerificationError(
                "VERIFICATION_BINDING_INVALID",
                location + ("verifier_type",),
                "the verifier type is not supported",
            )
        if len(binding.config) != 0:
            raise VerificationError(
                "VERIFICATION_BINDING_INVALID",
                location + ("config",),
                "a json_rule binding must not carry a rule definition",
            )
    return rules


def required_all_aggregate(results: list[RuleEvaluation], required: list[bool]) -> VerificationStatus:
    """Aggregate required items only: FAIL beats INCONCLUSIVE, otherwise PASS."""
    required_statuses = [
        result.status for result, is_required in zip(results, required) if is_required
    ]
    if not required_statuses:
        raise VerificationError(
            "VERIFICATION_BINDING_INVALID", ("<set>",), "at least one required item is required"
        )
    if VerificationStatus.FAIL in required_statuses:
        return VerificationStatus.FAIL
    if VerificationStatus.INCONCLUSIVE in required_statuses:
        return VerificationStatus.INCONCLUSIVE
    if all(status is VerificationStatus.PASS for status in required_statuses):
        return VerificationStatus.PASS
    raise VerificationError(
        "VERIFICATION_INPUT_INVALID", ("<set>",), "the aggregated status is not defined"
    )


def aggregate_reason(status: VerificationStatus) -> str:
    """The fixed overall reason code for one aggregated status."""
    if status is VerificationStatus.FAIL:
        return REQUIRED_FAILED
    if status is VerificationStatus.INCONCLUSIVE:
        return REQUIRED_INCONCLUSIVE
    return REQUIRED_PASSED


def _outcome_verification(index: int, evaluation: RuleEvaluation) -> OutcomeVerification:
    """Reuse the existing Core record for one evaluated item."""
    try:
        return OutcomeVerification(
            schema_version="0.1",
            outcome_index=index,
            status=evaluation.status,
            has_actual_value=evaluation.has_actual_value,
            actual_value=evaluation.actual_value,
            reason=evaluation.reason,
        )
    except ValidationError as exc:
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("results", index),
            "the evaluated item does not match the declared schema",
            cause=exc,
        ) from exc


class StepVerifier:
    """Verify one committed contract against one piece of evidence.

    The class holds no state and takes no constructor argument, so the same request always
    produces the same verdict.
    """

    def verify_step(self, *, contract: ActionContract,
                    evidence: VerificationEvidence) -> VerificationResult:
        """Return the ordered step verdict of the committed contract."""
        revalidate_argument(evidence, VerificationEvidence, "evidence")
        rules = prevalidate_contract(contract)
        outcomes = list(contract.expected_outcomes)
        evaluator = JsonRuleEvaluator()
        evaluations = [
            evaluator.evaluate(rule=rule, evidence=evidence) for rule in rules
        ]
        results = [
            _outcome_verification(index, evaluation)
            for index, evaluation in enumerate(evaluations)
        ]
        status = required_all_aggregate(evaluations, [outcome.required for outcome in outcomes])
        try:
            return VerificationResult(
                schema_version="0.1",
                status=status,
                verifier=STEP_VERIFIER,
                outcome_results=results,
                reason=aggregate_reason(status),
                metadata={
                    "scope": "step",
                    "contract_id": contract.contract_id,
                    "goal_id": contract.goal_id,
                    "observation_id": evidence.observation.observation_id,
                    "state_complete": evidence.state_complete,
                },
            )
        except ValidationError as exc:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("result",),
                "the step result does not match the declared schema",
                cause=exc,
            ) from exc
