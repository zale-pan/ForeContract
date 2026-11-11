"""The independent task-completion oracle.

:class:`TaskCompletionOracle` answers a different question from the step verifier: not "did the
committed action's outcomes verify" but "do independently supplied completion conditions hold
for this evidence". It is therefore given a :class:`TaskCompletionCriteria`, never a forecast,
never an action contract and never a completion hint.

Passing every step is not task completion, and a planner's ``task_completion_hint`` has no
influence here. An unmet condition means the current criteria are not satisfied; it is not a
verdict that the task can never succeed, and this round sets no terminal state.
"""

from __future__ import annotations

from pydantic import ValidationError

from forecontract.models.outcomes import (
    OutcomeVerification,
    VerificationResult,
    VerificationStatus,
)
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import (
    RuleEvaluation,
    TaskCompletionCriteria,
    VerificationEvidence,
    validate_declared_instance,
)
from forecontract.verification.rules import JsonRuleEvaluator, revalidate_argument
from forecontract.verification.step import (
    _outcome_verification,
    aggregate_reason,
    prevalidate_outcome,
    required_all_aggregate,
)

__all__ = ["TaskCompletionOracle"]

#: The fixed verifier name of a task verdict.
TASK_VERIFIER = "json_task"


def prevalidate_criteria(criteria: object) -> list[object]:
    """Validate the whole criteria record and return its rules in condition list order."""
    if not isinstance(criteria, TaskCompletionCriteria):
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID", ("criteria",), "validated criteria are required"
        )
    validate_declared_instance(criteria, tuple(TaskCompletionCriteria.model_fields))
    conditions = list(criteria.conditions)
    if not conditions:
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("criteria", "conditions"),
            "at least one completion condition is required",
        )
    if not any(condition.required for condition in conditions):
        raise VerificationError(
            "VERIFICATION_BINDING_INVALID",
            ("criteria", "conditions"),
            "at least one completion condition must be required",
        )
    return [
        prevalidate_outcome(condition, ("criteria", "conditions", index))
        for index, condition in enumerate(conditions)
    ]


class TaskCompletionOracle:
    """Evaluate independently supplied completion criteria against one piece of evidence.

    The class holds no state and takes no constructor argument, so the same request always
    produces the same verdict, and it cannot consult a forecast even indirectly.
    """

    def evaluate_completion(self, *, criteria: TaskCompletionCriteria,
                            evidence: VerificationEvidence) -> VerificationResult:
        """Return the task-completion verdict for the supplied criteria."""
        revalidate_argument(evidence, VerificationEvidence, "evidence")
        rules = prevalidate_criteria(criteria)
        conditions = list(criteria.conditions)
        evaluator = JsonRuleEvaluator()
        evaluations: list[RuleEvaluation] = [
            evaluator.evaluate(rule=rule, evidence=evidence) for rule in rules
        ]
        results: list[OutcomeVerification] = [
            _outcome_verification(index, evaluation)
            for index, evaluation in enumerate(evaluations)
        ]
        status = required_all_aggregate(
            evaluations, [condition.required for condition in conditions]
        )
        try:
            return VerificationResult(
                schema_version="0.1",
                status=status,
                verifier=TASK_VERIFIER,
                outcome_results=results,
                reason=aggregate_reason(status),
                metadata={
                    "scope": "task",
                    "goal_id": criteria.goal_id,
                    "observation_id": evidence.observation.observation_id,
                    "state_complete": evidence.state_complete,
                },
            )
        except ValidationError as exc:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("result",),
                "the task result does not match the declared schema",
                cause=exc,
            ) from exc


_ = VerificationStatus
