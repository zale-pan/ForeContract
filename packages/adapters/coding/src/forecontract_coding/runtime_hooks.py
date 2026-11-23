"""Pure synchronous Core contract validation hooks and task-criteria validator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
from pydantic import ValidationError

from forecontract_coding.models import (
    CODING_ENVIRONMENT_TYPE,
    CODING_VERIFIER_TYPE,
    SUPPORTED_CONDITION_TYPES,
    SUPPORTED_OUTCOME_TYPES,
    CodingFileWrittenPayload,
    CodingScopeValidPayload,
    CodingTask,
    CodingTaskVerifiedPayload,
    CodingToolResultPayload,
    CodingWritePreconditionPayload,
    thaw_json_value,
)

__all__ = [
    "CodingContractHooksFactory",
    "CodingContractValidationHooks",
    "CodingTaskCriteriaValidator",
]


def _attempt(call: Callable[[], Any]) -> tuple[bool, Any]:
    try:
        return True, call()
    except Exception:  # noqa: BLE001
        return False, None


class CodingTaskCriteriaValidator:
    """Reject empty, all-optional, or ``json_rule``/non-Coding task completion criteria."""

    def __init__(self, *, task: CodingTask | None = None) -> None:
        self._task = CodingTask.model_validate(task) if task is not None else None

    def validate(self, *, criteria: TaskCompletionCriteria) -> None:
        if not isinstance(criteria, TaskCompletionCriteria):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("criteria",),
                "TaskCompletionCriteria instance is required",
            )
        ok, rebuilt = _attempt(
            lambda: TaskCompletionCriteria.model_validate(criteria.model_dump(mode="python"))
        )
        if not ok or rebuilt is None:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("criteria",),
                "task criteria do not match declared schema",
            )
        conditions = list(rebuilt.conditions)
        if not conditions or not any(cond.required for cond in conditions):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("criteria", "conditions"),
                "task criteria require at least one required condition",
            )
        for index, cond in enumerate(conditions):
            if cond.outcome_type != "coding_task_verified":
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("criteria", "conditions", index, "outcome_type"),
                    "task criteria support only coding_task_verified",
                )
            try:
                parsed = CodingTaskVerifiedPayload.model_validate(thaw_json_value(cond.payload))
            except ValidationError:
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("criteria", "conditions", index, "payload"),
                    "invalid coding_task_verified task criteria payload",
                ) from None
            if self._task is not None:
                if (
                    parsed.verification_command_ids is not None
                    and parsed.verification_command_ids != self._task.verification_command_ids
                ):
                    raise VerificationError(
                        "VERIFICATION_INPUT_INVALID",
                        ("criteria", "conditions", index, "payload", "verification_command_ids"),
                        "criteria verification_command_ids do not match task",
                    )
                if (
                    parsed.commands_digest is not None
                    and parsed.commands_digest != self._task.commands_digest()
                ):
                    raise VerificationError(
                        "VERIFICATION_INPUT_INVALID",
                        ("criteria", "conditions", index, "payload", "commands_digest"),
                        "criteria commands_digest does not match task",
                    )


class CodingContractValidationHooks:
    """Synchronous pure ``ContractValidationHooks`` implementation for Coding."""

    def __init__(self, *, state_complete: bool) -> None:
        if type(state_complete) is not bool:
            raise InvalidActionPayloadError("state_complete must be an explicit strict bool")
        self._state_complete = state_complete

    @property
    def state_complete(self) -> bool:
        return self._state_complete

    def validate_precondition(self, condition: Precondition) -> None:
        if not isinstance(condition, Precondition):
            raise InvalidActionPayloadError("Precondition instance is required")
        ok, rebuilt = _attempt(
            lambda: Precondition.model_validate(condition.model_dump(mode="python"))
        )
        if not ok or rebuilt is None:
            raise InvalidActionPayloadError("invalid Precondition schema")
        if rebuilt.condition_type not in SUPPORTED_CONDITION_TYPES:
            raise InvalidActionPayloadError("unsupported coding precondition condition_type")
        model = (
            CodingScopeValidPayload
            if rebuilt.condition_type == "coding_scope_valid"
            else CodingWritePreconditionPayload
        )
        ok_p, _ = _attempt(lambda: model.model_validate(thaw_json_value(rebuilt.payload)))
        if not ok_p:
            raise InvalidActionPayloadError("invalid coding precondition payload")

    def validate_expected_outcome(self, outcome: ExpectedOutcome) -> None:
        if not isinstance(outcome, ExpectedOutcome):
            raise InvalidActionPayloadError("ExpectedOutcome instance is required")
        ok, rebuilt = _attempt(
            lambda: ExpectedOutcome.model_validate(outcome.model_dump(mode="python"))
        )
        if not ok or rebuilt is None:
            raise InvalidActionPayloadError("invalid ExpectedOutcome schema")
        if rebuilt.outcome_type not in SUPPORTED_OUTCOME_TYPES:
            raise InvalidActionPayloadError("unsupported coding outcome_type")
        if rebuilt.outcome_type == "coding_tool_result":
            model = CodingToolResultPayload
        elif rebuilt.outcome_type == "coding_file_written":
            model = CodingFileWrittenPayload
        else:
            model = CodingTaskVerifiedPayload
        ok_p, _ = _attempt(lambda: model.model_validate(thaw_json_value(rebuilt.payload)))
        if not ok_p:
            raise InvalidActionPayloadError("invalid coding expected outcome payload")

    def validate_verification_binding(
        self, binding: VerificationBinding, outcome: ExpectedOutcome
    ) -> None:
        if not isinstance(binding, VerificationBinding):
            raise InvalidActionPayloadError("VerificationBinding instance is required")
        ok, rebuilt = _attempt(
            lambda: VerificationBinding.model_validate(binding.model_dump(mode="python"))
        )
        if not ok or rebuilt is None:
            raise InvalidActionPayloadError("invalid VerificationBinding schema")
        if rebuilt.verifier_type != CODING_VERIFIER_TYPE or len(rebuilt.config) != 0:
            raise InvalidActionPayloadError(
                "coding verification binding must declare verifier_type='coding_rule' and empty config"
            )
        self.validate_expected_outcome(outcome)

    def check_precondition(
        self, condition: Precondition, observation: Observation
    ) -> VerificationStatus:
        self.validate_precondition(condition)
        if not isinstance(observation, Observation):
            raise InvalidActionPayloadError("Observation instance is required")
        if observation.environment_type != CODING_ENVIRONMENT_TYPE:
            return VerificationStatus.FAIL
        if not self._state_complete:
            return VerificationStatus.INCONCLUSIVE
        state = observation.state
        if not isinstance(state, dict):
            return VerificationStatus.FAIL
        if state.get("observation_overflow") is True:
            return VerificationStatus.INCONCLUSIVE

        if condition.condition_type == "coding_scope_valid":
            return (
                VerificationStatus.PASS
                if state.get("scope_ok") is True
                else VerificationStatus.FAIL
            )

        parsed = CodingWritePreconditionPayload.model_validate(
            thaw_json_value(condition.payload)
        )
        files = state.get("files")
        if not isinstance(files, list):
            return VerificationStatus.FAIL
        match = next(
            (item for item in files if isinstance(item, dict) and item.get("path") == parsed.path),
            None,
        )
        if match is None or match.get("writable") is not True:
            return VerificationStatus.FAIL
        if match.get("operation") != parsed.operation:
            return VerificationStatus.FAIL
        if parsed.operation == "create":
            if match.get("exists") is not False or parsed.expected_sha256 is not None:
                return VerificationStatus.FAIL
            return VerificationStatus.PASS
        if match.get("exists") is not True or match.get("sha256") != parsed.expected_sha256:
            return VerificationStatus.FAIL
        return VerificationStatus.PASS


class CodingContractHooksFactory:
    """Create ``CodingContractValidationHooks`` carrying explicit ``state_complete``."""

    def create(self, *, evidence: VerificationEvidence) -> CodingContractValidationHooks:
        if not isinstance(evidence, VerificationEvidence):
            raise InvalidActionPayloadError("VerificationEvidence instance is required")
        ok, rebuilt = _attempt(
            lambda: VerificationEvidence.model_validate(evidence.model_dump(mode="python"))
        )
        if not ok or rebuilt is None or type(rebuilt.state_complete) is not bool:
            raise InvalidActionPayloadError("evidence must carry explicit boolean state_complete")
        return CodingContractValidationHooks(state_complete=rebuilt.state_complete)
