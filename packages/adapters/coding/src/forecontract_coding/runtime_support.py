"""Pure synchronous domain support implementing Core's ``RuntimeDomainSupport``."""

from __future__ import annotations

import math
from typing import Any

from forecontract.contracts.models import (
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import (
    ActionSpec,
    ExpectedOutcome,
    Forecast,
    Precondition,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome, ExecutionStatus
from forecontract.recovery.models import (
    EFFECT_APPLIED,
    EFFECT_NOT_APPLIED,
    EFFECT_UNKNOWN,
    SAFETY_NON_IDEMPOTENT,
    SAFETY_READ_ONLY,
    ActionSafetyDeclaration,
)
from forecontract.runtime.models import ContractPreparation
from forecontract.verification.models import VerificationEvidence
from pydantic import ValidationError

from forecontract_coding.errors import (
    CODING_INPUT_INVALID,
    CODING_VERIFICATION_INVALID,
    CodingError,
)
from forecontract_coding.models import (
    CODING_ADAPTER_NAME,
    CODING_ENVIRONMENT_TYPE,
    CODING_VERIFIER_TYPE,
    SCHEMA_VERSION,
    SUPPORTED_ACTIONS,
    SUPPORTED_OUTCOME_TYPES,
    CodingFileWrittenPayload,
    CodingTask,
    CodingTaskVerifiedPayload,
    CodingToolResultPayload,
    RunCommandPayload,
    WriteFilePayload,
    thaw_json_value,
)

__all__ = ["CodingDomainSupport", "validate_coding_expected_outcome"]


def validate_coding_expected_outcome(
    *,
    outcome: ExpectedOutcome,
    action: ActionSpec,
    location: tuple[str | int, ...],
) -> None:
    """Validate an expected outcome and ensure it applies to the paired action."""
    if not isinstance(outcome, ExpectedOutcome):
        raise CodingError(
            CODING_VERIFICATION_INVALID,
            "ExpectedOutcome instance is required",
            location=location,
        )
    otype = outcome.outcome_type
    if otype not in SUPPORTED_OUTCOME_TYPES:
        raise CodingError(
            CODING_VERIFICATION_INVALID,
            f"unsupported coding outcome_type: {otype}",
            location=location + ("outcome_type",),
        )
    if otype == "coding_tool_result":
        try:
            parsed = CodingToolResultPayload.model_validate(thaw_json_value(outcome.payload))
        except ValidationError as exc:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "invalid coding_tool_result payload",
                location=location + ("payload",),
            ) from exc
        if parsed.action_type != action.action_type:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "coding_tool_result.action_type does not match proposed action",
                location=location + ("payload", "action_type"),
            )
        if (
            parsed.path is not None
            and action.action_type in ("read_file", "write_file")
            and action.payload.get("path") != parsed.path
        ):
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "coding_tool_result.path does not match action payload path",
                location=location + ("payload", "path"),
            )
        if (
            parsed.command_id is not None
            and action.action_type == "run_command"
            and action.payload.get("command_id") != parsed.command_id
        ):
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "coding_tool_result.command_id does not match run_command payload",
                location=location + ("payload", "command_id"),
            )
    elif otype == "coding_file_written":
        if action.action_type != "write_file":
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "coding_file_written applies only to write_file actions",
                location=location + ("outcome_type",),
            )
        try:
            parsed = CodingFileWrittenPayload.model_validate(thaw_json_value(outcome.payload))
        except ValidationError as exc:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "invalid coding_file_written payload",
                location=location + ("payload",),
            ) from exc
        try:
            write_payload = WriteFilePayload.model_validate(thaw_json_value(action.payload))
        except ValidationError as exc:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "invalid write_file action payload",
                location=("forecast", "proposed_action", "payload"),
            ) from exc
        if parsed.path != write_payload.path:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "coding_file_written.path must match write_file.path",
                location=location + ("payload", "path"),
            )
    else:
        # coding_task_verified
        if action.action_type != "verify_task":
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "coding_task_verified applies only to verify_task step actions",
                location=location + ("outcome_type",),
            )
        try:
            CodingTaskVerifiedPayload.model_validate(thaw_json_value(outcome.payload))
        except ValidationError as exc:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "invalid coding_task_verified payload",
                location=location + ("payload",),
            ) from exc


def _observation_state_is_complete(observation: Observation) -> bool:
    if observation.environment_type != CODING_ENVIRONMENT_TYPE:
        return False
    state = observation.state
    if not isinstance(state, dict):
        return False
    if state.get("observation_overflow") is True:
        return False
    if state.get("scope_ok") is not True:
        return False
    rev = state.get("workspace_revision")
    if not isinstance(rev, str) or not rev.strip():
        return False
    if not isinstance(state.get("files"), list):
        return False
    return isinstance(state.get("acceptance"), dict)


class CodingDomainSupport:
    """Pure, I/O-free ``RuntimeDomainSupport`` for Coding actions and recovery."""

    def __init__(self, *, task: CodingTask | None = None) -> None:
        self._task = CodingTask.model_validate(task) if task is not None else None

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        if not isinstance(observation, Observation):
            raise CodingError(
                CODING_INPUT_INVALID,
                "Observation instance is required",
                location=("observation",),
            )
        try:
            validated = Observation.model_validate(observation.model_dump(mode="python"))
        except ValidationError as exc:
            raise CodingError(
                CODING_INPUT_INVALID,
                "invalid Observation record",
                location=("observation",),
            ) from exc
        complete = _observation_state_is_complete(validated)
        return VerificationEvidence(
            schema_version=SCHEMA_VERSION,
            observation=validated,
            state_complete=complete,
        )

    def _compute_step_timeout_ms(self, action: ActionSpec) -> float:
        run_cap = float(self._task.run_timeout_ms) if self._task is not None else 30000.0
        if action.action_type in ("read_file", "search_text", "write_file"):
            return min(5000.0, run_cap)
        if action.action_type == "run_command" and self._task is not None:
            parsed = RunCommandPayload.model_validate(thaw_json_value(action.payload))
            spec = self._task.commands.get(parsed.command_id)
            if spec is not None:
                # Allow the Job runner's internal timeout + cleanup to complete before Core cancels the await.
                desired = float(spec.timeout_ms) + 1500.0
                return min(desired, run_cap)
        if action.action_type == "verify_task" and self._task is not None:
            total_cmd = sum(
                float(self._task.commands[cid].timeout_ms)
                for cid in self._task.verification_command_ids
            )
            desired = total_cmd + 2000.0
            return min(desired, run_cap)
        val = min(10000.0, run_cap)
        if not math.isfinite(val) or val <= 0:
            return 1000.0
        return val

    def prepare_contract(
        self,
        *,
        forecast: Forecast,
        evidence: VerificationEvidence,
    ) -> ContractPreparation:
        if not isinstance(forecast, Forecast):
            raise CodingError(
                CODING_INPUT_INVALID,
                "Forecast instance is required",
                location=("forecast",),
            )
        if not isinstance(evidence, VerificationEvidence):
            raise CodingError(
                CODING_INPUT_INVALID,
                "VerificationEvidence instance is required",
                location=("evidence",),
            )
        action = forecast.proposed_action
        if action.action_type not in SUPPORTED_ACTIONS:
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                f"unsupported coding action_type: {action.action_type}",
                location=("forecast", "proposed_action", "action_type"),
            )
        outcomes = list(forecast.expected_outcomes)
        if not outcomes or not any(item.required for item in outcomes):
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "expected_outcomes must contain at least one required outcome",
                location=("forecast", "expected_outcomes"),
            )
        for index, outcome in enumerate(outcomes):
            validate_coding_expected_outcome(
                outcome=outcome,
                action=action,
                location=("forecast", "expected_outcomes", index),
            )

        bindings = [
            VerificationBinding(
                schema_version=SCHEMA_VERSION,
                outcome_index=index,
                verifier_type=CODING_VERIFIER_TYPE,
                config={},
            )
            for index in range(len(outcomes))
        ]
        preconditions = [
            Precondition(
                schema_version=SCHEMA_VERSION,
                condition_type="coding_scope_valid",
                payload={},
                required=True,
            )
        ]
        timeout_ms = self._compute_step_timeout_ms(action)
        return ContractPreparation(
            schema_version=SCHEMA_VERSION,
            preconditions=preconditions,
            verification_policy=VerificationPolicy(
                schema_version=SCHEMA_VERSION,
                bindings=bindings,
            ),
            recovery_policy=RecoveryPolicy(
                schema_version=SCHEMA_VERSION,
                mode="bounded",
                max_retries=0,
                allow_replan=True,
            ),
            timeout_ms=timeout_ms,
        )

    def declare_action_safety(self, *, contract: object) -> ActionSafetyDeclaration:
        contract_id = getattr(contract, "contract_id", None)
        adapter_name = getattr(contract, "adapter_name", None)
        action = getattr(contract, "action", None)
        action_type = getattr(action, "action_type", None)
        if (
            not isinstance(contract_id, str)
            or not contract_id.strip()
            or adapter_name != CODING_ADAPTER_NAME
            or not isinstance(action_type, str)
            or action_type not in SUPPORTED_ACTIONS
        ):
            raise CodingError(
                CODING_INPUT_INVALID,
                "committed contract does not describe a valid Coding action",
                location=("contract",),
            )
        safety = (
            SAFETY_READ_ONLY
            if action_type in ("read_file", "search_text")
            else SAFETY_NON_IDEMPOTENT
        )
        return ActionSafetyDeclaration(
            schema_version=SCHEMA_VERSION,
            contract_id=contract_id,
            adapter_name=adapter_name,
            action_type=action_type,
            retry_safety=safety,
        )

    def assess_effect(
        self,
        *,
        contract: object,
        actual_outcome: ActualOutcome,
        evidence: VerificationEvidence,
    ) -> str:
        if not isinstance(actual_outcome, ActualOutcome) or not isinstance(
            evidence, VerificationEvidence
        ):
            raise CodingError(
                CODING_INPUT_INVALID,
                "valid ActualOutcome and VerificationEvidence are required",
                location=("assess_effect",),
            )
        if not evidence.state_complete:
            return EFFECT_UNKNOWN
        if actual_outcome.status is ExecutionStatus.UNKNOWN:
            return EFFECT_UNKNOWN

        action = getattr(contract, "action", None)
        action_type = getattr(action, "action_type", None)
        meta = getattr(action, "metadata", {}) or {}
        action_id = meta.get("action_id") if isinstance(meta, dict) else None
        if not action_id and hasattr(meta, "get"):
            action_id = meta.get("action_id")

        if actual_outcome.status is ExecutionStatus.FAILED:
            if action_type in ("read_file", "search_text"):
                return EFFECT_NOT_APPLIED
            err_code = actual_outcome.error.code if actual_outcome.error is not None else None
            if action_type == "write_file" and err_code in (
                "CODING_FILE_CONFLICT",
                "CODING_FILE_NOT_FOUND",
                "CODING_SCOPE_VIOLATION",
            ):
                return EFFECT_NOT_APPLIED
            if action_type in ("run_command", "verify_task") and err_code == "CODING_PROCESS_START_FAILED":
                return EFFECT_NOT_APPLIED
            return EFFECT_UNKNOWN

        # ExecutionStatus.SUCCEEDED: correlate last_tool_result with contracted action_id & live revision
        state: Any = evidence.observation.state
        if not isinstance(state, dict):
            return EFFECT_UNKNOWN
        last = state.get("last_tool_result")
        if not isinstance(last, dict):
            return EFFECT_UNKNOWN
        if not isinstance(action_id, str) or last.get("action_id") != action_id:
            return EFFECT_UNKNOWN
        if last.get("after_revision") != state.get("workspace_revision"):
            return EFFECT_UNKNOWN
        return EFFECT_APPLIED
