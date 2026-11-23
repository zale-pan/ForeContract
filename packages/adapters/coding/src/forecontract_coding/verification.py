"""Pure synchronous ``CodingStepVerifier`` and ``CodingTaskCompletionOracle``."""

from __future__ import annotations

from typing import Any, Final

from forecontract.contracts.models import ActionContract
from forecontract.models.actions import ExpectedOutcome
from forecontract.models.outcomes import (
    OutcomeVerification,
    VerificationResult,
    VerificationStatus,
)
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from pydantic import ValidationError

from forecontract_coding.models import (
    CODING_ADAPTER_NAME,
    CODING_ENVIRONMENT_TYPE,
    SCHEMA_VERSION,
    SUPPORTED_OUTCOME_TYPES,
    CodingFileWrittenPayload,
    CodingTask,
    CodingTaskVerifiedPayload,
    CodingToolResultPayload,
    WriteFilePayload,
    is_valid_sha256_hex,
    thaw_json_value,
)

__all__ = [
    "CODING_STEP_VERIFIER",
    "CODING_TASK_ORACLE",
    "CodingStepVerifier",
    "CodingTaskCompletionOracle",
]

CODING_STEP_VERIFIER: Final[str] = "coding_step_verifier"
CODING_TASK_ORACLE: Final[str] = "coding_task_completion_oracle"

REQUIRED_PASSED: Final[str] = "REQUIRED_PASSED"
REQUIRED_FAILED: Final[str] = "REQUIRED_FAILED"
REQUIRED_INCONCLUSIVE: Final[str] = "REQUIRED_INCONCLUSIVE"


def _aggregate_status(
    statuses: list[VerificationStatus], required_flags: list[bool]
) -> VerificationStatus:
    req = [s for s, is_req in zip(statuses, required_flags, strict=True) if is_req]
    if any(s is VerificationStatus.FAIL for s in req):
        return VerificationStatus.FAIL
    if any(s is VerificationStatus.INCONCLUSIVE for s in req):
        return VerificationStatus.INCONCLUSIVE
    return VerificationStatus.PASS


def _reason_for(status: VerificationStatus) -> str:
    if status is VerificationStatus.PASS:
        return REQUIRED_PASSED
    if status is VerificationStatus.INCONCLUSIVE:
        return REQUIRED_INCONCLUSIVE
    return REQUIRED_FAILED


def _check_evidence_shape(evidence: object) -> tuple[VerificationEvidence, dict[str, Any]]:
    if not isinstance(evidence, VerificationEvidence):
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("evidence",),
            "VerificationEvidence instance is required",
        )
    try:
        rebuilt = VerificationEvidence.model_validate(evidence.model_dump(mode="python"))
    except ValidationError:
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("evidence",),
            "VerificationEvidence does not match declared schema",
        ) from None
    if rebuilt.observation.environment_type != CODING_ENVIRONMENT_TYPE:
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("evidence", "observation", "environment_type"),
            "observation environment_type must be coding",
        )
    if not isinstance(rebuilt.observation.state, dict):
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("evidence", "observation", "state"),
            "observation state must be a dictionary",
        )
    return rebuilt, rebuilt.observation.state


def _check_scope_clean(state: dict[str, Any]) -> bool:
    """Verify host state reports clean workspace scope and empty violation list."""
    if state.get("scope_ok") is not True:
        return False
    violations = state.get("scope_violations")
    return isinstance(violations, list) and len(violations) == 0


def _evaluate_acceptance_record(
    *,
    state: dict[str, Any],
    state_complete: bool,
    payload: CodingTaskVerifiedPayload,
    task: CodingTask | None,
    require_action_id: str | None = None,
) -> tuple[VerificationStatus, str]:
    """Evaluate only host-generated acceptance evidence for the current revision."""
    if not state_complete or state.get("observation_overflow") is True:
        return VerificationStatus.INCONCLUSIVE, "incomplete_evidence"
    if not _check_scope_clean(state):
        return VerificationStatus.FAIL, "scope_violation"
    if state.get("acceptance_sources_ok") is not True:
        return VerificationStatus.FAIL, "test_source_changed"

    src_hashes = state.get("acceptance_source_hashes")
    if (
        not isinstance(src_hashes, dict)
        or not src_hashes
        or any(
            not isinstance(k, str) or not k.strip() or not is_valid_sha256_hex(v)
            for k, v in src_hashes.items()
        )
    ):
        return VerificationStatus.FAIL, "missing_acceptance_source_hashes"

    live_rev = state.get("workspace_revision")
    if not isinstance(live_rev, str) or not live_rev.strip():
        return VerificationStatus.FAIL, "missing_workspace_revision"

    acceptance = state.get("acceptance")
    if not isinstance(acceptance, dict):
        return VerificationStatus.FAIL, "missing_acceptance_record"

    status = acceptance.get("status")
    if status == "unknown":
        stale = acceptance.get("stale_reason") or "acceptance_status_unknown"
        return VerificationStatus.FAIL, str(stale)
    if status != "passed":
        return VerificationStatus.FAIL, f"acceptance_not_passed:{status}"
    if acceptance.get("stale_reason") is not None:
        return VerificationStatus.FAIL, "acceptance_marked_stale"

    acc_rev = acceptance.get("revision")
    if not isinstance(acc_rev, str) or acc_rev != live_rev:
        return VerificationStatus.FAIL, "acceptance_revision_mismatch"

    acc_action_id = acceptance.get("action_id")
    if not isinstance(acc_action_id, str) or not acc_action_id.strip():
        return VerificationStatus.FAIL, "missing_acceptance_action_id"
    if require_action_id is not None and acc_action_id != require_action_id:
        return VerificationStatus.FAIL, "acceptance_action_id_mismatch"

    cmd_ids = acceptance.get("command_ids")
    if (
        not isinstance(cmd_ids, list)
        or not cmd_ids
        or any(not isinstance(cid, str) or not cid.strip() for cid in cmd_ids)
        or len(set(cmd_ids)) != len(cmd_ids)
    ):
        return VerificationStatus.FAIL, "invalid_acceptance_command_ids"

    if payload.verification_command_ids is not None and cmd_ids != payload.verification_command_ids:
        return VerificationStatus.FAIL, "criteria_command_ids_mismatch"
    if task is not None and cmd_ids != task.verification_command_ids:
        return VerificationStatus.FAIL, "task_command_ids_mismatch"

    digest = acceptance.get("commands_digest")
    if not isinstance(digest, str) or not digest.strip():
        return VerificationStatus.FAIL, "missing_commands_digest"
    if payload.commands_digest is not None and digest != payload.commands_digest:
        return VerificationStatus.FAIL, "criteria_commands_digest_mismatch"
    if task is not None and digest != task.commands_digest():
        return VerificationStatus.FAIL, "task_commands_digest_mismatch"

    results = acceptance.get("results")
    if not isinstance(results, list) or len(results) != len(cmd_ids):
        return VerificationStatus.FAIL, "incomplete_command_results"

    for expected_cid, item in zip(cmd_ids, results, strict=True):
        if not isinstance(item, dict):
            return VerificationStatus.FAIL, "invalid_command_result_entry"
        if item.get("command_id") != expected_cid:
            return VerificationStatus.FAIL, "command_result_order_or_id_mismatch"
        if item.get("action_id") != acc_action_id:
            return VerificationStatus.FAIL, "command_result_action_id_mismatch"
        if item.get("status") != "passed" or item.get("exit_code") != 0:
            return VerificationStatus.FAIL, f"command_failed:{expected_cid}"
        if item.get("report_valid") is not True:
            return VerificationStatus.FAIL, f"invalid_report:{expected_cid}"
        if item.get("before_revision") != live_rev or item.get("after_revision") != live_rev:
            return VerificationStatus.FAIL, f"command_revision_mismatch:{expected_cid}"
        if (
            not isinstance(item.get("tests_executed"), int)
            or isinstance(item.get("tests_executed"), bool)
            or item.get("tests_executed") <= 0
        ):
            return VerificationStatus.FAIL, f"no_tests_executed:{expected_cid}"
        if item.get("failures") != 0 or item.get("errors") != 0 or item.get("skipped") != 0:
            return VerificationStatus.FAIL, f"nonzero_failures_or_skips:{expected_cid}"
        cmd_hashes = item.get("test_source_hashes")
        if (
            not isinstance(cmd_hashes, dict)
            or not cmd_hashes
            or any(
                not isinstance(k, str)
                or not is_valid_sha256_hex(v)
                or src_hashes.get(k) != v
                for k, v in cmd_hashes.items()
            )
        ):
            return VerificationStatus.FAIL, f"test_source_hashes_mismatch:{expected_cid}"

        if task is not None:
            spec = task.commands[expected_cid]
            expected_pairs = {(t.classname, t.name) for t in spec.expected_tests}
            verified_raw = item.get("verified_tests")
            if not isinstance(verified_raw, list):
                return VerificationStatus.FAIL, f"missing_verified_tests:{expected_cid}"
            observed_pairs: set[tuple[str, str]] = set()
            for pair in verified_raw:
                if (
                    not isinstance(pair, dict)
                    or not isinstance(pair.get("classname"), str)
                    or not isinstance(pair.get("name"), str)
                ):
                    return VerificationStatus.FAIL, f"invalid_verified_test_item:{expected_cid}"
                observed_pairs.add((pair["classname"], pair["name"]))
            if observed_pairs != expected_pairs:
                return VerificationStatus.FAIL, f"verified_test_set_mismatch:{expected_cid}"

    return VerificationStatus.PASS, "task_acceptance_verified"


class CodingStepVerifier:
    """Stateless pure step verifier for Coding contracts."""

    def verify_step(
        self, *, contract: ActionContract, evidence: VerificationEvidence
    ) -> VerificationResult:
        if not isinstance(contract, ActionContract):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("contract",),
                "ActionContract instance is required",
            )
        rebuilt_evidence, state = _check_evidence_shape(evidence)
        if contract.adapter_name != CODING_ADAPTER_NAME:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("contract", "adapter_name"),
                "contract adapter_name must be coding_workspace",
            )
        outcomes = list(contract.expected_outcomes)
        if not outcomes or not any(item.required for item in outcomes):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("contract", "expected_outcomes"),
                "contract requires at least one required expected outcome",
            )
        meta = dict(contract.action.metadata)
        action_id = meta.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip():
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("contract", "action", "metadata", "action_id"),
                "contract action requires non-blank action_id metadata",
            )

        action_type = contract.action.action_type
        action_payload = dict(contract.action.payload)
        state_complete = rebuilt_evidence.state_complete
        obs_id = rebuilt_evidence.observation.observation_id

        results: list[OutcomeVerification] = []
        statuses: list[VerificationStatus] = []

        for index, outcome in enumerate(outcomes):
            if not isinstance(outcome, ExpectedOutcome):
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("contract", "expected_outcomes", index),
                    "ExpectedOutcome instance is required",
                )
            otype = outcome.outcome_type
            if otype not in SUPPORTED_OUTCOME_TYPES:
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("contract", "expected_outcomes", index, "outcome_type"),
                    f"unsupported coding outcome_type: {otype}",
                )
            status, detail = self._evaluate_step_outcome(
                index=index,
                outcome=outcome,
                action_type=action_type,
                action_payload=action_payload,
                action_id=action_id,
                state=state,
                state_complete=state_complete,
            )
            statuses.append(status)
            results.append(
                OutcomeVerification(
                    schema_version=SCHEMA_VERSION,
                    outcome_index=index,
                    status=status,
                    has_actual_value=False,
                    actual_value=None,
                    reason=detail,
                )
            )

        agg = _aggregate_status(statuses, [item.required for item in outcomes])
        return VerificationResult(
            schema_version=SCHEMA_VERSION,
            status=agg,
            verifier=CODING_STEP_VERIFIER,
            outcome_results=results,
            reason=_reason_for(agg),
            metadata={
                "scope": "step",
                "contract_id": contract.contract_id,
                "goal_id": contract.goal_id,
                "observation_id": obs_id,
                "state_complete": state_complete,
            },
        )

    def _evaluate_step_outcome(
        self,
        *,
        index: int,
        outcome: ExpectedOutcome,
        action_type: str,
        action_payload: dict[str, Any],
        action_id: str,
        state: dict[str, Any],
        state_complete: bool,
    ) -> tuple[VerificationStatus, str]:
        otype = outcome.outcome_type
        last = state.get("last_tool_result")
        live_rev = state.get("workspace_revision")

        if otype == "coding_tool_result":
            try:
                parsed = CodingToolResultPayload.model_validate(thaw_json_value(outcome.payload))
            except ValidationError:
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("contract", "expected_outcomes", index, "payload"),
                    "invalid coding_tool_result payload",
                ) from None
            if parsed.action_type != action_type:
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("contract", "expected_outcomes", index, "payload", "action_type"),
                    "coding_tool_result does not match contracted action_type",
                )
            if not state_complete or state.get("observation_overflow") is True:
                return VerificationStatus.INCONCLUSIVE, "incomplete_evidence"
            if not _check_scope_clean(state):
                return VerificationStatus.FAIL, "scope_violation"
            if not isinstance(last, dict):
                return VerificationStatus.FAIL, "missing_last_tool_result"
            if last.get("action_id") != action_id:
                return VerificationStatus.FAIL, "action_id_mismatch"
            if last.get("action_type") != action_type:
                return VerificationStatus.FAIL, "action_type_mismatch"
            if last.get("after_revision") != live_rev:
                return VerificationStatus.FAIL, "revision_mismatch"
            if parsed.path is not None and last.get("path") != parsed.path:
                return VerificationStatus.FAIL, "path_mismatch"
            if parsed.command_id is not None and last.get("command_id") != parsed.command_id:
                return VerificationStatus.FAIL, "command_id_mismatch"
            if parsed.require_success and last.get("tool_succeeded") is not True:
                return VerificationStatus.FAIL, "tool_execution_not_succeeded"
            return VerificationStatus.PASS, "tool_result_verified"

        if otype == "coding_file_written":
            if action_type != "write_file":
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("contract", "expected_outcomes", index, "outcome_type"),
                    "coding_file_written applies only to write_file",
                )
            try:
                parsed = CodingFileWrittenPayload.model_validate(thaw_json_value(outcome.payload))
                write_req = WriteFilePayload.model_validate(thaw_json_value(action_payload))
            except ValidationError:
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("contract", "expected_outcomes", index, "payload"),
                    "invalid coding_file_written payload",
                ) from None
            if parsed.path != write_req.path:
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("contract", "expected_outcomes", index, "payload", "path"),
                    "coding_file_written path must match contracted write_file path",
                )
            if not state_complete or state.get("observation_overflow") is True:
                return VerificationStatus.INCONCLUSIVE, "incomplete_evidence"
            if not _check_scope_clean(state):
                return VerificationStatus.FAIL, "scope_violation"
            if not isinstance(last, dict):
                return VerificationStatus.FAIL, "missing_last_tool_result"
            if last.get("action_id") != action_id or last.get("action_type") != "write_file":
                return VerificationStatus.FAIL, "action_id_or_type_mismatch"
            if last.get("path") != parsed.path:
                return VerificationStatus.FAIL, "written_path_mismatch"
            if last.get("tool_succeeded") is not True or last.get("read_back_match") is not True:
                return VerificationStatus.FAIL, "write_read_back_failed"
            written_sha = last.get("written_sha256")
            read_back_sha = last.get("read_back_sha256")
            if not isinstance(written_sha, str) or written_sha != read_back_sha:
                return VerificationStatus.FAIL, "read_back_sha_mismatch"
            if (
                parsed.expected_after_sha256 is not None
                and read_back_sha != parsed.expected_after_sha256
            ):
                return VerificationStatus.FAIL, "expected_after_sha_mismatch"
            files = state.get("files")
            if not isinstance(files, list):
                return VerificationStatus.FAIL, "missing_files_inventory"
            entry = next(
                (f for f in files if isinstance(f, dict) and f.get("path") == parsed.path),
                None,
            )
            if entry is None or entry.get("exists") is not True or entry.get("sha256") != read_back_sha:
                return VerificationStatus.FAIL, "live_file_sha_mismatch"
            return VerificationStatus.PASS, "file_write_read_back_verified"

        # coding_task_verified step outcome
        if action_type != "verify_task":
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("contract", "expected_outcomes", index, "outcome_type"),
                "coding_task_verified step outcome applies only to verify_task",
            )
        try:
            parsed = CodingTaskVerifiedPayload.model_validate(thaw_json_value(outcome.payload))
        except ValidationError:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("contract", "expected_outcomes", index, "payload"),
                "invalid coding_task_verified payload",
            ) from None
        if not state_complete or state.get("observation_overflow") is True:
            return VerificationStatus.INCONCLUSIVE, "incomplete_evidence"
        if not isinstance(last, dict) or last.get("action_id") != action_id:
            return VerificationStatus.FAIL, "verify_action_id_mismatch"
        return _evaluate_acceptance_record(
            state=state,
            state_complete=state_complete,
            payload=parsed,
            task=None,
            require_action_id=action_id,
        )


class CodingTaskCompletionOracle:
    """Stateless independent task-completion oracle reading only criteria & host evidence."""

    def __init__(self, *, task: CodingTask | None = None) -> None:
        self._task = CodingTask.model_validate(task) if task is not None else None

    def evaluate_completion(
        self, *, criteria: TaskCompletionCriteria, evidence: VerificationEvidence
    ) -> VerificationResult:
        if not isinstance(criteria, TaskCompletionCriteria):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("criteria",),
                "TaskCompletionCriteria instance is required",
            )
        try:
            rebuilt_criteria = TaskCompletionCriteria.model_validate(
                criteria.model_dump(mode="python")
            )
        except ValidationError:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("criteria",),
                "TaskCompletionCriteria does not match declared schema",
            ) from None
        conditions = list(rebuilt_criteria.conditions)
        if not conditions or not any(item.required for item in conditions):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("criteria", "conditions"),
                "task criteria require at least one required condition",
            )

        rebuilt_evidence, state = _check_evidence_shape(evidence)
        obs_id = rebuilt_evidence.observation.observation_id
        state_complete = rebuilt_evidence.state_complete

        results: list[OutcomeVerification] = []
        statuses: list[VerificationStatus] = []
        for index, cond in enumerate(conditions):
            if cond.outcome_type != "coding_task_verified":
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("criteria", "conditions", index, "outcome_type"),
                    "task oracle accepts only coding_task_verified criteria",
                )
            try:
                parsed = CodingTaskVerifiedPayload.model_validate(thaw_json_value(cond.payload))
            except ValidationError:
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("criteria", "conditions", index, "payload"),
                    "invalid coding_task_verified payload",
                ) from None
            status, detail = _evaluate_acceptance_record(
                state=state,
                state_complete=state_complete,
                payload=parsed,
                task=self._task,
                require_action_id=None,
            )
            statuses.append(status)
            results.append(
                OutcomeVerification(
                    schema_version=SCHEMA_VERSION,
                    outcome_index=index,
                    status=status,
                    has_actual_value=False,
                    actual_value=None,
                    reason=detail,
                )
            )

        agg = _aggregate_status(statuses, [item.required for item in conditions])
        return VerificationResult(
            schema_version=SCHEMA_VERSION,
            status=agg,
            verifier=CODING_TASK_ORACLE,
            outcome_results=results,
            reason=_reason_for(agg),
            metadata={
                "scope": "task",
                "goal_id": rebuilt_criteria.goal_id,
                "observation_id": obs_id,
                "state_complete": state_complete,
            },
        )
