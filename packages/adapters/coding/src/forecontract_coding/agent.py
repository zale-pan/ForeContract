"""Native Core ``RuntimeEngine`` + ``LLMPlanner`` assembly entry point for Coding."""

from __future__ import annotations

import hashlib
from typing import Any

from forecontract.models.actions import ExpectedOutcome, Forecast
from forecontract.models.observations import Observation
from forecontract.recovery import RecoveryEngine
from forecontract.runtime import (
    MonotonicBudgetSource,
    RunRequest,
    RuntimeEngine,
    RuntimeTimingPolicy,
)
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm import LLMPlanner, PlannerConfig, PlannerContext
from forecontract_llm.provider import ModelProvider

from forecontract_coding.adapter import CodingAdapter
from forecontract_coding.errors import CODING_INPUT_INVALID, CodingError
from forecontract_coding.models import (
    SCHEMA_VERSION,
    AcceptanceSummary,
    CodingConfig,
    CodingRunResult,
    CodingTask,
)
from forecontract_coding.runtime_hooks import (
    CodingContractHooksFactory,
    CodingTaskCriteriaValidator,
)
from forecontract_coding.runtime_support import CodingDomainSupport
from forecontract_coding.verification import (
    CODING_STEP_VERIFIER,
    CodingStepVerifier,
    CodingTaskCompletionOracle,
)

__all__ = ["run_coding_task"]

_ACTION_GUIDANCE: dict[str, dict[str, Any]] = {
    "read_file": {
        "description": (
            "Read a line-bounded UTF-8 slice (up to 16 KiB) from task.read_paths, "
            "with optional line_byte_offset + expected_sha256 for long-line continuation."
        ),
        "required": ["path", "start_line", "max_lines"],
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 200},
            "line_byte_offset": {"type": "integer", "minimum": 0},
            "expected_sha256": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
    "search_text": {
        "description": "Literal non-regex search across a subset of task.read_paths sorted by (path, line).",
        "required": ["query", "paths", "max_matches"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "paths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "max_matches": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "additionalProperties": False,
    },
    "write_file": {
        "description": "Create or atomically replace an authorized path with full UTF-8 content.",
        "required": ["path", "expected_sha256", "content"],
        "properties": {
            "path": {"type": "string"},
            "expected_sha256": {"type": ["string", "null"]},
            "content": {"type": "string"},
        },
        "additionalProperties": False,
    },
    "run_command": {
        "description": "Execute one pre-registered host command_id without custom args, env, or cwd.",
        "required": ["command_id"],
        "properties": {"command_id": {"type": "string"}},
        "additionalProperties": False,
    },
    "verify_task": {
        "description": "Run all host acceptance commands and verify JUnit XML results against current revision.",
        "required": [],
        "properties": {},
        "additionalProperties": False,
    },
}

_OUTCOME_GUIDANCE: dict[str, dict[str, Any]] = {
    "coding_tool_result": {
        "description": "Verifies tool execution correlation with contracted action_id and live revision.",
        "required": ["action_type"],
        "properties": {
            "action_type": {"type": "string"},
            "path": {"type": "string"},
            "command_id": {"type": "string"},
            "require_success": {"type": "boolean"},
        },
        "additionalProperties": False,
    },
    "coding_file_written": {
        "description": "Verifies write_file read-back SHA-256 matches written content and observation.",
        "required": ["path"],
        "properties": {
            "path": {"type": "string"},
            "expected_after_sha256": {"type": "string"},
        },
        "additionalProperties": False,
    },
    "coding_task_verified": {
        "description": "Verifies full acceptance command suite passed cleanly for current revision.",
        "required": [],
        "properties": {
            "verification_command_ids": {"type": "array", "items": {"type": "string"}},
            "commands_digest": {"type": "string"},
        },
        "additionalProperties": False,
    },
}


class _HostActionIdPlanner:
    """Binds a host-generated unique ``action_id`` onto each decoded LLM ``Forecast``."""

    def __init__(self, *, inner: LLMPlanner, run_id: str) -> None:
        self._inner = inner
        self._run_id = run_id
        self._run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
        self._seq = 0

    async def plan(self, observation: Observation) -> Forecast:
        forecast = await self._inner.plan(observation)
        self._seq += 1
        action_id = f"host-{self._run_digest}-act-{self._seq:04d}"
        bound_action = forecast.proposed_action.model_copy(
            update={"metadata": {"action_id": action_id}}
        )
        return forecast.model_copy(update={"proposed_action": bound_action})


async def run_coding_task(
    *,
    task: CodingTask,
    config: CodingConfig,
    provider: ModelProvider,
    planner_config: PlannerConfig,
    trace_writer: object,
    clock: object | None = None,
    monotonic_clock: object | None = None,
) -> CodingRunResult:
    """Assemble and run one bounded native Core + LLM Coding session."""
    if not isinstance(task, CodingTask):
        raise CodingError(
            CODING_INPUT_INVALID,
            "CodingTask instance is required",
            location=("task",),
        )
    validated_task = CodingTask.model_validate(task.model_dump(mode="python"))
    if not isinstance(config, CodingConfig):
        raise CodingError(
            CODING_INPUT_INVALID,
            "CodingConfig instance is required",
            location=("config",),
        )
    validated_config = CodingConfig.model_validate(config.model_dump(mode="python"))
    validated_planner_config = PlannerConfig.model_validate(planner_config)

    adapter = CodingAdapter(task=validated_task, config=validated_config)

    criteria = TaskCompletionCriteria(
        schema_version=SCHEMA_VERSION,
        goal_id=validated_task.goal_id,
        conditions=[
            ExpectedOutcome(
                schema_version=SCHEMA_VERSION,
                outcome_type="coding_task_verified",
                payload={
                    "verification_command_ids": list(validated_task.verification_command_ids),
                    "commands_digest": validated_task.commands_digest(),
                },
                required=True,
            )
        ],
    )
    run_request = RunRequest(
        schema_version=SCHEMA_VERSION,
        run_id=validated_task.run_id,
        goal_id=validated_task.goal_id,
        task_criteria=criteria,
        max_steps=validated_task.max_steps,
        max_planning_calls=validated_task.max_planning_calls,
        max_retries=0,
        max_replans=validated_task.max_replans,
    )

    planner_context = PlannerContext(
        schema_version=SCHEMA_VERSION,
        run_id=validated_task.run_id,
        goal_id=validated_task.goal_id,
        goal_text=validated_task.goal_text,
        task_criteria=criteria,
        capabilities=adapter.capabilities(),
        action_guidance=_ACTION_GUIDANCE,
        outcome_guidance=_OUTCOME_GUIDANCE,
    )
    llm_planner = LLMPlanner(
        provider=provider,
        context=planner_context,
        config=validated_planner_config,
    )
    bound_planner = _HostActionIdPlanner(inner=llm_planner, run_id=validated_task.run_id)

    domain_support = CodingDomainSupport(task=validated_task)
    hooks_factory = CodingContractHooksFactory()
    criteria_validator = CodingTaskCriteriaValidator(task=validated_task)
    step_verifier = CodingStepVerifier()
    task_oracle = CodingTaskCompletionOracle(task=validated_task)
    recovery_engine = RecoveryEngine(step_verifier_name=CODING_STEP_VERIFIER)
    budget_source = MonotonicBudgetSource(
        total_budget_ms=float(validated_task.run_timeout_ms),
        monotonic_clock=monotonic_clock,
    )
    timing_policy = RuntimeTimingPolicy(
        schema_version=SCHEMA_VERSION,
        run_timeout_ms=float(validated_task.run_timeout_ms),
        cleanup_timeout_ms=float(max(100, validated_config.process_cleanup_ms)),
        finalize_timeout_ms=1000.0,
    )

    engine = RuntimeEngine(
        adapter=adapter,
        planner=bound_planner,
        domain_support=domain_support,
        budget_source=budget_source,
        trace_writer=trace_writer,
        step_verifier=step_verifier,
        task_oracle=task_oracle,
        recovery_engine=recovery_engine,
        clock=clock,
        timing_policy=timing_policy,
        monotonic_clock=monotonic_clock,
        contract_hooks_factory=hooks_factory,
        task_criteria_validator=criteria_validator,
    )

    try:
        run_result = await engine.run(request=run_request)
    finally:
        try:
            await llm_planner.aclose()
        finally:
            await adapter.stop()

    final_revision = adapter.current_workspace_revision()
    baseline_revision = adapter.baseline_revision or final_revision
    changed_paths = adapter.current_changed_paths()
    acceptance: AcceptanceSummary = adapter.acceptance_summary

    return CodingRunResult(
        schema_version=SCHEMA_VERSION,
        run_result=run_result,
        baseline_workspace_revision=baseline_revision,
        final_workspace_revision=final_revision,
        changed_paths=changed_paths,
        acceptance=acceptance,
        terminal_status=run_result.terminal_status.value,
        reason_code=run_result.reason_code.value,
    )
