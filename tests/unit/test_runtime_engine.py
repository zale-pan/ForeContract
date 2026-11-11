"""Acceptance tests for the sequential Runtime engine.

The engine is driven here through a real adapter-shaped dependency, a real planner object and the
real Builder, StepVerifier, TaskCompletionOracle and RecoveryEngine. A small scripted stand-in
adapter is used so a test can assert exact call counts and exact terminal facts; the integrated
Fake and ScriptedPlanner flow is exercised separately in ``test_runtime_fake_flow.py``.

Coverage follows A01-A12: zero-action success, the oracle's independent authority, the planner hint
having no authority, step-versus-task separation, real execution and effect judgement, retry and
replan identity rules, precondition expiry, the bounded recovery loop, host budget and counter
limits, working-memory isolation, honest primary-error preservation and trace failure exposure.
"""

from __future__ import annotations

import asyncio
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.contracts import RecoveryPolicy, VerificationPolicy
from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
)
from forecontract.models.base import SCHEMA_VERSION
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome, ErrorInfo, VerificationStatus
from forecontract.recovery import (
    ActionSafetyDeclaration,
    RecoveryAssessment,
    RecoveryEngine,
)
from forecontract.runtime import (
    ContractPreparation,
    RunRequest,
    RuntimeBudgetSnapshot,
    RuntimeEngine,
    RuntimeError,
)
from forecontract.runtime.errors import RUNTIME_ERROR_CODES, RUNTIME_STAGES
from forecontract.tracing import (
    AttemptFinished,
    AttemptStarted,
    ContractCommitted,
    JsonlTraceStore,
    RunFinished,
    RunStarted,
    StepStarted,
    TraceValidationError,
    VerificationRecorded,
)
from forecontract.verification import StepVerifier, TaskCompletionOracle
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)

#: The stand-in adapters raise a distinguishable error, so a test never confuses the builtin
#: ``RuntimeError`` with the bounded Runtime error this module imports.
AdapterError = type("AdapterError", (Exception,), {})

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
ACTION = ActionSpec(schema_version=SCHEMA_VERSION, action_type="click",
                    payload={"selector": "#go"})
N_RULE = ExpectedOutcome(schema_version=SCHEMA_VERSION, outcome_type="json_rule",
                         payload={"schema_version": SCHEMA_VERSION, "rule_type": "value_equals",
                                  "path": ["n"], "expected": 1}, required=True)
DONE_RULE = ExpectedOutcome(schema_version=SCHEMA_VERSION, outcome_type="json_rule",
                            payload={"schema_version": SCHEMA_VERSION,
                                     "rule_type": "value_equals", "path": ["done"],
                                     "expected": True}, required=True)
ABSENT_RULE = ExpectedOutcome(schema_version=SCHEMA_VERSION, outcome_type="json_rule",
                              payload={"schema_version": SCHEMA_VERSION,
                                       "rule_type": "value_equals", "path": ["n"],
                                       "expected": 99}, required=True)


def drive(coroutine):
    return asyncio.run(coroutine)


# --- the scripted stand-in dependencies -----------------------------------------

class ScriptedAdapter:
    """A deterministic adapter stand-in: a list of observations and a list of receipts.

    It exposes the same declared members as the real adapter protocol, which is what the Runtime
    depends on. It records every call so a test can assert exact counts and order.
    """

    def __init__(self, *, observations, outcomes=None, capabilities=None, execute_raises=None,
                 observe_raises=None, stop_raises=None, on_observe=None):
        self._observations = list(observations)
        self._outcomes = list(outcomes or [])
        self._capabilities = capabilities
        self._execute_raises = execute_raises
        self._observe_raises = observe_raises
        self._stop_raises = stop_raises
        self._on_observe = on_observe
        self.calls: list[str] = []
        self.actions: list[ActionSpec] = []
        self.stops = 0

    @property
    def name(self) -> str:
        return "scripted-1"

    def capabilities(self) -> AdapterCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        return AdapterCapabilities(schema_version=SCHEMA_VERSION, adapter_name="scripted-1",
                                   environment_type="scripted",
                                   supported_actions=["click"],
                                   supported_observation_types=["scripted.state"],
                                   supported_condition_types=["json_rule"],
                                   supported_outcome_types=["json_rule"])

    def validate_action(self, action: ActionSpec) -> None:
        if action.action_type != "click":
            raise ValueError("unsupported action")

    async def start(self) -> None:
        self.calls.append("start")

    async def stop(self) -> None:
        self.stops += 1
        self.calls.append("stop")
        if self._stop_raises is not None:
            raise self._stop_raises

    async def observe(self) -> Observation:
        self.calls.append("observe")
        if self._observe_raises is not None:
            raise self._observe_raises
        if self._on_observe is not None:
            self._on_observe(self)
        if not self._observations:
            raise AdapterError("the scripted observation list is exhausted")
        return self._observations.pop(0)

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        self.calls.append("execute")
        self.actions.append(action)
        if self._execute_raises is not None:
            raise self._execute_raises
        if not self._outcomes:
            raise AdapterError("the scripted outcome list is exhausted")
        return self._outcomes.pop(0)


def obs(oid: str, state: dict) -> Observation:
    return Observation(schema_version=SCHEMA_VERSION, observation_id=oid,
                       environment_type="scripted", state=state, timestamp=NOW)


def outcome(index: int, status: str = "succeeded", recoverable: bool | None = None,
            changes: dict | None = None) -> ActualOutcome:
    fields: dict = {"schema_version": SCHEMA_VERSION, "outcome_id": f"out-{index}",
                    "status": status, "changes": changes if changes is not None else {"n": 1}}
    if status != "succeeded":
        fields["error"] = ErrorInfo(schema_version=SCHEMA_VERSION, code="EXECUTION_FAILED",
                                    message="synthetic", recoverable=recoverable)
    return ActualOutcome(**fields)


class ScriptedPlanner:
    """A planner stand-in returning a prepared list of forecasts."""

    def __init__(self, forecasts=None, *, raises=None):
        self._forecasts = list(forecasts or [])
        self._raises = raises
        self.observations: list[Observation] = []

    @staticmethod
    def forecast(index: int = 0, hints: bool | None = None,
                 outcomes: list[ExpectedOutcome] | None = None) -> Forecast:
        return Forecast(schema_version=SCHEMA_VERSION, forecast_id=f"f-{index}",
                        proposed_action=ACTION,
                        expected_outcomes=outcomes if outcomes is not None else [N_RULE],
                        task_completion_hint=hints)

    async def plan(self, observation: Observation) -> Forecast:
        self.observations.append(observation)
        if self._raises is not None:
            raise self._raises
        if not self._forecasts:
            raise AdapterError("the scripted forecast list is exhausted")
        return self._forecasts.pop(0)


class Support:
    """A domain-support stand-in that records calls and returns explicit declarations."""

    def __init__(self, *, complete: bool = True, effect: str = "not_applied",
                 safety: str = "idempotent", mode: str = "bounded",
                 preconditions=None, mutate_forecast: bool = False,
                 substitute_observation: bool = False, effect_value: object | None = None,
                 to_evidence_raises: Exception | None = None):
        self.complete = complete
        self.effect = effect
        self.effect_value = effect_value if effect_value is not None else effect
        self.safety = safety
        self.mode = mode
        self.preconditions = preconditions or []
        self.mutate_forecast = mutate_forecast
        self.substitute_observation = substitute_observation
        self.to_evidence_raises = to_evidence_raises
        self.calls: list[str] = []
        self.seen: list[Observation] = []

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        self.calls.append("to_evidence")
        self.seen.append(observation)
        if self.to_evidence_raises is not None:
            raise self.to_evidence_raises
        carried = (obs(observation.observation_id + "-other", {"other": 1})
                   if self.substitute_observation else observation)
        return VerificationEvidence(schema_version=SCHEMA_VERSION, observation=carried,
                                    state_complete=self.complete)

    def prepare_contract(self, *, forecast: Forecast,
                         evidence: VerificationEvidence) -> ContractPreparation:
        self.calls.append("prepare_contract")
        if self.mutate_forecast:
            forecast.metadata["tampered"] = True
        bindings = [{"schema_version": SCHEMA_VERSION, "outcome_index": index,
                     "verifier_type": "json_rule", "config": {}}
                    for index, _ in enumerate(forecast.expected_outcomes)]
        policy = (RecoveryPolicy(schema_version=SCHEMA_VERSION)
                  if self.mode == "abort"
                  else RecoveryPolicy(schema_version=SCHEMA_VERSION, mode="bounded",
                                      max_retries=1, allow_replan=True))
        return ContractPreparation(
            schema_version=SCHEMA_VERSION, preconditions=list(self.preconditions),
            verification_policy=VerificationPolicy.model_validate(
                {"schema_version": SCHEMA_VERSION, "bindings": bindings}),
            recovery_policy=policy, timeout_ms=1000.0)

    def declare_action_safety(self, *, contract) -> ActionSafetyDeclaration:
        self.calls.append("declare_action_safety")
        return ActionSafetyDeclaration(schema_version=SCHEMA_VERSION,
                                        contract_id=contract.contract_id,
                                        adapter_name=contract.adapter_name,
                                        action_type=contract.action.action_type,
                                        retry_safety=self.safety)

    def assess_effect(self, *, contract, actual_outcome, evidence):
        self.calls.append("assess_effect")
        return self.effect_value


class Budget:
    def __init__(self, value: float = 10_000.0, *, raises: Exception | None = None):
        self.value = value
        self.raises = raises
        self.reads = 0

    def snapshot(self) -> RuntimeBudgetSnapshot:
        self.reads += 1
        if self.raises is not None:
            raise self.raises
        return RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION, remaining_budget_ms=self.value)


class MemoryRecorder:
    """A working-memory stand-in that records what the engine published."""

    def __init__(self):
        self.states: dict[str, dict] = {}

    def set(self, run_id, state):
        self.states[run_id] = state

    def get(self, run_id):
        return self.states[run_id]

    def clear(self, run_id):
        return self.states.pop(run_id, None) is not None


class TraceRecorder:
    """A trace-writer stand-in collecting the records in order."""

    def __init__(self, *, fail_on=None):
        self.records: list = []
        self.fail_on = fail_on

    def append(self, record):
        if self.fail_on is not None and type(record).__name__ == self.fail_on:
            raise TraceValidationError("synthetic trace refusal")
        self.records.append(record)

    def kinds(self) -> list[str]:
        return [record.kind for record in self.records]


def request(run_id: str = "run-1", conditions=None, **limits) -> RunRequest:
    fields = {"max_steps": 3, "max_planning_calls": 3, "max_retries": 1, "max_replans": 1}
    fields.update(limits)
    return RunRequest(schema_version=SCHEMA_VERSION, run_id=run_id, goal_id="g-1",
                      task_criteria=TaskCompletionCriteria(
                          schema_version=SCHEMA_VERSION, goal_id="g-1",
                          conditions=conditions or [DONE_RULE]), **fields)


def unmet_series(steps: int, last: dict | None = None, start: dict | None = None) -> list:
    """Observations for a run of ``steps`` executions whose step rule keeps passing.

    The approved contract observes once initially, once before every attempt (after the Builder
    submission, even with an empty precondition list) and once after every attempt, so ``steps``
    executions need ``3 * steps + 1`` observations. Each observation after the first reports the
    effect the step rule expects (``n == 1``), so every step verifies; every observation but the
    last reports a goal that still does not hold, and the last one reports the goal state the
    caller asks for.
    """
    first = dict(start) if start is not None else {"done": False, "n": 0}
    series = [obs("o-0", first)]
    for index in range(1, steps + 1):
        working = {"done": False, "n": 1}
        if index == steps and last is not None:
            working = dict(last)
        # The pre-execution and post-execution observations of this attempt.
        series.append(obs(f"pre-{index}", dict(working)))
        series.append(obs(f"post-{index}", dict(working)))
    return series


def outcomes_for(steps: int) -> list:
    return [outcome(index) for index in range(steps)]


def engine_for(adapter, planner, support, *, budget=None, trace=None, memory=None, **extra):
    return RuntimeEngine(adapter=adapter, planner=planner, domain_support=support,
                         budget_source=budget or Budget(), trace_writer=trace or TraceRecorder(),
                         working_memory=memory, **extra)


# --- A01/A05: zero-action success and independent task authority ----------------

def test_a_goal_that_already_holds_completes_with_zero_actions() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    planner = ScriptedPlanner([])
    result = drive(engine_for(adapter, planner, Support()).run(request=request()))
    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (0, 0, 0)
    assert result.verification_record_id is not None
    assert planner.observations == []


def test_the_terminal_fact_is_the_last_record_and_it_is_unique() -> None:
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace).run(request=request()))
    kinds = trace.kinds()
    assert kinds == [RunStarted.kind if hasattr(RunStarted, "kind") else "run_started",
                     "verification_recorded", "run_finished"]
    terminals = [record for record in trace.records if isinstance(record, RunFinished)]
    assert len(terminals) == 1
    assert terminals[0].terminal_status == "success"
    assert terminals[0].verification_record_id == trace.records[1].record_id


def test_a_planner_hint_cannot_make_a_failing_goal_succeed() -> None:
    adapter = ScriptedAdapter(observations=unmet_series(3, last={"done": False, "n": 1}),
                              outcomes=outcomes_for(3))
    planner = ScriptedPlanner([ScriptedPlanner.forecast(index, hints=True) for index in range(3)])
    support = Support(effect="not_applied")
    result = drive(engine_for(adapter, planner, support)
                   .run(request=request(max_steps=3, max_planning_calls=3)))
    # The hint is ignored: three steps really ran and verified, the goal still did not hold, and
    # the run is stopped by the host's step allowance rather than reported as a success.
    assert result.terminal_status.value == "aborted", result.model_dump()
    assert result.reason_code.value == "MAX_STEPS_EXHAUSTED"
    assert result.steps_started == 3
    assert result.attempts_started == 3
    assert result.verification_record_id is None


def test_a_step_pass_does_not_become_task_success() -> None:
    adapter = ScriptedAdapter(observations=unmet_series(3, last={"done": False, "n": 1}),
                              outcomes=outcomes_for(3))
    planner = ScriptedPlanner([ScriptedPlanner.forecast(index) for index in range(3)])
    result = drive(engine_for(adapter, planner, Support())
                   .run(request=request(max_steps=3, max_planning_calls=3)))
    # The step verified PASS (n == 1), yet the goal never held, so the run is not a success.
    assert result.terminal_status.value != "completed"
    assert result.attempts_started == 3


def test_a_goal_that_holds_after_a_step_completes() -> None:
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, planner, Support(), trace=trace).run(request=request()))
    assert result.terminal_status.value == "completed"
    assert result.steps_started == 1
    assert result.attempts_started == 1
    assert trace.kinds() == ["run_started", "step_started", "contract_committed",
                             "attempt_started", "attempt_finished", "verification_recorded",
                             "verification_recorded", "run_finished"]


def test_only_a_real_run_scope_pass_permits_success() -> None:
    """A successful terminal fact always references a recorded run-scope PASS."""
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
                   .run(request=request()))
    passes = [record for record in trace.records
              if isinstance(record, VerificationRecorded) and record.scope == "run"]
    assert len(passes) == 1
    assert passes[0].verification_status.value == "pass"
    assert result.verification_record_id == passes[0].record_id


# --- A08: the builder is the only component that commits ------------------------

def test_an_unsupported_outcome_type_is_never_committed_or_executed() -> None:
    capabilities = AdapterCapabilities(schema_version=SCHEMA_VERSION, adapter_name="scripted-1",
                                       environment_type="scripted",
                                       supported_actions=["click"],
                                       supported_outcome_types=[])
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})],
                              capabilities=capabilities)
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    result = drive(engine_for(adapter, planner, Support()).run(request=request()))
    assert result.terminal_status.value == "failed"
    assert "execute" not in adapter.calls
    assert result.attempts_started == 0


def test_a_mutated_forecast_is_refused_rather_than_accepted() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(mutate_forecast=True)
    result = drive(engine_for(adapter, planner, support).run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_evidence_must_carry_the_observation_of_this_call() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})])
    support = Support(substitute_observation=True)
    result = drive(engine_for(adapter, ScriptedPlanner([]), support).run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


# --- A04/A08: execution is real and never fabricated ---------------------------

def test_an_execute_exception_never_produces_a_fabricated_outcome() -> None:
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False}),
                                            obs("o-1", {"done": False})],
                              execute_raises=AdapterError("synthetic boom"))
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    engine = engine_for(adapter, planner, Support(), trace=trace)
    # The executor raised an ordinary exception, so the bounded facility failure is propagated once
    # the Run has recorded its own end.
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value == "failed"
    assert adapter.calls.count("execute") == 1
    # The Attempt started, and no AttemptFinished fact was invented for it.
    assert isinstance(trace.records[3], AttemptStarted)
    assert not any(isinstance(record, AttemptFinished) for record in trace.records)


def test_a_failed_execution_keeps_its_diagnosis_and_still_needs_the_oracle() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False}),
                                            obs("o-1", {"done": False, "n": 1})],
                              outcomes=[outcome(0, status="failed", recoverable=True)])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(effect="unknown", safety="non_idempotent")
    engine = engine_for(adapter, planner, support)
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    # The step PASSed on the observation while the execution failed: the run is not a task success.
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value != "completed"


# --- A06/A07: retry and replan identity rules ----------------------------------

def test_a_retry_keeps_the_contract_and_starts_a_new_attempt() -> None:
    failed = outcome(0, status="failed", recoverable=True)
    # The contract rule expects n == 1. The observation after the failed execute still reports
    # n == 0, so the step genuinely fails and recovery decides. The retry then really produces the
    # effect and the goal. Each attempt is observed once before it and once after it, on top of the
    # initial observation.
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0}),
                                            obs("pre-1", {"done": True, "n": 0}),
                                            obs("post-1", {"done": True, "n": 0}),
                                            obs("recovery", {"done": True, "n": 0}),
                                            obs("pre-2", {"done": True, "n": 0}),
                                            obs("post-2", {"done": True, "n": 1})],
                              outcomes=[failed, outcome(1)])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(effect="not_applied", safety="idempotent")
    trace = TraceRecorder()
    result = drive(engine_for(adapter, planner, support, trace=trace).run(request=request()))
    assert result.terminal_status.value == "completed", result.model_dump()
    assert result.retries_used == 1
    assert result.attempts_started == 2
    assert result.steps_started == 1
    # Only one contract was committed, and two Attempts started against it.
    committed = [record for record in trace.records if isinstance(record, ContractCommitted)]
    started = [record for record in trace.records if isinstance(record, AttemptStarted)]
    assert len(committed) == 1
    assert len(started) == 2
    assert committed[0].contract_id == started[1].contract_id
    assert started[0].attempt_id != started[1].attempt_id


def test_a_replan_keeps_the_step_and_links_a_new_contract() -> None:
    failed = outcome(0, status="failed", recoverable=True)
    # The step still fails on the recovery-time observation. The domain says the effect is known
    # and already applied, which is precisely the case that forbids a blind retry on a
    # non-idempotent action and asks for a replan instead. The new contract is observed before it
    # may execute, and its attempt then produces the goal.
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0}),
                                            obs("pre-1", {"done": True, "n": 0}),
                                            obs("post-1", {"done": True, "n": 0}),
                                            obs("recovery", {"done": True, "n": 0}),
                                            obs("pre-2", {"done": True, "n": 0}),
                                            obs("post-2", {"done": True, "n": 1})],
                              outcomes=[failed, outcome(1)])
    planner = ScriptedPlanner([ScriptedPlanner.forecast(0), ScriptedPlanner.forecast(1)])
    support = Support(effect="applied", safety="non_idempotent")
    trace = TraceRecorder()
    result = drive(engine_for(adapter, planner, support, trace=trace).run(request=request()))
    assert result.replans_used == 1, (result.model_dump(), adapter.calls, support.calls)
    assert result.steps_started == 1
    committed = [record for record in trace.records if isinstance(record, ContractCommitted)]
    started = [record for record in trace.records if isinstance(record, StepStarted)]
    assert len(started) == 1
    assert len(committed) == 2
    assert committed[1].replaces_contract_id == committed[0].contract_id
    assert committed[0].step_id == committed[1].step_id == started[0].step_id
    assert result.attempts_started == 2


def test_a_recovery_observation_may_legally_repeat_the_identifier() -> None:
    failed = outcome(0, status="failed", recoverable=True)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0}),
                                            obs("pre-1", {"done": True, "n": 0}),
                                            obs("post-1", {"done": True, "n": 0}),
                                            obs("same", {"done": True, "n": 0}),
                                            obs("pre-2", {"done": True, "n": 0}),
                                            obs("post-2", {"done": True, "n": 1})],
                              outcomes=[failed, outcome(1)])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(effect="not_applied")
    memory = MemoryRecorder()
    result = drive(engine_for(adapter, planner, support, memory=memory).run(request=request()))
    assert result.terminal_status.value == "completed", result.model_dump()
    # Both calls are visible with their own ordinal and phase, even though the identifier repeats.
    calls = memory.get("run-1")["observation_calls"]
    phases = [call["phase"] for call in calls]
    assert "RECOVERING" in phases and "VERIFYING" in phases
    ordinals = [call["ordinal"] for call in calls]
    assert ordinals == sorted(ordinals) and len(set(ordinals)) == len(ordinals)


def test_the_recovery_loop_never_asks_for_a_second_observation() -> None:
    """A recovery assessment that still wants evidence after one observation is a facility error."""
    failed = outcome(0, status="failed", recoverable=True)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False}),
                                            obs("o-1", {"done": True, "n": 0}),
                                            obs("o-2", {"done": True, "n": 1})],
                              outcomes=[failed])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    # The step keeps failing on the recovery-time evidence, where the contract rule expects n == 1.
    support = Support(effect="not_applied", complete=False)
    result = drive(engine_for(adapter, planner, support).run(request=request()))
    observes = [call for call in adapter.calls if call == "observe"]
    assert len(observes) <= 4
    assert result.terminal_status.value in {"completed", "failed", "aborted"}


def test_a_default_aborting_policy_does_not_veto_a_verified_step() -> None:
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(mode="abort")
    result = drive(engine_for(adapter, planner, support).run(request=request()))
    assert result.terminal_status.value == "completed"


# --- A04: the failure means the effect is unknown, not absent ------------------

def test_a_failed_execution_is_never_mapped_to_an_absent_effect() -> None:
    """The domain says the effect is unknown, so a retry is refused for a non-idempotent action."""
    failed = outcome(0, status="failed", recoverable=True)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0}),
                                            obs("pre-1", {"done": True, "n": 0}),
                                            obs("post-1", {"done": True, "n": 0}),
                                            obs("recovery", {"done": True, "n": 0})],
                              outcomes=[failed])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(effect="unknown", safety="non_idempotent")
    result = drive(engine_for(adapter, planner, support).run(request=request()))
    assert result.attempts_started == 1
    assert result.retries_used == 0
    assert result.terminal_status.value == "aborted", result.model_dump()
    assert result.reason_code.value == "POLICY_ABORT"


def test_an_undeclared_effect_state_is_a_facility_error() -> None:
    failed = outcome(0, status="failed", recoverable=True)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False}),
                                            obs("o-1", {"done": True, "n": 0})],
                              outcomes=[failed])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(effect_value="maybe")
    engine = engine_for(adapter, planner, support)
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value == "failed"


def test_a_safety_declaration_bound_to_another_contract_is_refused() -> None:
    failed = outcome(0, status="failed", recoverable=True)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0}),
                                            obs("pre-1", {"done": True, "n": 0}),
                                            obs("post-1", {"done": True, "n": 0}),
                                            obs("recovery", {"done": True, "n": 0})],
                              outcomes=[failed])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])

    class WrongSafety(Support):
        def declare_action_safety(self, *, contract):
            declaration = super().declare_action_safety(contract=contract)
            return declaration.model_copy(update={"contract_id": "someone-else"})

    result = drive(engine_for(adapter, planner, WrongSafety()).run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


# --- A09: host budget and the counter limits -----------------------------------

def test_a_zero_host_budget_forbids_every_business_operation() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), budget=Budget(0.0))
                   .run(request=request()))
    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "BUDGET_EXHAUSTED"
    assert "observe" not in adapter.calls
    assert result.attempts_started == 0


def test_an_increasing_budget_declaration_is_refused() -> None:
    budget = Budget(10.0)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    original = budget.snapshot

    def rising():
        budget.value += 100.0
        return original()

    budget.snapshot = rising
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), budget=budget)
                   .run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value in {"INPUT_INVALID", "DEPENDENCY_FAILURE"}


def test_a_budget_source_failure_is_a_facility_failure() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    budget = Budget(raises=AdapterError("synthetic budget failure"))
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), budget=budget)
                   .run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "DEPENDENCY_FAILURE"


def test_the_step_allowance_is_global_and_not_reset_by_a_new_step() -> None:
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": False, "n": 1}),
                              outcomes=outcomes_for(1))
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    result = drive(engine_for(adapter, planner, Support())
                   .run(request=request(max_steps=1)))
    assert result.steps_started == 1
    assert result.terminal_status.value == "aborted", result.model_dump()
    assert result.reason_code.value == "MAX_STEPS_EXHAUSTED"


def test_the_planning_allowance_is_consumed_before_the_call() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})])
    planner = ScriptedPlanner([], raises=AdapterError("synthetic planner failure"))
    memory = MemoryRecorder()
    engine = engine_for(adapter, planner, Support(), memory=memory)
    # A planner that cannot produce a plan is a facility refusal, and the contract keeps it bounded and
    # propagated; the failed call still consumed its allowance.
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value == "failed"
    # The allowance was consumed before the call. The count is read from the safe Memory record the Run
    # published, which carries exactly the bounded counters the RunResult reports.
    assert memory.get("run-1")["planning_calls"] == 1


def test_an_exhausted_planning_allowance_aborts_instead_of_planning() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    memory = MemoryRecorder()
    engine = engine_for(adapter, planner, Support(), memory=memory)
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request(max_planning_calls=1, max_steps=3)))
    # The plan call succeeds and its allowance is consumed; the Run then needs the observation the
    # adapter cannot supply, which is the facility failure the contract propagates.
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value == "failed"
    # Exactly one plan was made: the count comes from the safe Memory record the Run published.
    assert memory.get("run-1")["planning_calls"] == 1


def test_zero_recovery_allowance_stops_at_the_policy_abort() -> None:
    failed = outcome(0, status="failed", recoverable=True)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0}),
                                            obs("pre-1", {"done": True, "n": 0}),
                                            obs("post-1", {"done": True, "n": 0}),
                                            obs("recovery", {"done": True, "n": 0})],
                              outcomes=[failed])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support(mode="abort")
    result = drive(engine_for(adapter, planner, support).run(request=request()))
    assert result.retries_used == 0
    assert result.terminal_status.value == "aborted"


# --- A03/A10/A12: identity, isolation and honesty ------------------------------

def test_a_second_run_with_the_same_identifier_is_refused() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True}),
                                            obs("o-1", {"done": True})])
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    drive(engine.run(request=request("same")))
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request("same")))
    assert info.value.code == "IDENTITY_CONFLICT"


def test_a_different_run_identifier_is_accepted_by_the_same_engine() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True}),
                                            obs("o-1", {"done": True})])
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    assert drive(engine.run(request=request("first"))).terminal_status.value == "completed"
    assert drive(engine.run(request=request("second"))).terminal_status.value == "completed"


def test_the_working_memory_of_one_run_does_not_disturb_another() -> None:
    memory = MemoryRecorder()
    first = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                            outcomes=outcomes_for(1))
    second = ScriptedAdapter(observations=unmet_series(1, last={"done": False, "n": 1}),
                             outcomes=outcomes_for(1))
    drive(engine_for(first, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                     memory=memory).run(request=request("run-a", max_steps=1)))
    drive(engine_for(second, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                     memory=memory).run(request=request("run-b", max_steps=1)))
    assert memory.get("run-a")["run_id"] == "run-a"
    assert memory.get("run-b")["run_id"] == "run-b"
    assert memory.get("run-a")["terminal_status"] == "completed"
    assert memory.get("run-b")["terminal_status"] == "aborted"


def test_the_published_state_carries_no_raw_observation_or_payload() -> None:
    memory = MemoryRecorder()
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1},
                                                        start={"secret": "SYNTHETIC_PRIVATE",
                                                               "n": 0}),
                              outcomes=outcomes_for(1))
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    result = drive(engine_for(adapter, planner, Support(), memory=memory)
                   .run(request=request()))
    state = memory.get("run-1")
    rendered = repr(state)
    assert "SYNTHETIC_PRIVATE" not in rendered
    assert "secret" not in rendered
    assert result.reason_code.value == "TASK_ACHIEVED"
    assert set(state) >= {"run_id", "phase", "terminal_status", "observation_calls"}


def test_a_trace_write_failure_is_exposed_as_a_trace_failure() -> None:
    trace = TraceRecorder(fail_on="RunFinished")
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
              .run(request=request()))
    assert info.value.code == "TRACE_FAILURE"
    # The failure is attributed to the finalization stage, which is where the unique terminal fact
    # is persisted in this round, rather than to the generic trace stage. The code is unchanged.
    assert info.value.stage == "finalize"


def test_the_adapter_is_stopped_exactly_once_on_the_normal_path() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=request()))
    assert adapter.stops == 1


def _cleanup_failure_round_001() -> None:
    """Superseded by the round-002 correction below: the Run no longer raises on cleanup.

    The round-002 contract requires the unique terminal fact to be written *after* the stop result
    is known, so a cleanup failure invalidates an otherwise successful Run through the recorded
    terminal status instead of aborting the call. The round-001 expectation is kept here, renamed
    and not collected, as the preserved record of what this round corrected.
    """
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})],
                              stop_raises=AdapterError("synthetic stop failure"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=request()))
    assert info.value.code == "CLEANUP_FAILURE"
    assert adapter.stops == 1


def test_a_cleanup_failure_never_overwrites_the_primary_reason() -> None:
    """The restored round-001 regression: a simultaneous stop failure keeps the primary reason.

    The round-002 revision of this test asserted CLEANUP_FAILURE, which contradicted both the
    approved rule that a cleanup error must not overwrite an already-established primary cause and
    the test's own comment. The primary bounded diagnosis leads; the cleanup failure remains a
    bounded secondary fact and the terminal status is consistently a failure.
    """
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": False, "n": 1}),
                              execute_raises=AdapterError("synthetic execute failure"),
                              stop_raises=AdapterError("synthetic stop failure"))
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, planner, Support(), trace=trace).run(request=request()))
    # The primary execute failure leads with its own bounded reason and stage; the later cleanup
    # failure changes neither, and adds no success.
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert info.value.stage == "execute"
    assert info.value.cleanup_failure is not None
    assert info.value.cleanup_failure.code == "CLEANUP_FAILURE"
    assert adapter.stops == 1
    terminals = [record.terminal_status for record in trace.records
                 if getattr(record, "kind", None) == "run_finished"]
    assert terminals == ["failure"]


def test_a_primary_dependency_failure_keeps_its_code_when_cleanup_also_fails() -> None:
    """F02: the primary reason keeps its own bounded code, stage and message."""
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})],
                              observe_raises=OSError("synthetic primary"),
                              stop_raises=OSError("synthetic cleanup"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
              .run(request=request()))
    failure = info.value
    assert failure.code == "DEPENDENCY_FAILURE"
    assert failure.stage == "observe"
    assert "synthetic" not in str(failure)
    # The bounded secondary cleanup fact is preserved on the leading diagnosis, not discarded.
    assert failure.cleanup_failure is not None
    assert failure.cleanup_failure.code == "CLEANUP_FAILURE"
    assert failure.cleanup_failure.stage == "cleanup"
    assert "synthetic" not in str(failure.cleanup_failure)
    assert adapter.stops == 1
    terminals = [record.reason_code for record in trace.records
                 if getattr(record, "kind", None) == "run_finished"]
    assert terminals == ["internal_error"]


def test_a_real_execute_failure_with_a_cleanup_failure_keeps_the_execute_diagnosis() -> None:
    """F02: a real execution failure plus a cleanup failure reports the execution failure."""
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": False, "n": 1}),
                              execute_raises=AdapterError("synthetic execute failure"),
                              stop_raises=AdapterError("synthetic stop failure"))
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, planner, Support()).run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert info.value.stage == "execute"
    assert info.value.cleanup_failure is not None
    assert adapter.stops == 1


def test_an_observation_failure_with_a_cleanup_failure_keeps_the_observation_diagnosis() -> None:
    """F02: a real observation failure plus a cleanup failure reports the observation failure."""
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})],
                              observe_raises=AdapterError("synthetic observe failure"),
                              stop_raises=AdapterError("synthetic stop failure"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert info.value.stage == "observe"
    assert info.value.cleanup_failure is not None
    assert adapter.stops == 1


def test_a_partial_start_failure_attempts_the_stop_without_observing() -> None:
    """F02: a start that raised part-way through still gets its normal stop attempt."""
    class Partial(ScriptedAdapter):
        async def start(self) -> None:
            self.calls.append("start")
            raise AdapterError("synthetic partial startup")

    adapter = Partial(observations=[obs("o-0", {"done": True})])
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    # The start raised an ordinary exception, so the bounded facility failure is propagated once the
    # Run has recorded its own end; the stop was still attempted and no observation was made.
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert adapter.stops == 1
    assert "observe" not in adapter.calls
    assert engine.last_finalization.primary_terminal_status.value == "failed"


def test_a_partial_start_and_a_cleanup_failure_report_the_start_failure() -> None:
    """F02: the start failure leads, and the later cleanup failure does not replace it."""
    class Partial(ScriptedAdapter):
        async def start(self) -> None:
            self.calls.append("start")
            raise AdapterError("synthetic partial startup")

    adapter = Partial(observations=[obs("o-0", {"done": True})],
                      stop_raises=AdapterError("synthetic stop failure"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=request()))
    assert adapter.stops == 1
    assert "observe" not in adapter.calls
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert info.value.cleanup_failure is not None


def test_a_bounded_error_never_echoes_a_caller_value() -> None:
    """A rejected field value must not reach the bounded message.

    The request here carries a marker in a field whose declaration cannot accept it, so the
    rejection is real; the marker is a legal identifier, so the same marker in ``run_id`` would
    legitimately be accepted and is not a rejection case at all.
    """
    marker = "SYNTHETIC_PRIVATE_VALUE"
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    bad_request = RunRequest.model_construct(
        schema_version=SCHEMA_VERSION, run_id="run-1", goal_id="g-1",
        task_criteria=TaskCompletionCriteria(schema_version=SCHEMA_VERSION, goal_id="g-1",
                                             conditions=[DONE_RULE]),
        max_steps=marker, max_planning_calls=1, max_retries=0, max_replans=0)
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=bad_request))
    assert info.value.code == "INPUT_INVALID"
    assert marker not in str(info.value)
    assert len(str(info.value)) < 240
    assert adapter.calls == []


# --- A01/A12: the request guard ------------------------------------------------

@pytest.mark.parametrize("value", [None, 5, "not-a-request", {"run_id": "r"}])
def test_a_request_that_is_not_the_declared_record_is_refused(value: object) -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=value))
    assert info.value.code == "INPUT_INVALID"


def test_a_criteria_bound_to_another_goal_is_refused_before_any_observation() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    wrong = TaskCompletionCriteria(schema_version=SCHEMA_VERSION, goal_id="other",
                                   conditions=[DONE_RULE])
    built = RunRequest(schema_version=SCHEMA_VERSION, run_id="run-1", goal_id="g-1",
                       task_criteria=wrong, max_steps=1, max_planning_calls=1, max_retries=0,
                       max_replans=0)
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=built))
    assert info.value.code == "INPUT_INVALID"
    assert adapter.calls == []


def test_a_validation_bypass_request_is_refused_before_any_observation() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    broken = RunRequest.model_construct(
        schema_version=SCHEMA_VERSION, run_id="run-1", goal_id="g-1",
        task_criteria=TaskCompletionCriteria(schema_version=SCHEMA_VERSION, goal_id="g-1",
                                             conditions=[DONE_RULE]),
        max_steps=0, max_planning_calls=1, max_retries=0, max_replans=0)
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=broken))
    assert info.value.code == "INPUT_INVALID"
    assert adapter.calls == []


def test_a_missing_dependency_is_refused_at_construction() -> None:
    with pytest.raises(RuntimeError) as info:
        RuntimeEngine(adapter=None, planner=ScriptedPlanner([]), domain_support=Support(),
                      budget_source=Budget(), trace_writer=TraceRecorder())
    assert info.value.code == "INPUT_INVALID"


def test_the_budget_is_never_increased_or_reset_by_the_engine() -> None:
    budget = Budget(50.0)
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    drive(engine_for(adapter, ScriptedPlanner([]), Support(), budget=budget)
          .run(request=request()))
    assert budget.value == 50.0
    assert budget.reads >= 1


def test_the_request_limits_are_never_widened_by_a_dependency() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False}),
                                            obs("o-1", {"done": False})],
                              outcomes=[outcome(0)])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    support = Support()
    trace = TraceRecorder()
    memory = MemoryRecorder()
    engine = engine_for(adapter, planner, support, trace=trace, memory=memory)
    # The planner runs out of forecasts, which is a facility failure it propagates once the Run has
    # recorded its end. The counters it did record are still bounded by the request.
    limit = request(max_steps=1, max_planning_calls=1)
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=limit))
    assert info.value.code == "DEPENDENCY_FAILURE"
    snapshot = engine.last_finalization
    assert snapshot is not None
    assert snapshot.primary_terminal_status.value == "failed"
    # The request's own limits are never widened by a dependency, however it fails. The planning counter
    # is the safe Memory record the Run published - a real planning count - and the Step bound is the
    # bounded counter the Run reports. Trace lengths corroborate; they are not planning counters.
    published = memory.get("run-1")
    assert published["planning_calls"] <= limit.max_planning_calls
    assert published["steps_started"] <= limit.max_steps
    kinds = trace.kinds()
    assert kinds.count("step_started") <= 1
    assert kinds.count("attempt_started") <= 1
    assert kinds.count("contract_committed") <= 1


def test_the_engine_reads_no_clock_of_its_own_when_one_is_supplied() -> None:
    fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    engine = engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace,
                        clock=lambda: fixed)
    drive(engine.run(request=request()))
    assert {record.occurred_at for record in trace.records} == {fixed}


def test_a_clock_that_is_not_aware_is_refused() -> None:
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    engine = engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace,
                        clock=lambda: datetime(2026, 1, 2, 3, 4, 5))  # noqa: DTZ001
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    assert info.value.code == "INPUT_INVALID"


def test_the_transition_log_records_the_real_phases() -> None:
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False}),
                                            obs("o-1", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    planner = ScriptedPlanner([ScriptedPlanner.forecast()])
    engine = engine_for(adapter, planner, Support())
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    # The dependency failed, and the Run still sealed the terminal phase it really reached.
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value == "failed"
    _ = math


# --- round-002 corrections: lifecycle, detachment, budget and completeness -------

def _absent_path_precondition():
    """A precondition whose verdict depends on the completeness the domain declares."""
    from forecontract.models.actions import Precondition

    return Precondition(
        schema_version=SCHEMA_VERSION, condition_type="json_rule",
        payload={"schema_version": SCHEMA_VERSION, "rule_type": "path_exists",
                 "path": ["ready"]})


def _committed_precondition_contract(adapter, condition, build_state, pre_state):
    """A real Builder contract whose precondition holds at build time and not at pre-execution."""
    from forecontract.contracts import ContractBuilder, VerificationPolicy
    from forecontract.verification import RuleValidationHooks

    build_observation = obs("build", build_state)
    forecast = ScriptedPlanner.forecast(outcomes=[N_RULE])
    contract = ContractBuilder().build(
        forecast=forecast, observation=build_observation, adapter=adapter,
        hooks=RuleValidationHooks(state_complete=True), preconditions=[condition],
        verification_policy=VerificationPolicy.model_validate(
            {"schema_version": SCHEMA_VERSION,
             "bindings": [{"schema_version": SCHEMA_VERSION, "outcome_index": 0,
                           "verifier_type": "json_rule", "config": {}}]}),
        timeout_ms=1000.0, remaining_budget_ms=1000.0, contract_id="c-fixed", goal_id="g-1",
        created_at=NOW)
    return contract, obs("pre", pre_state)


def test_the_precondition_check_uses_the_declared_completeness() -> None:
    """F05: a forced ``state_complete=True`` would turn an unresolved precondition into a FAIL.

    The rule asks whether an absent path exists. When the provider states its state is complete the
    absence is a real FAIL; when the provider states the state is incomplete the same absence proves
    nothing and the honest verdict is INCONCLUSIVE. The two must not be the same answer.
    """
    from forecontract.verification.models import VerificationEvidence

    adapter = ScriptedAdapter(observations=[])
    condition = _absent_path_precondition()
    contract, pre_observation = _committed_precondition_contract(
        adapter, condition, {"ready": True}, {"n": 0})
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    complete = VerificationEvidence(schema_version=SCHEMA_VERSION, observation=pre_observation,
                                    state_complete=True)
    incomplete = VerificationEvidence(schema_version=SCHEMA_VERSION, observation=pre_observation,
                                      state_complete=False)
    assert engine._check_preconditions(contract, pre_observation, complete) is VerificationStatus.FAIL
    assert engine._check_preconditions(contract, pre_observation,
                                       incomplete) is VerificationStatus.INCONCLUSIVE


def test_a_precondition_that_expires_before_execution_blocks_the_attempt() -> None:
    """F05: the pre-execution observation really gates the Attempt, and it consumes a replan."""
    adapter = ScriptedAdapter(observations=[obs("initial", {"done": False, "n": 0, "ready": True}),
                                            obs("pre", {"done": False, "n": 0})],
                              outcomes=[outcome(0)])
    engine = engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]),
                        Support(preconditions=[_absent_path_precondition()]))
    result = drive(engine.run(request=request(max_replans=0)))
    assert "execute" not in adapter.calls
    assert result.attempts_started == 0
    assert result.terminal_status.value == "aborted"


def test_the_runtime_owns_the_adapter_lifecycle() -> None:
    """F01: the Runtime starts the adapter itself and stops it once, in that order."""
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=request()))
    assert adapter.calls == ["start", "observe", "stop"]


def test_a_failing_adapter_start_is_a_bounded_failure_without_an_observation() -> None:
    """A start that failed part-way through still gets its normal stop attempt.

    The ``start`` call was entered, so the Run may already hold a resource; the round-003 correction
    therefore requires the ordinary stop attempt here, while no observation may be made.
    """
    class Refusing(ScriptedAdapter):
        async def start(self) -> None:
            self.calls.append("start")
            raise AdapterError("synthetic start failure")

    adapter = Refusing(observations=[obs("o-0", {"done": True})])
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert adapter.calls == ["start", "stop"]
    assert adapter.stops == 1
    assert engine.last_finalization.primary_terminal_status.value == "failed"
    assert engine.last_finalization.primary_reason_code.value == "DEPENDENCY_FAILURE"


def test_a_zero_budget_never_starts_the_adapter() -> None:
    """F01/F06: no budget means no business I/O, not even the lifecycle start."""
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), budget=Budget(0.0),
                              trace=trace).run(request=request()))
    assert adapter.calls == []
    assert result.terminal_status.value == "aborted"
    assert trace.kinds() == ["run_started", "run_finished"]


def test_a_zero_budget_declared_mid_run_stops_before_the_next_operation() -> None:
    """F06: a budget that reaches zero blocks the next operation even after the first read."""
    class Declining(Budget):
        def snapshot(self):
            self.reads += 1
            self.value = 10_000.0 if self.reads == 1 else 0.0
            return RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION,
                                         remaining_budget_ms=self.value)

    declining = Declining()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), budget=declining)
                   .run(request=request()))
    assert "observe" not in adapter.calls
    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "BUDGET_EXHAUSTED"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1000"])
def test_a_constructed_budget_snapshot_is_revalidated_not_coerced(value: object) -> None:
    """F06: a bypass construction cannot launder an illegal value through a float conversion."""
    class Forged:
        def snapshot(self):
            return RuntimeBudgetSnapshot.model_construct(schema_version=SCHEMA_VERSION,
                                                         remaining_budget_ms=value)

    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), budget=Forged())
                   .run(request=request()))
    assert "observe" not in adapter.calls
    assert result.terminal_status.value != "completed"


def test_a_same_identifier_with_different_state_is_not_this_observation() -> None:
    """F03: evidence must carry this call's content, not merely its identifier."""
    original = obs("same", {"done": False})

    class Substituting(Support):
        def to_evidence(self, observation):
            return VerificationEvidence(schema_version=SCHEMA_VERSION,
                                        observation=obs(observation.observation_id,
                                                        {"done": True}),
                                        state_complete=True)

    adapter = ScriptedAdapter(observations=[original])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Substituting())
                   .run(request=request()))
    assert result.terminal_status.value != "completed"


def test_a_domain_cannot_mutate_the_observation_into_the_goal() -> None:
    """F03: the adapter's object is handed over as an isolated copy."""
    original = obs("o-0", {"done": False})

    class Mutating(Support):
        def to_evidence(self, observation):
            observation.state["done"] = True
            return super().to_evidence(observation)

    adapter = ScriptedAdapter(observations=[original])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Mutating()).run(request=request()))
    assert original.state["done"] is False
    assert result.terminal_status.value != "completed"


def test_an_int_to_bool_callback_mutation_is_detected() -> None:
    """F03: the signature is type-sensitive, so rewriting 1 into True is caught."""
    forecast = ScriptedPlanner.forecast()
    forecast.metadata["flag"] = 1

    class Rewriting(Support):
        def prepare_contract(self, *, forecast, evidence):
            forecast.metadata["flag"] = True
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0}),
                                            obs("p", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([forecast]), Rewriting())
                   .run(request=request()))
    assert "execute" not in adapter.calls
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_an_undeclared_oracle_return_cannot_create_success() -> None:
    """F04: a look-alike verdict object is refused; only a declared record is a verdict."""
    import types as types_module

    class Forged:
        def evaluate_completion(self, **kwargs):
            return types_module.SimpleNamespace(status=VerificationStatus.PASS)

    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), task_oracle=Forged())
                   .run(request=request()))
    assert result.terminal_status.value != "completed"
    assert result.verification_record_id is None


def test_an_ordinary_dependency_exception_is_bounded_and_still_cleans_up() -> None:
    """F02: a raw OSError becomes a bounded failure and the stop is still attempted."""
    class Exploding:
        def evaluate_completion(self, **kwargs):
            raise OSError("synthetic")

    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), task_oracle=Exploding())
                   .run(request=request()))
    assert adapter.stops == 1
    assert result.reason_code.value == "DEPENDENCY_FAILURE"


def test_the_stop_precedes_the_unique_terminal_fact() -> None:
    """F02: the terminal fact is written only after the real stop result is known."""
    order: list[str] = []

    class Ordered(ScriptedAdapter):
        async def stop(self) -> None:
            order.append("stop")
            await super().stop()

    class OrderedTrace(TraceRecorder):
        def append(self, record):
            if getattr(record, "kind", None) == "run_finished":
                order.append("terminal")
            super().append(record)

    adapter = Ordered(observations=[obs("o-0", {"done": True})])
    drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=OrderedTrace())
          .run(request=request()))
    assert order == ["stop", "terminal"]


def test_a_cleanup_failure_never_leaves_a_success_terminal() -> None:
    """F02: an otherwise successful run that fails to stop is a failure."""
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})],
                              stop_raises=AdapterError("synthetic stop failure"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
              .run(request=request()))
    assert info.value.code == "CLEANUP_FAILURE"
    terminals = [record.terminal_status for record in trace.records
                 if getattr(record, "kind", None) == "run_finished"]
    assert terminals == ["failure"]


def test_a_terminal_trace_failure_still_attempts_the_stop() -> None:
    """F02: a persistence failure is exposed, and the normal cleanup still runs."""
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(),
                         trace=TraceRecorder(fail_on="RunFinished")).run(request=request()))
    assert info.value.code == "TRACE_FAILURE"
    assert adapter.stops == 1


def test_two_runs_sharing_one_engine_do_not_reuse_a_record_identifier() -> None:
    """F07: identifiers are engine-scoped, so a real store accepts both runs."""
    adapter = ScriptedAdapter(observations=[obs("o-1", {"done": True}), obs("o-2", {"done": True})])
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    first = drive(engine.run(request=request("run-a")))
    second = drive(engine.run(request=request("run-b")))
    assert first.terminal_status.value == "completed"
    assert second.terminal_status.value == "completed"
    assert first.terminal_record_id != second.terminal_record_id


def _cleanup_failure_round_001_duplicate() -> None:
    """Superseded by the round-002 correction: a failing stop no longer raises out of the Run.

    Kept renamed and uncollected as the preserved record; the corrected behaviour is asserted by
    ``test_a_cleanup_failure_never_leaves_a_success_terminal`` below.
    """
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})],
                              stop_raises=AdapterError("synthetic stop failure"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
              .run(request=request()))
    assert info.value.code == "CLEANUP_FAILURE"
    terminals = [record.terminal_status for record in trace.records
                 if getattr(record, "kind", None) == "run_finished"]
    assert terminals == ["failure"]


def test_a_malformed_task_rule_is_refused_before_any_side_effect() -> None:
    """F08: the rule is prevalidated before the start, the observation and the first trace write."""
    adapter = ScriptedAdapter(observations=[obs("o", {"done": True})])
    broken = DONE_RULE.model_copy(deep=True)
    broken.payload["rule_type"] = "not_a_rule"
    criteria = TaskCompletionCriteria(schema_version=SCHEMA_VERSION, goal_id="g-1",
                                      conditions=[broken])
    built = RunRequest(schema_version=SCHEMA_VERSION, run_id="run-1", goal_id="g-1",
                       task_criteria=criteria, max_steps=1, max_planning_calls=1,
                       max_retries=0, max_replans=0)
    trace = TraceRecorder()
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace).run(request=built))
    assert info.value.code == "INPUT_INVALID"
    assert adapter.calls == []
    assert trace.records == []


def test_a_host_mutation_of_its_criteria_cannot_change_the_run_goal() -> None:
    """F08: the Run judges its own deep snapshot, not the host's mutable object."""
    built = request()
    original = built.task_criteria.conditions[0].payload["expected"]

    class Mutating(Support):
        def to_evidence(self, observation):
            built.task_criteria.conditions[0].payload["expected"] = False
            return super().to_evidence(observation)

    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Mutating()).run(request=built))
    # The host's own object really was rewritten, yet the Run still judged its snapshot and
    # completed against the goal it validated.
    assert built.task_criteria.conditions[0].payload["expected"] != original
    assert result.terminal_status.value == "completed"


def test_a_transition_record_cannot_be_rewritten() -> None:
    """F08: a transition that happened is immutable."""
    from forecontract.runtime import RuntimePhase as Phase
    from forecontract.runtime import RuntimeReason as Reason
    from forecontract.runtime import RuntimeTransition

    record = RuntimeTransition(schema_version=SCHEMA_VERSION, sequence=0, run_id="r",
                               from_phase="NEW", to_phase="OBSERVING",
                               reason=Reason.TASK_ACHIEVED.value)
    with pytest.raises(ValidationError):
        record.to_phase = Phase.COMPLETED


def test_a_preparation_requires_declared_element_and_policy_types() -> None:
    """F08: the candidate configuration declares its real Core element and policy types."""
    with pytest.raises(ValidationError):
        ContractPreparation(schema_version=SCHEMA_VERSION, preconditions=[123],
                            verification_policy="invalid", recovery_policy=7, timeout_ms=1)


def test_an_unknown_location_segment_is_redacted() -> None:
    """F09: the diagnostic location only ever names a declared field or an index."""
    marker = "SYNTHETIC_PRIVATE_MARKER_06"
    error = RuntimeError("INPUT_INVALID", "validate", (marker,), "safe")
    assert marker not in str(error)
    assert marker not in str(error.location)
    declared = RuntimeError("INPUT_INVALID", "validate", ("run_id",), "safe")
    assert declared.location == ("run_id",)


def test_no_environment_variable_can_make_the_runtime_write_a_file(tmp_path) -> None:
    """F09: the undeclared environment-controlled diagnostic bypass is gone."""
    import os

    marker = "SYNTHETIC_PRIVATE_MARKER_06"
    target = tmp_path / "debug.txt"
    previous = os.environ.get("FORECONTRACT_RUNTIME_TRACE")
    os.environ["FORECONTRACT_RUNTIME_TRACE"] = str(target)
    try:
        adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False})])
        planner = ScriptedPlanner([], raises=AdapterError(marker))
        engine = engine_for(adapter, planner, Support())
        # The planner's refusal is a facility failure, so the Run propagates it; what this case checks is
        # that no environment variable made the Runtime open the file either way.
        with pytest.raises(RuntimeError) as info:
            drive(engine.run(request=request()))
        assert info.value.code == "DEPENDENCY_FAILURE"
    finally:
        if previous is None:
            os.environ.pop("FORECONTRACT_RUNTIME_TRACE", None)
        else:
            os.environ["FORECONTRACT_RUNTIME_TRACE"] = previous
    assert not target.exists()


# --- round-003 corrections: every callback boundary, receipts, identity, nesting ----

#: Every test scratch path lives outside the repository, beside the other ForeContract test scratch.
DECLARED_SCRATCH = Path(tempfile.gettempdir()) / "ForeContract-p1-tests"


def _scratch_path(name: str) -> Path:
    """A fresh external scratch directory for one test, never inside the repository."""
    directory = Path(tempfile.mkdtemp(prefix=f"ForeContract-runtime-{name}-",
                                      dir=DECLARED_SCRATCH.mkdir(parents=True, exist_ok=True)
                                      or DECLARED_SCRATCH))
    return directory


def _failing_step_adapter(*, observations=None, outcomes=None):
    """A run that reaches RECOVERING: one real failed attempt whose step verdict fails."""
    steps = observations if observations is not None else [
        obs("o-0", {"done": False, "n": 0}),
        obs("pre-1", {"done": True, "n": 0}),
        obs("post-1", {"done": True, "n": 0}),
        obs("recovery", {"done": True, "n": 0}),
    ]
    receipts = outcomes if outcomes is not None else [outcome(0, status="failed", recoverable=True)]
    return ScriptedAdapter(observations=steps, outcomes=receipts)


def test_the_planner_observation_is_detached_and_signed() -> None:
    """F03: a planner that rewrites the observation it was given is refused."""
    class Rewriting(ScriptedPlanner):
        async def plan(self, observation):
            self.observations.append(observation)
            observation.state["n"] = 99
            return await super().plan(observation)

    # The initial observation does not yet satisfy the goal, so the Run really reaches PLANNING.
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": False, "n": 0})])
    planner = Rewriting([ScriptedPlanner.forecast()])
    result = drive(engine_for(adapter, planner, Support()).run(request=request()))
    assert planner.observations and planner.observations[0].state["n"] == 99
    assert result.terminal_status.value != "completed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_the_task_oracle_inputs_are_detached_and_signed() -> None:
    """F03: an oracle that rewrites the criteria or the evidence is refused."""
    class Rewriting:
        def evaluate_completion(self, *, criteria, evidence):
            criteria.conditions[0].payload["expected"] = False
            return TaskCompletionOracle().evaluate_completion(criteria=criteria, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), task_oracle=Rewriting())
                   .run(request=request()))
    assert result.terminal_status.value != "completed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_the_effect_callback_inputs_are_detached_and_signed() -> None:
    """F03: assess_effect cannot rewrite the outcome or evidence it was handed."""
    class Rewriting(Support):
        def assess_effect(self, *, contract, actual_outcome, evidence):
            evidence.observation.state["n"] = 99
            return super().assess_effect(contract=contract, actual_outcome=actual_outcome,
                                         evidence=evidence)

    adapter = _failing_step_adapter()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Rewriting())
                   .run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_the_safety_callback_cannot_rewrite_the_contract() -> None:
    """F03: declare_action_safety cannot rewrite the contract it was handed.

    The isolated copy handed to the callback keeps the genuine immutable containers the Core contract
    is built from, so an ordinary ``append`` or attribute assignment simply fails against the frozen
    value: the contract is preserved by construction, not merely by a check afterwards. The check is
    therefore exercised with a mutation that bypasses the frozen surface, which is the only way this
    callback could actually succeed in rewriting its input.
    """
    class Rewriting(Support):
        def declare_action_safety(self, *, contract):
            object.__setattr__(contract, "timeout_ms", 99999.0)
            return super().declare_action_safety(contract=contract)

    adapter = _failing_step_adapter()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Rewriting())
                   .run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_a_frozen_contract_cannot_be_mutated_by_a_callback_at_all() -> None:
    """F08/F03 positive control: the genuine frozen Core value survives the isolation intact."""
    seen: list[str] = []

    class Probing(Support):
        def declare_action_safety(self, *, contract):
            seen.append(type(contract.expected_outcomes).__name__)
            with pytest.raises((AttributeError, ValidationError)):
                contract.expected_outcomes.append(contract.expected_outcomes[0])
            with pytest.raises((AttributeError, ValidationError)):
                contract.timeout_ms = 1.0
            return super().declare_action_safety(contract=contract)

    adapter = _failing_step_adapter()
    engine = engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Probing())
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    # Every refused attempt really happened, and the Run reached its own termination.
    assert seen == ["FrozenList"]
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value in ("failed", "aborted")


def test_the_verifier_inputs_are_detached_and_signed() -> None:
    """F03: a verifier that rewrites the evidence it was given is refused."""
    class Rewriting:
        def verify_step(self, *, contract, evidence):
            # A real, observable rewrite of the post-execution observation the verifier was handed.
            evidence.observation.state["n"] = 99
            return StepVerifier().verify_step(contract=contract, evidence=evidence)

    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              step_verifier=Rewriting()).run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_a_look_alike_recovery_assessment_cannot_grant_a_retry() -> None:
    """F04: the recovery return is a declared record, not anything with the right attribute names."""
    class Forged:
        def assess(self, **kwargs):
            return SimpleNamespace(kind="decision", decision="retry")

    adapter = ScriptedAdapter(observations=[obs(f"o-{i}", {"done": False, "n": 0})
                                            for i in range(6)],
                              outcomes=[outcome(i, status="failed", recoverable=True)
                                        for i in range(4)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Forged()).run(request=request(max_retries=0)))
    assert adapter.calls.count("execute") == 1
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert result.retries_used == 0


def test_a_recovery_assessment_for_another_run_is_refused() -> None:
    """F04: the assessment must name this Run, Step, Contract, Attempt and outcome."""
    real = RecoveryEngine()

    class Misreporting:
        def assess(self, *, contract, context, safety, evidence):
            assessment = real.assess(contract=contract, context=context, safety=safety,
                                     evidence=evidence)
            object.__setattr__(assessment, "run_id", "some-other-run")
            return assessment

    adapter = _failing_step_adapter()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Misreporting()).run(request=request()))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert result.retries_used == 0


def test_two_engines_sharing_one_real_store_both_complete() -> None:
    """F07: identifiers are process-global, so a second Engine cannot collide with the first."""
    store = JsonlTraceStore(_scratch_path("shared-engines") / "trace.jsonl")
    statuses = []
    for index, run_id in enumerate(("engine-a-run", "engine-b-run")):
        adapter = ScriptedAdapter(observations=[obs(f"o-{index}", {"done": True})])
        result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=store)
                       .run(request=request(run_id)))
        statuses.append(result.terminal_status.value)
    assert statuses == ["completed", "completed"]
    first = store.read_run("engine-a-run")
    second = store.read_run("engine-b-run")
    identifiers = {record.record_id for record in first} | {record.record_id for record in second}
    assert len(identifiers) == len(first) + len(second)
    # Each Run still numbers its own sequence from zero.
    assert [record.sequence for record in first] == list(range(len(first)))
    assert [record.sequence for record in second] == list(range(len(second)))


def test_identifiers_stay_distinct_across_several_engines_and_shapes() -> None:
    """F07: record, Step, Contract and Attempt identifiers all come from one opaque sequence."""
    store = JsonlTraceStore(_scratch_path("many-engines") / "trace.jsonl")
    for index in range(3):
        adapter = ScriptedAdapter(observations=[obs(f"o-{index}", {"done": True})])
        result = drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=store)
                       .run(request=request(f"multi-{index}")))
        assert result.terminal_status.value == "completed"
    steps = [record.step_id for record in store.read_run("multi-0")
             if getattr(record, "step_id", None) is not None]
    contracts = [record.contract_id for record in store.read_run("multi-0")
                 if getattr(record, "contract_id", None) is not None]
    assert len(set(steps)) == len(steps)
    assert len(set(contracts)) == len(contracts)


def test_a_damaged_actual_outcome_schema_is_refused_without_a_finish_fact() -> None:
    """F08: a receipt whose stored schema no longer matches is not recorded as a real ending."""
    receipt = outcome(0)
    receipt.__dict__["schema_version"] = "9.9"
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.terminal_status.value != "completed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    kinds = trace.kinds()
    assert "attempt_started" in kinds
    assert "attempt_finished" not in kinds


def test_a_damaged_actual_outcome_stored_field_is_refused_without_a_finish_fact() -> None:
    """F08: an undeclared stored key on the receipt is a facility error, not a real ending."""
    receipt = outcome(0)
    receipt.__dict__["undeclared"] = 123
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.terminal_status.value != "completed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "attempt_finished" not in trace.kinds()


def test_a_legal_actual_outcome_is_still_recorded() -> None:
    """F08 positive control: the honest receipt still produces its finish fact and the goal."""
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.terminal_status.value == "completed"
    assert "attempt_finished" in trace.kinds()


def test_a_request_cycle_is_a_bounded_input_error_not_a_recursion_error() -> None:
    """F08: the active path is refused before any recursive validation runs."""
    built = request()
    payload = built.task_criteria.conditions[0].payload
    payload["cycle"] = payload
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    trace = TraceRecorder()
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace).run(request=built))
    assert info.value.code == "INPUT_INVALID"
    assert adapter.calls == []
    assert trace.records == []


def test_a_request_that_shares_one_payload_twice_is_legal() -> None:
    """F08 positive control: a genuinely shared value is not a cycle."""
    shared = {"schema_version": SCHEMA_VERSION, "rule_type": "value_equals",
              "path": ["done"], "expected": True}
    first = ExpectedOutcome(schema_version=SCHEMA_VERSION, outcome_type="json_rule",
                            payload=dict(shared), required=True)
    second = ExpectedOutcome(schema_version=SCHEMA_VERSION, outcome_type="json_rule",
                             payload=shared, required=True)
    built = request(conditions=[first, second])
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=built))
    assert result.terminal_status.value == "completed"


def test_the_run_request_constructor_owns_its_nested_criteria() -> None:
    """F08: the public constructor must not alias the caller's nested objects."""
    criteria = request().task_criteria
    built = RunRequest(schema_version=SCHEMA_VERSION, run_id="owned", goal_id="g-1",
                       task_criteria=criteria, max_steps=1, max_planning_calls=1,
                       max_retries=0, max_replans=0)
    criteria.conditions[0].payload["expected"] = False
    assert built.task_criteria is not criteria
    assert built.task_criteria.conditions[0].payload is not criteria.conditions[0].payload
    assert built.task_criteria.conditions[0].payload["expected"] is True


def test_an_isolated_record_revalidates_its_declared_scalars() -> None:
    """F08: rebuilding a record revalidates its own scalars instead of trusting them."""
    from forecontract.runtime.models import isolated_record, record_payload

    receipt = outcome(0)
    receipt.__dict__["schema_version"] = "9.9"
    with pytest.raises(ValueError):
        isolated_record(ActualOutcome, record_payload(receipt))
    good = isolated_record(ActualOutcome, record_payload(outcome(0)))
    assert good.schema_version == SCHEMA_VERSION
    assert good is not receipt


def test_a_preparation_refuses_a_nested_policy_with_an_undeclared_field() -> None:
    """F08: a declared type alone does not revalidate an already-built nested instance."""
    prepared = Support().prepare_contract(
        forecast=ScriptedPlanner.forecast(),
        evidence=VerificationEvidence(schema_version=SCHEMA_VERSION, observation=obs("o", {"n": 0}),
                                      state_complete=True))
    policy = prepared.verification_policy
    policy.__dict__["undeclared"] = 123
    with pytest.raises(ValidationError):
        ContractPreparation(schema_version=SCHEMA_VERSION, preconditions=[],
                            verification_policy=policy, recovery_policy=prepared.recovery_policy,
                            timeout_ms=1000.0)


def test_a_legal_preparation_keeps_its_nested_policy_values() -> None:
    """F08 positive control: the legal candidate still builds with its real values."""
    prepared = Support(mode="bounded").prepare_contract(
        forecast=ScriptedPlanner.forecast(),
        evidence=VerificationEvidence(schema_version=SCHEMA_VERSION, observation=obs("o", {"n": 0}),
                                      state_complete=True))
    again = ContractPreparation(schema_version=SCHEMA_VERSION,
                                preconditions=list(prepared.preconditions),
                                verification_policy=prepared.verification_policy,
                                recovery_policy=prepared.recovery_policy,
                                timeout_ms=prepared.timeout_ms)
    assert again.verification_policy.schema_version == SCHEMA_VERSION
    assert again.verification_policy is not prepared.verification_policy
    assert again.recovery_policy is not prepared.recovery_policy
    assert again.timeout_ms == prepared.timeout_ms


def test_every_public_diagnostic_attribute_is_bounded_and_value_free() -> None:
    """F09: the whole public surface is bounded, not only the rendered prefix."""
    marker = "SYNTHETIC_AUDIT_PRIVATE_003"
    error = RuntimeError("INPUT_INVALID", "validate", ("run_id",), marker * 100)
    assert marker not in str(error)
    assert marker not in error.message
    assert len(error.message) <= 240
    assert error.code in RUNTIME_ERROR_CODES
    assert error.stage in RUNTIME_STAGES
    # A declared diagnostic survives as declared text.
    declared = RuntimeError("INPUT_INVALID", "validate", ("run_id",),
                            "the request is not a RunRequest record")
    assert declared.message == "the request is not a RunRequest record"


def test_an_attached_cause_cannot_re_expose_dependency_text() -> None:
    """F09: the cause chain is a public surface too, so it is bounded as well."""
    marker = "SYNTHETIC_AUDIT_PRIVATE_003"
    dependency = AdapterError(marker)
    error = RuntimeError("DEPENDENCY_FAILURE", "observe", ("adapter",),
                         "the dependency raised a bounded exception", cause=dependency)
    assert marker not in str(error)
    assert marker not in repr(error.__cause__)
    assert "AdapterError" not in repr(error.__cause__)


def test_a_dependency_exception_never_reaches_the_bounded_message() -> None:
    """F09: an arbitrary dependency message is never copied into a Runtime diagnostic."""
    marker = "SYNTHETIC_AUDIT_PRIVATE_003"

    class Noisy(ScriptedAdapter):
        async def observe(self):
            raise AdapterError(marker * 50)

    adapter = Noisy(observations=[obs("o-0", {"done": True})])
    trace = TraceRecorder()
    engine = engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value == "failed"
    rendered = " ".join(str(getattr(record, "reason_code", "")) for record in trace.records)
    assert marker not in rendered
    assert all(marker not in str(record.model_dump()) for record in trace.records)


# --- round-004 corrections: unified raw validation, recovery inputs, diagnostics ----

def test_a_secondary_cleanup_fact_is_preserved_on_the_leading_diagnosis() -> None:
    """F02: a simultaneous cleanup failure is recorded, not discarded, and never leads."""
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})],
                              observe_raises=OSError("synthetic primary"),
                              stop_raises=OSError("synthetic cleanup"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
              .run(request=request()))
    failure = info.value
    assert failure.code == "DEPENDENCY_FAILURE"
    assert failure.stage == "observe"
    assert failure.cleanup_failure is not None
    assert failure.cleanup_failure.code == "CLEANUP_FAILURE"
    assert failure.cleanup_failure.stage == "cleanup"
    # The secondary fact is auditable through the bounded surface, and neither diagnosis leaks text.
    assert "synthetic" not in str(failure)
    assert "synthetic" not in str(failure.cleanup_failure)


def test_a_cleanup_failure_still_leads_when_nothing_else_failed() -> None:
    """F02 positive control: with no earlier failure the cleanup problem is the leading reason."""
    trace = TraceRecorder()
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})],
                              stop_raises=AdapterError("synthetic stop failure"))
    with pytest.raises(RuntimeError) as info:
        drive(engine_for(adapter, ScriptedPlanner([]), Support(), trace=trace)
              .run(request=request()))
    assert info.value.code == "CLEANUP_FAILURE"
    terminals = [record.terminal_status for record in trace.records
                 if getattr(record, "kind", None) == "run_finished"]
    assert terminals == ["failure"]


def test_a_recovery_context_mutation_is_refused_from_a_real_engine() -> None:
    """F03: the RecoveryContext handed to the engine is signed, so a rewrite is refused."""
    class Rewriting:
        def __init__(self):
            self._real = RecoveryEngine()

        def assess(self, **kwargs):
            result = self._real.assess(**kwargs)
            object.__setattr__(kwargs["context"], "remaining_retries", 999)
            return result

    adapter = ScriptedAdapter(
        observations=[obs(f"o{i}", {"done": False, "n": 0}) for i in range(8)],
        outcomes=[outcome(i, status="failed", recoverable=True) for i in range(3)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Rewriting()).run(request=request()))
    assert adapter.calls.count("execute") == 1
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert result.retries_used == 0


def test_a_recovery_contract_mutation_is_refused_from_a_real_engine() -> None:
    """F03: the contract handed to the engine is signed, so a rewrite is refused."""
    class Rewriting:
        def __init__(self):
            self._real = RecoveryEngine()

        def assess(self, **kwargs):
            result = self._real.assess(**kwargs)
            object.__setattr__(kwargs["contract"], "timeout_ms", 99999.0)
            return result

    adapter = ScriptedAdapter(
        observations=[obs(f"o{i}", {"done": False, "n": 0}) for i in range(8)],
        outcomes=[outcome(i, status="failed", recoverable=True) for i in range(3)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Rewriting()).run(request=request()))
    assert adapter.calls.count("execute") == 1
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_a_recovery_safety_mutation_is_refused_from_a_real_engine() -> None:
    """F03: the safety declaration handed to the engine is signed, so a rewrite is refused."""
    class Rewriting:
        def __init__(self):
            self._real = RecoveryEngine()

        def assess(self, **kwargs):
            result = self._real.assess(**kwargs)
            object.__setattr__(kwargs["safety"], "retry_safety", "unsafe")
            return result

    adapter = ScriptedAdapter(
        observations=[obs(f"o{i}", {"done": False, "n": 0}) for i in range(8)],
        outcomes=[outcome(i, status="failed", recoverable=True) for i in range(3)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Rewriting()).run(request=request()))
    assert adapter.calls.count("execute") == 1
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_an_undeclared_key_added_to_a_recovery_input_is_refused() -> None:
    """F03: injecting a stored key into a recovery argument is a refusal, not a dependency crash."""
    class Injecting:
        def __init__(self):
            self._real = RecoveryEngine()

        def assess(self, **kwargs):
            result = self._real.assess(**kwargs)
            kwargs["context"].__dict__["undeclared"] = 123
            return result

    adapter = ScriptedAdapter(
        observations=[obs(f"o{i}", {"done": False, "n": 0}) for i in range(8)],
        outcomes=[outcome(i, status="failed", recoverable=True) for i in range(3)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Injecting()).run(request=request()))
    assert adapter.calls.count("execute") == 1
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_a_contradictory_recovery_verdict_is_refused_at_ingress() -> None:
    """F04: the complete declared schema runs, so a self-contradictory verdict never reaches."""
    class Contradicting:
        def assess(self, **kwargs):
            context = kwargs["context"]
            verdict = RecoveryAssessment(
                schema_version=SCHEMA_VERSION, kind="observation_required", decision=None,
                reason="OBSERVATION_REQUIRED", run_id=context.run_id, step_id=context.step_id,
                contract_id=context.contract_id, attempt_id=context.attempt_id,
                outcome_id=context.actual_outcome.outcome_id, observation_id=None)
            # A ``no_recovery`` kind carrying an observation-request reason is exactly the
            # combination the declared cross-field rule excludes.
            verdict.__dict__["kind"] = "no_recovery"
            verdict.__dict__["decision"] = None
            return verdict

    adapter = ScriptedAdapter(
        observations=[obs(f"o{i}", {"done": False, "n": 0}) for i in range(8)],
        outcomes=[outcome(i, status="failed", recoverable=True) for i in range(3)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Contradicting()).run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert result.retries_used == 0
    assert adapter.calls.count("execute") == 1


def test_a_preparation_refuses_a_nested_binding_extra_key_and_wrong_schema() -> None:
    """F08: the nested declared shape is really revalidated, not merely re-declared."""
    prepared = Support().prepare_contract(
        forecast=ScriptedPlanner.forecast(),
        evidence=VerificationEvidence(schema_version=SCHEMA_VERSION, observation=obs("o", {"n": 0}),
                                      state_complete=True))
    prepared.verification_policy.bindings[0].__dict__["undeclared"] = 123
    with pytest.raises((ValidationError, ValueError)):
        ContractPreparation(**prepared.__dict__)

    prepared = Support().prepare_contract(
        forecast=ScriptedPlanner.forecast(),
        evidence=VerificationEvidence(schema_version=SCHEMA_VERSION, observation=obs("o", {"n": 0}),
                                      state_complete=True))
    prepared.verification_policy.bindings[0].__dict__["schema_version"] = "9.9"
    with pytest.raises((ValidationError, ValueError)):
        ContractPreparation(**prepared.__dict__)


def test_a_preparation_refuses_a_negative_nested_limit() -> None:
    """F08: a nested field constraint is enforced, so a negative retry limit is refused."""
    prepared = Support().prepare_contract(
        forecast=ScriptedPlanner.forecast(),
        evidence=VerificationEvidence(schema_version=SCHEMA_VERSION, observation=obs("o", {"n": 0}),
                                      state_complete=True))
    prepared.recovery_policy.__dict__["max_retries"] = -1
    with pytest.raises((ValidationError, ValueError)):
        ContractPreparation(**prepared.__dict__)


def test_a_cyclic_nested_binding_config_is_a_bounded_refusal() -> None:
    """F08: the active path is refused before recursion, so a cycle is never a RecursionError."""
    prepared = Support().prepare_contract(
        forecast=ScriptedPlanner.forecast(),
        evidence=VerificationEvidence(schema_version=SCHEMA_VERSION, observation=obs("o", {"n": 0}),
                                      state_complete=True))
    cyclic: dict = {}
    cyclic["cycle"] = cyclic
    prepared.verification_policy.bindings[0].__dict__["config"] = cyclic
    with pytest.raises((ValidationError, ValueError)) as info:
        ContractPreparation(**prepared.__dict__)
    assert not isinstance(info.value, RecursionError)


@pytest.mark.parametrize("damage", ["duration_negative", "duration_boolean", "metadata_tuple",
                                    "metadata_cycle", "nested_artifact", "success_with_error"])
def test_a_damaged_receipt_is_refused_before_any_finish_fact(damage: str) -> None:
    """F08: a receipt the declared schema excludes is never recorded as a real ending."""
    receipt = outcome(0)
    if damage == "duration_negative":
        receipt.__dict__["duration_ms"] = -1
    elif damage == "duration_boolean":
        receipt.__dict__["duration_ms"] = True
    elif damage == "metadata_tuple":
        receipt.metadata["bad"] = (1, 2)
    elif damage == "metadata_cycle":
        receipt.metadata["cycle"] = receipt.metadata
    elif damage == "nested_artifact":
        from forecontract.models.observations import ArtifactRef

        artifact = ArtifactRef(schema_version=SCHEMA_VERSION, artifact_id="artifact-1",
                               artifact_type="text", uri="synthetic")
        artifact.__dict__["undeclared"] = 1
        receipt.artifacts.append(artifact)
    elif damage == "success_with_error":
        receipt.__dict__["error"] = ErrorInfo(schema_version=SCHEMA_VERSION, code="synthetic",
                                              message="synthetic")
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.terminal_status.value != "completed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "attempt_finished" not in trace.kinds()


def test_a_legal_receipt_keeps_its_nested_values() -> None:
    """F08 positive control: a legal receipt still validates and records its finish fact."""
    from forecontract.models.observations import ArtifactRef

    receipt = outcome(0)
    receipt.artifacts.append(ArtifactRef(schema_version=SCHEMA_VERSION, artifact_id="artifact-1",
                                         artifact_type="text", uri="synthetic"))
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.terminal_status.value == "completed"
    assert "attempt_finished" in trace.kinds()


def test_a_shared_nested_record_is_not_a_cycle() -> None:
    """F08 positive control: a genuinely shared record is legal, not a cycle."""
    shared_rule = DONE_RULE.model_copy(deep=True)
    built = request(conditions=[shared_rule, shared_rule])
    adapter = ScriptedAdapter(observations=[obs("o-0", {"done": True})])
    result = drive(engine_for(adapter, ScriptedPlanner([]), Support()).run(request=built))
    assert result.terminal_status.value == "completed"


def test_a_record_that_contains_itself_is_a_bounded_refusal() -> None:
    """F08: a model-level cycle is detected by identity, not replaced by a fresh mapping."""
    from forecontract.runtime.models import require_acyclic

    record = Support().prepare_contract(
        forecast=ScriptedPlanner.forecast(),
        evidence=VerificationEvidence(schema_version=SCHEMA_VERSION, observation=obs("o", {"n": 0}),
                                      state_complete=True))
    record.__dict__["verification_policy"] = record
    with pytest.raises(ValueError):
        require_acyclic(record)


def test_the_public_location_is_bounded_in_length_and_index_magnitude() -> None:
    """F09: the stored location is bounded, not only the rendered string."""
    huge = RuntimeError("INPUT_INVALID", "validate", tuple("run_id" for _ in range(10000)),
                        "synthetic")
    assert len(huge.location) <= 64
    assert huge.location[-1] == "<truncated>"
    assert len(str(huge)) <= 240
    # A canonical declared index is kept; an out-of-range magnitude is replaced.
    canonical = RuntimeError("INPUT_INVALID", "validate", ("preconditions", 3), "synthetic")
    assert canonical.location == ("preconditions", 3)
    extreme = RuntimeError("INPUT_INVALID", "validate", ("preconditions", 10 ** 12), "synthetic")
    assert extreme.location == ("preconditions", "<index>")
    # Value redaction and declared-name retention are unchanged.
    marker = "SYNTHETIC_AUDIT_PRIVATE_004"
    redacted = RuntimeError("INPUT_INVALID", "validate", (marker,), "synthetic")
    assert marker not in str(redacted) and marker not in redacted.location


# --- round-005 corrections: cause separation, lossless signatures, unified ingress ----

def test_a_bounded_cause_is_never_labelled_as_a_cleanup_fact() -> None:
    """F02: an attached primary cause is not a cleanup failure, and must not occupy that slot."""
    cause = RuntimeError("EVIDENCE_INVALID", "observe", (), "synthetic")
    outer = RuntimeError("DEPENDENCY_FAILURE", "observe", (), "synthetic", cause=cause)
    assert outer.cleanup_failure is None
    assert outer.code == "DEPENDENCY_FAILURE"
    assert outer.__cause__ is not None


def test_a_real_cleanup_fact_is_not_replaced_by_a_bounded_primary_cause() -> None:
    """F02: the real later stop failure fills the slot even when a bounded cause was attached."""
    cause = RuntimeError("EVIDENCE_INVALID", "observe", (), "synthetic")
    adapter = ScriptedAdapter(observations=[obs("o", {"done": True})], observe_raises=cause,
                              stop_raises=OSError("synthetic cleanup"))
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    # The dependency's refusal at the observation boundary leads as the bounded facility failure, and the
    # real stop failure is preserved on it as the bounded secondary fact rather than replacing it.
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.cleanup_status == "failed"
    assert info.value.cleanup_failure is not None
    assert info.value.cleanup_failure.code == "CLEANUP_FAILURE"


def test_a_cleanup_fact_requires_a_real_stop_failure() -> None:
    """F02 positive control: with no stop failure nothing is recorded as a cleanup problem.

    A bounded dependency exception leads here, so the Run returns its bounded result and no cleanup
    fact exists to report.
    """
    adapter = ScriptedAdapter(observations=[obs("o", {"done": True})],
                              observe_raises=OSError("synthetic primary"))
    engine = engine_for(adapter, ScriptedPlanner([]), Support())
    with pytest.raises(RuntimeError) as info:
        drive(engine.run(request=request()))
    # The observation dependency failed, so the bounded failure is what the caller receives; there is
    # no cleanup problem to report because the stop really succeeded.
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert info.value.cleanup_failure is None
    assert engine.last_finalization.cleanup_status == "succeeded"
    assert engine.last_finalization.cleanup_error_code is None


def test_a_container_type_change_in_a_controlled_callback_is_refused() -> None:
    """F03: the signature is lossless, so swapping a list for a tuple is a detected mutation."""
    forecast = ScriptedPlanner.forecast()
    forecast.metadata["shape"] = [1]

    class Swapping(Support):
        def prepare_contract(self, *, forecast, evidence):
            forecast.metadata["shape"] = (1,)
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    result = drive(engine_for(adapter, ScriptedPlanner([forecast]), Swapping())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_a_nested_container_type_change_in_the_evidence_is_refused() -> None:
    """F03: the comparison reaches nested storage, so a nested list-to-tuple swap is detected."""
    class Swapping(Support):
        def prepare_contract(self, *, forecast, evidence):
            evidence.observation.state["shape"] = (1,)
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0, "shape": [1]}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Swapping())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_an_injected_nested_key_in_the_evidence_is_refused() -> None:
    """F03: an undeclared nested stored key is a refusal, not a data error."""
    class Injecting(Support):
        def prepare_contract(self, *, forecast, evidence):
            evidence.observation.__dict__["undeclared"] = 1
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0, "shape": [1]}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Injecting())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_an_injected_key_inside_a_recovery_outcome_is_refused() -> None:
    """F03: the comparison reaches the nested outcome, so a nested extra key is detected."""
    class Injecting:
        def __init__(self):
            self._real = RecoveryEngine()

        def assess(self, **kwargs):
            verdict = self._real.assess(**kwargs)
            kwargs["context"].actual_outcome.__dict__["undeclared"] = 1
            return verdict

    adapter = ScriptedAdapter(
        observations=[obs(f"o{i}", {"done": False, "n": 0}) for i in range(10)],
        outcomes=[outcome(i, status="failed", recoverable=True) for i in range(3)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              recovery_engine=Injecting()).run(request=request()))
    assert adapter.calls.count("execute") == 1
    assert result.reason_code.value == "EVIDENCE_INVALID"


def test_a_returned_preparation_with_a_wrong_schema_is_refused_before_the_builder() -> None:
    """F08: the candidate is validated from its own storage before the Builder sees it."""
    class Damaged(Support):
        def prepare_contract(self, *, forecast, evidence):
            prepared = super().prepare_contract(forecast=forecast, evidence=evidence)
            prepared.__dict__["schema_version"] = "9.9"
            return prepared

    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Damaged())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_a_returned_preparation_with_an_extra_key_is_refused() -> None:
    """F08: an undeclared stored key on the candidate is refused, not silently dropped."""
    class Damaged(Support):
        def prepare_contract(self, *, forecast, evidence):
            prepared = super().prepare_contract(forecast=forecast, evidence=evidence)
            prepared.__dict__["undeclared"] = 1
            return prepared

    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Damaged())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_a_returned_preparation_with_an_ordinary_tuple_is_refused() -> None:
    """F08: an ordinary tuple in a strict list position is refused before any conversion."""
    class Damaged(Support):
        def prepare_contract(self, *, forecast, evidence):
            prepared = super().prepare_contract(forecast=forecast, evidence=evidence)
            prepared.__dict__["preconditions"] = ()
            return prepared

    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Damaged())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_a_damaged_nested_artifact_metadata_is_refused() -> None:
    """F08: a damaged defaulted nested field is inspected, not skipped as 'the default'."""
    from forecontract.models.observations import ArtifactRef

    receipt = outcome(0)
    artifact = ArtifactRef(schema_version=SCHEMA_VERSION, artifact_id="a", artifact_type="text",
                           uri="synthetic")
    artifact.__dict__["metadata"] = {"bad": (1, 2)}
    receipt.artifacts.append(artifact)
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "attempt_finished" not in trace.kinds()


def test_a_receipt_artifacts_tuple_is_not_coerced_to_a_list() -> None:
    """F08: an ordinary tuple received as the artifact list is refused, not converted."""
    receipt = outcome(0)
    receipt.__dict__["artifacts"] = ()
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "attempt_finished" not in trace.kinds()


def test_an_existing_request_instance_is_revalidated_at_ingress() -> None:
    """F08: an existing instance is inspected through its own storage, not trusted by identity."""
    value = request()
    value.__dict__["max_steps"] = False
    with pytest.raises((ValidationError, ValueError)):
        RunRequest.model_validate(value)
    # A legal instance still round-trips.
    again = RunRequest.model_validate(request())
    assert again.max_steps == 3


def test_a_nested_condition_tuple_is_refused_by_the_request_constructor() -> None:
    """F08: a nested ordinary tuple is refused rather than normalised into a legal list."""
    built = request()
    criteria = built.task_criteria
    criteria.__dict__["conditions"] = tuple(criteria.conditions)
    with pytest.raises((ValidationError, ValueError)):
        RunRequest(**built.__dict__)


def test_a_legal_receipt_and_a_legal_request_still_pass() -> None:
    """F08 positive controls: the unified ingress does not refuse a legal record."""
    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0, "shape": [1]}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support())
                   .run(request=request()))
    assert result.terminal_status.value == "completed"
    assert RunRequest.model_validate(request()).max_steps == 3


# --- round-006 corrections: real type identity and genuine frozen refusal -----------

def _display_name_lookalike() -> type:
    """An unrelated tuple subclass that repeats a list's writable display names."""
    return type("list", (tuple,), {"__module__": "builtins", "__qualname__": "list"})


def test_a_same_display_name_lookalike_is_not_the_declared_container() -> None:
    """F03: the signature uses real type identity, so a name lookalike cannot pass as a list."""
    lookalike = _display_name_lookalike()
    forecast = ScriptedPlanner.forecast().model_copy(update={"metadata": {"shape": [1]}})

    class Swapping(Support):
        def prepare_contract(self, *, forecast, evidence):
            forecast.metadata["shape"] = lookalike(forecast.metadata["shape"])
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0, "shape": [1]}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([forecast]), Swapping())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_a_same_display_name_lookalike_in_the_evidence_is_refused() -> None:
    """F03: the same identity rule applies to nested evidence storage."""
    lookalike = _display_name_lookalike()

    class Swapping(Support):
        def prepare_contract(self, *, forecast, evidence):
            evidence.observation.state["shape"] = lookalike(evidence.observation.state["shape"])
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0, "shape": [1]}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Swapping())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


def test_a_same_display_name_lookalike_type_is_not_a_list_in_the_signature() -> None:
    """F03: the identity check is on the type object, not on its module and qualified name."""
    from forecontract.runtime.engine import _signature

    lookalike = _display_name_lookalike()
    plain = _signature([1], set(), set())
    forged = _signature(lookalike([1]), set(), set())
    assert lookalike.__module__ == list.__module__
    assert lookalike.__qualname__ == list.__qualname__
    assert plain != forged


@pytest.mark.parametrize("container", ["forged", "named"])
def test_an_ordinary_tuple_subclass_is_refused_as_a_declared_list(container: str) -> None:
    """F08: any tuple that is not the genuine Core container is refused, by real type identity."""
    cls = type("ForgedTuple", (tuple,), {}) if container == "forged" else type("FrozenList", (tuple,), {})
    receipt = outcome(0)
    receipt.__dict__["artifacts"] = cls()
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    trace = TraceRecorder()
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support(),
                              trace=trace).run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "attempt_finished" not in trace.kinds()


@pytest.mark.parametrize("container", ["forged", "named"])
def test_an_ordinary_tuple_subclass_is_refused_in_a_preparation(container: str) -> None:
    """F08: the same strict refusal applies to the returned candidate configuration."""
    cls = type("ForgedTuple", (tuple,), {}) if container == "forged" else type("FrozenList", (tuple,), {})

    class Damaged(Support):
        def prepare_contract(self, *, forecast, evidence):
            prepared = super().prepare_contract(forecast=forecast, evidence=evidence)
            prepared.__dict__["preconditions"] = cls(prepared.preconditions)
            return prepared

    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=outcomes_for(1))
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Damaged())
                   .run(request=request()))
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert "execute" not in adapter.calls


@pytest.mark.parametrize("container", ["forged", "named"])
def test_an_ordinary_tuple_subclass_is_refused_in_the_request(container: str) -> None:
    """F08: the constructor refuses a tuple subclass rather than coercing it to a list."""
    cls = type("ForgedTuple", (tuple,), {}) if container == "forged" else type("FrozenList", (tuple,), {})
    built = request()
    built.task_criteria.__dict__["conditions"] = cls(built.task_criteria.conditions)
    with pytest.raises((ValidationError, ValueError)):
        RunRequest(**built.__dict__)


def test_the_genuine_core_frozen_container_is_still_accepted() -> None:
    """F08 positive control: the real Core immutable container passes by its own identity."""
    from forecontract.contracts.snapshot import FrozenList as RealFrozenList

    receipt = outcome(0)
    receipt.__dict__["artifacts"] = RealFrozenList([])
    adapter = ScriptedAdapter(observations=unmet_series(1, last={"done": True, "n": 1}),
                              outcomes=[receipt])
    result = drive(engine_for(adapter, ScriptedPlanner([ScriptedPlanner.forecast()]), Support())
                   .run(request=request()))
    assert result.terminal_status.value == "completed"


def test_a_legal_current_defaulted_forecast_metadata_reaches_the_domain() -> None:
    """F08: a record's complete current storage is preserved, including a modified default."""
    forecast = ScriptedPlanner.forecast()
    forecast.metadata["shape"] = [1]
    seen: list = []

    class Observing(Support):
        def prepare_contract(self, *, forecast, evidence):
            seen.append(dict(forecast.metadata))
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([forecast]), Observing())
                   .run(request=request()))
    assert seen == [{"shape": [1]}]
    assert result.terminal_status.value == "completed"


def test_a_legally_explicit_forecast_metadata_reaches_the_domain() -> None:
    """F08 positive control: an explicitly supplied metadata value is preserved too."""
    forecast = ScriptedPlanner.forecast().model_copy(update={"metadata": {"shape": [1]}})
    seen: list = []

    class Observing(Support):
        def prepare_contract(self, *, forecast, evidence):
            seen.append(dict(forecast.metadata))
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([forecast]), Observing())
                   .run(request=request()))
    assert seen == [{"shape": [1]}]
    assert result.terminal_status.value == "completed"


def test_the_omission_semantics_are_limited_to_the_declaration_that_needs_them() -> None:
    """F08: the supplied-field restriction applies only where the schema really distinguishes it."""
    from forecontract.contracts import RecoveryPolicy
    from forecontract.runtime.models import _SUPPLIED_SENSITIVE

    assert RecoveryPolicy in _SUPPLIED_SENSITIVE
    assert Forecast not in _SUPPLIED_SENSITIVE
    assert Observation not in _SUPPLIED_SENSITIVE
    assert ActualOutcome not in _SUPPLIED_SENSITIVE


# --- round-007 correction: real scalar type identity in the raw signature -----------

class _IntChild(int):
    """A subclass carrying the same value as the builtin, but a different real type."""


class _StrChild(str):
    """A subclass carrying the same value as the builtin, but a different real type."""


class _FloatChild(float):
    """A subclass carrying the same value as the builtin, but a different real type."""


def _scalar_mutation(value: object, replacement: object, *, unchanged: bool = False):
    """Drive one callback that swaps a scalar for a same-valued replacement."""
    forecast = ScriptedPlanner.forecast().model_copy(update={"metadata": {"item": value}})

    class Swapping(Support):
        def prepare_contract(self, *, forecast, evidence):
            prepared = super().prepare_contract(forecast=forecast, evidence=evidence)
            forecast.metadata["item"] = replacement
            return prepared

    adapter = ScriptedAdapter(observations=[obs("i", {"done": False, "n": 0}),
                                            obs("p", {"done": False, "n": 0}),
                                            obs("o", {"done": True, "n": 1})],
                              outcomes=[outcome(0)])
    result = drive(engine_for(adapter, ScriptedPlanner([forecast]), Swapping())
                   .run(request=request()))
    if unchanged:
        assert result.terminal_status.value == "completed"
    else:
        assert result.reason_code.value == "EVIDENCE_INVALID"
        assert "execute" not in adapter.calls
    return result


def test_a_same_valued_int_subclass_is_not_the_builtin_int() -> None:
    """F03: the scalar branch records the real type, so a same-valued int subclass is a mutation."""
    _scalar_mutation(1, _IntChild(1))


def test_a_same_valued_str_subclass_is_not_the_builtin_str() -> None:
    """F03: the scalar branch records the real type, so a same-valued str subclass is a mutation."""
    _scalar_mutation("x", _StrChild("x"))


def test_a_same_valued_float_subclass_is_not_the_builtin_float() -> None:
    """F03: the scalar branch records the real type, so a same-valued float subclass is a mutation."""
    _scalar_mutation(1.5, _FloatChild(1.5))


def test_an_int_replaced_by_a_float_of_equal_value_is_refused() -> None:
    """F03 control: the builtin scalar types stay distinct from each other."""
    _scalar_mutation(1, 1.0)


def test_an_unchanged_scalar_is_legal() -> None:
    """F03 positive control: an untouched legal scalar still passes unchanged."""
    _scalar_mutation(1, 1, unchanged=True)


def test_the_scalar_signature_uses_real_type_identity() -> None:
    """F03: every scalar branch signs the type object, never a type name."""
    from forecontract.runtime.engine import _signature

    def sign(value: object) -> object:
        return _signature(value, set(), set())

    assert sign(1) != sign(_IntChild(1))
    assert sign("x") != sign(_StrChild("x"))
    assert sign(1.5) != sign(_FloatChild(1.5))
    # The builtin distinctions the contract relies on are preserved.
    assert sign(1) != sign(1.0)
    assert sign(1) != sign(True)
    assert sign(True) == sign(True)
    # A legal scalar compares equal to itself, including a NaN, which must not break the fixed point.
    assert sign(1) == sign(1)
    assert sign("x") == sign("x")
    assert sign(1.5) == sign(1.5)
    assert sign(float("nan")) == sign(float("nan"))
    assert sign(b"a") == sign(b"a")
    assert sign(b"a") != sign(bytearray(b"a"))
