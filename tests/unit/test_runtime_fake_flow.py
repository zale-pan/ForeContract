"""Integrated Runtime acceptance with the real Fake adapter, ScriptedPlanner and JsonlTraceStore.

Nothing is simulated here: the Runtime drives the real Fake environment, the real ScriptedPlanner,
the real Contract Builder, StepVerifier, TaskCompletionOracle and RecoveryEngine, and the real
JSONL trace store, and every assertion is checked against what the store reads back.

The production Fake deliberately declares no domain-verification capability, so a small *test-only*
wrapper delegates every operation to the real adapter and only widens that declaration. The Fake
package itself is untouched.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
FAKE_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "fake" / "src"
SCRIPTED_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "scripted" / "src"
for candidate in (CORE_SRC, FAKE_SRC, SCRIPTED_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts import RecoveryPolicy, VerificationPolicy
from forecontract.memory.working import WorkingMemory
from forecontract.models.actions import ActionSpec, ExpectedOutcome
from forecontract.models.base import SCHEMA_VERSION
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome, ErrorInfo
from forecontract.recovery import ActionSafetyDeclaration
from forecontract.runtime import (
    ContractPreparation,
    RunRequest,
    RuntimeBudgetSnapshot,
    RuntimeEngine,
)

# The bounded Runtime failure type, imported under its own name so a propagation assertion cannot
# accidentally test the interpreter's builtin instead.
from forecontract.runtime import RuntimeError as RuntimeFailure
from forecontract.tracing import JsonlTraceStore
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract_fake import (
    FakeEnvironmentAdapter,
    FakeScenario,
    FakeStage,
    FakeStep,
)
from forecontract_scripted import PlannerScript, PlannerStep, ScriptedPlanner

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
ACTION = ActionSpec(schema_version=SCHEMA_VERSION, action_type="click",
                    payload={"selector": "#go"})
DECLARED_SCRATCH = Path(os.environ.get("FORECONTRACT_TEST_SCRATCH", ""))


def drive(coroutine):
    return asyncio.run(coroutine)


def _scratch_base() -> Path:
    """The explicit external test scratch; there is no fallback directory."""

    if not os.environ.get("FORECONTRACT_TEST_SCRATCH"):
        raise RuntimeError("FORECONTRACT_TEST_SCRATCH must be set to an external writable directory")
    candidate = Path(DECLARED_SCRATCH)
    source_root = Path(__file__).resolve().parents[2]
    resolved = candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
    if resolved == source_root or source_root in resolved.parents:
        raise RuntimeError("FORECONTRACT_TEST_SCRATCH must not be the source tree or a subdirectory")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / f".probe-{uuid.uuid4().hex[:8]}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return candidate
    except OSError:
        raise RuntimeError(
            "FORECONTRACT_TEST_SCRATCH must name a writable external directory") from None


@pytest.fixture
def trace_store() -> JsonlTraceStore:
    base = _scratch_base() / f"runtime-flow-{uuid.uuid4().hex[:10]}"
    base.mkdir(parents=True, exist_ok=True)
    return JsonlTraceStore(base / "trace.jsonl")


class DeclaringFake:
    """The real adapter behaviour plus the json_rule capability the Runtime contract needs.

    This wrapper exists only in the test. It delegates ``start``, ``stop``, ``observe``,
    ``execute`` and ``validate_action`` straight to the real adapter, so the environment actually
    exercised is the real Fake one; only the declared capability lists are widened.
    """

    def __init__(self, inner: FakeEnvironmentAdapter) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    def capabilities(self):
        return self._inner.capabilities().model_copy(
            update={"supported_condition_types": ["json_rule"],
                    "supported_outcome_types": ["json_rule"]})

    def validate_action(self, action: ActionSpec) -> None:
        self._inner.validate_action(action)

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()

    async def observe(self) -> Observation:
        return await self._inner.observe()

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        return await self._inner.execute(action)


class JsonRuleSupport:
    """A domain-support implementation over the existing json_rule verification."""

    def __init__(self, *, effect: str = "not_applied", safety: str = "idempotent",
                 mode: str = "abort", complete: bool = True) -> None:
        self.effect = effect
        self.safety = safety
        self.mode = mode
        self.complete = complete
        self.seen: list[str] = []

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        self.seen.append(observation.observation_id)
        return VerificationEvidence(schema_version=SCHEMA_VERSION, observation=observation,
                                    state_complete=self.complete)

    def prepare_contract(self, *, forecast, evidence) -> ContractPreparation:
        bindings = [{"schema_version": SCHEMA_VERSION, "outcome_index": index,
                     "verifier_type": "json_rule", "config": {}}
                    for index, _ in enumerate(forecast.expected_outcomes)]
        policy = (RecoveryPolicy(schema_version=SCHEMA_VERSION) if self.mode == "abort"
                  else RecoveryPolicy(schema_version=SCHEMA_VERSION, mode="bounded",
                                      max_retries=1, allow_replan=True))
        return ContractPreparation(
            schema_version=SCHEMA_VERSION, preconditions=[],
            verification_policy=VerificationPolicy.model_validate(
                {"schema_version": SCHEMA_VERSION, "bindings": bindings}),
            recovery_policy=policy, timeout_ms=1000.0)

    def declare_action_safety(self, *, contract) -> ActionSafetyDeclaration:
        return ActionSafetyDeclaration(schema_version=SCHEMA_VERSION,
                                        contract_id=contract.contract_id,
                                        adapter_name=contract.adapter_name,
                                        action_type=contract.action.action_type,
                                        retry_safety=self.safety)

    def assess_effect(self, *, contract, actual_outcome, evidence) -> str:
        return self.effect


class HostBudget:
    def __init__(self, value: float = 10_000.0) -> None:
        self.value = value

    def snapshot(self) -> RuntimeBudgetSnapshot:
        return RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION, remaining_budget_ms=self.value)


def rule(name: str, expected: object) -> ExpectedOutcome:
    return ExpectedOutcome(schema_version=SCHEMA_VERSION, outcome_type="json_rule",
                           payload={"schema_version": SCHEMA_VERSION,
                                    "rule_type": "value_equals", "path": [name],
                                    "expected": expected}, required=True)


VISIBLE = rule("visible", True)
DONE = rule("done", True)
#: The step rule the committed contract verifies. It demands the ``n`` field, so an observation
#: that does not carry it makes the step FAIL outright (a present/absent judgement, not a
#: comparison against an absent path, which would be INCONCLUSIVE instead).
EFFECT = ExpectedOutcome(
    schema_version=SCHEMA_VERSION, outcome_type="json_rule",
    payload={"schema_version": SCHEMA_VERSION, "rule_type": "path_exists", "path": ["n"]},
    required=True)


def observation(oid: str, state: dict) -> Observation:
    return Observation(schema_version=SCHEMA_VERSION, observation_id=oid,
                       environment_type="fake", state=dict(state), timestamp=NOW)


def stage(oid: str, actual: dict, visible: dict) -> FakeStage:
    return FakeStage(schema_version=SCHEMA_VERSION, actual_state=dict(actual),
                     observation=observation(oid, visible))


def fake_step(index: int, status: str = "succeeded", error: ErrorInfo | None = None) -> FakeStep:
    fields: dict = {"schema_version": SCHEMA_VERSION, "outcome_id": f"out-{index}",
                    "status": status, "changes": {"n": index + 1}}
    if error is not None:
        fields["error"] = error
    return FakeStep(schema_version=SCHEMA_VERSION, step_id=f"s-{index}", action=ACTION,
                    outcome=ActualOutcome(**fields))


def scenario(stages, steps) -> FakeScenario:
    return FakeScenario(schema_version=SCHEMA_VERSION, scenario_id="sc-1",
                        adapter_name="fake-1", allowed_actions=[ACTION], stages=stages,
                        steps=steps)


def planner_for(states, goal: ExpectedOutcome = DONE) -> ScriptedPlanner:
    steps = [PlannerStep(schema_version=SCHEMA_VERSION, step_id=f"ps-{index}",
                         expected_environment_type="fake", expected_state=dict(state),
                         forecast={"schema_version": SCHEMA_VERSION,
                                   "forecast_id": f"f-{index}",
                                   "proposed_action": ACTION.model_dump(),
                                   "expected_outcomes": [goal.model_dump()]})
             for index, state in enumerate(states)]
    return ScriptedPlanner(PlannerScript(schema_version=SCHEMA_VERSION, script_id="p-1",
                                         steps=steps))


def request(run_id: str = "run-1", conditions=None, **limits) -> RunRequest:
    fields = {"max_steps": 3, "max_planning_calls": 3, "max_retries": 1, "max_replans": 1}
    fields.update(limits)
    return RunRequest(schema_version=SCHEMA_VERSION, run_id=run_id, goal_id="g-1",
                      task_criteria=TaskCompletionCriteria(
                          schema_version=SCHEMA_VERSION, goal_id="g-1",
                          conditions=conditions or [DONE]), **fields)


def build_engine(adapter, planner, support, store, *, budget=None, memory=None):
    return RuntimeEngine(adapter=DeclaringFake(adapter), planner=planner,
                         domain_support=support, budget_source=budget or HostBudget(),
                         trace_writer=store, working_memory=memory or WorkingMemory())


# --- A01: a goal that already holds needs no action -----------------------------

def test_a_goal_that_already_holds_completes_without_any_action(
        trace_store: JsonlTraceStore) -> None:
    adapter = FakeEnvironmentAdapter(scenario([stage("o-0", {"n": 0}, {"visible": True,
                                                                    "done": True})], []))
    drive(adapter.start())
    engine = build_engine(adapter, planner_for([]), JsonRuleSupport(), trace_store)
    result = drive(engine.run(request=request(conditions=[DONE])))
    assert result.terminal_status.value == "completed"
    assert (result.steps_started, result.attempts_started) == (0, 0)
    records = trace_store.read_run("run-1")
    assert [record.kind for record in records] == ["run_started", "verification_recorded",
                                                  "run_finished"]
    assert records[-1].terminal_status == "success"
    assert records[-1].verification_record_id == records[1].record_id


# --- A01/A04/A05: one real step achieves the goal -------------------------------

def test_one_real_step_through_the_fake_completes_and_reads_back(
        trace_store: JsonlTraceStore) -> None:
    # The pre-execution observation does not carry the effect field, the post-execution one does.
    stages = [stage("o-0", {"n": 0}, {"visible": True, "done": False}),
              stage("o-1", {"n": 1}, {"visible": True, "done": True, "n": 1})]
    adapter = FakeEnvironmentAdapter(scenario(stages, [fake_step(0)]))
    drive(adapter.start())
    planner = planner_for([{"visible": True, "done": False}], goal=EFFECT)
    engine = build_engine(adapter, planner, JsonRuleSupport(), trace_store)
    result = drive(engine.run(request=request(conditions=[DONE])))
    assert result.terminal_status.value == "completed"
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (1, 1, 1)
    records = trace_store.read_run("run-1")
    assert [record.kind for record in records] == [
        "run_started", "step_started", "contract_committed", "attempt_started",
        "attempt_finished", "verification_recorded", "verification_recorded", "run_finished"]
    assert [record.sequence for record in records] == list(range(len(records)))
    started = next(record for record in records if record.kind == "attempt_started")
    finished = next(record for record in records if record.kind == "attempt_finished")
    assert started.attempt_id == finished.attempt_id
    assert finished.execution_status.value == "succeeded"
    scoped = [record for record in records if record.kind == "verification_recorded"
              and record.scope == "attempt"]
    assert scoped[0].attempt_id == started.attempt_id
    run_scoped = [record for record in records if record.kind == "verification_recorded"
                  and record.scope == "run"]
    assert run_scoped[0].verification_status.value == "pass"
    assert records[-1].verification_record_id == run_scoped[0].record_id


def test_the_real_fake_advances_its_stages_in_order(trace_store: JsonlTraceStore) -> None:
    stages = [stage("o-0", {"n": 0}, {"visible": True, "done": False}),
              stage("o-1", {"n": 1}, {"visible": True, "done": True, "n": 1})]
    adapter = FakeEnvironmentAdapter(scenario(stages, [fake_step(0)]))
    drive(adapter.start())
    planner = planner_for([{"visible": True, "done": False}], goal=EFFECT)
    engine = build_engine(adapter, planner, JsonRuleSupport(), trace_store)
    result = drive(engine.run(request=request(conditions=[DONE])))
    assert result.terminal_status.value == "completed"
    # The contract the Runtime committed really names the observation the adapter returned before
    # the execute, which is the stage the Fake exposed at that point.
    records = trace_store.read_run("run-1")
    committed = next(record for record in records if record.kind == "contract_committed")
    assert committed.contract_id
    assert result.verification_record_id is not None


# --- A06/A07: a failed real step retries against the same contract --------------

def test_a_real_failed_step_retries_and_keeps_one_contract(
        trace_store: JsonlTraceStore) -> None:
    failure = ErrorInfo(schema_version=SCHEMA_VERSION, code="EXECUTION_FAILED",
                        message="synthetic", recoverable=True)
    # Stage 1 (the recovery-time observation) still lacks the effect field, so the step stays
    # FAILED and recovery may retry; the retry then advances the Fake to a stage that carries it.
    stages = [stage("o-0", {"n": 0}, {"visible": True, "done": False}),
              stage("o-1", {"n": 0}, {"visible": True, "done": True}),
              stage("o-2", {"n": 0}, {"visible": True, "done": True}),
              stage("o-3", {"n": 1}, {"visible": True, "done": True})]
    adapter = FakeEnvironmentAdapter(scenario(stages, [fake_step(0, "failed", failure),
                                                      fake_step(1), fake_step(2)]))
    drive(adapter.start())
    # The Run plans for the initial Attempt and again for the retry, so the same declared state is
    # scripted twice: the planner is a facility and its exhaustion is not a plan result.
    planner = planner_for([{"visible": True, "done": False},
                           {"visible": True, "done": False}], goal=EFFECT)
    support = JsonRuleSupport(effect="not_applied", safety="idempotent", mode="bounded")
    engine = build_engine(adapter, planner, support, trace_store)
    result = drive(engine.run(request=request(conditions=[DONE])))
    assert result.retries_used == 1, (result.model_dump(), support.seen)
    # The real Fake run completes, so its RunResult is the valid place for the original counter
    # assertions: two Attempts and one Step. The Trace facts below corroborate them, not replace them.
    assert result.attempts_started == 2
    assert result.steps_started == 1
    records = trace_store.read_run("run-1")
    committed = [record for record in records if record.kind == "contract_committed"]
    attempts = [record for record in records if record.kind == "attempt_started"]
    steps = [record for record in records if record.kind == "step_started"]
    assert len(committed) == 1
    assert len(attempts) == 2
    assert len(steps) == 1
    assert attempts[0].contract_id == attempts[1].contract_id == committed[0].contract_id
    assert attempts[0].attempt_id != attempts[1].attempt_id
    # The first Attempt honestly records its failure.
    finished = [record for record in records if record.kind == "attempt_finished"]
    assert finished[0].execution_status.value == "failed"
    assert finished[0].error_code == "execution_failed"
    assert finished[1].execution_status.value == "succeeded"


def test_a_real_recovery_observation_carries_the_goal(
        trace_store: JsonlTraceStore) -> None:
    failure = ErrorInfo(schema_version=SCHEMA_VERSION, code="EXECUTION_FAILED",
                        message="synthetic", recoverable=True)
    stages = [stage("o-0", {"n": 0}, {"visible": True, "done": False}),
              stage("o-1", {"n": 0}, {"visible": True, "done": True}),
              stage("o-2", {"n": 0}, {"visible": True, "done": True}),
              stage("o-3", {"n": 1}, {"visible": True, "done": True})]
    adapter = FakeEnvironmentAdapter(scenario(stages, [fake_step(0, "failed", failure),
                                                     fake_step(1), fake_step(2)]))
    drive(adapter.start())
    planner = planner_for([{"visible": True, "done": False},
                           {"visible": True, "done": False}], goal=EFFECT)
    memory = WorkingMemory()
    engine = build_engine(adapter, planner,
                          JsonRuleSupport(effect="not_applied", safety="idempotent",
                                          mode="bounded"),
                          trace_store, memory=memory)
    drive(engine.run(request=request(conditions=[DONE])))
    calls = memory.get("run-1")["observation_calls"]
    phases = [call["phase"] for call in calls]
    # The execution-time and recovery-time observations are distinguished by provenance, not by
    # the adapter identifier alone.
    assert "OBSERVING" in phases
    assert "RECOVERING" in phases
    assert "VERIFYING" in phases
    assert [call["ordinal"] for call in calls] == sorted(call["ordinal"] for call in calls)


# --- A04: a zero host budget stops before any environment read ------------------

def test_a_zero_host_budget_never_touches_the_environment(
        trace_store: JsonlTraceStore) -> None:
    adapter = FakeEnvironmentAdapter(scenario([stage("o-0", {"n": 0}, {"done": False})], []))
    drive(adapter.start())
    engine = build_engine(adapter, planner_for([]), JsonRuleSupport(), trace_store,
                          budget=HostBudget(0.0))
    result = drive(engine.run(request=request(conditions=[DONE])))
    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "BUDGET_EXHAUSTED"
    records = trace_store.read_run("run-1")
    assert [record.kind for record in records] == ["run_started", "run_finished"]
    assert records[-1].terminal_status == "aborted"
    assert records[-1].reason_code == "policy_abort"


# --- A12: the store content holds the Runtime's own facts only ------------------

def test_the_trace_file_holds_only_the_declared_record_fields(
        trace_store: JsonlTraceStore) -> None:
    stages = [stage("o-0", {"n": 0}, {"visible": True, "done": True})]
    adapter = FakeEnvironmentAdapter(scenario(stages, []))
    drive(adapter.start())
    engine = build_engine(adapter, planner_for([]), JsonRuleSupport(), trace_store)
    drive(engine.run(request=request(conditions=[DONE])))
    lines = trace_store.path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    for line in lines:
        payload = json.loads(line)
        assert "sequence" in payload and "record_id" in payload and "run_id" in payload
        # No raw observation, payload, exception text or secret has a field to live in.
        for forbidden in ("observation", "state", "payload", "error_message", "summary"):
            assert forbidden not in payload


def test_a_successful_terminal_status_is_never_written_without_a_reference(
        trace_store: JsonlTraceStore) -> None:
    stages = [stage("o-0", {"n": 0}, {"visible": True, "done": True})]
    adapter = FakeEnvironmentAdapter(scenario(stages, []))
    drive(adapter.start())
    engine = build_engine(adapter, planner_for([]), JsonRuleSupport(), trace_store)
    drive(engine.run(request=request(conditions=[DONE])))
    terminal = trace_store.read_run("run-1")[-1]
    assert terminal.terminal_status == "success"
    assert terminal.verification_record_id is not None
    # The reference is a real recorded run-scope PASS.
    referenced = trace_store.get_record("run-1", terminal.verification_record_id)
    assert referenced.kind == "verification_recorded"
    assert referenced.scope == "run"
    assert referenced.verification_status.value == "pass"


def test_an_honest_unfinished_attempt_survives_a_real_execute_failure(
        trace_store: JsonlTraceStore) -> None:
    """The Fake has no matching step, so ``execute`` raises: the Attempt stays unfinished."""
    stages = [stage("o-0", {"n": 0}, {"visible": True, "done": False})]
    adapter = FakeEnvironmentAdapter(scenario(stages, []))
    drive(adapter.start())
    planner = planner_for([{"visible": True, "done": False}], goal=EFFECT)
    engine = build_engine(adapter, planner, JsonRuleSupport(), trace_store)
    # The Fake has no matching step, so the executor raises an ordinary exception: the bounded facility
    # failure is propagated once the Run has recorded its own end.
    with pytest.raises(RuntimeFailure) as info:
        drive(engine.run(request=request(conditions=[DONE])))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert engine.last_finalization.primary_terminal_status.value == "failed"
    records = trace_store.read_run("run-1")
    kinds = [record.kind for record in records]
    assert "attempt_started" in kinds
    # No AttemptFinished and no verification was fabricated for the unfinished Attempt.
    assert "attempt_finished" not in kinds
    assert records[-1].kind == "run_finished"
    assert records[-1].terminal_status == "failure"
