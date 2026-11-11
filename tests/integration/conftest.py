"""Shared, offline-only fixtures for the P2-T08 integration suite.

Everything here is built from the real public components: the real Fake environment, the real ScriptedPlanner,
the real Contract Builder, the real StepVerifier and TaskCompletionOracle, the real RecoveryEngine, the real
Runtime and the real JSONL Trace and Memory stores. Nothing in this module is a stand-in for a product
component, and no unit-test helper is imported: the suite drives the components the way a host would.

Two host responsibilities are made explicit here because the product deliberately leaves them to the host.
The first is the domain binding: the Runtime calls ``to_evidence``, ``prepare_contract``,
``declare_action_safety`` and ``assess_effect`` on a domain support object, and this suite supplies them
rather than expecting a default. The second is the capability declaration: the Fake declares its own
supported outcome types, so a thin proxy adds the ``json_rule`` binding the host is asking for while every
lifecycle, observation, validation and execution call still goes to the real Fake. The proxy never reads the
Fake's private state and never decides task completion.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The suite runs against the checkout's own sources so it exercises the code under review. This is an import
#: path for the test process only; it changes no global collection rule and no root configuration.
for part in ("packages/core/src", "packages/adapters/fake/src", "packages/planners/scripted/src"):
    candidate = str(REPO_ROOT / part)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from forecontract.contracts import (
    ContractBuilder,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.memory import (
    JsonlEpisodicMemory,
    MemoryItem,
    TraceSourceResolver,
    WorkingMemory,
)
from forecontract.models import (
    ActionSpec,
    ActualOutcome,
    ErrorInfo,
    ExpectedOutcome,
    Forecast,
    Observation,
    Precondition,
)
from forecontract.recovery import ActionSafetyDeclaration
from forecontract.runtime import (
    ContractPreparation,
    MonotonicBudgetSource,
    RunRequest,
    RuntimeEngine,
    RuntimeTimingPolicy,
)
from forecontract.tracing import JsonlTraceStore
from forecontract.verification import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract_fake import (
    FakeEnvironmentAdapter,
    FakeScenario,
    FakeStage,
    FakeStep,
)
from forecontract_scripted import (
    PlannerScript,
    PlannerStep,
    ScriptedPlanner,
)

SCHEMA = "0.1"

#: A fixed timestamp. Identifiers are allowed to vary between runs, but the facts this suite asserts are
#: fixed and declared here and in each test, never read back from the run under test.
FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

#: The synthetic marker C12 requires to be absent from every artefact. It is deliberately not a secret.
SENSITIVE_MARKER = "P2T08-SYNTHETIC-SENSITIVE-MARKER"

#: The observation-state flag the synthetic task is about.
FLAG = "done"


def json_rule_condition(flag: str = FLAG, value: object = True) -> ExpectedOutcome:
    """A ``json_rule`` completion condition evaluated against an observation's own state."""
    return ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="json_rule",
        payload={"schema_version": SCHEMA, "rule_type": "value_equals",
                 "path": [flag], "expected": value})


def path_exists_condition(flag: str = FLAG) -> ExpectedOutcome:
    """A ``path_exists`` condition, used for the incomplete-evidence case."""
    return ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="json_rule",
        payload={"schema_version": SCHEMA, "rule_type": "path_exists", "path": [flag]})


def action(action_type: str = "mark_done", payload: dict | None = None) -> ActionSpec:
    return ActionSpec(schema_version=SCHEMA, action_type=action_type, payload=payload or {})


def observation(observation_id: str, state: dict) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="fake", state=state, timestamp=FIXED_TIME)


def succeeded(outcome_id: str, state_after: dict) -> ActualOutcome:
    return ActualOutcome(schema_version=SCHEMA, outcome_id=outcome_id, status="succeeded",
                         state_after=state_after)


def failed(outcome_id: str, *, message: str = "the action did not apply") -> ActualOutcome:
    """A failed outcome with the error code the Fake's own scenario boundary permits for ``failed``."""
    return ActualOutcome(schema_version=SCHEMA, outcome_id=outcome_id, status="failed",
                         error=ErrorInfo(schema_version=SCHEMA, code="EXECUTION_FAILED",
                                         message=message))


def unknown(outcome_id: str, *, message: str = "the response was lost") -> ActualOutcome:
    """An unknown outcome with the error code the Fake's boundary permits for ``unknown``."""
    return ActualOutcome(schema_version=SCHEMA, outcome_id=outcome_id, status="unknown",
                         error=ErrorInfo(schema_version=SCHEMA, code="RESPONSE_LOST",
                                         message=message))


def outcome_status(outcome_id: str, status: str) -> ActualOutcome:
    return ActualOutcome(schema_version=SCHEMA, outcome_id=outcome_id, status=status)


class HostDomain:
    """The explicit host domain binding the Runtime requires, with no hidden defaults.

    The declared verifier type, the recovery policy and the retry-safety verdict are all supplied by the test
    that builds the run, so every case states its own contract instead of inheriting one.
    """

    def __init__(self, *, verifier_type: str = "json_rule", mode: str = "bounded",
                 max_retries: int | None = 1, allow_replan: bool | None = False,
                 retry_safety: str = "idempotent", effect: str | None = None,
                 preconditions: list[Precondition] | None = None,
                 timeout_ms: float = 5000.0,
                 state_complete: bool = True) -> None:
        self.verifier_type = verifier_type
        self.mode = mode
        self.max_retries = max_retries
        self.allow_replan = allow_replan
        self.retry_safety = retry_safety
        # ``effect`` pins one verdict for a case that deliberately exercises a single recovery branch. Left
        # unset, the binding answers from the evidence it was handed rather than from a value declared before
        # the run.
        self.fixed_effect = effect
        self.preconditions = list(preconditions or [])
        self.timeout_ms = timeout_ms
        self.state_complete = state_complete
        self.assess_calls: list[tuple[str, str]] = []

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        return VerificationEvidence(schema_version=SCHEMA, observation=observation,
                                    state_complete=self.state_complete)

    def prepare_contract(self, *, forecast: Forecast,
                         evidence: VerificationEvidence) -> ContractPreparation:
        return ContractPreparation(
            schema_version=SCHEMA,
            preconditions=[item.model_copy(deep=True) for item in self.preconditions],
            verification_policy=VerificationPolicy(
                schema_version=SCHEMA,
                bindings=[VerificationBinding(schema_version=SCHEMA, outcome_index=index,
                                              verifier_type=self.verifier_type)
                          for index in range(len(forecast.expected_outcomes))]),
            recovery_policy=RecoveryPolicy(schema_version=SCHEMA, mode=self.mode,
                                           max_retries=self.max_retries,
                                           allow_replan=self.allow_replan),
            timeout_ms=self.timeout_ms)

    def declare_action_safety(self, *, contract: object) -> ActionSafetyDeclaration:
        return ActionSafetyDeclaration(
            schema_version=SCHEMA, contract_id=contract.contract_id,
            adapter_name=contract.adapter_name, action_type=contract.action.action_type,
            retry_safety=self.retry_safety)

    def assess_effect(self, *, contract: object, actual_outcome: ActualOutcome,
                      evidence: VerificationEvidence) -> str:
        """Report whether the action's effect applied, judged from the evidence really received.

        The order is declared and each step states only what it knows:

        * a case may pin the verdict, which is how a single recovery branch is exercised on purpose;
        * without complete evidence there is nothing to judge against, so the answer is ``unknown``;
        * the receipt's own ``state_after``, when the adapter reported one, is the environment's word and
          decides the answer on its own;
        * otherwise a succeeded execution is evidence that the effect applied and a failed or unknown one is
          evidence that it did not - a failed or unknown receipt never claims the effect applied.
        """
        self.assess_calls.append((contract.contract_id, actual_outcome.outcome_id))
        if self.fixed_effect is not None:
            return self.fixed_effect
        if not evidence.state_complete:
            return "unknown"
        reported = actual_outcome.state_after
        if isinstance(reported, dict) and FLAG in reported:
            return "applied" if reported.get(FLAG) is True else "not_applied"
        # The latest valid observation is decisive: if it carries the effect field, that field - not the
        # receipt's status - decides. A failed or unknown receipt therefore cannot downgrade evidence that
        # already shows the effect applied.
        observed = evidence.observation.state
        if FLAG not in observed:
            # Nothing observed carries the effect field, so nothing here can say whether it applied.
            return "unknown"
        return "applied" if observed.get(FLAG) is True else "not_applied"


class DeclaredAdapter:
    """A thin proxy over the real Fake that explicitly declares the host's ``json_rule`` binding.

    Lifecycle, observation, action validation and execution are delegated to the real adapter unchanged. Only
    the capability declaration is widened, and only to the outcome types the caller names, because the
    product requires the declaration to match the domain support actually offered.

    ``outcomes`` optionally records the real outcomes the adapter returned, so a test can prove from the
    outside that an execution really happened without reading the Fake's private state.
    """

    def __init__(self, inner: FakeEnvironmentAdapter, *,
                 outcome_types: tuple[str, ...] = ("json_rule",)) -> None:
        self._inner = inner
        self._outcome_types = tuple(outcome_types)
        self.outcomes: list[ActualOutcome] = []
        self.actions: list[ActionSpec] = []
        self.observations: list[Observation] = []
        self.starts = 0
        self.stops = 0

    @property
    def name(self) -> str:
        return self._inner.capabilities().adapter_name

    def capabilities(self):
        real = self._inner.capabilities()
        declared = list(self._outcome_types)
        return real.model_copy(update={"supported_outcome_types": declared})

    def validate_action(self, action_spec: ActionSpec) -> None:
        self._inner.validate_action(action_spec)

    async def start(self) -> None:
        self.starts += 1
        await self._inner.start()

    async def stop(self) -> None:
        self.stops += 1
        await self._inner.stop()

    async def observe(self) -> Observation:
        seen = await self._inner.observe()
        self.observations.append(seen)
        return seen

    async def execute(self, action_spec: ActionSpec) -> ActualOutcome:
        self.actions.append(action_spec)
        outcome = await self._inner.execute(action_spec)
        self.outcomes.append(outcome)
        return outcome


def no_step_scenario(*, action_spec: ActionSpec, state: dict,
                     scenario_id: str = "scenario") -> FakeScenario:
    """A scenario with no scripted step, so the single observation is the only stage.

    The Fake requires exactly one more stage than steps, so a case that must never execute anything declares
    one stage and one allowed action and nothing else.
    """
    return FakeScenario(
        schema_version=SCHEMA, scenario_id=scenario_id, adapter_name="fake",
        allowed_actions=[action_spec],
        stages=[FakeStage(schema_version=SCHEMA, actual_state=state,
                          observation=observation("obs-only", state))],
        steps=[])


def fake_scenario(*, action_spec: ActionSpec, start_state: dict, end_state: dict,
                  outcome: ActualOutcome | None = None,
                  scripted: list[tuple[ActionSpec, ActualOutcome, dict]] | None = None,
                  scenario_id: str = "scenario") -> FakeScenario:
    """A synthetic Fake scenario.

    The Fake requires exactly one more stage than steps, so the declared observations are derived from that
    rule rather than guessed: a scenario with N scripted steps carries the initial observation plus one
    observation after each step. ``scripted`` lets a case declare several real actions; a single action is
    the common case and is expressed by ``action_spec`` plus ``outcome``.
    """
    if scripted is None:
        scripted = [(action_spec, outcome if outcome is not None
                     else succeeded("fake-outcome-1", end_state), end_state)]

    steps: list[FakeStep] = []
    stages: list[FakeStage] = []
    state = start_state
    stages.append(FakeStage(schema_version=SCHEMA, actual_state=state,
                            observation=observation("obs-before", state)))
    for index, (step_action, step_outcome, after_state) in enumerate(scripted, start=1):
        steps.append(FakeStep(schema_version=SCHEMA, step_id=f"fake-step-{index}",
                              action=step_action, outcome=step_outcome))
        state = after_state
        stages.append(FakeStage(schema_version=SCHEMA, actual_state=state,
                                observation=observation(f"obs-after-{index}", state)))
    return FakeScenario(
        schema_version=SCHEMA, scenario_id=scenario_id, adapter_name="fake",
        allowed_actions=[step_action for step_action, _, _ in scripted],
        stages=stages, steps=steps)


def forecast_for(action_spec: ActionSpec, *, forecast_id: str = "forecast-1",
                 conditions: list[ExpectedOutcome] | None = None,
                 hint: bool | None = True) -> Forecast:
    return Forecast(schema_version=SCHEMA, forecast_id=forecast_id, proposed_action=action_spec,
                    expected_outcomes=list(conditions or [json_rule_condition()]),
                    task_completion_hint=hint)


def planner_script(steps: list[tuple[str, dict, Forecast]], *,
                   script_id: str = "script") -> PlannerScript:
    return PlannerScript(
        schema_version=SCHEMA, script_id=script_id,
        steps=[PlannerStep(schema_version=SCHEMA, step_id=step_id,
                           expected_environment_type="fake", expected_state=state, forecast=plan)
               for step_id, state, plan in steps])


def request_for(*, run_id: str = "run-1", goal_id: str = "goal-1",
                conditions: list[ExpectedOutcome] | None = None, max_steps: int = 4,
                max_planning_calls: int = 3, max_retries: int = 2,
                max_replans: int = 1) -> RunRequest:
    return RunRequest(schema_version=SCHEMA, run_id=run_id, goal_id=goal_id,
                      task_criteria=TaskCompletionCriteria(
                          schema_version=SCHEMA, goal_id=goal_id,
                          conditions=list(conditions or [json_rule_condition()])),
                      max_steps=max_steps, max_planning_calls=max_planning_calls,
                      max_retries=max_retries, max_replans=max_replans)


def build_engine(*, tmp_path: Path, adapter, planner, domain: HostDomain | None = None,
                 budget_ms: float = 30000.0, run_timeout_ms: float | None = 20000.0,
                 cleanup_timeout_ms: float = 2000.0, finalize_timeout_ms: float = 2000.0,
                 trace_name: str = "trace.jsonl", memory: object | None = None,
                 monotonic_clock: object | None = None,
                 trace_writer: object | None = None,
                 step_verifier: object | None = None,
                 recovery_engine: object | None = None,
                 budget_source: object | None = None):
    """Assemble the real closed loop. Returns the engine, its trace store and its working memory.

    Two of the four already-verified components are forwarded as the host may supply them: the
    ``RuntimeEngine``'s public constructor already accepts ``step_verifier`` and ``recovery_engine``,
    so this helper only passes the caller's object through unchanged. Left unset, the constructor's
    own default - the real ``StepVerifier`` and the real ``RecoveryEngine`` - is used.
    """
    trace = trace_writer if trace_writer is not None else JsonlTraceStore(tmp_path / trace_name)
    working = memory if memory is not None else WorkingMemory()
    engine = RuntimeEngine(
        adapter=adapter, planner=planner, domain_support=domain or HostDomain(),
        budget_source=budget_source if budget_source is not None else
        MonotonicBudgetSource(total_budget_ms=budget_ms,
                              monotonic_clock=monotonic_clock),
        trace_writer=trace, working_memory=working, builder=ContractBuilder(),
        step_verifier=step_verifier, recovery_engine=recovery_engine,
        timing_policy=RuntimeTimingPolicy(schema_version=SCHEMA, run_timeout_ms=run_timeout_ms,
                                          cleanup_timeout_ms=cleanup_timeout_ms,
                                          finalize_timeout_ms=finalize_timeout_ms),
        monotonic_clock=monotonic_clock)
    return engine, trace, working


def read_records(trace: JsonlTraceStore, run_id: str):
    """Read a run's records back from the JSONL file, not from any in-memory recorder."""
    return trace.read_run(run_id)


def kinds(records: object) -> list[str]:
    return [record.kind for record in records]


def terminal_records(records: object) -> list[object]:
    return [record for record in records if record.kind == "run_finished"]


def project_run(trace: JsonlTraceStore, tmp_path: Path, run_id: str,
                *, source_record_id: str | None = None,
                memory_id: str = "memory-1", result: str | None = None):
    """Host-side Episodic projection from a confirmed trace source.

    The host resolves the source first, then writes the item; the product is never asked to write Episodic
    Memory itself, and a source that cannot be resolved produces no item.
    """
    resolver = TraceSourceResolver(trace)
    if source_record_id is None:
        records = trace.read_run(run_id)
        finished = terminal_records(records)
        if not finished:
            return None, resolver
        source_record_id = finished[0].record_id
    resolved = resolver.resolve(run_id, source_record_id)
    episodic = JsonlEpisodicMemory(tmp_path / "episodic.jsonl", source_resolver=resolver)
    episodic.append(MemoryItem(schema_version=SCHEMA, memory_id=memory_id, run_id=run_id,
                               source_level=resolved.source_level,
                               source_record_id=resolved.record_id,
                               result=result if result is not None else resolved.result,
                               created_at=FIXED_TIME))
    return episodic, resolver


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL artefact as plain lines, so a test can prove what the files really contain."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """An external temporary directory. Every artefact this suite writes lives under it."""
    return tmp_path


@pytest.fixture()
def done_action() -> ActionSpec:
    return action()


@pytest.fixture()
def closed_loop(workdir: Path, done_action: ActionSpec):
    """One real success run: initial state unmet, one real action, oracle PASS."""
    scenario = fake_scenario(action_spec=done_action, start_state={FLAG: False},
                             end_state={FLAG: True})
    domain = HostDomain()
    engine, trace, memory = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done_action))])),
        domain=domain)
    return engine, trace, memory, domain
