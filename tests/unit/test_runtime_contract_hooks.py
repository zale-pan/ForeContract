"""Unit acceptance of the optional Core contract-hooks factory boundary.

The tests drive the real Runtime with the real Fake adapter, ScriptedPlanner, Contract Builder,
StepVerifier, RecoveryEngine and trace store; only the injected hooks are test doubles, because
the factory boundary is exactly what this task adds. They assert the two creation sites, the
fresh evidence, the default compatibility, the isolation and error mapping, the cancellation
semantics and the fixed Run deadline.
"""

from __future__ import annotations

import asyncio
import datetime
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

REPO = Path(__file__).resolve().parents[2]
INTEGRATION = REPO / "tests" / "integration"
for candidate in (REPO / "packages" / "core" / "src",
                  REPO / "packages" / "adapters" / "fake" / "src",
                  REPO / "packages" / "planners" / "scripted" / "src",
                  INTEGRATION):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from conftest import (
    SCHEMA,
    DeclaredAdapter,
    HostDomain,
    action,
    fake_scenario,
    forecast_for,
    observation,
    planner_script,
    request_for,
    unknown,
)
from forecontract.contracts import ContractBuilder, ContractValidationHooks
from forecontract.memory.working import WorkingMemory
from forecontract.models.actions import ExpectedOutcome, Precondition
from forecontract.models.base import SCHEMA_VERSION
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.runtime import (
    MonotonicBudgetSource,
    RuntimeEngine,
    RuntimeTimingPolicy,
)
from forecontract.runtime import RuntimeError as RuntimeFailure
from forecontract.runtime.protocols import ContractHooksFactory
from forecontract.tracing import JsonlTraceStore
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import VerificationEvidence
from forecontract_fake import FakeEnvironmentAdapter
from forecontract_scripted import ScriptedPlanner

DONE = "done"
PRECONDITION = Precondition(
    schema_version=SCHEMA, condition_type="json_rule",
    payload={"schema_version": SCHEMA, "rule_type": "value_equals",
             "path": [DONE], "expected": True},
    required=True)


class WideAdapter(DeclaredAdapter):
    """The real Fake adapter with the host's json_rule condition binding also declared."""

    def capabilities(self):
        real = super().capabilities()
        return real.model_copy(update={"supported_condition_types": ["json_rule"],
                                       "supported_outcome_types": ["json_rule"]})


def make_scenario(*, start: bool = False):
    return fake_scenario(action_spec=action(), start_state={DONE: start},
                         end_state={DONE: True})


def make_script():
    return planner_script([("plan-1", {DONE: False}, forecast_for(action()))])


class Clock:
    """A declared monotonic timeline the test may advance deliberately."""

    def __init__(self) -> None:
        self.value = 0

    def now_ns(self) -> int:
        return self.value


class GoodHooks:
    """The four required synchronized methods, returning a declared verdict."""

    def __init__(self, factory: Factory) -> None:
        self._factory = factory

    def validate_precondition(self, condition: object) -> None:
        return None

    def validate_expected_outcome(self, outcome: object) -> None:
        return None

    def validate_verification_binding(self, binding: object, outcome: object) -> None:
        return None

    def check_precondition(self, condition: object, observation: object) -> VerificationStatus:
        self._factory.checks += 1
        if self._factory.on_check is not None:
            self._factory.on_check(condition, observation, self._factory)
        verdict = self._factory.verdict
        if isinstance(verdict, BaseException):
            raise verdict
        if callable(verdict):
            return verdict()
        return verdict


class Factory:
    """A hooks factory that records every evidence object it is handed."""

    def __init__(self, *, hooks_type=GoodHooks, verdict=VerificationStatus.PASS,
                 on_check=None, on_create=None, create_impl=None) -> None:
        self.calls: list[object] = []
        self.checks = 0
        self.verdict = verdict
        self.on_check = on_check
        self.on_create = on_create
        self._hooks_type = hooks_type
        self._create_impl = create_impl

    def create(self, *, evidence):
        self.calls.append(evidence)
        if self.on_create is not None:
            self.on_create(self, evidence)
        if self._create_impl is not None:
            return self._create_impl(self, evidence)
        return self._hooks_type(self)


class MissingCreate:
    pass


def build(tmp_path: Path, *, factory=None, domain=None, clock=None, run_timeout_ms=20000.0,
          budget_ms=30000.0, scenario=None, script=None, criteria_validator=None,
          adapter=None, memory=None):
    trace = JsonlTraceStore(tmp_path / "trace.jsonl")
    engine = RuntimeEngine(
        adapter=adapter or WideAdapter(FakeEnvironmentAdapter(scenario or make_scenario())),
        planner=ScriptedPlanner(script or make_script()),
        domain_support=domain or HostDomain(),
        budget_source=MonotonicBudgetSource(total_budget_ms=budget_ms,
                                            monotonic_clock=clock),
        trace_writer=trace,
        working_memory=memory or WorkingMemory(),
        builder=ContractBuilder(),
        timing_policy=RuntimeTimingPolicy(
            schema_version=SCHEMA_VERSION, run_timeout_ms=run_timeout_ms,
            cleanup_timeout_ms=2000.0, finalize_timeout_ms=2000.0),
        monotonic_clock=clock,
        contract_hooks_factory=factory,
        task_criteria_validator=criteria_validator,
    )
    return engine, trace


def run(engine, request):
    return asyncio.run(engine.run(request=request))


def test_the_default_json_path_is_completely_unchanged(tmp_path: Path) -> None:
    engine, _trace = build(tmp_path)
    result = run(engine, request_for(max_steps=4))
    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    assert result.steps_started == 1
    assert result.attempts_started == 1


def test_the_factory_is_created_at_both_boundaries_with_fresh_evidence(tmp_path: Path) -> None:
    factory = Factory()
    engine, _trace = build(tmp_path, factory=factory)
    result = run(engine, request_for())
    assert result.terminal_status.value == "completed"
    assert len(factory.calls) == 2
    assert factory.calls[0] is not factory.calls[1]
    assert all(call.state_complete is True for call in factory.calls)


def test_a_changed_pre_execution_verdict_prevents_the_attempt(tmp_path: Path) -> None:
    factory = Factory()
    factory.verdict = VerificationStatus.PASS

    def on_check(condition, observation, current):
        current.verdict = VerificationStatus.FAIL if current.checks >= 2 else \
            VerificationStatus.PASS

    factory.on_check = on_check
    engine, trace = build(
        tmp_path, factory=factory,
        domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=1, max_replans=0))
    assert result.terminal_status.value == "aborted"
    assert result.steps_started == 1
    assert result.attempts_started == 0
    kinds = [record.kind for record in trace.read_run("run-1")]
    assert "attempt_started" not in kinds


@pytest.mark.parametrize("verdict", ["pass", 1, None, object()])
def test_a_wrong_hook_verdict_is_a_dependency_failure(tmp_path: Path, verdict: object) -> None:
    factory = Factory(verdict=verdict)
    engine, _trace = build(tmp_path, factory=factory,
                          domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "DEPENDENCY_FAILURE"
    assert result.attempts_started == 0


class AwaitableCheckHooks(GoodHooks):
    """Return an awaitable only from the pre-execution re-check, never from the build gate."""

    def check_precondition(self, condition: object, observation: object):
        self._factory.checks += 1
        if self._factory.checks >= 2:
            async def awaitable() -> VerificationStatus:
                return VerificationStatus.PASS

            return awaitable()
        return VerificationStatus.PASS


def test_an_awaitable_hook_verdict_is_refused(tmp_path: Path) -> None:
    factory = Factory(hooks_type=AwaitableCheckHooks)
    engine, _trace = build(tmp_path, factory=factory,
                           domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "DEPENDENCY_FAILURE"
    assert result.attempts_started == 0


def test_a_hook_that_mutates_its_observation_is_refused(tmp_path: Path) -> None:
    factory = Factory()

    def on_check(condition, observation, current):
        if current.checks >= 2:
            observation.state[DONE] = True

    factory.on_check = on_check
    engine, _trace = build(tmp_path, factory=factory,
                          domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert result.attempts_started == 0


def test_a_hook_that_mutates_its_condition_is_refused(tmp_path: Path) -> None:
    factory = Factory()

    def on_check(condition, observation, current):
        if current.checks >= 2:
            condition.payload["expected"] = True

    factory.on_check = on_check
    engine, _trace = build(tmp_path, factory=factory,
                          domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert result.attempts_started == 0


def test_an_ordinary_hook_exception_is_bounded(tmp_path: Path) -> None:
    factory = Factory(verdict=RuntimeFailure("DEPENDENCY_FAILURE", "validate", ("x",),
                                             "an ordinary hook failure"))
    engine, _trace = build(tmp_path, factory=factory,
                          domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "DEPENDENCY_FAILURE"
    assert result.attempts_started == 0


def test_a_missing_factory_method_is_a_dependency_failure(tmp_path: Path) -> None:
    engine, _trace = build(tmp_path, factory=MissingCreate())
    result = run(engine, request_for())
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "DEPENDENCY_FAILURE"
    assert result.attempts_started == 0


def test_an_awaitable_factory_return_is_refused(tmp_path: Path) -> None:
    async def create_impl(factory, evidence):
        return GoodHooks(factory)

    factory = Factory(create_impl=create_impl)
    engine, _trace = build(tmp_path, factory=factory)
    result = run(engine, request_for())
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "DEPENDENCY_FAILURE"
    assert result.attempts_started == 0


class MutatingFactory(Factory):
    """A factory that rewrites the detached evidence it was handed."""

    def __init__(self, mutation) -> None:
        super().__init__()
        self._mutation = mutation

    def create(self, *, evidence):
        self.calls.append(evidence)
        self._mutation(evidence)
        return GoodHooks(self)


def test_a_factory_that_mutates_the_evidence_is_refused(tmp_path: Path) -> None:
    def mutation(evidence) -> None:
        evidence.observation.state[DONE] = True

    engine, _trace = build(tmp_path, factory=MutatingFactory(mutation))
    result = run(engine, request_for())
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert result.attempts_started == 0


def test_a_factory_that_rewrites_a_one_as_true_is_refused(tmp_path: Path) -> None:
    def mutation(evidence) -> None:
        evidence.observation.state["numeric"] = True

    scenario = fake_scenario(action_spec=action(), start_state={DONE: False, "numeric": 1},
                             end_state={DONE: True, "numeric": 1})
    script = planner_script([("plan-1", {DONE: False, "numeric": 1}, forecast_for(action()))])
    engine, _trace = build(tmp_path, factory=MutatingFactory(mutation), scenario=scenario,
                          script=script)
    result = run(engine, request_for())
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


@pytest.mark.parametrize("factory", [GoodHooks(None), lambda: GoodHooks(None)])
def test_a_factory_without_create_is_a_dependency_failure(tmp_path: Path, factory: object) -> None:
    # A plain callable is not a factory either: only the declared create keyword is accepted.
    engine, _trace = build(tmp_path, factory=factory)
    result = run(engine, request_for())
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "DEPENDENCY_FAILURE"


def test_the_factory_cannot_commit_or_execute_after_the_deadline(tmp_path: Path) -> None:
    clock = Clock()

    def on_create(current, evidence) -> None:
        if len(current.calls) >= 2:
            clock.value = 2_000_000_000

    factory = Factory(on_create=on_create)
    engine, _trace = build(tmp_path, factory=factory, clock=clock, run_timeout_ms=1000.0,
                          domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    assert result.terminal_status.value == "timeout"
    assert result.reason_code.value == "DEADLINE_EXCEEDED"
    assert result.attempts_started == 0


def test_a_cancellation_from_the_factory_propagates_as_cancellation(tmp_path: Path) -> None:
    factory = Factory(verdict=asyncio.CancelledError())
    engine, _trace = build(tmp_path, factory=factory,
                          domain=HostDomain(preconditions=[PRECONDITION]))
    with pytest.raises(asyncio.CancelledError):
        run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    finalization = engine.last_finalization
    assert finalization is not None
    assert finalization.primary_terminal_status.value == "cancelled"
    assert finalization.primary_reason_code.value == "USER_CANCELLED"


def test_a_hook_that_mutates_a_nested_container_is_refused(tmp_path: Path) -> None:
    factory = Factory()

    def on_check(condition, observation, current):
        if current.checks >= 2:
            observation.state["nested"] = {"a": [1, 3]}

    factory.on_check = on_check
    scenario = fake_scenario(action_spec=action(),
                             start_state={DONE: False, "nested": {"a": [1, 2]}},
                             end_state={DONE: True, "nested": {"a": [1, 2]}})
    script = planner_script([("plan-1", {DONE: False, "nested": {"a": [1, 2]}},
                              forecast_for(action()))])
    engine, _trace = build(tmp_path, factory=factory, scenario=scenario, script=script,
                          domain=HostDomain(preconditions=[PRECONDITION]))
    result = run(engine, request_for(max_steps=4, max_planning_calls=2, max_replans=0))
    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"


# --- F02: the new factory/hooks boundary exposes no reachable original context ----------------


class RaisingFactory:
    """A factory whose ``create()`` call fails with one synthetic marker."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def create(self, *, evidence):
        raise ValueError(self.marker)


class PropertyRaisingFactory:
    """A factory whose ``create`` attribute read fails with one synthetic marker."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    @property
    def create(self):
        raise ValueError(self.marker)


def _reachable_text(error: BaseException) -> str:
    """Every text reachable along ``__cause__``, ``__context__`` and suppressed context."""
    parts: list[str] = []
    current: BaseException | None = error
    while current is not None:
        parts.append(repr(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__
    return "\n".join(parts)


def _evidence_for_hooks() -> VerificationEvidence:
    observation = Observation(
        schema_version=SCHEMA, observation_id="obs-hooks", environment_type="fake",
        state={DONE: False},
        timestamp=datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc))
    return VerificationEvidence(schema_version=SCHEMA, observation=observation,
                                state_complete=True)


def test_a_factory_call_failure_has_no_reachable_original_context(tmp_path: Path) -> None:
    marker = "STEWARD_SYNTHETIC_FACTORY_CALL"
    engine, _trace = build(tmp_path, factory=RaisingFactory(marker))
    with pytest.raises(RuntimeFailure) as raised:
        engine._factory_hooks(_evidence_for_hooks(), None, "build")
    assert raised.value.code == "DEPENDENCY_FAILURE"
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert marker not in _reachable_text(raised.value)


def test_a_factory_property_failure_has_no_reachable_original_context(tmp_path: Path) -> None:
    marker = "STEWARD_SYNTHETIC_FACTORY_PROPERTY"
    engine, _trace = build(tmp_path, factory=PropertyRaisingFactory(marker))
    with pytest.raises(RuntimeFailure) as raised:
        engine._factory_hooks(_evidence_for_hooks(), None, "build")
    assert raised.value.code == "DEPENDENCY_FAILURE"
    assert raised.value.__context__ is None
    assert marker not in _reachable_text(raised.value)


def test_a_precondition_failure_has_no_reachable_original_context(tmp_path: Path) -> None:
    marker = "STEWARD_SYNTHETIC_PRECONDITION"
    factory = Factory(verdict=Exception(marker))
    engine, _trace = build(tmp_path, factory=factory)
    contract = SimpleNamespace(preconditions=[PRECONDITION])
    observation = _evidence_for_hooks().observation
    with pytest.raises(RuntimeFailure) as raised:
        engine._injected_preconditions(contract, observation, _evidence_for_hooks(), None)
    assert raised.value.code == "DEPENDENCY_FAILURE"
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert marker not in _reachable_text(raised.value)


def test_a_factory_cancellation_keeps_its_base_exception_identity(tmp_path: Path) -> None:
    class CancellingFactory:
        def create(self, *, evidence):
            raise asyncio.CancelledError()

    engine, _trace = build(tmp_path, factory=CancellingFactory())
    with pytest.raises(asyncio.CancelledError):
        engine._factory_hooks(_evidence_for_hooks(), None, "build")


# --- F05: the independent criteria prevalidation ---------------------------------------------

BAD_CONDITION = ExpectedOutcome(
    schema_version=SCHEMA, outcome_type="json_rule",
    payload={"schema_version": SCHEMA, "rule_type": "no_such_json_rule", "path": ["x"]},
    required=True,
)


class RecordingValidator:
    """A validator that records the record it was handed and returns ``None``."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def validate(self, *, criteria) -> None:
        self.calls.append(criteria)


class RejectingValidator(RecordingValidator):
    """A validator that reports unusable domain criteria through the Core input error."""

    def validate(self, *, criteria) -> None:
        super().validate(criteria=criteria)
        raise VerificationError("VERIFICATION_INPUT_INVALID", ("criteria", "conditions"),
                                "the task criteria are not a usable domain rule set")


class OrdinaryValidator(RecordingValidator):
    def __init__(self, marker: str) -> None:
        super().__init__()
        self.marker = marker

    def validate(self, *, criteria) -> None:
        super().validate(criteria=criteria)
        raise ValueError(self.marker)


class MissingValidate:
    pass


class AwaitableValidator(RecordingValidator):
    def validate(self, *, criteria):
        self.calls.append(criteria)
        async def never_awaited() -> None:
            return None
        return never_awaited()


class ReturningValidator(RecordingValidator):
    def validate(self, *, criteria):
        self.calls.append(criteria)
        return "pass"


class MutatingValidator(RecordingValidator):
    def validate(self, *, criteria) -> None:
        super().validate(criteria=criteria)
        criteria.conditions[0].payload["path"] = ["mutated"]


def test_the_factory_protocol_declares_the_existing_hooks_type() -> None:
    hints = get_type_hints(ContractHooksFactory.create)
    assert hints["return"] is ContractValidationHooks


def test_malformed_criteria_are_refused_before_start_even_with_a_factory(
        tmp_path: Path) -> None:
    """F05: the factory no longer bypasses the Core criteria prevalidation."""
    factory = Factory()
    engine, _trace = build(tmp_path, factory=factory)
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for(conditions=[BAD_CONDITION]))
    assert raised.value.code == "INPUT_INVALID"
    assert factory.calls == []
    assert not (tmp_path / "trace.jsonl").exists()


def test_an_injected_validator_runs_once_before_anything_starts(tmp_path: Path) -> None:
    validator = RecordingValidator()
    engine, _trace = build(tmp_path, criteria_validator=validator)
    request = request_for()
    result = run(engine, request)
    assert result.terminal_status.value == "completed"
    assert len(validator.calls) == 1
    assert validator.calls[0] is not request.task_criteria


def test_a_validator_rejection_is_input_invalid_before_start(tmp_path: Path) -> None:
    validator = RejectingValidator()
    engine, _trace = build(tmp_path, criteria_validator=validator)
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for())
    assert raised.value.code == "INPUT_INVALID"
    assert len(validator.calls) == 1
    assert not (tmp_path / "trace.jsonl").exists()


def test_a_validator_missing_its_method_is_a_dependency_failure(tmp_path: Path) -> None:
    engine, _trace = build(tmp_path, criteria_validator=MissingValidate())
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for())
    assert raised.value.code == "DEPENDENCY_FAILURE"
    assert raised.value.__context__ is None


def test_a_validator_ordinary_exception_is_bounded(tmp_path: Path) -> None:
    marker = "STEWARD_SYNTHETIC_CRITERIA_VALIDATOR"
    engine, _trace = build(tmp_path, criteria_validator=OrdinaryValidator(marker))
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for())
    assert raised.value.code == "DEPENDENCY_FAILURE"
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert marker not in _reachable_text(raised.value)


def test_an_awaitable_validator_return_is_refused(tmp_path: Path) -> None:
    engine, _trace = build(tmp_path, criteria_validator=AwaitableValidator())
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for())
    assert raised.value.code == "DEPENDENCY_FAILURE"


def test_a_non_none_validator_return_is_refused(tmp_path: Path) -> None:
    engine, _trace = build(tmp_path, criteria_validator=ReturningValidator())
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for())
    assert raised.value.code == "DEPENDENCY_FAILURE"


def test_a_validator_that_mutates_the_criteria_is_refused(tmp_path: Path) -> None:
    engine, _trace = build(tmp_path, criteria_validator=MutatingValidator())
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for())
    assert raised.value.code == "EVIDENCE_INVALID"


def test_a_validator_cancellation_keeps_its_base_exception_identity(tmp_path: Path) -> None:
    class CancellingValidator:
        def validate(self, *, criteria) -> None:
            raise asyncio.CancelledError()

    engine, _trace = build(tmp_path, criteria_validator=CancellingValidator())
    with pytest.raises(asyncio.CancelledError):
        run(engine, request_for())


# --- F01 rollback: an unknown execution is never judged on stale pre-attempt evidence ---------


class StaleEvidenceAdapter(WideAdapter):
    """A real-shaped adapter whose post-attempt observation fails after an unknown execution.

    The declared sequence is the one the round-002 probe reproduced: the opening observation shows
    the goal unmet, the pre-execution re-observation shows it met (the stale evidence that could
    authorise a false PASS), the execution's effect is unknown while the fixture's real
    post-execution state stays unmet, and the real post-attempt observation fails with an ordinary
    error. A correct Runtime must propagate that facility failure; it may not fall back to the
    pre-attempt snapshot.
    """

    def __init__(self, inner: FakeEnvironmentAdapter, *, states: list[object]) -> None:
        super().__init__(inner)
        self._states = list(states)
        self.observe_calls = 0

    async def observe(self) -> Observation:
        self.observe_calls += 1
        state = self._states[min(self.observe_calls, len(self._states)) - 1]
        if isinstance(state, BaseException):
            raise state
        seen = observation(f"obs-stale-control-{self.observe_calls}", state)
        self.observations.append(seen)
        return seen

    async def execute(self, action_spec):
        self.actions.append(action_spec)
        result = unknown("outcome-stale-control")
        self.outcomes.append(result)
        return result


def _stale_control(tmp_path: Path, memory=None):
    """Build the stale-evidence control: pre-attempt evidence met, real state unmet, observe fails."""
    marker = "STEWARD_SYNTHETIC_POST_ATTEMPT_OBSERVATION"
    scenario = fake_scenario(action_spec=action(), start_state={DONE: False},
                             end_state={DONE: False})
    adapter = StaleEvidenceAdapter(
        FakeEnvironmentAdapter(scenario),
        states=[{DONE: False}, {DONE: True}, OSError(marker)],
    )
    engine, trace = build(tmp_path, adapter=adapter, scenario=scenario, memory=memory)
    return engine, trace, adapter


def test_an_unknown_execution_does_not_swallow_a_failed_post_attempt_observation(
        tmp_path: Path) -> None:
    """F01 rollback: the real post-attempt observation failure stays a facility failure.

    The pre-attempt snapshot carries the goal holding, and the fixture's real post-execution state
    does not. Judging the step on that stale snapshot would write a step PASS and possibly
    TASK_ACHIEVED, so the frozen behaviour - propagate the observation failure - is required.
    """
    engine, trace, adapter = _stale_control(tmp_path)
    with pytest.raises(RuntimeFailure) as raised:
        run(engine, request_for())
    assert raised.value.code == "DEPENDENCY_FAILURE"
    assert adapter.observe_calls == 3
    finalization = engine.last_finalization
    assert finalization is not None
    assert finalization.primary_terminal_status.value == "failed"
    assert finalization.primary_reason_code.value == "DEPENDENCY_FAILURE"
    records = list(trace.read_run("run-1"))
    kinds = [record.kind for record in records]
    assert "attempt_started" in kinds and "attempt_finished" in kinds
    # No step verdict was produced from the stale snapshot, and no success terminal was written.
    assert "verification_recorded" not in kinds
    assert records[-1].kind == "run_finished"
    assert records[-1].terminal_status == "failure"
    assert records[-1].reason_code == "internal_error"


def test_a_failed_post_attempt_observation_keeps_its_real_call_provenance(
        tmp_path: Path) -> None:
    """F01 rollback: an observation that never returned stays in the published call record."""
    memory = WorkingMemory()
    engine, _trace, _adapter = _stale_control(tmp_path, memory=memory)
    with pytest.raises(RuntimeFailure):
        run(engine, request_for())
    state = memory.get("run-1")
    calls = state["observation_calls"]
    assert len(calls) == 3
    assert [call["observation_id"] for call in calls] == [
        "obs-stale-control-1", "obs-stale-control-2", None]
    assert [call["phase"] for call in calls] == [
        "OBSERVING", "CHECKING_PRECONDITIONS", "VERIFYING"]
    assert [call["ordinal"] for call in calls] == sorted(
        call["ordinal"] for call in calls)
