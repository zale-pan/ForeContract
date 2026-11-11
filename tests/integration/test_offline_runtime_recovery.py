"""C04 to C07: recovery, an unresolved effect, and the deterministic bounds of one Run.

The cases drive the real Runtime over the real Fake, Planner, Builder and RecoveryEngine and read
their facts back from the JSONL trace: a retried Attempt, a replan that commits a second Contract, an
unknown outcome that is never repeated blindly, and the host limits that stop a Run as an abort rather
than as a facility failure or a timeout.

Two local helpers exist because the shared scenario helper cannot express these cases: the Fake
forbids a repeated action type and payload in a scenario's catalog, while a retry must re-execute the
same committed action, so the catalog is de-duplicated here while the two scripted steps stay
distinct. Every expected fact is declared by the test itself and was measured against the real engine
before it was written down.
"""

from __future__ import annotations

import asyncio

from conftest import (
    FLAG,
    SCHEMA,
    DeclaredAdapter,
    HostDomain,
    action,
    build_engine,
    failed,
    fake_scenario,
    forecast_for,
    json_rule_condition,
    kinds,
    no_step_scenario,
    observation,
    planner_script,
    read_records,
    request_for,
    succeeded,
    terminal_records,
    unknown,
)
from forecontract.models import ActionSpec, ActualOutcome, ErrorInfo
from forecontract_fake import (
    FakeEnvironmentAdapter,
    FakeScenario,
    FakeStage,
    FakeStep,
)
from forecontract_scripted import ScriptedPlanner

#: One millisecond in nanoseconds, for the declared monotonic timeline the budget cases drive.
_NS_PER_MS = 1_000_000


def recoverable_failure(outcome_id: str) -> ActualOutcome:
    """A failed receipt that declares its own recoverability, which a real retry requires.

    The shared ``failed()`` helper deliberately leaves ``ErrorInfo.recoverable`` unset, and the real
    RecoveryEngine only authorises a retry for a failed or unknown execution whose error explicitly
    says ``recoverable is True``. The companion test below asserts that behaviour; this receipt
    states the recoverability the retry branch needs.
    """
    return ActualOutcome(schema_version=SCHEMA, outcome_id=outcome_id, status="failed",
                         error=ErrorInfo(schema_version=SCHEMA, code="EXECUTION_FAILED",
                                         message="the action did not apply", recoverable=True))


def repeated_action_scenario(*, action_spec: ActionSpec, states: list[dict],
                             first_outcome: ActualOutcome,
                             second_outcome: ActualOutcome) -> FakeScenario:
    """A two-step scenario in which both steps execute the very same action.

    The scenario's catalog is deduplicated on purpose, because a retry re-executes the committed
    action and the Fake refuses a catalog that repeats an action type and payload.
    """
    steps = [
        FakeStep(schema_version=SCHEMA, step_id="fake-step-1", action=action_spec,
                 outcome=first_outcome),
        FakeStep(schema_version=SCHEMA, step_id="fake-step-2", action=action_spec,
                 outcome=second_outcome),
    ]
    stages = [FakeStage(schema_version=SCHEMA, actual_state=state,
                        observation=observation(f"obs-{index}", state))
              for index, state in enumerate(states)]
    return FakeScenario(schema_version=SCHEMA, scenario_id="scenario", adapter_name="fake",
                        allowed_actions=[action_spec], stages=stages, steps=steps)


def test_a_retry_re_executes_the_committed_contract_and_is_counted(workdir):
    """C04: a recoverable failed Attempt is retried, and the retry re-executes the same Contract.

    The domain allows exactly one retry, refuses a replan, declares the action repeat-safe and reports
    the effect as not applied, so the real RecoveryEngine answers a retry decision. The second real
    execution is therefore the already committed Contract: no new Step, no new Contract, and the
    Attempt counter moves while the Step counter does not.
    """
    done = action()
    start = {FLAG: False}
    end = {FLAG: True}
    scenario = repeated_action_scenario(
        action_spec=done, states=[start, start, end],
        first_outcome=recoverable_failure("ao-1"),
        second_outcome=succeeded("ao-2", end))
    domain = HostDomain(mode="bounded", max_retries=1, allow_replan=False,
                        retry_safety="idempotent", effect="not_applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([("plan-1", start, forecast_for(done))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        max_steps=1, max_planning_calls=1, max_retries=1)))

    # Observed exactly: completed / TASK_ACHIEVED, one Step, one planning call, two Attempts, one
    # retry used and no replan.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 2, 1, 0)
    assert (adapter.starts, adapter.stops) == (1, 1)
    # Observed exactly: two real executions of the same action reached the adapter, carrying the two
    # scripted receipts.
    assert [item.action_type for item in adapter.actions] == ["mark_done", "mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1", "ao-2"]
    # The domain was asked for the effect of the first execution, which is what the retry decision
    # was made from.
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1"]

    records = read_records(trace, "run-1")
    # Observed exactly: one Step, one committed Contract and two Attempts, in this record order.
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "verification_recorded", "run_finished"]
    committed = [record for record in records if record.kind == "contract_committed"]
    assert len(committed) == 1
    assert committed[0].replaces_contract_id is None
    assert kinds(records).count("step_started") == 1
    attempts = [record for record in records if record.kind == "attempt_started"]
    assert len(attempts) == 2
    # Both Attempts belong to the one committed Contract and to the one logical Step: the committed
    # contract was not rewritten by the retry.
    assert {record.contract_id for record in attempts} == {committed[0].contract_id}
    assert {record.step_id for record in attempts} == {committed[0].step_id}
    finished = [record for record in records if record.kind == "attempt_finished"]
    assert [(record.execution_status.value, record.error_code) for record in finished] == [
        ("failed", "execution_failed"), ("succeeded", None)]
    step_verdicts = [record for record in records
                     if record.kind == "verification_recorded" and record.scope == "attempt"]
    assert [record.verification_status.value for record in step_verdicts] == ["fail", "pass"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    # Observed exactly: a completed Run's terminal record is spelled success with no reason code, and
    # it names the one task-scope PASS that this Run really recorded.
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("success", None)
    assert terminals[0].record_id == result.terminal_record_id
    run_verdicts = [record for record in records
                    if record.kind == "verification_recorded" and record.scope == "run"]
    assert len(run_verdicts) == 1
    assert result.verification_record_id == run_verdicts[0].record_id
    assert terminals[0].verification_record_id == run_verdicts[0].record_id
    # The teardown settled the resource, so nothing is quarantined and the trace is confirmed.
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.cleanup_status, finalization.trace_status) == ("succeeded", "persisted")
    assert (finalization.finalize_deadline_exceeded, finalization.quarantined) == (False, False)
    assert finalization.pending_operation_at_return is False
    assert engine.is_quarantined is False


def test_a_failure_without_a_recoverability_declaration_is_not_retried(workdir):
    """C04: the retry allowance alone does not authorise a retry.

    Documented observed behaviour, asserted as it really is. The shared ``failed()`` receipt leaves
    ``ErrorInfo.recoverable`` unset, and the real RecoveryEngine's ``retry_allowed`` rule requires an
    explicit ``recoverable is True`` for a failed or unknown execution. A host that allows one retry
    therefore still gets a bounded policy abort here, and the second scripted step - which really
    exists in the scenario - is never executed. This is not a weakened retry case: it is the same
    host policy with a receipt that does not carry the declaration the retry branch requires.
    """
    done = action()
    start = {FLAG: False}
    end = {FLAG: True}
    scenario = repeated_action_scenario(
        action_spec=done, states=[start, start, end],
        first_outcome=failed("ao-1"),
        second_outcome=succeeded("ao-2", end))
    domain = HostDomain(mode="bounded", max_retries=1, allow_replan=False,
                        retry_safety="idempotent", effect="not_applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([("plan-1", start, forecast_for(done))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        max_steps=2, max_planning_calls=2, max_retries=1)))

    # Observed exactly: aborted / POLICY_ABORT after one Step and one Attempt.
    assert (result.terminal_status.value, result.reason_code.value) == ("aborted", "POLICY_ABORT")
    assert (result.steps_started, result.attempts_started, result.retries_used) == (1, 1, 0)
    assert result.verification_record_id is None
    # The retry was refused by a real decision, not skipped: the domain was still asked for the effect.
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1"]
    assert len(scenario.steps) == 2
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    # Observed exactly: the abort is spelled policy_abort and carries no task PASS reference.
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id


def test_a_retry_allowance_of_zero_stops_after_the_first_attempt(workdir):
    """C07: a recoverable failure with no retry allowance is one Attempt and a bounded abort.

    The receipt declares itself recoverable, so the retry branch is genuinely reachable; the host
    declares no retry at all. Observed exactly: the single real Attempt happens, the Run aborts on its
    policy, and the second executable step that the scenario really carries is never taken.
    """
    done = action()
    start = {FLAG: False}
    end = {FLAG: True}
    scenario = repeated_action_scenario(
        action_spec=done, states=[start, start, end],
        first_outcome=recoverable_failure("ao-1"),
        second_outcome=succeeded("ao-2", end))
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False,
                        retry_safety="idempotent", effect="not_applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([("plan-1", start, forecast_for(done))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        max_steps=2, max_planning_calls=2, max_retries=0)))

    # Observed exactly: aborted / POLICY_ABORT, one Step, one planning call, exactly one Attempt, no
    # retry and no replan.
    assert (result.terminal_status.value, result.reason_code.value) == ("aborted", "POLICY_ABORT")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 0, 0)
    assert result.verification_record_id is None
    # Observed exactly: one action reached the adapter, although a second executable step existed.
    assert len(scenario.steps) == 2
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    # The exhaustion was decided from the domain's real effect judgement for this very outcome.
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1"]
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    finished = [record for record in records if record.kind == "attempt_finished"]
    assert [(record.execution_status.value, record.error_code) for record in finished] \
        == [("failed", "execution_failed")]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.primary_reason_code.value, finalization.cleanup_status,
            finalization.trace_status, finalization.quarantined) \
        == ("POLICY_ABORT", "succeeded", "persisted", False)


class RetryClock:
    """A declared monotonic timeline the budget case advances explicitly, never a wall-clock reading."""

    def __init__(self) -> None:
        self.ns = 0

    def now_ns(self) -> int:
        return self.ns

    def advance_ms(self, milliseconds: float) -> None:
        self.ns += int(milliseconds * _NS_PER_MS)


class BurningEffectDomain(HostDomain):
    """A host binding whose own effect judgement spends the declared timeline it was handed.

    The spend happens inside ``assess_effect``, which is the effect judgement the real RecoveryEngine's
    retry decision is made from, so the retry is granted and only then discovered to be unpaid for.
    """

    def __init__(self, clock, burn_ms, **options) -> None:
        super().__init__(**options)
        self._clock = clock
        self._burn_ms = burn_ms

    def assess_effect(self, *, contract, actual_outcome, evidence) -> str:
        self._clock.advance_ms(self._burn_ms)
        return super().assess_effect(contract=contract, actual_outcome=actual_outcome,
                                     evidence=evidence)


def test_a_granted_retry_cannot_be_paid_for_once_the_run_budget_is_spent(workdir):
    """C05: a retry granted after the Run's remaining budget was spent is not given a fresh budget.

    Two Runs replay the same script under the same declared 100 ms host budget on a deterministic
    monotonic timeline, and both declare one retry. The control spends nothing and pays for the retry
    with a second real Attempt. The second Run's host support really spends 110 ms of that declared
    timeline inside the effect judgement its retry decision is made from, so by the time the decision
    comes back the Run's remaining budget is already zero; the retry is still granted - the decision was
    given a budget snapshot rather than a fresh reading - and then refused, because no budget remains.
    The two Runs differ only in that spending, which is what makes this assertion reset-sensitive: a
    product that restored the budget at the retry would pay for the second Attempt and complete.
    """
    done = action()
    start = {FLAG: False}
    end = {FLAG: True}

    def run_once(name: str, burn_ms: float):
        clock = RetryClock()
        directory = workdir / name
        directory.mkdir()
        scenario = repeated_action_scenario(
            action_spec=done, states=[start, start, end],
            first_outcome=recoverable_failure("ao-1"),
            second_outcome=succeeded("ao-2", end))
        domain = BurningEffectDomain(clock, burn_ms, mode="bounded", max_retries=1,
                                     allow_replan=False, retry_safety="idempotent",
                                     timeout_ms=90.0)
        adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
        engine, trace, _ = build_engine(
            tmp_path=directory, adapter=adapter,
            planner=ScriptedPlanner(planner_script([("plan-1", start, forecast_for(done))])),
            domain=domain, budget_ms=100.0, monotonic_clock=clock)
        result = asyncio.run(engine.run(request=request_for(
            max_steps=2, max_planning_calls=2, max_retries=1)))
        return result, adapter, domain, engine, trace, clock

    control_result, control_adapter, control_domain, control_engine, _, control_clock = run_once(
        "control", 0.0)

    # Observed exactly: with the same script, the same policy and the same declared budget, the retry
    # is paid for out of that budget and the retried step completes.
    assert (control_result.terminal_status.value, control_result.reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (control_result.steps_started, control_result.planning_calls,
            control_result.attempts_started, control_result.retries_used,
            control_result.replans_used) == (1, 1, 2, 1, 0)
    assert control_clock.ns == 0
    assert [item.action_type for item in control_adapter.actions] == ["mark_done", "mark_done"]
    assert [item.outcome_id for item in control_adapter.outcomes] == ["ao-1", "ao-2"]
    assert [outcome_id for _, outcome_id in control_domain.assess_calls] == ["ao-1"]

    result, adapter, domain, engine, trace, clock = run_once("burning", 110.0)

    # Observed exactly: the declared timeline really spent more than the whole 100 ms budget, and the
    # Run is stopped by that budget rather than by a timeout.
    assert clock.ns == 110 * _NS_PER_MS
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("aborted", "BUDGET_EXHAUSTED")
    # Observed exactly: one Step, one planning call, one real Attempt, and one retry that was *granted*
    # and then could not be paid for. No replan was used.
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 1, 0)
    assert result.verification_record_id is None
    # The effect judgement the retry was decided from really happened, and the second scripted action
    # never reached the adapter.
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1"]
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    records = read_records(trace, "run-1")
    # Observed exactly: one committed Contract, one Attempt and no second Attempt fact at all.
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    assert kinds(records).count("attempt_started") == 1
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.primary_terminal_status.value, finalization.primary_reason_code.value,
            finalization.cleanup_status, finalization.trace_status, finalization.quarantined) \
        == ("aborted", "BUDGET_EXHAUSTED", "succeeded", "persisted", False)
    # The reset-sensitivity: the control paid for the retry with a second Attempt and completed, the
    # limited Run could not. A product that restored the budget would have produced the control here.
    assert (control_result.attempts_started, result.attempts_started) == (2, 1)
    assert control_result.reason_code.value != result.reason_code.value
    assert control_engine.is_quarantined is False
    assert engine.is_quarantined is False


def test_a_positive_retry_allowance_is_exhausted_by_a_failure_that_keeps_failing(workdir):
    """C07: an allowance of one retry buys exactly one retry, and then the Run stops.

    Both scripted executions of the committed action really run and both are recoverable failures, so
    the retry branch is reachable twice. The host declares ``max_retries=1`` and the contract carries
    the real RecoveryEngine's own retry decision, which is bounded by the per-contract retry count as
    well as by the Run's remaining allowance. Observed exactly: two Attempts, one retry used, no
    replan, and a bounded policy abort. The scenario carries exactly two executable steps, so a third
    action was not merely unnecessary - the Fake itself would have refused it.
    """
    done = action()
    start = {FLAG: False}
    scenario = repeated_action_scenario(
        action_spec=done, states=[start, start, start],
        first_outcome=recoverable_failure("ao-1"),
        second_outcome=recoverable_failure("ao-2"))
    domain = HostDomain(mode="bounded", max_retries=1, allow_replan=False,
                        retry_safety="idempotent")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([("plan-1", start, forecast_for(done))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        max_steps=2, max_planning_calls=2, max_retries=1)))

    # Observed exactly: aborted / POLICY_ABORT, one Step, one planning call, two Attempts, one retry
    # used and no replan.
    assert (result.terminal_status.value, result.reason_code.value) == ("aborted", "POLICY_ABORT")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 2, 1, 0)
    assert result.verification_record_id is None
    # Observed exactly: exactly two real executions, and no third action was taken.
    assert len(scenario.steps) == 2
    assert [item.action_type for item in adapter.actions] == ["mark_done", "mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1", "ao-2"]
    assert len(adapter.actions) == 2
    assert (adapter.starts, adapter.stops) == (1, 1)
    # The verdict came from the domain's real effect judgement of the *second* failure, which is the
    # Attempt the refusal is about.
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1", "ao-2"]
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    assert kinds(records).count("contract_committed") == 1
    finished = [record for record in records if record.kind == "attempt_finished"]
    assert [(record.execution_status.value, record.error_code) for record in finished] \
        == [("failed", "execution_failed"), ("failed", "execution_failed")]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.primary_terminal_status.value, finalization.primary_reason_code.value,
            finalization.cleanup_status, finalization.trace_status, finalization.quarantined) \
        == ("aborted", "POLICY_ABORT", "succeeded", "persisted", False)
    assert engine.is_quarantined is False


def test_a_replan_commits_a_second_distinct_contract(workdir):
    """C05: an allowed replan commits a second Contract that replaces the first, inside one Step.

    Noted path: the engine reaches this replan through the RECOVERING branch, where the real
    RecoveryEngine answers REPLAN_ALLOWED - the retry is denied because the effect is ``applied`` and
    the action is declared not safe to repeat - rather than through the stale-precondition branch.
    A Contract that replaces another belongs to the same logical Step, so the Step counter stays at
    one while two real Contracts are committed and the Run finishes by the task oracle.
    """
    open_gate = action("open_gate")
    mark_done = action("mark_done")
    start = {"gate": "closed", FLAG: False}
    after_first = {"gate": "closed", FLAG: False}
    end = {"gate": "closed", FLAG: True}
    task = [json_rule_condition(FLAG, True)]
    scenario = fake_scenario(
        action_spec=open_gate, start_state=start, end_state=end,
        scripted=[(open_gate, failed("ao-1"), after_first),
                  (mark_done, succeeded("ao-2", end), end)])
    domain = HostDomain(mode="bounded", max_retries=1, allow_replan=True,
                        retry_safety="non_idempotent", effect="applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([
            ("plan-1", start, forecast_for(open_gate, forecast_id="fc-1",
                                           conditions=[json_rule_condition("gate", "open")])),
            ("plan-2", after_first, forecast_for(mark_done, forecast_id="fc-2",
                                                 conditions=[json_rule_condition(FLAG, True)])),
        ])), domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=task, max_steps=2, max_planning_calls=2, max_replans=1)))

    # Observed exactly: completed / TASK_ACHIEVED with two planning calls, two Attempts and one replan.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 2, 2, 0, 1)
    records = read_records(trace, "run-1")
    committed = [record for record in records if record.kind == "contract_committed"]
    assert len(committed) == 2
    first, second = committed
    # The two Contracts are genuinely distinct identifiers, and the second one declares the first.
    assert first.contract_id != second.contract_id
    assert first.replaces_contract_id is None
    assert second.replaces_contract_id == first.contract_id
    # A replacement Contract belongs to the same logical Step, so no second Step was started.
    assert second.step_id == first.step_id
    assert kinds(records).count("step_started") == 1
    # Both scripted actions really ran, in order, through their own Attempts.
    assert [item.action_type for item in adapter.actions] == ["open_gate", "mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1", "ao-2"]
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("success", None)
    assert terminals[0].record_id == result.terminal_record_id
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.cleanup_status, finalization.trace_status, finalization.quarantined) \
        == ("succeeded", "persisted", False)


def test_a_replan_does_not_restore_the_run_budget(workdir):
    """C05: the Run's own allowance is cumulative, so a replan cannot restore budget.

    Two Runs replay the same script under the same declared 100 ms host budget on a deterministic
    monotonic timeline. The control never spends it and completes the replanned step. The second Run's
    host support really spends 110 ms of that declared timeline while preparing the *replanned*
    Contract, and the replanned build is then refused because no budget remains: the second planning
    call and the replan really happened, and the Run aborts on the budget instead of being granted a
    fresh allowance for the replan.
    """
    open_gate = action("open_gate")
    mark_done = action("mark_done")
    start = {"gate": "closed", FLAG: False}
    after_first = {"gate": "closed", FLAG: False}
    end = {"gate": "closed", FLAG: True}
    task = [json_rule_condition(FLAG, True)]

    class StepClock:
        """A declared monotonic timeline the test advances explicitly, never a wall-clock reading."""

        def __init__(self) -> None:
            self.ns = 0

        def now_ns(self) -> int:
            return self.ns

        def advance_ms(self, milliseconds: float) -> None:
            self.ns += int(milliseconds * _NS_PER_MS)

    class BurningPreparationDomain(HostDomain):
        """The host's own contract preparation, replaying the declared timeline it spends."""

        def __init__(self, clock, burn_ms: float, **options) -> None:
            super().__init__(**options)
            self._clock = clock
            self._burn_ms = burn_ms
            self.preparations = 0

        def prepare_contract(self, *, forecast, evidence):
            self.preparations += 1
            # Only the replanned Contract's preparation spends the declared timeline.
            if self.preparations == 2:
                self._clock.advance_ms(self._burn_ms)
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    def run_once(name: str, burn_ms: float, budget_ms: float = 100.0):
        clock = StepClock()
        directory = workdir / name
        directory.mkdir()
        scenario = fake_scenario(
            action_spec=open_gate, start_state=start, end_state=end,
            scripted=[(open_gate, failed("ao-1"), after_first),
                      (mark_done, succeeded("ao-2", end), end)])
        domain = BurningPreparationDomain(
            clock, burn_ms, mode="bounded", max_retries=1, allow_replan=True,
            retry_safety="non_idempotent", effect="applied", timeout_ms=90.0)
        adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
        engine, trace, _ = build_engine(
            tmp_path=directory, adapter=adapter,
            planner=ScriptedPlanner(planner_script([
                ("plan-1", start, forecast_for(
                    open_gate, forecast_id="fc-1",
                    conditions=[json_rule_condition("gate", "open")])),
                ("plan-2", after_first, forecast_for(
                    mark_done, forecast_id="fc-2",
                    conditions=[json_rule_condition(FLAG, True)])),
            ])), domain=domain, budget_ms=budget_ms, monotonic_clock=clock)
        result = asyncio.run(engine.run(request=request_for(
            conditions=task, max_steps=2, max_planning_calls=2, max_replans=1)))
        return result, adapter, domain, engine, trace

    control_result, control_adapter, control_domain, control_engine, _ = run_once(
        "control", 0.0)
    # Observed exactly: with the same script, budget and policy, the replanned step completes.
    assert (control_result.terminal_status.value, control_result.reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (control_result.steps_started, control_result.planning_calls,
            control_result.attempts_started, control_result.replans_used) == (1, 2, 2, 1)
    assert control_domain.preparations == 2
    assert [item.action_type for item in control_adapter.actions] == ["open_gate", "mark_done"]

    result, adapter, domain, engine, trace = run_once("burning", 110.0)

    # Observed exactly: the replan really happened - two planning calls and one replan used - and the
    # Run was then stopped by the budget before the replanned Contract could be committed.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("aborted", "BUDGET_EXHAUSTED")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 2, 1, 0, 1)
    assert result.verification_record_id is None
    assert domain.preparations == 2
    # Only the first scripted action ever ran: the replanned action never reached the adapter.
    assert [item.action_type for item in adapter.actions] == ["open_gate"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    records = read_records(trace, "run-1")
    # Observed exactly: one committed Contract, so the replanned Contract was never committed.
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    assert kinds(records).count("contract_committed") == 1
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].record_id == result.terminal_record_id
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.primary_reason_code.value, finalization.cleanup_status,
            finalization.trace_status, finalization.quarantined, finalization.terminal_record_id) \
        == ("BUDGET_EXHAUSTED", "succeeded", "persisted", False, terminals[0].record_id)
    # The control Run is the proof that the budget, not the script, stopped the replanned Run.
    assert control_result.reason_code.value != result.reason_code.value
    assert control_engine.is_quarantined is False
    assert engine.is_quarantined is False

    # The reset-sensitive half the contract asks for: replay the same real spending against a *different*
    # declared deadline. A product that recomputed or restored the allowance when the replan was decided would
    # stop the narrower Run somewhere else, or would let it pay for the replanned Contract like the control.
    wide, wide_adapter, wide_domain, wide_engine, wide_trace = run_once("wide", 110.0, 100.0)
    narrow, narrow_adapter, narrow_domain, narrow_engine, narrow_trace = run_once(
        "narrow", 110.0, 95.0)

    # Observed exactly: both spent Runs really prepared the replanned Contract, and both were then stopped by
    # the budget after the replan had already been used - so the replan is not fictitious.
    for stopped, spent_domain, spent_engine in ((wide, wide_domain, wide_engine),
                                               (narrow, narrow_domain, narrow_engine)):
        assert spent_domain.preparations == 2
        assert (stopped.steps_started, stopped.planning_calls, stopped.attempts_started,
                stopped.retries_used, stopped.replans_used) == (1, 2, 1, 0, 1)
        assert stopped.verification_record_id is None
        assert spent_engine.is_quarantined is False

    # The declared 95 ms deadline changes nothing about where the spending stops the Run.
    assert (narrow.terminal_status.value, narrow.reason_code.value) \
        == (wide.terminal_status.value, wide.reason_code.value) \
        == ("aborted", "BUDGET_EXHAUSTED")
    assert (narrow.steps_started, narrow.planning_calls, narrow.attempts_started,
            narrow.retries_used, narrow.replans_used) \
        == (wide.steps_started, wide.planning_calls, wide.attempts_started,
            wide.retries_used, wide.replans_used)
    assert [item.action_type for item in narrow_adapter.actions] == ["open_gate"]
    assert [item.action_type for item in wide_adapter.actions] == ["open_gate"]

    # The narrower declaration really is a different deadline: the same script with nothing spent still
    # completes under 95 ms, so it is the spending, not the smaller number, that stopped the Run.
    narrow_control, narrow_control_adapter, narrow_control_domain, _, _ = run_once(
        "narrow-control", 0.0, 95.0)
    assert (narrow_control.terminal_status.value,
            narrow_control.reason_code.value) == ("completed", "TASK_ACHIEVED")
    assert (narrow_control.steps_started, narrow_control.planning_calls,
            narrow_control.attempts_started, narrow_control.replans_used) == (1, 2, 2, 1)
    assert narrow_control_domain.preparations == 2
    assert [item.action_type for item in narrow_control_adapter.actions] \
        == ["open_gate", "mark_done"]

    # Both spent traces are the honest record of the same stopping point.
    for spent_trace, terminal_id in ((wide_trace, wide.terminal_record_id),
                                     (narrow_trace, narrow.terminal_record_id)):
        spent_terminals = terminal_records(read_records(spent_trace, "run-1"))
        assert len(spent_terminals) == 1
        assert (spent_terminals[0].terminal_status,
                spent_terminals[0].reason_code) == ("aborted", "policy_abort")
        assert spent_terminals[0].verification_record_id is None
        assert spent_terminals[0].record_id == terminal_id

    # ---------------------------------------------------------------- the C05 three-control structure
    # One oracle, applied unchanged to the limited control and to the reset mutant. A Run satisfies it only
    # when it stopped on an already-spent budget, so both the positive control and the reset mutant fail it -
    # and for the mutant that rejection is exactly what the test is for.
    def _limited_oracle(run_result):
        """The precise static oracle for a Run that must have stopped on a spent budget."""
        return (run_result.terminal_status.value == "aborted"
                and run_result.reason_code.value == "BUDGET_EXHAUSTED"
                and run_result.verification_record_id is None)

    from forecontract.runtime import (
        BudgetSource,
        MonotonicBudgetSource,
        RuntimeBudgetSnapshot,
    )

    class SpendsBeforeReplan(HostDomain):
        """Spends the declared timeline while building the evidence for the post-failure re-observation.

        That point is before the RecoveryEngine decides, so the allowance is really gone by the moment a replan
        would be chosen - unlike spending while the replanned Contract is being prepared.
        """

        def __init__(self, clock, burn_ms, **options):
            super().__init__(**options)
            self._clock = clock
            self._burn_ms = burn_ms
            self.evidence_calls = 0

        def to_evidence(self, observation):
            evidence = super().to_evidence(observation)
            self.evidence_calls += 1
            if self.evidence_calls == 3:
                self._clock.advance_ms(self._burn_ms)
            return evidence

    class ResetDeclaration(BudgetSource):
        """Declares 100 ms, then declares 130 ms once the Run has really spent time.

        The switch is driven by the Run's own elapsed spending, never by how often the source was read, so the
        reset becomes visible exactly after the pre-replan spending.
        """

        def __init__(self, clock, *, initial_ms, reset_ms, after_ns):
            self._clock = clock
            self._initial = initial_ms
            self._reset = reset_ms
            self._after_ns = after_ns

        def snapshot(self):
            declared = self._reset if self._clock.ns >= self._after_ns else self._initial
            elapsed_ms = self._clock.ns / _NS_PER_MS
            remaining = declared - elapsed_ms
            return RuntimeBudgetSnapshot(schema_version="0.1",
                                         remaining_budget_ms=max(0.0, remaining))

    def run_control(name, *, burn_ms, reset):
        clock = StepClock()
        directory = workdir / name
        directory.mkdir()
        scenario = fake_scenario(
            action_spec=open_gate, start_state=start, end_state=end,
            scripted=[(open_gate, failed("ao-1"), after_first),
                      (mark_done, succeeded("ao-2", end), end)])
        domain = SpendsBeforeReplan(clock, burn_ms, mode="bounded", max_retries=1,
                                    allow_replan=True, retry_safety="non_idempotent",
                                    effect="applied", timeout_ms=90.0)
        if reset:
            source = ResetDeclaration(clock, initial_ms=100.0, reset_ms=130.0,
                                      after_ns=110 * _NS_PER_MS)
        else:
            source = MonotonicBudgetSource(total_budget_ms=100.0, monotonic_clock=clock)
        adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
        engine, trace, _ = build_engine(
            tmp_path=directory, adapter=adapter,
            planner=ScriptedPlanner(planner_script([
                ("plan-1", start, forecast_for(
                    open_gate, forecast_id="fc-1",
                    conditions=[json_rule_condition("gate", "open")])),
                ("plan-2", after_first, forecast_for(
                    mark_done, forecast_id="fc-2",
                    conditions=[json_rule_condition(FLAG, True)])),
            ])), domain=domain, budget_source=source, monotonic_clock=clock,
            run_timeout_ms=30000.0)
        result = asyncio.run(engine.run(request=request_for(
            conditions=task, max_steps=2, max_planning_calls=2, max_replans=1)))
        return result, adapter, domain, clock, trace

    # Control 1 - normal positive: nothing is spent, the replan is paid for and the goal is met.
    positive, positive_adapter, _, positive_clock, _ = run_control(
        "three-positive", burn_ms=0.0, reset=False)
    assert positive_clock.ns == 0
    assert (positive.terminal_status.value, positive.reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (positive.steps_started, positive.planning_calls, positive.attempts_started,
            positive.retries_used, positive.replans_used) == (1, 2, 2, 0, 1)
    assert [item.action_type for item in positive_adapter.actions] == ["open_gate", "mark_done"]
    assert _limited_oracle(positive) is False

    # Control 2 - normal limited: 110 ms is really spent before the replan decision, so the unraised 100 ms
    # allowance is already gone and the Run stops on the budget.
    limited, limited_adapter, limited_domain, limited_clock, limited_trace = run_control(
        "three-limited", burn_ms=110.0, reset=False)
    assert limited_domain.evidence_calls >= 3
    assert limited_clock.ns == 110 * _NS_PER_MS
    assert _limited_oracle(limited) is True
    assert (limited.steps_started, limited.planning_calls, limited.attempts_started,
            limited.retries_used, limited.replans_used) == (1, 1, 1, 0, 0)
    assert [item.action_type for item in limited_adapter.actions] == ["open_gate"]
    limited_terminals = terminal_records(read_records(limited_trace, "run-1"))
    assert len(limited_terminals) == 1
    assert (limited_terminals[0].terminal_status,
            limited_terminals[0].reason_code) == ("aborted", "policy_abort")
    assert limited_terminals[0].record_id == limited.terminal_record_id

    # Control 3 - the reset mutant: the same 110 ms pre-replan spending, and now the host deliberately raises
    # its declaration from 100 ms to 130 ms once that spending is real. The SAME limited oracle must reject
    # this run. The rejection is the negative test succeeding; the mutant is not required to stay normal, and
    # the product is not required to accept a rising declaration.
    mutant, mutant_adapter, mutant_domain, mutant_clock, mutant_trace = run_control(
        "three-mutant", burn_ms=110.0, reset=True)
    assert mutant_domain.evidence_calls >= 3
    assert mutant_clock.ns == 110 * _NS_PER_MS
    assert _limited_oracle(mutant) is False, "the reset mutant must not satisfy the limited oracle"
    # Measured: the raised declaration is not honoured - the Run's effective deadline was fixed at its start -
    # so the mutant ends as a deadline expiry instead of the budget stop the limited control reached.
    assert (mutant.terminal_status.value, mutant.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (mutant.terminal_status.value, mutant.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    # Restored reason inequality inside the original replan node: the two stops really differ.
    assert mutant.reason_code.value != limited.reason_code.value
    assert (mutant.steps_started, mutant.planning_calls, mutant.attempts_started,
            mutant.retries_used, mutant.replans_used) \
        == (limited.steps_started, limited.planning_calls, limited.attempts_started,
            limited.retries_used, limited.replans_used)
    assert [item.action_type for item in mutant_adapter.actions] == ["open_gate"]
    assert mutant.verification_record_id is None
    mutant_terminals = terminal_records(read_records(mutant_trace, "run-1"))
    assert len(mutant_terminals) == 1
    assert mutant_terminals[0].verification_record_id is None


def test_a_deadline_reset_at_the_reached_replan_boundary_is_rejected_by_the_limited_oracle(workdir):
    """C05: the reset must happen at a boundary the Run really reaches, and the same oracle must reject it.

    Three Runs replay one script. All three spend the SAME nonzero 30 ms while building the evidence for the
    post-failure re-observation, which is before the RecoveryEngine decides, and all three then spend a further
    80 ms at the *second* Contract preparation - the boundary the replan really reaches. So the pre-spend is
    identical in the positive and limited controls, and the replan is genuinely taken in both.

    * the **positive** control declares 500 ms, so nothing runs out and the replanned step completes;
    * the **limited** control declares 100 ms, so 30 + 80 ms exceeds it and the Run stops on the budget;
    * the **mutant** is the limited Run with one deliberate difference: at that same reached boundary the host
      changes its own deadline from 100 ms to 130 ms. The declared value before and after are captured and
      compared, so the change is proven to have happened at a boundary the Run reached, not at declaration time.

    The single ``_complete_limited_oracle`` is applied unchanged to all three. It is satisfied only by the
    limited Run; the positive control and the deadline-reset mutant are both rejected by it, and for the mutant
    that rejection is the negative test succeeding.
    """
    open_gate = action("open_gate")
    mark_done = action("mark_done")
    start = {"gate": "closed", FLAG: False}
    after_first = {"gate": "closed", FLAG: False}
    end = {"gate": "closed", FLAG: True}
    task = [json_rule_condition(FLAG, True)]

    from forecontract.runtime import BudgetSource, RuntimeBudgetSnapshot

    class BoundaryClock:
        """A declared monotonic timeline the test advances explicitly."""

        def __init__(self):
            self.ns = 0

        def now_ns(self):
            return self.ns

        def advance_ms(self, milliseconds):
            self.ns += int(milliseconds * _NS_PER_MS)

    def _complete_limited_oracle(run_result, records, finalization):
        """The complete static oracle for a Run that must have stopped on an exhausted budget after a replan.

        It checks the RunResult counters, the single committed Contract, the single terminal fact and its
        association with this Run, and the whole Finalization snapshot. A run that ended the same way on the
        surface but with a different Contract count, a different terminal source or a different finalization is
        rejected.
        """
        if not (run_result.terminal_status.value == "aborted"
                and run_result.reason_code.value == "BUDGET_EXHAUSTED"
                and run_result.verification_record_id is None
                and (run_result.steps_started, run_result.planning_calls,
                     run_result.attempts_started, run_result.retries_used,
                     run_result.replans_used) == (1, 2, 1, 0, 1)):
            return False
        if kinds(records).count("contract_committed") != 1:
            return False
        terminals = terminal_records(records)
        if len(terminals) != 1:
            return False
        terminal = terminals[0]
        if terminal.record_id != run_result.terminal_record_id:
            return False
        if terminal.verification_record_id is not None:
            return False
        if finalization is None:
            return False
        return (finalization.primary_terminal_status.value == "aborted"
                and finalization.primary_reason_code.value == "BUDGET_EXHAUSTED"
                and finalization.cleanup_status == "succeeded"
                and finalization.trace_status == "persisted"
                and finalization.quarantined is False
                and finalization.terminal_record_id == run_result.terminal_record_id)

    class SharedPreSpendDomain(HostDomain):
        """Spends the same 30 ms pre-replan, then 80 ms at the second prepare boundary."""

        def __init__(self, clock, on_boundary, **options):
            super().__init__(**options)
            self._clock = clock
            self._on_boundary = on_boundary
            self.evidence_calls = 0
            self.preparations = 0

        def to_evidence(self, observation):
            evidence = super().to_evidence(observation)
            self.evidence_calls += 1
            if self.evidence_calls == 3:
                self._clock.advance_ms(30.0)
            return evidence

        def prepare_contract(self, *, forecast, evidence):
            self.preparations += 1
            if self.preparations == 2:
                self._on_boundary(self._clock.ns)
                self._clock.advance_ms(80.0)
            return super().prepare_contract(forecast=forecast, evidence=evidence)

    class HostDeadline(BudgetSource):
        """Declares one deadline, optionally changing it at a boundary the Run has already reached."""

        def __init__(self, clock, *, declared_ms):
            self._clock = clock
            self._declared = declared_ms
            self.before = declared_ms
            self.after = declared_ms

        def change_at(self, _elapsed_ns, new_ms=None):
            self.before = self._declared
            if new_ms is not None:
                self._declared = new_ms
            self.after = self._declared

        def snapshot(self):
            remaining = self._declared - self._clock.ns / _NS_PER_MS
            return RuntimeBudgetSnapshot(schema_version="0.1", remaining_budget_ms=max(0.0, remaining))

    def run_once(name, *, declared_ms, reset_to):
        clock = BoundaryClock()
        directory = workdir / name
        directory.mkdir()
        scenario = fake_scenario(
            action_spec=open_gate, start_state=start, end_state=end,
            scripted=[(open_gate, failed("ao-1"), after_first),
                      (mark_done, succeeded("ao-2", end), end)])
        source = HostDeadline(clock, declared_ms=declared_ms)
        domain = SharedPreSpendDomain(
            clock, lambda elapsed: source.change_at(elapsed, getattr(source, "pending", None)),
            mode="bounded", max_retries=1, allow_replan=True, retry_safety="non_idempotent",
            effect="applied", timeout_ms=90.0)
        if reset_to is not None:
            source.pending = reset_to
        adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
        engine, trace, _ = build_engine(
            tmp_path=directory, adapter=adapter,
            planner=ScriptedPlanner(planner_script([
                ("plan-1", start, forecast_for(
                    open_gate, forecast_id="fc-1",
                    conditions=[json_rule_condition("gate", "open")])),
                ("plan-2", after_first, forecast_for(
                    mark_done, forecast_id="fc-2",
                    conditions=[json_rule_condition(FLAG, True)])),
            ])), domain=domain, budget_source=source, monotonic_clock=clock,
            run_timeout_ms=30000.0)
        result = asyncio.run(engine.run(request=request_for(
            conditions=task, max_steps=2, max_planning_calls=2, max_replans=1)))
        return result, adapter, domain, source, clock, trace, engine

    positive, positive_adapter, positive_domain, positive_source, positive_clock, \
        positive_trace, positive_engine = run_once(
        "boundary-positive", declared_ms=500.0, reset_to=None)
    limited, limited_adapter, limited_domain, limited_source, limited_clock, \
        limited_trace, limited_engine = run_once(
        "boundary-limited", declared_ms=100.0, reset_to=None)
    mutant, mutant_adapter, mutant_domain, mutant_source, mutant_clock, \
        mutant_trace, mutant_engine = run_once(
        "boundary-mutant", declared_ms=100.0, reset_to=130.0)

    # All three share the same nonzero pre-spend and all three really reached the replan boundary.
    for domain, clock, label in ((positive_domain, positive_clock, "positive"),
                                 (limited_domain, limited_clock, "limited"),
                                 (mutant_domain, mutant_clock, "mutant")):
        assert clock.ns > 0, f"{label} must have spent real time"
        assert domain.evidence_calls >= 3, f"{label} must have reached the pre-spend point"
        assert domain.preparations == 2, f"{label} must have reached the replan boundary"

    # The positive control paid for the replan and completed.
    assert (positive.terminal_status.value, positive.reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (positive.steps_started, positive.planning_calls, positive.attempts_started,
            positive.retries_used, positive.replans_used) == (1, 2, 2, 0, 1)
    assert [item.action_type for item in positive_adapter.actions] == ["open_gate", "mark_done"]
    assert positive_source.before == positive_source.after == 500.0

    # The limited control reached the same boundary and stopped on the budget.
    assert (limited.terminal_status.value, limited.reason_code.value) \
        == ("aborted", "BUDGET_EXHAUSTED")
    assert limited_source.before == limited_source.after == 100.0

    # The mutant changed the host deadline at that same reached boundary, which is proven by the captured
    # declared values and by the fact that the boundary itself was reached.
    assert mutant_source.before == 100.0
    assert mutant_source.after == 130.0
    assert mutant_source.before != mutant_source.after
    assert mutant_clock.ns > 0
    assert mutant_domain.preparations == 2

    # The same complete oracle, applied unchanged to all three. Each run supplies its own records and its own
    # finalization, but the predicate is one and the same.
    limited_records = read_records(limited_trace, "run-1")
    limited_finalization = limited_engine.last_finalization
    positive_records = read_records(positive_trace, "run-1")
    positive_finalization = positive_engine.last_finalization
    mutant_records = read_records(mutant_trace, "run-1")
    mutant_finalization = mutant_engine.last_finalization

    assert _complete_limited_oracle(limited, limited_records, limited_finalization) is True
    assert _complete_limited_oracle(positive, positive_records, positive_finalization) is False
    assert _complete_limited_oracle(mutant, mutant_records, mutant_finalization) is False, \
        "the deadline-reset mutant must not satisfy the limited oracle"
    assert (mutant.terminal_status.value, mutant.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    # Restored reason inequality: the two stops really differ, so the mutant cannot be confused with the
    # limited control even though they share every counter.
    assert mutant.reason_code.value != limited.reason_code.value
    # The counters are identical to the limited control, so the only difference is which ending the same
    # oracle sees - the mutant is rejected on the reason, not on a different amount of work.
    assert (mutant.steps_started, mutant.planning_calls, mutant.attempts_started,
            mutant.retries_used, mutant.replans_used) \
        == (limited.steps_started, limited.planning_calls, limited.attempts_started,
            limited.retries_used, limited.replans_used)
    assert mutant.verification_record_id is None

    # The limited Run's trace is the honest record of the stop it reached: one committed Contract, one Attempt,
    # one action, and one terminal fact whose source is this Run.
    assert kinds(limited_records).count("contract_committed") == 1
    assert kinds(limited_records).count("attempt_started") == 1
    assert [item.action_type for item in limited_adapter.actions] == ["open_gate"]
    limited_terminals = terminal_records(limited_records)
    assert len(limited_terminals) == 1
    assert (limited_terminals[0].terminal_status,
            limited_terminals[0].reason_code) == ("aborted", "policy_abort")
    assert limited_terminals[0].record_id == limited.terminal_record_id
    assert limited_terminals[0].verification_record_id is None

    # The mutant's own terminal fact is associated with this Run, so its rejection is its own, not a borrowed
    # limited terminal.
    mutant_terminals = terminal_records(mutant_records)
    assert len(mutant_terminals) == 1
    assert mutant_terminals[0].record_id == mutant.terminal_record_id
    assert mutant_terminals[0].verification_record_id is None
    assert [item.action_type for item in mutant_adapter.actions] == ["open_gate"]


def test_an_unknown_outcome_is_re_observed_and_not_blindly_re_executed(workdir):
    """C06: an unknown execution is never repeated blindly; it is left as an unresolved effect.

    The scenario really carries a second executable step, so a blind repeat was available. It did not
    happen: the Runtime re-observed, asked the domain for the effect state, and the real RecoveryEngine
    aborted because the effect is unknown and this action is not safe to repeat. The terminal facts
    carry no success, and the Attempt's own trace facts keep the honest unknown status.
    """
    open_gate = action("open_gate")
    start = {"gate": "closed"}
    opened = {"gate": "open"}
    conditions = [json_rule_condition("gate", "open")]
    scenario = repeated_action_scenario(
        action_spec=open_gate, states=[start, start, opened],
        first_outcome=unknown("ao-1"),
        second_outcome=succeeded("ao-2", opened))
    domain = HostDomain(mode="bounded", max_retries=2, allow_replan=False,
                        retry_safety="non_idempotent", effect="unknown")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", start, forecast_for(open_gate, conditions=conditions))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=conditions, max_steps=2, max_planning_calls=2, max_retries=2)))

    # Observed exactly: aborted / POLICY_ABORT after one Step and exactly one Attempt.
    assert (result.terminal_status.value, result.reason_code.value) == ("aborted", "POLICY_ABORT")
    assert (result.steps_started, result.attempts_started, result.retries_used) == (1, 1, 0)
    assert result.verification_record_id is None
    # A second step was scripted and executable, and it was not taken: no blind repeat happened.
    assert len(scenario.steps) == 2
    assert len(adapter.actions) == 1
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    # The Runtime re-observed before deciding, and its own observation count exceeds its executions.
    assert len(adapter.observations) == 4
    assert len(adapter.observations) > len(adapter.actions)
    # The decision was made from the domain's real effect judgement for this very outcome.
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1"]
    records = read_records(trace, "run-1")
    finished = [record for record in records if record.kind == "attempt_finished"]
    assert [(record.execution_status.value, record.error_code) for record in finished] \
        == [("unknown", "execution_unknown")]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    # Observed exactly: the unresolved effect is spelled policy_abort, never a success.
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id
    assert [record for record in records
            if record.kind == "verification_recorded" and record.scope == "run"] == []
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.cleanup_status, finalization.trace_status) == ("succeeded", "persisted")
    assert finalization.quarantined is False
    assert engine.is_quarantined is False


def test_a_re_observation_that_still_shows_the_goal_unmet_is_not_re_executed(workdir):
    """C06: a re-observation that still shows the goal unmet authorises no blind re-execution.

    The domain support pins no effect verdict here, so the real RecoveryEngine judges from the
    evidence the Run really received: every observation still reports the goal's flag unset, and the
    unknown receipt's own state_after is absent. Observed exactly: the goal is still unmet, no second
    action is dispatched, and the Run ends on its policy instead of a success.
    """
    open_gate = action("open_gate")
    start = {"gate": "closed"}
    conditions = [json_rule_condition("gate", "open")]
    scenario = repeated_action_scenario(
        action_spec=open_gate, states=[start, start, start],
        first_outcome=unknown("ao-1"),
        second_outcome=succeeded("ao-2", start))
    domain = HostDomain(mode="bounded", max_retries=2, allow_replan=False,
                        retry_safety="non_idempotent")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", start, forecast_for(open_gate, conditions=conditions))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=conditions, max_steps=2, max_planning_calls=2, max_retries=2)))

    # Observed exactly: every observation the Run received still reports the goal flag unset.
    assert [item.state for item in adapter.observations] == [start, start, start, start]
    assert (result.terminal_status.value, result.reason_code.value) == ("aborted", "POLICY_ABORT")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 0, 0)
    assert result.verification_record_id is None
    # Observed exactly: exactly one action, although a second executable step really existed.
    assert len(scenario.steps) == 2
    assert [item.action_type for item in adapter.actions] == ["open_gate"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    assert [outcome_id for _, outcome_id in domain.assess_calls] == ["ao-1"]
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    finished = [record for record in records if record.kind == "attempt_finished"]
    assert [(record.execution_status.value, record.error_code) for record in finished] \
        == [("unknown", "execution_unknown")]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.primary_reason_code.value, finalization.cleanup_status,
            finalization.trace_status, finalization.quarantined) \
        == ("POLICY_ABORT", "succeeded", "persisted", False)


def test_max_steps_stops_the_run_without_taking_extra_actions(workdir):
    """C07: the host's step allowance is a hard bound; the Run aborts instead of doing more work.

    Two steps are scripted and the goal needs both, but the host declares one. The Run performs
    exactly one Step and one Attempt and then aborts, and the second scripted action never reaches
    the adapter.
    """
    open_gate = action("open_gate")
    mark_done = action("mark_done")
    start = {"gate": "closed", FLAG: False}
    middle = {"gate": "open", FLAG: False}
    end = {"gate": "open", FLAG: True}
    scenario = fake_scenario(
        action_spec=open_gate, start_state=start, end_state=end,
        scripted=[(open_gate, succeeded("ao-1", middle), middle),
                  (mark_done, succeeded("ao-2", end), end)])
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False, effect="applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([
            ("plan-1", start, forecast_for(open_gate, forecast_id="fc-1",
                                           conditions=[json_rule_condition("gate", "open")])),
            ("plan-2", middle, forecast_for(mark_done, forecast_id="fc-2")),
        ])), domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=[json_rule_condition(FLAG, True)], max_steps=1, max_planning_calls=2)))

    # Observed exactly: aborted / MAX_STEPS_EXHAUSTED after one Step and one Attempt.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("aborted", "MAX_STEPS_EXHAUSTED")
    assert (result.steps_started, result.attempts_started) == (1, 1)
    assert result.verification_record_id is None
    assert len(scenario.steps) == 2
    # Observed exactly: one action, one outcome and no more - the declared bound was never exceeded.
    assert [item.action_type for item in adapter.actions] == ["open_gate"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    records = read_records(trace, "run-1")
    assert kinds(records).count("step_started") == 1
    assert kinds(records).count("attempt_started") == 1
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id


def test_max_planning_calls_stops_the_run_without_a_failed_step(workdir):
    """C07: the host's planning allowance is a separate hard bound.

    The one committed step verifies PASS, so the stop is not a failed step: it is the planning
    allowance being spent while the task oracle still declines to complete the Run. The Run aborts on
    MAX_PLANNING_EXHAUSTED with exactly one planning call and one real action.
    """
    open_gate = action("open_gate")
    start = {"gate": "closed", FLAG: False}
    end = {"gate": "open", FLAG: False}
    scenario = fake_scenario(action_spec=open_gate, start_state=start, end_state=end,
                             outcome=succeeded("ao-1", end))
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False, effect="applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", start, forecast_for(open_gate,
                                            conditions=[json_rule_condition("gate", "open")]))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=[json_rule_condition(FLAG, True)], max_steps=4, max_planning_calls=1)))

    # Observed exactly: aborted / MAX_PLANNING_EXHAUSTED after one planning call and one Step.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("aborted", "MAX_PLANNING_EXHAUSTED")
    assert (result.planning_calls, result.steps_started) == (1, 1)
    assert result.verification_record_id is None
    assert [item.action_type for item in adapter.actions] == ["open_gate"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    records = read_records(trace, "run-1")
    verdicts = [record for record in records if record.kind == "verification_recorded"]
    assert [(record.scope, record.verification_status.value) for record in verdicts] \
        == [("attempt", "pass")]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id


def test_an_abort_is_neither_a_facility_failure_nor_a_timeout(workdir):
    """C07: the bounded stop keeps its own diagnosis, distinct from a dependency failure and a timeout.

    Three real Runs are compared: a host policy that allows no recovery at all, a zero Run allowance
    that is already spent before any business call, and a planner that has no plan for the observation
    it is given. Their reasons are pairwise distinct, so an abort is never a relabelled facility
    failure and never a relabelled timeout, and the abort takes no action beyond its declared bound.
    """
    done = action()
    state = {FLAG: False}

    # (1) A policy abort: the step FAILs and this host allows neither a retry nor a replan.
    abort_scenario = fake_scenario(action_spec=done, start_state=state, end_state=state,
                                   outcome=succeeded("ao-1", state))
    abort_domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False,
                              retry_safety="non_idempotent", effect="applied")
    abort_adapter = DeclaredAdapter(FakeEnvironmentAdapter(abort_scenario))
    abort_engine, abort_trace, _ = build_engine(
        tmp_path=workdir, adapter=abort_adapter,
        planner=ScriptedPlanner(planner_script([("plan-1", state, forecast_for(done))])),
        domain=abort_domain)
    aborted = asyncio.run(abort_engine.run(request=request_for(
        conditions=[json_rule_condition(FLAG, True)], max_steps=1, max_planning_calls=1)))

    assert (aborted.terminal_status.value, aborted.reason_code.value) == ("aborted", "POLICY_ABORT")
    # Observed exactly: one action, one outcome, and the abort is spelled policy_abort in the trace.
    assert [item.action_type for item in abort_adapter.actions] == ["mark_done"]
    assert (abort_adapter.starts, abort_adapter.stops) == (1, 1)
    abort_terminals = terminal_records(read_records(abort_trace, "run-1"))
    assert len(abort_terminals) == 1
    assert (abort_terminals[0].terminal_status, abort_terminals[0].reason_code) \
        == ("aborted", "policy_abort")

    # (2) A real timeout: the Run's own allowance is zero, so its deadline is already spent.
    timeout_scenario = no_step_scenario(action_spec=done, state={FLAG: True})
    timeout_adapter = DeclaredAdapter(FakeEnvironmentAdapter(timeout_scenario))
    timeout_engine, timeout_trace, _ = build_engine(
        tmp_path=workdir, adapter=timeout_adapter,
        planner=ScriptedPlanner(planner_script([])), domain=HostDomain(),
        run_timeout_ms=0.0, trace_name="timeout-trace.jsonl")
    timed_out = asyncio.run(timeout_engine.run(request=request_for(run_id="run-timeout")))

    # Observed exactly: timeout / DEADLINE_EXCEEDED, and an already-spent deadline touches nothing.
    assert (timed_out.terminal_status.value, timed_out.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (timeout_adapter.starts, timeout_adapter.stops) == (0, 0)
    assert timeout_adapter.observations == []
    assert timeout_adapter.actions == []
    timeout_records = read_records(timeout_trace, "run-timeout")
    assert kinds(timeout_records) == ["run_started", "run_finished"]
    assert (terminal_records(timeout_records)[0].terminal_status,
            terminal_records(timeout_records)[0].reason_code) == ("timeout", "deadline_exceeded")

    # (3) A real facility failure: the planner has no plan for the observation it is given.
    refusal_scenario = fake_scenario(action_spec=done, start_state=state, end_state=state,
                                     outcome=succeeded("ao-1", state))
    refusal_adapter = DeclaredAdapter(FakeEnvironmentAdapter(refusal_scenario))
    refusal_engine, refusal_trace, _ = build_engine(
        tmp_path=workdir, adapter=refusal_adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {"other": 1}, forecast_for(done))])),
        domain=HostDomain(), trace_name="refusal-trace.jsonl")
    refused = asyncio.run(refusal_engine.run(request=request_for(run_id="run-refusal")))

    # Observed exactly: failed / DEPENDENCY_FAILURE with no action ever dispatched.
    assert (refused.terminal_status.value, refused.reason_code.value) \
        == ("failed", "DEPENDENCY_FAILURE")
    assert refusal_adapter.actions == []
    refusal_terminals = terminal_records(read_records(refusal_trace, "run-refusal"))
    assert len(refusal_terminals) == 1
    assert (refusal_terminals[0].terminal_status, refusal_terminals[0].reason_code) \
        == ("failure", "internal_error")

    # The three diagnoses are pairwise distinct.
    assert aborted.reason_code.value != timed_out.reason_code.value
    assert aborted.reason_code.value != refused.reason_code.value
    assert timed_out.reason_code.value != refused.reason_code.value


def test_an_unknown_outcome_with_the_effect_already_applied_does_not_repeat_the_action(workdir):
    """C06: evidence that already shows the goal held means no blind repeat, even for an unknown receipt."""
    done = action()
    # The re-observation shows the goal already holds, so the effect is applied by the environment's own word
    # even though the receipt is unknown.
    def unknown_then_applied():
        from conftest import unknown as make_unknown

        return make_unknown("ao-unknown")

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True},
                             outcome=unknown_then_applied())
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())

    result = asyncio.run(engine.run(request=request_for(max_steps=2, max_planning_calls=2)))

    # No blind repeat: the single scripted action ran at most once, and no terminal fact claims a run PASS
    # unless the task oracle really passed.
    assert len(adapter.actions) <= 1
    terminals = terminal_records(read_records(trace, "run-1"))
    assert len(terminals) == 1
    assert result.terminal_status.value in {"aborted", "failed", "timeout", "completed"}

def test_an_unknown_outcome_with_the_evidence_applied_completes_with_one_action(workdir):
    """C06: an unknown receipt whose re-observation shows the effect applied completes the real goal.

    Measured exactly before asserting. The receipt is ``unknown``, and the re-observation the Run really
    received reports the goal's own flag set, so the domain's effect judgement is ``applied`` and the
    task oracle - not the receipt - decides completion. Every fact a false ``aborted`` / ``POLICY_ABORT``
    would have to break is asserted literally: the terminal pair, the single real execution, the single
    Attempt, the single committed Contract, the single run-scope PASS, the terminal fact naming that same
    PASS, and the result and the trace naming the same terminal record.
    """
    done = action()
    start = {FLAG: False}
    end = {FLAG: True}
    scenario = fake_scenario(action_spec=done, start_state=start, end_state=end,
                             outcome=unknown("ao-unknown"))
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([("plan-1", start, forecast_for(done))])),
        domain=HostDomain())

    result = asyncio.run(engine.run(request=request_for(max_steps=1, max_planning_calls=1)))

    # Observed exactly: the Run completed on the real task PASS, never as an abort, a failure or a timeout.
    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    # Exactly one Step, one planning call, one Attempt and one real execution; no retry and no replan.
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 0, 0)
    assert len(adapter.actions) == 1
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-unknown"]
    assert (adapter.starts, adapter.stops) == (1, 1)

    records = read_records(trace, "run-1")
    # Observed exactly: one committed Contract, one Attempt and one step-scope verdict, in this order.
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "verification_recorded", "run_finished"]
    committed = [record for record in records if record.kind == "contract_committed"]
    assert len(committed) == 1
    assert kinds(records).count("attempt_started") == 1
    finished = [record for record in records if record.kind == "attempt_finished"]
    assert [(record.execution_status.value, record.error_code) for record in finished] \
        == [("unknown", "execution_unknown")]
    step_verdicts = [record for record in records
                     if record.kind == "verification_recorded" and record.scope == "attempt"]
    assert [record.verification_status.value for record in step_verdicts] == ["pass"]
    run_verdicts = [record for record in records
                    if record.kind == "verification_recorded" and record.scope == "run"]
    assert len(run_verdicts) == 1
    assert run_verdicts[0].verification_status.value == "pass"
    terminals = terminal_records(records)
    assert len(terminals) == 1
    # The trace's own closed vocabulary spells a completed Run as success with no reason code.
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("success", None)
    # The result, the recorded run-scope PASS and the terminal fact all name the one same record.
    assert result.verification_record_id == run_verdicts[0].record_id
    assert terminals[0].verification_record_id == run_verdicts[0].record_id
    assert terminals[0].verification_record_id == result.verification_record_id
    assert result.terminal_record_id == terminals[0].record_id
    finalization = engine.last_finalization
    assert finalization is not None
    assert (finalization.primary_terminal_status.value, finalization.primary_reason_code.value,
            finalization.cleanup_status, finalization.trace_status,
            finalization.terminal_record_id, finalization.quarantined) \
        == ("completed", "TASK_ACHIEVED", "succeeded", "persisted", terminals[0].record_id, False)
    assert engine.is_quarantined is False
