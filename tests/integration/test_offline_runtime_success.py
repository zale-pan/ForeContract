"""C01 and C02: the closed success loop, and the two ways a Run must not claim success.

These cases drive the real Runtime over the real Fake, Planner, Builder, Verifier and Oracle, and read the
facts back from the JSONL trace rather than from any in-memory recorder. Every expected fact is declared by
the test before the run; identifiers are allowed to vary, but the association graph, the counts, the result
labels and the uniqueness of the terminal fact are asserted independently.
"""

from __future__ import annotations

import asyncio

from conftest import (
    FLAG,
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
    planner_script,
    read_records,
    request_for,
    succeeded,
    terminal_records,
)
from forecontract_fake import FakeEnvironmentAdapter
from forecontract_scripted import ScriptedPlanner


def test_an_already_satisfied_goal_performs_no_execution(workdir):
    """C01: when the first real observation already satisfies the oracle, nothing is executed."""
    done = action()
    scenario = no_step_scenario(action_spec=done, state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(tmp_path=workdir, adapter=adapter,
                                    planner=ScriptedPlanner(planner_script([])), domain=domain)

    result = asyncio.run(engine.run(request=request_for(conditions=[json_rule_condition()])))

    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    # The goal was met by the pre-execution observation: no step, no attempt, no execution, no planning.
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (0, 0, 0)
    assert adapter.actions == []
    assert domain.assess_calls == []
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "verification_recorded", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert terminals[0].terminal_status == "success"
    assert terminals[0].verification_record_id == result.verification_record_id


def test_a_single_real_action_completes_the_goal(workdir):
    """C01: one real action and exactly one real Attempt, and only the oracle decides completion."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain)

    result = asyncio.run(engine.run(request=request_for()))

    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 0, 0)
    assert result.verification_record_id is not None
    assert result.terminal_record_id is not None
    # Exactly one real execution reached the adapter, and it is the action the planner proposed.
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["fake-outcome-1"]

    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "verification_recorded", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    finished = terminals[0]
    assert finished.terminal_status == "success"
    # The terminal fact names a run-scope PASS that really exists in the same trace.
    passes = [record for record in records
              if record.kind == "verification_recorded" and record.scope == "run"]
    assert len(passes) == 1
    assert finished.verification_record_id == passes[0].record_id
    assert {record.run_id for record in records} == {"run-1"}
    identifiers = [record.record_id for record in records]
    assert len(set(identifiers)) == len(identifiers)
    assert [record.sequence for record in records] == list(range(len(records)))


def test_several_actions_are_counted_exactly_as_scripted(workdir):
    """C01: a two-action script produces exactly two steps and two real executions."""
    open_gate = action("open_gate")
    mark_done = action("mark_done")
    start = {FLAG: False}
    middle = {"gate": "open"}
    end = {"gate": "open", FLAG: True}

    scenario = fake_scenario(
        action_spec=open_gate, start_state=start, end_state=end,
        scripted=[(open_gate, succeeded("ao-1", middle), middle),
                  (mark_done, succeeded("ao-2", end), end)])
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([
            ("plan-1", start, forecast_for(open_gate, forecast_id="fc-1",
                                           conditions=[json_rule_condition("gate", "open")])),
            ("plan-2", middle, forecast_for(mark_done, forecast_id="fc-2",
                                            conditions=[json_rule_condition()])),
        ])), domain=domain)

    result = asyncio.run(engine.run(request=request_for(max_steps=3, max_planning_calls=2)))

    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (2, 2, 2)
    assert [item.action_type for item in adapter.actions] == ["open_gate", "mark_done"]
    records = read_records(trace, "run-1")
    assert kinds(records).count("step_started") == 2
    assert kinds(records).count("contract_committed") == 2
    assert kinds(records).count("attempt_started") == 2
    assert len(terminal_records(records)) == 1


def test_a_verified_step_whose_goal_still_does_not_hold_is_not_a_success(workdir):
    """C02: a step PASS is not a completion; only the task oracle may complete a Run."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: False},
                             outcome=failed("ao-1"))
    domain = HostDomain(effect="applied")
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain)

    result = asyncio.run(engine.run(request=request_for(max_steps=1, max_planning_calls=1)))

    assert result.terminal_status.value != "completed"
    assert result.reason_code.value != "TASK_ACHIEVED"
    records = read_records(trace, "run-1")
    assert len(terminal_records(records)) == 1
    # No terminal fact claims a run-scope PASS.
    assert terminal_records(records)[0].verification_record_id is None


def test_a_planner_completion_hint_never_completes_a_goal(workdir):
    """C02: ``task_completion_hint=True`` is a proposal, never a completion signal."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: False},
                             outcome=failed("ao-1"))
    domain = HostDomain(effect="applied")
    engine, _trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done, hint=True))])), domain=domain)
    result = asyncio.run(engine.run(request=request_for(max_steps=1, max_planning_calls=1)))

    assert result.terminal_status.value != "completed"
    assert result.reason_code.value != "TASK_ACHIEVED"
    # The hint added no completion signal to the schema: RunResult still carries only its approved fields.
    assert set(type(result).model_fields) == {
        "schema_version", "run_id", "goal_id", "terminal_status", "reason_code",
        "steps_started", "planning_calls", "attempts_started", "retries_used", "replans_used",
        "verification_record_id", "terminal_record_id"}
