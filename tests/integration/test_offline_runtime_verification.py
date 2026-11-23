"""C03: the verification semantics of one Run, from the completion conditions to the two entries.

Every case drives the real Runtime over the real Fake, Planner, Builder, StepVerifier and
TaskCompletionOracle, and reads its facts back from the JSONL trace. The cases keep four things
apart that are easy to conflate:

* a required completion condition that FAILs is not a success, whatever else holds;
* a path the observed state does not reach is INCONCLUSIVE, or a definite FAIL once the domain
  declares the state complete - never a PASS, and a missing field is not a damaged structure;
* a damaged evidence structure is a facility refusal rather than a verdict at all;
* a step PASS and a task PASS are decided by two independent entries, and only the second one may
  complete a Run.

One case (F02) leaves the Runtime out of it on purpose: it calls the domain binding's own
``assess_effect`` directly and compares the whole evidence-by-status matrix against a literal table,
for both the example's binding and this suite's.

Every expected fact is declared by the test itself; identifiers come from the run, and the counts,
labels and read-back records are asserted independently.
"""

from __future__ import annotations

import asyncio
import importlib.util
from types import SimpleNamespace

from conftest import (
    FLAG,
    REPO_ROOT,
    SCHEMA,
    DeclaredAdapter,
    HostDomain,
    action,
    build_engine,
    fake_scenario,
    forecast_for,
    json_rule_condition,
    kinds,
    no_step_scenario,
    observation,
    outcome_status,
    path_exists_condition,
    planner_script,
    read_records,
    request_for,
    succeeded,
    terminal_records,
)
from forecontract.models import Observation
from forecontract.verification import VerificationEvidence
from forecontract_fake import FakeEnvironmentAdapter
from forecontract_scripted import ScriptedPlanner

#: The example the CLI cases run, loaded here as a module so its own ``DemoDomain`` can be called
#: directly. The file is read, never modified: loading it is how this case proves the shipped example
#: really carries the binding the matrix describes.
DEMO_PATH = REPO_ROOT / "examples" / "offline_runtime_demo.py"

#: The observation states of the matrix: the effect field set, the effect field unset, and an
#: observation that does not carry the field at all.
EFFECT_STATES = {
    "flag_set": {FLAG: True},
    "flag_unset": {FLAG: False},
    "no_field": {"other": True},
}

#: The three closed execution statuses of an actual outcome.
EFFECT_STATUSES = ("succeeded", "failed", "unknown")

#: The exact verdict each matrix cell must return, declared before the calls are made. The receipt's own
#: status never decides a cell whose observation carries the field, and a missing field is ``unknown``
#: whatever the receipt says.
EXPECTED_EFFECTS = {
    ("flag_set", "succeeded"): "applied",
    ("flag_set", "failed"): "applied",
    ("flag_set", "unknown"): "applied",
    ("flag_unset", "succeeded"): "not_applied",
    ("flag_unset", "failed"): "not_applied",
    ("flag_unset", "unknown"): "not_applied",
    ("no_field", "succeeded"): "unknown",
    ("no_field", "failed"): "unknown",
    ("no_field", "unknown"): "unknown",
}


def load_demo_domain():
    """Load the shipped example as a module and return its own ``DemoDomain`` class."""
    spec = importlib.util.spec_from_file_location("forecontract_offline_runtime_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DemoDomain



def test_the_domain_effect_judgement_answers_every_evidence_and_status_cell_directly():
    """F02: the domain binding's own ``assess_effect``, called directly, for every matrix cell.

    The two bindings are the example's ``DemoDomain`` - loaded from the very file the CLI cases run -
    and this suite's ``HostDomain``. Neither is driven through the Runtime here: the call is made
    directly, with an evidence record whose observation either carries the effect field set, carries it
    unset, or does not carry it at all, crossed with the three closed execution statuses. The complete
    matrix is compared to a literal table, so every cell is an exact, falsifiable answer:

    * an observation that really shows the effect applied says ``applied`` even when the receipt is
      ``failed`` - a failed receipt may not downgrade evidence the environment already supplied;
    * an observation that shows the field unset says ``not_applied`` for every status;
    * an observation that does not carry the field at all says ``unknown`` for every status, which is a
      statement about the evidence rather than a guess from the receipt.
    """
    demo_domain = load_demo_domain()()
    host_domain = HostDomain()
    contract = SimpleNamespace(contract_id="rt-contract-matrix")

    def matrix_for(domain) -> dict[tuple[str, str], str]:
        observed: dict[tuple[str, str], str] = {}
        for state_name, state in EFFECT_STATES.items():
            for status in EFFECT_STATUSES:
                evidence = VerificationEvidence(
                    schema_version=SCHEMA, observation=observation(f"obs-{state_name}", state),
                    state_complete=True)
                outcome = outcome_status(f"ao-{state_name}-{status}", status)
                observed[(state_name, status)] = domain.assess_effect(
                    contract=contract, actual_outcome=outcome, evidence=evidence)
        return observed

    demo_effects = matrix_for(demo_domain)
    host_effects = matrix_for(host_domain)

    # Every cell of the literal table, for both bindings.
    assert demo_effects == EXPECTED_EFFECTS
    assert host_effects == EXPECTED_EFFECTS
    assert demo_effects == host_effects
    # The two cells the contract calls out explicitly, asserted on their own as well.
    assert demo_effects[("flag_set", "failed")] == "applied"
    assert host_effects[("flag_set", "failed")] == "applied"
    assert demo_effects[("no_field", "succeeded")] == "unknown"
    assert host_effects[("no_field", "succeeded")] == "unknown"
    # The nine cells are genuinely distinct calls, not one answer reused: every cell really was assessed.
    assert len(host_effects) == 9
    assert [outcome_id for _, outcome_id in host_domain.assess_calls] == [
        f"ao-{state_name}-{status}"
        for state_name in EFFECT_STATES for status in EFFECT_STATUSES]


def test_a_failing_required_condition_cannot_complete_the_run(workdir):
    """C03: a required condition the real observation does not satisfy stops the Run short.

    The state is legal and observed; it simply does not satisfy the declared condition. The step
    verdict is FAIL, the real RecoveryEngine refuses a recovery because this host allows neither a
    retry nor a replan, and the Run ends as a bounded policy abort. No terminal fact may claim a
    run-scope PASS, because no run-scope PASS was ever recorded.
    """
    done = action()
    state = {FLAG: False}
    scenario = fake_scenario(action_spec=done, start_state=state, end_state=state,
                             outcome=succeeded("ao-1", state))
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False, effect="applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script([("plan-1", state, forecast_for(done))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=[json_rule_condition(FLAG, True)], max_steps=1, max_planning_calls=1)))

    # Observed terminal diagnosis: with no retry and no replan the recovery assessment is
    # RECOVERY_NOT_ALLOWED, and the engine publishes a recovery abort as this bounded policy abort.
    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "POLICY_ABORT"
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 0, 0)
    assert result.verification_record_id is None
    # The real action ran exactly once, and the observed state still fails the declared condition.
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert adapter.observations[-1].state == {FLAG: False}

    records = read_records(trace, "run-1")
    committed = [record for record in records if record.kind == "contract_committed"]
    assert len(committed) == 1
    # The domain was really asked what the effect of that execution was, so the refusal is a decision
    # about this Attempt rather than a verdict reached without evidence.
    assert [(contract_id, outcome_id) for contract_id, outcome_id in domain.assess_calls] \
        == [(committed[0].contract_id, "ao-1")]
    step_verdicts = [record for record in records
                     if record.kind == "verification_recorded" and record.scope == "attempt"]
    assert [record.verification_status.value for record in step_verdicts] == ["fail"]
    # No record anywhere in this Run claims a run-scope PASS.
    assert [record for record in records
            if record.kind == "verification_recorded" and record.scope == "run"] == []

    terminals = terminal_records(records)
    assert len(terminals) == 1
    finished = terminals[0]
    assert finished.terminal_status != "success"
    assert finished.verification_record_id is None
    assert finished.reason_code == "policy_abort"


def test_an_absent_field_yields_inconclusive_not_pass(workdir):
    """C03: a condition whose path the state does not reach is INCONCLUSIVE, never a PASS.

    ``path_exists_condition("missing_flag")`` is judged against a state that really lacks that key.
    The domain declares the state incomplete, which is exactly the case in which an absent path is
    insufficient evidence rather than a definite absence: the step verdict and the task verdict are
    both INCONCLUSIVE, so the Run cannot complete as if the goal held. The structure itself is intact
    - a legal observation of a legal JSON object - and that is a different thing from damaged evidence,
    which the next cases cover.
    """
    done = action()
    condition = path_exists_condition("missing_flag")
    state = {FLAG: False}
    scenario = fake_scenario(action_spec=done, start_state=state, end_state=state,
                             outcome=succeeded("ao-1", state))
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False, effect="applied",
                        state_complete=False)
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", state, forecast_for(done, conditions=[condition]))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=[condition], max_steps=1, max_planning_calls=1)))

    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "POLICY_ABORT"
    assert result.verification_record_id is None
    # The state the adapter really returned is a well-formed object that simply lacks the declared key.
    observed = adapter.observations[-1].state
    assert isinstance(observed, dict)
    assert "missing_flag" not in observed

    records = read_records(trace, "run-1")
    verdicts = [record for record in records if record.kind == "verification_recorded"]
    assert [record.verification_status.value for record in verdicts] == ["inconclusive"]
    assert [record.scope for record in verdicts] == ["attempt"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert terminals[0].terminal_status != "success"
    assert terminals[0].verification_record_id is None


def test_an_absent_field_beside_complete_evidence_is_a_fail_not_a_pass(workdir):
    """C03: with the state declared complete, the same absent path is a definite FAIL.

    This is the other half of the distinction: absence is a verdict either way - INCONCLUSIVE while
    the domain admits its knowledge is incomplete, FAIL once the domain declares the state complete -
    and it is never a PASS. Only the completeness declaration the domain really supplied changes it.
    """
    done = action()
    condition = path_exists_condition("missing_flag")
    state = {FLAG: False}
    scenario = fake_scenario(action_spec=done, start_state=state, end_state=state,
                             outcome=succeeded("ao-1", state))
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False, effect="applied",
                        state_complete=True)
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", state, forecast_for(done, conditions=[condition]))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=[condition], max_steps=1, max_planning_calls=1)))

    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "POLICY_ABORT"
    assert result.verification_record_id is None
    records = read_records(trace, "run-1")
    verdicts = [record for record in records if record.kind == "verification_recorded"]
    assert [record.verification_status.value for record in verdicts] == ["fail"]
    assert "missing_flag" not in adapter.observations[-1].state
    assert terminal_records(records)[0].verification_record_id is None


class SubstitutedEvidenceDomain(HostDomain):
    """A domain that answers with evidence for a substitute observation.

    The substitute repeats the observation identifier while carrying different content, which is the
    shape the Runtime documents as a damaged evidence structure rather than as a missing field.
    """

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        substituted = observation.model_copy(deep=True)
        substituted.state = {"impostor": True}
        return VerificationEvidence(schema_version=SCHEMA, observation=substituted,
                                    state_complete=self.state_complete)


def test_damaged_evidence_is_refused_instead_of_being_verified(workdir):
    """C03: a damaged evidence structure is a facility refusal, not an INCONCLUSIVE verdict.

    The two diagnoses are genuinely different facts: an absent field is judged on real evidence and
    recorded as a verdict, while evidence that does not carry the observation it claims is refused
    before any Step, Contract or verification fact exists. Neither can be a PASS.
    """
    done = action()
    scenario = no_step_scenario(action_spec=done, state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(tmp_path=workdir, adapter=adapter,
                                    planner=ScriptedPlanner(planner_script([])),
                                    domain=SubstitutedEvidenceDomain())

    result = asyncio.run(engine.run(request=request_for()))

    assert result.terminal_status.value == "failed"
    assert result.reason_code.value == "EVIDENCE_INVALID"
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (0, 0, 0)
    assert result.verification_record_id is None
    assert adapter.actions == []
    records = read_records(trace, "run-1")
    # No verdict of any kind was produced: the Run refused the evidence before judging anything.
    assert kinds(records) == ["run_started", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert terminals[0].terminal_status == "failure"
    assert terminals[0].verification_record_id is None


def test_evidence_that_satisfies_every_declared_condition_completes_the_run(workdir):
    """C03: when every declared condition holds on real evidence, the oracle PASS completes the Run.

    Nothing has to be executed here, so the case isolates the completion semantics: two declared
    conditions, one equality rule and one path-existence rule, both satisfied by the single real
    observation the adapter returned.
    """
    done = action()
    conditions = [json_rule_condition(FLAG, True), path_exists_condition("gate")]
    scenario = no_step_scenario(action_spec=done, state={FLAG: True, "gate": "open"})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(tmp_path=workdir, adapter=adapter,
                                    planner=ScriptedPlanner(planner_script([])),
                                    domain=HostDomain())

    result = asyncio.run(engine.run(request=request_for(conditions=conditions)))

    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (0, 0, 0)
    assert adapter.actions == []
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "verification_recorded", "run_finished"]
    run_verdicts = [record for record in records
                    if record.kind == "verification_recorded" and record.scope == "run"]
    assert len(run_verdicts) == 1
    assert run_verdicts[0].verification_status.value == "pass"
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert terminals[0].terminal_status == "success"
    # The terminal fact, the recorded run-scope PASS and the returned result name the same record.
    assert terminals[0].verification_record_id == run_verdicts[0].record_id
    assert result.verification_record_id == run_verdicts[0].record_id
    assert result.terminal_record_id == terminals[0].record_id
    finalization = engine.last_finalization
    assert (finalization.cleanup_status, finalization.trace_status) == ("succeeded", "persisted")
    assert finalization.quarantined is False
    assert engine.is_quarantined is False


def test_a_satisfied_condition_cannot_offset_an_unsatisfied_one(workdir):
    """C03: one satisfied declared condition never buys off another unsatisfied one.

    Both conditions are required. The observed state really satisfies the first one, so the failure
    is not an artefact of missing evidence: the second one is present and false, and the required-all
    aggregate is FAIL.
    """
    done = action()
    satisfied = json_rule_condition("gate", "open")
    unsatisfied = json_rule_condition(FLAG, True)
    state = {"gate": "open", FLAG: False}
    scenario = fake_scenario(action_spec=done, start_state=state, end_state=state,
                             outcome=succeeded("ao-1", state))
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False, effect="applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", state, forecast_for(done, conditions=[satisfied, unsatisfied]))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=[satisfied, unsatisfied], max_steps=1, max_planning_calls=1)))

    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "POLICY_ABORT"
    assert result.verification_record_id is None
    # One declared condition really holds in the state the adapter returned ...
    observed = adapter.observations[-1].state
    assert observed["gate"] == "open"
    # ... and the other one is present and false, so the aggregate cannot pass.
    assert observed[FLAG] is False
    records = read_records(trace, "run-1")
    verdicts = [record for record in records if record.kind == "verification_recorded"]
    assert [record.verification_status.value for record in verdicts] == ["fail"]
    assert [record.scope for record in verdicts] == ["attempt"]
    assert terminal_records(records)[0].verification_record_id is None


def test_a_passing_step_does_not_by_itself_complete_the_run(workdir):
    """C03: the step verifier and the task oracle are two independent entries.

    The step PASSes - the committed Contract's own condition holds for the real post-execution
    observation - and that PASS is recorded in the trace at the step scope. The Run still fails to
    complete, because the separate task-completion criteria are not satisfied, and the terminal fact
    carries no run-scope PASS. A step PASS therefore cannot stand in for completion.
    """
    open_gate = action("open_gate")
    step_condition = path_exists_condition("gate")
    start = {"gate": "closed", FLAG: False}
    end = {"gate": "open", FLAG: False}
    scenario = fake_scenario(action_spec=open_gate, start_state=start, end_state=end,
                             outcome=succeeded("ao-1", end))
    domain = HostDomain(mode="bounded", max_retries=0, allow_replan=False, effect="applied")
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", start, forecast_for(open_gate, conditions=[step_condition]))])),
        domain=domain)

    result = asyncio.run(engine.run(request=request_for(
        conditions=[json_rule_condition(FLAG, True)], max_steps=1, max_planning_calls=1)))

    # The Run stopped on its declared step allowance after the task oracle declined to pass it.
    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "MAX_STEPS_EXHAUSTED"
    assert result.reason_code.value != "TASK_ACHIEVED"
    assert result.steps_started == 1
    assert result.verification_record_id is None
    records = read_records(trace, "run-1")
    step_verdicts = [record for record in records
                     if record.kind == "verification_recorded" and record.scope == "attempt"]
    assert [record.verification_status.value for record in step_verdicts] == ["pass"]
    assert [record for record in records
            if record.kind == "verification_recorded" and record.scope == "run"] == []
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert terminals[0].terminal_status != "success"
    assert terminals[0].verification_record_id is None
    # The oracle was consulted on the same real observation and did not pass it: the step PASS alone
    # decided only the step.
    assert adapter.observations[-1].state == end
