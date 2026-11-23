"""C08, C09, C10 and C12: budget and deadline boundaries, cancellation, and the bounded teardown.

Every fact asserted here was measured against the real Runtime before it was written down, and each
assertion names one exact value or one exact relation between values the Run itself produced: the
terminal status and reason, the cleanup and trace verdicts, the record identifier, the quarantine
decision and the real adapter call counts. No assertion accepts a set that contains a vacuous member,
and no expected value is read back from the output under test.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from conftest import (
    FLAG,
    SENSITIVE_MARKER,
    DeclaredAdapter,
    HostDomain,
    action,
    build_engine,
    fake_scenario,
    forecast_for,
    kinds,
    no_step_scenario,
    planner_script,
    read_jsonl,
    read_records,
    request_for,
    succeeded,
    terminal_records,
)
from forecontract.recovery import RecoveryEngine
from forecontract.runtime.errors import RuntimeError
from forecontract.tracing import JsonlTraceStore
from forecontract.verification import StepVerifier
from forecontract_fake import FakeEnvironmentAdapter
from forecontract_scripted import ScriptedPlanner


class TerminalAppendStore(JsonlTraceStore):
    """The real JSONL store, refusing the single terminal append in one of two declared ways.

    Every other record is written by the real store, so the file on disk is the product's own
    artefact. ``write_first`` is the only difference between the two cases: one refuses the terminal
    append before anything is written, the other really writes the record and then refuses.
    """

    def __init__(self, path, *, write_first: bool) -> None:
        super().__init__(path)
        self.write_first = write_first
        self.refusals = 0
        self.written_terminal = 0

    def append(self, record):
        if record.kind == "run_finished":
            self.refusals += 1
            if self.write_first:
                super().append(record)
                self.written_terminal += 1
                raise OSError("the store lost the acknowledgement")
            raise OSError("the store refused the terminal record")
        return super().append(record)


def test_a_run_budget_that_is_already_spent_aborts_without_touching_the_adapter(workdir):
    """C08: a host budget that declares zero cannot be turned into a real timeout claim."""
    done = action()
    # A single-stage scenario means nothing may execute; a spent budget must refuse before any call.
    scenario = no_step_scenario(action_spec=done, state={FLAG: False})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(tmp_path=workdir, adapter=adapter,
                                    planner=ScriptedPlanner(planner_script([])), domain=domain,
                                    budget_ms=0.0)

    result = asyncio.run(engine.run(request=request_for()))

    # Observed exactly: aborted / BUDGET_EXHAUSTED with every counter still zero.
    assert (result.terminal_status.value, result.reason_code.value) == ("aborted", "BUDGET_EXHAUSTED")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (0, 0, 0, 0, 0)
    assert result.verification_record_id is None
    # Observed exactly: no lifecycle call, no observation and no action reached the adapter.
    assert (adapter.starts, adapter.stops) == (0, 0)
    assert adapter.actions == []
    assert adapter.observations == []
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    # Observed exactly: the trace's own closed vocabulary renders this abort as policy_abort.
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("aborted", "policy_abort")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == result.terminal_record_id
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: aborted / BUDGET_EXHAUSTED, cleanup not_required, trace persisted, not
    # quarantined and no elapsed finalization allowance.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("aborted", "BUDGET_EXHAUSTED")
    assert (final.pending_operation_at_return, final.cleanup_status, final.cleanup_error_code) \
        == (False, "not_required", None)
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("persisted", None, terminals[0].record_id)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, False)
    assert engine.is_quarantined is False


def test_a_zero_run_allowance_is_an_immediate_deadline_without_a_lifecycle_call(workdir):
    """C08: an explicit zero run allowance is a real timeout that performs no lifecycle call."""
    done = action()
    scenario = no_step_scenario(action_spec=done, state={FLAG: False})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(tmp_path=workdir, adapter=adapter,
                                    planner=ScriptedPlanner(planner_script([])), domain=domain,
                                    run_timeout_ms=0.0)

    result = asyncio.run(engine.run(request=request_for()))

    # Observed exactly: timeout / DEADLINE_EXCEEDED, not an abort and not an operation timeout.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (0, 0, 0)
    assert result.verification_record_id is None
    assert (adapter.starts, adapter.stops) == (0, 0)
    assert adapter.actions == []
    assert adapter.observations == []
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    # Observed exactly: a deadline expiry keeps the trace's own deadline_exceeded code.
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("timeout", "deadline_exceeded")
    assert terminals[0].record_id == result.terminal_record_id
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the Run never acquired a resource, so its cleanup was not required.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (final.cleanup_status, final.cleanup_error_code) == ("not_required", None)
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("persisted", None, terminals[0].record_id)
    assert (final.finalize_deadline_exceeded, final.quarantined, final.pending_operation_at_return) \
        == (False, False, False)
    assert engine.is_quarantined is False


def test_a_run_bound_by_a_real_allowance_cannot_take_a_second_action(workdir):
    """C08: with the allowance already spent, no phase may start another action."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        run_timeout_ms=0.0)

    result = asyncio.run(engine.run(request=request_for()))

    # A real action was scripted and available; observed exactly: not one call was made.
    assert len(scenario.steps) == 1
    assert adapter.actions == []
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (0, 0, 0)
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("timeout", "deadline_exceeded")
    assert terminals[0].record_id == result.terminal_record_id


def test_a_caller_cancellation_is_propagated_and_the_terminal_fact_stays_unique(workdir):
    """C09: ``CancelledError`` reaches the caller and the teardown still records one terminal fact.

    The cancellation is delivered from inside the operation that is really in flight - the adapter
    cancels the Engine's own task and then waits cooperatively - so the exercised boundary is the
    declared one rather than whichever await a wall-clock race happened to reach.
    """
    done = action()

    class CooperativeExecute(DeclaredAdapter):
        def __init__(self, inner) -> None:
            super().__init__(inner)
            self.tasks: list[asyncio.Task] = []
            self.holder: dict[str, asyncio.Task] = {}

        async def execute(self, action_spec):
            self.actions.append(action_spec)
            self.tasks.append(asyncio.current_task())
            self.holder["engine"].cancel()
            await asyncio.sleep(3600)

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = CooperativeExecute(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain)

    async def cancel_run():
        task = asyncio.ensure_future(engine.run(request=request_for()))
        adapter.holder["engine"] = task
        try:
            await task
        finally:
            # Release the deliberately blocked operation so no task outlives this test.
            for pending in adapter.tasks:
                await asyncio.gather(pending, return_exceptions=True)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_run())

    # Observed exactly: one start, one real action, one cooperative stop and two observations.
    assert (adapter.starts, adapter.stops) == (1, 1)
    assert len(adapter.actions) == 1
    assert len(adapter.observations) == 2
    records = read_records(trace, "run-1")
    # Observed exactly: the cancelled Attempt keeps its AttemptStarted fact and gains no ending.
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("cancelled", "user_cancelled")
    assert terminals[0].verification_record_id is None
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the operation confirmed its cancellation inside the allowance, so this Run is
    # a cleanly closed cancellation rather than a quarantined one.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("cancelled", "USER_CANCELLED")
    assert (final.pending_operation_at_return, final.cleanup_status, final.cleanup_error_code) \
        == (False, "succeeded", None)
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("persisted", None, terminals[0].record_id)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, False)
    assert engine.is_quarantined is False


def test_a_cancellation_the_operation_refuses_keeps_the_task_owned_and_quarantined(workdir):
    """C09: an operation that refuses prompt cancellation is never claimed to have stopped.

    The adapter cancels the Engine's own task while its own execution is in flight and then refuses
    the cancellation the teardown delivers, so the Run returns while that operation is still
    unfinished. The Engine must keep owning it, must not claim the resource stopped, and the late
    result must not write a new trace record or change the sealed terminal fact.
    """
    done = action()

    class RefusingExecute(DeclaredAdapter):
        def __init__(self, inner, barrier) -> None:
            super().__init__(inner)
            self._barrier = barrier
            self.tasks: list[asyncio.Task] = []
            self.holder: dict[str, asyncio.Task] = {}
            self.refusals = 0

        async def execute(self, action_spec):
            self.actions.append(action_spec)
            self.tasks.append(asyncio.current_task())
            self.holder["engine"].cancel()
            # Deliberate refusal: the request to cancel is caught and the barrier is awaited again.
            while not self._barrier.is_set():
                try:
                    await self._barrier.wait()
                except asyncio.CancelledError:
                    self.refusals += 1
                    continue
            late = succeeded("ao-late", {FLAG: True})
            self.outcomes.append(late)
            return late

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    barrier = asyncio.Event()
    adapter = RefusingExecute(FakeEnvironmentAdapter(scenario), barrier)
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        cleanup_timeout_ms=50.0)
    sealed: dict[str, object] = {}

    async def cancel_run():
        task = asyncio.ensure_future(engine.run(request=request_for()))
        adapter.holder["engine"] = task
        try:
            await task
        finally:
            # The Run has already returned while this operation is unfinished: record the sealed
            # facts, then release the barrier so the delayed result really arrives and no task is
            # left hanging.
            before = read_records(trace, "run-1")
            sealed["kinds"] = kinds(before)
            sealed["terminal"] = terminal_records(before)[0]
            barrier.set()
            for pending in adapter.tasks:
                await asyncio.gather(pending, return_exceptions=True)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_run())

    # Observed exactly: the teardown delivered one cancellation and the operation refused it.
    assert adapter.refusals == 1
    # Observed exactly: the delayed result really was produced after the Run returned.
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-late"]
    assert len(adapter.actions) == 1
    # Observed exactly: the stop was skipped, because the operation never confirmed it stopped.
    assert (adapter.starts, adapter.stops) == (1, 0)
    sealed_terminal = sealed["terminal"]
    after = read_records(trace, "run-1")
    # The late result wrote no new record and changed no sealed terminal fact.
    assert kinds(after) == sealed["kinds"]
    assert sealed["kinds"] == ["run_started", "step_started", "contract_committed",
                               "attempt_started", "run_finished"]
    terminals = terminal_records(after)
    assert len(terminals) == 1
    assert terminals[0].record_id == sealed_terminal.record_id
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("cancelled", "user_cancelled")
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the Engine publishes the still-pending operation and quarantines the resource.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("cancelled", "USER_CANCELLED")
    assert final.pending_operation_at_return is True
    assert final.cleanup_status == "skipped_pending_operation"
    assert final.cleanup_error_code.value == "CLEANUP_FAILURE"
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("persisted", None, terminals[0].record_id)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, True)
    assert engine.is_quarantined is True


def test_a_late_result_leaves_the_whole_sealed_trace_and_finalization_unchanged(workdir):
    """C09: the complete sealed Trace and Finalization are identical before and after a late result.

    The adapter cancels the Engine's task from inside its own execution and then refuses that
    cancellation, so the Run returns while its operation is still unfinished. The *whole* sealed record
    set - every record's own ``record_id``, ``kind`` and ``sequence`` - and the whole
    ``RuntimeFinalization`` dump are snapshotted while that operation is still blocked, then the
    operation is released and really produces its late result. The comparison is list-for-list and
    key-for-key, so a late result that appended a record, reused an identifier, moved a sequence or
    rewrote a finalization field would fail here even though the record count was unchanged.
    """
    done = action()

    class RefusingLateExecute(DeclaredAdapter):
        """The real Fake with one declared refusal: the cancellation is caught and awaited again."""

        def __init__(self, inner, barrier) -> None:
            super().__init__(inner)
            self._barrier = barrier
            self.tasks: list[asyncio.Task] = []
            self.holder: dict[str, asyncio.Task] = {}
            self.refusals = 0

        async def execute(self, action_spec):
            self.actions.append(action_spec)
            self.tasks.append(asyncio.current_task())
            self.holder["engine"].cancel()
            while not self._barrier.is_set():
                try:
                    await self._barrier.wait()
                except asyncio.CancelledError:
                    self.refusals += 1
                    continue
            late = succeeded("ao-late", {FLAG: True})
            self.outcomes.append(late)
            return late

    barrier = asyncio.Event()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    adapter = RefusingLateExecute(FakeEnvironmentAdapter(scenario), barrier)
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])),
        domain=HostDomain(), cleanup_timeout_ms=50.0)
    snapshot: dict[str, object] = {}

    async def cancel_run():
        task = asyncio.ensure_future(engine.run(request=request_for()))
        adapter.holder["engine"] = task
        try:
            await task
        finally:
            # The Run has already returned while this operation is still blocked: snapshot the complete
            # sealed facts before anything is released, so the comparison below is about stability.
            # Every field of every record is detached: the shared envelope, the timestamp, the associations and
            # each variant's own payload. A three-field projection would not be the sealed Trace.
            snapshot["records"] = [record.model_dump(mode="json")
                                   for record in read_records(trace, "run-1")]
            snapshot["finalization"] = engine.last_finalization.model_dump()
            snapshot["released_when_snapshotted"] = barrier.is_set()
            barrier.set()
            for pending in adapter.tasks:
                await asyncio.gather(pending, return_exceptions=True)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_run())

    # The late result really arrived after the snapshot, and the operation really refused the teardown's
    # one cancellation request.
    assert snapshot["released_when_snapshotted"] is False
    assert adapter.refusals == 1
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-late"]
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert (adapter.starts, adapter.stops) == (1, 0)

    sealed_records = snapshot["records"]
    sealed_finalization = snapshot["finalization"]
    # Observed exactly: five sealed records, one per fact the Run really wrote, in this order and with
    # these sequence numbers, and every record identifier is unique.
    assert [record["kind"] for record in sealed_records] == ["run_started", "step_started",
                                                            "contract_committed", "attempt_started",
                                                            "run_finished"]
    assert [record["sequence"] for record in sealed_records] == [0, 1, 2, 3, 4]
    assert len({record["record_id"] for record in sealed_records}) == 5
    # The detached snapshot really carries the whole envelope and every variant's own fields, which is what
    # makes the after-comparison a comparison of the sealed Trace rather than of an identifier projection.
    envelope = {"schema_version", "record_id", "run_id", "sequence", "occurred_at", "kind"}
    assert all(envelope <= set(record) for record in sealed_records)
    # The terminal variant carries the fields that make it a terminal fact, so the snapshot is not a projection.
    assert {"terminal_status", "reason_code"} <= set(sealed_records[4])
    assert sealed_finalization["terminal_record_id"] == sealed_records[-1]["record_id"]

    # Re-read everything after the late result and compare the complete snapshot, field for field: a late
    # result that appended a record, reused an identifier, moved a sequence, changed a timestamp, broke an
    # association or rewrote any variant payload fails here even though the record count is unchanged.
    after_records = [record.model_dump(mode="json") for record in read_records(trace, "run-1")]
    assert after_records == sealed_records
    # The whole diagnostic snapshot is compared the same way: every field of the RuntimeFinalization read back
    # after the late result must equal the one detached while the operation was still blocked. A late result
    # that rewrote any field - pending state, cleanup verdict, trace verdict, terminal identifier, the overdue
    # flag or the quarantine verdict - fails here, and the field-by-field assertions below are corroboration
    # rather than a substitute for this equality.
    assert engine.last_finalization.model_dump() == sealed_finalization

    # The sealed finalization is the cancelled ending this Run really reached, and the release changed
    # none of it.
    assert (sealed_finalization["primary_terminal_status"].value,
            sealed_finalization["primary_reason_code"].value) \
        == ("cancelled", "USER_CANCELLED")
    assert (sealed_finalization["pending_operation_at_return"],
            sealed_finalization["cleanup_status"],
            sealed_finalization["cleanup_error_code"].value,
            sealed_finalization["trace_status"], sealed_finalization["trace_error_code"],
            sealed_finalization["finalize_deadline_exceeded"],
            sealed_finalization["quarantined"]) \
        == (True, "skipped_pending_operation", "CLEANUP_FAILURE", "persisted", None, False, True)
    terminals = terminal_records(read_records(trace, "run-1"))
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("cancelled", "user_cancelled")
    assert terminals[0].record_id == sealed_records[-1]["record_id"]
    assert engine.is_quarantined is True


def test_a_stop_failure_keeps_the_primary_reason_and_quarantines_the_resource(workdir):
    """C10: a failed stop is a bounded cleanup failure; the resource is not settled and is quarantined."""
    done = action()

    class FailingStop(DeclaredAdapter):
        async def stop(self):
            self.stops += 1
            raise OSError("the environment refused to stop")

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = FailingStop(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain)

    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=request_for()))

    # Observed exactly: a raising stop is one attempted stop, and the raised diagnosis is the
    # cleanup failure itself with its own bounded location and message.
    assert adapter.stops == 1
    assert (adapter.starts, len(adapter.actions)) == (1, 1)
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("CLEANUP_FAILURE", "cleanup", ("<field>",))
    assert info.value.message == "the adapter stop operation failed"
    records = read_records(trace, "run-1")
    terminals = terminal_records(records)
    assert len(terminals) == 1
    # Observed exactly: this Run really reached success and its task PASS was recorded, so the trace
    # terminal fact is the cleanup failure the Run was published as. The mapping of CLEANUP_FAILURE
    # onto the closed trace vocabulary is `internal_error`.
    assert [record.kind for record in records] == [
        "run_started", "step_started", "contract_committed", "attempt_started",
        "attempt_finished", "verification_recorded", "verification_recorded", "run_finished"]
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("failure", "internal_error")
    assert terminals[0].verification_record_id is None
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the cleanup failed, the resource is not settled, and this is really what the
    # Engine reports for a raising stop.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("failed", "CLEANUP_FAILURE")
    assert (final.pending_operation_at_return, final.cleanup_status) == (False, "failed")
    assert final.cleanup_error_code.value == "CLEANUP_FAILURE"
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("persisted", None, terminals[0].record_id)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, True)
    assert engine.is_quarantined is True


def test_a_stop_that_outlives_its_allowance_times_out_and_is_quarantined(workdir):
    """C10: a stop that outlives the cleanup allowance is a timeout, never a settled resource."""
    done = action()
    barrier = asyncio.Event()

    class BlockingStop(DeclaredAdapter):
        def __init__(self, inner, gate) -> None:
            super().__init__(inner)
            self._gate = gate
            self.tasks: list[asyncio.Task] = []

        async def stop(self):
            self.stops += 1
            self.tasks.append(asyncio.current_task())
            await self._gate.wait()

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = BlockingStop(FakeEnvironmentAdapter(scenario), barrier)
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        cleanup_timeout_ms=50.0)

    async def run_it():
        try:
            await engine.run(request=request_for())
        finally:
            # Release the stop that deliberately outlived its allowance, so nothing hangs.
            barrier.set()
            for pending in adapter.tasks:
                await asyncio.gather(pending, return_exceptions=True)

    with pytest.raises(RuntimeError) as info:
        asyncio.run(run_it())

    # Observed exactly: the stop was really issued once and really outlived the allowance.
    assert adapter.stops == 1
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("CLEANUP_FAILURE", "cleanup", ("<field>",))
    assert info.value.message == "the adapter stop operation did not finish in time"
    records = read_records(trace, "run-1")
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("failure", "internal_error")
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the candidate success is published as the cleanup failure that decided it,
    # the cleanup is reported as timed_out and the unfinished stop keeps the resource quarantined.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("failed", "CLEANUP_FAILURE")
    assert final.pending_operation_at_return is True
    assert final.cleanup_status == "timed_out"
    assert final.cleanup_error_code.value == "CLEANUP_FAILURE"
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("persisted", None, terminals[0].record_id)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, True)
    assert engine.is_quarantined is True


def test_a_trace_that_refuses_every_record_is_reported_without_faking_a_terminal_id(workdir):
    """C10: a trace store that refuses cannot produce a fabricated persisted terminal fact."""
    done = action()

    class RefusingTrace:
        def append(self, record):
            raise OSError("the trace store refused the record")

        def read_run(self, run_id):
            return ()

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, _, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        trace_writer=RefusingTrace())

    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=request_for()))

    # Observed exactly: the store refused the very first record, so the Run never entered the adapter.
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("TRACE_FAILURE", "finalize", ("<field>",))
    assert info.value.message == "the terminal fact could not be persisted"
    assert (adapter.starts, adapter.stops) == (0, 0)
    assert adapter.actions == []
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the append raised, so the Runtime cannot know whether the record reached the
    # file. The honest label is ``unknown`` - explicitly *not* a claim that nothing was written - and
    # no terminal record is named without a confirmed append return.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("failed", "TRACE_FAILURE")
    assert (final.pending_operation_at_return, final.cleanup_status, final.cleanup_error_code) \
        == (False, "not_required", None)
    assert final.trace_status == "unknown"
    assert final.trace_error_code.value == "TRACE_FAILURE"
    assert final.terminal_record_id is None
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, True)
    assert engine.is_quarantined is True


def test_a_terminal_append_that_raises_before_writing_is_reported_unknown(workdir):
    """C10: a terminal append that raises before writing is uncertain, and names no record."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    store = TerminalAppendStore(workdir / "trace.jsonl", write_first=False)
    engine, _, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        trace_writer=store)

    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=request_for()))

    # Observed exactly: the refusal was on the single terminal append, which never reached the file.
    assert store.refusals == 1
    assert store.written_terminal == 0
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("TRACE_FAILURE", "finalize", ("<field>",))
    assert info.value.message == "the terminal fact could not be persisted"
    records = read_records(store, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "verification_recorded"]
    assert terminal_records(records) == []
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the Run's business outcome really was success, and the trace verdict is the
    # honest ``unknown`` with no confirmed identifier and no invented elapsed allowance.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (final.cleanup_status, final.cleanup_error_code) == ("succeeded", None)
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("unknown", "TRACE_FAILURE", None)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, True)
    assert engine.is_quarantined is True


def test_a_terminal_append_that_writes_then_raises_is_reported_unknown(workdir):
    """C10: a terminal append that really wrote and then raised is still reported ``unknown``.

    The record is on disk, so the Run may not claim nothing was written; but the append never
    returned, so the Runtime may not name a terminal record either. Both statements are asserted
    against the real file.
    """
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    store = TerminalAppendStore(workdir / "trace.jsonl", write_first=True)
    engine, _, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        trace_writer=store)

    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=request_for()))

    # Observed exactly: the store really wrote the terminal record and then refused.
    assert store.refusals == 1
    assert store.written_terminal == 1
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("TRACE_FAILURE", "finalize", ("<field>",))
    records = read_records(store, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "verification_recorded", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("success", None)
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the same ``unknown`` verdict as the unwritten case, and an identifier no
    # confirmed append return exists for is never named.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (final.cleanup_status, final.cleanup_error_code) == ("succeeded", None)
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("unknown", "TRACE_FAILURE", None)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, True)
    assert engine.is_quarantined is True


def test_a_refusing_working_memory_is_reported_without_claiming_a_successful_memory(workdir):
    """C10: a Working Memory that refuses is a bounded facility failure, not a silent success."""
    done = action()

    class RefusingMemory:
        def __init__(self) -> None:
            self.calls = 0
            self.stored: dict[str, object] = {}

        def set(self, run_id, state):
            self.calls += 1
            raise OSError("the working memory refused the state")

        def get(self, run_id):
            raise KeyError(run_id)

        def clear(self, run_id):
            return False

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    memory = RefusingMemory()
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        memory=memory)

    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=request_for()))

    # Observed exactly: the refusal is reported at the Memory boundary of the finalizing segment.
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("TRACE_FAILURE", "finalize", ("memory",))
    assert info.value.message == "the terminal fact could not be persisted"
    # Observed exactly: both retained Memory writes - the safe diagnosis and the publication of the
    # confirmed terminal identifier - were attempted and neither succeeded, so the Engine claims no
    # successful terminal memory.
    assert (memory.calls, memory.stored) == (2, {})
    records = read_records(trace, "run-1")
    terminals = terminal_records(records)
    assert len(terminals) == 1
    run_verdicts = [record for record in records
                    if record.kind == "verification_recorded" and record.scope == "run"]
    assert len(run_verdicts) == 1
    # The terminal fact was still written by the real store, and it names the real run-scope PASS.
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("success", None)
    assert terminals[0].verification_record_id == run_verdicts[0].record_id
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: the business outcome and the trace write both really succeeded; the refusal
    # is carried by the raised diagnosis, not by a rewritten snapshot.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (final.cleanup_status, final.cleanup_error_code) == ("succeeded", None)
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("persisted", None, terminals[0].record_id)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (False, False)
    assert engine.is_quarantined is False


def test_a_refused_memory_write_charges_the_real_finalization_allowance(workdir):
    """C10: a refused Memory write is charged against the real finalization allowance.

    The Memory write blocks for 50 ms while the declared allowance is 10 ms, so the refusal really
    spent the allowance. Observed exactly: the Run reports the elapsed allowance, never attempts the
    terminal append, and claims no successful terminal memory.
    """
    done = action()

    class SlowRefusingMemory:
        def __init__(self) -> None:
            self.calls = 0
            self.stored: dict[str, object] = {}

        def set(self, run_id, state):
            self.calls += 1
            time.sleep(0.05)
            raise OSError("the working memory refused the state")

        def get(self, run_id):
            raise KeyError(run_id)

        def clear(self, run_id):
            return False

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    memory = SlowRefusingMemory()
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain,
        memory=memory, finalize_timeout_ms=10.0)

    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=request_for()))

    # Observed exactly: the refusal consumed the allowance, so no terminal append was attempted and
    # the raised diagnosis is this segment's bounded persistence failure.
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("TRACE_FAILURE", "finalize", ("<field>",))
    assert info.value.message == "the terminal fact could not be persisted"
    assert (memory.calls, memory.stored) == (1, {})
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "verification_recorded"]
    assert terminal_records(records) == []
    final = engine.last_finalization
    assert final is not None
    # Observed exactly: not_attempted with the elapsed-allowance fact set, and the resource is
    # quarantined because no terminal fact was persisted.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value) \
        == ("completed", "TASK_ACHIEVED")
    assert (final.cleanup_status, final.cleanup_error_code) == ("succeeded", None)
    assert (final.trace_status, final.trace_error_code, final.terminal_record_id) \
        == ("not_attempted", "TRACE_FAILURE", None)
    assert (final.finalize_deadline_exceeded, final.quarantined) == (True, True)
    assert engine.is_quarantined is True


def test_a_second_engine_refuses_the_quarantined_adapter_without_calling_it(workdir):
    """C10: a resource an earlier Run failed to stop is not handed to a second Engine.

    The first Run's stop raises, so the adapter object's resource stays occupied. A *second* Engine
    built on the very same adapter object must refuse before it starts anything.
    """
    done = action()

    class FailingStop(DeclaredAdapter):
        async def stop(self):
            self.stops += 1
            raise OSError("the environment refused to stop")

    # The observation is the one this planner has no plan for, so the first Run fails before any
    # action is dispatched and its occupied resource is the only thing the two Engines share.
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: False})
    domain = HostDomain()
    adapter = FailingStop(FakeEnvironmentAdapter(scenario))
    first, _, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {"other": 1}, forecast_for(done))])), domain=domain,
        trace_name="first-trace.jsonl")

    with pytest.raises(RuntimeError) as first_info:
        asyncio.run(first.run(request=request_for(run_id="run-1")))

    # Observed exactly: the planning refusal leads, the failed cleanup is its bounded secondary fact,
    # and the resource is quarantined rather than released.
    assert (first_info.value.code, first_info.value.stage) == ("DEPENDENCY_FAILURE", "plan")
    assert first_info.value.cleanup_failure is not None
    assert first_info.value.cleanup_failure.code == "CLEANUP_FAILURE"
    first_final = first.last_finalization
    assert first_final is not None
    assert (first_final.primary_terminal_status.value, first_final.primary_reason_code.value) \
        == ("failed", "DEPENDENCY_FAILURE")
    assert (first_final.cleanup_status, first_final.cleanup_error_code.value) \
        == ("failed", "CLEANUP_FAILURE")
    assert (first_final.trace_status, first_final.quarantined) == ("persisted", True)
    assert first.is_quarantined is True
    # Observed exactly: the first Run made its single start, no action ever ran, and its stop raised.
    assert (adapter.starts, adapter.stops) == (1, 1)
    assert adapter.actions == []

    second, _, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {"other": 1}, forecast_for(done))])), domain=domain,
        trace_name="second-trace.jsonl")

    with pytest.raises(RuntimeError) as second_info:
        asyncio.run(second.run(request=request_for(run_id="run-2")))

    # Observed exactly: a new Engine refuses the occupied resource before any adapter call.
    assert (second_info.value.code, second_info.value.stage, second_info.value.location) \
        == ("DEPENDENCY_FAILURE", "validate", ("<field>",))
    assert second_info.value.message == "the adapter resource is occupied by an unsettled Run"
    assert second.is_quarantined is True
    # A refused Run never reaches a diagnosis of its own, so it cannot erase the first Run's.
    assert second.last_finalization is None
    # The second Engine performed no adapter call at all: the counts are exactly the first Run's own
    # one start and one stop, so the refused Run added zero of each, and no action was ever dispatched.
    assert (adapter.starts, adapter.stops) == (1, 1)
    assert adapter.actions == []


def test_no_artifact_of_a_failing_run_carries_the_sensitive_marker(workdir):
    """C12: even a facility-failing run writes no sensitive marker into its trace."""
    done = action()

    class FailingStop(DeclaredAdapter):
        async def stop(self):
            self.stops += 1
            raise OSError(f"refused {SENSITIVE_MARKER}")

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    domain = HostDomain()
    adapter = FailingStop(FakeEnvironmentAdapter(scenario))
    engine, _, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=domain)

    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=request_for()))

    # Observed exactly: the marker-carrying refusal is the same bounded cleanup failure as any other.
    assert adapter.stops == 1
    assert (info.value.code, info.value.stage, info.value.location) \
        == ("CLEANUP_FAILURE", "cleanup", ("<field>",))
    written = json.dumps(read_jsonl(workdir / "trace.jsonl"), ensure_ascii=False)
    assert SENSITIVE_MARKER not in written
    assert "api_key" not in written.lower()
    final = engine.last_finalization
    assert final is not None
    assert (final.primary_reason_code.value, final.cleanup_status, final.quarantined) \
        == ("CLEANUP_FAILURE", "failed", True)


def test_a_planner_wait_that_outlives_the_run_deadline_is_a_deadline_timeout(workdir):
    """C08: an asynchronous plan call that runs past D is a whole-Run deadline timeout."""
    done = action()
    gate = asyncio.Event()
    entered = []

    class WaitingPlanner:
        """A planner that yields the loop and then waits until the host releases it."""

        async def plan(self, observation):
            entered.append(observation.observation_id)
            await gate.wait()
            raise AssertionError("the barrier must never be released before the deadline")

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter, planner=WaitingPlanner(), domain=HostDomain(),
        run_timeout_ms=40.0)

    async def drive():
        try:
            return await engine.run(request=request_for())
        finally:
            gate.set()

    result = asyncio.run(drive())

    assert (result.terminal_status.value, result.reason_code.value) == ("timeout",
                                                                       "DEADLINE_EXCEEDED")
    assert result.steps_started == 0
    assert result.attempts_started == 0
    # The phase really reached the planner, so the timeout is at the wait rather than before the call.
    assert entered == ["obs-before"]
    assert adapter.actions == []
    terminals = terminal_records(read_records(trace, "run-1"))
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("timeout",
                                                                       "deadline_exceeded")


def test_an_environment_call_that_never_returns_is_a_whole_run_deadline_timeout(workdir):
    """C08: a real environment call that never returns ends as the whole-Run deadline timeout.

    Two Runs share one declared 400 ms Run allowance and one blocking environment call: the first blocks
    in ``observe``, the second in ``execute``. Both really enter the phase they block in - the second
    really starts an Attempt and really dispatches its action - so the diagnosis is the Run's own
    deadline and not a step, a Contract or a lifecycle outcome. Both operations are released in the
    ``finally`` so no task outlives the test.
    """
    done = action()
    entered: list[str] = []

    class NeverReturning(DeclaredAdapter):
        """The real Fake with one declared barrier: the named call waits until the host cancels it."""

        def __init__(self, inner, *, blocks: str) -> None:
            super().__init__(inner)
            self._blocks = blocks
            self.tasks: list[asyncio.Task] = []

        async def _wait_forever(self, label: str) -> None:
            entered.append(label)
            self.tasks.append(asyncio.current_task())
            await asyncio.Event().wait()

        async def observe(self):
            if self._blocks == "observe":
                await self._wait_forever("observe")
            return await super().observe()

        async def execute(self, action_spec):
            if self._blocks == "execute":
                # The real proxy records the action before it really executes it, so the action this Run
                # dispatched is recorded here before the barrier is entered.
                self.actions.append(action_spec)
                await self._wait_forever("execute")
            return await super().execute(action_spec)

    def run_blocked(name: str, blocks: str):
        del entered[:]
        directory = workdir / name
        directory.mkdir()
        scenario = fake_scenario(action_spec=done, start_state={FLAG: False},
                                 end_state={FLAG: True})
        adapter = NeverReturning(FakeEnvironmentAdapter(scenario), blocks=blocks)
        engine, trace, _ = build_engine(
            tmp_path=directory, adapter=adapter,
            planner=ScriptedPlanner(planner_script(
                [("plan-1", {FLAG: False}, forecast_for(done))])),
            domain=HostDomain(), run_timeout_ms=400.0)

        async def drive():
            try:
                return await engine.run(request=request_for())
            finally:
                # Release every deliberately blocked operation, so no task outlives this test.
                for pending in adapter.tasks:
                    pending.cancel()
                for pending in adapter.tasks:
                    await asyncio.gather(pending, return_exceptions=True)

        result = asyncio.run(drive())
        return result, adapter, engine, trace, list(entered)

    # (1) the observation never returns: the Run's whole deadline is the only bound that can stop it.
    observed_result, observed_adapter, observed_engine, observed_trace, observed_entered = run_blocked(
        "blocked-observe", "observe")

    assert (observed_result.terminal_status.value, observed_result.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (observed_result.steps_started, observed_result.planning_calls,
            observed_result.attempts_started) == (0, 0, 0)
    assert observed_result.verification_record_id is None
    # The phase really reached the adapter's observation and nothing was executed.
    assert observed_entered == ["observe"]
    assert (observed_adapter.starts, observed_adapter.stops) == (1, 1)
    assert observed_adapter.actions == []
    assert observed_adapter.outcomes == []
    records = read_records(observed_trace, "run-1")
    assert kinds(records) == ["run_started", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("timeout", "deadline_exceeded")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == observed_result.terminal_record_id
    final = observed_engine.last_finalization
    assert final is not None
    # The blocked observation acknowledged its cancellation inside the cleanup allowance, so this Run is
    # a cleanly closed timeout rather than a quarantined one.
    assert (final.primary_terminal_status.value, final.primary_reason_code.value,
            final.pending_operation_at_return, final.cleanup_status, final.trace_status,
            final.quarantined) \
        == ("timeout", "DEADLINE_EXCEEDED", False, "succeeded", "persisted", False)
    assert observed_engine.is_quarantined is False

    # (2) the execution never returns: the action really was dispatched and still nothing completes.
    executed_result, executed_adapter, executed_engine, executed_trace, executed_entered = run_blocked(
        "blocked-execute", "execute")

    assert (executed_result.terminal_status.value, executed_result.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (executed_result.steps_started, executed_result.planning_calls,
            executed_result.attempts_started) == (1, 1, 1)
    assert executed_result.verification_record_id is None
    # The phase really reached the adapter's execution, and exactly one action was dispatched.
    assert executed_entered == ["execute"]
    assert [item.action_type for item in executed_adapter.actions] == ["mark_done"]
    assert executed_adapter.outcomes == []
    assert (executed_adapter.starts, executed_adapter.stops) == (1, 1)
    records = read_records(executed_trace, "run-1")
    # The Attempt keeps the AttemptStarted fact it really wrote and gains no fabricated ending.
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("timeout", "deadline_exceeded")
    assert terminals[0].verification_record_id is None
    assert terminals[0].record_id == executed_result.terminal_record_id
    final = executed_engine.last_finalization
    assert final is not None
    assert (final.primary_terminal_status.value, final.primary_reason_code.value,
            final.pending_operation_at_return, final.cleanup_status, final.trace_status,
            final.quarantined) \
        == ("timeout", "DEADLINE_EXCEEDED", False, "succeeded", "persisted", False)
    assert executed_engine.is_quarantined is False


class SleepingStepVerifier:
    """The real ``StepVerifier``, preceded by a synchronous delay no event loop can interrupt.

    A synchronous dependency is charged when it returns: the delay blocks the loop, so the Run's own
    deadline is crossed while the call is still inside the verifier.
    """

    def __init__(self, delay_seconds: float) -> None:
        self._inner = StepVerifier()
        self._delay_seconds = delay_seconds
        self.calls = 0

    def verify_step(self, *, contract, evidence):
        self.calls += 1
        time.sleep(self._delay_seconds)
        return self._inner.verify_step(contract=contract, evidence=evidence)


class SleepingRecoveryEngine:
    """The real ``RecoveryEngine``, preceded by a synchronous delay no event loop can interrupt."""

    def __init__(self, delay_seconds: float) -> None:
        self._inner = RecoveryEngine()
        self._delay_seconds = delay_seconds
        self.calls = 0

    def assess(self, *, contract, context, safety, evidence=None):
        self.calls += 1
        time.sleep(self._delay_seconds)
        return self._inner.assess(contract=contract, context=context, safety=safety,
                                  evidence=evidence)


def test_a_synchronous_step_verifier_that_returns_late_is_a_deadline_timeout(workdir):
    """C08: a synchronous StepVerifier that returns only after the deadline cannot complete the Run.

    The verifier is the real one behind a synchronous 600 ms delay, and the Run declares a 400 ms
    allowance, so the deadline is already crossed when the verdict comes back. The verdict the verifier
    really produced is not accepted as a success: the Run ends as the whole-Run deadline timeout, with
    exactly one action taken, exactly one terminal fact, and the result naming that same fact.
    """
    done = action()
    verifier = SleepingStepVerifier(0.6)
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])),
        domain=HostDomain(), step_verifier=verifier, run_timeout_ms=400.0)

    result = asyncio.run(engine.run(request=request_for()))

    # Observed exactly: timeout / DEADLINE_EXCEEDED, not a completion and not a facility failure.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 0, 0)
    assert result.verification_record_id is None
    # The verifier really was entered exactly once, so the timeout is at the late return and not before it.
    assert verifier.calls == 1
    # Exactly one action reached the adapter and was executed.
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["fake-outcome-1"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    step_verdicts = [record for record in records
                     if record.kind == "verification_recorded" and record.scope == "attempt"]
    assert [record.verification_status.value for record in step_verdicts] == ["pass"]
    # No record anywhere in this Run claims a run-scope PASS.
    assert [record for record in records
            if record.kind == "verification_recorded" and record.scope == "run"] == []
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("timeout", "deadline_exceeded")
    assert terminals[0].verification_record_id is None
    assert result.terminal_record_id == terminals[0].record_id
    final = engine.last_finalization
    assert final is not None
    assert (final.primary_terminal_status.value, final.primary_reason_code.value,
            final.cleanup_status, final.trace_status, final.terminal_record_id,
            final.quarantined) \
        == ("timeout", "DEADLINE_EXCEEDED", "succeeded", "persisted", terminals[0].record_id,
            False)
    assert engine.is_quarantined is False


def test_a_synchronous_recovery_engine_that_returns_late_is_a_deadline_timeout(workdir):
    """C08: a synchronous RecoveryEngine that returns only after the deadline stops the Run there.

    The recovery engine is the real one behind a synchronous 600 ms delay, and the Run declares a
    400 ms allowance. The first assessment really happens and really asks for the post-attempt
    observation, and the Run cannot act on it: the whole-Run deadline is already spent, so the third
    observation that decision needs is never taken, and the Run ends as the deadline timeout.
    """
    done = action()
    state = {FLAG: False}
    recovery = SleepingRecoveryEngine(0.6)
    scenario = fake_scenario(action_spec=done, start_state=state, end_state=state,
                             outcome=succeeded("ao-1", state))
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", state, forecast_for(done))])),
        domain=HostDomain(mode="bounded", max_retries=1, allow_replan=False),
        recovery_engine=recovery, run_timeout_ms=400.0)

    result = asyncio.run(engine.run(request=request_for(max_steps=2, max_planning_calls=2,
                                                        max_retries=1)))

    # Observed exactly: timeout / DEADLINE_EXCEEDED with one Step, one Attempt and no retry or replan.
    assert (result.terminal_status.value, result.reason_code.value) \
        == ("timeout", "DEADLINE_EXCEEDED")
    assert (result.steps_started, result.planning_calls, result.attempts_started,
            result.retries_used, result.replans_used) == (1, 1, 1, 0, 0)
    assert result.verification_record_id is None
    # The recovery engine really was entered exactly once: the deadline stopped the decision it made.
    assert recovery.calls == 1
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert [item.outcome_id for item in adapter.outcomes] == ["ao-1"]
    assert (adapter.starts, adapter.stops) == (1, 1)
    records = read_records(trace, "run-1")
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "run_finished"]
    step_verdicts = [record for record in records
                     if record.kind == "verification_recorded" and record.scope == "attempt"]
    assert [record.verification_status.value for record in step_verdicts] == ["fail"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) \
        == ("timeout", "deadline_exceeded")
    assert terminals[0].verification_record_id is None
    assert result.terminal_record_id == terminals[0].record_id
    final = engine.last_finalization
    assert final is not None
    assert (final.primary_terminal_status.value, final.primary_reason_code.value,
            final.cleanup_status, final.trace_status, final.terminal_record_id,
            final.quarantined) \
        == ("timeout", "DEADLINE_EXCEEDED", "succeeded", "persisted", terminals[0].record_id,
            False)
    assert engine.is_quarantined is False


def test_an_execute_that_outlives_the_committed_contract_is_an_operation_timeout(workdir):
    """C08: an asynchronous execute that outlives the Contract's local allowance times out locally."""
    done = action()
    gate = asyncio.Event()
    entered = []

    class BarrierAdapter(DeclaredAdapter):
        async def execute(self, action_spec):
            self.actions.append(action_spec)
            entered.append(action_spec.action_type)
            await gate.wait()
            return succeeded("ao-late", {FLAG: True})

    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    adapter = BarrierAdapter(FakeEnvironmentAdapter(scenario))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=adapter,
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])),
        domain=HostDomain(timeout_ms=40.0), run_timeout_ms=30000.0)

    async def drive():
        try:
            return await engine.run(request=request_for())
        finally:
            gate.set()

    result = asyncio.run(drive())

    # Measured exactly: the committed contract's own allowance elapsed first, so the RunResult keeps the
    # local reason ``OPERATION_TIMEOUT`` and the trace's closed vocabulary collapses both timeout reasons
    # onto the single label ``deadline_exceeded`` (verified against the engine's own reason-to-label map).
    assert result.terminal_status.value == "timeout"
    assert result.reason_code.value == "OPERATION_TIMEOUT"
    assert entered == ["mark_done"]
    terminals = terminal_records(read_records(trace, "run-1"))
    assert len(terminals) == 1
    assert (terminals[0].terminal_status, terminals[0].reason_code) == ("timeout",
                                                                       "deadline_exceeded")
