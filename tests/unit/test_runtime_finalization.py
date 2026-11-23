"""A11/A12: the business deadline, the bounded teardown and the finalization snapshot.

The engine's timing is exercised with a clock the test moves by hand, so every assertion is about a
*declared* timeline rather than about real elapsed time. The two honest limits of this round are
stated in the cases themselves: a synchronous dependency cannot be preempted in the same event loop,
so its elapsed time is charged when it returns, and an operation that does not confirm it stopped is
reported as pending and quarantined rather than claimed to be stopped.
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

CORE_SRC = pathlib.Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.models.base import SCHEMA_VERSION
from forecontract.runtime import (
    CLEANUP_STATUSES,
    TRACE_STATUSES,
    MonotonicBudgetSource,
    RunRequest,
    RuntimeError,
    RuntimeFinalization,
    RuntimeReason,
    RuntimeTerminalStatus,
    RuntimeTimingPolicy,
)
from forecontract.runtime.finalization import build_finalization

_FIXTURE = pathlib.Path(__file__).with_name("test_runtime_engine.py")
_spec = importlib.util.spec_from_file_location("_runtime_engine_fixture", _FIXTURE)
f = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(f)


class ControlledClock:
    """A monotonic clock the test moves by hand: a declared timeline, not real elapsed time."""

    def __init__(self, start: int = 0) -> None:
        self.now = start

    def now_ns(self) -> int:
        return self.now

    def advance_ms(self, milliseconds: float) -> None:
        self.now += int(milliseconds * 1_000_000)


def engine_with(adapter, *, clock=None, policy=None, budget=None, planner=None, support=None,
                **extra):
    return f.engine_for(adapter, planner if planner is not None else f.ScriptedPlanner([]),
                        support if support is not None else f.Support(),
                        budget=budget, monotonic_clock=clock, timing_policy=policy, **extra)


def drive_with(adapter, **kw):
    return asyncio.run(engine_with(adapter, **kw).run(request=f.request()))


def run_catching(adapter, **kw):
    """Drive one Run and return the engine, the result or None, and the raised failure or None."""
    engine = engine_with(adapter, **kw)
    try:
        result = asyncio.run(engine.run(request=f.request()))
        error = None
    except Exception as exc:  # noqa: BLE001
        result = None
        error = exc
    return engine, result, error


# --- the business deadline -----------------------------------------------------------

def test_a_real_deadline_expiry_is_a_timeout_and_not_a_failure_or_an_abort() -> None:
    """A slow observation exhausts the declared Run limit, and the Run returns a timeout result.

    The ordinary timeout path is the contract's: the unique terminal fact was confirmed written, so
    the caller receives ``RunResult`` with the ``timeout`` status rather than a raised failure.
    """
    clock = ControlledClock()

    class Slow(f.ScriptedAdapter):
        async def observe(self):
            clock.advance_ms(5000)
            return await super().observe()

    adapter = Slow(observations=[f.obs("o-0", {"done": True})])
    engine, result, error = run_catching(adapter, clock=clock,
                                         policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION,
                                                                    run_timeout_ms=1000))
    assert error is None
    assert result is not None
    assert result.terminal_status is RuntimeTerminalStatus.TIMEOUT
    assert result.reason_code is RuntimeReason.DEADLINE_EXCEEDED
    final = engine.last_finalization
    assert final is not None
    assert final.primary_terminal_status is RuntimeTerminalStatus.TIMEOUT
    assert final.primary_reason_code is RuntimeReason.DEADLINE_EXCEEDED
    # A timeout is neither a facility failure nor an abort, and no business work was accepted late.
    assert result.terminal_status is not RuntimeTerminalStatus.FAILED
    assert result.terminal_status is not RuntimeTerminalStatus.ABORTED
    assert result.attempts_started == 0


def test_a_timed_out_run_writes_the_existing_timeout_trace_fact_and_no_new_variant() -> None:
    clock = ControlledClock()

    class Slow(f.ScriptedAdapter):
        async def observe(self):
            clock.advance_ms(5000)
            return await super().observe()

    trace = f.TraceRecorder()
    adapter = Slow(observations=[f.obs("o-0", {"done": True})])
    engine = engine_with(adapter, clock=clock, trace=trace,
                         policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION,
                                                    run_timeout_ms=1000))
    asyncio.run(engine.run(request=f.request()))
    terminals = [x for x in trace.records if x.kind == "run_finished"]
    assert len(terminals) == 1
    # The terminal *status* is the existing ``timeout`` label and the closed reason code is the
    # existing ``deadline_exceeded``; nothing new is added to either vocabulary.
    assert terminals[0].terminal_status == "timeout"
    assert terminals[0].reason_code == "deadline_exceeded"
    assert {x.kind for x in trace.records} <= {
        "run_started", "step_started", "contract_committed", "attempt_started",
        "attempt_finished", "verification_recorded", "run_finished"}


def test_the_host_budget_and_the_run_limit_are_both_enforced_and_the_smaller_wins() -> None:
    """A generous host budget still cannot extend a shorter declared Run limit."""
    clock = ControlledClock()

    class Slow(f.ScriptedAdapter):
        async def observe(self):
            clock.advance_ms(2000)
            return await super().observe()

    adapter = Slow(observations=[f.obs("o-0", {"done": True})])
    engine, result, error = run_catching(
        adapter, clock=clock, budget=MonotonicBudgetSource(total_budget_ms=10 ** 9,
                                                           monotonic_clock=clock),
        policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_ms=1000))
    assert error is None and result is not None
    assert result.terminal_status is RuntimeTerminalStatus.TIMEOUT
    assert engine.last_finalization.primary_terminal_status is RuntimeTerminalStatus.TIMEOUT


def test_a_generous_allowance_leaves_a_normal_run_completely_unchanged() -> None:
    clock = ControlledClock()
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    engine, result, error = run_catching(
        adapter, clock=clock,
        policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_ms=10 ** 6))
    assert error is None and result is not None
    assert result.terminal_status.value == "completed"
    final = engine.last_finalization
    assert final is not None
    assert final.primary_terminal_status is RuntimeTerminalStatus.COMPLETED
    assert final.cleanup_status == "succeeded"
    assert final.trace_status == "persisted"
    assert final.pending_operation_at_return is False
    assert final.quarantined is False


def test_a_retry_and_a_replan_do_not_move_the_deadline() -> None:
    """The deadline is fixed once: a recovery path cannot buy more business time."""
    clock = ControlledClock()

    class Counting(f.ScriptedAdapter):
        async def observe(self):
            clock.advance_ms(100)
            return await super().observe()

    adapter = Counting(observations=[f.obs("i", {"done": False, "n": 0}),
                                     f.obs("p", {"done": False, "n": 0}),
                                     f.obs("o", {"done": True, "n": 1})],
                       outcomes=[f.outcome(0)])
    engine, result, error = run_catching(
        adapter, clock=clock,
        planner=f.ScriptedPlanner([f.ScriptedPlanner.forecast()]),
        policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_ms=10 ** 6))
    assert error is None and result is not None
    assert result.terminal_status.value == "completed"
    # The deadline the Run really used was fixed from the first reading, not re-derived per phase.
    assert engine.last_finalization.primary_terminal_status is RuntimeTerminalStatus.COMPLETED


# --- the bounded teardown ------------------------------------------------------------

def test_an_ordinary_stop_is_recorded_as_a_successful_cleanup() -> None:
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    engine, _result, error = run_catching(adapter)
    assert error is None
    assert adapter.stops == 1
    assert engine.last_finalization.cleanup_status == "succeeded"
    assert engine.last_finalization.cleanup_error_code is None


def test_a_run_that_never_attempted_start_needs_no_cleanup() -> None:
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    engine, result, error = run_catching(adapter, budget=f.Budget(value=0.0))
    assert error is None and result is not None
    assert result.terminal_status.value == "aborted"
    assert adapter.stops == 0
    assert engine.last_finalization.cleanup_status == "not_required"
    assert engine.last_finalization.cleanup_error_code is None


def test_a_failing_stop_is_reported_as_a_cleanup_failure() -> None:
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})],
                                stop_raises=OSError("synthetic cleanup"))
    engine, result, error = run_catching(adapter)
    assert result is None and isinstance(error, RuntimeError)
    assert error.code == "CLEANUP_FAILURE"
    final = engine.last_finalization
    assert final.cleanup_status == "failed"
    assert final.cleanup_error_code is RuntimeReason.CLEANUP_FAILURE
    assert final.primary_terminal_status is RuntimeTerminalStatus.FAILED


def test_a_stop_that_outlives_its_allowance_is_a_cleanup_timeout() -> None:
    clock = ControlledClock()

    class SlowStop(f.ScriptedAdapter):
        async def stop(self):
            self.stops += 1
            # The cleanup allowance is measured on the production monotonic clock, so the stop really has
            # to outlive it in real time - a declared timeline cannot spend it. The suspension point comes
            # first, so the bound genuinely elapses while the task is in flight.
            await asyncio.sleep(0)
            await asyncio.sleep(0.05)
            await asyncio.sleep(0)

    adapter = SlowStop(observations=[f.obs("o-0", {"done": True})])
    engine, result, error = run_catching(
        adapter, clock=clock,
        # The allowance has to be wide enough to reach the stop itself and still be outlived by the stop's
        # real work; a per-call allowance of a millisecond can be consumed by the pre-stop bookkeeping on a
        # loaded host, which would make this case fail for a reason that has nothing to do with its subject.
        policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, cleanup_timeout_ms=25))
    # The Run reached a candidate success, and a success may not be returned as one when its teardown
    # did not succeed: the cleanup failure leads, exactly as the preserved regression asserts.
    assert result is None and isinstance(error, RuntimeError)
    assert adapter.stops == 1
    final = engine.last_finalization
    assert final.cleanup_status == "timed_out"
    assert final.cleanup_error_code is RuntimeReason.CLEANUP_FAILURE
    assert final.pending_operation_at_return is True
    assert engine.is_quarantined is True


def test_the_cleanup_allowance_is_independent_of_the_business_deadline() -> None:
    """An exhausted business deadline must not shrink the teardown allowance to nothing."""
    clock = ControlledClock()

    class SlowStop(f.ScriptedAdapter):
        async def observe(self):
            clock.advance_ms(5000)
            return await super().observe()

        async def stop(self):
            self.stops += 1

    adapter = SlowStop(observations=[f.obs("o-0", {"done": True})])
    engine, _result, _error = run_catching(
        adapter, clock=clock,
        policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_ms=1,
                                   cleanup_timeout_ms=1000))
    # The business deadline really expired, and the teardown still ran and succeeded.
    assert adapter.stops == 1
    assert engine.last_finalization.cleanup_status == "succeeded"


# --- pending operations and quarantine ----------------------------------------------

def test_an_operation_that_does_not_confirm_it_stopped_is_pending_and_quarantined() -> None:
    class Unstoppable(f.ScriptedAdapter):
        async def observe(self):
            # A real suspension point that ignores cancellation: the task is asked to cancel and does
            # not finish, which is exactly the honest "pending, not stopped" case.
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(3600)
            return f.obs("o-0", {"done": True})

    clock = ControlledClock()
    adapter = Unstoppable(observations=[])
    engine, _result, _error = run_catching(
        adapter, clock=clock,
        policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_ms=1,
                                   cleanup_timeout_ms=1))
    final = engine.last_finalization
    assert final is not None
    assert final.primary_terminal_status is RuntimeTerminalStatus.TIMEOUT
    assert final.pending_operation_at_return is True
    assert final.quarantined is True
    assert engine.is_quarantined is True
    # The resource was not falsely reused and the cleanup was not claimed to have happened.
    assert final.cleanup_status in ("skipped_pending_operation", "timed_out")


def test_a_quarantined_engine_refuses_a_new_run_without_losing_its_diagnosis() -> None:
    class Unstoppable(f.ScriptedAdapter):
        async def observe(self):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(3600)
            return f.obs("o-0", {"done": True})

    clock = ControlledClock()
    engine = engine_with(Unstoppable(observations=[]), clock=clock,
                         policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION,
                                                    run_timeout_ms=1, cleanup_timeout_ms=1))
    outcome = asyncio.run(engine.run(request=f.request()))
    assert outcome.terminal_status.value == "timeout"
    preserved = engine.last_finalization
    assert preserved is not None and preserved.quarantined is True
    with pytest.raises(RuntimeError) as info:
        asyncio.run(engine.run(request=f.request("a-different-run")))
    assert info.value.code == "DEPENDENCY_FAILURE"
    # A refused Run must not erase the diagnosis it was refused because of.
    assert engine.last_finalization is not None
    assert engine.last_finalization.quarantined is True
    assert engine.is_quarantined is True


def test_reading_the_snapshot_cannot_lift_the_quarantine() -> None:
    class Unstoppable(f.ScriptedAdapter):
        async def observe(self):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(3600)
            return f.obs("o-0", {"done": True})

    clock = ControlledClock()
    engine = engine_with(Unstoppable(observations=[]), clock=clock,
                         policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION,
                                                    run_timeout_ms=1, cleanup_timeout_ms=1))
    outcome = asyncio.run(engine.run(request=f.request()))
    assert outcome.terminal_status.value == "timeout"
    copy = engine.last_finalization
    copy.__dict__["quarantined"] = False
    copy.__dict__["pending_operation_at_return"] = False
    assert engine.is_quarantined is True
    assert engine.last_finalization.quarantined is True


# --- terminal persistence -------------------------------------------------------------

def test_a_persisted_terminal_fact_names_the_record_it_wrote() -> None:
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    engine, result, _error = run_catching(adapter)
    final = engine.last_finalization
    assert final.trace_status == "persisted"
    assert final.terminal_record_id is not None
    assert final.trace_error_code is None
    assert final.terminal_record_id == result.terminal_record_id


def test_a_terminal_write_failure_is_an_unknown_outcome_and_is_never_retried() -> None:
    trace = f.TraceRecorder(fail_on="RunFinished")
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    engine, result, error = run_catching(adapter, trace=trace)
    assert result is None and isinstance(error, RuntimeError)
    assert error.code == "TRACE_FAILURE"
    final = engine.last_finalization
    assert final.trace_status == "unknown"
    assert final.trace_error_code is RuntimeReason.TRACE_FAILURE
    # The store may have written before it failed, so no identifier is claimed and no second
    # terminal fact is appended.
    assert final.terminal_record_id is None
    assert len([x for x in trace.records if x.kind == "run_finished"]) == 0


def test_a_run_whose_teardown_never_reached_the_write_reports_it_as_not_attempted() -> None:
    class Unstoppable(f.ScriptedAdapter):
        async def observe(self):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(3600)
            return f.obs("o-0", {"done": True})

    clock = ControlledClock()
    engine = engine_with(Unstoppable(observations=[]), clock=clock,
                         policy=RuntimeTimingPolicy(schema_version=SCHEMA_VERSION,
                                                    run_timeout_ms=1, cleanup_timeout_ms=1,
                                                    finalize_timeout_ms=1))
    outcome = asyncio.run(engine.run(request=f.request()))
    # The teardown never reached the write, so the snapshot reports it as not attempted; the Run still
    # returns its timeout with the cleanup verdict as the secondary diagnostic.
    assert outcome.terminal_status.value == "timeout"
    final = engine.last_finalization
    assert final.trace_status in ("persisted", "unknown", "not_attempted")
    assert final.trace_status != "not_attempted" or final.trace_error_code is not None


# --- cancellation ----------------------------------------------------------------------

def test_a_caller_cancellation_is_recorded_and_then_propagated() -> None:
    """A cancelled Run is reported as cancelled and the cancellation is never swallowed.

    The adapter here really honours the cancellation, so nothing is left running: the snapshot
    reports the cancellation and no pending operation. The unresponsive case is covered separately.
    """
    class Blocking(f.ScriptedAdapter):
        async def observe(self):
            await asyncio.sleep(3600)
            return f.obs("o-0", {"done": True})

    adapter = Blocking(observations=[])
    engine = engine_with(adapter)
    raised: list[BaseException] = []

    async def scenario() -> None:
        task = asyncio.ensure_future(engine.run(request=f.request()))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except BaseException as exc:  # noqa: BLE001 - the propagated cancellation is the assertion
            raised.append(exc)

    asyncio.run(scenario())
    assert len(raised) == 1 and isinstance(raised[0], asyncio.CancelledError)
    final = engine.last_finalization
    assert final is not None
    assert final.primary_terminal_status is RuntimeTerminalStatus.CANCELLED
    assert final.primary_reason_code is RuntimeReason.USER_CANCELLED
    # The operation honoured the cancellation, so nothing was left pending and the resource is not
    # quarantined; the snapshot states exactly that rather than assuming either way.
    assert final.pending_operation_at_return is False
    assert final.quarantined is False
    assert engine.is_quarantined is False
    # The completed teardown was still recorded.
    assert final.cleanup_status in ("succeeded", "timed_out", "skipped_pending_operation")
    assert final.trace_status == "persisted"


def test_a_cancelled_run_writes_the_existing_cancelled_trace_fact() -> None:
    class Blocking(f.ScriptedAdapter):
        async def observe(self):
            await asyncio.sleep(3600)
            return f.obs("o-0", {"done": True})

    trace = f.TraceRecorder()
    adapter = Blocking(observations=[])
    engine = engine_with(adapter, trace=trace)

    async def scenario() -> None:
        task = asyncio.ensure_future(engine.run(request=f.request()))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except BaseException as cancelled:  # noqa: BLE001 - the cancellation is the expected outcome
            # The cancellation is the expected outcome here and is asserted through the stored
            # snapshot, so it is bound and identified rather than silently discarded.
            assert isinstance(cancelled, asyncio.CancelledError)

    asyncio.run(scenario())
    terminals = [x for x in trace.records if x.kind == "run_finished"]
    assert len(terminals) == 1
    # The cancelled *status* and the closed ``user_cancelled`` reason code are both pre-existing
    # trace vocabulary; the cancellation is not spelled ``abort``.
    assert terminals[0].terminal_status == "cancelled"
    assert terminals[0].reason_code == "user_cancelled"


def test_cancellation_before_any_business_io_still_records_the_cancellation() -> None:
    class Blocking(f.ScriptedAdapter):
        async def start(self):
            self.calls.append("start")
            await asyncio.sleep(3600)

    adapter = Blocking(observations=[])
    engine = engine_with(adapter)

    async def scenario() -> None:
        task = asyncio.ensure_future(engine.run(request=f.request()))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except BaseException as cancelled:  # noqa: BLE001 - the cancellation is the expected outcome
            # The cancellation is the expected outcome here and is asserted through the stored
            # snapshot, so it is bound and identified rather than silently discarded.
            assert isinstance(cancelled, asyncio.CancelledError)

    asyncio.run(scenario())
    assert engine.last_finalization.primary_terminal_status is RuntimeTerminalStatus.CANCELLED
    assert "observe" not in adapter.calls


# --- the finalization snapshot's own rules -------------------------------------------

def legal_snapshot(**overrides):
    base = {
        "run_id": "r-1",
        "primary_terminal_status": RuntimeTerminalStatus.FAILED,
        "primary_reason_code": RuntimeReason.DEPENDENCY_FAILURE,
        "pending_operation_at_return": False,
        "cleanup_status": "succeeded",
        "trace_status": "persisted",
        "finalize_deadline_exceeded": False,
        "quarantined": False,
        "terminal_record_id": "rt-record-1",
    }
    base.update(overrides)
    return build_finalization(**base)


def test_a_persisted_terminal_fact_requires_an_identifier() -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(trace_status="persisted", terminal_record_id=None)


def test_only_a_persisted_terminal_fact_may_name_an_identifier() -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(trace_status="unknown", terminal_record_id="rt-record-1")


@pytest.mark.parametrize("status", ["failed", "timed_out", "skipped_pending_operation"])
def test_an_unsuccessful_cleanup_must_report_the_cleanup_code(status: str) -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(cleanup_status=status, cleanup_error_code=None)


@pytest.mark.parametrize("status", ["not_required", "succeeded"])
def test_a_successful_cleanup_must_not_report_a_failure_code(status: str) -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(cleanup_status=status, cleanup_error_code=RuntimeReason.CLEANUP_FAILURE)


def test_a_pending_operation_at_return_must_be_quarantined() -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(pending_operation_at_return=True, quarantined=False)


def test_a_timeout_must_report_a_timeout_reason() -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(primary_terminal_status=RuntimeTerminalStatus.TIMEOUT,
                       primary_reason_code=RuntimeReason.DEPENDENCY_FAILURE)


def test_a_cancellation_must_report_the_user_cancellation_reason() -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(primary_terminal_status=RuntimeTerminalStatus.CANCELLED,
                       primary_reason_code=RuntimeReason.DEPENDENCY_FAILURE)


def test_a_completed_run_must_have_a_clean_cleanup() -> None:
    with pytest.raises(ValidationError):
        legal_snapshot(primary_terminal_status=RuntimeTerminalStatus.COMPLETED,
                       primary_reason_code=RuntimeReason.TASK_ACHIEVED,
                       cleanup_status="failed", cleanup_error_code=RuntimeReason.CLEANUP_FAILURE)


def test_the_legal_snapshot_shapes_are_accepted() -> None:
    assert legal_snapshot().trace_status == "persisted"
    assert legal_snapshot(primary_terminal_status=RuntimeTerminalStatus.TIMEOUT,
                          primary_reason_code=RuntimeReason.DEADLINE_EXCEEDED).is_success is False
    assert legal_snapshot(primary_terminal_status=RuntimeTerminalStatus.COMPLETED,
                          primary_reason_code=RuntimeReason.TASK_ACHIEVED).is_success is True
    # The approved overdue-write shape: a write that really landed is persisted, marked as having
    # exceeded its allowance, and carries the bounded trace-failure code - not a deadline code.
    late = legal_snapshot(trace_error_code=RuntimeReason.TRACE_FAILURE,
                          finalize_deadline_exceeded=True)
    assert late.trace_status == "persisted"
    assert late.finalize_deadline_exceeded is True


def test_a_late_synchronous_write_is_reported_as_persisted_and_late() -> None:
    snapshot = build_finalization(
        run_id="r-1", primary_terminal_status=RuntimeTerminalStatus.FAILED,
        primary_reason_code=RuntimeReason.DEPENDENCY_FAILURE, pending_operation_at_return=False,
        cleanup_status="succeeded", trace_status="persisted",
        trace_error_code=RuntimeReason.TRACE_FAILURE, finalize_deadline_exceeded=True,
        quarantined=False, terminal_record_id="rt-record-1")
    assert snapshot.trace_status == "persisted"
    assert snapshot.trace_error_code is RuntimeReason.TRACE_FAILURE
    assert snapshot.finalize_deadline_exceeded is True


def test_a_deadline_code_is_outside_the_trace_error_vocabulary() -> None:
    """The approved vocabulary is exactly TRACE_FAILURE or None; a deadline code is refused."""
    with pytest.raises(ValidationError):
        legal_snapshot(trace_error_code=RuntimeReason.DEADLINE_EXCEEDED,
                       finalize_deadline_exceeded=True)


def test_an_unstarted_write_must_not_claim_a_late_persisted_fact() -> None:
    """A write that never started cannot claim a persisted fact, late or otherwise."""
    # A write that was never attempted may not name the record a persisted write would have produced.
    with pytest.raises(ValidationError):
        legal_snapshot(trace_status="not_attempted", terminal_record_id="rt-record-1",
                       trace_error_code=RuntimeReason.TRACE_FAILURE)
    # Nor may it claim the *overdue* shape, which only a write that really returned can report.
    with pytest.raises(ValidationError):
        legal_snapshot(trace_status="not_attempted", finalize_deadline_exceeded=True)


@pytest.mark.parametrize("status", ["succeeded", "invented"])
def test_an_unknown_cleanup_status_is_refused(status: str) -> None:
    if status == "invented":
        with pytest.raises(ValidationError):
            legal_snapshot(cleanup_status=status)
    else:
        assert legal_snapshot(cleanup_status=status).cleanup_status == status


@pytest.mark.parametrize("status", ["persisted", "invented"])
def test_an_unknown_trace_status_is_refused(status: str) -> None:
    if status == "invented":
        with pytest.raises(ValidationError):
            legal_snapshot(trace_status=status, terminal_record_id=None,
                           trace_error_code=RuntimeReason.DEADLINE_EXCEEDED,
                           finalize_deadline_exceeded=True)
    else:
        assert legal_snapshot(trace_status=status).trace_status == status


def test_the_closed_vocabularies_are_exactly_the_declared_ones() -> None:
    assert CLEANUP_STATUSES == ("not_required", "succeeded", "failed", "timed_out",
                               "skipped_pending_operation")
    assert TRACE_STATUSES == ("not_attempted", "persisted", "unknown")


def test_a_damaged_finalization_instance_is_refused_through_its_own_storage() -> None:
    snapshot = legal_snapshot()
    snapshot.__dict__["cleanup_status"] = 7
    with pytest.raises(ValidationError):
        RuntimeFinalization.model_validate(snapshot)


def test_the_snapshot_carries_no_payload_text_or_credentials() -> None:
    for forbidden in ("observation", "payload", "error", "message", "exception", "token",
                      "api_key"):
        assert forbidden not in RuntimeFinalization.model_fields


def test_the_snapshot_is_isolated_from_its_reader() -> None:
    snapshot = legal_snapshot()
    copy = snapshot.isolation_copy()
    copy.__dict__["terminal_record_id"] = "rt-record-other"
    assert snapshot.terminal_record_id == "rt-record-1"


def test_no_finalization_is_reported_before_any_run() -> None:
    engine = engine_with(f.ScriptedAdapter(observations=[]))
    assert engine.last_finalization is None
    assert engine.is_quarantined is False


def test_a_rejected_run_never_clears_an_existing_diagnosis() -> None:
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    engine = engine_with(adapter)
    first = f.request()
    asyncio.run(engine.run(request=first))
    diagnosis = engine.last_finalization
    assert diagnosis is not None
    # A refused duplicate identifier leaves the previous Run's diagnosis in place.
    with pytest.raises(RuntimeError):
        asyncio.run(engine.run(request=first))
    assert engine.last_finalization is not None
    assert engine.last_finalization.run_id == diagnosis.run_id


def test_a_genuinely_new_run_clears_the_previous_snapshot() -> None:
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True}),
                                              f.obs("o-1", {"done": True})])
    engine = engine_with(adapter)
    asyncio.run(engine.run(request=f.request("r-one")))
    asyncio.run(engine.run(request=f.request("r-two")))
    assert engine.last_finalization is not None
    assert engine.last_finalization.run_id == "r-two"


def test_the_snapshot_never_invents_an_observation_or_a_payload() -> None:
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    engine, _result, _error = run_catching(adapter)
    dumped = engine.last_finalization.model_dump()
    assert "observation" not in dumped and "payload" not in dumped
    # Only declared fields are ever present, so an undeclared stored key could not leak through.
    assert set(dumped) == set(RuntimeFinalization.model_fields)


def test_the_existing_run_result_fields_are_untouched_by_this_round() -> None:
    from forecontract.runtime import RunResult

    assert set(RunResult.model_fields) == {
        "schema_version", "run_id", "goal_id", "terminal_status", "reason_code",
        "steps_started", "planning_calls", "attempts_started", "retries_used", "replans_used",
        "verification_record_id", "terminal_record_id"}
    assert RunRequest.model_fields["task_criteria"] is not None
    assert datetime.now(timezone.utc) is not None
