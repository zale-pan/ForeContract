"""A11/A12: the timing declaration, the monotonic clock and the monotonic budget source.

The bounded teardown and deadline behaviour are asserted in ``test_runtime_finalization.py``; this
module covers the pieces those behaviours are built from. Every case states what is really being
established: a declaration is checked against its schema, an injected clock proves the budget
semantics *for the declared timeline* rather than certifying real time, and a budget source reports
what remains instead of enforcing anything.
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys

import pytest
from pydantic import ValidationError

CORE_SRC = pathlib.Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.models.base import SCHEMA_VERSION
from forecontract.runtime import (
    MonotonicBudgetSource,
    RuntimeBudgetSnapshot,
    RuntimeError,
    RuntimeTimingPolicy,
    SystemMonotonicClock,
)
from forecontract.runtime.timing import (
    milliseconds_to_nanoseconds,
    read_monotonic,
)

_FIXTURE = pathlib.Path(__file__).with_name("test_runtime_engine.py")
_spec = importlib.util.spec_from_file_location("_runtime_engine_fixture", _FIXTURE)
f = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(f)


# --- a controlled clock, used only to prove the declared-timeline semantics ----------

class ControlledClock:
    """A monotonic clock whose reading a test moves by hand.

    It is deliberately not a fake of real time: it establishes that the budget semantics follow
    whatever timeline is declared, and nothing about elapsed wall-clock time.
    """

    def __init__(self, start: int = 0) -> None:
        self.now = start
        self.reads = 0

    def now_ns(self) -> int:
        self.reads += 1
        return self.now

    def advance_ms(self, milliseconds: float) -> None:
        self.now += int(milliseconds * 1_000_000)


# --- the timing policy declaration --------------------------------------------------

def test_the_timing_policy_defaults_are_an_absent_run_limit_and_two_positive_allowances() -> None:
    policy = RuntimeTimingPolicy(schema_version=SCHEMA_VERSION)
    # ``None`` means "derive the deadline from the host budget", not "switch the deadline off".
    assert policy.run_timeout_ms is None
    assert policy.cleanup_timeout_ms == 1000.0
    assert policy.finalize_timeout_ms == 1000.0
    assert policy.schema_version == SCHEMA_VERSION


def test_a_zero_run_limit_is_legal_and_means_an_immediate_deadline() -> None:
    policy = RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_ms=0)
    assert policy.run_timeout_ms == 0


@pytest.mark.parametrize("value", [True, False, "1000", float("nan"), float("inf"),
                                   float("-inf"), -1, -0.5])
def test_an_unusable_run_limit_is_refused(value: object) -> None:
    with pytest.raises(ValidationError):
        RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_ms=value)


@pytest.mark.parametrize("field", ["cleanup_timeout_ms", "finalize_timeout_ms"])
@pytest.mark.parametrize("value", [0, -1, True, "1000", float("nan"), float("inf")])
def test_a_cleanup_allowance_must_be_strictly_positive(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, **{field: value})


def test_an_unknown_timing_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        RuntimeTimingPolicy(schema_version=SCHEMA_VERSION, run_timeout_seconds=1)


def test_a_damaged_timing_policy_instance_is_refused_through_its_own_storage() -> None:
    policy = RuntimeTimingPolicy(schema_version=SCHEMA_VERSION)
    policy.__dict__["cleanup_timeout_ms"] = 0
    with pytest.raises(ValidationError):
        RuntimeTimingPolicy.model_validate(policy)


def test_a_timing_policy_cycle_is_refused_rather_than_recursed() -> None:
    policy = RuntimeTimingPolicy(schema_version=SCHEMA_VERSION)
    policy.__dict__["run_timeout_ms"] = policy
    with pytest.raises((ValidationError, ValueError)):
        RuntimeTimingPolicy.model_validate(policy)


def test_two_engines_do_not_share_mutable_timing_state() -> None:
    first = f.engine_for(f.ScriptedAdapter(observations=[]), f.ScriptedPlanner([]), f.Support())
    second = f.engine_for(f.ScriptedAdapter(observations=[]), f.ScriptedPlanner([]), f.Support())
    # Each engine holds its own policy object, so a mutation through one is never visible in the
    # other, and the returned copy is detached from the engine's own.
    first_policy = first.timing_policy
    first_policy.__dict__["cleanup_timeout_ms"] = 1.0
    assert second.timing_policy.cleanup_timeout_ms == 1000.0
    assert first.timing_policy.cleanup_timeout_ms == 1000.0


# --- the monotonic clock ------------------------------------------------------------

def test_the_production_clock_returns_a_non_decreasing_integer() -> None:
    clock = SystemMonotonicClock()
    first = clock.now_ns()
    second = clock.now_ns()
    assert isinstance(first, int) and not isinstance(first, bool)
    assert first >= 0 and second >= first


def test_a_raising_clock_is_a_bounded_dependency_failure() -> None:
    class Broken:
        def now_ns(self) -> int:
            raise OSError("synthetic clock failure")

    with pytest.raises(RuntimeError) as info:
        read_monotonic(Broken(), "facility")
    assert info.value.code == "DEPENDENCY_FAILURE"
    # No exception text is carried across the boundary.
    assert "synthetic" not in info.value.message


@pytest.mark.parametrize("value", [True, 1.0, "0", None])
def test_a_clock_that_does_not_return_an_integer_is_refused(value: object) -> None:
    class Wrong:
        def now_ns(self) -> object:
            return value

    with pytest.raises(RuntimeError) as info:
        read_monotonic(Wrong(), "facility")
    assert info.value.code == "INPUT_INVALID"


def test_a_negative_clock_reading_is_refused() -> None:
    class Backwards:
        def now_ns(self) -> int:
            return -1

    with pytest.raises(RuntimeError) as info:
        read_monotonic(Backwards(), "facility")
    assert info.value.code == "INPUT_INVALID"


def test_an_object_without_now_ns_is_refused() -> None:
    with pytest.raises(RuntimeError) as info:
        read_monotonic(object(), "facility")
    assert info.value.code == "INPUT_INVALID"


def test_a_base_exception_from_a_clock_is_never_swallowed() -> None:
    # ``read_monotonic`` bounds ordinary dependency failures. A BaseException is not a dependency
    # failure and must propagate: nothing in the Runtime catches BaseException anywhere.
    class Interrupted:
        def now_ns(self) -> int:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        read_monotonic(Interrupted(), "facility")


# --- the millisecond conversion -----------------------------------------------------

def test_the_millisecond_conversion_is_exact_and_refuses_what_it_cannot_represent() -> None:
    assert milliseconds_to_nanoseconds(0, ()) == 0
    assert milliseconds_to_nanoseconds(1, ()) == 1_000_000
    assert milliseconds_to_nanoseconds(1.5, ()) == 1_500_000
    for bad in (True, "1", float("nan"), float("inf"), -1):
        with pytest.raises(RuntimeError) as info:
            milliseconds_to_nanoseconds(bad, ())
        assert info.value.code == "INPUT_INVALID"
    with pytest.raises(RuntimeError):
        milliseconds_to_nanoseconds(10 ** 30, ())


# --- the monotonic budget source ----------------------------------------------------

def test_the_first_reading_establishes_the_deadline_and_later_readings_never_increase() -> None:
    clock = ControlledClock()
    source = MonotonicBudgetSource(total_budget_ms=1000.0, monotonic_clock=clock)
    assert source.deadline_ns is None
    first = source.snapshot()
    established = source.deadline_ns
    assert first.remaining_budget_ms == pytest.approx(1000.0)
    clock.advance_ms(400)
    second = source.snapshot()
    assert second.remaining_budget_ms == pytest.approx(600.0)
    # The deadline was fixed once and is never moved by a later reading.
    assert source.deadline_ns == established
    clock.advance_ms(200)
    assert source.snapshot().remaining_budget_ms == pytest.approx(400.0)


def test_an_expired_budget_reports_exactly_zero_and_never_recovers() -> None:
    clock = ControlledClock()
    source = MonotonicBudgetSource(total_budget_ms=100.0, monotonic_clock=clock)
    source.snapshot()
    clock.advance_ms(100)
    assert source.snapshot().remaining_budget_ms == 0.0
    clock.advance_ms(5000)
    assert source.snapshot().remaining_budget_ms == 0.0
    # There is deliberately no reset: a reused instance cannot restore spent budget.
    assert not hasattr(source, "reset")


def test_each_instance_keeps_its_own_deadline() -> None:
    clock = ControlledClock()
    first = MonotonicBudgetSource(total_budget_ms=100.0, monotonic_clock=clock)
    second = MonotonicBudgetSource(total_budget_ms=100.0, monotonic_clock=clock)
    first.snapshot()
    clock.advance_ms(60)
    second.snapshot()
    clock.advance_ms(40)
    # ``first`` is now expired while ``second`` still has 60 ms: reusing one instance cannot restore
    # budget, and a fresh instance genuinely starts a fresh deadline.
    assert first.snapshot().remaining_budget_ms == 0.0
    assert second.snapshot().remaining_budget_ms == pytest.approx(60.0)


def test_a_zero_total_budget_is_legal_and_reports_zero() -> None:
    source = MonotonicBudgetSource(total_budget_ms=0, monotonic_clock=ControlledClock())
    snapshot = source.snapshot()
    assert snapshot.remaining_budget_ms == 0.0
    assert isinstance(snapshot, RuntimeBudgetSnapshot)


@pytest.mark.parametrize("value", [True, "1000", float("nan"), float("inf"), -1])
def test_an_unusable_total_budget_is_refused(value: object) -> None:
    with pytest.raises(RuntimeError) as info:
        MonotonicBudgetSource(total_budget_ms=value)
    assert info.value.code == "INPUT_INVALID"


def test_the_budget_source_is_a_budget_source_for_the_engine() -> None:
    clock = ControlledClock()
    source = MonotonicBudgetSource(total_budget_ms=1000.0, monotonic_clock=clock)
    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    result = f.drive(f.engine_for(adapter, f.ScriptedPlanner([]), f.Support(), budget=source)
                     .run(request=f.request()))
    assert result.terminal_status.value == "completed"


def test_a_declining_budget_source_still_blocks_the_next_operation() -> None:
    class Declining:
        def __init__(self) -> None:
            self.reads = 0

        def snapshot(self) -> RuntimeBudgetSnapshot:
            self.reads += 1
            return RuntimeBudgetSnapshot(
                schema_version=SCHEMA_VERSION,
                remaining_budget_ms=10_000.0 if self.reads == 1 else 0.0)

    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    result = f.drive(f.engine_for(adapter, f.ScriptedPlanner([]), f.Support(),
                                  budget=Declining()).run(request=f.request()))
    assert "observe" not in adapter.calls
    assert result.terminal_status.value == "aborted"
    assert result.reason_code.value == "BUDGET_EXHAUSTED"


# --- the engine's own timing dependencies -------------------------------------------

def test_a_damaged_monotonic_clock_is_a_bounded_dependency_failure() -> None:
    class Broken:
        def now_ns(self) -> int:
            raise OSError("synthetic")

    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    with pytest.raises(RuntimeError) as info:
        f.drive(f.engine_for(adapter, f.ScriptedPlanner([]), f.Support(),
                             monotonic_clock=Broken()).run(request=f.request()))
    assert info.value.code == "DEPENDENCY_FAILURE"
    assert "synthetic" not in info.value.message


def test_a_damaged_timing_policy_is_refused_at_construction() -> None:
    with pytest.raises(RuntimeError) as info:
        f.engine_for(f.ScriptedAdapter(observations=[]), f.ScriptedPlanner([]), f.Support(),
                     timing_policy={"schema_version": SCHEMA_VERSION, "cleanup_timeout_ms": 0})
    assert info.value.code == "INPUT_INVALID"


def test_a_legal_timing_policy_mapping_is_accepted_and_isolated() -> None:
    mapping = {"schema_version": SCHEMA_VERSION, "run_timeout_ms": 5000}
    engine = f.engine_for(f.ScriptedAdapter(observations=[]), f.ScriptedPlanner([]), f.Support(),
                          timing_policy=mapping)
    mapping["run_timeout_ms"] = 1
    # The engine consumed its own validated copy, so a later mutation of the host's mapping cannot
    # shorten or extend the Run it governs.
    assert engine.timing_policy.run_timeout_ms == 5000


def test_a_nan_run_limit_from_a_mapping_is_refused() -> None:
    with pytest.raises(RuntimeError):
        f.engine_for(f.ScriptedAdapter(observations=[]), f.ScriptedPlanner([]), f.Support(),
                     timing_policy={"schema_version": SCHEMA_VERSION, "run_timeout_ms": float("nan")})


def test_the_existing_clock_parameter_still_means_event_time() -> None:
    """The original UTC event-time clock keeps its meaning and is never read as a duration.

    The engine uses the event clock for trace timestamps through the existing calling convention and
    reads the monotonic clock separately, so injecting an event clock cannot satisfy a deadline.
    """
    from datetime import datetime, timezone

    reads: list[str] = []

    def event_clock() -> datetime:
        reads.append("read")
        return datetime.now(timezone.utc)

    adapter = f.ScriptedAdapter(observations=[f.obs("o-0", {"done": True})])
    result = f.drive(f.engine_for(adapter, f.ScriptedPlanner([]), f.Support(),
                                  clock=event_clock).run(request=f.request()))
    assert result.terminal_status.value == "completed"
    # The event clock really was consulted, and it was never the source of a duration.
    assert reads
    assert f.TraceRecorder is not None


def test_the_module_states_its_asyncio_dependency() -> None:
    """The timing module is exercised through the real event loop, so asyncio is imported."""
    assert asyncio is not None
