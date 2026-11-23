"""The timing policy, the monotonic clock protocol and a monotonic budget source.

This module supplies the timing pieces the Runtime needs and nothing more. Three things are worth
stating plainly, because they bound what the rest of this task can claim:

* a monotonic clock is a *declaration*. The production clock reads the standard library's monotonic
  counter, and an injected clock lets a test exercise the budget semantics against a declared
  timeline. Neither authenticates real time, and neither is an operating-system guarantee.
* a deadline is only meaningful because the Runtime checks it. A synchronous dependency cannot be
  preempted in the same event loop, so its elapsed time is charged when it returns; the clock does
  not interrupt it. Nothing here wraps a synchronous call in an uncancellable background thread.
* the budget source reports what remains. It does not enforce anything: the bounds on how long the
  Runtime waits are the engine's, applied at the boundaries the contract names.
"""

from __future__ import annotations

import math
import time
from typing import Annotated, Protocol

from pydantic import BeforeValidator, Field

from forecontract.models.base import SCHEMA_VERSION, SchemaVersion, StrictInt
from forecontract.runtime.errors import RuntimeError
from forecontract.runtime.models import (
    RuntimeBudgetSnapshot,
    RuntimeRecord,
    isolated_record,
    stored_payload,
)

__all__ = [
    "MonotonicBudgetSource",
    "MonotonicClock",
    "MonotonicReading",
    "RuntimeTimingPolicy",
    "SystemMonotonicClock",
    "milliseconds_to_nanoseconds",
    "read_monotonic",
]

#: Nanoseconds in one millisecond. The conversion is explicit so a millisecond declaration can never
#: be silently treated as a nanosecond count.
_NS_PER_MS = 1_000_000

#: The largest nanosecond count the Runtime will represent. Beyond this an int is not a usable
#: timeline: converting it would either overflow or silently wrap, so it is refused instead.
_MAX_NS = 1 << 62


def _finite_number(value: object) -> object:
    """Accept only a finite JSON number: never a bool, a string or a non-finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Pydantic requires ValueError here so the rejection becomes a validation error.
        raise ValueError("must be a JSON number")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("must be a finite JSON number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("must be a finite non-negative JSON number")
    return value


#: A millisecond duration: a finite non-negative number that is not a bool.
TimeoutMs = Annotated[float, BeforeValidator(_finite_number), Field(allow_inf_nan=False, ge=0)]

#: A positive millisecond duration: finite, not a bool, and strictly greater than zero.
PositiveTimeoutMs = Annotated[
    float, BeforeValidator(_finite_number), Field(allow_inf_nan=False, gt=0)
]

#: A monotonic reading: a strict non-negative integer, so a bool and a float are both refused.
Nanoseconds = Annotated[int, StrictInt, Field(ge=0, strict=True)]

#: The default cleanup allowance, in milliseconds.
DEFAULT_CLEANUP_TIMEOUT_MS = 1000.0

#: The default finalization allowance, in milliseconds.
DEFAULT_FINALIZE_TIMEOUT_MS = 1000.0


def milliseconds_to_nanoseconds(value: object, location: tuple[str | int, ...]) -> int:
    """Convert a millisecond declaration to nanoseconds with a finite and overflow check.

    A value that is not a finite non-negative number, or whose nanosecond form cannot be represented,
    is refused rather than truncated into a legal-looking count. Truncating would turn an illegal
    declaration into a valid deadline, which is exactly the failure this check exists to prevent.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("INPUT_INVALID", "validate", location,
                           "the millisecond declaration is not a JSON number")
    try:
        number = float(value)
    except OverflowError:
        raise RuntimeError("INPUT_INVALID", "validate", location,
                           "the millisecond declaration is not a finite number") from None
    if not math.isfinite(number) or number < 0:
        raise RuntimeError("INPUT_INVALID", "validate", location,
                           "the millisecond declaration is not a finite non-negative number")
    nanoseconds = number * _NS_PER_MS
    if not math.isfinite(nanoseconds) or nanoseconds > _MAX_NS:
        raise RuntimeError("INPUT_INVALID", "validate", location,
                           "the millisecond declaration is outside the representable range")
    return int(nanoseconds)


class MonotonicClock(Protocol):
    """The synchronous, no-I/O monotonic clock the Runtime reads.

    A reading is a strict non-negative integer count of nanoseconds and must never decrease within
    one clock. The protocol states a calling convention; it cannot certify that a given clock is
    honest, and an injected clock proves only that the budget semantics follow the declared timeline.
    """

    def now_ns(self) -> int:
        """Return the current monotonic reading, in nanoseconds."""
        ...


class SystemMonotonicClock:
    """The production clock, backed by the standard library's monotonic counter.

    ``time.monotonic_ns`` is monotonic and unaffected by wall-clock adjustments, which is what a
    duration measurement needs. It says nothing about real elapsed time under an adjusted or virtual
    machine clock, and no claim beyond "monotonic within this process" is made.
    """

    __slots__ = ()

    def now_ns(self) -> int:
        """Return the current monotonic reading, in nanoseconds."""
        return int(time.monotonic_ns())

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        return "SystemMonotonicClock()"


class MonotonicReading:
    """A monotonic clock reader that enforces the non-decreasing rule on every reading.

    A clock is *declared* monotonic, and the declaration is checked rather than trusted: the first
    reading establishes the baseline and every later reading must be greater than or equal to it. A
    reading that went backwards would make a duration negative and a remaining budget grow, which is
    exactly the failure a deadline must not have, so a regression is one bounded refusal. The reader
    also owns the two ways a reading can be unusable, so one object states the whole clock contract.
    """

    __slots__ = ("_clock", "_last", "_stage")

    def __init__(self, clock: object, *, stage: str = "facility") -> None:
        self._clock = clock
        self._last: int | None = None
        self._stage = stage

    def read(self, location: tuple[str | int, ...] = ("monotonic_clock",)) -> int:
        """Return the next reading, refusing an unusable or non-monotonic one."""
        value = read_monotonic(self._clock, self._stage, location)
        if self._last is not None and value < self._last:
            # A reading that went backwards is never clamped to the previous one: clamping would hide
            # the regression and hand the caller a budget the clock does not support.
            raise RuntimeError("INPUT_INVALID", self._stage, location,
                               "the monotonic clock reading went backwards")
        self._last = value
        return value

    @property
    def last(self) -> int | None:
        """The most recent accepted reading, or ``None`` before the first one."""
        return self._last


def read_monotonic(clock: object, stage: str,
                   location: tuple[str | int, ...] = ("monotonic_clock",)) -> int:
    """Read a monotonic clock and bound every way the reading can be unusable.

    A clock that raises is a dependency failure; a clock that returns a bool, a float, a negative
    count or a non-integer is an invalid declaration. No exception text is carried: the reading is
    either a usable nanosecond count or one of two bounded failures.
    """
    reader = getattr(clock, "now_ns", None)
    if not callable(reader):
        raise RuntimeError("INPUT_INVALID", stage, location,
                           "the monotonic clock does not provide now_ns")
    try:
        value = reader()
    except Exception as exc:  # noqa: BLE001 - a clock failure is a bounded dependency failure
        raise RuntimeError("DEPENDENCY_FAILURE", stage, location,
                           "the monotonic clock raised an exception", cause=exc) from None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("INPUT_INVALID", stage, location,
                           "the monotonic clock did not return an integer reading")
    if value < 0:
        raise RuntimeError("INPUT_INVALID", stage, location,
                           "the monotonic clock returned a negative reading")
    return value


class RuntimeTimingPolicy(RuntimeRecord):
    """The bounded time allowances one Run is governed by.

    ``run_timeout_ms`` is optional and ``None`` means the business deadline is derived from the first
    legal host budget rather than switched off: a host that declares a budget still gets a deadline.
    ``cleanup_timeout_ms`` and ``finalize_timeout_ms`` are the two independent allowances for the
    bounded teardown, and both must be strictly positive. The record is a closed schema, so a bool, a
    numeric string, a non-finite number, an unknown key, a damaged nested instance or a cycle is
    refused by the same unified boundary every other Runtime record uses.
    """

    schema_version: SchemaVersion
    run_timeout_ms: TimeoutMs | None = None
    cleanup_timeout_ms: PositiveTimeoutMs = DEFAULT_CLEANUP_TIMEOUT_MS
    finalize_timeout_ms: PositiveTimeoutMs = DEFAULT_FINALIZE_TIMEOUT_MS


class MonotonicBudgetSource:
    """A host budget that counts down against a monotonic clock.

    The first :meth:`snapshot` establishes the deadline once, and every later reading reports what is
    left of it, never an increase. When the deadline has passed the reported remainder is exactly
    zero. There is deliberately no reset: a reused instance cannot restore budget it has already
    spent, which is why an independent instance per Run is the intended use.

    The instance only *reports* the remaining budget. The Runtime still reads it at the agreed
    boundaries and is what constrains how long an operation may wait; this class is not an
    enforcement mechanism and does not interrupt anything.
    """

    __slots__ = ("_deadline_ns", "_reader", "_total_ns")

    def __init__(self, *, total_budget_ms: float, monotonic_clock: object | None = None) -> None:
        if isinstance(total_budget_ms, bool) or not isinstance(total_budget_ms, (int, float)):
            raise RuntimeError("INPUT_INVALID", "validate", ("total_budget_ms",),
                               "the total budget is not a JSON number")
        try:
            number = float(total_budget_ms)
        except OverflowError:
            raise RuntimeError("INPUT_INVALID", "validate", ("total_budget_ms",),
                               "the total budget is not a finite number") from None
        if not math.isfinite(number) or number < 0:
            raise RuntimeError("INPUT_INVALID", "validate", ("total_budget_ms",),
                               "the total budget is not a finite non-negative number")
        self._total_ns = milliseconds_to_nanoseconds(number, ("total_budget_ms",))
        clock = SystemMonotonicClock() if monotonic_clock is None else monotonic_clock
        # A regression in the clock must never be read as extra budget, so the source carries its own
        # monotonic reader and refuses a backwards reading instead of reporting more remaining time.
        self._reader = MonotonicReading(clock, stage="validate")
        self._deadline_ns: int | None = None

    def snapshot(self) -> RuntimeBudgetSnapshot:
        """Report what remains of the budget this source established on its first reading."""
        now = self._reader.read(("budget_source", "monotonic_clock"))
        if self._deadline_ns is None:
            self._deadline_ns = now + self._total_ns
        remaining_ns = self._deadline_ns - now
        if remaining_ns <= 0:
            return RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION, remaining_budget_ms=0.0)
        return RuntimeBudgetSnapshot(schema_version=SCHEMA_VERSION,
                                     remaining_budget_ms=remaining_ns / _NS_PER_MS)

    @property
    def deadline_ns(self) -> int | None:
        """The deadline this source established, or ``None`` before the first reading."""
        return self._deadline_ns


def timing_policy_from(value: object) -> RuntimeTimingPolicy:
    """Rebuild a timing policy from its own storage, refusing anything the schema excludes."""
    if isinstance(value, RuntimeTimingPolicy):
        payload = stored_payload(value)
    elif isinstance(value, dict):
        payload = value
    else:
        raise RuntimeError("INPUT_INVALID", "validate", ("timing_policy",),
                           "the timing policy is not a declared record")
    try:
        return isolated_record(RuntimeTimingPolicy, payload, validate_fully=True)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("INPUT_INVALID", "validate", ("timing_policy",),
                           "the timing policy does not match its declared schema",
                           cause=exc) from None
