"""Per-instance, generation-scoped mutable state behind one lock.

One :class:`SiteState` belongs to exactly one :class:`~test_site.server.LocalTestSite`. It owns
the counters, the phase-event set, the scenario gates, the authenticated session set and the
active-request count. Every read or write happens under one re-entrant condition lock, and
:meth:`SiteState.snapshot` returns a detached frozen record rather than a live view.

The state is not shared between instances. ``reset`` is the only successful way to change the
scenario of a running instance: it requires an idle site, advances the generation and revokes
every old session, so an old form or an old cookie cannot reach the new generation.
"""

from __future__ import annotations

import secrets
import threading
import time
from enum import Enum
from typing import Final

from test_site.models import (
    LocalSiteError,
    Scenario,
    SiteEvent,
    SiteOracleSnapshot,
    _is_scenario,
)

__all__ = ["GateOutcome", "SiteState"]

_GATED_SCENARIOS: Final[frozenset[Scenario]] = frozenset(
    {Scenario.DELAYED_LOGIN, Scenario.COMMIT_THEN_STALL}
)


class GateOutcome(Enum):
    """Why one scenario gate wait ended."""

    RELEASED = "released"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"


def _wait_milliseconds(value: object) -> int:
    """Validate a host wait budget in ``1..60000`` without coercion.

    ``type(value) is int`` rejects ``bool`` as well, and unlike ``isinstance(value, bool)`` it
    never reads a caller-owned ``__class__`` or ``__getattribute__``.
    """
    if type(value) is not int:
        raise LocalSiteError("SITE_CONTROL_INVALID")
    if value < 1 or value > 60000:
        raise LocalSiteError("SITE_CONTROL_INVALID")
    return value


def _pause_within(deadline: float) -> bool:
    """Yield briefly without requesting a wait past ``deadline`` (high-resolution clock)."""
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return False
    time.sleep(0.001 if remaining >= 0.05 else 0)
    return True


def _acquire_within(lock: object, deadline: float) -> bool:
    """Acquire one internal coordination lock without waiting past ``deadline``.

    Windows rounds a positive lock timeout up to a ~15ms timer tick, which would silently extend
    a short cleanup budget. A non-blocking acquisition inside a bounded poll instead observes the
    caller's one absolute ``perf_counter`` deadline and returns ``False`` as soon as it expires.
    """
    while True:
        if lock.acquire(blocking=False):  # type: ignore[attr-defined]
            return True
        if not _pause_within(deadline):
            return False


class SiteState:
    """The mutable, locked state of one local test site instance."""

    def __init__(self, scenario: Scenario) -> None:
        if not _is_scenario(scenario):
            raise LocalSiteError("SITE_CONTROL_INVALID")
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._request_deadline = threading.local()
        self._scenario = scenario
        self._generation = 1
        self._running = False
        self._login_requests = 0
        self._auth_successes = 0
        self._auth_failures = 0
        self._commits = 0
        self._active_requests = 0
        self._blocked_waiters = 0
        self._events: set[SiteEvent] = set()
        self._sessions: set[str] = set()
        self._gate_open = scenario not in _GATED_SCENARIOS
        self._overlay_active = scenario is Scenario.BLOCKED_SUBMIT

    # ------------------------------------------------------------------ lifecycle

    def mark_running(self) -> None:
        """Publish the running fact; the listener is already bound at this point."""
        with self._condition:
            self._running = True
            self._condition.notify_all()

    def mark_stopping(self, deadline: float | None = None) -> bool:
        """Stop accepting new work and wake every gate and event waiter without committing.

        ``deadline`` is the caller's one absolute cleanup deadline. When supplied, even acquiring
        this state's coordination lock is bounded by it: a contended lock makes the attempt return
        ``False`` instead of delaying the whole cleanup past its budget. Without a deadline the
        call keeps the ordinary blocking semantics used by the start-failure path.
        """
        if deadline is None:
            with self._condition:
                self._running = False
                self._condition.notify_all()
            return True
        if not _acquire_within(self._lock, deadline):
            return False
        try:
            self._running = False
            self._condition.notify_all()
            return True
        finally:
            self._lock.release()

    @property
    def running(self) -> bool:
        """Whether this generation still accepts requests."""
        with self._lock:
            return self._running

    @property
    def scenario(self) -> Scenario:
        """The current generation's mode."""
        with self._lock:
            return self._scenario

    @property
    def generation(self) -> int:
        """The current generation number."""
        with self._lock:
            return self._generation

    def begin_request(self) -> bool:
        """Register one in-flight request, refusing work after the stop begins."""
        with self._condition:
            if not self._running:
                return False
            self._active_requests += 1
            self._condition.notify_all()
            return True

    def end_request(self) -> None:
        """Retire one in-flight request and wake any reset waiter."""
        with self._condition:
            if self._active_requests > 0:
                self._active_requests -= 1
            self._condition.notify_all()

    def wait_for_idle(self, deadline: float) -> bool:
        """Wait for every in-flight request to retire under the caller's one cleanup deadline.

        Cleanup uses this bounded wait so a request woken by the stop can deliver its fixed
        pre-commit refusal before its socket is closed. It never creates a fresh budget and
        returns ``False`` as soon as the passed absolute deadline has expired. Every acquisition
        of the coordination lock in this loop is itself bounded by the same deadline, so a lock
        held by another party cannot make cleanup wait past its budget. The wait polls on the
        high-resolution monotonic performance counter instead of using a lock timeout: on
        Windows a positive lock-wait timeout is rounded up to a ~15ms timer tick, which would
        silently extend a short cleanup budget.
        """
        while True:
            if not _acquire_within(self._lock, deadline):
                return False
            try:
                if self._active_requests == 0:
                    return True
            finally:
                self._lock.release()
            if not _pause_within(deadline):
                return False

    def active_request_count(self, deadline: float | None = None) -> int:
        """Read the live in-flight count within ``deadline`` without a false zero.

        The return value is always the stored integer. When the coordination lock cannot be
        acquired before the deadline the last stored value is still a real observed count (a
        single integer read), never an invented zero and never a fake idle proof.
        """
        if deadline is None:
            with self._lock:
                return self._active_requests
        if not _acquire_within(self._lock, deadline):
            return self._active_requests
        try:
            return self._active_requests
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ facts

    def snapshot(self) -> SiteOracleSnapshot:
        """Return a consistent snapshot detached from every mutable field."""
        with self._lock:
            return SiteOracleSnapshot(
                generation=self._generation,
                scenario=self._scenario,
                running=self._running,
                login_requests=self._login_requests,
                auth_successes=self._auth_successes,
                auth_failures=self._auth_failures,
                commits=self._commits,
                active_requests=self._active_requests,
                response_blocked=self._blocked_waiters > 0,
                authenticated_sessions=len(self._sessions),
            )

    def wait_for(self, event: SiteEvent, timeout_ms: int) -> bool:
        """Wait for one fixed event in the generation observed at the call boundary."""
        if type(event) is not SiteEvent:
            raise LocalSiteError("SITE_CONTROL_INVALID")
        budget_ms = _wait_milliseconds(timeout_ms)
        deadline = time.monotonic() + budget_ms / 1000.0
        with self._condition:
            if event in self._events:
                return True
            generation = self._generation
            while self._running and self._generation == generation:
                if event in self._events:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return event in self._events and self._generation == generation

    def generation_matches(self, generation: object) -> bool:
        """Whether a canonical submitted generation is the live one."""
        with self._lock:
            return (
                self._running
                and type(generation) is int
                and generation == self._generation
            )

    def record_login_received(self) -> bool:
        """Count one structurally valid login and publish ``LOGIN_RECEIVED``."""
        with self._condition:
            if not self._running:
                return False
            self._login_requests += 1
            self._events.add(SiteEvent.LOGIN_RECEIVED)
            self._condition.notify_all()
            return True

    def record_auth_failure(self) -> None:
        """Count one valid but incorrect synthetic credential pair."""
        with self._condition:
            if self._running:
                self._auth_failures += 1
                self._condition.notify_all()

    def bind_request_deadline(self, deadline: float) -> None:
        """Bind the live request's one absolute deadline to the current request thread.

        The server does this once per connection at transport setup. The critical commit then
        rechecks the same immutable deadline even though its call carries no argument, so a
        delay injected at the commit entry cannot make a stale pre-check commit anymore.
        """
        self._request_deadline.value = deadline

    def commit_session(self, deadline: float | None = None) -> str | None:
        """Commit exactly one successful login and register its current-generation session.

        The one absolute request deadline is rechecked *inside* this coordination boundary, at
        the critical point where the counters and the session would change. A commit that would
        start after the budget expired therefore returns ``None`` and creates no normal commit,
        no session and no ``LOGIN_COMMITTED`` fact. ``deadline`` may be supplied explicitly; a
        live request instead binds it to its own thread at transport setup. ``None`` means no
        request budget applies (a direct host call outside a request).
        """
        if deadline is None:
            deadline = getattr(self._request_deadline, "value", None)
        with self._condition:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            if not self._running:
                return None
            token = secrets.token_hex(16)
            self._sessions.add(token)
            self._auth_successes += 1
            self._commits += 1
            self._events.add(SiteEvent.LOGIN_COMMITTED)
            self._condition.notify_all()
            return token

    def session_is_valid(self, token: object) -> bool:
        """Validate an authentication fact on the server, never from a client claim."""
        with self._lock:
            return (
                self._running
                and isinstance(token, str)
                and token in self._sessions
            )

    def overlay_is_active(self) -> bool:
        """Whether the current page generation currently paints the pointer overlay."""
        with self._lock:
            return self._overlay_active

    # ------------------------------------------------------------------ gates and controls

    def gate_wait(self, deadline: float) -> GateOutcome:
        """Wait at this generation's scenario gate until release, timeout or stop.

        The caller passes one absolute monotonic request deadline; this method never creates a
        fresh full budget. A stop wake is distinct from a release: only a release may let the
        caller commit.
        """
        with self._condition:
            if self._scenario not in _GATED_SCENARIOS or self._gate_open:
                return GateOutcome.RELEASED
            if not self._running:
                return GateOutcome.STOPPED
            self._blocked_waiters += 1
            self._events.add(SiteEvent.RESPONSE_BLOCKED)
            self._condition.notify_all()
            try:
                while self._running and not self._gate_open:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._gate_open:
                    return GateOutcome.RELEASED
                if not self._running:
                    return GateOutcome.STOPPED
                return GateOutcome.TIMED_OUT
            finally:
                self._blocked_waiters -= 1
                self._condition.notify_all()

    def release(self) -> None:
        """Release this generation's gate or overlay; idempotent while running."""
        with self._condition:
            if not self._running:
                raise LocalSiteError("SITE_NOT_RUNNING")
            if self._scenario in _GATED_SCENARIOS:
                self._gate_open = True
            elif self._scenario is Scenario.BLOCKED_SUBMIT:
                self._overlay_active = False
            self._condition.notify_all()

    def reset(self, scenario: Scenario) -> int:
        """Advance to a clean generation, refusing atomically while any request is active."""
        if not _is_scenario(scenario):
            raise LocalSiteError("SITE_CONTROL_INVALID")
        with self._condition:
            if not self._running:
                raise LocalSiteError("SITE_NOT_RUNNING")
            if self._active_requests != 0:
                raise LocalSiteError("SITE_BUSY")
            self._generation += 1
            self._scenario = scenario
            self._login_requests = 0
            self._auth_successes = 0
            self._auth_failures = 0
            self._commits = 0
            self._blocked_waiters = 0
            self._events.clear()
            self._sessions.clear()
            self._gate_open = scenario not in _GATED_SCENARIOS
            self._overlay_active = scenario is Scenario.BLOCKED_SUBMIT
            self._condition.notify_all()
            return self._generation
