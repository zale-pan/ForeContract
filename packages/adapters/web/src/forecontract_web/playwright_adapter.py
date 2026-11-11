"""The Playwright Web adapter: strict configuration, lifecycle and execution.

``PlaywrightWebAdapter`` implements the existing Core :class:`~forecontract.protocols.adapter.
Adapter` protocol exactly - a fixed ``name``, synchronous no-I/O ``capabilities()`` and
``validate_action()``, and asynchronous ``start()``/``stop()``/``observe()``/``execute()`` - with
no deadline, context, retry or cancellation parameter and no Core, Runtime or Planner change.

The constructor accepts only a :class:`~forecontract_web.browser_config.PlaywrightWebConfig`,
performs no I/O, strictly revalidates the supplied configuration (including a supplied model
instance) and stores a detached copy. Importing this module never imports Playwright: the
optional dependency is loaded lazily by ``start()``, so a schema-only install can import, construct
and validate the adapter, and ``start()`` reports the bounded
``WEB_BROWSER_DEPENDENCY_MISSING`` code instead of a raw ``ImportError``.

One instance owns one Playwright driver, browser, context and active page for one Run. The
lifecycle state table is enforced before any backend I/O:

======================  =====================================================================
state / call            result
======================  =====================================================================
NEW / start             STARTING, acquire, then RUNNING only after every resource is ready
RUNNING / start         idempotent; no second resource set
STARTING / start        WEB_ADAPTER_BUSY before a second backend operation
STARTING / observe      WEB_ADAPTER_BUSY before a second backend operation
STARTING / execute      WEB_ADAPTER_BUSY before a second backend operation
non-RUNNING / observe   WEB_ADAPTER_NOT_RUNNING before I/O
non-RUNNING / execute   WEB_ADAPTER_NOT_RUNNING before I/O
NEW / stop              idempotent STOPPED, no resource creation
STOPPED / stop          idempotent STOPPED
RUNNING / stop          STOPPING, bounded release, then STOPPED if complete
STARTING / stop         cancel the owned acquisition, release partial resources, no fresh launch
FAILED / stop           a new host-authorized bounded cleanup attempt over the remaining handles
STOPPING / stop         join the same cleanup attempt; its deadline is never reset
STOPPED / start         WEB_ADAPTER_STOPPED; a new Run needs a new instance
cleanup failure         FAILED with a bounded cleanup error exposed through ``cleanup_error``
======================  =====================================================================

A concurrent ``observe``/``execute`` on the same instance is rejected with
``WEB_ADAPTER_BUSY`` before it reaches the browser. A host cancellation or a start failure never
leaves a RUNNING instance: the adapter releases the partial resources inside a bounded cleanup
attempt, exposes a cleanup failure through the read-only ``cleanup_error`` and re-raises the
original cancellation or start failure rather than masking it. A host cancellation keeps its
original type and identity even when a concurrent ``stop`` has claimed the instance, and ``stop``
cancels and finishes the adapter-owned in-flight observation/execution await inside the one
bounded cleanup attempt instead of returning while that work can still succeed later.
``asyncio.CancelledError``, ``KeyboardInterrupt`` and ``SystemExit`` always propagate and are never
forged into an ``ActualOutcome``.

Every whole ``start``/``observe``/``execute`` call computes one absolute monotonic deadline and
runs its backend work as one owned task that is awaited only until that deadline. A backend call
that delays its own cancellation therefore cannot extend the public cap, and work that finishes
late is abandoned, tracked and unable to establish a visible success or a RUNNING instance.
Cleanup is one absolute allowance for the whole attempt; a ``stop()`` that joins an attempt in
progress never resets it. A close that is only confirmed after the attempt has ended keeps its
handle and a visible bounded failure, so a later host-authorized ``stop`` can retry it instead of
a lost resource being reported as a clean ``STOPPED``.

The optional per-target observation allowance (``observation_target_timeout_ms``) is one absolute
deadline per target group: the native reads and the necessary cancellation settling share it, it
is never reset and no second allowance is granted. A settled cancellation is classified rather
than consumed - the adapter's own ordinary cancellation is the one bounded ``TARGET_QUERY_FAILED``
record; an observed ``SessionLost``, an internal defect or a process-control result propagates, and
the session is revalidated before any successful snapshot.

The optional startup allowance (``startup_timeout_ms``) bounds only the browser resource
acquisition of ``start``: one absolute deadline for that one attempt, with no phase reset, retry,
warm-up or reused session. ``None`` keeps the historical behaviour, where ``start`` is bounded by
``operation_timeout_ms``. ``observe`` and ``execute`` keep using ``operation_timeout_ms``, the
per-target allowance keeps the whole-operation bound, and cleanup keeps its own independent
``cleanup_timeout_ms``; the startup value never widens any of them. A startup timeout still runs
the existing bounded failed-start cleanup, so the host-observed whole-call elapsed time may
include that cleanup and is not evidence of the native browser-acquisition duration alone.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Final

from forecontract.models.actions import ActionSpec, AdapterCapabilities
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome

from forecontract_web import _browser_actions, _browser_observation
from forecontract_web._browser_session import (
    BrowserSession,
    BudgetExpired,
    CleanupResult,
    PlaywrightBackend,
    SessionLost,
)
from forecontract_web.browser_config import PlaywrightWebConfig, revalidated_config
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import (
    WEB_ACTION_TYPES,
    WEB_OBSERVATION_TYPES,
    RoleTarget,
    TestIdTarget,
)
from forecontract_web.rule_capabilities import build_web_rule_capabilities
from forecontract_web.rule_models import WEB_RULE_TYPES
from forecontract_web.validation import parse_web_action, validate_web_action

__all__ = ["PlaywrightWebAdapter"]

_SCHEMA_VERSION: Final[str] = "0.1"
_ADAPTER_NAME: Final[str] = "playwright_web"
_ENVIRONMENT_TYPE: Final[str] = "web"

#: The largest internal part of one target allowance that is reserved for the bounded
#: cancellation wind-down. The reserve is carved *out of* the approved target allowance; it
#: never adds a second allowance and never extends the whole observation.
_SETTLE_RESERVE_SECONDS: Final[float] = 0.050

#: The six lifecycle states. One instance is never restartable after a terminal stop.
_NEW: Final[str] = "new"
_STARTING: Final[str] = "starting"
_RUNNING: Final[str] = "running"
_STOPPING: Final[str] = "stopping"
_STOPPED: Final[str] = "stopped"
_FAILED: Final[str] = "failed"


class _ProcessControl:
    """Transport container for a host process-control exception raised inside an owned task.

    ``asyncio`` re-raises ``KeyboardInterrupt`` and ``SystemExit`` from a task directly into the
    event loop, where the caller of the public method could no longer observe them. An owned
    operation therefore returns the same exception object to its awaiting task, which re-raises
    it unchanged: the type, identity and propagation semantics stay the caller's. This is
    transport, not conversion - the exception is never turned into a public diagnostic.
    """

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


async def _guarded(coroutine):
    """Run one owned operation so process-control exceptions stay with the awaiting caller."""
    try:
        return await coroutine
    except (KeyboardInterrupt, SystemExit) as exc:
        return _ProcessControl(exc)


def _create_session(config: PlaywrightWebConfig) -> BrowserSession:
    """Build the owned session for a validated configuration.

    This function is pure - it imports no optional dependency and performs no I/O - and exists as
    a single named seam so a unit double can exercise the real lifecycle code without a browser.
    """
    return BrowserSession(PlaywrightBackend(config._canonical_origins))


class PlaywrightWebAdapter:
    """A headless Chromium adapter for the six Web actions and one Web snapshot."""

    def __init__(self, config: PlaywrightWebConfig) -> None:
        self._config = revalidated_config(config)
        self._state = _NEW
        self._session: BrowserSession | None = None
        self._start_task: asyncio.Task | None = None
        self._active_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_deadline: float | None = None
        self._cleanup_result: CleanupResult | None = None
        self._cleanup_error: WebBrowserError | None = None
        self._late_tasks: set[asyncio.Task] = set()
        self._stop_requested = False
        self._active = False

    # ------------------------------------------------------------------ declaration

    @property
    def name(self) -> str:
        """The fixed adapter name of this round."""
        return _ADAPTER_NAME

    @property
    def cleanup_error(self) -> WebBrowserError | None:
        """The most recent bounded cleanup failure, or ``None`` when cleanup was complete.

        The property is read-only. It keeps a failure that happened while a primary cancellation
        or a start failure was being preserved visible to the host after that primary exception
        has propagated.
        """
        return self._cleanup_error

    def capabilities(self) -> AdapterCapabilities:
        """Return one fresh declaration of the six actions, the snapshot and the five rules."""
        return build_web_rule_capabilities(
            adapter_name=_ADAPTER_NAME,
            supported_actions=list(WEB_ACTION_TYPES),
            supported_observation_types=list(WEB_OBSERVATION_TYPES),
            supported_condition_types=list(WEB_RULE_TYPES),
            supported_outcome_types=list(WEB_RULE_TYPES),
        )

    def validate_action(self, action: ActionSpec) -> None:
        """Reject an unsupported or malformed action before any browser interaction.

        The decision order and exceptions are exactly those of the pure Web contract: the Core
        envelope first, the unsupported action type second and the selected payload last. The call
        performs no I/O, changes no state and never modifies the supplied object.
        """
        validate_web_action(action)

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Acquire the owned browser resources, or report the bounded start failure."""
        state = self._state
        if state == _RUNNING:
            return
        if state == _STARTING or state == _STOPPING:
            raise WebBrowserError("WEB_ADAPTER_BUSY", "start")
        if state == _STOPPED or state == _FAILED:
            raise WebBrowserError("WEB_ADAPTER_STOPPED", "start")

        session = _create_session(self._config)
        self._session = session
        self._state = _STARTING
        # One absolute deadline for the whole browser acquisition. ``effective_startup_timeout_ms``
        # is the declared startup allowance, or the historical operation allowance when none was
        # declared: this is the only call that applies the startup budget, and it is never reset,
        # extended, retried or followed by a warm-up.
        deadline = time.monotonic() + self._config.effective_startup_timeout_ms / 1000.0
        task = asyncio.ensure_future(_guarded(session.start()))
        self._start_task = task

        primary: BaseException | None = None
        internal_cancel = False
        host_cancelled = False
        try:
            completed = await self._await_until(task, deadline)
        except asyncio.CancelledError as exc:
            # The host cancelled the start caller itself; ``_await_until`` already abandoned the
            # owned acquisition and the partial resources are released below. A concurrent
            # ``stop`` must not turn this original cancellation into a start failure: the host
            # cancellation keeps its type and identity.
            primary = exc
            host_cancelled = True
        else:
            if completed:
                try:
                    outcome = task.result()
                except asyncio.CancelledError as exc:
                    # The owned acquisition task itself was cancelled by a concurrent stop; the
                    # start caller receives the fixed bounded start failure.
                    primary = exc
                    internal_cancel = True
                except Exception as exc:  # noqa: BLE001 - released below, then re-raised
                    primary = exc
                else:
                    if isinstance(outcome, _ProcessControl):
                        primary = outcome.error
                    elif self._state == _STARTING:
                        self._start_task = None
                        self._state = _RUNNING
                        return
                    else:
                        # A concurrent stop cancelled the acquisition; its cleanup owns the
                        # handles.
                        internal_cancel = True
            else:
                # One operation budget for the whole start: a late or cancellation-delayed
                # acquisition can never establish a RUNNING instance afterwards.
                primary = WebBrowserError("WEB_BROWSER_START_FAILED", "start")

        self._start_task = None
        if self._state == _STARTING:
            self._state = _STOPPING
        self._ensure_cleanup()
        try:
            await self._join_cleanup()
        except asyncio.CancelledError:
            # A second cancellation: the bounded cleanup continues in its own task, and the
            # primary cancellation is still what propagates from this call.
            pass
        self._settle_cleanup()
        self._absorb_late_failure()
        if host_cancelled and primary is not None:
            raise primary
        if internal_cancel:
            raise WebBrowserError("WEB_BROWSER_START_FAILED", "start")
        if primary is None:
            raise WebBrowserError("WEB_BROWSER_START_FAILED", "start")
        raise primary

    async def stop(self) -> None:
        """Release every owned resource inside one bounded cleanup attempt."""
        state = self._state
        if state == _NEW or state == _STOPPED:
            self._state = _STOPPED
            return
        if state == _STOPPING:
            await self._join_cleanup()
            self._settle_cleanup()
            self._absorb_late_failure()
            self._raise_cleanup_error()
            return
        if state == _FAILED:
            # One new host-authorized cleanup attempt over the remaining owned handles.
            self._cleanup_task = None
            self._cleanup_result = None
            self._cleanup_deadline = None

        self._state = _STOPPING
        self._stop_requested = True
        self._cancel_owned_tasks()
        self._ensure_cleanup()
        await self._join_active_operation()
        await self._join_acquisition()
        await self._join_cleanup()
        self._settle_cleanup()
        self._absorb_late_failure()
        self._raise_cleanup_error()

    # ------------------------------------------------------------------ interaction

    async def observe(self) -> Observation:
        """Read one explicit-target Web snapshot from the owned page."""
        self._require_running("observe")
        self._active = True
        task: asyncio.Task | None = None
        operation = asyncio.current_task()
        try:
            session = self._session
            if session is None:
                raise WebBrowserError("WEB_BROWSER_UNAVAILABLE", "observe")
            deadline = time.monotonic() + self._config.operation_timeout_ms / 1000.0
            target_timeout_ms = self._config.observation_target_timeout_ms
            if target_timeout_ms is None:
                # The historical path: one target list bounded only by the whole operation.
                task = asyncio.ensure_future(
                    _guarded(
                        _browser_observation.collect(
                            session,
                            self._config.observation_targets,
                            deadline,
                        )
                    )
                )
            else:
                # The optional path: each target group gets its own absolute slice, while the
                # whole observation keeps the same operation deadline as the historical path.
                task = asyncio.ensure_future(
                    _guarded(
                        self._observe_targets_with_budget(
                            session, deadline, target_timeout_ms)
                    )
                )
            # The owned active await is the public observe call itself: a concurrent ``stop``
            # cancels this task, which abandons the inner backend await inside the one cleanup
            # allowance and makes the operation end as a cancellation - never as a late success.
            self._active_task = operation
            state = None
            code = None
            completed = await self._await_until(task, deadline)
            if not completed:
                # The one whole-operation deadline expired, even if a cancellation-delayed read
                # is still winding down: the public call is done.
                code = "WEB_OBSERVE_TIMEOUT"
            else:
                try:
                    outcome = task.result()
                except BudgetExpired:
                    code = "WEB_OBSERVE_TIMEOUT"
                except SessionLost:
                    code = "WEB_BROWSER_UNAVAILABLE"
                else:
                    if isinstance(outcome, _ProcessControl):
                        raise outcome.error
                    state = outcome
            if code is not None:
                raise WebBrowserError(code, "observe")
            return Observation(
                schema_version=_SCHEMA_VERSION,
                observation_id=f"web-observation-{uuid.uuid4().hex}",
                environment_type=_ENVIRONMENT_TYPE,
                state=state.model_dump(mode="json"),  # type: ignore[union-attr]
                timestamp=datetime.now(timezone.utc),
            )
        finally:
            if operation is not None and self._active_task is operation:
                self._active_task = None
            self._active = False

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        """Validate and execute one Web action exactly once."""
        self._require_running("execute")
        self._active = True
        started = time.monotonic()
        task: asyncio.Task | None = None
        operation = asyncio.current_task()
        try:
            deadline = started + self._config.operation_timeout_ms / 1000.0
            validated = parse_web_action(action)
            session = self._session
            if session is None:
                raise WebBrowserError("WEB_BROWSER_UNAVAILABLE", "execute")
            task = asyncio.ensure_future(
                _guarded(_browser_actions.perform(session, validated, deadline, started))
            )
            # Same ownership as observe: stop cancels this public await, so a backend call that
            # suppresses its own cancellation can never turn into a visible SUCCEEDED outcome.
            self._active_task = operation
            completed = await self._await_until(task, deadline)
            if not completed:
                # Nothing after the deadline can be confirmed, so the conservative execution
                # fact is UNKNOWN; the abandoned call is never replayed and never re-joined.
                return _browser_actions.budget_timeout_outcome(started)
            outcome = task.result()
            if isinstance(outcome, _ProcessControl):
                raise outcome.error
            return outcome
        finally:
            if operation is not None and self._active_task is operation:
                self._active_task = None
            self._active = False

    # ------------------------------------------------------------------ internals

    async def _observe_targets_with_budget(self, session: BrowserSession, deadline: float,
                                           target_timeout_ms: float) -> object:
        """Collect one snapshot with one absolute allowance per target group.

        Every target is a separately owned subtask awaited only until its own one absolute
        deadline. That single allowance covers the whole group (count, handle and attributes)
        *and* the necessary cancellation settling: the read portion is interrupted early enough
        to reserve a bounded wind-down inside the very same deadline, and the deadline is never
        reset, extended or followed by a second allowance. A group that settles inside it becomes
        the existing ``unavailable``/``TARGET_QUERY_FAILED`` record while the independently
        completed targets keep their real content; a group whose cancellation does not converge
        inside that deadline is abandoned to the adapter's existing late-task ownership and the
        whole observation ends as the existing facility failure.

        Ownership is explicit: the in-flight subtask is either awaited to completion or handed to
        the adapter's existing late-task bookkeeping. A settled subtask is classified by what it
        actually is: the adapter's own ordinary cancellation, an observed ``SessionLost``, an
        internal defect or a process-control result. Only the first is ordinary evidence; the
        others propagate and are never published as an unavailable target. A session that is no
        longer usable can therefore never back a successful snapshot.
        """
        page, types = _browser_observation.snapshot_page(session)
        url = _browser_observation.snapshot_url(session, page)
        target_seconds = target_timeout_ms / 1000.0
        records: list = []
        current: asyncio.Task | None = None
        try:
            for target in self._config.observation_targets:
                now = time.monotonic()
                if now >= deadline:
                    raise BudgetExpired()
                group_deadline = min(deadline, now + target_seconds)
                # One absolute allowance per group, internally arranged so that the bounded
                # cancellation wind-down stays inside it. This is a split of the approved
                # allowance, never an additional one.
                reserve = min((group_deadline - now) / 4.0, _SETTLE_RESERVE_SECONDS)
                work_deadline = group_deadline - reserve
                current = asyncio.ensure_future(
                    _guarded(_browser_observation.query_target(
                        session, page, target, group_deadline, types)))
                if await self._await_owned(current, work_deadline):
                    finished_target = current
                    current = None
                    outcome = finished_target.result()
                    if isinstance(outcome, _ProcessControl):
                        raise outcome.error
                    records.append(outcome)
                    continue
                # The group's read portion is spent: cancel and confirm the convergence inside the
                # same absolute deadline. Nothing is reset and no second allowance is granted; a
                # late completion is never read back into the snapshot.
                current.cancel()
                if not await self._await_owned(current, group_deadline):
                    self._abandon(current)
                    current = None
                    raise BudgetExpired()
                settled = current
                current = None
                records.append(self._settled_cancellation(settled, session, target))
            if not session.is_connected():
                # Revalidate the whole session before any successful snapshot: a facility that
                # disappeared during this call cannot back a collected record.
                raise SessionLost()
        finally:
            if current is not None:
                self._abandon(current)
        return _browser_observation.build_state(url, records)

    def _settled_cancellation(self, task: asyncio.Task, session: BrowserSession,
                              target: TestIdTarget | RoleTarget) -> object:
        """Classify one target subtask that settled after the adapter cancelled it.

        Only the adapter's own ordinary cancellation request is evidence of a bounded local
        failure. An observed ``SessionLost``, an internal defect or a process-control result is
        propagated instead of being consumed, and a target that happened to complete late is
        never read back into the snapshot.
        """
        if task.cancelled():
            if not session.is_connected():
                raise SessionLost()
            return _browser_observation.unavailable(target)
        try:
            outcome = task.result()
        except SessionLost:
            raise
        except BudgetExpired:
            outcome = None
        if isinstance(outcome, _ProcessControl):
            raise outcome.error
        if not session.is_connected():
            raise SessionLost()
        return _browser_observation.unavailable(target)

    async def _await_owned(self, task: asyncio.Task, deadline: float) -> bool:
        """Await one owned subtask only until an absolute deadline, without abandoning it.

        The caller decides whether a still-running subtask is a bounded local failure or a
        cancellation that failed to converge; this helper never creates a task and never leaves
        one untracked, because the caller either reads its result or hands it to ``_abandon``.
        """
        left = deadline - time.monotonic()
        if left <= 0:
            return task.done()
        done, _ = await asyncio.wait({task}, timeout=left)
        return task in done

    async def _await_until(self, task: asyncio.Task, deadline: float) -> bool:
        """Wait for one owned task only until the one whole-operation deadline.

        :func:`asyncio.wait` returns at the deadline even when the awaited work resists its own
        cancellation, so a cancellation-delayed backend call can never extend the public cap.
        An expired wait abandons (cancels and tracks) the task instead of joining it; nothing
        that finishes late can then establish a visible success or a RUNNING instance.
        """
        left = deadline - time.monotonic()
        if left <= 0:
            self._abandon(task)
            return False
        try:
            done, _ = await asyncio.wait({task}, timeout=left)
        except asyncio.CancelledError:
            self._abandon(task)
            raise
        if task in done:
            return True
        self._abandon(task)
        return False

    def _abandon(self, task: asyncio.Task) -> None:
        """Cancel and track one owned task that outlived its one deadline."""
        task.cancel()
        self._late_tasks.add(task)

        def settled(finished: asyncio.Task) -> None:
            self._late_tasks.discard(finished)
            if not finished.cancelled():
                try:
                    finished.exception()
                except BaseException:  # noqa: BLE001, S110 - a late failure is not a public outcome
                    pass
            # A late handle whose close already failed is reported honestly instead of being
            # consumed here: the resource and the bounded cleanup error stay visible so that a
            # later host-authorized ``stop`` can retry the remaining handles.
            self._note_late_session_failure()

        task.add_done_callback(settled)

    def _cancel_owned_tasks(self) -> None:
        """Cancel every owned in-flight backend task before the bounded release."""
        start_task = self._start_task
        if start_task is not None and not start_task.done():
            start_task.cancel()
        active_task = self._active_task
        if active_task is not None and not active_task.done():
            active_task.cancel()
        for task in list(self._late_tasks):
            if not task.done():
                task.cancel()

    async def _join_active_operation(self) -> None:
        """Cancel and finish one owned in-flight observe/execute await inside the cleanup bound.

        ``stop`` cancels the adapter-owned active task before this join. The task is awaited only
        until the one absolute cleanup deadline, so a backend call that delays its own
        cancellation cannot extend the attempt and cannot establish a visible success after
        ``stop``. A task that outlives the deadline is abandoned and tracked, never silently
        joined outside the budget.
        """
        task = self._active_task
        if task is None:
            return
        if task.done():
            if not task.cancelled():
                try:
                    task.exception()
                except BaseException:  # noqa: BLE001, S110 - the operation caller owns its result
                    pass
            return
        deadline = self._cleanup_deadline
        if deadline is None:
            return
        left = deadline - time.monotonic()
        if left > 0:
            try:
                await asyncio.wait({task}, timeout=left)
            except asyncio.CancelledError:
                self._abandon(task)
                raise
        if not task.done():
            self._abandon(task)
        elif not task.cancelled():
            try:
                task.exception()
            except BaseException:  # noqa: BLE001, S110 - the operation caller owns its result
                pass

    async def _join_acquisition(self) -> None:
        """Join a cancelled acquisition only inside the one cleanup bound.

        The start caller receives its own primary failure; this join only prevents a concurrent
        close from racing the acquisition. A cancellation-delayed acquisition is abandoned at
        the cleanup deadline instead of resetting or extending the attempt.
        """
        task = self._start_task
        if task is None:
            return
        self._start_task = None
        if not task.done():
            task.cancel()
        deadline = self._cleanup_deadline
        if deadline is None:
            return
        left = deadline - time.monotonic()
        if left > 0:
            try:
                await asyncio.wait({task}, timeout=left)
            except asyncio.CancelledError:
                self._abandon(task)
                raise
        if not task.done():
            self._abandon(task)
        elif not task.cancelled():
            try:
                task.exception()
            except BaseException:  # noqa: BLE001, S110 - the start caller owns the primary failure
                pass

    def _note_late_session_failure(self) -> None:
        """Surface a cleanup failure that appeared after the attempt had already settled."""
        if self._state in (_NEW, _STARTING, _RUNNING, _STOPPING):
            return
        self._absorb_late_failure()

    def _absorb_late_failure(self) -> None:
        """Report a late handle whose close failed instead of discarding it silently.

        A handle obtained by an abandoned acquisition can fail its close after the one cleanup
        attempt has already ended. That resource is kept on the session for a later attempt; the
        adapter must not keep claiming ``STOPPED`` with no error, so the visible bounded cleanup
        failure and the ``FAILED`` state are established here. A subsequent host-authorized
        ``stop`` then retries exactly that remaining handle.
        """
        if self._state in (_NEW, _STARTING, _RUNNING, _STOPPING):
            return
        session = self._session
        if session is None or not session.has_pending_cleanup():
            return
        failure = session.late_failure
        if failure is None:
            failure = WebBrowserError("WEB_BROWSER_CLEANUP_FAILED", "cleanup")
        if self._cleanup_error is None:
            self._cleanup_error = failure
        self._cleanup_result = CleanupResult(False, self._cleanup_error)
        self._state = _FAILED

    def _require_running(self, operation: str) -> None:
        """Enforce the state and single-operation table before any backend I/O."""
        state = self._state
        if state == _STARTING:
            raise WebBrowserError("WEB_ADAPTER_BUSY", operation)
        if state != _RUNNING:
            raise WebBrowserError("WEB_ADAPTER_NOT_RUNNING", operation)
        if self._active:
            raise WebBrowserError("WEB_ADAPTER_BUSY", operation)

    def _ensure_cleanup(self) -> None:
        """Create the one cleanup attempt for this state transition, with its own deadline."""
        if self._cleanup_task is not None:
            return
        if self._session is None:
            self._cleanup_result = CleanupResult(True, None)
            return
        deadline = time.monotonic() + self._config.cleanup_timeout_ms / 1000.0
        self._cleanup_deadline = deadline
        self._cleanup_task = asyncio.ensure_future(self._cleanup(deadline))

    async def _cleanup(self, deadline: float) -> CleanupResult:
        session = self._session
        if session is None:
            result = CleanupResult(True, None)
        else:
            result = await session.cleanup(deadline)
        self._cleanup_result = result
        self._cleanup_error = result.error
        return result

    async def _join_cleanup(self) -> None:
        """Join the current bounded cleanup attempt without cancelling or resetting it."""
        task = self._cleanup_task
        if task is None:
            return
        await asyncio.shield(task)

    def _settle_cleanup(self) -> None:
        result = self._cleanup_result
        if result is not None and result.complete:
            self._state = _STOPPED
        else:
            self._state = _FAILED

    def _raise_cleanup_error(self) -> None:
        error = self._cleanup_error
        if error is not None:
            raise error
