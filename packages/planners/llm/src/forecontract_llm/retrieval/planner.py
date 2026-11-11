"""The bounded retrieval planner: real evidence in, one Core Forecast out.

:class:`EvidenceLLMPlanner` implements the existing Core ``Planner`` protocol exactly:
``async plan(observation: Observation) -> Forecast``, plus ``async aclose() ->
PlannerCloseResult``. One instance is bound to one Run. A planning attempt opens a frozen
evidence scope, then permits at most ``max_reads`` authorized reads and at most
``min(max_model_calls, max_reads + 1)`` provider calls inside one monotonic deadline and the
fixed C05 byte, depth, node, count and token limits.

Every external call (provider, reader, audit sink) is started only after its intent was
persisted, is awaited under the remaining deadline, and is cancelled with one bounded cleanup
window when the host cancels, the deadline expires or the planner is closed. A call task that
does not settle is retained, the planner becomes ``poisoned`` and a later plan is refused; the
close result then reports the truth instead of a fabricated drain. An external cancellation is
always re-raised as ``CancelledError``; only this object's own close-triggered cancellation is
converted to ``invalid_state``.

F01/F04: evaluating an external descriptor, preparing an event and creating its awaitable are
synchronous steps that can outlast the absolute deadline they run under. The same absolute
deadline is therefore re-read *after* that preparation and *before* any task is scheduled or
awaited; an operation whose window already expired is never started, an unscheduled coroutine is
closed instead of leaked, and an unwritten terminal record is reported as unconverged. A
``RetrievalError`` raised by an external facility at those boundaries is normalised by facility
kind under this planner's own request identity; it is never re-raised with the facility's
classification, message or request identifier, while an external ``CancelledError`` keeps its
identity and the planner's own budget, binding and selection errors keep their own codes.

No API key is read, no network connection is opened, no SDK is imported and no production path exists in
this distribution. The planner calls no adapter, decides no terminal state and never turns a
malformed or unauthorized answer into an action.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
import uuid
from typing import Final

from forecontract.models.actions import Forecast
from forecontract.models.observations import Observation

from forecontract_llm.models import ModelResponse, PlannerCloseResult, PlannerContext

from . import codec
from .audit import build_event
from .errors import RetrievalError
from .models import (
    PLAN_REQUEST_MAX_BYTES,
    PLAN_RESPONSE_MAX_BYTES,
    PLAN_RESULT_MAX_BYTES,
    RESULT_MAX_BYTES,
    SCHEMA_VERSION,
    SCOPE_MAX_BYTES,
    EvidenceReadRequest,
    EvidenceReadResult,
    EvidenceScope,
    RetrievalConfig,
    RetrievalModelRequest,
    copy_record,
    effective_base_config,
    measure_record,
)

__all__ = ["EvidenceLLMPlanner"]

_READY: Final[str] = "ready"
_BUSY: Final[str] = "busy"
_POISONED: Final[str] = "poisoned"
_CLOSED: Final[str] = "closed"

_EXTERNAL_FAILURES: Final[dict[str, tuple[str, str]]] = {
    "provider": ("provider_failed", "provider"),
    "reader": ("reader_failed", "reader"),
    "audit": ("audit_failed", "audit"),
}


def _fail(code: str, stage: str, request_id: str | None = None) -> RetrievalError:
    return RetrievalError(code, stage, request_id=request_id)


class EvidenceLLMPlanner:
    """One retrieval planner bound to exactly one Run's context and its injected interfaces."""

    def __init__(
        self,
        *,
        provider: object,
        context: PlannerContext,
        config: RetrievalConfig,
        reader: object,
        audit: object,
    ) -> None:
        self._provider = self._require_async(provider, "complete", "provider")
        self._reader = self._require_async(reader, "read", "reader")
        self._require_callable(reader, "open_scope", "reader")
        self._require_callable(reader, "close_scope", "reader")
        self._audit = self._require_async(audit, "append", "audit")
        self._context = copy_record(context, PlannerContext, "invalid_input", "input")
        self._config = copy_record(config, RetrievalConfig, "invalid_input", "input")
        self._base = effective_base_config(self._config.base)
        self._calls_used = 0
        self._sequence = 0
        self._current_plan_id = str(uuid.uuid4())
        self._busy = False
        self._closed = False
        self._poisoned = False
        self._audit_unconverged = False
        self._closed_event = False
        self._task: asyncio.Task | None = None
        self._plan_task: asyncio.Task | None = None
        self._close_cancelled_plan: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()
        self._orphans: set[asyncio.Task] = set()
        self._cleanup_deadline: float | None = None

    # ------------------------------------------------------------------ construction guards

    @staticmethod
    def _require_async(owner: object, name: str, stage: str) -> object:
        try:
            method = getattr(owner, name)
        except Exception:  # noqa: BLE001 - a hostile descriptor must not leak its text
            raise _fail("invalid_input", stage) from None
        if not callable(method) or not inspect.iscoroutinefunction(method):
            raise _fail("invalid_input", stage)
        return owner

    @staticmethod
    def _require_callable(owner: object, name: str, stage: str) -> None:
        try:
            method = getattr(owner, name)
        except Exception:  # noqa: BLE001 - a hostile descriptor must not leak its text
            raise _fail("invalid_input", stage) from None
        if not callable(method):
            raise _fail("invalid_input", stage)

    # ------------------------------------------------------------------ public state

    @property
    def calls_used(self) -> int:
        """How many provider attempts this instance has really started."""
        return self._calls_used

    @property
    def state(self) -> str:
        """One of ``ready``, ``busy``, ``poisoned`` or ``closed``."""
        if self._closed:
            return _CLOSED
        if self._poisoned:
            return _POISONED
        if self._busy:
            return _BUSY
        return _READY

    @property
    def audit_unconverged(self) -> bool:
        """Whether a terminal audit record could not be persisted inside its bound."""
        return self._audit_unconverged

    # ------------------------------------------------------------------ plan

    async def plan(self, observation: Observation) -> Forecast:
        """Return one Core Forecast for this observation, or a bounded facility failure."""
        if self._closed:
            raise _fail("invalid_state", "lifecycle")
        if self._poisoned:
            raise _fail("invalid_state", "lifecycle")
        if self._busy:
            raise _fail("invalid_state", "lifecycle")
        self._busy = True
        self._plan_task = asyncio.current_task()
        self._cleanup_deadline = None
        plan_id = str(uuid.uuid4())
        self._current_plan_id = plan_id
        scope: EvidenceScope | None = None
        try:
            deadline = time.monotonic() + (self._base.call_timeout_ms / 1000.0)
            current = copy_record(observation, Observation, "invalid_input", "input")
            if current.environment_type != self._context.capabilities.environment_type:
                raise _fail("binding_mismatch", "context")
            scope = self._open_scope(current, deadline)
            await self._emit(
                "plan_started",
                deadline=deadline,
                scope_id=scope.scope_id,
                payload={
                    "observation_id": current.observation_id,
                    "max_reads": self._config.max_reads,
                    "max_model_calls": self._config.max_model_calls,
                    "instance_max_calls": self._base.max_calls,
                },
            )
            exchanges: list[EvidenceReadResult] = []
            seen: set[tuple[str, str, int | None, int | None]] = set()
            reads_used = 0
            request_total = 0
            response_total = 0
            result_total = 0
            allowed_calls = min(self._config.max_model_calls, self._config.max_reads + 1)
            for turn in range(1, allowed_calls + 1):
                self._check_deadline(deadline)
                request_id = str(uuid.uuid4())
                request, request_size = codec.build_request(
                    context=self._context,
                    observation=current,
                    config=self._config,
                    request_id=request_id,
                    plan_id=plan_id,
                    turn=turn,
                    scope=scope,
                    exchanges=exchanges,
                )
                request_total += request_size
                if request_total > PLAN_REQUEST_MAX_BYTES:
                    raise _fail("budget_exceeded", "budget", request_id)
                await self._emit(
                    "request_prepared",
                    deadline=deadline,
                    request_id=request_id,
                    turn=turn,
                    scope_id=scope.scope_id,
                    payload=request.model_dump(mode="json"),
                )
                if self._calls_used >= self._base.max_calls:
                    raise _fail("budget_exceeded", "budget", request_id)
                self._calls_used += 1
                response = await self._call_provider(request, deadline)
                try:
                    checked = codec.check_response(
                        response=response, request=request, config=self._config)
                except RetrievalError:
                    # F05: a response that arrived but failed its shell/binding checks is
                    # still a real return. Persist it (bounded) before the plan ends.
                    await self._emit(
                        "provider_returned",
                        deadline=deadline,
                        request_id=request_id,
                        turn=turn,
                        scope_id=scope.scope_id,
                        payload=self._provider_return_record(response, request),
                    )
                    raise
                response_total += checked.content_bytes
                await self._emit(
                    "provider_returned",
                    deadline=deadline,
                    request_id=request_id,
                    turn=turn,
                    scope_id=scope.scope_id,
                    payload=self._provider_payload(checked.response, request),
                )
                if response_total > PLAN_RESPONSE_MAX_BYTES:
                    raise _fail("budget_exceeded", "budget", request_id)
                if checked.response.status == "refused":
                    raise _fail("provider_refused", "provider", request_id)
                self._check_deadline(deadline)
                content = checked.response.content
                if not isinstance(content, str):
                    raise _fail("invalid_response", "response", request_id)
                message = codec.decode_message(
                    content=content,
                    request=request,
                    config=self._config,
                )
                if isinstance(message, codec.ForecastMessage):
                    await self._emit(
                        "forecast_ready",
                        deadline=deadline,
                        request_id=request_id,
                        turn=turn,
                        scope_id=scope.scope_id,
                        payload={"forecast": message.forecast.model_dump(mode="json")},
                    )
                    self._check_deadline(deadline)
                    if self._closed:
                        raise _fail("invalid_state", "lifecycle")
                    return message.forecast
                view = self._authorize(scope, message.evidence_id, message.view_id)
                if message.text_range is not None and view.format == "json":
                    raise _fail("selection_invalid", "selection", request_id)
                key = (
                    message.evidence_id,
                    message.view_id,
                    None if message.text_range is None else message.text_range.start,
                    None if message.text_range is None else message.text_range.end,
                )
                if key in seen:
                    raise _fail("duplicate_read", "selection", request_id)
                if reads_used >= self._config.max_reads:
                    raise _fail("budget_exceeded", "budget", request_id)
                reads_used += 1  # noqa: SIM113 - only a legal unique read is charged
                seen.add(key)
                read_request = EvidenceReadRequest(
                    schema_version=SCHEMA_VERSION,
                    scope_id=scope.scope_id,
                    run_id=request.run_id,
                    plan_id=request.plan_id,
                    turn=request.turn,
                    request_id=request.request_id,
                    evidence_id=message.evidence_id,
                    view_id=message.view_id,
                    text_range=message.text_range,
                )
                await self._emit(
                    "read_requested",
                    deadline=deadline,
                    request_id=request_id,
                    turn=turn,
                    scope_id=scope.scope_id,
                    payload=read_request.model_dump(mode="json"),
                )
                result = await self._call_reader(read_request, deadline)
                self._check_deadline(deadline)
                validated, result_size = self._check_result(
                    result, read_request, scope, result_total)
                result_total += result_size
                # F03: only the detached, revalidated copy is measured, audited and sent on.
                await self._emit(
                    "read_returned",
                    deadline=deadline,
                    request_id=request_id,
                    turn=turn,
                    scope_id=scope.scope_id,
                    payload={
                        "result": validated.model_dump(mode="json"),
                        "source": {
                            "kind": "registered_snapshot",
                            "registration_time_snapshot": True,
                            "live_disk_state": False,
                            "raw_sha256": validated.raw_sha256,
                            "raw_bytes": validated.raw_bytes,
                        },
                    },
                )
                exchanges.append(validated)
            raise _fail("budget_exceeded", "budget")
        except RetrievalError as error:
            if not (error.code == "invalid_state" and error.stage == "lifecycle"):
                await self._terminal_failure(error)
            raise
        except asyncio.CancelledError:
            if self._close_owns_cancellation(asyncio.current_task()):
                raise _fail("invalid_state", "lifecycle") from None
            await self._terminal_cancelled()
            raise
        finally:
            if scope is not None:
                with contextlib.suppress(Exception):
                    self._reader.close_scope(scope.scope_id)
            if self._close_cancelled_plan is asyncio.current_task():
                self._close_cancelled_plan = None
            self._task = None
            self._plan_task = None
            self._busy = False

    def _open_scope(self, observation: Observation, deadline: float) -> EvidenceScope:
        self._check_deadline(deadline)
        try:
            scope = self._reader.open_scope(self._context.run_id, observation.observation_id)
        except Exception:  # noqa: BLE001 - F04: an externally forged failure is normalised
            raise _fail("reader_failed", "reader") from None
        self._check_deadline(deadline)
        if not isinstance(scope, EvidenceScope):
            raise _fail("reader_failed", "reader")
        current = copy_record(scope, EvidenceScope, "binding_mismatch", "binding")
        if current.run_id != self._context.run_id:
            raise _fail("binding_mismatch", "binding")
        if current.observation_id != observation.observation_id:
            raise _fail("binding_mismatch", "binding")
        # F02: the planner enforces its own directory budget before it trusts the reader's
        # check or calls the provider. Exactly the cap is legal; one byte more is refused.
        measure_record(
            current,
            max_bytes=SCOPE_MAX_BYTES,
            max_depth=self._base.max_json_depth,
            max_nodes=self._base.max_json_nodes,
            code="budget_exceeded",
            stage="scope",
        )
        return current

    @staticmethod
    def _authorize(scope: EvidenceScope, evidence_id: str, view_id: str) -> object:
        entry = next(
            (item for item in scope.entries if item.evidence_id == evidence_id), None)
        if entry is None:
            raise _fail("access_denied", "grant")
        view = next((item for item in entry.views if item.view_id == view_id), None)
        if view is None:
            raise _fail("access_denied", "grant")
        return view

    def _check_result(
        self,
        result: object,
        request: EvidenceReadRequest,
        scope: EvidenceScope,
        prior_total: int,
    ) -> tuple[EvidenceReadResult, int]:
        if not isinstance(result, EvidenceReadResult):
            raise _fail("reader_failed", "reader", request.request_id)
        current = copy_record(result, EvidenceReadResult, "invalid_response", "reader")
        if (
            current.scope_id != request.scope_id
            or current.run_id != request.run_id
            or current.plan_id != request.plan_id
            or current.turn != request.turn
            or current.request_id != request.request_id
            or current.evidence_id != request.evidence_id
            or current.view_id != request.view_id
        ):
            raise _fail("binding_mismatch", "binding", request.request_id)
        entry = next(
            (item for item in scope.entries if item.evidence_id == request.evidence_id), None)
        if entry is None:
            raise _fail("access_denied", "grant", request.request_id)
        view = next(
            (item for item in entry.views if item.view_id == request.view_id), None)
        if view is None:
            raise _fail("access_denied", "grant", request.request_id)
        if (
            current.source_observation_id != entry.source_observation_id
            or current.raw_sha256 != entry.raw_sha256
            or current.raw_bytes != entry.raw_bytes
            or list(current.selector) != list(view.selector)
            or current.format != view.format
        ):
            raise _fail("binding_mismatch", "binding", request.request_id)
        requested = request.text_range
        returned = current.text_range
        left = None if requested is None else (requested.start, requested.end)
        right = None if returned is None else (returned.start, returned.end)
        if left != right:
            raise _fail("binding_mismatch", "binding", request.request_id)
        size = measure_record(
            current,
            max_bytes=RESULT_MAX_BYTES,
            max_depth=self._base.max_json_depth,
            max_nodes=self._base.max_json_nodes,
            code="budget_exceeded",
            stage="budget",
        )
        if prior_total + size > PLAN_RESULT_MAX_BYTES:
            raise _fail("budget_exceeded", "budget", request.request_id)
        return current, size

    # ------------------------------------------------------------------ external calls

    async def _call_provider(
        self, request: RetrievalModelRequest, deadline: float) -> object:
        try:
            complete = self._provider.complete
            awaitable = complete(request.model_copy(deep=True))
        except Exception:  # noqa: BLE001 - F04: a forged external failure is normalised
            raise _fail("provider_failed", "provider", request.request_id) from None
        return await self._external(
            awaitable,
            deadline=deadline,
            kind="provider",
            request_id=request.request_id,
        )

    async def _call_reader(
        self, request: EvidenceReadRequest, deadline: float) -> object:
        try:
            read = self._reader.read
            awaitable = read(request.model_copy(deep=True))
        except Exception:  # noqa: BLE001 - F04: a forged external failure is normalised
            raise _fail("reader_failed", "reader", request.request_id) from None
        return await self._external(
            awaitable,
            deadline=deadline,
            kind="reader",
            request_id=request.request_id,
        )

    @staticmethod
    def _dispose(awaitable: object) -> None:
        """Close an awaitable that was created but must never be scheduled (F01)."""
        with contextlib.suppress(BaseException):
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()

    async def _external(
        self,
        awaitable: object,
        *,
        deadline: float,
        kind: str,
        request_id: str | None = None,
    ) -> object:
        """Await one external operation under the deadline, normalising every failure.

        F01: the awaitable was created by a synchronous external evaluation that may itself
        have consumed the whole deadline. The remaining time is re-read here, before a task
        is scheduled, so an already-expired operation is never started: its coroutine is
        closed and the caller sees the deadline failure.
        """
        failure_code, failure_stage = _EXTERNAL_FAILURES[kind]
        if deadline - time.monotonic() <= 0.0:
            self._dispose(awaitable)
            self._begin_cleanup()
            await self._drain_cleanup()
            raise _fail("timeout", "deadline", request_id)
        try:
            task = asyncio.ensure_future(awaitable)
        except Exception:  # noqa: BLE001 - a coroutine that cannot be scheduled is bounded
            self._dispose(awaitable)
            raise _fail(failure_code, failure_stage, request_id) from None
        self._task = task
        self._inflight.add(task)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._begin_cleanup()
                task.cancel()
                await self._drain_cleanup()
                raise _fail("timeout", "deadline", request_id)
            try:
                await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError:
                self._begin_cleanup()
                if not task.done():
                    task.cancel()
                    self._retain(task)
                raise
            if not task.done():
                self._begin_cleanup()
                task.cancel()
                await self._drain_cleanup()
                raise _fail("timeout", "deadline", request_id)
            try:
                return task.result()
            except asyncio.CancelledError:
                raise _fail(failure_code, failure_stage, request_id) from None
            except Exception:  # noqa: BLE001 - one bounded facility failure
                raise _fail(failure_code, failure_stage, request_id) from None
        finally:
            self._inflight.discard(task)

    # ------------------------------------------------------------------ audit

    def _next_event(
        self,
        kind: str,
        *,
        payload: object,
        request_id: str | None = None,
        turn: int | None = None,
        scope_id: str | None = None,
    ):
        self._sequence += 1
        return build_event(
            run_id=self._context.run_id,
            plan_id=self._current_plan_id,
            sequence=self._sequence,
            kind=kind,
            request_id=request_id,
            turn=turn,
            scope_id=scope_id,
            payload=payload,
        )


    async def _emit(
        self,
        kind: str,
        *,
        deadline: float,
        payload: object,
        request_id: str | None = None,
        turn: int | None = None,
        scope_id: str | None = None,
    ) -> None:
        try:
            append = self._audit.append
            event = self._next_event(
                kind, payload=payload, request_id=request_id, turn=turn, scope_id=scope_id)
            awaitable = append(event)
        except Exception:  # noqa: BLE001 - F04: a forged external failure is normalised
            raise _fail("audit_failed", "audit", request_id) from None
        await self._external(awaitable, deadline=deadline, kind="audit",
                             request_id=request_id)

    # ------------------------------------------------------------------ cleanup

    def _begin_cleanup(self) -> None:
        """Start the single cleanup window that covers every outstanding operation."""
        if self._cleanup_deadline is None:
            self._cleanup_deadline = (
                time.monotonic() + (self._base.cleanup_timeout_ms / 1000.0))

    def _remaining_cleanup(self) -> float:
        if self._cleanup_deadline is None:
            return 0.0
        return max(0.0, self._cleanup_deadline - time.monotonic())

    async def _drain_cleanup(self) -> None:
        """Wait the remaining cleanup window for every outstanding external operation."""
        tasks = {task for task in self._inflight | self._orphans if not task.done()}
        if not tasks:
            for task in list(self._orphans):
                if task.done():
                    self._consume(task)
                    self._orphans.discard(task)
            return
        remaining = self._remaining_cleanup()
        if remaining <= 0.0:
            for task in tasks:
                self._retain(task)
            self._poisoned = True
            return
        try:
            await asyncio.wait(tasks, timeout=remaining)
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
                    self._retain(task)
            self._poisoned = True
            raise
        for task in tasks:
            if task.done():
                self._consume(task)
                self._orphans.discard(task)
            else:
                self._retain(task)
                self._poisoned = True

    def _mark_poisoned_if_unconverged(self) -> None:
        """F01: an unsettled operation or terminal audit failure forbids a later plan."""
        if self._audit_unconverged or any(
                not task.done() for task in self._inflight | self._orphans):
            self._poisoned = True

    async def _terminal_failure(self, error: RetrievalError) -> None:
        """Persist one bounded failure record; never mask the original failure."""
        self._begin_cleanup()
        await self._drain_cleanup()
        if error.code == "audit_failed":
            self._audit_unconverged = True
            self._poisoned = True
            return
        await self._best_effort_event(
            "plan_failed",
            payload={"code": error.code, "stage": error.stage},
            request_id=error.request_id,
        )
        self._mark_poisoned_if_unconverged()

    async def _terminal_cancelled(self) -> None:
        """Persist one bounded cancellation record; never turn it into another error."""
        self._begin_cleanup()
        try:
            await self._drain_cleanup()
        except asyncio.CancelledError:
            self._audit_unconverged = True
            self._poisoned = True
            return
        await self._best_effort_event(
            "plan_cancelled", payload={"reason": "external_cancellation"})
        self._mark_poisoned_if_unconverged()

    async def _best_effort_event(
        self,
        kind: str,
        *,
        payload: object,
        request_id: str | None = None,
        turn: int | None = None,
        scope_id: str | None = None,
    ) -> None:
        """Try once, inside the bounded cleanup window, to persist a terminal record.

        F01/F05: the window is checked before a new audit task is started, so an exhausted
        window never starts work it cannot wait for. F01: looking up the sink's ``append``,
        building the event and creating the awaitable are synchronous steps that can consume
        the whole window themselves, so the *same* absolute deadline is re-read after that
        preparation and before any task is scheduled; an already-expired window closes the
        unscheduled coroutine instead of starting work outside its bound, and the record is
        reported as unconverged. A task that is cancelled, raises or does not settle inside
        that same window is retained as an unconverged terminal record; it is never retried
        with a fresh window.
        """
        remaining = self._remaining_cleanup()
        if remaining <= 0.0:
            self._audit_unconverged = True
            return
        try:
            append = self._audit.append
            event = self._next_event(
                kind, payload=payload, request_id=request_id, turn=turn, scope_id=scope_id)
            awaitable = append(event)
        except Exception:  # noqa: BLE001 - an unusable sink is an unconverged terminal record
            self._audit_unconverged = True
            return
        remaining = self._remaining_cleanup()
        if remaining <= 0.0:
            self._dispose(awaitable)
            self._audit_unconverged = True
            return
        try:
            task = asyncio.ensure_future(awaitable)
        except Exception:  # noqa: BLE001 - an unusable sink is an unconverged terminal record
            self._dispose(awaitable)
            self._audit_unconverged = True
            return
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                self._retain(task)
            self._audit_unconverged = True
            raise
        if not task.done():
            task.cancel()
            self._retain(task)
            self._audit_unconverged = True
            return
        failure = None
        if task.cancelled():
            self._audit_unconverged = True
        else:
            with contextlib.suppress(BaseException):
                failure = task.exception()
            if failure is not None:
                self._audit_unconverged = True
        self._consume(task)

    def _retain(self, task: asyncio.Task) -> None:
        if task.done():
            self._consume(task)
            return
        self._orphans.add(task)
        task.add_done_callback(self._forget)

    def _forget(self, task: asyncio.Task) -> None:
        self._orphans.discard(task)
        self._consume(task)

    @staticmethod
    def _consume(task: asyncio.Task) -> None:
        with contextlib.suppress(BaseException):
            task.exception()

    def _close_owns_cancellation(self, task: asyncio.Task | None) -> bool:
        """True only for the single cancellation request this planner's close owns."""
        return (
            self._closed
            and task is not None
            and self._close_cancelled_plan is task
            and task.cancelling() == 1
        )

    def _check_deadline(self, deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise _fail("timeout", "deadline")

    def _provider_payload(self, response: ModelResponse, request: RetrievalModelRequest) -> dict:
        usage = response.usage
        return {
            "request_id": response.request_id,
            "status": response.status,
            "content": response.content,
            "usage": None if usage is None else {
                "schema_version": usage.schema_version,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
            "usage_known": usage is not None,
            "request_max_output_tokens": request.base_request.max_output_tokens,
        }

    def _provider_return_record(
        self, response: object, request: RetrievalModelRequest) -> dict:
        """Bound and describe a provider value that did not pass the response shell gate.

        The value really arrived, so it is not disguised as an absent return. A declared
        record is copied through its own storage (never through an overridable serializer)
        and its actual content is recorded while it fits the response byte cap; a value that
        is not a declared response, or whose content cannot be represented or is oversized,
        is recorded as an explicit rejection instead of being silently dropped.
        """
        try:
            current = copy_record(response, ModelResponse, "invalid_response", "response")
        except RetrievalError:
            return {
                "request_id": None,
                "status": "rejected",
                "content": None,
                "usage": None,
                "usage_known": False,
                "request_max_output_tokens": request.base_request.max_output_tokens,
                "rejection": "not_a_declared_response",
            }
        payload = self._provider_payload(current, request)
        content = current.content
        if isinstance(content, str):
            try:
                encoded = content.encode("utf-8")
            except UnicodeEncodeError:
                payload.update({
                    "content": None,
                    "content_bytes": 0,
                    "content_omitted": True,
                    "rejection": "invalid_text",
                })
                return payload
            if len(encoded) > self._base.max_response_bytes:
                payload.update({
                    "content": None,
                    "content_bytes": len(encoded),
                    "content_omitted": True,
                    "rejection": "response_too_large",
                })
        elif content is not None:
            payload.update({
                "content": None,
                "content_omitted": True,
                "rejection": "invalid_content",
            })
        return payload

    # ------------------------------------------------------------------ close

    def _pending_count(
        self, plan_task: asyncio.Task | None, current: asyncio.Task | None) -> int:
        """Count every unresolved external operation; never a fabricated zero."""
        unresolved = {task for task in self._inflight | self._orphans if not task.done()}
        if plan_task is not None and plan_task is not current and not plan_task.done():
            unresolved.add(plan_task)
        return len(unresolved)

    async def aclose(self) -> PlannerCloseResult:
        """Stop accepting work, cancel outstanding operations and report the real drain.

        F01: a repeated close reuses the same cleanup window instead of granting a new one.
        The reported ``pending_calls`` is recomputed after the close record was attempted,
        because that record is itself an external operation; if it is still running it is
        counted, retained and the planner is poisoned.
        """
        self._closed = True
        self._begin_cleanup()
        tasks: set[asyncio.Task] = {
            task for task in self._inflight | self._orphans if not task.done()
        }
        for task in tasks:
            task.cancel()
        plan_task = self._plan_task
        current = asyncio.current_task()
        if (
            plan_task is not None
            and plan_task is not current
            and not plan_task.done()
            and plan_task.cancelling() == 0
        ):
            self._close_cancelled_plan = plan_task
            plan_task.cancel()
            tasks.add(plan_task)
        if tasks:
            remaining = self._remaining_cleanup()
            if remaining > 0.0:
                try:
                    await asyncio.wait(tasks, timeout=remaining)
                except asyncio.CancelledError:
                    for task in tasks:
                        if task.done():
                            self._consume(task)
                        else:
                            self._retain(task)
                    raise
        pending = self._pending_count(plan_task, current)
        if not self._closed_event:
            self._closed_event = True
            await self._best_effort_event(
                "plan_closed",
                payload={"drained": pending == 0, "pending_calls": pending},
            )
        # The close record was an external operation too. Recompute the truth after it.
        pending = self._pending_count(plan_task, current)
        if pending:
            self._poisoned = True
        self._task = None
        return PlannerCloseResult(
            schema_version=SCHEMA_VERSION,
            drained=pending == 0,
            pending_calls=pending,
        )
