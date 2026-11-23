"""The offline structured LLMPlanner.

:class:`LLMPlanner` implements the existing Core ``Planner`` protocol exactly:
``async plan(observation: Observation) -> Forecast``. It keeps the Core lifecycle semantics of
its own single provider call - one call per ``plan``, no hidden retry, a bounded timeout and a
bounded cleanup - and it never fabricates an action.

A successful call returns a *proposal*. The Contract Builder and the domain still decide
whether it may be committed and whether anything may be executed; the planner calls no
adapter, reads no environment and decides no terminal state. When the outcome is unknown the
planner raises the fixed facility error instead of guessing.

The provider is injected and must be offline and cooperative. This module starts no thread,
no process and no network connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
import uuid

from forecontract.models.actions import Forecast
from forecontract.models.observations import Observation

from forecontract_llm.codec import build_request, decode_response
from forecontract_llm.errors import PlannerError
from forecontract_llm.models import (
    ModelResponse,
    PlannerCloseResult,
    PlannerConfig,
    PlannerContext,
    read_declared_copy,
)

__all__ = ["LLMPlanner"]

_READY = "ready"
_BUSY = "busy"
_POISONED = "poisoned"
_CLOSED = "closed"


def _fail(code: str, message: str, location: tuple[str | int, ...]) -> PlannerError:
    return PlannerError(code, message, location=location)


class LLMPlanner:
    """One planner bound to exactly one Run's context and one injected offline provider.

    The host creates a new planner for each Run and guarantees that the context identifiers
    and task criteria match that Run's request. Core does not hand the planner a ``RunRequest``,
    so the planner cannot authenticate that binding and does not claim to.
    """

    def __init__(
        self,
        *,
        provider: object,
        context: PlannerContext,
        config: PlannerConfig,
    ) -> None:
        try:
            complete = getattr(provider, "complete", None)
        except Exception:  # noqa: BLE001 - a hostile descriptor must not leak its text
            raise _fail("INPUT_INVALID", "the provider does not offer an asynchronous complete",
                        ("provider",)) from None
        if not callable(complete) or not inspect.iscoroutinefunction(complete):
            raise _fail("INPUT_INVALID", "the provider does not offer an asynchronous complete",
                        ("provider",))
        self._provider = provider
        self._context = read_declared_copy(context, PlannerContext, "INPUT_INVALID")
        self._config = read_declared_copy(config, PlannerConfig, "INPUT_INVALID")
        self._calls_used = 0
        self._busy = False
        self._closed = False
        self._poisoned = False
        self._task: asyncio.Task | None = None
        self._plan_task: asyncio.Task | None = None
        self._close_cancelled_plan: asyncio.Task | None = None
        self._orphans: set[asyncio.Task] = set()

    @property
    def calls_used(self) -> int:
        """How many provider calls this planner has actually started."""
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

    async def plan(self, observation: Observation) -> Forecast:
        """Return one structured forecast for this observation, or a bounded failure.

        Host cancellation keeps its own identity. A cancellation requested by :meth:`aclose`
        ends the in-flight plan as ``CLOSED``; a cancellation observed from the host first - or
        one the host delivers while the close-triggered cleanup is still draining - is re-raised
        as ``CancelledError``, so neither a close that follows an observed external cancel nor a
        close that started first can rewrite the caller's cancellation.
        """
        if self._closed:
            raise _fail("CLOSED", "the planner is closed", ("lifecycle",))
        if self._poisoned:
            raise _fail("POISONED", "an earlier provider call could not be settled",
                        ("lifecycle",))
        if self._busy:
            raise _fail("BUSY", "another planning call is already in progress", ("lifecycle",))
        self._busy = True
        self._plan_task = asyncio.current_task()
        try:
            try:
                deadline = time.monotonic() + (self._config.call_timeout_ms / 1000.0)
                current = read_declared_copy(observation, Observation, "INPUT_INVALID")
                if current.environment_type != self._context.capabilities.environment_type:
                    raise _fail(
                        "CONTEXT_MISMATCH",
                        "the observation environment does not match the declared capabilities",
                        ("environment_type",),
                    )
                request = build_request(
                    context=self._context,
                    observation=current,
                    config=self._config,
                    request_id=str(uuid.uuid4()),
                )
                if self._calls_used >= self._config.max_calls:
                    raise _fail("BUDGET_EXCEEDED", "the configured call allowance is exhausted",
                                ("calls",))
                if time.monotonic() >= deadline:
                    raise _fail("TIMEOUT",
                                "the planning deadline expired before the provider call",
                                ("deadline",))
                self._calls_used += 1
                task = self._start_call(request)
                self._task = task
                try:
                    response = await self._await_provider(task, deadline)
                except PlannerError:
                    raise
                except Exception:  # noqa: BLE001 - one bounded provider failure
                    raise _fail("PROVIDER_FAILURE", "the provider call failed",
                                ("provider",)) from None
                if not isinstance(response, ModelResponse):
                    raise _fail("PROVIDER_FAILURE",
                                "the provider did not return a response record", ("provider",))
                if time.monotonic() >= deadline:
                    raise _fail("TIMEOUT", "the provider answered after the planning deadline",
                                ("deadline",))
                forecast = decode_response(response=response, request=request,
                                           config=self._config)
                if time.monotonic() >= deadline:
                    raise _fail("TIMEOUT",
                                "the forecast was produced after the planning deadline",
                                ("deadline",))
                if self._closed:
                    raise _fail("CLOSED",
                                "the planner was closed before the forecast was returned",
                                ("lifecycle",))
                return forecast
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if self._close_owns_cancellation(current_task):
                    raise _fail("CLOSED",
                                "the planner was closed while the call was in flight",
                                ("lifecycle",)) from None
                raise
        finally:
            if self._close_cancelled_plan is asyncio.current_task():
                self._close_cancelled_plan = None
            self._plan_task = None
            self._busy = False

    def _close_owns_cancellation(self, task: asyncio.Task | None) -> bool:
        """True only for the single cancellation request this planner's close owns.

        ``Task.cancelling()`` counts every pending cancellation for the task. ``aclose`` sends
        exactly one when it owns the plan; a host cancellation delivered later - even while the
        close-triggered cleanup is still draining - raises the count. That observed external
        cancellation must keep its ``CancelledError``; only the unmodified close-owned count may
        be rewritten as ``CLOSED``.
        """
        return (self._closed
                and task is not None
                and self._close_cancelled_plan is task
                and task.cancelling() == 1)

    def _start_call(self, request: object) -> asyncio.Task:
        """Start exactly one provider call task behind the untrusted provider boundary."""
        try:
            complete = self._provider.complete
        except Exception:  # noqa: BLE001 - a hostile descriptor must not leak its text
            raise _fail("PROVIDER_FAILURE", "the provider could not be invoked",
                        ("provider",)) from None
        try:
            return asyncio.create_task(complete(request.model_copy(deep=True)))
        except Exception:  # noqa: BLE001 - a broken transport is one bounded failure
            raise _fail("PROVIDER_FAILURE", "the provider could not be invoked",
                        ("provider",)) from None

    async def _await_provider(self, task: asyncio.Task, deadline: float) -> ModelResponse:
        """Await the single call task under the absolute deadline and a bounded cleanup.

        An error raised by the provider - including a ``PlannerError`` it forged itself - is
        normalised at this boundary into the fixed ``PROVIDER_FAILURE``; only the planner's own
        deadline and lifecycle errors keep their codes. Host cancellation is re-raised after a
        bounded drain so the caller keeps its ``CancelledError``.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            task.cancel()
            settled = await self._drain(task)
            self._settle(settled)
            raise _fail("TIMEOUT", "the planning deadline expired before the provider answered",
                        ("deadline",))
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                if not task.done():
                    task.cancel()
                try:
                    settled = await self._drain(task)
                except asyncio.CancelledError:
                    self._retain(task)
                    settled = False
                self._settle(settled)
                raise
            raise
        if not task.done():
            task.cancel()
            settled = await self._drain(task)
            self._settle(settled)
            raise _fail("TIMEOUT", "the provider did not answer before the planning deadline",
                        ("deadline",))
        return self._provider_result(task)

    @staticmethod
    def _provider_result(task: asyncio.Task) -> ModelResponse:
        """Read one settled provider task, normalising every provider-origin failure."""
        try:
            return task.result()
        except asyncio.CancelledError:
            raise _fail("PROVIDER_FAILURE", "the provider call did not complete",
                        ("provider",)) from None
        except PlannerError:
            raise _fail("PROVIDER_FAILURE", "the provider call failed",
                        ("provider",)) from None
        except Exception:  # noqa: BLE001 - a provider failure is one bounded code
            raise _fail("PROVIDER_FAILURE", "the provider call failed",
                        ("provider",)) from None

    async def _drain(self, task: asyncio.Task) -> bool:
        """Wait one cleanup allowance for one call task and report whether it settled."""
        if task.done():
            self._consume(task)
            return True
        try:
            await asyncio.wait(
                {task}, timeout=self._config.cleanup_timeout_ms / 1000.0
            )
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            self._retain(task)
            raise
        if task.done():
            self._consume(task)
            return True
        self._retain(task)
        return False

    def _settle(self, settled: bool) -> None:
        """A call task that did not settle leaves the planner poisoned."""
        if not settled:
            self._poisoned = True

    def _retain(self, task: asyncio.Task) -> None:
        """Keep a reference to an unsettled task and consume its eventual exception."""
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

    async def aclose(self) -> PlannerCloseResult:
        """Stop accepting work, request cancellation and report the truth of the drain.

        Idempotent. Each call waits at most one cleanup allowance and states how many provider
        call tasks are still pending. It does not claim to kill uncooperative code.
        """
        self._closed = True
        provider_tasks: set[asyncio.Task] = set()
        candidates: set[asyncio.Task] = set(self._orphans)
        if self._task is not None:
            candidates.add(self._task)
        for task in candidates:
            if task.done():
                self._consume(task)
                self._orphans.discard(task)
            else:
                provider_tasks.add(task)
        plan_task = self._plan_task
        current = asyncio.current_task()
        for task in provider_tasks:
            task.cancel()
        if (plan_task is not None and plan_task is not current and not plan_task.done()
                and plan_task.cancelling() == 0):
            # Only a plan that has not already observed a host cancellation is cancelled here,
            # and only this object may later report that cancellation as CLOSED. An external
            # cancellation keeps its CancelledError even when aclose follows it.
            self._close_cancelled_plan = plan_task
            plan_task.cancel()
        wait_for: set[asyncio.Task] = set(provider_tasks)
        if plan_task is not None and plan_task is not current and not plan_task.done():
            wait_for.add(plan_task)
        if wait_for:
            try:
                await asyncio.wait(
                    wait_for, timeout=self._config.cleanup_timeout_ms / 1000.0
                )
            except asyncio.CancelledError:
                for task in provider_tasks:
                    if task.done():
                        self._consume(task)
                        self._orphans.discard(task)
                    else:
                        self._retain(task)
                self._task = None
                raise
        pending = 0
        for task in provider_tasks:
            if task.done():
                self._consume(task)
                self._orphans.discard(task)
            else:
                self._retain(task)
                pending += 1
        self._task = None
        return PlannerCloseResult(
            schema_version="0.1",
            drained=pending == 0,
            pending_calls=pending,
        )
