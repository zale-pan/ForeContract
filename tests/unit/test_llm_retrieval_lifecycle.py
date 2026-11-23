"""Acceptance tests for cancellation, timeouts, poisoning and honest close.

C06: an external cancellation keeps its ``CancelledError``; a close-triggered cancellation is
reported as a closed lifecycle; an operation that does not settle inside the single cleanup
window is retained, the planner is poisoned and ``aclose`` reports the truth instead of a
fabricated drain. A termination audit failure never masks the original cancellation.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

import pytest
from forecontract.models import AdapterCapabilities, ExpectedOutcome, Observation
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm.models import ModelResponse, PlannerConfig, PlannerContext
from forecontract_llm.retrieval import (
    EvidenceReadResult,
    EvidenceScope,
    RetrievalConfig,
    RetrievalError,
)
from forecontract_llm.retrieval.models import (
    EvidenceScopeEntry,
    ScopeView,
    canonical_json_bytes,
)
from forecontract_llm.retrieval.planner import EvidenceLLMPlanner

SCHEMA = "0.1"
RUN = "run-1"
OBS = "obs-1"
HEX = "a" * 64


def _context() -> PlannerContext:
    capabilities = AdapterCapabilities(
        schema_version=SCHEMA, adapter_name="synthetic", environment_type="offline",
        supported_actions=["mark_done"], supported_outcome_types=["json_rule"])
    criteria = TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id="goal-1",
        conditions=[ExpectedOutcome(schema_version=SCHEMA, outcome_type="json_rule")])
    return PlannerContext(
        schema_version=SCHEMA, run_id=RUN, goal_id="goal-1", goal_text="finish",
        task_criteria=criteria, capabilities=capabilities,
        action_guidance={"mark_done": {"payload": {"type": "object"}}},
        outcome_guidance={"json_rule": {"payload": {"type": "object"}}})


def _config(*, call_timeout_ms: float = 5_000, cleanup_timeout_ms: float = 60,
            max_reads: int = 2) -> RetrievalConfig:
    return RetrievalConfig(
        schema_version=SCHEMA,
        base=PlannerConfig(schema_version=SCHEMA, call_timeout_ms=call_timeout_ms,
                           cleanup_timeout_ms=cleanup_timeout_ms),
        max_reads=max_reads)


def _observation() -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=OBS,
                       environment_type="offline",
                       state={"done": False, "order": {"status": "pending"}},
                       timestamp=datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc))


def _scope() -> EvidenceScope:
    return EvidenceScope(
        schema_version=SCHEMA, scope_id="scope-1", run_id=RUN, observation_id=OBS,
        entries=[EvidenceScopeEntry(
            schema_version=SCHEMA, evidence_id="evidence-1",
            source_observation_id=OBS, kind="observation", raw_sha256=HEX, raw_bytes=10,
            views=[ScopeView(schema_version=SCHEMA, view_id="status",
                             selector=["state", "order", "status"], format="text")])])


class Reader:
    def __init__(self, *, delay: float = 0.0, cooperative: bool = True) -> None:
        self.scope = _scope()
        self.delay = delay
        self.cooperative = cooperative
        self.reads = []
        self.closed: list[str] = []
        self.release = asyncio.Event()

    def open_scope(self, run_id: str, observation_id: str) -> EvidenceScope:
        return self.scope

    async def read(self, request):
        self.reads.append(request)
        if not self.delay:
            return self._result(request)
        if self.cooperative:
            await asyncio.sleep(self.delay)
        else:
            while True:
                try:
                    await self.release.wait()
                    return self._result(request)
                except asyncio.CancelledError:
                    continue
        return self._result(request)

    def _result(self, request) -> EvidenceReadResult:
        value = "pending"
        raw = canonical_json_bytes(value)
        return EvidenceReadResult(
            schema_version=SCHEMA, scope_id=request.scope_id, run_id=request.run_id,
            plan_id=request.plan_id, turn=request.turn, request_id=request.request_id,
            evidence_id=request.evidence_id, view_id=request.view_id, status="found",
            source_observation_id=OBS, raw_sha256=HEX, raw_bytes=10,
            selector=["state", "order", "status"], format="text",
            text_range=request.text_range, partial=False, value=value,
            value_sha256=__import__("hashlib").sha256(raw).hexdigest(),
            value_bytes=len(raw))

    def close_scope(self, scope_id: str) -> None:
        self.closed.append(scope_id)


class Sink:
    def __init__(self, *, fail_on: set[str] | None = None, uncooperative: bool = False) -> None:
        self.events = []
        self.fail_on = fail_on or set()
        self.uncooperative = uncooperative
        self.release = asyncio.Event()

    async def append(self, event) -> None:
        if event.kind in self.fail_on:
            raise RuntimeError("the audit sink is unavailable")
        if self.uncooperative:
            while True:
                try:
                    await self.release.wait()
                    return
                except asyncio.CancelledError:
                    continue
        self.events.append(event)


class Provider:
    def __init__(self, *, delay: float = 0.0, cooperative: bool = True,
                 answer: object = None) -> None:
        self.delay = delay
        self.cooperative = cooperative
        self.answer = answer
        self.calls = 0
        self.started = asyncio.Event()
        self.requests: list = []
        self.release = asyncio.Event()

    async def complete(self, request):
        self.calls += 1
        self.requests.append(request)
        self.started.set()
        if self.delay:
            if self.cooperative:
                await asyncio.sleep(self.delay)
            else:
                while True:
                    try:
                        await self.release.wait()
                        break
                    except asyncio.CancelledError:
                        continue
        if self.answer is not None:
            return self.answer(request)
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="ok",
                             content=json.dumps({
                                 "schema_version": SCHEMA, "run_id": request.run_id,
                                 "plan_id": request.plan_id, "turn": request.turn,
                                 "request_id": request.request_id, "kind": "forecast",
                                 "forecast": {
                                     "schema_version": SCHEMA,
                                     "proposed_action": {
                                         "schema_version": SCHEMA,
                                         "action_type": "mark_done", "payload": {}},
                                     "expected_outcomes": [
                                         {"schema_version": SCHEMA,
                                          "outcome_type": "json_rule",
                                          "payload": {"schema_version": SCHEMA,
                                                      "rule_type": "value_equals",
                                                      "path": ["done"],
                                                      "expected": True}}],
                                     "task_completion_hint": True}}))


def _planner(provider, *, reader: Reader | None = None, sink: Sink | None = None,
             config: RetrievalConfig | None = None) -> tuple[EvidenceLLMPlanner,
                                                             Reader, Sink]:
    current_reader = reader or Reader()
    current_sink = sink or Sink()
    planner = EvidenceLLMPlanner(provider=provider, context=_context(),
                                 config=config or _config(), reader=current_reader,
                                 audit=current_sink)
    return planner, current_reader, current_sink


def _kinds(sink: Sink) -> list[str]:
    return [event.kind for event in sink.events]


def _read_content(request) -> str:
    return json.dumps({
        "schema_version": SCHEMA, "run_id": request.run_id, "plan_id": request.plan_id,
        "turn": request.turn, "request_id": request.request_id, "kind": "read_evidence",
        "read": {"evidence_id": "evidence-1", "view_id": "status", "text_range": None}})


def test_a_host_cancellation_keeps_its_cancelled_error() -> None:
    """C06: a host cancellation is re-raised and recorded as a cancellation."""
    sink = Sink()
    provider = Provider(delay=5)
    planner, _reader, _sink = _planner(provider, sink=sink)

    async def main():
        task = asyncio.create_task(planner.plan(_observation()))
        await provider.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await planner.aclose()

    close = asyncio.run(main())
    assert _kinds(sink) == ["plan_started", "request_prepared", "plan_cancelled",
                            "plan_closed"]
    assert (close.drained, close.pending_calls) == (True, 0)


def test_a_close_triggered_cancellation_is_a_closed_lifecycle() -> None:
    """C06: only this object's own close cancellation is reported as ``invalid_state``."""
    sink = Sink()
    provider = Provider(delay=5)
    planner, _reader, _sink = _planner(provider, sink=sink)

    async def main():
        task = asyncio.create_task(planner.plan(_observation()))
        await provider.started.wait()
        close = await planner.aclose()
        with pytest.raises(RetrievalError) as info:
            await task
        return close, info.value

    close, error = asyncio.run(main())
    assert (error.code, error.stage) == ("invalid_state", "lifecycle")
    assert (close.drained, close.pending_calls) == (True, 0)
    assert _kinds(sink) == ["plan_started", "request_prepared", "plan_closed"]


def test_an_uncooperative_provider_is_retained_and_poisoned() -> None:
    """C06: an operation that ignores cancellation is kept, poison and reported honestly."""
    provider = Provider(delay=10, cooperative=False)
    planner, _reader, _sink = _planner(
        provider, config=_config(call_timeout_ms=40, cleanup_timeout_ms=40))

    async def main():
        started = time.monotonic()
        with pytest.raises(RetrievalError) as info:
            await planner.plan(_observation())
        state_after_plan = planner.state
        close = await planner.aclose()
        elapsed = time.monotonic() - started
        provider.release.set()
        await asyncio.sleep(0.05)
        return info.value, close, elapsed, state_after_plan

    error, close, elapsed, state_after_plan = asyncio.run(main())
    assert (error.code, error.stage) == ("timeout", "deadline")
    assert close.drained is False
    assert close.pending_calls >= 1
    assert state_after_plan == "poisoned"
    assert elapsed < 5.0
    with pytest.raises(RetrievalError) as info:
        asyncio.run(planner.plan(_observation()))
    assert (info.value.code, info.value.stage) == ("invalid_state", "lifecycle")


def test_a_late_reader_result_cannot_publish_a_forecast() -> None:
    """C06: a timed-out read is cancelled and never becomes evidence for a forecast."""
    provider = Provider()
    reader = Reader(delay=0.3)
    planner, reader, _sink = _planner(
        provider, reader=reader, config=_config(call_timeout_ms=60, cleanup_timeout_ms=40,
                                                max_reads=2))

    async def main():
        provider.answer = lambda request: ModelResponse(
            schema_version=SCHEMA, request_id=request.request_id, status="ok",
            content=json.dumps({
                "schema_version": SCHEMA, "run_id": request.run_id,
                "plan_id": request.plan_id, "turn": request.turn,
                "request_id": request.request_id, "kind": "read_evidence",
                "read": {"evidence_id": "evidence-1", "view_id": "status",
                         "text_range": None}}))
        try:
            return await planner.plan(_observation())
        finally:
            await planner.aclose()

    with pytest.raises(RetrievalError) as info:
        asyncio.run(main())
    assert (info.value.code, info.value.stage) == ("timeout", "deadline")
    assert provider.calls == 1


def test_a_reader_timeout_is_a_bounded_deadline_failure() -> None:
    """C06: an uncooperative reader is cancelled inside the plan deadline."""
    provider = Provider()
    reader = Reader(delay=10, cooperative=False)
    planner, reader, _sink = _planner(
        provider, reader=reader, config=_config(call_timeout_ms=60, cleanup_timeout_ms=30))

    async def main():
        provider.answer = lambda request: ModelResponse(
            schema_version=SCHEMA, request_id=request.request_id, status="ok",
            content=_read_content(request))
        try:
            await planner.plan(_observation())
        except RetrievalError as error:
            state_after_plan = planner.state
            await planner.aclose()
            reader.release.set()
            await asyncio.sleep(0.05)
            return error, state_after_plan
        raise AssertionError("the uncooperative reader did not time out")

    error, state_after_plan = asyncio.run(main())
    assert (error.code, error.stage) == ("timeout", "deadline")
    assert state_after_plan == "poisoned"
    assert planner.audit_unconverged is True


def test_an_uncooperative_audit_sink_poisons_the_instance() -> None:
    """C07/C06: a sink that never settles is bounded and honestly reported."""
    sink = Sink(uncooperative=True)
    provider = Provider()
    planner, _reader, _sink = _planner(
        provider, sink=sink, config=_config(call_timeout_ms=60, cleanup_timeout_ms=30))

    async def main():
        with pytest.raises(RetrievalError) as info:
            await planner.plan(_observation())
        state_after_plan = planner.state
        close = await planner.aclose()
        sink.release.set()
        await asyncio.sleep(0.05)
        return info.value, close, state_after_plan

    error, close, state_after_plan = asyncio.run(main())
    assert error.code == "timeout"
    assert close.drained is False
    assert close.pending_calls >= 1
    assert state_after_plan == "poisoned"


def test_a_cancellation_is_not_masked_by_an_audit_failure() -> None:
    """C06: a failed terminal audit never replaces the host's CancelledError."""
    sink = Sink(fail_on={"plan_cancelled"})
    provider = Provider(delay=5)
    planner, _reader, _sink = _planner(provider, sink=sink)

    async def main():
        task = asyncio.create_task(planner.plan(_observation()))
        await provider.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await planner.aclose()
        return planner.audit_unconverged

    assert asyncio.run(main()) is True


def test_close_is_idempotent_and_emits_one_close_record() -> None:
    """C06: a second close is harmless and does not fabricate a second terminal record."""
    provider = Provider()
    planner, _reader, sink = _planner(provider, config=_config(max_reads=0))

    async def main():
        await planner.plan(_observation())
        first = await planner.aclose()
        second = await planner.aclose()
        return first, second

    first, second = asyncio.run(main())
    assert (first.drained, second.drained) == (True, True)
    assert (first.pending_calls, second.pending_calls) == (0, 0)
    assert _kinds(sink).count("plan_closed") == 1


def test_a_close_on_a_fresh_planner_is_legal() -> None:
    """C06: closing an unused planner reports a clean drain and one close record."""
    planner, _reader, sink = _planner(Provider())

    async def main():
        return await planner.aclose()

    close = asyncio.run(main())
    assert (close.drained, close.pending_calls) == (True, 0)
    assert _kinds(sink) == ["plan_closed"]
    assert planner.state == "closed"


def test_a_reader_read_is_not_called_after_its_scope_was_closed() -> None:
    """C06: the plan closes its scope in the same finally that clears the plan."""
    provider = Provider()
    planner, reader, _sink = _planner(provider, config=_config(max_reads=0))

    async def main():
        await planner.plan(_observation())
        await planner.aclose()

    asyncio.run(main())
    assert reader.closed == ["scope-1"]


def test_an_external_cancel_after_close_keeps_a_cancelled_error() -> None:
    """C06: a host cancellation observed after close is still a cancellation."""
    provider = Provider(delay=5)
    planner, _reader, _sink = _planner(provider)

    async def main():
        task = asyncio.create_task(planner.plan(_observation()))
        await provider.started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await planner.aclose()

    asyncio.run(main())
    with pytest.raises(RetrievalError) as info:
        asyncio.run(planner.plan(_observation()))
    assert info.value.code == "invalid_state"


def test_close_reports_pending_audit() -> None:
    """F01: a close record that does not settle is a real pending call, not a clean drain.

    The close audit append is itself an external operation. It shares the one cleanup
    window; when it is still running afterwards the close result must say so and retain it.
    """
    release = asyncio.Event()
    entered = asyncio.Event()

    class SlowCloseSink(Sink):
        async def append(self, event) -> None:
            if event.kind == "plan_closed":
                entered.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
            await super().append(event)

    sink = SlowCloseSink()
    planner, _reader, _sink = _planner(
        Provider(), sink=sink, config=_config(cleanup_timeout_ms=40))

    async def main():
        close = await planner.aclose()
        retained = len([task for task in planner._orphans if not task.done()])
        release.set()
        await asyncio.sleep(0.02)
        return close, retained

    close, retained = asyncio.run(main())
    assert entered.is_set() is True
    assert close.drained is False
    assert close.pending_calls >= 1
    assert retained >= 1
    assert planner.audit_unconverged is True
    assert planner.state == "closed"


def test_terminal_audit_unsettled_poisons() -> None:
    """F01: an unsettled terminal audit poisons the instance and blocks a second plan."""
    release = asyncio.Event()

    class SlowFailureSink(Sink):
        async def append(self, event) -> None:
            if event.kind == "plan_failed":
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
            await super().append(event)

    class FailingProvider(Provider):
        async def complete(self, request):
            self.calls += 1
            raise RuntimeError("synthetic provider failure")

    sink = SlowFailureSink()
    provider = FailingProvider()
    planner, _reader, _sink = _planner(
        provider, sink=sink, config=_config(call_timeout_ms=500, cleanup_timeout_ms=30))

    async def main():
        with pytest.raises(RetrievalError) as first:
            await planner.plan(_observation())
        state = planner.state
        with pytest.raises(RetrievalError) as second:
            await planner.plan(_observation())
        release.set()
        await asyncio.sleep(0.02)
        await planner.aclose()
        return first.value, state, second.value

    first, state, second = asyncio.run(main())
    assert (first.code, first.stage) == ("provider_failed", "provider")
    assert state == "poisoned"
    assert (second.code, second.stage) == ("invalid_state", "lifecycle")
    assert provider.calls == 1


def test_terminal_audit_cancel_marks_unconverged() -> None:
    """F01: a sink that raises CancelledError is an unconverged terminal record."""
    class CancelledCloseSink(Sink):
        async def append(self, event) -> None:
            if event.kind == "plan_closed":
                raise asyncio.CancelledError()
            await super().append(event)

    planner, _reader, _sink = _planner(Provider(), sink=CancelledCloseSink())
    close = asyncio.run(planner.aclose())
    assert planner.audit_unconverged is True
    assert close.drained is True
    assert close.pending_calls == 0


def test_reader_call_failure_sanitized() -> None:
    """F04: a synchronous ``read`` shape is refused at construction, and a call-time
    ``ValueError`` carrying facility text is normalised to ``reader_failed`` without echo.
    """
    class SyncReader:
        def open_scope(self, run_id: str, observation_id: str):
            return _scope()

        def read(self, request):
            raise ValueError("SYNTHETIC-SENSITIVE-DETAIL")

        def close_scope(self, scope_id: str) -> None:
            return None

    with pytest.raises(RetrievalError) as rejected:
        EvidenceLLMPlanner(provider=Provider(), context=_context(), config=_config(),
                           reader=SyncReader(), audit=Sink())
    assert (rejected.value.code, rejected.value.stage) == ("invalid_input", "reader")
    assert "SYNTHETIC-SENSITIVE-DETAIL" not in str(rejected.value)

    reader = Reader()
    planner = EvidenceLLMPlanner(
        provider=Provider(answer=lambda request: ModelResponse(
            schema_version=SCHEMA, request_id=request.request_id, status="ok",
            content=_read_content(request))),
        context=_context(), config=_config(), reader=reader, audit=Sink())

    def leaking_read(request):
        raise ValueError("SYNTHETIC-SENSITIVE-DETAIL")

    reader.read = leaking_read
    with pytest.raises(RetrievalError) as failed:
        asyncio.run(planner.plan(_observation()))
    assert (failed.value.code, failed.value.stage) == ("reader_failed", "reader")
    assert "SYNTHETIC-SENSITIVE-DETAIL" not in str(failed.value)


def test_a_cancelled_plan_records_the_cleanup_facts_in_one_window() -> None:
    """C06/C07: the close record is written inside the bounded cleanup window."""
    sink = Sink()
    provider = Provider(delay=5)
    planner, _reader, _sink = _planner(provider, sink=sink,
                                       config=_config(cleanup_timeout_ms=100))

    async def main():
        task = asyncio.create_task(planner.plan(_observation()))
        await provider.started.wait()
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            started = time.monotonic()
            close = await planner.aclose()
            elapsed = time.monotonic() - started
        return close, elapsed

    close, elapsed = asyncio.run(main())
    assert close.drained is True
    assert elapsed < 2.0
    kinds = _kinds(sink)
    assert kinds.index("plan_cancelled") < kinds.index("plan_closed")


# --------------------------------------------------------------------------- F01 deadline recheck


class BlockingLookupProvider(Provider):
    """A legal async provider whose ``complete`` descriptor blocks synchronously first."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.lookup_delay = delay

    @property
    def complete(self):
        time.sleep(self.lookup_delay)
        return self._complete

    async def _complete(self, request):
        return await super().complete(request)


class BlockingLookupSink:
    """A legal async sink whose ``append`` descriptor blocks synchronously first."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.events = []
        self.started = 0

    @property
    def append(self):
        time.sleep(self.delay)
        return self._append

    async def _append(self, event) -> None:
        self.started += 1
        self.events.append(event)


class FailingProvider:
    """A provider whose body only ever fails, so every plan needs a terminal record."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        raise RuntimeError("synthetic provider failure")


def test_a_synchronous_lookup_that_expires_the_deadline_starts_no_call() -> None:
    """F01: a synchronous descriptor evaluation that outlasts the plan deadline must not
    schedule the already-prepared provider coroutine outside its deadline.

    The provider is legal (an async method after a 60 ms synchronous lookup) and the plan
    deadline is 25 ms. The remaining time is re-read after the awaitable exists, so the coroutine
    body never runs, the prepared coroutine is closed rather than leaked, and the caller sees
    the deadline failure with an honest close.
    """
    provider = BlockingLookupProvider(delay=0.06)
    planner, _reader, _sink = _planner(
        provider, config=_config(call_timeout_ms=25, cleanup_timeout_ms=100, max_reads=0))

    async def main():
        try:
            with pytest.raises(RetrievalError) as info:
                await planner.plan(_observation())
            error = info.value
        finally:
            close = await planner.aclose()
        return error, close

    error, close = asyncio.run(main())
    assert (error.code, error.stage) == ("timeout", "deadline")
    assert provider.calls == 0
    assert provider.requests == []
    # The terminal record itself had a 100 ms window and a fast sink, so it converged.
    assert planner.audit_unconverged is False
    assert (close.drained, close.pending_calls) == (True, 0)


def test_a_terminal_audit_is_not_started_after_the_cleanup_window_expired() -> None:
    """F01: a 60 ms synchronous ``append`` lookup cannot start work after a 25 ms window.

    The sink is legal (an async method is returned), but the lookup itself spends the whole
    cleanup window. The same absolute cleanup deadline is re-read after the event and awaitable
    were prepared; the unscheduled coroutine is closed, no audit body runs, the record is
    reported as unconverged and the close still tells the truth.
    """
    sink = BlockingLookupSink(delay=0.06)
    provider = FailingProvider()
    planner, _reader, _sink = _planner(
        provider, sink=sink, config=_config(call_timeout_ms=5_000, cleanup_timeout_ms=25))

    async def main():
        try:
            with pytest.raises(RetrievalError) as info:
                await planner.plan(_observation())
            error = info.value
        finally:
            close = await planner.aclose()
        return error, close

    error, close = asyncio.run(main())
    assert (error.code, error.stage) == ("provider_failed", "provider")
    assert planner.audit_unconverged is True
    # Only the two in-plan records ran; the terminal ``plan_failed``/``plan_closed`` bodies
    # were never started because their window had already expired.
    assert sink.started == 2
    assert _kinds(sink) == ["plan_started", "request_prepared"]
    assert (close.drained, close.pending_calls) == (True, 0)


def test_a_terminal_audit_prepared_inside_the_window_still_converges() -> None:
    """F01 positive control: the same blocking lookup inside a wider window still persists.

    The lookup costs 5 ms and the cleanup window is 200 ms, so the re-read deadline is still in
    the future: the terminal record really runs, nothing is marked unconverged and no window is
    reset.
    """
    sink = BlockingLookupSink(delay=0.005)
    provider = FailingProvider()
    planner, _reader, _sink = _planner(
        provider, sink=sink, config=_config(call_timeout_ms=5_000, cleanup_timeout_ms=200))

    async def main():
        try:
            with pytest.raises(RetrievalError) as info:
                await planner.plan(_observation())
            error = info.value
        finally:
            close = await planner.aclose()
        return error, close

    error, close = asyncio.run(main())
    assert (error.code, error.stage) == ("provider_failed", "provider")
    assert planner.audit_unconverged is False
    assert sink.started == 4
    assert _kinds(sink) == ["plan_started", "request_prepared", "plan_failed", "plan_closed"]
    assert (close.drained, close.pending_calls) == (True, 0)


# --------------------------------------------------------------------------- F04 normalisation

_LEAK = "SYNTHETIC-SENSITIVE-DETAIL-REQUEST"


class ForgingReader(Reader):
    """Legal at construction; forges an ``access_denied`` ``RetrievalError`` at call time."""

    def __init__(self) -> None:
        super().__init__()
        self.forge_at_call = False
        self.forged = False

    @property
    def read(self):
        if self.forge_at_call:
            self.forged = True
            raise RetrievalError("access_denied", "grant", request_id=_LEAK)
        return self._read

    async def _read(self, request):
        return await super().read(request)


class ForgingProvider(Provider):
    """Legal at construction; forges an ``access_denied`` ``RetrievalError`` at call time."""

    def __init__(self) -> None:
        super().__init__()
        self.forge_at_call = False

    @property
    def complete(self):
        if self.forge_at_call:
            raise RetrievalError("access_denied", "grant", request_id=_LEAK)
        return self._complete

    async def _complete(self, request):
        return await super().complete(request)


class ForgingSink(Sink):
    """Legal at construction; forges an ``access_denied`` ``RetrievalError`` on a lookup."""

    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0
        self.forge_from_lookup: int | None = None

    @property
    def append(self):
        self.lookups += 1
        if self.forge_from_lookup is not None and self.lookups >= self.forge_from_lookup:
            raise RetrievalError("access_denied", "grant", request_id=_LEAK)
        return self._append

    async def _append(self, event):
        await super().append(event)


class CancellingReader(Reader):
    """Legal at construction; raises an external ``CancelledError`` at call time."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel_at_call = False

    @property
    def read(self):
        if self.cancel_at_call:
            raise asyncio.CancelledError()
        return self._read

    async def _read(self, request):
        return await super().read(request)


def _read_answer(request):
    return ModelResponse(schema_version=SCHEMA, request_id=request.request_id, status="ok",
                         content=_read_content(request))


def test_a_forged_reader_error_is_normalised_to_reader_failed() -> None:
    """F04: an external ``RetrievalError`` at the reader boundary is not re-raised as itself.

    The reader is legal at construction and raises ``access_denied``/``grant`` with a synthetic
    request identifier when ``read`` is evaluated. The plan must report the bounded
    ``reader_failed``/``reader`` code under the planner's own request identity and must not echo
    the facility text.
    """
    provider = Provider(answer=_read_answer)
    reader = ForgingReader()
    planner, reader, _sink = _planner(provider, reader=reader)
    reader.forge_at_call = True

    async def main():
        try:
            with pytest.raises(RetrievalError) as info:
                await planner.plan(_observation())
            return info.value
        finally:
            await planner.aclose()

    error = asyncio.run(main())
    assert reader.forged is True
    assert (error.code, error.stage) == ("reader_failed", "reader")
    assert error.request_id == provider.requests[-1].request_id
    assert _LEAK not in str(error)
    assert _LEAK not in repr(error)


def test_a_forged_provider_error_is_normalised_to_provider_failed() -> None:
    """F04: the same normalisation applies at the provider boundary, not only the reader."""
    provider = ForgingProvider()
    planner, _reader, sink = _planner(provider)
    provider.forge_at_call = True

    async def main():
        try:
            with pytest.raises(RetrievalError) as info:
                await planner.plan(_observation())
            return info.value
        finally:
            await planner.aclose()

    error = asyncio.run(main())
    assert (error.code, error.stage) == ("provider_failed", "provider")
    prepared = [event for event in sink.events if event.kind == "request_prepared"]
    assert error.request_id == prepared[-1].request_id
    assert _LEAK not in str(error)
    assert _LEAK not in repr(error)


def test_a_forged_audit_error_is_normalised_to_audit_failed() -> None:
    """F04: a forged ``RetrievalError`` raised by the audit lookup is an audit failure.

    Only the second lookup forges, so the failure lands on the ``request_prepared`` record and
    the provider is never called. The forged ``access_denied`` classification and its synthetic
    request identifier must not reach the caller.
    """
    provider = Provider()
    sink = ForgingSink()
    planner, _reader, sink = _planner(provider, sink=sink)
    sink.forge_from_lookup = 3

    async def main():
        try:
            with pytest.raises(RetrievalError) as info:
                await planner.plan(_observation())
            return info.value
        finally:
            await planner.aclose()

    error = asyncio.run(main())
    assert (error.code, error.stage) == ("audit_failed", "audit")
    assert error.request_id is not None
    assert error.request_id != _LEAK
    assert provider.calls == 0
    assert _kinds(sink) == ["plan_started"]
    assert planner.audit_unconverged is True
    assert _LEAK not in str(error)


def test_a_synchronous_external_cancellation_keeps_its_identity() -> None:
    """F04: normalising external ``RetrievalError`` must not swallow a ``CancelledError``.

    The reader raises ``CancelledError`` while its async ``read`` is being evaluated. That is a
    real cancellation, not a facility failure: it reaches the host unchanged, a cancellation
    record is attempted inside the cleanup window and the close reports a clean drain.
    """
    provider = Provider(answer=_read_answer)
    reader = CancellingReader()
    planner, reader, sink = _planner(provider, reader=reader)
    reader.cancel_at_call = True

    async def main():
        try:
            with pytest.raises(asyncio.CancelledError):
                await planner.plan(_observation())
        finally:
            close = await planner.aclose()
        return close

    close = asyncio.run(main())
    assert "plan_cancelled" in _kinds(sink)
    assert (close.drained, close.pending_calls) == (True, 0)


def test_an_internal_grant_refusal_is_not_rewritten_as_a_reader_failure() -> None:
    """F04 negative control: the planner's own refusal keeps its own classification.

    An ungranted evidence identifier is refused by the planner before any external call; the
    reader normalisation must not turn that internal ``access_denied``/``grant`` into
    ``reader_failed``, and the reader is still never called.
    """
    provider = Provider(answer=lambda request: ModelResponse(
        schema_version=SCHEMA, request_id=request.request_id, status="ok",
        content=json.dumps({
            "schema_version": SCHEMA, "run_id": request.run_id, "plan_id": request.plan_id,
            "turn": request.turn, "request_id": request.request_id,
            "kind": "read_evidence",
            "read": {"evidence_id": "not-granted", "view_id": "status",
                     "text_range": None}})))
    reader = Reader()
    planner, reader, _sink = _planner(provider, reader=reader)

    async def main():
        try:
            with pytest.raises(RetrievalError) as info:
                await planner.plan(_observation())
            return info.value
        finally:
            await planner.aclose()

    error = asyncio.run(main())
    assert (error.code, error.stage) == ("access_denied", "grant")
    assert reader.reads == []

