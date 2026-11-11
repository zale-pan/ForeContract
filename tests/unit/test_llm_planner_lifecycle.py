"""Acceptance tests for the planner lifecycle, timeout and cancellation semantics.

The tests drive real coroutines with deterministic in-process providers, including providers
that ignore cancellation. They assert the closed state vocabulary, the single charge per call,
the bounded cleanup, the honest close result and the refusal to publish a late answer.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
LLM_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "llm" / "src"
for candidate in (CORE_SRC, LLM_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.models import AdapterCapabilities, ExpectedOutcome, Observation
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm import (
    LLMPlanner,
    ModelRequest,
    ModelResponse,
    PlannerCloseResult,
    PlannerConfig,
    PlannerContext,
    PlannerError,
)

SCHEMA = "0.1"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def context() -> PlannerContext:
    capabilities = AdapterCapabilities(
        schema_version=SCHEMA, adapter_name="synthetic", environment_type="fake",
        supported_actions=["mark_done"], supported_outcome_types=["json_rule"],
    )
    return PlannerContext(
        schema_version=SCHEMA, run_id="run-1", goal_id="goal-1",
        goal_text="finish the synthetic task",
        task_criteria=TaskCompletionCriteria(
            schema_version=SCHEMA, goal_id="goal-1",
            conditions=[ExpectedOutcome(schema_version=SCHEMA, outcome_type="json_rule")],
        ),
        capabilities=capabilities,
        action_guidance={"mark_done": {"payload": {"type": "object"}}},
        outcome_guidance={"json_rule": {"payload": {"type": "object"}}},
    )


def observation() -> Observation:
    return Observation(
        schema_version=SCHEMA, observation_id="obs-1", environment_type="fake",
        state={"done": False}, timestamp=NOW,
    )


def config(**overrides: object) -> PlannerConfig:
    values: dict[str, object] = {"schema_version": SCHEMA}
    values.update(overrides)
    return PlannerConfig(**values)


def answer(request: ModelRequest) -> ModelResponse:
    content = json.dumps({
        "schema_version": SCHEMA,
        "proposed_action": {
            "schema_version": SCHEMA, "action_type": "mark_done", "payload": {},
        },
        "expected_outcomes": [
            {"schema_version": SCHEMA, "outcome_type": "json_rule", "payload": {}},
        ],
    })
    return ModelResponse(schema_version=SCHEMA, request_id=request.request_id, status="ok",
                         content=content)


class SlowProvider:
    """A provider that records its entry and never answers before the planner deadline."""

    def __init__(self) -> None:
        self.calls = 0
        self.cancelled = False
        self.entered = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        self.entered.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return answer(request)


class GatedProvider:
    """A provider that waits until the host releases it."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        await self.gate.wait()
        return answer(request)


class StubbornProvider:
    """A provider that records entry and keeps running until the host explicitly releases it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        self.entered.set()
        while not self.release.is_set():
            try:
                await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                continue
        return answer(request)


class SelfCancellingProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await asyncio.sleep(0.05)
        return answer(request)


def test_initial_state_is_ready_with_zero_calls() -> None:
    """C04: a constructed planner is ready and has spent nothing."""

    class Provider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return answer(request)

    planner = LLMPlanner(provider=Provider(), context=context(), config=config())
    assert (planner.calls_used, planner.state) == (0, "ready")


def test_a_concurrent_plan_is_refused_while_busy() -> None:
    """C04: a second overlapping plan is BUSY immediately and spends no second call."""
    provider = GatedProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=5000, cleanup_timeout_ms=50))

    async def main() -> None:
        first = asyncio.create_task(planner.plan(observation()))
        await asyncio.sleep(0)
        with pytest.raises(PlannerError) as info:
            await planner.plan(observation())
        assert info.value.code == "BUSY"
        assert planner.state == "busy"
        provider.gate.set()
        forecast = await first
        assert forecast.metadata["planner"] == "llm"

    asyncio.run(main())
    assert provider.calls == 1
    assert (planner.calls_used, planner.state) == (1, "ready")
    assert asyncio.run(planner.aclose()).drained is True


def test_a_timeout_is_charged_once_and_leaves_a_drained_ready_planner() -> None:
    """C04/F07: entry is confirmed before the deadline; a settled worker does not poison."""
    provider = SlowProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=200, cleanup_timeout_ms=100))

    async def main() -> None:
        task = asyncio.create_task(planner.plan(observation()))
        await asyncio.wait_for(provider.entered.wait(), 1.0)
        with pytest.raises(PlannerError) as info:
            await task
        assert info.value.code == "TIMEOUT"

    asyncio.run(main())
    assert provider.entered.is_set()
    assert provider.calls == 1
    assert provider.cancelled is True
    assert (planner.calls_used, planner.state) == (1, "ready")
    assert asyncio.run(planner.aclose()).drained is True


def test_a_deadline_expired_during_request_assembly_charges_nothing() -> None:
    """C04/F07: the absolute deadline includes request assembly; no call means no charge.

    This is the legitimate pre-provider expiry branch: the provider is never entered, so the
    planner stays ready with zero calls instead of being poisoned.
    """
    provider = SlowProvider()
    big_state = {f"key-{index}": index for index in range(20000)}
    planner = LLMPlanner(
        provider=provider,
        context=context(),
        config=config(call_timeout_ms=0.5, cleanup_timeout_ms=50,
                      max_json_nodes=1_000_000, max_request_bytes=1_000_000),
    )
    heavy = Observation(
        schema_version=SCHEMA, observation_id="obs-big", environment_type="fake",
        state=big_state, timestamp=NOW,
    )
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(heavy))
    assert info.value.code == "TIMEOUT"
    assert provider.calls == 0
    assert provider.entered.is_set() is False
    assert (planner.calls_used, planner.state) == (0, "ready")


def test_a_non_cooperative_provider_poisons_the_planner() -> None:
    """C04/F07: after confirmed entry, an unsettled timeout poisons and refuses a new call."""
    provider = StubbornProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=200, cleanup_timeout_ms=20))

    async def main() -> None:
        task = asyncio.create_task(planner.plan(observation()))
        await asyncio.wait_for(provider.entered.wait(), 1.0)
        with pytest.raises(PlannerError) as info:
            await task
        assert info.value.code == "TIMEOUT"
        assert provider.calls == 1
        assert planner.state == "poisoned"
        with pytest.raises(PlannerError) as poisoned:
            await planner.plan(observation())
        assert poisoned.value.code == "POISONED"
        # Release the worker so the test loop can finish; the planner is already poisoned.
        provider.release.set()
        await asyncio.sleep(0.05)
        await planner.aclose()

    asyncio.run(main())
    assert provider.entered.is_set()
    assert provider.calls == 1
    assert planner.calls_used == 1


def test_the_cleanup_allowance_bounds_the_wait() -> None:
    """C04/F07: a confirmed non-cooperative task is not awaited without a bound."""
    provider = StubbornProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=200, cleanup_timeout_ms=20))

    async def main() -> None:
        started = time.monotonic()
        task = asyncio.create_task(planner.plan(observation()))
        await asyncio.wait_for(provider.entered.wait(), 1.0)
        with pytest.raises(PlannerError):
            await task
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert planner.state == "poisoned"
        provider.release.set()
        await asyncio.sleep(0.05)
        await planner.aclose()

    asyncio.run(main())
    assert provider.entered.is_set()


def test_an_external_cancellation_keeps_its_type_and_cleans_up() -> None:
    """C04: a host cancellation is not wrapped; the call is charged once and the task stops."""
    provider = SlowProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=5000, cleanup_timeout_ms=100))

    async def main() -> None:
        task = asyncio.create_task(planner.plan(observation()))
        await asyncio.wait_for(provider.entered.wait(), 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert planner.state == "ready"
        assert provider.cancelled is True
        result = await planner.aclose()
        assert result.drained is True

    asyncio.run(main())
    assert provider.calls == 1
    assert planner.calls_used == 1


def test_an_observed_external_cancel_wins_over_a_close_that_follows() -> None:
    """F03: a cancellation observed first keeps its identity even when aclose follows."""
    async def main() -> None:
        entered = asyncio.Event()
        unwinding = asyncio.Event()
        release = asyncio.Event()

        class Provider:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    unwinding.set()
                    while not release.is_set():
                        try:
                            await release.wait()
                        except asyncio.CancelledError:
                            continue
                    raise
                return answer(request)

        planner = LLMPlanner(provider=Provider(), context=context(),
                             config=config(call_timeout_ms=5000, cleanup_timeout_ms=20))
        plan_task = asyncio.create_task(planner.plan(observation()))
        await asyncio.wait_for(entered.wait(), 1.0)
        plan_task.cancel("synthetic-outer-cancel")
        await asyncio.wait_for(unwinding.wait(), 1.0)
        close_task = asyncio.create_task(planner.aclose())
        try:
            with pytest.raises(asyncio.CancelledError):
                await plan_task
            report = await asyncio.wait_for(close_task, 1.0)
        finally:
            release.set()
            await asyncio.sleep(0.02)
            await planner.aclose()
        assert (report.drained, report.pending_calls) == (False, 1)
        assert planner.state == "closed"

    asyncio.run(main())


def test_an_external_cancel_during_close_cleanup_keeps_its_cancellation() -> None:
    """F03: a host cancel observed while close-owned cleanup drains keeps CancelledError."""
    async def main() -> None:
        entered = asyncio.Event()
        unwinding = asyncio.Event()
        release = asyncio.Event()

        class Provider:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    unwinding.set()
                    while not release.is_set():
                        try:
                            await release.wait()
                        except asyncio.CancelledError:
                            continue
                    raise
                return answer(request)

        planner = LLMPlanner(provider=Provider(), context=context(),
                             config=config(call_timeout_ms=5000, cleanup_timeout_ms=100))
        plan_task = asyncio.create_task(planner.plan(observation()))
        await asyncio.wait_for(entered.wait(), 1.0)
        close_task = asyncio.create_task(planner.aclose())
        await asyncio.wait_for(unwinding.wait(), 1.0)
        await asyncio.sleep(0)
        assert plan_task.cancelling() == 1
        plan_task.cancel("external-during-close-cleanup")
        try:
            with pytest.raises(asyncio.CancelledError):
                await plan_task
            report = await asyncio.wait_for(close_task, 1.0)
        finally:
            release.set()
            await asyncio.sleep(0.02)
            await planner.aclose()
        assert plan_task.cancelling() == 2
        assert planner.state == "closed"
        assert planner.calls_used == 1
        assert (report.drained, report.pending_calls) == (False, 1)

    asyncio.run(main())


def test_a_self_cancelled_provider_is_a_provider_failure() -> None:
    """C04: a provider that cancels itself is a failure, not a host cancellation."""
    planner = LLMPlanner(provider=SelfCancellingProvider(), context=context(),
                         config=config(call_timeout_ms=5000, cleanup_timeout_ms=100))
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "PROVIDER_FAILURE"
    assert (planner.calls_used, planner.state) == (1, "ready")


def test_a_system_exit_from_the_provider_is_not_swallowed() -> None:
    """C04: SystemExit keeps its identity; the planner returns to ready and spends one call."""

    class Exiting:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise SystemExit("synthetic exit")

    planner = LLMPlanner(provider=Exiting(), context=context(), config=config())
    with pytest.raises(SystemExit):
        asyncio.run(planner.plan(observation()))
    assert (planner.calls_used, planner.state) == (1, "ready")


def test_aclose_on_an_idle_planner_is_idempotent() -> None:
    """C04: closing an idle planner is clean, closed forever and idempotent."""

    class Provider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return answer(request)

    planner = LLMPlanner(provider=Provider(), context=context(), config=config())

    async def main() -> None:
        first = await planner.aclose()
        second = await planner.aclose()
        assert isinstance(first, PlannerCloseResult)
        assert (first.drained, first.pending_calls) == (True, 0)
        assert (second.drained, second.pending_calls) == (True, 0)
        assert planner.state == "closed"

    asyncio.run(main())


def test_a_closed_planner_refuses_a_new_plan() -> None:
    """C04: closed is permanent and is checked before any input validation."""

    class Provider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return answer(request)

    planner = LLMPlanner(provider=Provider(), context=context(), config=config())
    asyncio.run(planner.aclose())
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "CLOSED"
    assert planner.calls_used == 0


def test_aclose_during_a_call_ends_the_plan_as_closed() -> None:
    """C04: an in-flight plan ends CLOSED; no forecast is published after the close."""
    provider = GatedProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=5000, cleanup_timeout_ms=20))

    async def main() -> None:
        task = asyncio.create_task(planner.plan(observation()))
        await asyncio.sleep(0.01)
        result = await planner.aclose()
        assert planner.state == "closed"
        with pytest.raises(PlannerError) as info:
            await task
        assert info.value.code == "CLOSED"
        assert result.pending_calls == 0
        assert result.drained is True

    asyncio.run(main())
    assert planner.calls_used == 1
    assert provider.calls == 1


def test_a_late_answer_after_close_is_never_published() -> None:
    """C04: a provider that ignores cancellation cannot publish a forecast after close."""
    provider = StubbornProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=5000, cleanup_timeout_ms=10))

    async def main() -> None:
        task = asyncio.create_task(planner.plan(observation()))
        await asyncio.sleep(0.01)
        first_close = await planner.aclose()
        assert (first_close.drained, first_close.pending_calls) == (False, 1)
        with pytest.raises(PlannerError) as info:
            await task
        assert info.value.code == "CLOSED"
        provider.release.set()
        await asyncio.sleep(0.05)
        second_close = await planner.aclose()
        assert second_close.drained is True
        assert planner.state == "closed"

    asyncio.run(main())
    assert planner.calls_used == 1


def test_aclose_reports_an_unsettled_call_then_drains_it() -> None:
    """C04/F07: pending is reported honestly and a later close observes the real settlement."""
    provider = StubbornProvider()
    planner = LLMPlanner(provider=provider, context=context(),
                         config=config(call_timeout_ms=200, cleanup_timeout_ms=10))

    async def main() -> None:
        task = asyncio.create_task(planner.plan(observation()))
        await asyncio.wait_for(provider.entered.wait(), 1.0)
        with pytest.raises(PlannerError):
            await task
        assert planner.state == "poisoned"
        first = await planner.aclose()
        assert (first.drained, first.pending_calls) == (False, 1)
        provider.release.set()
        await asyncio.sleep(0.05)
        second = await planner.aclose()
        assert (second.drained, second.pending_calls) == (True, 0)

    asyncio.run(main())
    assert planner.calls_used == 1


def test_calls_used_never_decrements_across_failures() -> None:
    """C04: refusals, malformed answers and successes all stay counted."""
    answers = iter([
        ModelResponse(schema_version=SCHEMA, request_id="req-x", status="refused", content=None),
        None,
    ])
    seen: list[int] = []

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            item = next(answers)
            if item is None:
                return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                                     status="ok", content="not json")
            return item

    planner = LLMPlanner(provider=Provider(), context=context(), config=config())
    with pytest.raises(PlannerError):
        asyncio.run(planner.plan(observation()))
    seen.append(planner.calls_used)
    with pytest.raises(PlannerError):
        asyncio.run(planner.plan(observation()))
    seen.append(planner.calls_used)
    assert seen == [1, 2]
    assert planner.state == "ready"
