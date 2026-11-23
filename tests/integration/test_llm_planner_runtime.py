"""Acceptance tests for the offline LLMPlanner through the real Core Runtime.

The suite drives the real Fake adapter, the real Contract Builder, the real StepVerifier and
TaskCompletionOracle, the real RecoveryEngine and the real Runtime over a deterministic
in-process provider. The host combines the same RunRequest and PlannerContext; no source
directory is injected into an installed consumer here.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_SRC = REPO_ROOT / "packages" / "planners" / "llm" / "src"
INTEGRATION_DIR = Path(__file__).resolve().parent
for candidate in (LLM_SRC, INTEGRATION_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from conftest import (
    FLAG,
    DeclaredAdapter,
    HostDomain,
    action,
    build_engine,
    failed,
    fake_scenario,
    json_rule_condition,
    request_for,
)
from forecontract.models import AdapterCapabilities, ExpectedOutcome
from forecontract.runtime.errors import RuntimeError as RuntimeFailure
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_fake import FakeEnvironmentAdapter
from forecontract_llm import (
    LLMPlanner,
    ModelRequest,
    ModelResponse,
    PlannerConfig,
    PlannerContext,
)

SCHEMA = "0.1"


def planner_context(*, capabilities: AdapterCapabilities,
                    criteria: TaskCompletionCriteria,
                    run_id: str = "run-1", goal_id: str = "goal-1") -> PlannerContext:
    return PlannerContext(
        schema_version=SCHEMA,
        run_id=run_id,
        goal_id=goal_id,
        goal_text="finish the synthetic task",
        task_criteria=criteria,
        capabilities=capabilities,
        action_guidance={
            name: {"payload": {"type": "object"}}
            for name in capabilities.supported_actions
        },
        outcome_guidance={
            name: {"payload": {"type": "object"}}
            for name in capabilities.supported_outcome_types
        },
    )


def planner_for(provider: object, *, adapter: DeclaredAdapter,
                request) -> LLMPlanner:
    return LLMPlanner(
        provider=provider,
        context=planner_context(capabilities=adapter.capabilities(),
                                criteria=request.task_criteria,
                                run_id=request.run_id, goal_id=request.goal_id),
        config=PlannerConfig(schema_version=SCHEMA,
                             call_timeout_ms=5000, cleanup_timeout_ms=100),
    )


def proposal(action_spec, *, outcomes: list[ExpectedOutcome] | None = None,
             hint: bool | None = None) -> str:
    body = {
        "schema_version": SCHEMA,
        "proposed_action": action_spec.model_dump(mode="json"),
        "expected_outcomes": [
            item.model_dump(mode="json")
            for item in (outcomes if outcomes is not None else [json_rule_condition()])
        ],
    }
    if hint is not None:
        body["task_completion_hint"] = hint
    return json.dumps(body)


class ContentProvider:
    """A deterministic in-process provider that records the request it received."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            schema_version=SCHEMA, request_id=request.request_id, status="ok",
            content=self._content,
        )


def test_the_real_core_engine_runs_a_forecast_from_the_offline_provider(workdir) -> None:
    """C01/C07: one real Runtime Run completes through the plan -> build -> execute loop."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False},
                             end_state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for()
    provider = ContentProvider(proposal(done, hint=True))
    planner = planner_for(provider, adapter=adapter, request=request)
    engine, trace, _ = build_engine(tmp_path=workdir, adapter=adapter, planner=planner,
                                    domain=HostDomain())

    async def main():
        try:
            return await engine.run(request=request)
        finally:
            return_value = await planner.aclose()
            assert return_value.drained is True

    result = asyncio.run(main())

    assert (result.terminal_status.value, result.reason_code.value) == (
        "completed", "TASK_ACHIEVED")
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (1, 1, 1)
    assert [item.action_type for item in adapter.actions] == ["mark_done"]
    assert len(provider.requests) == 1
    assert planner.calls_used == 1
    assert planner.state == "closed"
    records = trace.read_run("run-1")
    assert [record.kind for record in records].count("contract_committed") == 1
    assert inspect.iscoroutinefunction(LLMPlanner.plan)
    assert list(inspect.signature(LLMPlanner.plan).parameters) == ["self", "observation"]


def test_the_host_context_and_the_run_request_use_the_same_inputs(workdir) -> None:
    """C01: the host supplies one RunRequest and one matching PlannerContext."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False},
                             end_state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for(run_id="run-42", goal_id="goal-42")
    provider = ContentProvider(proposal(done))
    planner = planner_for(provider, adapter=adapter, request=request)
    engine, _trace, _ = build_engine(tmp_path=workdir, adapter=adapter, planner=planner,
                                     domain=HostDomain())

    async def main():
        result = await engine.run(request=request)
        await planner.aclose()
        return result

    result = asyncio.run(main())
    assert result.run_id == "run-42"
    seen = provider.requests[0]
    assert seen.model_context.run_id == "run-42"
    assert seen.model_context.goal_id == "goal-42"
    assert seen.model_context.task_criteria.model_dump(mode="json") == \
        request.task_criteria.model_dump(mode="json")
    assert seen.observation.observation_id == "obs-before"
    assert seen.observation.state == {FLAG: False}


def test_a_shorter_run_deadline_wins_over_a_longer_planner_deadline(workdir) -> None:
    """C07: the Core Run deadline can be shorter and really cancels the provider wait."""

    class NeverAnswering:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            await asyncio.sleep(5)
            return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                                 status="ok", content="{}")

    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False},
                             end_state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for()
    provider = NeverAnswering()
    planner = LLMPlanner(
        provider=provider,
        context=planner_context(capabilities=adapter.capabilities(),
                                criteria=request.task_criteria),
        config=PlannerConfig(schema_version=SCHEMA, call_timeout_ms=5000,
                             cleanup_timeout_ms=100),
    )
    engine, _trace, _ = build_engine(tmp_path=workdir, adapter=adapter, planner=planner,
                                     domain=HostDomain(), run_timeout_ms=40.0)

    async def main():
        try:
            return await engine.run(request=request)
        finally:
            await planner.aclose()

    result = asyncio.run(main())
    assert (result.terminal_status.value, result.reason_code.value) == (
        "timeout", "DEADLINE_EXCEEDED")
    assert adapter.actions == []
    assert provider.calls == 1
    assert planner.calls_used == 1
    assert planner.state == "closed"


def test_a_planner_hint_cannot_make_a_failing_goal_succeed(workdir) -> None:
    """C03/C07: the hint is a proposal; the Core oracle still decides."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False},
                             end_state={FLAG: False}, outcome=failed("ao-1"))
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for(max_steps=1, max_planning_calls=1)
    provider = ContentProvider(proposal(done, hint=True))
    planner = planner_for(provider, adapter=adapter, request=request)
    engine, _trace, _ = build_engine(tmp_path=workdir, adapter=adapter, planner=planner,
                                     domain=HostDomain(effect="applied"))

    async def main():
        try:
            return await engine.run(request=request)
        finally:
            await planner.aclose()

    result = asyncio.run(main())
    assert result.terminal_status.value != "completed"
    assert result.reason_code.value != "TASK_ACHIEVED"
    assert result.attempts_started == 1
    assert [item.action_type for item in adapter.actions] == ["mark_done"]


def test_a_builder_rejection_leaves_zero_execution_side_effects(workdir) -> None:
    """C01/C03: a proposal still has to pass the Builder; a rejection executes nothing."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False},
                             end_state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for(max_steps=1, max_planning_calls=1)
    outside_catalog = action(payload={"selector": "#not-in-the-catalog"})
    provider = ContentProvider(proposal(outside_catalog))
    planner = planner_for(provider, adapter=adapter, request=request)
    engine, trace, _ = build_engine(tmp_path=workdir, adapter=adapter, planner=planner,
                                    domain=HostDomain())

    async def main():
        try:
            return await engine.run(request=request)
        finally:
            await planner.aclose()

    result = asyncio.run(main())
    assert (result.terminal_status.value, result.reason_code.value) == (
        "failed", "EVIDENCE_INVALID")
    assert (result.steps_started, result.attempts_started) == (0, 0)
    assert adapter.actions == []
    assert (adapter.starts, adapter.stops) == (1, 1)
    assert [record.kind for record in trace.read_run("run-1")] == [
        "run_started", "run_finished"]
    assert planner.calls_used == 1


def test_a_malformed_answer_is_a_planning_facility_failure_with_no_effect(workdir) -> None:
    """C05: a malformed model answer aborts planning; no Attempt and no execution follow."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False},
                             end_state={FLAG: True})
    adapter = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for()
    provider = ContentProvider("this is not the declared JSON object")
    planner = planner_for(provider, adapter=adapter, request=request)
    engine, _trace, _ = build_engine(tmp_path=workdir, adapter=adapter, planner=planner,
                                     domain=HostDomain())

    async def main():
        try:
            with pytest.raises(RuntimeFailure) as info:
                await engine.run(request=request)
            return info.value
        finally:
            close = await planner.aclose()
            assert close.drained is True

    error = asyncio.run(main())
    assert (error.code, error.stage) == ("DEPENDENCY_FAILURE", "plan")
    assert adapter.actions == []
    assert (adapter.starts, adapter.stops) == (1, 1)
    assert (planner.calls_used, planner.state) == (1, "closed")
