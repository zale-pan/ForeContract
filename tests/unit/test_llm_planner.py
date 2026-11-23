"""Acceptance tests for the decision behaviour of the offline LLMPlanner.

The planner is driven through its real public ``plan`` method with injected deterministic
in-process providers. The cases prove that it makes exactly one bounded call, charges exactly
once, binds each answer to the current request, never coerces an unknown or malformed answer
into an action, and keeps the trusted context in its own storage.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
LLM_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "llm" / "src"
for candidate in (CORE_SRC, LLM_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.models import (
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
    Observation,
)
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm import (
    LLMPlanner,
    ModelRequest,
    ModelResponse,
    PlannerConfig,
    PlannerContext,
    PlannerError,
)

SCHEMA = "0.1"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
SECRET = "SYNTHETIC-PRIVATE-PROVIDER-MARKER"


def capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        schema_version=SCHEMA,
        adapter_name="synthetic",
        environment_type="fake",
        supported_actions=["mark_done"],
        supported_outcome_types=["json_rule"],
    )


def context(**overrides: object) -> PlannerContext:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "run_id": "run-1",
        "goal_id": "goal-1",
        "goal_text": "finish the synthetic task",
        "task_criteria": TaskCompletionCriteria(
            schema_version=SCHEMA, goal_id="goal-1",
            conditions=[ExpectedOutcome(schema_version=SCHEMA, outcome_type="json_rule")],
        ),
        "capabilities": capabilities(),
        "action_guidance": {"mark_done": {"payload": {"type": "object"}}},
        "outcome_guidance": {"json_rule": {"payload": {"type": "object"}}},
    }
    values.update(overrides)
    return PlannerContext(**values)


def config(**overrides: object) -> PlannerConfig:
    values: dict[str, object] = {"schema_version": SCHEMA}
    values.update(overrides)
    return PlannerConfig(**values)


def observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "observation_id": "obs-1",
        "environment_type": "fake",
        "state": {"done": False},
        "timestamp": NOW,
    }
    values.update(overrides)
    return Observation(**values)


def body(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "proposed_action": {
            "schema_version": SCHEMA,
            "action_type": "mark_done",
            "payload": {"selector": "#go"},
        },
        "expected_outcomes": [
            {"schema_version": SCHEMA, "outcome_type": "json_rule",
             "payload": {"rule_type": "value_equals", "path": ["done"]}},
        ],
    }
    values.update(overrides)
    return values


def content(**overrides: object) -> str:
    return json.dumps(body(**overrides))


class RecordingProvider:
    """A deterministic in-process provider that records the exact request it received."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self._responder(request)


def respond_ok(text: str | None = None):
    def responder(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            schema_version=SCHEMA, request_id=request.request_id, status="ok",
            content=content() if text is None else text,
        )
    return responder


def build_planner(*, responder=None, planner_config: PlannerConfig | None = None,
                  planner_context: PlannerContext | None = None) -> tuple[LLMPlanner, RecordingProvider]:
    provider = RecordingProvider(responder if responder is not None else respond_ok())
    planner = LLMPlanner(
        provider=provider,
        context=planner_context if planner_context is not None else context(),
        config=planner_config if planner_config is not None else config(),
    )
    return planner, provider


def test_a_normal_plan_returns_one_core_forecast() -> None:
    """C01/C03: one call produces one proposal bound to the current request."""
    planner, provider = build_planner()
    forecast = asyncio.run(planner.plan(observation()))

    assert isinstance(forecast, Forecast)
    assert forecast.proposed_action.action_type == "mark_done"
    assert forecast.metadata["planner"] == "llm"
    assert forecast.metadata["run_id"] == "run-1"
    assert forecast.metadata["goal_id"] == "goal-1"
    assert forecast.metadata["observation_id"] == "obs-1"
    assert forecast.metadata["request_id"] == forecast.forecast_id
    assert (planner.calls_used, planner.state) == (1, "ready")
    assert len(provider.requests) == 1
    assert asyncio.run(planner.aclose()).drained is True


def test_the_planner_generates_a_fresh_uuid_request_identifier_per_call() -> None:
    """C02: the identifier is planner-generated, not chosen by the response."""
    planner, provider = build_planner()
    first = asyncio.run(planner.plan(observation()))
    second = asyncio.run(planner.plan(observation(state={"done": True})))
    assert first.forecast_id != second.forecast_id
    uuid.UUID(first.forecast_id)
    uuid.UUID(second.forecast_id)
    assert [item.request_id for item in provider.requests] == [
        first.forecast_id, second.forecast_id
    ]
    assert planner.calls_used == 2


def test_the_provider_receives_the_complete_observation_and_trusted_context() -> None:
    """C03: the request carries the full state and the host-supplied context."""
    planner, provider = build_planner()
    seen = observation(state={"done": False, "nested": {"count": 2}})
    asyncio.run(planner.plan(seen))
    request = provider.requests[0]
    assert request.observation.state == {"done": False, "nested": {"count": 2}}
    assert request.model_context.run_id == "run-1"
    assert request.model_context.goal_id == "goal-1"
    assert request.model_context.task_criteria.goal_id == "goal-1"
    assert request.model_context.capabilities.supported_actions == ["mark_done"]


def test_an_environment_mismatch_is_refused_before_the_provider_call() -> None:
    """C03: a mismatch is an input/context error, not a provider opportunity."""
    planner, provider = build_planner()
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation(environment_type="other")))
    assert info.value.code == "CONTEXT_MISMATCH"
    assert provider.requests == []
    assert (planner.calls_used, planner.state) == (0, "ready")


def test_an_invalid_observation_is_refused_before_the_provider_call() -> None:
    """C05: an invalid observation never consumes a call and never reaches the provider."""
    planner, provider = build_planner()
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(object()))  # type: ignore[arg-type]
    assert info.value.code == "INPUT_INVALID"
    assert provider.requests == []
    assert planner.calls_used == 0


def test_a_refusal_is_the_fixed_provider_refused_error() -> None:
    """C05: a refusal is a facility failure, not a fabricated plan."""
    def responder(request: ModelRequest) -> ModelResponse:
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="refused", content=None)

    planner, provider = build_planner(responder=responder)
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "PROVIDER_REFUSED"
    assert len(provider.requests) == 1
    assert (planner.calls_used, planner.state) == (1, "ready")


def test_a_malformed_answer_is_the_fixed_response_error() -> None:
    """C03: malformed content never becomes an action and is charged exactly once."""
    planner, provider = build_planner(responder=respond_ok("not json"))
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "RESPONSE_INVALID"
    assert len(provider.requests) == 1
    assert (planner.calls_used, planner.state) == (1, "ready")


def test_a_provider_exception_is_a_bounded_facility_error() -> None:
    """C05: the provider text is not echoed and no cause is exposed."""
    def responder(request: ModelRequest) -> ModelResponse:
        raise RuntimeError(f"leaked {SECRET}")

    planner, provider = build_planner(responder=responder)
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    error = info.value
    assert error.code == "PROVIDER_FAILURE"
    assert SECRET not in str(error)
    assert SECRET not in repr(error)
    assert error.__cause__ is None
    assert len(provider.requests) == 1
    assert (planner.calls_used, planner.state) == (1, "ready")


def test_a_provider_descriptor_that_raises_is_a_safe_input_error() -> None:
    """F04: a raising ``complete`` descriptor is bounded at construction, text-free."""
    class HostileProvider:
        @property
        def complete(self) -> object:
            raise ValueError(f"leaked {SECRET}")

    with pytest.raises(PlannerError) as info:
        LLMPlanner(provider=HostileProvider(), context=context(), config=config())
    error = info.value
    assert error.code == "INPUT_INVALID"
    assert SECRET not in str(error)
    assert SECRET not in repr(error)
    assert error.__cause__ is None


def test_a_provider_raised_planner_error_is_a_safe_provider_failure() -> None:
    """F04: a provider cannot forge a planner code or smuggle its own message out."""
    class HostileProvider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise PlannerError("UNSUPPORTED_ACTION", f"leaked {SECRET}",
                               location=("provider",))

    planner = LLMPlanner(provider=HostileProvider(), context=context(), config=config())
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    error = info.value
    assert error.code == "PROVIDER_FAILURE"
    assert SECRET not in str(error)
    assert SECRET not in repr(error)
    assert error.__cause__ is None
    assert (planner.calls_used, planner.state) == (1, "ready")
    assert asyncio.run(planner.aclose()).drained is True


def test_a_provider_that_returns_a_non_record_is_a_protocol_failure() -> None:
    """C05: a transport that breaks the declared interface is a provider failure."""
    planner, _provider = build_planner(responder=lambda request: {"content": content()})
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "PROVIDER_FAILURE"
    assert planner.calls_used == 1


def test_the_call_allowance_is_consumed_once_and_then_exhausted() -> None:
    """C04: a call is charged once, and an exhausted allowance calls no provider."""
    planner, provider = build_planner(planner_config=config(max_calls=2))
    asyncio.run(planner.plan(observation()))
    asyncio.run(planner.plan(observation()))
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "BUDGET_EXCEEDED"
    assert len(provider.requests) == 2
    assert planner.calls_used == 2


def test_a_failed_call_is_never_retried() -> None:
    """C04: one plan means at most one provider call, with no hidden retry."""
    calls: list[str] = []

    def responder(request: ModelRequest) -> ModelResponse:
        calls.append(request.request_id)
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="ok", content="{}")

    planner, _provider = build_planner(responder=responder)
    with pytest.raises(PlannerError):
        asyncio.run(planner.plan(observation()))
    assert len(calls) == 1
    assert planner.calls_used == 1


def test_a_declared_output_token_overrun_is_refused() -> None:
    """C04: a declared overrun is a budget failure and the call is still charged."""
    def responder(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            schema_version=SCHEMA, request_id=request.request_id, status="ok",
            content=content(),
            usage={"schema_version": SCHEMA, "input_tokens": 1, "output_tokens": 99},
        )

    planner, _provider = build_planner(
        responder=responder, planner_config=config(max_output_tokens=10)
    )
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "BUDGET_EXCEEDED"
    assert planner.calls_used == 1


def test_an_out_of_capability_action_is_refused_not_coerced() -> None:
    """C03: a well-formed but undeclared action is refused as an unsupported action."""
    planner, _provider = build_planner(
        responder=respond_ok(content(proposed_action={
            "schema_version": SCHEMA, "action_type": "delete_everything"}))
    )
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "UNSUPPORTED_ACTION"


def test_an_out_of_capability_outcome_is_refused_not_coerced() -> None:
    """C03: an undeclared outcome type is refused as an unsupported outcome."""
    planner, _provider = build_planner(
        responder=respond_ok(content(expected_outcomes=[
            {"schema_version": SCHEMA, "outcome_type": "invented"}]))
    )
    with pytest.raises(PlannerError) as info:
        asyncio.run(planner.plan(observation()))
    assert info.value.code == "UNSUPPORTED_OUTCOME"


def test_a_provider_mutation_of_its_request_cannot_change_the_decoded_forecast() -> None:
    """C02: the provider is handed a detached copy; the binding uses the trusted request."""
    def responder(request: ModelRequest) -> ModelResponse:
        original = request.request_id
        request.__dict__["request_id"] = "mutated"
        return ModelResponse(schema_version=SCHEMA, request_id=original, status="ok",
                             content=content())

    planner, _provider = build_planner(responder=responder)
    forecast = asyncio.run(planner.plan(observation()))
    assert forecast.forecast_id != "mutated"
    assert forecast.metadata["request_id"] == forecast.forecast_id
    assert planner.calls_used == 1


def test_the_planner_keeps_its_own_copy_of_the_trusted_context() -> None:
    """C02: a later caller mutation of the context does not rewrite the planner's input."""
    planner_context = context()
    planner, provider = build_planner(planner_context=planner_context)
    planner_context.goal_id = "goal-2"
    planner_context.goal_text = "changed"
    forecast = asyncio.run(planner.plan(observation()))
    assert forecast.metadata["goal_id"] == "goal-1"
    assert provider.requests[0].model_context.goal_id == "goal-1"


def test_a_completion_hint_is_only_carried_as_a_proposal() -> None:
    """C03: the hint is forwarded but the planner claims no completion itself."""
    planner, _provider = build_planner(
        responder=respond_ok(content(task_completion_hint=True, confidence=0.99))
    )
    forecast = asyncio.run(planner.plan(observation()))
    assert forecast.task_completion_hint is True
    assert forecast.confidence == 0.99
    assert forecast.metadata["planner"] == "llm"
    assert planner.state == "ready"


def test_the_planner_uses_no_credential_or_transport_field() -> None:
    """C01: the request vocabulary has no API key, provider URL or tool permission."""
    keys = set(ModelRequest.model_fields)
    assert keys == {
        "schema_version", "request_id", "model_context", "observation", "response_schema",
        "max_output_tokens",
    }
    for forbidden in ("api_key", "url", "headers", "tools", "auth"):
        assert all(forbidden not in name for name in keys)
