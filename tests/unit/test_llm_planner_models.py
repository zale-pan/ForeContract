"""Acceptance tests for the strict records of the offline LLM planner.

Every fixture is synthetic and embedded here. The tests build real records and assert the
declared invariants rather than the mere absence of exceptions: the closed root surface, the
explicit schema version, strict scalars, the guidance/capability binding, the response
status/content verdict, the close-result consistency, and the refusal of a forged or mutated
record at the public planner entry.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
LLM_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "llm" / "src"
for candidate in (CORE_SRC, LLM_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import forecontract_llm
from forecontract.models import (
    AdapterCapabilities,
    ExpectedOutcome,
    Observation,
)
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm import (
    LLMPlanner,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PlannerCloseResult,
    PlannerConfig,
    PlannerContext,
    PlannerError,
)
from forecontract_llm.codec import build_request
from forecontract_llm.provider import OfflineProvider

SCHEMA = "0.1"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)

EXPECTED_ROOT_SURFACE = [
    "LLMPlanner",
    "ModelProvider",
    "PlannerContext",
    "PlannerConfig",
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
    "PlannerCloseResult",
    "PlannerError",
]


def capabilities(
    *,
    actions: tuple[str, ...] = ("mark_done",),
    outcomes: tuple[str, ...] = ("json_rule",),
    environment_type: str = "fake",
) -> AdapterCapabilities:
    return AdapterCapabilities(
        schema_version=SCHEMA,
        adapter_name="synthetic",
        environment_type=environment_type,
        supported_actions=list(actions),
        supported_outcome_types=list(outcomes),
    )


def criteria(goal_id: str = "goal-1") -> TaskCompletionCriteria:
    return TaskCompletionCriteria(
        schema_version=SCHEMA,
        goal_id=goal_id,
        conditions=[ExpectedOutcome(schema_version=SCHEMA, outcome_type="json_rule")],
    )


def context(**overrides: object) -> PlannerContext:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "run_id": "run-1",
        "goal_id": "goal-1",
        "goal_text": "finish the synthetic task",
        "task_criteria": criteria(),
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


def usage(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "input_tokens": 3,
        "output_tokens": 4,
    }
    values.update(overrides)
    return values


def response(**overrides: object) -> ModelResponse:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "request_id": "req-1",
        "status": "ok",
        "content": "{}",
    }
    values.update(overrides)
    return ModelResponse(**values)


def request(**overrides: object) -> ModelRequest:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "request_id": "req-1",
        "model_context": context(),
        "observation": observation(),
        "response_schema": {"type": "object"},
        "max_output_tokens": 64,
    }
    values.update(overrides)
    return ModelRequest(**values)


class _AsyncProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return response(request_id=request.request_id)


class _SyncProvider:
    def complete(self, request: ModelRequest) -> ModelResponse:
        return response(request_id=request.request_id)


def test_the_root_exports_are_exactly_the_declared_public_surface() -> None:
    """C02: the root package exports exactly the nine agreed names."""
    assert set(forecontract_llm.__all__) == set(EXPECTED_ROOT_SURFACE)
    assert forecontract_llm.__all__ == sorted(EXPECTED_ROOT_SURFACE)
    for name in EXPECTED_ROOT_SURFACE:
        assert hasattr(forecontract_llm, name)
    public = {
        name
        for name in dir(forecontract_llm)
        if name != "annotations"
        and not name.startswith("_")
        and not isinstance(getattr(forecontract_llm, name), types.ModuleType)
    }
    assert public == set(EXPECTED_ROOT_SURFACE)
    # The codec helpers are deliberately not part of the root export set.
    assert "build_request" not in forecontract_llm.__all__
    assert "decode_response" not in forecontract_llm.__all__


def test_every_public_record_requires_the_explicit_schema_version() -> None:
    """C02: no record can be built without the explicit version literal."""
    cases = [
        lambda: PlannerContext(
            run_id="run-1", goal_id="goal-1", goal_text="text", task_criteria=criteria(),
            capabilities=capabilities(), action_guidance={"mark_done": {"x": 1}},
            outcome_guidance={"json_rule": {"x": 1}},
        ),
        lambda: PlannerConfig(),
        lambda: ModelUsage(input_tokens=1, output_tokens=1),
        lambda: ModelRequest(
            request_id="req-1", model_context=context(), observation=observation(),
            response_schema={}, max_output_tokens=1,
        ),
        lambda: ModelResponse(request_id="req-1", status="ok", content="{}"),
        lambda: PlannerCloseResult(drained=True, pending_calls=0),
    ]
    for build in cases:
        with pytest.raises(ValidationError):
            build()


@pytest.mark.parametrize("version", ["1.0", "0.10", "0.2", "", 0.1, True, None])
def test_an_unknown_schema_version_is_refused(version: object) -> None:
    """C02: the version literal accepts only the exact string ``0.1``."""
    with pytest.raises(ValidationError):
        config(schema_version=version)


@pytest.mark.parametrize("extra", [{"unknown": 1}, {"metadata": {}}, {"payload": {}}])
def test_an_unknown_field_is_refused_on_the_context(extra: dict[str, object]) -> None:
    """C02: extra=forbid applies to every record."""
    with pytest.raises(ValidationError):
        context(**extra)


@pytest.mark.parametrize(
    "record_builder",
    [
        lambda extra: config(**extra),
        lambda extra: request(**extra),
        lambda extra: ModelResponse(schema_version=SCHEMA, request_id="r", status="ok",
                                    content="{}", **extra),
        lambda extra: ModelUsage(schema_version=SCHEMA, input_tokens=1, output_tokens=1, **extra),
        lambda extra: PlannerCloseResult(schema_version=SCHEMA, drained=True, pending_calls=0,
                                         **extra),
    ],
)
def test_an_unknown_field_is_refused_on_every_other_record(record_builder) -> None:
    """C02: no record silently accepts an undeclared key."""
    with pytest.raises(ValidationError):
        record_builder({"unknown": 1})


def test_the_config_defaults_are_the_declared_limits() -> None:
    """C02: the documented defaults are the actual default values."""
    planner_config = config()
    assert planner_config.call_timeout_ms == 30000
    assert planner_config.cleanup_timeout_ms == 100
    assert planner_config.max_calls == 20
    assert planner_config.max_request_bytes == 262144
    assert planner_config.max_response_bytes == 65536
    assert planner_config.max_output_tokens == 2048
    assert planner_config.max_json_depth == 32
    assert planner_config.max_json_nodes == 10000


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "5", None])
def test_a_count_limit_is_a_strict_positive_integer(value: object) -> None:
    """C02: integer limits never accept a boolean, a float, a string or zero."""
    for field in (
        "max_calls",
        "max_request_bytes",
        "max_response_bytes",
        "max_output_tokens",
        "max_json_depth",
        "max_json_nodes",
    ):
        with pytest.raises(ValidationError):
            config(**{field: value})


@pytest.mark.parametrize(
    "value", [0, -1, -0.5, True, False, "30000", float("nan"), float("inf"), None]
)
def test_a_timeout_is_a_finite_positive_number(value: object) -> None:
    """C02: a timeout accepts a positive int/float and refuses a boolean or a string."""
    for field in ("call_timeout_ms", "cleanup_timeout_ms"):
        with pytest.raises(ValidationError):
            config(**{field: value})


def test_a_timeout_accepts_an_integer_or_a_float() -> None:
    """C02: both numeric spellings of a positive timeout are legal."""
    planner_config = config(call_timeout_ms=5, cleanup_timeout_ms=0.5)
    assert planner_config.call_timeout_ms == 5.0
    assert planner_config.cleanup_timeout_ms == 0.5


def test_guidance_must_name_exactly_the_declared_capability_keys() -> None:
    """C02: the guidance key set equals the capability key set, not a superset or subset."""
    with pytest.raises(ValidationError):
        context(action_guidance={"mark_done": {"x": 1}, "extra": {"x": 1}})
    with pytest.raises(ValidationError):
        context(action_guidance={})
    with pytest.raises(ValidationError):
        context(outcome_guidance={"other": {"x": 1}})


@pytest.mark.parametrize("value", [[], "text", 5, {}, None])
def test_each_guidance_value_must_be_a_non_empty_json_object(value: object) -> None:
    """C02: a guidance entry is a non-empty JSON object describing its payload."""
    with pytest.raises(ValidationError):
        context(action_guidance={"mark_done": value})


@pytest.mark.parametrize("value", ["", "   ", 7, True, None])
def test_a_context_identifier_must_be_a_non_blank_string(value: object) -> None:
    """C02: the trusted identifiers are strict non-blank strings."""
    for field in ("run_id", "goal_id", "goal_text"):
        with pytest.raises(ValidationError):
            context(**{field: value})


def test_the_context_requires_the_task_criteria_and_capabilities() -> None:
    """C02: the context carries the existing Core criteria and capability records."""
    with pytest.raises(ValidationError):
        PlannerContext(
            schema_version=SCHEMA, run_id="run-1", goal_id="goal-1", goal_text="text",
            capabilities=capabilities(), action_guidance={"mark_done": {"x": 1}},
            outcome_guidance={"json_rule": {"x": 1}},
        )


def test_a_response_status_and_content_are_one_verdict() -> None:
    """C02: ok requires string content; refused requires no content at all."""
    assert response(status="ok", content="{}").content == "{}"
    assert response(status="refused", content=None).content is None
    with pytest.raises(ValidationError):
        response(status="ok", content=None)
    with pytest.raises(ValidationError):
        response(status="ok", content=5)
    with pytest.raises(ValidationError):
        response(status="refused", content="{}")
    with pytest.raises(ValidationError):
        response(status="maybe", content="{}")


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None])
def test_usage_is_a_strict_non_negative_integer(value: object) -> None:
    """C02: a token count is a strict non-negative integer, never a bool or a float."""
    with pytest.raises(ValidationError):
        ModelUsage(schema_version=SCHEMA, input_tokens=value, output_tokens=0)
    with pytest.raises(ValidationError):
        ModelUsage(schema_version=SCHEMA, input_tokens=0, output_tokens=value)


def test_usage_accepts_zero_and_a_missing_usage_stays_absent() -> None:
    """C02: zero is legal, and an absent usage is not manufactured as zero."""
    assert ModelUsage(schema_version=SCHEMA, input_tokens=0, output_tokens=0).output_tokens == 0
    assert response().usage is None


@pytest.mark.parametrize(
    ("drained", "pending"),
    [(True, 1), (True, 2), (False, 0)],
)
def test_a_close_result_requires_a_consistent_drain(drained: bool, pending: int) -> None:
    """C02: drained and pending_calls are one verdict."""
    with pytest.raises(ValidationError):
        PlannerCloseResult(schema_version=SCHEMA, drained=drained, pending_calls=pending)
    assert PlannerCloseResult(
        schema_version=SCHEMA, drained=pending == 0, pending_calls=pending
    ).drained == (pending == 0)


@pytest.mark.parametrize("value", ["", "   ", 5, True, None])
def test_a_request_identifier_must_be_a_non_blank_string(value: object) -> None:
    """C02: the planner-generated request identifier is a non-blank string."""
    with pytest.raises(ValidationError):
        request(request_id=value)


def test_a_request_requires_every_declared_structured_field() -> None:
    """C02: the request is a complete structured record, not a provider message."""
    for field in ("model_context", "observation", "response_schema", "max_output_tokens"):
        values = {
            "request_id": "req-1", "model_context": context(), "observation": observation(),
            "response_schema": {"type": "object"}, "max_output_tokens": 8,
        }
        values.pop(field)
        with pytest.raises(ValidationError):
            ModelRequest(schema_version=SCHEMA, **values)


def test_a_forged_instance_is_refused_by_the_record_ingress() -> None:
    """C02: model_validate of an existing instance revalidates instead of trusting it."""
    forged = PlannerConfig.model_construct(schema_version=SCHEMA, max_calls=-1)
    with pytest.raises(PlannerError) as info:
        PlannerConfig.model_validate(forged)
    assert info.value.code == "INPUT_INVALID"
    forged_response = ModelResponse.model_construct(
        schema_version=SCHEMA, request_id="req-1", status="ok", content=None, usage=None,
    )
    with pytest.raises(PlannerError) as info:
        ModelResponse.model_validate(forged_response)
    assert info.value.code == "RESPONSE_INVALID"
    legal = config()
    rebuilt = PlannerConfig.model_validate(legal)
    assert rebuilt.max_calls == 20
    assert rebuilt is not legal


def test_assignment_validation_rejects_a_damaged_field_and_keeps_the_old_value() -> None:
    """C02: assignment is validated and a rejected assignment leaves the record intact."""
    planner_config = config()
    with pytest.raises(ValidationError):
        planner_config.max_calls = True
    assert planner_config.max_calls == 20
    current = context()
    with pytest.raises(ValidationError):
        current.run_id = 7
    assert current.run_id == "run-1"


def test_a_refused_response_assignment_keeps_the_previous_verdict() -> None:
    """F01: a refused status/content change is rolled back; no half-valid response remains."""
    current = response(status="refused", content=None)
    with pytest.raises(ValidationError):
        current.status = "ok"
    assert (current.status, current.content) == ("refused", None)
    with pytest.raises(ValidationError):
        current.content = "{}"
    assert (current.status, current.content) == ("refused", None)


def test_a_refused_close_assignment_keeps_the_previous_drain() -> None:
    """F01: a refused pending/drained change is rolled back; the close result stays consistent."""
    current = PlannerCloseResult(schema_version=SCHEMA, drained=True, pending_calls=0)
    with pytest.raises(ValidationError):
        current.pending_calls = 1
    assert (current.drained, current.pending_calls) == (True, 0)
    with pytest.raises(ValidationError):
        current.drained = False
    assert (current.drained, current.pending_calls) == (True, 0)


def test_a_refused_context_assignment_keeps_the_previous_guidance() -> None:
    """F01: a refused guidance assignment is rolled back to the exact previous record."""
    current = context()
    before = current.model_dump()
    with pytest.raises(ValidationError):
        current.action_guidance = {}
    assert current.action_guidance == {"mark_done": {"payload": {"type": "object"}}}
    assert current.model_dump() == before


def test_a_legal_assignment_still_updates_the_record_atomically() -> None:
    """F01: an accepted assignment is committed, so the rollback does not freeze the record."""
    current = context()
    current.goal_id = "goal-2"
    assert current.goal_id == "goal-2"
    assert current.model_dump()["goal_id"] == "goal-2"


def test_the_context_owns_a_detached_copy_of_the_capabilities() -> None:
    """F01: the caller keeps no alias into the record's nested state."""
    supplied = capabilities()
    current = context(capabilities=supplied)
    assert current.capabilities is not supplied
    supplied.supported_actions.append("injected")
    assert current.capabilities.supported_actions == ["mark_done"]


def test_a_forged_nested_capability_is_refused_at_construction() -> None:
    """F01: a stored tuple that the nested Core field forbids is not accepted by the parent."""
    forged = capabilities()
    forged.__dict__["supported_actions"] = ("mark_done",)
    with pytest.raises(PlannerError) as info:
        context(capabilities=forged)
    assert info.value.code == "INPUT_INVALID"


def test_a_virtual_dict_view_cannot_hide_a_forged_nested_capability() -> None:
    """F02: nested ingress reads real storage; a masked view cannot launder a forged tuple."""
    class VirtualCapabilities(AdapterCapabilities):
        def __getattribute__(self, name: str) -> object:
            if name == "__dict__":
                return dict(object.__getattribute__(self, "__dict__"),
                            supported_actions=["mark_done"])
            return super().__getattribute__(name)

    forged = VirtualCapabilities(
        schema_version=SCHEMA,
        adapter_name="synthetic",
        environment_type="fake",
        supported_actions=["mark_done"],
        supported_outcome_types=["json_rule"],
    )
    object.__getattribute__(forged, "__dict__")["supported_actions"] = ("mark_done",)
    with pytest.raises(PlannerError) as info:
        context(capabilities=forged)
    assert info.value.code == "INPUT_INVALID"


def test_a_forged_context_is_refused_by_the_planner_constructor() -> None:
    """C02: a model_construct or __dict__ forgery is revalidated, never trusted."""
    forged_values = context().model_dump(mode="python")
    forged = PlannerContext.model_construct(**forged_values)
    forged.__dict__["run_id"] = 7
    with pytest.raises(PlannerError) as info:
        LLMPlanner(provider=_AsyncProvider(), context=forged, config=config())
    assert info.value.code == "INPUT_INVALID"


def test_a_forged_config_is_refused_by_the_planner_constructor() -> None:
    """C02: a damaged stored limit is refused at the public entry."""
    forged = PlannerConfig.model_construct(schema_version=SCHEMA, max_calls=-3)
    with pytest.raises(PlannerError) as info:
        LLMPlanner(provider=_AsyncProvider(), context=context(), config=forged)
    assert info.value.code == "INPUT_INVALID"


def test_a_legal_context_builds_a_ready_planner() -> None:
    """C02: a legal record still passes every gate."""
    planner = LLMPlanner(provider=_AsyncProvider(), context=context(), config=config())
    assert (planner.calls_used, planner.state) == (0, "ready")


def test_a_synchronous_provider_is_refused_at_construction() -> None:
    """C02: the injected transport must be the declared asynchronous interface."""
    with pytest.raises(PlannerError) as info:
        LLMPlanner(provider=_SyncProvider(), context=context(), config=config())
    assert info.value.code == "INPUT_INVALID"
    with pytest.raises(PlannerError):
        LLMPlanner(provider=object(), context=context(), config=config())


def test_the_offline_provider_replays_validated_detached_responses() -> None:
    """The in-process reference double is deterministic and performs no I/O."""
    built = build_request(context=context(), observation=observation(), config=config(),
                          request_id="req-1")
    first = response(request_id="req-1", content='{"answer": 1}')
    provider = OfflineProvider([first])
    seen = asyncio.run(provider.complete(built))
    assert seen.content == '{"answer": 1}'
    assert provider.received[0].request_id == "req-1"
    with pytest.raises(PlannerError) as info:
        asyncio.run(provider.complete(built))
    assert info.value.code == "PROVIDER_FAILURE"


def test_the_provider_protocol_declares_one_async_complete() -> None:
    """C02: the injected interface is exactly one coroutine method."""
    import inspect

    assert inspect.iscoroutinefunction(_AsyncProvider.complete)
    assert list(inspect.signature(_AsyncProvider.complete).parameters) == ["self", "request"]


_STORAGE_DESCRIPTOR = BaseModel.__dict__["__dict__"]


def _real_storage(record: BaseModel) -> dict[str, object]:
    """The record's actual stored fields, read without any subclass attribute hook."""
    return _STORAGE_DESCRIPTOR.__get__(record, type(record))


def _masking_subclass(declared: type, visible: dict[str, object]) -> type:
    """A record subclass whose ``__dict__`` view differs from the real stored fields.

    The getter answers with ``visible`` layered over the actual storage, and the setter
    delegates to the base ``__dict__`` descriptor, so construction and mutation still reach the
    object's real storage. Only the reported view is sanitised.
    """
    class Masked(declared):
        @property
        def __dict__(self) -> dict[str, object]:
            return {**_real_storage(self), **visible}

        @__dict__.setter
        def __dict__(self, value: dict[str, object]) -> None:
            _STORAGE_DESCRIPTOR.__set__(self, value)

    return Masked


def test_a_dict_property_subclass_cannot_hide_a_forged_nested_capability() -> None:
    """F02: nested ingress reads real storage, not a subclass ``__dict__`` property's view."""
    masked = _masking_subclass(AdapterCapabilities, {"supported_actions": ["mark_done"]})
    forged = masked(**capabilities().model_dump())
    _real_storage(forged)["supported_actions"] = ("mark_done",)
    assert isinstance(_real_storage(forged)["supported_actions"], tuple)
    assert forged.__dict__["supported_actions"] == ["mark_done"]
    with pytest.raises(PlannerError) as info:
        context(capabilities=forged)
    assert info.value.code == "INPUT_INVALID"


def test_a_dict_property_subclass_is_judged_by_its_real_storage() -> None:
    """F02: a masking view cannot make a legally stored record look invalid."""
    masked = _masking_subclass(PlannerConfig, {"max_calls": False})
    supplied = masked(schema_version=SCHEMA)
    assert supplied.__dict__["max_calls"] is False
    assert _real_storage(supplied)["max_calls"] == 20
    planner = LLMPlanner(provider=_AsyncProvider(), context=context(), config=supplied)
    assert (planner.calls_used, planner.state) == (0, "ready")


def test_a_dict_property_subclass_is_rebuilt_as_the_declared_record() -> None:
    """C02: the copied record is the declared base type read from the real stored fields."""
    masked = _masking_subclass(PlannerContext, {"goal_text": "visible-only text"})
    supplied = masked(**context().model_dump(mode="python"))
    assert supplied.__dict__["goal_text"] == "visible-only text"
    rebuilt = PlannerContext.model_validate(supplied)
    assert type(rebuilt) is PlannerContext
    assert rebuilt.goal_text == "finish the synthetic task"
    assert rebuilt.capabilities.supported_actions == ["mark_done"]
    assert rebuilt is not supplied

