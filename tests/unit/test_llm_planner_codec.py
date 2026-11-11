"""Acceptance tests for the strict codec between model output and the Core vocabulary.

The cases cover both directions: an assembled request is measured and bound to the trusted
context, and a provider answer is rejected on every malformed, duplicated, oversized,
out-of-capability or out-of-budget shape instead of being coerced into an action.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
LLM_SRC = Path(__file__).resolve().parents[2] / "packages" / "planners" / "llm" / "src"
for candidate in (CORE_SRC, LLM_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.models import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Observation,
)
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm import (
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PlannerConfig,
    PlannerContext,
    PlannerError,
)
from forecontract_llm.codec import build_request, decode_response

SCHEMA = "0.1"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        schema_version=SCHEMA,
        adapter_name="synthetic",
        environment_type="fake",
        supported_actions=["mark_done", "open_gate"],
        supported_outcome_types=["json_rule", "gate_open"],
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
        "action_guidance": {
            "mark_done": {"payload": {"type": "object"}},
            "open_gate": {"payload": {"type": "object"}},
        },
        "outcome_guidance": {
            "json_rule": {"payload": {"type": "object"}},
            "gate_open": {"payload": {"type": "object"}},
        },
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
        "state": {"done": False, "nested": {"count": 2}},
        "timestamp": NOW,
    }
    values.update(overrides)
    return Observation(**values)


def make_request(*, planner_config: PlannerConfig | None = None,
                 request_id: str = "req-1") -> ModelRequest:
    return build_request(
        context=context(),
        observation=observation(),
        config=planner_config if planner_config is not None else config(),
        request_id=request_id,
    )


REQUEST = make_request()


def body(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": SCHEMA,
        "proposed_action": {
            "schema_version": SCHEMA,
            "action_type": "mark_done",
            "payload": {"selector": "#go"},
        },
        "expected_outcomes": [
            {
                "schema_version": SCHEMA,
                "outcome_type": "json_rule",
                "payload": {"rule_type": "value_equals", "path": ["done"]},
            }
        ],
    }
    values.update(overrides)
    return values


def text(**overrides: object) -> str:
    return json.dumps(body(**overrides))


def answer(content: str, *, request_value: ModelRequest | None = None,
           usage_value: ModelUsage | None = None) -> ModelResponse:
    current = request_value if request_value is not None else REQUEST
    return ModelResponse(
        schema_version=SCHEMA,
        request_id=current.request_id,
        status="ok",
        content=content,
        usage=usage_value,
    )


def decode(content: str, *, planner_config: PlannerConfig | None = None,
           request_value: ModelRequest | None = None,
           usage_value: ModelUsage | None = None) -> object:
    return decode_response(
        response=answer(content, request_value=request_value, usage_value=usage_value),
        request=request_value if request_value is not None else REQUEST,
        config=planner_config if planner_config is not None else config(),
    )


def assert_rejected(content: str, code: str, *, planner_config: PlannerConfig | None = None,
                    request_value: ModelRequest | None = None,
                    usage_value: ModelUsage | None = None) -> PlannerError:
    with pytest.raises(PlannerError) as info:
        decode(content, planner_config=planner_config, request_value=request_value,
               usage_value=usage_value)
    assert info.value.code == code
    return info.value


def test_build_request_produces_the_declared_structured_request() -> None:
    """C02/C03: the request is structured, versioned, measured and identifier-bound."""
    built = make_request(request_id="req-42")
    assert built.schema_version == SCHEMA
    assert built.request_id == "req-42"
    assert built.model_context.run_id == "run-1"
    assert built.model_context.goal_id == "goal-1"
    assert built.observation.observation_id == "obs-1"
    assert built.max_output_tokens == 2048
    schema = built.response_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"schema_version", "proposed_action", "expected_outcomes"}
    assert schema["properties"]["schema_version"] == {"const": SCHEMA}
    assert schema["properties"]["proposed_action"]["additionalProperties"] is False
    assert schema["properties"]["expected_outcomes"]["minItems"] == 1
    # No transport-only, credential or tool field exists on the request record.
    assert set(ModelRequest.model_fields) == {
        "schema_version", "request_id", "model_context", "observation", "response_schema",
        "max_output_tokens",
    }


def test_the_advertised_outcome_schema_matches_the_core_outcome_shape() -> None:
    """F05: the schema advertises exactly the fields Core's ExpectedOutcome actually has."""
    outcome = make_request().response_schema["properties"]["expected_outcomes"]["items"]
    assert outcome["additionalProperties"] is False
    assert set(outcome["properties"]) == {
        "schema_version", "outcome_type", "payload", "required",
    }
    assert "metadata" not in outcome["properties"]


def test_an_outcome_metadata_field_is_refused_even_when_empty() -> None:
    """F05: Core has no outcome metadata, so any such key is an undeclared extra field."""
    content = json.dumps({**body(), "expected_outcomes": [
        {"schema_version": SCHEMA, "outcome_type": "json_rule", "metadata": {}}]})
    assert_rejected(content, "RESPONSE_INVALID")


def test_build_request_carries_the_complete_current_observation() -> None:
    """C03: the full observation state is carried, not a summary or a subset."""
    built = make_request()
    assert built.observation.state == {"done": False, "nested": {"count": 2}}
    assert built.observation.timestamp == NOW


def test_build_request_revalidates_its_context() -> None:
    """C02: the codec is not a way around the planner's input gates."""
    forged = PlannerContext.model_construct(**context().model_dump(mode="python"))
    forged.__dict__["run_id"] = 7
    with pytest.raises(PlannerError) as info:
        build_request(context=forged, observation=observation(), config=config(),
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_build_request_revalidates_its_configuration() -> None:
    """C02: a damaged limit is refused before any request is assembled."""
    forged = PlannerConfig.model_construct(schema_version=SCHEMA, max_calls=0)
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=observation(), config=forged,
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_build_request_revalidates_its_observation() -> None:
    """C02: a forged observation is refused rather than measured."""
    forged = Observation.model_construct(
        schema_version=SCHEMA, observation_id="obs-1", environment_type="fake",
        state={"done": True}, timestamp=NOW,
    )
    forged.__dict__["state"] = b"not-json"
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=forged, config=config(),
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_a_masked_config_serializer_cannot_hide_a_forged_stored_limit() -> None:
    """F02: a subclass whose model_dump lies cannot launder the actual stored value."""
    class MaskedConfig(PlannerConfig):
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            return config().model_dump(*args, **kwargs)

    forged = MaskedConfig(schema_version=SCHEMA)
    forged.__dict__["max_calls"] = False
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=observation(), config=forged,
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_a_masked_observation_serializer_cannot_hide_a_forged_stored_environment() -> None:
    """F02: the observation ingress reads real storage, not an overridable serialiser."""
    legal = observation()

    class MaskedObservation(Observation):
        def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
            return legal.model_dump(*args, **kwargs)

    forged = MaskedObservation(**legal.model_dump())
    forged.__dict__["environment_type"] = False
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=forged, config=config(),
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_a_virtual_dict_view_cannot_hide_a_forged_config_limit() -> None:
    """F02: an overridden __getattribute__ for __dict__ has no authority over real storage."""
    class VirtualConfig(PlannerConfig):
        def __getattribute__(self, name: str) -> object:
            if name == "__dict__":
                return dict(object.__getattribute__(self, "__dict__"), max_calls=1)
            return super().__getattribute__(name)

    forged = VirtualConfig(schema_version=SCHEMA)
    object.__getattribute__(forged, "__dict__")["max_calls"] = False
    assert object.__getattribute__(forged, "__dict__")["max_calls"] is False
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=observation(), config=forged,
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_a_virtual_dict_view_cannot_hide_a_forged_observation_environment() -> None:
    """F02: the observation ingress reads real storage even when a subclass masks its view."""
    legal = observation()

    class VirtualObservation(Observation):
        def __getattribute__(self, name: str) -> object:
            if name == "__dict__":
                return dict(object.__getattribute__(self, "__dict__"),
                            environment_type="fake")
            return super().__getattribute__(name)

    forged = VirtualObservation(**legal.model_dump())
    object.__getattribute__(forged, "__dict__")["environment_type"] = False
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=forged, config=config(),
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_build_request_refuses_a_cyclic_observation() -> None:
    """C04: a cycle in a supplied record is a controlled refusal, not recursion."""
    state: dict[str, object] = {}
    state["self"] = state
    forged = Observation.model_construct(
        schema_version=SCHEMA, observation_id="obs-1", environment_type="fake",
        state=state, timestamp=NOW,
    )
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=forged, config=config(),
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_build_request_refuses_an_environment_mismatch() -> None:
    """C03: an environment mismatch is refused before any provider could be called."""
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=observation(environment_type="other"),
                      config=config(), request_id="req-1")
    assert info.value.code == "CONTEXT_MISMATCH"
    assert info.value.location == ("environment_type",)


def test_build_request_refuses_a_request_over_the_byte_limit() -> None:
    """C04: the measured request includes the response schema and must fit the byte limit."""
    with pytest.raises(PlannerError) as info:
        make_request(planner_config=config(max_request_bytes=64))
    assert info.value.code == "REQUEST_TOO_LARGE"
    assert info.value.location == ("bytes",)


def test_build_request_refuses_a_request_over_the_depth_limit() -> None:
    """C04: the request depth is measured from its root container."""
    with pytest.raises(PlannerError) as info:
        make_request(planner_config=config(max_json_depth=3))
    assert info.value.code == "REQUEST_TOO_LARGE"
    assert info.value.location == ("response_schema",)


def test_build_request_refuses_a_request_over_the_node_limit() -> None:
    """C04: the request node count includes containers and scalars."""
    with pytest.raises(PlannerError) as info:
        make_request(planner_config=config(max_json_nodes=6))
    assert info.value.code == "REQUEST_TOO_LARGE"


@pytest.mark.parametrize("request_id", ["", "   ", 7, True, None])
def test_build_request_refuses_an_invalid_request_identifier(request_id: object) -> None:
    """C02: the identifier is supplied by the planner and must be a non-blank string."""
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=observation(), config=config(),
                      request_id=request_id)
    assert info.value.code == "INPUT_INVALID"


def test_decode_response_builds_the_forecast_from_trusted_context() -> None:
    """C03: the forecast identifier, metadata and optional hints come from trusted sources."""
    content = text(intent="reach the goal", decision_summary=None, confidence=0.75,
                   task_completion_hint=True)
    forecast = decode(content)
    assert forecast.forecast_id == "req-1"
    assert forecast.schema_version == SCHEMA
    assert forecast.proposed_action.action_type == "mark_done"
    assert forecast.proposed_action.payload == {"selector": "#go"}
    assert [item.outcome_type for item in forecast.expected_outcomes] == ["json_rule"]
    assert forecast.intent == "reach the goal"
    assert forecast.decision_summary is None
    assert forecast.confidence == 0.75
    assert forecast.task_completion_hint is True
    assert forecast.metadata == {
        "planner": "llm",
        "run_id": "run-1",
        "goal_id": "goal-1",
        "observation_id": "obs-1",
        "request_id": "req-1",
    }


def test_decode_response_rejects_a_replayed_response() -> None:
    """C03: a response bound to another request is never accepted."""
    old = answer("{}", request_value=make_request(request_id="req-old"))
    with pytest.raises(PlannerError) as info:
        decode_response(response=old, request=REQUEST, config=config())
    assert info.value.code == "RESPONSE_INVALID"
    assert info.value.location == ("request_id",)


def test_decode_response_refuses_a_refusal() -> None:
    """C05: a refusal is the fixed PROVIDER_REFUSED error, never a plan."""
    refused = ModelResponse(schema_version=SCHEMA, request_id="req-1", status="refused",
                            content=None)
    with pytest.raises(PlannerError) as info:
        decode_response(response=refused, request=REQUEST, config=config())
    assert info.value.code == "PROVIDER_REFUSED"


def test_decode_response_rejects_a_forged_response_instance() -> None:
    """C02: an instance built by model_construct is revalidated, not trusted."""
    forged = ModelResponse.model_construct(
        schema_version=SCHEMA, request_id="req-1", status="ok", content=None, usage=None,
    )
    with pytest.raises(PlannerError) as info:
        decode_response(response=forged, request=REQUEST, config=config())
    assert info.value.code == "RESPONSE_INVALID"


def test_decode_response_rejects_a_non_record_answer() -> None:
    """C05: a plain mapping is not a response record."""
    with pytest.raises(PlannerError) as info:
        decode_response(response={"content": text()}, request=REQUEST, config=config())
    assert info.value.code == "RESPONSE_INVALID"


def test_decode_response_refuses_content_over_the_byte_limit() -> None:
    """C05: the content UTF-8 length is bounded independently of its JSON shape."""
    error = assert_rejected(text(), "RESPONSE_TOO_LARGE",
                            planner_config=config(max_response_bytes=8))
    assert error.location == ("bytes",)


@pytest.mark.parametrize("content", ["", "   ", "[]", '"text"', "null", "1", "true"])
def test_decode_response_requires_exactly_one_json_object(content: str) -> None:
    """C03: the content must be a JSON object, not a scalar, a list or nothing."""
    assert_rejected(content, "RESPONSE_INVALID")


@pytest.mark.parametrize(
    "content",
    [
        "```json\n" + json.dumps(body()) + "\n```",
        json.dumps(body()) + " trailing",
        json.dumps(body()) + json.dumps(body()),
        "prefix " + json.dumps(body()),
        '{"schema_version":"0.1","proposed_action":{},"expected_outcomes":[]}',
    ],
)
def test_decode_response_refuses_fences_trailing_text_and_multiple_objects(content: str) -> None:
    """C03: no Markdown fence, trailing text, extra object or malformed document is accepted."""
    assert_rejected(content, "RESPONSE_INVALID")


@pytest.mark.parametrize(
    "content",
    [
        '{"schema_version":"0.1","schema_version":"0.1","proposed_action":{},"expected_outcomes":[]}',
        '{"schema_version":"0.1","proposed_action":{"schema_version":"0.1","action_type":"mark_done","payload":{"a":1,"a":2}},"expected_outcomes":[{"schema_version":"0.1","outcome_type":"json_rule"}]}',
        '{"schema_version":"0.1","proposed_action":{"schema_version":"0.1","action_type":"mark_done"},"expected_outcomes":[{"schema_version":"0.1","outcome_type":"json_rule","payload":{"nested":{"b":1,"b":2}}}]}',
    ],
)
def test_decode_response_refuses_duplicate_keys_at_any_depth(content: str) -> None:
    """C03: a duplicate object key is refused, not silently overwritten."""
    assert_rejected(content, "RESPONSE_INVALID")


@pytest.mark.parametrize(
    "content",
    [
        '{"schema_version":"0.1","proposed_action":{},"expected_outcomes":[],"confidence":NaN}',
        '{"schema_version":"0.1","proposed_action":{},"expected_outcomes":[],"confidence":Infinity}',
        '{"schema_version":"0.1","proposed_action":{},"expected_outcomes":[],"confidence":-Infinity}',
    ],
)
def test_decode_response_refuses_non_finite_numbers(content: str) -> None:
    """C03: NaN and Infinity are not JSON and are never parsed."""
    assert_rejected(content, "RESPONSE_INVALID")


def test_decode_response_refuses_an_invalid_unicode_surrogate() -> None:
    """C03: a lone surrogate is invalid Unicode, not a string to keep."""
    content = text(intent="\ud800")
    assert_rejected(content, "RESPONSE_INVALID")
    content_escape = text()[:-1] + ',"intent":"\\ud800"}'
    assert_rejected(content_escape, "RESPONSE_INVALID")


def test_decode_response_refuses_content_over_the_depth_limit() -> None:
    """C03: the depth limit is enforced before any expensive model validation."""
    deep: dict[str, object] = {}
    node: dict[str, object] = deep
    for _ in range(12):
        child: dict[str, object] = {}
        node["child"] = child
        node = child
    content = text(proposed_action={
        "schema_version": SCHEMA, "action_type": "mark_done", "payload": deep,
    })
    assert_rejected(content, "RESPONSE_INVALID", planner_config=config(max_json_depth=6))


def test_decode_response_refuses_content_over_the_node_limit() -> None:
    """C03: container and scalar nodes are counted against the configured limit."""
    assert_rejected(text(), "RESPONSE_INVALID", planner_config=config(max_json_nodes=5))


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"schema_version": "0.1"}),
        json.dumps({**body(), "forecast_id": "forged"}),
        json.dumps({**body(), "metadata": {"run_id": "other"}}),
        json.dumps({**body(), "planner": "llm"}),
    ],
)
def test_decode_response_refuses_missing_or_extra_top_level_keys(content: str) -> None:
    """C03: the top-level key set is exactly the declared one."""
    assert_rejected(content, "RESPONSE_INVALID")


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({**body(), "proposed_action": {
            "schema_version": SCHEMA, "action_type": "mark_done", "metadata": {"x": 1}}}),
        json.dumps({**body(), "expected_outcomes": [
            {"schema_version": SCHEMA, "outcome_type": "json_rule", "metadata": {"x": 1}}]}),
    ],
)
def test_decode_response_refuses_non_empty_nested_metadata(content: str) -> None:
    """C03: a nested Core metadata object must be empty, so no control data can be smuggled."""
    assert_rejected(content, "RESPONSE_INVALID")


@pytest.mark.parametrize(
    "outcomes",
    [[], {}, "text", [{"schema_version": SCHEMA, "outcome_type": "json_rule"}, 7]],
)
def test_decode_response_requires_a_non_empty_list_of_outcomes(outcomes: object) -> None:
    """C03: expected_outcomes is a strict non-empty JSON list of Core outcome records."""
    assert_rejected(json.dumps({**body(), "expected_outcomes": outcomes}), "RESPONSE_INVALID")


@pytest.mark.parametrize(
    "mutated",
    [
        {"proposed_action": {"schema_version": SCHEMA, "action_type": 5}},
        {"proposed_action": {"schema_version": SCHEMA, "action_type": "   "}},
        {"proposed_action": {"schema_version": "0.2", "action_type": "mark_done"}},
        {"expected_outcomes": [{"schema_version": SCHEMA, "outcome_type": True}]},
        {"expected_outcomes": [{"schema_version": SCHEMA, "outcome_type": "json_rule",
                                "required": "true"}]},
        {"intent": 7},
        {"decision_summary": False},
        {"task_completion_hint": "true"},
        {"schema_version": 0.1},
    ],
)
def test_decode_response_never_coerces_a_scalar_type(mutated: dict[str, object]) -> None:
    """C03: a number, a boolean or a numeric string never becomes a string, bool or version."""
    assert_rejected(json.dumps({**body(), **mutated}), "RESPONSE_INVALID")


@pytest.mark.parametrize("confidence", [True, False, "0.5", 1.5, -0.1, 2, [0.5]])
def test_decode_response_requires_a_finite_confidence_between_zero_and_one(
    confidence: object,
) -> None:
    """C03: confidence is a finite number in [0, 1] and a boolean is not a number."""
    assert_rejected(json.dumps({**body(), "confidence": confidence}), "RESPONSE_INVALID")


def test_decode_response_accepts_the_boundary_confidences() -> None:
    """C03: zero and one are inside the closed interval."""
    assert decode(json.dumps({**body(), "confidence": 0})).confidence == 0.0
    assert decode(json.dumps({**body(), "confidence": 1})).confidence == 1.0
    assert decode(json.dumps({**body(), "confidence": 0.5})).confidence == 0.5


def test_decode_response_refuses_an_unsupported_action() -> None:
    """C03: a declared name outside supported_actions is an unsupported action."""
    content = json.dumps({**body(), "proposed_action": {
        "schema_version": SCHEMA, "action_type": "delete_everything"}})
    error = assert_rejected(content, "UNSUPPORTED_ACTION")
    assert error.location == ("action",)


def test_decode_response_refuses_an_unsupported_outcome_with_its_index() -> None:
    """C03: every outcome type must be declared; the rejection names its index."""
    content = json.dumps({**body(), "expected_outcomes": [
        {"schema_version": SCHEMA, "outcome_type": "json_rule"},
        {"schema_version": SCHEMA, "outcome_type": "undeclared"},
    ]})
    error = assert_rejected(content, "UNSUPPORTED_OUTCOME")
    assert error.location == ("outcomes", 1)


def test_decode_response_refuses_a_declared_output_token_overrun() -> None:
    """C04: a declared output-token overrun is a budget failure, not a free plan."""
    over = ModelUsage(schema_version=SCHEMA, input_tokens=1, output_tokens=9)
    error = assert_rejected(text(), "BUDGET_EXCEEDED",
                            planner_config=config(max_output_tokens=8), usage_value=over)
    assert error.location == ("usage", "output_tokens")
    exact = ModelUsage(schema_version=SCHEMA, input_tokens=1, output_tokens=8)
    decode(text(), planner_config=config(max_output_tokens=8), usage_value=exact)
    decode(text())


def test_decode_response_errors_never_echo_the_response_body() -> None:
    """C05: the fixed error carries no model content and no arbitrary key as a path."""
    secret = "SYNTHETIC-PRIVATE-RESPONSE-MARKER"
    content = json.dumps({**body(), secret: {"value": secret}})
    error = assert_rejected(content, "RESPONSE_INVALID")
    assert secret not in str(error)
    assert secret not in repr(error)
    assert all(isinstance(part, (int, str)) for part in error.location)
    payload_secret = "SYNTHETIC-PRIVATE-PAYLOAD-MARKER"
    content = json.dumps({**body(), "proposed_action": {
        "schema_version": SCHEMA, "action_type": payload_secret,
        "payload": {"value": payload_secret}}})
    error = assert_rejected(content, "UNSUPPORTED_ACTION")
    assert payload_secret not in str(error)
    assert payload_secret not in repr(error)


def test_decode_response_returns_a_detached_forecast() -> None:
    """C02: the caller cannot reach through the returned forecast into the response."""
    content = text()
    first = decode(content)
    first.metadata["planner"] = "tampered"
    first.proposed_action.payload["selector"] = "#tampered"
    second = decode(content)
    assert second.metadata["planner"] == "llm"
    assert second.proposed_action.payload == {"selector": "#go"}
    assert first is not second


def test_optional_response_fields_default_to_absent() -> None:
    """C03: the optional proposal fields are not invented when the model omits them."""
    forecast = decode(text())
    assert forecast.intent is None
    assert forecast.decision_summary is None
    assert forecast.confidence is None
    assert forecast.task_completion_hint is None
    assert forecast.metadata["planner"] == "llm"
    assert isinstance(forecast.proposed_action, ActionSpec)
    assert all(isinstance(item, ExpectedOutcome) for item in forecast.expected_outcomes)


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


def test_a_dict_property_subclass_cannot_hide_a_forged_config_limit() -> None:
    """F02: the ingress reads real storage, not a subclass ``__dict__`` property's view."""
    masked = _masking_subclass(PlannerConfig, {"max_calls": 1})
    forged = masked(schema_version=SCHEMA)
    _real_storage(forged)["max_calls"] = False
    assert _real_storage(forged)["max_calls"] is False
    assert forged.__dict__["max_calls"] == 1
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=observation(), config=forged,
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_a_dict_property_subclass_cannot_hide_a_forged_observation_environment() -> None:
    """F02: a masked ``__dict__`` property cannot launder a stored observation environment."""
    legal = observation()
    masked = _masking_subclass(Observation, {"environment_type": "fake"})
    forged = masked(**legal.model_dump())
    _real_storage(forged)["environment_type"] = False
    assert _real_storage(forged)["environment_type"] is False
    assert forged.__dict__["environment_type"] == "fake"
    with pytest.raises(PlannerError) as info:
        build_request(context=context(), observation=forged, config=config(),
                      request_id="req-1")
    assert info.value.code == "INPUT_INVALID"


def test_a_dict_property_subclass_is_judged_by_its_real_storage() -> None:
    """F02: a lie in the visible view cannot reject or change a legally stored record."""
    masked = _masking_subclass(PlannerConfig, {"max_calls": False})
    supplied = masked(schema_version=SCHEMA)
    assert supplied.__dict__["max_calls"] is False
    assert _real_storage(supplied)["max_calls"] == 20
    built = build_request(context=context(), observation=observation(), config=supplied,
                          request_id="req-1")
    assert isinstance(built, ModelRequest)
    assert built.max_output_tokens == 2048

