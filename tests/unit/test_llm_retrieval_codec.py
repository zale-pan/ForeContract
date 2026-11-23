"""Acceptance tests for the strict retrieval message codec.

C02/C05/C06: one measured request carries the original Core request, the frozen scope and the
previous read results; one answer must be exactly one of the two declared message shapes and
must match the run/plan/turn/request identity. A forecast is still decided by the original
Core codec, and no refusal is ever coerced into a forecast.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest
from forecontract.models import AdapterCapabilities, ExpectedOutcome, Observation
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm.models import (
    ModelResponse,
    ModelUsage,
    PlannerConfig,
    PlannerContext,
)
from forecontract_llm.retrieval import (
    EvidenceReadResult,
    EvidenceScope,
    RetrievalConfig,
    RetrievalError,
    codec,
)
from forecontract_llm.retrieval.models import (
    EvidenceScopeEntry,
    ScopeView,
    canonical_json_bytes,
)

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


def _observation(state: dict | None = None, observation_id: str = OBS) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="offline",
                       state=state if state is not None else {"done": False},
                       timestamp=datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc))


def _config(**overrides: object) -> RetrievalConfig:
    base = {"schema_version": SCHEMA}
    base.update(overrides.pop("base", {}))
    return RetrievalConfig(schema_version=SCHEMA, base=PlannerConfig(**base), **overrides)


def _scope(*, run_id: str = RUN, observation_id: str = OBS,
           evidence_id: str = "evidence-1", view_id: str = "status",
           selector: list[object] | None = None, format: str = "text",
           entries: list[EvidenceScopeEntry] | None = None) -> EvidenceScope:
    if entries is None:
        entries = [EvidenceScopeEntry(
            schema_version=SCHEMA, evidence_id=evidence_id,
            source_observation_id=observation_id, kind="observation",
            raw_sha256=HEX, raw_bytes=10,
            views=[ScopeView(schema_version=SCHEMA, view_id=view_id,
                             selector=selector or ["state", "order", "status"],
                             format=format)])]
    return EvidenceScope(schema_version=SCHEMA, scope_id="scope-1", run_id=run_id,
                         observation_id=observation_id, entries=entries)


def _result(*, turn: int = 1, plan_id: str = "plan-1", run_id: str = RUN,
            scope_id: str = "scope-1", evidence_id: str = "evidence-1",
            view_id: str = "status", value: object = "pending",
            format: str = "text", text_range: object = None,
            selector: list[object] | None = None) -> EvidenceReadResult:
    raw = canonical_json_bytes(value)
    return EvidenceReadResult(
        schema_version=SCHEMA, scope_id=scope_id, run_id=run_id, plan_id=plan_id,
        turn=turn, request_id="request-1", evidence_id=evidence_id, view_id=view_id,
        status="found", source_observation_id=OBS, raw_sha256=HEX, raw_bytes=10,
        selector=selector or ["state", "order", "status"], format=format,
        text_range=text_range, partial=False, value=value,
        value_sha256=hashlib.sha256(raw).hexdigest(), value_bytes=len(raw))


def _build(*, scope: EvidenceScope | None = None, exchanges: list[EvidenceReadResult] | None = None,
           config: RetrievalConfig | None = None, observation: Observation | None = None,
           turn: int = 1, request_id: str = "request-1", plan_id: str = "plan-1"):
    return codec.build_request(
        context=_context(),
        observation=observation or _observation(),
        config=config or _config(),
        request_id=request_id,
        plan_id=plan_id,
        turn=turn,
        scope=scope or _scope(),
        exchanges=list(exchanges or []),
    )


def _forecast_body(*, action_type: str = "mark_done", outcome_type: str = "json_rule",
                   payload: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA,
        "proposed_action": {"schema_version": SCHEMA, "action_type": action_type,
                            "payload": payload or {}},
        "expected_outcomes": [{"schema_version": SCHEMA, "outcome_type": outcome_type,
                               "payload": {"schema_version": SCHEMA,
                                           "rule_type": "value_equals",
                                           "path": ["done"], "expected": True}}],
        "task_completion_hint": True,
    }


def _envelope(request: object, kind: str, extra: dict) -> str:
    document = {"schema_version": SCHEMA, "run_id": request.run_id,
                "plan_id": request.plan_id, "turn": request.turn,
                "request_id": request.request_id, "kind": kind}
    document.update(extra)
    return json.dumps(document)


def _forecast_content(request: object, **body_overrides: object) -> str:
    return _envelope(request, "forecast", {"forecast": _forecast_body(**body_overrides)})


def _read_content(request: object, **read_overrides: object) -> str:
    read = {"evidence_id": "evidence-1", "view_id": "status", "text_range": None}
    read.update(read_overrides)
    return _envelope(request, "read_evidence", {"read": read})


def _decode(request: object, content: str, config: RetrievalConfig | None = None):
    return codec.decode_message(content=content, request=request, config=config or _config())


def _error(call) -> RetrievalError:
    with pytest.raises(RetrievalError) as info:
        call()
    return info.value


# --------------------------------------------------------------------------- request side


def test_build_request_binds_the_base_request_and_the_frozen_scope() -> None:
    """C02: the envelope and its base request carry one identity, one schema and the scope."""
    config = _config(base={"max_output_tokens": 4096})
    request, size = _build(config=config)
    assert request.request_id == "request-1"
    assert request.run_id == RUN
    assert request.base_request.request_id == request.request_id
    assert request.base_request.model_context.run_id == request.run_id
    assert request.base_request.observation.observation_id == OBS
    assert request.base_request.response_schema == request.response_schema
    assert request.scope.run_id == RUN
    assert request.base_request.max_output_tokens == 2048
    assert size == len(canonical_json_bytes(request.model_dump(mode="json")))
    assert size > 0


def test_build_request_charges_every_exchange_without_trimming_it() -> None:
    """C02/C05: the previous read results are re-sent in full, not summarised away."""
    _build(turn=1)
    exchange = _result(turn=1)
    request, size = _build(turn=2, exchanges=[exchange])
    assert request.exchanges[0].value == "pending"
    assert request.exchanges[0].value_sha256 == exchange.value_sha256
    assert size > codec.build_request(
        context=_context(), observation=_observation(), config=_config(),
        request_id="request-1", plan_id="plan-1", turn=2, scope=_scope(),
        exchanges=[])[1]


@pytest.mark.parametrize("scope_kwargs", [
    {"run_id": "run-2"},
    {"observation_id": "obs-2"},
])
def test_build_request_refuses_a_scope_from_another_run_or_observation(
        scope_kwargs: dict) -> None:
    """C02/C03: the scope must belong to this Run and this current Observation."""
    error = _error(lambda: _build(scope=_scope(**scope_kwargs)))
    assert (error.code, error.stage) == ("binding_mismatch", "binding")


def test_build_request_refuses_an_exchange_from_another_plan_or_a_replayed_turn() -> None:
    """C02: an exchange must precede the current turn and belong to this plan."""
    error = _error(lambda: _build(turn=1, exchanges=[_result(turn=1)]))
    assert (error.code, error.stage) == ("invalid_input", "request")
    error = _error(lambda: _build(turn=2, exchanges=[_result(turn=1, plan_id="plan-2")]))
    assert (error.code, error.stage) == ("invalid_input", "request")
    error = _error(lambda: _build(turn=2, exchanges=[_result(turn=1, scope_id="scope-2")]))
    assert (error.code, error.stage) == ("invalid_input", "request")


def test_build_request_enforces_the_single_request_byte_limit() -> None:
    """C05: the complete request is measured; an oversize one is refused before any call."""
    big = _observation(state={"blob": "x" * 300_000})
    error = _error(lambda: _build(observation=big))
    assert (error.code, error.stage) == ("budget_exceeded", "request")


def test_build_request_enforces_the_stricter_json_depth_limit() -> None:
    """C05: the base depth cap is not widened by the retrieval envelope."""
    config = _config(base={"max_json_depth": 4, "max_json_nodes": 10_000})
    deep = _observation(state={"a": {"b": {"c": {"d": {"e": 1}}}}})
    error = _error(lambda: _build(config=config, observation=deep))
    assert error.code == "budget_exceeded"


def test_the_response_schema_documents_the_two_message_kinds() -> None:
    """C02: the advertised schema has two branches and forbids the other branch's field."""
    request, _ = _build()
    schema = request.response_schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version", "run_id", "plan_id", "turn", "request_id", "kind"}
    assert len(schema["oneOf"]) == 2
    branches = json.dumps(schema["oneOf"])
    assert "forecast" in branches and "read" in branches
    assert "required" in schema["oneOf"][0] and "required" in schema["oneOf"][1]


# --------------------------------------------------------------------------- forecast branch


def test_a_forecast_message_is_decoded_by_the_original_core_codec() -> None:
    """C02: the original decoder still builds the Forecast and keeps its metadata."""
    request, _ = _build()
    message = _decode(request, _forecast_content(request))
    assert isinstance(message, codec.ForecastMessage)
    forecast = message.forecast
    assert forecast.forecast_id == request.request_id
    assert forecast.proposed_action.action_type == "mark_done"
    assert [item.outcome_type for item in forecast.expected_outcomes] == ["json_rule"]
    assert forecast.task_completion_hint is True
    assert forecast.metadata == {
        "planner": "llm", "run_id": RUN, "goal_id": "goal-1",
        "observation_id": OBS, "request_id": request.request_id,
    }


@pytest.mark.parametrize("body_kwargs", [
    {"action_type": "open_the_airlock"},
    {"outcome_type": "invented_outcome"},
])
def test_the_forecast_branch_still_obeys_the_capability_gates(body_kwargs: dict) -> None:
    """C02: an out-of-vocabulary action or outcome is refused by the original codec."""
    request, _ = _build()
    content = _forecast_content(request, **body_kwargs)
    error = _error(lambda: _decode(request, content))
    assert (error.code, error.stage) == ("invalid_response", "response")


def test_the_forecast_branch_refuses_an_extra_or_missing_body_field() -> None:
    """C02: the original Forecast shape is not widened by the envelope."""
    request, _ = _build()
    body = _forecast_body()
    body["injected"] = 1
    content = _envelope(request, "forecast", {"forecast": body})
    assert _error(lambda: _decode(request, content)).code == "invalid_response"
    body = _forecast_body()
    del body["expected_outcomes"]
    content = _envelope(request, "forecast", {"forecast": body})
    assert _error(lambda: _decode(request, content)).code == "invalid_response"


# --------------------------------------------------------------------------- read branch


def test_a_read_message_is_returned_with_its_requested_view() -> None:
    """C02: the read branch carries exactly the identity and the optional range."""
    request, _ = _build()
    message = _decode(request, _read_content(request))
    assert isinstance(message, codec.ReadMessage)
    assert (message.evidence_id, message.view_id, message.text_range) == (
        "evidence-1", "status", None)
    ranged = _decode(request, _read_content(
        request, text_range={"start": 1, "end": 3}))
    assert (ranged.text_range.start, ranged.text_range.end) == (1, 3)


@pytest.mark.parametrize("read_kwargs", [
    {"text_range": {"start": 3, "end": 1}},
    {"text_range": {"start": True, "end": 2}},
    {"text_range": {"start": 0, "end": 1.5}},
    {"text_range": {"start": -1, "end": 2}},
])
def test_a_read_range_is_strictly_ordered_non_negative_integers(read_kwargs: dict) -> None:
    """C02/C03: a bad interval is a malformed message, never a silent clamp."""
    request, _ = _build()
    content = _read_content(request, **read_kwargs)
    assert _error(lambda: _decode(request, content)).code == "invalid_response"


def test_a_read_message_requires_exactly_its_three_fields() -> None:
    """C02: a batch, a partial read or an invented field is not the declared shape."""
    request, _ = _build()
    content = _envelope(request, "read_evidence", {"read": {"evidence_id": "evidence-1"}})
    assert _error(lambda: _decode(request, content)).code == "invalid_response"
    content = _envelope(request, "read_evidence", {
        "read": {"evidence_id": "evidence-1", "view_id": "status", "text_range": None,
                 "extra": 1}})
    assert _error(lambda: _decode(request, content)).code == "invalid_response"
    content = _envelope(request, "read_evidence", {"read": [
        {"evidence_id": "evidence-1", "view_id": "status", "text_range": None}]})
    assert _error(lambda: _decode(request, content)).code == "invalid_response"


# --------------------------------------------------------------------------- envelope


@pytest.mark.parametrize("field, value, code", [
    ("run_id", "run-2", "binding_mismatch"),
    ("plan_id", "plan-2", "binding_mismatch"),
    ("turn", 2, "binding_mismatch"),
    ("request_id", "request-2", "binding_mismatch"),
    ("turn", True, "binding_mismatch"),
])
def test_the_envelope_must_match_run_plan_turn_and_request(
        field: str, value: object, code: str) -> None:
    """C02: a replayed or foreign answer is refused before its branch is used."""
    request, _ = _build()
    document = json.loads(_forecast_content(request))
    document[field] = value
    error = _error(lambda: _decode(request, json.dumps(document)))
    assert error.code == code


def test_the_envelope_refuses_an_invented_or_missing_common_field() -> None:
    """C02: every common field is required and no unknown field is allowed."""
    request, _ = _build()
    document = json.loads(_forecast_content(request))
    document["injected"] = 1
    assert _error(lambda: _decode(request, json.dumps(document))).code == "invalid_response"
    document = json.loads(_forecast_content(request))
    del document["turn"]
    assert _error(lambda: _decode(request, json.dumps(document))).code == "invalid_response"
    document = json.loads(_forecast_content(request))
    document["kind"] = "both"
    assert _error(lambda: _decode(request, json.dumps(document))).code == "invalid_response"


def test_the_envelope_refuses_crossing_or_mixing_the_two_branches() -> None:
    """C02: only the branch field of the declared kind may appear."""
    request, _ = _build()
    document = json.loads(_forecast_content(request))
    document["read"] = {"evidence_id": "evidence-1", "view_id": "status",
                        "text_range": None}
    assert _error(lambda: _decode(request, json.dumps(document))).code == "invalid_response"
    document = json.loads(_read_content(request))
    document["forecast"] = _forecast_body()
    assert _error(lambda: _decode(request, json.dumps(document))).code == "invalid_response"


@pytest.mark.parametrize("content", [
    '```json\n{}\n```',
    '{"a": 1} trailing',
    '{"schema_version": "0.1", "schema_version": "0.1"}',
    '[{"kind": "forecast"}]',
    '{"a": NaN}',
    '{"a": "\\ud800"}',
    "",
])
def test_content_must_be_one_strict_json_object(content: str) -> None:
    """C02/C05: fences, trailing text, duplicate keys, NaN and lone surrogates are refused."""
    request, _ = _build()
    error = _error(lambda: _decode(request, content))
    assert (error.code, error.stage) == ("invalid_response", "response")


def test_a_duplicate_key_inside_the_forecast_body_is_refused() -> None:
    """C02: the strict parse covers the nested body, not only the envelope."""
    request, _ = _build()
    content = ('{"schema_version": "0.1", "run_id": "run-1", "plan_id": "plan-1", '
               '"turn": 1, "request_id": "request-1", "kind": "forecast", "forecast": '
               '{"schema_version": "0.1", "schema_version": "0.1", "proposed_action": '
               '{"schema_version": "0.1", "action_type": "mark_done"}, '
               '"expected_outcomes": []}}')
    assert _error(lambda: _decode(request, content)).code == "invalid_response"


# --------------------------------------------------------------------------- response shell


def test_check_response_binds_the_response_to_the_current_request() -> None:
    """C02/C06: a response with another request identity is a binding failure."""
    request, _ = _build()
    response = ModelResponse(schema_version=SCHEMA, request_id="request-2", status="ok",
                             content=_forecast_content(request))
    error = _error(lambda: codec.check_response(
        response=response, request=request, config=_config()))
    assert (error.code, error.stage) == ("binding_mismatch", "binding")
    assert error.request_id == request.request_id


def test_a_refused_response_is_a_checked_shell_for_the_audit() -> None:
    """C02/C07: a refusal is a real returned shell, classified after it is recorded."""
    request, _ = _build()
    response = ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="refused")
    checked = codec.check_response(response=response, request=request, config=_config())
    assert (checked.response.status, checked.content_bytes) == ("refused", 0)


def test_the_response_size_and_token_budget_are_enforced() -> None:
    """C05: the content bytes and a declared output overrun are fixed refusals."""
    request, _ = _build()
    huge = ModelResponse(schema_version=SCHEMA, request_id=request.request_id, status="ok",
                         content="x" * 70_000)
    assert _error(lambda: codec.check_response(
        response=huge, request=request, config=_config())).code == "invalid_response"
    over = ModelResponse(
        schema_version=SCHEMA, request_id=request.request_id, status="ok",
        content=_forecast_content(request),
        usage=ModelUsage(schema_version=SCHEMA, input_tokens=1, output_tokens=4096))
    error = _error(lambda: codec.check_response(
        response=over, request=request, config=_config()))
    assert (error.code, error.stage) == ("budget_exceeded", "budget")


def test_a_missing_usage_is_legal_and_unknown() -> None:
    """C05: a missing usage is not a zero cost and is not fabricated."""
    request, _ = _build()
    response = ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="ok", content=_forecast_content(request))
    checked = codec.check_response(response=response, request=request, config=_config())
    assert checked.response.usage is None
    assert checked.content_bytes > 0


def test_a_non_response_record_is_a_provider_failure() -> None:
    """C06: a facility that does not return the declared shell is normalised."""
    request, _ = _build()
    error = _error(lambda: codec.check_response(
        response=object(), request=request, config=_config()))
    assert (error.code, error.stage) == ("provider_failed", "provider")


def test_the_request_is_not_mutated_by_a_decode() -> None:
    """C02: the planner's own request record is never rewritten by the codec."""
    request, _ = _build()
    before = request.model_dump(mode="json")
    _decode(request, _forecast_content(request))
    assert request.model_dump(mode="json") == before
