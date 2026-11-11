"""The strict boundary between a model answer and the Core action/observation vocabulary.

``build_request`` turns the trusted planning context, the complete current observation and
the caller's limits into one :class:`~forecontract_llm.models.ModelRequest`. ``decode_response``
turns one provider answer into one :class:`forecontract.models.actions.Forecast`, or raises a
:class:`~forecontract_llm.errors.PlannerError`.

Nothing here calls a provider, reads an environment variable, imports an SDK or performs I/O.
The response side is strict in both directions: content must be exactly one JSON object with
no duplicate key, no non-finite number, no lone surrogate, no trailing text and no Markdown
fence, and the decoded action and outcomes must stay inside the declared capabilities. A
malformed or out-of-vocabulary answer is the fixed facility error, never a coerced action.
"""

from __future__ import annotations

import json
import math
from typing import Any

from forecontract.models.actions import ActionSpec, ExpectedOutcome, Forecast
from forecontract.models.observations import Observation
from pydantic import ValidationError

from forecontract_llm.errors import PlannerError
from forecontract_llm.models import (
    ModelRequest,
    ModelResponse,
    PlannerConfig,
    PlannerContext,
    measure_json,
    read_declared_copy,
    request_byte_length,
)

__all__ = ["build_request", "decode_response"]

_SCHEMA_VERSION = "0.1"
_REQUIRED_TOP = frozenset({"schema_version", "proposed_action", "expected_outcomes"})
_ALLOWED_TOP = _REQUIRED_TOP | frozenset(
    {"intent", "decision_summary", "confidence", "task_completion_hint"}
)


class _RejectedJSON(ValueError):
    """An internal marker for content the strict parser must refuse."""


def _fail(code: str, message: str, location: tuple[str | int, ...]) -> PlannerError:
    return PlannerError(code, message, location=location)


def _action_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "action_type"],
        "properties": {
            "schema_version": {"const": _SCHEMA_VERSION},
            "action_type": {"type": "string", "minLength": 1},
            "payload": {"type": "object"},
            "metadata": {"type": "object", "maxProperties": 0},
        },
    }


def _outcome_schema() -> dict[str, object]:
    """The advertised shape of one ``forecontract.models.actions.ExpectedOutcome``.

    Core's ``ExpectedOutcome`` has no ``metadata`` field, so the schema advertised to the model
    must not invent one; a response that carries it is refused as an undeclared field.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "outcome_type"],
        "properties": {
            "schema_version": {"const": _SCHEMA_VERSION},
            "outcome_type": {"type": "string", "minLength": 1},
            "payload": {"type": "object"},
            "required": {"type": "boolean"},
        },
    }


def _response_schema() -> dict[str, object]:
    """Describe the single legal response shape for the provider.

    The schema is documentation handed to the provider; the decoder below enforces the same
    rules directly and never trusts the schema to have been honoured.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "proposed_action", "expected_outcomes"],
        "properties": {
            "schema_version": {"const": _SCHEMA_VERSION},
            "proposed_action": _action_schema(),
            "expected_outcomes": {
                "type": "array",
                "minItems": 1,
                "items": _outcome_schema(),
            },
            "intent": {"type": ["string", "null"]},
            "decision_summary": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "task_completion_hint": {"type": ["boolean", "null"]},
        },
    }


def build_request(
    *,
    context: PlannerContext,
    observation: Observation,
    config: PlannerConfig,
    request_id: str,
) -> ModelRequest:
    """Build and measure one structured request from strictly revalidated inputs.

    The context, configuration and observation are revalidated and copied here, so the codec
    is not a way around the planner's input gates. The observation environment must match the
    declared capability environment, and the assembled request must stay inside the configured
    byte, depth and node limits before any provider can be called.
    """
    planner_context = read_declared_copy(context, PlannerContext, "INPUT_INVALID")
    planner_config = read_declared_copy(config, PlannerConfig, "INPUT_INVALID")
    current = read_declared_copy(observation, Observation, "INPUT_INVALID")
    if not isinstance(request_id, str) or not request_id.strip():
        raise _fail("INPUT_INVALID", "the request identifier must be a non-blank string",
                    ("request_id",))
    if current.environment_type != planner_context.capabilities.environment_type:
        raise _fail(
            "CONTEXT_MISMATCH",
            "the observation environment does not match the declared capability environment",
            ("environment_type",),
        )
    schema = _response_schema()
    measure_json(
        schema,
        max_depth=planner_config.max_json_depth,
        max_nodes=planner_config.max_json_nodes,
        code="REQUEST_TOO_LARGE",
        location=("response_schema",),
    )
    try:
        request = ModelRequest(
            schema_version=_SCHEMA_VERSION,
            request_id=request_id,
            model_context=planner_context,
            observation=current,
            response_schema=schema,
            max_output_tokens=planner_config.max_output_tokens,
        )
    except ValidationError:
        raise _fail("INPUT_INVALID", "the assembled request does not match the declared schema",
                    ("request",)) from None
    payload = request.model_dump(mode="json")
    measure_json(
        payload,
        max_depth=planner_config.max_json_depth,
        max_nodes=planner_config.max_json_nodes,
        code="REQUEST_TOO_LARGE",
        location=("request",),
    )
    if request_byte_length(request) > planner_config.max_request_bytes:
        raise _fail("REQUEST_TOO_LARGE", "the request exceeds the configured byte limit",
                    ("bytes",))
    return request


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _RejectedJSON("duplicate key")
        result[key] = value
    return result


def _reject_constant(_text: str) -> Any:
    raise _RejectedJSON("non-finite number")


def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in text)


def _check_surrogates(value: object) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _has_surrogate(item):
                raise _fail("RESPONSE_INVALID", "the response contains an invalid Unicode string",
                            ("content",))
        elif isinstance(item, dict):
            for key, child in item.items():
                if _has_surrogate(key):
                    raise _fail("RESPONSE_INVALID",
                                "the response contains an invalid Unicode string", ("content",))
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)


def _raw_depth_within(text: str, limit: int) -> bool:
    """Cheaply bound the nesting depth before the recursive JSON parser runs."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > limit:
                return False
        elif character in "}]":
            depth -= 1
    return True


def _parse_content(content: str, config: PlannerConfig) -> object:
    if not _raw_depth_within(content, config.max_json_depth):
        raise _fail("RESPONSE_INVALID", "the response exceeds the configured JSON depth limit",
                    ("content",))
    try:
        return json.loads(
            content,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError):
        raise _fail("RESPONSE_INVALID", "the response content is not legal JSON",
                    ("content",)) from None


def _empty_metadata(raw: dict[str, Any]) -> None:
    if "metadata" not in raw:
        return
    metadata = raw["metadata"]
    if not isinstance(metadata, dict) or metadata:
        raise _fail("RESPONSE_INVALID", "a nested Core record must not carry metadata",
                    ("metadata",))


def _reject_outcome_metadata(raw: dict[str, Any]) -> None:
    """Refuse any ``metadata`` key on an expected outcome: Core has no such field."""
    if "metadata" in raw:
        raise _fail("RESPONSE_INVALID",
                    "an expected outcome must not carry a metadata field", ("metadata",))


def _optional_text(parsed: dict[str, Any], key: str) -> str | None:
    if key not in parsed:
        return None
    value = parsed[key]
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise _fail("RESPONSE_INVALID", "an optional text field must be a string or null",
                ("content",))


def _optional_flag(parsed: dict[str, Any], key: str) -> bool | None:
    if key not in parsed:
        return None
    value = parsed[key]
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise _fail("RESPONSE_INVALID", "an optional flag must be a boolean or null",
                ("content",))


def _optional_confidence(parsed: dict[str, Any]) -> float | None:
    if "confidence" not in parsed:
        return None
    value = parsed["confidence"]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail("RESPONSE_INVALID", "confidence must be a finite number between zero and one",
                    ("content",))
    try:
        number = float(value)
    except OverflowError:
        raise _fail("RESPONSE_INVALID", "confidence must be a finite number between zero and one",
                    ("content",)) from None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise _fail("RESPONSE_INVALID", "confidence must be a finite number between zero and one",
                    ("content",))
    return number


def _decode_body(
    parsed: dict[str, Any],
) -> tuple[ActionSpec, list[ExpectedOutcome], str | None, str | None, float | None, bool | None]:
    unknown = set(parsed) - _ALLOWED_TOP
    if unknown:
        raise _fail("RESPONSE_INVALID", "the response carries an undeclared top-level key",
                    ("schema",))
    missing = _REQUIRED_TOP - set(parsed)
    if missing:
        raise _fail("RESPONSE_INVALID", "the response is missing a required top-level key",
                    ("schema",))
    if not isinstance(parsed.get("schema_version"), str):
        raise _fail("RESPONSE_INVALID", "the response schema version is not the declared string",
                    ("schema_version",))
    if parsed["schema_version"] != _SCHEMA_VERSION:
        raise _fail("RESPONSE_INVALID", "the response schema version is not the declared value",
                    ("schema_version",))
    action_raw = parsed["proposed_action"]
    if not isinstance(action_raw, dict):
        raise _fail("RESPONSE_INVALID", "the proposed action is not a JSON object",
                    ("action",))
    _empty_metadata(action_raw)
    try:
        action = ActionSpec.model_validate(action_raw)
    except ValidationError:
        raise _fail("RESPONSE_INVALID", "the proposed action does not match the Core shape",
                    ("action",)) from None
    outcomes_raw = parsed["expected_outcomes"]
    if not isinstance(outcomes_raw, list) or not outcomes_raw:
        raise _fail("RESPONSE_INVALID",
                    "expected_outcomes must be a non-empty JSON list", ("outcomes",))
    outcomes: list[ExpectedOutcome] = []
    for item in outcomes_raw:
        if not isinstance(item, dict):
            raise _fail("RESPONSE_INVALID", "an expected outcome is not a JSON object",
                        ("outcomes",))
        _reject_outcome_metadata(item)
        try:
            outcomes.append(ExpectedOutcome.model_validate(item))
        except ValidationError:
            raise _fail("RESPONSE_INVALID",
                        "an expected outcome does not match the Core shape", ("outcomes",)) from None
    intent = _optional_text(parsed, "intent")
    decision_summary = _optional_text(parsed, "decision_summary")
    confidence = _optional_confidence(parsed)
    task_completion_hint = _optional_flag(parsed, "task_completion_hint")
    return action, outcomes, intent, decision_summary, confidence, task_completion_hint


def decode_response(
    *,
    response: ModelResponse,
    request: ModelRequest,
    config: PlannerConfig,
) -> Forecast:
    """Decode exactly one provider answer into one Core forecast.

    The checks run in the agreed order: response shell and binding, refusal, content bytes and
    syntax, strict shape, capability bounds, declared output-token budget. Every rejection is
    the fixed facility error; no partial forecast is ever returned.
    """
    current_request = read_declared_copy(request, ModelRequest, "INPUT_INVALID")
    planner_config = read_declared_copy(config, PlannerConfig, "INPUT_INVALID")
    if not isinstance(response, ModelResponse):
        raise _fail("RESPONSE_INVALID", "the provider did not return a response record",
                    ("response",))
    current_response = read_declared_copy(response, ModelResponse, "RESPONSE_INVALID")
    if current_response.request_id != current_request.request_id:
        raise _fail("RESPONSE_INVALID", "the response does not belong to the current request",
                    ("request_id",))
    if current_response.status == "refused":
        raise _fail("PROVIDER_REFUSED", "the provider refused the request", ("response",))
    content = current_response.content
    if not isinstance(content, str):
        raise _fail("RESPONSE_INVALID", "an ok response must carry string content",
                    ("content",))
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError:
        raise _fail("RESPONSE_INVALID", "the response contains an invalid Unicode string",
                    ("content",)) from None
    if len(encoded) > planner_config.max_response_bytes:
        raise _fail("RESPONSE_TOO_LARGE", "the response exceeds the configured byte limit",
                    ("bytes",))
    parsed = _parse_content(content, planner_config)
    _check_surrogates(parsed)
    if not isinstance(parsed, dict):
        raise _fail("RESPONSE_INVALID", "the response content is not a JSON object",
                    ("content",))
    measure_json(
        parsed,
        max_depth=planner_config.max_json_depth,
        max_nodes=planner_config.max_json_nodes,
        code="RESPONSE_INVALID",
        location=("content",),
    )
    action, outcomes, intent, decision_summary, confidence, hint = _decode_body(parsed)
    capabilities = current_request.model_context.capabilities
    if action.action_type not in capabilities.supported_actions:
        raise _fail("UNSUPPORTED_ACTION", "the proposed action is outside the declared capabilities",
                    ("action",))
    for index, outcome in enumerate(outcomes):
        if outcome.outcome_type not in capabilities.supported_outcome_types:
            raise _fail(
                "UNSUPPORTED_OUTCOME",
                "a proposed outcome is outside the declared capabilities",
                ("outcomes", index),
            )
    usage = current_response.usage
    if usage is not None and usage.output_tokens > planner_config.max_output_tokens:
        raise _fail("BUDGET_EXCEEDED", "the declared output-token budget was exceeded",
                    ("usage", "output_tokens"))
    context = current_request.model_context
    return Forecast(
        schema_version=_SCHEMA_VERSION,
        forecast_id=current_request.request_id,
        proposed_action=action,
        expected_outcomes=outcomes,
        intent=intent,
        decision_summary=decision_summary,
        confidence=confidence,
        task_completion_hint=hint,
        metadata={
            "planner": "llm",
            "run_id": context.run_id,
            "goal_id": context.goal_id,
            "observation_id": current_request.observation.observation_id,
            "request_id": current_request.request_id,
        },
    )
