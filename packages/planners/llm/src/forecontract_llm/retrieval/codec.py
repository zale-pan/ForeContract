"""The strict message protocol between the planner and one retrieval provider.

``build_request`` composes the complete, measured :class:`RetrievalModelRequest` from the
trusted context, the current observation, the frozen evidence directory and every read result
already obtained. The Core part is built by the original ``forecontract_llm.codec.build_request``
and then re-wrapped: the same request identifier, the same Run and the same response schema
appear in both the envelope and its ``base_request``.

``decode_message`` accepts exactly one of the two declared message shapes. Every common field
is required, the branch field of the other kind is forbidden, duplicate keys, non-finite
numbers, lone surrogates, code fences and trailing text are refused, and the run/plan/turn/
request binding must match the current request. A forecast body is handed back to the original
codec through a restored request whose response schema is the original Forecast schema, so the
original action, outcome and capability gates still decide.

This module performs no I/O and never rewrites a caller's record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

from forecontract.models.actions import Forecast
from pydantic import ValidationError

from forecontract_llm import codec as base_codec
from forecontract_llm.errors import PlannerError
from forecontract_llm.models import ModelRequest, ModelResponse

from .errors import RetrievalError
from .models import (
    OUTPUT_TOKENS_MAX,
    SCHEMA_VERSION,
    EvidenceReadResult,
    EvidenceScope,
    RetrievalConfig,
    RetrievalModelRequest,
    TextRange,
    copy_record,
    effective_base_config,
    measure_record,
    parse_strict_json,
)

__all__ = [
    "CheckedResponse",
    "ForecastMessage",
    "ReadMessage",
    "build_request",
    "check_response",
    "decode_message",
    "response_schema",
]

#: The two legal message kinds.
MESSAGE_KINDS: Final[tuple[str, ...]] = ("forecast", "read_evidence")

_COMMON_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "run_id",
    "plan_id",
    "turn",
    "request_id",
    "kind",
)

_PLANNER_TO_RETRIEVAL: Final[dict[str, tuple[str, str]]] = {
    "PROVIDER_REFUSED": ("provider_refused", "provider"),
    "BUDGET_EXCEEDED": ("budget_exceeded", "budget"),
    "TIMEOUT": ("timeout", "deadline"),
    "CONTEXT_MISMATCH": ("binding_mismatch", "context"),
    "REQUEST_TOO_LARGE": ("budget_exceeded", "request"),
}


@dataclass(frozen=True, slots=True)
class ForecastMessage:
    """A decoded forecast branch; the forecast already passed the original Core codec."""

    forecast: Forecast


@dataclass(frozen=True, slots=True)
class ReadMessage:
    """A decoded read branch, exactly as the provider requested it."""

    evidence_id: str
    view_id: str
    text_range: TextRange | None


def response_schema(forecast_schema: dict[str, object]) -> dict[str, object]:
    """Describe the single legal retrieval message, embedding the original Forecast schema.

    The schema is documentation handed to the provider. The decoder below enforces the same
    rules directly and never trusts the schema to have been honoured.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_COMMON_FIELDS),
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "run_id": {"type": "string", "minLength": 1},
            "plan_id": {"type": "string", "minLength": 1},
            "turn": {"type": "integer", "minimum": 1},
            "request_id": {"type": "string", "minLength": 1},
            "kind": {"enum": list(MESSAGE_KINDS)},
            "forecast": forecast_schema,
            "read": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "view_id", "text_range"],
                "properties": {
                    "evidence_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "view_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "text_range": {
                        "oneOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["start", "end"],
                                "properties": {
                                    "start": {"type": "integer", "minimum": 0},
                                    "end": {"type": "integer", "minimum": 0},
                                },
                            },
                        ]
                    },
                },
            },
        },
        "oneOf": [
            {
                "required": ["kind", "forecast"],
                "not": {"required": ["read"]},
                "properties": {"kind": {"const": "forecast"}},
            },
            {
                "required": ["kind", "read"],
                "not": {"required": ["forecast"]},
                "properties": {"kind": {"const": "read_evidence"}},
            },
        ],
    }


def _normalise(
    error: PlannerError,
    *,
    stage: str,
    default: tuple[str, str] | None = None,
) -> RetrievalError:
    fallback = default if default is not None else ("invalid_input", stage)
    code, mapped_stage = _PLANNER_TO_RETRIEVAL.get(error.code, fallback)
    return RetrievalError(code, mapped_stage)


def build_request(
    *,
    context: object,
    observation: object,
    config: RetrievalConfig,
    request_id: str,
    plan_id: str,
    turn: int,
    scope: EvidenceScope,
    exchanges: list[EvidenceReadResult],
) -> tuple[RetrievalModelRequest, int]:
    """Build and measure one complete retrieval request; return it with its byte length.

    The original Core codec builds and measures the base request, so the declared context,
    observation, byte, depth and node gates are the already-verified ones. The envelope then
    carries the frozen scope and the previous read results; their complete content is charged,
    never silently trimmed.
    """
    current_config = copy_record(config, RetrievalConfig, "invalid_input", "input")
    current_scope = copy_record(scope, EvidenceScope, "invalid_input", "input")
    base = effective_base_config(current_config.base)
    try:
        base_request = base_codec.build_request(
            context=context,
            observation=observation,
            config=base,
            request_id=request_id,
        )
    except PlannerError as exc:
        raise _normalise(exc, stage="request") from None
    except (ValidationError, TypeError, ValueError):
        raise RetrievalError("invalid_input", "input") from None
    run_id = base_request.model_context.run_id
    if current_scope.run_id != run_id:
        raise RetrievalError("binding_mismatch", "binding")
    if current_scope.observation_id != base_request.observation.observation_id:
        raise RetrievalError("binding_mismatch", "binding")
    schema = response_schema(base_request.response_schema)
    try:
        measured_base = ModelRequest(
            schema_version=SCHEMA_VERSION,
            request_id=request_id,
            model_context=base_request.model_context,
            observation=base_request.observation,
            response_schema=schema,
            max_output_tokens=min(base_request.max_output_tokens, OUTPUT_TOKENS_MAX),
        )
    except (ValidationError, TypeError, ValueError):
        raise RetrievalError("invalid_input", "request") from None
    try:
        request = RetrievalModelRequest(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            plan_id=plan_id,
            turn=turn,
            request_id=request_id,
            base_request=measured_base,
            scope=current_scope,
            exchanges=[copy_record(item, EvidenceReadResult, "invalid_input", "input")
                       for item in exchanges],
            response_schema=schema,
        )
    except (ValidationError, TypeError, ValueError):
        raise RetrievalError("invalid_input", "request") from None
    size = measure_record(
        request,
        max_bytes=base.max_request_bytes,
        max_depth=base.max_json_depth,
        max_nodes=base.max_json_nodes,
        code="budget_exceeded",
        stage="request",
    )
    return request, size


@dataclass(frozen=True, slots=True)
class CheckedResponse:
    """A revalidated response shell and the measured size of its content."""

    response: ModelResponse
    content_bytes: int


def check_response(
    *,
    response: object,
    request: RetrievalModelRequest,
    config: RetrievalConfig,
) -> CheckedResponse:
    """Validate the response shell, its binding and its size; return the revalidated copy."""
    current_config = copy_record(config, RetrievalConfig, "invalid_input", "input")
    base = effective_base_config(current_config.base)
    if not isinstance(response, ModelResponse):
        raise RetrievalError("provider_failed", "provider")
    current = copy_record(response, ModelResponse, "invalid_response", "response")
    if current.request_id != request.request_id:
        raise RetrievalError("binding_mismatch", "binding", request_id=request.request_id)
    usage = current.usage
    if usage is not None and usage.output_tokens > request.base_request.max_output_tokens:
        raise RetrievalError("budget_exceeded", "budget", request_id=request.request_id)
    if current.status == "refused":
        return CheckedResponse(response=current, content_bytes=0)
    content = current.content
    if not isinstance(content, str):
        raise RetrievalError("invalid_response", "response", request_id=request.request_id)
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError:
        raise RetrievalError("invalid_response", "response", request_id=request.request_id,
                             ) from None
    if len(encoded) > base.max_response_bytes:
        raise RetrievalError("invalid_response", "response", request_id=request.request_id)
    return CheckedResponse(response=current, content_bytes=len(encoded))


def _decode_forecast(
    body: dict[str, object],
    request: RetrievalModelRequest,
    config: RetrievalConfig,
) -> Forecast:
    """Restore the original codec's response schema and let the original gate decide."""
    base = effective_base_config(config.base)
    try:
        restored = base_codec.build_request(
            context=request.base_request.model_context,
            observation=request.base_request.observation,
            config=base,
            request_id=request.request_id,
        )
    except PlannerError as exc:
        raise _normalise(exc, stage="response",
                         default=("invalid_response", "response")) from None
    except (ValidationError, TypeError, ValueError):
        raise RetrievalError("invalid_response", "response") from None
    try:
        content = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False)
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        raise RetrievalError("invalid_response", "response") from None
    try:
        response = ModelResponse(
            schema_version=SCHEMA_VERSION,
            request_id=request.request_id,
            status="ok",
            content=content,
        )
    except (ValidationError, TypeError, ValueError):
        raise RetrievalError("invalid_response", "response") from None
    try:
        forecast = base_codec.decode_response(
            response=response, request=restored, config=base)
    except PlannerError as exc:
        raise _normalise(exc, stage="response",
                         default=("invalid_response", "response")) from None
    except (ValidationError, TypeError, ValueError):
        raise RetrievalError("invalid_response", "response") from None
    if forecast.forecast_id != request.request_id:
        raise RetrievalError("invalid_response", "response")
    return forecast


def _is_strict_int(value: object) -> bool:
    return type(value) is int


def _decode_read(raw: object) -> ReadMessage:
    if not isinstance(raw, dict):
        raise RetrievalError("invalid_response", "response")
    if set(raw) != {"evidence_id", "view_id", "text_range"}:
        raise RetrievalError("invalid_response", "response")
    evidence_id = raw["evidence_id"]
    view_id = raw["view_id"]
    for value in (evidence_id, view_id):
        if type(value) is not str or not 1 <= len(value) <= 128:
            raise RetrievalError("invalid_response", "response")
    text_range_raw = raw["text_range"]
    if text_range_raw is None:
        return ReadMessage(evidence_id=evidence_id, view_id=view_id, text_range=None)
    if not isinstance(text_range_raw, dict) or set(text_range_raw) != {"start", "end"}:
        raise RetrievalError("invalid_response", "response")
    start = text_range_raw["start"]
    end = text_range_raw["end"]
    if not _is_strict_int(start) or not _is_strict_int(end) or start < 0 or end < start:
        raise RetrievalError("invalid_response", "response")
    try:
        text_range = TextRange(schema_version=SCHEMA_VERSION, start=start, end=end)
    except (ValidationError, TypeError, ValueError):
        raise RetrievalError("invalid_response", "response") from None
    return ReadMessage(evidence_id=evidence_id, view_id=view_id, text_range=text_range)


def decode_message(
    *,
    content: str,
    request: RetrievalModelRequest,
    config: RetrievalConfig,
) -> ForecastMessage | ReadMessage:
    """Decode exactly one of the two legal messages, bound to the current request."""
    current_config = copy_record(config, RetrievalConfig, "invalid_input", "input")
    base = effective_base_config(current_config.base)
    parsed = _parse(content, base.max_json_depth, base.max_json_nodes)
    if not isinstance(parsed, dict):
        raise RetrievalError("invalid_response", "response")
    kind = parsed.get("kind")
    if kind not in MESSAGE_KINDS:
        raise RetrievalError("invalid_response", "response")
    branch = "forecast" if kind == "forecast" else "read"
    other = "read" if kind == "forecast" else "forecast"
    allowed = set(_COMMON_FIELDS) | {branch}
    if set(parsed) - allowed or other in parsed:
        raise RetrievalError("invalid_response", "response")
    for name in _COMMON_FIELDS:
        if name not in parsed:
            raise RetrievalError("invalid_response", "response")
    if parsed["schema_version"] != SCHEMA_VERSION:
        raise RetrievalError("invalid_response", "response")
    for name, expected in (("run_id", request.run_id), ("plan_id", request.plan_id),
                           ("request_id", request.request_id)):
        if parsed[name] != expected:
            raise RetrievalError("binding_mismatch", "binding", request_id=request.request_id)
    turn = parsed["turn"]
    if not _is_strict_int(turn) or turn != request.turn:
        raise RetrievalError("binding_mismatch", "binding", request_id=request.request_id)
    if branch not in parsed:
        raise RetrievalError("invalid_response", "response")
    if kind == "read_evidence":
        return _decode_read(parsed["read"])
    body = parsed["forecast"]
    if not isinstance(body, dict):
        raise RetrievalError("invalid_response", "response")
    return ForecastMessage(forecast=_decode_forecast(body, request, current_config))


def _parse(content: str, max_depth: int, max_nodes: int) -> object:
    """Strict JSON parse of one response content, with the agreed error classification."""
    return parse_strict_json(content, max_depth=max_depth, max_nodes=max_nodes,
                             code="invalid_response", stage="response")
