"""Internal validation records and helpers for offline verification extensions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from forecontract.models.outcomes import VerificationStatus
from forecontract.verification.errors import VerificationError

CompositeItemKind = Literal["deterministic", "model"]
CompositeMode = Literal["all", "any", "weighted"]
ModelResponseState = Literal["ok", "refused", "unavailable", "timeout"]

COMPOSITE_ITEM_KEYS: tuple[str, ...] = (
    "id",
    "status",
    "kind",
    "critical",
    "weight",
    "source_id",
    "reason",
)
MODEL_IDENTITY_KEYS: tuple[str, ...] = (
    "request_id",
    "model_id",
    "prompt_version",
    "evidence_id",
)
MODEL_RESPONSE_KEYS: tuple[str, ...] = (
    *MODEL_IDENTITY_KEYS,
    "state",
    "response_id",
    "status",
    "confidence",
    "reason",
)
VALID_STATUS_VALUES: frozenset[str] = frozenset({"pass", "fail", "inconclusive"})
VALID_ITEM_KINDS: frozenset[str] = frozenset({"deterministic", "model"})
VALID_COMPOSITE_MODES: frozenset[str] = frozenset({"all", "any", "weighted"})
VALID_MODEL_STATES: frozenset[str] = frozenset({"ok", "refused", "unavailable", "timeout"})


@dataclass(frozen=True, slots=True)
class CompositeItemRecord:
    """Validated internal representation of one composite verification item."""

    id: str
    status: VerificationStatus
    kind: CompositeItemKind
    critical: bool
    weight: float
    source_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ValidatedCompositeInput:
    """Validated internal representation of a composite verification call."""

    items: tuple[CompositeItemRecord, ...]
    mode: CompositeMode
    threshold: float | None


@dataclass(frozen=True, slots=True)
class ValidatedModelInput:
    """Validated internal representation of an offline model verification call."""

    state: ModelResponseState
    status: VerificationStatus | None
    confidence: float | None
    reason: str


def _require_exact_dict_keys(
    raw: object,
    *,
    required_keys: tuple[str, ...],
    location: tuple[str | int, ...],
    object_message: str,
) -> dict[str, object]:
    if type(raw) is not dict:
        raise VerificationError("VERIFICATION_INPUT_INVALID", location, object_message)
    for key in raw:
        if type(key) is not str:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                location,
                "mapping keys must be strings",
            )
    for key in required_keys:
        if key not in raw:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                (*location, key),
                "required field is missing",
            )
    allowed = frozenset(required_keys)
    for key in raw:
        if key not in allowed:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                (*location, "<unknown_field>"),
                "unexpected field is not allowed",
            )
    return raw


def _require_strict_str(
    value: object,
    *,
    min_len: int,
    max_len: int,
    location: tuple[str | int, ...],
    message: str,
) -> str:
    if type(value) is not str:
        raise VerificationError("VERIFICATION_INPUT_INVALID", location, message)
    if not (min_len <= len(value) <= max_len) or not value.strip():
        raise VerificationError("VERIFICATION_INPUT_INVALID", location, message)
    return value


def _require_finite_real(
    value: object,
    *,
    location: tuple[str | int, ...],
    code: str,
    message: str,
) -> float:
    if isinstance(value, bool):
        raise VerificationError(code, location, message)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VerificationError(code, location, message)
        return value
    raise VerificationError(code, location, message)


def validate_composite_input(
    *,
    items: list[dict[str, object]],
    mode: str,
    threshold: float | None,
) -> ValidatedCompositeInput:
    """Validate structure, bindings and aggregation rules in strict order."""
    if type(items) is not list or not (1 <= len(items) <= 128):
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("items",),
            "items must be a plain list of 1 to 128 entries",
        )

    records: list[CompositeItemRecord] = []
    for index, raw_item in enumerate(items):
        item_loc: tuple[str | int, ...] = ("items", index)
        mapping = _require_exact_dict_keys(
            raw_item,
            required_keys=COMPOSITE_ITEM_KEYS,
            location=item_loc,
            object_message="each composite item must be a mapping",
        )
        item_id = _require_strict_str(
            mapping["id"],
            min_len=1,
            max_len=128,
            location=("items", index, "id"),
            message="item id must be a non-blank string up to 128 characters",
        )
        raw_status = mapping["status"]
        if type(raw_status) is not str or raw_status not in VALID_STATUS_VALUES:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("items", index, "status"),
                "item status must be pass, fail or inconclusive",
            )
        raw_kind = mapping["kind"]
        if type(raw_kind) is not str or raw_kind not in VALID_ITEM_KINDS:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("items", index, "kind"),
                "item kind must be deterministic or model",
            )
        raw_critical = mapping["critical"]
        if type(raw_critical) is not bool:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("items", index, "critical"),
                "item critical flag must be a boolean",
            )
        weight = _require_finite_real(
            mapping["weight"],
            location=("items", index, "weight"),
            code="VERIFICATION_INPUT_INVALID",
            message="item weight must be a finite non-negative number",
        )
        if weight < 0:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("items", index, "weight"),
                "item weight must be a finite non-negative number",
            )
        source_id = _require_strict_str(
            mapping["source_id"],
            min_len=1,
            max_len=128,
            location=("items", index, "source_id"),
            message="item source_id must be a non-blank string up to 128 characters",
        )
        reason = _require_strict_str(
            mapping["reason"],
            min_len=1,
            max_len=1024,
            location=("items", index, "reason"),
            message="item reason must be a non-blank string up to 1024 characters",
        )
        kind_literal: CompositeItemKind = "deterministic" if raw_kind == "deterministic" else "model"
        records.append(
            CompositeItemRecord(
                id=item_id,
                status=VerificationStatus(raw_status),
                kind=kind_literal,
                critical=raw_critical,
                weight=weight,
                source_id=source_id,
                reason=reason,
            )
        )

    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if record.kind == "model" and record.critical:
            raise VerificationError(
                "VERIFICATION_BINDING_INVALID",
                ("items", index, "critical"),
                "a model verification item cannot be marked critical",
            )
        if record.id in seen_ids:
            raise VerificationError(
                "VERIFICATION_BINDING_INVALID",
                ("items", index, "id"),
                "composite item identifiers must be unique",
            )
        seen_ids.add(record.id)

    if type(mode) is not str or mode not in VALID_COMPOSITE_MODES:
        raise VerificationError(
            "VERIFICATION_RULE_INVALID",
            ("mode",),
            "aggregation mode must be all, any or weighted",
        )

    mode_literal: CompositeMode
    if mode == "all":
        mode_literal = "all"
    elif mode == "any":
        mode_literal = "any"
    else:
        mode_literal = "weighted"

    if mode_literal in ("all", "any"):
        if threshold is not None:
            raise VerificationError(
                "VERIFICATION_RULE_INVALID",
                ("threshold",),
                "threshold is only permitted in weighted mode",
            )
        validated_threshold: float | None = None
    else:
        raw_threshold = _require_finite_real(
            threshold,
            location=("threshold",),
            code="VERIFICATION_RULE_INVALID",
            message="weighted mode requires a finite threshold in (0, 1]",
        )
        if not (0 < raw_threshold <= 1):
            raise VerificationError(
                "VERIFICATION_RULE_INVALID",
                ("threshold",),
                "weighted mode requires a finite threshold in (0, 1]",
            )
        validated_threshold = float(raw_threshold)
        if not any(record.weight > 0 for record in records):
            raise VerificationError(
                "VERIFICATION_RULE_INVALID",
                ("items",),
                "weighted mode requires a positive total weight",
            )

    return ValidatedCompositeInput(
        items=tuple(records),
        mode=mode_literal,
        threshold=validated_threshold,
    )


def validate_model_input(
    *,
    request: dict[str, object],
    response: dict[str, object],
) -> ValidatedModelInput:
    """Validate offline model request and response structure before identity binding."""
    req_map = _require_exact_dict_keys(
        request,
        required_keys=MODEL_IDENTITY_KEYS,
        location=("request",),
        object_message="model request must be a mapping",
    )
    for key in MODEL_IDENTITY_KEYS:
        _require_strict_str(
            req_map[key],
            min_len=1,
            max_len=128,
            location=("request", key),
            message="request identifier must be a non-blank string up to 128 characters",
        )

    resp_map = _require_exact_dict_keys(
        response,
        required_keys=MODEL_RESPONSE_KEYS,
        location=("response",),
        object_message="model response must be a mapping",
    )
    for key in MODEL_IDENTITY_KEYS:
        _require_strict_str(
            resp_map[key],
            min_len=1,
            max_len=128,
            location=("response", key),
            message="response identifier must be a non-blank string up to 128 characters",
        )

    raw_state = resp_map["state"]
    if type(raw_state) is not str or raw_state not in VALID_MODEL_STATES:
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            ("response", "state"),
            "response state must be ok, refused, unavailable or timeout",
        )

    reason = _require_strict_str(
        resp_map["reason"],
        min_len=1,
        max_len=1024,
        location=("response", "reason"),
        message="response reason must be a non-blank string up to 1024 characters",
    )

    state_literal: ModelResponseState
    validated_status: VerificationStatus | None
    validated_confidence: float | None

    if raw_state == "ok":
        state_literal = "ok"
        _require_strict_str(
            resp_map["response_id"],
            min_len=1,
            max_len=128,
            location=("response", "response_id"),
            message="ok response requires a non-blank response_id up to 128 characters",
        )
        raw_status = resp_map["status"]
        if type(raw_status) is not str or raw_status not in VALID_STATUS_VALUES:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("response", "status"),
                "ok response status must be pass, fail or inconclusive",
            )
        validated_status = VerificationStatus(raw_status)
        raw_confidence = resp_map["confidence"]
        if raw_confidence is None:
            validated_confidence = None
        else:
            confidence_val = _require_finite_real(
                raw_confidence,
                location=("response", "confidence"),
                code="VERIFICATION_INPUT_INVALID",
                message="response confidence must be None or a finite number in [0, 1]",
            )
            if not (0 <= confidence_val <= 1):
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("response", "confidence"),
                    "response confidence must be None or a finite number in [0, 1]",
                )
            validated_confidence = float(confidence_val)
    elif raw_state == "refused":
        state_literal = "refused"
        _require_strict_str(
            resp_map["response_id"],
            min_len=1,
            max_len=128,
            location=("response", "response_id"),
            message="refused response requires a non-blank response_id up to 128 characters",
        )
        if resp_map["status"] != "inconclusive":
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("response", "status"),
                "refused response status must be inconclusive",
            )
        if resp_map["confidence"] is not None:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("response", "confidence"),
                "refused response confidence must be None",
            )
        validated_status = VerificationStatus.INCONCLUSIVE
        validated_confidence = None
    else:
        state_literal = "unavailable" if raw_state == "unavailable" else "timeout"
        if resp_map["response_id"] is not None:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("response", "response_id"),
                "unavailable or timeout response_id must be None",
            )
        if resp_map["status"] is not None:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("response", "status"),
                "unavailable or timeout status must be None",
            )
        if resp_map["confidence"] is not None:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("response", "confidence"),
                "unavailable or timeout confidence must be None",
            )
        validated_status = None
        validated_confidence = None

    for key in MODEL_IDENTITY_KEYS:
        if resp_map[key] != req_map[key]:
            raise VerificationError(
                "VERIFICATION_BINDING_INVALID",
                ("response", key),
                "response identity does not match request",
            )

    return ValidatedModelInput(
        state=state_literal,
        status=validated_status,
        confidence=validated_confidence,
        reason=reason,
    )
