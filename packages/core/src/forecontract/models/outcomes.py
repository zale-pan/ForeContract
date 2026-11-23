"""Outcome-side Core models: execution status, errors and verification results.

Execution success, verification status and task success are three separate notions. This
module records execution and verification facts only; it makes no task-completion claim.

The cross-field invariants are enforced by field validators rather than an after-model
validator: a field validator runs before the new value is stored, so a rejected assignment
leaves the previous valid state untouched instead of leaving an invalid live object.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, ValidationInfo, field_validator

from forecontract.models.base import (
    MODEL_CONFIG,
    NonBlankStr,
    NonNegativeFloat,
    NonNegativeInt,
    SchemaVersion,
    StrictBool,
    StrictStr,
    _json_enum_only,
)
from forecontract.models.json_types import JsonObject, JsonValue
from forecontract.models.observations import ArtifactRef

__all__ = [
    "ActualOutcome",
    "ErrorInfo",
    "ExecutionStatus",
    "OutcomeVerification",
    "VerificationResult",
    "VerificationStatus",
]


class ExecutionStatus(str, Enum):
    """How an attempted action actually ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    """The verification verdict for one outcome or for a whole result."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


ExecutionStatusField = Annotated[ExecutionStatus, BeforeValidator(_json_enum_only)]
VerificationStatusField = Annotated[VerificationStatus, BeforeValidator(_json_enum_only)]
ArtifactList = Annotated[list[ArtifactRef], Field(strict=True)]


class ErrorInfo(BaseModel):
    """An execution-failure description. It carries no Runtime stage information."""

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    code: NonBlankStr
    message: StrictStr
    recoverable: StrictBool | None = None
    details: JsonObject = Field(default_factory=dict)


class ActualOutcome(BaseModel):
    """What actually happened when an action was executed.

    An optional error is allowed for FAILED and UNKNOWN outcomes; a SUCCEEDED outcome must
    not carry an execution-failure ErrorInfo.
    """

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    outcome_id: NonBlankStr
    status: ExecutionStatusField
    state_before: JsonObject | None = None
    state_after: JsonObject | None = None
    changes: JsonObject = Field(default_factory=dict)
    artifacts: ArtifactList = Field(default_factory=list)
    error: ErrorInfo | None = None
    duration_ms: NonNegativeFloat | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("status")
    @classmethod
    def _status_allows_error(cls, value: ExecutionStatus, info: ValidationInfo) -> ExecutionStatus:
        if value is ExecutionStatus.SUCCEEDED and info.data.get("error") is not None:
            raise ValueError("SUCCEEDED must not carry an execution-failure ErrorInfo")
        return value

    @field_validator("error")
    @classmethod
    def _error_allowed_for_status(cls, value: ErrorInfo | None, info: ValidationInfo) -> ErrorInfo | None:
        if value is not None and info.data.get("status") is ExecutionStatus.SUCCEEDED:
            raise ValueError("SUCCEEDED must not carry an execution-failure ErrorInfo")
        return value


class OutcomeVerification(BaseModel):
    """Verification of one expected outcome.

    ``has_actual_value=False`` requires ``actual_value=None``. ``has_actual_value=True``
    with ``actual_value=None`` records an observed JSON null, while a non-null value
    records an observed value; that distinction survives both roundtrips.
    """

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    outcome_index: NonNegativeInt
    status: VerificationStatusField
    has_actual_value: StrictBool = False
    actual_value: JsonValue = None
    reason: StrictStr | None = None

    @field_validator("has_actual_value")
    @classmethod
    def _flag_requires_null(cls, value: bool, info: ValidationInfo) -> bool:
        if not value and info.data.get("actual_value") is not None:
            raise ValueError("has_actual_value=False requires actual_value=None")
        return value

    @field_validator("actual_value")
    @classmethod
    def _value_requires_flag(cls, value: object, info: ValidationInfo) -> object:
        if value is not None and info.data.get("has_actual_value") is False:
            raise ValueError("has_actual_value=False requires actual_value=None")
        return value


class VerificationResult(BaseModel):
    """An aggregate verification record. It deliberately has no confidence field."""

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    status: VerificationStatusField
    verifier: NonBlankStr
    outcome_results: Annotated[list[OutcomeVerification], Field(strict=True)] = Field(default_factory=list)
    reason: StrictStr | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("outcome_results")
    @classmethod
    def _unique_outcome_index(cls, value: list[OutcomeVerification]) -> list[OutcomeVerification]:
        seen: set[int] = set()
        for result in value:
            if result.outcome_index in seen:
                raise ValueError(f"duplicate outcome_index: {result.outcome_index}")
            seen.add(result.outcome_index)
        return value
