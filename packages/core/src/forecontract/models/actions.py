"""Action-side Core models: actions, preconditions, forecasts and capabilities.

Core declares the shape of an action and of an adapter's declared abilities. It does not
understand DOM, selectors or any browser payload schema, and it implements no domain
parsing.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from forecontract.models.base import (
    MODEL_CONFIG,
    NonBlankStr,
    SchemaVersion,
    StrictBool,
    StrictStr,
    UnitFloat,
)
from forecontract.models.json_types import JsonObject

__all__ = [
    "ActionSpec",
    "AdapterCapabilities",
    "ExpectedOutcome",
    "Forecast",
    "Precondition",
]

CAPABILITY_LISTS = (
    "supported_actions",
    "supported_observation_types",
    "supported_condition_types",
    "supported_outcome_types",
)


class ActionSpec(BaseModel):
    """A candidate action and its opaque payload."""

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    action_type: NonBlankStr
    payload: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)


class Precondition(BaseModel):
    """A precondition an environment may require before an action."""

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    condition_type: NonBlankStr
    payload: JsonObject = Field(default_factory=dict)
    required: StrictBool = True


class ExpectedOutcome(BaseModel):
    """An outcome an action is expected to produce."""

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    outcome_type: NonBlankStr
    payload: JsonObject = Field(default_factory=dict)
    required: StrictBool = True


class Forecast(BaseModel):
    """A proposed action with its expected outcomes.

    ``expected_outcomes`` is required but may be empty: whether a candidate may be
    committed is a Contract Builder decision, not a property of this model.
    ``task_completion_hint`` is advisory and is never proof of task success.
    """

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    forecast_id: NonBlankStr
    proposed_action: ActionSpec
    expected_outcomes: Annotated[list[ExpectedOutcome], Field(strict=True)]
    intent: StrictStr | None = None
    decision_summary: StrictStr | None = None
    confidence: UnitFloat | None = None
    task_completion_hint: StrictBool | None = None
    metadata: JsonObject = Field(default_factory=dict)


class AdapterCapabilities(BaseModel):
    """What an adapter declares it can observe and do.

    Every capability list may be empty. Elements must contain a non-whitespace character,
    duplicates are rejected and the given order is preserved.
    """

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    adapter_name: NonBlankStr
    environment_type: NonBlankStr
    supported_actions: Annotated[list[NonBlankStr], Field(strict=True)]
    supported_observation_types: Annotated[list[NonBlankStr], Field(strict=True)] = Field(default_factory=list)
    supported_condition_types: Annotated[list[NonBlankStr], Field(strict=True)] = Field(default_factory=list)
    supported_outcome_types: Annotated[list[NonBlankStr], Field(strict=True)] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator(*CAPABILITY_LISTS)
    @classmethod
    def _reject_duplicate_entries(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("capability entries must be unique")
        return value
