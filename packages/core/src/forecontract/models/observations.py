"""Observation-side Core models: artifact references and environment observations."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from forecontract.models.base import (
    MODEL_CONFIG,
    NonBlankStr,
    SchemaVersion,
    StrictStr,
    Timestamp,
)
from forecontract.models.json_types import JsonObject

__all__ = ["ArtifactRef", "Observation"]


class ArtifactRef(BaseModel):
    """A reference to an artifact produced or consumed by an environment."""

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    artifact_id: NonBlankStr
    artifact_type: NonBlankStr
    uri: NonBlankStr
    metadata: JsonObject = Field(default_factory=dict)


class Observation(BaseModel):
    """A single environment observation."""

    model_config = MODEL_CONFIG

    schema_version: SchemaVersion
    observation_id: NonBlankStr
    environment_type: NonBlankStr
    state: JsonObject
    timestamp: Timestamp
    summary: StrictStr | None = None
    artifacts: Annotated[list[ArtifactRef], Field(strict=True)] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)
