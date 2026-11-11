"""Public versioned Core data models.

Import the models from ``forecontract.models``. Every public model requires an explicit
``schema_version="0.1"`` at every nesting level, forbids unknown fields and validates
assignment. See ``forecontract.models.base`` for the strict scalar rules.
"""

from __future__ import annotations

from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
    Precondition,
)
from forecontract.models.json_types import JsonObject, JsonValue
from forecontract.models.observations import ArtifactRef, Observation
from forecontract.models.outcomes import (
    ActualOutcome,
    ErrorInfo,
    ExecutionStatus,
    OutcomeVerification,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "ActionSpec",
    "ActualOutcome",
    "AdapterCapabilities",
    "ArtifactRef",
    "ErrorInfo",
    "ExecutionStatus",
    "ExpectedOutcome",
    "Forecast",
    "JsonObject",
    "JsonValue",
    "Observation",
    "OutcomeVerification",
    "Precondition",
    "VerificationResult",
    "VerificationStatus",
]
