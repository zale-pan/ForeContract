"""Shared validation primitives for the versioned Core data models.

Every public model requires ``schema_version: Literal["0.1"]`` with no default, rejects
extra fields and validates assignments. Scalars are strict: strings never accept numbers
or booleans, booleans never accept strings or integers, integers never accept booleans or
floats, and numbers are finite and range-checked.

The promised ingress contract is honoured deliberately. ``model_dump_json()`` round-trips
through ``model_validate_json()``, and ``model_dump(mode="json")`` round-trips through
``model_validate()`` because the timestamp type accepts an ISO-8601 string and enums accept
their JSON string values. A blanket strict model configuration would break that ingress,
so strictness is expressed per field instead.

Assignment validation covers field assignment only. In-place mutation of a nested dict or
list is not observed by Pydantic; components must revalidate payloads at their own ingress
boundary. This module claims no deep immutability and no Contract semantics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BeforeValidator, ConfigDict, Field

__all__ = [
    "MODEL_CONFIG",
    "SCHEMA_VERSION",
    "FiniteFloat",
    "NonBlankStr",
    "NonNegativeFloat",
    "NonNegativeInt",
    "SchemaVersion",
    "StrictBool",
    "StrictInt",
    "StrictStr",
    "Timestamp",
    "UnitFloat",
]

SCHEMA_VERSION = "0.1"
SchemaVersion = Literal["0.1"]

MODEL_CONFIG = ConfigDict(extra="forbid", validate_assignment=True)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain a non-whitespace character")
    return value


def _json_enum_only(value: object) -> object:
    """Accept only an enum member or its JSON string value.

    Lax Enum validation would also accept ``bytes`` and decode it, so enum-annotated fields
    use this before-validator to reject unrelated input types while keeping the JSON string
    representation that both promised roundtrips rely on.
    """
    if isinstance(value, str):
        return value
    raise ValueError("enum input must be the member or its JSON string value")


def _aware_timestamp(value: object) -> object:
    """Accept a timezone-aware datetime or its ISO-8601 JSON representation only."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO-8601 datetime string") from exc
    raise ValueError("timestamp must be a timezone-aware datetime or an ISO-8601 string")


NonBlankStr = Annotated[str, Field(strict=True), AfterValidator(_non_blank)]
StrictStr = Annotated[str, Field(strict=True)]
StrictBool = Annotated[bool, Field(strict=True)]
StrictInt = Annotated[int, Field(strict=True)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Timestamp = Annotated[AwareDatetime, BeforeValidator(_aware_timestamp)]
