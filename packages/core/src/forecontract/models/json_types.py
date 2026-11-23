"""Recursive JSON value types shared by the versioned Core models.

``JsonValue`` accepts only JSON null, boolean, integer, finite float, string, list and
string-keyed object values. ``JsonObject`` is restricted to a JSON object whose values are
``JsonValue``. Bytes, ``datetime`` instances, tuples, sets, arbitrary Python objects,
non-string keys, ``NaN`` and infinities are rejected.

The type is expressed as a recursive Pydantic core schema rather than a single plain
validator for two reasons: a failure reports the offending nested field or index in the
error location, and the validated payload stays an ordinary ``dict``/``list`` that
round-trips through both ``model_dump_json()`` and ``model_dump(mode="json")``.
"""

from __future__ import annotations

from typing import Any

from pydantic_core import core_schema

__all__ = ["JsonObject", "JsonValue"]

_VALUE_REF = "forecontract-json-value"
_OBJECT_REF = "forecontract-json-object"


def _tag(value: Any) -> str:
    """Select the JSON branch from the runtime type of the raw input."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "invalid"


def _unsupported(value: Any) -> Any:
    # Pydantic validators must raise ValueError to become validation errors.
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def _value_schema() -> core_schema.CoreSchema:
    reference = core_schema.definition_reference_schema(_VALUE_REF)
    return core_schema.tagged_union_schema(
        {
            "null": core_schema.none_schema(),
            "bool": core_schema.bool_schema(strict=True),
            "int": core_schema.int_schema(strict=True),
            "float": core_schema.float_schema(strict=True, allow_inf_nan=False),
            "str": core_schema.str_schema(strict=True),
            "list": core_schema.list_schema(reference, strict=True),
            "dict": core_schema.dict_schema(core_schema.str_schema(strict=True), reference),
            "invalid": core_schema.no_info_plain_validator_function(_unsupported),
        },
        discriminator=_tag,
        ref=_VALUE_REF,
    )


def _object_schema() -> core_schema.CoreSchema:
    return core_schema.dict_schema(
        core_schema.str_schema(strict=True),
        core_schema.definition_reference_schema(_VALUE_REF),
        ref=_OBJECT_REF,
    )


def _definitions() -> list[core_schema.CoreSchema]:
    return [_value_schema(), _object_schema()]


class JsonValue:
    """A JSON value: null, boolean, integer, finite float, string, list or object."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.definitions_schema(
            core_schema.definition_reference_schema(_VALUE_REF), _definitions()
        )


class JsonObject:
    """A JSON object with string keys and ``JsonValue`` values."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.definitions_schema(
            core_schema.definition_reference_schema(_OBJECT_REF), _definitions()
        )
