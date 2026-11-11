"""Closed Memory records and the strict ingress gate they share with the stores.

Two records exist in this round:

* :class:`MemoryItem` - one immutable, versioned Memory fact. It carries a source locator, a
  mapped result and an optional predecessor. It carries no summary, no metadata, no body, no
  observation, no outcome payload, no URL and no exception text: the content stays in the
  trace, and the record only indexes it.
* :class:`ResolvedSource` - the read-only output of resolving one real trace record. It is a
  report of what the resolver read, never a substitute for resolving.

Both are closed, strict and frozen, and both go through :func:`validated_instance` at every
ingress. That gate reads the instance's own storage: a key that is not a declared field is a
rejection, so an ordinary ``model_copy``/``model_construct`` cannot smuggle a field past the
schema, and a rejected candidate never reaches memory or disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Final, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from forecontract.memory.errors import MemoryValidationError
from forecontract.models.base import SchemaVersion, Timestamp

__all__ = [
    "MEMORY_RESULTS",
    "SOURCE_LEVELS",
    "MemoryItem",
    "ResolvedSource",
    "normalised_optional_id",
    "validated_instance",
]

#: The closed result vocabulary. A Memory result must equal what the real source maps to.
MEMORY_RESULTS: Final[tuple[str, ...]] = (
    "success",
    "failure",
    "unknown",
    "cancelled",
    "timeout",
    "aborted",
)

#: The two source levels. They are never mixed.
SOURCE_LEVELS: Final[tuple[str, ...]] = ("run", "attempt")

MemoryResult = Literal["success", "failure", "unknown", "cancelled", "timeout", "aborted"]
SourceLevel = Literal["run", "attempt"]

#: A non-blank identifier. The ``min_length`` guard is backed by the validator below, which is
#: what rejects a whitespace-only string and a non-string without coercion.
Identifier = Annotated[str, Field(strict=True, min_length=1)]

_RECORD_CONFIG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
ModelT = TypeVar("ModelT", bound=BaseModel)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("an identifier must not be blank")
    return value


class _ClosedRecord(BaseModel):
    """The shared strict envelope: frozen, no extra fields, no usable echo in a message.

    The core-schema hook below is what makes instance ingress honest. Pydantic's own
    ``model_validate`` accepts an instance of the right class as it stands, so a copied or
    constructed object carrying an unknown key, an undeclared enum value or a wrong scalar type
    would otherwise pass untouched. Detected instances are therefore re-validated from their
    stored fields, and every rejection is bounded and value-free (F01, F04).
    """

    model_config = _RECORD_CONFIG

    def __init__(self, **data: object) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise MemoryValidationError(_bounded_message(exc, type(self)), cause=exc) from None

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: object, handler: object) -> object:
        """Wrap the declared schema so an instance is re-validated from its stored fields."""
        schema = handler(source_type)
        return core_schema.no_info_wrap_validator_function(cls._reject_invalid_instance, schema)

    @classmethod
    def model_validate(cls, obj: object, **kwargs: object) -> object:
        """The bounded public Python ingress for one record.

        The whole boundary is wrapped, not only the field validators: a non-record Python input
        such as ``1``, ``[]``, ``"text"`` or ``None`` fails before any field is reached, and an
        instance is re-validated from its stored fields by the core-schema hook. Every failure
        becomes a bounded ``MemoryValidationError`` that exposes no rejected value, key or path
        through ``str``, ``repr``, ``errors`` or ``json``; the underlying pydantic error stays
        in-process as the cause and is never serialized (F01).
        """
        try:
            return super().model_validate(obj, **kwargs)
        except MemoryValidationError:
            raise
        except ValidationError as exc:
            raise MemoryValidationError(_bounded_message(exc, cls), cause=exc)

    @classmethod
    def model_validate_json(cls, json_data: object, **kwargs: object) -> object:
        """The bounded public JSON ingress for one record, including JSON syntax failures."""
        try:
            return super().model_validate_json(json_data, **kwargs)
        except MemoryValidationError:
            raise
        except ValidationError as exc:
            raise MemoryValidationError(_bounded_message(exc, cls), cause=exc)

    @classmethod
    def _reject_invalid_instance(cls, value: object, handler: object) -> object:
        """Reject a detected instance that does not match its own declared schema."""
        if isinstance(value, cls):
            state = _validated_stored_state(cls, value)
            _identifier_guard(cls, state)
            return handler(state)
        return handler(value)

    @field_validator("schema_version", "memory_id", "run_id", "source_record_id", "supersedes_id",
                     "record_id", "step_id", "contract_id", "attempt_id", check_fields=False)
    @classmethod
    def _identifier_shape(cls, value: object) -> object:
        """Reject a non-string identifier before any field-dependent validator can touch it."""
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("the identifier must be a string")  # noqa: TRY004 - a schema rejection, not a Python type error
        if not value.strip():
            raise ValueError("an identifier must not be blank")
        return value


def _stored_keys(record: BaseModel) -> list[object]:
    """Return the ``__dict__`` keys of an instance, or an empty list when it has none."""
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        return []
    return list(stored)


def _validated_stored_state(model: type[BaseModel], record: object) -> dict:
    """Rebuild one record's state from its stored fields, rejecting anything undeclared.

    A real pydantic instance keeps its bookkeeping in class slots, which this function never
    reads, so an ordinary instance passes. Any ``__dict__`` key that is not a declared field is
    a rejection, including a non-string key and a bookkeeping-looking name injected as a
    dictionary entry.
    """
    if not isinstance(record, BaseModel) or not isinstance(record, model):
        raise MemoryValidationError("the supplied value is not a validated memory record")
    allowed = frozenset(model.model_fields)
    for key in _stored_keys(record):
        if not (isinstance(key, str) and key in allowed):
            raise MemoryValidationError(
                "the record carries a stored key that is not a declared field"
            )
    return {name: getattr(record, name, None) for name in model.model_fields}


def _identifier_guard(model: type[BaseModel], state: dict) -> None:
    """Reject a declared identifier that is not a non-blank string, before derived validators."""
    for name in model.model_fields:
        if not (name in _IDENTIFIER_FIELDS or name.endswith("_id")):
            continue
        value = state.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise MemoryValidationError("an identifier must be a non-blank string")
        if not value.strip():
            raise MemoryValidationError("an identifier must not be blank")


#: Field names that are identifiers but do not end in ``_id``.
_IDENTIFIER_FIELDS: Final[frozenset[str]] = frozenset({"schema_version"})


def _safe_location(parts: object, model: type[BaseModel]) -> str:
    """Render only declared field names, list indices and a bounded placeholder."""
    allowed = frozenset(model.model_fields)
    rendered: list[str] = []
    for part in (parts or ()):  # type: ignore[union-attr]
        if isinstance(part, bool):
            rendered.append("<value>")
        elif isinstance(part, int):
            rendered.append(f"[{part}]")
        elif isinstance(part, str) and part in allowed:
            rendered.append(part)
        else:
            rendered.append("<field>")
    return ".".join(rendered)[:120] or "<record>"


def _declared_field(parts: object, model: type[BaseModel]) -> str:
    """Return the first declared field named by a location, or an empty string."""
    allowed = frozenset(model.model_fields)
    for part in (parts or ()):  # type: ignore[union-attr]
        if isinstance(part, str) and part in allowed:
            return part
    return ""


def _bounded_message(exc: ValidationError, model: type[BaseModel] | None = None) -> str:
    """Describe a schema rejection with a bounded category and a safe field location.

    The rejected value never appears, and a location is named only when it is a declared field
    of the record class, so an attacker-controlled key cannot be read back from the message
    (F01). The whole message stays short. A whole-boundary failure - a non-record Python input
    or JSON syntax that never parsed - is described by its category alone, because there is no
    declared field to locate.
    """
    errors = exc.errors()
    if not errors:
        return "the supplied record does not match the declared schema"
    first = errors[0]
    error_type = str(first.get("type") or "invalid")
    category = _category(error_type)
    if category == "invalid":
        field = _declared_field(first.get("loc"), model) if model is not None else ""
        category = f"the declared field {field} is invalid" if field else "the value does not match the declared schema"
    if error_type in _WHOLE_INPUT_TYPES:
        # A top-level type failure or an unparsable document: there is no declared location.
        return f"invalid memory record: {category}"[:240]
    location = _safe_location(first.get("loc"), model) if model is not None else "<record>"
    return f"invalid memory record at {location}: {category}"[:240]


#: Error types that describe the whole input rather than one declared field.
_WHOLE_INPUT_TYPES: Final[frozenset[str]] = frozenset({
    "model_type", "model_attributes_type", "dataclass_type", "dict_type",
    "json_invalid", "json_type", "is_instance_of",
})


_CATEGORIES: Final[dict[str, str]] = {
    "extra_forbidden": "an unexpected field is not accepted",
    "missing": "a required field is missing",
    "literal_error": "a value is outside the closed set this field accepts",
    "enum": "a value is outside the closed set this field accepts",
    "string_type": "a field expects a string",
    "string_too_short": "a field expects a non-blank string",
    "int_type": "a field expects an integer",
    "int_parsing": "a field expects an integer",
    "greater_than_equal": "a number is outside the accepted range",
    "bool_type": "a field expects a boolean",
    "bool_parsing": "a field expects a boolean",
    "datetime_type": "a field expects a timestamp",
    "datetime_parsing": "a field expects a timestamp",
    "datetime_from_date_parsing": "a field expects a timestamp",
    "value_error": "a field rule was violated",
    "union_tag_invalid": "a value is outside the closed set this field accepts",
    "strict": "a value has the wrong JSON type",
    "model_type": "the input is not an object this record can be built from",
    "model_attributes_type": "the input is not an object this record can be built from",
    "dict_type": "the input is not an object this record can be built from",
    "dataclass_type": "the input is not an object this record can be built from",
    "is_instance_of": "the input is not an object this record can be built from",
    "json_invalid": "the input is not a valid JSON document",
    "json_type": "the input is not a JSON string",
}


def _category(error_type: str) -> str:
    return _CATEGORIES.get(error_type, "the value does not match the declared schema")


def normalised_optional_id(value: object) -> object:
    """Accept a declared association field as either ``None`` or an empty container.

    Only ``None`` and an empty ``list``/``tuple`` mean *absent*. Anything else is passed through
    unchanged so the schema decides, which keeps a forged candidate a rejection instead of
    something this helper quietly repairs.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return None
    return value


def validated_instance(model: type[ModelT], record: object) -> ModelT:
    """Return the freshly validated copy of one Memory record.

    The gate compares every key in the instance's own storage with the declared field names of
    the requested class. A real pydantic instance keeps its bookkeeping in class slots, not in
    ``__dict__``, so an ordinary instance passes while an unknown key, a non-string key or an
    injected bookkeeping-looking key is rejected before anything is read or written. Passing an
    already validated record into this function is how the stores consume their candidate.
    """
    state = _validated_stored_state(model, record)
    _identifier_guard(model, state)
    try:
        return model.model_validate(state)
    except ValidationError as exc:
        raise MemoryValidationError(_bounded_message(exc, model), cause=exc) from exc


class MemoryItem(_ClosedRecord):
    """One versioned Memory fact for one logical experience.

    ``source_record_id`` locates the real trace record the fact indexes, and ``result`` must
    equal the label that record maps to. ``supersedes_id`` optionally names the current chain
    head this version replaces; the stores check that relation against the whole file.
    """

    schema_version: SchemaVersion
    memory_id: Identifier
    run_id: Identifier
    source_level: SourceLevel
    source_record_id: Identifier
    result: MemoryResult
    created_at: Timestamp
    supersedes_id: Identifier | None = None

    @model_validator(mode="after")
    def _blank_identifiers(self) -> MemoryItem:
        for name in ("memory_id", "run_id", "source_record_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("an identifier must be a non-blank string")
        if self.supersedes_id is not None and (
            not isinstance(self.supersedes_id, str) or not self.supersedes_id.strip()
        ):
            raise ValueError("an identifier must be a non-blank string")
        if not isinstance(self.created_at, datetime):
            raise ValueError("the timestamp must be a datetime")  # noqa: TRY004 - a schema rejection
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("the timestamp must be timezone-aware")
        return self

    def utc_key(self) -> tuple[datetime, str]:
        """The stable ordering key: the UTC instant, then the memory identifier."""
        return (self.created_at.astimezone(timezone.utc), self.memory_id)


class ResolvedSource(_ClosedRecord):
    """The read-only result of resolving one real trace record.

    The first four fields name what was read; the three association identifiers are ``None``
    for a run-level fact and copy the real Attempt associations for an attempt-level fact.
    """

    schema_version: SchemaVersion
    run_id: Identifier
    record_id: Identifier
    source_level: SourceLevel
    step_id: Identifier | None = None
    contract_id: Identifier | None = None
    attempt_id: Identifier | None = None
    result: MemoryResult

    @model_validator(mode="after")
    def _scope_consistency(self) -> ResolvedSource:
        for name in ("run_id", "record_id", "step_id", "contract_id", "attempt_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("an identifier must be a non-blank string")
        present = [self.step_id, self.contract_id, self.attempt_id]
        if self.source_level == "run" and any(item is not None for item in present):
            raise ValueError("a run-level source must not carry an attempt association")
        if self.source_level == "attempt" and any(item is None for item in present):
            raise ValueError("an attempt-level source needs its step, contract and attempt")
        return self
