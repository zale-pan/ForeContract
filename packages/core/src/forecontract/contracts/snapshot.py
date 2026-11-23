"""Deep immutable snapshots for committed contract content.

Pydantic's ``frozen=True`` blocks attribute assignment but leaves a nested ``dict`` or
``list`` mutable, so a committed contract could still be changed through an ordinary
supported API. This module supplies the missing half: validated JSON payloads are stored in
immutable containers (``FrozenMapping`` / ``FrozenList``) and the candidate models are
re-validated into frozen subclasses of the existing Core models. Mutation attempts are then
rejected by Python itself instead of being silently accepted, while ``model_dump()``,
``model_dump(mode="json")`` and ``model_dump_json()`` keep the agreed JSON object/array
shape and both roundtrips.

The frozen subclasses never exist as caller-visible input types: the contract fields are
declared with the existing Core models, so a caller still passes an ordinary ``ActionSpec``
or ``Precondition`` and Pydantic re-validates it before the snapshot is taken.

Revalidation always goes through the strict python-mode snapshot, never through JSON text:
a serializer would happily turn ``NaN`` or ``infinity`` into ``null`` and hide an illegal
in-place edit. ``detach`` therefore dumps to python objects and validates those, so bytes,
non-string keys and non-finite floats are detected instead of normalised away.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Annotated, Any, Self, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PlainSerializer

from forecontract.models.actions import ActionSpec, ExpectedOutcome, Precondition
from forecontract.models.json_types import JsonObject

__all__ = [
    "FrozenActionSpec",
    "FrozenExpectedOutcome",
    "FrozenJsonObject",
    "FrozenList",
    "FrozenMapping",
    "FrozenPrecondition",
    "detach",
    "freeze_action",
    "freeze_expected_outcome",
    "freeze_expected_outcomes",
    "freeze_json",
    "freeze_precondition",
    "freeze_preconditions",
    "thaw_json",
]

ModelT = TypeVar("ModelT", bound=BaseModel)


class FrozenMapping(Mapping):
    """A read-only JSON object that still compares equal to a plain ``dict``."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping) -> None:
        object.__setattr__(self, "_data", data)

    def __getitem__(self, key: object) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._data)!r})"

    __hash__ = None


class FrozenList(tuple):
    """A read-only JSON array that still compares equal to a plain ``list``."""

    def __new__(cls, values: Iterable = ()) -> Self:
        return super().__new__(cls, values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return len(self) == len(other) and all(left == right for left, right in zip(self, other))
        return NotImplemented

    __hash__ = None


def freeze_json(value: Any) -> Any:
    """Return an immutable deep copy of an already validated JSON value."""
    if isinstance(value, dict):
        return FrozenMapping({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    """Return an ordinary JSON structure for serialization."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_json(item) for item in value]
    return value


FrozenJsonObject = Annotated[
    JsonObject,
    AfterValidator(freeze_json),
    PlainSerializer(thaw_json, return_type=dict, when_used="always"),
]

_FROZEN_CONFIG = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class FrozenActionSpec(ActionSpec):
    """An ``ActionSpec`` whose payload and metadata can no longer be edited."""

    model_config = _FROZEN_CONFIG

    payload: FrozenJsonObject = Field(default_factory=dict)
    metadata: FrozenJsonObject = Field(default_factory=dict)


class FrozenPrecondition(Precondition):
    """A ``Precondition`` whose payload can no longer be edited."""

    model_config = _FROZEN_CONFIG

    payload: FrozenJsonObject = Field(default_factory=dict)


class FrozenExpectedOutcome(ExpectedOutcome):
    """An ``ExpectedOutcome`` whose payload can no longer be edited."""

    model_config = _FROZEN_CONFIG

    payload: FrozenJsonObject = Field(default_factory=dict)


def detach(model_type: type[ModelT], value: object) -> ModelT:
    """Revalidate and detach caller-owned model data.

    ``value`` may be an instance of ``model_type`` or a plain mapping. The python-mode dump
    is re-validated through the declared strict schema, which is what detects an in-place
    contamination that a model instance would otherwise carry straight through. A
    ``ValidationError`` or ``TypeError`` here means the input is malformed.

    Serializer warnings are suppressed on purpose: a contaminated payload is expected to
    look wrong to the serializer, and the revalidation below - never the serializer - is
    what decides whether the data is acceptable.
    """
    if isinstance(value, model_type):
        payload = value.model_dump(mode="python", warnings=False)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"{model_type.__name__} instance or mapping required")
    return model_type.model_validate(payload)


def freeze_action(value: ActionSpec) -> FrozenActionSpec:
    """Return the frozen snapshot of a validated action candidate."""
    return FrozenActionSpec.model_validate(value.model_dump(mode="python", warnings=False))


def freeze_precondition(value: Precondition) -> FrozenPrecondition:
    """Return the frozen snapshot of a validated precondition."""
    return FrozenPrecondition.model_validate(value.model_dump(mode="python", warnings=False))


def freeze_expected_outcome(value: ExpectedOutcome) -> FrozenExpectedOutcome:
    """Return the frozen snapshot of a validated expected outcome."""
    return FrozenExpectedOutcome.model_validate(value.model_dump(mode="python", warnings=False))


def freeze_preconditions(values: Iterable[Precondition]) -> FrozenList:
    """Return an immutable sequence of frozen preconditions in the supplied order."""
    return FrozenList(freeze_precondition(value) for value in values)


def freeze_expected_outcomes(values: Iterable[ExpectedOutcome]) -> FrozenList:
    """Return an immutable sequence of frozen outcomes in the supplied order."""
    return FrozenList(freeze_expected_outcome(value) for value in values)
