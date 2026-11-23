"""The five versioned records of the verification package.

Every record requires an explicit ``schema_version="0.1"``, rejects unknown versions and extra
fields and uses strict field types. A path is an exact list of segments relative to an
observation state - strict strings for JSON object keys and non-negative strict integers for
JSON array indices - with no wildcards, fallbacks, conversions or expressions.

Two properties are deliberate:

* ``VerificationEvidence.state_complete`` is a required strict boolean with no default. The
  provider states whether the supplied state covers the whole range the caller's rules need.
  The verifier never infers completeness from a field count, from metadata, from an adapter
  name or from an execution status, and it never authenticates the provider.
* A record is revalidated from its own stored fields and stored keys at every public ingress,
  including an existing instance. Content that reaches a public entry point already invalid
  through an ordinary copy or a constructed instance is therefore refused rather than quietly
  laundered by serialization.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from forecontract.models.actions import ExpectedOutcome
from forecontract.models.base import SchemaVersion, StrictBool, StrictInt
from forecontract.models.json_types import JsonValue
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus, _json_enum_only
from forecontract.verification.errors import VerificationError

__all__ = [
    "PathExistsRule",
    "RuleEvaluation",
    "TaskCompletionCriteria",
    "ValueEqualsRule",
    "VerificationEvidence",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", validate_assignment=True)

#: The two supported rule type names. No other rule exists.
VALUE_EQUALS = "value_equals"
PATH_EXISTS = "path_exists"
RULE_TYPES = (VALUE_EQUALS, PATH_EXISTS)

#: The fixed reason codes a rule evaluation may carry.
VALUE_MATCH = "VALUE_MATCH"
VALUE_MISMATCH = "VALUE_MISMATCH"
PATH_PRESENT = "PATH_PRESENT"
PATH_ABSENT = "PATH_ABSENT"
EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"


def _non_blank_path_segment(value: object) -> object:
    """A path segment is a strict string or a non-negative strict integer, nothing else.

    A string segment is never trimmed and never split, so a blank or blank-only key is a legal
    exact object key. A boolean, a float and a negative integer are all refused: ``True`` is not
    the index ``1`` and ``"0"`` is not the index ``0``. The annotation is deliberately ``object``
    so that this check, rather than a schema union, decides - it rejects a boolean, a float, a
    negative integer, ``None``, a list and a mapping with the same bounded facility error.
    """
    if isinstance(value, bool):
        raise ValueError("a path segment must not be a boolean")  # noqa: TRY004
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        if value < 0:
            raise ValueError("an array index must not be negative")
        return value
    raise ValueError("a path segment must be a string or a non-negative integer")


PathSegment = Annotated[object, AfterValidator(_non_blank_path_segment)]

#: The only sequence a path field accepts. A tuple is deliberately not a list here: the caller
#: contract names a non-empty list, and a frozen Core payload that holds a tuple is normalised by
#: the rule parser before it reaches this field (F01).
NonEmptyPath = Annotated[
    list[PathSegment], Field(strict=True, min_length=1)
]


def normalise_path_segments(payload: dict) -> dict:
    """Return a rule payload whose declared path is a fresh list, whatever sequence it holds.

    A committed contract stores its payload in a deeply immutable mapping, where a JSON array
    becomes a tuple. That tuple is the same data the caller wrote, so it is rebuilt as a list
    here - and only here, inside the facility - while a caller who passes a tuple to a public
    record constructor or parser is refused by the strict path field.
    """
    path = payload.get("path")
    if isinstance(path, tuple):
        return {**payload, "path": list(path)}
    return payload

#: The Core enum ingress rule: only a member or its JSON string value, never bytes and never a
#: value the lax enum decoder would coerce (F01).
StatusField = Annotated[VerificationStatus, BeforeValidator(_json_enum_only)]


def _reject_blank_identifier(value: object) -> object:
    """Reject a blank or blank-only identifier without rewriting the caller value."""
    if isinstance(value, str) and not value.strip():
        raise ValueError("an identifier must not be blank")
    return value


_non_blank_identifier = AfterValidator(_reject_blank_identifier)

#: A goal identifier: a strict non-blank string, preserved verbatim.
GoalId = Annotated[str, Field(strict=True), _non_blank_identifier]

#: Distinguishes "this field had no previous value" from "the previous value was null".
_MISSING = object()

#: The private keys pydantic itself stores on an instance; they are not declared fields.
_STORAGE_KEYS = frozenset({"__pydantic_fields_set__", "__pydantic_extra__",
                           "__pydantic_private__"})


def _validate_updated_fields(copy: object, updated: dict) -> None:
    """Refuse a ``model_copy`` update whose value does not match its declared field schema.

    The record is rebuilt from its own plain projection with the update applied, so every
    declared rule applies to the updated value - including the strict shape of a path, which a
    standalone field adapter would accept as a sequence.
    """
    fields = type(copy).model_fields
    for name in updated:
        if name not in fields:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("<record>",),
                "an updated key is not a declared field",
            )
    state = copy.model_dump(mode="python", warnings=False)
    state.update(updated)
    try:
        type(copy).model_validate(state)
    except VerificationError:
        raise
    except ValidationError as exc:
        errors = exc.errors()
        raw = tuple(errors[0].get("loc") or ()) if errors else ()
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            tuple(part for part in raw if isinstance(part, str) and part in fields) or ("<record>",),
            "an updated field does not match the declared schema",
            cause=exc,
        ) from exc


def _detach_nested(data: object) -> object:
    """Validate and detach every nested record a caller hands to a public ingress (F02, F04).

    A nested model instance is checked against its own declared fields, stored keys and stored
    scalar types, then rebuilt from that state, so a record produced by an ordinary copy or a
    constructed instance cannot hide an invalid field, an undeclared key, a cyclic reference or a
    wrong basic type, and the new record owns its nested data rather than aliasing the caller's.
    A mapping is passed through untouched: the schema itself validates it and builds fresh
    instances.
    """
    if isinstance(data, dict):
        active: set[int] = set()
        return {key: _detached(data[key], active) for key in data}
    return data


def _detached(value: object, active: set[int]) -> object:
    """Return a detached copy of one nested value, refusing a cycle in the raw container (F02).

    ``active`` holds the identities on the current descent only, and each identity is removed
    once its value is finished. A raw list that reaches itself is therefore refused with one
    bounded facility error, while a value legitimately referenced twice - two attributes sharing
    one list, or a repeated model instance - is copied once per reference and still accepted,
    because a reference is only a cycle when it is still being descended.
    """
    if isinstance(value, BaseModel):
        identity = id(value)
        if identity in active:
            raise VerificationError("VERIFICATION_INPUT_INVALID", ("<record>",),
                                    "the supplied record contains a cyclic reference")
        active.add(identity)
        try:
            return _detached_model(value, active)
        finally:
            active.discard(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise VerificationError("VERIFICATION_INPUT_INVALID", ("<record>",),
                                    "the supplied value contains a cyclic reference")
        active.add(identity)
        try:
            return [_detached(item, active) for item in value]
        finally:
            active.discard(identity)
    return value


def _detached_model(value: BaseModel, active: set[int]) -> object:
    """Validate one nested record from its own stored fields, then rebuild it detached."""
    declared = tuple(type(value).model_fields)
    validate_declared_instance(value, declared)
    stored = getattr(value, "__dict__", None)
    if not isinstance(stored, dict):
        return value
    # Rebuilding from the stored fields drops an undeclared key rather than carrying it, so the
    # extra key is refused here: the closed schema of the outer record would never see it.
    for key in stored:
        if key not in declared and key not in _STORAGE_KEYS:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("<record>",),
                "the supplied record carries a key its schema does not declare",
            )
    state = {key: _detached(item, active) for key, item in stored.items()
             if key not in _STORAGE_KEYS}
    return type(value).model_validate(state)


def _validated_assignment(value: object) -> object:
    """Validate a value being assigned, refusing an invalid nested record before it is stored.

    Only a model or a list of models needs this: the schema itself already checks every plain
    JSON value. A nested record is validated from its own stored fields, so an instance produced
    by ``model_copy`` or ``model_construct`` cannot carry an invalid field or an undeclared key
    into the record through an assignment.
    """
    if isinstance(value, BaseModel):
        return _detached(value, set())
    if isinstance(value, list) and any(isinstance(item, BaseModel) for item in value):
        return _detached(value, set())
    return value


class _VerificationRecord(BaseModel):
    """The shared closed envelope for the new records.

    Three guarantees are enforced here rather than in each record, because all five records need
    them: a nested record handed in by the caller is validated from its own stored fields before
    the schema sees it and is then rebuilt inside this record, so invalid nested content is
    refused at every ingress and the record owns its nested state instead of aliasing the
    caller's; and a failed assignment leaves the record exactly as it was.
    """

    model_config = _MODEL_CONFIG

    #: Guards the nested rebuild so that detaching does not recurse through itself.
    _detaching: ClassVar[bool] = False

    def __init__(self, /, **data: object) -> None:
        # ``self`` is positional-only, exactly as pydantic declares its own constructor, so a
        # caller mapping that carries an unknown key named ``self`` is forwarded intact and
        # rejected by the closed schema instead of colliding with the parameter.
        if not type(self)._detaching:
            type(self)._detaching = True
            try:
                data = _detach_nested(data)
            finally:
                type(self)._detaching = False
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise _as_input_error(exc, tuple(type(self).model_fields)) from None

    def __setattr__(self, name: str, value: object) -> None:
        """Assign one declared field transactionally.

        Pydantic stores the new value before an after-model validator can reject it, so a failed
        assignment would otherwise leave an invalid record with no way back. The previous state
        is restored whenever the assignment is refused, and a legal assignment still takes
        effect.
        """
        stored = self.__dict__
        if name.startswith("_") or not isinstance(stored, dict) or name not in type(self).model_fields:
            super().__setattr__(name, value)
            return
        # An assigned nested record must pass the same original-shape check construction applies,
        # before anything is committed, so a forged nested instance is refused rather than stored
        # and the previous valid record is untouched (F02, F03).
        value = _validated_assignment(value)
        snapshot = {key: stored[key] for key in stored}
        extras = getattr(self, "__pydantic_extra__", None)
        if name not in stored:
            snapshot[name] = _MISSING
        try:
            super().__setattr__(name, value)
        except Exception:
            # A failed assignment may have replaced the storage dictionary and stored the new
            # value before an after-model validator refused it, so the prior valid state is put
            # back into whatever dictionary the instance now uses.
            current = self.__dict__
            if isinstance(current, dict):
                for key in [key for key in current if key not in snapshot]:
                    del current[key]
                current.update(snapshot)
            if isinstance(extras, dict):
                self.__pydantic_extra__ = extras
            raise

    @classmethod
    def model_validate(cls, obj: object, **kwargs: object) -> object:
        """Public mapping and instance ingress, wrapped as a bounded facility error.

        An existing instance is validated from its own stored fields like any other nested record
        and then rebuilt, so the returned record never shares mutable nested data with the
        argument (F04); ``model_validate`` on a valid record returns an equal, independent record.
        """
        if isinstance(obj, cls):
            obj = _detached(obj, set())
        else:
            obj = _detach_nested(obj)
        try:
            return super().model_validate(obj, **kwargs)
        except VerificationError:
            raise
        except ValidationError as exc:
            raise _as_input_error(exc, tuple(cls.model_fields)) from None

    def model_copy(self, **kwargs: object) -> object:  # type: ignore[override]
        """Copy a record, checking any updated field against its declared schema.

        ``model_copy`` writes the update straight into the instance dictionary, so without this
        an updated field would never meet its own schema and an invalid value could be carried
        into a record that every other ingress refuses.
        """
        updated = kwargs.get("update")
        copy = BaseModel.model_copy(self, **kwargs)  # type: ignore[arg-type]
        if isinstance(updated, dict):
            _validate_updated_fields(copy, updated)
        return copy

    @classmethod
    def model_validate_json(cls, json_data: object, **kwargs: object) -> object:
        """Public JSON ingress, including a document that never parsed."""
        try:
            return super().model_validate_json(json_data, **kwargs)
        except VerificationError:
            raise
        except ValidationError as exc:
            raise _as_input_error(exc, tuple(cls.model_fields)) from None


def _as_input_error(exc: BaseException, allowed: tuple[str, ...]) -> VerificationError:
    """Convert a schema rejection into the bounded input error, keeping the cause in-process.

    Only a real schema rejection is converted. A programming fault of another kind is re-raised
    unchanged so that it can never be mistaken for a business verdict.
    """
    if not isinstance(exc, ValidationError):
        raise exc
    errors = exc.errors()
    location = tuple(errors[0].get("loc") or ()) if errors else ()
    safe = tuple(part for part in location if not isinstance(part, str) or part in allowed)
    return VerificationError(
        "VERIFICATION_INPUT_INVALID",
        safe or ("<record>",),
        "the supplied record does not match the declared schema",
        cause=exc,
    )


def validate_declared_instance(record: object, allowed: tuple[str, ...]) -> None:
    """Revalidate one record from its declared fields and actual stored keys.

    An unknown stored key, a stored field that no longer matches the schema and a genuinely
    cyclic reference are all refused here. A record is revalidated through the pydantic core
    validator - not through the class constructor, which would recurse - and that validator runs
    in strict mode, so a stored ``"yes"`` is refused where a ``bool`` is declared rather than
    being coerced. A legal shared reference is visited once per path.
    """
    if not isinstance(record, BaseModel):
        return
    try:
        _validate_model(record, ())
    except _StrictRejection as exc:
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            exc.location or ("<record>",),
            "the record does not match the declared schema",
            cause=exc.cause,
        ) from exc


class _StrictRejection(Exception):
    """An internal carrier for a strict schema rejection found while walking a record."""

    def __init__(self, message: str, location: tuple[str | int, ...],
                 cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.location = tuple(location)
        self.cause = cause


def _validate_model(record: BaseModel, seen: tuple[tuple[int, type], ...]) -> None:
    """Revalidate one record, then walk the record classes its fields declare.

    Two checks run here. First, every scalar the record actually stores must already be of its
    declared basic type, so a stored ``"yes"`` where a ``bool`` is declared is refused instead of
    being coerced. Second, the record is rebuilt from that stored state through its own class, so
    the declared schema decides everything else - including the Core legal JSON forms, such as a
    timestamp string or an enum's string value, which stay acceptable in a plain dump.
    """
    identity = (id(record), type(record))
    if identity in seen:
        raise _StrictRejection("the record contains a cyclic reference", ("<record>",))
    model = type(record)
    fields = model.model_fields
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        return
    for key in stored:
        if not (isinstance(key, str) and key in fields):
            raise _StrictRejection(
                "the record carries a stored key that is not a declared field", ("<record>",)
            )
    for name, declared in fields.items():
        if name in stored:
            _require_declared_scalar(getattr(record, name, None), declared.annotation, name)
    try:
        # Rebuild from the record's own plain projection. ``model_dump`` restores the structures
        # the declared schemas expect - including Core's immutable contract containers - while the
        # stored-key walk above has already decided which keys may be present at all.
        model.model_validate(model.model_dump(record, mode="python", warnings=False))
    except ValidationError as exc:
        errors = exc.errors()
        raw = tuple(errors[0].get("loc") or ()) if errors else ()
        raise _StrictRejection(
            "the stored fields do not match the declared schema",
            tuple(part for part in raw if not isinstance(part, str) or part in fields),
            exc,
        ) from exc
    nested_seen = seen + (identity,)
    for name, declared in fields.items():
        if name in stored:
            _walk_field(getattr(record, name, None), declared.annotation, nested_seen)


def _require_declared_scalar(value: object, annotation: object,
                             name: str) -> None:
    """Refuse a stored scalar whose basic type is not the declared one.

    This is the narrow guard the declared schemas cannot express, because the same schemas must
    keep accepting the Core legal JSON forms.
    """
    if annotation is bool and not isinstance(value, bool):
        raise _StrictRejection("a stored field has the wrong basic type", (name,))
    if annotation is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise _StrictRejection("a stored field has the wrong basic type", (name,))
    if annotation is float and (isinstance(value, bool) or not isinstance(value, float)):
        raise _StrictRejection("a stored field has the wrong basic type", (name,))
    if annotation is str and not isinstance(value, str):
        raise _StrictRejection("a stored field has the wrong basic type", (name,))


def _walk(value: object, seen: tuple[tuple[int, type], ...]) -> None:
    """Walk one declared nested argument: a model is revalidated, a list is visited in order."""
    if isinstance(value, BaseModel):
        _validate_model(value, seen)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk(item, seen)


def _walk_field(value: object, annotation: object, seen: tuple[tuple[int, type], ...]) -> None:
    if getattr(annotation, "__origin__", None) is list:
        if not isinstance(value, list):
            return
        for item in value:
            _walk(item, seen)
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _walk(value, seen)


class ValueEqualsRule(_VerificationRecord):
    """A rule asserting that one exact path equals one expected JSON value."""

    schema_version: SchemaVersion
    rule_type: Literal["value_equals"]
    path: NonEmptyPath
    expected: JsonValue

    @model_validator(mode="after")
    def _rule_is_well_formed(self) -> ValueEqualsRule:
        _require_rule_path(self.path, type(self).model_fields, "path")
        return self


class PathExistsRule(_VerificationRecord):
    """A rule asserting that one exact path is present.

    No ``expected`` field exists: an expected value on a presence rule is an unknown field and
    is rejected by the closed schema.
    """

    schema_version: SchemaVersion
    rule_type: Literal["path_exists"]
    path: NonEmptyPath

    @model_validator(mode="after")
    def _rule_is_well_formed(self) -> PathExistsRule:
        _require_rule_path(self.path, type(self).model_fields, "path")
        return self


def _require_rule_path(path: object, allowed: dict, field: str) -> None:
    """Reject an unusable path as a rule error, before any evidence is consulted."""
    if not isinstance(path, list) or not path:
        raise VerificationError(
            "VERIFICATION_RULE_INVALID", (field,), "the rule path must be a non-empty list"
        )
    for index, segment in enumerate(path):
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise VerificationError(
                "VERIFICATION_RULE_INVALID",
                (field, index),
                "a path segment must be a string or a non-negative integer",
            )
        if isinstance(segment, int) and segment < 0:
            raise VerificationError(
                "VERIFICATION_RULE_INVALID", (field, index), "an array index must not be negative"
            )
    del allowed


class VerificationEvidence(_VerificationRecord):
    """One observation together with the provider's explicit completeness declaration."""

    schema_version: SchemaVersion
    observation: Observation
    state_complete: StrictBool


class RuleEvaluation(_VerificationRecord):
    """The bounded result of evaluating one rule against one piece of evidence.

    The status, the value presence, the value and the reason are one verdict, not four
    independent fields: the agreed truth table is enforced here so a self-contradictory record
    cannot be built through any ingress.
    """

    schema_version: SchemaVersion
    status: StatusField
    has_actual_value: StrictBool
    actual_value: JsonValue
    reason: Literal["VALUE_MATCH", "VALUE_MISMATCH", "PATH_PRESENT", "PATH_ABSENT",
                    "EVIDENCE_INSUFFICIENT"]

    @model_validator(mode="after")
    def _verdict_is_consistent(self) -> RuleEvaluation:
        """Reject any combination the truth table does not contain.

        Each reason code carries its own presence requirement: a match, a mismatch and a present
        path all report a value, while an absent path and insufficient evidence report none. Only
        an inconclusive verdict may report insufficient evidence.
        """
        status = self.status
        present = self.has_actual_value
        reason = self.reason
        if not present and self.actual_value is not None:
            raise ValueError("an absent actual value must be null")
        if reason in (VALUE_MATCH, VALUE_MISMATCH, PATH_PRESENT):
            if not present:
                raise ValueError("this verdict requires an observed value")
        elif reason == PATH_ABSENT and present:
            raise ValueError("an absent path cannot carry a value")
        if status is VerificationStatus.PASS:
            # A PASS verdict can never report a mismatch: the frozen truth table pairs a mismatch
            # with FAIL and a match with PASS.
            if reason in (VALUE_MISMATCH, PATH_ABSENT, EVIDENCE_INSUFFICIENT):
                raise ValueError("a PASS verdict cannot report this reason")
        elif status is VerificationStatus.FAIL:
            if reason in (VALUE_MATCH, PATH_PRESENT, EVIDENCE_INSUFFICIENT):
                raise ValueError("a FAIL verdict cannot report this reason")
        elif present or reason != EVIDENCE_INSUFFICIENT:
            raise ValueError("an inconclusive verdict reports insufficient evidence only")
        return self


class TaskCompletionCriteria(_VerificationRecord):
    """Independently supplied task-completion conditions.

    ``conditions`` reuses the existing :class:`ExpectedOutcome` shape and must contain at least
    one required condition, so an empty or optional-only set can never yield a PASS.
    """

    schema_version: SchemaVersion
    goal_id: GoalId
    conditions: Annotated[list[ExpectedOutcome], Field(strict=True, min_length=1)]

    @model_validator(mode="after")
    def _at_least_one_required(self) -> TaskCompletionCriteria:
        if not any(condition.required for condition in self.conditions):
            raise ValueError("at least one completion condition must be required")
        return self


_ = (StrictInt,)
