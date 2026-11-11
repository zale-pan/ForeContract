"""The closed records of the deterministic recovery decisions package.

Every record is explicit about its schema version, rejects unknown fields, uses strict field
types and validates its own cross-field rules. All of them share one envelope that enforces four
properties the caller can rely on through ordinary API use:

* an original nested record is inspected from its own stored fields, stored extra keys and stored
  scalar types before any dependent validator or serialization runs, so a record produced by an
  ordinary copy cannot hide an invalid field;
* a raw container cycle is refused with one bounded facility error before the walk recurses, so a
  malformed request is never a recursion failure;
* the record owns its nested data: mutating the caller's nested mapping or list afterwards cannot
  change the record, and mutating the record cannot pollute the caller;
* a refused assignment leaves the previous valid record exactly as it was.

These are ordinary API guarantees only. Reflection, private-attribute surgery and deliberate
construction that bypasses validation are outside them, and nothing here claims process or
operating-system isolation.
"""

from __future__ import annotations

import datetime as datetime_module
import math
import sys
from enum import Enum
from typing import Annotated, ClassVar, Literal, Union, get_origin

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic.types import AwareDatetime

from forecontract.contracts.snapshot import FrozenList, FrozenMapping
from forecontract.models.base import (
    SCHEMA_VERSION,
    NonBlankStr,
    SchemaVersion,
    StrictBool,
)
from forecontract.models.json_types import JsonObject
from forecontract.models.json_types import JsonValue as JsonValueType
from forecontract.models.outcomes import (
    ActualOutcome,
    VerificationResult,
    VerificationStatus,
)
from forecontract.recovery.errors import RecoveryError

__all__ = [
    "ActionSafetyDeclaration",
    "RecoveryAssessment",
    "RecoveryContext",
    "RecoveryEvidence",
    "validate_committed_shape",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", validate_assignment=True)

#: The three assessment kinds. Only the third carries a decision.
KIND_OBSERVATION_REQUIRED = "observation_required"
KIND_NO_RECOVERY = "no_recovery"
KIND_DECISION = "decision"
ASSESSMENT_KINDS = (KIND_OBSERVATION_REQUIRED, KIND_NO_RECOVERY, KIND_DECISION)

#: The three recovery decisions.
DECISION_RETRY = "retry"
DECISION_REPLAN = "replan"
DECISION_ABORT = "abort"
DECISIONS = (DECISION_RETRY, DECISION_REPLAN, DECISION_ABORT)

#: The closed reason vocabulary of a recovery assessment.
STOP_REQUESTED = "STOP_REQUESTED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
POLICY_ABORT = "POLICY_ABORT"
OBSERVATION_REQUIRED = "OBSERVATION_REQUIRED"
STEP_VERIFIED = "STEP_VERIFIED"
STEP_VERIFIED_AFTER_EXECUTION_ANOMALY = "STEP_VERIFIED_AFTER_EXECUTION_ANOMALY"
VERIFICATION_INCONCLUSIVE = "VERIFICATION_INCONCLUSIVE"
EFFECT_UNRESOLVED = "EFFECT_UNRESOLVED"
RETRY_ALLOWED = "RETRY_ALLOWED"
REPLAN_ALLOWED = "REPLAN_ALLOWED"
RECOVERY_NOT_ALLOWED = "RECOVERY_NOT_ALLOWED"
RECOVERY_REASONS = (
    STOP_REQUESTED,
    BUDGET_EXHAUSTED,
    POLICY_ABORT,
    OBSERVATION_REQUIRED,
    STEP_VERIFIED,
    STEP_VERIFIED_AFTER_EXECUTION_ANOMALY,
    VERIFICATION_INCONCLUSIVE,
    EFFECT_UNRESOLVED,
    RETRY_ALLOWED,
    REPLAN_ALLOWED,
    RECOVERY_NOT_ALLOWED,
)

#: The bounded domain effect states. ``unknown`` is an explicit unresolved effect.
EFFECT_NOT_APPLIED = "not_applied"
EFFECT_APPLIED = "applied"
EFFECT_UNKNOWN = "unknown"
EFFECT_STATES = (EFFECT_NOT_APPLIED, EFFECT_APPLIED, EFFECT_UNKNOWN)

#: The repeat-safety declaration values. ``unknown`` is an explicit unknown, not a safe value.
SAFETY_READ_ONLY = "read_only"
SAFETY_IDEMPOTENT = "idempotent"
SAFETY_NON_IDEMPOTENT = "non_idempotent"
SAFETY_UNKNOWN = "unknown"
RETRY_SAFETY_VALUES = (SAFETY_READ_ONLY, SAFETY_IDEMPOTENT, SAFETY_NON_IDEMPOTENT, SAFETY_UNKNOWN)

#: The repeat-safety values that make a repeated effect-bearing action safe on their own.
SAFE_TO_REPEAT = (SAFETY_READ_ONLY, SAFETY_IDEMPOTENT)

#: The reasons that can only be produced after post-attempt evidence was supplied and validated.
#: They must therefore report the associated observation, never a null.
EVIDENCE_DEPENDENT_REASONS = (
    STEP_VERIFIED,
    STEP_VERIFIED_AFTER_EXECUTION_ANOMALY,
    VERIFICATION_INCONCLUSIVE,
    EFFECT_UNRESOLVED,
    RETRY_ALLOWED,
    REPLAN_ALLOWED,
    RECOVERY_NOT_ALLOWED,
)

#: The reasons an immediate branch produces without needing any post-attempt evidence. When
#: evidence was nevertheless supplied and validated, the associated observation is still reported.
IMMEDIATE_REASONS = (STOP_REQUESTED, BUDGET_EXHAUSTED, POLICY_ABORT)

#: The exact two-field shape of a committed abort policy, preserved byte for byte.
ABORT_POLICY_DUMP = {"schema_version": "0.1", "mode": "abort"}

#: The private attributes pydantic stores on an instance; they are not declared fields.
_STORAGE_KEYS = frozenset({"__pydantic_fields_set__", "__pydantic_extra__",
                           "__pydantic_private__"})

#: Distinguishes "this field had no previous value" from "the previous value was null".
_MISSING = object()


def _finite_non_negative(value: object) -> object:
    """Accept a finite non-negative JSON number: never a bool, a string or a non-finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a finite non-negative number")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("must be a finite non-negative number")
    return value


#: A caller-supplied remaining budget: a finite non-negative ``int`` or ``float``, never a bool.
RemainingBudget = Annotated[
    int | float, BeforeValidator(_finite_non_negative), Field(allow_inf_nan=False, ge=0)
]

#: A strict non-negative integer: a bool is not the integer zero or one.
StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

#: An enum ingress that accepts only a member or its JSON string value. Bytes are refused
#: instead of being decoded into an enumerated name.
def _str_enum_only(value: object) -> object:
    """Accept only a string, so a bytes value can never be decoded into an enum name."""
    if isinstance(value, str):
        return value
    raise ValueError("the value must be an enumerated name")


def _closed_name(value: object) -> object:
    """Route a member of a closed name vocabulary, refusing anything that is not a name."""
    if isinstance(value, str):
        return value
    raise RecoveryError("INPUT_INVALID", ("<record>",),
                        "the supplied value is not a member of a closed name vocabulary")


def _closed_name_or_none(value: object) -> object:
    """The same closed-name ingress, with an explicit null for "no decision"."""
    if value is None or isinstance(value, str):
        return value
    raise RecoveryError("INPUT_INVALID", ("<record>",),
                        "the supplied value is not a member of a closed name vocabulary")


def _status_name_only(value: object) -> object:
    """Accept a Core status member or its JSON name, never bytes and never a coercion."""
    if isinstance(value, VerificationStatus):
        return value
    if isinstance(value, str):
        return value
    raise RecoveryError("INPUT_INVALID", ("<record>",),
                        "the supplied value is not a member of a closed name vocabulary")


RetrySafetyField = Annotated[
    Literal["read_only", "idempotent", "non_idempotent", "unknown"],
    BeforeValidator(_str_enum_only),
]
EffectStateField = Annotated[
    Literal["not_applied", "applied", "unknown"],
    BeforeValidator(_str_enum_only),
]
#: A Core outcome status reached through the same strict names-only ingress, so a bytes value can
#: never be decoded into a verdict by any local enum boundary.
StatusField = Annotated[VerificationStatus, BeforeValidator(_status_name_only)]


def _verification_evidence_type() -> type:
    """The existing verification evidence record, imported lazily to keep the packages acyclic."""
    from forecontract.verification import VerificationEvidence

    return VerificationEvidence


def _as_verification_evidence(value: object) -> object:
    """Accept a real evidence record or its own serialized mapping, and rebuild it.

    An already-valid instance is rebuilt through the verification ingress so the local record owns
    its nested observation, and a serialized mapping is reconstructed from its original structure.
    A damaged structure is refused rather than washed through a dump, and the refusal is reported
    as this package's evidence error rather than as a raw dependency error.
    """
    from forecontract.verification import VerificationError

    try:
        return _verification_evidence_type().model_validate(value)
    except RecoveryError:
        raise
    except (VerificationError, ValidationError) as exc:
        raise RecoveryError("EVIDENCE_INVALID", ("verification_evidence",),
                            "the supplied verification evidence is not structurally valid",
                            cause=exc) from exc


#: The nested evidence field: a real record, or its own serialized mapping.
VerificationEvidenceField = Annotated[object, BeforeValidator(_as_verification_evidence)]


def _key_location(key: object, index: int = 0) -> str | int:
    """Render a container key for a diagnostic path without ever echoing caller data.

    A mapping key is raw caller JSON, so it is never published: a key that is a legal Python
    identifier becomes a fixed key placeholder and any other key becomes a fixed index
    placeholder. Array indices are already structural and are kept.
    """
    if isinstance(key, str) and key.isidentifier():
        return "<key>"
    if isinstance(key, bool):
        return "<key>"
    if isinstance(key, int):
        return key
    return index


def _validated_fields(value: BaseModel, location: tuple[str | int, ...], code: str,
                      active: set[int]) -> None:
    """Refuse an undeclared stored key or a wrong stored scalar, walking every stored field.

    ``active`` is the one descent set shared by the whole entry, so a record reached again through
    any depth of nested models and containers is refused as a cycle before anything recurses.
    """
    declared = tuple(type(value).model_fields)
    stored = getattr(value, "__dict__", None)
    if not isinstance(stored, dict):
        return
    for key in stored:
        if key not in declared and key not in _STORAGE_KEYS:
            raise RecoveryError(code, location,
                                "the supplied record carries a key its schema does not declare")
    for key, previous in stored.items():
        if key not in declared:
            continue
        _check_stored(previous, key, location, code, active)


def _check_stored(value: object, name: str, location: tuple[str | int, ...], code: str,
                  active: set[int]) -> None:
    """Refuse a stored scalar whose basic type is not the declared one and walk nested data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    _walk_container(value, location + (name,), code, active)


def _enter(value: object, location: tuple[str | int, ...], code: str,
           active: set[int]) -> int:
    """Register a record or container on the current descent, refusing a repeat."""
    identity = id(value)
    if identity in active:
        raise RecoveryError(code, location,
                            "the supplied record contains a cyclic reference")
    active.add(identity)
    return identity


def _walk_container(value: object, location: tuple[str | int, ...], code: str,
                   active: set[int]) -> None:
    """Walk one stored container on the shared descent, refusing a cycle before recursing."""
    if isinstance(value, BaseModel):
        identity = _enter(value, location, code, active)
        try:
            _validated_fields(value, location, code, active)
        finally:
            active.discard(identity)
        return
    if isinstance(value, dict):
        identity = _enter(value, location, code, active)
        try:
            for position, key in enumerate(value):
                _walk_container(value[key], location + (_key_location(key, position),), code,
                                active)
        finally:
            active.discard(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = _enter(value, location, code, active)
        try:
            for index, item in enumerate(value):
                _walk_container(item, location + (index,), code, active)
        finally:
            active.discard(identity)


def _detached(value: object, code: str, location: tuple[str | int, ...],
              active: set[int]) -> object:
    """Return a detached copy of one nested value, refusing a container cycle.

    ``active`` carries the identities of every record on the current descent and is threaded
    through every model field, so a record that reaches itself through its own fields is refused
    before the walk recurses while a value legitimately referenced twice is still accepted. Each
    leaf container is walked on a fresh descent path, so a shared reference between two sibling
    fields is not mistaken for a cycle.
    """
    if isinstance(value, BaseModel):
        identity = _enter(value, location, code, active)
        _validated_fields(value, location, code, active)
        stored = getattr(value, "__dict__", None)
        if not isinstance(stored, dict):
            active.discard(identity)
            return value
        try:
            state = {key: _detached(item, code, location, active)
                     for key, item in stored.items() if key not in _STORAGE_KEYS}
            try:
                return type(value).model_validate(state)
            except RecoveryError:
                raise
            except ValidationError as exc:
                # A nested Core record that cannot be rebuilt from its own stored fields is
                # damaged input, never a raw dependency error.
                raise RecoveryError(code, location,
                                    "a nested record does not match its declared schema",
                                    cause=exc) from exc
        finally:
            active.discard(identity)
    if isinstance(value, (list, tuple)):
        identity = _enter(value, location, code, active)
        try:
            return [_detached(item, code, location, active) for item in value]
        finally:
            active.discard(identity)
    if isinstance(value, dict):
        identity = _enter(value, location, code, active)
        try:
            detached: dict = {}
            for position, key in enumerate(value):
                detached[key] = _detached(value[key], code,
                                          location + (_key_location(key, position),), active)
            return detached
        finally:
            active.discard(identity)
    return value


def _detach_nested(data: object, code: str, location: tuple[str | int, ...]) -> object:
    """Validate and detach every nested record a caller hands to a public ingress."""
    if isinstance(data, dict):
        active: set[int] = set()
        detached: dict = {}
        for position, key in enumerate(data):
            detached[key] = _detached(data[key], code,
                                      location + (_key_location(key, position),), active)
        return detached
    return data


def _as_facility_error(exc: ValidationError, code: str,
                       fields: tuple[str, ...]) -> RecoveryError:
    """Convert a dependency validation failure into one bounded facility error.

    Only a declared field name survives from the dependency's own location, so a location can
    never carry a caller-supplied mapping key.
    """
    errors = exc.errors()
    raw = tuple(errors[0].get("loc") or ()) if errors else ()
    location = tuple(part for part in raw if isinstance(part, int) or part in fields)
    return RecoveryError(code, location or ("<record>",),
                         "the supplied record does not match the declared schema", cause=exc)


def _cycle_free(value: object, active: set[int]) -> bool:
    """Whether a value is a finite JSON structure free of a container cycle."""
    if isinstance(value, BaseModel):
        return _sound_record(value, active)
    identity = id(value)
    if isinstance(value, dict):
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(_cycle_free(value[key], active) for key in value)
        finally:
            active.discard(identity)
    if isinstance(value, (list, tuple)):
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(_cycle_free(item, active) for item in value)
        finally:
            active.discard(identity)
    if isinstance(value, (bytes, bytearray)):
        return False
    if value is None or isinstance(value, (bool, int, float, str, datetime_module.datetime)):
        return True
    # The Core immutable containers are neither a mapping nor a sequence by inheritance, so they
    # are walked through their own interface rather than rejected as an unknown object.
    if callable(getattr(value, "items", None)):
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(_cycle_free(value[key], active) for key in value)
        finally:
            active.discard(identity)
    if isinstance(value, (dict, list, tuple)) or (
            hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray))):
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        try:
            if isinstance(value, dict):
                return all(_cycle_free(value[key], active) for key in value)
            return all(_cycle_free(item, active) for item in value)
        finally:
            active.discard(identity)
    return False


def validate_committed_shape(value: object, location: tuple[str | int, ...],
                             code: str) -> None:
    """Validate a committed record production-side shape before the recovery module consumes it.

    The Contract Builder and its snapshot module are read-only for this round, and the existing
    committed records do not re-derive their own stored state when they are handed an instance, so
    a record produced by an ordinary copy could otherwise be consumed as if it were valid. This
    checks the record's declared fields, its undeclared stored keys, its stored scalar types, its
    schema and its nested cycles, and refuses anything that does not match, without modifying,
    freezing or dumping the record. A legal deeply immutable contract and its containers are
    accepted unchanged.
    """
    if not isinstance(value, BaseModel) or not _sound_record(value, set()):
        raise RecoveryError(code, location,
                            "the supplied committed record does not match its declared schema")
    if not _contract_is_intrinsically_sound(value):
        raise RecoveryError(code, location,
                            "the supplied committed record does not satisfy its own structure")


def _scalar_matches(value: object, info: object) -> bool:
    """Whether a scalar is acceptable for a declared field, including its numeric bounds.

    Only an exact scalar annotation and the metadata constraints that go with it are enforced
    here: a union such as ``int | float``, a strict string or boolean, a timestamp, an enumerated
    name and the positivity or non-negativity of a JSON number. A number that cannot be converted
    at all, such as an integer far outside the finite float range, is refused as a bounded
    mismatch rather than allowed to raise. Model fields, sequences and the Core immutable
    containers are left to the structural walk, because their production schemas transform a
    validated value into a frozen snapshot that the raw schema would not accept a second time.
    """
    annotation = getattr(info, "annotation", None)
    metadata = list(getattr(info, "metadata", ()) or ())
    union = getattr(annotation, "__args__", ())
    if annotation in (int, float) or set(union) == {int, float}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            number = float(value)
        except (OverflowError, ValueError):
            return False
        if not math.isfinite(number):
            return False
        if any(getattr(limit, "gt", None) == 0 for limit in metadata) and number <= 0:
            return False
        return not (any(getattr(limit, "ge", None) == 0 for limit in metadata)
                    and number < 0)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is datetime_module.datetime:
        return _aware_timestamp(value)
    return True


def _aware_timestamp(value: object) -> bool:
    """Whether a committed timestamp is a real timezone-aware datetime."""
    if not isinstance(value, datetime_module.datetime):
        return False
    return value.tzinfo is not None and value.utcoffset() is not None


def _annotation_args(annotation: object) -> tuple:
    """The declared type arguments of an annotation, unwrapping the Core wrappers.

    A production field is written as ``Annotated[T, AfterValidator(...)]``, as a strict list of a
    model, or as an ``AfterValidator``-derived frozen type. The first two unwrap through their
    arguments; the frozen type has none, so the declared model is recovered from the validation
    function the Core snapshot module attached, which is looked up in that function's own module.
    """
    current = annotation
    for _ in range(8):
        args = getattr(current, "__args__", ())
        if args:
            origin = getattr(current, "__origin__", None)
            if getattr(current, "__metadata__", None) is not None:
                current = args[0]
                continue
            if origin is not None:
                return args
            current = args[0]
            continue
    return ()


def _frozen_origin(annotation: object) -> type | None:
    """The declared model a frozen annotation stands for, if any.

    The Core snapshot module builds a frozen subclass from the declared model and attaches its
    freezing function through an ``AfterValidator``. The declared model can be recovered either
    because a field of the frozen type is annotated with it, or because the frozen type is named
    after it, which is how the module names every one of them.
    """
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        return None
    for name, member in vars(annotation).items():
        metadata = getattr(member, "__metadata__", None)
        if metadata is None:
            continue
        for item in metadata:
            function = getattr(item, "func", None)
            module = getattr(function, "__module__", None) if function else None
            for candidate in vars(sys.modules.get(module, object())).values():
                if (isinstance(candidate, type) and issubclass(candidate, BaseModel)
                        and candidate is not annotation
                        and getattr(candidate, name, None) is not None):
                    return candidate
    if annotation.__name__.startswith("Frozen"):
        declared = annotation.__name__[len("Frozen"):]
        for candidate in vars(sys.modules.get(annotation.__module__, object())).values():
            if (isinstance(candidate, type) and candidate is not annotation
                    and candidate.__name__ == declared
                    and not candidate.__name__.startswith("Frozen")):
                return candidate
    return None


def _is_frozen_mapping(value: object) -> bool:
    """Whether a value is the Core immutable JSON object container, by real type identity.

    A class name or a module label is not identity: an unrelated mapping that merely repeats the
    production class name is an arbitrary mapping and is checked as one.
    """
    return isinstance(value, FrozenMapping)


def _requires_non_blank(metadata: object) -> bool:
    """Whether a field carries the Core non-blank validator, which a plain string would pass.

    A blank value can only be refused by the validator function the Core model attaches, because
    the declared annotation of such a field is a plain string. A committed record that was built
    by a validation-bypass construction never ran that validator, so the check has to be repeated
    here from the same metadata.
    """
    for item in list(metadata or ()):
        function = getattr(item, "func", None)
        if function is not None and getattr(function, "__name__", "") == "_non_blank":
            return True
    return False


def _raw_cycle_precheck(value: object, active: set[int], done: set[int],
                        location: tuple[str | int, ...]) -> bool:
    """Walk a raw stored value once, refusing a cycle and accepting a repeated reference.

    This runs before any recursive declared-type or JSON-content validation, so a self-referencing
    mapping or a mixed mapping-and-list cycle can never reach a recursive validator. ``active``
    holds the identities of the current descent, so a value that reaches itself is a cycle, and
    ``done`` holds the identities already fully walked, so a value legitimately referenced twice
    is accepted instead of being mistaken for one. Both sets are threaded through every model field
    and container, which also covers a cycle that crosses a model boundary.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, bytearray,
                                          datetime_module.datetime)):
        return True
    if isinstance(value, BaseModel):
        identity = id(value)
        if identity in done:
            return True
        if identity in active:
            return False
        declared = type(value).model_fields
        stored = getattr(value, "__dict__", None)
        if not isinstance(stored, dict):
            return False
        active.add(identity)
        try:
            for key in declared:
                if key not in stored:
                    return False
                if not _raw_cycle_precheck(stored[key], active, done, location + (key,)):
                    return False
        finally:
            active.discard(identity)
        done.add(identity)
        return True
    if callable(getattr(value, "items", None)):
        identity = id(value)
        if identity in done:
            return True
        if identity in active:
            return False
        active.add(identity)
        try:
            for key in value:
                if not _raw_cycle_precheck(value[key], active, done, location + ("<key>",)):
                    return False
        finally:
            active.discard(identity)
        done.add(identity)
        return True
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in done:
            return True
        if identity in active:
            return False
        active.add(identity)
        try:
            for item in value:
                if not _raw_cycle_precheck(item, active, done, location + ("<item>",)):
                    return False
        finally:
            active.discard(identity)
        done.add(identity)
        return True
    return True


def _declared_type_matches(value: object, annotation: object, metadata: object = (),
                           active: frozenset[int] = frozenset(),
                           done: set[int] | None = None) -> bool:
    """Whether a value matches its declared annotation, recursively.

    ``active`` and ``done`` are the shared descent sets of the caller, so a container that reaches
    itself is refused here too rather than recursing, while a repeated reference is accepted.
    Scalars are compared exactly, so an integer field refuses a float and a non-blank string field
    refuses a blank one; a sequence is compared element by element against the declared element
    type; and a JSON object is compared key by key and value by value. Only a real list, or the
    Core frozen sequence a committed contract actually holds, stands in for a declared sequence: an
    arbitrary tuple does not.
    """
    if annotation is None:
        return True
    if done is None:
        done = set()
    if annotation is str:
        return isinstance(value, str) and not (_requires_non_blank(metadata)
                                               and not value.strip())
    origin = get_origin(annotation)
    arguments = _annotation_args(annotation)
    if isinstance(annotation, type) and _frozen_origin(annotation) is not None:
        # The declared model accepts both the frozen subclass the snapshot module produced and the
        # declared model itself, because the frozen class is a subclass of the declared one.
        return isinstance(value, annotation) and _sound_record(value, set())
    if origin is Annotated or hasattr(annotation, "__metadata__"):
        return (_declared_type_matches(value, arguments[0], active=active)
                if arguments else True)
    if origin is Union or str(origin) == "typing.Union":
        if value is None and type(None) in arguments:
            return True
        return any(_declared_type_matches(value, item, active=active, done=done)
                   for item in arguments if item is not type(None))
    if origin in (list,):
        element = arguments[0] if arguments else object
        if isinstance(value, list) or _is_frozen_sequence(value):
            identity = id(value)
            if identity in done:
                return True
            if identity in active:
                return False
            descent = active | {identity}
            return all(_declared_type_matches(item, element, active=descent, done=done)
                       for item in value)
        return False
    if annotation is AwareDatetime:
        return _aware_timestamp(value)
    if annotation is JsonObject or _is_frozen_mapping(value):
        return _json_object_matches(value, active, done)
    if origin is dict:
        return _json_object_matches(value, active, done)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return True
        return isinstance(value, str) and value in {member.value for member in annotation}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return isinstance(value, BaseModel) and isinstance(value, annotation) \
            and _sound_record(value, set())
    if annotation is float or _is_number_union(arguments):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            number = float(value)
        except (OverflowError, ValueError):
            # An integer far outside the finite float range cannot be a JSON finite number.
            return False
        if not math.isfinite(number):
            return False
        if _numeric_limit(metadata, "gt") == 0 and number <= 0:
            return False
        return _numeric_limit(metadata, "ge") != 0 or number >= 0
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is datetime_module.datetime:
        return _aware_timestamp(value)
    if annotation is object or annotation is JsonValueType:
        return _json_value_matches(value, active, done)
    if isinstance(annotation, type):
        return isinstance(value, annotation)
    return True


def _is_frozen_sequence(value: object) -> bool:
    """Whether a value is the Core immutable sequence a committed contract holds.

    Only the real production class counts. An unrelated tuple subclass that merely repeats the
    ``FrozenList`` name is an arbitrary tuple and is refused wherever a JSON array or a declared
    sequence is required, which is why identity is compared instead of a name or a module label.
    """
    return isinstance(value, FrozenList)


def _is_number_union(arguments: tuple) -> bool:
    """Whether a declared union is the JSON number union, so an int or a float is legal."""
    return set(arguments) == {int, float}


def _numeric_limit(metadata: object, name: str) -> object:
    """The declared numeric limit of a field, or ``None`` when the constraint is absent."""
    for item in list(metadata or ()):
        limit = getattr(item, name, None)
        if limit is not None:
            return limit
    return None


def _json_object_matches(value: object, active: frozenset[int] = frozenset(),
                         done: set[int] | None = None) -> bool:
    """Whether a value is a JSON object: a mapping with string keys and finite JSON values."""
    if not callable(getattr(value, "items", None)):
        return False
    if done is None:
        done = set()
    identity = id(value)
    if identity in done:
        return True
    if identity in active:
        return False
    descent = active | {identity}
    return all(isinstance(key, str) and _json_value_matches(item, descent, done)
               for key, item in value.items())


def _json_value_matches(value: object, active: frozenset[int] = frozenset(),
                        done: set[int] | None = None) -> bool:
    """Whether a value is a finite JSON value: no bytes, no timestamp, no unknown object.

    A JSON array is a real list, never an arbitrary tuple: the only tuple that may stand in for one
    is the Core immutable sequence a committed contract actually holds. ``done`` records the
    identities already walked, so a value referenced by two keys is accepted rather than mistaken
    for a cycle.
    """
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (bytes, bytearray, datetime_module.datetime)):
        return False
    if done is None:
        done = set()
    if isinstance(value, list) or _is_frozen_sequence(value):
        identity = id(value)
        if identity in done:
            return True
        if identity in active:
            return False
        descent = active | {identity}
        return all(_json_value_matches(item, descent, done) for item in value)
    if callable(getattr(value, "items", None)):
        return _json_object_matches(value, active, done)
    return False


def _sound_record(value: BaseModel, active: set[int]) -> bool:
    """Whether one committed record and everything it stores is structurally sound.

    The raw structural descent runs first and in full, so a cycle anywhere in the stored data is
    refused before any recursive declared-type or JSON-content validation can recurse into it.
    """
    if isinstance(value, (bytes, bytearray)):
        return False
    identity = id(value)
    if identity in active:
        return False
    model = type(value)
    declared = model.model_fields
    stored = getattr(value, "__dict__", None)
    if not isinstance(stored, dict):
        return False
    for key in stored:
        if key not in declared and key not in _STORAGE_KEYS:
            return False
    for key in declared:
        if key not in stored:
            return False
    if not _raw_cycle_precheck(value, set(), set(), ()):
        return False
    active.add(identity)
    try:
        for key in declared:
            info = declared[key]
            if not _declared_type_matches(stored[key], info.annotation, info.metadata):
                return False
            if not _declared_value_matches(stored[key], key, active):
                return False
    finally:
        active.discard(identity)
    return True


def _declared_value_matches(value: object, name: str, active: set[int]) -> bool:
    """Whether one stored value is acceptable for its declared field.

    The structural walk accepts a committed string, number, timestamp, enumerated name or
    immutable container of those, and refuses a bytes value, an unknown object or a cyclic
    container. The scalar and nested-model checks above cover what the walk alone cannot express.
    """
    if name == "schema_version":
        return value == SCHEMA_VERSION
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, datetime_module.datetime):
        return _aware_timestamp(value)
    if isinstance(value, (bytes, bytearray)):
        return False
    if isinstance(value, BaseModel):
        return _sound_record(value, active)
    # The Core immutable containers are neither a mapping nor a sequence by inheritance, so they
    # are reached through their own interface before the plain JSON containers.
    if callable(getattr(value, "items", None)):
        return _cycle_free(value, active)
    if isinstance(value, (dict, list, tuple)) or (
            hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray))):
        return _cycle_free(value, active)
    return False


def _contract_is_intrinsically_sound(value: object) -> bool:
    """Whether a committed contract satisfies its own intrinsic structure.

    ``ActionContract`` requires at least one required expected outcome and exactly one
    verification binding for every outcome, and a committed abort policy must carry no bounded
    control. A record that is not a contract and not a policy is unaffected by this check; a
    record that is a policy is judged only on its control consistency. The module refuses a
    damaged record at its own boundary without redefining or rewriting the record.
    """
    if isinstance(value, _policy_type()):
        return _policy_controls_are_consistent(value)
    outcomes = getattr(value, "expected_outcomes", None)
    policy = getattr(value, "recovery_policy", None)
    bindings = getattr(getattr(value, "verification_policy", None), "bindings", None)
    if outcomes is None or bindings is None or policy is None:
        return True
    try:
        if not any(outcome.required for outcome in outcomes):
            return False
        indices = [binding.outcome_index for binding in bindings]
        if len(set(indices)) != len(indices):
            return False
        if sorted(indices) != list(range(len(outcomes))):
            return False
    except AttributeError:
        return False
    return _policy_controls_are_consistent(policy)


def _policy_type() -> type:
    """The committed recovery policy class, imported lazily to keep the modules acyclic."""
    from forecontract.contracts.models import RecoveryPolicy

    return RecoveryPolicy


def _policy_controls_are_consistent(policy: object) -> bool:
    """Whether a committed policy's stored controls agree with its own mode."""
    mode = getattr(policy, "mode", None)
    max_retries = getattr(policy, "max_retries", None)
    allow_replan = getattr(policy, "allow_replan", None)
    if mode == "abort":
        return max_retries is None and allow_replan is None
    if mode == "bounded":
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            return False
        return isinstance(allow_replan, bool)
    return False


class _RecoveryRecord(BaseModel):
    """The shared closed envelope of the recovery records."""

    model_config = _MODEL_CONFIG

    #: The facility code this record reports for damaged input.
    _error_code: ClassVar[str] = "INPUT_INVALID"

    #: Guards the nested rebuild so that detaching does not recurse through itself.
    _detaching: ClassVar[bool] = False

    def __init__(self, /, **data: object) -> None:
        # ``self`` is positional-only, exactly as pydantic declares its own constructor, so a
        # caller mapping carrying an unknown key named ``self`` is forwarded intact and rejected
        # by the closed schema instead of colliding with the parameter.
        if not type(self)._detaching:
            type(self)._detaching = True
            try:
                data = _detach_nested(data, type(self)._error_code, ())
            finally:
                type(self)._detaching = False
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise _as_facility_error(exc, type(self)._error_code,
                                     tuple(type(self).model_fields)) from None

    def __setattr__(self, name: str, value: object) -> None:
        """Assign one declared field transactionally after validating nested content."""
        stored = self.__dict__
        if name.startswith("_") or not isinstance(stored, dict) or name not in type(self).model_fields:
            super().__setattr__(name, value)
            return
        if isinstance(value, BaseModel) or (isinstance(value, list)
                                            and any(isinstance(item, BaseModel)
                                                    for item in value)):
            value = _detached(value, type(self)._error_code, (name,), set())
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

        An existing instance is validated from its own stored fields and rebuilt, so the returned
        record never shares mutable nested data with the argument.
        """
        if isinstance(obj, cls):
            obj = _detached(obj, cls._error_code, (), set())
        else:
            obj = _detach_nested(obj, cls._error_code, ())
        try:
            return super().model_validate(obj, **kwargs)
        except RecoveryError:
            raise
        except ValidationError as exc:
            raise _as_facility_error(exc, cls._error_code, tuple(cls.model_fields)) from None

    @classmethod
    def model_validate_json(cls, json_data: object, **kwargs: object) -> object:
        """Public JSON ingress, including a document that never parsed."""
        try:
            return super().model_validate_json(json_data, **kwargs)
        except RecoveryError:
            raise
        except ValidationError as exc:
            raise _as_facility_error(exc, cls._error_code, tuple(cls.model_fields)) from None

    def model_copy(self, **kwargs: object) -> object:  # type: ignore[override]
        """Copy a record, checking any updated field against its declared schema.

        ``model_copy`` writes the update straight into the instance dictionary, so without this an
        updated field would never meet its own schema and an invalid value could be carried into a
        record that every other ingress refuses.
        """
        updated = kwargs.get("update")
        copy = BaseModel.model_copy(self, **kwargs)  # type: ignore[arg-type]
        if isinstance(updated, dict):
            _validate_updated_fields(copy, updated)
        return copy


def _validate_updated_fields(copy: object, updated: dict) -> None:
    """Refuse a ``model_copy`` update whose value does not match its declared field schema."""
    fields = type(copy).model_fields
    for name in updated:
        if name not in fields:
            raise RecoveryError("INPUT_INVALID", ("<record>",),
                                "an updated key is not a declared field")
    state = copy.model_dump(mode="python", warnings=False)
    state.update(updated)
    try:
        type(copy).model_validate(state)
    except RecoveryError:
        raise
    except ValidationError as exc:
        errors = exc.errors()
        raw = tuple(errors[0].get("loc") or ()) if errors else ()
        raise RecoveryError(
            "INPUT_INVALID",
            tuple(part for part in raw if isinstance(part, str) and part in fields)
            or ("<record>",),
            "an updated field does not match the declared schema",
            cause=exc,
        ) from exc


class RecoveryContext(_RecoveryRecord):
    """The caller-supplied current attempt, budget and stop intent.

    The budget and the counters belong to the caller: this module reads them and never rewrites,
    decrements or resets them. No clock is read and no identifier is generated.
    """

    schema_version: SchemaVersion
    run_id: NonBlankStr
    step_id: NonBlankStr
    contract_id: NonBlankStr
    attempt_id: NonBlankStr
    actual_outcome: ActualOutcome
    remaining_budget_ms: RemainingBudget
    remaining_retries: StrictNonNegativeInt
    remaining_replans: StrictNonNegativeInt
    contract_retry_count: StrictNonNegativeInt
    stop_requested: StrictBool


class ActionSafetyDeclaration(_RecoveryRecord):
    """The domain-provided repeat-safety declaration of the contracted action.

    A declaration is supplied by the adapter or domain layer; it is never inferred from the action
    name, an error message, ``recoverable`` or a planner hint. ``unknown`` is a legal explicit
    unknown and does not mean the action may be repeated safely.
    """

    schema_version: SchemaVersion
    contract_id: NonBlankStr
    adapter_name: NonBlankStr
    action_type: NonBlankStr
    retry_safety: RetrySafetyField


class RecoveryEvidence(_RecoveryRecord):
    """The post-attempt evidence of the current attempt.

    ``effect_state`` is a domain judgement about the effect, so a failed step does not imply that
    nothing happened. ``preconditions_status`` is a judgement about current state; an earlier
    PASS taken before the attempt cannot be passed off as a new one.
    """

    _error_code: ClassVar[str] = "EVIDENCE_INVALID"

    schema_version: SchemaVersion
    run_id: NonBlankStr
    step_id: NonBlankStr
    contract_id: NonBlankStr
    attempt_id: NonBlankStr
    outcome_id: NonBlankStr
    verification_evidence: VerificationEvidenceField
    step_result: VerificationResult
    effect_state: EffectStateField
    preconditions_status: StatusField

    @model_validator(mode="after")
    def _evidence_is_well_formed(self) -> RecoveryEvidence:
        """Require a real verification evidence record with an explicitly complete state."""
        if not isinstance(self.verification_evidence, _verification_evidence_type()):
            # Pydantic requires ValueError here so the rejection becomes a validation error.
            raise ValueError(  # noqa: TRY004
                "verification_evidence must be a VerificationEvidence record")
        return self


class RecoveryAssessment(_RecoveryRecord):
    """The bounded recovery decision for one attempt.

    The kind, the decision and the reason are one verdict, not three free fields: the legal
    combinations are enforced here, so a self-contradictory assessment cannot be built. An
    out-of-vocabulary name is a facility error rather than a dependency schema error, so a caller
    never sees a raw pydantic failure from the public closed vocabularies.
    """

    schema_version: SchemaVersion
    kind: Annotated[Literal["observation_required", "no_recovery", "decision"],
                    BeforeValidator(_closed_name)]
    decision: Annotated[Literal["retry", "replan", "abort"] | None,
                        BeforeValidator(_closed_name_or_none)]
    reason: Annotated[
        Literal[
            "STOP_REQUESTED",
            "BUDGET_EXHAUSTED",
            "POLICY_ABORT",
            "OBSERVATION_REQUIRED",
            "STEP_VERIFIED",
            "STEP_VERIFIED_AFTER_EXECUTION_ANOMALY",
            "VERIFICATION_INCONCLUSIVE",
            "EFFECT_UNRESOLVED",
            "RETRY_ALLOWED",
            "REPLAN_ALLOWED",
            "RECOVERY_NOT_ALLOWED",
        ],
        BeforeValidator(_closed_name),
    ]
    run_id: NonBlankStr
    step_id: NonBlankStr
    contract_id: NonBlankStr
    attempt_id: NonBlankStr
    outcome_id: NonBlankStr
    observation_id: NonBlankStr | None

    @model_validator(mode="after")
    def _verdict_is_consistent(self) -> RecoveryAssessment:
        """Reject any kind, decision, reason and observation combination the contract excludes.

        Every reason that depends on the evidence requires a non-null observation association, and
        only an observation request reports none. An immediate stop, exhausted budget or policy
        abort makes no extra demand on the runtime, but when evidence was supplied and validated
        the associated observation is still reported, so a decision and the evidence it was made
        from always agree.
        """
        kind = self.kind
        decision = self.decision
        reason = self.reason
        if reason == OBSERVATION_REQUIRED:
            if kind != KIND_OBSERVATION_REQUIRED or decision is not None:
                raise ValueError("an observation request carries no decision")
        elif reason in (STEP_VERIFIED, STEP_VERIFIED_AFTER_EXECUTION_ANOMALY):
            if kind != KIND_NO_RECOVERY or decision is not None:
                raise ValueError("a verified step is not a recovery decision")
        elif reason == RETRY_ALLOWED:
            if kind != KIND_DECISION or decision != DECISION_RETRY:
                raise ValueError("a retry reason requires the retry decision")
        elif reason == REPLAN_ALLOWED:
            if kind != KIND_DECISION or decision != DECISION_REPLAN:
                raise ValueError("a replan reason requires the replan decision")
        elif kind != KIND_DECISION or decision != DECISION_ABORT:
            raise ValueError("this reason requires the abort decision")
        if kind == KIND_OBSERVATION_REQUIRED:
            if self.observation_id is not None:
                raise ValueError("an observation request reports no observation")
        elif reason in EVIDENCE_DEPENDENT_REASONS and self.observation_id is None:
            raise ValueError("this verdict requires the observation it was judged on")
        return self


#: The verdict lines the assessment may carry, used by the engine to construct legal records.
