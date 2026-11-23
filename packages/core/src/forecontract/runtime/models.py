"""Strict records, the closed phase vocabulary and the ingress shape guard of the Runtime.

Every Runtime record is a local, ``schema_version="0.1"``, closed-shape pydantic model: an unknown
field, a non-string identifier, a boolean in a numeric position, a non-finite number, a fractional
count or an out-of-vocabulary phase name is refused. Nothing here carries a raw observation, a
forecast payload, a free-text exception or a secret, because the records are the only things the
Runtime hands back to a host.

The four Core records the Runtime *carries* (a task criteria, an observation, a policy) are owned by
their own modules and keep their own already-verified validation. This module only checks that the
value is the declared Core type and that the Runtime-owned JSON payload it stores is still finite
JSON, so a copied or constructed instance cannot smuggle a tuple, a timestamp or a cycle past the
boundary before it is serialized.
"""

from __future__ import annotations

import datetime as datetime_module
import math
from enum import Enum
from typing import Annotated, Any, ClassVar, Union, get_args, get_origin

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from forecontract.contracts.models import RecoveryPolicy, VerificationPolicy
from forecontract.contracts.snapshot import FrozenList, FrozenMapping
from forecontract.models.actions import Precondition
from forecontract.models.base import (
    NonBlankStr,
    SchemaVersion,
    StrictInt,
)
from forecontract.verification.models import TaskCompletionCriteria

__all__ = [
    "PHASES",
    "TERMINAL_PHASES",
    "ContractPreparation",
    "RunRequest",
    "RunResult",
    "RuntimeBudgetSnapshot",
    "RuntimePhase",
    "RuntimeReason",
    "RuntimeRecord",
    "RuntimeTerminalStatus",
    "RuntimeTransition",
    "detached_state",
    "isolated_record",
    "plain_containers",
    "record_payload",
    "require_acyclic",
    "require_declared_record",
    "require_json_shape",
    "stored_payload",
    "validate_declared_fields",
    "validated_budget_value",
]

#: ``extra="forbid"`` refuses an unknown field; ``validate_assignment`` makes a later assignment go
#: through the same schema, so a valid record cannot be turned into an invalid one in place. The
#: records are deliberately *not* frozen: the Runtime holds one live progress record per Run, and a
#: frozen shell would only push the same mutation into a replacement object without adding a
#: guarantee.
RECORD_CONFIG = ConfigDict(extra="forbid", validate_assignment=True)

#: A record that must not change after it is built. A transition that happened is a fact, so the
#: frozen configuration refuses a later rewrite of its target phase instead of accepting a claim
#: about a transition that never took place.
FROZEN_RECORD_CONFIG = ConfigDict(extra="forbid", frozen=True)

#: The closed phase vocabulary. The ordered tuple is the single public statement of the names.
PHASES: tuple[str, ...] = (
    "NEW",
    "OBSERVING",
    "CHECKING_TASK",
    "PLANNING",
    "BUILDING",
    "CHECKING_PRECONDITIONS",
    "EXECUTING",
    "VERIFYING",
    "RECOVERING",
    "CANCELLED",
    "TIMED_OUT",
    "COMPLETED",
    "FAILED",
    "ABORTED",
)

#: The terminal phases. A terminal phase has no outgoing transition at all.
TERMINAL_PHASES: frozenset[str] = frozenset({"COMPLETED", "FAILED", "ABORTED", "CANCELLED",
                                             "TIMED_OUT"})

#: The phases in which a Run may still perform business work. These are the phases from which a
#: genuine cancellation, a real deadline expiry or a facility failure may reach a terminal phase.
ACTIVE_PHASES: tuple[str, ...] = (
    "NEW",
    "OBSERVING",
    "CHECKING_TASK",
    "PLANNING",
    "BUILDING",
    "CHECKING_PRECONDITIONS",
    "EXECUTING",
    "VERIFYING",
    "RECOVERING",
)


class RuntimePhase(str, Enum):
    """One discrete phase of a single sequential Run."""

    NEW = "NEW"
    OBSERVING = "OBSERVING"
    CHECKING_TASK = "CHECKING_TASK"
    PLANNING = "PLANNING"
    BUILDING = "BUILDING"
    CHECKING_PRECONDITIONS = "CHECKING_PRECONDITIONS"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class RuntimeTerminalStatus(str, Enum):
    """The terminal status of a Run, in the existing trace vocabulary.

    ``cancelled`` and ``timeout`` are existing literals of the trace enum; this round is the first to
    produce them from a real trigger, and neither is spelled ``abort``.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RuntimeReason(str, Enum):
    """The bounded Runtime reason vocabulary.

    These are the Runtime's own diagnosis. The trace terminal fact reuses the existing closed
    trace code set instead, so no custom reason is ever forced into that older schema.
    """

    TASK_ACHIEVED = "TASK_ACHIEVED"
    MAX_STEPS_EXHAUSTED = "MAX_STEPS_EXHAUSTED"
    MAX_PLANNING_EXHAUSTED = "MAX_PLANNING_EXHAUSTED"
    MAX_REPLANS_EXHAUSTED = "MAX_REPLANS_EXHAUSTED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    POLICY_ABORT = "POLICY_ABORT"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    EVIDENCE_INVALID = "EVIDENCE_INVALID"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    TRACE_FAILURE = "TRACE_FAILURE"
    CLEANUP_FAILURE = "CLEANUP_FAILURE"
    USER_CANCELLED = "USER_CANCELLED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    OPERATION_TIMEOUT = "OPERATION_TIMEOUT"


def _closed_name(value: object) -> object:
    """Accept only a member or the exact string name of a closed Runtime vocabulary."""
    if isinstance(value, str):
        return value
    if isinstance(value, RuntimePhase):
        return value.value
    raise ValueError("the supplied value is outside the closed Runtime vocabulary")


PhaseField = Annotated[RuntimePhase, BeforeValidator(_closed_name)]
ReasonField = Annotated[RuntimeReason, BeforeValidator(_closed_name)]
TerminalField = Annotated[RuntimeTerminalStatus, BeforeValidator(_closed_name)]

#: A count is a strict non-negative integer: a bool or a float is never a count.
Count = Annotated[int, StrictInt, Field(ge=0, strict=True)]


def _finite_non_negative(value: object) -> object:
    """Accept only a finite non-negative JSON number: never a bool, a string or a non-finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Pydantic requires ValueError here so the rejection becomes a validation error.
        raise ValueError("must be a JSON number")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as exc:
        # An integer far outside the finite float range is a range error, not a crash.
        raise ValueError("must be a finite JSON number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("must be a finite non-negative JSON number")
    return value


#: A budget snapshot value: a finite non-negative number that is not a bool.
BudgetValue = Annotated[
    float, BeforeValidator(_finite_non_negative), Field(allow_inf_nan=False, ge=0)
]

#: A host limit: a strict positive integer, so a bool and a float are both refused.
PositiveCount = Annotated[int, StrictInt, Field(gt=0, strict=True)]


class _RuntimeOwnedRecord(BaseModel):
    """The shared envelope of every Runtime-owned record.

    A record that already exists is not trusted for its identity. Pydantic's default is to accept an
    instance of the exact model class without revalidating it, which means a value written straight
    into that instance's storage bypasses every schema check. ``revalidate_instances="always"``
    closes that: an existing instance is inspected through the same declared schema as a raw mapping,
    together with the before-validator below, which converts it to its real stored payload. A legal
    record still round-trips; a damaged one is refused.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              revalidate_instances="always")

    _error_code: ClassVar[str] = "INPUT_INVALID"

    @model_validator(mode="before")
    @classmethod
    def _own_the_actual_storage(cls, value: object) -> object:
        """Re-derive an existing instance from its real storage before its schema runs."""
        if isinstance(value, BaseModel):
            return stored_payload(value)
        return value


class RuntimeRecord(_RuntimeOwnedRecord):
    """The shared envelope: a closed shape that requires the explicit schema version."""


class RunRequest(RuntimeRecord):
    """What a host asks one Run to do, and the limits it declares for that Run.

    Every limit belongs to the host: the planner and the domain support objects receive them and
    can never widen them. ``task_criteria`` is the existing Core record, so the acceptance
    conditions keep the already-verified JSON-rule semantics.

    The nested criteria are rebuilt from their detached payload rather than kept as the caller's
    object. Pydantic validation of an already-valid nested instance keeps that instance, so the
    host's own ``conditions`` list and rule payload would stay aliased into this request and a later
    mutation of the host's object would silently change the goal the Run is judged against.
    """

    schema_version: SchemaVersion
    run_id: NonBlankStr
    goal_id: NonBlankStr
    task_criteria: TaskCompletionCriteria
    max_steps: PositiveCount
    max_planning_calls: PositiveCount
    max_retries: Count
    max_replans: Count

    @model_validator(mode="before")
    @classmethod
    def _own_the_nested_criteria(cls, value: object) -> object:
        """Refuse an active-path cycle first, then replace the nested criteria with an owned copy.

        The cycle walk runs before anything else, because every later step reads the record's own
        storage and a self-referential graph would otherwise fail as an interpreter recursion error
        instead of one bounded input error.
        """
        if isinstance(value, BaseModel):
            # An existing instance is not trusted for its own storage: reading it into a fresh
            # mapping is what makes the schema below run again, including over a field that was
            # filled from a default and then overwritten in place.
            value = stored_payload(value)
        require_acyclic(value, ("request",))
        if isinstance(value, dict):
            criteria = value.get("task_criteria")
            if isinstance(criteria, TaskCompletionCriteria):
                detached = dict(value)
                detached["task_criteria"] = isolated_record(
                    TaskCompletionCriteria, record_payload(criteria))
                return detached
        return value


class ContractPreparation(RuntimeRecord):
    """A domain-proposed candidate configuration for one Contract.

    This is a candidate only. It never becomes a committed Contract by itself: the Runtime always
    submits it to the existing Builder, which is the only component that may commit one.

    Every element is declared with its real Core type, so a bare integer in the precondition list,
    a string where a verification policy belongs and an integer where a recovery policy belongs are
    all refused by the declared schema rather than accepted as opaque values. The nested policies are
    re-derived from their own stored shapes, because a declared type alone would let an already-built
    policy instance carry an undeclared stored field past this constructor.
    """

    schema_version: SchemaVersion
    preconditions: Annotated[list[Precondition], Field(strict=True)]
    verification_policy: VerificationPolicy
    recovery_policy: RecoveryPolicy
    timeout_ms: Annotated[float, BeforeValidator(_finite_non_negative), Field(gt=0)]

    @model_validator(mode="before")
    @classmethod
    def _revalidate_nested_declarations(cls, value: object) -> object:
        """Refuse an active-path cycle first, then re-derive every nested declared record.

        The cycle walk runs before any read of the nested storage, so a self-referential binding
        config is one bounded refusal rather than a recursion failure while the record is being
        inspected.
        """
        require_acyclic(value, ("preparation",))
        if not isinstance(value, dict):
            return value
        detached = dict(value)
        for name, declared in (("preconditions", Precondition),
                               ("verification_policy", VerificationPolicy),
                               ("recovery_policy", RecoveryPolicy)):
            member = detached.get(name)
            if isinstance(member, list):
                detached[name] = [
                    isolated_record(declared, record_payload(item))
                    if isinstance(item, BaseModel) else item
                    for item in member
                ]
            elif isinstance(member, BaseModel):
                detached[name] = isolated_record(declared, record_payload(member))
        return detached


class RuntimeBudgetSnapshot(RuntimeRecord):
    """The host's declaration of how much budget remains for this Run.

    This is a host *declaration*, not a measurement: the Runtime reads it at the agreed boundaries
    and refuses an illegal or increasing value, but nothing here proves that wall-clock time was
    actually measured. A production monotonic clock and real await deadlines remain a later task.
    """

    schema_version: SchemaVersion
    remaining_budget_ms: BudgetValue


class RunResult(RuntimeRecord):
    """The safe outcome summary of one Run.

    It carries associations and counters only. A raw observation, an actual-outcome payload, a
    forecast and a free-text exception have no field here, and neither does an implicit success
    flag: success is expressed by ``terminal_status`` plus the referenced verification record.
    """

    schema_version: SchemaVersion
    run_id: NonBlankStr
    goal_id: NonBlankStr
    terminal_status: TerminalField
    reason_code: ReasonField
    steps_started: Count
    planning_calls: Count
    attempts_started: Count
    retries_used: Count
    replans_used: Count
    verification_record_id: NonBlankStr | None = None
    terminal_record_id: NonBlankStr


class RuntimeTransition(RuntimeRecord):
    """One immutable local transition record.

    The record is frozen: a transition that happened is a fact, so rewriting its target phase
    afterwards would let a caller claim a transition that never took place. This is Runtime-local
    diagnostic data. It deliberately does **not** extend the existing trace event schema: the seven
    trace variants and their strict shapes are unchanged, and this record is never written to a
    trace store.
    """

    model_config = FROZEN_RECORD_CONFIG

    schema_version: SchemaVersion
    sequence: Count
    run_id: NonBlankStr
    from_phase: PhaseField
    to_phase: PhaseField
    reason: ReasonField


# --- ingress shape guard -------------------------------------------------------------

#: The declared JSON payload types of the Core records the Runtime carries. A value stored under
#: one of these names is re-checked as finite JSON, so a constructed instance cannot smuggle a
#: tuple, a set, a bytes value, a timestamp or a cycle into a field that will be serialized.
_JSON_ADAPTER = TypeAdapter(Any)


def detached_state(record: BaseModel) -> dict[str, Any]:
    """Return a record's declared raw field values after rejecting an undeclared stored key.

    ``model_dump`` silently drops an attribute pydantic does not know about, which is exactly the
    smuggling route a copied or constructed instance can take. The check therefore starts from the
    instance's own storage and compares every key with the declared field allowlist before any
    normalization, so an injected key is a rejection rather than something quietly discarded.
    """
    declared = type(record).model_fields
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        # A plain ValueError keeps the rejection a bounded record error the caller already
        # translates, instead of a TypeError that would escape as an unexpected facility failure.
        raise ValueError("the record does not expose its stored fields")  # noqa: TRY004
    for key in stored:
        if not (isinstance(key, str) and key in declared):
            raise ValueError("the record carries a stored key that is not a declared field")
    state: dict[str, Any] = {}
    for name in declared:
        if name not in stored:
            raise ValueError("a declared field is missing from the stored record")
        state[name] = stored[name]
    return state


def record_payload(record: object) -> dict[str, Any]:
    """The declared raw stored payload of a record.

    ``__dict__`` is the instance's own storage, so it exposes a key that ``model_dump`` would quietly
    drop: an undeclared stored field is refused here rather than silently discarded. This is the
    shape a callback argument is signed from and the shape a returned record is rebuilt from. It is
    deliberately *not* a JSON check: a legitimate Core record may carry an aware timestamp or a
    genuine frozen container, and those are values the approved contract preserves. Genuine JSON is
    required only where the record really has to be JSON, which is checked separately.
    """
    return detached_state(record)  # type: ignore[arg-type]


def stored_payload(record: BaseModel) -> dict[str, Any]:
    """Every field the record actually holds, including one filled from a default.

    Pydantic stores every declared field in ``__dict__``, so a defaulted field is real storage and an
    invalid value written into it has to be seen. An undeclared stored key is refused rather than
    dropped. This is the payload the unified boundary validates: skipping a defaulted field would let
    a caller hide a damaged value behind "the schema defaulted it", which is precisely the smuggling
    route this function exists to close.

    The one place where the storage is *not* the whole story is re-supplying a value to a Core record
    whose schema cares whether a field was supplied at all. That is handled where the record is
    rebuilt rather than by discarding storage here, so nothing is dropped before it is checked.
    """
    declared = type(record).model_fields
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        raise ValueError("the record does not expose its stored fields")  # noqa: TRY004
    for key in stored:
        if not (isinstance(key, str) and key in declared):
            raise ValueError("the record carries a stored key that is not a declared field")
    return {name: value for name, value in stored.items()}


def plain_containers(value: object, active: set[int] | None = None) -> object:
    """A copy in which every mapping and sequence became a plain mapping or list, cycle-safe.

    A genuine Core ``FrozenList``/``FrozenMapping`` is a legitimate immutable value this contract
    preserves, but it cannot be handed to a recursive rebuild unchanged: a frozen mapping reached
    through a record's field would otherwise pass through as itself and its *own* nested members would
    never be inspected, which is exactly how a nested declared-shape violation hides. Any object with
    ``items()`` is treated as a mapping and any other sequence as a list, so the walk reaches every
    level regardless of which derived container is used. Because the walk is explicit and
    identity-tracked, a container that reaches itself is reported as a cycle instead of recursing,
    and a value genuinely shared by two branches is copied twice rather than refused.

    Two container kinds are deliberately *rebuilt as themselves* rather than flattened, because
    flattening them would be lossy rather than normalising:

    * a mapping whose iteration yields pairs (a frozen mapping) keeps that behaviour, so the members
      it reports are isolated without silently turning it into an ordered plain mapping;
    * a named tuple is a record-like product with a real type, so its members are isolated while its
      type and field names are kept.
    """
    if active is None:
        active = set()
    if isinstance(value, BaseModel):
        marker = id(value)
        if marker in active:
            raise ValueError("the value contains a record that reaches itself")
        active.add(marker)
        try:
            return plain_containers(stored_payload(value), active)
        finally:
            active.discard(marker)
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        return value
    items = getattr(value, "items", None)
    if callable(items):
        marker = id(value)
        if marker in active:
            raise ValueError("the value contains a mapping that reaches itself")
        active.add(marker)
        try:
            # A plain mapping is the normalised form for *every* mapping, including the Core frozen
            # representation: the frozen form is a derived read-only wrapper whose constructor takes
            # a mapping, so rebuilding it here would only re-wrap the same data while making the
            # copy harder to inspect. The declared field validators restore whatever representation
            # the record itself uses.
            return {key: plain_containers(item, active) for key, item in items()}
        finally:
            active.discard(marker)
    if isinstance(value, tuple):
        marker = id(value)
        if marker in active:
            raise ValueError("the value contains a sequence that reaches itself")
        active.add(marker)
        try:
            if hasattr(value, "_fields"):
                # A named tuple is a record-like product: keep its type and field names.
                return type(value)(*(plain_containers(item, active) for item in value))
            # A plain tuple and a derived tuple such as the Core frozen sequence are both replayed
            # as a plain list, because every declared sequence the Runtime carries is declared as a
            # list and a tuple would be refused by that declaration rather than isolated by it.
            return [plain_containers(item, active) for item in value]
        finally:
            active.discard(marker)
    if isinstance(value, list):
        return [plain_containers(item, active) for item in value]
    return value


def deep_detached(value: object) -> object:
    """A deep copy that shares nothing with the original and never builds an invalid record.

    This used to rebuild a nested record with ``model_construct``, which *launders* an undeclared
    stored key: the copy simply does not carry it. It now descends into the record's own storage and
    returns plain containers, so a nested key that is not declared survives the walk and is visible
    to whatever validation runs next. Nothing here creates a record from unchecked fields.
    """
    return plain_containers(value)


def _field_adapter(declared: type, name: str) -> TypeAdapter:
    """A validator for one declared field that keeps the field's own declaration intact.

    ``field.annotation`` alone loses the ``Field`` metadata - ``strict=True``, ``gt=0``, the
    ``BeforeValidator`` hooks - so validating a bare annotation would silently accept exactly the
    values the declaration exists to refuse. The field's rebuilt annotation and its metadata are
    recombined into one ``Annotated`` declaration instead, which is what the record itself declares.

    A field declared as a nested record is handled differently and deliberately. Running a nested
    Core record's *whole* schema here would re-run its cross-field rule with an explicitly supplied
    default that the record never actually carried - the recovery policy in abort mode refuses a
    supplied bounded control while permitting an omitted one - so a legal contract would be refused.
    The nested record is instead validated member by member, which still catches an undeclared nested
    key, a wrong nested schema, a nested constraint violation and a nested type error, and the
    validated members are rebuilt with ``model_construct`` so the omission semantics survive. Field
    metadata is kept and only an ``AfterValidator`` is dropped: those do not constrain a value, they
    reshape it into the Core frozen representation, which has already happened for these values.
    """
    field = declared.model_fields[name]
    annotation = field.rebuild_annotation()
    origin = get_origin(annotation) or annotation
    if isinstance(origin, type) and issubclass(origin, BaseModel):
        return TypeAdapter(Annotated[(BaseModel, _NestedRecordValidator(origin))])
    keep = tuple(item for item in field.metadata
                 if type(item).__name__ != "AfterValidator")
    if keep:
        annotation = Annotated[(annotation, *keep)]
    return TypeAdapter(annotation)


class _NestedRecordValidator:
    """Validate one nested record member by member and rebuild it without its cross-field rule."""

    def __init__(self, declared: type) -> None:
        self._declared = declared

    def __get_pydantic_core_schema__(self, source: type, handler: Any) -> Any:
        from pydantic_core import core_schema

        declared = self._declared

        def validate(value: Any) -> Any:
            if isinstance(value, BaseModel):
                payload = stored_payload(value)
                supplied = getattr(value, "model_fields_set", None)
            elif isinstance(value, dict):
                payload = value
                supplied = None
            else:
                raise TypeError("the nested member is not a declared record")
            checked = validate_declared_fields(declared, payload)
            return declared.model_construct(**_supplied_only(declared, checked, supplied))

        return core_schema.no_info_plain_validator_function(
            validate, json_schema_input_schema=handler.generate_schema(dict))


def _unwrap(annotation: object) -> list[object]:
    """Every type an annotation can denote, with ``Annotated`` wrappers and unions flattened.

    ``Annotated[X, ...]`` carries its real type as the first argument and a union carries its members,
    so a check that only looks at the outermost object would miss the declared type of an optional
    JSON payload and of every optional record. Both are unwrapped here.
    """
    found: list[object] = []
    pending = [annotation]
    seen = 0
    while pending and seen < 64:
        seen += 1
        item = pending.pop()
        origin = get_origin(item)
        if origin is Annotated or str(origin) == "typing.Annotated":
            pending.extend(get_args(item))
            continue
        if origin is Union or str(origin).endswith("Union"):
            pending.extend(get_args(item))
            continue
        found.append(origin or item)
    return found


def _declares_json(field: Any) -> bool:
    """Whether a declared field is a free JSON payload, which must really be JSON.

    A field whose declaration is an opaque JSON object has no schema of its own to validate against,
    so the only meaningful check is that the stored value really is finite, cycle-free JSON - a
    tuple, a set, a bytes value or a cycle satisfies no JSON declaration at all. The declaration is
    recognised by the real declared type rather than by name, so this applies to every such field of
    every Core record without changing any Core model.
    """
    for candidate in (field.annotation, field.rebuild_annotation()):
        for item in _unwrap(candidate):
            if getattr(item, "__name__", "") in ("JsonObject", "JsonValue", "JsonArray"):
                return True
    return False


def _genuine_frozen(value: object) -> bool:
    """Whether a value is the genuine Core immutable container, by real type identity.

    The comparison is against the actual class object, never against a type's module, qualified name
    or name text. Those attributes are writable, so an unrelated class can repeat them and pass a
    name-based test while being a completely different type. Only the real Core container is the
    declared immutable sequence this contract preserves.
    """
    return type(value) in (FrozenList, FrozenMapping)


def _declares_sequence(field: Any) -> bool:
    """Whether a declared field is a sequence, which must not arrive as an ordinary tuple.

    A JSON array is a real ``list``; the only tuple a declared sequence accepts is the genuine Core
    immutable sequence, which carries its own type. An unrelated tuple is a plain tuple with no such
    identity, so it is refused *before* any normalisation could turn it into an acceptable list.
    """
    for candidate in (field.annotation, field.rebuild_annotation()):
        for item in _unwrap(candidate):
            if item in (list, tuple) or getattr(item, "__name__", "") == "FrozenList":
                return True
    return False


def validate_declared_fields(declared: type, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate every stored member of a payload against its own complete field declaration.

    Each member is checked on its own so a derived Core record's cross-field validator cannot be
    tripped by a re-supplied default. Every declared scalar, sequence member, nested record, strict
    type, constraint and hook is exercised, and an undeclared stored key is refused before any of
    that, so nothing is dropped before it is seen.

    Two checks deliberately run on the *raw* stored value rather than on a normalised copy, because
    normalising first would launder exactly what the declaration refuses:

    * a member declared as JSON must really be finite, cycle-free JSON - an ordinary tuple, a set, a
      bytes value or a cycle satisfies no JSON declaration;
    * a member declared as a sequence must not be an ordinary tuple. The ordinary tuple is refused
      before conversion, while the genuine Core immutable sequence keeps its own identity and passes.
    """
    checked: dict[str, Any] = {}
    for name, member in payload.items():
        field = declared.model_fields.get(name)
        if field is None:
            raise ValueError("the record carries a stored key that is not a declared field")
        if _declares_json(field):
            require_json_shape(member, (name,))
        elif _declares_sequence(field) and isinstance(member, tuple) \
                and not _genuine_frozen(member):
            # Every tuple that is not the genuine Core immutable sequence is refused here, before
            # any conversion could turn it into an acceptable list. The test is on the real type, not
            # on a name: a class can repeat another type's display names without becoming that type,
            # and only the actual Core container is the declared immutable representation.
            raise ValueError(f"the stored field {name!r} is an ordinary tuple, not a declared list")
        # A sequence of declared records is walked into: a member that is itself a damaged record
        # would otherwise pass through, because pydantic keeps an existing instance of the right class
        # rather than revalidating it. Its own storage is checked here so the damage is refused.
        nested = _nested_record_type(field)
        if nested is not None and isinstance(member, (list, tuple)):
            for item in member:
                if isinstance(item, BaseModel):
                    validate_declared_fields(nested, stored_payload(item))
        try:
            checked[name] = _field_adapter(declared, name).validate_python(
                plain_containers(member))
        except (ValidationError, ValueError, TypeError) as exc:
            raise ValueError(f"the stored field {name!r} does not match its declaration") from exc
    return checked


def _nested_record_type(field: Any) -> type | None:
    """The declared record type of a sequence field, when its members are declared records."""
    for candidate in (field.annotation, field.rebuild_annotation()):
        for item in (candidate, *get_args(candidate)):
            origin = get_origin(item)
            if origin in (list, tuple):
                for inner in get_args(item):
                    base = get_origin(inner) or inner
                    if isinstance(base, type) and issubclass(base, BaseModel):
                        return base
    return None


def isolated_record(declared: type, payload: dict[str, Any],
                    *, validate_fully: bool = False,
                    supplied: object = None) -> Any:
    """Validate a record through its declared schema and return a detached instance.

    Two validation depths are available and both are real schema validation:

    * the default checks every stored member against its own complete field declaration, which
      re-runs the strict types, the numeric constraints, the ``BeforeValidator`` hooks, the nested
      declarations and the ``extra`` allowances while leaving a derived record's cross-field rule to
      the record that owns it;
    * ``validate_fully=True`` additionally re-validates the whole record at once, which is what a
      return whose *cross-field* consistency is part of the contract needs - a contradictory recovery
      verdict has to be refused at ingress, and only the whole-record schema knows it is
      contradictory.

    ``supplied`` is the set of fields the original record recorded as set. Validation always covers
    every stored field; the set only limits which values are replayed when the copy is rebuilt, so a
    record whose schema distinguishes an omitted field from an explicitly supplied one keeps that
    distinction. Both paths start from the record's own storage, so an undeclared stored key is a
    refusal rather than something a copy quietly drops, and both return a freshly built instance that
    shares no container with the object it came from. A cycle is a bounded rejection rather than a
    ``RecursionError``.
    """
    require_acyclic(payload)
    checked = validate_declared_fields(declared, payload)
    if validate_fully:
        # The whole record is re-validated from the values that were just checked, so the record's
        # own field validators and its cross-field invariants run as well: an outcome that says it
        # succeeded while carrying an error, or a recovery verdict whose kind and reason contradict
        # each other, is refused here rather than accepted and acted on. The whole validated payload
        # is used, including a defaulted field, because a damaged *default* is exactly what a
        # supplied-only replay would hide; ``model_construct`` cannot be used for this tier at all,
        # since it would skip those validators.
        return declared.model_validate(checked)
    # Every stored field has already been validated - including one filled from a default - so a
    # damaged defaulted value cannot hide behind "the schema defaulted it". The rebuild then supplies
    # only the fields the record really recorded as set: a Core record may treat "the caller supplied
    # this" as meaningful (the recovery policy refuses an explicitly supplied control in abort mode
    # while allowing an omitted one), so replaying a defaulted field would change the record's
    # meaning rather than reproduce it.
    return declared.model_construct(**_supplied_only(declared, checked, supplied))


#: The declared records whose schema really distinguishes an *omitted* field from an explicitly
#: supplied one. Only these need the supplied-field restriction when a copy is rebuilt: for every
#: other record the current storage is the truth, and dropping a defaulted value would lose data the
#: record legitimately holds. The Core recovery policy is the one such declaration in this round: in
#: abort mode it refuses an explicitly supplied bounded control while accepting an omitted one.
_SUPPLIED_SENSITIVE: frozenset[type] = frozenset({RecoveryPolicy})


def _supplied_only(declared: type, checked: dict[str, Any], supplied: object) -> dict[str, Any]:
    """The validated values a rebuilt copy replays.

    Validation always runs over every stored field, including a defaulted one, so a damaged defaulted
    value is still refused. Here the question is only what the *copy* carries. For a record whose
    schema distinguishes an omitted field from an explicitly supplied one - the recovery policy - the
    copy replays the fields the original recorded as set, so that distinction survives. For every
    other record the complete validated storage is replayed: a default the record legitimately
    populated and then modified in place is real data, and filtering it by the field-set marker would
    silently drop it from the copy the dependency receives.
    """
    if declared not in _SUPPLIED_SENSITIVE:
        return dict(checked)
    if not isinstance(supplied, (set, frozenset)):
        return dict(checked)
    keep = {name for name in supplied if name in declared.model_fields and name in checked}
    return {name: checked[name] for name in keep} if keep else dict(checked)


def require_acyclic(value: object, location: tuple[str | int, ...] = ()) -> None:
    """Refuse a record or container that reaches itself, using an explicit stack.

    The walk is iterative on purpose: a guard that recursed would itself raise ``RecursionError`` on
    a deep or self-referential graph, which is exactly the failure it exists to prevent. Cycle
    detection therefore has to be a *precondition* of any recursive validation. A record is followed
    through its own storage *by identity* rather than being replaced by a fresh mapping, so a record
    that genuinely contains itself is detected; a value legitimately shared by two branches is not a
    cycle, because the active-path set is unwound as the walk leaves each node.
    """
    stack: list[tuple[object, bool]] = [(value, False)]
    active: set[int] = set()
    while stack:
        node, leaving = stack.pop()
        if isinstance(node, BaseModel):
            marker = id(node)
            if leaving:
                active.discard(marker)
                continue
            if marker in active:
                raise ValueError("the value contains a record that reaches itself")
            active.add(marker)
            stack.append((node, True))
            stack.extend((getattr(node, name, None), False)
                         for name in type(node).model_fields)
            continue
        if not isinstance(node, (dict, list, tuple)):
            continue
        marker = id(node)
        if leaving:
            active.discard(marker)
            continue
        if marker in active:
            raise ValueError("the value contains a container that reaches itself")
        active.add(marker)
        stack.append((node, True))
        if isinstance(node, dict):
            stack.extend((item, False) for item in node.values())
        else:
            stack.extend((item, False) for item in node)
    _ = location


def _json_shape(value: object, active: set[int], done: set[int]) -> bool:
    """Whether a stored value is finite JSON, cycle-free and shareable.

    ``active`` holds the identities of the current descent, so a value that reaches itself is a
    cycle; ``done`` holds the identities already fully checked, so a value legitimately referenced
    twice is accepted instead of being mistaken for a cycle.
    """
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (bytes, bytearray, datetime_module.datetime)):
        return False
    if isinstance(value, (list, tuple)):
        if isinstance(value, tuple) and not isinstance(value, FrozenList):
            # A JSON array is a real list; the only tuple accepted is the genuine Core immutable
            # sequence a committed contract holds. An unrelated tuple subclass that merely repeats
            # that class name is an arbitrary tuple and is refused on real type identity.
            return False
        marker = id(value)
        if marker in done:
            return True
        if marker in active:
            return False
        active.add(marker)
        try:
            return all(_json_shape(item, active, done) for item in value)
        finally:
            active.discard(marker)
            done.add(marker)
    if isinstance(value, BaseModel):
        marker = id(value)
        if marker in done:
            return True
        if marker in active:
            return False
        active.add(marker)
        try:
            for name in type(value).model_fields:
                if not _json_shape(getattr(value, name, None), active, done):
                    return False
            return True
        finally:
            active.discard(marker)
            done.add(marker)
    if callable(getattr(value, "items", None)):
        # A genuine Core immutable mapping is the committed production representation of a JSON
        # object; any other mapping is checked with the same JSON rules.
        marker = id(value)
        if marker in done:
            return True
        if marker in active:
            return False
        active.add(marker)
        try:
            for key in value:
                if not isinstance(key, str):
                    return False
                if not _json_shape(value[key], active, done):
                    return False
            return True
        finally:
            active.discard(marker)
            done.add(marker)
    return False


def _model_is_sound(value: BaseModel, active: set[int]) -> bool:
    """Whether a Core record still validates from its own stored fields.

    The record is rebuilt through its own schema rather than inspected field by field, so the
    already-verified validation of that module decides, and a copied or constructed instance is
    re-derived instead of trusted on class identity. A defensive recursion guard turns a
    pathological structure into a bounded rejection instead of an interpreter failure.
    """
    try:
        state = detached_state(value)
    except ValueError:
        return False
    marker = id(value)
    if marker in active:
        return False
    active.add(marker)
    try:
        type(value).model_validate(state)
    except (ValidationError, ValueError, TypeError, RecursionError):
        return False
    finally:
        active.discard(marker)
    return True


def require_declared_record(value: object, expected: type[BaseModel],
                            location: tuple[str | int, ...]) -> BaseModel:
    """Return ``value`` when it is a sound instance of ``expected``, else raise ``ValueError``.

    The class is compared by real type identity, never by name: a look-alike type with the same
    name is not the declared record and is refused. The instance is then re-derived through its own
    schema, so a validation-bypass construction is caught before the Runtime consumes it.
    """
    if not isinstance(value, expected):
        # The caller translates one bounded rejection kind, so a ValueError keeps a wrong type and
        # a damaged instance on the same reporting path.
        raise ValueError("the value is not the declared record type")  # noqa: TRY004
    if not _model_is_sound(value, set()):
        raise ValueError("the record does not match its declared schema")
    _ = location
    return value


def validated_budget_value(snapshot: object,
                           location: tuple[str | int, ...]) -> float:
    """Return the finite non-negative budget of a revalidated snapshot, else raise ``ValueError``.

    The snapshot is first re-derived from its own stored fields through its declared schema, so a
    bypass construction is refused instead of being consumed. A boolean, a numeric string, an
    infinity, a NaN, a negative number, an unknown field or a cycle is therefore rejected here
    rather than laundered through a float coercion downstream.
    """
    if not isinstance(snapshot, RuntimeBudgetSnapshot):
        raise ValueError("the value is not a budget snapshot record")  # noqa: TRY004
    if not _model_is_sound(snapshot, set()):
        raise ValueError("the budget snapshot does not match its declared schema")
    try:
        state = detached_state(snapshot)
    except ValueError as exc:
        raise ValueError("the budget snapshot does not expose its declared fields") from exc
    try:
        sound = RuntimeBudgetSnapshot.model_validate(state)
    except (ValidationError, ValueError, TypeError):
        raise ValueError("the budget snapshot is not a finite non-negative number") from None
    _ = location
    return float(sound.remaining_budget_ms)


def require_json_shape(value: object, location: tuple[str | int, ...]) -> None:
    """Refuse a stored value that is not finite, cycle-free, shareable JSON."""
    _ = location
    if not _json_shape(value, set(), set()):
        raise ValueError("the stored value is not finite JSON")
