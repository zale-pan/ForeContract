"""Versioned structured-tracing records.

One physical JSONL line is one :class:`Event`, and ``record_id`` *is* the event identity: there
is no second identity to keep in step. The public envelope of every event is

    schema_version, record_id, run_id, sequence, occurred_at, kind

plus the fields the closed variant adds. ``sequence`` is a strict non-negative integer that
starts at ``0`` and increases by one per Run; ``occurred_at`` is a timezone-aware timestamp;
every identifier is a non-blank opaque string supplied by the caller.

The variant set is closed. Unknown ``kind`` values, extra fields, non-string identifiers,
booleans in integer positions, tuples where JSON arrays are required and non-finite numbers
are all rejected, so a normalisation can never hide an invalid key, container or scalar type.
``run_id``, ``step_id``, ``contract_id``, ``attempt_id`` and ``verification_record_id`` are
association identifiers only: no raw execution payload, observation, exception message,
metadata or URI has a field to live in.

The two source levels a consumer may select are described by :data:`SOURCE_LEVELS`: an
*attempt* is located by ``(run_id, step_id, contract_id, attempt_id)`` and a *run* by
``run_id`` alone. They are never mixed. A start record whose matching terminal fact is absent
stays *unknown*; it is never promoted to success, and a verification record can never become
a success source. The explicit label mappings live in :data:`ATTEMPT_RESULT_LABELS`,
:data:`TERMINAL_RESULT_LABELS` and the two ``*_label`` helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal, NamedTuple, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from forecontract.models.base import (
    NonBlankStr,
    NonNegativeInt,
    SchemaVersion,
    Timestamp,
)
from forecontract.models.outcomes import (
    ExecutionStatus,
    ExecutionStatusField,
    VerificationStatusField,
)
from forecontract.tracing.errors import TraceValidationError

__all__ = [
    "ATTEMPT_RESULT_LABELS",
    "ATTEMPT_RESULT_TO_LABEL",
    "ATTEMPT_SOURCE_LEVEL",
    "CLOSED_CODES",
    "EVENT_KINDS",
    "EVENT_MODELS",
    "RUN_SOURCE_LEVEL",
    "SOURCE_LEVELS",
    "TERMINAL_RESULT_LABELS",
    "TERMINAL_RESULT_TO_LABEL",
    "TERMINAL_STATUSES",
    "AttemptFinished",
    "AttemptRef",
    "AttemptStarted",
    "ClosedCode",
    "ContractCommitted",
    "Event",
    "RunFinished",
    "RunStarted",
    "StepStarted",
    "TraceRecord",
    "VerificationRecorded",
    "attempt_label",
    "attempt_ref",
    "detached_state",
    "parse_record",
    "source_location",
    "terminal_label",
]

#: The closed set of event kinds. A record's ``kind`` selects exactly one variant.
EVENT_KINDS: Final[tuple[str, ...]] = (
    "run_started",
    "step_started",
    "contract_committed",
    "attempt_started",
    "attempt_finished",
    "verification_recorded",
    "run_finished",
)

#: The closed set of reason/error codes. No free-text description is accepted.
CLOSED_CODES: Final[tuple[str, ...]] = (
    "execution_failed",
    "execution_unknown",
    "verification_failed",
    "verification_inconclusive",
    "user_cancelled",
    "deadline_exceeded",
    "policy_abort",
    "internal_error",
)

#: The closed set of terminal statuses.
TERMINAL_STATUSES: Final[tuple[str, ...]] = (
    "success",
    "failure",
    "cancelled",
    "timeout",
    "aborted",
)

#: ``attempt_finished.execution_status`` -> result label. The label describes the action
#: result only; it never claims that a Run succeeded.
ATTEMPT_RESULT_LABELS: Final[dict[str, str]] = {
    "succeeded": "success",
    "failed": "failure",
    "unknown": "unknown",
}
ATTEMPT_RESULT_TO_LABEL: Final[dict[str, str]] = dict(ATTEMPT_RESULT_LABELS)

#: ``run_finished.terminal_status`` -> result label, one to one.
TERMINAL_RESULT_LABELS: Final[dict[str, str]] = {
    "success": "success",
    "failure": "failure",
    "cancelled": "cancelled",
    "timeout": "timeout",
    "aborted": "aborted",
}
TERMINAL_RESULT_TO_LABEL: Final[dict[str, str]] = dict(TERMINAL_RESULT_LABELS)

#: The two source levels a consumer may select. They are never mixed.
ATTEMPT_SOURCE_LEVEL: Final[str] = "attempt"
RUN_SOURCE_LEVEL: Final[str] = "run"
SOURCE_LEVELS: Final[tuple[str, ...]] = (ATTEMPT_SOURCE_LEVEL, RUN_SOURCE_LEVEL)

#: The record and caller-supplied identity types are non-blank opaque strings.
Identifier = NonBlankStr
Sequence = NonNegativeInt
ClosedCode = Literal[
    "execution_failed",
    "execution_unknown",
    "verification_failed",
    "verification_inconclusive",
    "user_cancelled",
    "deadline_exceeded",
    "policy_abort",
    "internal_error",
]
VerificationScope = Literal["attempt", "run"]
TerminalStatus = Literal["success", "failure", "cancelled", "timeout", "aborted"]

#: Events are immutable: a committed record is a fact, not a mutable working object. The
#: container is not a deep-immutability claim in the Contract sense - every field is a scalar
#: or a string, so a frozen shell is already a complete guarantee for a record.
#:
#: ``hide_input_in_errors`` keeps a rejected value out of a rendered message, and the model
#: wrapper below reduces every schema rejection to a bounded category plus a safe location, so
#: an attacker-controlled field name or value is never echoed back (F04).
RECORD_CONFIG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

#: The declared field names of the record contract. A validation error location may mention
#: only these names, a list index or a closed kind value; anything else is replaced by a
#: placeholder. The set is built from the models after they are defined.
_SAFE_LOCATION_NAMES: set[str] = set()
_SAFE_LOCATION_VALUES: set[str] = set(EVENT_KINDS) | {"attempt", "run"}

#: The bounded category text for each schema error type the tracing models can raise.
_ERROR_CATEGORIES: Final[dict[str, str]] = {
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
    "value_error": "a cross-field rule was violated",
    "union_tag_invalid": "the record kind is not a known event variant",
    "union_tag_not_found": "the record kind is not a known event variant",
    "strict": "a value has the wrong JSON type",
}
_UNCATEGORISED = "the value does not match the declared schema"


def _bounded_category(error_type: str) -> str:
    """Map a schema error type onto a bounded sentence, never the raw message."""
    return _ERROR_CATEGORIES.get(error_type, _UNCATEGORISED)


def _safe_location(location: tuple[object, ...]) -> str:
    """Render an error location using only declared names, indices and closed kinds."""
    parts: list[str] = []
    for part in location:
        if isinstance(part, bool):
            parts.append("<value>")
        elif isinstance(part, int):
            parts.append(f"[{part}]")
        elif isinstance(part, str) and (part in _SAFE_LOCATION_NAMES or part in _SAFE_LOCATION_VALUES):
            parts.append(("." if parts else "") + part)
        else:
            parts.append(".<field>" if parts else "<record>")
    return "".join(parts) or "<record>"


def _safe_location_parts(location: tuple[object, ...]) -> tuple[object, ...]:
    """Replace every location segment that is not a declared name, index or closed kind.

    The public error surface rebuilds its own location from the raw input, so sanitising the
    location text is not enough: the structured ``loc`` and the JSON rendering have to lose an
    attacker-controlled field name too, while a declared field location is kept as it is (F04).
    """
    safe: list[object] = []
    for part in location:
        if isinstance(part, bool):
            safe.append("<value>")
        elif isinstance(part, int) or (
            isinstance(part, str) and (part in _SAFE_LOCATION_NAMES or part in _SAFE_LOCATION_VALUES)
        ):
            safe.append(part)
        else:
            safe.append("<field>")
    return tuple(safe)


def _safe_error(error: dict, source_type: str) -> dict:
    """Reduce one schema error to a bounded category and a safe location.

    Only ``value_error`` is used for the rebuilt error, because the richer pydantic types
    require their own ``ctx`` payload; nothing from the rejected value is carried over.
    """
    location = _safe_location_parts(tuple(error.get("loc") or ()))
    return {
        "type": "value_error",
        "loc": location,
        "msg": _bounded_category(source_type),
        "input": None,
        "ctx": {"error": ValueError(_bounded_category(source_type))},
    }


def _sanitised_validation_error(exc: ValidationError) -> ValidationError:
    """Build the bounded replacement for one schema rejection.

    Only the error category and the safe location survive; the rejected value and any
    attacker-controlled field name are dropped instead of being transcribed (F04).
    """
    errors = exc.errors()
    if not errors:
        return ValidationError.from_exception_data(
            "TraceRecord",
            [{"type": "value_error", "loc": (), "msg": _UNCATEGORISED, "input": None,
              "ctx": {"error": ValueError(_UNCATEGORISED)}}],
        )
    return ValidationError.from_exception_data(
        exc.title, [_safe_error(error, str(error.get("type") or "invalid")) for error in errors]
    )


def bounded_schema_message(exc: ValidationError) -> str:
    """Describe a schema rejection with a bounded category and a safe field location.

    The category is taken from the bounded text carried by the error itself, so a rebuilt
    rejection still names the real category instead of the one used to rebuild it.
    """
    errors = exc.errors()
    if not errors:
        return "the record does not match the declared schema"
    first = errors[0]
    location = tuple(first.get("loc") or ())
    category = str(first.get("msg") or "").strip()
    if category not in set(_ERROR_CATEGORIES.values()) and category != _UNCATEGORISED:
        category = _bounded_category(str(first.get("type") or "invalid"))
    return f"invalid record at {_safe_location(location)}: {category}"


class AttemptRef(NamedTuple):
    """The attempt experience key ``(run_id, step_id, contract_id, attempt_id)``."""

    run_id: str
    step_id: str
    contract_id: str
    attempt_id: str


class _BaseEvent(BaseModel):
    """The shared record envelope. Subclasses add their closed variant fields."""

    model_config = RECORD_CONFIG

    schema_version: SchemaVersion
    record_id: Identifier
    run_id: Identifier
    sequence: Sequence
    occurred_at: Timestamp
    kind: str

    @model_validator(mode="wrap")
    @classmethod
    def _bounded_validation(cls, data: object, handler) -> object:
        """Reduce every schema rejection to a bounded, value-free error before it escapes.

        The wrapper sits on the shared envelope, so construction, mapping loading, JSON
        loading, the discriminated union and a copied or constructed instance all report the
        same bounded category and safe location instead of echoing a rejected key or value.
        """
        try:
            return handler(data)
        except ValidationError as exc:
            raise _sanitised_validation_error(exc) from None

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in EVENT_KINDS:
            raise ValueError("the record kind is not a known event variant")
        return value


class RunStarted(_BaseEvent):
    """A Run is created. It carries no step, contract or attempt reference."""

    kind: Literal["run_started"] = "run_started"


class StepStarted(_BaseEvent):
    """A new step starts inside an existing Run.

    ``step_id`` must not have been created in this Run before; the store checks that against
    the recorded history.
    """

    kind: Literal["step_started"] = "step_started"
    step_id: Identifier


class ContractCommitted(_BaseEvent):
    """A new Contract version becomes the current one for an existing step.

    ``replaces_contract_id`` is an optional, same-Run, same-Step reference to the Contract
    this commit replaces. Self-reference and a replacement that is not the current version of
    that step are rejected by the store.
    """

    kind: Literal["contract_committed"] = "contract_committed"
    step_id: Identifier
    contract_id: Identifier
    replaces_contract_id: Identifier | None = None


class AttemptStarted(_BaseEvent):
    """A new Attempt starts against the current Contract of an existing step."""

    kind: Literal["attempt_started"] = "attempt_started"
    step_id: Identifier
    contract_id: Identifier
    attempt_id: Identifier


class AttemptFinished(_BaseEvent):
    """An Attempt ends. It carries the reusable execution status and an optional safe code.

    The record carries no outcome payload: raw actual outcome content has no field here.
    """

    kind: Literal["attempt_finished"] = "attempt_finished"
    step_id: Identifier
    contract_id: Identifier
    attempt_id: Identifier
    execution_status: ExecutionStatusField
    error_code: ClosedCode | None = None


class VerificationRecorded(_BaseEvent):
    """A verification fact for one finished Attempt, or for the Run as a whole.

    ``scope="attempt"`` requires the step, contract and attempt references and they must
    identify an Attempt that already finished. ``scope="run"`` must not carry any of them.
    ``verification_status`` reuses the existing Core ``VerificationStatus`` member unchanged.
    """

    kind: Literal["verification_recorded"] = "verification_recorded"
    scope: VerificationScope
    verification_status: VerificationStatusField
    step_id: Identifier | None = None
    contract_id: Identifier | None = None
    attempt_id: Identifier | None = None

    @model_validator(mode="after")
    def _scope_fields(self) -> VerificationRecorded:
        """Bind the scope to its reference fields after every field, default or not, exists.

        A field validator on ``attempt_id`` would be skipped whenever the caller omits it, so
        the rule is enforced on the finished model instead.
        """
        present = [self.step_id, self.contract_id, self.attempt_id]
        if self.scope == "attempt" and any(item is None for item in present):
            raise ValueError("an attempt-scoped verification needs its step, contract and attempt")
        if self.scope == "run" and any(item is not None for item in present):
            raise ValueError("a run-scoped verification must not carry an attempt reference")
        return self


class RunFinished(_BaseEvent):
    """The unique terminal fact of a Run.

    ``success`` must reference a previously recorded run-scoped ``pass`` verification. That is
    a reference-consistency check only: no verification algorithm runs here and nothing about
    the referenced record is re-evaluated.
    """

    kind: Literal["run_finished"] = "run_finished"
    terminal_status: TerminalStatus
    reason_code: ClosedCode | None = None
    verification_record_id: Identifier | None = None

    @model_validator(mode="after")
    def _success_needs_reference(self) -> RunFinished:
        """A successful terminal fact must name the run-scoped PASS it relies on.

        The reference is an optional field, so the rule is enforced on the finished model: a
        value validator would simply never run when the caller omits the reference.
        """
        if self.terminal_status == "success" and self.verification_record_id is None:
            raise ValueError("a successful terminal fact must reference a run-scoped pass record")
        return self


#: The closed record union. ``kind`` is the discriminant, so an unknown variant and an extra
#: field both fail instead of being absorbed by a permissive union branch.
TraceRecord: TypeAlias = Annotated[
    RunStarted
    | StepStarted
    | ContractCommitted
    | AttemptStarted
    | AttemptFinished
    | VerificationRecorded
    | RunFinished,
    Field(discriminator="kind"),
]

#: ``kind`` -> concrete model, used by callers that need the class without a match statement.
EVENT_MODELS: Final[dict[str, type[_BaseEvent]]] = {
    "run_started": RunStarted,
    "step_started": StepStarted,
    "contract_committed": ContractCommitted,
    "attempt_started": AttemptStarted,
    "attempt_finished": AttemptFinished,
    "verification_recorded": VerificationRecorded,
    "run_finished": RunFinished,
}

# The safe-location allowlist is the declared field names of the closed record contract. It is
# derived from the models themselves so it can never drift from the schema (F04).
_SAFE_LOCATION_NAMES.update(
    name for model in EVENT_MODELS.values() for name in model.model_fields
)

#: The seven concrete record classes of the closed union. Ingress accepts an instance only
#: when it is one of these, so a foreign model is never returned as a record (F03).
_RECORD_CLASSES: Final[frozenset[type]] = frozenset(EVENT_MODELS.values())

#: The single public name for one record: the union is the model surface.
Event: TypeAlias = TraceRecord


def attempt_ref(record: TraceRecord) -> AttemptRef:
    """Return the attempt experience key of a record that carries one.

    Raises :class:`ValueError` when the record is not an attempt-scoped fact, because asking
    for an attempt key of a run-level record would silently mix the two source levels.
    """
    if isinstance(record, (AttemptStarted, AttemptFinished)):
        return AttemptRef(record.run_id, record.step_id, record.contract_id, record.attempt_id)
    if isinstance(record, VerificationRecorded) and record.scope == "attempt":
        return AttemptRef(record.run_id, record.step_id, record.contract_id, record.attempt_id)
    raise ValueError("the record does not carry an attempt reference")


def source_location(record: TraceRecord) -> tuple[str, ...]:
    """Return the source location key of a record.

    Attempt-scoped facts are located by ``(run_id, step_id, contract_id, attempt_id)`` and
    every other fact by ``(run_id, record_id)``. The two shapes have different lengths, so a
    consumer cannot confuse one level with the other.
    """
    if isinstance(record, (AttemptStarted, AttemptFinished)):
        return (record.run_id, record.step_id, record.contract_id, record.attempt_id)
    if isinstance(record, VerificationRecorded) and record.scope == "attempt":
        return (record.run_id, record.step_id, record.contract_id, record.attempt_id)
    return (record.run_id, record.record_id)


def attempt_label(status: ExecutionStatus) -> str:
    """Map an attempt execution status to its explicit result label."""
    return ATTEMPT_RESULT_LABELS[status.value]


def terminal_label(status: str) -> str:
    """Map a terminal status to its explicit result label."""
    return TERMINAL_RESULT_LABELS[status]


def parse_record(payload: object) -> TraceRecord:
    """Validate one decoded JSON value as a record.

    An already-built record is re-validated from its detached fields, so an instance that was
    assembled by ``model_copy`` or ``model_construct`` - including one carrying an unknown
    attribute - cannot pass ingress on class identity (F03). The rejection never transcribes
    the rejected value or an attacker-controlled field name: only a declared field location
    and the schema error category are reported (F04).
    """
    if isinstance(payload, BaseModel):
        if type(payload) not in _RECORD_CLASSES:
            # An arbitrary model is not a record of the closed union, so it must not be
            # returned as one just because it is a BaseModel (F03).
            raise TraceValidationError("the supplied object is not a record of the closed union")
        state = detached_state(payload)
        try:
            return type(payload).model_validate(state)
        except ValidationError as exc:
            raise TraceValidationError(bounded_schema_message(exc), cause=exc) from None
    try:
        return _RECORD_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise TraceValidationError(bounded_schema_message(exc), cause=exc) from None


def _field_names(record: BaseModel) -> frozenset[str]:
    """The declared field names of a record, as a set the raw keys can be compared against."""
    return frozenset(type(record).model_fields)


def _unknown_raw_keys(record: BaseModel) -> list[object]:
    """Return the ``__dict__`` keys that are not a declared field of this record class.

    Nothing else belongs in a record's ``__dict__``. A real pydantic instance keeps its
    bookkeeping in the class's slots (``__pydantic_fields_set__``, ``__pydantic_extra__`` and
    ``__pydantic_private__``), not as dictionary entries, so a ``__dict__`` entry with one of
    those names can only have been injected by a caller. A non-string key is always unknown,
    because a record field name is a string. The check looks at the instance's real storage
    only; it never infers internal status from a name pattern and never discards a key
    silently.
    """
    names = _field_names(record)
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        return []
    return [key for key in stored if not (isinstance(key, str) and key in names)]


def detached_state(record: object) -> dict:
    """Return the declared raw fields of a record after checking its dictionary keys.

    ``model_dump`` drops an attribute pydantic does not know about, which is exactly the
    smuggling route a copied or constructed instance can take. The check therefore starts from
    the instance's own storage: every ``__dict__`` key is compared with the exact declared
    field allowlist before any normalization, and a key that is unknown, non-string, an
    invented dunder name or one of the real bookkeeping names used as a dictionary entry is a
    rejection rather than something to drop. Genuine bookkeeping keeps working because it lives
    in the class slots, which this check never touches, so an ordinary instance is unaffected
    while a user-injected entry is never accepted (F03).
    """
    if isinstance(record, BaseModel):
        unknown = _unknown_raw_keys(record)
        if unknown:
            raise TraceValidationError("the record carries a stored key that is not a declared field")
        state: dict = {name: getattr(record, name, None) for name in type(record).model_fields}
        extra = getattr(record, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            state.update(extra)
        return state
    if isinstance(record, Mapping):
        return dict(record)
    raise TraceValidationError("the supplied object is not a record of the closed union")


_RECORD_ADAPTER: Final = TypeAdapter(TraceRecord)
