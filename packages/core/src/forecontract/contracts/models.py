"""Public contract models for the deterministic Contract Builder.

``VerificationBinding``, ``VerificationPolicy``, ``RecoveryPolicy`` and ``ActionContract``
are the versioned shapes a committed contract is made of. Every one of them requires an
explicit ``schema_version="0.1"``, rejects unknown versions and extra fields, uses strict
field types and reuses the existing Core model, JSON and timestamp validation without
editing those modules.

Committed content is deeply immutable. The models are frozen, the JSON payloads are held in
immutable containers and the existing Core candidate models are re-validated into frozen
snapshots (see ``forecontract.contracts.snapshot``). Mutating a caller-owned object after
the build, or mutating anything obtained from the contract, its dumps or another
deserialized instance, therefore cannot change the committed contract. That is an ordinary
API guarantee only: reflection, private-attribute surgery and deliberate validation-bypass
construction are outside it, and nothing here claims OS-level isolation.

``ActionContract`` also enforces its own intrinsic structure - at least one required
expected outcome and exactly one verification binding for every outcome. It cannot prove
live domain validity, that preconditions currently hold, that a runtime budget is still
available, or that an authentic Builder produced a loaded record. A loaded record is data,
never permission to execute it.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_serializer,
    model_serializer,
    model_validator,
)

from forecontract.contracts.snapshot import (
    FrozenJsonObject,
    FrozenList,
    freeze_action,
    freeze_expected_outcomes,
    freeze_preconditions,
)
from forecontract.models.actions import ActionSpec, ExpectedOutcome, Precondition
from forecontract.models.base import (
    NonBlankStr,
    NonNegativeInt,
    SchemaVersion,
    StrictBool,
    Timestamp,
)

__all__ = [
    "ActionContract",
    "PositiveFiniteNumber",
    "RecoveryPolicy",
    "VerificationBinding",
    "VerificationPolicy",
]

# Frozen models validate their defaults too (F01). Pydantic does not validate a default
# unless it is told to, so an omitted ``config`` or ``metadata`` would otherwise be stored
# as the plain ``dict`` the factory returned and stay mutable after the commit. Validating
# defaults routes them through the same strict JSON snapshot as an explicit value, so an
# omitted payload is frozen exactly like a supplied one.
_FROZEN_CONFIG = ConfigDict(extra="forbid", frozen=True, validate_default=True)


#: The bounded controls the caller supplied on the record currently being validated. An explicit
#: null is a supplied control; an omitted field is not, so the two can never be confused.
_SUPPLIED_CONTROLS: ContextVar[frozenset[str]] = ContextVar("_supplied_controls", default=frozenset())


def _json_number_only(value: object) -> object:
    """Accept only a finite JSON number: never a bool, a string or a non-finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Pydantic requires ValueError here so the rejection becomes a validation error.
        raise ValueError("must be a JSON number")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as exc:
        # An integer far outside the finite float range is a range error, not a crash (F04).
        # Returning the converted float also keeps the conversion away from the field schema,
        # which would otherwise overflow a second time.
        raise ValueError("must be a finite JSON number") from exc
    if not math.isfinite(number):
        raise ValueError("must be a finite JSON number")
    return number


PositiveFiniteNumber = Annotated[
    float, BeforeValidator(_json_number_only), Field(gt=0, allow_inf_nan=False)
]
"""A strictly positive finite JSON number, accepting ``int`` but never ``bool`` or ``str``."""


def _dump_models(value: object) -> list:
    """Serialize an immutable model sequence back to the agreed JSON array shape."""
    return [item.model_dump() for item in value]


def _freeze_binding_sequence(value: list) -> FrozenList:
    """Hold an already frozen binding sequence in an immutable container."""
    return FrozenList(value)


class VerificationBinding(BaseModel):
    """One verifier bound to exactly one expected outcome of the candidate action."""

    model_config = _FROZEN_CONFIG

    schema_version: SchemaVersion
    outcome_index: NonNegativeInt
    verifier_type: NonBlankStr
    config: FrozenJsonObject = Field(default_factory=dict)


class VerificationPolicy(BaseModel):
    """The ordered verification bindings of a contract.

    The order supplied by the caller is preserved and emitted unchanged; completeness and
    uniqueness are decided against the candidate by the Builder, never silently repaired.
    """

    model_config = _FROZEN_CONFIG

    schema_version: SchemaVersion
    bindings: Annotated[
        list[VerificationBinding],
        Field(strict=True),
        AfterValidator(_freeze_binding_sequence),
    ]

    @field_serializer("bindings", when_used="always")
    def _serialize_bindings(self, value: object) -> list:
        return _dump_models(value)


class RecoveryPolicy(BaseModel):
    """The explicit recovery policy a contract is committed with.

    ``abort`` is the default and the only policy the Builder creates on its own: it performs no
    recovery and makes no retry or replanning claim. Its committed JSON shape is exactly the
    original two fields, ``{"schema_version": "0.1", "mode": "abort"}``; the bounded controls are
    not leaked as nulls or defaults.

    ``bounded`` requires an explicit ``max_retries`` and an explicit ``allow_replan``. Neither is
    defaulted, so no contract can acquire a retry or a replan budget by omission. ``max_retries``
    of zero is legal and means the contract has no retry allowance at all.
    """

    model_config = _FROZEN_CONFIG

    schema_version: SchemaVersion
    mode: Literal["abort", "bounded"] = "abort"
    max_retries: NonNegativeInt | None = None
    allow_replan: StrictBool | None = None

    @model_validator(mode="after")
    def _controls_match_the_mode(self) -> RecoveryPolicy:
        """Require the bounded controls for ``bounded`` and forbid them for ``abort``.

        An aborting policy refuses any explicitly supplied bounded control, including an explicit
        null: the presence of the input is what is rejected, not merely a non-null value, so a
        caller cannot smuggle a retry or replan allowance into an abort policy. Omission stays
        legal, including the rebuilt record that an internal revalidation produces.
        """
        if self.mode == "abort":
            if _SUPPLIED_CONTROLS.get() & {"max_retries", "allow_replan"}:
                raise ValueError("an aborting policy does not accept bounded controls")
            return self
        if self.max_retries is None:
            raise ValueError("a bounded policy requires an explicit max_retries")
        if self.allow_replan is None:
            raise ValueError("a bounded policy requires an explicit allow_replan")
        return self

    def __init__(self, /, **data: object) -> None:
        # Record which bounded controls the caller actually supplied, so the after-validator can
        # tell an explicit null apart from an omitted field. Only a mapping can carry an explicit
        # key, so any other input shape leaves the set empty.
        supplied = frozenset(key for key in ("max_retries", "allow_replan") if key in data)
        token = _SUPPLIED_CONTROLS.set(supplied)
        try:
            super().__init__(**data)
        finally:
            _SUPPLIED_CONTROLS.reset(token)

    @model_serializer(mode="plain", when_used="always")
    def _serialize_policy(self) -> dict:
        """Emit the exact agreed shape: two fields for ``abort``, four for ``bounded``.

        The bounded controls are omitted entirely in abort mode rather than emitted as nulls or
        defaults, so every pre-existing abort dump, including ``model_dump_json`` and the nested
        contract dump, keeps its original byte-for-byte structure.
        """
        if self.mode == "abort":
            return {"schema_version": self.schema_version, "mode": "abort"}
        return {
            "schema_version": self.schema_version,
            "mode": "bounded",
            "max_retries": self.max_retries,
            "allow_replan": self.allow_replan,
        }


class ActionContract(BaseModel):
    """The immutable, versioned record of one accepted action candidate.

    IDs and ``created_at`` come from the caller: the Builder never reads a clock and never
    generates an identifier. The contract records a decision, not an execution and not task
    success.
    """

    model_config = _FROZEN_CONFIG

    schema_version: SchemaVersion
    contract_id: NonBlankStr
    goal_id: NonBlankStr
    forecast_id: NonBlankStr
    observation_id: NonBlankStr
    adapter_name: NonBlankStr
    environment_type: NonBlankStr
    action: Annotated[ActionSpec, AfterValidator(freeze_action)]
    preconditions: Annotated[
        list[Precondition],
        Field(strict=True),
        AfterValidator(freeze_preconditions),
    ]
    expected_outcomes: Annotated[
        list[ExpectedOutcome],
        Field(strict=True),
        AfterValidator(freeze_expected_outcomes),
    ]
    verification_policy: VerificationPolicy
    recovery_policy: RecoveryPolicy
    timeout_ms: PositiveFiniteNumber
    created_at: Timestamp
    metadata: FrozenJsonObject = Field(default_factory=dict)

    @field_serializer("action", when_used="always")
    def _serialize_action(self, value: object) -> dict:
        return value.model_dump()

    @field_serializer("preconditions", "expected_outcomes", when_used="always")
    def _serialize_models(self, value: object) -> list:
        return _dump_models(value)

    @model_validator(mode="after")
    def _check_intrinsic_structure(self) -> ActionContract:
        """Reject a structurally incomplete contract, whatever route built it."""
        outcomes = self.expected_outcomes
        if not any(outcome.required for outcome in outcomes):
            raise ValueError("a contract requires at least one required expected outcome")
        indices = [binding.outcome_index for binding in self.verification_policy.bindings]
        if len(set(indices)) != len(indices):
            raise ValueError("verification bindings must not repeat an outcome index")
        if sorted(indices) != list(range(len(outcomes))):
            raise ValueError("every expected outcome needs exactly one verification binding")
        return self
