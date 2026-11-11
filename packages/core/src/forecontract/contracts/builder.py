"""Deterministic Contract Builder: fixed gates, fixed rejection order, immutable commit.

``ContractBuilder.build`` turns one caller-supplied candidate into one committed
:class:`~forecontract.contracts.models.ActionContract`. It is synchronous, it reads no
clock, it generates no identifier and it never touches the environment: it calls only
``adapter.name``, ``adapter.capabilities()`` and ``adapter.validate_action()``, never
``start``, ``stop``, ``observe`` or ``execute``.

The gate order is part of the contract, and so is what each gate may reject:

1. revalidate and detach the caller models, identifiers, timestamp and metadata
   (``INPUT_INVALID``);
2. revalidate the adapter capabilities and compare names and environment types
   (``ADAPTER_MISMATCH``), then reject any action, condition or outcome type the adapter
   does not declare (``UNSUPPORTED_CAPABILITY``);
3. call ``validate_action`` and then the domain hooks, in the supplied order, with detached
   copies of every argument (``INVALID_DOMAIN_PAYLOAD``, ``VERIFICATION_BINDING_INVALID``,
   ``DEPENDENCY_FAILURE``);
4. require at least one required expected outcome and exactly one binding per outcome
   (``EXPECTED_OUTCOME_REQUIRED``, ``VERIFICATION_BINDING_INVALID``);
5. check the two budget arguments (``BUDGET_INVALID``);
6. evaluate every precondition in order (``PRECONDITION_UNSATISFIED``);
7. commit the deep immutable snapshot of exactly the accepted candidate.

A structurally invalid binding index is rejected at gate 3 before any hook is called with
it, because calling a dependency with data that cannot be interpreted would not be a
deterministic decision. Nothing is repaired on the way: outcomes are never replaced,
optional conditions are never dropped and bindings are never remapped.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from datetime import datetime

from pydantic import BaseModel, TypeAdapter, ValidationError

from forecontract.contracts.errors import ContractBuildError
from forecontract.contracts.models import (
    ActionContract,
    RecoveryPolicy,
    VerificationPolicy,
)
from forecontract.contracts.snapshot import detach
from forecontract.models.actions import (
    AdapterCapabilities,
    Forecast,
    Precondition,
)
from forecontract.models.base import SCHEMA_VERSION
from forecontract.models.json_types import JsonObject
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.protocols.adapter import (
    ActionValidationError,
    UnsupportedActionError,
)

__all__ = ["ContractBuilder"]

_JSON_OBJECT = TypeAdapter(JsonObject)


def _validation_location(location: tuple[str | int, ...], exc: ValidationError) -> tuple[str | int, ...]:
    """Prepend the gate location to the reported schema location of the first error."""
    errors = exc.errors()
    if not errors:
        return location
    return tuple(location) + tuple(errors[0].get("loc", ()))


def _validation_message(exc: ValidationError) -> str:
    """Describe a schema failure without copying any rejected value into the text."""
    errors = exc.errors()
    if not errors:
        return "the supplied data does not match the declared schema"
    first = errors[0]
    return f"schema violation {first.get('type')}: {first.get('msg')}"


def _detach(
    model_type: type[BaseModel],
    value: object,
    location: tuple[str | int, ...],
    code: str = "INPUT_INVALID",
) -> BaseModel:
    """Revalidate and detach one caller-owned model, mapping failures to ``code``."""
    try:
        return detach(model_type, value)
    except ValidationError as exc:
        raise ContractBuildError(
            code, _validation_location(location, exc), _validation_message(exc)
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ContractBuildError(code, location, f"{model_type.__name__} instance is required") from exc


def _recovery_policy(value: object) -> RecoveryPolicy:
    """Validate and detach the optional committed recovery policy.

    An omitted policy commits the original fixed abort policy, so the default behaviour and its
    exact committed shape never change. A supplied policy is revalidated from its own fields and
    detached, so mutating the caller's object afterwards cannot change the committed contract and
    an invalid or incomplete policy is rejected rather than repaired.
    """
    if value is None:
        return RecoveryPolicy(schema_version=SCHEMA_VERSION, mode="abort")
    return _detach(RecoveryPolicy, value, ("recovery_policy",))  # type: ignore[return-value]


def _identifier(value: object, location: tuple[str | int, ...]) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ContractBuildError("INPUT_INVALID", location, "a non-blank string identifier is required")
    return value


def _timestamp(value: object, location: tuple[str | int, ...]) -> datetime:
    """Accept a timezone-aware datetime or its own ISO-8601 JSON representation."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractBuildError(
                "INPUT_INVALID", location, "an ISO-8601 datetime string is required"
            ) from exc
    else:
        raise ContractBuildError(
            "INPUT_INVALID", location, "a timezone-aware datetime or ISO-8601 string is required"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractBuildError("INPUT_INVALID", location, "the timestamp must be timezone-aware")
    return parsed


def _metadata(value: object, location: tuple[str | int, ...]) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractBuildError("INPUT_INVALID", location, "a JSON object is required")
    try:
        return _JSON_OBJECT.validate_python(dict(value))
    except ValidationError as exc:
        raise ContractBuildError(
            "INPUT_INVALID", _validation_location(location, exc), _validation_message(exc)
        ) from exc


def _budget(value: object, location: tuple[str | int, ...]) -> float:
    """A budget argument is a strictly positive finite JSON number, never a bool or string.

    An integer whose magnitude is outside the finite float range is a range error at this
    gate, so it is reported as BUDGET_INVALID with its cause preserved instead of escaping
    as an OverflowError (F04).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractBuildError(
            "BUDGET_INVALID", location, "a strictly positive finite JSON number is required"
        )
    try:
        number = float(value)
    except OverflowError as exc:
        raise ContractBuildError(
            "BUDGET_INVALID", location, "the budget is outside the finite numeric range"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise ContractBuildError(
            "BUDGET_INVALID", location, "a strictly positive finite JSON number is required"
        )
    return number


def _plain(value: object) -> object:
    """A plain, serializer-free view of a model, container or scalar.

    Reading the declared fields directly avoids invoking a serializer on a value a callback
    may have damaged, and keeps the comparison independent of JSON encoding. A list is never
    rewritten as a tuple, because that substitution is itself a change the comparison has to
    see (F03).
    """
    if isinstance(value, BaseModel):
        return {name: _plain(getattr(value, name, None)) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _key_signature(key: object) -> object:
    """A JSON-type-aware signature of one mapping key.

    The key's type is part of its identity: a JSON object key is always a string, so replacing
    the string key ``"1"`` with the integer key ``1`` is a replacement even though Python
    considers those keys equal. Converting the key with ``str()`` here is what hid that change
    (F03), so the raw key is kept instead.
    """
    if isinstance(key, str):
        return ("str", key)
    return _signature(key)


def _signature(value: object) -> object:
    """A JSON-type-aware signature of a plain value.

    Plain ``==`` cannot see a callback that rewrites ``True`` as ``1``, ``1`` as ``1.0``, a
    list as a tuple with equal elements, or a string key as an equal integer key, because
    Python considers those equal. The signature therefore keeps the JSON type of every scalar,
    the container kind of every array, and the type of every object key. Object keys are
    sorted by their signature so key order stays irrelevant, as it is in JSON.
    """
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Mapping):
        items = sorted(
            ((_key_signature(key), _signature(item)) for key, item in value.items()),
            key=lambda pair: (repr(pair[0]), repr(pair[1])),
        )
        return ("object", tuple(items))
    if isinstance(value, list):
        return ("list", tuple(_signature(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_signature(item) for item in value))
    return ("other", type(value).__name__, repr(value))


def _argument_signature(value: object) -> object:
    """The comparable signature of one callback argument."""
    return _signature(_plain(value))


def _hook(
    hooks: object,
    name: str,
    arguments: tuple[tuple[object, tuple[str | int, ...]], ...],
    location: tuple[str | int, ...],
    code: str,
    *,
    returns_status: bool = False,
) -> object:
    """Call one injected hook with detached copies and police its contract.

    A rejection is signalled with the project's domain-validation exception hierarchy. Any
    other exception, a wrong return value or a change to a supplied argument is an
    unexpected dependency failure reported at the location of the offending argument, and
    the caller's own objects are never handed over.
    """
    method = getattr(hooks, name, None)
    if not callable(method):
        raise ContractBuildError(
            "DEPENDENCY_FAILURE", location, f"hooks do not provide {name}()"
        )
    clones = [copy.deepcopy(argument) for argument, _ in arguments]
    before = [_argument_signature(clone) for clone in clones]
    try:
        result = method(*clones)
    except ActionValidationError as exc:
        raise ContractBuildError(code, location, f"{name} rejected the supplied data") from exc
    except Exception as exc:
        raise ContractBuildError(
            "DEPENDENCY_FAILURE", location, f"{name} raised {type(exc).__name__}"
        ) from exc
    for index, clone in enumerate(clones):
        if _argument_signature(clone) != before[index]:
            raise ContractBuildError(
                "DEPENDENCY_FAILURE", arguments[index][1], f"{name} mutated its argument"
            )
    if returns_status:
        if not isinstance(result, VerificationStatus):
            raise ContractBuildError(
                "DEPENDENCY_FAILURE", location, f"{name} must return a VerificationStatus member"
            )
        return result
    if result is not None:
        raise ContractBuildError("DEPENDENCY_FAILURE", location, f"{name} must return None")
    return None


def _capabilities(adapter: object) -> AdapterCapabilities:
    method = getattr(adapter, "capabilities", None)
    if not callable(method):
        raise ContractBuildError(
            "DEPENDENCY_FAILURE", ("adapter", "capabilities"), "the adapter does not provide capabilities()"
        )
    try:
        declared = method()
    except Exception as exc:
        raise ContractBuildError(
            "DEPENDENCY_FAILURE", ("adapter", "capabilities"),
            f"adapter capabilities() raised {type(exc).__name__}",
        ) from exc
    return _detach(AdapterCapabilities, declared, ("adapter", "capabilities"), "DEPENDENCY_FAILURE")


def _adapter_name(adapter: object) -> str:
    try:
        name = adapter.name
    except Exception as exc:
        raise ContractBuildError(
            "DEPENDENCY_FAILURE", ("adapter", "name"), f"adapter name raised {type(exc).__name__}"
        ) from exc
    if not isinstance(name, str) or not name.strip():
        raise ContractBuildError(
            "DEPENDENCY_FAILURE", ("adapter", "name"), "the adapter name must be a non-blank string"
        )
    return name


class ContractBuilder:
    """Builds one immutable :class:`ActionContract` through the fixed gate order."""

    def build(
        self,
        *,
        forecast: Forecast,
        observation: Observation,
        adapter: object,
        hooks: object,
        preconditions: list[Precondition],
        verification_policy: VerificationPolicy,
        timeout_ms: float,
        remaining_budget_ms: float,
        contract_id: str,
        goal_id: str,
        created_at: datetime | str,
        metadata: Mapping | None = None,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> ActionContract:
        """Reject deterministically or return the deep immutable accepted snapshot.

        ``recovery_policy`` is optional and keyword-only. Omitting it commits the original fixed
        abort policy, so every pre-existing call keeps its previous behaviour and its exact
        committed shape. A supplied policy is validated from its own fields, detached from the
        caller's mutable data and committed inside the contract's deep immutable snapshot; it is
        never re-read or overridden per assessment.
        """
        # Gate 1: revalidate and detach every caller-supplied structure.
        forecast_snapshot = _detach(Forecast, forecast, ("forecast",))
        observation_snapshot = _detach(Observation, observation, ("observation",))
        if not isinstance(preconditions, list):
            raise ContractBuildError(
                "INPUT_INVALID", ("preconditions",), "a list of preconditions is required"
            )
        condition_snapshots = [
            _detach(Precondition, condition, ("preconditions", index))
            for index, condition in enumerate(preconditions)
        ]
        policy_snapshot = _detach(VerificationPolicy, verification_policy, ("verification_policy",))
        recovery_snapshot = _recovery_policy(recovery_policy)
        metadata_snapshot = _metadata(metadata, ("metadata",))
        contract_snapshot = _identifier(contract_id, ("contract_id",))
        goal_snapshot = _identifier(goal_id, ("goal_id",))
        created_at_snapshot = _timestamp(created_at, ("created_at",))

        # Gate 2: the adapter declaration decides what this candidate may even contain.
        capabilities = _capabilities(adapter)
        if _adapter_name(adapter) != capabilities.adapter_name:
            raise ContractBuildError(
                "ADAPTER_MISMATCH", ("adapter", "name"),
                "the adapter name does not match its declared capabilities",
            )
        if observation_snapshot.environment_type != capabilities.environment_type:
            raise ContractBuildError(
                "ADAPTER_MISMATCH", ("observation", "environment_type"),
                "the observation environment type does not match the declared capabilities",
            )
        action = forecast_snapshot.proposed_action
        outcomes = list(forecast_snapshot.expected_outcomes)
        if action.action_type not in capabilities.supported_actions:
            raise ContractBuildError(
                "UNSUPPORTED_CAPABILITY", ("forecast", "proposed_action", "action_type"),
                "the adapter does not declare support for this action type",
            )
        for index, condition in enumerate(condition_snapshots):
            if condition.condition_type not in capabilities.supported_condition_types:
                raise ContractBuildError(
                    "UNSUPPORTED_CAPABILITY", ("preconditions", index, "condition_type"),
                    "the adapter does not declare support for this condition type",
                )
        for index, outcome in enumerate(outcomes):
            if outcome.outcome_type not in capabilities.supported_outcome_types:
                raise ContractBuildError(
                    "UNSUPPORTED_CAPABILITY", ("forecast", "expected_outcomes", index, "outcome_type"),
                    "the adapter does not declare support for this outcome type",
                )

        # Gate 3: domain callbacks, on detached copies, in the supplied order.
        validate_action = getattr(adapter, "validate_action", None)
        if not callable(validate_action):
            raise ContractBuildError(
                "DEPENDENCY_FAILURE", ("adapter", "validate_action"),
                "the adapter does not provide validate_action()",
            )
        action_signature = _argument_signature(action)
        action_clone = action.model_copy(deep=True)
        try:
            returned = validate_action(action_clone)
        except UnsupportedActionError as exc:
            raise ContractBuildError(
                "UNSUPPORTED_CAPABILITY", ("forecast", "proposed_action"),
                "the adapter rejected this action type as unsupported",
            ) from exc
        except ActionValidationError as exc:
            raise ContractBuildError(
                "INVALID_DOMAIN_PAYLOAD", ("forecast", "proposed_action", "payload"),
                "the adapter rejected the action payload",
            ) from exc
        except Exception as exc:
            raise ContractBuildError(
                "DEPENDENCY_FAILURE", ("adapter", "validate_action"),
                f"adapter validate_action raised {type(exc).__name__}",
            ) from exc
        if returned is not None:
            raise ContractBuildError(
                "DEPENDENCY_FAILURE", ("adapter", "validate_action"), "validate_action must return None"
            )
        if _argument_signature(action_clone) != action_signature:
            raise ContractBuildError(
                "DEPENDENCY_FAILURE", ("forecast", "proposed_action"),
                "validate_action mutated the supplied action",
            )
        for index, condition in enumerate(condition_snapshots):
            _hook(hooks, "validate_precondition", ((condition, ("preconditions", index)),),
                  ("preconditions", index), "INVALID_DOMAIN_PAYLOAD")
        for index, outcome in enumerate(outcomes):
            _hook(hooks, "validate_expected_outcome", ((outcome, ("forecast", "expected_outcomes", index)),),
                  ("forecast", "expected_outcomes", index), "INVALID_DOMAIN_PAYLOAD")
        bindings = list(policy_snapshot.bindings)
        for index, binding in enumerate(bindings):
            if binding.outcome_index >= len(outcomes):
                raise ContractBuildError(
                    "VERIFICATION_BINDING_INVALID",
                    ("verification_policy", "bindings", index, "outcome_index"),
                    "the binding index is outside the candidate expected outcomes",
                )
            _hook(hooks, "validate_verification_binding",
                  ((binding, ("verification_policy", "bindings", index)),
                   (outcomes[binding.outcome_index], ("forecast", "expected_outcomes", binding.outcome_index))),
                  ("verification_policy", "bindings", index), "VERIFICATION_BINDING_INVALID")

        # Gate 4: the structural completeness of the candidate.
        if not any(outcome.required for outcome in outcomes):
            raise ContractBuildError(
                "EXPECTED_OUTCOME_REQUIRED", ("forecast", "expected_outcomes"),
                "a contract requires at least one required expected outcome",
            )
        indices = [binding.outcome_index for binding in bindings]
        if len(set(indices)) != len(indices) or sorted(indices) != list(range(len(outcomes))):
            raise ContractBuildError(
                "VERIFICATION_BINDING_INVALID", ("verification_policy", "bindings"),
                "every expected outcome needs exactly one verification binding",
            )

        # Gate 5: the caller-observed budget, never a newly created deadline.
        timeout = _budget(timeout_ms, ("timeout_ms",))
        remaining = _budget(remaining_budget_ms, ("remaining_budget_ms",))
        if timeout > remaining:
            raise ContractBuildError(
                "BUDGET_INVALID", ("timeout_ms",), "timeout_ms exceeds the remaining budget"
            )

        # Gate 6: evaluate every precondition, in order, against a detached observation.
        for index, condition in enumerate(condition_snapshots):
            verdict = _hook(hooks, "check_precondition",
                            ((condition, ("preconditions", index)), (observation_snapshot, ("observation",))),
                            ("preconditions", index), "INVALID_DOMAIN_PAYLOAD", returns_status=True)
            if condition.required and verdict is not VerificationStatus.PASS:
                raise ContractBuildError(
                    "PRECONDITION_UNSATISFIED", ("preconditions", index),
                    f"a required precondition evaluated to {verdict.value}",
                )

        # Gate 7: commit exactly the accepted candidate as a deep immutable snapshot.
        try:
            return ActionContract(
                schema_version=SCHEMA_VERSION,
                contract_id=contract_snapshot,
                goal_id=goal_snapshot,
                forecast_id=forecast_snapshot.forecast_id,
                observation_id=observation_snapshot.observation_id,
                adapter_name=capabilities.adapter_name,
                environment_type=capabilities.environment_type,
                action=action,
                preconditions=condition_snapshots,
                expected_outcomes=outcomes,
                verification_policy=policy_snapshot,
                recovery_policy=recovery_snapshot,
                timeout_ms=timeout,
                created_at=created_at_snapshot,
                metadata=metadata_snapshot,
            )
        except ValidationError as exc:
            raise ContractBuildError(
                "INPUT_INVALID",
                _validation_location(("contract",), exc),
                "the accepted candidate could not be committed as a contract",
            ) from exc
