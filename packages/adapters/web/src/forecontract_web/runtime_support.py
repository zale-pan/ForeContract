"""Strict, explicit Web ``RuntimeDomainSupport`` and its per-action policy records.

``WebActionPolicy`` is a frozen, strict, unknown-field-rejecting record with exactly seven
fields: the full Core ``ActionSpec``, the pre-execution preconditions, the repeat-safety
declaration, the two effect-proof groups, the Core recovery policy and one positive timeout.
``WebRuntimeDomainSupport`` matches an action by the complete, type-sensitive content of that
``ActionSpec`` (metadata included), states evidence completeness only for the explicitly
required URL and targets, prepares a candidate contract without changing the forecast, declares
repeat safety from the matched policy alone and judges an effect only from the supplied evidence.

Every method is synchronous and performs no I/O. The support never reads a live page, a private
adapter or site state, never calls the adapter, never writes a trace or a file and never decides
that a task succeeded: that verdict belongs to the independent task oracle and the host's own
criteria.

The fixed public error codes of this module are ``WEB_RUNTIME_SUPPORT_INVALID`` and
``WEB_RUNTIME_POLICY_NOT_FOUND``. A diagnostic carries only a fixed sentence and a bounded
structural location, never a supplied value. A converted ordinary failure is raised only after
its handler has finished, so the public error carries neither ``__cause__`` nor ``__context__``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Annotated, Final, Literal

from forecontract.contracts.models import (
    ActionContract,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import (
    ActionSpec,
    ExpectedOutcome,
    Forecast,
    Precondition,
)
from forecontract.models.base import SCHEMA_VERSION
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome
from forecontract.recovery.models import ActionSafetyDeclaration
from forecontract.runtime.models import ContractPreparation
from forecontract.verification.models import VerificationEvidence
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from forecontract_web.models import RoleTarget, TestIdTarget
from forecontract_web.rule_models import WEB_RULE_TYPES
from forecontract_web.rules import (
    WebRuleEvaluator,
    WebVerificationInputInvalid,
    _prepare_rule,
    _revalidate_evidence,
    _revalidated_model,
)
from forecontract_web.validation import parse_web_action, parse_web_observation
from forecontract_web.verification import WEB_RULE_BINDING

__all__ = ["WebActionPolicy", "WebRuntimeDomainSupport", "WebRuntimeSupportError"]

_SUPPORT_INVALID: Final[str] = "WEB_RUNTIME_SUPPORT_INVALID"
_POLICY_NOT_FOUND: Final[str] = "WEB_RUNTIME_POLICY_NOT_FOUND"

_POLICY_MESSAGE: Final[str] = "the web action policy is not usable"
_SUPPORT_MESSAGE: Final[str] = "the web runtime domain support configuration is not usable"
_NOT_FOUND_MESSAGE: Final[str] = "no web action policy matches this action"
_EVIDENCE_MESSAGE: Final[str] = "the supplied web evidence is not usable"
_OUTCOME_MESSAGE: Final[str] = "the supplied actual outcome is not usable"

_FIELD_LOCATION: Final[str] = "<field>"
_MAX_LOCATION: Final[int] = 8


def _format_location(location: tuple[str | int, ...]) -> str:
    """Render a bounded structural location without any supplied value."""
    text = ""
    for part in location[:_MAX_LOCATION]:
        if isinstance(part, int) and not isinstance(part, bool):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text or "<root>"


def _attempt(call: Callable[[], object]) -> tuple[bool, object]:
    """Run one caller-owned read and leave the handler before any replacement is raised."""
    try:
        return True, call()
    except Exception:  # noqa: BLE001 - converted to a bounded refusal outside the handler
        return False, None


class WebRuntimeSupportError(Exception):
    """A bounded Web runtime-support facility failure carrying one fixed code.

    The constructor accepts only the two fixed codes. An unknown code collapses to
    ``WEB_RUNTIME_SUPPORT_INVALID`` so caller text can never become a public code. The error is
    always raised outside its handler, so it has no ``__cause__``/``__context__`` and carries no
    supplied value.
    """

    code: str = _SUPPORT_INVALID

    def __init__(self, message: str, *, code: str | None = None,
                 location: tuple[str | int, ...] = ()) -> None:
        self.message = str(message)
        self.location = tuple(location)
        if code is not None and code in (_SUPPORT_INVALID, _POLICY_NOT_FOUND):
            self.code = code
        super().__init__(
            f"{self.code} at {_format_location(self.location)}: {self.message}"
        )


def _invalid(location: tuple[str | int, ...] = (),
             message: str = _SUPPORT_MESSAGE) -> WebRuntimeSupportError:
    """Build one fixed support-invalid error outside a handler."""
    return WebRuntimeSupportError(message, code=_SUPPORT_INVALID, location=location)


def _not_found() -> WebRuntimeSupportError:
    """Build one fixed policy-not-found error outside a handler."""
    return WebRuntimeSupportError(_NOT_FOUND_MESSAGE, code=_POLICY_NOT_FOUND)


_POLICY_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True,
                            validate_assignment=True, hide_input_in_errors=True)


def _finite_positive_milliseconds(value: object) -> object:
    """Accept a finite positive JSON number, never a bool or a numeric string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_ms must be a JSON number")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("timeout_ms must be finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError("timeout_ms must be finite and positive")
    return number


def _real_type(value: object) -> type:
    """The real type slot, never a caller-owned ``__class__`` property."""
    return type(value)


def _json_key(key: object) -> object:
    """A key signature that keeps the real type and never calls a caller comparison."""
    if _real_type(key) is str:
        return (str, key)
    if _real_type(key) is int:
        return (int, key)
    if _real_type(key) is bool:
        return (bool, key)
    if _real_type(key) is float:
        return (float, repr(key))
    if key is None:
        return ("none",)
    return ("opaque", _real_type(key), repr(key))


def _canonical(value: object, active: set[int]) -> object:
    """A cycle-safe, type-sensitive canonical form of a Core/Web value."""
    if isinstance(value, BaseModel):
        marker = id(value)
        if marker in active:
            return ("cycle",)
        active.add(marker)
        try:
            return tuple((name, _canonical(getattr(value, name, None), active))
                         for name in type(value).model_fields)
        finally:
            active.discard(marker)
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            return ("cycle",)
        active.add(marker)
        try:
            pairs = [(_json_key(key), _canonical(item, active))
                     for key, item in value.items()]
        finally:
            active.discard(marker)
        pairs.sort(key=repr)
        return ("mapping", tuple(pairs))
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            return ("cycle",)
        active.add(marker)
        try:
            return ("sequence", tuple(_canonical(item, active) for item in value))
        finally:
            active.discard(marker)
    if value is None:
        return ("none",)
    real = _real_type(value)
    if real is float:
        return (float, repr(value))
    return (real, value)


def _action_signature(action: object) -> object:
    """The complete type-sensitive identity of one action, metadata included."""
    return _canonical(action, set())


def _safe_policy_location(exc: ValidationError) -> tuple[str | int, ...]:
    """A bounded structural location of one policy schema rejection."""
    errors = exc.errors()
    if not errors:
        return ()
    declared = frozenset(WebActionPolicy.model_fields)
    location: list[str | int] = []
    for part in errors[0].get("loc") or ():
        if isinstance(part, bool):
            location.append(_FIELD_LOCATION)
        elif isinstance(part, int):
            location.append(part if 0 <= part <= 99 else _FIELD_LOCATION)
        elif isinstance(part, str) and part in declared:
            location.append(part)
        else:
            location.append(_FIELD_LOCATION)
    return tuple(location[:_MAX_LOCATION])


class WebActionPolicy(BaseModel):
    """One explicit, immutable policy for exactly one complete Web action.

    The policy never infers anything from an action type name: its ``action`` must be one of the
    real six Web payloads, and matching elsewhere compares the complete type-sensitive content,
    including ``metadata``.
    """

    model_config = _POLICY_CONFIG

    action: ActionSpec
    preconditions: Annotated[list[Precondition], Field(strict=True)]
    retry_safety: Literal["read_only", "idempotent", "non_idempotent", "unknown"]
    applied_when: Annotated[list[ExpectedOutcome], Field(strict=True)]
    not_applied_when: Annotated[list[ExpectedOutcome], Field(strict=True)]
    recovery_policy: RecoveryPolicy
    timeout_ms: float

    def __init__(self, /, **data: object) -> None:
        location: tuple[str | int, ...] = ()
        failure: ValidationError | None = None
        try:
            super().__init__(**data)
        except ValidationError as exc:
            failure = exc
            location = _safe_policy_location(exc)
        if failure is not None:
            raise _invalid(location) from None

    @field_validator("action", mode="before")
    @classmethod
    def _real_web_action(cls, value: object) -> ActionSpec:
        ok, parsed = _attempt(lambda: parse_web_action(value))  # type: ignore[arg-type]
        if not ok:
            raise ValueError("the action is not a valid Web action")
        return parsed  # type: ignore[return-value]

    @field_validator("timeout_ms", mode="before")
    @classmethod
    def _strict_timeout(cls, value: object) -> object:
        return _finite_positive_milliseconds(value)

    @model_validator(mode="after")
    def _all_parts_are_usable(self) -> WebActionPolicy:
        """Revalidate every nested record and every Web rule before the policy is accepted."""

        def rebuild() -> tuple[list, list, list, RecoveryPolicy]:
            preconditions = [
                _revalidated_model(item, Precondition, _POLICY_MESSAGE,
                                   WebVerificationInputInvalid,
                                   ("preconditions", index))
                for index, item in enumerate(self.preconditions)
            ]
            for index, item in enumerate(preconditions):
                _prepare_rule(item.condition_type, item.payload,
                              location=("preconditions", index, "payload"))
            applied = self._effect_group(self.applied_when, "applied_when")
            not_applied = self._effect_group(self.not_applied_when, "not_applied_when")
            recovery = _revalidated_model(
                self.recovery_policy, RecoveryPolicy, _POLICY_MESSAGE,
                WebVerificationInputInvalid, ("recovery_policy",))
            return preconditions, applied, not_applied, recovery

        ok, built = _attempt(rebuild)
        if not ok:
            raise ValueError("a nested policy record is not usable")
        preconditions, applied, not_applied, recovery = built  # type: ignore[misc]
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "applied_when", applied)
        object.__setattr__(self, "not_applied_when", not_applied)
        object.__setattr__(self, "recovery_policy", recovery)
        return self

    @staticmethod
    def _effect_group(items: list[ExpectedOutcome], field: str) -> list[ExpectedOutcome]:
        """Revalidate one effect group; every supplied item is required and a Web rule."""
        rebuilt: list[ExpectedOutcome] = []
        for index, item in enumerate(items):
            condition = _revalidated_model(
                item, ExpectedOutcome, _POLICY_MESSAGE, WebVerificationInputInvalid,
                (field, index))
            if condition.outcome_type not in WEB_RULE_TYPES:
                raise WebVerificationInputInvalid(
                    "the effect condition is not a Web rule", path=(field, index))
            if condition.required is not True:
                raise WebVerificationInputInvalid(
                    "every supplied effect condition must be required",
                    path=(field, index, "required"))
            _prepare_rule(condition.outcome_type, condition.payload,
                          location=(field, index, "payload"))
            rebuilt.append(condition)
        return rebuilt


class WebRuntimeDomainSupport:
    """The explicit, pure Web domain companion of the Core Runtime.

    The configuration is stored as detached immutable tuples. Only the declared target records,
    the two fixed booleans/values and the seven-field policies are accepted; a callback, script,
    regex, page object or oracle handle is not a declared field and is refused.
    """

    __slots__ = ("_action_policies", "_adapter_name", "_policies", "_require_url",
                 "_targets")

    def __init__(self, *, adapter_name: str, required_targets: list[TestIdTarget | RoleTarget],
                 require_url: bool, action_policies: list[WebActionPolicy],
                 **extra: object) -> None:
        if extra:
            # No undeclared configuration is accepted: a callback, script, regex, page object or
            # oracle handle is simply not one of the four declared keywords.
            raise _invalid((_FIELD_LOCATION,))
        self._adapter_name = self._name(adapter_name)
        self._require_url = self._boolean(require_url)
        self._targets = self._validate_targets(required_targets)
        policies = self._validate_policies(action_policies)
        self._action_policies = tuple(policies)
        self._policies = tuple((_action_signature(policy.action), policy) for policy in policies)

    # ------------------------------------------------------------------ validation

    @staticmethod
    def _name(value: object) -> str:
        if _real_type(value) is not str:
            raise _invalid(("adapter_name",))
        ok, blank = _attempt(lambda: not value.strip())
        if not ok or blank:
            raise _invalid(("adapter_name",))
        return value

    @staticmethod
    def _boolean(value: object) -> bool:
        if _real_type(value) is not bool:
            raise _invalid(("require_url",))
        return value

    @staticmethod
    def _target_model(value: object, index: int) -> type:
        """Select the declared locator model from the real strategy value."""
        if _real_type(value) is TestIdTarget:
            return TestIdTarget
        if _real_type(value) is RoleTarget:
            return RoleTarget
        if isinstance(value, Mapping):
            ok, strategy = _attempt(lambda: value.get("strategy"))
            if ok and _real_type(strategy) is str and str.__eq__(strategy, "test_id"):
                return TestIdTarget
            if ok and _real_type(strategy) is str and str.__eq__(strategy, "role"):
                return RoleTarget
        raise _invalid(("required_targets", index))

    @staticmethod
    def _target_key(target: object) -> tuple:
        if _real_type(target) is TestIdTarget:
            fields = tuple(TestIdTarget.model_fields)
        elif _real_type(target) is RoleTarget:
            fields = tuple(RoleTarget.model_fields)
        else:
            raise _invalid(("required_targets",))
        ok, key = _attempt(lambda: tuple(
            (name, getattr(target, name)) for name in sorted(fields)))
        if not ok:
            raise _invalid(("required_targets",))
        return key

    def _validate_targets(self, value: object) -> tuple:
        if not isinstance(value, list):
            raise _invalid(("required_targets",))
        targets: list = []
        seen: set[tuple] = set()
        for index, item in enumerate(value):
            model = self._target_model(item, index)

            def rebuild(item: object = item, model: type = model,
                        index: int = index) -> object:
                if isinstance(item, Mapping):
                    candidate = model.model_validate(dict(item))
                    return _revalidated_model(
                        candidate, model, _SUPPORT_MESSAGE,
                        WebVerificationInputInvalid, ("required_targets", index))
                return _revalidated_model(
                    item, model, _SUPPORT_MESSAGE,
                    WebVerificationInputInvalid, ("required_targets", index))

            ok, rebuilt = _attempt(rebuild)
            if not ok:
                raise _invalid(("required_targets", index))
            key = self._target_key(rebuilt)
            if key in seen:
                raise _invalid(("required_targets", index))
            seen.add(key)
            targets.append(rebuilt)
        return tuple(targets)

    @staticmethod
    def _validate_policies(value: object) -> list[WebActionPolicy]:
        if not isinstance(value, list) or not value:
            raise _invalid(("action_policies",))
        policies: list[WebActionPolicy] = []
        seen: set[object] = set()
        for index, item in enumerate(value):
            if not isinstance(item, WebActionPolicy):
                raise _invalid(("action_policies", index))
            ok, rebuilt = _attempt(lambda item=item, index=index: _revalidated_model(
                item, WebActionPolicy, _POLICY_MESSAGE, WebVerificationInputInvalid,
                ("action_policies", index)))
            if not ok:
                raise _invalid(("action_policies", index))
            signature = _action_signature(rebuilt.action)  # type: ignore[attr-defined]
            if signature in seen:
                raise _invalid(("action_policies", index))
            seen.add(signature)
            policies.append(rebuilt)  # type: ignore[arg-type]
        return policies

    # ------------------------------------------------------------------ protocol

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        """State completeness for this observation's explicit required URL and targets."""
        ok, snapshot = _attempt(lambda: parse_web_observation(observation))
        if not ok:
            raise _invalid(("observation",))
        complete = True
        if self._require_url and snapshot.url is None:  # type: ignore[attr-defined]
            complete = False
        records = {self._target_key(record.target): record  # type: ignore[attr-defined]
                   for record in snapshot.targets}  # type: ignore[attr-defined]
        for target in self._targets:
            record = records.get(self._target_key(target))
            if record is None or record.status != "collected":
                complete = False
                break
        ok, evidence = _attempt(lambda: VerificationEvidence(
            schema_version=SCHEMA_VERSION,
            observation=observation,
            state_complete=complete,
        ))
        if not ok:
            raise _invalid(("observation",))
        return evidence  # type: ignore[return-value]

    def prepare_contract(self, *, forecast: Forecast,
                         evidence: VerificationEvidence) -> ContractPreparation:
        """Return the matched policy's preconditions, bindings and recovery configuration."""
        ok, rebuilt = _attempt(lambda: _revalidated_model(
            forecast, Forecast, _EVIDENCE_MESSAGE, WebVerificationInputInvalid,
            ("forecast",)))
        if not ok:
            raise _invalid(("forecast",))
        policy = self._policy_for(rebuilt.proposed_action)  # type: ignore[attr-defined]
        bindings = [
            VerificationBinding(
                schema_version=SCHEMA_VERSION,
                outcome_index=index,
                verifier_type=WEB_RULE_BINDING,
            )
            for index in range(len(rebuilt.expected_outcomes))  # type: ignore[attr-defined]
        ]
        ok, preparation = _attempt(lambda: ContractPreparation(
            schema_version=SCHEMA_VERSION,
            preconditions=list(policy.preconditions),
            verification_policy=VerificationPolicy(
                schema_version=SCHEMA_VERSION, bindings=bindings),
            recovery_policy=policy.recovery_policy,
            timeout_ms=policy.timeout_ms,
        ))
        if not ok:
            raise _invalid(("preparation",))
        return preparation  # type: ignore[return-value]

    def declare_action_safety(self, *, contract: object) -> ActionSafetyDeclaration:
        """Declare the matched policy's repeat safety, bound to this committed contract."""
        rebuilt = self._contract(contract)
        if rebuilt.adapter_name != self._adapter_name:
            raise _invalid(("contract", "adapter_name"))
        policy = self._policy_for(rebuilt.action)
        ok, declaration = _attempt(lambda: ActionSafetyDeclaration(
            schema_version=SCHEMA_VERSION,
            contract_id=rebuilt.contract_id,
            adapter_name=self._adapter_name,
            action_type=rebuilt.action.action_type,
            retry_safety=policy.retry_safety,
        ))
        if not ok:
            raise _invalid(("safety",))
        return declaration  # type: ignore[return-value]

    def assess_effect(self, *, contract: object, actual_outcome: ActualOutcome,
                      evidence: VerificationEvidence) -> str:
        """Judge the effect from the matched policy's conditions and the supplied evidence only.

        The complete evidence and its Observation are validated first, before the two effect groups
        are inspected. An empty group therefore means "this policy states no judgement", never
        "skip the evidence check": a damaged evidence record is a fixed facility error even when
        both groups are empty, while a legal evidence record with no declared judgement is the
        documented ``unknown``.
        """
        rebuilt = self._contract(contract)
        policy = self._policy_for(rebuilt.action)
        ok, _ = _attempt(lambda: _revalidated_model(
            actual_outcome, ActualOutcome, _OUTCOME_MESSAGE,
            WebVerificationInputInvalid, ("actual_outcome",)))
        if not ok:
            raise _invalid(("actual_outcome",))
        rebuilt_evidence = self._validated_evidence(evidence)
        applied = self._group_passes(policy.applied_when, rebuilt_evidence, "applied_when")
        not_applied = self._group_passes(policy.not_applied_when, rebuilt_evidence,
                                         "not_applied_when")
        applied_all = applied is True
        not_applied_all = not_applied is True
        if applied_all and not not_applied_all:
            return "applied"
        if not_applied_all and not applied_all:
            return "not_applied"
        return "unknown"

    # ------------------------------------------------------------------ internals

    def _contract(self, contract: object) -> ActionContract:
        ok, rebuilt = _attempt(lambda: _revalidated_model(
            contract, ActionContract, _EVIDENCE_MESSAGE,
            WebVerificationInputInvalid, ("contract",)))
        if not ok:
            raise _invalid(("contract",))
        return rebuilt  # type: ignore[return-value]

    @staticmethod
    def _validated_evidence(evidence: object) -> VerificationEvidence:
        """Strictly revalidate the whole evidence and its Web Observation, or refuse it.

        This runs before any effect-group decision, including the case where both groups are empty.
        The nested evidence and snapshot checks are the same ones the accepted Web rule evaluator
        performs, so a structurally damaged record can never be reported as the ``unknown`` that a
        legal "no declared judgement" policy returns.
        """
        ok, rebuilt = _attempt(lambda: _revalidated_model(
            evidence, VerificationEvidence, _EVIDENCE_MESSAGE,
            WebVerificationInputInvalid, ("evidence",)))
        if not ok:
            raise _invalid(("evidence",))
        ok, _ = _attempt(lambda: _revalidate_evidence(rebuilt))
        if not ok:
            raise _invalid(("evidence",))
        return rebuilt  # type: ignore[return-value]

    def _policy_for(self, action: object) -> WebActionPolicy:
        signature = _action_signature(action)
        for known, policy in self._policies:
            if known == signature:
                return policy
        raise _not_found()

    def _group_passes(self, conditions: list[ExpectedOutcome],
                      evidence: VerificationEvidence, field: str) -> bool | None:
        """Return whether every supplied condition passes; ``None`` for an empty group."""
        if not conditions:
            return None
        ok, rebuilt_evidence = _attempt(lambda: _revalidated_model(
            evidence, VerificationEvidence, _EVIDENCE_MESSAGE,
            WebVerificationInputInvalid, ("evidence",)))
        if not ok:
            raise _invalid(("evidence",))
        evaluator = WebRuleEvaluator()
        for index, condition in enumerate(conditions):
            ok, result = _attempt(lambda condition=condition: evaluator.evaluate(
                rule_type=condition.outcome_type,
                payload=condition.payload,
                evidence=rebuilt_evidence,
            ))
            if not ok:
                raise _invalid((field, index))
            if result.status.value != "pass":  # type: ignore[attr-defined]
                return False
        return True
