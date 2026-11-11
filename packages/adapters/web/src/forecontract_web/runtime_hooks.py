"""Web contract-validation hooks for the Core Contract Builder.

``WebContractValidationHooks`` is the Web domain implementation of the existing four-method
``ContractValidationHooks`` protocol. It validates the five declared Web rule payloads through
the same public rule parser the accepted Web verifier uses and evaluates a precondition with the
already-verified :class:`~forecontract_web.rules.WebRuleEvaluator`; it does not copy a looser
parser, read a live page, access adapter private state or evaluate a task.

``WebContractHooksFactory`` creates one hooks instance from the *current* evidence at each of the
Runtime's two boundaries (contract build and pre-execution re-check). It reads the evidence's
explicit ``state_complete`` declaration and never substitutes ``True`` for an explicit ``False``.

The module is synchronous, performs no I/O, imports no browser package and exposes no callback
hook. A malformed structure is reported through the project's existing
:class:`~forecontract.protocols.adapter.InvalidActionPayloadError` domain rejection, so the Builder
keeps its established ``INVALID_DOMAIN_PAYLOAD`` mapping. A converted ordinary failure is raised
only after its handler has finished, so the public error carries neither ``__cause__`` nor
``__context__``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from forecontract.contracts.models import VerificationBinding
from forecontract.models.actions import ExpectedOutcome, Precondition
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.protocols.adapter import InvalidActionPayloadError
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)

from forecontract_web.rule_models import WEB_RULE_TYPES, WebRuleEvaluation
from forecontract_web.rules import (
    WebRuleEvaluator,
    WebVerificationInputInvalid,
    _prepare_rule,
    _revalidated_model,
)
from forecontract_web.verification import (
    WEB_RULE_BINDING,
    _revalidate_criteria,
)

__all__ = ["WebContractHooksFactory", "WebContractValidationHooks",
           "WebTaskCriteriaValidator"]

_SCHEMA_VERSION: Final[str] = "0.1"

#: Fixed, value-free diagnostics. A rejected payload never echoes a URL, text, locator or key.
_PAYLOAD_MESSAGE: Final[str] = "the domain payload is not a usable web rule definition"
_BINDING_MESSAGE: Final[str] = "the binding is not a usable web_rule binding"
_EVIDENCE_MESSAGE: Final[str] = "the supplied observation cannot be used as web verification evidence"
_CRITERIA_MESSAGE: Final[str] = "the task criteria are not a usable web task definition"


def _attempt(call: Callable[[], object]) -> tuple[bool, object]:
    """Run one caller-owned read and leave the handler before the replacement is raised."""
    try:
        return True, call()
    except Exception:  # noqa: BLE001 - converted to a bounded rejection outside the handler
        return False, None


def _domain_rejection(message: str) -> InvalidActionPayloadError:
    """Build one fixed domain rejection outside any exception handler."""
    return InvalidActionPayloadError(message)


def _revalidate_core(record: object, model: type, location: tuple[str | int, ...]) -> object:
    """Revalidate one Core record from its actual stored fields, or reject it."""
    if not isinstance(record, model):
        raise _domain_rejection(_PAYLOAD_MESSAGE)
    ok, rebuilt = _attempt(lambda: _revalidated_model(
        record, model, _PAYLOAD_MESSAGE, WebVerificationInputInvalid, location))
    if not ok:
        raise _domain_rejection(_PAYLOAD_MESSAGE)
    return rebuilt  # type: ignore[return-value]


def _validate_rule(payload_type: object, payload: object) -> None:
    """Reject an unknown rule type or a payload outside the selected rule schema."""
    if not isinstance(payload_type, str) or payload_type not in WEB_RULE_TYPES:
        raise _domain_rejection("the web rule type is not supported")
    ok, _ = _attempt(lambda: _prepare_rule(payload_type, payload))
    if not ok:
        raise _domain_rejection(_PAYLOAD_MESSAGE)


class WebTaskCriteriaValidator:
    """Prevalidate a Run's complete Web task-criteria rule set before the Runtime starts.

    The validator reuses the same structural criteria revalidation and the same five-rule parser
    the accepted :class:`~forecontract_web.verification.WebTaskCompletionOracle` uses, so it cannot
    accept a rule the oracle would reject and it does not copy a looser parser. Every condition is
    parsed, including the optional ones, because an unusable optional condition is still unusable
    evidence rather than something to ignore.

    Success returns ``None`` and decides nothing about completion. An unusable criteria record,
    rule type or payload is reported as the existing Core
    :class:`~forecontract.verification.errors.VerificationError` with the fixed
    ``VERIFICATION_INPUT_INVALID`` code, which the Runtime maps onto its own pre-start
    ``INPUT_INVALID``. The replacement is raised after its handler has finished, so it carries
    neither ``__cause__`` nor ``__context__`` and no supplied value.
    """

    def validate(self, *, criteria: TaskCompletionCriteria) -> None:
        """Return ``None`` when every declared Web criterion is usable, or reject the criteria."""
        ok, rebuilt = _attempt(lambda: _revalidate_criteria(criteria))
        if not ok:
            raise VerificationError("VERIFICATION_INPUT_INVALID", ("criteria",),
                                    _CRITERIA_MESSAGE)
        conditions = rebuilt.conditions  # type: ignore[attr-defined]
        ok, _ = _attempt(lambda: [
            _prepare_rule(condition.outcome_type, condition.payload,
                          location=("criteria", "conditions", index, "payload"))
            for index, condition in enumerate(conditions)
        ])
        if not ok:
            raise VerificationError("VERIFICATION_INPUT_INVALID", ("criteria",),
                                    _CRITERIA_MESSAGE)


class WebContractValidationHooks:
    """Validate and evaluate the five Web rules for one explicitly stated completeness.

    ``state_complete`` is a required strict boolean and is the caller's statement about the
    evidence this instance will judge. The value is never read from an observation's metadata and
    is never defaulted.
    """

    def __init__(self, *, state_complete: bool) -> None:
        if type(state_complete) is not bool:
            raise WebVerificationInputInvalid(
                "state_complete must be an explicit boolean",
                path=("state_complete",),
            )
        self._state_complete = state_complete
        self._evaluator = WebRuleEvaluator()

    @property
    def state_complete(self) -> bool:
        """The explicit completeness statement this instance was built with."""
        return self._state_complete

    def validate_precondition(self, condition: Precondition) -> None:
        """Reject a precondition that is not a usable Web rule condition."""
        rebuilt = _revalidate_core(condition, Precondition, ("condition",))
        _validate_rule(rebuilt.condition_type, rebuilt.payload)  # type: ignore[attr-defined]

    def validate_expected_outcome(self, outcome: ExpectedOutcome) -> None:
        """Reject an expected outcome that is not a usable Web rule condition."""
        rebuilt = _revalidate_core(outcome, ExpectedOutcome, ("outcome",))
        _validate_rule(rebuilt.outcome_type, rebuilt.payload)  # type: ignore[attr-defined]

    def validate_verification_binding(self, binding: VerificationBinding,
                                      outcome: ExpectedOutcome) -> None:
        """Require the fixed ``web_rule`` verifier with an empty configuration."""
        rebuilt = _revalidate_core(binding, VerificationBinding,
                                   ("verification_policy", "bindings"))
        _revalidate_core(outcome, ExpectedOutcome, ("outcome",))
        if rebuilt.verifier_type != WEB_RULE_BINDING:  # type: ignore[attr-defined]
            raise _domain_rejection(_BINDING_MESSAGE)
        if len(rebuilt.config) != 0:  # type: ignore[attr-defined]
            raise _domain_rejection(_BINDING_MESSAGE)

    def check_precondition(self, condition: Precondition,
                           observation: Observation) -> VerificationStatus:
        """Evaluate one Web precondition against this instance's explicit completeness."""
        rebuilt = _revalidate_core(condition, Precondition, ("condition",))
        _validate_rule(rebuilt.condition_type, rebuilt.payload)  # type: ignore[attr-defined]
        if not isinstance(observation, Observation):
            raise _domain_rejection(_EVIDENCE_MESSAGE)
        ok, evidence = _attempt(lambda: VerificationEvidence(
            schema_version=_SCHEMA_VERSION,
            observation=observation,
            state_complete=self._state_complete,
        ))
        if not ok:
            raise _domain_rejection(_EVIDENCE_MESSAGE)
        ok, result = _attempt(lambda: self._evaluator.evaluate(
            rule_type=rebuilt.condition_type,  # type: ignore[attr-defined]
            payload=rebuilt.payload,  # type: ignore[attr-defined]
            evidence=evidence,
        ))
        if not ok:
            raise _domain_rejection(_EVIDENCE_MESSAGE)
        if not isinstance(result, WebRuleEvaluation):
            raise _domain_rejection(_EVIDENCE_MESSAGE)
        return result.status


class WebContractHooksFactory:
    """Create one Web hooks instance from exactly the evidence supplied for this boundary."""

    def create(self, *, evidence: VerificationEvidence) -> WebContractValidationHooks:
        """Return hooks carrying the evidence's explicit completeness declaration."""
        if not isinstance(evidence, VerificationEvidence):
            raise WebVerificationInputInvalid(_EVIDENCE_MESSAGE, path=("evidence",))
        ok, rebuilt = _attempt(lambda: _revalidated_model(
            evidence, VerificationEvidence, _EVIDENCE_MESSAGE,
            WebVerificationInputInvalid, ("evidence",)))
        if not ok or type(getattr(rebuilt, "state_complete", None)) is not bool:
            raise WebVerificationInputInvalid(
                "the evidence does not state its completeness explicitly",
                path=("evidence", "state_complete"),
            )
        return WebContractValidationHooks(state_complete=rebuilt.state_complete)  # type: ignore[attr-defined]
