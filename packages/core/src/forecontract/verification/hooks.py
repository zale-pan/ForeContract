"""The Builder-facing domain validation hooks for JSON rules.

The existing Contract Builder injects a synchronous :class:`ContractValidationHooks`
implementation. :class:`RuleValidationHooks` is that implementation for the ``json_rule``
domain, and it implements the four existing methods without changing the protocol or the
Builder.

``check_precondition`` receives only an :class:`Observation`, so it cannot read a completeness
declaration from it. The declaration is therefore an explicit required constructor argument:
one hooks instance asserts one completeness statement for every observation it is asked about.
A caller whose evidence completeness changes builds a new instance. The value is never taken
from observation metadata and is never silently defaulted to ``True``.

A domain rejection leaves this module as the project's existing safe
:class:`InvalidActionPayloadError`, so the Builder handles it through its established domain
rejection path instead of turning a malformed payload into a dependency failure. Any other
exception is left alone: an internal programming fault is never swallowed as a business answer.
"""

from __future__ import annotations

from forecontract.contracts.models import VerificationBinding
from forecontract.models.actions import ExpectedOutcome, Precondition
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.protocols.adapter import InvalidActionPayloadError
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import VerificationEvidence
from forecontract.verification.rules import JsonRuleEvaluator
from forecontract.verification.step import JSON_RULE, parse_json_rule

__all__ = ["RuleValidationHooks"]


class RuleValidationHooks:
    """Validate and evaluate ``json_rule`` payloads for the Contract Builder.

    ``state_complete`` is required and strict: it is the caller's explicit statement about the
    evidence this instance will be used with.
    """

    def __init__(self, *, state_complete: bool) -> None:
        if not isinstance(state_complete, bool):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("state_complete",),
                "state_complete must be a boolean",
            )
        self._state_complete = state_complete
        self._evaluator = JsonRuleEvaluator()

    @property
    def state_complete(self) -> bool:
        """The explicit completeness statement this instance was built with."""
        return self._state_complete

    def validate_precondition(self, condition: Precondition) -> None:
        """Reject a precondition that is not a usable json_rule condition."""
        self._check_domain_payload(
            condition.condition_type, condition.payload, ("condition", "payload")
        )

    def validate_expected_outcome(self, outcome: ExpectedOutcome) -> None:
        """Reject an expected outcome that is not a usable json_rule condition."""
        self._check_domain_payload(
            outcome.outcome_type, outcome.payload, ("outcome", "payload")
        )

    def validate_verification_binding(self, binding: VerificationBinding,
                                      outcome: ExpectedOutcome) -> None:
        """Reject a binding that points elsewhere or carries a second rule definition."""
        if binding.verifier_type != JSON_RULE:
            raise InvalidActionPayloadError(
                "the binding must name the json_rule verifier"
            )
        if len(binding.config) != 0:
            raise InvalidActionPayloadError(
                "a json_rule binding must not carry a rule definition"
            )
        del outcome

    def check_precondition(self, condition: Precondition,
                           observation: Observation) -> VerificationStatus:
        """Evaluate one json_rule precondition against this instance's completeness statement."""
        rule = self._rule_or_domain_error(
            condition.condition_type, condition.payload, ("condition", "payload")
        )
        evidence = self._evidence(observation)
        return self._evaluator.evaluate(rule=rule, evidence=evidence).status

    def _evidence(self, observation: Observation) -> VerificationEvidence:
        try:
            return VerificationEvidence(
                schema_version="0.1",
                observation=observation,
                state_complete=self._state_complete,
            )
        except VerificationError as exc:
            raise InvalidActionPayloadError(
                "the supplied observation cannot be used as verification evidence"
            ) from exc

    def _rule_or_domain_error(self, payload_type: object, payload: object,
                              location: tuple[str | int, ...]) -> object:
        try:
            return self._parse(payload_type, payload, location)
        except VerificationError as exc:
            raise InvalidActionPayloadError(
                "the domain payload is not a usable json_rule definition"
            ) from exc

    def _check_domain_payload(self, payload_type: object, payload: object,
                              location: tuple[str | int, ...]) -> None:
        try:
            self._parse(payload_type, payload, location)
        except VerificationError as exc:
            raise InvalidActionPayloadError(
                "the domain payload is not a usable json_rule definition"
            ) from exc

    @staticmethod
    def _parse(payload_type: object, payload: object,
               location: tuple[str | int, ...]) -> object:
        if payload_type != JSON_RULE:
            raise VerificationError(
                "VERIFICATION_RULE_INVALID",
                (location[0], "type"),
                "the domain type is not supported",
            )
        return parse_json_rule(payload, location)
