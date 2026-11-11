"""The synchronous domain-validation protocol the Contract Builder depends on.

Core declares the interface and never parses a domain payload itself. A caller injects an
implementation of this protocol, together with an existing
:class:`~forecontract.protocols.adapter.Adapter`, and the Builder calls them in a fixed
order at fixed gates.

Every method is synchronous and declared to perform no I/O. The dependency is trusted: this
protocol is not a sandbox, it does not prove that a future implementation is pure or
honest, and it does not make a domain payload valid. It only gives Core a place to ask.

How a rejection is signalled: the validation methods return ``None`` and reject by raising
the project's domain-validation exception hierarchy
(:class:`~forecontract.protocols.adapter.ActionValidationError`). Any other exception, a
non-``None`` return from a ``None``-returning method, or a mutation of the supplied argument
is treated by the Builder as an unexpected dependency failure rather than as a domain
verdict. ``check_precondition`` is the one method that returns a verdict and must return a
:class:`~forecontract.models.outcomes.VerificationStatus` member.
"""

from __future__ import annotations

from typing import Protocol

from forecontract.contracts.models import VerificationBinding
from forecontract.models.actions import ExpectedOutcome, Precondition
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus

__all__ = ["ContractValidationHooks"]


class ContractValidationHooks(Protocol):
    """Domain validation callbacks injected into the Contract Builder."""

    def validate_precondition(self, condition: Precondition) -> None:
        """Reject a precondition whose domain payload is malformed."""
        ...

    def validate_expected_outcome(self, outcome: ExpectedOutcome) -> None:
        """Reject an expected outcome whose domain payload is malformed."""
        ...

    def validate_verification_binding(
        self, binding: VerificationBinding, outcome: ExpectedOutcome
    ) -> None:
        """Reject a binding whose verifier or configuration cannot verify this outcome."""
        ...

    def check_precondition(
        self, condition: Precondition, observation: Observation
    ) -> VerificationStatus:
        """Evaluate one precondition against the supplied observation."""
        ...
