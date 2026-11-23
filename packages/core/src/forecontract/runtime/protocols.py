"""The dependency protocols a Runtime is injected with.

These are Core-declared structural contracts. The Runtime depends on them and never on a concrete
adapter, planner, store or domain package, so Core keeps no dependency on Fake, ScriptedPlanner,
Web, Coding or any external automation framework.

A protocol states an expected calling convention. It cannot certify that an implementation is
honest: nothing here authenticates a provider, reads a hidden state or proves that an observation
describes the real world. What the Runtime can do is hand every dependency an isolated input, ask
for an explicit declaration instead of inferring one, and check that the dependency did not mutate
what it was given.

The domain support protocol exists so a domain can supply the four things only the domain knows -
the completeness of its evidence, its preconditions and policies, the repeat safety of its action
and the effect state of an outcome - while the Runtime keeps the authority to decide when a Run
succeeds and how the host's limits are spent.
"""

from __future__ import annotations

from typing import Protocol

from forecontract.contracts.validation import ContractValidationHooks
from forecontract.models.actions import ActionSpec, Forecast
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome
from forecontract.recovery.models import ActionSafetyDeclaration
from forecontract.runtime.models import ContractPreparation, RuntimeBudgetSnapshot
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)

__all__ = ["BudgetSource", "ContractHooksFactory", "Planner", "RuntimeDomainSupport",
           "TaskCriteriaValidator"]


class Planner(Protocol):
    """The existing asynchronous planning contract: one observation in, one Forecast out.

    The real :class:`forecontract_scripted.ScriptedPlanner` satisfies this protocol. A forecast is
    a *proposal*: its ``task_completion_hint`` is advisory and can never make a Run succeed, and a
    proposed action still has to be committed by the Builder before anything may run.
    """

    async def plan(self, observation: Observation) -> Forecast:
        """Return a candidate action for the supplied observation."""
        ...


class BudgetSource(Protocol):
    """A synchronous host declaration of the budget that remains for a Run.

    The Runtime reads it at the agreed boundaries and refuses an illegal or increasing value. This
    is a *declaration*: it does not prove that wall-clock time was measured, and a production
    monotonic clock with real await deadlines is deliberately out of scope for this task.
    """

    def snapshot(self) -> RuntimeBudgetSnapshot:
        """Return the current remaining budget for this Run."""
        ...


class RuntimeDomainSupport(Protocol):
    """The synchronous, no-I/O domain companion of the Runtime.

    Every method must be a pure function of its arguments together with explicit domain
    configuration: it must not read a hidden environment state, call an adapter, mutate anything it
    was handed or decide final task success. The Runtime checks the pre- and post-call structure of
    every argument and refuses a call that mutated its input rather than accepting a contaminated
    snapshot.
    """

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        """State explicitly whether this observation's evidence is complete.

        The returned evidence must carry the very observation that this ``observe`` call returned.
        Substituting a different observation, hiding state or fabricating history is a facility
        failure, and completeness must be stated, never inferred.
        """
        ...

    def prepare_contract(self, *, forecast: Forecast,
                         evidence: VerificationEvidence) -> ContractPreparation:
        """Propose the domain preconditions and policies for a forecast.

        The result is a candidate only. The Runtime submits it to the Builder, which decides.
        """
        ...

    def declare_action_safety(self, *, contract: object) -> ActionSafetyDeclaration:
        """Declare the repeat safety of the contracted action.

        The declaration must be bound to this contract's identifier, adapter name and action type.
        Repeat safety is never inferred from the action name, an error message, a ``recoverable``
        flag or a planner hint.
        """
        ...

    def assess_effect(self, *, contract: object, actual_outcome: ActualOutcome,
                      evidence: VerificationEvidence) -> str:
        """Judge the effect state of an outcome: ``not_applied``, ``applied`` or ``unknown``.

        A failed execution must not be mapped to ``not_applied`` automatically: whether an effect
        happened is a domain judgement about the world, not a restatement of the execution status.
        """
        ...


class ContractHooksFactory(Protocol):
    """The optional factory that supplies Builder-facing domain validation hooks per evidence.

    The Runtime calls :meth:`create` once at contract build time and once again immediately before
    execution, each time with the evidence that belongs to *that* boundary. A hooks instance is
    therefore never reused across phases: the completeness declaration and the observation a hook
    will eventually judge must be the current ones, not a cached earlier pair.

    The returned object is the existing four-method ``ContractValidationHooks`` contract - it is
    not a new Builder interface. ``validate_precondition``, ``validate_expected_outcome`` and
    ``validate_verification_binding`` return ``None``; ``check_precondition`` returns a
    ``VerificationStatus``. The factory is a synchronous, side-effect-free extension point, and
    the Runtime hands it a detached copy of the current evidence.
    """

    def create(self, *, evidence: VerificationEvidence) -> ContractValidationHooks:
        """Return one hooks instance for exactly this piece of evidence."""
        ...


class TaskCriteriaValidator(Protocol):
    """The optional synchronous prevalidation of a Run's task criteria.

    The Runtime calls :meth:`validate` once, before it writes a trace record, starts the adapter or
    observes anything, so an unusable domain rule is refused before any resource is acquired. The
    validator is independent of :class:`ContractHooksFactory`: supplying a factory never bypasses
    this prevalidation and supplying a validator never changes the factory's two call sites.

    A successful call returns ``None`` and performs no I/O. An illegal criterion is signalled with
    the existing Core :class:`~forecontract.verification.errors.VerificationError` carrying
    ``VERIFICATION_INPUT_INVALID``; the Runtime maps it to its own ``INPUT_INVALID``. A missing
    method, a readable-attribute failure, an awaitable or non-``None`` return and any other ordinary
    exception are a ``DEPENDENCY_FAILURE``. A validator that mutates the record it was handed is an
    ``EVIDENCE_INVALID``. ``BaseException`` keeps its cancellation/interrupt identity.

    The validator decides only whether the criteria are usable. It never evaluates completion,
    never returns a verdict and never calls the task oracle.
    """

    def validate(self, *, criteria: TaskCompletionCriteria) -> None:
        """Return ``None`` for usable criteria, or reject them."""
        ...


def action_is_declared(declaration: ActionSafetyDeclaration, contract_id: str, adapter_name: str,
                       action: ActionSpec) -> bool:
    """Whether a safety declaration is exactly bound to this contract, adapter and action type.

    The Runtime calls this instead of trusting a declaration that was produced for another
    contract, which would otherwise let a retry decision be made from an unrelated declaration.
    """
    return (declaration.contract_id == contract_id
            and declaration.adapter_name == adapter_name
            and declaration.action_type == action.action_type)
