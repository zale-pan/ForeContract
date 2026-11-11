"""The Runtime state machine: which phase transitions are legal, and nothing else.

The machine is a pure table. It reads no clock, performs no I/O, holds no mutable state and makes
no decision about *why* a transition happens: it only answers whether the transition is legal. That
separation matters because the same table is what makes an illegal transition a bounded failure
that leaves the phase untouched, rather than a partially applied state change.

Two of the contract's rules live here rather than in the engine:

* a terminal phase has no outgoing transition at all, so a finished Run can never be resumed; and
* an in-phase activity such as the post-execution observation during ``VERIFYING``, or the
  recovery-time observation during ``RECOVERING``, is *not* a transition. The phase stays where it
  is and the pending operation is tracked separately, so a phase name is never used to carry
  attempt association and an activity can never be mistaken for a phase change.
"""

from __future__ import annotations

from typing import Final

from forecontract.runtime.errors import RuntimeError
from forecontract.runtime.models import (
    ACTIVE_PHASES,
    PHASES,
    TERMINAL_PHASES,
    RuntimePhase,
)

__all__ = ["LEGAL_TRANSITIONS", "RuntimeStateMachine", "is_legal_transition"]

#: The complete legal transition table. Every entry is an explicit edge of the approved contract;
#: a pair that is absent is illegal, so the table is closed by construction rather than by a
#: separately maintained deny list.
LEGAL_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    "NEW": frozenset({"OBSERVING"}),
    "OBSERVING": frozenset({"CHECKING_TASK"}),
    "CHECKING_TASK": frozenset({"COMPLETED", "PLANNING"}),
    "PLANNING": frozenset({"BUILDING"}),
    "BUILDING": frozenset({"CHECKING_PRECONDITIONS"}),
    "CHECKING_PRECONDITIONS": frozenset({"EXECUTING", "PLANNING"}),
    "EXECUTING": frozenset({"VERIFYING"}),
    "VERIFYING": frozenset({"CHECKING_TASK", "RECOVERING"}),
    "RECOVERING": frozenset({"CHECKING_TASK", "CHECKING_PRECONDITIONS", "PLANNING"}),
    # A terminal phase is final. The empty set is the explicit statement that nothing follows, and it
    # applies equally to the five terminal phases: a finished Run can never be resumed.
    "COMPLETED": frozenset(),
    "FAILED": frozenset(),
    "ABORTED": frozenset(),
    "CANCELLED": frozenset(),
    "TIMED_OUT": frozenset(),
}

#: The phases from which a genuine facility failure may reach ``FAILED``, from which an explicit
#: policy or limit stop may reach ``ABORTED``, and from which a delivered cancellation or a real
#: deadline expiry may reach ``CANCELLED`` or ``TIMED_OUT``. They are added at runtime so the base
#: table stays exactly the normal-flow graph and the real-trigger edges are stated once.
_ACTIVE_PHASES: Final[tuple[str, ...]] = ACTIVE_PHASES

for _phase in _ACTIVE_PHASES:
    LEGAL_TRANSITIONS[_phase] = LEGAL_TRANSITIONS[_phase] | {
        "FAILED", "ABORTED", "CANCELLED", "TIMED_OUT"}

#: Every declared phase must appear exactly once, so the table can never drift from the vocabulary.
if set(LEGAL_TRANSITIONS) != set(PHASES):  # pragma: no cover - import-time consistency guard
    raise RuntimeError("INVALID_TRANSITION", "facility", (),
                       "the transition table does not cover the declared phases")


def is_legal_transition(source: object, target: object) -> bool:
    """Whether ``source -> target`` is one of the approved edges.

    The comparison is on the closed phase vocabulary, so a caller-supplied string or a foreign
    object is simply not a phase and therefore not a legal source or target.
    """
    try:
        source_name = RuntimePhase(source).value if not isinstance(source, RuntimePhase) else source.value
        target_name = RuntimePhase(target).value if not isinstance(target, RuntimePhase) else target.value
    except ValueError:
        return False
    return target_name in LEGAL_TRANSITIONS.get(source_name, frozenset())


class RuntimeStateMachine:
    """The current phase of one Run, with the legal edges enforced on every move.

    The machine owns exactly one piece of state, the current phase. An illegal move raises a
    bounded :class:`~forecontract.runtime.errors.RuntimeError` and leaves the phase untouched, so a
    rejected transition can never leave a half-applied state behind.
    """

    __slots__ = ("_phase",)

    def __init__(self, phase: RuntimePhase = RuntimePhase.NEW) -> None:
        if not isinstance(phase, RuntimePhase):
            raise RuntimeError("INPUT_INVALID", "validate", ("phase",),
                               "the initial phase is not a declared phase")
        self._phase = phase

    @property
    def phase(self) -> RuntimePhase:
        """The current phase."""
        return self._phase

    @property
    def is_terminal(self) -> bool:
        """Whether the Run has reached a final phase."""
        return self._phase.value in TERMINAL_PHASES

    @property
    def is_active(self) -> bool:
        """Whether the Run may still perform business work."""
        return not self.is_terminal

    def can_move_to(self, target: RuntimePhase) -> bool:
        """Whether moving to ``target`` is legal from the current phase."""
        return is_legal_transition(self._phase, target)

    def move_to(self, target: RuntimePhase) -> RuntimePhase:
        """Move to ``target`` when the edge is legal, else refuse and change nothing.

        Returns the new phase so the caller records the transition it just made instead of
        re-reading the machine and risking a mismatch.
        """
        if not isinstance(target, RuntimePhase):
            raise RuntimeError("INPUT_INVALID", "validate", ("phase",),
                               "the target phase is not a declared phase")
        if not self.can_move_to(target):
            raise RuntimeError(
                "INVALID_TRANSITION", "facility", (self._phase.value, target.value),
                "the transition is not a declared edge",
            )
        self._phase = target
        return self._phase
