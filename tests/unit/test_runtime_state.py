"""Acceptance tests for the Runtime state machine.

The machine is a pure transition table, so these tests assert the exact legal edge set, that an
illegal edge is refused while leaving the phase untouched, that a terminal phase has no exit at
all, and that an in-phase activity never fabricates a transition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.runtime import PHASES, TERMINAL_PHASES, RuntimeError, RuntimePhase
from forecontract.runtime.state import (
    LEGAL_TRANSITIONS,
    RuntimeStateMachine,
    is_legal_transition,
)

#: The exact normal-flow edges of the approved contract, stated independently of the table.
NORMAL_EDGES = {
    ("NEW", "OBSERVING"),
    ("OBSERVING", "CHECKING_TASK"),
    ("CHECKING_TASK", "COMPLETED"),
    ("CHECKING_TASK", "PLANNING"),
    ("PLANNING", "BUILDING"),
    ("BUILDING", "CHECKING_PRECONDITIONS"),
    ("CHECKING_PRECONDITIONS", "EXECUTING"),
    ("CHECKING_PRECONDITIONS", "PLANNING"),
    ("EXECUTING", "VERIFYING"),
    ("VERIFYING", "CHECKING_TASK"),
    ("VERIFYING", "RECOVERING"),
    ("RECOVERING", "CHECKING_TASK"),
    ("RECOVERING", "CHECKING_PRECONDITIONS"),
    ("RECOVERING", "PLANNING"),
}

ACTIVE_PHASES = tuple(phase for phase in PHASES if phase not in TERMINAL_PHASES)


def test_the_table_covers_every_declared_phase_exactly_once() -> None:
    assert set(LEGAL_TRANSITIONS) == set(PHASES)
    assert len(LEGAL_TRANSITIONS) == len(PHASES)


def test_the_normal_flow_edges_are_exactly_the_approved_ones() -> None:
    declared = {(source, target) for source, targets in LEGAL_TRANSITIONS.items()
                for target in targets}
    # The approval adds two real-trigger terminals, so every active phase can now also reach
    # ``CANCELLED`` and ``TIMED_OUT``. The normal-flow edges and the two original failure edges are
    # unchanged; only these two are added, which is asserted exactly rather than by relaxing the
    # comparison.
    real_trigger_edges = {(source, "CANCELLED") for source in ACTIVE_PHASES} | {
        (source, "TIMED_OUT") for source in ACTIVE_PHASES}
    failure_edges = {(source, "FAILED") for source in ACTIVE_PHASES} | {
        (source, "ABORTED") for source in ACTIVE_PHASES}
    assert declared == NORMAL_EDGES | failure_edges | real_trigger_edges


def test_every_active_phase_can_reach_both_failure_terminals() -> None:
    """The original name and its original subject are restored.

    The assertion still states exactly what its name says - every active phase can reach both failure
    terminals - and the two real-trigger terminals the approval added are asserted alongside it rather
    than replacing it, so no earlier coverage is lost.
    """
    for phase in ACTIVE_PHASES:
        assert LEGAL_TRANSITIONS[phase] >= {"FAILED", "ABORTED"}, phase
        assert LEGAL_TRANSITIONS[phase] >= {"CANCELLED", "TIMED_OUT"}, phase


@pytest.mark.parametrize("phase", sorted(TERMINAL_PHASES))
def test_a_terminal_phase_has_no_outgoing_transition(phase: str) -> None:
    assert LEGAL_TRANSITIONS[phase] == frozenset()
    machine = RuntimeStateMachine(RuntimePhase(phase))
    assert machine.is_terminal
    assert not machine.is_active
    for target in RuntimePhase:
        assert not machine.can_move_to(target)
        with pytest.raises(RuntimeError) as info:
            machine.move_to(target)
        assert info.value.code == "INVALID_TRANSITION"
        assert machine.phase is RuntimePhase(phase)


def test_an_illegal_move_is_refused_and_leaves_the_phase_untouched() -> None:
    machine = RuntimeStateMachine()
    assert machine.phase is RuntimePhase.NEW
    with pytest.raises(RuntimeError) as info:
        machine.move_to(RuntimePhase.PLANNING)
    assert info.value.code == "INVALID_TRANSITION"
    assert info.value.stage == "facility"
    assert machine.phase is RuntimePhase.NEW
    # The legal move still works from the unchanged state.
    assert machine.move_to(RuntimePhase.OBSERVING) is RuntimePhase.OBSERVING


def test_the_whole_normal_path_is_walkable_in_order() -> None:
    machine = RuntimeStateMachine()
    for target in (RuntimePhase.OBSERVING, RuntimePhase.CHECKING_TASK, RuntimePhase.PLANNING,
                   RuntimePhase.BUILDING, RuntimePhase.CHECKING_PRECONDITIONS,
                   RuntimePhase.EXECUTING, RuntimePhase.VERIFYING, RuntimePhase.CHECKING_TASK,
                   RuntimePhase.COMPLETED):
        assert machine.move_to(target) is target
    assert machine.is_terminal


def test_the_recovery_paths_are_walkable() -> None:
    for decision_path in (RuntimePhase.CHECKING_TASK, RuntimePhase.CHECKING_PRECONDITIONS,
                          RuntimePhase.PLANNING):
        machine = RuntimeStateMachine(RuntimePhase.VERIFYING)
        machine.move_to(RuntimePhase.RECOVERING)
        assert machine.move_to(decision_path) is decision_path


def test_a_failure_reaches_the_terminal_from_every_active_phase() -> None:
    for phase in ACTIVE_PHASES:
        for terminal in (RuntimePhase.FAILED, RuntimePhase.ABORTED):
            machine = RuntimeStateMachine(RuntimePhase(phase))
            assert machine.move_to(terminal) is terminal


@pytest.mark.parametrize("source,target", [
    ("NEW", "PLANNING"), ("NEW", "EXECUTING"), ("OBSERVING", "PLANNING"),
    ("CHECKING_TASK", "EXECUTING"), ("PLANNING", "CHECKING_TASK"),
    ("BUILDING", "EXECUTING"), ("EXECUTING", "CHECKING_TASK"),
    ("COMPLETED", "OBSERVING"), ("FAILED", "PLANNING"), ("ABORTED", "NEW"),
])
def test_an_undeclared_edge_is_not_legal(source: str, target: str) -> None:
    assert not is_legal_transition(source, target)
    machine = RuntimeStateMachine(RuntimePhase(source))
    with pytest.raises(RuntimeError):
        machine.move_to(RuntimePhase(target))
    assert machine.phase is RuntimePhase(source)


def test_an_in_phase_activity_is_not_a_transition() -> None:
    """The post-execution observation during VERIFYING, and the recovery observation during
    RECOVERING, keep the phase. The table has no self-edge, so an activity can never be recorded
    as a phase change and an Attempt association can never be carried by a phase name."""
    for phase in (RuntimePhase.VERIFYING, RuntimePhase.RECOVERING):
        assert phase.value not in LEGAL_TRANSITIONS[phase.value]
        assert not is_legal_transition(phase, phase)
        machine = RuntimeStateMachine(phase)
        with pytest.raises(RuntimeError):
            machine.move_to(phase)
        assert machine.phase is phase


def test_the_machine_refuses_a_value_that_is_not_a_declared_phase() -> None:
    with pytest.raises(RuntimeError) as info:
        RuntimeStateMachine("NOT_A_PHASE")  # type: ignore[arg-type]
    assert info.value.code == "INPUT_INVALID"
    machine = RuntimeStateMachine()
    with pytest.raises(RuntimeError) as other:
        machine.move_to("NOT_A_PHASE")  # type: ignore[arg-type]
    assert other.value.code == "INPUT_INVALID"
    assert machine.phase is RuntimePhase.NEW


def test_the_legality_query_refuses_a_non_phase_value() -> None:
    assert not is_legal_transition("NOT_A_PHASE", "OBSERVING")
    assert not is_legal_transition(object(), RuntimePhase.OBSERVING)
    assert not is_legal_transition(RuntimePhase.NEW, object())
    assert is_legal_transition(RuntimePhase.NEW, RuntimePhase.OBSERVING)
    # A declared phase is accepted by name as well as by member.
    assert is_legal_transition("NEW", "OBSERVING")
