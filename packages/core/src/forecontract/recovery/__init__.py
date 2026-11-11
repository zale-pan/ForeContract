"""Deterministic recovery decisions for one committed contract attempt.

This subpackage answers what a runtime may do after one attempt, and nothing more. It is
synchronous and stateless: the same valid inputs always produce the same
:class:`~forecontract.recovery.models.RecoveryAssessment`.

It never executes or observes anything, never loops or waits, never reads a clock, never
generates an identifier, never resets or consumes a budget, never writes a trace or memory
record, and never calls the planner or the builder. Association checks prove that the supplied
data agrees with itself and with the committed contract; they do not authenticate a provider,
prove that an observation really happened or guarantee any ordering. The runtime owns the actual
execution, the provenance of every observation and the terminal state of a run.

A verified step is not task success, a failed step does not prove that the effect was absent, and
a recoverable error alone never authorises a retry.
"""

from __future__ import annotations

from forecontract.recovery.engine import RecoveryEngine
from forecontract.recovery.errors import RecoveryError
from forecontract.recovery.models import (
    ActionSafetyDeclaration,
    RecoveryAssessment,
    RecoveryContext,
    RecoveryEvidence,
)

__all__ = [
    "ActionSafetyDeclaration",
    "RecoveryAssessment",
    "RecoveryContext",
    "RecoveryEngine",
    "RecoveryError",
    "RecoveryEvidence",
]

__version__ = "0.1.0"
