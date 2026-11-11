"""Bounded, deterministic JSON-rule verification and an independent task-completion oracle.

The package offers three distinct evaluation surfaces over one shared rule engine:

* :class:`~forecontract.verification.rules.JsonRuleEvaluator` turns one rule and one explicit
  piece of evidence into a bounded verdict;
* :class:`~forecontract.verification.step.StepVerifier` verifies a committed contract's own
  expected outcomes, using the contract's index-bound verification policy;
* :class:`~forecontract.verification.oracle.TaskCompletionOracle` evaluates independently
  supplied completion criteria and never consults a forecast, a contract or a completion hint.

Only two rules exist - ``value_equals`` and ``path_exists`` - over an exact path relative to an
observation state. Every call is synchronous, deterministic and stateless. Insufficient valid
evidence is a verdict; damaged input, an unusable rule and an unusable binding are
:class:`~forecontract.verification.errors.VerificationError` facility failures and are never
reported as a verdict.

What this package is not: a Runtime, a recovery or retry policy, an observation source, a trace
or Memory writer, a terminal-state authority, or proof of task success. A PASS here means the
supplied rules held for the supplied evidence, and nothing more.
"""

from __future__ import annotations

from forecontract.verification.errors import VerificationError
from forecontract.verification.hooks import RuleValidationHooks
from forecontract.verification.models import (
    PathExistsRule,
    RuleEvaluation,
    TaskCompletionCriteria,
    ValueEqualsRule,
    VerificationEvidence,
)
from forecontract.verification.oracle import TaskCompletionOracle
from forecontract.verification.rules import JsonRuleEvaluator
from forecontract.verification.step import StepVerifier

__version__ = "0.1.0"

__all__ = [
    "JsonRuleEvaluator",
    "PathExistsRule",
    "RuleEvaluation",
    "RuleValidationHooks",
    "StepVerifier",
    "TaskCompletionCriteria",
    "TaskCompletionOracle",
    "ValueEqualsRule",
    "VerificationError",
    "VerificationEvidence",
]
