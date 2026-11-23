"""Bounded, deterministic JSON-rule verification, offline extension verifiers and a completion oracle.

The package offers distinct evaluation surfaces over explicit evidence inputs:

* :class:`~forecontract.verification.rules.JsonRuleEvaluator` turns one rule and one explicit
  piece of evidence into a bounded verdict;
* :class:`~forecontract.verification.step.StepVerifier` verifies a committed contract's own
  expected outcomes, using the contract's index-bound verification policy;
* :class:`~forecontract.verification.oracle.TaskCompletionOracle` evaluates independently
  supplied completion criteria and never consults a forecast, a contract or a completion hint;
* :class:`~forecontract.verification.composite.CompositeVerifier` aggregates deterministic and
  offline model verdicts under ``all``, ``any`` and ``weighted`` modes while enforcing critical
  deterministic gates;
* :class:`~forecontract.verification.model.ModelVerifier` validates and interprets externally
  supplied model verification payloads offline, recording declared request/response provenance
  without performing remote calls or treating non-deterministic model scores as critical gates.

Only two deterministic JSON rules exist - ``value_equals`` and ``path_exists`` - over an exact path
relative to an observation state. Every call is synchronous, deterministic and stateless.
Insufficient valid evidence is a verdict; damaged input, an unusable rule and an unusable binding
are :class:`~forecontract.verification.errors.VerificationError` facility failures and are never
reported as a verdict.

What this package is not: a Runtime, a recovery or retry policy, an observation source, a network
model client, a trace or Memory writer, a terminal-state authority, or proof of task success. A
PASS here means the supplied rules or evidence inputs held under the verifier's contract, and
nothing more.
"""

from __future__ import annotations

from forecontract.verification.composite import CompositeVerifier
from forecontract.verification.errors import VerificationError
from forecontract.verification.hooks import RuleValidationHooks
from forecontract.verification.model import ModelVerifier
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
    "CompositeVerifier",
    "JsonRuleEvaluator",
    "ModelVerifier",
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
