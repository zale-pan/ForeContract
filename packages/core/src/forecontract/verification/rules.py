"""The shared, stateless JSON rule evaluator.

:class:`JsonRuleEvaluator` is the one place where a rule is turned into a bounded verdict. It
is synchronous, deterministic and free of state, callbacks and registration, and it is used
unchanged by the step verifier and by the task-completion oracle.

Order of work, always: validate the whole rule and the whole evidence, resolve the exact path,
then compare. A malformed rule or malformed evidence is a facility error and can never be
reported as FAIL or INCONCLUSIVE, because insufficient evidence and broken input are different
things.
"""

from __future__ import annotations

import math

from forecontract.models.json_types import JsonObject
from forecontract.models.outcomes import VerificationStatus
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import (
    EVIDENCE_INSUFFICIENT,
    PATH_ABSENT,
    PATH_EXISTS,
    PATH_PRESENT,
    VALUE_EQUALS,
    VALUE_MATCH,
    VALUE_MISMATCH,
    PathExistsRule,
    RuleEvaluation,
    ValueEqualsRule,
    VerificationEvidence,
    validate_declared_instance,
)

__all__ = ["JsonRuleEvaluator"]

#: A sentinel distinguishing "the path is absent" from "the path holds JSON null".
_ABSENT = object()


def _is_finite(value: float) -> bool:
    """Whether a float is a finite JSON number."""
    return math.isfinite(value)


def revalidate_argument(record: object, model: object, location: str) -> None:
    """Revalidate one component argument from its own stored fields, before anything else."""
    models = model if isinstance(model, tuple) else (model,)
    if not isinstance(record, models):
        expected = " or ".join(item.__name__ for item in models)
        raise VerificationError(
            "VERIFICATION_INPUT_INVALID",
            (location,),
            f"a validated {expected} is required",
        )
    validate_declared_instance(record, tuple(type(record).model_fields))


def _require_json(value: object, seen: tuple[int, ...] = ()) -> object:
    """Return the value if it is finite JSON, otherwise a bounded input error.

    The declared observation state is already a ``JsonObject``, so this guards the raw object a
    caller can still reach: a container that is genuinely cyclic, a non-finite float, bytes, a
    non-string object key and any other non-JSON content are refused. A legal shared reference is
    visited once per path, because only containers on the current path are remembered.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not _is_finite(value):
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("observation", "state"),
                "the observation state is not finite JSON",
            )
        return value
    if isinstance(value, (list, dict)):
        if id(value) in seen:
            raise VerificationError(
                "VERIFICATION_INPUT_INVALID",
                ("observation", "state"),
                "the observation state contains a cyclic reference",
            )
        nested_seen = seen + (id(value),)
        if isinstance(value, list):
            return [_require_json(item, nested_seen) for item in value]
        for key, item in value.items():
            if not isinstance(key, str):
                raise VerificationError(
                    "VERIFICATION_INPUT_INVALID",
                    ("observation", "state"),
                    "the observation state has a non-string object key",
                )
            _require_json(item, nested_seen)
        return value
    raise VerificationError(
        "VERIFICATION_INPUT_INVALID",
        ("observation", "state"),
        "the observation state is not JSON",
    )


def _resolve(state: JsonObject, path: list) -> object:
    """Walk the exact path, or return ``_ABSENT``.

    A key that is missing, an index past the end of an array, an array index used on an object,
    a string key used on an array and any attempt to continue inside a scalar or null all mean
    the path is not present. That is a legal verdict, never a malformed rule.
    """
    current: object = state
    for segment in path:
        if isinstance(segment, bool):
            return _ABSENT
        if isinstance(segment, str):
            if not isinstance(current, dict) or segment not in current:
                return _ABSENT
            current = current[segment]
            continue
        if not isinstance(current, list) or segment >= len(current):
            return _ABSENT
        current = current[segment]
    return current


def json_equal(left: object, right: object) -> bool:
    """Type-aware recursive JSON equality, as the agreed rule semantics require.

    ``True``, ``1`` and ``1.0`` are three distinct values. Object key order is irrelevant and
    array order is significant. Two objects are equal only when they carry exactly the same
    string keys.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, float) and isinstance(right, float):
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_equal(item, other) for item, other in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if len(left) != len(right) or set(left) != set(right):
            return False
        return all(json_equal(left[key], right[key]) for key in left)
    return False


class JsonRuleEvaluator:
    """Evaluate one JSON rule against one piece of explicit evidence.

    The class holds no state and takes no constructor argument, so repeated calls with equal
    arguments always produce equal results and no callback can be injected.
    """

    def evaluate(
        self, *, rule: ValueEqualsRule | PathExistsRule, evidence: VerificationEvidence
    ) -> RuleEvaluation:
        """Return the bounded verdict of the rule truth table for this rule and evidence."""
        revalidate_argument(evidence, VerificationEvidence, "evidence")
        revalidate_argument(rule, (ValueEqualsRule, PathExistsRule), "rule")
        state = evidence.observation.state
        _require_json(state)
        rule_type = rule.rule_type
        path = list(rule.path)
        found = _resolve(state, path)

        if rule_type == VALUE_EQUALS:
            if found is _ABSENT:
                # Equality can never be decided from a path the evidence does not reach,
                # whatever the completeness declaration says.
                return self._result(VerificationStatus.INCONCLUSIVE, False, None,
                                    EVIDENCE_INSUFFICIENT)
            matched = json_equal(found, rule.expected)
            if matched:
                return self._result(VerificationStatus.PASS, True, found, VALUE_MATCH)
            return self._result(VerificationStatus.FAIL, True, found, VALUE_MISMATCH)

        if rule_type == PATH_EXISTS:
            if found is not _ABSENT:
                # An observed JSON null is a present value.
                return self._result(VerificationStatus.PASS, True, found, PATH_PRESENT)
            if evidence.state_complete:
                return self._result(VerificationStatus.FAIL, False, None, PATH_ABSENT)
            return self._result(VerificationStatus.INCONCLUSIVE, False, None,
                                EVIDENCE_INSUFFICIENT)

        raise VerificationError(
            "VERIFICATION_RULE_INVALID", ("rule", "rule_type"), "the rule type is not supported"
        )

    @staticmethod
    def _result(status: VerificationStatus, present: bool, actual: object,
                reason: str) -> RuleEvaluation:
        """Build the agreed verdict record, rejecting a self-contradictory combination."""
        return RuleEvaluation(
            schema_version="0.1",
            status=status,
            has_actual_value=present,
            actual_value=actual,
            reason=reason,
        )


_ = (PATH_EXISTS,)
