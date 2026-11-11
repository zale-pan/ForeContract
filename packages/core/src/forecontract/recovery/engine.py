"""The stateless deterministic recovery decision engine.

:class:`RecoveryEngine` answers one question: given the current attempt, the committed policy, the
domain repeat-safety declaration and, once available, the post-attempt evidence, what may the
runtime do next? It returns a bounded :class:`~forecontract.recovery.models.RecoveryAssessment`
and nothing else.

The engine is synchronous and stateless. The same valid inputs always produce the same
assessment. It never executes or observes anything, never loops or waits, never reads a clock,
never generates an identifier, never resets or consumes a budget, never writes a trace or memory
record, and never calls the planner or the builder. The runtime owns the actual ordering, the
provenance of every observation and the terminal state of a run.

A failed step never proves that the effect was absent, and a recoverable error alone never
authorises a retry. Nothing here claims task success: a verified step only means that step's
expectations held for the evidence supplied.
"""

from __future__ import annotations

from typing import Final

from forecontract.recovery.errors import RecoveryError
from forecontract.recovery.models import (
    BUDGET_EXHAUSTED,
    DECISION_ABORT,
    DECISION_REPLAN,
    DECISION_RETRY,
    EFFECT_APPLIED,
    EFFECT_NOT_APPLIED,
    EFFECT_UNKNOWN,
    EFFECT_UNRESOLVED,
    KIND_DECISION,
    KIND_NO_RECOVERY,
    KIND_OBSERVATION_REQUIRED,
    OBSERVATION_REQUIRED,
    POLICY_ABORT,
    RECOVERY_NOT_ALLOWED,
    REPLAN_ALLOWED,
    RETRY_ALLOWED,
    SAFE_TO_REPEAT,
    STEP_VERIFIED,
    STEP_VERIFIED_AFTER_EXECUTION_ANOMALY,
    STOP_REQUESTED,
    VERIFICATION_INCONCLUSIVE,
    ActionSafetyDeclaration,
    RecoveryAssessment,
    RecoveryContext,
    RecoveryEvidence,
    validate_committed_shape,
)

__all__ = ["RecoveryEngine"]

#: The step verifier identity a host expects by default. A host that composes a different real
#: verifier states its own identity at construction time; the engine never learns one from a
#: supplied verdict or from evidence, so a result cannot promote itself into an accepted verifier.
DEFAULT_STEP_VERIFIER: Final[str] = "json_step"

#: The longest configured verifier identity. A name is a declared dependency label, not a data
#: channel, so an unbounded one is refused rather than stored.
_MAX_STEP_VERIFIER_LENGTH: Final[int] = 128

#: The metadata keys a committed step result exposes, and the values it must agree on.
_STEP_SCOPE: Final[str] = "step"
_METADATA_KEYS: Final[tuple[str, ...]] = (
    "contract_id",
    "goal_id",
    "observation_id",
    "scope",
    "state_complete",
)

_POLICY_MODES: Final[tuple[str, ...]] = ("abort", "bounded")


class RecoveryEngine:
    """The stateless recovery decision engine.

    One instance carries no decision state between calls, so a caller may keep or discard it
    freely. The one thing an instance does hold is the *declared* identity of the step verifier it
    depends on. That identity is generic configuration supplied by the host, never a whitelist that
    grows automatically and never a value read from a supplied verdict: the engine only checks that
    the step result it is handed was produced by the verifier the host said it composed.
    """

    def __init__(self, *, step_verifier_name: str = DEFAULT_STEP_VERIFIER) -> None:
        """Bind the engine to one host-declared step-verifier identity.

        The default keeps the original Core ``json_step`` contract byte-for-byte. The argument must
        be a built-in non-blank string of 1..128 characters; anything else is refused with the
        existing bounded ``POLICY_INVALID`` diagnostic, which carries neither the supplied value nor
        a reachable original exception chain.
        """
        if not _usable_verifier_name(step_verifier_name):
            raise RecoveryError("POLICY_INVALID", ("step_verifier_name",),
                                "the step verifier identity is not a usable configured name")
        self._step_verifier_name = step_verifier_name

    def assess(
        self,
        *,
        contract: object,
        context: RecoveryContext,
        safety: ActionSafetyDeclaration,
        evidence: RecoveryEvidence | None = None,
    ) -> RecoveryAssessment:
        """Return the bounded recovery assessment for one attempt.

        Every supplied input is validated first, in the fixed order, so a malformed request is
        never hidden by an earlier stop, budget or policy branch. ``evidence=None`` is the legal
        "no post-attempt evidence yet" request; damaged evidence is not.

        The configured verifier identity is re-checked on every call, so an instance whose
        configuration became unusable after construction refuses the request instead of silently
        falling back to a different label.
        """
        self._require_configured_identity()
        record = self._validated(contract, context, safety, evidence)
        policy = record["policy"]
        # A validated observation is reported by the assessment whatever branch decides, so the
        # caller can correlate the decision with the evidence it was made from. No identity is
        # generated: the identifier is the one the supplied observation already carried.
        observed = record.get("observation_id")

        # 2. The stop intent, the exhausted budget and an aborting policy stop immediately and
        #    need no further input, so they are decided before any observation is requested. They
        #    never demand extra observation, but they keep the association they were given.
        if record["stop_requested"]:
            return self._build(record, KIND_DECISION, DECISION_ABORT, STOP_REQUESTED, observed)
        if record["remaining_budget_ms"] <= 0:
            return self._build(record, KIND_DECISION, DECISION_ABORT, BUDGET_EXHAUSTED, observed)
        if policy["mode"] == "abort":
            return self._build(record, KIND_DECISION, DECISION_ABORT, POLICY_ABORT, observed)

        # 3. A bounded policy first asks for the post-attempt evidence of the current attempt.
        if evidence is None:
            return self._build(record, KIND_OBSERVATION_REQUIRED, None, OBSERVATION_REQUIRED,
                               None)

        # 4. A verified step is not a recovery decision. The execution status only separates the
        #    clean case from the anomaly diagnosis; it says nothing about the user's task.
        if record["step_status"] == "pass":
            reason = (STEP_VERIFIED if record["execution_status"] == "succeeded"
                      else STEP_VERIFIED_AFTER_EXECUTION_ANOMALY)
            return self._build(record, KIND_NO_RECOVERY, None, reason, record["observation_id"])

        # 5. An inconclusive step and an unresolved effect of a state-changing action abort. An
        #    ambiguous situation is never repackaged as "nothing happened".
        if record["step_status"] == "inconclusive":
            return self._build(record, KIND_DECISION, DECISION_ABORT, VERIFICATION_INCONCLUSIVE,
                               record["observation_id"])
        if record["effect_state"] == EFFECT_UNKNOWN and not record["safe_to_repeat"]:
            return self._build(record, KIND_DECISION, DECISION_ABORT, EFFECT_UNRESOLVED,
                               record["observation_id"])

        # 6. A retry needs every condition at once, and no unbounded path exists.
        if record["retry_allowed"]:
            return self._build(record, KIND_DECISION, DECISION_RETRY, RETRY_ALLOWED,
                               record["observation_id"])

        # 7. Otherwise a replan may apply when the current state is known and complete.
        if record["replan_allowed"]:
            return self._build(record, KIND_DECISION, DECISION_REPLAN, REPLAN_ALLOWED,
                               record["observation_id"])

        reason = (EFFECT_UNRESOLVED if record["effect_state"] == EFFECT_UNKNOWN
                  else RECOVERY_NOT_ALLOWED)
        return self._build(record, KIND_DECISION, DECISION_ABORT, reason,
                           record["observation_id"])

    # ------------------------------------------------------------------ validation

    def _require_configured_identity(self) -> None:
        """Re-check the host-declared verifier identity before every assessment.

        The configuration is copied at construction and never inferred again from a verdict, so
        the only way it can become unusable is external damage to this instance. That is refused
        with the same fixed policy diagnostic and without exposing the damaged value.
        """
        if not _usable_verifier_name(getattr(self, "_step_verifier_name", None)):
            raise RecoveryError("POLICY_INVALID", ("step_verifier_name",),
                                "the step verifier identity is not a usable configured name")

    def _validated(self, contract: object, context: RecoveryContext,
                   safety: ActionSafetyDeclaration,
                   evidence: RecoveryEvidence | None) -> dict:
        """Validate every supplied input and its associations, refusing a damaged request."""
        committed = _committed_contract(contract)
        context = _typed(context, RecoveryContext, "context")
        safety = _typed(safety, ActionSafetyDeclaration, "safety")
        if evidence is not None:
            evidence = _typed(evidence, RecoveryEvidence, "evidence")

        _require_same(context.contract_id, committed.contract_id, "context", "contract_id")
        _require_same(safety.contract_id, committed.contract_id, "safety", "contract_id")
        _require_same(safety.adapter_name, committed.adapter_name, "safety", "adapter_name")
        _require_same(safety.action_type, committed.action.action_type, "safety", "action_type")

        record = {
            "contract": committed,
            "context": context,
            "safety": safety,
            "policy": _committed_policy(committed),
            "policy_declared": _policy_fields(committed.recovery_policy),
            "stop_requested": bool(context.stop_requested),
            "remaining_budget_ms": float(context.remaining_budget_ms),
            "remaining_retries": int(context.remaining_retries),
            "remaining_replans": int(context.remaining_replans),
            "contract_retry_count": int(context.contract_retry_count),
            "execution_status": context.actual_outcome.status.value,
            "recoverable": _recoverable(context.actual_outcome),
        }
        if evidence is None:
            return record

        # The evidence must describe this context, this contract, this attempt and this outcome.
        for name in ("run_id", "step_id", "contract_id", "attempt_id"):
            _require_same(getattr(evidence, name), getattr(context, name), "evidence", name)
        _require_same(evidence.outcome_id, context.actual_outcome.outcome_id,
                      "evidence", "outcome_id")

        observation, state_complete = _observation(evidence, committed, record)
        record["observation_id"] = observation.observation_id
        record["state_complete"] = state_complete
        record["step_status"] = _step_status(
            evidence, committed, observation, state_complete, self._step_verifier_name)
        record["effect_state"] = evidence.effect_state
        record["preconditions_status"] = evidence.preconditions_status.value
        record["safe_to_repeat"] = safety.retry_safety in SAFE_TO_REPEAT

        retryable_failure = (record["execution_status"] in ("failed", "unknown")
                             and record["recoverable"] is True)
        record["retry_allowed"] = bool(
            record["policy"]["mode"] == "bounded"
            and record["policy"]["max_retries"] is not None
            and record["contract_retry_count"] < record["policy"]["max_retries"]
            and record["remaining_retries"] > 0
            and record["remaining_budget_ms"] > 0
            and record["preconditions_status"] == "pass"
            and record["state_complete"]
            and retryable_failure
            and (record["effect_state"] == EFFECT_NOT_APPLIED or record["safe_to_repeat"])
        )
        record["replan_allowed"] = bool(
            not record["retry_allowed"]
            and record["policy"]["allow_replan"] is True
            and record["remaining_replans"] > 0
            and record["remaining_budget_ms"] > 0
            and record["state_complete"]
            and record["effect_state"] in (EFFECT_NOT_APPLIED, EFFECT_APPLIED)
        )
        return record

    # ------------------------------------------------------------------ output

    def _build(self, record: dict, kind: str, decision: str | None, reason: str,
               observation_id: str | None) -> RecoveryAssessment:
        """Build one legal assessment for the recorded attempt, generating no identifier."""
        try:
            return RecoveryAssessment(
                schema_version="0.1",
                kind=kind,
                decision=decision,
                reason=reason,
                run_id=record["context"].run_id,
                step_id=record["context"].step_id,
                contract_id=record["context"].contract_id,
                attempt_id=record["context"].attempt_id,
                outcome_id=record["context"].actual_outcome.outcome_id,
                observation_id=observation_id,
            )
        except RecoveryError:
            raise
        except Exception as exc:  # an unbuildable verdict is a facility error
            raise RecoveryError("INPUT_INVALID", ("<assessment>",),
                                "the assessment for this request could not be constructed",
                                cause=exc) from exc


def _typed(value: object, expected: type, name: str) -> object:
    """Return a validated record of the expected type, or refuse the request."""
    if not isinstance(value, expected):
        raise RecoveryError("INPUT_INVALID", (name,),
                            "the supplied value is not the declared record")
    return expected.model_validate(value)


def _committed_contract(contract: object) -> object:
    """Return a validated committed contract, or refuse the request.

    The contract's own production-side shape is checked first, because a committed record does not
    re-derive its stored state when it is handed an instance: an ordinary copy that carries a wrong
    schema, an impossible budget or an undeclared stored field is refused here, before the stop,
    budget and policy branches and before anything is consumed or serialized.
    """
    from forecontract.contracts.models import ActionContract

    if not isinstance(contract, ActionContract):
        raise RecoveryError("INPUT_INVALID", ("contract",),
                            "a committed ActionContract is required")
    # The committed policy is checked first so a damaged policy is reported as a policy error
    # rather than as a damaged contract.
    policy = getattr(contract, "recovery_policy", None)
    if policy is not None:
        validate_committed_shape(policy, ("contract", "recovery_policy"), "POLICY_INVALID")
    validate_committed_shape(contract, ("contract",), "INPUT_INVALID")
    return contract


def _committed_policy(contract: object) -> dict:
    """Return the committed recovery policy as a plain mapping, or refuse it."""
    policy = getattr(contract, "recovery_policy", None)
    if policy is None:
        raise RecoveryError("POLICY_INVALID", ("recovery_policy",),
                            "the contract carries no recovery policy")
    return _policy_fields(policy)


def _policy_fields(policy: object) -> dict:
    """Read one committed policy, requiring an explicit bounded configuration."""
    mode = getattr(policy, "mode", None)
    if mode not in _POLICY_MODES:
        raise RecoveryError("POLICY_INVALID", ("recovery_policy", "mode"),
                            "the committed recovery mode is not recognised")
    if mode == "abort":
        return {"mode": "abort", "max_retries": None, "allow_replan": None}
    max_retries = getattr(policy, "max_retries", None)
    allow_replan = getattr(policy, "allow_replan", None)
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise RecoveryError("POLICY_INVALID", ("recovery_policy", "max_retries"),
                            "a bounded policy requires an explicit non-negative retry limit")
    if not isinstance(allow_replan, bool):
        raise RecoveryError("POLICY_INVALID", ("recovery_policy", "allow_replan"),
                            "a bounded policy requires an explicit replan declaration")
    return {"mode": "bounded", "max_retries": max_retries, "allow_replan": allow_replan}


def _recoverable(outcome: object) -> bool | None:
    """Read the explicit recoverability of a failed execution, or its honest absence."""
    error = getattr(outcome, "error", None)
    if error is None:
        return None
    return getattr(error, "recoverable", None)


def _require_same(value: object, expected: object, name: str, field: str) -> None:
    """Refuse an association that does not match the committed contract or the context."""
    if value != expected:
        raise RecoveryError("EVIDENCE_INVALID", (name, field),
                            "the supplied association does not match the current attempt")


def _observation(evidence: RecoveryEvidence, contract: object,
                 record: dict) -> tuple[object, bool]:
    """Return the validated observation and its completeness declaration."""
    from forecontract.verification import VerificationEvidence

    piece = evidence.verification_evidence
    if not isinstance(piece, VerificationEvidence):
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "verification_evidence"),
                            "a VerificationEvidence record is required")
    try:
        piece = VerificationEvidence.model_validate(piece)
    except RecoveryError:
        raise
    except Exception as exc:  # damaged evidence is a facility error
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "verification_evidence"),
                            "the verification evidence is not structurally valid",
                            cause=exc) from exc
    observation = piece.observation
    if not hasattr(observation, "observation_id") or not hasattr(observation, "state"):
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "verification_evidence",
                                                "observation"),
                            "the verification evidence carries no usable observation")
    if piece.state_complete is not True and piece.state_complete is not False:
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "verification_evidence",
                                                "state_complete"),
                            "the observation completeness must be declared explicitly")
    _ = contract
    _ = record
    return observation, bool(piece.state_complete)


def _step_status(evidence: RecoveryEvidence, contract: object, observation: object,
                 state_complete: bool, step_verifier_name: str) -> str:
    """Require the step result to be this contract's committed step verdict."""
    result = evidence.step_result
    if result.verifier != step_verifier_name:
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "step_result", "verifier"),
                            "the step result did not come from the configured step verifier")
    metadata = result.metadata
    if not _has_keys(metadata, _METADATA_KEYS):
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "step_result", "metadata"),
                            "the step result is missing a required association")
    if metadata["scope"] != _STEP_SCOPE:
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "step_result", "metadata", "scope"),
                            "a task-completion result cannot stand in for a step verdict")
    if metadata["contract_id"] != contract.contract_id:
        raise RecoveryError("EVIDENCE_INVALID",
                            ("evidence", "step_result", "metadata", "contract_id"),
                            "the step result belongs to another contract")
    if metadata["goal_id"] != contract.goal_id:
        raise RecoveryError("EVIDENCE_INVALID",
                            ("evidence", "step_result", "metadata", "goal_id"),
                            "the step result belongs to another goal")
    if metadata["observation_id"] != observation.observation_id:
        raise RecoveryError("EVIDENCE_INVALID",
                            ("evidence", "step_result", "metadata", "observation_id"),
                            "the step result was judged on another observation")
    if metadata["state_complete"] is not state_complete:
        raise RecoveryError("EVIDENCE_INVALID",
                            ("evidence", "step_result", "metadata", "state_complete"),
                            "the step result and the evidence disagree about completeness")
    expected = len(contract.expected_outcomes)
    results = result.outcome_results
    if len(results) != expected:
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "step_result", "outcome_results"),
                            "the step result does not cover every expected outcome")
    indices = [item.outcome_index for item in results]
    if sorted(indices) != list(range(expected)):
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "step_result", "outcome_results"),
                            "the step result does not match the committed outcome indices")
    status = result.status.value
    if status not in ("pass", "fail", "inconclusive"):
        raise RecoveryError("EVIDENCE_INVALID", ("evidence", "step_result", "status"),
                            "the step result carries no known verdict")
    _require_aggregate_consistency(result, contract, results)
    return status


def _require_aggregate_consistency(result: object, contract: object,
                                   results: list) -> None:
    """Require the overall verdict to be the required-all aggregate of the item results.

    The step verdict is consumed, never redefined: this checks that the supplied overall status
    agrees with the committed required flags and the per-item three-state results, so a
    contradictory record cannot be used for a no_recovery verdict or a business abort.
    """
    required = [index for index, outcome in enumerate(contract.expected_outcomes)
                if outcome.required]
    if not required:
        return
    verdicts = {item.outcome_index: item.status.value for item in results}
    required_results = [verdicts.get(index) for index in required]
    if any(verdict == "fail" for verdict in required_results):
        expected = "fail"
    elif any(verdict == "inconclusive" for verdict in required_results):
        expected = "inconclusive"
    else:
        expected = "pass"
    if result.status.value != expected:
        raise RecoveryError("EVIDENCE_INVALID",
                            ("evidence", "step_result", "status"),
                            "the step verdict contradicts the required outcome results")


def _usable_verifier_name(value: object) -> bool:
    """Whether a value is a built-in, non-blank, bounded verifier identity.

    Only the real built-in ``str`` type is accepted: a ``str`` subclass can override ``strip`` and
    answer a membership test however it likes, so its own comparison is never run. No coercion is
    attempted, and nothing about the supplied value is included in a refusal.
    """
    if type(value) is not str:
        return False
    if len(value) < 1 or len(value) > _MAX_STEP_VERIFIER_LENGTH:
        return False
    return bool(value.strip())


def _has_keys(mapping: object, keys: tuple[str, ...]) -> bool:
    """Whether a metadata mapping carries every required association key."""
    try:
        return all(key in mapping for key in keys)
    except TypeError:
        return False
