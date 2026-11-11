"""Acceptance tests for the deterministic recovery decision engine.

These tests own the fixed decision order, the closed output vocabulary and the association
checks. Every case drives a real committed :class:`ActionContract` and a real
:class:`StepVerifier` result, so the engine is exercised against the same objects a runtime
would hand it.
"""

from __future__ import annotations

import datetime
import itertools
import json
import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
for candidate in (CORE_SRC, Path(__file__).resolve().parents[2] / "packages" / "adapters" / "fake"
                  / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts import (
    ContractBuilder,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models.actions import (
    ActionSpec,
    AdapterCapabilities,
    ExpectedOutcome,
    Forecast,
)
from forecontract.models.observations import Observation
from forecontract.models.outcomes import (
    ActualOutcome,
    ErrorInfo,
    VerificationResult,
    VerificationStatus,
)
from forecontract.recovery import (
    ActionSafetyDeclaration,
    RecoveryContext,
    RecoveryEngine,
    RecoveryError,
    RecoveryEvidence,
)
from forecontract.verification import StepVerifier, VerificationEvidence

NOW = datetime.datetime(2026, 10, 10, 12, 0, tzinfo=datetime.timezone.utc)
MARKER = "SYNTHETIC_PRIVATE_VALUE"
PAYLOAD = {"schema_version": "0.1", "rule_type": "path_exists", "path": ["a"]}
ENGINE = RecoveryEngine()


class _Adapter:
    name = "fake-1"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(schema_version="0.1", adapter_name="fake-1",
                                   environment_type="fake", supported_actions=["click"],
                                   supported_condition_types=["json_rule"],
                                   supported_outcome_types=["json_rule"])

    def validate_action(self, candidate: ActionSpec) -> None:
        del candidate


class _Hooks:
    def validate_precondition(self, precondition: object) -> None:
        del precondition

    def validate_expected_outcome(self, outcome: object) -> None:
        del outcome

    def validate_verification_binding(self, binding: object, outcome: object) -> None:
        del binding, outcome

    def check_precondition(self, precondition: object, observed: Observation):
        del precondition, observed
        return VerificationStatus.PASS


def build(*, policy: RecoveryPolicy | None = None, contract_id: str = "c-1",
          observation_id: str = "obs-1", goal_id: str = "g-1",
          action_type: str = "click") -> object:
    outcome = ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", required=True,
                              payload=PAYLOAD)
    extra = {} if policy is None else {"recovery_policy": policy}
    return ContractBuilder().build(
        forecast=Forecast(schema_version="0.1", forecast_id="f-1", proposed_action=ActionSpec(
            schema_version="0.1", action_type=action_type, payload={}),
            expected_outcomes=[outcome]),
        observation=Observation(schema_version="0.1", observation_id=observation_id,
                                environment_type="fake", state={"a": 1}, timestamp=NOW),
        adapter=_Adapter(), hooks=_Hooks(), preconditions=[],
        verification_policy=VerificationPolicy(schema_version="0.1", bindings=[
            VerificationBinding(schema_version="0.1", outcome_index=0,
                                verifier_type="json_rule")]),
        timeout_ms=100, remaining_budget_ms=250, contract_id=contract_id, goal_id=goal_id,
        created_at=NOW, **extra,
    )


def bounded(max_retries: int = 2, allow_replan: bool = True) -> RecoveryPolicy:
    return RecoveryPolicy(schema_version="0.1", mode="bounded", max_retries=max_retries,
                          allow_replan=allow_replan)


def context(contract: object, *, status: str = "failed", recoverable: object = True,
            budget: float = 100.0, retries: int = 1, replans: int = 1, count: int = 0,
            stop: bool = False) -> RecoveryContext:
    error = None if status == "succeeded" else ErrorInfo(
        schema_version="0.1", code="E", message="m", recoverable=recoverable)
    return RecoveryContext(
        schema_version="0.1", run_id="r-1", step_id="s-1", contract_id=contract.contract_id,
        attempt_id="a-1",
        actual_outcome=ActualOutcome(schema_version="0.1", outcome_id="o-1", status=status,
                                     error=error),
        remaining_budget_ms=budget, remaining_retries=retries, remaining_replans=replans,
        contract_retry_count=count, stop_requested=stop)


def safety(contract: object, retry_safety: str = "idempotent") -> ActionSafetyDeclaration:
    return ActionSafetyDeclaration(schema_version="0.1", contract_id=contract.contract_id,
                                   adapter_name=contract.adapter_name,
                                   action_type=contract.action.action_type,
                                   retry_safety=retry_safety)


def evidence(contract: object, ctx: RecoveryContext, *, state: dict | None = None,
             complete: bool = True, effect: str = "not_applied",
             preconditions: VerificationStatus = VerificationStatus.PASS,
             observation_id: str = "obs-1") -> RecoveryEvidence:
    observed = Observation(schema_version="0.1", observation_id=observation_id,
                           environment_type="fake",
                           state={"a": 1} if state is None else state, timestamp=NOW)
    piece = VerificationEvidence(schema_version="0.1", observation=observed,
                                 state_complete=complete)
    return RecoveryEvidence(
        schema_version="0.1", run_id=ctx.run_id, step_id=ctx.step_id,
        contract_id=ctx.contract_id, attempt_id=ctx.attempt_id,
        outcome_id=ctx.actual_outcome.outcome_id, verification_evidence=piece,
        step_result=StepVerifier().verify_step(contract=contract, evidence=piece),
        effect_state=effect, preconditions_status=preconditions)


def assess(contract: object, ctx: RecoveryContext, *, retry_safety: str = "idempotent",
           piece: RecoveryEvidence | None = None):
    return ENGINE.assess(contract=contract, context=ctx, safety=safety(contract, retry_safety),
                         evidence=piece)


def forged(model: object, **mutation: object) -> object:
    """Build a record that carries a damaged state, bypassing the guarded copy.

    ``model_copy`` validates its update, so a forged record is built from the raw stored
    dictionary exactly as a validation-bypass construction would.
    """
    stored = dict(model.__dict__)
    stored.update(mutation)
    return type(model).model_construct(**stored)


def forged_record(model: object, **mutation: object) -> object:
    """The same validation-bypass construction, named for readability at the call site."""
    stored = dict(model.__dict__)
    stored.update(mutation)
    return type(model).model_construct(**stored)


# --- A03: statelessness, the observation request and determinism ---

def test_a_bounded_policy_first_requests_the_post_attempt_evidence() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    result = assess(contract, ctx)
    assert result.kind == "observation_required"
    assert result.decision is None
    assert result.reason == "OBSERVATION_REQUIRED"
    assert result.observation_id is None
    assert result.run_id == "r-1" and result.step_id == "s-1"
    assert result.contract_id == "c-1" and result.attempt_id == "a-1"
    assert result.outcome_id == "o-1"


def test_the_engine_is_stateless_and_deterministic() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx)
    first = assess(contract, ctx, piece=piece)
    second = assess(contract, ctx, piece=piece)
    assert first == second and first is not second
    assert RecoveryEngine().assess(contract=contract, context=ctx,
                                   safety=safety(contract), evidence=piece) == first


def test_the_engine_does_not_mutate_its_inputs_or_the_budget() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx)
    before_context = ctx.model_dump(mode="json")
    before_evidence = piece.model_dump(mode="json")
    policy_before = contract.recovery_policy.model_dump()
    assess(contract, ctx, piece=piece)
    assert ctx.model_dump(mode="json") == before_context
    assert piece.model_dump(mode="json") == before_evidence
    assert contract.recovery_policy.model_dump() == policy_before
    assert ctx.remaining_budget_ms == 100.0


def test_the_engine_takes_no_callbacks_and_performs_no_io() -> None:
    import inspect

    source = inspect.getsource(RecoveryEngine.assess)
    for forbidden in ("open(", "sleep", "time.", "datetime.now", "uuid", "print(", "input("):
        assert forbidden not in source


# --- A04: the fixed precedence ---

def test_an_aborting_policy_decides_without_asking_for_evidence() -> None:
    contract = build()
    result = assess(contract, context(contract))
    assert (result.kind, result.decision, result.reason) == ("decision", "abort", "POLICY_ABORT")
    assert result.observation_id is None


def test_a_stop_request_wins_over_everything_else() -> None:
    contract = build(policy=bounded())
    ctx = context(contract, stop=True, budget=0)
    result = assess(contract, ctx, piece=evidence(contract, ctx))
    assert result.reason == "STOP_REQUESTED"


def test_an_exhausted_budget_wins_over_the_policy_and_the_evidence() -> None:
    contract = build(policy=bounded())
    ctx = context(contract, budget=0)
    assert assess(contract, ctx).reason == "BUDGET_EXHAUSTED"
    assert assess(contract, ctx, piece=evidence(contract, ctx)).reason == "BUDGET_EXHAUSTED"
    aborting = build()
    assert assess(aborting, context(aborting, budget=0)).reason == "BUDGET_EXHAUSTED"
    assert assess(aborting, context(aborting, stop=True, budget=0)).reason == "STOP_REQUESTED"


def test_a_damaged_request_is_refused_before_any_earlier_branch() -> None:
    contract = build(policy=bounded())
    ctx = context(contract, stop=True)
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=contract, context=ctx, safety=safety(contract),
                      evidence=RecoveryEvidence.model_construct(
                          schema_version="0.1", run_id="other", step_id="s-1",
                          contract_id="c-1", attempt_id="a-1", outcome_id="o-1",
                          verification_evidence=None, step_result=None,
                          effect_state="not_applied",
                          preconditions_status=VerificationStatus.PASS))


# --- A05: a verified step is not task success ---

@pytest.mark.parametrize("status,reason", [
    ("succeeded", "STEP_VERIFIED"),
    ("failed", "STEP_VERIFIED_AFTER_EXECUTION_ANOMALY"),
    ("unknown", "STEP_VERIFIED_AFTER_EXECUTION_ANOMALY"),
])
def test_a_passing_step_is_no_recovery_and_keeps_the_anomaly_diagnosis(status: str,
                                                                       reason: str) -> None:
    contract = build(policy=bounded())
    ctx = context(contract, status=status)
    result = assess(contract, ctx, piece=evidence(contract, ctx))
    assert (result.kind, result.decision, result.reason) == ("no_recovery", None, reason)
    assert result.observation_id == "obs-1"


def test_a_passing_step_is_not_reported_as_task_success_anywhere() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    result = assess(contract, ctx, piece=evidence(contract, ctx))
    payload = result.model_dump(mode="json")
    assert sorted(payload) == ["attempt_id", "contract_id", "decision", "kind",
                               "observation_id", "outcome_id", "reason", "run_id",
                               "schema_version", "step_id"]
    assert "success" not in json.dumps(payload)
    assert "task" not in json.dumps(payload)


# --- A06: a failure does not prove an absent effect ---

def test_an_inconclusive_step_aborts_and_is_never_masked() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    for effect in ("not_applied", "applied", "unknown"):
        piece = evidence(contract, ctx, state={"b": 2}, complete=False, effect=effect,
                         preconditions=VerificationStatus.INCONCLUSIVE)
        result = assess(contract, ctx, piece=piece)
        assert result.reason == "VERIFICATION_INCONCLUSIVE"
        assert result.decision == "abort"


def test_an_unresolved_effect_of_a_state_changing_action_aborts() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx, state={"b": 2}, effect="unknown",
                     preconditions=VerificationStatus.FAIL)
    for retry_safety in ("non_idempotent", "unknown"):
        result = assess(contract, ctx, retry_safety=retry_safety, piece=piece)
        assert (result.kind, result.decision, result.reason) == (
            "decision", "abort", "EFFECT_UNRESOLVED")


@pytest.mark.parametrize("recoverable,allowed", [(True, True), (False, False), (None, False)])
def test_only_an_explicit_recoverable_failure_may_retry(recoverable: object,
                                                        allowed: bool) -> None:
    contract = build(policy=bounded())
    ctx = context(contract, recoverable=recoverable)
    piece = evidence(contract, ctx, state={"b": 2})
    result = assess(contract, ctx, piece=piece)
    assert (result.reason == "RETRY_ALLOWED") is allowed


def test_an_explicitly_safe_action_may_repeat_an_unresolved_effect() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx, state={"b": 2}, effect="unknown",
                     preconditions=VerificationStatus.PASS)
    for retry_safety in ("read_only", "idempotent"):
        assert assess(contract, ctx, retry_safety=retry_safety,
                      piece=piece).reason == "RETRY_ALLOWED"


def test_a_succeeded_execution_with_a_failing_step_does_not_retry_on_its_own() -> None:
    contract = build(policy=bounded())
    ctx = context(contract, status="succeeded")
    piece = evidence(contract, ctx, state={"b": 2}, preconditions=VerificationStatus.PASS)
    result = assess(contract, ctx, piece=piece)
    assert result.reason != "RETRY_ALLOWED"


# --- A07: the parameterised oracle ---

def test_the_retry_matrix_covers_safety_effect_and_execution() -> None:
    """A parameterised fixture oracle over safety x effect x execution status."""
    contract = build(policy=bounded())
    checked = 0
    for retry_safety, effect, status, recoverable in itertools.product(
        ("read_only", "idempotent", "non_idempotent", "unknown"),
        ("not_applied", "applied", "unknown"),
        ("failed", "unknown", "succeeded"),
        (True, False),
    ):
        ctx = context(contract, status=status, recoverable=recoverable)
        piece = evidence(contract, ctx, state={"b": 2}, effect=effect,
                         preconditions=VerificationStatus.PASS)
        result = assess(contract, ctx, retry_safety=retry_safety, piece=piece)
        expected = (status in ("failed", "unknown") and recoverable is True
                    and (effect == "not_applied" or retry_safety in ("read_only", "idempotent")))
        assert (result.reason == "RETRY_ALLOWED") is expected, (
            retry_safety, effect, status, recoverable, result.reason)
        checked += 1
    assert checked == 72


def test_a_replan_falls_back_when_a_retry_is_not_available() -> None:
    contract = build(policy=bounded())
    ctx = context(contract, recoverable=False)
    piece = evidence(contract, ctx, state={"b": 2}, effect="applied",
                     preconditions=VerificationStatus.FAIL)
    assert assess(contract, ctx, piece=piece).reason == "REPLAN_ALLOWED"
    disabled = build(policy=bounded(allow_replan=False))
    ctx = context(disabled, recoverable=False)
    piece = evidence(disabled, ctx, state={"b": 2}, effect="applied",
                     preconditions=VerificationStatus.FAIL)
    assert assess(disabled, ctx, piece=piece).reason == "RECOVERY_NOT_ALLOWED"


def test_an_incomplete_state_never_authorises_a_retry() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx, state={"b": 2}, complete=False,
                     preconditions=VerificationStatus.PASS)
    assert assess(contract, ctx, piece=piece).reason != "RETRY_ALLOWED"


def test_a_failing_current_precondition_never_authorises_a_retry() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx, state={"b": 2}, effect="not_applied",
                     preconditions=VerificationStatus.FAIL)
    assert assess(contract, ctx, piece=piece).reason != "RETRY_ALLOWED"


# --- A08: associations ---

def test_a_foreign_contract_or_action_is_refused() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=contract, context=ctx, safety=ActionSafetyDeclaration(
            schema_version="0.1", contract_id="other", adapter_name="fake-1",
            action_type="click", retry_safety="idempotent"))
    assert info.value.code == "EVIDENCE_INVALID"
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=contract, context=ctx, safety=ActionSafetyDeclaration(
            schema_version="0.1", contract_id="c-1", adapter_name="other",
            action_type="click", retry_safety="idempotent"))
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=contract, context=ctx, safety=ActionSafetyDeclaration(
            schema_version="0.1", contract_id="c-1", adapter_name="fake-1",
            action_type="other", retry_safety="idempotent"))


def test_a_context_from_another_contract_is_refused() -> None:
    contract = build(policy=bounded())
    other = context(build(policy=bounded(), contract_id="c-2", observation_id="obs-2"))
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=contract, context=other, safety=safety(contract))


@pytest.mark.parametrize("field,value", [
    ("run_id", "other"), ("step_id", "other"), ("contract_id", "other"),
    ("attempt_id", "other"), ("outcome_id", "other"),
])
def test_evidence_must_describe_this_attempt(field: str, value: str) -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx)
    damaged = forged(piece, **{field: value})
    with pytest.raises(RecoveryError) as info:
        assess(contract, ctx, piece=damaged)
    assert info.value.code == "EVIDENCE_INVALID"


def test_a_task_completion_result_cannot_stand_in_for_a_step_verdict() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx)
    for mutation in ({"verifier": "json_task"}, {"metadata": {**piece.step_result.metadata,
                                                             "scope": "task"}},
                     {"metadata": {**piece.step_result.metadata, "contract_id": "other"}},
                     {"metadata": {**piece.step_result.metadata, "goal_id": "other"}},
                     {"metadata": {**piece.step_result.metadata, "observation_id": "other"}},
                     {"metadata": {**piece.step_result.metadata, "state_complete": False}}):
        forged = VerificationResult.model_construct(
            **{**dict(piece.step_result.__dict__), **mutation})
        with pytest.raises(RecoveryError) as info:
            assess(contract, ctx, piece=piece.model_copy(update={"step_result": forged}))
        assert info.value.code == "EVIDENCE_INVALID"


def test_a_step_result_must_cover_every_committed_outcome() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx)
    stored = dict(piece.step_result.__dict__)
    stored["outcome_results"] = []
    forged = VerificationResult.model_construct(**stored)
    with pytest.raises(RecoveryError):
        assess(contract, ctx, piece=piece.model_copy(update={"step_result": forged}))


def test_a_foreign_or_missing_metadata_key_is_refused() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx)
    for metadata in ({}, {"scope": "step"}, "not-a-mapping"):
        stored = dict(piece.step_result.__dict__)
        stored["metadata"] = metadata
        forged = VerificationResult.model_construct(**stored)
        with pytest.raises(RecoveryError):
            assess(contract, ctx, piece=piece.model_copy(update={"step_result": forged}))


# --- A09: bounded budgets and counters ---

@pytest.mark.parametrize("count,retries,max_retries,allowed", [
    (0, 1, 2, True),
    (1, 1, 2, True),
    (2, 1, 2, False),
    (3, 5, 2, False),
    (0, 0, 2, False),
    (0, 1, 0, False),
    (0, 5, 0, False),
])
def test_the_retry_limit_is_the_joint_bound_of_policy_counter_and_budget(
    count: int, retries: int, max_retries: int, allowed: bool,
) -> None:
    contract = build(policy=bounded(max_retries=max_retries))
    ctx = context(contract, count=count, retries=retries)
    piece = evidence(contract, ctx, state={"b": 2})
    assert (assess(contract, ctx, piece=piece).reason == "RETRY_ALLOWED") is allowed


@pytest.mark.parametrize("replans,allow_replan,budget,expected", [
    (1, True, 100.0, "REPLAN_ALLOWED"),
    (0, True, 100.0, "RECOVERY_NOT_ALLOWED"),
    (1, False, 100.0, "RECOVERY_NOT_ALLOWED"),
])
def test_the_replan_limit_is_the_joint_bound_of_policy_counter_and_budget(
    replans: int, allow_replan: bool, budget: float, expected: str,
) -> None:
    contract = build(policy=bounded(allow_replan=allow_replan))
    ctx = context(contract, recoverable=False, replans=replans, budget=budget)
    piece = evidence(contract, ctx, state={"b": 2}, effect="applied",
                     preconditions=VerificationStatus.FAIL)
    assert assess(contract, ctx, piece=piece).reason == expected


@pytest.mark.parametrize("budget", [0, 0.0, -0.0])
def test_a_zero_budget_stops_immediately(budget: float) -> None:
    contract = build(policy=bounded())
    ctx = context(contract, budget=budget)
    assert assess(contract, ctx).reason == "BUDGET_EXHAUSTED"


def test_a_large_finite_budget_is_accepted_and_never_reduced() -> None:
    contract = build(policy=bounded())
    ctx = context(contract, budget=1e15)
    piece = evidence(contract, ctx, state={"b": 2})
    assert assess(contract, ctx, piece=piece).reason == "RETRY_ALLOWED"
    assert ctx.remaining_budget_ms == 1e15


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), float("-inf"), -1, True])
def test_a_non_finite_or_negative_budget_is_refused(budget: object) -> None:
    contract = build(policy=bounded())
    with pytest.raises(RecoveryError):
        context(contract, budget=budget)


# --- A10: bounded errors and a closed output ---

def test_every_facility_error_is_bounded_and_never_echoes_a_supplied_value() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    for call in (
        lambda: ENGINE.assess(contract={"a": 1}, context=ctx, safety=safety(contract)),
        lambda: ENGINE.assess(contract=contract, context={"run_id": MARKER},
                              safety=safety(contract)),
        lambda: ENGINE.assess(contract=contract, context=ctx,
                              safety={"contract_id": MARKER}),
    ):
        with pytest.raises(RecoveryError) as info:
            call()
        assert info.value.code in ("INPUT_INVALID", "POLICY_INVALID", "EVIDENCE_INVALID")
        assert len(str(info.value)) < 240
        assert MARKER not in str(info.value)
        assert "0x" not in str(info.value)


def test_a_damaged_committed_policy_is_a_policy_error() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    damaged = RecoveryPolicy.model_construct(schema_version="0.1", mode="bounded",
                                             max_retries=None, allow_replan=True)
    forged = type(contract).model_construct(
        **{**dict(contract.__dict__), "recovery_policy": damaged})
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=forged, context=ctx, safety=safety(contract))
    assert info.value.code in ("INPUT_INVALID", "POLICY_INVALID")


def test_the_output_carries_no_free_payload() -> None:
    contract = build(policy=bounded())
    ctx = context(contract)
    result = assess(contract, ctx, piece=evidence(contract, ctx, state={"b": 2}))
    payload = result.model_dump(mode="json")
    assert set(payload) == {"schema_version", "kind", "decision", "reason", "run_id", "step_id",
                            "contract_id", "attempt_id", "outcome_id", "observation_id"}
    assert payload["schema_version"] == "0.1"
    assert all(isinstance(value, (str, type(None))) for value in payload.values())
    # No returned value carries a state, an error payload or a free-text message.
    assert MARKER not in json.dumps(payload)


def test_an_invalid_output_combination_can_never_be_returned() -> None:
    """Every assessment the engine returns satisfies the closed combination rule."""
    seen = set()
    for stop, budget, policy, piece_kind, retry_safety, effect, status, recoverable in (
        (True, 100.0, bounded(), None, "idempotent", "not_applied", "failed", True),
        (False, 0, bounded(), None, "idempotent", "not_applied", "failed", True),
        (False, 100.0, None, None, "idempotent", "not_applied", "failed", True),
        (False, 100.0, bounded(), "none", "idempotent", "not_applied", "failed", True),
        (False, 100.0, bounded(), "pass", "idempotent", "not_applied", "succeeded", True),
        (False, 100.0, bounded(), "pass", "idempotent", "not_applied", "failed", True),
        (False, 100.0, bounded(), "pass", "idempotent", "not_applied", "unknown", True),
        (False, 100.0, bounded(), "fail", "non_idempotent", "unknown", "failed", True),
        (False, 100.0, bounded(), "retry", "idempotent", "not_applied", "failed", True),
        (False, 100.0, bounded(), "replan", "non_idempotent", "applied", "failed", False),
        (False, 100.0, bounded(), "abort", "non_idempotent", "applied", "failed", False),
    ):
        chosen = build(policy=policy) if policy is not None else build()
        ctx = context(chosen, budget=budget, stop=stop, status=status, recoverable=recoverable)
        piece = None
        if piece_kind is not None:
            state = {"b": 2} if piece_kind in ("fail", "retry", "replan", "abort") else None
            complete = piece_kind != "inconclusive"
            preconditions = (VerificationStatus.INCONCLUSIVE if piece_kind == "abort"
                             else VerificationStatus.PASS)
            piece = evidence(chosen, ctx, state=state, complete=complete, effect=effect,
                             preconditions=preconditions)
        result = assess(chosen, ctx, retry_safety=retry_safety, piece=piece)
        # The record itself enforces the closed combination, so a returned value cannot break it.
        assert result.kind in ("observation_required", "no_recovery", "decision")
        assert result.reason in (
            "STOP_REQUESTED", "BUDGET_EXHAUSTED", "POLICY_ABORT", "OBSERVATION_REQUIRED",
            "STEP_VERIFIED", "STEP_VERIFIED_AFTER_EXECUTION_ANOMALY",
            "VERIFICATION_INCONCLUSIVE", "EFFECT_UNRESOLVED", "RETRY_ALLOWED",
            "REPLAN_ALLOWED", "RECOVERY_NOT_ALLOWED")
        if result.kind == "observation_required":
            assert result.decision is None and result.observation_id is None
        elif result.kind == "no_recovery":
            assert result.decision is None and result.observation_id is not None
        else:
            assert result.decision in ("retry", "replan", "abort")
        seen.add(result.reason)
    assert len(seen) >= 6


_ = (ActualOutcome, json, RecoveryContext)


# --- round-002 corrections: F01 committed shape, F05 aggregate, F06 association, F07/F08 ---

@pytest.mark.parametrize("field,value", [
    ("schema_version", "9.9"),
    ("timeout_ms", -1),
    ("timeout_ms", 0),
    ("contract_id", 19),
    ("adapter_name", b"fake-1"),
])
def test_a_damaged_committed_contract_is_refused(field: str, value: object) -> None:
    """F01: a committed copy that no longer matches its own schema is refused at the boundary."""
    contract = build(policy=bounded())
    stored = dict(contract.__dict__)
    stored[field] = value
    damaged = type(contract).model_construct(**stored)
    if field == "timeout_ms" and not isinstance(value, bool) and float(value) > 0:
        raise AssertionError("the case must be an invalid value")
    with pytest.raises(RecoveryError) as info:
        assess(damaged, context(damaged))
    assert info.value.code == "INPUT_INVALID"


def test_a_contract_with_an_undeclared_stored_field_is_refused() -> None:
    contract = build(policy=bounded())
    damaged = type(contract).model_construct(**dict(contract.__dict__))
    damaged.__dict__[MARKER] = 1
    with pytest.raises(RecoveryError) as info:
        assess(damaged, context(damaged))
    assert MARKER not in str(info.value)
    assert MARKER not in str(info.value.location)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "9.9"),
    ("max_retries", True),
    ("allow_replan", 1),
])
def test_a_damaged_committed_policy_is_refused(field: str, value: object) -> None:
    """F01: the committed policy is inspected in its original shape, not merely projected."""
    from forecontract.contracts import RecoveryPolicy

    contract = build(policy=bounded())
    stored = dict(contract.recovery_policy.__dict__)
    stored[field] = value
    damaged_policy = RecoveryPolicy.model_construct(**stored)
    damaged = type(contract).model_construct(
        **{**dict(contract.__dict__), "recovery_policy": damaged_policy})
    with pytest.raises(RecoveryError) as info:
        assess(damaged, context(damaged))
    assert info.code if False else info.value.code in ("INPUT_INVALID", "POLICY_INVALID")


def test_a_policy_with_an_undeclared_stored_field_is_refused() -> None:
    from forecontract.contracts import RecoveryPolicy

    contract = build(policy=bounded())
    damaged_policy = RecoveryPolicy.model_construct(**dict(contract.recovery_policy.__dict__))
    damaged_policy.__dict__[MARKER] = 1
    damaged = type(contract).model_construct(
        **{**dict(contract.__dict__), "recovery_policy": damaged_policy})
    with pytest.raises(RecoveryError) as info:
        assess(damaged, context(damaged))
    assert info.value.code == "POLICY_INVALID"
    assert MARKER not in str(info.value)


def test_a_damaged_contract_is_refused_before_any_earlier_branch() -> None:
    """F01: the boundary check precedes the stop, budget and policy short circuits."""
    contract = build(policy=bounded())
    stored = dict(contract.__dict__)
    stored["schema_version"] = "9.9"
    damaged = type(contract).model_construct(**stored)
    for ctx in (context(damaged, stop=True), context(damaged, budget=0),
                context(damaged)):
        with pytest.raises(RecoveryError):
            assess(damaged, ctx)


@pytest.mark.parametrize("aggregate", ["pass", "inconclusive"])
def test_a_contradictory_step_aggregate_is_refused(aggregate: str) -> None:
    """F05: a required FAIL cannot be reported as an overall PASS or INCONCLUSIVE."""
    contract = build(policy=bounded())
    ctx = context(contract)
    piece = evidence(contract, ctx, state={"b": 2})
    stored = dict(piece.step_result.__dict__)
    stored["status"] = VerificationStatus(aggregate)
    forged = VerificationResult.model_construct(**stored)
    with pytest.raises(RecoveryError) as info:
        assess(contract, ctx, piece=forged_record(piece, step_result=forged))
    assert info.value.code == "EVIDENCE_INVALID"


def test_a_consistent_step_aggregate_is_still_accepted() -> None:
    """F05 control: the real verdicts the step verifier produces are accepted unchanged."""
    contract = build(policy=bounded())
    for state, expected in (({"b": 2}, "fail"), ({"a": 1}, "pass")):
        ctx = context(contract, status="succeeded" if expected == "pass" else "failed")
        piece = evidence(contract, ctx, state=state)
        assert piece.step_result.status.value == expected
        result = assess(contract, ctx, piece=piece)
        assert result.reason in ("STEP_VERIFIED", "STEP_VERIFIED_AFTER_EXECUTION_ANOMALY",
                                 "RETRY_ALLOWED", "REPLAN_ALLOWED", "RECOVERY_NOT_ALLOWED",
                                 "EFFECT_UNRESOLVED", "VERIFICATION_INCONCLUSIVE")


@pytest.mark.parametrize("why", ["stop", "budget", "abort_policy"])
def test_a_terminal_branch_still_reports_the_supplied_observation(why: str) -> None:
    """F06: an immediate stop keeps the association of the evidence it was given."""
    contract = build() if why == "abort_policy" else build(policy=bounded())
    ctx = context(contract, stop=why == "stop", budget=0 if why == "budget" else 100)
    piece = evidence(contract, ctx, state={}, observation_id="obs-after")
    result = assess(contract, ctx, piece=piece)
    assert result.observation_id == "obs-after"
    assert result.reason in ("STOP_REQUESTED", "BUDGET_EXHAUSTED", "POLICY_ABORT")


@pytest.mark.parametrize("why", ["stop", "budget", "abort_policy"])
def test_a_terminal_branch_without_evidence_reports_no_observation(why: str) -> None:
    """F06 control: no evidence supplied means a null association, never an invented one."""
    contract = build() if why == "abort_policy" else build(policy=bounded())
    ctx = context(contract, stop=why == "stop", budget=0 if why == "budget" else 100)
    result = assess(contract, ctx)
    assert result.observation_id is None


@pytest.mark.parametrize("reason,decision", [
    ("RETRY_ALLOWED", "retry"),
    ("REPLAN_ALLOWED", "replan"),
    ("VERIFICATION_INCONCLUSIVE", "abort"),
    ("EFFECT_UNRESOLVED", "abort"),
    ("RECOVERY_NOT_ALLOWED", "abort"),
    ("STEP_VERIFIED", None),
    ("STEP_VERIFIED_AFTER_EXECUTION_ANOMALY", None),
])
def test_every_evidence_dependent_verdict_requires_an_observation(reason: str,
                                                                 decision: str | None) -> None:
    """F06: a verdict that depends on the evidence must report the observation it used."""
    from forecontract.recovery import RecoveryAssessment

    kind = "no_recovery" if decision is None else "decision"
    with pytest.raises(RecoveryError):
        RecoveryAssessment(schema_version="0.1", kind=kind, decision=decision, reason=reason,
                           run_id="r-1", step_id="s-1", contract_id="c-1", attempt_id="a-1",
                           outcome_id="o-1", observation_id=None)
    assert RecoveryAssessment(schema_version="0.1", kind=kind, decision=decision, reason=reason,
                              run_id="r-1", step_id="s-1", contract_id="c-1", attempt_id="a-1",
                              outcome_id="o-1", observation_id="obs-1").observation_id == "obs-1"


@pytest.mark.parametrize("route", ["constructor", "mapping", "instance", "component"])
def test_a_model_cycle_is_a_bounded_error_on_every_route(route: str) -> None:
    """F07: a record that reaches itself through its own field is refused, not a recursion crash."""
    contract = build(policy=bounded())
    ctx = context(contract)
    damaged_outcome = ActualOutcome.model_construct(**dict(ctx.actual_outcome.__dict__))
    damaged_outcome.__dict__["changes"] = {"cycle": damaged_outcome}
    data = {**dict(ctx.__dict__), "actual_outcome": damaged_outcome}
    with pytest.raises(RecoveryError) as info:
        if route == "constructor":
            RecoveryContext(**data)
        elif route == "mapping":
            RecoveryContext.model_validate(data)
        elif route == "instance":
            RecoveryContext.model_validate(RecoveryContext.model_construct(**data))
        else:
            assess(contract, RecoveryContext.model_construct(**data))
    assert len(str(info.value)) < 240
    assert MARKER not in str(info.value.location)


def test_a_container_cycle_is_a_bounded_error_and_shared_references_are_still_legal() -> None:
    """F07 control: the same bound holds for a raw container, and legal sharing still passes."""
    contract = build(policy=bounded())
    ctx = context(contract)
    loop: dict = {}
    loop[MARKER] = loop
    damaged_outcome = ActualOutcome.model_construct(**{**dict(ctx.actual_outcome.__dict__),
                                                       "changes": loop})
    with pytest.raises(RecoveryError):
        RecoveryContext(**{**dict(ctx.__dict__), "actual_outcome": damaged_outcome})
    shared = [1]
    legal = ActualOutcome.model_construct(**{**dict(ctx.actual_outcome.__dict__),
                                             "changes": {"a": shared, "b": shared}})
    rebuilt = RecoveryContext(**{**dict(ctx.__dict__), "actual_outcome": legal})
    assert rebuilt.actual_outcome.changes == {"a": [1], "b": [1]}


def test_an_error_location_never_echoes_a_caller_key() -> None:
    """F08: a diagnostic path reports a fixed placeholder instead of a supplied mapping key."""
    contract = build(policy=bounded())
    ctx = context(contract)
    loop: dict = {}
    loop[MARKER] = loop
    damaged = ActualOutcome.model_construct(**{**dict(ctx.actual_outcome.__dict__),
                                               "changes": loop})
    with pytest.raises(RecoveryError) as info:
        RecoveryContext(**{**dict(ctx.__dict__), "actual_outcome": damaged})
    text = str(info.value)
    location = str(info.value.location)
    assert MARKER not in text
    assert MARKER not in location
    assert len(text) < 240
    assert info.value.code == "INPUT_INVALID"
    assert "<key>" in location
    # An ordinary declaration error keeps its usable structural location.
    with pytest.raises(RecoveryError) as other:
        RecoveryContext(**{**dict(context(contract).__dict__), "run_id": 17})
    assert other.value.location in (("run_id",), ("<record>",))


# --- round-003 corrections: F01 full committed shape, F07 arbitrary-length cycles ---

def _linked_outcomes(size: int) -> ActualOutcome:
    """A chain of ``size`` attempt outcomes whose last one points back at the first."""
    contract = build(policy=bounded())
    ctx = context(contract)
    nodes = [ActualOutcome.model_construct(**dict(ctx.actual_outcome.__dict__))
             for _ in range(size)]
    for index, node in enumerate(nodes):
        node.__dict__["changes"] = {"next": nodes[(index + 1) % size]}
    return nodes[0]


@pytest.mark.parametrize("size", [2, 3, 5, 8])
@pytest.mark.parametrize("route", ["constructor", "mapping", "instance", "component"])
def test_an_arbitrary_length_model_cycle_is_bounded_on_every_route(size: int,
                                                                  route: str) -> None:
    """F07: a cycle of any length is refused with one bounded error, never a recursion failure."""
    contract = build(policy=bounded())
    data = {**dict(context(contract).__dict__), "actual_outcome": _linked_outcomes(size)}
    with pytest.raises(RecoveryError) as info:
        if route == "constructor":
            RecoveryContext(**data)
        elif route == "mapping":
            RecoveryContext.model_validate(data)
        elif route == "instance":
            RecoveryContext.model_validate(RecoveryContext.model_construct(**data))
        else:
            assess(contract, RecoveryContext.model_construct(**data))
    assert "maximum recursion depth" not in str(info.value)
    assert len(str(info.value)) < 240


@pytest.mark.parametrize("size", [2, 3, 5])
def test_a_mixed_model_and_container_cycle_is_bounded(size: int) -> None:
    """F07: the bound also holds when a cycle alternates records and raw containers."""
    first = ActualOutcome.model_construct(**dict(_linked_outcomes(2).__dict__))
    second = ActualOutcome.model_construct(**dict(_linked_outcomes(2).__dict__))
    first.__dict__["changes"] = {"a": [{"b": second}]}
    second.__dict__["changes"] = {"c": [first]}
    contract = build(policy=bounded())
    with pytest.raises(RecoveryError) as info:
        RecoveryContext(**{**dict(context(contract).__dict__), "actual_outcome": first})
    assert info.value.code == "INPUT_INVALID"
    assert size


def test_the_committed_contract_is_fully_validated_at_the_engine_boundary() -> None:
    """F01: every malformed committed input is a bounded facility error, not a crash."""
    import datetime

    from forecontract.contracts import RecoveryPolicy

    contract = build(policy=bounded())
    ctx = context(contract)

    def damaged(**mutation):
        stored = dict(contract.__dict__)
        stored.update(mutation)
        return type(contract).model_construct(**stored)

    cases = {
        "blank identifier": damaged(contract_id="   "),
        "naive timestamp": damaged(created_at=datetime.datetime(2026, 1, 1)),  # noqa: DTZ001
        "wrong nested model": damaged(action=contract.recovery_policy),
        "no required outcome": damaged(expected_outcomes=[]),
        "policy is not a model": damaged(recovery_policy={}),
        "overflowing budget": damaged(timeout_ms=10 ** 400),
    }
    # The declaration is built from the legal contract, because a damaged contract cannot supply
    # the attributes the declaration reads; the engine must refuse the contract regardless.
    valid_safety = safety(contract)
    for label, broken in cases.items():
        with pytest.raises(RecoveryError) as info:
            ENGINE.assess(contract=broken, context=ctx, safety=valid_safety)
        # A damaged contract is refused with a bounded facility error; which of the three codes
        # reports it depends on the first inconsistency the boundary reaches.
        assert info.value.code in ("INPUT_INVALID", "POLICY_INVALID", "EVIDENCE_INVALID"), label
        assert len(str(info.value)) < 240, label
        assert "maximum recursion depth" not in str(info.value), label
    # A legal positive finite float stays legal through the committed shape.
    stored = dict(contract.__dict__)
    stored["timeout_ms"] = 1.5
    assert ENGINE.assess(contract=type(contract).model_construct(**stored), context=ctx,
                         safety=valid_safety).reason == "OBSERVATION_REQUIRED"
    # A committed abort policy must carry no stored bounded control.
    for field, value in (("max_retries", 1), ("allow_replan", True)):
        stored_policy = dict(RecoveryPolicy(schema_version="0.1", mode="abort").__dict__)
        stored_policy[field] = value
        broken_policy = RecoveryPolicy.model_construct(**stored_policy)
        with pytest.raises(RecoveryError) as info:
            ENGINE.assess(contract=damaged(recovery_policy=broken_policy), context=ctx,
                          safety=valid_safety)
        assert info.value.code == "POLICY_INVALID", field


# --- round-004 correction: the whole original declared field shape, not a sample ---

def _deep_damaged_contract(field: str, value: object):
    """A committed contract whose one declared field holds a value its own schema refuses."""
    contract = build(policy=bounded())
    stored = dict(contract.__dict__)
    stored[field] = value
    return type(contract).model_construct(**stored)


def _deep_damaged_nested(container: str, field: str, value: object):
    """A committed contract whose one nested committed record holds a refused value."""
    contract = build(policy=bounded())
    if container == "expected_outcomes":
        original = contract.expected_outcomes[0]
    else:
        original = contract.verification_policy.bindings[0]
    stored_record = dict(original.__dict__)
    stored_record[field] = value
    record = type(original).model_construct(**stored_record)
    stored = dict(contract.__dict__)
    if container == "expected_outcomes":
        stored["expected_outcomes"] = [record]
    else:
        stored_policy = dict(contract.verification_policy.__dict__)
        stored_policy["bindings"] = [record]
        stored["verification_policy"] = type(contract.verification_policy).model_construct(
            **stored_policy)
    return type(contract).model_construct(**stored)


def _deep_damaged_action(field: str, value: object):
    """A committed contract whose frozen action holds a value its own schema refuses."""
    contract = build(policy=bounded())
    stored_action = dict(contract.action.__dict__)
    stored_action[field] = value
    stored = dict(contract.__dict__)
    stored["action"] = type(contract.action).model_construct(**stored_action)
    return type(contract).model_construct(**stored)


@pytest.mark.parametrize("field,value,label", [
    ("environment_type", "   ", "blank environment type"),
    ("preconditions", "", "preconditions is not a sequence"),
    ("metadata", [], "metadata is a list"),
    ("metadata", {"x": float("nan")}, "metadata holds NaN"),
    ("metadata", {1: "x"}, "metadata has an integer key"),
    ("metadata", {"x": datetime.datetime.now(datetime.timezone.utc)}, "metadata holds a datetime"),
])
def test_a_contract_field_that_breaks_its_declared_shape_is_refused(field: str, value: object,
                                                                   label: str) -> None:
    """F01: every declared contract field is checked, not only the listed examples."""
    contract = build(policy=bounded())
    ctx = context(contract)
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=_deep_damaged_contract(field, value), context=ctx,
                      safety=safety(contract))
    assert len(str(info.value)) < 240, label
    assert "maximum recursion depth" not in str(info.value), label


@pytest.mark.parametrize("field,value,label", [
    ("outcome_index", 0.0, "a float index"),
    ("verifier_type", "   ", "a blank verifier type"),
    ("config", [], "a list config"),
    ("config", {"x": float("inf")}, "an infinite config value"),
])
def test_a_binding_field_that_breaks_its_declared_shape_is_refused(field: str, value: object,
                                                                  label: str) -> None:
    """F01: an outcome index is a strict integer and a binding config is a finite JSON object."""
    contract = build(policy=bounded())
    ctx = context(contract)
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=_deep_damaged_nested("verification_policy", field, value),
                      context=ctx, safety=safety(contract))
    assert len(str(info.value)) < 240, label


@pytest.mark.parametrize("field,value,label", [
    ("outcome_type", "   ", "a blank outcome type"),
    ("payload", [], "a list payload"),
    ("payload", {"x": float("nan")}, "a NaN payload value"),
    ("payload", {1: "x"}, "an integer payload key"),
])
def test_an_expected_outcome_field_that_breaks_its_declared_shape_is_refused(
        field: str, value: object, label: str) -> None:
    """F01: an expected outcome payload is a finite JSON object with string keys."""
    contract = build(policy=bounded())
    ctx = context(contract)
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=_deep_damaged_nested("expected_outcomes", field, value),
                      context=ctx, safety=safety(contract))
    assert len(str(info.value)) < 240, label


@pytest.mark.parametrize("field,value,label", [
    ("payload", [], "a list action payload"),
    ("payload", {"x": float("nan")}, "a NaN action payload value"),
    ("payload", {1: "x"}, "an integer action payload key"),
    ("metadata", [], "a list action metadata"),
])
def test_an_action_field_that_breaks_its_declared_shape_is_refused(field: str, value: object,
                                                                  label: str) -> None:
    """F01: the frozen action keeps its declared JSON payload and metadata shape."""
    contract = build(policy=bounded())
    ctx = context(contract)
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=_deep_damaged_action(field, value), context=ctx,
                      safety=safety(contract))
    assert len(str(info.value)) < 240, label


def test_a_sequence_element_of_the_wrong_model_is_refused() -> None:
    """F01: a sequence element must be the declared model, not an unrelated record."""
    contract = build(policy=bounded())
    ctx = context(contract)
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=_deep_damaged_contract("preconditions",
                                                      [contract.recovery_policy]),
                      context=ctx, safety=safety(contract))
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=_deep_damaged_contract("expected_outcomes",
                                                      [contract.recovery_policy]),
                      context=ctx, safety=safety(contract))


def test_a_look_alike_record_of_another_model_is_refused_in_a_sequence() -> None:
    """F01: the declared element class decides, so a different but similar record is refused."""
    from forecontract.models.actions import Precondition

    contract = build(policy=bounded())
    ctx = context(contract)
    # A condition is a legal element of the preconditions sequence, so it is accepted there and
    # refused where the declared element is an expected outcome.
    look_alike = Precondition(schema_version="0.1", condition_type="x", required=True)
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=_deep_damaged_contract("expected_outcomes", [look_alike]),
                      context=ctx, safety=safety(contract))


def test_a_sequence_element_of_the_wrong_model_inside_a_committed_list_is_refused() -> None:
    """F01: the same rule holds for an element reached inside a committed immutable sequence."""
    contract = build(policy=bounded())
    ctx = context(contract)
    stored = dict(contract.__dict__)
    stored["preconditions"] = (contract.recovery_policy,)
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=type(contract).model_construct(**stored), context=ctx,
                      safety=safety(contract))


def test_the_deep_field_checks_keep_every_legal_committed_shape() -> None:
    """F01 control: the legal deeply frozen contract and a legal fractional budget still pass."""
    contract = build(policy=bounded())
    ctx = context(contract)
    assert ENGINE.assess(contract=contract, context=ctx,
                         safety=safety(contract)).reason == "OBSERVATION_REQUIRED"
    stored = dict(contract.__dict__)
    stored["timeout_ms"] = 1.5
    assert ENGINE.assess(contract=type(contract).model_construct(**stored), context=ctx,
                         safety=safety(contract)).reason == "OBSERVATION_REQUIRED"
    stored = dict(contract.__dict__)
    stored["metadata"] = {"nested": [1, {"a": None}], "n": 1.5, "ok": True}
    assert ENGINE.assess(contract=type(contract).model_construct(**stored), context=ctx,
                         safety=safety(contract)).reason == "OBSERVATION_REQUIRED"
    stored = dict(contract.__dict__)
    stored["environment_type"] = "fake"
    assert ENGINE.assess(contract=type(contract).model_construct(**stored), context=ctx,
                         safety=safety(contract)).reason == "OBSERVATION_REQUIRED"


# --- round-005 corrections: strict JSON arrays and a pre-validated cycle prewalk ---

def _put_at(location: str, value: object, contract: object):
    """A committed contract whose one nested JSON field holds the supplied value."""
    stored = dict(contract.__dict__)
    if location == "metadata":
        stored["metadata"] = value
    elif location == "action.payload":
        stored_action = dict(contract.action.__dict__)
        stored_action["payload"] = value
        stored["action"] = type(contract.action).model_construct(**stored_action)
    else:
        stored_binding = dict(contract.verification_policy.bindings[0].__dict__)
        stored_binding["config"] = value
        binding = type(contract.verification_policy.bindings[0]).model_construct(
            **stored_binding)
        stored_policy = dict(contract.verification_policy.__dict__)
        stored_policy["bindings"] = [binding]
        stored["verification_policy"] = type(contract.verification_policy).model_construct(
            **stored_policy)
    return type(contract).model_construct(**stored)


@pytest.mark.parametrize("location", ["metadata", "action.payload", "binding.config"])
def test_an_arbitrary_tuple_is_not_a_json_array(location: str) -> None:
    """F01: the original strict JSON semantics accept only a real array or the frozen sequence."""
    contract = build(policy=bounded())
    ctx = context(contract)
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=_put_at(location, {"a": (1,)}, contract), context=ctx,
                      safety=safety(contract))
    assert len(str(info.value)) < 240
    # A real list and the Core frozen sequence both stand for a JSON array.
    from forecontract.contracts.snapshot import FrozenList

    for legal in ([1], FrozenList([1])):
        candidate = _put_at(location, {"a": legal, "b": legal}, contract)
        assert ENGINE.assess(contract=candidate, context=ctx,
                             safety=safety(contract)).reason == "OBSERVATION_REQUIRED"


@pytest.mark.parametrize("shape", ["dict_self", "mixed_list"])
@pytest.mark.parametrize("branch", ["normal", "stop", "zero_budget", "abort"])
@pytest.mark.parametrize("location", ["metadata", "action.payload", "binding.config"])
def test_a_json_field_cycle_is_bounded_before_every_branch(location: str, branch: str,
                                                          shape: str) -> None:
    """F07: a self-referencing or mixed cycle in a JSON field is refused before any branch."""
    contract = build(policy=bounded())
    loop: dict = {}
    if shape == "dict_self":
        loop["edge"] = loop
    else:
        items = [loop]
        loop["edge"] = items
    candidate = _put_at(location, loop, contract)
    if branch == "abort":
        stored = dict(candidate.__dict__)
        stored["recovery_policy"] = RecoveryPolicy(schema_version="0.1", mode="abort")
        candidate = type(candidate).model_construct(**stored)
    ctx = context(candidate, stop=branch == "stop",
                  budget=0 if branch == "zero_budget" else 100)
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=candidate, context=ctx, safety=safety(candidate))
    assert "maximum recursion depth" not in str(info.value)
    assert len(str(info.value)) < 240


@pytest.mark.parametrize("field", ["preconditions", "expected_outcomes", "bindings"])
def test_a_sequence_field_accepts_the_frozen_sequence_not_a_tuple(field: str) -> None:
    """F01: a declared sequence field distinguishes the Core frozen sequence from a tuple."""
    contract = build(policy=bounded())
    ctx = context(contract)
    if field == "bindings":
        original = list(contract.verification_policy.bindings)
        stored_policy = dict(contract.verification_policy.__dict__)
        stored_policy["bindings"] = tuple(original)
        broken = type(contract).model_construct(
            **{**dict(contract.__dict__),
               "verification_policy": type(contract.verification_policy).model_construct(
                   **stored_policy)})
    else:
        broken = type(contract).model_construct(
            **{**dict(contract.__dict__), field: tuple(getattr(contract, field))})
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=broken, context=ctx, safety=safety(contract))
    # The committed frozen representation itself is legal.
    assert ENGINE.assess(contract=contract, context=ctx,
                         safety=safety(contract)).reason == "OBSERVATION_REQUIRED"


def test_a_shared_reference_inside_a_json_object_is_not_a_cycle() -> None:
    """F07 control: the same array under two keys is a shared reference, not a cycle."""
    contract = build(policy=bounded())
    ctx = context(contract)
    shared = [1]
    for location in ("metadata", "action.payload", "binding.config"):
        candidate = _put_at(location, {"a": shared, "b": shared}, contract)
        assert ENGINE.assess(contract=candidate, context=ctx,
                             safety=safety(contract)).reason == "OBSERVATION_REQUIRED"


# --- round-006 correction: real Core frozen-type identity, not a class name ---

def _same_name_tuple_subclass() -> type:
    """A tuple subclass that only repeats the production class name and is unrelated to it."""
    from forecontract.contracts.snapshot import FrozenList

    look_alike = type("FrozenList", (tuple,), {"__module__": __name__})
    assert not issubclass(look_alike, FrozenList)
    assert look_alike.__name__ == FrozenList.__name__
    return look_alike


def _put_sequence(field: str, value: object, contract: object):
    """A committed contract whose one declared sequence field holds the supplied value."""
    if field == "bindings":
        stored_policy = dict(contract.verification_policy.__dict__)
        stored_policy["bindings"] = value
        return type(contract).model_construct(
            **{**dict(contract.__dict__),
               "verification_policy": type(contract.verification_policy).model_construct(
                   **stored_policy)})
    return type(contract).model_construct(**{**dict(contract.__dict__), field: value})


@pytest.mark.parametrize("location", ["metadata", "action.payload", "binding.config"])
def test_a_same_name_tuple_subclass_is_not_the_frozen_sequence(location: str) -> None:
    """F01: a JSON array must be a real list or the genuine Core frozen sequence."""
    from forecontract.contracts.snapshot import FrozenList

    contract = build(policy=bounded())
    ctx = context(contract)
    look_alike = _same_name_tuple_subclass()
    with pytest.raises(RecoveryError) as info:
        ENGINE.assess(contract=_put_at(location, {"a": look_alike([1])}, contract), context=ctx,
                      safety=safety(contract))
    assert "maximum recursion depth" not in str(info.value)
    assert len(str(info.value)) < 240
    for legal in ([1], FrozenList([1])):
        candidate = _put_at(location, {"a": legal}, contract)
        assert ENGINE.assess(contract=candidate, context=ctx,
                             safety=safety(contract)).reason == "OBSERVATION_REQUIRED"


@pytest.mark.parametrize("field", ["preconditions", "expected_outcomes", "bindings"])
def test_a_same_name_tuple_subclass_is_not_a_declared_sequence(field: str) -> None:
    """F01: a declared sequence field accepts the genuine frozen sequence, not a look-alike."""
    from forecontract.contracts.snapshot import FrozenList

    contract = build(policy=bounded())
    ctx = context(contract)
    if field == "bindings":
        source = list(contract.verification_policy.bindings)
    else:
        source = list(getattr(contract, field))
    look_alike = _same_name_tuple_subclass()
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=_put_sequence(field, look_alike(source), contract), context=ctx,
                      safety=safety(contract))
    with pytest.raises(RecoveryError):
        ENGINE.assess(contract=_put_sequence(field, tuple(source), contract), context=ctx,
                      safety=safety(contract))
    # A real list and the genuine production class both remain legal.
    for legal in (source, FrozenList(source)):
        assert ENGINE.assess(contract=_put_sequence(field, legal, contract), context=ctx,
                             safety=safety(contract)).reason == "OBSERVATION_REQUIRED"


def test_the_frozen_type_recognition_uses_identity_not_a_name() -> None:
    """F01 control: the recognition helper compares real classes, not names or module labels."""
    from forecontract.contracts.snapshot import FrozenList, FrozenMapping
    from forecontract.recovery import models as recovery_models

    look_alike = _same_name_tuple_subclass()
    assert recovery_models._is_frozen_sequence(FrozenList([1]))
    assert not recovery_models._is_frozen_sequence(look_alike([1]))
    assert not recovery_models._is_frozen_sequence((1,))
    assert not recovery_models._is_frozen_sequence([1])
    assert recovery_models._is_frozen_mapping(FrozenMapping({"a": 1}))
    assert not recovery_models._is_frozen_mapping({"a": 1})

    same_name_mapping = type("FrozenMapping", (dict,), {"__module__": __name__})
    assert not recovery_models._is_frozen_mapping(same_name_mapping({"a": 1}))
