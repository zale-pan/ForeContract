"""Integration tests combining deterministic rules, ModelVerifier and CompositeVerifier."""

from __future__ import annotations

import copy
import datetime
import gc
import os
import socket
import subprocess

import pytest
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.verification import (
    CompositeVerifier,
    JsonRuleEvaluator,
    ModelVerifier,
    TaskCompletionCriteria,
    TaskCompletionOracle,
    ValueEqualsRule,
    VerificationError,
    VerificationEvidence,
)


@pytest.fixture
def isolated_verifiers(monkeypatch: pytest.MonkeyPatch) -> tuple[
    JsonRuleEvaluator,
    ModelVerifier,
    CompositeVerifier,
    TaskCompletionOracle,
]:
    def forbid_external(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline verification must not perform network or process actions")

    monkeypatch.setattr(socket.socket, "connect", forbid_external)
    monkeypatch.setattr(socket, "create_connection", forbid_external)
    monkeypatch.setattr(subprocess, "Popen", forbid_external)
    return (
        JsonRuleEvaluator(),
        ModelVerifier(),
        CompositeVerifier(),
        TaskCompletionOracle(),
    )


def test_deterministic_and_offline_model_verdicts_compose_without_external_side_effects(
    isolated_verifiers: tuple[
        JsonRuleEvaluator,
        ModelVerifier,
        CompositeVerifier,
        TaskCompletionOracle,
    ],
) -> None:
    evaluator, model_verifier, composite_verifier, oracle = isolated_verifiers
    initial_state = (gc.get_threshold(), gc.isenabled(), dict(os.environ))

    now = datetime.datetime(2026, 11, 13, 8, 0, tzinfo=datetime.timezone.utc)
    observation = Observation(
        schema_version="0.1",
        observation_id="obs-int-1",
        environment_type="fake",
        state={"order": {"status": "submitted", "total": 42}},
        timestamp=now,
    )
    evidence = VerificationEvidence(
        schema_version="0.1",
        observation=observation,
        state_complete=True,
    )
    det_rule = ValueEqualsRule(
        schema_version="0.1",
        rule_type="value_equals",
        path=["order", "status"],
        expected="submitted",
    )
    det_eval = evaluator.evaluate(rule=det_rule, evidence=evidence)
    assert det_eval.status is VerificationStatus.PASS

    req = {
        "request_id": "req-int-1",
        "model_id": "offline-auditor-v1",
        "prompt_version": "p-1",
        "evidence_id": observation.observation_id,
    }
    ok_resp = {
        **req,
        "state": "ok",
        "response_id": "resp-ok-1",
        "status": "pass",
        "confidence": 0.9,
        "reason": "receipt summary matches submitted order",
    }
    refused_resp = {
        **req,
        "state": "refused",
        "response_id": "resp-ref-1",
        "status": "inconclusive",
        "confidence": None,
        "reason": "policy refusal on secondary check",
    }
    timeout_resp = {
        **req,
        "state": "timeout",
        "response_id": None,
        "status": None,
        "confidence": None,
        "reason": "offline recorded upstream timeout",
    }

    before_inputs = copy.deepcopy((req, ok_resp, refused_resp, timeout_resp))
    ok_model = model_verifier.verify(request=req, response=ok_resp)
    refused_model = model_verifier.verify(request=req, response=refused_resp)
    timeout_model = model_verifier.verify(request=req, response=timeout_resp)

    assert ok_model.status is VerificationStatus.PASS
    assert refused_model.status is VerificationStatus.INCONCLUSIVE
    assert timeout_model.status is VerificationStatus.INCONCLUSIVE

    items = [
        {
            "id": "det-order-status",
            "status": det_eval.status.value,
            "kind": "deterministic",
            "critical": True,
            "weight": 2.0,
            "source_id": observation.observation_id,
            "reason": det_eval.reason,
        },
        {
            "id": "mod-ok",
            "status": ok_model.status.value,
            "kind": "model",
            "critical": False,
            "weight": 1.0,
            "source_id": "resp-ok-1",
            "reason": ok_model.reason,
        },
        {
            "id": "mod-timeout",
            "status": timeout_model.status.value,
            "kind": "model",
            "critical": False,
            "weight": 1.0,
            "source_id": req["request_id"],
            "reason": timeout_model.reason,
        },
    ]
    items_snapshot = copy.deepcopy(items)

    weighted_pass = composite_verifier.verify(items=items, mode="weighted", threshold=0.75)
    assert weighted_pass.status is VerificationStatus.PASS
    assert weighted_pass.reason == "COMPOSITE_PASSED"
    assert weighted_pass.metadata["lower"] == 0.75
    assert weighted_pass.metadata["upper"] == 1.0

    weighted_inc = composite_verifier.verify(items=items, mode="weighted", threshold=0.8)
    assert weighted_inc.status is VerificationStatus.INCONCLUSIVE
    assert weighted_inc.reason == "COMPOSITE_INCONCLUSIVE"
    assert weighted_inc.metadata["lower"] == 0.75
    assert weighted_inc.metadata["upper"] == 1.0

    assert composite_verifier.verify(
        items=items, mode="weighted", threshold=0.8
    ).model_dump(mode="json") == weighted_inc.model_dump(mode="json")
    assert items == items_snapshot
    assert (req, ok_resp, refused_resp, timeout_resp) == before_inputs
    assert (gc.get_threshold(), gc.isenabled(), dict(os.environ)) == initial_state

    criteria = TaskCompletionCriteria(
        schema_version="0.1",
        goal_id="goal-int-1",
        conditions=[
            {
                "schema_version": "0.1",
                "outcome_type": "json_rule",
                "required": True,
                "payload": {
                    "schema_version": "0.1",
                    "rule_type": "value_equals",
                    "path": ["order", "status"],
                    "expected": "completed",
                },
            }
        ],
    )
    oracle_result = oracle.evaluate_completion(criteria=criteria, evidence=evidence)
    assert oracle_result.status is VerificationStatus.FAIL
    assert weighted_pass.status is VerificationStatus.PASS

    with pytest.raises(VerificationError) as exc_info:
        composite_verifier.verify(
            items=[
                {
                    "id": "illegal-critical-model",
                    "status": ok_model.status.value,
                    "kind": "model",
                    "critical": True,
                    "weight": 1.0,
                    "source_id": "resp-ok-1",
                    "reason": ok_model.reason,
                }
            ],
            mode="all",
        )
    assert exc_info.value.code == "VERIFICATION_BINDING_INVALID"
