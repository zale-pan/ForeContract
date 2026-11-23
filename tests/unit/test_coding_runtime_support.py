"""Unit tests for pure domain support, hooks, step verifier, and oracle (C17, C18, C19)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from forecontract.contracts import (
    ActionContract,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.models import (
    ActionSpec,
    ExpectedOutcome,
    Forecast,
    Observation,
    Precondition,
    VerificationStatus,
)
from forecontract.protocols.adapter import InvalidActionPayloadError
from forecontract.verification import (
    TaskCompletionCriteria,
    VerificationError,
    VerificationEvidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (
    REPO_ROOT / "packages" / "core" / "src",
    REPO_ROOT / "packages" / "planners" / "llm" / "src",
    REPO_ROOT / "packages" / "adapters" / "coding" / "src",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract_coding import (
    CODING_VERIFIER_TYPE,
    CodingContractHooksFactory,
    CodingDomainSupport,
    CodingError,
    CodingStepVerifier,
    CodingTask,
    CodingTaskCompletionOracle,
    CodingTaskCriteriaValidator,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
)


def _sample_task(tmp_path: Path) -> CodingTask:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    return CodingTask(
        run_id="run-pure",
        goal_id="goal-pure",
        goal_text="pure checks",
        workspace_root=str(ws),
        read_paths=["calc.py"],
        write_paths=[WriteTarget(path="calc.py", operation="modify")],
        commands={
            "acc": CommandSpec(
                command_id="acc",
                executable=sys.executable,
                args=["--junitxml={result_dir}/junit.xml", str(tmp_path / "ext.py")],
                cwd=str(ws),
                timeout_ms=5000,
                purpose="acceptance",
                expected_tests=[ExpectedTestCase(classname="ext", name="test_ok")],
            )
        },
        verification_command_ids=["acc"],
        max_steps=5,
        max_planning_calls=5,
        max_replans=2,
        run_timeout_ms=10000,
    )


def _obs(
    *,
    rev: str = "rev-1",
    action_id: str = "act-1",
    action_type: str = "write_file",
    path: str = "calc.py",
    file_sha: str | None = None,
    acceptance_status: str = "not_run",
    acceptance_rev: str | None = None,
    acceptance_action_id: str | None = None,
    task: CodingTask | None = None,
    overflow: bool = False,
    scope_ok: bool = True,
    scope_violations: list[dict[str, object]] | None = None,
    acceptance_sources_ok: bool = True,
    source_hashes: dict[str, str] | None = None,
    result_source_hashes: dict[str, str] | None = None,
) -> Observation:
    fsha = file_sha or hashlib.sha256(b"x = 1\n").hexdigest()
    cmd_ids = list(task.verification_command_ids) if task is not None else ["acc"]
    digest = task.commands_digest() if task is not None else "0" * 64
    base_src_hashes = (
        dict(source_hashes)
        if source_hashes is not None
        else {"ext.py": hashlib.sha256(b"def test_ok(): pass\n").hexdigest()}
    )
    cmd_src_hashes = (
        dict(result_source_hashes)
        if result_source_hashes is not None
        else dict(base_src_hashes)
    )
    results = []
    if acceptance_status == "passed":
        results = [
            {
                "command_id": "acc",
                "action_id": acceptance_action_id or "act-v1",
                "exit_code": 0,
                "status": "passed",
                "report_valid": True,
                "before_revision": acceptance_rev or rev,
                "after_revision": acceptance_rev or rev,
                "tests_executed": 1,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "test_source_hashes": cmd_src_hashes,
                "verified_tests": [{"classname": "ext", "name": "test_ok"}],
            }
        ]
    return Observation(
        schema_version="0.1",
        observation_id="obs-1",
        environment_type="coding",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        state={
            "schema_version": "0.1",
            "workspace_revision": rev,
            "scope_ok": scope_ok,
            "scope_violations": list(scope_violations or []),
            "acceptance_sources_ok": acceptance_sources_ok,
            "acceptance_source_hashes": base_src_hashes,
            "observation_overflow": overflow,
            "files": [
                {
                    "path": path,
                    "exists": True,
                    "sha256": fsha,
                    "bytes": 6,
                    "readable": True,
                    "writable": True,
                    "operation": "modify",
                }
            ],
            "changed_paths": [],
            "acceptance": {
                "status": acceptance_status,
                "revision": acceptance_rev,
                "action_id": acceptance_action_id or ("act-v1" if acceptance_status == "passed" else None),
                "command_ids": cmd_ids,
                "commands_digest": digest,
                "stale_reason": None,
                "results": results,
            },
            "last_tool_result": {
                "action_id": action_id,
                "action_type": action_type,
                "path": path,
                "written_sha256": fsha,
                "read_back_sha256": fsha,
                "read_back_match": True,
                "after_revision": rev,
                "tool_succeeded": True,
            },
            "history": [],
            "history_truncated": False,
        },
    )


def _committed_contract(
    *,
    action: ActionSpec,
    outcomes: list[ExpectedOutcome],
) -> ActionContract:
    return ActionContract(
        schema_version="0.1",
        contract_id="c-1",
        goal_id="goal-pure",
        forecast_id="f-1",
        observation_id="obs-1",
        adapter_name="coding_workspace",
        environment_type="coding",
        action=action,
        preconditions=[],
        expected_outcomes=outcomes,
        verification_policy=VerificationPolicy(
            schema_version="0.1",
            bindings=[
                VerificationBinding(
                    schema_version="0.1",
                    outcome_index=i,
                    verifier_type=CODING_VERIFIER_TYPE,
                    config={},
                )
                for i in range(len(outcomes))
            ],
        ),
        recovery_policy=RecoveryPolicy(
            schema_version="0.1", mode="bounded", max_retries=0, allow_replan=True
        ),
        timeout_ms=5000.0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_c17_pure_domain_methods_perform_no_io_and_reject_bypass_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_io(*args, **kwargs):
        raise AssertionError("pure domain layer must never perform disk or process I/O")

    monkeypatch.setattr(Path, "read_bytes", _no_io)
    monkeypatch.setattr(Path, "write_bytes", _no_io)
    monkeypatch.setattr(subprocess, "run", _no_io)
    monkeypatch.setattr(subprocess, "Popen", _no_io)

    task = _sample_task(tmp_path)
    support = CodingDomainSupport(task=task)
    hooks = CodingContractHooksFactory().create(
        evidence=VerificationEvidence(
            schema_version="0.1", observation=_obs(task=task), state_complete=True
        )
    )
    criteria_val = CodingTaskCriteriaValidator(task=task)

    # Reject json_rule bypass, empty criteria, and all-optional criteria
    for bad_conditions in (
        [],
        [ExpectedOutcome(schema_version="0.1", outcome_type="coding_task_verified", payload={}, required=False)],
        [ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", payload={"rule_type": "value_equals", "path": ["done"], "expected": True}, required=True)],
    ):
        with pytest.raises(VerificationError):
            crit = TaskCompletionCriteria(schema_version="0.1", goal_id="goal-pure", conditions=bad_conditions)
            criteria_val.validate(criteria=crit)

    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_expected_outcome(
            ExpectedOutcome(schema_version="0.1", outcome_type="json_rule", payload={"path": ["done"]})
        )
    with pytest.raises(InvalidActionPayloadError):
        hooks.validate_precondition(
            Precondition(schema_version="0.1", condition_type="json_rule", payload={"path": ["done"]})
        )

    # Reject inapplicable expected outcome pairing in prepare_contract and step verifier
    bad_forecast = Forecast(
        schema_version="0.1",
        forecast_id="f-bad",
        proposed_action=ActionSpec(
            schema_version="0.1",
            action_type="read_file",
            payload={"path": "calc.py", "start_line": 1, "max_lines": 5},
            metadata={"action_id": "a-read"},
        ),
        expected_outcomes=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="coding_file_written",
                payload={"path": "calc.py"},
                required=True,
            )
        ],
    )
    ev = support.to_evidence(_obs(task=task))
    with pytest.raises(CodingError):
        support.prepare_contract(forecast=bad_forecast, evidence=ev)


def test_c18_verifier_and_oracle_reject_forged_action_file_revision_and_pass(
    tmp_path: Path,
) -> None:
    task = _sample_task(tmp_path)
    verifier = CodingStepVerifier()
    oracle = CodingTaskCompletionOracle(task=task)

    contract = _committed_contract(
        action=ActionSpec(
            schema_version="0.1",
            action_type="write_file",
            payload={
                "path": "calc.py",
                "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
                "content": "x = 1\n",
            },
            metadata={"action_id": "act-expected"},
        ),
        outcomes=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="coding_file_written",
                payload={"path": "calc.py"},
                required=True,
            )
        ],
    )

    # Forged stale action_id rejected
    stale_ev = VerificationEvidence(
        schema_version="0.1",
        observation=_obs(action_id="act-older", task=task),
        state_complete=True,
    )
    assert verifier.verify_step(contract=contract, evidence=stale_ev).status is VerificationStatus.FAIL

    # Forged different file path result rejected
    wrong_file_ev = VerificationEvidence(
        schema_version="0.1",
        observation=_obs(action_id="act-expected", path="other.py", task=task),
        state_complete=True,
    )
    assert verifier.verify_step(contract=contract, evidence=wrong_file_ev).status is VerificationStatus.FAIL

    # Matching action_id + file + read-back passes
    good_ev = VerificationEvidence(
        schema_version="0.1",
        observation=_obs(action_id="act-expected", path="calc.py", task=task),
        state_complete=True,
    )
    assert verifier.verify_step(contract=contract, evidence=good_ev).status is VerificationStatus.PASS

    # Task oracle rejects passed acceptance bound to another workspace revision
    criteria = TaskCompletionCriteria(
        schema_version="0.1",
        goal_id="goal-pure",
        conditions=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="coding_task_verified",
                payload={},
                required=True,
            )
        ],
    )
    stale_acc = VerificationEvidence(
        schema_version="0.1",
        observation=_obs(
            rev="rev-new",
            acceptance_status="passed",
            acceptance_rev="rev-old",
            acceptance_action_id="act-v1",
            task=task,
        ),
        state_complete=True,
    )
    assert oracle.evaluate_completion(criteria=criteria, evidence=stale_acc).status is VerificationStatus.FAIL

    # Genuine same-revision passed acceptance succeeds
    pass_acc = VerificationEvidence(
        schema_version="0.1",
        observation=_obs(
            rev="rev-new",
            acceptance_status="passed",
            acceptance_rev="rev-new",
            acceptance_action_id="act-v1",
            task=task,
        ),
        state_complete=True,
    )
    assert oracle.evaluate_completion(criteria=criteria, evidence=pass_acc).status is VerificationStatus.PASS


def test_c19_overflow_marks_state_incomplete_and_refuses_pass(tmp_path: Path) -> None:
    task = _sample_task(tmp_path)
    support = CodingDomainSupport(task=task)
    oracle = CodingTaskCompletionOracle(task=task)
    obs = _obs(
        rev="rev-1",
        acceptance_status="passed",
        acceptance_rev="rev-1",
        acceptance_action_id="act-v1",
        task=task,
        overflow=True,
    )
    ev = support.to_evidence(obs)
    assert ev.state_complete is False
    criteria = TaskCompletionCriteria(
        schema_version="0.1",
        goal_id="goal-pure",
        conditions=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="coding_task_verified",
                payload={},
                required=True,
            )
        ],
    )
    assert oracle.evaluate_completion(criteria=criteria, evidence=ev).status is VerificationStatus.INCONCLUSIVE


def test_d06_oracle_and_verifier_reject_tampered_test_source_hashes(tmp_path: Path) -> None:
    task = _sample_task(tmp_path)
    oracle = CodingTaskCompletionOracle(task=task)
    criteria = TaskCompletionCriteria(
        schema_version="0.1",
        goal_id="goal-pure",
        conditions=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="coding_task_verified",
                payload={},
                required=True,
            )
        ],
    )

    bad_flag_ev = VerificationEvidence(
        schema_version="0.1",
        observation=_obs(
            rev="rev-1",
            acceptance_status="passed",
            acceptance_rev="rev-1",
            task=task,
            acceptance_sources_ok=False,
        ),
        state_complete=True,
    )
    assert oracle.evaluate_completion(criteria=criteria, evidence=bad_flag_ev).status is VerificationStatus.FAIL

    mismatch_ev = VerificationEvidence(
        schema_version="0.1",
        observation=_obs(
            rev="rev-1",
            acceptance_status="passed",
            acceptance_rev="rev-1",
            task=task,
            source_hashes={"ext.py": "1" * 64},
            result_source_hashes={"ext.py": "2" * 64},
        ),
        state_complete=True,
    )
    assert oracle.evaluate_completion(criteria=criteria, evidence=mismatch_ev).status is VerificationStatus.FAIL


def test_d08_step_verifier_and_oracle_reject_scope_violations(tmp_path: Path) -> None:
    task = _sample_task(tmp_path)
    verifier = CodingStepVerifier()
    oracle = CodingTaskCompletionOracle(task=task)
    criteria = TaskCompletionCriteria(
        schema_version="0.1",
        goal_id="goal-pure",
        conditions=[
            ExpectedOutcome(
                schema_version="0.1",
                outcome_type="coding_task_verified",
                payload={},
                required=True,
            )
        ],
    )
    violation = [{"path": "unapproved.py", "reason": "unauthorized_create", "before_sha256": None, "after_sha256": "a" * 64}]

    for obs in (
        _obs(rev="rev-1", acceptance_status="passed", acceptance_rev="rev-1", task=task, scope_ok=False),
        _obs(rev="rev-1", acceptance_status="passed", acceptance_rev="rev-1", task=task, scope_ok=True, scope_violations=violation),
    ):
        ev = VerificationEvidence(schema_version="0.1", observation=obs, state_complete=True)
        assert oracle.evaluate_completion(criteria=criteria, evidence=ev).status is VerificationStatus.FAIL

        write_contract = _committed_contract(
            action=ActionSpec(
                schema_version="0.1",
                action_type="write_file",
                payload={
                    "path": "calc.py",
                    "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
                    "content": "x = 1\n",
                },
                metadata={"action_id": "act-1"},
            ),
            outcomes=[
                ExpectedOutcome(
                    schema_version="0.1",
                    outcome_type="coding_file_written",
                    payload={"path": "calc.py"},
                    required=True,
                ),
                ExpectedOutcome(
                    schema_version="0.1",
                    outcome_type="coding_tool_result",
                    payload={"action_type": "write_file", "path": "calc.py"},
                    required=True,
                ),
            ],
        )
        assert verifier.verify_step(contract=write_contract, evidence=ev).status is VerificationStatus.FAIL
