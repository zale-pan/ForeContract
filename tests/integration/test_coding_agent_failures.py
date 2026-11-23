"""Adversarial & failure-mode Coding Agent integration tests (C12, C13, C14, C15, C20)."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest
from forecontract.models import ActionSpec, ExpectedOutcome, VerificationStatus
from forecontract.runtime.errors import RuntimeError as ForeContractRuntimeError
from forecontract.tracing import JsonlTraceStore
from forecontract.verification import TaskCompletionCriteria
from forecontract_llm import ModelResponse, PlannerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (
    REPO_ROOT / "packages" / "core" / "src",
    REPO_ROOT / "packages" / "planners" / "llm" / "src",
    REPO_ROOT / "packages" / "adapters" / "coding" / "src",
    Path(__file__).resolve().parent,
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from coding_helpers import (
    FIXED_CALC_SOURCE,
    SequenceProvider,
    make_sum_even_task,
    setup_sum_even_workspace,
)
from forecontract_coding import (
    CodingAdapter,
    CodingConfig,
    CodingDomainSupport,
    CodingTask,
    CodingTaskCompletionOracle,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
    run_coding_task,
)


def test_c12_completion_hint_true_and_fake_report_never_complete_without_verify(
    tmp_path: Path,
) -> None:
    ws, test_file, out = setup_sum_even_workspace(tmp_path)
    task, cfg = make_sum_even_task(
        ws=ws,
        test_file=test_file,
        out=out,
        extra_read_paths=["report.json"],
        extra_write_targets=[WriteTarget(path="report.json", operation="create")],
        max_steps=3,
        max_planning_calls=3,
    )

    def _plan(idx: int, req):
        files = {f["path"]: f for f in req.observation.state["files"]}
        if idx == 0:
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "calculator.py",
                        "expected_sha256": files["calculator.py"]["sha256"],
                        "content": FIXED_CALC_SOURCE,
                    },
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_file_written",
                        "payload": {"path": "calculator.py"},
                        "required": True,
                    }
                ],
                "task_completion_hint": True,
            }
        # Model writes a fake report.json and keeps setting task_completion_hint=True
        return {
            "schema_version": "0.1",
            "proposed_action": {
                "schema_version": "0.1",
                "action_type": "write_file" if idx == 1 else "read_file",
                "payload": (
                    {
                        "path": "report.json",
                        "expected_sha256": None,
                        "content": '{"acceptance": "passed"}',
                    }
                    if idx == 1
                    else {"path": "calculator.py", "start_line": 1, "max_lines": 10}
                ),
                "metadata": {},
            },
            "expected_outcomes": [
                {
                    "schema_version": "0.1",
                    "outcome_type": "coding_tool_result",
                    "payload": {
                        "action_type": "write_file" if idx == 1 else "read_file"
                    },
                    "required": True,
                }
            ],
            "task_completion_hint": True,
        }

    result = asyncio.run(
        run_coding_task(
            task=task,
            config=cfg,
            provider=SequenceProvider(_plan),
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=5),
            trace_writer=JsonlTraceStore(tmp_path / "trace_c12.jsonl"),
        )
    )
    assert result.terminal_status != "completed"
    assert result.acceptance.status == "not_run"


def test_c13_two_acceptance_commands_one_failing_never_completes(tmp_path: Path) -> None:
    ws, ext_pass, out = setup_sum_even_workspace(tmp_path)
    ext_fail = ext_pass.parent / "test_extra_fail.py"
    ext_fail.write_text("import calculator\ndef test_extra_contract(): assert calculator.sum_even([2, 4]) == 999\n", encoding="utf-8")
    basetemp = out / "basetemp"
    basetemp.mkdir(parents=True, exist_ok=True)

    task = CodingTask(
        run_id="run-c13",
        goal_id="goal-c13",
        goal_text="Both acceptance suites must pass.",
        workspace_root=str(ws),
        read_paths=["calculator.py"],
        write_paths=[WriteTarget(path="calculator.py", operation="modify")],
        commands={
            "acc_one": CommandSpec(
                command_id="acc_one",
                executable=sys.executable,
                args=[
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--import-mode=importlib",
                    f"--rootdir={ext_pass.parent}",
                    f"--confcutdir={ext_pass.parent}",
                    f"--basetemp={basetemp}",
                    "--junitxml={result_dir}/junit.xml",
                    str(ext_pass),
                ],
                cwd=str(ws),
                env={"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                timeout_ms=10000,
                purpose="acceptance",
                expected_tests=[
                    ExpectedTestCase(classname="test_calc_acceptance", name="test_even_basic"),
                    ExpectedTestCase(classname="test_calc_acceptance", name="test_even_edge_cases"),
                ],
            ),
            "acc_two": CommandSpec(
                command_id="acc_two",
                executable=sys.executable,
                args=[
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--import-mode=importlib",
                    f"--rootdir={ext_fail.parent}",
                    f"--confcutdir={ext_fail.parent}",
                    f"--basetemp={basetemp}",
                    "--junitxml={result_dir}/junit.xml",
                    str(ext_fail),
                ],
                cwd=str(ws),
                env={"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                timeout_ms=10000,
                purpose="acceptance",
                expected_tests=[
                    ExpectedTestCase(classname="test_extra_fail", name="test_extra_contract"),
                ],
            ),
        },
        verification_command_ids=["acc_one", "acc_two"],
        max_steps=4,
        max_planning_calls=4,
        max_replans=1,
        run_timeout_ms=60000,
    )

    def _plan(idx: int, req):
        files = {f["path"]: f for f in req.observation.state["files"]}
        if idx == 0:
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "calculator.py",
                        "expected_sha256": files["calculator.py"]["sha256"],
                        "content": FIXED_CALC_SOURCE,
                    },
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_file_written",
                        "payload": {"path": "calculator.py"},
                        "required": True,
                    }
                ],
            }
        if idx == 1:
            # Running only the first acceptance subset cannot satisfy task completion
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "run_command",
                    "payload": {"command_id": "acc_one"},
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_tool_result",
                        "payload": {"action_type": "run_command", "command_id": "acc_one"},
                        "required": True,
                    }
                ],
            }
        return {
            "schema_version": "0.1",
            "proposed_action": {
                "schema_version": "0.1",
                "action_type": "verify_task",
                "payload": {},
                "metadata": {},
            },
            "expected_outcomes": [
                {
                    "schema_version": "0.1",
                    "outcome_type": "coding_task_verified",
                    "payload": {},
                    "required": True,
                }
            ],
        }

    result = asyncio.run(
        run_coding_task(
            task=task,
            config=CodingConfig(output_root=str(out)),
            provider=SequenceProvider(_plan),
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=6),
            trace_writer=JsonlTraceStore(tmp_path / "trace_c13.jsonl"),
        )
    )
    assert result.terminal_status != "completed"
    assert result.acceptance.status == "failed"
    assert [entry["command_id"] for entry in result.acceptance.results] == ["acc_one", "acc_two"]
    assert result.acceptance.results[0]["status"] == "passed"
    assert result.acceptance.results[1]["status"] == "failed"


def test_c15_edit_after_verify_forces_reverification_before_completion(tmp_path: Path) -> None:
    # 1. Editing during verify_task invalidates green pytest exit 0 and blocks completion.
    ws1, _, out1 = setup_sum_even_workspace(tmp_path / "case_mid_verify")
    ext1 = tmp_path / "ext_mid_verify"
    ext1.mkdir(parents=True, exist_ok=True)
    basetemp1 = out1 / "basetemp"
    basetemp1.mkdir(parents=True, exist_ok=True)
    (ws1 / "note.txt").write_bytes(b"v0\n")
    note_target = ws1 / "note.txt"
    tfile1 = ext1 / "test_calc_mutating.py"
    tfile1.write_text(
        "from pathlib import Path\n"
        "import calculator\n"
        "def test_even_and_mutate():\n"
        f"    Path({str(note_target)!r}).write_bytes(b'v1\\n')\n"
        "    assert calculator.sum_even([1, 2, 3, 4]) == 6\n",
        encoding="utf-8",
    )
    mut_cmd = CommandSpec(
        command_id="pytest_acc",
        executable=sys.executable,
        args=[
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
            f"--rootdir={ext1}",
            f"--confcutdir={ext1}",
            f"--basetemp={basetemp1}",
            "--junitxml={result_dir}/junit.xml",
            str(tfile1),
        ],
        cwd=str(ws1),
        env={"PYTHONPATH": str(ws1), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        timeout_ms=15000,
        purpose="acceptance",
        expected_tests=[ExpectedTestCase(classname="test_calc_mutating", name="test_even_and_mutate")],
    )
    task1 = CodingTask(
        run_id="run-c15-mid",
        goal_id="goal-c15-mid",
        goal_text="Fix calculator and pass verification without workspace mutation.",
        workspace_root=str(ws1),
        read_paths=["calculator.py", "note.txt"],
        write_paths=[WriteTarget(path="calculator.py", operation="modify")],
        commands={"pytest_acc": mut_cmd},
        verification_command_ids=["pytest_acc"],
        max_steps=5,
        max_planning_calls=6,
        max_replans=2,
        run_timeout_ms=60000,
    )

    def _plan_fix_then_verify(idx: int, req):
        files = {f["path"]: f for f in req.observation.state["files"]}
        if idx == 0:
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "calculator.py",
                        "expected_sha256": files["calculator.py"]["sha256"],
                        "content": FIXED_CALC_SOURCE,
                    },
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_file_written",
                        "payload": {"path": "calculator.py"},
                        "required": True,
                    }
                ],
            }
        return {
            "schema_version": "0.1",
            "proposed_action": {
                "schema_version": "0.1",
                "action_type": "verify_task",
                "payload": {},
                "metadata": {},
            },
            "expected_outcomes": [
                {
                    "schema_version": "0.1",
                    "outcome_type": "coding_task_verified",
                    "payload": {},
                    "required": True,
                }
            ],
        }

    res_mut = asyncio.run(
        run_coding_task(
            task=task1,
            config=CodingConfig(output_root=str(out1)),
            provider=SequenceProvider(_plan_fix_then_verify),
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=6),
            trace_writer=JsonlTraceStore(tmp_path / "trace_c15_mid.jsonl"),
        )
    )
    assert res_mut.terminal_status == "aborted"
    assert res_mut.acceptance.status == "unknown"
    assert res_mut.acceptance.stale_reason == "workspace_changed_during_verification"

    # 2. Editing after verify_task passes clears acceptance, fails oracle, and requires re-verification.
    ws2, test_file2, out2 = setup_sum_even_workspace(tmp_path / "case_post_verify")
    task2, cfg2 = make_sum_even_task(ws=ws2, test_file=test_file2, out=out2)
    adapter2 = CodingAdapter(task=task2, config=cfg2)
    oracle2 = CodingTaskCompletionOracle(task=task2)
    support2 = CodingDomainSupport(task=task2)
    criteria2 = TaskCompletionCriteria(
        schema_version="0.1",
        goal_id=task2.goal_id,
        conditions=[
            ExpectedOutcome(
                schema_version="0.1", outcome_type="coding_task_verified", payload={}, required=True
            )
        ],
    )

    async def _post_verify_edit_flow() -> None:
        await adapter2.start()
        try:
            obs0 = await adapter2.observe()
            sha0 = obs0.state["files"][0]["sha256"]
            await adapter2.execute(
                ActionSpec(
                    schema_version="0.1",
                    action_type="write_file",
                    payload={"path": "calculator.py", "expected_sha256": sha0, "content": FIXED_CALC_SOURCE},
                    metadata={"action_id": "c15-w1"},
                )
            )
            await adapter2.execute(
                ActionSpec(
                    schema_version="0.1",
                    action_type="verify_task",
                    payload={},
                    metadata={"action_id": "c15-v1"},
                )
            )
            obs_passed = await adapter2.observe()
            assert obs_passed.state["acceptance"]["status"] == "passed"
            assert oracle2.evaluate_completion(
                criteria=criteria2, evidence=support2.to_evidence(obs_passed)
            ).status is VerificationStatus.PASS

            sha1 = obs_passed.state["files"][0]["sha256"]
            await adapter2.execute(
                ActionSpec(
                    schema_version="0.1",
                    action_type="write_file",
                    payload={
                        "path": "calculator.py",
                        "expected_sha256": sha1,
                        "content": FIXED_CALC_SOURCE + "# post-verify edit\n",
                    },
                    metadata={"action_id": "c15-w2"},
                )
            )
            obs_cleared = await adapter2.observe()
            assert obs_cleared.state["acceptance"]["status"] == "unknown"
            assert obs_cleared.state["acceptance"]["stale_reason"] == "workspace_revision_changed"
            assert oracle2.evaluate_completion(
                criteria=criteria2, evidence=support2.to_evidence(obs_cleared)
            ).status is VerificationStatus.FAIL

            await adapter2.execute(
                ActionSpec(
                    schema_version="0.1",
                    action_type="verify_task",
                    payload={},
                    metadata={"action_id": "c15-v2"},
                )
            )
            obs_reverified = await adapter2.observe()
            assert obs_reverified.state["acceptance"]["status"] == "passed"
            assert oracle2.evaluate_completion(
                criteria=criteria2, evidence=support2.to_evidence(obs_reverified)
            ).status is VerificationStatus.PASS
        finally:
            await adapter2.stop()

    asyncio.run(_post_verify_edit_flow())


def test_c20_and_d10_timeout_unknown_does_not_replay_deadline_and_bad_plans_stop_cleanly(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws_c20"
    ext = tmp_path / "ext_c20"
    out = tmp_path / "out_c20"
    basetemp = out / "basetemp"
    ws.mkdir()
    ext.mkdir()
    out.mkdir()
    basetemp.mkdir()
    (ws / "app.py").write_text("x = 1\n", encoding="utf-8")
    ext_test = ext / "test_acc.py"
    ext_test.write_text("def test_ok(): assert True\n", encoding="utf-8")
    marker = out / "run_counter.txt"
    ready_file = out / "ready.txt"
    pid_file = out / "child_pid.txt"

    def _pid_is_alive(pid: int) -> bool:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.GetExitCodeProcess.restype = wintypes.BOOL
            k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            h = k32.OpenProcess(0x00101000, False, pid)
            if not h:
                return False
            try:
                code = wintypes.DWORD(0)
                if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return False
                return code.value == 259  # STILL_ACTIVE
            finally:
                k32.CloseHandle(h)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _build_task(
        *,
        run_id: str,
        timeout_ms: int = 15000,
        run_timeout_ms: int = 20000,
        max_steps: int = 5,
        max_planning_calls: int = 5,
        max_replans: int = 3,
    ) -> CodingTask:
        return CodingTask(
            run_id=run_id,
            goal_id="goal-c20",
            goal_text="Timeout command must not be replayed automatically.",
            workspace_root=str(ws),
            read_paths=["app.py"],
            write_paths=[WriteTarget(path="app.py", operation="modify")],
            commands={
                "hang_cmd": CommandSpec(
                    command_id="hang_cmd",
                    executable=sys.executable,
                    args=[
                        "-c",
                        (
                            f"import os, pathlib, time; "
                            f"pathlib.Path(r'{pid_file}').write_text(str(os.getpid()), encoding='utf-8'); "
                            f"p = pathlib.Path(r'{marker}'); "
                            f"p.write_text((p.read_text() if p.exists() else '') + '1\\n'); "
                            f"pathlib.Path(r'{ready_file}').write_text('ready', encoding='utf-8'); "
                            f"time.sleep(30)"
                        ),
                    ],
                    cwd=str(ws),
                    timeout_ms=timeout_ms,
                    purpose="diagnostic",
                ),
                "acc": CommandSpec(
                    command_id="acc",
                    executable=sys.executable,
                    args=[
                        "-B",
                        "-m",
                        "pytest",
                        "-q",
                        f"--rootdir={ext}",
                        f"--confcutdir={ext}",
                        f"--basetemp={basetemp}",
                        "--junitxml={result_dir}/junit.xml",
                        str(ext_test),
                    ],
                    cwd=str(ws),
                    timeout_ms=5000,
                    purpose="acceptance",
                    expected_tests=[ExpectedTestCase(classname="test_acc", name="test_ok")],
                ),
            },
            verification_command_ids=["acc"],
            max_steps=max_steps,
            max_planning_calls=max_planning_calls,
            max_replans=max_replans,
            run_timeout_ms=run_timeout_ms,
        )

    def _plan(idx: int, req):
        return {
            "schema_version": "0.1",
            "proposed_action": {
                "schema_version": "0.1",
                "action_type": "run_command",
                "payload": {"command_id": "hang_cmd"},
                "metadata": {},
            },
            "expected_outcomes": [
                {
                    "schema_version": "0.1",
                    "outcome_type": "coding_tool_result",
                    "payload": {"action_type": "run_command", "command_id": "hang_cmd"},
                    "required": True,
                }
            ],
        }

    # 1. Deterministic bounded ready handshake BEFORE controlled host cancellation
    trace_cancel = tmp_path / "trace_c20_cancel.jsonl"

    async def _run_ready_then_cancel() -> None:
        run_task = asyncio.create_task(
            run_coding_task(
                task=_build_task(run_id="run-c20-ready-cancel", timeout_ms=15000),
                config=CodingConfig(output_root=str(out)),
                provider=SequenceProvider(_plan),
                planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=5),
                trace_writer=JsonlTraceStore(trace_cancel),
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if (
                ready_file.exists()
                and ready_file.read_text(encoding="utf-8") == "ready"
                and pid_file.exists()
                and pid_file.read_text(encoding="utf-8").strip().isdigit()
            ):
                break
            if run_task.done():
                raise AssertionError(f"run_task ended unexpectedly before ready: {run_task}")
            await asyncio.sleep(0.02)
        else:
            run_task.cancel()
            raise AssertionError("child process did not signal ready within bounded wait")

        assert not run_task.done()
        assert len(marker.read_text(encoding="utf-8").splitlines()) == 1
        child_pid = int(pid_file.read_text(encoding="utf-8").strip())

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=5.0)

        stop_deadline = time.monotonic() + 5.0
        while time.monotonic() < stop_deadline and _pid_is_alive(child_pid):
            await asyncio.sleep(0.02)
        assert not _pid_is_alive(child_pid)

    asyncio.run(_run_ready_then_cancel())
    assert len(marker.read_text(encoding="utf-8").splitlines()) == 1
    time.sleep(0.15)
    assert len(marker.read_text(encoding="utf-8").splitlines()) == 1

    # 2. Timeout AFTER confirmed ready handshake -> stops cleanly (POLICY_ABORT) with zero replay
    marker.unlink()
    ready_file.unlink()
    pid_file.unlink()

    async def _run_observed_timeout():
        run_task = asyncio.create_task(
            run_coding_task(
                task=_build_task(run_id="run-c20-timeout", timeout_ms=700),
                config=CodingConfig(output_root=str(out)),
                provider=SequenceProvider(_plan),
                planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=5),
                trace_writer=JsonlTraceStore(tmp_path / "trace_c20.jsonl"),
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ready_file.exists() and ready_file.read_text(encoding="utf-8") == "ready":
                break
            await asyncio.sleep(0.02)
        else:
            run_task.cancel()
            raise AssertionError("timeout subcase did not reach ready handshake before exit")
        return await asyncio.wait_for(run_task, timeout=10.0)

    result = asyncio.run(_run_observed_timeout())
    assert result.terminal_status == "aborted"
    assert result.reason_code == "POLICY_ABORT"
    assert ready_file.exists() and ready_file.read_text(encoding="utf-8") == "ready"
    lines = marker.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    time.sleep(0.15)
    assert len(marker.read_text(encoding="utf-8").splitlines()) == 1

    # 3. Pre-start / already-expired deadline stops before launching command (zero launches/effects)
    marker.unlink()
    ready_file.unlink()
    pid_file.unlink()

    class _SteppingMonotonicClock:
        def __init__(self) -> None:
            self.calls = 0

        def now_ns(self) -> int:
            self.calls += 1
            if self.calls == 1:
                return 1_000_000_000
            return 100_000_000_000  # exceeds 100ms run_timeout_ms immediately on check

    res_early = asyncio.run(
        run_coding_task(
            task=_build_task(run_id="run-c20-early", run_timeout_ms=100),
            config=CodingConfig(output_root=str(out)),
            provider=SequenceProvider(_plan),
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=5),
            trace_writer=JsonlTraceStore(tmp_path / "trace_c20_early.jsonl"),
            monotonic_clock=_SteppingMonotonicClock(),
        )
    )
    assert res_early.terminal_status != "completed"
    assert not marker.exists()
    assert not ready_file.exists()

    # 4. Unparseable / bad planner outputs stop cleanly within planning budget with zero side effects
    bad_calls = 0

    class _BadPlanProvider:
        async def complete(self, req) -> ModelResponse:
            nonlocal bad_calls
            bad_calls += 1
            return ModelResponse(
                schema_version="0.1",
                request_id=req.request_id,
                status="ok",
                content="{not valid json at all",
            )

    trace_bad = JsonlTraceStore(tmp_path / "trace_c20_bad.jsonl")
    with pytest.raises(ForeContractRuntimeError) as exc_bad:
        asyncio.run(
            run_coding_task(
                task=_build_task(run_id="run-c20-bad", max_planning_calls=3, max_replans=1),
                config=CodingConfig(output_root=str(out)),
                provider=_BadPlanProvider(),
                planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=3),
                trace_writer=trace_bad,
            )
        )
    assert exc_bad.value.code == "DEPENDENCY_FAILURE"
    assert 1 <= bad_calls <= 3
    assert not marker.exists()
