"""Integration tests for complete native Coding Agent loops (C10, C11, C16, C21, C24)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from forecontract.runtime.errors import RuntimeError as ForeContractRuntimeError
from forecontract.tracing import JsonlTraceStore
from forecontract_llm import LLMPlanner, ModelResponse, PlannerConfig

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
    INITIAL_CALC_SOURCE,
    INITIAL_METRICS_SOURCE,
    METRICS_ACCEPTANCE_TEST,
    ReactiveSumEvenProvider,
    SequenceProvider,
    VariantBReactiveProvider,
    make_sum_even_task,
    setup_sum_even_workspace,
)
from forecontract_coding import (
    CodingConfig,
    CodingTask,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
    run_coding_task,
)


def test_c10_and_c24_full_coding_loop_preserves_edit_and_trace(tmp_path: Path) -> None:
    ws, test_file, out = setup_sum_even_workspace(tmp_path)
    ext_before = test_file.read_bytes()
    before_sha = hashlib.sha256(INITIAL_CALC_SOURCE.encode("utf-8")).hexdigest()
    expected_after_sha = hashlib.sha256(FIXED_CALC_SOURCE.encode("utf-8")).hexdigest()

    task, cfg = make_sum_even_task(ws=ws, test_file=test_file, out=out)
    provider = ReactiveSumEvenProvider()
    trace_path = tmp_path / "trace.jsonl"
    result = asyncio.run(
        run_coding_task(
            task=task,
            config=cfg,
            provider=provider,
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=12),
            trace_writer=JsonlTraceStore(trace_path),
        )
    )

    assert result.terminal_status == "completed"
    assert result.acceptance.status == "passed"
    assert result.acceptance.revision == result.final_workspace_revision
    assert result.final_workspace_revision != result.baseline_workspace_revision
    assert len(provider.received_requests) == 5

    # Verify the fourth request (after diagnostic failure) received exit_code=1 + current file SHA
    diag_req = provider.received_requests[3]
    assert diag_req.observation.state["last_tool_result"]["exit_code"] == 1
    calc_entry = next(
        f for f in diag_req.observation.state["files"] if f["path"] == "calculator.py"
    )
    assert calc_entry["sha256"] != before_sha

    # C24: actual workspace file preserved, accurate diff, trace persisted, external test untouched
    assert (ws / "calculator.py").read_text(encoding="utf-8") == FIXED_CALC_SOURCE
    assert test_file.read_bytes() == ext_before
    assert not (ws / ".git").exists()
    assert len(result.changed_paths) == 1
    diff = result.changed_paths[0]
    assert diff.path == "calculator.py"
    assert diff.before_sha256 == before_sha
    assert diff.after_sha256 == expected_after_sha
    trace_lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(trace_lines) >= 5


def test_c11_variant_b_different_file_function_and_data(tmp_path: Path) -> None:
    ws = tmp_path / "ws_var_b"
    ext = tmp_path / "ext_var_b"
    out = tmp_path / "out_var_b"
    basetemp = out / "basetemp"
    (ws / "metrics").mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)
    basetemp.mkdir(parents=True, exist_ok=True)

    (ws / "metrics" / "__init__.py").write_text("", encoding="utf-8")
    (ws / "metrics" / "aggregator.py").write_text(INITIAL_METRICS_SOURCE, encoding="utf-8")
    test_file = ext / "test_metrics_acceptance.py"
    test_file.write_text(METRICS_ACCEPTANCE_TEST, encoding="utf-8")

    task = CodingTask(
        run_id="run-var-b",
        goal_id="goal-var-b",
        goal_text="Implement weighted_positive_total in metrics/aggregator.py.",
        workspace_root=str(ws),
        read_paths=["metrics/__init__.py", "metrics/aggregator.py"],
        write_paths=[WriteTarget(path="metrics/aggregator.py", operation="modify")],
        commands={
            "diag_metrics": CommandSpec(
                command_id="diag_metrics",
                executable=sys.executable,
                args=[
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--import-mode=importlib",
                    f"--rootdir={ext}",
                    f"--confcutdir={ext}",
                    f"--basetemp={basetemp}",
                    str(test_file),
                ],
                cwd=str(ws),
                env={"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                timeout_ms=15000,
                purpose="diagnostic",
            ),
            "acc_metrics": CommandSpec(
                command_id="acc_metrics",
                executable=sys.executable,
                args=[
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--import-mode=importlib",
                    f"--rootdir={ext}",
                    f"--confcutdir={ext}",
                    f"--basetemp={basetemp}",
                    "--junitxml={result_dir}/junit.xml",
                    str(test_file),
                ],
                cwd=str(ws),
                env={"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                timeout_ms=15000,
                purpose="acceptance",
                expected_tests=[
                    ExpectedTestCase(classname="test_metrics_acceptance", name="test_weighted_basic"),
                    ExpectedTestCase(
                        classname="test_metrics_acceptance", name="test_weighted_threshold_and_empty"
                    ),
                ],
            ),
        },
        verification_command_ids=["acc_metrics"],
        max_steps=12,
        max_planning_calls=12,
        max_replans=4,
        run_timeout_ms=60000,
    )
    provider = VariantBReactiveProvider()
    result = asyncio.run(
        run_coding_task(
            task=task,
            config=CodingConfig(output_root=str(out)),
            provider=provider,
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=12),
            trace_writer=JsonlTraceStore(tmp_path / "trace_b.jsonl"),
        )
    )
    assert result.terminal_status == "completed"
    assert result.acceptance.status == "passed"
    assert result.changed_paths[0].path == "metrics/aggregator.py"


def test_c16_developer_test_writable_cannot_override_external_acceptance(tmp_path: Path) -> None:
    ws, test_file, out = setup_sum_even_workspace(tmp_path)
    ext_before = test_file.read_bytes()

    task, cfg = make_sum_even_task(
        ws=ws,
        test_file=test_file,
        out=out,
        extra_read_paths=["tests_dev/test_local.py"],
        extra_write_targets=[WriteTarget(path="tests_dev/test_local.py", operation="create")],
    )

    def _plan(idx: int, req):
        files = {f["path"]: f for f in req.observation.state["files"]}
        if idx == 0:
            # Create relaxed dev test + leave bug in calculator.py
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "tests_dev/test_local.py",
                        "expected_sha256": None,
                        "content": "def test_trivially_green():\n    assert True\n",
                    },
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_file_written",
                        "payload": {"path": "tests_dev/test_local.py"},
                        "required": True,
                    }
                ],
            }
        if idx == 1:
            # External acceptance test STILL catches buggy calculator.py even though dev test is green!
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
        if idx == 2:
            # Now fix calculator.py genuinely
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

    result = asyncio.run(
        run_coding_task(
            task=task,
            config=cfg,
            provider=SequenceProvider(_plan),
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=10),
            trace_writer=JsonlTraceStore(tmp_path / "trace_c16.jsonl"),
        )
    )
    assert (ws / "tests_dev" / "test_local.py").exists()
    assert test_file.read_bytes() == ext_before
    assert result.terminal_status == "completed"


def test_c24_cli_demo_runs_offline_and_exits_zero(tmp_path: Path) -> None:
    demo = REPO_ROOT / "examples" / "coding_agent_demo.py"
    demo_root = tmp_path / "demo_run"
    env = dict(os.environ)
    pp = os.pathsep.join(
        [
            str(REPO_ROOT / "packages" / "core" / "src"),
            str(REPO_ROOT / "packages" / "planners" / "llm" / "src"),
            str(REPO_ROOT / "packages" / "adapters" / "coding" / "src"),
        ]
    )
    env["PYTHONPATH"] = pp
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-B", str(demo), "--root", str(demo_root)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["terminal_status"] == "completed"
    assert payload["acceptance"]["status"] == "passed"


def test_c21_and_d11_run_coding_task_context_and_lifecycle_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forecontract_coding import agent as agent_mod

    captured_contexts: list[object] = []
    planner_aclose_calls = 0
    adapter_stop_calls = 0

    orig_init = LLMPlanner.__init__
    orig_aclose = LLMPlanner.aclose
    orig_stop = agent_mod.CodingAdapter.stop

    def _spy_init(self, *, provider, context, config):
        captured_contexts.append(context)
        return orig_init(self, provider=provider, context=context, config=config)

    async def _spy_aclose(self):
        nonlocal planner_aclose_calls
        planner_aclose_calls += 1
        return await orig_aclose(self)

    async def _spy_stop(self):
        nonlocal adapter_stop_calls
        adapter_stop_calls += 1
        return await orig_stop(self)

    monkeypatch.setattr(LLMPlanner, "__init__", _spy_init)
    monkeypatch.setattr(LLMPlanner, "aclose", _spy_aclose)
    monkeypatch.setattr(agent_mod.CodingAdapter, "stop", _spy_stop)

    # 1. Normal completion path: PlannerContext matches CodingTask & borrowed provider is never closed
    ws1, test1, out1 = setup_sum_even_workspace(tmp_path / "normal")
    task1, cfg1 = make_sum_even_task(ws=ws1, test_file=test1, out=out1, run_id="run-c21-normal")

    class _NormalProvider(ReactiveSumEvenProvider):
        def __init__(self) -> None:
            super().__init__()
            self.provider_close_call_count = 0

        async def aclose(self) -> None:
            self.provider_close_call_count += 1

    p1 = _NormalProvider()
    res1 = asyncio.run(
        run_coding_task(
            task=task1,
            config=cfg1,
            provider=p1,
            planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=12),
            trace_writer=JsonlTraceStore(tmp_path / "trace_c21_1.jsonl"),
        )
    )
    assert res1.terminal_status == "completed"
    assert len(captured_contexts) == 1
    ctx = captured_contexts[0]
    assert ctx.run_id == task1.run_id
    assert ctx.goal_id == task1.goal_id
    assert ctx.task_criteria.goal_id == task1.goal_id
    assert "line_byte_offset" in ctx.action_guidance["read_file"]["properties"]
    assert "expected_sha256" in ctx.action_guidance["read_file"]["properties"]
    assert planner_aclose_calls == 1
    assert adapter_stop_calls >= 1
    assert p1.provider_close_call_count == 0

    # 2. Provider exception path: owned planner & adapter still close, borrowed provider untouched
    ws2, test2, out2 = setup_sum_even_workspace(tmp_path / "err")
    task2, cfg2 = make_sum_even_task(ws=ws2, test_file=test2, out=out2, run_id="run-c21-err")

    class _ExplodingProvider:
        def __init__(self) -> None:
            self.provider_close_call_count = 0

        async def complete(self, request) -> ModelResponse:
            raise RuntimeError("boom during complete")

        async def aclose(self) -> None:
            self.provider_close_call_count += 1

    p2 = _ExplodingProvider()
    before_aclose, before_stop = planner_aclose_calls, adapter_stop_calls
    with pytest.raises(ForeContractRuntimeError) as exc_p2:
        asyncio.run(
            run_coding_task(
                task=task2,
                config=cfg2,
                provider=p2,
                planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=4),
                trace_writer=JsonlTraceStore(tmp_path / "trace_c21_2.jsonl"),
            )
        )
    assert exc_p2.value.code == "DEPENDENCY_FAILURE"
    assert planner_aclose_calls == before_aclose + 1
    assert adapter_stop_calls >= before_stop + 1
    assert p2.provider_close_call_count == 0

    # 3. Host asyncio.Task cancellation path: owned planner & adapter still close cleanly
    ws3, test3, out3 = setup_sum_even_workspace(tmp_path / "cancel")
    task3, cfg3 = make_sum_even_task(ws=ws3, test_file=test3, out=out3, run_id="run-c21-cancel")

    class _BlockingProvider:
        def __init__(self) -> None:
            self.provider_close_call_count = 0
            self.entered = asyncio.Event()

        async def complete(self, request) -> ModelResponse:
            self.entered.set()
            await asyncio.sleep(10)
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            self.provider_close_call_count += 1

    p3 = _BlockingProvider()
    before_aclose3, before_stop3 = planner_aclose_calls, adapter_stop_calls

    async def _run_and_cancel() -> None:
        t = asyncio.create_task(
            run_coding_task(
                task=task3,
                config=cfg3,
                provider=p3,
                planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=4),
                trace_writer=JsonlTraceStore(tmp_path / "trace_c21_3.jsonl"),
            )
        )
        await p3.entered.wait()
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(_run_and_cancel())
    assert planner_aclose_calls == before_aclose3 + 1
    assert adapter_stop_calls >= before_stop3 + 1
    assert p3.provider_close_call_count == 0


def test_c01_c16_d03_logical_run_id_and_command_id_preserve_identity(tmp_path: Path) -> None:
    from forecontract_coding.models import validate_safe_single_segment

    for logical_run_id in ("team/session", "run-normal"):
        case_dir = tmp_path / hashlib.sha256(logical_run_id.encode("utf-8")).hexdigest()[:8]
        case_dir.mkdir()
        ws, test_file, out = setup_sum_even_workspace(case_dir)
        basetemp = out / "basetemp"
        basetemp.mkdir(parents=True, exist_ok=True)
        task = CodingTask(
            run_id=logical_run_id,
            goal_id="goal-logical-run",
            goal_text="Diagnose with logical command IDs, fix sum_even, and verify.",
            workspace_root=str(ws),
            read_paths=["calculator.py"],
            write_paths=[WriteTarget(path="calculator.py", operation="modify")],
            commands={
                "check:unit": CommandSpec(
                    command_id="check:unit",
                    executable=sys.executable,
                    args=["-c", "print('diag-ok')"],
                    cwd=str(ws),
                    timeout_ms=5000,
                    purpose="diagnostic",
                ),
                "unit/tests": CommandSpec(
                    command_id="unit/tests",
                    executable=sys.executable,
                    args=[
                        "-B",
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "--import-mode=importlib",
                        f"--rootdir={test_file.parent}",
                        f"--confcutdir={test_file.parent}",
                        f"--basetemp={basetemp}",
                        "--junitxml={result_dir}/junit.xml",
                        str(test_file),
                    ],
                    cwd=str(ws),
                    env={"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                    timeout_ms=15000,
                    purpose="acceptance",
                    expected_tests=[
                        ExpectedTestCase(classname="test_calc_acceptance", name="test_even_basic"),
                        ExpectedTestCase(
                            classname="test_calc_acceptance", name="test_even_edge_cases"
                        ),
                    ],
                ),
            },
            verification_command_ids=["unit/tests"],
            max_steps=8,
            max_planning_calls=8,
            max_replans=3,
            run_timeout_ms=60000,
        )

        def _plan(idx: int, req):
            files = {f["path"]: f for f in req.observation.state["files"]}
            if idx == 0:
                return {
                    "schema_version": "0.1",
                    "proposed_action": {
                        "schema_version": "0.1",
                        "action_type": "run_command",
                        "payload": {"command_id": "check:unit"},
                        "metadata": {},
                    },
                    "expected_outcomes": [
                        {
                            "schema_version": "0.1",
                            "outcome_type": "coding_tool_result",
                            "payload": {
                                "action_type": "run_command",
                                "command_id": "check:unit",
                            },
                            "required": True,
                        }
                    ],
                }
            if idx == 1:
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

        trace_path = case_dir / "trace.jsonl"
        result = asyncio.run(
            run_coding_task(
                task=task,
                config=CodingConfig(output_root=str(out)),
                provider=SequenceProvider(_plan),
                planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=8),
                trace_writer=JsonlTraceStore(trace_path),
            )
        )
        assert result.terminal_status == "completed"
        assert result.acceptance.status == "passed"
        assert result.acceptance.command_ids == ["unit/tests"]
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert events
        assert all(ev["run_id"] == logical_run_id for ev in events)
        assert result.acceptance.action_id is not None
        assert (
            validate_safe_single_segment(result.acceptance.action_id, field_name="action_id")
            == result.acceptance.action_id
        )
        cmd_dirs = list((out / "cmd_results").iterdir())
        verify_dirs = list((out / "verify").iterdir())
        assert len(cmd_dirs) == 1
        assert len(verify_dirs) == 1
        for d in (*cmd_dirs, *verify_dirs):
            safe_aid = d.name.split("__", 1)[0]
            assert validate_safe_single_segment(safe_aid, field_name="action_id") == safe_aid
            assert "/" not in safe_aid
        assert (ws / "calculator.py").read_text(encoding="utf-8") == FIXED_CALC_SOURCE
