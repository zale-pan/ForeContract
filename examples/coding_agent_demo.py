"""Offline CLI demo exercising ``forecontract_coding.run_coding_task`` cleanly."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from forecontract.tracing import JsonlTraceStore
from forecontract_coding import (
    CodingConfig,
    CodingTask,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
    run_coding_task,
)
from forecontract_llm import ModelRequest, ModelResponse, PlannerConfig

_INITIAL_CALC = '''"""Synthetic demo calculator module."""


def sum_even(values):
    """Return the sum of even integers in values."""
    raise NotImplementedError("TODO")
'''

_ACCEPTANCE_TEST = """import calculator


def test_basic_even():
    assert calculator.sum_even([1, 2, 3, 4]) == 6


def test_empty_and_negatives():
    assert calculator.sum_even([]) == 0
    assert calculator.sum_even([-4, -3, 0, 5]) == -4
"""

_CORRECT_CALC = '''"""Synthetic demo calculator module."""


def sum_even(values):
    """Return the sum of even integers in values."""
    return sum(x for x in values if x % 2 == 0)
'''


class OfflineFeedbackDemoProvider:
    """Offline reactive provider reading real file SHA and pytest diagnostics from ModelRequest."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._fixed_bug = False

    def _plan_json(self, req: ModelRequest) -> dict[str, object]:
        state = req.observation.state
        last = state.get("last_tool_result")
        files = {entry["path"]: entry for entry in state["files"]}
        acc = state["acceptance"]

        if last is None:
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "read_file",
                    "payload": {"path": "calculator.py", "start_line": 1, "max_lines": 50},
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_tool_result",
                        "payload": {"action_type": "read_file", "path": "calculator.py"},
                        "required": True,
                    }
                ],
            }

        if not self._fixed_bug and last.get("action_type") == "read_file":
            current_sha = files["calculator.py"]["sha256"]
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "calculator.py",
                        "expected_sha256": current_sha,
                        "content": (
                            '"""Synthetic demo calculator module."""\n\n\n'
                            "def sum_even(values):\n"
                            "    return sum(values)\n"
                        ),
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

        if not self._fixed_bug and last.get("action_type") == "write_file":
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "run_command",
                    "payload": {"command_id": "pytest_unit"},
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_tool_result",
                        "payload": {"action_type": "run_command", "command_id": "pytest_unit"},
                        "required": True,
                    }
                ],
            }

        if not self._fixed_bug and last.get("action_type") == "run_command":
            self._fixed_bug = True
            current_sha = files["calculator.py"]["sha256"]
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "calculator.py",
                        "expected_sha256": current_sha,
                        "content": _CORRECT_CALC,
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

        if acc.get("status") != "passed":
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
        raise RuntimeError("unexpected extra planning call after pass")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            schema_version="0.1",
            request_id=request.request_id,
            status="ok",
            content=json.dumps(self._plan_json(request)),
        )


def _setup_demo_dir(root: Path) -> tuple[Path, Path, Path, Path]:
    ws = root / "workspace"
    ext = root / "external_tests"
    out = root / "output"
    trace = root / "trace.jsonl"
    ws.mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (ws / "calculator.py").write_text(_INITIAL_CALC, encoding="utf-8")
    test_file = ext / "test_acceptance.py"
    test_file.write_text(_ACCEPTANCE_TEST, encoding="utf-8")
    return ws, test_file, out, trace


async def run_demo(root: Path) -> dict[str, object]:
    """Run the complete offline coding demo inside ``root`` and return JSON-safe summary."""
    ws, test_file, out, trace_path = _setup_demo_dir(root)
    basetemp = out / "pytest_tmp"
    basetemp.mkdir(parents=True, exist_ok=True)
    task = CodingTask(
        schema_version="0.1",
        run_id="demo-coding-run-1",
        goal_id="demo-coding-goal-1",
        goal_text="Implement sum_even in calculator.py so all external tests pass.",
        workspace_root=str(ws),
        read_paths=["calculator.py"],
        write_paths=[WriteTarget(path="calculator.py", operation="modify")],
        commands={
            "pytest_unit": CommandSpec(
                command_id="pytest_unit",
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
                    ExpectedTestCase(classname="test_acceptance", name="test_basic_even"),
                    ExpectedTestCase(classname="test_acceptance", name="test_empty_and_negatives"),
                ],
            )
        },
        verification_command_ids=["pytest_unit"],
        max_steps=10,
        max_planning_calls=10,
        max_replans=3,
        run_timeout_ms=60000,
    )
    config = CodingConfig(schema_version="0.1", output_root=str(out))
    provider = OfflineFeedbackDemoProvider()
    planner_cfg = PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=10)
    trace_writer = JsonlTraceStore(trace_path)
    result = await run_coding_task(
        task=task,
        config=config,
        provider=provider,
        planner_config=planner_cfg,
        trace_writer=trace_writer,
    )
    return result.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ForeContract offline Coding Agent demo")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional isolated working root; defaults to temporary directory under FORECONTRACT_CODING_SCRATCH or system temp.",
    )
    args = parser.parse_args(argv)
    if args.root is not None:
        summary = asyncio.run(run_demo(args.root))
    else:
        scratch = os.environ.get("FORECONTRACT_CODING_SCRATCH")
        parent = Path(scratch) if scratch else None
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fc_coding_demo_", dir=parent) as tmp:
            summary = asyncio.run(run_demo(Path(tmp)))
    print(json.dumps(summary, indent=2))
    return 0 if summary["terminal_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
