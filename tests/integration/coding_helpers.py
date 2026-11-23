"""Shared offline synthetic workspace fixtures and reactive test providers (C10-C16)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from forecontract_coding import (
    CodingConfig,
    CodingTask,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
)
from forecontract_llm import ModelRequest, ModelResponse

INITIAL_CALC_SOURCE = '''"""Synthetic calculator fixture."""


def sum_even(values):
    """Return the sum of even numbers in ``values``."""
    raise NotImplementedError("TODO")
'''

FIXED_CALC_SOURCE = '''"""Synthetic calculator fixture."""


def sum_even(values):
    """Return the sum of even numbers in ``values``."""
    return sum(x for x in values if x % 2 == 0)
'''

CALC_ACCEPTANCE_TEST = """import calculator


def test_even_basic():
    assert calculator.sum_even([1, 2, 3, 4]) == 6


def test_even_edge_cases():
    assert calculator.sum_even([]) == 0
    assert calculator.sum_even([-4, -3, 0, 5]) == -4
    assert calculator.sum_even(x for x in [2, 3, 8, 9]) == 10
"""

INITIAL_METRICS_SOURCE = '''"""Variant B metrics fixture."""


def weighted_positive_total(pairs, min_weight=2):
    """Sum value * weight for positive items whose weight >= min_weight."""
    raise NotImplementedError("TODO")
'''

METRICS_ACCEPTANCE_TEST = """from metrics.aggregator import weighted_positive_total


def test_weighted_basic():
    assert weighted_positive_total([(1, 10), (2, 5), (3, -4)]) == 10


def test_weighted_threshold_and_empty():
    assert weighted_positive_total([], min_weight=3) == 0
    assert weighted_positive_total([(4, 7), (2, 7)], min_weight=3) == 28
"""


def setup_sum_even_workspace(root: Path) -> tuple[Path, Path, Path]:
    """Create isolated Variant A workspace + external acceptance test + output dir."""
    ws = root / "ws_sum_even"
    ext = root / "ext_sum_even"
    out = root / "out_sum_even"
    ws.mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (ws / "calculator.py").write_bytes(INITIAL_CALC_SOURCE.encode("utf-8"))
    test_file = ext / "test_calc_acceptance.py"
    test_file.write_bytes(CALC_ACCEPTANCE_TEST.encode("utf-8"))
    return ws, test_file, out


def make_sum_even_task(
    *,
    ws: Path,
    test_file: Path,
    out: Path,
    run_id: str = "run-sum-even",
    extra_commands: dict[str, CommandSpec] | None = None,
    extra_write_targets: list[WriteTarget] | None = None,
    extra_read_paths: list[str] | None = None,
    max_steps: int = 12,
    max_planning_calls: int = 12,
    max_replans: int = 4,
    run_timeout_ms: int = 60000,
) -> tuple[CodingTask, CodingConfig]:
    basetemp = out / "basetemp"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands: dict[str, CommandSpec] = {
        "pytest_acc": CommandSpec(
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
                ExpectedTestCase(classname="test_calc_acceptance", name="test_even_edge_cases"),
            ],
        ),
        "pytest_diag": CommandSpec(
            command_id="pytest_diag",
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
                str(test_file),
            ],
            cwd=str(ws),
            env={"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            timeout_ms=15000,
            purpose="diagnostic",
        ),
    }
    if extra_commands:
        commands.update(extra_commands)

    read_paths = ["calculator.py", *(extra_read_paths or [])]
    write_paths = [
        WriteTarget(path="calculator.py", operation="modify"),
        *(extra_write_targets or []),
    ]
    task = CodingTask(
        run_id=run_id,
        goal_id=f"goal-{run_id}",
        goal_text="Implement sum_even(values) in calculator.py to pass external tests.",
        workspace_root=str(ws),
        read_paths=read_paths,
        write_paths=write_paths,
        commands=commands,
        verification_command_ids=["pytest_acc"],
        max_steps=max_steps,
        max_planning_calls=max_planning_calls,
        max_replans=max_replans,
        run_timeout_ms=run_timeout_ms,
    )
    return task, CodingConfig(output_root=str(out))


class ReactiveSumEvenProvider:
    """Offline ModelProvider implementing real read -> buggy write -> pytest exit1 -> fix -> verify."""

    def __init__(self) -> None:
        self.received_requests: list[ModelRequest] = []
        self._wrote_buggy = False
        self._ran_diag = False
        self._wrote_fixed = False

    def _build_body(self, req: ModelRequest) -> dict[str, Any]:
        state = req.observation.state
        last = state.get("last_tool_result")
        files = {entry["path"]: entry for entry in state["files"]}

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

        if not self._wrote_buggy:
            self._wrote_buggy = True
            sha = files["calculator.py"]["sha256"]
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "calculator.py",
                        "expected_sha256": sha,
                        "content": "def sum_even(values):\n    return sum(values)\n",
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

        if not self._ran_diag:
            self._ran_diag = True
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "run_command",
                    "payload": {"command_id": "pytest_diag"},
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_tool_result",
                        "payload": {"action_type": "run_command", "command_id": "pytest_diag"},
                        "required": True,
                    }
                ],
            }

        if not self._wrote_fixed:
            # Must react to actual exit_code=1 diagnostic feedback carried inside observation
            assert last["action_type"] == "run_command"
            assert last["exit_code"] == 1
            assert "assert 10 == 6" in last["stdout"]
            self._wrote_fixed = True
            sha = files["calculator.py"]["sha256"]
            return {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": "calculator.py",
                        "expected_sha256": sha,
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

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.received_requests.append(request)
        return ModelResponse(
            schema_version="0.1",
            request_id=request.request_id,
            status="ok",
            content=json.dumps(self._build_body(request)),
        )


class VariantBReactiveProvider:
    """Variant B provider discovering path, function, and fix purely from ModelRequest."""

    def __init__(self) -> None:
        self.received_requests: list[ModelRequest] = []
        self._step = 0
        self._discovered_path: str | None = None
        self._discovered_fn: str | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.received_requests.append(request)
        state = request.observation.state
        files = {entry["path"]: entry for entry in state["files"]}
        last = state.get("last_tool_result")
        writable_paths = [p for p, info in files.items() if info["writable"]]
        assert len(writable_paths) == 1
        target_path = writable_paths[0]

        if self._step == 0:
            self._step += 1
            body = {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "search_text",
                    "payload": {"query": "NotImplementedError", "paths": list(files.keys()), "max_matches": 5},
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_tool_result",
                        "payload": {"action_type": "search_text"},
                        "required": True,
                    }
                ],
            }
        elif self._step == 1:
            assert last and last["action_type"] == "search_text" and last["matches"]
            self._discovered_path = last["matches"][0]["path"]
            self._step += 1
            body = {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "read_file",
                    "payload": {"path": self._discovered_path, "start_line": 1, "max_lines": 50},
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_tool_result",
                        "payload": {"action_type": "read_file", "path": self._discovered_path},
                        "required": True,
                    }
                ],
            }
        elif self._step == 2:
            assert last and last["action_type"] == "read_file"
            for line in last["content"].splitlines():
                if line.startswith("def "):
                    self._discovered_fn = line.split("def ", 1)[1].split("(", 1)[0]
                    break
            assert self._discovered_fn is not None
            self._step += 1
            body = {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": target_path,
                        "expected_sha256": files[target_path]["sha256"],
                        "content": f"def {self._discovered_fn}(pairs, min_weight=2):\n    return sum(w * v for w, v in pairs)\n",
                    },
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_file_written",
                        "payload": {"path": target_path},
                        "required": True,
                    }
                ],
            }
        elif self._step == 3:
            self._step += 1
            body = {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "run_command",
                    "payload": {"command_id": "diag_metrics"},
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_tool_result",
                        "payload": {"action_type": "run_command", "command_id": "diag_metrics"},
                        "required": True,
                    }
                ],
            }
        elif self._step == 4:
            assert last and last["exit_code"] == 1
            self._step += 1
            assert self._discovered_fn is not None
            body = {
                "schema_version": "0.1",
                "proposed_action": {
                    "schema_version": "0.1",
                    "action_type": "write_file",
                    "payload": {
                        "path": target_path,
                        "expected_sha256": files[target_path]["sha256"],
                        "content": (
                            f"def {self._discovered_fn}(pairs, min_weight=2):\n"
                            "    return sum(w * v for w, v in pairs if v > 0 and w >= min_weight)\n"
                        ),
                    },
                    "metadata": {},
                },
                "expected_outcomes": [
                    {
                        "schema_version": "0.1",
                        "outcome_type": "coding_file_written",
                        "payload": {"path": target_path},
                        "required": True,
                    }
                ],
            }
        else:
            body = {
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

        return ModelResponse(
            schema_version="0.1",
            request_id=request.request_id,
            status="ok",
            content=json.dumps(body),
        )


class SequenceProvider:
    """Test helper that replays a callable sequence of response payloads per turn."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self.received_requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.received_requests.append(request)
        body = self._handler(len(self.received_requests) - 1, request)
        return ModelResponse(
            schema_version="0.1",
            request_id=request.request_id,
            status="ok",
            content=body if isinstance(body, str) else json.dumps(body),
        )
