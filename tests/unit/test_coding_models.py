"""Unit tests for Coding models, fixed v1 limits, and C01 pre-execution validation."""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest
from forecontract.tracing import JsonlTraceStore
from forecontract_llm import PlannerConfig
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (
    REPO_ROOT / "packages" / "core" / "src",
    REPO_ROOT / "packages" / "planners" / "llm" / "src",
    REPO_ROOT / "packages" / "adapters" / "coding" / "src",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract_coding import (
    HISTORY_EXCERPT_BYTES,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_HISTORY_ENTRIES,
    MAX_OBSERVATION_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_PROCESS_CLEANUP_MS,
    MAX_TOTAL_FILE_BYTES,
    CodingConfig,
    CodingError,
    CodingTask,
    CommandSpec,
    ExpectedTestCase,
    ReadFilePayload,
    RunCommandPayload,
    SearchTextPayload,
    VerifyTaskPayload,
    WriteFilePayload,
    WriteTarget,
    run_coding_task,
)


def _valid_command(tmp_path: Path, *, cmd_id: str = "accept") -> CommandSpec:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    return CommandSpec(
        command_id=cmd_id,
        executable=sys.executable,
        args=["-B", "-m", "pytest", "--junitxml={result_dir}/junit.xml", str(tmp_path / "ext.py")],
        cwd=str(ws),
        env={"PYTHONPATH": str(ws)},
        timeout_ms=5000,
        purpose="acceptance",
        expected_tests=[ExpectedTestCase(classname="ext", name="test_one")],
    )


def _valid_task_kwargs(tmp_path: Path) -> dict[str, object]:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "app.py").write_text("x = 1\n", encoding="utf-8")
    return {
        "schema_version": "0.1",
        "run_id": "run-c01",
        "goal_id": "goal-c01",
        "goal_text": "update app.py",
        "workspace_root": str(ws),
        "read_paths": ["app.py"],
        "write_paths": [WriteTarget(path="app.py", operation="modify")],
        "commands": {"accept": _valid_command(tmp_path)},
        "verification_command_ids": ["accept"],
        "max_steps": 5,
        "max_planning_calls": 5,
        "max_replans": 2,
        "run_timeout_ms": 15000,
    }


def test_c01_valid_task_and_config_defaults_accepted(tmp_path: Path) -> None:
    task = CodingTask(**_valid_task_kwargs(tmp_path))
    out = tmp_path / "out"
    config = CodingConfig(output_root=str(out))
    assert task.schema_version == "0.1"
    assert config.max_files == MAX_FILES
    assert config.max_total_file_bytes == MAX_TOTAL_FILE_BYTES
    assert config.max_file_bytes == MAX_FILE_BYTES
    assert config.max_observation_bytes == MAX_OBSERVATION_BYTES
    assert config.max_output_bytes == MAX_OUTPUT_BYTES
    assert config.max_history_entries == MAX_HISTORY_ENTRIES
    assert config.history_excerpt_bytes == HISTORY_EXCERPT_BYTES
    assert config.process_cleanup_ms == MAX_PROCESS_CLEANUP_MS


def test_c01_config_allows_tightening_and_rejects_relaxation(tmp_path: Path) -> None:
    out = str(tmp_path / "out")
    tight = CodingConfig(
        output_root=out,
        max_files=10,
        max_total_file_bytes=4096,
        max_file_bytes=1024,
        max_observation_bytes=4096,
        max_output_bytes=1024,
        max_history_entries=3,
        history_excerpt_bytes=256,
        process_cleanup_ms=500,
    )
    assert tight.max_files == 10

    for field_name, relaxed in (
        ("max_files", MAX_FILES + 1),
        ("max_total_file_bytes", MAX_TOTAL_FILE_BYTES + 1),
        ("max_file_bytes", MAX_FILE_BYTES + 1),
        ("max_observation_bytes", MAX_OBSERVATION_BYTES + 1),
        ("max_output_bytes", MAX_OUTPUT_BYTES + 1),
        ("max_history_entries", MAX_HISTORY_ENTRIES + 1),
        ("history_excerpt_bytes", HISTORY_EXCERPT_BYTES + 1),
        ("process_cleanup_ms", MAX_PROCESS_CLEANUP_MS + 1),
    ):
        with pytest.raises(ValidationError):
            CodingConfig(output_root=out, **{field_name: relaxed})


def test_c01_rejects_extra_fields_bool_budgets_duplicate_commands_and_dangling_ids(
    tmp_path: Path,
) -> None:
    base = _valid_task_kwargs(tmp_path)

    # Extra unknown field
    with pytest.raises(ValidationError):
        CodingTask(**{**base, "unexpected_field": True})

    # Boolean in integer budget positions
    for field_name in ("max_steps", "max_planning_calls", "max_replans", "run_timeout_ms"):
        with pytest.raises(ValidationError):
            CodingTask(**{**base, field_name: True})

    # Non-positive timeout
    with pytest.raises(ValidationError):
        CodingTask(**{**base, "run_timeout_ms": 0})

    # Mismatched commands dict key vs CommandSpec.command_id (case-insensitive duplicate too)
    cmd = _valid_command(tmp_path, cmd_id="Accept")
    with pytest.raises(ValidationError):
        CodingTask(**{**base, "commands": {"accept": cmd}})

    # Duplicate verification_command_ids
    with pytest.raises(ValidationError):
        CodingTask(**{**base, "verification_command_ids": ["accept", "accept"]})

    # Dangling verification_command_ids
    with pytest.raises(ValidationError):
        CodingTask(**{**base, "verification_command_ids": ["missing_cmd"]})

    # Diagnostic command referenced as verification_command_ids
    diag = CommandSpec(
        command_id="diag",
        executable=sys.executable,
        args=["-V"],
        cwd=str(tmp_path / "ws"),
        env={},
        timeout_ms=2000,
        purpose="diagnostic",
        expected_tests=[],
    )
    with pytest.raises(ValidationError):
        CodingTask(**{**base, "commands": {"diag": diag}, "verification_command_ids": ["diag"]})


def test_c01_command_spec_rejects_empty_acceptance_tests_credentials_and_bad_placeholders(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValidationError):
        CommandSpec(
            command_id="acc",
            executable=sys.executable,
            args=["--junitxml={result_dir}/junit.xml"],
            cwd=str(ws),
            timeout_ms=1000,
            purpose="acceptance",
            expected_tests=[],
        )
    with pytest.raises(ValidationError):
        CommandSpec(
            command_id="acc",
            executable=sys.executable,
            args=["--junitxml={other_var}/junit.xml", "x.py"],
            cwd=str(ws),
            timeout_ms=1000,
            purpose="acceptance",
            expected_tests=[ExpectedTestCase(classname="x", name="test_a")],
        )
    with pytest.raises(ValidationError):
        CommandSpec(
            command_id="diag",
            executable=sys.executable,
            args=["-V"],
            cwd=str(ws),
            env={"OPENAI_API_KEY": "secret"},
            timeout_ms=1000,
            purpose="diagnostic",
        )


def test_c01_payload_strict_models_reject_extra_fields_and_invalid_bounds() -> None:
    assert ReadFilePayload(path="a.py", start_line=1, max_lines=10).path == "a.py"
    assert SearchTextPayload(query="def", paths=["a.py"], max_matches=5).max_matches == 5
    assert WriteFilePayload(path="a.py", expected_sha256=None, content="x").content == "x"
    assert RunCommandPayload(command_id="diag").command_id == "diag"
    VerifyTaskPayload()

    with pytest.raises(ValidationError):
        RunCommandPayload(command_id="diag", args=["--extra"])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        VerifyTaskPayload(extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ReadFilePayload(path="a.py", start_line=True, max_lines=10)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CodingConfig(output_root="relative/out", max_files=math.nan)  # type: ignore[arg-type]


def test_c01_invalid_config_rejects_at_zero_provider_calls(tmp_path: Path) -> None:
    calls = 0

    class _CountingProvider:
        async def complete(self, request):
            nonlocal calls
            calls += 1
            raise AssertionError("provider must not be called")

    task = CodingTask(**_valid_task_kwargs(tmp_path))
    # output_root inside workspace_root must fail BEFORE provider/files/subprocess
    bad_cfg = CodingConfig(output_root=str(Path(task.workspace_root) / "nested_out"))
    with pytest.raises(CodingError):
        asyncio.run(
            run_coding_task(
                task=task,
                config=bad_cfg,
                provider=_CountingProvider(),
                planner_config=PlannerConfig(schema_version="0.1"),
                trace_writer=JsonlTraceStore(tmp_path / "trace.jsonl"),
            )
        )
    assert calls == 0


def test_d01_read_file_payload_byte_offset_and_expected_sha_validation() -> None:
    sha = "a" * 64
    default_req = ReadFilePayload(path="a.py", start_line=1, max_lines=10)
    assert default_req.line_byte_offset == 0
    assert default_req.expected_sha256 is None

    cont_req = ReadFilePayload(
        path="a.py",
        start_line=1,
        max_lines=10,
        line_byte_offset=16384,
        expected_sha256=sha,
    )
    assert cont_req.line_byte_offset == 16384
    assert cont_req.expected_sha256 == sha

    with pytest.raises(ValidationError):
        ReadFilePayload(path="a.py", start_line=1, max_lines=10, line_byte_offset=10)
    with pytest.raises(ValidationError):
        ReadFilePayload(path="a.py", start_line=1, max_lines=10, line_byte_offset=-1)
    with pytest.raises(ValidationError):
        ReadFilePayload(
            path="a.py",
            start_line=1,
            max_lines=10,
            line_byte_offset=True,  # type: ignore[arg-type]
            expected_sha256=sha,
        )
    with pytest.raises(ValidationError):
        ReadFilePayload(
            path="a.py",
            start_line=1,
            max_lines=10,
            line_byte_offset=0,
            expected_sha256="NOT-HEX",
        )


@pytest.mark.parametrize(
    "logical_cid",
    ["check-unit", "check:unit", "unit/tests", "CON", "cmd."],
)
def test_d03_command_spec_accepts_logical_command_ids(
    tmp_path: Path, logical_cid: str
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    spec = CommandSpec(
        command_id=logical_cid,
        executable=sys.executable,
        args=["-V"],
        cwd=str(ws),
        timeout_ms=1000,
        purpose="diagnostic",
    )
    assert spec.command_id == logical_cid


@pytest.mark.parametrize("blank_cid", ["", "   ", "\t\n"])
def test_d03_command_spec_rejects_blank_command_id(
    tmp_path: Path, blank_cid: str
) -> None:
    ws = tmp_path / "ws_blank"
    ws.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValidationError):
        CommandSpec(
            command_id=blank_cid,
            executable=sys.executable,
            args=["-V"],
            cwd=str(ws),
            timeout_ms=1000,
            purpose="diagnostic",
        )


@pytest.mark.parametrize("bad_action_id", ["../bad", "a/b", "CON", "cmd.", "cmd:1"])
def test_d03_validate_safe_single_segment_still_rejects_unsafe_action_id(
    bad_action_id: str,
) -> None:
    from forecontract_coding.models import validate_safe_single_segment

    with pytest.raises(ValueError):
        validate_safe_single_segment(bad_action_id, field_name="action_id")
