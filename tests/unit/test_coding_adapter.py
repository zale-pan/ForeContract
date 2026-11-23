"""Unit tests for ``CodingAdapter`` covering C02, C06, C07, C14, C15, C18, and C19."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from forecontract.models import ActionSpec, ExecutionStatus
from forecontract.protocols.adapter import (
    InvalidActionPayloadError,
    UnsupportedActionError,
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
    CodingAdapter,
    CodingConfig,
    CodingError,
    CodingTask,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
)


def _setup_adapter(
    tmp_path: Path,
    *,
    test_body: str = "import mod\ndef test_val(): assert mod.VAL == 42\n",
    expected_tests: list[ExpectedTestCase] | None = None,
    max_observation_bytes: int = 64 * 1024,
    max_history_entries: int = 8,
    history_excerpt_bytes: int = 2048,
) -> tuple[CodingAdapter, Path, Path, Path]:
    ws = tmp_path / "ws"
    out = tmp_path / "out"
    ext = tmp_path / "ext"
    ws.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)
    (ws / "mod.py").write_bytes(b"VAL = 0\n")
    test_file = ext / "test_ext.py"
    test_file.write_text(test_body, encoding="utf-8")
    basetemp = out / "basetemp"
    basetemp.mkdir(parents=True, exist_ok=True)

    tests = (
        expected_tests
        if expected_tests is not None
        else [ExpectedTestCase(classname="test_ext", name="test_val")]
    )
    commands = {
        "accept": CommandSpec(
            command_id="accept",
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
            timeout_ms=10000,
            purpose="acceptance",
            expected_tests=tests,
        ),
        "diag_exit1": CommandSpec(
            command_id="diag_exit1",
            executable=sys.executable,
            args=["-c", "import sys; print('fail detail'); sys.exit(1)"],
            cwd=str(ws),
            timeout_ms=5000,
            purpose="diagnostic",
        ),
        "diag_echo_argv": CommandSpec(
            command_id="diag_echo_argv",
            executable=sys.executable,
            args=["-c", "import sys; print('FIXED-OK')"],
            cwd=str(ws),
            timeout_ms=5000,
            purpose="diagnostic",
        ),
    }
    task = CodingTask(
        run_id="run-adp",
        goal_id="goal-adp",
        goal_text="adapter verification",
        workspace_root=str(ws),
        read_paths=["mod.py"],
        write_paths=[WriteTarget(path="mod.py", operation="modify")],
        commands=commands,
        verification_command_ids=["accept"],
        max_steps=10,
        max_planning_calls=10,
        max_replans=3,
        run_timeout_ms=60000,
    )
    cfg = CodingConfig(
        output_root=str(out),
        max_observation_bytes=max_observation_bytes,
        max_history_entries=max_history_entries,
        history_excerpt_bytes=history_excerpt_bytes,
    )
    adapter = CodingAdapter(task=task, config=cfg)
    return adapter, ws, out, test_file


def _act(action_type: str, payload: dict[str, object], aid: str) -> ActionSpec:
    return ActionSpec(
        schema_version="0.1",
        action_type=action_type,
        payload=payload,
        metadata={"action_id": aid},
    )


def test_c02_dynamic_write_file_allows_distinct_contents_without_preregistration(
    tmp_path: Path,
) -> None:
    adapter, ws, _, _ = _setup_adapter(tmp_path)

    async def _run() -> None:
        await adapter.start()
        try:
            obs0 = await adapter.observe()
            sha0 = obs0.state["files"][0]["sha256"]
            out1 = await adapter.execute(
                _act("write_file", {"path": "mod.py", "expected_sha256": sha0, "content": "VAL = 11\n"}, "a1")
            )
            assert out1.status is ExecutionStatus.SUCCEEDED
            assert (ws / "mod.py").read_text(encoding="utf-8") == "VAL = 11\n"

            sha1 = hashlib.sha256(b"VAL = 11\n").hexdigest()
            out2 = await adapter.execute(
                _act("write_file", {"path": "mod.py", "expected_sha256": sha1, "content": "VAL = 99\n"}, "a2")
            )
            assert out2.status is ExecutionStatus.SUCCEEDED
            assert (ws / "mod.py").read_text(encoding="utf-8") == "VAL = 99\n"
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_c06_rejects_command_injection_and_unknown_command_id(tmp_path: Path) -> None:
    adapter, _, _, _ = _setup_adapter(tmp_path)

    with pytest.raises(UnsupportedActionError):
        adapter.validate_action(_act("shell", {"cmd": "dir"}, "a1"))

    with pytest.raises(InvalidActionPayloadError):
        adapter.validate_action(
            _act("run_command", {"command_id": "diag_echo_argv", "args": ["-V"]}, "a1")
        )
    with pytest.raises(InvalidActionPayloadError):
        adapter.validate_action(
            _act("run_command", {"command_id": "diag_echo_argv", "env": {"X": "1"}}, "a2")
        )
    with pytest.raises(InvalidActionPayloadError):
        adapter.validate_action(
            _act("run_command", {"command_id": "diag_echo_argv", "cwd": "C:\\"}, "a3")
        )
    with pytest.raises(InvalidActionPayloadError):
        adapter.validate_action(_act("run_command", {"command_id": "unknown_cmd"}, "a4"))

    async def _run() -> None:
        await adapter.start()
        try:
            res = await adapter.execute(
                _act("run_command", {"command_id": "diag_echo_argv"}, "a5")
            )
            assert res.status is ExecutionStatus.SUCCEEDED
            obs = await adapter.observe()
            assert obs.state["last_tool_result"]["stdout"].strip() == "FIXED-OK"
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_c07_and_c18_exit1_continues_and_action_id_replay_forbidden(tmp_path: Path) -> None:
    adapter, _, _, _ = _setup_adapter(tmp_path)

    async def _run() -> None:
        await adapter.start()
        try:
            res = await adapter.execute(
                _act("run_command", {"command_id": "diag_exit1"}, "dup-id")
            )
            assert res.status is ExecutionStatus.SUCCEEDED
            obs = await adapter.observe()
            assert obs.state["last_tool_result"]["exit_code"] == 1
            assert obs.state["acceptance"]["status"] == "not_run"

            with pytest.raises(InvalidActionPayloadError):
                adapter.validate_action(
                    _act("read_file", {"path": "mod.py", "start_line": 1, "max_lines": 5}, "dup-id")
                )
            with pytest.raises(InvalidActionPayloadError):
                await adapter.execute(
                    _act("read_file", {"path": "mod.py", "start_line": 1, "max_lines": 5}, "dup-id")
                )
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_c14_skipped_and_zero_test_cases_never_pass(tmp_path: Path) -> None:
    # 1. All skipped test case
    adapter_skip, _ws, _, _ = _setup_adapter(
        tmp_path / "skip_case",
        test_body="import pytest\n@pytest.mark.skip(reason='skip')\ndef test_val(): pass\n",
    )

    async def _run_skip() -> None:
        await adapter_skip.start()
        try:
            await adapter_skip.execute(_act("verify_task", {}, "v1"))
            obs = await adapter_skip.observe()
            assert obs.state["acceptance"]["status"] == "failed"
        finally:
            await adapter_skip.stop()

    asyncio.run(_run_skip())

    # 2. Zero collected test cases (exit 5 from pytest or mismatched empty execution)
    adapter_zero, _, _, _ = _setup_adapter(
        tmp_path / "zero_case",
        test_body="# empty file with no tests\n",
    )

    async def _run_zero() -> None:
        await adapter_zero.start()
        try:
            await adapter_zero.execute(_act("verify_task", {}, "v2"))
            obs = await adapter_zero.observe()
            assert obs.state["acceptance"]["status"] != "passed"
        finally:
            await adapter_zero.stop()

    asyncio.run(_run_zero())


def test_c15_post_verify_edit_invalidates_acceptance_immediately(tmp_path: Path) -> None:
    adapter, ws, _, _ = _setup_adapter(tmp_path)

    async def _run() -> None:
        await adapter.start()
        try:
            sha0 = hashlib.sha256(b"VAL = 0\n").hexdigest()
            await adapter.execute(
                _act("write_file", {"path": "mod.py", "expected_sha256": sha0, "content": "VAL = 42\n"}, "w1")
            )
            await adapter.execute(_act("verify_task", {}, "v1"))
            obs_pass = await adapter.observe()
            assert obs_pass.state["acceptance"]["status"] == "passed"

            # External modification breaks passed acceptance immediately upon next observe()
            (ws / "mod.py").write_bytes(b"VAL = 0\n")
            obs_stale = await adapter.observe()
            assert obs_stale.state["acceptance"]["status"] == "unknown"
            assert obs_stale.state["acceptance"]["stale_reason"] == "workspace_revision_changed"
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_c19_bounded_history_and_observation_overflow_degrades_safely(tmp_path: Path) -> None:
    # 1. History > 8 preserves max_history_entries and history_truncated flag
    adapter, _, _, _ = _setup_adapter(tmp_path / "hist", max_history_entries=4, history_excerpt_bytes=64)

    async def _run_hist() -> None:
        await adapter.start()
        try:
            for i in range(6):
                await adapter.execute(
                    _act("read_file", {"path": "mod.py", "start_line": 1, "max_lines": 5}, f"r-{i}")
                )
            obs = await adapter.observe()
            assert len(obs.state["history"]) == 4
            assert obs.state["history_truncated"] is True
            assert obs.state["omitted_history_count"] == 2
        finally:
            await adapter.stop()

    asyncio.run(_run_hist())

    # 2. Tight max_observation_bytes with multi-file workspace triggers explicit incomplete overflow observation
    ws_ov = tmp_path / "ov_ws"
    out_ov = tmp_path / "ov_out"
    ext_ov = tmp_path / "ov_ext"
    bt_ov = out_ov / "bt"
    ws_ov.mkdir(parents=True)
    ext_ov.mkdir(parents=True)
    bt_ov.mkdir(parents=True)
    extra_names = [f"module_{i}.py" for i in range(6)]
    for name in extra_names:
        (ws_ov / name).write_bytes(b"VAL = 0\n")
    tfile_ov = ext_ov / "test_ov.py"
    tfile_ov.write_text("def test_ok(): assert True\n", encoding="utf-8")
    task_ov = CodingTask(
        run_id="run-ov",
        goal_id="goal-ov",
        goal_text="overflow check",
        workspace_root=str(ws_ov),
        read_paths=extra_names,
        write_paths=[WriteTarget(path=extra_names[0], operation="modify")],
        commands={
            "accept": CommandSpec(
                command_id="accept",
                executable=sys.executable,
                args=[
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    f"--rootdir={ext_ov}",
                    f"--confcutdir={ext_ov}",
                    f"--basetemp={bt_ov}",
                    "--junitxml={result_dir}/junit.xml",
                    str(tfile_ov),
                ],
                cwd=str(ws_ov),
                timeout_ms=5000,
                purpose="acceptance",
                expected_tests=[ExpectedTestCase(classname="test_ov", name="test_ok")],
            )
        },
        verification_command_ids=["accept"],
        max_steps=5,
        max_planning_calls=5,
        max_replans=1,
        run_timeout_ms=10000,
    )
    adapter_ov = CodingAdapter(
        task=task_ov,
        config=CodingConfig(output_root=str(out_ov), max_observation_bytes=1200),
    )

    async def _run_overflow() -> None:
        await adapter_ov.start()
        try:
            obs = await adapter_ov.observe()
            assert obs.state["observation_overflow"] is True
            assert obs.state["acceptance"]["status"] == "unknown"
            assert len(json.dumps(obs.model_dump(mode="json")).encode("utf-8")) <= 1200
        finally:
            await adapter_ov.stop()

    asyncio.run(_run_overflow())

    # 3. Pathologically tiny max_observation_bytes refuses silent omission with CODING_OBSERVATION_OVERFLOW
    adapter_tiny, _, _, _ = _setup_adapter(tmp_path / "tiny", max_observation_bytes=512)

    async def _run_tiny() -> None:
        await adapter_tiny.start()
        try:
            with pytest.raises(CodingError) as exc_info:
                await adapter_tiny.observe()
            assert exc_info.value.code == "CODING_OBSERVATION_OVERFLOW"
        finally:
            await adapter_tiny.stop()

    asyncio.run(_run_tiny())


def test_d02_adapter_read_file_exposes_next_line_byte_offset_in_last_tool_result(
    tmp_path: Path,
) -> None:
    adapter, ws, _, _ = _setup_adapter(tmp_path)
    long_line = ("x" * 17000) + "-tail\n"
    (ws / "mod.py").write_bytes(long_line.encode("utf-8"))
    sha = hashlib.sha256(long_line.encode("utf-8")).hexdigest()

    async def _run() -> None:
        await adapter.start()
        try:
            r1 = await adapter.execute(
                _act("read_file", {"path": "mod.py", "start_line": 1, "max_lines": 1}, "rd-1")
            )
            assert r1.status is ExecutionStatus.SUCCEEDED
            obs1 = await adapter.observe()
            last1 = obs1.state["last_tool_result"]
            assert last1["next_start_line"] == 1
            assert isinstance(last1["next_line_byte_offset"], int)
            assert last1["next_line_byte_offset"] > 0

            r2 = await adapter.execute(
                _act(
                    "read_file",
                    {
                        "path": "mod.py",
                        "start_line": 1,
                        "max_lines": 1,
                        "line_byte_offset": last1["next_line_byte_offset"],
                        "expected_sha256": sha,
                    },
                    "rd-2",
                )
            )
            assert r2.status is ExecutionStatus.SUCCEEDED
            obs2 = await adapter.observe()
            last2 = obs2.state["last_tool_result"]
            assert last2["truncated"] is False
            assert last2["next_line_byte_offset"] is None
            assert last1["content"] + last2["content"] == long_line
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_d03_logical_command_id_encoded_executes_and_preserves_identity(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws_logical"
    out = tmp_path / "out_logical"
    ext = tmp_path / "ext_logical"
    ws.mkdir()
    out.mkdir()
    ext.mkdir()
    (ws / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    ext_test = ext / "test_acc.py"
    ext_test.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    basetemp = out / "basetemp"
    basetemp.mkdir(parents=True, exist_ok=True)

    task = CodingTask(
        run_id="team/session",
        goal_id="goal-logical-cmd",
        goal_text="Execute commands with logical IDs cleanly.",
        workspace_root=str(ws),
        read_paths=["app.py"],
        write_paths=[],
        commands={
            "check:unit": CommandSpec(
                command_id="check:unit",
                executable=sys.executable,
                args=[
                    "-c",
                    (
                        "import pathlib, sys; "
                        "p = pathlib.Path(sys.argv[1]); "
                        "p.write_text('check:unit', encoding='utf-8'); "
                        "print('ok:check:unit')"
                    ),
                    "{result_dir}/marker.txt",
                ],
                cwd=str(ws),
                timeout_ms=5000,
                purpose="diagnostic",
            ),
            "unit/tests": CommandSpec(
                command_id="unit/tests",
                executable=sys.executable,
                args=[
                    "-c",
                    (
                        "import pathlib, sys; "
                        "p = pathlib.Path(sys.argv[1]); "
                        "p.write_text('unit/tests', encoding='utf-8'); "
                        "print('ok:unit/tests')"
                    ),
                    "{result_dir}/marker.txt",
                ],
                cwd=str(ws),
                timeout_ms=5000,
                purpose="diagnostic",
            ),
            "acc:default": CommandSpec(
                command_id="acc:default",
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
                    str(ext_test),
                ],
                cwd=str(ws),
                env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                timeout_ms=10000,
                purpose="acceptance",
                expected_tests=[ExpectedTestCase(classname="test_acc", name="test_ok")],
            ),
        },
        verification_command_ids=["acc:default"],
        max_steps=8,
        max_planning_calls=8,
        max_replans=3,
        run_timeout_ms=30000,
    )
    adapter = CodingAdapter(task=task, config=CodingConfig(output_root=str(out)))

    async def _run() -> None:
        await adapter.start()
        try:
            res_colon = await adapter.execute(
                _act("run_command", {"command_id": "check:unit"}, "act-cmd-1")
            )
            assert res_colon.status is ExecutionStatus.SUCCEEDED
            assert res_colon.state_after["command_id"] == "check:unit"
            obs1 = await adapter.observe()
            assert obs1.state["last_tool_result"]["command_id"] == "check:unit"
            assert obs1.state["history"][-1]["target"] == "check:unit"

            res_slash = await adapter.execute(
                _act("run_command", {"command_id": "unit/tests"}, "act-cmd-2")
            )
            assert res_slash.status is ExecutionStatus.SUCCEEDED
            assert res_slash.state_after["command_id"] == "unit/tests"
            obs2 = await adapter.observe()
            assert obs2.state["last_tool_result"]["command_id"] == "unit/tests"
            assert obs2.state["history"][-1]["target"] == "unit/tests"

            res_verify = await adapter.execute(_act("verify_task", {}, "act-ver-1"))
            assert res_verify.status is ExecutionStatus.SUCCEEDED
            obs3 = await adapter.observe()
            assert obs3.state["acceptance"]["status"] == "passed"
            assert obs3.state["acceptance"]["command_ids"] == ["acc:default"]
            assert obs3.state["acceptance"]["results"][0]["command_id"] == "acc:default"

            cmd_dirs = sorted(p.name for p in (out / "cmd_results").iterdir())
            verify_dirs = sorted(p.name for p in (out / "verify").iterdir())
            assert len(cmd_dirs) == 2
            assert len(verify_dirs) == 1
            assert (
                (out / "cmd_results" / cmd_dirs[0] / "marker.txt").read_text(encoding="utf-8")
                != (out / "cmd_results" / cmd_dirs[1] / "marker.txt").read_text(encoding="utf-8")
            )
        finally:
            await adapter.stop()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "bad_action_id",
    ["../escape", "a/b", "C:\\evil", "foo:ads", "CON", "act."],
)
def test_d03_unsafe_action_id_rejected_before_mkdir_or_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_action_id: str
) -> None:
    adapter, _, out, _ = _setup_adapter(tmp_path)
    spawns = 0

    async def _never_spawn(*args, **kwargs):
        nonlocal spawns
        spawns += 1
        raise AssertionError("subprocess must never spawn for unsafe action_id")

    from forecontract_coding import adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "run_command_in_job", _never_spawn)

    async def _run() -> None:
        await adapter.start()
        try:
            before_entries = {p.name for p in out.iterdir()}
            for action_type, payload in (
                ("run_command", {"command_id": "diag_echo_argv"}),
                ("verify_task", {}),
            ):
                act = _act(action_type, payload, bad_action_id)
                with pytest.raises(InvalidActionPayloadError):
                    adapter.validate_action(act)
                with pytest.raises(InvalidActionPayloadError):
                    await adapter.execute(act)
            assert spawns == 0
            assert {p.name for p in out.iterdir()} == before_entries
        finally:
            await adapter.stop()

    asyncio.run(_run())


def test_d04_result_dir_allocation_preserves_prior_runs_without_rmtree(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    out = tmp_path / "out"
    ext = tmp_path / "ext"
    ws.mkdir()
    out.mkdir()
    ext.mkdir()
    (ws / "mod.py").write_bytes(b"VAL = 42\n")
    test_file = ext / "test_ext.py"
    test_file.write_text("import mod\ndef test_val(): assert mod.VAL == 42\n", encoding="utf-8")
    basetemp = out / "basetemp"
    basetemp.mkdir()

    def _build_adapter() -> CodingAdapter:
        task = CodingTask(
            run_id="run-d04",
            goal_id="goal-d04",
            goal_text="result dir isolation",
            workspace_root=str(ws),
            read_paths=["mod.py"],
            write_paths=[WriteTarget(path="mod.py", operation="modify")],
            commands={
                "accept": CommandSpec(
                    command_id="accept",
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
                    timeout_ms=10000,
                    purpose="acceptance",
                    expected_tests=[ExpectedTestCase(classname="test_ext", name="test_val")],
                )
            },
            verification_command_ids=["accept"],
            max_steps=6,
            max_planning_calls=6,
            max_replans=2,
            run_timeout_ms=30000,
        )
        return CodingAdapter(task=task, config=CodingConfig(output_root=str(out)))

    adp1 = _build_adapter()
    adp2 = _build_adapter()

    async def _run() -> None:
        await adp1.start()
        try:
            r1 = await adp1.execute(_act("verify_task", {}, "shared-id"))
            assert r1.status is ExecutionStatus.SUCCEEDED
            first_xmls = list((out / "verify").rglob("junit.xml"))
            assert len(first_xmls) == 1
            sentinel = first_xmls[0].parent / "keep_me.txt"
            sentinel.write_text("prior-invocation", encoding="utf-8")
        finally:
            await adp1.stop()

        await adp2.start()
        try:
            r2 = await adp2.execute(_act("verify_task", {}, "shared-id"))
            assert r2.status is ExecutionStatus.SUCCEEDED
            all_xmls = list((out / "verify").rglob("junit.xml"))
            assert len(all_xmls) == 2
            assert sentinel.exists() and sentinel.read_text(encoding="utf-8") == "prior-invocation"
        finally:
            await adp2.stop()

    asyncio.run(_run())


def test_d05_acceptance_source_rejected_at_init_when_invalid(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    out = tmp_path / "out"
    ext = tmp_path / "ext"
    ws.mkdir()
    out.mkdir()
    ext.mkdir()
    (ws / "mod.py").write_bytes(b"VAL = 42\n")
    (ws / "in_ws_test.py").write_text("def test_val(): assert True\n", encoding="utf-8")
    (out / "in_out_test.py").write_text("def test_val(): assert True\n", encoding="utf-8")
    real_ext = ext / "real_ext.py"
    real_ext.write_text("def test_val(): assert True\n", encoding="utf-8")

    # Symlink / hardlink / inside workspace / inside output / missing .py
    bad_targets: list[Path] = [
        ext / "missing.py",
        ws / "in_ws_test.py",
        out / "in_out_test.py",
    ]
    hl_ext = ext / "hardlink_in_ws.py"
    os.link(ws / "mod.py", hl_ext)
    bad_targets.append(hl_ext)

    sym_ext = ext / "sym_test.py"
    sym_res = subprocess.run(
        ["cmd", "/c", "mklink", str(sym_ext), str(real_ext)],
        capture_output=True,
        text=True,
        check=False,
    )
    if sym_res.returncode == 0 and sym_ext.exists():
        bad_targets.append(sym_ext)

    basetemp = out / "bt"
    basetemp.mkdir()
    for idx, target in enumerate(bad_targets):
        task = CodingTask(
            run_id=f"run-d05-{idx}",
            goal_id="goal-d05",
            goal_text="reject bad acceptance source",
            workspace_root=str(ws),
            read_paths=["mod.py"],
            write_paths=[WriteTarget(path="mod.py", operation="modify")],
            commands={
                "accept": CommandSpec(
                    command_id="accept",
                    executable=sys.executable,
                    args=[
                        "-B",
                        "-m",
                        "pytest",
                        "-q",
                        f"--rootdir={target.parent}",
                        f"--confcutdir={target.parent}",
                        f"--basetemp={basetemp}",
                        "--junitxml={result_dir}/junit.xml",
                        str(target),
                    ],
                    cwd=str(ws),
                    timeout_ms=5000,
                    purpose="acceptance",
                    expected_tests=[ExpectedTestCase(classname=target.stem, name="test_val")],
                )
            },
            verification_command_ids=["accept"],
            max_steps=4,
            max_planning_calls=4,
            max_replans=1,
            run_timeout_ms=10000,
        )
        with pytest.raises(CodingError):
            CodingAdapter(task=task, config=CodingConfig(output_root=str(out)))


def test_d06_acceptance_source_mutation_before_during_after_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forecontract_coding import adapter as adapter_mod

    # Case A: Mutated BEFORE verify_task -> zero process starts, unknown acceptance
    adp_before, ws_a, _, ext_a = _setup_adapter(tmp_path / "case_a")
    (ws_a / "mod.py").write_bytes(b"VAL = 42\n")
    spawns_before = 0
    real_run = adapter_mod.run_command_in_job

    async def _count_spawns(*args, **kwargs):
        nonlocal spawns_before
        spawns_before += 1
        return await real_run(*args, **kwargs)

    async def _run_a() -> None:
        await adp_before.start()
        try:
            ext_a.write_text("def test_val(): assert True\n", encoding="utf-8")
            monkeypatch.setattr(adapter_mod, "run_command_in_job", _count_spawns)
            out = await adp_before.execute(_act("verify_task", {}, "va-1"))
            assert out.status is ExecutionStatus.FAILED
            assert out.error is not None and out.error.code == "CODING_VERIFICATION_INVALID"
            assert spawns_before == 0
            obs = await adp_before.observe()
            assert obs.state["acceptance"]["status"] == "unknown"
            assert obs.state["acceptance"]["stale_reason"] == "test_source_changed"
            assert obs.state["acceptance_sources_ok"] is False
        finally:
            monkeypatch.undo()
            await adp_before.stop()

    asyncio.run(_run_a())

    # Case B: Mutated DURING verify_task -> execution result marked unknown
    adp_during, ws_b, _, _ext_b = _setup_adapter(
        tmp_path / "case_b",
        test_body=(
            "import pathlib\n"
            "def test_val():\n"
            "    p = pathlib.Path(__file__)\n"
            "    p.write_text('def test_val(): assert True\\n', encoding='utf-8')\n"
            "    assert True\n"
        ),
    )
    (ws_b / "mod.py").write_bytes(b"VAL = 42\n")

    async def _run_b() -> None:
        await adp_during.start()
        try:
            out = await adp_during.execute(_act("verify_task", {}, "vb-1"))
            assert out.status is ExecutionStatus.UNKNOWN
            obs = await adp_during.observe()
            assert obs.state["acceptance"]["status"] == "unknown"
            assert obs.state["acceptance"]["stale_reason"] == "test_source_changed"
        finally:
            await adp_during.stop()

    asyncio.run(_run_b())

    # Case C: Passed verification invalidated immediately upon observe() after source tamper without re-run
    adp_after, _ws_c, _, ext_c = _setup_adapter(tmp_path / "case_c")

    async def _run_c() -> None:
        await adp_after.start()
        try:
            sha0 = hashlib.sha256(b"VAL = 0\n").hexdigest()
            await adp_after.execute(
                _act("write_file", {"path": "mod.py", "expected_sha256": sha0, "content": "VAL = 42\n"}, "wc-1")
            )
            res = await adp_after.execute(_act("verify_task", {}, "vc-1"))
            assert res.status is ExecutionStatus.SUCCEEDED
            obs_ok = await adp_after.observe()
            assert obs_ok.state["acceptance"]["status"] == "passed"
            assert obs_ok.state["acceptance_sources_ok"] is True

            ext_c.write_text("def test_val(): assert False\n", encoding="utf-8")
            obs_bad = await adp_after.observe()
            assert obs_bad.state["acceptance"]["status"] == "unknown"
            assert obs_bad.state["acceptance"]["stale_reason"] == "test_source_changed"
            assert obs_bad.state["acceptance_sources_ok"] is False
        finally:
            await adp_after.stop()

    asyncio.run(_run_c())


def test_d07_workspace_scope_inventory_detects_create_modify_delete_and_overflow(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    out = tmp_path / "out"
    ext = tmp_path / "ext"
    ws.mkdir()
    out.mkdir()
    ext.mkdir()
    (ws / "mod.py").write_bytes(b"VAL = 0\n")
    (ws / "ro.py").write_bytes(b"RO = 1\n")
    (ws / "locked.py").write_bytes(b"LOCKED = 1\n")
    test_file = ext / "test_ext.py"
    test_file.write_text("import mod\ndef test_val(): assert mod.VAL == 42\n", encoding="utf-8")
    basetemp = out / "basetemp"
    basetemp.mkdir()

    task = CodingTask(
        run_id="run-d07",
        goal_id="goal-d07",
        goal_text="scope checks",
        workspace_root=str(ws),
        read_paths=["mod.py", "ro.py"],
        write_paths=[WriteTarget(path="mod.py", operation="modify")],
        commands={
            "accept": CommandSpec(
                command_id="accept",
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
                    str(test_file),
                ],
                cwd=str(ws),
                env={"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                timeout_ms=10000,
                purpose="acceptance",
                expected_tests=[ExpectedTestCase(classname="test_ext", name="test_val")],
            )
        },
        verification_command_ids=["accept"],
        max_steps=6,
        max_planning_calls=6,
        max_replans=2,
        run_timeout_ms=30000,
    )
    adapter = CodingAdapter(task=task, config=CodingConfig(output_root=str(out)))

    async def _run() -> None:
        await adapter.start()
        try:
            sha0 = hashlib.sha256(b"VAL = 0\n").hexdigest()
            await adapter.execute(
                _act("write_file", {"path": "mod.py", "expected_sha256": sha0, "content": "VAL = 42\n"}, "w1")
            )
            await adapter.execute(_act("verify_task", {}, "v1"))
            obs_clean = await adapter.observe()
            assert obs_clean.state["scope_ok"] is True
            assert obs_clean.state["scope_violations"] == []
            assert obs_clean.state["acceptance"]["status"] == "passed"
            assert [c["path"] for c in obs_clean.state["changed_paths"]] == ["mod.py"]

            # Introduce unauthorized create + unlisted modify + read-only delete
            (ws / "unapproved.py").write_bytes(b"ROGUE = 1\n")
            (ws / "locked.py").write_bytes(b"LOCKED = 999\n")
            (ws / "ro.py").unlink()

            obs_bad = await adapter.observe()
            assert obs_bad.state["scope_ok"] is False
            assert obs_bad.state["acceptance"]["status"] == "unknown"
            assert obs_bad.state["acceptance"]["stale_reason"] == "scope_violation"
            violations = obs_bad.state["scope_violations"]
            assert [(v["path"], v["reason"]) for v in violations] == [
                ("locked.py", "unauthorized_modify"),
                ("ro.py", "unauthorized_delete"),
                ("unapproved.py", "unauthorized_create"),
            ]
            # Violating files must NOT be silently deleted or auto-restored
            assert (ws / "unapproved.py").exists()
            assert (ws / "locked.py").read_bytes() == b"LOCKED = 999\n"
            assert not (ws / "ro.py").exists()
        finally:
            await adapter.stop()

    asyncio.run(_run())

    # Observation overflow while scope_violations is non-empty keeps scope_ok=False & bounded list
    adp_ov = CodingAdapter(
        task=task,
        config=CodingConfig(output_root=str(out), max_observation_bytes=1200),
    )

    async def _run_ov() -> None:
        await adp_ov.start()
        try:
            (ws / "extra_overflow.py").write_bytes(b"X = 1\n")
            obs_ov = await adp_ov.observe()
            assert obs_ov.state["observation_overflow"] is True
            assert obs_ov.state["scope_ok"] is False
            assert len(obs_ov.state["scope_violations"]) >= 1
            assert len(json.dumps(obs_ov.model_dump(mode="json")).encode("utf-8")) <= 1200
        finally:
            await adp_ov.stop()

    asyncio.run(_run_ov())
