"""Unit tests for ``forecontract_coding.process`` covering C06, C07, C08, C09."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (
    REPO_ROOT / "packages" / "core" / "src",
    REPO_ROOT / "packages" / "planners" / "llm" / "src",
    REPO_ROOT / "packages" / "adapters" / "coding" / "src",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract_coding.models import CommandSpec
from forecontract_coding.process import (
    build_minimal_windows_env,
    run_command_in_job,
)


def test_c06_minimal_env_excludes_parent_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-must-not-pass")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-leak-must-not-pass")
    monkeypatch.setenv("CUSTOM_HOST_VAR", "must-not-inherit")

    spec = CommandSpec(
        command_id="diag",
        executable=sys.executable,
        args=["-c", "import os, json; print(json.dumps(dict(os.environ)))"],
        cwd=str(tmp_path),
        env={"SAFE_FLAG": "1"},
        timeout_ms=5000,
        purpose="diagnostic",
    )
    built = build_minimal_windows_env(spec=spec, temp_dir=tmp_path / "tmp")
    assert "OPENAI_API_KEY" not in built
    assert "DEEPSEEK_API_KEY" not in built
    assert "CUSTOM_HOST_VAR" not in built
    assert built["SAFE_FLAG"] == "1"

    res = asyncio.run(
        run_command_in_job(
            spec=spec,
            argv=(spec.executable, *spec.args),
            temp_dir=tmp_path / "tmp",
            max_output_bytes=32 * 1024,
            process_cleanup_ms=2000,
        )
    )
    assert res.exit_code == 0
    assert "sk-leak-must-not-pass" not in res.stdout
    assert "ds-leak-must-not-pass" not in res.stdout
    assert "must-not-inherit" not in res.stdout


def test_c07_real_exit0_and_exit1_preserve_both_streams(tmp_path: Path) -> None:
    for code in (0, 1):
        spec = CommandSpec(
            command_id=f"exit_{code}",
            executable=sys.executable,
            args=[
                "-c",
                f"import sys; sys.stdout.write('out-{code}\\n'); sys.stderr.write('err-{code}\\n'); sys.exit({code})",
            ],
            cwd=str(tmp_path),
            timeout_ms=5000,
            purpose="diagnostic",
        )
        res = asyncio.run(
            run_command_in_job(
                spec=spec,
                argv=(spec.executable, *spec.args),
                temp_dir=tmp_path / "tmp",
                max_output_bytes=32 * 1024,
                process_cleanup_ms=2000,
            )
        )
        assert res.exit_code == code
        assert res.timed_out is False
        assert f"out-{code}" in res.stdout
        assert f"err-{code}" in res.stderr


def test_c08_large_dual_stream_output_bounded_without_pipe_deadlock(tmp_path: Path) -> None:
    limit = 4096
    spec = CommandSpec(
        command_id="flood",
        executable=sys.executable,
        args=[
            "-c",
            (
                "import sys\n"
                "for _ in range(400):\n"
                "    sys.stdout.write('A' * 512 + '\\n')\n"
                "    sys.stderr.write('B' * 512 + '\\n')\n"
                "sys.stdout.flush(); sys.stderr.flush()\n"
            ),
        ],
        cwd=str(tmp_path),
        timeout_ms=10000,
        purpose="diagnostic",
    )
    res = asyncio.run(
        run_command_in_job(
            spec=spec,
            argv=(spec.executable, *spec.args),
            temp_dir=tmp_path / "tmp",
            max_output_bytes=limit,
            process_cleanup_ms=2000,
        )
    )
    assert res.exit_code == 0
    assert res.stdout_truncated is True
    assert res.stderr_truncated is True
    assert len(res.stdout.encode("utf-8")) <= limit
    assert len(res.stderr.encode("utf-8")) <= limit
    assert res.stdout_bytes_seen > limit
    assert res.stderr_bytes_seen > limit


def test_c08_timeout_terminates_job_cleanly(tmp_path: Path) -> None:
    spec = CommandSpec(
        command_id="hang",
        executable=sys.executable,
        args=["-c", "import time; time.sleep(30)"],
        cwd=str(tmp_path),
        timeout_ms=250,
        purpose="diagnostic",
    )
    res = asyncio.run(
        run_command_in_job(
            spec=spec,
            argv=(spec.executable, *spec.args),
            temp_dir=tmp_path / "tmp",
            max_output_bytes=32 * 1024,
            process_cleanup_ms=2000,
        )
    )
    assert res.timed_out is True
    assert res.exit_code is None
    assert res.cleanup_ok is True
    assert res.active_processes_remaining == 0


def test_c09_cancel_stops_grandchild_writer_and_leaves_sentinel_alive(tmp_path: Path) -> None:
    heartbeat = tmp_path / "grandchild_heartbeat.txt"
    grandchild_script = tmp_path / "grandchild.py"
    grandchild_script.write_text(
        (
            "import pathlib, time\n"
            f"p = pathlib.Path(r'{heartbeat}')\n"
            "for i in range(10000):\n"
            "    with p.open('a', encoding='utf-8') as f:\n"
            "        f.write(f'{i}\\n')\n"
            "    time.sleep(0.02)\n"
        ),
        encoding="utf-8",
    )
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-B', r'{grandchild_script}'])\n"
            "time.sleep(60)\n"
        ),
        encoding="utf-8",
    )

    # Start independent sentinel process OUTSIDE the coding Job
    sentinel = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        spec = CommandSpec(
            command_id="spawn_tree",
            executable=sys.executable,
            args=["-B", str(parent_script)],
            cwd=str(tmp_path),
            timeout_ms=15000,
            purpose="diagnostic",
        )

        async def _cancel_flow() -> float:
            task = asyncio.create_task(
                run_command_in_job(
                    spec=spec,
                    argv=(spec.executable, *spec.args),
                    temp_dir=tmp_path / "tmp",
                    max_output_bytes=32 * 1024,
                    process_cleanup_ms=2000,
                )
            )
            deadline = time.monotonic() + 5.0
            while not heartbeat.exists() or heartbeat.stat().st_size == 0:
                if time.monotonic() >= deadline:
                    raise AssertionError("grandchild never started writing")
                await asyncio.sleep(0.02)
            t0 = time.monotonic()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return (time.monotonic() - t0) * 1000.0

        elapsed_ms = asyncio.run(_cancel_flow())
        assert elapsed_ms <= 2500.0
        size_after_cancel = heartbeat.stat().st_size
        time.sleep(0.25)
        assert heartbeat.stat().st_size == size_after_cancel
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)
