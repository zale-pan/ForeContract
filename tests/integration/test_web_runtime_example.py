"""CLI and external-directory acceptance for the Web Runtime example.

The tests invoke the example as a real subprocess from an external working directory. They prove
the exact five-field stdout object, the exit-code contract (0 success, 2 normal non-completion,
130 propagated cancellation, 1 facility/output failure, 2 argparse), the external new-or-empty
output-directory rule and the presence of the real trace, evidence index and Working Memory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "web_runtime_demo.py"
FIVE_FIELDS = {"result", "terminal", "trace_path", "evidence_index_path",
               "working_memory_path"}


def environment() -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(REPO), str(REPO / "examples"),
             str(REPO / "packages" / "core" / "src"),
             str(REPO / "packages" / "adapters" / "web" / "src"),
             str(REPO / "packages" / "adapters" / "fake" / "src"),
             str(REPO / "packages" / "planners" / "scripted" / "src")]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def invoke(args: list[str], *, cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", str(DEMO), *args],
        cwd=cwd, env=environment(), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )


def parse_single_object(completed: subprocess.CompletedProcess) -> dict:
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, completed.stdout
    payload = json.loads(lines[0])
    assert set(payload) == FIVE_FIELDS
    return payload


def test_normal_cli_prints_one_five_field_record_and_exits_zero(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    output = tmp_path / "normal"
    output.mkdir()
    completed = invoke(["--profile", "normal", "--output-dir", str(output)], cwd=external)
    assert completed.returncode == 0, completed.stderr
    payload = parse_single_object(completed)
    assert payload["result"]["terminal_status"] == "completed"
    assert payload["terminal"] == {"terminal_status": "completed",
                                   "reason_code": "TASK_ACHIEVED",
                                   "source": "run_result"}
    for key in ("trace_path", "evidence_index_path", "working_memory_path"):
        assert Path(payload[key]).is_file(), key
    index = json.loads(Path(payload["evidence_index_path"]).read_text(encoding="utf-8"))
    assert index["cleanup"]["adapter_cleanup_error"] is None
    assert not any(path.name == "normal" for path in REPO.iterdir())


def test_wrong_credentials_exits_two_without_claiming_task_success(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    output = tmp_path / "wrong"
    output.mkdir()
    completed = invoke(["--profile", "wrong_credentials", "--output-dir", str(output)],
                       cwd=external)
    assert completed.returncode == 2, completed.stderr
    payload = parse_single_object(completed)
    assert payload["result"]["terminal_status"] == "aborted"
    assert payload["result"]["reason_code"] == "POLICY_ABORT"
    assert payload["result"]["terminal_status"] != "completed"


def test_cancellation_is_recorded_and_exits_one_hundred_thirty(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    output = tmp_path / "cancel"
    output.mkdir()
    completed = invoke(["--profile", "cancel_after_received", "--output-dir", str(output)],
                       cwd=external)
    assert completed.returncode == 130, (completed.returncode, completed.stderr)
    payload = parse_single_object(completed)
    assert payload["result"] is None
    assert payload["terminal"] == {"terminal_status": "cancelled",
                                   "reason_code": "USER_CANCELLED",
                                   "source": "last_finalization"}
    index = json.loads(Path(payload["evidence_index_path"]).read_text(encoding="utf-8"))
    assert index["call_interrupted"] == "CancelledError"
    assert index["last_finalization"]["primary_terminal_status"] == "cancelled"


def test_overlay_released_runs_a_new_generation_and_exits_zero(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    output = tmp_path / "overlay-released"
    output.mkdir()
    completed = invoke(["--profile", "overlay_released", "--output-dir", str(output)],
                       cwd=external)
    assert completed.returncode == 0, completed.stderr
    payload = parse_single_object(completed)
    assert payload["result"]["terminal_status"] == "completed"
    index = json.loads(Path(payload["evidence_index_path"]).read_text(encoding="utf-8"))
    previous = json.loads((output / "previous-run" / "evidence-index.json")
                          .read_text(encoding="utf-8"))
    assert previous["result"]["terminal_status"] == "aborted"
    assert index["result"]["terminal_status"] == "completed"


def test_an_unknown_profile_is_an_argparse_error_and_runs_nothing(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    output = tmp_path / "never-created"
    completed = invoke(["--profile", "no-such-profile", "--output-dir", str(output)],
                       cwd=external)
    assert completed.returncode == 2
    assert completed.stdout.strip() == ""
    assert not output.exists()


def test_a_missing_output_directory_is_an_argparse_error(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    completed = invoke(["--profile", "normal"], cwd=external)
    assert completed.returncode == 2
    assert completed.stdout.strip() == ""


def test_a_non_empty_output_directory_is_refused_with_exit_one(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    completed = invoke(["--profile", "normal", "--output-dir", str(output)], cwd=external)
    assert completed.returncode == 1
    assert completed.stdout.strip() == ""
    assert (output / "existing.txt").read_text(encoding="utf-8") == "do not overwrite"
    assert not (output / "trace.jsonl").exists()


def test_a_repository_output_directory_is_refused_with_exit_one(tmp_path: Path) -> None:
    external = tmp_path / "cwd"
    external.mkdir()
    forbidden = REPO / "examples" / "should-never-be-created"
    assert not forbidden.exists()
    completed = invoke(["--profile", "normal", "--output-dir", str(forbidden)], cwd=external)
    assert completed.returncode == 1
    assert not forbidden.exists()
