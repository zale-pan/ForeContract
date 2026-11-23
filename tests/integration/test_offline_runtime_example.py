"""C12 and the minimal-example contract: the CLI, its output shape and its safety rules.

The example is exercised as a real subprocess against an external output directory, so these cases prove the
process-level contract rather than the behaviour of an imported function: the exit code, the exact three
top-level stdout fields, the real ``RunResult`` payload, the read-back artefacts, and the refusal to touch the
repository or a non-empty directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import (
    REPO_ROOT,
    SENSITIVE_MARKER,
    kinds,
    read_jsonl,
    read_records,
    terminal_records,
)
from forecontract.tracing import JsonlTraceStore

DEMO = REPO_ROOT / "examples" / "offline_runtime_demo.py"
SOURCE_PATHS = [str(REPO_ROOT / part) for part in
                ("packages/core/src", "packages/adapters/fake/src",
                 "packages/planners/scripted/src")]


def run_demo(output_dir: Path) -> subprocess.CompletedProcess:
    """Run the example as a separate process, importing the real distributions the way a host would."""
    return run_demo_script(DEMO, output_dir)


def run_demo_script(script: Path, output_dir: Path) -> subprocess.CompletedProcess:
    """Run one example script - the real one or a faulted copy - as a separate process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ";".join(SOURCE_PATHS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", str(script), "--output-dir", str(output_dir)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=False)


#: C12/F01: the three declared stages, and the exact example source line each fault case replaces. Each
#: replacement raises an ``OSError`` carrying a unique, deliberately non-secret marker, so a process that
#: echoed the dependency text would be caught by the marker rather than by a guess about its wording.
FAULT_REPLACEMENTS = {
    "output_preparation": (
        "            target.mkdir(parents=True)",
        '            raise OSError("P2T08-CLI-FAULT-OUTPUT-PREPARATION")'),
    "runtime": (
        "        result = asyncio.run(engine.run(request=request))",
        '        raise OSError("P2T08-CLI-FAULT-RUNTIME")'),
    "projection": (
        "        episodic = JsonlEpisodicMemory(episodic_path, source_resolver=resolver)",
        '        raise OSError("P2T08-CLI-FAULT-PROJECTION")'),
}


def run_faulted_demo(workdir: Path, stage: str) -> subprocess.CompletedProcess:
    """Copy the example into external scratch with exactly one stage line replaced by a raising one.

    The copy is two directories below the scratch root and the output target is elsewhere in it, so the
    copy's own repository check does not refuse the target and the fault under test is the one that fires.
    """
    original_line, faulted_line = FAULT_REPLACEMENTS[stage]
    original = DEMO.read_text(encoding="utf-8")
    # The premise is checked, not assumed: the exact line this case replaces occurs exactly once.
    assert original.count(original_line) == 1
    copied = workdir / "copied" / "src" / DEMO.name
    copied.parent.mkdir(parents=True)
    copied.write_text(original.replace(original_line, faulted_line), encoding="utf-8")
    return run_demo_script(copied, workdir / "faulted-output")


def assert_bounded_cli_failure(completed: subprocess.CompletedProcess, *,
                               marker: str, diagnostic: str) -> None:
    """The exact process-level contract of a stage fault: a bounded exit, no summary, no dependency text."""
    # Measured exactly: the example's one declared failure exit code, never a success and never a crash.
    assert completed.returncode == 1
    # No success summary was printed, so the caller is never handed a trace path for a Run that failed.
    assert "trace_path" not in completed.stdout
    assert completed.stdout.strip() == ""
    stderr = completed.stderr
    # None of the raised marker, its class name or a traceback reaches the caller: the diagnostic is fixed.
    assert marker not in stderr
    assert "OSError" not in stderr
    assert "Traceback" not in stderr
    assert stderr.strip() == f"offline runtime demo failed: {diagnostic}"


def test_a_fault_while_the_output_directory_is_prepared_is_a_bounded_cli_failure(workdir: Path):
    """C12/F01: a filesystem fault during output preparation ends in the fixed 'output' diagnostic."""
    completed = run_faulted_demo(workdir, "output_preparation")

    assert_bounded_cli_failure(
        completed, marker="P2T08-CLI-FAULT-OUTPUT-PREPARATION",
        diagnostic="the requested output directory could not be prepared")
    # The fault really struck the preparation stage: the target was never created and nothing was written.
    assert not (workdir / "faulted-output").exists()


def test_a_fault_during_the_runtime_run_is_a_bounded_cli_failure(workdir: Path):
    """C12/F01: a dependency fault inside the Run ends in the fixed 'runtime' diagnostic."""
    completed = run_faulted_demo(workdir, "runtime")

    assert_bounded_cli_failure(
        completed, marker="P2T08-CLI-FAULT-RUNTIME",
        diagnostic="the offline runtime run did not complete")
    # The output directory was prepared, so the fault really struck the Run and not the preparation.
    assert (workdir / "faulted-output").is_dir()


def test_a_fault_during_the_episodic_projection_is_a_bounded_cli_failure(workdir: Path):
    """C12/F01: a dependency fault inside the host projection ends in the fixed 'projection' diagnostic."""
    completed = run_faulted_demo(workdir, "projection")

    assert_bounded_cli_failure(
        completed, marker="P2T08-CLI-FAULT-PROJECTION",
        diagnostic="the host episodic projection did not complete")
    # The Run really completed and wrote its own trace before the projection stage failed.
    assert (workdir / "faulted-output" / "trace.jsonl").is_file()
    assert not (workdir / "faulted-output" / "episodic.jsonl").exists()


def test_the_example_prints_exactly_the_three_documented_fields(workdir: Path):
    """The stdout object has exactly ``result``, ``trace_path`` and ``episodic_memory_path``."""
    target = workdir / "demo-out"
    completed = run_demo(target)

    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, completed.stdout
    summary = json.loads(lines[0])
    assert set(summary) == {"result", "trace_path", "episodic_memory_path"}
    assert Path(summary["trace_path"]).is_file()
    assert Path(summary["episodic_memory_path"]).is_file()


def test_the_demo_result_is_a_real_completed_run_result(workdir: Path):
    """C01/C12: the embedded result is a real RunResult with the declared success facts."""
    target = workdir / "demo-out"
    completed = run_demo(target)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)

    result = summary["result"]
    assert set(result) == {
        "schema_version", "run_id", "goal_id", "terminal_status", "reason_code",
        "steps_started", "planning_calls", "attempts_started", "retries_used", "replans_used",
        "verification_record_id", "terminal_record_id"}
    assert result["terminal_status"] == "completed"
    assert result["reason_code"] == "TASK_ACHIEVED"
    # Exactly one real action and Attempt, one planning call, no retry and no replan.
    assert (result["steps_started"], result["planning_calls"], result["attempts_started"],
            result["retries_used"], result["replans_used"]) == (1, 1, 1, 0, 0)


def test_the_demo_trace_and_projection_read_back_consistently(workdir: Path):
    """The written Trace really contains the run, and the projection cites a real source."""
    target = workdir / "demo-out"
    completed = run_demo(target)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    result = summary["result"]

    trace = JsonlTraceStore(Path(summary["trace_path"]))
    records = read_records(trace, result["run_id"])
    assert kinds(records) == ["run_started", "step_started", "contract_committed",
                              "attempt_started", "attempt_finished", "verification_recorded",
                              "verification_recorded", "run_finished"]
    terminals = terminal_records(records)
    assert len(terminals) == 1
    assert terminals[0].terminal_status == "success"
    assert terminals[0].verification_record_id == result["verification_record_id"]

    items = read_jsonl(Path(summary["episodic_memory_path"]))
    assert len(items) == 1
    item = items[0]
    assert item["run_id"] == result["run_id"]
    assert item["source_record_id"] == result["terminal_record_id"]
    assert item["result"] == "success"
    # Every projected source record really exists in the trace it names.
    assert item["source_record_id"] in {record.record_id for record in records}


def test_the_artifacts_carry_no_sensitive_marker(workdir: Path):
    """C12: no artefact may contain the synthetic sensitive marker, and none may carry an API key."""
    target = workdir / "demo-out"
    completed = run_demo(target)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)

    for path in (Path(summary["trace_path"]), Path(summary["episodic_memory_path"]),
                 target / "working-memory.json"):
        text = path.read_text(encoding="utf-8")
        assert SENSITIVE_MARKER not in text
        assert "api_key" not in text.lower()
    assert SENSITIVE_MARKER not in completed.stdout
    # The example does not embed provenance text either: the summary is the three documented fields only.
    assert "sk-" not in completed.stdout


def test_the_example_refuses_to_write_inside_the_repository(workdir: Path):
    """The example resolves its target and refuses a repository-internal path without writing."""
    inside = REPO_ROOT / "examples" / "_demo_should_not_exist"
    completed = run_demo(inside)

    assert completed.returncode != 0
    assert "repository" in completed.stderr.lower()
    # Nothing was created and no success summary was printed.
    assert not inside.exists()
    assert "trace_path" not in completed.stdout


def test_the_example_refuses_a_non_empty_directory(workdir: Path):
    """A pre-existing non-empty target is refused, and its existing content is left untouched."""
    target = workdir / "occupied"
    target.mkdir()
    kept = target / "keep.txt"
    kept.write_text("existing content", encoding="utf-8")

    completed = run_demo(target)

    assert completed.returncode != 0
    assert "non-empty" in completed.stderr.lower() or "empty" in completed.stderr.lower()
    assert kept.read_text(encoding="utf-8") == "existing content"
    assert sorted(item.name for item in target.iterdir()) == ["keep.txt"]
    assert "trace_path" not in completed.stdout


def test_the_example_accepts_an_empty_directory(workdir: Path):
    """An existing but empty target is legal, which is the documented precondition."""
    target = workdir / "empty"
    target.mkdir()

    completed = run_demo(target)

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout.splitlines()[0])
    assert Path(summary["trace_path"]).parent == target
