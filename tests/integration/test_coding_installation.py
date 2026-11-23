"""Acceptance tests for offline building and isolated installation of ``forecontract-coding`` (C23)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PROJECT = REPO_ROOT / "packages" / "core"
LLM_PROJECT = REPO_ROOT / "packages" / "planners" / "llm"
CODING_PROJECT = REPO_ROOT / "packages" / "adapters" / "coding"
CODING_SRC = CODING_PROJECT / "src" / "forecontract_coding"

CODING_MODULES = {
    "__init__.py",
    "adapter.py",
    "agent.py",
    "errors.py",
    "models.py",
    "process.py",
    "runtime_hooks.py",
    "runtime_support.py",
    "verification.py",
    "workspace.py",
}


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(args: list[str], *, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _require_dir_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"the acceptance environment must define {name}")
    path = Path(value)
    assert path.is_dir(), f"{name} must be an existing directory"
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


@pytest.fixture(scope="module")
def coding_archives(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Copy Core/LLM/Coding to isolated temp dirs and build wheels + sdists offline."""
    _require_dir_env("FORECONTRACT_PYDANTIC_WHEELHOUSE")
    work = tmp_path_factory.mktemp("coding-src-build")
    out = work / "dist"
    out.mkdir()
    for name, project in (
        ("core", CORE_PROJECT),
        ("llm", LLM_PROJECT),
        ("coding", CODING_PROJECT),
    ):
        shutil.copytree(
            project,
            work / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc", "build", "dist"),
        )
        res = _run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(out), str(work / name)],
            cwd=work,
        )
        assert res.returncode == 0, res.stdout + res.stderr

    wheels = sorted(out.glob("*.whl"))
    sdists = sorted(out.glob("*.tar.gz"))
    assert len(wheels) == 3
    assert len(sdists) == 3
    selected: dict[str, Path] = {"out": out}
    for whl in wheels:
        if whl.name.startswith("forecontract_core"):
            selected["core_wheel"] = whl
        elif whl.name.startswith("forecontract_llm"):
            selected["llm_wheel"] = whl
        elif whl.name.startswith("forecontract_coding"):
            selected["coding_wheel"] = whl
    for sdist in sdists:
        if sdist.name.startswith("forecontract_coding"):
            selected["coding_sdist"] = sdist
    return selected


def test_c23_coding_wheel_and_sdist_match_metadata_and_source_bytes(
    coding_archives: dict[str, Path],
) -> None:
    wheel = coding_archives["coding_wheel"]
    sdist = coding_archives["coding_sdist"]

    with zipfile.ZipFile(wheel) as zf:
        metadata = zf.read("forecontract_coding-0.1.0.dist-info/METADATA").decode("utf-8")
        modules = {
            name.split("forecontract_coding/", 1)[1]: zf.read(name)
            for name in zf.namelist()
            if name.startswith("forecontract_coding/") and not name.endswith("/")
        }
    assert set(modules) == CODING_MODULES
    for rel, payload in modules.items():
        assert payload == (CODING_SRC / rel).read_bytes(), rel

    name_field = next(l.split(":", 1)[1].strip() for l in metadata.splitlines() if l.startswith("Name:"))
    ver_field = next(l.split(":", 1)[1].strip() for l in metadata.splitlines() if l.startswith("Version:"))
    py_field = next(l.split(":", 1)[1].strip() for l in metadata.splitlines() if l.startswith("Requires-Python:"))
    reqs = {
        str(Requirement(l.split(":", 1)[1].strip()))
        for l in metadata.splitlines()
        if l.startswith("Requires-Dist:")
    }
    assert name_field == "forecontract-coding"
    assert ver_field == "0.1.0"
    assert py_field == ">=3.12"
    assert reqs == {
        str(Requirement("forecontract-core>=0.1,<0.2")),
        str(Requirement("forecontract-llm>=0.1,<0.2")),
        str(Requirement("pydantic>=2,<3")),
    }

    with tarfile.open(sdist) as tf:
        sdist_mods = {}
        for m in tf.getmembers():
            if m.isfile() and "/src/forecontract_coding/" in m.name:
                rel = m.name.split("/src/forecontract_coding/", 1)[1]
                f = tf.extractfile(m)
                assert f is not None
                sdist_mods[rel] = f.read()
    assert set(sdist_mods) == CODING_MODULES
    for rel, payload in sdist_mods.items():
        assert payload == (CODING_SRC / rel).read_bytes(), rel


def test_c23_fresh_clean_venv_runs_coding_agent_without_repo_pythonpath(
    coding_archives: dict[str, Path], tmp_path: Path
) -> None:
    wheelhouse = _require_dir_env("FORECONTRACT_PYDANTIC_WHEELHOUSE")
    venv_dir = tmp_path / "clean-venv"
    res = _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    py = _venv_python(venv_dir)

    install = _run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--disable-pip-version-check",
            "--no-input",
            str(coding_archives["core_wheel"]),
            str(coding_archives["llm_wheel"]),
            str(coding_archives["coding_wheel"]),
        ],
        cwd=tmp_path,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    consumer = tmp_path / "consumer.py"
    pytest_python = sys.executable
    consumer.write_text(
        f"""import asyncio, importlib.metadata, json, sys
from pathlib import Path
import forecontract_coding
from forecontract.tracing import JsonlTraceStore
from forecontract_coding import (
    CodingConfig, CodingTask, CommandSpec, ExpectedTestCase, WriteTarget, run_coding_task
)
from forecontract_llm import ModelRequest, ModelResponse, PlannerConfig

repo_root = Path({str(REPO_ROOT)!r}).resolve()
venv_root = Path({str(venv_dir)!r}).resolve()
pkg_path = Path(forecontract_coding.__file__).resolve()
assert venv_root in pkg_path.parents, f"expected site-packages import, got {{pkg_path}}"
for entry in sys.path:
    if entry:
        p = Path(entry).resolve()
        assert p != repo_root and repo_root not in p.parents, f"checkout leaked into sys.path: {{entry}}"

dists = {{d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
assert "forecontract-coding" in dists and dists["forecontract-coding"] == "0.1.0"
assert "playwright" not in dists and "torch" not in dists

root = Path({str(tmp_path / "installed_run")!r})
ws = root / "ws"
ext = root / "ext"
out = root / "out"
bt = out / "bt"
ws.mkdir(parents=True); ext.mkdir(parents=True); bt.mkdir(parents=True)
(ws / "calc.py").write_text("def double(n): raise NotImplementedError()\\n", encoding="utf-8")
tfile = ext / "test_acc.py"
tfile.write_text("import calc\\ndef test_double(): assert calc.double(5) == 10\\n", encoding="utf-8")

class Provider:
    def __init__(self): self.i = 0
    async def complete(self, req: ModelRequest) -> ModelResponse:
        self.i += 1
        files = {{f["path"]: f for f in req.observation.state["files"]}}
        if self.i == 1:
            body = {{
                "schema_version": "0.1",
                "proposed_action": {{
                    "schema_version": "0.1", "action_type": "write_file",
                    "payload": {{"path": "calc.py", "expected_sha256": files["calc.py"]["sha256"],
                                "content": "def double(n): return n * 2\\n"}},
                    "metadata": {{}},
                }},
                "expected_outcomes": [{{"schema_version": "0.1", "outcome_type": "coding_file_written", "payload": {{"path": "calc.py"}}, "required": True}}],
            }}
        else:
            body = {{
                "schema_version": "0.1",
                "proposed_action": {{"schema_version": "0.1", "action_type": "verify_task", "payload": {{}}, "metadata": {{}}}},
                "expected_outcomes": [{{"schema_version": "0.1", "outcome_type": "coding_task_verified", "payload": {{}}, "required": True}}],
            }}
        return ModelResponse(schema_version="0.1", request_id=req.request_id, status="ok", content=json.dumps(body))

task = CodingTask(
    run_id="inst-1", goal_id="goal-inst", goal_text="double", workspace_root=str(ws),
    read_paths=["calc.py"], write_paths=[WriteTarget(path="calc.py", operation="modify")],
    commands={{
        "acc": CommandSpec(
            command_id="acc", executable={pytest_python!r},
            args=["-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
                  f"--rootdir={{ext}}", f"--confcutdir={{ext}}", f"--basetemp={{bt}}",
                  "--junitxml={{result_dir}}/junit.xml", str(tfile)],
            cwd=str(ws), env={{"PYTHONPATH": str(ws), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}},
            timeout_ms=10000, purpose="acceptance",
            expected_tests=[ExpectedTestCase(classname="test_acc", name="test_double")],
        )
    }},
    verification_command_ids=["acc"], max_steps=6, max_planning_calls=6, max_replans=2, run_timeout_ms=30000,
)
result = asyncio.run(run_coding_task(
    task=task, config=CodingConfig(output_root=str(out)), provider=Provider(),
    planner_config=PlannerConfig(schema_version="0.1", call_timeout_ms=5000, max_calls=6),
    trace_writer=JsonlTraceStore(root / "trace.jsonl"),
))
print(json.dumps({{
    "terminal_status": result.terminal_status,
    "acceptance_status": result.acceptance.status,
    "pkg_path": str(pkg_path),
}}))
""",
        encoding="utf-8",
    )
    run_res = _run([str(py), "-B", str(consumer)], cwd=tmp_path)
    assert run_res.returncode == 0, run_res.stdout + run_res.stderr
    summary = json.loads(run_res.stdout)
    assert summary["terminal_status"] == "completed"
    assert summary["acceptance_status"] == "passed"
    assert _is_within(Path(summary["pkg_path"]), venv_dir)
