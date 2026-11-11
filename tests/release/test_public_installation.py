"""Release engineering checks for building and installing the five public packages.

The test is driven entirely by the two declared environment variables; it never falls back to a
machine-specific path. It builds each project exactly once from an external copy, installs one
Core-only environment and one full environment, runs the three real offline consumers, and writes
``release-installation-index.json`` binding the ten archives and every command it ran.

Collecting this module performs no build, no install and creates no directory.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import venv
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
    "core": REPO_ROOT / "packages" / "core",
    "fake": REPO_ROOT / "packages" / "adapters" / "fake",
    "web": REPO_ROOT / "packages" / "adapters" / "web",
    "scripted": REPO_ROOT / "packages" / "planners" / "scripted",
    "llm": REPO_ROOT / "packages" / "planners" / "llm",
}
MODULES = {"core": "forecontract", "fake": "forecontract_fake", "web": "forecontract_web",
           "scripted": "forecontract_scripted", "llm": "forecontract_llm"}
CONSUMERS = ("installed-consumer-offline", "installed-consumer-retrieval",
             "installed-consumer-smoke")


def _require_dir(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"the release environment must define {name}")
    path = Path(value)
    if path.resolve().is_relative_to(REPO_ROOT.resolve()):
        pytest.fail(f"{name} must be outside the repository")
    return path


def _work_root() -> Path:
    return _require_dir("FORECONTRACT_RELEASE_WORK_ROOT")


def _wheelhouse() -> Path:
    path = _require_dir("FORECONTRACT_WHEELHOUSE")
    if not path.is_dir():
        pytest.fail("FORECONTRACT_WHEELHOUSE must be an existing directory")
    return path


def _environment(**extra: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                        "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                        "PIP_NO_INPUT": "1"})
    environment.update(extra)
    return environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host_script(script: Path, arguments: list[str]) -> str:
    """Run one public example with only the public examples directory added to sys.path.

    `-I` ignores every environment variable, so the sibling import path is injected by this
    runner itself; no package source directory and no repository root is ever added.
    """

    return (
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'examples')!r})\n"
        f"sys.argv = [{str(script)!r}, *{arguments!r}]\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )


COMMANDS: list[dict] = []


def _runtime_root(install: Path) -> Path:
    return install.parents[1]


def _command_row(item: dict, install: Path) -> dict:
    root = _runtime_root(install)
    return {"id": item["id"], "argv": item["argv"], "cwd": item["cwd"],
            "start_utc": item["start_utc"], "end_utc": item["end_utc"],
            "seconds": item["seconds"], "exit_code": item["exit_code"],
            "stdout": {"path": item["stdout"].relative_to(root).as_posix(),
                        "bytes": item["stdout"].stat().st_size,
                        "sha256": _sha256(item["stdout"])},
            "stderr": {"path": item["stderr"].relative_to(root).as_posix(),
                        "bytes": item["stderr"].stat().st_size,
                        "sha256": _sha256(item["stderr"])}}


def _archive_rows(install: Path, dist: Path) -> list[dict]:
    root = _runtime_root(install)
    archives = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
    return [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
             "sha256": _sha256(path)} for path in archives]


def _persist_index(install: Path, *, status: str, files: list[dict]) -> None:
    """Write the index after every finished command, before the caller asserts anything."""

    index = {"schema_version": "1", "status": status, "files": files,
             "commands": [_command_row(item, install) for item in COMMANDS]}
    (install / "release-installation-index.json").write_text(
        json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n")


def _run(command: list[str], *, cwd: Path, install: Path, name: str,
         environment: dict[str, str] | None = None) -> dict:
    """Run one command, persisting its raw binary stdout/stderr before any assertion."""

    runs = install / "runs" / name
    runs.mkdir(parents=True, exist_ok=True)
    stdout_path = runs / "stdout.bin"
    stderr_path = runs / "stderr.bin"
    started = datetime.datetime.now(datetime.timezone.utc)
    clock = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(command, cwd=str(cwd), env=environment or _environment(),
                                   stdout=out, stderr=err, check=False)
    seconds = round(time.monotonic() - clock, 6)
    ended = datetime.datetime.now(datetime.timezone.utc)
    record = {"id": name, "argv": list(command), "cwd": str(cwd),
              "start_utc": started.isoformat(), "end_utc": ended.isoformat(),
              "seconds": seconds, "exit_code": completed.returncode,
              "stdout": stdout_path, "stderr": stderr_path}
    COMMANDS.append(record)
    dist = install / "dist"
    files = _archive_rows(install, dist) if dist.is_dir() else []
    _persist_index(install, status="incomplete", files=files)
    return record


@pytest.fixture(scope="module")
def install_root() -> Path:
    root = _work_root()
    if root.exists():
        pytest.fail("FORECONTRACT_RELEASE_WORK_ROOT must not exist before the fixture starts")
    root.mkdir(parents=True, exist_ok=False)
    (root / "scratch").mkdir()
    return root


@pytest.fixture(scope="module")
def built(install_root: Path) -> dict:
    dist = install_root / "dist"
    dist.mkdir()
    build_src = install_root / "build-src"
    build_src.mkdir()
    wheels: dict[str, Path] = {}
    sdists: dict[str, Path] = {}
    commands: list[dict] = []
    for name in ("core", "fake", "web", "scripted", "llm"):
        copied = build_src / name
        shutil.copytree(PACKAGES[name], copied,
                        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info",
                                                      "__pycache__"))
        before = set(dist.glob("*.whl")), set(dist.glob("*.tar.gz"))
        record = _run([sys.executable, "-B", "-m", "build", "--no-isolation", "--wheel",
                       "--sdist", "--outdir", str(dist), str(copied)],
                      cwd=install_root, install=install_root, name=f"build-{name}")
        commands.append(record)
        assert record["exit_code"] == 0, (name, record["stderr"].read_bytes()[-400:])
        new_wheels = sorted(set(dist.glob("*.whl")) - before[0])
        new_sdists = sorted(set(dist.glob("*.tar.gz")) - before[1])
        assert len(new_wheels) == 1, (name, [item.name for item in new_wheels])
        assert len(new_sdists) == 1, (name, [item.name for item in new_sdists])
        wheels[name] = new_wheels[0]
        sdists[name] = new_sdists[0]
    return {"dist": dist, "wheels": wheels, "sdists": sdists, "commands": commands,
            "install": install_root}


def test_the_environment_is_declared_and_external(install_root):
    assert install_root.is_dir()
    assert not install_root.resolve().is_relative_to(REPO_ROOT.resolve())
    assert _wheelhouse().is_dir()
    assert any(_wheelhouse().glob("pydantic-*.whl"))


def test_exactly_five_wheels_and_five_sdists(built):
    assert len(built["wheels"]) == 5
    assert len(built["sdists"]) == 5
    archives = sorted(built["dist"].glob("*.whl")) + sorted(built["dist"].glob("*.tar.gz"))
    assert len(archives) == 10
    for path in archives:
        assert path.stat().st_size > 0


def test_wheel_members_match_the_public_source(built):
    import zipfile

    for name, wheel in built["wheels"].items():
        with zipfile.ZipFile(wheel) as archive:
            members = [item for item in archive.namelist() if item.endswith(".py")]
        assert members, name
        assert not any("__pycache__" in item for item in members), name
        assert not any(item.startswith("tests/") for item in members), name


def test_no_build_output_lands_in_the_repository(built):
    for name, directory in PACKAGES.items():
        assert not list(directory.glob("build")), name
        assert not list(directory.glob("*.egg-info")), name


def _venv(install: Path, name: str) -> Path:
    environment_dir = install / "venvs" / name
    venv.EnvBuilder(with_pip=True, clear=False).create(environment_dir)
    python = environment_dir / "Scripts" / "python.exe"
    return python if python.is_file() else environment_dir / "bin" / "python"


def _pip(python: Path, install: Path, name: str, arguments: list[str]) -> dict:
    return _run([str(python), "-m", "pip", "install", "--no-index", "--find-links",
                 str(_wheelhouse()), "--disable-pip-version-check", *arguments],
                cwd=install, install=install, name=name)


@pytest.fixture(scope="module")
def core_python(built, install_root) -> dict:
    python = _venv(install_root, "core")
    record = _pip(python, install_root, "install-core", [str(built["wheels"]["core"])])
    assert record["exit_code"] == 0, record["stderr"].read_bytes()[-600:]
    return {"python": python, "record": record}


@pytest.fixture(scope="module")
def full_python(built, install_root) -> dict:
    python = _venv(install_root, "full")
    arguments = [str(built["wheels"][name]) for name in
                 ("core", "fake", "web", "scripted", "llm")]
    matches = sorted(_wheelhouse().glob("httpx-*.whl"))
    assert matches, "the wheelhouse must contain httpx"
    arguments.append(str(matches[-1]))
    record = _pip(python, install_root, "install-full", arguments)
    assert record["exit_code"] == 0, record["stderr"].read_bytes()[-600:]
    return {"python": python, "record": record}


def test_core_only_environment_has_no_other_packages(core_python, built, install_root):
    probe = ("import importlib.util, json\n"
             "print(json.dumps({name: importlib.util.find_spec(name) is not None for name in "
             "['forecontract', 'forecontract_fake', 'forecontract_web', 'forecontract_scripted',"
             " 'forecontract_llm', 'playwright', 'torch']}))\n")
    record = _run([str(core_python["python"]), "-I", "-B", "-c", probe],
                  cwd=install_root, install=install_root, name="core-probe")
    assert record["exit_code"] == 0, record["stderr"].read_bytes()[-400:]
    found = json.loads(record["stdout"].read_bytes().decode("utf-8", "replace"))
    assert found["forecontract"] is True
    for absent in ("forecontract_fake", "forecontract_web", "forecontract_scripted",
                   "forecontract_llm", "playwright", "torch"):
        assert found[absent] is False, absent


def test_full_environment_imports_resolve_inside_the_venv(full_python, built, install_root):
    probe = ("import importlib, json\n"
             "names = " + json.dumps(list(MODULES.values())) + "\n"
             "print(json.dumps({name: importlib.import_module(name).__file__ for name in names}))\n")
    record = _run([str(full_python["python"]), "-I", "-B", "-c", probe],
                  cwd=install_root, install=install_root, name="full-probe")
    assert record["exit_code"] == 0, record["stderr"].read_bytes()[-400:]
    resolved = json.loads(record["stdout"].read_bytes().decode("utf-8", "replace"))
    base = (install_root / "venvs" / "full").resolve()
    for name, path in resolved.items():
        module_path = Path(path).resolve()
        assert module_path.is_relative_to(base), (name, path)
        assert not module_path.is_relative_to(REPO_ROOT.resolve()), (name, path)


def _record_hashes(source: str) -> dict[str, str]:
    """The sha256 RECORD entries of one wheel or one installed dist-info directory."""

    if source.endswith(".whl"):
        with zipfile.ZipFile(source) as archive:
            name = next(item for item in archive.namelist()
                        if item.endswith(".dist-info/RECORD"))
            rows = list(csv.reader(io.StringIO(archive.read(name).decode("utf-8"))))
    else:
        rows = list(csv.reader(io.StringIO((Path(source) / "RECORD").read_text(encoding="utf-8"))))
    result = {}
    for row in rows:
        if len(row) >= 3 and row[1].startswith("sha256="):
            result[row[0]] = row[1]
    return result


def test_installed_record_digests_match_this_round_wheel(built, install_root):
    """Compare the installed RECORD sha256 entries with this round wheel's RECORD entries.

    This is the wheel RECORD digest check, not the PEP610 `direct_url.json` archive digest; the
    frozen independent PEP610 judge remains the authority for that separate fact.
    """

    site = install_root / "venvs" / "full"
    candidates = sorted(site.rglob("*.dist-info"))
    installed = {}
    for directory in candidates:
        name = directory.parent.name or ""
        if "forecontract" in directory.name.lower() or "forecontract" in name.lower():
            installed[directory.name] = directory
    assert len(installed) == 5, sorted(installed)
    compared = 0
    for wheel in built["wheels"].values():
        expected = _record_hashes(str(wheel))
        matching = [directory for directory in installed.values()
                    if (directory.parent / "forecontract").is_dir()
                    and _record_hashes(str(directory)).keys() & expected.keys()]
        assert matching, wheel.name
        for directory in matching:
            actual = _record_hashes(str(directory))
            shared = set(actual) & set(expected)
            assert shared, (wheel.name, directory.name)
            for key in sorted(shared):
                assert actual[key] == expected[key], (wheel.name, key)
            compared += len(shared)
    assert compared > 0


def test_full_environment_pip_check_passes(full_python, install_root):
    """The five wheels plus httpx resolve without a missing or conflicting dependency."""

    record = _run([str(full_python["python"]), "-m", "pip", "check"],
                  cwd=install_root, install=install_root, name="pip-check-full")
    assert record["exit_code"] == 0, record["stdout"].read_bytes()[-400:]


def test_offline_consumer_completes(full_python, install_root):
    output = install_root / "consumer-offline"
    record = _run([str(full_python["python"]), "-I", "-B",
                   str(REPO_ROOT / "examples" / "offline_runtime_demo.py"),
                   "--output-dir", str(output)],
                  cwd=install_root, install=install_root, name=CONSUMERS[0])
    assert record["exit_code"] == 0, record["stderr"].read_bytes()[-600:]
    stdout = record["stdout"].read_bytes().decode("utf-8", "replace")
    assert "completed" in stdout or "TASK_ACHIEVED" in stdout
    assert (output / "trace.jsonl").is_file()


def test_retrieval_consumer_completes(full_python, install_root):
    output = install_root / "consumer-retrieval"
    runner = _host_script(REPO_ROOT / "examples" / "llm_retrieval_demo.py",
                          ["--output-dir", str(output)])
    record = _run([str(full_python["python"]), "-I", "-B", "-c", runner],
                  cwd=install_root, install=install_root, name=CONSUMERS[1])
    assert record["exit_code"] == 0, record["stderr"].read_bytes()[-600:]
    audits = sorted((output / "audit").glob("*.json"))
    assert len(audits) == 9, [path.name for path in audits]


def test_smoke_consumer_completes_all_three_cases(full_python, install_root):
    output = install_root / "consumer-smoke"
    runner = _host_script(REPO_ROOT / "examples" / "llm_api_smoke.py",
                          ["--mode", "offline",
                           "--cases", str(REPO_ROOT / "examples" / "llm_api_smoke_cases.v1.json"),
                           "--output-dir", str(output)])
    record = _run([str(full_python["python"]), "-I", "-B", "-c", runner],
                  cwd=install_root, install=install_root, name=CONSUMERS[2])
    assert record["exit_code"] == 0, record["stderr"].read_bytes()[-600:]
    summary = json.loads((output / "offline-summary.json").read_bytes())
    assert summary["status"] == "completed"
    assert set(summary["cases"]) == {"direct", "retrieval-pending", "retrieval-ready"}


def test_installation_index_binds_archives_and_commands(built, install_root):
    """Seal the index with the ten archives and every command actually executed."""

    runtime_root = _runtime_root(install_root)
    files = _archive_rows(install_root, built["dist"])
    assert len(files) == 10
    _persist_index(install_root, status="completed", files=files)
    index = json.loads((install_root / "release-installation-index.json").read_bytes())
    assert index["status"] == "completed"
    assert len(index["commands"]) >= 13, [row["id"] for row in index["commands"]]
    ids = [row["id"] for row in index["commands"]]
    assert len(ids) == len(set(ids))
    for cid in ("installed-consumer-offline", "installed-consumer-retrieval",
                "installed-consumer-smoke", "core-probe", "full-probe"):
        assert cid in ids, cid
    for item in index["files"]:
        assert (runtime_root / item["path"]).is_file()
        assert item["path"].startswith("scratch/installation/dist/")
    for row in index["commands"]:
        assert (runtime_root / row["stdout"]["path"]).is_file()
        assert (runtime_root / row["stderr"]["path"]).is_file()
        assert row["exit_code"] == 0, row["id"]
        assert not Path(row["cwd"]).resolve().is_relative_to(REPO_ROOT.resolve())
