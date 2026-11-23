"""Unit tests for offline Web verification replay, diffing and input/output rejection."""

from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Self
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = (
    "packages/core/src",
    "packages/adapters/fake/src",
    "packages/adapters/web/src",
    "packages/planners/scripted/src",
    "packages/planners/llm/src",
    "examples",
)


def _fixtures_root() -> Path:
    env_path = os.environ.get("P5_REPLAY_FIXTURES")
    if env_path:
        return Path(env_path)
    return REPO_ROOT / "artifacts/verification/P5-T01/round-001-inputs/fixtures"


def _load_deps() -> tuple[Any, Any, Any]:
    for candidate in _SOURCES:
        entry = str(REPO_ROOT / candidate)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import replay_trace
    from forecontract_web.verification import (
        WebStepVerifier,
        WebTaskCompletionOracle,
    )

    return replay_trace, WebStepVerifier, WebTaskCompletionOracle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _rehash_member(root: Path, rel_path: str) -> None:
    index_path = root / "evidence-index.json"
    idx = _read_json(index_path)
    target = root / rel_path
    new_sha = _sha256(target)
    new_size = target.stat().st_size

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("path") == rel_path:
                if "sha256" in node:
                    node["sha256"] = new_sha
                if "bytes" in node:
                    node["bytes"] = new_size
            for val in node.values():
                _visit(val)
        elif isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(idx)
    _write_json(index_path, idx)


def _copy_normal(tmp_path: Path, name: str = "case") -> Path:
    dest = tmp_path / name
    shutil.copytree(_fixtures_root() / "normal", dest)
    return dest


def test_import_performs_no_io_or_subprocess() -> None:
    _load_deps()
    spec = importlib.util.spec_from_file_location(
        "replay_trace_isolated_import",
        REPO_ROOT / "examples/replay_trace.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        patch.object(Path, "read_bytes", side_effect=AssertionError("import read_bytes")),
        patch.object(Path, "read_text", side_effect=AssertionError("import read_text")),
        patch.object(subprocess, "Popen", side_effect=AssertionError("import Popen")),
    ):
        spec.loader.exec_module(module)
    assert hasattr(module, "replay_evidence")
    assert hasattr(module, "ReplayError")


def test_differences_distinguish_missing_from_null_and_arrays() -> None:
    replay_trace, _, _ = _load_deps()
    diffs_missing = replay_trace._compute_differences({}, {"a": None})
    assert len(diffs_missing) == 1
    assert diffs_missing[0] == {
        "path": "/a",
        "recorded_present": False,
        "recomputed_present": True,
        "recorded": None,
        "recomputed": None,
    }

    diffs_array = replay_trace._compute_differences([1, 2], [1])
    assert len(diffs_array) == 1
    assert diffs_array[0] == {
        "path": "/1",
        "recorded_present": True,
        "recomputed_present": False,
        "recorded": 2,
        "recomputed": None,
    }

    diffs_escape = replay_trace._compute_differences({"a/b~c": 1}, {"a/b~c": 2})
    assert diffs_escape[0]["path"] == "/a~1b~0c"


def test_subresult_difference_detected_and_cli_exits_2(tmp_path: Path) -> None:
    replay_trace, _, _ = _load_deps()
    root = _copy_normal(tmp_path, "diff_case")
    idx = _read_json(root / "evidence-index.json")
    rel = idx["verifications"][0]["path"]
    snap_path = root / rel
    snap = _read_json(snap_path)
    snap["payload"]["result"]["outcome_results"][0]["reason"] = "REPLAY_TEST_DIFFERENCE"
    _write_json(snap_path, snap)
    _rehash_member(root, rel)

    report = replay_trace.replay_evidence(root)
    assert report["replay_status"] == "different"
    assert report["difference_count"] == 1
    assert any(
        d["path"] == "/outcome_results/0/reason"
        and d["recorded"] == "REPLAY_TEST_DIFFERENCE"
        for row in report["verifications"]
        for d in row["differences"]
    )

    out_file = tmp_path / "diff_report.json"
    exit_code = replay_trace.main(["--input", str(root), "--output", str(out_file)])
    assert exit_code == 2
    assert out_file.is_file()
    persisted = _read_json(out_file)
    assert persisted == report


@pytest.mark.parametrize(
    "mode",
    ["missing", "unlisted", "extra", "duplicate", "association", "no_index", "bad_schema", "bad_kind", "bad_verifier"],
)
def test_inventory_and_schema_violations_rejected_before_verifiers(
    tmp_path: Path,
    mode: str,
) -> None:
    replay_trace, step_cls, task_cls = _load_deps()
    root = _copy_normal(tmp_path, f"inv_{mode}")
    idx_path = root / "evidence-index.json"
    idx = _read_json(idx_path)
    rel = idx["verifications"][0]["path"]

    if mode == "missing":
        (root / rel).unlink()
    elif mode == "unlisted":
        idx["snapshots"] = [x for x in idx["snapshots"] if x["path"] != rel]
        _write_json(idx_path, idx)
    elif mode == "extra":
        (root / "snapshots/extra.json").write_bytes(b"{}\n")
    elif mode == "duplicate":
        idx["verifications"].append(copy.deepcopy(idx["verifications"][0]))
        _write_json(idx_path, idx)
    elif mode == "association":
        idx["verifications"][0]["attempt_id"] = "other-run-attempt"
        _write_json(idx_path, idx)
    elif mode == "no_index":
        idx_path.unlink()
    elif mode == "bad_schema":
        idx["schema_version"] = "9.9"
        _write_json(idx_path, idx)
    elif mode == "bad_kind":
        idx["kind"] = "unknown_kind"
        _write_json(idx_path, idx)
    elif mode == "bad_verifier":
        idx["verifications"][0]["verifier"] = "unknown_verifier"
        _write_json(idx_path, idx)

    with (
        patch.object(step_cls, "verify_step", side_effect=AssertionError("step verifier called")) as step_mock,
        patch.object(task_cls, "evaluate_completion", side_effect=AssertionError("task oracle called")) as task_mock,
    ):
        with pytest.raises(replay_trace.ReplayError) as exc_info:
            replay_trace.replay_evidence(root)
        assert exc_info.value.code == "REPLAY_INPUT_INVALID"
        assert step_mock.call_count == 0
        assert task_mock.call_count == 0

    out_path = tmp_path / f"out_{mode}.json"
    assert replay_trace.main(["--input", str(root), "--output", str(out_path)]) == 3
    assert not out_path.exists()


@pytest.mark.parametrize("target", ["index", "snapshot", "memory", "trace"])
@pytest.mark.parametrize("kind", ["duplicate", "bom", "nan", "hash"])
def test_malformed_json_and_hash_mismatches_rejected_without_leaking_sentinel(
    tmp_path: Path,
    target: str,
    kind: str,
) -> None:
    replay_trace, step_cls, task_cls = _load_deps()
    root = _copy_normal(tmp_path, f"mal_{target}_{kind}")
    idx = _read_json(root / "evidence-index.json")
    rel = {
        "index": "evidence-index.json",
        "snapshot": idx["verifications"][0]["path"],
        "memory": idx["working_memory"]["path"],
        "trace": idx["trace"]["path"],
    }[target]
    file_path = root / rel
    raw = file_path.read_bytes()
    if kind == "duplicate":
        raw = raw.replace(b"{", b'{"probe":"SENSITIVE_SENTINEL","probe":1,', 1)
    elif kind == "bom":
        raw = b"\xef\xbb\xbf" + raw
    elif kind == "nan":
        raw = raw.replace(b"{", b'{"probe":NaN,', 1)
    else:
        raw += b" "
    file_path.write_bytes(raw)
    if kind != "hash" and target != "index":
        _rehash_member(root, rel)

    with (
        patch.object(step_cls, "verify_step", side_effect=AssertionError("verifier invoked")) as step_mock,
        patch.object(task_cls, "evaluate_completion", side_effect=AssertionError("oracle invoked")) as task_mock,
    ):
        with pytest.raises(replay_trace.ReplayError) as exc_info:
            replay_trace.replay_evidence(root)
        assert exc_info.value.code == "REPLAY_INPUT_INVALID"
        assert "SENSITIVE_SENTINEL" not in str(exc_info.value)
        assert step_mock.call_count == 0
        assert task_mock.call_count == 0


def test_path_traversal_and_junction_rejected(tmp_path: Path) -> None:
    replay_trace, _, _ = _load_deps()
    for bad_path, label in [("../outside.json", "traversal"), ("C:/outside.json", "abs")]:
        root = _copy_normal(tmp_path, f"path_{label}")
        idx = _read_json(root / "evidence-index.json")
        idx["trace"]["path"] = bad_path
        _write_json(root / "evidence-index.json", idx)
        with pytest.raises(replay_trace.ReplayError) as exc_info:
            replay_trace.replay_evidence(root)
        assert exc_info.value.code == "REPLAY_INPUT_INVALID"

    outside = tmp_path / "outside_target"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"intact")
    j_root = _copy_normal(tmp_path, "junction_case")
    link = j_root / "linked"
    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail("Junction creation failed in test scratch")
    with pytest.raises(replay_trace.ReplayError) as exc_info:
        replay_trace.replay_evidence(j_root)
    assert exc_info.value.code == "REPLAY_INPUT_INVALID"
    assert sentinel.read_bytes() == b"intact"

    out_via_link = link / "report.json"
    code = replay_trace.main(["--input", str(_fixtures_root() / "normal"), "--output", str(out_via_link)])
    assert code == 4
    assert not out_via_link.exists()


def test_concurrent_input_modification_rejected(tmp_path: Path) -> None:
    replay_trace, step_cls, _ = _load_deps()
    root = _copy_normal(tmp_path, "concurrent")
    orig_verify = step_cls.verify_step

    def _mutate(self: Any, **kwargs: Any) -> Any:
        res = orig_verify(self, **kwargs)
        trace_p = root / "trace.jsonl"
        trace_p.write_bytes(trace_p.read_bytes() + b" ")
        return res

    out_file = tmp_path / "concurrent_out.json"
    with patch.object(step_cls, "verify_step", _mutate):
        with pytest.raises(replay_trace.ReplayError) as exc_info:
            replay_trace.replay_evidence(root)
        assert exc_info.value.code == "REPLAY_INPUT_INVALID"

    root2 = _copy_normal(tmp_path, "concurrent2")

    def _mutate2(self: Any, **kwargs: Any) -> Any:
        res = orig_verify(self, **kwargs)
        trace_p = root2 / "trace.jsonl"
        trace_p.write_bytes(trace_p.read_bytes() + b" ")
        return res

    with patch.object(step_cls, "verify_step", _mutate2):
        assert replay_trace.main(["--input", str(root2), "--output", str(out_file)]) == 3
    assert not out_file.exists()


def test_output_path_errors_exit_4_and_preserve_existing_files(tmp_path: Path) -> None:
    replay_trace, _, _ = _load_deps()
    root = _copy_normal(tmp_path, "out_err")
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"preserve-me")
    assert replay_trace.main(["--input", str(root), "--output", str(existing)]) == 4
    assert existing.read_bytes() == b"preserve-me"

    inside = root / "inside.json"
    assert replay_trace.main(["--input", str(root), "--output", str(inside)]) == 4
    assert not inside.exists()

    blocker = tmp_path / "file_parent"
    blocker.write_bytes(b"blocker")
    assert replay_trace.main(["--input", str(root), "--output", str(blocker / "report.json")]) == 4
    assert blocker.read_bytes() == b"blocker"

    partial_dir = tmp_path / "partial_dir"
    partial_out = partial_dir / "partial.json"
    orig_open = open
    partial_hits: list[int] = []

    class _PartialWriter:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        def write(self, data: bytes | str) -> int:
            chunk = data[:17]
            self._wrapped.write(chunk)
            self._wrapped.flush()
            partial_hits.append(len(chunk))
            raise OSError(errno.ENOSPC, "simulated partial write failure")

        def __enter__(self) -> Self:
            self._wrapped.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> Any:
            return self._wrapped.__exit__(exc_type, exc, tb)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    def _patched_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = orig_open(file, mode, *args, **kwargs)
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            return _PartialWriter(handle)
        return handle

    with patch("builtins.open", side_effect=_patched_open):
        assert replay_trace.main(["--input", str(root), "--output", str(partial_out)]) == 4
    assert partial_hits == [17]
    assert not partial_out.exists()
    assert list(partial_dir.iterdir()) == []


def test_deterministic_serialization_and_interrupt_propagation(tmp_path: Path) -> None:
    replay_trace, step_cls, _ = _load_deps()
    root = _copy_normal(tmp_path, "det")
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    assert replay_trace.main(["--input", str(root), "--output", str(one)]) == 0
    assert replay_trace.main(["--input", str(root), "--output", str(two)]) == 0
    raw = one.read_bytes()
    assert raw == two.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")

    idx = _read_json(root / "evidence-index.json")
    idx["verifications"].reverse()
    _write_json(root / "evidence-index.json", idx)
    reversed_report = replay_trace.replay_evidence(root)
    assert reversed_report["verifications"] == _read_json(one)["verifications"]

    for irq in (KeyboardInterrupt, SystemExit):
        with (
            patch.object(step_cls, "verify_step", side_effect=irq),
            pytest.raises(irq),
        ):
            replay_trace.replay_evidence(_fixtures_root() / "normal")
