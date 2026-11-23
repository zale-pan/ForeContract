"""Unit tests for ``forecontract_coding.workspace`` covering C03, C04, and C05."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
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

from forecontract_coding.errors import (
    CODING_FILE_CONFLICT,
    CODING_INPUT_INVALID,
    CODING_SCOPE_VIOLATION,
    CodingError,
)
from forecontract_coding.models import (
    CodingConfig,
    CodingTask,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
    canonicalize_relative_path,
)
from forecontract_coding.workspace import ManagedWorkspace, sha256_bytes


def _make_task(
    tmp_path: Path,
    *,
    read_paths: list[str],
    write_paths: list[WriteTarget],
    max_file_bytes: int = 64 * 1024,
) -> tuple[CodingTask, CodingConfig, Path, Path]:
    ws = tmp_path / "ws"
    out = tmp_path / "out"
    ws.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    ext = tmp_path / "external_tests"
    ext.mkdir(parents=True, exist_ok=True)
    test_file = ext / "test_acc.py"
    test_file.write_text("def test_ok(): assert True\n", encoding="utf-8")

    task = CodingTask(
        run_id="run-ws",
        goal_id="goal-ws",
        goal_text="workspace tests",
        workspace_root=str(ws),
        read_paths=read_paths,
        write_paths=write_paths,
        commands={
            "acc": CommandSpec(
                command_id="acc",
                executable=sys.executable,
                args=[
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "--junitxml={result_dir}/junit.xml",
                    str(test_file),
                ],
                cwd=str(ws),
                timeout_ms=5000,
                purpose="acceptance",
                expected_tests=[ExpectedTestCase(classname="test_acc", name="test_ok")],
            )
        },
        verification_command_ids=["acc"],
        max_steps=5,
        max_planning_calls=5,
        max_replans=2,
        run_timeout_ms=20000,
    )
    cfg = CodingConfig(output_root=str(out), max_file_bytes=max_file_bytes)
    return task, cfg, ws, test_file


def test_c03_read_pagination_multibyte_utf8_and_sorted_search_truncation(
    tmp_path: Path,
) -> None:
    task, cfg, ws, _ = _make_task(
        tmp_path,
        read_paths=["b.py", "a.py", "big.txt"],
        write_paths=[],
    )
    (ws / "a.py").write_text("# 注释 alpha\nx = 'hit'\ny = 'hit'\n", encoding="utf-8")
    (ws / "b.py").write_text("z = 'hit'\n# 中文结尾\n", encoding="utf-8")
    # 25 KiB text guarantees >16 KiB single-read cap cleanly across UTF-8 multibyte lines
    big_lines = [f"line-{i:04d}-测试文本\n" for i in range(1200)]
    (ws / "big.txt").write_text("".join(big_lines), encoding="utf-8")

    mw = ManagedWorkspace(task=task, config=cfg)
    chunk1 = mw.read_file(path="a.py", start_line=1, max_lines=2)
    assert chunk1.sha256 == sha256_bytes((ws / "a.py").read_bytes())
    assert chunk1.truncated is True
    assert chunk1.next_start_line == 3
    assert "注释 alpha" in chunk1.content

    chunk_big1 = mw.read_file(path="big.txt", start_line=1, max_lines=200)
    assert len(chunk_big1.content.encode("utf-8")) <= 16 * 1024
    assert chunk_big1.truncated is True
    assert chunk_big1.next_start_line is not None and chunk_big1.next_start_line > 1
    chunk_big2 = mw.read_file(path="big.txt", start_line=chunk_big1.next_start_line, max_lines=200)
    assert chunk_big2.start_line == chunk_big1.next_start_line

    search_res = mw.search_text(query="hit", paths=["b.py", "a.py"], max_matches=2)
    assert search_res.truncated is True
    assert search_res.searched_paths == ("a.py", "b.py")
    assert [(m.path, m.line_number) for m in search_res.matches] == [("a.py", 2), ("a.py", 3)]
    assert search_res.next_cursor == {"path": "b.py", "line_number": 1}


def test_c04_create_modify_conflict_and_atomic_preserve(tmp_path: Path) -> None:
    task, cfg, ws, _ = _make_task(
        tmp_path,
        read_paths=["mod.py", "created.py"],
        write_paths=[
            WriteTarget(path="mod.py", operation="modify"),
            WriteTarget(path="created.py", operation="create"),
        ],
    )
    (ws / "mod.py").write_bytes(b"v1\n")
    mw = ManagedWorkspace(task=task, config=cfg)

    rev_0, snaps_0, _ = mw.snapshot()
    # Absent create target must be included in deterministic revision
    created_snap = next(s for s in snaps_0 if s.path == "created.py")
    assert created_snap.exists is False and created_snap.sha256 is None

    # Create succeeds cleanly with read-back check
    c_out = mw.write_file(path="created.py", expected_sha256=None, content="created=1\n")
    assert c_out.read_back_match is True
    assert c_out.before_sha256 is None
    assert c_out.read_back_sha256 == hashlib.sha256(b"created=1\n").hexdigest()
    assert c_out.after_revision != rev_0

    # Duplicate create fails cleanly
    with pytest.raises(CodingError) as exc_info:
        mw.write_file(path="created.py", expected_sha256=None, content="overwrite\n")
    assert exc_info.value.code == CODING_FILE_CONFLICT
    assert (ws / "created.py").read_bytes() == b"created=1\n"

    # Simulate external user edit before write_file
    (ws / "mod.py").write_bytes(b"user-edited\n")
    stale_sha = hashlib.sha256(b"v1\n").hexdigest()
    with pytest.raises(CodingError) as conflict_err:
        mw.write_file(path="mod.py", expected_sha256=stale_sha, content="agent-overwrite\n")
    assert conflict_err.value.code == CODING_FILE_CONFLICT
    assert (ws / "mod.py").read_bytes() == b"user-edited\n"

    # Correct current SHA succeeds
    cur_sha = hashlib.sha256(b"user-edited\n").hexdigest()
    m_out = mw.write_file(path="mod.py", expected_sha256=cur_sha, content="v2\n")
    assert m_out.read_back_match is True
    assert (ws / "mod.py").read_bytes() == b"v2\n"


@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside.txt",
        "sub/../../outside.txt",
        "/abs.py",
        "\\abs.py",
        "C:\\Windows\\win.ini",
        "C:rel.txt",
        "\\\\server\\share\\file.py",
        "app.py:hidden_ads",
        "CON",
        "nul.txt",
        "sub/COM1.py",
        "lpt9",
        "file.py ",
        "file.py.",
        "dir /sub.py",
    ],
)
def test_c05_dangerous_relative_paths_rejected(bad_path: str) -> None:
    with pytest.raises(ValueError):
        canonicalize_relative_path(bad_path)


def test_c05_case_collision_hardlink_junction_and_sentinel_untouched(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    sentinel = tmp_path / "outside_sentinel.txt"
    sentinel.write_bytes(b"SENSITIVE-OUTSIDE-BYTES")
    sentinel_sha = sha256_bytes(sentinel.read_bytes())

    # Case-insensitive collision rejected in CodingTask
    with pytest.raises(ValueError):
        CodingTask(
            run_id="r",
            goal_id="g",
            goal_text="t",
            workspace_root=str(ws),
            read_paths=["Foo.py", "foo.py"],
            write_paths=[],
            commands={
                "acc": CommandSpec(
                    command_id="acc",
                    executable=sys.executable,
                    args=["--junitxml={result_dir}/junit.xml", str(tmp_path / "x.py")],
                    cwd=str(ws),
                    timeout_ms=5000,
                    purpose="acceptance",
                    expected_tests=[ExpectedTestCase(classname="x", name="test_x")],
                )
            },
            verification_command_ids=["acc"],
            max_steps=2,
            max_planning_calls=2,
            max_replans=1,
            run_timeout_ms=10000,
        )

    # Multi-hardlink writable target rejected
    task, cfg, ws2, ext_test = _make_task(
        tmp_path / "hardlink_case",
        read_paths=["target.py"],
        write_paths=[WriteTarget(path="target.py", operation="modify")],
    )
    ext_before = ext_test.read_bytes()
    target = ws2 / "target.py"
    target.write_bytes(b"orig\n")
    link_outside = tmp_path / "hardlink_alias.py"
    os.link(target, link_outside)
    mw = ManagedWorkspace(task=task, config=cfg)
    with pytest.raises(CodingError) as hl_err:
        mw.snapshot()
    assert hl_err.value.code == CODING_SCOPE_VIOLATION

    # Directory junction inside workspace rejected
    task_j, cfg_j, ws_j, _ = _make_task(
        tmp_path / "junction_case",
        read_paths=["linked_dir/secret.py"],
        write_paths=[WriteTarget(path="linked_dir/secret.py", operation="modify")],
    )
    outside_dir = tmp_path / "outside_target_dir"
    outside_dir.mkdir()
    (outside_dir / "secret.py").write_bytes(b"escape\n")
    res = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(ws_j / "linked_dir"), str(outside_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    mw_j = ManagedWorkspace(task=task_j, config=cfg_j)
    with pytest.raises(CodingError) as j_err:
        mw_j.write_file(
            path="linked_dir/secret.py",
            expected_sha256=sha256_bytes(b"escape\n"),
            content="pwn\n",
        )
    assert j_err.value.code == CODING_SCOPE_VIOLATION
    assert (outside_dir / "secret.py").read_bytes() == b"escape\n"
    assert sha256_bytes(sentinel.read_bytes()) == sentinel_sha
    assert ext_test.read_bytes() == ext_before


def test_d02_long_single_line_byte_offset_pagination_crlf_and_errors(tmp_path: Path) -> None:
    task, cfg, ws, _ = _make_task(
        tmp_path,
        read_paths=["long.txt"],
        write_paths=[WriteTarget(path="long.txt", operation="modify")],
    )
    # Force multibyte 3-byte ('汉') and 4-byte ('😀') chars right across 16 KiB boundaries + CRLF
    line1 = ("a" * 16382) + "汉" + ("b" * 16381) + "😀" + "end\r\n"
    line2 = "second-line-ok\r\n"
    raw_bytes = (line1 + line2).encode("utf-8")
    (ws / "long.txt").write_bytes(raw_bytes)
    expected_sha = sha256_bytes(raw_bytes)

    mw = ManagedWorkspace(task=task, config=cfg)
    chunks: list[str] = []
    start_line = 1
    offset = 0
    sha_arg: str | None = None
    for _ in range(10):
        chunk = mw.read_file(
            path="long.txt",
            start_line=start_line,
            max_lines=1,
            line_byte_offset=offset,
            expected_sha256=sha_arg,
        )
        assert len(chunk.content.encode("utf-8")) <= 16 * 1024
        chunks.append(chunk.content)
        if not chunk.truncated:
            assert chunk.next_start_line is None
            assert chunk.next_line_byte_offset is None
            break
        assert chunk.next_start_line is not None
        assert chunk.next_line_byte_offset is not None
        if chunk.next_start_line == start_line:
            assert chunk.next_line_byte_offset > offset
        start_line = chunk.next_start_line
        offset = chunk.next_line_byte_offset
        sha_arg = chunk.sha256
    else:
        raise AssertionError("pagination did not reach EOF cleanly")

    assert "".join(chunks).encode("utf-8") == raw_bytes
    assert "\r\n" in "".join(chunks)

    # Rejects missing expected_sha256 when line_byte_offset > 0
    with pytest.raises(CodingError) as exc_missing_sha:
        mw.read_file(path="long.txt", start_line=1, max_lines=1, line_byte_offset=10)
    assert exc_missing_sha.value.code == CODING_INPUT_INVALID

    # Rejects mid-UTF-8 continuation byte (line1 starts with 16382 'a' then 3-byte '汉' at 16382..16385)
    with pytest.raises(CodingError) as exc_mid_utf8:
        mw.read_file(
            path="long.txt",
            start_line=1,
            max_lines=1,
            line_byte_offset=16383,
            expected_sha256=expected_sha,
        )
    assert exc_mid_utf8.value.code == CODING_INPUT_INVALID

    # Rejects out-of-range offset or non-existent line with offset > 0
    with pytest.raises(CodingError) as exc_oob:
        mw.read_file(
            path="long.txt",
            start_line=2,
            max_lines=1,
            line_byte_offset=999,
            expected_sha256=expected_sha,
        )
    assert exc_oob.value.code == CODING_INPUT_INVALID

    # Detects concurrent edit on continuation (both mid-line and offset=0 with expected_sha256)
    (ws / "long.txt").write_bytes(b"changed\r\n")
    with pytest.raises(CodingError) as exc_conflict_mid:
        mw.read_file(
            path="long.txt",
            start_line=1,
            max_lines=1,
            line_byte_offset=2,
            expected_sha256=expected_sha,
        )
    assert exc_conflict_mid.value.code == CODING_FILE_CONFLICT

    with pytest.raises(CodingError) as exc_conflict_zero:
        mw.read_file(
            path="long.txt",
            start_line=1,
            max_lines=1,
            line_byte_offset=0,
            expected_sha256=expected_sha,
        )
    assert exc_conflict_zero.value.code == CODING_FILE_CONFLICT


def test_d09_write_file_exclusive_commit_boundary_and_concurrent_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, cfg, ws, _ = _make_task(
        tmp_path,
        read_paths=["target.py", "fresh.py"],
        write_paths=[
            WriteTarget(path="target.py", operation="modify"),
            WriteTarget(path="fresh.py", operation="create"),
        ],
    )
    target = ws / "target.py"
    target.write_bytes(b"initial = 'very long original line for truncation test'\n")
    mw = ManagedWorkspace(task=task, config=cfg)

    # 1. Shorter replacement truncates cleanly with exact read-back match
    sha_init = sha256_bytes(target.read_bytes())
    out_short = mw.write_file(path="target.py", expected_sha256=sha_init, content="x=1\n")
    assert out_short.read_back_match is True
    assert target.read_bytes() == b"x=1\n"

    # 2. Peer commits before our exclusive commit opens handle -> CODING_FILE_CONFLICT & peer preserved
    sha_v2 = sha256_bytes(b"x=1\n")
    real_commit = ManagedWorkspace._commit_modify_exclusive

    def _peer_before_lock(self, *, file_path, parts, norm, before_sha, new_bytes):
        file_path.write_bytes(b"peer_committed = True\n")
        return real_commit(
            self,
            file_path=file_path,
            parts=parts,
            norm=norm,
            before_sha=before_sha,
            new_bytes=new_bytes,
        )

    monkeypatch.setattr(ManagedWorkspace, "_commit_modify_exclusive", _peer_before_lock)
    with pytest.raises(CodingError) as conflict_err:
        mw.write_file(path="target.py", expected_sha256=sha_v2, content="agent_lost = True\n")
    assert conflict_err.value.code == CODING_FILE_CONFLICT
    assert target.read_bytes() == b"peer_committed = True\n"
    monkeypatch.undo()

    # 3. Real commit-boundary fault injection + concurrent reader isolation + OS peer contention
    if os.name == "nt":
        from forecontract_coding import workspace as ws_mod

        old_bytes = b"A" * 24 + b"TAIL_OLD_DATA\n"
        new_bytes = b"B" * 24 + b"NEW_SHORT\n"
        sha_peer = sha256_bytes(b"peer_committed = True\n")
        mw.write_file(
            path="target.py",
            expected_sha256=sha_peer,
            content=old_bytes.decode("utf-8"),
        )
        assert target.read_bytes() == old_bytes
        sha_old = sha256_bytes(old_bytes)

        # 3a. Fault mid-write inside _win32_write_all after partial 8 bytes -> TxF rollback keeps old
        real_write_all = ws_mod._win32_write_all
        reader_seen_mid_fault: list[bytes] = []

        def _partial_then_fail(handle: int, data: bytes) -> None:
            real_write_all(handle, data[:8])
            reader_seen_mid_fault.append(target.read_bytes())
            raise OSError("simulated io failure after partial 8-byte write")

        monkeypatch.setattr(ws_mod, "_win32_write_all", _partial_then_fail)
        try:
            with pytest.raises(OSError, match="simulated io failure"):
                mw.write_file(
                    path="target.py",
                    expected_sha256=sha_old,
                    content=new_bytes.decode("utf-8"),
                )
        finally:
            monkeypatch.undo()

        observed_after_partial = target.read_bytes()
        assert observed_after_partial == old_bytes
        assert observed_after_partial != new_bytes[:8] + old_bytes[8:]
        assert reader_seen_mid_fault == [old_bytes]

        # 3b. Fault inside CommitTransaction -> TxF rollback keeps old cleanly
        real_commit_txn = ws_mod._win32_commit_transaction

        def _fail_commit(txn: int) -> None:
            raise OSError("simulated commit failure")

        monkeypatch.setattr(ws_mod, "_win32_commit_transaction", _fail_commit)
        try:
            with pytest.raises(OSError, match="simulated commit failure"):
                mw.write_file(
                    path="target.py",
                    expected_sha256=sha_old,
                    content=new_bytes.decode("utf-8"),
                )
        finally:
            monkeypatch.undo()
        assert target.read_bytes() == old_bytes

        # 3c. Fault during create -> TxF rollback leaves new target absent
        def _partial_create_fail(handle: int, data: bytes) -> None:
            real_write_all(handle, data[:6])
            raise OSError("simulated create io failure")

        monkeypatch.setattr(ws_mod, "_win32_write_all", _partial_create_fail)
        try:
            with pytest.raises(OSError, match="simulated create io failure"):
                mw.write_file(
                    path="fresh.py",
                    expected_sha256=None,
                    content="partial_should_never_exist = True\n",
                )
        finally:
            monkeypatch.undo()
        assert not (ws / "fresh.py").exists()

        # 3d. Concurrent reader visibility + external OS write/os.replace contention inside commit
        peer_replace_src = tmp_path / "peer_replace_src.py"
        peer_replace_src.write_bytes(b"peer_replace_overwrite\n")
        peer_write_blocked = False
        peer_replace_blocked = False
        commit_entered = threading.Event()
        release_commit = threading.Event()
        seen_by_reader: list[bytes] = []
        reader_errors: list[BaseException] = []
        stop_reader = threading.Event()

        def _reader_loop() -> None:
            while not stop_reader.is_set():
                try:
                    seen_by_reader.append(target.read_bytes())
                except OSError as exc:  # pragma: no cover - asserted empty below
                    reader_errors.append(exc)

        def _contend_inside_txf(txn: int) -> None:
            nonlocal peer_write_blocked, peer_replace_blocked
            commit_entered.set()
            assert target.read_bytes() == old_bytes
            proc_write = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    f"import pathlib; pathlib.Path(r'{target}').write_bytes(b'race_overwrite\\n')",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            peer_write_blocked = proc_write.returncode != 0
            try:
                os.replace(peer_replace_src, target)
            except OSError:
                peer_replace_blocked = True
            assert release_commit.wait(timeout=5.0)
            real_commit_txn(txn)

        reader_thread = threading.Thread(target=_reader_loop, daemon=True)
        reader_thread.start()
        monkeypatch.setattr(ws_mod, "_win32_commit_transaction", _contend_inside_txf)
        try:
            def _allow_commit() -> None:
                if commit_entered.wait(timeout=5.0):
                    release_commit.set()

            releaser = threading.Thread(target=_allow_commit, daemon=True)
            releaser.start()
            committed = mw.write_file(
                path="target.py",
                expected_sha256=sha_old,
                content=new_bytes.decode("utf-8"),
            )
            releaser.join(timeout=2.0)
        finally:
            stop_reader.set()
            reader_thread.join(timeout=2.0)
            monkeypatch.undo()

        assert peer_write_blocked is True
        assert peer_replace_blocked is True
        assert committed.read_back_match is True
        assert target.read_bytes() == new_bytes
        assert not reader_errors
        assert seen_by_reader
        assert set(seen_by_reader).issubset({old_bytes, new_bytes})
        assert target.read_bytes() == new_bytes

    # 4. Normal exclusive create still verifies atomic visibility
    created = mw.write_file(path="fresh.py", expected_sha256=None, content="created = 2\n")
    assert created.read_back_match is True
    assert (ws / "fresh.py").read_bytes() == b"created = 2\n"
