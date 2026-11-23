"""Guarded workspace file access, path/link safety checks, revisions, and atomic writes."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from forecontract_coding.errors import (
    CODING_FILE_CONFLICT,
    CODING_FILE_NOT_FOUND,
    CODING_INPUT_INVALID,
    CODING_PATH_CONFLICT,
    CODING_SCOPE_VIOLATION,
    CODING_VERIFICATION_INVALID,
    CodingError,
)
from forecontract_coding.models import (
    MAX_FILE_BYTES,
    MAX_READ_CHUNK_BYTES,
    MAX_TOTAL_FILE_BYTES,
    CodingConfig,
    CodingTask,
    CommandSpec,
    WriteTarget,
    canonicalize_relative_path,
    is_valid_sha256_hex,
)

__all__ = [
    "FileSnapshot",
    "ManagedWorkspace",
    "ReadChunk",
    "ScopeViolation",
    "SearchMatch",
    "SearchResponse",
    "WriteOutcome",
    "collect_and_verify_acceptance_sources",
    "commit_file_atomic",
    "compute_workspace_revision",
    "ensure_clean_subdir",
    "extract_acceptance_source_paths",
    "sha256_bytes",
    "validate_task_and_config_paths",
]

_FILE_ATTRIBUTE_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Point-in-time state of one managed workspace file."""

    path: str
    exists: bool
    sha256: str | None
    byte_length: int
    readable: bool
    writable: bool
    operation: str | None


@dataclass(frozen=True, slots=True)
class ReadChunk:
    """Bounded UTF-8 file read response."""

    path: str
    sha256: str
    start_line: int
    end_line: int | None
    total_lines: int
    content: str
    truncated: bool
    next_start_line: int | None
    next_line_byte_offset: int | None = None


@dataclass(frozen=True, slots=True)
class ScopeViolation:
    """One unauthorized workspace delta or failed scan relative to baseline."""

    path: str
    reason: str
    before_sha256: str | None
    after_sha256: str | None


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """Single deterministic literal text match."""

    path: str
    line_number: int
    column: int
    line_text: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Bounded literal search result across authorized paths."""

    query: str
    matches: tuple[SearchMatch, ...]
    truncated: bool
    searched_paths: tuple[str, ...]
    next_cursor: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    """Verified outcome of an atomic file write with mandatory read-back check."""

    path: str
    operation: str
    before_sha256: str | None
    written_sha256: str
    read_back_sha256: str
    read_back_match: bool
    before_revision: str
    after_revision: str


def sha256_bytes(data: bytes) -> str:
    """Return lowercase SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    """Return True if ``path`` exists as a symlink, NTFS junction, or other reparse point."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    if attrs & _FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            if is_junction():
                return True
        except OSError:
            return True
    return False


def _ensure_clean_dir_chain(root: Path, rel_parts: tuple[str, ...]) -> None:
    """Verify ``root`` and every existing component below it are non-reparse directories/file."""
    if _is_reparse_or_symlink(root):
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "workspace_root must not be a symlink, junction, or reparse point",
            location=("workspace_root",),
        )
    if not root.exists() or not root.is_dir():
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "workspace_root must be an existing real directory",
            location=("workspace_root",),
        )
    cur = root
    resolved_root = os.path.realpath(root)
    for index, part in enumerate(rel_parts):
        cur = cur / part
        if not os.path.lexists(cur):
            # Missing tail components cannot traverse outside yet; stop here.
            break
        if _is_reparse_or_symlink(cur):
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "managed path contains a symlink, junction, or reparse point",
                location=("path", "/".join(rel_parts[: index + 1])),
            )
        is_last = index == len(rel_parts) - 1
        if not is_last and not cur.is_dir():
            raise CodingError(
                CODING_PATH_CONFLICT,
                "parent path component is not a directory",
                location=("path", "/".join(rel_parts[: index + 1])),
            )
        real_cur = os.path.realpath(cur)
        try:
            Path(real_cur).relative_to(Path(resolved_root))
        except ValueError:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "resolved path escapes workspace_root",
                location=("path", "/".join(rel_parts)),
            ) from None


def _ensure_no_ancestor_reparse(path: Path) -> None:
    """Check ``path`` and all existing ancestors do not cross a reparse point."""
    cur = path
    checked: set[Path] = set()
    while cur not in checked:
        checked.add(cur)
        if os.path.lexists(cur) and _is_reparse_or_symlink(cur):
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "directory chain contains a symlink, junction, or reparse point",
                location=("path", str(cur.name)),
            )
        parent = cur.parent
        if parent == cur:
            break
        cur = parent


def _paths_overlap(a: Path, b: Path) -> bool:
    """Return True if resolved ``a`` and ``b`` are equal or one contains the other."""
    ra = Path(os.path.realpath(a))
    rb = Path(os.path.realpath(b))
    return ra == rb or ra in rb.parents or rb in ra.parents


def ensure_clean_subdir(root: Path, rel_parts: tuple[str, ...]) -> Path:
    """Safely verify or create ``rel_parts`` below non-reparse ``root`` without ``rmtree``."""
    _ensure_no_ancestor_reparse(root)
    if not root.exists() or not root.is_dir() or _is_reparse_or_symlink(root):
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "root directory must be an existing non-reparse directory",
            location=("output_root",),
        )
    real_root = Path(os.path.realpath(root))
    cur = root
    for part in rel_parts:
        cur = cur / part
        if os.path.lexists(cur):
            if _is_reparse_or_symlink(cur) or not cur.is_dir():
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "directory chain contains a symlink, junction, or non-directory",
                    location=("output_root", part),
                )
        else:
            cur.mkdir(exist_ok=False)
        real_cur = Path(os.path.realpath(cur))
        if real_cur != real_root and real_root not in real_cur.parents:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "directory chain escapes root",
                location=("output_root", part),
            )
    return cur


_PYTEST_VALUE_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "-m",
        "-p",
        "-k",
        "-o",
        "-c",
        "-W",
        "--rootdir",
        "--confcutdir",
        "--basetemp",
        "--junitxml",
        "--import-mode",
    }
)


def extract_acceptance_source_paths(spec: CommandSpec) -> tuple[Path, ...]:
    """Deterministically extract fixed ``.py`` test source paths from an acceptance command."""
    sources: list[Path] = []
    args = list(spec.args)
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--":
            idx += 1
            while idx < len(args):
                tail = args[idx]
                if not tail.lower().endswith(".py") or "{result_dir}" in tail:
                    raise CodingError(
                        CODING_VERIFICATION_INVALID,
                        "acceptance command contains non-fixed or non-.py test target",
                        location=("task", "commands", spec.command_id, "args"),
                    )
                p = Path(tail)
                if not p.is_absolute():
                    p = Path(spec.cwd) / p
                sources.append(p)
                idx += 1
            break
        if arg.startswith("-"):
            if "=" in arg:
                idx += 1
                continue
            if arg in _PYTEST_VALUE_FLAGS:
                if idx + 1 >= len(args):
                    raise CodingError(
                        CODING_VERIFICATION_INVALID,
                        "acceptance command option is missing value",
                        location=("task", "commands", spec.command_id, "args"),
                    )
                idx += 2
                continue
            idx += 1
            continue
        if (
            "{result_dir}" in arg
            or any(ch in arg for ch in "*?[")
            or "::" in arg
            or not arg.lower().endswith(".py")
        ):
            raise CodingError(
                CODING_VERIFICATION_INVALID,
                "acceptance command must explicitly reference fixed .py source files",
                location=("task", "commands", spec.command_id, "args"),
            )
        p = Path(arg)
        if not p.is_absolute():
            p = Path(spec.cwd) / p
        sources.append(p)
        idx += 1

    if not sources:
        raise CodingError(
            CODING_VERIFICATION_INVALID,
            "acceptance command must declare at least one fixed external .py test file",
            location=("task", "commands", spec.command_id, "args"),
        )
    return tuple(sources)


def collect_and_verify_acceptance_sources(
    task: CodingTask, config: CodingConfig
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Validate and hash all fixed external acceptance test files at startup."""
    ws = Path(task.workspace_root)
    out = Path(config.output_root)
    real_ws = Path(os.path.realpath(ws))
    real_out = Path(os.path.realpath(out))
    write_candidates: list[Path] = [
        ws.joinpath(*wt.path.split("/")) for wt in task.write_paths
    ]

    flat_hashes: dict[str, str] = {}
    by_cmd: dict[str, tuple[str, ...]] = {}
    for cid in task.verification_command_ids:
        spec = task.commands[cid]
        raw_paths = extract_acceptance_source_paths(spec)
        cmd_keys: list[str] = []
        for src in raw_paths:
            _ensure_no_ancestor_reparse(src)
            if not os.path.lexists(src):
                raise CodingError(
                    CODING_VERIFICATION_INVALID,
                    "acceptance test source file does not exist",
                    location=("task", "commands", cid, "args"),
                )
            if _is_reparse_or_symlink(src):
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "acceptance test source must not be a symlink, junction, or reparse point",
                    location=("task", "commands", cid, "args"),
                )
            st = os.lstat(src)
            if not stat.S_ISREG(st.st_mode):
                raise CodingError(
                    CODING_VERIFICATION_INVALID,
                    "acceptance test source must be a regular file",
                    location=("task", "commands", cid, "args"),
                )
            if st.st_nlink > 1:
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "acceptance test source must not have multiple hardlinks",
                    location=("task", "commands", cid, "args"),
                )
            if st.st_size > MAX_TOTAL_FILE_BYTES:
                raise CodingError(
                    CODING_VERIFICATION_INVALID,
                    "acceptance test source exceeds bounded size limit",
                    location=("task", "commands", cid, "args"),
                )
            real_src = Path(os.path.realpath(src))
            if real_src == real_ws or real_ws in real_src.parents:
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "acceptance test source must remain outside workspace_root",
                    location=("task", "commands", cid, "args"),
                )
            if real_src == real_out or real_out in real_src.parents:
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "acceptance test source must remain outside output_root",
                    location=("task", "commands", cid, "args"),
                )
            for wt_path in write_candidates:
                if os.path.lexists(wt_path):
                    try:
                        if os.path.samefile(real_src, wt_path):
                            raise CodingError(
                                CODING_SCOPE_VIOLATION,
                                "acceptance test source must not overlap write_paths",
                                location=("task", "commands", cid, "args"),
                            )
                    except OSError as exc:
                        raise CodingError(
                            CODING_SCOPE_VIOLATION,
                            "unable to verify physical identity of acceptance test source",
                            location=("task", "commands", cid, "args"),
                        ) from exc

            key = str(real_src)
            if key not in flat_hashes:
                flat_hashes[key] = sha256_bytes(real_src.read_bytes())
            cmd_keys.append(key)
        by_cmd[cid] = tuple(cmd_keys)
    return dict(sorted(flat_hashes.items())), by_cmd


def validate_task_and_config_paths(task: CodingTask, config: CodingConfig) -> None:
    """Verify host workspace/output/command directories and external test sources before work."""
    ws = Path(task.workspace_root)
    out = Path(config.output_root)
    _ensure_no_ancestor_reparse(ws)
    if not ws.exists() or not ws.is_dir():
        raise CodingError(
            CODING_INPUT_INVALID,
            "workspace_root must exist and be a directory",
            location=("task", "workspace_root"),
        )
    if os.path.lexists(out):
        _ensure_no_ancestor_reparse(out)
        if not out.is_dir():
            raise CodingError(
                CODING_INPUT_INVALID,
                "output_root must be a directory",
                location=("config", "output_root"),
            )
    else:
        _ensure_no_ancestor_reparse(out.parent)
    if _paths_overlap(ws, out):
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "output_root must remain outside workspace_root",
            location=("config", "output_root"),
        )

    managed = task.managed_paths()
    if len(managed) > config.max_files:
        raise CodingError(
            CODING_INPUT_INVALID,
            "managed file count exceeds config.max_files",
            location=("task", "read_paths"),
        )

    real_ws = Path(os.path.realpath(ws))
    real_out = Path(os.path.realpath(out))
    for cid, spec in task.commands.items():
        exe = Path(spec.executable)
        if not os.path.exists(exe) or not exe.is_file() or _is_reparse_or_symlink(exe):
            raise CodingError(
                CODING_INPUT_INVALID,
                "command executable must be an existing regular file",
                location=("task", "commands", cid, "executable"),
            )
        cwd_path = Path(spec.cwd)
        if not cwd_path.exists() or not cwd_path.is_dir():
            raise CodingError(
                CODING_INPUT_INVALID,
                "command cwd must exist and be a directory",
                location=("task", "commands", cid, "cwd"),
            )
        _ensure_no_ancestor_reparse(cwd_path)
        real_cwd = Path(os.path.realpath(cwd_path))
        in_ws = real_cwd == real_ws or real_ws in real_cwd.parents
        in_out = real_cwd == real_out or real_out in real_cwd.parents
        if not (in_ws or in_out):
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "command cwd must reside inside workspace_root or output_root",
                location=("task", "commands", cid, "cwd"),
            )

    collect_and_verify_acceptance_sources(task, config)


def compute_workspace_revision(snapshots: tuple[FileSnapshot, ...]) -> str:
    """Compute a deterministic SHA-256 digest of all managed files, existence, and SHAs."""
    ordered = sorted(snapshots, key=lambda item: item.path)
    payload = [
        {"path": item.path, "exists": item.exists, "sha256": item.sha256}
        for item in ordered
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _truncate_utf8_prefix(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    cut = encoded[:max_bytes]
    while cut:
        try:
            return cut.decode("utf-8"), True
        except UnicodeDecodeError:
            cut = cut[:-1]
    return "", True


def _slice_utf8_prefix_bytes(encoded: bytes, max_bytes: int) -> tuple[bytes, str]:
    """Return the longest valid UTF-8 byte prefix of ``encoded`` within ``max_bytes``."""
    if len(encoded) <= max_bytes:
        return encoded, encoded.decode("utf-8")
    cut = encoded[:max_bytes]
    while cut:
        try:
            return cut, cut.decode("utf-8")
        except UnicodeDecodeError:
            cut = cut[:-1]
    # Guard for absurdly small synthetic max_bytes (< UTF-8 byte width of first char):
    # contract guarantees >= 1 character progress without splitting UTF-8 codepoints.
    text = encoded.decode("utf-8")
    first_char_bytes = text[0].encode("utf-8")
    return first_char_bytes, text[0]


if os.name == "nt":
    from ctypes import wintypes

    _WIN32_INVALID_HANDLE = wintypes.HANDLE(-1).value
    _WIN32_GENERIC_READ = 0x80000000
    _WIN32_GENERIC_WRITE = 0x40000000
    _WIN32_FILE_SHARE_READ = 0x00000001
    _WIN32_CREATE_NEW = 1
    _WIN32_OPEN_EXISTING = 3
    _WIN32_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _WIN32_FILE_ATTRIBUTE_DIRECTORY = 0x0010
    _WIN32_FILE_BEGIN = 0
    _WIN32_ERR_FILE_NOT_FOUND = 2
    _WIN32_ERR_SHARING_VIOLATION = 32
    _WIN32_ERR_LOCK_VIOLATION = 33
    _WIN32_ERR_FILE_EXISTS = 80
    _WIN32_ERR_ALREADY_EXISTS = 183
    _WIN32_ERR_TRANSACTIONAL_CONFLICT = 6800

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    _kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    ]
    _kernel32.SetFilePointerEx.restype = wintypes.BOOL
    _kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    _kernel32.WriteFile.restype = wintypes.BOOL
    _kernel32.WriteFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    _kernel32.SetEndOfFile.restype = wintypes.BOOL
    _kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    _kernel32.FlushFileBuffers.restype = wintypes.BOOL
    _kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CreateFileTransactedW.restype = wintypes.HANDLE
    _kernel32.CreateFileTransactedW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]

    _ktmw32 = ctypes.WinDLL("ktmw32", use_last_error=True)
    _ktmw32.CreateTransaction.restype = wintypes.HANDLE
    _ktmw32.CreateTransaction.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPCWSTR,
    ]
    _ktmw32.CommitTransaction.restype = wintypes.BOOL
    _ktmw32.CommitTransaction.argtypes = [wintypes.HANDLE]
    _ktmw32.RollbackTransaction.restype = wintypes.BOOL
    _ktmw32.RollbackTransaction.argtypes = [wintypes.HANDLE]

    def _win32_create_transaction() -> int:
        txn = _ktmw32.CreateTransaction(None, None, 0, 0, 0, 0, None)
        if txn == _WIN32_INVALID_HANDLE:
            raise OSError(ctypes.get_last_error(), "CreateTransaction failed")
        return int(txn)

    def _win32_commit_transaction(txn: int) -> None:
        if not _ktmw32.CommitTransaction(txn):
            raise OSError(ctypes.get_last_error(), "CommitTransaction failed")

    def _win32_rollback_transaction(txn: int) -> None:
        _ktmw32.RollbackTransaction(txn)

    def _win32_write_all(handle: int, data: bytes) -> None:
        offset = 0
        total = len(data)
        while offset < total:
            chunk = data[offset:]
            wbuf = (ctypes.c_ubyte * len(chunk)).from_buffer_copy(chunk)
            written = wintypes.DWORD(0)
            if not _kernel32.WriteFile(handle, wbuf, len(chunk), ctypes.byref(written), None):
                err = ctypes.get_last_error()
                raise OSError(err, "WriteFile failed")
            if written.value == 0:
                raise OSError("WriteFile wrote zero bytes")
            offset += written.value

    def _win32_read_all(handle: int, expected_size: int) -> bytes:
        if expected_size == 0:
            return b""
        buf = (ctypes.c_ubyte * expected_size)()
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining > 0:
            got = wintypes.DWORD(0)
            if not _kernel32.ReadFile(handle, buf, remaining, ctypes.byref(got), None):
                err = ctypes.get_last_error()
                raise OSError(err, "ReadFile failed")
            if got.value == 0:
                break
            chunks.append(bytes(buf[: got.value]))
            remaining -= got.value
        return b"".join(chunks)


def _inspect_regular_file_bytes(
    norm: str, file_path: Path, *, writable: bool, max_file_bytes: int
) -> bytes | None:
    if not os.path.lexists(file_path):
        return None
    if _is_reparse_or_symlink(file_path):
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "managed file must not be a symlink, junction, or reparse point",
            location=("path", norm),
        )
    st = os.lstat(file_path)
    if not stat.S_ISREG(st.st_mode):
        raise CodingError(
            CODING_PATH_CONFLICT,
            "managed target is not a regular file",
            location=("path", norm),
        )
    if writable and st.st_nlink > 1:
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "writable target must not have multiple hardlinks",
            location=("path", norm),
        )
    if st.st_size > max_file_bytes:
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "managed file exceeds config.max_file_bytes",
            location=("path", norm),
        )
    data = file_path.read_bytes()
    if len(data) > max_file_bytes:
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "managed file bytes exceed config.max_file_bytes",
            location=("path", norm),
        )
    return data


def _commit_create_exclusive(
    *, file_path: Path, new_bytes: bytes, norm: str
) -> None:
    """Create a brand-new file atomically inside a Win32 TxF transaction (or O_EXCL on POSIX)."""
    if os.name == "nt":
        txn = _win32_create_transaction()
        handle: int | None = None
        committed = False
        try:
            opened = _kernel32.CreateFileTransactedW(
                str(file_path),
                _WIN32_GENERIC_READ | _WIN32_GENERIC_WRITE,
                0,
                None,
                _WIN32_CREATE_NEW,
                _WIN32_FILE_FLAG_OPEN_REPARSE_POINT,
                None,
                txn,
                None,
                None,
            )
            if opened == _WIN32_INVALID_HANDLE:
                err = ctypes.get_last_error()
                if err in (
                    _WIN32_ERR_FILE_EXISTS,
                    _WIN32_ERR_ALREADY_EXISTS,
                    _WIN32_ERR_SHARING_VIOLATION,
                    _WIN32_ERR_LOCK_VIOLATION,
                    _WIN32_ERR_TRANSACTIONAL_CONFLICT,
                ):
                    raise CodingError(
                        CODING_FILE_CONFLICT,
                        "concurrent file creation detected before create commit",
                        location=("path", norm),
                    )
                raise OSError(err, "CreateFileTransactedW create commit failed")
            handle = int(opened)
            _win32_write_all(handle, new_bytes)
            if not _kernel32.SetEndOfFile(handle):
                raise OSError(ctypes.get_last_error(), "SetEndOfFile failed")
            if not _kernel32.FlushFileBuffers(handle):
                raise OSError(ctypes.get_last_error(), "FlushFileBuffers failed")
            _kernel32.CloseHandle(handle)
            handle = None
            _win32_commit_transaction(txn)
            committed = True
        finally:
            if handle is not None:
                _kernel32.CloseHandle(handle)
            if not committed:
                _win32_rollback_transaction(txn)
            _kernel32.CloseHandle(txn)
        return

    try:
        fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        raise CodingError(
            CODING_FILE_CONFLICT,
            "concurrent file creation detected before create commit",
            location=("path", norm),
        ) from None
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(new_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _commit_modify_exclusive(
    *,
    root: Path,
    file_path: Path,
    parts: tuple[str, ...],
    norm: str,
    before_sha: str,
    new_bytes: bytes,
    max_file_bytes: int,
) -> None:
    """Hold TxF-transacted exclusive writer lock across final SHA check and atomic commit."""
    if os.name == "nt":
        txn = _win32_create_transaction()
        handle: int | None = None
        committed = False
        try:
            opened = _kernel32.CreateFileTransactedW(
                str(file_path),
                _WIN32_GENERIC_READ | _WIN32_GENERIC_WRITE,
                _WIN32_FILE_SHARE_READ,
                None,
                _WIN32_OPEN_EXISTING,
                _WIN32_FILE_FLAG_OPEN_REPARSE_POINT,
                None,
                txn,
                None,
                None,
            )
            if opened == _WIN32_INVALID_HANDLE:
                err = ctypes.get_last_error()
                if err == _WIN32_ERR_FILE_NOT_FOUND:
                    raise CodingError(
                        CODING_FILE_NOT_FOUND,
                        "modify target disappeared before exclusive commit",
                        location=("path", norm),
                    )
                if err in (
                    _WIN32_ERR_SHARING_VIOLATION,
                    _WIN32_ERR_LOCK_VIOLATION,
                    _WIN32_ERR_TRANSACTIONAL_CONFLICT,
                ):
                    raise CodingError(
                        CODING_FILE_CONFLICT,
                        "concurrent file access detected at commit boundary",
                        location=("path", norm),
                    )
                raise CodingError(
                    CODING_FILE_CONFLICT,
                    f"unable to acquire transacted commit handle (winerr={err})",
                    location=("path", norm),
                )
            handle = int(opened)
            _ensure_clean_dir_chain(root, parts)
            info = _BY_HANDLE_FILE_INFORMATION()
            if not _kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
                raise OSError(ctypes.get_last_error(), "GetFileInformationByHandle failed")
            if info.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "managed file must not be a symlink, junction, or reparse point",
                    location=("path", norm),
                )
            if info.dwFileAttributes & _WIN32_FILE_ATTRIBUTE_DIRECTORY:
                raise CodingError(
                    CODING_PATH_CONFLICT,
                    "managed target is not a regular file",
                    location=("path", norm),
                )
            if info.nNumberOfLinks > 1:
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "writable target must not have multiple hardlinks",
                    location=("path", norm),
                )
            current_size = (info.nFileSizeHigh << 32) | info.nFileSizeLow
            if current_size > max_file_bytes:
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "managed file exceeds config.max_file_bytes",
                    location=("path", norm),
                )
            if not _kernel32.SetFilePointerEx(handle, 0, None, _WIN32_FILE_BEGIN):
                raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
            current_bytes = _win32_read_all(handle, current_size)
            if sha256_bytes(current_bytes) != before_sha:
                raise CodingError(
                    CODING_FILE_CONFLICT,
                    "concurrent file modification detected inside commit boundary",
                    location=("path", norm),
                )
            if not _kernel32.SetFilePointerEx(handle, 0, None, _WIN32_FILE_BEGIN):
                raise OSError(ctypes.get_last_error(), "SetFilePointerEx rewind failed")
            _win32_write_all(handle, new_bytes)
            if not _kernel32.SetEndOfFile(handle):
                raise OSError(ctypes.get_last_error(), "SetEndOfFile failed")
            if not _kernel32.FlushFileBuffers(handle):
                raise OSError(ctypes.get_last_error(), "FlushFileBuffers failed")
            _kernel32.CloseHandle(handle)
            handle = None
            _win32_commit_transaction(txn)
            committed = True
        finally:
            if handle is not None:
                _kernel32.CloseHandle(handle)
            if not committed:
                _win32_rollback_transaction(txn)
            _kernel32.CloseHandle(txn)
        return

    _ensure_clean_dir_chain(root, parts)
    latest_data = _inspect_regular_file_bytes(
        norm, file_path, writable=True, max_file_bytes=max_file_bytes
    )
    if latest_data is None or sha256_bytes(latest_data) != before_sha:
        raise CodingError(
            CODING_FILE_CONFLICT,
            "concurrent file modification detected immediately before replace",
            location=("path", norm),
        )
    temp_name = f".forecontract_tmp_{uuid.uuid4().hex}.tmp"
    temp_path = file_path.parent / temp_name
    try:
        tfd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(tfd, "wb") as handle:
            handle.write(new_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, file_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _commit_file_atomic_core(
    *,
    root: Path,
    norm: str,
    file_path: Path,
    operation: Literal["create", "modify"],
    expected_sha256: str | None,
    new_bytes: bytes,
    max_file_bytes: int,
    inspect_hook: Any,
    create_hook: Any,
    modify_hook: Any,
) -> WriteOutcome:
    """Shared guarded atomic create/modify sequence with read-back check."""
    parts = tuple(norm.split("/"))
    _ensure_clean_dir_chain(root, parts)
    current_data = inspect_hook(norm, file_path, writable=True)

    if operation == "create":
        if expected_sha256 is not None:
            raise CodingError(
                CODING_FILE_CONFLICT,
                "create operation requires expected_sha256=null",
                location=("payload", "expected_sha256"),
            )
        if current_data is not None:
            raise CodingError(
                CODING_FILE_CONFLICT,
                "create target already exists on disk",
                location=("path", norm),
            )
    else:
        if expected_sha256 is None:
            raise CodingError(
                CODING_FILE_CONFLICT,
                "modify operation requires non-null expected_sha256",
                location=("payload", "expected_sha256"),
            )
        if current_data is None:
            raise CodingError(
                CODING_FILE_NOT_FOUND,
                "modify target does not exist on disk",
                location=("path", norm),
            )
        current_sha = sha256_bytes(current_data)
        if current_sha != expected_sha256:
            raise CodingError(
                CODING_FILE_CONFLICT,
                "pre-write SHA-256 mismatch; file changed concurrently or expectation stale",
                location=("payload", "expected_sha256"),
            )

    cur_dir = root
    for segment in parts[:-1]:
        cur_dir = cur_dir / segment
        if os.path.lexists(cur_dir):
            if _is_reparse_or_symlink(cur_dir) or not cur_dir.is_dir():
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "unsafe parent directory during write",
                    location=("path", norm),
                )
        else:
            cur_dir.mkdir(exist_ok=False)

    before_sha = sha256_bytes(current_data) if current_data is not None else None
    expected_new_sha = sha256_bytes(new_bytes)

    if operation == "create":
        create_hook(file_path=file_path, new_bytes=new_bytes, norm=norm)
    else:
        assert before_sha is not None
        modify_hook(
            file_path=file_path,
            parts=parts,
            norm=norm,
            before_sha=before_sha,
            new_bytes=new_bytes,
        )

    _ensure_clean_dir_chain(root, parts)
    read_back = inspect_hook(norm, file_path, writable=True)
    if read_back is None:
        raise CodingError(
            CODING_FILE_CONFLICT,
            "written file disappeared before read-back check",
            location=("path", norm),
        )
    read_back_sha = sha256_bytes(read_back)
    return WriteOutcome(
        path=norm,
        operation=operation,
        before_sha256=before_sha,
        written_sha256=expected_new_sha,
        read_back_sha256=read_back_sha,
        read_back_match=(read_back_sha == expected_new_sha),
        before_revision=before_sha if before_sha is not None else ("0" * 64),
        after_revision=read_back_sha,
    )


def commit_file_atomic(
    *,
    workspace_root: Path,
    path: str,
    operation: Literal["create", "modify"],
    expected_sha256: str | None,
    content: bytes,
    max_file_bytes: int = 65536,
) -> WriteOutcome:
    """Safely create or atomically replace one workspace file with Win32 TxF/read-back checks."""
    if not isinstance(workspace_root, Path):
        raise CodingError(
            CODING_INPUT_INVALID,
            "workspace_root must be a pathlib.Path",
            location=("workspace_root",),
        )
    if operation not in ("create", "modify"):
        raise CodingError(
            CODING_INPUT_INVALID,
            "operation must be 'create' or 'modify'",
            location=("operation",),
        )
    if not isinstance(content, bytes):
        raise CodingError(
            CODING_INPUT_INVALID,
            "content must be raw bytes",
            location=("content",),
        )
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes < 1
        or max_file_bytes > MAX_FILE_BYTES
    ):
        raise CodingError(
            CODING_INPUT_INVALID,
            f"max_file_bytes must be an integer in 1..{MAX_FILE_BYTES}",
            location=("max_file_bytes",),
        )
    if len(content) > max_file_bytes:
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            "new file content exceeds max_file_bytes",
            location=("payload", "content"),
        )
    try:
        norm = canonicalize_relative_path(path)
    except (TypeError, ValueError) as exc:
        raise CodingError(
            CODING_SCOPE_VIOLATION,
            str(exc),
            location=("path",),
        ) from None

    _ensure_no_ancestor_reparse(workspace_root)
    parts = tuple(norm.split("/"))
    file_path = workspace_root.joinpath(*parts)

    def _inspect(item_norm: str, item_path: Path, *, writable: bool) -> bytes | None:
        return _inspect_regular_file_bytes(
            item_norm, item_path, writable=writable, max_file_bytes=max_file_bytes
        )

    def _modify(
        *,
        file_path: Path,
        parts: tuple[str, ...],
        norm: str,
        before_sha: str,
        new_bytes: bytes,
    ) -> None:
        _commit_modify_exclusive(
            root=workspace_root,
            file_path=file_path,
            parts=parts,
            norm=norm,
            before_sha=before_sha,
            new_bytes=new_bytes,
            max_file_bytes=max_file_bytes,
        )

    return _commit_file_atomic_core(
        root=workspace_root,
        norm=norm,
        file_path=file_path,
        operation=operation,
        expected_sha256=expected_sha256,
        new_bytes=content,
        max_file_bytes=max_file_bytes,
        inspect_hook=_inspect,
        create_hook=_commit_create_exclusive,
        modify_hook=_modify,
    )


class ManagedWorkspace:
    """Encapsulates strict path-scoped read, search, write, and revision computation."""

    def __init__(self, *, task: CodingTask, config: CodingConfig) -> None:
        self._task = task
        self._config = config
        validate_task_and_config_paths(task, config)
        self._root = Path(task.workspace_root)
        self._read_set: frozenset[str] = frozenset(task.read_paths)
        self._write_map: dict[str, WriteTarget] = {
            target.path: target for target in task.write_paths
        }
        self._managed_paths: tuple[str, ...] = task.managed_paths()

    @property
    def root(self) -> Path:
        return self._root

    def _resolve_managed_path(
        self, raw_path: str, *, require_read: bool = False, require_write: bool = False
    ) -> tuple[str, Path, WriteTarget | None]:
        try:
            norm = canonicalize_relative_path(raw_path)
        except ValueError as exc:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                str(exc),
                location=("path",),
            ) from None
        if require_read and norm not in self._read_set:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "path is outside task.read_paths",
                location=("path", norm),
            )
        target_spec = self._write_map.get(norm)
        if require_write and target_spec is None:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "path is outside task.write_paths",
                location=("path", norm),
            )
        if not require_read and not require_write and norm not in self._managed_paths:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "path is outside managed workspace scope",
                location=("path", norm),
            )
        parts = tuple(norm.split("/"))
        _ensure_clean_dir_chain(self._root, parts)
        return norm, self._root.joinpath(*parts), target_spec

    def _inspect_regular_file(self, norm: str, file_path: Path, *, writable: bool) -> bytes | None:
        return _inspect_regular_file_bytes(
            norm, file_path, writable=writable, max_file_bytes=self._config.max_file_bytes
        )

    def snapshot(self) -> tuple[str, tuple[FileSnapshot, ...], bool]:
        """Inspect all managed paths cleanly and return ``(revision, files, scope_ok)``."""
        snapshots: list[FileSnapshot] = []
        total_bytes = 0
        for norm in self._managed_paths:
            writable_spec = self._write_map.get(norm)
            writable = writable_spec is not None
            readable = norm in self._read_set
            operation = writable_spec.operation if writable_spec is not None else None
            parts = tuple(norm.split("/"))
            _ensure_clean_dir_chain(self._root, parts)
            file_path = self._root.joinpath(*parts)
            data = self._inspect_regular_file(norm, file_path, writable=writable)
            if data is None:
                snapshots.append(
                    FileSnapshot(
                        path=norm,
                        exists=False,
                        sha256=None,
                        byte_length=0,
                        readable=readable,
                        writable=writable,
                        operation=operation,
                    )
                )
            else:
                total_bytes += len(data)
                snapshots.append(
                    FileSnapshot(
                        path=norm,
                        exists=True,
                        sha256=sha256_bytes(data),
                        byte_length=len(data),
                        readable=readable,
                        writable=writable,
                        operation=operation,
                    )
                )
        if total_bytes > self._config.max_total_file_bytes:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "total managed file bytes exceed config.max_total_file_bytes",
                location=("workspace",),
            )
        ordered = tuple(snapshots)
        return compute_workspace_revision(ordered), ordered, True

    def _scan_workspace_files(
        self, *, fail_on_error: bool
    ) -> tuple[bool, dict[str, str], list[ScopeViolation]]:
        """Walk ``workspace_root`` deterministically up to config file/byte ceilings."""
        if _is_reparse_or_symlink(self._root) or not self._root.is_dir():
            if fail_on_error:
                raise CodingError(
                    CODING_SCOPE_VIOLATION,
                    "workspace_root must be a real non-reparse directory",
                    location=("workspace_root",),
                )
            return False, {}, [ScopeViolation(".", "scope_scan_failed", None, None)]

        inventory: dict[str, str] = {}
        scan_violations: list[ScopeViolation] = []
        scan_ok = True
        file_count = 0
        total_bytes = 0
        stack: list[tuple[Path, tuple[str, ...]]] = [(self._root, ())]

        while stack:
            cur_dir, rel_parts = stack.pop()
            try:
                with os.scandir(cur_dir) as it:
                    entries = sorted(it, key=lambda item: item.name)
            except OSError as exc:
                rel = "/".join(rel_parts) if rel_parts else "."
                if fail_on_error:
                    raise CodingError(
                        CODING_SCOPE_VIOLATION,
                        "unable to scan workspace directory cleanly",
                        location=("path", rel),
                    ) from exc
                scan_ok = False
                scan_violations.append(ScopeViolation(rel, "scope_scan_failed", None, None))
                continue

            subdirs: list[tuple[Path, tuple[str, ...]]] = []
            for entry in entries:
                child_parts = (*rel_parts, entry.name)
                child_rel = "/".join(child_parts)
                child_path = Path(entry.path)
                if _is_reparse_or_symlink(child_path):
                    if fail_on_error:
                        raise CodingError(
                            CODING_SCOPE_VIOLATION,
                            "workspace contains symlink, junction, or reparse point",
                            location=("path", child_rel),
                        )
                    scan_ok = False
                    scan_violations.append(
                        ScopeViolation(child_rel, "scope_scan_failed", None, None)
                    )
                    continue
                try:
                    st = os.lstat(child_path)
                except OSError as exc:
                    if fail_on_error:
                        raise CodingError(
                            CODING_SCOPE_VIOLATION,
                            "unable to stat workspace entry",
                            location=("path", child_rel),
                        ) from exc
                    scan_ok = False
                    scan_violations.append(
                        ScopeViolation(child_rel, "scope_scan_failed", None, None)
                    )
                    continue

                if stat.S_ISDIR(st.st_mode):
                    subdirs.append((child_path, child_parts))
                elif stat.S_ISREG(st.st_mode):
                    file_count += 1
                    if file_count > self._config.max_files:
                        if fail_on_error:
                            raise CodingError(
                                CODING_SCOPE_VIOLATION,
                                "workspace file count exceeds config.max_files",
                                location=("path", child_rel),
                            )
                        scan_ok = False
                        scan_violations.append(
                            ScopeViolation(child_rel, "scope_scan_failed", None, None)
                        )
                        return False, inventory, scan_violations
                    if child_rel in self._write_map and st.st_nlink > 1:
                        if fail_on_error:
                            raise CodingError(
                                CODING_SCOPE_VIOLATION,
                                "writable target must not have multiple hardlinks",
                                location=("path", child_rel),
                            )
                        scan_ok = False
                        scan_violations.append(
                            ScopeViolation(child_rel, "scope_scan_failed", None, None)
                        )
                        continue
                    if st.st_size > self._config.max_file_bytes:
                        if fail_on_error:
                            raise CodingError(
                                CODING_SCOPE_VIOLATION,
                                "workspace file exceeds config.max_file_bytes",
                                location=("path", child_rel),
                            )
                        scan_ok = False
                        scan_violations.append(
                            ScopeViolation(child_rel, "scope_scan_failed", None, None)
                        )
                        continue
                    try:
                        data = child_path.read_bytes()
                    except OSError as exc:
                        if fail_on_error:
                            raise CodingError(
                                CODING_SCOPE_VIOLATION,
                                "unable to read workspace file cleanly",
                                location=("path", child_rel),
                            ) from exc
                        scan_ok = False
                        scan_violations.append(
                            ScopeViolation(child_rel, "scope_scan_failed", None, None)
                        )
                        continue
                    if len(data) > self._config.max_file_bytes:
                        if fail_on_error:
                            raise CodingError(
                                CODING_SCOPE_VIOLATION,
                                "workspace file bytes exceed config.max_file_bytes",
                                location=("path", child_rel),
                            )
                        scan_ok = False
                        scan_violations.append(
                            ScopeViolation(child_rel, "scope_scan_failed", None, None)
                        )
                        continue
                    total_bytes += len(data)
                    if total_bytes > self._config.max_total_file_bytes:
                        if fail_on_error:
                            raise CodingError(
                                CODING_SCOPE_VIOLATION,
                                "total workspace bytes exceed config.max_total_file_bytes",
                                location=("workspace",),
                            )
                        scan_ok = False
                        scan_violations.append(
                            ScopeViolation(child_rel, "scope_scan_failed", None, None)
                        )
                        return False, inventory, scan_violations
                    inventory[child_rel] = sha256_bytes(data)
                else:
                    if fail_on_error:
                        raise CodingError(
                            CODING_PATH_CONFLICT,
                            "workspace contains unsupported non-regular entry",
                            location=("path", child_rel),
                        )
                    scan_ok = False
                    scan_violations.append(
                        ScopeViolation(child_rel, "scope_scan_failed", None, None)
                    )
            stack.extend(reversed(subdirs))

        return scan_ok, dict(sorted(inventory.items())), scan_violations

    def capture_scope_baseline(self) -> dict[str, str]:
        """Capture bounded full-workspace inventory at session start (fail closed)."""
        _, inventory, _ = self._scan_workspace_files(fail_on_error=True)
        return inventory

    def compute_full_revision(
        self,
        *,
        snapshots: tuple[FileSnapshot, ...],
        inventory: dict[str, str],
        scope_ok: bool,
    ) -> str:
        """Compute deterministic digest covering managed slots plus full workspace inventory."""
        managed_payload = [
            {"path": item.path, "exists": item.exists, "sha256": item.sha256}
            for item in sorted(snapshots, key=lambda entry: entry.path)
        ]
        inventory_payload = [
            {"path": p, "sha256": inventory[p]} for p in sorted(inventory)
        ]
        encoded = json.dumps(
            {
                "managed": managed_payload,
                "inventory": inventory_payload,
                "scope_ok": scope_ok,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def inspect_scope(
        self, baseline_inventory: dict[str, str]
    ) -> tuple[bool, tuple[ScopeViolation, ...], dict[str, str]]:
        """Compare live workspace inventory against baseline without auto-restoring."""
        scan_ok, live_inventory, scan_violations = self._scan_workspace_files(
            fail_on_error=False
        )
        violations: list[ScopeViolation] = list(scan_violations)
        scan_failed_paths = {item.path for item in scan_violations}

        for path in sorted(set(baseline_inventory) | set(live_inventory)):
            if path in scan_failed_paths:
                continue
            in_base = path in baseline_inventory
            in_live = path in live_inventory
            writable = path in self._write_map
            if not in_base and in_live:
                if not writable:
                    violations.append(
                        ScopeViolation(
                            path=path,
                            reason="unauthorized_create",
                            before_sha256=None,
                            after_sha256=live_inventory[path],
                        )
                    )
            elif in_base and not in_live:
                # Deletion is never authorized even for writable paths (create|modify only).
                violations.append(
                    ScopeViolation(
                        path=path,
                        reason="unauthorized_delete",
                        before_sha256=baseline_inventory[path],
                        after_sha256=None,
                    )
                )
            elif (
                in_base
                and in_live
                and live_inventory[path] != baseline_inventory[path]
                and not writable
            ):
                violations.append(
                    ScopeViolation(
                        path=path,
                        reason="unauthorized_modify",
                        before_sha256=baseline_inventory[path],
                        after_sha256=live_inventory[path],
                    )
                )

        violations.sort(key=lambda item: (item.path, item.reason))
        scope_ok = bool(scan_ok and not violations)
        return scope_ok, tuple(violations), live_inventory

    def read_file(
        self,
        *,
        path: str,
        start_line: int,
        max_lines: int,
        line_byte_offset: int = 0,
        expected_sha256: str | None = None,
    ) -> ReadChunk:
        """Read UTF-8 text bounded by ``max_lines``, 16 KiB, and optional line byte offset."""
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
            raise CodingError(
                CODING_INPUT_INVALID,
                "start_line must be a positive integer",
                location=("payload", "start_line"),
            )
        if isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1:
            raise CodingError(
                CODING_INPUT_INVALID,
                "max_lines must be a positive integer",
                location=("payload", "max_lines"),
            )
        if (
            isinstance(line_byte_offset, bool)
            or not isinstance(line_byte_offset, int)
            or line_byte_offset < 0
        ):
            raise CodingError(
                CODING_INPUT_INVALID,
                "line_byte_offset must be a non-negative integer",
                location=("payload", "line_byte_offset"),
            )
        if expected_sha256 is not None and not is_valid_sha256_hex(expected_sha256):
            raise CodingError(
                CODING_INPUT_INVALID,
                "expected_sha256 must be 64 lowercase hex characters",
                location=("payload", "expected_sha256"),
            )
        if line_byte_offset > 0 and expected_sha256 is None:
            raise CodingError(
                CODING_INPUT_INVALID,
                "expected_sha256 is required when line_byte_offset > 0",
                location=("payload", "expected_sha256"),
            )

        norm, file_path, _ = self._resolve_managed_path(path, require_read=True)
        data = self._inspect_regular_file(norm, file_path, writable=(norm in self._write_map))
        if data is None:
            raise CodingError(
                CODING_FILE_NOT_FOUND,
                "target file does not exist",
                location=("path", norm),
            )
        file_sha = sha256_bytes(data)
        if expected_sha256 is not None and file_sha != expected_sha256:
            raise CodingError(
                CODING_FILE_CONFLICT,
                "file SHA-256 changed before read_file continuation",
                location=("payload", "expected_sha256"),
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CodingError(
                CODING_INPUT_INVALID,
                "file content is not valid UTF-8",
                location=("path", norm),
            ) from exc

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        if start_line > total_lines:
            if line_byte_offset > 0:
                raise CodingError(
                    CODING_INPUT_INVALID,
                    "line_byte_offset specified past end of file",
                    location=("payload", "line_byte_offset"),
                )
            return ReadChunk(
                path=norm,
                sha256=file_sha,
                start_line=start_line,
                end_line=None,
                total_lines=total_lines,
                content="",
                truncated=False,
                next_start_line=None,
                next_line_byte_offset=None,
            )

        first_line_bytes = lines[start_line - 1].encode("utf-8")
        if line_byte_offset >= len(first_line_bytes):
            raise CodingError(
                CODING_INPUT_INVALID,
                "line_byte_offset out of range for target line",
                location=("payload", "line_byte_offset"),
            )
        if (first_line_bytes[line_byte_offset] & 0xC0) == 0x80:
            raise CodingError(
                CODING_INPUT_INVALID,
                "line_byte_offset must align on a UTF-8 character boundary",
                location=("payload", "line_byte_offset"),
            )

        max_chunk = min(self._config.max_file_bytes, MAX_READ_CHUNK_BYTES)
        selected: list[str] = []
        used_bytes = 0
        truncated = False
        next_start: int | None = None
        next_offset: int | None = None

        rem_first = first_line_bytes[line_byte_offset:]
        if len(rem_first) > max_chunk:
            consumed_bytes, prefix = _slice_utf8_prefix_bytes(rem_first, max_chunk)
            selected.append(prefix)
            new_offset = line_byte_offset + len(consumed_bytes)
            if new_offset < len(first_line_bytes):
                truncated = True
                next_start = start_line
                next_offset = new_offset
                line_idx = total_lines  # skip loop; mid-line continuation
            else:
                line_idx = start_line
        else:
            selected.append(rem_first.decode("utf-8"))
            used_bytes = len(rem_first)
            line_idx = start_line
            limit = min(total_lines, (start_line - 1) + max_lines)
            while line_idx < limit:
                line = lines[line_idx]
                line_len = len(line.encode("utf-8"))
                if used_bytes + line_len <= max_chunk:
                    selected.append(line)
                    used_bytes += line_len
                    line_idx += 1
                else:
                    truncated = True
                    next_start = line_idx + 1
                    next_offset = 0
                    break

        if next_start is None:
            if line_idx < total_lines:
                truncated = True
                next_start = line_idx + 1
                next_offset = 0
            else:
                truncated = False
                next_start = None
                next_offset = None

        end_line = start_line + len(selected) - 1
        return ReadChunk(
            path=norm,
            sha256=file_sha,
            start_line=start_line,
            end_line=end_line,
            total_lines=total_lines,
            content="".join(selected),
            truncated=truncated,
            next_start_line=next_start,
            next_line_byte_offset=next_offset,
        )

    def search_text(
        self, *, query: str, paths: list[str], max_matches: int
    ) -> SearchResponse:
        """Perform deterministic non-regex literal search over authorized readable paths."""
        resolved: list[tuple[str, Path]] = []
        for item in paths:
            norm, file_path, _ = self._resolve_managed_path(item, require_read=True)
            resolved.append((norm, file_path))
        resolved.sort(key=lambda pair: pair[0])

        matches: list[SearchMatch] = []
        truncated = False
        next_cursor: dict[str, Any] | None = None

        for norm, file_path in resolved:
            data = self._inspect_regular_file(norm, file_path, writable=(norm in self._write_map))
            if data is None:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CodingError(
                    CODING_INPUT_INVALID,
                    "search target file is not valid UTF-8",
                    location=("path", norm),
                ) from exc
            file_sha = sha256_bytes(data)
            for idx, line in enumerate(text.splitlines(), start=1):
                col = line.find(query)
                if col == -1:
                    continue
                if len(matches) >= max_matches:
                    truncated = True
                    next_cursor = {"path": norm, "line_number": idx}
                    break
                excerpt, _ = _truncate_utf8_prefix(line, 512)
                matches.append(
                    SearchMatch(
                        path=norm,
                        line_number=idx,
                        column=col + 1,
                        line_text=excerpt,
                        file_sha256=file_sha,
                    )
                )
            if truncated:
                break

        return SearchResponse(
            query=query,
            matches=tuple(matches),
            truncated=truncated,
            searched_paths=tuple(pair[0] for pair in resolved),
            next_cursor=next_cursor,
        )

    def _commit_create_exclusive(
        self, *, file_path: Path, new_bytes: bytes, norm: str
    ) -> None:
        """Create a brand-new file atomically inside a Win32 TxF transaction (or O_EXCL on POSIX)."""
        _commit_create_exclusive(file_path=file_path, new_bytes=new_bytes, norm=norm)

    def _commit_modify_exclusive(
        self,
        *,
        file_path: Path,
        parts: tuple[str, ...],
        norm: str,
        before_sha: str,
        new_bytes: bytes,
    ) -> None:
        """Hold TxF-transacted exclusive writer lock across final SHA check and atomic commit."""
        _commit_modify_exclusive(
            root=self._root,
            file_path=file_path,
            parts=parts,
            norm=norm,
            before_sha=before_sha,
            new_bytes=new_bytes,
            max_file_bytes=self._config.max_file_bytes,
        )

    def write_file(
        self, *, path: str, expected_sha256: str | None, content: str
    ) -> WriteOutcome:
        """Safely create or atomically replace one authorized file with mandatory read-back."""
        norm, file_path, target_spec = self._resolve_managed_path(path, require_write=True)
        assert target_spec is not None

        new_bytes = content.encode("utf-8")
        if len(new_bytes) > self._config.max_file_bytes:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "new file content exceeds config.max_file_bytes",
                location=("payload", "content"),
            )

        before_rev, snapshots, _ = self.snapshot()
        current_total = sum(s.byte_length for s in snapshots if s.path != norm)
        if current_total + len(new_bytes) > self._config.max_total_file_bytes:
            raise CodingError(
                CODING_SCOPE_VIOLATION,
                "resulting total workspace bytes would exceed config.max_total_file_bytes",
                location=("payload", "content"),
            )

        outcome = _commit_file_atomic_core(
            root=self._root,
            norm=norm,
            file_path=file_path,
            operation=target_spec.operation,
            expected_sha256=expected_sha256,
            new_bytes=new_bytes,
            max_file_bytes=self._config.max_file_bytes,
            inspect_hook=self._inspect_regular_file,
            create_hook=self._commit_create_exclusive,
            modify_hook=self._commit_modify_exclusive,
        )
        after_rev, _, _ = self.snapshot()
        return WriteOutcome(
            path=outcome.path,
            operation=outcome.operation,
            before_sha256=outcome.before_sha256,
            written_sha256=outcome.written_sha256,
            read_back_sha256=outcome.read_back_sha256,
            read_back_match=outcome.read_back_match,
            before_revision=before_rev,
            after_revision=after_rev,
        )
