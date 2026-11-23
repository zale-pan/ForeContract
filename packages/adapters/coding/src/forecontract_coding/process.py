"""Windows Job-Object-backed subprocess runner using Python stdlib only.

Every child is created suspended with redirected pipes, assigned to a fresh Windows Job
Object before its main thread is resumed, drained incrementally with a strict per-stream
UTF-8 byte limit, and forcibly terminated (entire process tree) within ``process_cleanup_ms``
on timeout or cancellation.
"""

from __future__ import annotations

import asyncio
import ctypes
import msvcrt
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from forecontract_coding.errors import (
    CODING_PROCESS_CLEANUP_FAILED,
    CODING_PROCESS_START_FAILED,
    CodingError,
)
from forecontract_coding.models import CommandSpec

__all__ = [
    "ProcessExecutionResult",
    "build_minimal_windows_env",
    "run_command_in_job",
]

_CREATE_SUSPENDED: Final[int] = 0x00000004
_CREATE_UNICODE_ENVIRONMENT: Final[int] = 0x00000400
_CREATE_NEW_PROCESS_GROUP: Final[int] = 0x00000200
_CREATE_NO_WINDOW: Final[int] = 0x08000000

_HANDLE_FLAG_INHERIT: Final[int] = 0x00000001
_STARTF_USESTDHANDLES: Final[int] = 0x00000100
_GENERIC_READ: Final[int] = 0x80000000
_FILE_SHARE_READ_WRITE: Final[int] = 0x00000003
_OPEN_EXISTING: Final[int] = 3

_JOB_OBJECT_EXTENDED_LIMIT_INFO: Final[int] = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFO: Final[int] = 1
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final[int] = 0x00002000

_WAIT_OBJECT_0: Final[int] = 0x00000000
_INVALID_HANDLE_VALUE: Final[int] = ctypes.c_void_p(-1).value or -1

_SAFE_WIN_BASE_KEYS: Final[tuple[str, ...]] = (
    "COMSPEC",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "WINDIR",
)


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


@dataclass(frozen=True, slots=True)
class ProcessExecutionResult:
    """Bounded outcome of running one CommandSpec inside a Windows Job."""

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_bytes_seen: int
    stderr_bytes_seen: int
    timed_out: bool
    cancelled: bool
    cleanup_ok: bool
    active_processes_remaining: int
    duration_ms: float
    output_delivery_failed: bool = False


def _kernel32() -> ctypes.WinDLL:
    if os.name != "nt":
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            "Windows Job runner is supported only on Windows hosts",
            location=("process",),
        )
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    dll.CreateJobObjectW.restype = wintypes.HANDLE
    dll.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    dll.SetInformationJobObject.restype = wintypes.BOOL
    dll.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    dll.QueryInformationJobObject.restype = wintypes.BOOL
    dll.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    dll.AssignProcessToJobObject.restype = wintypes.BOOL
    dll.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    dll.TerminateJobObject.restype = wintypes.BOOL
    dll.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    dll.TerminateProcess.restype = wintypes.BOOL
    dll.ResumeThread.argtypes = [wintypes.HANDLE]
    dll.ResumeThread.restype = wintypes.DWORD
    dll.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    dll.WaitForSingleObject.restype = wintypes.DWORD
    dll.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    dll.GetExitCodeProcess.restype = wintypes.BOOL
    dll.CreatePipe.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.POINTER(_SECURITY_ATTRIBUTES),
        wintypes.DWORD,
    ]
    dll.CreatePipe.restype = wintypes.BOOL
    dll.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
    dll.SetHandleInformation.restype = wintypes.BOOL
    dll.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_SECURITY_ATTRIBUTES),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    dll.CreateFileW.restype = wintypes.HANDLE
    dll.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.POINTER(_SECURITY_ATTRIBUTES),
        ctypes.POINTER(_SECURITY_ATTRIBUTES),
        wintypes.BOOL,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPCWSTR,
        ctypes.POINTER(_STARTUPINFOW),
        ctypes.POINTER(_PROCESS_INFORMATION),
    ]
    dll.CreateProcessW.restype = wintypes.BOOL
    dll.CloseHandle.argtypes = [wintypes.HANDLE]
    dll.CloseHandle.restype = wintypes.BOOL
    return dll


def build_minimal_windows_env(
    *, spec: CommandSpec, temp_dir: Path
) -> dict[str, str]:
    """Construct a minimal, explicit, credential-free environment for a child command."""
    env: dict[str, str] = {}
    for key in _SAFE_WIN_BASE_KEYS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    sys_root = env.get("SYSTEMROOT") or os.environ.get("SystemRoot") or r"C:\Windows"
    exe_dir = str(Path(spec.executable).parent)
    env["PATH"] = f"{exe_dir};{sys_root}\\System32;{sys_root}"
    temp_str = str(temp_dir)
    env["TEMP"] = temp_str
    env["TMP"] = temp_str
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    for key, val in spec.env.items():
        env[key] = val
    return env


def _encode_env_block(env: dict[str, str]) -> ctypes.Array[ctypes.c_wchar]:
    items = sorted(env.items(), key=lambda kv: kv[0].upper())
    joined = "".join(f"{k}={v}\x00" for k, v in items) + "\x00"
    return ctypes.create_unicode_buffer(joined)


def _decode_bounded_utf8(raw: bytearray, limit: int) -> str:
    chunk = bytes(raw[:limit])
    while chunk:
        try:
            return chunk.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            # Trim incomplete trailing sequence if cut mid-codepoint; otherwise replace.
            if exc.start >= max(0, len(chunk) - 4):
                chunk = chunk[: exc.start]
            else:
                return chunk.decode("utf-8", errors="replace")
    return ""


def _read_pipe_bounded(
    fd: int,
    limit: int,
    out: dict[str, object],
    stream_name: str = "stdout",
    delivery_queue: queue.Queue[tuple[str, bytes] | None] | None = None,
    delivery_failed: threading.Event | None = None,
) -> None:
    buf = bytearray()
    total = 0
    truncated = False
    try:
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break
            total += len(chunk)
            if (
                delivery_queue is not None
                and delivery_failed is not None
                and not delivery_failed.is_set()
            ):
                try:
                    delivery_queue.put_nowait((stream_name, bytes(chunk)))
                except queue.Full:
                    delivery_failed.set()
            if len(buf) < limit:
                room = limit - len(buf)
                if len(chunk) <= room:
                    buf.extend(chunk)
                else:
                    buf.extend(chunk[:room])
                    truncated = True
            else:
                truncated = True
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    out["raw"] = buf
    out["total"] = total
    out["truncated"] = truncated


def _drain_output_queue(
    delivery_queue: queue.Queue[tuple[str, bytes] | None],
    on_output: Callable[[str, bytes], None],
    delivery_failed: threading.Event,
) -> None:
    while True:
        item = delivery_queue.get()
        if item is None:
            return
        if delivery_failed.is_set():
            continue
        stream, chunk = item
        try:
            on_output(stream, chunk)
        except Exception:  # noqa: BLE001 - isolate hostile or crashing callbacks
            delivery_failed.set()


def _finish_dispatcher(
    delivery_queue: queue.Queue[tuple[str, bytes] | None] | None,
    dispatch_thread: threading.Thread | None,
    delivery_failed: threading.Event | None,
    timeout_s: float,
) -> None:
    if delivery_queue is None or dispatch_thread is None or delivery_failed is None:
        return
    try:
        delivery_queue.put_nowait(None)
    except queue.Full:
        delivery_failed.set()
        while True:
            try:
                delivery_queue.get_nowait()
            except queue.Empty:
                break
        try:
            delivery_queue.put_nowait(None)
        except queue.Full:
            pass
    dispatch_thread.join(timeout=timeout_s)
    if dispatch_thread.is_alive():
        delivery_failed.set()


def _query_active_processes(k32: ctypes.WinDLL, h_job: wintypes.HANDLE) -> int:
    acct = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
    retlen = wintypes.DWORD(0)
    ok = k32.QueryInformationJobObject(
        h_job,
        _JOB_OBJECT_BASIC_ACCOUNTING_INFO,
        ctypes.byref(acct),
        ctypes.sizeof(acct),
        ctypes.byref(retlen),
    )
    if not ok:
        return -1
    return int(acct.ActiveProcesses)


def _cleanup_job_sync(
    k32: ctypes.WinDLL,
    h_job: wintypes.HANDLE | None,
    h_proc: wintypes.HANDLE | None,
    cleanup_ms: int,
) -> tuple[bool, int]:
    """Terminate the entire Job Object and verify zero surviving processes within budget."""
    active = 0
    deadline = time.monotonic() + (cleanup_ms / 1000.0)
    if h_job is not None:
        k32.TerminateJobObject(h_job, 1)
        while True:
            active = _query_active_processes(k32, h_job)
            if active == 0:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        k32.CloseHandle(h_job)
    if h_proc is not None:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        k32.WaitForSingleObject(h_proc, min(remaining_ms, 50))
        k32.CloseHandle(h_proc)
    return (active == 0), active


async def run_command_in_job(
    *,
    spec: CommandSpec,
    argv: tuple[str, ...],
    temp_dir: Path,
    max_output_bytes: int,
    process_cleanup_ms: int,
    on_output: Callable[[str, bytes], None] | None = None,
) -> ProcessExecutionResult:
    """Execute ``argv`` inside a dedicated Windows Job Object with bounded I/O and teardown."""
    if on_output is not None and not callable(on_output):
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            "on_output must be callable when provided",
            location=("process", "on_output"),
        )
    k32 = _kernel32()
    temp_dir.mkdir(parents=True, exist_ok=True)
    env_map = build_minimal_windows_env(spec=spec, temp_dir=temp_dir)
    env_block = _encode_env_block(env_map)
    cmdline = subprocess.list2cmdline(list(argv))

    sa = _SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(_SECURITY_ATTRIBUTES)
    sa.lpSecurityDescriptor = None
    sa.bInheritHandle = True

    out_read = wintypes.HANDLE()
    out_write = wintypes.HANDLE()
    err_read = wintypes.HANDLE()
    err_write = wintypes.HANDLE()
    h_nul: wintypes.HANDLE | None = None
    h_job: wintypes.HANDLE | None = None
    h_proc: wintypes.HANDLE | None = None
    h_thread: wintypes.HANDLE | None = None
    out_fd: int | None = None
    err_fd: int | None = None
    out_thread: threading.Thread | None = None
    err_thread: threading.Thread | None = None

    def _safe_close(handle: wintypes.HANDLE | None) -> None:
        if handle is None:
            return
        raw = getattr(handle, "value", handle)
        if raw and raw != _INVALID_HANDLE_VALUE:
            k32.CloseHandle(handle)

    if not k32.CreatePipe(ctypes.byref(out_read), ctypes.byref(out_write), ctypes.byref(sa), 0):
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            "failed to create stdout pipe",
            location=("process", "pipe"),
        )
    if not k32.CreatePipe(ctypes.byref(err_read), ctypes.byref(err_write), ctypes.byref(sa), 0):
        _safe_close(out_read)
        _safe_close(out_write)
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            "failed to create stderr pipe",
            location=("process", "pipe"),
        )

    k32.SetHandleInformation(out_read, _HANDLE_FLAG_INHERIT, 0)
    k32.SetHandleInformation(err_read, _HANDLE_FLAG_INHERIT, 0)
    h_nul = k32.CreateFileW(
        "NUL", _GENERIC_READ, _FILE_SHARE_READ_WRITE, ctypes.byref(sa), _OPEN_EXISTING, 0, None
    )

    h_job = k32.CreateJobObjectW(None, None)
    if not h_job:
        for h in (out_read, out_write, err_read, err_write, h_nul):
            _safe_close(h)
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            "failed to create Windows Job Object",
            location=("process", "job"),
        )

    ext = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    ext.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(
        h_job, _JOB_OBJECT_EXTENDED_LIMIT_INFO, ctypes.byref(ext), ctypes.sizeof(ext)
    ):
        for h in (h_job, out_read, out_write, err_read, err_write, h_nul):
            _safe_close(h)
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            "failed to configure KILL_ON_JOB_CLOSE on Windows Job Object",
            location=("process", "job"),
        )

    si = _STARTUPINFOW()
    si.cb = ctypes.sizeof(_STARTUPINFOW)
    si.dwFlags = _STARTF_USESTDHANDLES
    si.hStdInput = h_nul
    si.hStdOutput = out_write
    si.hStdError = err_write
    pi = _PROCESS_INFORMATION()

    started_ns = time.monotonic_ns()
    created = k32.CreateProcessW(
        spec.executable,
        ctypes.create_unicode_buffer(cmdline),
        None,
        None,
        True,
        _CREATE_SUSPENDED
        | _CREATE_UNICODE_ENVIRONMENT
        | _CREATE_NEW_PROCESS_GROUP
        | _CREATE_NO_WINDOW,
        ctypes.byref(env_block),
        spec.cwd,
        ctypes.byref(si),
        ctypes.byref(pi),
    )

    # Close child-owned write handles and NUL handle in parent immediately.
    _safe_close(out_write)
    _safe_close(err_write)
    _safe_close(h_nul)

    if not created:
        _safe_close(out_read)
        _safe_close(err_read)
        _safe_close(h_job)
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            f"CreateProcessW failed with WinError {ctypes.get_last_error()}",
            location=("process", "spawn"),
        )

    h_proc = pi.hProcess
    h_thread = pi.hThread

    # Assign suspended child into Job BEFORE resuming main thread so no unmanaged code runs.
    if not k32.AssignProcessToJobObject(h_job, h_proc):
        k32.TerminateProcess(h_proc, 1)
        _safe_close(h_thread)
        _safe_close(h_proc)
        _safe_close(h_job)
        _safe_close(out_read)
        _safe_close(err_read)
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            f"AssignProcessToJobObject failed with WinError {ctypes.get_last_error()}",
            location=("process", "job_assign"),
        )

    assert out_read.value is not None and err_read.value is not None
    out_fd = msvcrt.open_osfhandle(out_read.value, os.O_RDONLY)
    err_fd = msvcrt.open_osfhandle(err_read.value, os.O_RDONLY)
    out_box: dict[str, object] = {"raw": bytearray(), "total": 0, "truncated": False}
    err_box: dict[str, object] = {"raw": bytearray(), "total": 0, "truncated": False}
    delivery_queue: queue.Queue[tuple[str, bytes] | None] | None = None
    delivery_failed: threading.Event | None = None
    dispatch_thread: threading.Thread | None = None
    if on_output is not None:
        delivery_queue = queue.Queue(maxsize=512)
        delivery_failed = threading.Event()
        dispatch_thread = threading.Thread(
            target=_drain_output_queue,
            args=(delivery_queue, on_output, delivery_failed),
            daemon=True,
        )
        dispatch_thread.start()

    out_thread = threading.Thread(
        target=_read_pipe_bounded,
        args=(out_fd, max_output_bytes, out_box, "stdout", delivery_queue, delivery_failed),
        daemon=True,
    )
    err_thread = threading.Thread(
        target=_read_pipe_bounded,
        args=(err_fd, max_output_bytes, err_box, "stderr", delivery_queue, delivery_failed),
        daemon=True,
    )
    out_thread.start()
    err_thread.start()

    if k32.ResumeThread(h_thread) == 0xFFFFFFFF:
        _safe_close(h_thread)
        _cleanup_job_sync(k32, h_job, h_proc, process_cleanup_ms)
        _finish_dispatcher(delivery_queue, dispatch_thread, delivery_failed, 0.05)
        raise CodingError(
            CODING_PROCESS_START_FAILED,
            "ResumeThread failed after job assignment",
            location=("process", "resume"),
        )
    _safe_close(h_thread)
    h_thread = None

    timeout_deadline = time.monotonic() + (spec.timeout_ms / 1000.0)
    timed_out = False
    cancelled = False
    exit_code: int | None = None
    cleanup_ok = True
    remaining_active = 0

    try:
        while True:
            wait_res = k32.WaitForSingleObject(h_proc, 0)
            if wait_res == _WAIT_OBJECT_0:
                code_val = wintypes.DWORD(0)
                k32.GetExitCodeProcess(h_proc, ctypes.byref(code_val))
                raw_code = int(code_val.value)
                exit_code = raw_code - (1 << 32) if raw_code >= (1 << 31) else raw_code
                break
            if time.monotonic() >= timeout_deadline:
                timed_out = True
                break
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        cancelled = True
        cleanup_ok, remaining_active = _cleanup_job_sync(k32, h_job, h_proc, process_cleanup_ms)
        h_job = None
        h_proc = None
        join_limit = max(0.05, process_cleanup_ms / 1000.0)
        if out_thread is not None:
            out_thread.join(timeout=join_limit)
        if err_thread is not None:
            err_thread.join(timeout=join_limit)
        _finish_dispatcher(delivery_queue, dispatch_thread, delivery_failed, 0.05)
        if not cleanup_ok:
            raise CodingError(
                CODING_PROCESS_CLEANUP_FAILED,
                "descendant processes remained active after cancellation cleanup",
                location=("process", "cleanup"),
            ) from None
        raise

    if timed_out:
        cleanup_ok, remaining_active = _cleanup_job_sync(k32, h_job, h_proc, process_cleanup_ms)
        h_job = None
        h_proc = None
        join_limit = max(0.05, process_cleanup_ms / 1000.0)
        if out_thread is not None:
            out_thread.join(timeout=join_limit)
        if err_thread is not None:
            err_thread.join(timeout=join_limit)
        _finish_dispatcher(delivery_queue, dispatch_thread, delivery_failed, 0.05)
    else:
        # Process exited cleanly; close Job Object (killing any stray job survivors) and drain readers.
        cleanup_ok, remaining_active = _cleanup_job_sync(k32, h_job, h_proc, process_cleanup_ms)
        h_job = None
        h_proc = None
        join_limit = max(0.2, process_cleanup_ms / 1000.0)
        if out_thread is not None:
            out_thread.join(timeout=join_limit)
        if err_thread is not None:
            err_thread.join(timeout=join_limit)
        _finish_dispatcher(
            delivery_queue,
            dispatch_thread,
            delivery_failed,
            min(0.25, max(0.05, process_cleanup_ms / 1000.0)),
        )

    duration_ms = max(0.0, (time.monotonic_ns() - started_ns) / 1_000_000.0)
    stdout_text = _decode_bounded_utf8(out_box["raw"], max_output_bytes)  # type: ignore[arg-type]
    stderr_text = _decode_bounded_utf8(err_box["raw"], max_output_bytes)  # type: ignore[arg-type]

    return ProcessExecutionResult(
        argv=argv,
        exit_code=exit_code,
        stdout=stdout_text,
        stderr=stderr_text,
        stdout_truncated=bool(out_box["truncated"]),
        stderr_truncated=bool(err_box["truncated"]),
        stdout_bytes_seen=int(out_box["total"]),  # type: ignore[arg-type]
        stderr_bytes_seen=int(err_box["total"]),  # type: ignore[arg-type]
        timed_out=timed_out,
        cancelled=cancelled,
        cleanup_ok=cleanup_ok,
        active_processes_remaining=remaining_active,
        duration_ms=duration_ms,
        output_delivery_failed=bool(delivery_failed.is_set()) if delivery_failed is not None else False,
    )
