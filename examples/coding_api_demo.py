"""Bounded DeepSeek API Coding demo assembling Core + LLMPlanner + CodingAdapter."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final

from coding_agent_demo import _ACCEPTANCE_TEST
from deepseek_provider import ENDPOINT as DEEPSEEK_ENDPOINT
from deepseek_provider import DeepSeekProvider, make_client
from forecontract.tracing import JsonlTraceStore
from forecontract_coding import (
    CodingConfig,
    CodingRunResult,
    CodingTask,
    CommandSpec,
    ExpectedTestCase,
    WriteTarget,
    run_coding_task,
)
from forecontract_coding.workspace import ManagedWorkspace
from forecontract_llm import ModelProvider, PlannerConfig
from llm_api_smoke import _bounded_close, _ProviderRecorder

__all__ = [
    "ALLOWED_ENDPOINT",
    "ALLOWED_MODEL",
    "MAX_BUDGET_CNY",
    "MAX_OUTPUT_TOKENS",
    "MAX_REQUEST_BYTES",
    "MAX_REQUEST_SLOTS",
    "MAX_RESPONSE_BYTES",
    "PER_CALL_RESERVED_CNY",
    "CodingApiDemoError",
    "SessionBudgetGuard",
    "build_task",
    "main",
    "prepare_demo",
    "run_demo_with_client",
    "run_session",
    "validate_demo_evidence",
]

ALLOWED_ENDPOINT: Final[str] = "https://api.deepseek.com/chat/completions"
ALLOWED_MODEL: Final[str] = "deepseek-flash"
RUN_ID: Final[str] = "cod-t02-live-001"
GOAL_ID: Final[str] = "sum-even-fix"

MAX_REQUEST_SLOTS: Final[int] = 8
MAX_BUDGET_CNY: Final[Decimal] = Decimal(1)
INPUT_CNY_PER_MILLION: Final[Decimal] = Decimal(2)
OUTPUT_CNY_PER_MILLION: Final[Decimal] = Decimal(8)
PACKAGING_OVERHEAD_BYTES: Final[int] = 4096
MAX_REQUEST_BYTES: Final[int] = 32768
MAX_RESPONSE_BYTES: Final[int] = 65536
MAX_OUTPUT_TOKENS: Final[int] = 2048
PER_CALL_RESERVED_CNY: Final[Decimal] = (
    Decimal(MAX_REQUEST_BYTES + PACKAGING_OVERHEAD_BYTES) * INPUT_CNY_PER_MILLION
    + Decimal(MAX_OUTPUT_TOKENS) * OUTPUT_CNY_PER_MILLION
) / Decimal(1_000_000)

SINGLE_DEADLINE_SECONDS: Final[float] = 30.0
SESSION_DEADLINE_SECONDS: Final[float] = 300.0

_INITIAL_CALC: Final[str] = (
    '"""Synthetic calculator fixture for the live Coding demo."""\n\n\n'
    "def sum_even(values):\n"
    '    """Return the sum of the even integers in values."""\n'
    "    return sum(values)\n"
)
_INITIAL_CALC_SHA256: Final[str] = hashlib.sha256(_INITIAL_CALC.encode("utf-8")).hexdigest()
_ACCEPTANCE_TEST_SHA256: Final[str] = hashlib.sha256(_ACCEPTANCE_TEST.encode("utf-8")).hexdigest()

_GOAL_TEXT: Final[str] = (
    "先读取 calculator.py 并运行 pytest_unit 了解失败；"
    "修复 sum_even，使其只求输入整数中的偶数和，保留负数、零、空输入语义；"
    "自行决定代码修改，最后调用 verify_task。"
    "只能修改 calculator.py，不能修改测试或自选命令。"
)


class CodingApiDemoError(Exception):
    """Bounded host error; never surfaces credentials, HTTP headers, or client handles."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _check_node_no_reparse(node: Path) -> None:
    """Reject a single filesystem node if it exists and is a symlink or Windows reparse point."""
    if os.path.lexists(node):
        if node.is_symlink():
            raise CodingApiDemoError("demo path must not traverse a symlink or reparse point")
        if os.name == "nt":
            try:
                st = os.lstat(node)
            except OSError as exc:
                raise CodingApiDemoError(f"cannot inspect demo path component: {exc}") from None
            if getattr(st, "st_file_attributes", 0) & 0x0400:
                raise CodingApiDemoError("demo path must not contain a Windows reparse point")


def _check_no_reparse_ancestors(path: Path) -> None:
    """Reject symlinks/reparse points along uncollapsed path components before resolution."""
    cur = Path(path.anchor) if path.anchor else Path(".")
    if path.anchor:
        _check_node_no_reparse(cur)
    for part in path.parts:
        if part in (path.anchor, path.drive):
            continue
        cur = cur / part
        _check_node_no_reparse(cur)


def _validate_fresh_demo_root(root: Path) -> Path:
    """Validate raw unresolved demo root (no reparse/symlink, not existing) before resolving."""
    if not isinstance(root, (str, Path)):
        raise CodingApiDemoError("root must be a filesystem path")
    raw_str = os.fspath(root)
    if not isinstance(raw_str, str) or not raw_str.strip():
        raise CodingApiDemoError("root must not be blank")
    expanded = Path(raw_str).expanduser()
    # Check raw uncollapsed segments BEFORE abspath/resolve so '..' cannot cancel a junction/symlink.
    if expanded.is_absolute():
        _check_no_reparse_ancestors(expanded)
    else:
        anchor = Path(os.path.abspath(expanded.drive)) if expanded.drive else Path.cwd()
        _check_no_reparse_ancestors(anchor)
        cur = anchor
        for part in expanded.parts:
            if part in (expanded.anchor, expanded.drive):
                continue
            cur = cur / part
            _check_node_no_reparse(cur)
    abs_unresolved = Path(os.path.abspath(os.fspath(expanded)))
    _check_no_reparse_ancestors(abs_unresolved)
    if os.path.lexists(abs_unresolved):
        raise CodingApiDemoError("demo root already exists; refusing to overwrite")
    resolved = abs_unresolved.resolve()
    _check_no_reparse_ancestors(resolved)
    if os.path.lexists(resolved):
        raise CodingApiDemoError("demo root already exists; refusing to overwrite")
    return resolved


def _parse_positive_decimal(raw: str) -> Decimal:
    """Parse a finite strictly positive Decimal value without float precision loss."""
    if not isinstance(raw, str) or not raw.strip():
        raise CodingApiDemoError("budget_cny must be a finite positive decimal string")
    try:
        dec = Decimal(raw.strip())
    except InvalidOperation:
        raise CodingApiDemoError("budget_cny must be a valid decimal number") from None
    if not dec.is_finite() or dec <= Decimal(0):
        raise CodingApiDemoError("budget_cny must be finite and greater than zero")
    return dec


class SessionBudgetGuard:
    """Latched per-session request-slot, deadline, and Decimal reservation guard."""

    def __init__(
        self,
        *,
        max_requests: int = MAX_REQUEST_SLOTS,
        budget_cny: Decimal = MAX_BUDGET_CNY,
        session_deadline_seconds: float = SESSION_DEADLINE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(max_requests, bool) or not isinstance(max_requests, int):
            raise CodingApiDemoError("max_requests must be an integer")
        if max_requests < 1 or max_requests > MAX_REQUEST_SLOTS:
            raise CodingApiDemoError(f"max_requests must be between 1 and {MAX_REQUEST_SLOTS}")
        if not isinstance(budget_cny, Decimal) or not budget_cny.is_finite() or budget_cny <= 0:
            raise CodingApiDemoError("budget_cny must be a finite positive Decimal")
        if budget_cny > MAX_BUDGET_CNY:
            raise CodingApiDemoError(f"budget_cny exceeds ceiling {MAX_BUDGET_CNY}")
        if (
            isinstance(session_deadline_seconds, bool)
            or not isinstance(session_deadline_seconds, (int, float))
            or session_deadline_seconds <= 0
        ):
            raise CodingApiDemoError("session_deadline_seconds must be positive")

        self._max_requests = max_requests
        self._budget_cny = budget_cny
        self._clock = clock
        self._deadline_at = float(clock()) + float(session_deadline_seconds)
        self._inner: object | None = None

        self.slots_occupied: int = 0
        self.reserved_cny: Decimal = Decimal(0)
        self.cost_upper_bound_cny: Decimal | None = Decimal(0)
        self.stopped: bool = False
        self.stop_reason: str | None = None
        self._awaiting_settlement: bool = False

    def _latch(self, reason: str, *, unknown_cost: bool) -> CodingApiDemoError:
        self.stopped = True
        if self.stop_reason is None:
            self.stop_reason = reason
        if unknown_cost:
            self.cost_upper_bound_cny = None
        self._awaiting_settlement = False
        return CodingApiDemoError(reason)

    def remaining(self) -> float:
        return self._deadline_at - float(self._clock())

    def finalize_unsettled(self, *, reason: str = "unsettled_provider_call") -> None:
        if self._awaiting_settlement:
            self._latch(reason, unknown_cost=True)

    def before_send(self) -> None:
        """Reserve an irreversible slot and CNY headroom before transport executes."""
        if self._awaiting_settlement:
            raise self._latch("previous_call_unsettled", unknown_cost=True)
        if self.stopped:
            raise CodingApiDemoError(self.stop_reason or "session_guard_stopped")
        if float(self._clock()) >= self._deadline_at:
            raise self._latch("session_deadline_elapsed", unknown_cost=False)
        if self.slots_occupied >= self._max_requests:
            raise self._latch("request_cap_reached", unknown_cost=False)
        if self.reserved_cny + PER_CALL_RESERVED_CNY > self._budget_cny:
            raise self._latch("budget_insufficient_for_reservation", unknown_cost=False)

        self.slots_occupied += 1
        self.reserved_cny += PER_CALL_RESERVED_CNY
        self._awaiting_settlement = True

    def after_response(self, response: object) -> None:
        """Settle usage strictly against per-call reservation and latch on any anomaly."""
        self._awaiting_settlement = False
        if self.stopped:
            raise CodingApiDemoError(self.stop_reason or "session_guard_stopped")

        inner_status = getattr(self._inner, "last_http_status", None) if self._inner else None
        status = getattr(response, "status", None)
        if status != "ok" or (inner_status is not None and inner_status != 200):
            raise self._latch("provider_refused_or_http_error", unknown_cost=True)

        usage = getattr(response, "usage", None)
        if usage is None:
            raise self._latch("missing_provider_usage", unknown_cost=True)

        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        if (
            isinstance(in_tok, bool)
            or not isinstance(in_tok, int)
            or in_tok < 0
            or isinstance(out_tok, bool)
            or not isinstance(out_tok, int)
            or out_tok < 0
        ):
            raise self._latch("invalid_provider_usage", unknown_cost=True)

        if out_tok > MAX_OUTPUT_TOKENS:
            raise self._latch("output_tokens_exceeded", unknown_cost=True)

        call_cost = (
            Decimal(in_tok) * INPUT_CNY_PER_MILLION
            + Decimal(out_tok) * OUTPUT_CNY_PER_MILLION
        ) / Decimal(1_000_000)
        if call_cost > PER_CALL_RESERVED_CNY:
            raise self._latch("usage_exceeded_per_call_reservation", unknown_cost=True)

        if self.cost_upper_bound_cny is not None:
            next_total = self.cost_upper_bound_cny + call_cost
            if next_total > self._budget_cny:
                raise self._latch("cost_upper_bound_exceeded_budget", unknown_cost=True)
            self.cost_upper_bound_cny = next_total

        if float(self._clock()) >= self._deadline_at:
            raise self._latch("session_deadline_elapsed_at_settlement", unknown_cost=False)


def prepare_demo(root: Path) -> dict[str, Path]:
    """Create a brand-new demo root with fixed workspace, test, and empty log files."""
    target = _validate_fresh_demo_root(root)

    target.mkdir(parents=True, exist_ok=False)
    ws = target / "workspace"
    ext = target / "external_tests"
    out = target / "output"
    ws.mkdir(exist_ok=False)
    ext.mkdir(exist_ok=False)
    out.mkdir(exist_ok=False)

    calc_path = ws / "calculator.py"
    calc_path.write_bytes(_INITIAL_CALC.encode("utf-8"))
    test_path = ext / "test_acceptance.py"
    test_path.write_bytes(_ACCEPTANCE_TEST.encode("utf-8"))

    trace_path = target / "trace.jsonl"
    trace_path.touch(exist_ok=False)
    events_path = target / "provider-events.jsonl"
    events_path.touch(exist_ok=False)

    return {
        "workspace": ws,
        "acceptance_test": test_path,
        "output": out,
        "trace": trace_path,
        "provider_events": events_path,
    }


def _write_initial_records(root: Path, paths: dict[str, Path], task: CodingTask) -> None:
    """Persist initial task and immutable source hashes before running the loop."""
    before_doc = {
        "schema_version": "1",
        "initial_source_path": "workspace/calculator.py",
        "initial_source_sha256": _sha256_bytes(
            (paths["workspace"] / "calculator.py").read_bytes()
        ),
        "acceptance_test_path": "external_tests/test_acceptance.py",
        "acceptance_test_sha256": _sha256_bytes(paths["acceptance_test"].read_bytes()),
    }
    (root / "before.json").write_bytes(_canonical_json(before_doc) + b"\n")
    (root / "task.json").write_bytes(_canonical_json(task.model_dump(mode="json")) + b"\n")


def build_task(
    paths: dict[str, Path], *, python_executable: str
) -> tuple[CodingTask, CodingConfig, PlannerConfig]:
    """Construct the fixed CodingTask, CodingConfig, and PlannerConfig over ``paths``."""
    ws = Path(paths["workspace"])
    test_file = Path(paths["acceptance_test"])
    out = Path(paths["output"])
    exe = str(python_executable)

    task = CodingTask(
        schema_version="0.1",
        run_id=RUN_ID,
        goal_id=GOAL_ID,
        goal_text=_GOAL_TEXT,
        workspace_root=str(ws),
        read_paths=["calculator.py"],
        write_paths=[WriteTarget(path="calculator.py", operation="modify")],
        commands={
            "pytest_unit": CommandSpec(
                command_id="pytest_unit",
                executable=exe,
                args=[
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--import-mode=importlib",
                    f"--rootdir={test_file.parent}",
                    f"--confcutdir={test_file.parent}",
                    "--basetemp={result_dir}/pytest-temp",
                    "--junitxml={result_dir}/junit.xml",
                    str(test_file),
                ],
                cwd=str(ws),
                env={
                    "PYTHONPATH": str(ws),
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                },
                timeout_ms=15000,
                purpose="acceptance",
                expected_tests=[
                    ExpectedTestCase(classname="test_acceptance", name="test_basic_even"),
                    ExpectedTestCase(
                        classname="test_acceptance", name="test_empty_and_negatives"
                    ),
                ],
            )
        },
        verification_command_ids=["pytest_unit"],
        max_steps=8,
        max_planning_calls=8,
        max_replans=2,
        run_timeout_ms=300000,
    )
    config = CodingConfig(schema_version="0.1", output_root=str(out))
    planner_config = PlannerConfig(
        schema_version="0.1",
        max_calls=8,
        call_timeout_ms=30000,
        cleanup_timeout_ms=1000,
        max_request_bytes=MAX_REQUEST_BYTES,
        max_response_bytes=MAX_RESPONSE_BYTES,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return task, config, planner_config


async def run_session(
    *,
    task: CodingTask,
    config: CodingConfig,
    provider: ModelProvider,
    planner_config: PlannerConfig,
    trace_writer: object,
) -> CodingRunResult:
    """Delegate directly to existing ``forecontract_coding.run_coding_task``."""
    return await run_coding_task(
        task=task,
        config=config,
        provider=provider,
        planner_config=planner_config,
        trace_writer=trace_writer,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise TypeError("jsonl row must be an object")
        rows.append(parsed)
    return rows


def _extract_model_proposal(event: dict[str, Any]) -> dict[str, Any] | None:
    response = event.get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_request_payload(event: dict[str, Any]) -> dict[str, Any] | None:
    req = event.get("request")
    if not isinstance(req, dict):
        return None
    messages = req.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    second = messages[1]
    if not isinstance(second, dict) or not isinstance(second.get("content"), str):
        return None
    try:
        parsed = json.loads(second["content"])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _validate_provider_events(
    events: list[dict[str, Any]], issues: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate provider events, budgets, wire invariants, and parse req/proposal streams."""
    if not events:
        issues.append("empty:provider_events")
        return [], [], []
    if len(events) > MAX_REQUEST_SLOTS:
        issues.append("too_many_provider_calls")

    req_payloads: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    valid_events: list[dict[str, Any]] = []
    total_cost = Decimal(0)
    total_reserved = Decimal(0)

    for idx, ev in enumerate(events):
        if ev.get("endpoint") != ALLOWED_ENDPOINT:
            issues.append(f"invalid_endpoint:{idx}")
        wire = ev.get("request")
        if not isinstance(wire, dict):
            issues.append(f"missing_wire_request:{idx}")
        else:
            if wire.get("model") != ALLOWED_MODEL:
                issues.append(f"invalid_wire_model:{idx}")
            if wire.get("thinking") != {"type": "disabled"}:
                issues.append(f"invalid_wire_thinking:{idx}")
            if wire.get("stream") is not False:
                issues.append(f"invalid_wire_stream:{idx}")
            if wire.get("response_format") != {"type": "json_object"}:
                issues.append(f"invalid_wire_response_format:{idx}")
            max_tok = wire.get("max_tokens")
            if isinstance(max_tok, bool) or not isinstance(max_tok, int) or max_tok > MAX_OUTPUT_TOKENS:
                issues.append(f"invalid_wire_max_tokens:{idx}")
        req_bytes = ev.get("request_bytes")
        if isinstance(req_bytes, int) and req_bytes > MAX_REQUEST_BYTES:
            issues.append(f"request_bytes_exceeded:{idx}")

        if not ev.get("sent"):
            issues.append(f"provider_not_sent:{idx}")
        if ev.get("http_status") != 200 or ev.get("error_code") is not None:
            issues.append(f"provider_http_failure:{idx}")

        total_reserved += PER_CALL_RESERVED_CNY
        raw_resp = ev.get("response")
        usage = raw_resp.get("usage") if isinstance(raw_resp, dict) else None
        if not isinstance(usage, dict):
            issues.append(f"missing_usage:{idx}")
        else:
            in_tok = usage.get("prompt_tokens")
            out_tok = usage.get("completion_tokens")
            if (
                isinstance(in_tok, bool)
                or not isinstance(in_tok, int)
                or in_tok < 0
                or isinstance(out_tok, bool)
                or not isinstance(out_tok, int)
                or out_tok < 0
            ):
                issues.append(f"invalid_usage:{idx}")
            else:
                if out_tok > MAX_OUTPUT_TOKENS:
                    issues.append(f"usage_output_exceeded:{idx}")
                call_cost = (
                    Decimal(in_tok) * INPUT_CNY_PER_MILLION
                    + Decimal(out_tok) * OUTPUT_CNY_PER_MILLION
                ) / Decimal(1_000_000)
                if call_cost > PER_CALL_RESERVED_CNY:
                    issues.append(f"usage_exceeded_reservation:{idx}")
                total_cost += call_cost

        req_payload = _extract_request_payload(ev)
        proposal = _extract_model_proposal(ev)
        if req_payload is None:
            issues.append(f"unparseable_request_payload:{idx}")
        else:
            req_payloads.append(req_payload)
        if proposal is None:
            issues.append(f"unparseable_model_proposal:{idx}")
        else:
            proposals.append(proposal)
        valid_events.append(ev)

    if total_reserved > MAX_BUDGET_CNY or total_cost > MAX_BUDGET_CNY:
        issues.append("budget_exceeded")
    return valid_events, req_payloads, proposals


def _live_workspace_revision(task: CodingTask, config: CodingConfig) -> str:
    """Compute the full live workspace revision matching CodingWorkspaceAdapter."""
    ws = ManagedWorkspace(task=task, config=config)
    inventory = ws.capture_scope_baseline()
    _, snaps, _ = ws.snapshot()
    return ws.compute_full_revision(
        snapshots=snaps,
        inventory=inventory,
        scope_ok=True,
    )


def _extract_junit_arg_from_argv(argv: object) -> Path | None:
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return None
    matches = [item[len("--junitxml="):] for item in argv if item.startswith("--junitxml=")]
    if len(matches) != 1 or not matches[0]:
        return None
    return Path(matches[0])


def _extract_bound_junit_from_run_result(
    run_result: CodingRunResult | None, out_dir: Path
) -> Path | None:
    if run_result is None or not run_result.acceptance.action_id:
        return None
    for item in run_result.acceptance.results:
        if not isinstance(item, dict):
            continue
        if item.get("command_id") != "pytest_unit":
            continue
        junit_path = _extract_junit_arg_from_argv(item.get("argv"))
        if junit_path is None:
            return None
        try:
            _check_no_reparse_ancestors(junit_path)
            resolved_verify = (out_dir / "verify").resolve()
            resolved_junit = junit_path.resolve()
        except CodingApiDemoError:
            return None
        if resolved_verify not in resolved_junit.parents:
            return None
        return resolved_junit
    return None


def _norm_path_arg(raw: str) -> str:
    return os.path.normcase(os.path.normpath(raw))


def _argv_matches_expected(actual: object, expected: list[str]) -> bool:
    if not isinstance(actual, list) or not all(isinstance(item, str) for item in actual):
        return False
    if len(actual) != len(expected):
        return False
    last_idx = len(expected) - 1
    for idx, (act, exp) in enumerate(zip(actual, expected, strict=True)):
        if act == exp:
            continue
        if idx in (0, last_idx):
            if _norm_path_arg(act) != _norm_path_arg(exp):
                return False
            continue
        if exp.startswith(("--rootdir=", "--confcutdir=", "--basetemp=", "--junitxml=")):
            prefix, exp_val = exp.split("=", 1)
            if not act.startswith(prefix + "="):
                return False
            act_val = act.split("=", 1)[1]
            if "{result_dir}" in exp_val:
                if act_val.replace("\\", "/") != exp_val.replace("\\", "/"):
                    return False
            elif _norm_path_arg(act_val) != _norm_path_arg(exp_val):
                return False
            continue
        return False
    return True


def _validate_fixed_task_command(
    validated_task: CodingTask, target: Path, issues: list[str]
) -> None:
    target_resolved = target.resolve()
    expected_ws = target_resolved / "workspace"
    expected_test = target_resolved / "external_tests" / "test_acceptance.py"
    if _norm_path_arg(validated_task.workspace_root) != _norm_path_arg(str(expected_ws)):
        issues.append("task_workspace_root_mismatch")
    if list(validated_task.commands.keys()) != ["pytest_unit"]:
        issues.append("task_commands_keys_mismatch")
        return
    spec = validated_task.commands["pytest_unit"]
    exe_path = Path(spec.executable)
    if not exe_path.is_absolute() or not exe_path.stem.lower().startswith("python") or not exe_path.is_file():
        issues.append("task_command_executable_invalid")
    if spec.cwd is None or _norm_path_arg(spec.cwd) != _norm_path_arg(str(expected_ws)):
        issues.append("task_command_cwd_mismatch")
    if (
        not isinstance(spec.env, dict)
        or spec.env.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1"
        or _norm_path_arg(spec.env.get("PYTHONPATH", "")) != _norm_path_arg(str(expected_ws))
    ):
        issues.append("task_command_env_mismatch")
    expected_args = [
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--import-mode=importlib",
        f"--rootdir={expected_test.parent}",
        f"--confcutdir={expected_test.parent}",
        "--basetemp={result_dir}/pytest-temp",
        "--junitxml={result_dir}/junit.xml",
        str(expected_test),
    ]
    if not _argv_matches_expected([spec.executable, *spec.args], [spec.executable, *expected_args]):
        issues.append("task_command_args_mismatch")


def _files_sha_map(state: dict[str, Any]) -> dict[str, str]:
    raw_files = state.get("files")
    if not isinstance(raw_files, list):
        return {}
    return {
        entry["path"]: entry["sha256"]
        for entry in raw_files
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and isinstance(entry.get("sha256"), str)
    }


def _validate_write_feedback(
    req_payloads: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    issues: list[str],
) -> None:
    """Validate that every model write_file has matching next-request actual write feedback."""
    for w_idx, proposal in enumerate(proposals):
        act = proposal.get("proposed_action")
        if not isinstance(act, dict) or act.get("action_type") != "write_file":
            continue
        payload = act.get("payload") if isinstance(act.get("payload"), dict) else {}
        if payload.get("path") != "calculator.py" or not isinstance(payload.get("content"), str):
            issues.append(f"invalid:write_proposal_payload:{w_idx}")
            continue
        written_sha = _sha256_bytes(payload["content"].encode("utf-8"))
        if w_idx >= len(req_payloads):
            issues.append(f"missing:write_request_context:{w_idx}")
            continue
        pre_state = (req_payloads[w_idx].get("observation") or {}).get("state")
        if not isinstance(pre_state, dict):
            issues.append(f"missing:write_pre_state:{w_idx}")
            continue
        pre_rev = pre_state.get("workspace_revision")
        pre_calc_sha = _files_sha_map(pre_state).get("calculator.py")
        if not isinstance(pre_rev, str) or not pre_rev or not pre_calc_sha:
            issues.append(f"invalid:write_pre_state_identity:{w_idx}")
            continue
        if payload.get("expected_sha256") != pre_calc_sha:
            issues.append(f"write_expected_sha_mismatch:{w_idx}")

        if w_idx + 1 >= len(req_payloads):
            issues.append("missing:write_feedback_followup")
            continue
        next_state = (req_payloads[w_idx + 1].get("observation") or {}).get("state")
        if not isinstance(next_state, dict):
            issues.append(f"missing:write_feedback_state:{w_idx}")
            continue
        last_res = next_state.get("last_tool_result")
        if not isinstance(last_res, dict):
            issues.append(f"missing:write_feedback_result:{w_idx}")
            continue
        if (
            last_res.get("action_type") != "write_file"
            or last_res.get("path") != "calculator.py"
            or last_res.get("operation") != "modify"
        ):
            issues.append(f"invalid:write_feedback_identity:{w_idx}")
        if last_res.get("tool_succeeded") is not True:
            issues.append(f"invalid:write_feedback_not_succeeded:{w_idx}")
        if last_res.get("read_back_match") is not True:
            issues.append(f"invalid:write_feedback_read_back_mismatch:{w_idx}")
        if (
            last_res.get("before_sha256") != pre_calc_sha
            or last_res.get("written_sha256") != written_sha
            or last_res.get("read_back_sha256") != written_sha
        ):
            issues.append(f"invalid:write_feedback_sha_mismatch:{w_idx}")
        after_rev = last_res.get("after_revision")
        if (
            last_res.get("before_revision") != pre_rev
            or not isinstance(after_rev, str)
            or not after_rev
            or after_rev != next_state.get("workspace_revision")
        ):
            issues.append(f"invalid:write_feedback_revision_mismatch:{w_idx}")
        elif written_sha != pre_calc_sha and after_rev == pre_rev:
            issues.append(f"invalid:write_feedback_revision_unchanged:{w_idx}")
        if _files_sha_map(next_state).get("calculator.py") != written_sha:
            issues.append(f"invalid:write_followup_observation_sha_mismatch:{w_idx}")


def _validate_bound_junit_file(
    junit_path: Path, verify_dir: Path, action_id: str, issues: list[str]
) -> None:
    """Validate strictly the bound JUnit report file without globbing or mtime selection."""
    try:
        _check_no_reparse_ancestors(junit_path)
    except CodingApiDemoError:
        issues.append("reparse:bound_junit_xml")
        return
    if not verify_dir.is_dir():
        issues.append("missing:output_verify")
        return
    resolved_verify = verify_dir.resolve()
    resolved_junit = junit_path.resolve()
    if resolved_junit.parent.parent != resolved_verify:
        issues.append("invalid:bound_junit_outside_verify")
        return
    if not resolved_junit.parent.name.startswith(f"{action_id}__cmd-pytest_unit-"):
        issues.append("invalid:bound_junit_action_dir_mismatch")
        return
    if not resolved_junit.is_file():
        issues.append("missing:bound_junit_xml")
        return

    try:
        tree = ET.fromstring(resolved_junit.read_bytes())
    except ET.ParseError:
        issues.append("unparseable:junit_xml")
        return

    suites = [tree] if tree.tag == "testsuite" else list(tree.findall(".//testsuite"))
    if not suites:
        issues.append("missing:junit_testsuite")
        return

    cases: list[ET.Element] = []
    for suite in suites:
        cases.extend(suite.findall("testcase"))
    case_ids = {
        (tc.attrib.get("classname", ""), tc.attrib.get("name", "")) for tc in cases
    }
    expected_cases = {
        ("test_acceptance", "test_basic_even"),
        ("test_acceptance", "test_empty_and_negatives"),
    }
    if len(cases) != 2 or case_ids != expected_cases:
        issues.append("invalid:junit_cases")
        return
    for tc in cases:
        if (
            tc.find("failure") is not None
            or tc.find("error") is not None
            or tc.find("skipped") is not None
        ):
            issues.append("invalid:junit_case_not_passed")
            return


def _validate_trace_execution_and_oracle(
    trace_rows: list[dict[str, Any]], issues: list[str]
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Validate actual attempt execution pairs and exact run-finished verification oracle."""
    counts = {
        "steps_started": 0,
        "attempts_started": 0,
        "attempts_finished": 0,
    }
    if not trace_rows:
        issues.append("empty:trace_jsonl")
        return None, counts

    record_map: dict[str, dict[str, Any]] = {}
    started_steps: set[str] = set()
    committed_contracts: dict[str, str] = {}
    started_attempts: dict[str, tuple[str, str]] = {}
    finished_attempts: set[str] = set()
    attempt_passed: set[str] = set()

    for expected_seq, row in enumerate(trace_rows):
        if row.get("run_id") != RUN_ID:
            issues.append(f"trace_run_id_mismatch:{expected_seq}")
        if row.get("sequence") != expected_seq:
            issues.append(f"trace_sequence_mismatch:{expected_seq}")
        rec_id = row.get("record_id")
        if not isinstance(rec_id, str) or not rec_id:
            issues.append(f"trace_missing_record_id:{expected_seq}")
        elif rec_id in record_map:
            issues.append(f"trace_duplicate_record_id:{rec_id}")
        else:
            record_map[rec_id] = row

        kind = row.get("kind")
        if kind == "step_started":
            sid = row.get("step_id")
            if isinstance(sid, str) and sid:
                started_steps.add(sid)
        elif kind == "contract_committed":
            sid = row.get("step_id")
            cid = row.get("contract_id")
            if isinstance(sid, str) and sid in started_steps and isinstance(cid, str) and cid:
                committed_contracts[cid] = sid
            else:
                issues.append(f"trace_orphan_contract:{expected_seq}")
        elif kind == "attempt_started":
            sid = row.get("step_id")
            cid = row.get("contract_id")
            aid = row.get("attempt_id")
            if (
                isinstance(aid, str)
                and aid
                and isinstance(cid, str)
                and committed_contracts.get(cid) == sid
            ):
                started_attempts[aid] = (sid, cid)
            else:
                issues.append(f"trace_orphan_attempt_start:{expected_seq}")
        elif kind == "attempt_finished":
            sid = row.get("step_id")
            cid = row.get("contract_id")
            aid = row.get("attempt_id")
            if not isinstance(aid, str) or started_attempts.get(aid) != (sid, cid):
                issues.append(f"trace_orphan_attempt_finish:{expected_seq}")
            elif row.get("execution_status") != "succeeded":
                issues.append(f"trace_attempt_not_succeeded:{aid}")
            else:
                finished_attempts.add(aid)
        elif kind == "verification_recorded" and row.get("scope") == "attempt":
            sid = row.get("step_id")
            cid = row.get("contract_id")
            aid = row.get("attempt_id")
            if (
                isinstance(aid, str)
                and started_attempts.get(aid) == (sid, cid)
                and row.get("verification_status") == "pass"
            ):
                attempt_passed.add(aid)

    counts["steps_started"] = len(started_steps)
    counts["attempts_started"] = len(started_attempts)
    counts["attempts_finished"] = len(finished_attempts)

    if trace_rows[0].get("kind") != "run_started":
        issues.append("missing:trace_run_started")
    if not started_attempts or not finished_attempts:
        issues.append("missing:actual_execution_attempts")
    elif set(started_attempts) != finished_attempts:
        issues.append("mismatch:attempt_started_finished")
    elif set(started_attempts) != attempt_passed:
        issues.append("mismatch:attempt_verification_pass")

    last_trace = trace_rows[-1]
    if last_trace.get("kind") != "run_finished":
        issues.append("trace_not_finished")
        return last_trace, counts
    if last_trace.get("terminal_status") != "success":
        issues.append("trace_terminal_not_success")

    oracle_id = last_trace.get("verification_record_id")
    if not isinstance(oracle_id, str) or not oracle_id:
        issues.append("missing:run_verification_record_id")
    else:
        oracles = [r for r in trace_rows if r.get("record_id") == oracle_id]
        if len(oracles) != 1:
            issues.append("invalid:run_verification_oracle_count")
        else:
            oracle = oracles[0]
            if oracle.get("kind") != "verification_recorded" or oracle.get("scope") != "run":
                issues.append("invalid:run_verification_oracle_kind")
            elif oracle.get("verification_status") != "pass":
                issues.append("failed:run_verification_oracle")

    return last_trace, counts


def _validate_persisted_run_result(
    *,
    run_result_path: Path,
    out_dir: Path,
    validated_task: CodingTask | None,
    test_path: Path,
    disk_calc_sha: str,
    live_rev: str,
    events: list[dict[str, Any]],
    last_trace: dict[str, Any] | None,
    trace_counts: dict[str, int],
    issues: list[str],
) -> None:
    if not run_result_path.is_file():
        issues.append("missing:coding_run_result_json")
        return
    try:
        raw_doc = json.loads(run_result_path.read_bytes().decode("utf-8"))
        coding_res = CodingRunResult.model_validate(raw_doc)
    except Exception:  # noqa: BLE001 - strict validation error becomes an evidence issue
        issues.append("invalid:coding_run_result_json")
        return

    rr = coding_res.run_result
    if coding_res.terminal_status != "completed" or rr.terminal_status.value != "completed":
        issues.append("coding_run_result_not_completed")
    if rr.run_id != RUN_ID or rr.goal_id != GOAL_ID:
        issues.append("coding_run_result_identity_mismatch")
    if rr.steps_started != trace_counts["steps_started"]:
        issues.append("coding_run_result_steps_mismatch")
    if rr.attempts_started == 0 or rr.attempts_started != trace_counts["attempts_started"]:
        issues.append("coding_run_result_attempts_mismatch")
    if rr.planning_calls != len(events):
        issues.append("coding_run_result_planning_calls_mismatch")
    if last_trace is not None:
        if rr.terminal_record_id != last_trace.get("record_id"):
            issues.append("coding_run_result_terminal_record_mismatch")
        if rr.verification_record_id != last_trace.get("verification_record_id"):
            issues.append("coding_run_result_oracle_record_mismatch")

    if coding_res.final_workspace_revision != live_rev:
        issues.append("coding_run_result_final_revision_mismatch")
    changed_summary = [
        (c.path, c.operation, c.before_sha256, c.after_sha256)
        for c in coding_res.changed_paths
    ]
    if changed_summary != [("calculator.py", "modify", _INITIAL_CALC_SHA256, disk_calc_sha)]:
        issues.append("coding_run_result_changed_paths_mismatch")

    acc = coding_res.acceptance
    if acc.status != "passed":
        issues.append("coding_run_result_acceptance_not_passed")
    if acc.revision != live_rev:
        issues.append("coding_run_result_acceptance_revision_mismatch")
    if not acc.action_id:
        issues.append("missing:coding_run_result_acceptance_action_id")
    if acc.command_ids != ["pytest_unit"]:
        issues.append("coding_run_result_command_ids_mismatch")
    if validated_task is not None and acc.commands_digest != validated_task.commands_digest():
        issues.append("coding_run_result_commands_digest_mismatch")
    if len(acc.results) != 1 or not isinstance(acc.results[0], dict):
        issues.append("invalid:coding_run_result_acceptance_results")
        return

    cmd_res = acc.results[0]
    if cmd_res.get("command_id") != "pytest_unit":
        issues.append("coding_run_result_cmd_id_mismatch")
    if cmd_res.get("action_id") != acc.action_id:
        issues.append("coding_run_result_cmd_action_id_mismatch")
    if cmd_res.get("exit_code") != 0 or cmd_res.get("status") != "passed":
        issues.append("coding_run_result_cmd_not_passed")
    if cmd_res.get("before_revision") != live_rev or cmd_res.get("after_revision") != live_rev:
        issues.append("coding_run_result_cmd_revision_mismatch")

    test_hashes = cmd_res.get("test_source_hashes")
    resolved_test_str = str(test_path.resolve())
    if not isinstance(test_hashes, dict) or (
        test_hashes.get(str(test_path)) != _ACCEPTANCE_TEST_SHA256
        and test_hashes.get(resolved_test_str) != _ACCEPTANCE_TEST_SHA256
    ):
        issues.append("coding_run_result_test_source_hash_mismatch")

    bound_junit = _extract_junit_arg_from_argv(cmd_res.get("argv"))
    if bound_junit is None:
        issues.append("missing:bound_junit_argv")
        return
    if validated_task is not None and "pytest_unit" in validated_task.commands:
        spec = validated_task.commands["pytest_unit"]
        expected_argv = [
            spec.executable,
            *(arg.replace("{result_dir}", str(bound_junit.parent)) for arg in spec.args),
        ]
        if not _argv_matches_expected(cmd_res.get("argv"), expected_argv):
            issues.append("coding_run_result_cmd_argv_mismatch")
    if acc.action_id:
        _validate_bound_junit_file(bound_junit, out_dir / "verify", acc.action_id, issues)


def validate_demo_evidence(root: Path) -> list[str]:
    """Read raw provider/trace/JUnit/source/workspace files back and return any defect list.

    Never reads summary status/passed claims as proof of completion.
    """
    target = Path(root)
    issues: list[str] = []
    calc_path = target / "workspace" / "calculator.py"
    test_path = target / "external_tests" / "test_acceptance.py"
    before_path = target / "before.json"
    task_path = target / "task.json"
    trace_path = target / "trace.jsonl"
    events_path = target / "provider-events.jsonl"
    out_dir = target / "output"

    for label, p in (
        ("workspace_calc", calc_path),
        ("acceptance_test", test_path),
        ("before_json", before_path),
        ("task_json", task_path),
        ("trace_jsonl", trace_path),
        ("provider_events_jsonl", events_path),
    ):
        if not p.is_file():
            issues.append(f"missing:{label}")
    if not out_dir.is_dir():
        issues.append("missing:output_dir")
    if issues:
        return issues

    try:
        before_doc = json.loads(before_path.read_bytes().decode("utf-8"))
        task_doc = json.loads(task_path.read_bytes().decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return ["unparseable:metadata_json"]

    if before_doc.get("initial_source_sha256") != _INITIAL_CALC_SHA256:
        issues.append("initial_source_sha_mismatch")
    if before_doc.get("acceptance_test_sha256") != _ACCEPTANCE_TEST_SHA256:
        issues.append("acceptance_before_sha_mismatch")
    if _sha256_bytes(test_path.read_bytes()) != _ACCEPTANCE_TEST_SHA256:
        issues.append("acceptance_source_sha_changed")

    try:
        validated_task = CodingTask.model_validate(task_doc)
    except Exception:  # noqa: BLE001 - strict validation error becomes an evidence issue
        issues.append("invalid:task_json")
        validated_task = None
    if validated_task is not None:
        if validated_task.run_id != RUN_ID or validated_task.goal_id != GOAL_ID:
            issues.append("task_identity_mismatch")
        if validated_task.read_paths != ["calculator.py"]:
            issues.append("task_read_scope_mismatch")
        if [(wt.path, wt.operation) for wt in validated_task.write_paths] != [
            ("calculator.py", "modify")
        ]:
            issues.append("task_write_scope_mismatch")
        if validated_task.verification_command_ids != ["pytest_unit"]:
            issues.append("task_verification_commands_mismatch")
        _validate_fixed_task_command(validated_task, target, issues)

    last_trace: dict[str, Any] | None = None
    trace_counts = {"steps_started": 0, "attempts_started": 0, "attempts_finished": 0}
    try:
        trace_rows = _read_jsonl(trace_path)
    except (ValueError, TypeError, UnicodeDecodeError):
        issues.append("unparseable:trace_jsonl")
        trace_rows = []
    last_trace, trace_counts = _validate_trace_execution_and_oracle(trace_rows, issues)

    try:
        events = _read_jsonl(events_path)
    except (ValueError, TypeError, UnicodeDecodeError):
        issues.append("unparseable:provider_events_jsonl")
        events = []

    _, req_payloads, proposals = _validate_provider_events(events, issues)

    # Check causal coding feedback sequence across model requests and proposals
    read_indices: list[int] = []
    first_write_idx: int | None = None
    last_write_idx: int | None = None
    last_write_content_sha: str | None = None
    failed_diag_idx: int | None = None
    verify_idx: int | None = None

    for idx, proposal in enumerate(proposals):
        act = proposal.get("proposed_action")
        if not isinstance(act, dict):
            continue
        atype = act.get("action_type")
        payload = act.get("payload") if isinstance(act.get("payload"), dict) else {}
        if atype == "read_file" and payload.get("path") == "calculator.py":
            read_indices.append(idx)
        elif atype == "write_file" and payload.get("path") == "calculator.py":
            if first_write_idx is None:
                first_write_idx = idx
            last_write_idx = idx
            if isinstance(payload.get("content"), str):
                last_write_content_sha = _sha256_bytes(payload["content"].encode("utf-8"))
        elif atype == "run_command" and payload.get("command_id") == "pytest_unit":
            if idx + 1 < len(req_payloads):
                next_state = (req_payloads[idx + 1].get("observation") or {}).get("state") or {}
                last_res = next_state.get("last_tool_result") or {}
                if (
                    last_res.get("action_type") == "run_command"
                    and last_res.get("command_id") == "pytest_unit"
                    and last_res.get("exit_code") == 1
                ):
                    failed_diag_idx = idx
        elif atype == "verify_task":
            verify_idx = idx

    if not read_indices:
        issues.append("missing:real_read_file")
    if first_write_idx is None or last_write_idx is None or last_write_content_sha is None:
        issues.append("missing:real_write_file")
    elif not read_indices or read_indices[0] >= first_write_idx:
        issues.append("missing:read_before_first_write")
    if len(req_payloads) == len(proposals) == len(events):
        _validate_write_feedback(req_payloads, proposals, issues)

    if failed_diag_idx is None:
        issues.append("missing:failed_pytest_feedback")
    elif last_write_idx is not None and failed_diag_idx >= last_write_idx:
        issues.append("failed_pytest_not_before_final_fix")

    if verify_idx is None:
        issues.append("missing:verify_task")
    elif last_write_idx is not None and verify_idx <= last_write_idx:
        issues.append("verify_not_after_write")

    disk_calc_sha = _sha256_bytes(calc_path.read_bytes())
    if disk_calc_sha == _INITIAL_CALC_SHA256:
        issues.append("workspace_calculator_unchanged")
    if last_write_content_sha is not None and disk_calc_sha != last_write_content_sha:
        issues.append("written_content_sha_mismatch")

    ws_entries = sorted(p.relative_to(target / "workspace").as_posix() for p in (target / "workspace").rglob("*") if p.is_file())
    if ws_entries != ["calculator.py"]:
        issues.append("workspace_unexpected_files")

    cfg = CodingConfig(schema_version="0.1", output_root=str(out_dir))
    live_rev = _live_workspace_revision(validated_task, cfg) if validated_task is not None else ""

    _validate_persisted_run_result(
        run_result_path=out_dir / "coding-run-result.json",
        out_dir=out_dir,
        validated_task=validated_task,
        test_path=test_path,
        disk_calc_sha=disk_calc_sha,
        live_rev=live_rev,
        events=events,
        last_trace=last_trace,
        trace_counts=trace_counts,
        issues=issues,
    )

    # Verify live workspace revision against final verify request revision (and summary if present)
    if validated_task is not None and verify_idx is not None and verify_idx < len(req_payloads):
        verify_state = (req_payloads[verify_idx].get("observation") or {}).get("state") or {}
        verify_rev = verify_state.get("workspace_revision")
        if verify_rev != live_rev:
            issues.append("workspace_revision_mismatch_after_verify")
        summary_path = target / "summary.json"
        if summary_path.is_file():
            try:
                summary_doc = json.loads(summary_path.read_bytes().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                issues.append("unparseable:summary_json")
            else:
                if summary_doc.get("acceptance_revision") != live_rev:
                    issues.append("summary_acceptance_revision_mismatch")
                if summary_doc.get("final_workspace_revision") != live_rev:
                    issues.append("summary_final_revision_mismatch")

    return issues


def _compute_usage_and_cost(
    events: list[dict[str, Any]], guard: SessionBudgetGuard
) -> tuple[dict[str, int] | None, str | None]:
    if guard.cost_upper_bound_cny is None or not events:
        return None, None
    total_in = 0
    total_out = 0
    for ev in events:
        resp = ev.get("response")
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if not isinstance(usage, dict):
            return None, None
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        if (
            isinstance(in_tok, bool)
            or not isinstance(in_tok, int)
            or isinstance(out_tok, bool)
            or not isinstance(out_tok, int)
        ):
            return None, None
        total_in += in_tok
        total_out += out_tok
    return {"input_tokens": total_in, "output_tokens": total_out}, str(guard.cost_upper_bound_cny)


def _extract_junit_summary(bound_junit: Path | None) -> dict[str, Any] | None:
    if bound_junit is None or not bound_junit.is_file():
        return None
    try:
        tree = ET.fromstring(bound_junit.read_bytes())
    except ET.ParseError:
        return None
    suites = [tree] if tree.tag == "testsuite" else list(tree.findall(".//testsuite"))
    cases: list[tuple[str, str]] = []
    failures = errors = skipped = 0
    for suite in suites:
        for tc in suite.findall("testcase"):
            cases.append((tc.attrib.get("classname", ""), tc.attrib.get("name", "")))
            if tc.find("failure") is not None:
                failures += 1
            if tc.find("error") is not None:
                errors += 1
            if tc.find("skipped") is not None:
                skipped += 1
    return {
        "total": len(cases),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "cases": [{"classname": c, "name": n} for c, n in cases],
    }


def _extract_step_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for idx, ev in enumerate(events):
        proposal = _extract_model_proposal(ev)
        action = proposal.get("proposed_action") if isinstance(proposal, dict) else None
        action_type = action.get("action_type") if isinstance(action, dict) else None
        req_payload = _extract_request_payload(ev)
        state = (
            ((req_payload or {}).get("observation") or {}).get("state") or {}
        )
        steps.append(
            {
                "call_index": idx + 1,
                "action_type": action_type,
                "http_status": ev.get("http_status"),
                "sent": bool(ev.get("sent")),
                "response_model": ev.get("model"),
                "observed_last_tool_result": state.get("last_tool_result"),
            }
        )
    return steps


async def run_demo_with_client(
    *,
    root: Path,
    client: object,
    mode: str,
    model: str = ALLOWED_MODEL,
    max_requests: int = MAX_REQUEST_SLOTS,
    budget_cny: Decimal = MAX_BUDGET_CNY,
    session_deadline_seconds: float = SESSION_DEADLINE_SECONDS,
    single_deadline_seconds: float = SINGLE_DEADLINE_SECONDS,
    python_executable: str | None = None,
) -> tuple[dict[str, object], int]:
    """Execute one complete demo session using an injected AsyncClient (never reads env keys)."""
    started_at = time.monotonic()
    run_result: CodingRunResult | None = None
    failure_reason: str | None = None
    was_cancelled = False
    close_failure: object | None = None
    close_cancelled = False
    paths: dict[str, Path] | None = None
    target: Path | None = None
    task: CodingTask | None = None
    config: CodingConfig | None = None
    guard: SessionBudgetGuard | None = None

    try:
        if DEEPSEEK_ENDPOINT != ALLOWED_ENDPOINT:
            raise CodingApiDemoError("deepseek_provider endpoint drifted from contract endpoint")
        target = _validate_fresh_demo_root(root)
        paths = prepare_demo(target)
        py_exe = python_executable or sys.executable
        task, config, planner_config = build_task(paths, python_executable=py_exe)
        _write_initial_records(target, paths, task)

        guard = SessionBudgetGuard(
            max_requests=max_requests,
            budget_cny=budget_cny,
            session_deadline_seconds=session_deadline_seconds,
        )
        provider = DeepSeekProvider(
            client=client,
            model=model,
            max_request_bytes=MAX_REQUEST_BYTES,
            max_response_bytes=MAX_RESPONSE_BYTES,
            timeout_seconds=single_deadline_seconds,
            max_output_tokens_cap=MAX_OUTPUT_TOKENS,
            remaining_seconds=guard.remaining,
        )
        recording_provider = _ProviderRecorder(
            provider,
            paths["provider_events"],
            mode=mode,
            case_id=GOAL_ID,
            guard=guard,
        )
        trace_writer = JsonlTraceStore(paths["trace"])

        try:
            run_result = await run_session(
                task=task,
                config=config,
                provider=recording_provider,
                planner_config=planner_config,
                trace_writer=trace_writer,
            )
        except asyncio.CancelledError:
            was_cancelled = True
            failure_reason = "session_cancelled"
            guard.finalize_unsettled(reason="session_cancelled")
        except Exception as exc:  # noqa: BLE001 - bounded failure written cleanly before summary
            failure_reason = f"{type(exc).__name__}: {str(exc)[:120]}"
            guard.finalize_unsettled(reason=type(exc).__name__)
        finally:
            guard.finalize_unsettled()
    finally:
        try:
            close_failure = await _bounded_close(client)
        except asyncio.CancelledError:
            close_cancelled = True
            close_failure = CodingApiDemoError("cancelled_during_client_close")

    if target is None or paths is None or task is None or config is None or guard is None:
        raise CodingApiDemoError(failure_reason or "demo initialization failed")

    # Persist real CodingRunResult verbatim before evidence validation and summary
    if run_result is not None:
        (paths["output"] / "coding-run-result.json").write_bytes(
            _canonical_json(run_result.model_dump(mode="json")) + b"\n"
        )

    # Compute diff from actual disk state before reading back evidence
    after_calc_bytes = (paths["workspace"] / "calculator.py").read_bytes()
    after_calc_text = after_calc_bytes.decode("utf-8", errors="replace")
    diff_lines = list(
        difflib.unified_diff(
            _INITIAL_CALC.splitlines(keepends=True),
            after_calc_text.splitlines(keepends=True),
            fromfile="a/workspace/calculator.py",
            tofile="b/workspace/calculator.py",
        )
    )
    (target / "diff.patch").write_bytes("".join(diff_lines).encode("utf-8"))

    events = _read_jsonl(paths["provider_events"])
    usage_totals, cost_upper = _compute_usage_and_cost(events, guard)
    junit_summary = _extract_junit_summary(
        _extract_bound_junit_from_run_result(run_result, paths["output"])
    )
    live_rev = _live_workspace_revision(task, config)

    evidence_issues = validate_demo_evidence(target)
    if close_cancelled:
        evidence_issues.append("cancelled_during_client_close")
    elif close_failure is not None:
        evidence_issues.append(f"client_close_failed:{type(close_failure).__name__}")
    if was_cancelled:
        evidence_issues.append("session_cancelled")
    if guard.stopped and guard.stop_reason:
        evidence_issues.append(f"guard_stopped:{guard.stop_reason}")

    if was_cancelled or close_cancelled:
        terminal_status = "cancelled"
        reason_code = "CANCELLED"
    elif (
        run_result is not None
        and run_result.terminal_status == "completed"
        and close_failure is None
        and not guard.stopped
        and not evidence_issues
    ):
        terminal_status = "completed"
        reason_code = run_result.reason_code
    elif run_result is not None and run_result.terminal_status != "completed":
        terminal_status = run_result.terminal_status
        reason_code = run_result.reason_code
    else:
        terminal_status = "failed"
        reason_code = guard.stop_reason or failure_reason or "EVIDENCE_INCOMPLETE"

    verified = terminal_status == "completed" and not evidence_issues
    functional_passed = bool(
        run_result is not None
        and run_result.terminal_status == "completed"
        and junit_summary is not None
        and junit_summary.get("total") == 2
        and junit_summary.get("failures") == 0
        and junit_summary.get("errors") == 0
        and junit_summary.get("skipped") == 0
    )
    transport_attempts = sum(1 for ev in events if ev.get("sent"))
    summary: dict[str, object] = {
        "schema_version": "1",
        "mode": mode,
        "configured": True,
        "exercised": transport_attempts > 0,
        "verified": verified,
        "functional_passed": functional_passed,
        "workflow_evidence_satisfied": not evidence_issues,
        "terminal_status": terminal_status,
        "reason_code": reason_code,
        "failure_reason": failure_reason,
        "run_id": RUN_ID,
        "goal_id": GOAL_ID,
        "endpoint": ALLOWED_ENDPOINT,
        "requested_model": model,
        "response_models": [ev.get("model") for ev in events if ev.get("model")],
        "thinking": {"type": "disabled"},
        "reserved_call_slots": guard.slots_occupied,
        "transport_attempts": transport_attempts,
        "provider_calls": len(events),
        "usage": usage_totals,
        "reserved_cny": str(guard.reserved_cny),
        "cost_upper_bound_cny": cost_upper,
        "billed_cost_cny": None,
        "elapsed_seconds": round(time.monotonic() - started_at, 6),
        "initial_source_sha256": _INITIAL_CALC_SHA256,
        "final_source_sha256": _sha256_bytes(after_calc_bytes),
        "acceptance_test_sha256": _sha256_bytes(paths["acceptance_test"].read_bytes()),
        "acceptance_revision": (
            run_result.acceptance.revision
            if run_result is not None and run_result.acceptance is not None
            else live_rev
        ),
        "final_workspace_revision": live_rev,
        "changed_paths": (
            [entry.model_dump(mode="json") for entry in run_result.changed_paths]
            if run_result is not None
            else (
                [
                    {
                        "path": "calculator.py",
                        "operation": "modify",
                        "before_exists": True,
                        "after_exists": True,
                        "before_sha256": _INITIAL_CALC_SHA256,
                        "after_sha256": _sha256_bytes(after_calc_bytes),
                    }
                ]
                if _sha256_bytes(after_calc_bytes) != _INITIAL_CALC_SHA256
                else []
            )
        ),
        "junit": junit_summary,
        "steps": _extract_step_timeline(events),
        "evidence_issues": evidence_issues,
        "guard_stopped": guard.stopped,
        "guard_stop_reason": guard.stop_reason,
    }
    (target / "summary.json").write_bytes(_canonical_json(summary) + b"\n")

    if was_cancelled:
        raise asyncio.CancelledError()
    return summary, (0 if verified else 6)


def _validate_live_cli_args(args: argparse.Namespace) -> tuple[Decimal | None, str | None]:
    if args.model != ALLOWED_MODEL:
        return None, f"live mode requires --model {ALLOWED_MODEL}"
    if args.max_requests is None or args.max_requests < 1 or args.max_requests > MAX_REQUEST_SLOTS:
        return None, f"live mode requires --max-requests between 1 and {MAX_REQUEST_SLOTS}"
    if args.budget_cny is None:
        return None, f"live mode requires --budget-cny (<= {MAX_BUDGET_CNY})"
    try:
        budget = _parse_positive_decimal(args.budget_cny)
    except CodingApiDemoError as exc:
        return None, str(exc)
    if budget > MAX_BUDGET_CNY:
        return None, f"live budget_cny must not exceed {MAX_BUDGET_CNY}"
    return budget, None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``--mode prepare|live --root <dir>``."""
    parser = argparse.ArgumentParser(
        description="ForeContract COD-T02 Coding API demo (default: --mode prepare)."
    )
    parser.add_argument("--mode", choices=("prepare", "live"), default="prepare")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--budget-cny", type=str, default=None)
    args = parser.parse_args(argv)

    if args.mode == "prepare":
        try:
            paths = prepare_demo(args.root)
            target = Path(args.root).expanduser().resolve()
            task, _, _ = build_task(paths, python_executable=sys.executable)
            _write_initial_records(target, paths, task)
        except CodingApiDemoError as exc:
            print(json.dumps({"mode": "prepare", "status": "refused", "reason": str(exc)}))
            return 2
        doc = {
            "mode": "prepare",
            "status": "configured_not_run",
            "configured": True,
            "exercised": False,
            "verified": False,
            "root": str(target),
            "initial_source_sha256": _INITIAL_CALC_SHA256,
            "acceptance_test_sha256": _ACCEPTANCE_TEST_SHA256,
        }
        print(json.dumps(doc, ensure_ascii=False))
        return 0

    budget_cny, err_msg = _validate_live_cli_args(args)
    if err_msg is not None or budget_cny is None:
        print(
            json.dumps(
                {
                    "mode": "live",
                    "status": "refused",
                    "configured": False,
                    "exercised": False,
                    "verified": False,
                    "reason": err_msg,
                    "transport_attempts": 0,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 5

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        print(
            json.dumps(
                {
                    "mode": "live",
                    "status": "not_run",
                    "configured": False,
                    "exercised": False,
                    "verified": False,
                    "reason": "missing_local_credentials",
                    "transport_attempts": 0,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 4

    try:
        _validate_fresh_demo_root(args.root)
    except CodingApiDemoError as exc:
        print(
            json.dumps(
                {
                    "mode": "live",
                    "status": "refused",
                    "reason": str(exc),
                    "transport_attempts": 0,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2

    async def _run_live_session() -> tuple[dict[str, object], int]:
        client = make_client(api_key=api_key)
        return await run_demo_with_client(
            root=args.root,
            client=client,
            mode="live",
            model=args.model,
            max_requests=args.max_requests,
            budget_cny=budget_cny,
        )

    try:
        summary, exit_code = asyncio.run(_run_live_session())
    except CodingApiDemoError as exc:
        print(
            json.dumps(
                {
                    "mode": "live",
                    "status": "refused",
                    "reason": str(exc),
                    "transport_attempts": 0,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except asyncio.CancelledError:
        print(
            json.dumps(
                {
                    "mode": "live",
                    "status": "cancelled",
                    "verified": False,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 6

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
