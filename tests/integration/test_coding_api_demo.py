"""Offline L01-L12 integration tests for examples/coding_api_demo.py."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import coding_agent_demo
import coding_api_demo
import deepseek_provider
import httpx
import llm_api_smoke
import pytest
from coding_api_demo import (
    ALLOWED_ENDPOINT,
    ALLOWED_MODEL,
    MAX_OUTPUT_TOKENS,
    MAX_REQUEST_BYTES,
    MAX_REQUEST_SLOTS,
    MAX_RESPONSE_BYTES,
    PER_CALL_RESERVED_CNY,
    CodingApiDemoError,
    SessionBudgetGuard,
    build_task,
    main,
    prepare_demo,
    run_demo_with_client,
    run_session,
    validate_demo_evidence,
)
from forecontract_coding import CodingConfig, run_coding_task
from forecontract_coding.process import build_minimal_windows_env
from forecontract_coding.workspace import ManagedWorkspace
from forecontract_llm import ModelRequest, PlannerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

_FIXED_CALC = (
    '"""Synthetic calculator fixture for the live Coding demo."""\n\n\n'
    "def sum_even(values):\n"
    '    """Return the sum of the even integers in values."""\n'
    "    return sum(v for v in values if isinstance(v, int) and not isinstance(v, bool) and v % 2 == 0)\n"
)

_FROZEN_INITIAL_SHA256 = "e44957155b34ed10fdbd282aa7b0533ec9ef15bb0ceeea615aa9ec8eaba8e9e1"
_FROZEN_ACCEPTANCE_SHA256 = "20b0ace4367e9afb8f20bd85428a078633edb898589f98757d4faeb80626cb9c"

_FROZEN_EXISTING_SHA256: dict[str, str] = {
    "examples/coding_agent_demo.py": "afa5dd093ff2a4c4b46d367bd8ef12634650621638c37b35182aa9440a9db63d",
    "examples/deepseek_provider.py": "2904d82e1379be808c3b9f1c9e92d232f39bae73c29b7c941da0d06af33b0b4e",
    "examples/llm_api_smoke.py": "15e3148bc418390d7326ebc5b781b838b24e60fefca69df552c4933c55d1d4b3",
    "packages/adapters/coding/src/forecontract_coding/agent.py": "4d9463a8d0b9cd3edbd632bcdbe55d489009f80d9781e6197db8394f952c2867",
    "packages/adapters/coding/src/forecontract_coding/process.py": "f6058cdc239d2ecd24c603209ca1c5de3d9093c41012e38bf9ca4749b224f515",
    "packages/planners/llm/src/forecontract_llm/models.py": "00c3344fde253dcb369e0aac7a84d02646948d1bfbfc05b017643c07770f02ff",
    "artifacts/verification/P4-T03/round-021-suite/windows_job.py": "87ef2719f1404d5c8237898d3feaf530a3f54b26a89da66b4641f310b76c93ff",
    "docs/round-summaries/COD-T01/round-003.md": "63bab47a6b63c6c6066986041fecbc1eafab41f99f9837cdfbb29225d6d658d9",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line.decode("utf-8"))
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_bytes(payload.encode("utf-8"))


def _wrap_chat_completion(
    proposal: dict[str, Any],
    *,
    prompt_tokens: int = 180,
    completion_tokens: int = 45,
    model: str = "DeepSeek-V4.1-Flash-1017",
) -> httpx.Response:
    body = {
        "id": "cmpl-offline-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(proposal, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return httpx.Response(200, json=body)


def _action_proposal(
    action_type: str,
    payload: dict[str, Any],
    outcome_type: str,
    outcome_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "proposed_action": {
            "schema_version": "0.1",
            "action_type": action_type,
            "payload": payload,
            "metadata": {},
        },
        "expected_outcomes": [
            {
                "schema_version": "0.1",
                "outcome_type": outcome_type,
                "payload": outcome_payload,
                "required": True,
            }
        ],
    }


def _extract_model_state(request: httpx.Request) -> tuple[dict[str, Any], dict[str, Any]]:
    wire = json.loads(request.content.decode("utf-8"))
    model_req = json.loads(wire["messages"][1]["content"])
    state = (model_req.get("observation") or {}).get("state") or {}
    return wire, state


class _ScriptedCodingTransport:
    """Offline MockTransport driving real read -> pytest_unit(fail) -> write -> verify_task."""

    def __init__(self, *, skip_diagnostics: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.wire_bodies: list[dict[str, Any]] = []
        self.skip_diagnostics = skip_diagnostics

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        wire, state = _extract_model_state(request)
        self.wire_bodies.append(wire)

        last = state.get("last_tool_result")
        files = {
            entry["path"]: entry
            for entry in state.get("files", [])
            if isinstance(entry, dict) and "path" in entry
        }
        calc_sha = files["calculator.py"]["sha256"]

        if self.skip_diagnostics:
            if not last:
                return _wrap_chat_completion(
                    _action_proposal(
                        "write_file",
                        {
                            "path": "calculator.py",
                            "expected_sha256": calc_sha,
                            "content": _FIXED_CALC,
                        },
                        "coding_file_written",
                        {"path": "calculator.py"},
                    )
                )
            return _wrap_chat_completion(
                _action_proposal("verify_task", {}, "coding_task_verified", {})
            )

        if not last:
            return _wrap_chat_completion(
                _action_proposal(
                    "read_file",
                    {"path": "calculator.py", "start_line": 1, "max_lines": 50},
                    "coding_tool_result",
                    {"action_type": "read_file", "path": "calculator.py"},
                )
            )
        if last.get("action_type") == "read_file":
            return _wrap_chat_completion(
                _action_proposal(
                    "run_command",
                    {"command_id": "pytest_unit"},
                    "coding_tool_result",
                    {"action_type": "run_command", "command_id": "pytest_unit"},
                )
            )
        if last.get("action_type") == "run_command":
            return _wrap_chat_completion(
                _action_proposal(
                    "write_file",
                    {
                        "path": "calculator.py",
                        "expected_sha256": calc_sha,
                        "content": _FIXED_CALC,
                    },
                    "coding_file_written",
                    {"path": "calculator.py"},
                )
            )
        if last.get("action_type") == "write_file":
            return _wrap_chat_completion(
                _action_proposal("verify_task", {}, "coding_task_verified", {})
            )
        raise AssertionError(f"unexpected model turn after {last}")


def _make_offline_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=httpx.Timeout(10.0),
        follow_redirects=False,
        trust_env=False,
    )


def _forbid_credential_read(monkeypatch: pytest.MonkeyPatch) -> None:
    real_environ = os.environ

    class _NoKeyEnviron:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "DEEPSEEK_API_KEY":
                raise AssertionError("DEEPSEEK_API_KEY must not be read here")
            return real_environ.get(key, default)

        def __getitem__(self, key: str) -> str:
            if key == "DEEPSEEK_API_KEY":
                raise AssertionError("DEEPSEEK_API_KEY must not be read here")
            return real_environ[key]

        def __contains__(self, key: object) -> bool:
            if key == "DEEPSEEK_API_KEY":
                raise AssertionError("DEEPSEEK_API_KEY must not be read here")
            return key in real_environ

    class _OsProxy:
        environ = _NoKeyEnviron()

        def __getattr__(self, name: str) -> Any:
            return getattr(os, name)

    monkeypatch.setattr(coding_api_demo, "os", _OsProxy())


def _parse_raw_record(run_root: Path) -> dict[str, Any]:
    before_doc = json.loads((run_root / "before.json").read_bytes().decode("utf-8"))
    task_doc = json.loads((run_root / "task.json").read_bytes().decode("utf-8"))
    trace_rows = _read_jsonl(run_root / "trace.jsonl")
    events = _read_jsonl(run_root / "provider-events.jsonl")

    proposals: list[dict[str, Any]] = []
    req_payloads: list[dict[str, Any]] = []
    for ev in events:
        wire = ev["request"]
        req_payloads.append(json.loads(wire["messages"][1]["content"]))
        proposals.append(json.loads(ev["response"]["choices"][0]["message"]["content"]))

    read_idx: int | None = None
    first_write_idx: int | None = None
    write_payload_sha: str | None = None
    failed_pytest_code: int | None = None
    failed_present_in_next_req = False
    verify_idx: int | None = None

    for idx, prop in enumerate(proposals):
        act = prop["proposed_action"]
        atype = act["action_type"]
        payload = act["payload"]
        if atype == "read_file" and payload.get("path") == "calculator.py" and read_idx is None:
            read_idx = idx
        elif atype == "write_file" and payload.get("path") == "calculator.py":
            if first_write_idx is None:
                first_write_idx = idx
            write_payload_sha = _sha256_bytes(payload["content"].encode("utf-8"))
        elif atype == "run_command" and payload.get("command_id") == "pytest_unit":
            if idx + 1 < len(req_payloads):
                last_res = (
                    (req_payloads[idx + 1].get("observation") or {}).get("state") or {}
                ).get("last_tool_result") or {}
                if (
                    last_res.get("action_type") == "run_command"
                    and last_res.get("command_id") == "pytest_unit"
                ):
                    failed_pytest_code = last_res.get("exit_code")
                    failed_present_in_next_req = failed_pytest_code == 1
        elif atype == "verify_task":
            verify_idx = idx

    junit_files = sorted((run_root / "output" / "verify").glob("*/junit.xml"))
    latest_junit = max(junit_files, key=lambda p: (p.stat().st_mtime_ns, p.parent.name))
    tree = ET.fromstring(latest_junit.read_bytes())
    suites = [tree] if tree.tag == "testsuite" else list(tree.findall(".//testsuite"))
    cases = [tc for s in suites for tc in s.findall("testcase")]
    junit_ids = {(tc.attrib["classname"], tc.attrib["name"]) for tc in cases}

    from forecontract_coding import CodingTask

    task = CodingTask.model_validate(task_doc)
    cfg = CodingConfig(schema_version="0.1", output_root=str(run_root / "output"))
    ws = ManagedWorkspace(task=task, config=cfg)
    inventory = ws.capture_scope_baseline()
    _, snaps, _ = ws.snapshot()
    live_rev = ws.compute_full_revision(snapshots=snaps, inventory=inventory, scope_ok=True)

    verify_rev = None
    if verify_idx is not None:
        verify_rev = (
            (req_payloads[verify_idx].get("observation") or {}).get("state") or {}
        ).get("workspace_revision")

    last_trace = trace_rows[-1]
    terminal_map = {"success": "completed"}

    changed_paths = [
        p.relative_to(run_root / "workspace").as_posix()
        for p in sorted((run_root / "workspace").rglob("*"))
        if p.is_file()
    ]

    return {
        "initial_source_sha256": before_doc["initial_source_sha256"],
        "acceptance_source_sha256_before": before_doc["acceptance_test_sha256"],
        "acceptance_source_sha256_after": _sha256_bytes(
            (run_root / "external_tests" / "test_acceptance.py").read_bytes()
        ),
        "actual_changed_paths": changed_paths,
        "has_real_read_before_first_write": (
            read_idx is not None and first_write_idx is not None and read_idx < first_write_idx
        ),
        "failed_pytest_exit_code": failed_pytest_code,
        "failed_test_result_present_in_next_model_request": failed_present_in_next_req,
        "written_content_sha256": _sha256_bytes(
            (run_root / "workspace" / "calculator.py").read_bytes()
        ),
        "model_write_payload_sha256": write_payload_sha,
        "junit_case_ids": junit_ids,
        "junit_total": len(cases),
        "junit_failures": sum(1 for tc in cases if tc.find("failure") is not None),
        "junit_errors": sum(1 for tc in cases if tc.find("error") is not None),
        "junit_skips": sum(1 for tc in cases if tc.find("skipped") is not None),
        "acceptance_revision": verify_rev,
        "final_workspace_revision": live_rev,
        "terminal_status": terminal_map.get(
            last_trace.get("terminal_status", ""), last_trace.get("terminal_status")
        ),
        "transport_attempts": sum(1 for ev in events if ev.get("sent")),
        "provider_calls": len(events),
    }


def _dummy_model_request(*, state_extra: dict[str, Any] | None = None) -> ModelRequest:
    state: dict[str, Any] = {"dummy": True}
    if state_extra:
        state.update(state_extra)
    return ModelRequest.model_validate(
        {
            "schema_version": "0.1",
            "request_id": "req-offline-dummy-1",
            "model_context": {
                "schema_version": "0.1",
                "run_id": "cod-t02-live-001",
                "goal_id": "sum-even-fix",
                "goal_text": "fix sum_even",
                "task_criteria": {
                    "schema_version": "0.1",
                    "goal_id": "sum-even-fix",
                    "conditions": [
                        {
                            "schema_version": "0.1",
                            "outcome_type": "coding_task_verified",
                            "payload": {},
                            "required": True,
                        }
                    ],
                },
                "capabilities": {
                    "schema_version": "0.1",
                    "adapter_name": "coding_workspace",
                    "environment_type": "coding",
                    "supported_actions": ["read_file"],
                    "supported_observation_types": ["coding_state"],
                    "supported_condition_types": ["coding_scope_valid"],
                    "supported_outcome_types": ["coding_task_verified"],
                    "metadata": {},
                },
                "action_guidance": {"read_file": {"type": "object"}},
                "outcome_guidance": {"coding_task_verified": {"type": "object"}},
            },
            "observation": {
                "schema_version": "0.1",
                "observation_id": "obs-offline-1",
                "environment_type": "coding",
                "state": state,
                "timestamp": "2026-11-13T00:00:00Z",
                "summary": None,
                "artifacts": [],
                "metadata": {"state_type": "coding_state"},
            },
            "response_schema": {"type": "object"},
            "max_output_tokens": 2048,
        }
    )


def test_l01_cli_no_implicit_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fail_make_client(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("make_client must not be called in prepare/invalid/missing-key paths")

    monkeypatch.setattr(coding_api_demo, "make_client", _fail_make_client)
    _forbid_credential_read(monkeypatch)

    # Default (no --mode) -> prepare only, 0 HTTP, 0 key read
    prep_root = tmp_path / "prep_default"
    code = main(["--root", str(prep_root)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "prepare"
    assert out["status"] == "configured_not_run"
    assert out["exercised"] is False
    assert out["verified"] is False
    assert (prep_root / "workspace" / "calculator.py").is_file()

    # Invalid --mode live options rejected BEFORE credential read
    for bad_argv in (
        ["--mode", "live", "--root", str(tmp_path / "bad_model"), "--model", "deepseek-chat", "--max-requests", "8", "--budget-cny", "1"],
        ["--mode", "live", "--root", str(tmp_path / "bad_req"), "--model", "deepseek-flash", "--max-requests", "9", "--budget-cny", "1"],
        ["--mode", "live", "--root", str(tmp_path / "bad_req0"), "--model", "deepseek-flash", "--max-requests", "0", "--budget-cny", "1"],
        ["--mode", "live", "--root", str(tmp_path / "bad_budget"), "--model", "deepseek-flash", "--max-requests", "8", "--budget-cny", "1.01"],
        ["--mode", "live", "--root", str(tmp_path / "bad_budget0"), "--model", "deepseek-flash", "--max-requests", "8", "--budget-cny", "0"],
        ["--mode", "live", "--root", str(tmp_path / "bad_nan"), "--model", "deepseek-flash", "--max-requests", "8", "--budget-cny", "NaN"],
        ["--mode", "live", "--root", str(tmp_path / "bad_inf"), "--model", "deepseek-flash", "--max-requests", "8", "--budget-cny", "Infinity"],
    ):
        assert main(bad_argv) == 5
        err = json.loads(capsys.readouterr().err)
        assert err["status"] == "refused"
        assert err["transport_attempts"] == 0

    # Valid --mode live args but missing key -> returns not_run (exit 4), 0 HTTP
    monkeypatch.undo()
    monkeypatch.setattr(coding_api_demo, "make_client", _fail_make_client)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    missing_key_root = tmp_path / "live_missing_key"
    code_no_key = main(
        [
            "--mode",
            "live",
            "--root",
            str(missing_key_root),
            "--model",
            "deepseek-flash",
            "--max-requests",
            "8",
            "--budget-cny",
            "1",
        ]
    )
    assert code_no_key == 4
    err_no_key = json.loads(capsys.readouterr().err)
    assert err_no_key["status"] == "not_run"
    assert err_no_key["exercised"] is False
    assert err_no_key["transport_attempts"] == 0
    assert not missing_key_root.exists()


def test_l02_fixed_scope_and_fresh_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "existing_root"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel_bytes = b"keep-existing-root-bytes\n"
    sentinel.write_bytes(sentinel_bytes)

    with pytest.raises(CodingApiDemoError):
        prepare_demo(existing)
    assert sentinel.read_bytes() == sentinel_bytes

    assert main(["--mode", "prepare", "--root", str(existing)]) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["status"] == "refused"
    assert sentinel.read_bytes() == sentinel_bytes

    fresh = tmp_path / "fresh_root"
    paths = prepare_demo(fresh)
    assert set(paths.keys()) == {
        "workspace",
        "acceptance_test",
        "output",
        "trace",
        "provider_events",
    }
    assert _sha256_bytes((paths["workspace"] / "calculator.py").read_bytes()) == _FROZEN_INITIAL_SHA256
    assert _sha256_bytes(paths["acceptance_test"].read_bytes()) == _FROZEN_ACCEPTANCE_SHA256
    assert paths["trace"].read_bytes() == b""
    assert paths["provider_events"].read_bytes() == b""

    task, cfg, pcfg = build_task(paths, python_executable=sys.executable)
    assert task.run_id == "cod-t02-live-001"
    assert task.goal_id == "sum-even-fix"
    assert task.read_paths == ["calculator.py"]
    assert [(w.path, w.operation) for w in task.write_paths] == [("calculator.py", "modify")]
    assert list(task.commands.keys()) == ["pytest_unit"]
    assert task.verification_command_ids == ["pytest_unit"]
    assert task.max_steps == 8
    assert task.max_planning_calls == 8
    assert task.max_replans == 2
    assert pcfg.max_calls == 8
    assert pcfg.max_request_bytes == MAX_REQUEST_BYTES
    assert pcfg.max_response_bytes == MAX_RESPONSE_BYTES
    assert pcfg.max_output_tokens == MAX_OUTPUT_TOKENS
    assert Path(cfg.output_root) == paths["output"]


def test_l03_native_transport_feedback_loop(tmp_path: Path) -> None:
    run_root = tmp_path / "l03_loop"
    transport = _ScriptedCodingTransport()
    summary, exit_code = asyncio.run(
        run_demo_with_client(
            root=run_root,
            client=_make_offline_client(transport),
            mode="offline_test",
        )
    )
    assert exit_code == 0
    assert summary["verified"] is True

    record = _parse_raw_record(run_root)
    assert record["initial_source_sha256"] == _FROZEN_INITIAL_SHA256
    assert record["acceptance_source_sha256_before"] == _FROZEN_ACCEPTANCE_SHA256
    assert record["acceptance_source_sha256_after"] == _FROZEN_ACCEPTANCE_SHA256
    assert record["actual_changed_paths"] == ["calculator.py"]
    assert record["has_real_read_before_first_write"] is True
    assert record["failed_pytest_exit_code"] == 1
    assert record["failed_test_result_present_in_next_model_request"] is True
    assert record["written_content_sha256"] == record["model_write_payload_sha256"]
    assert exit_code == 0
    assert record["junit_case_ids"] == {
        ("test_acceptance", "test_basic_even"),
        ("test_acceptance", "test_empty_and_negatives"),
    }
    assert record["junit_total"] == 2
    assert record["junit_failures"] == 0
    assert record["junit_errors"] == 0
    assert record["junit_skips"] == 0
    assert record["acceptance_revision"] == record["final_workspace_revision"]
    assert record["terminal_status"] == "completed"
    assert 0 < record["transport_attempts"] <= int(summary["reserved_call_slots"]) <= MAX_REQUEST_SLOTS
    assert Decimal(str(summary["reserved_cny"])) <= Decimal(1)
    assert Decimal(str(summary["cost_upper_bound_cny"])) <= Decimal(1)
    assert validate_demo_evidence(run_root) == []


@pytest.mark.parametrize("cap", [1, 8])
def test_l04_request_cap_and_reservation(tmp_path: Path, cap: int) -> None:
    class _OkUsage:
        input_tokens = 100
        output_tokens = 20

    class _OkResp:
        status = "ok"
        usage = _OkUsage()

    guard = SessionBudgetGuard(max_requests=cap, budget_cny=Decimal(1))
    for _ in range(cap):
        guard.before_send()
        guard.after_response(_OkResp())

    assert guard.slots_occupied == cap
    expected_reserved = PER_CALL_RESERVED_CNY * cap
    assert guard.reserved_cny == expected_reserved
    assert expected_reserved <= Decimal(1)

    with pytest.raises(CodingApiDemoError, match="request_cap_reached"):
        guard.before_send()
    assert guard.stopped is True
    assert guard.slots_occupied == cap

    # Calling again after latch still does zero sends and never increments slots
    with pytest.raises(CodingApiDemoError):
        guard.before_send()
    assert guard.slots_occupied == cap
    assert guard.reserved_cny == expected_reserved

    # Budget < 0.090112 refuses on first send with zero slots occupied
    tight_guard = SessionBudgetGuard(max_requests=8, budget_cny=Decimal("0.090111"))
    with pytest.raises(CodingApiDemoError, match="budget_insufficient"):
        tight_guard.before_send()
    assert tight_guard.slots_occupied == 0
    assert tight_guard.reserved_cny == Decimal(0)
    assert tight_guard.stopped is True

    # Wired through run_demo_with_client: insufficient budget produces 0 HTTP requests
    zero_root = tmp_path / f"zero_budget_{cap}"
    zero_transport = _ScriptedCodingTransport()
    summary_zero, code_zero = asyncio.run(
        run_demo_with_client(
            root=zero_root,
            client=_make_offline_client(zero_transport),
            mode="offline_test",
            budget_cny=Decimal("0.05"),
        )
    )
    assert code_zero != 0
    assert len(zero_transport.requests) == 0
    assert summary_zero["reserved_call_slots"] == 0
    assert summary_zero["transport_attempts"] == 0
    assert summary_zero["verified"] is False

    # Wired with max_requests=1 stops after exactly 1 HTTP request
    one_root = tmp_path / f"one_cap_{cap}"
    one_transport = _ScriptedCodingTransport()
    summary_one, code_one = asyncio.run(
        run_demo_with_client(
            root=one_root,
            client=_make_offline_client(one_transport),
            mode="offline_test",
            max_requests=1,
        )
    )
    assert code_one != 0
    assert len(one_transport.requests) == 1
    assert summary_one["reserved_call_slots"] == 1
    assert summary_one["transport_attempts"] == 1
    assert summary_one["guard_stopped"] is True


@pytest.mark.parametrize(
    "scenario",
    [
        "http_500",
        "http_429",
        "timeout",
        "missing_usage",
        "invalid_usage",
        "usage_over_reservation",
        "output_over_cap",
    ],
)
def test_l05_unknown_usage_and_failure_stop(tmp_path: Path, scenario: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if scenario == "http_500":
            return httpx.Response(500, json={"error": {"message": "server error"}})
        if scenario == "http_429":
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        if scenario == "timeout":
            raise httpx.ReadTimeout("simulated read timeout", request=request)
        if scenario == "missing_usage":
            resp = _wrap_chat_completion(
                _action_proposal("read_file", {"path": "calculator.py"}, "coding_file_read", {"path": "calculator.py"})
            )
            data = json.loads(resp.content)
            data.pop("usage")
            return httpx.Response(200, json=data)
        if scenario == "invalid_usage":
            return _wrap_chat_completion(
                _action_proposal("read_file", {"path": "calculator.py"}, "coding_file_read", {"path": "calculator.py"}),
                prompt_tokens=-5,
                completion_tokens=10,
            )
        if scenario == "usage_over_reservation":
            return _wrap_chat_completion(
                _action_proposal("read_file", {"path": "calculator.py"}, "coding_file_read", {"path": "calculator.py"}),
                prompt_tokens=100_000_000,
                completion_tokens=10,
            )
        if scenario == "output_over_cap":
            return _wrap_chat_completion(
                _action_proposal("read_file", {"path": "calculator.py"}, "coding_file_read", {"path": "calculator.py"}),
                prompt_tokens=100,
                completion_tokens=MAX_OUTPUT_TOKENS + 1,
            )
        raise AssertionError(scenario)

    events_path = tmp_path / f"events_{scenario}.jsonl"
    guard = SessionBudgetGuard(max_requests=8, budget_cny=Decimal(1))
    provider = coding_api_demo.DeepSeekProvider(
        client=_make_offline_client(handler),
        model=ALLOWED_MODEL,
        max_request_bytes=MAX_REQUEST_BYTES,
        max_response_bytes=MAX_RESPONSE_BYTES,
        timeout_seconds=10.0,
        max_output_tokens_cap=MAX_OUTPUT_TOKENS,
        remaining_seconds=guard.remaining,
    )
    recorder = coding_api_demo._ProviderRecorder(
        provider,
        events_path,
        mode="offline_test",
        case_id=scenario,
        guard=guard,
    )

    with pytest.raises(CodingApiDemoError):
        asyncio.run(recorder.complete(_dummy_model_request()))

    assert calls == 1
    assert guard.stopped is True
    assert guard.cost_upper_bound_cny is None
    assert guard.reserved_cny == PER_CALL_RESERVED_CNY
    assert len(_read_jsonl(events_path)) == 1

    # Second attempt is blocked cleanly in before_send with 0 additional transport calls
    with pytest.raises(CodingApiDemoError):
        asyncio.run(recorder.complete(_dummy_model_request()))
    assert calls == 1
    assert len(_read_jsonl(events_path)) == 1


def test_l06_wire_limits_and_identity(tmp_path: Path) -> None:
    run_root = tmp_path / "l06_wire"
    transport = _ScriptedCodingTransport()
    summary, exit_code = asyncio.run(
        run_demo_with_client(
            root=run_root,
            client=_make_offline_client(transport),
            mode="offline_test",
        )
    )
    assert exit_code == 0
    assert summary["run_id"] == "cod-t02-live-001"
    assert summary["goal_id"] == "sum-even-fix"
    assert len(transport.requests) >= 4

    for req, wire in zip(transport.requests, transport.wire_bodies, strict=True):
        assert str(req.url) == ALLOWED_ENDPOINT
        assert req.method == "POST"
        assert len(req.content) <= MAX_REQUEST_BYTES
        assert wire["model"] == ALLOWED_MODEL
        assert wire["thinking"] == {"type": "disabled"}
        assert wire["stream"] is False
        assert wire["response_format"] == {"type": "json_object"}
        assert wire["max_tokens"] == MAX_OUTPUT_TOKENS
        model_req = ModelRequest.model_validate(json.loads(wire["messages"][1]["content"]))
        assert model_req.model_context.run_id == "cod-t02-live-001"
        assert model_req.model_context.goal_id == "sum-even-fix"

    # Oversize request (> 32768) refuses locally with 0 HTTP calls
    zero_calls = 0

    def _counting_handler(_req: httpx.Request) -> httpx.Response:
        nonlocal zero_calls
        zero_calls += 1
        return _wrap_chat_completion(
            _action_proposal(
                "read_file",
                {"path": "calculator.py", "start_line": 1, "max_lines": 50},
                "coding_tool_result",
                {"action_type": "read_file", "path": "calculator.py"},
            )
        )

    provider = coding_api_demo.DeepSeekProvider(
        client=_make_offline_client(_counting_handler),
        model=ALLOWED_MODEL,
        max_request_bytes=MAX_REQUEST_BYTES,
        max_response_bytes=MAX_RESPONSE_BYTES,
        timeout_seconds=10.0,
        max_output_tokens_cap=MAX_OUTPUT_TOKENS,
    )
    huge_req = _dummy_model_request(state_extra={"blob": "x" * (MAX_REQUEST_BYTES + 1024)})
    res_huge = asyncio.run(provider.complete(huge_req))
    assert res_huge.status == "refused"
    assert zero_calls == 0

    # Oversize response (> 65536) rejects boundedly with no retry
    big_calls = 0

    def _oversize_resp(_req: httpx.Request) -> httpx.Response:
        nonlocal big_calls
        big_calls += 1
        return httpx.Response(200, content=b"{" + b" " * (MAX_RESPONSE_BYTES + 512) + b"}")

    provider_big = coding_api_demo.DeepSeekProvider(
        client=_make_offline_client(_oversize_resp),
        model=ALLOWED_MODEL,
        max_request_bytes=MAX_REQUEST_BYTES,
        max_response_bytes=MAX_RESPONSE_BYTES,
        timeout_seconds=10.0,
        max_output_tokens_cap=MAX_OUTPUT_TOKENS,
    )
    res_big = asyncio.run(provider_big.complete(_dummy_model_request()))
    assert res_big.status == "refused"
    assert big_calls == 1

    # 302 redirect is never followed (follow_redirects=False)
    redir_calls = 0

    def _redirect_resp(_req: httpx.Request) -> httpx.Response:
        nonlocal redir_calls
        redir_calls += 1
        return httpx.Response(302, headers={"Location": "https://example.invalid/redirected"})

    provider_redir = coding_api_demo.DeepSeekProvider(
        client=_make_offline_client(_redirect_resp),
        model=ALLOWED_MODEL,
        max_request_bytes=MAX_REQUEST_BYTES,
        max_response_bytes=MAX_RESPONSE_BYTES,
        timeout_seconds=10.0,
        max_output_tokens_cap=MAX_OUTPUT_TOKENS,
    )
    res_redir = asyncio.run(provider_redir.complete(_dummy_model_request()))
    assert res_redir.status == "refused"
    assert provider_redir.last_http_status == 302
    assert redir_calls == 1


def test_l07_secrets_and_child_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_secret = "sk-fake-l07-sentinel-key-987654321"
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_secret)

    seen_auth_headers: list[str] = []
    inner_transport = _ScriptedCodingTransport()

    def _inspecting_handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        seen_auth_headers.append(auth)
        return inner_transport(request)

    def _fake_make_client(*, api_key: str) -> httpx.AsyncClient:
        assert api_key == fake_secret
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_inspecting_handler),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(10.0),
            follow_redirects=False,
            trust_env=False,
        )

    monkeypatch.setattr(coding_api_demo, "make_client", _fake_make_client)

    live_root = tmp_path / "l07_live_mocked"
    code = main(
        [
            "--mode",
            "live",
            "--root",
            str(live_root),
            "--model",
            "deepseek-flash",
            "--max-requests",
            "8",
            "--budget-cny",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert seen_auth_headers and all(h == f"Bearer {fake_secret}" for h in seen_auth_headers)
    assert fake_secret not in captured.out
    assert fake_secret not in captured.err
    assert "Authorization" not in captured.out

    # Scan every artifact file written under live_root for secret / auth leakage
    for path in live_root.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            assert fake_secret.encode("utf-8") not in raw, f"secret leaked into {path}"
            assert b"Authorization" not in raw, f"Authorization header leaked into {path}"

    # Verify child pytest env built by existing process.py never inherits DEEPSEEK_API_KEY
    from forecontract_coding import CodingTask

    task = CodingTask.model_validate(json.loads((live_root / "task.json").read_bytes()))
    child_env = build_minimal_windows_env(
        spec=task.commands["pytest_unit"],
        temp_dir=tmp_path / "child_tmp",
    )
    assert "DEEPSEEK_API_KEY" not in child_env
    assert set(task.commands["pytest_unit"].env.keys()) == {
        "PYTHONPATH",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }

    # Failure path with secret present also must not leak key, headers, or client repr
    def _fail_make_client(*, api_key: str) -> httpx.AsyncClient:
        assert api_key == fake_secret
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(500, json={"error": {"message": "upstream failure"}})
            ),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(10.0),
            follow_redirects=False,
            trust_env=False,
        )

    monkeypatch.setattr(coding_api_demo, "make_client", _fail_make_client)
    fail_root = tmp_path / "l07_fail_mocked"
    fail_code = main(
        [
            "--mode",
            "live",
            "--root",
            str(fail_root),
            "--model",
            "deepseek-flash",
            "--max-requests",
            "8",
            "--budget-cny",
            "1",
        ]
    )
    fail_cap = capsys.readouterr()
    assert fail_code != 0
    assert fake_secret not in fail_cap.out
    assert fake_secret not in fail_cap.err
    for path in fail_root.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            assert fake_secret.encode("utf-8") not in raw


def test_l08_deadline_cancel_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_api_smoke, "CLOSE_TIMEOUT_SECONDS", 0.15)
    close_entered = asyncio.Event()

    class _HangingCloseTransport(httpx.AsyncBaseTransport):
        def __init__(self, started_evt: asyncio.Event) -> None:
            self._started = started_evt

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self._started.set()
            await asyncio.sleep(5.0)
            return httpx.Response(200, json={})

        async def aclose(self) -> None:
            close_entered.set()
            await asyncio.sleep(5.0)

    async def _run_cancel_case() -> tuple[Path, list[asyncio.Task[Any]]]:
        started = asyncio.Event()
        cancel_root = tmp_path / "l08_cancelled"
        client = httpx.AsyncClient(
            transport=_HangingCloseTransport(started),
            timeout=httpx.Timeout(10.0),
            follow_redirects=False,
            trust_env=False,
        )
        task = asyncio.create_task(
            run_demo_with_client(
                root=cancel_root,
                client=client,
                mode="offline_test",
                session_deadline_seconds=30.0,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=3.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        return cancel_root, [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]

    cancel_root, dangling = asyncio.run(_run_cancel_case())
    assert close_entered.is_set()
    assert not dangling
    summary = json.loads((cancel_root / "summary.json").read_bytes().decode("utf-8"))
    assert summary["terminal_status"] == "cancelled"
    assert summary["verified"] is False
    assert (cancel_root / "diff.patch").is_file()

    # Short session deadline aborts cleanly as failed/refused without second transport send
    deadline_calls = 0

    def _slow_sync_handler(_req: httpx.Request) -> httpx.Response:
        nonlocal deadline_calls
        deadline_calls += 1
        return _wrap_chat_completion(
            _action_proposal("read_file", {"path": "calculator.py"}, "coding_file_read", {"path": "calculator.py"})
        )

    deadline_root = tmp_path / "l08_deadline"
    summary_dl, code_dl = asyncio.run(
        run_demo_with_client(
            root=deadline_root,
            client=_make_offline_client(_slow_sync_handler),
            mode="offline_test",
            session_deadline_seconds=0.05,
        )
    )
    assert code_dl != 0
    assert summary_dl["verified"] is False
    assert summary_dl["terminal_status"] != "completed"
    assert deadline_calls <= 1


def test_l09_evidence_before_summary(tmp_path: Path) -> None:
    good_root = tmp_path / "l09_good"
    summary, code = asyncio.run(
        run_demo_with_client(
            root=good_root,
            client=_make_offline_client(_ScriptedCodingTransport()),
            mode="offline_test",
        )
    )
    assert code == 0

    events = _read_jsonl(good_root / "provider-events.jsonl")
    trace_rows = _read_jsonl(good_root / "trace.jsonl")
    disk_summary = json.loads((good_root / "summary.json").read_bytes().decode("utf-8"))
    assert disk_summary == summary
    assert events and trace_rows
    assert disk_summary["provider_calls"] == len(events)
    assert disk_summary["transport_attempts"] == sum(1 for e in events if e.get("sent"))

    total_in = sum(e["response"]["usage"]["prompt_tokens"] for e in events)
    total_out = sum(e["response"]["usage"]["completion_tokens"] for e in events)
    assert disk_summary["usage"] == {"input_tokens": total_in, "output_tokens": total_out}
    assert disk_summary["final_source_sha256"] == _sha256_bytes(
        (good_root / "workspace" / "calculator.py").read_bytes()
    )
    diff_text = (good_root / "diff.patch").read_text(encoding="utf-8")
    assert "def sum_even(values):" not in diff_text or "-    return sum(values)" in diff_text
    assert "+    return sum(v for v in values" in diff_text

    # Failure run also writes raw provider/trace/diff BEFORE summary with non-completed status
    fail_root = tmp_path / "l09_fail"
    summary_fail, fail_code = asyncio.run(
        run_demo_with_client(
            root=fail_root,
            client=_make_offline_client(
                lambda _req: httpx.Response(500, json={"error": {"message": "boom"}})
            ),
            mode="offline_test",
        )
    )
    assert fail_code != 0
    assert (fail_root / "provider-events.jsonl").is_file()
    assert (fail_root / "trace.jsonl").is_file()
    assert (fail_root / "diff.patch").read_text(encoding="utf-8") == ""
    assert summary_fail["terminal_status"] != "completed"
    assert summary_fail["verified"] is False
    assert summary_fail["cost_upper_bound_cny"] is None


def test_l10_live_evidence_reader(tmp_path: Path) -> None:
    baseline_root = tmp_path / "l10_baseline"
    summary, code = asyncio.run(
        run_demo_with_client(
            root=baseline_root,
            client=_make_offline_client(_ScriptedCodingTransport()),
            mode="offline_test",
        )
    )
    assert code == 0
    assert summary["verified"] is True
    assert validate_demo_evidence(baseline_root) == []

    def _assert_mutated_then_restored(
        target_file: Path,
        mutate_bytes: Callable[[bytes], bytes],
        expected_substr: str,
    ) -> None:
        assert validate_demo_evidence(baseline_root) == []
        original = target_file.read_bytes()
        try:
            target_file.write_bytes(mutate_bytes(original))
            reasons = validate_demo_evidence(baseline_root)
            assert reasons
            assert any(expected_substr in reason for reason in reasons), reasons
        finally:
            target_file.write_bytes(original)
        assert validate_demo_evidence(baseline_root) == []

    events_path = baseline_root / "provider-events.jsonl"
    bound_junit = _locate_bound_junit(baseline_root)
    acc_test_path = baseline_root / "external_tests" / "test_acceptance.py"
    summary_path = baseline_root / "summary.json"
    trace_path = baseline_root / "trace.jsonl"

    def _mutate_jsonl(raw: bytes, transform: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> bytes:
        rows = [json.loads(line.decode("utf-8")) for line in raw.splitlines() if line.strip()]
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in transform(rows)).encode("utf-8")

    # 1. Missing read
    _assert_mutated_then_restored(
        events_path,
        lambda raw: _mutate_jsonl(raw, lambda evs: evs[1:]),
        "missing:real_read_file",
    )

    # 2. Missing failed pytest feedback before final fix
    def _drop_fail_fb(evs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        req_inner = json.loads(evs[2]["request"]["messages"][1]["content"])
        req_inner["observation"]["state"]["last_tool_result"]["exit_code"] = 0
        evs[2]["request"]["messages"][1]["content"] = json.dumps(req_inner)
        return evs

    _assert_mutated_then_restored(
        events_path,
        lambda raw: _mutate_jsonl(raw, _drop_fail_fb),
        "missing:failed_pytest_feedback",
    )

    # 3. Missing write
    _assert_mutated_then_restored(
        events_path,
        lambda raw: _mutate_jsonl(raw, lambda evs: [e for i, e in enumerate(evs) if i != 2]),
        "missing:real_write_file",
    )

    # 4. Missing verify
    _assert_mutated_then_restored(
        events_path,
        lambda raw: _mutate_jsonl(raw, lambda evs: evs[:-1]),
        "missing:verify_task",
    )

    # 5a. Omitted test in bound junit.xml
    _assert_mutated_then_restored(
        bound_junit,
        lambda _: (
            b'<testsuite tests="1" failures="0" errors="0" skipped="0">'
            b'<testcase classname="test_acceptance" name="test_basic_even"/>'
            b"</testsuite>"
        ),
        "invalid:junit_cases",
    )

    # 5b. Skipped test in bound junit.xml
    _assert_mutated_then_restored(
        bound_junit,
        lambda _: (
            b'<testsuite tests="2" failures="0" errors="0" skipped="1">'
            b'<testcase classname="test_acceptance" name="test_basic_even"/>'
            b'<testcase classname="test_acceptance" name="test_empty_and_negatives"><skipped/></testcase>'
            b"</testsuite>"
        ),
        "invalid:junit_case_not_passed",
    )

    # 6. Acceptance source SHA tampered
    _assert_mutated_then_restored(
        acc_test_path,
        lambda _: b"# tampered\n",
        "acceptance_source_sha_changed",
    )

    # 7a. Wrong revision in output summary
    def _bad_summary_rev(raw: bytes) -> bytes:
        doc = json.loads(raw.decode("utf-8"))
        doc["acceptance_revision"] = "deadbeef"
        return json.dumps(doc).encode("utf-8")

    _assert_mutated_then_restored(
        summary_path,
        _bad_summary_rev,
        "summary_acceptance_revision_mismatch",
    )

    # 7b. Wrong revision in verify request payload
    def _bad_verify_req_rev(evs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        v_req = json.loads(evs[-1]["request"]["messages"][1]["content"])
        v_req["observation"]["state"]["workspace_revision"] = "badrev"
        evs[-1]["request"]["messages"][1]["content"] = json.dumps(v_req)
        return evs

    _assert_mutated_then_restored(
        events_path,
        lambda raw: _mutate_jsonl(raw, _bad_verify_req_rev),
        "workspace_revision_mismatch_after_verify",
    )

    # 8. Fake completed in summary.json while raw trace is failure
    def _bad_trace_status(trows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trows[-1]["terminal_status"] = "failure"
        return trows

    _assert_mutated_then_restored(
        trace_path,
        lambda raw: _mutate_jsonl(raw, _bad_trace_status),
        "trace_terminal_not_success",
    )


def test_l11_existing_components_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert coding_api_demo.DeepSeekProvider is deepseek_provider.DeepSeekProvider
    assert coding_api_demo._ProviderRecorder is llm_api_smoke._ProviderRecorder
    assert coding_api_demo._bounded_close is llm_api_smoke._bounded_close
    assert coding_api_demo.ALLOWED_ENDPOINT == deepseek_provider.ENDPOINT
    assert coding_api_demo._ACCEPTANCE_TEST == coding_agent_demo._ACCEPTANCE_TEST

    for rel, expected_sha in _FROZEN_EXISTING_SHA256.items():
        actual_sha = _sha256_bytes((REPO_ROOT / rel).read_bytes())
        assert actual_sha == expected_sha, f"frozen file drifted: {rel}"

    # Verify run_session delegates directly to existing forecontract_coding.run_coding_task
    called: list[bool] = []

    async def _spy_run_coding_task(**kwargs: Any) -> Any:
        called.append(True)
        return await run_coding_task(**kwargs)

    monkeypatch.setattr(coding_api_demo, "run_coding_task", _spy_run_coding_task)
    paths = prepare_demo(tmp_path / "l11_delegate")
    task, cfg, pcfg = build_task(paths, python_executable=sys.executable)
    res = asyncio.run(
        run_session(
            task=task,
            config=cfg,
            provider=deepseek_provider.DeepSeekProvider(
                client=_make_offline_client(_ScriptedCodingTransport()),
                model=ALLOWED_MODEL,
            ),
            planner_config=PlannerConfig(
                schema_version="0.1",
                max_calls=pcfg.max_calls,
                call_timeout_ms=pcfg.call_timeout_ms,
                cleanup_timeout_ms=pcfg.cleanup_timeout_ms,
                max_request_bytes=pcfg.max_request_bytes,
                max_response_bytes=pcfg.max_response_bytes,
                max_output_tokens=pcfg.max_output_tokens,
            ),
            trace_writer=coding_api_demo.JsonlTraceStore(paths["trace"]),
        )
    )
    assert called == [True]
    assert res.terminal_status == "completed"


def test_l12_display_and_failure_truth(tmp_path: Path) -> None:
    # 1. Bypassing read/failing-pytest (only model self-claims verify_task after direct write)
    # cannot yield exit 0 or verified=True even when JUnit passes 2/2.
    bypass_root = tmp_path / "l12_bypass"
    summary_bypass, code_bypass = asyncio.run(
        run_demo_with_client(
            root=bypass_root,
            client=_make_offline_client(_ScriptedCodingTransport(skip_diagnostics=True)),
            mode="offline_test",
        )
    )
    assert code_bypass == 6
    assert summary_bypass["functional_passed"] is True
    assert summary_bypass["workflow_evidence_satisfied"] is False
    assert summary_bypass["verified"] is False
    assert summary_bypass["terminal_status"] == "failed"

    # 2. Normal failure preserves original/attempted files and raw outputs cleanly without auto-fix
    fail_root = tmp_path / "l12_normal_fail"

    def _bad_write_transport(req: httpx.Request) -> httpx.Response:
        _, state = _extract_model_state(req)
        last = state.get("last_tool_result")
        files = {e["path"]: e for e in state.get("files", [])}
        calc_sha = files["calculator.py"]["sha256"]
        if not last:
            return _wrap_chat_completion(
                _action_proposal(
                    "read_file",
                    {"path": "calculator.py", "start_line": 1, "max_lines": 50},
                    "coding_tool_result",
                    {"action_type": "read_file", "path": "calculator.py"},
                )
            )
        if last.get("action_type") == "read_file":
            return _wrap_chat_completion(
                _action_proposal(
                    "write_file",
                    {
                        "path": "calculator.py",
                        "expected_sha256": calc_sha,
                        "content": '"""Broken attempt."""\n\n\ndef sum_even(values):\n    return 999\n',
                    },
                    "coding_file_written",
                    {"path": "calculator.py"},
                )
            )
        return httpx.Response(500, json={"error": {"message": "provider failed"}})

    summary_fail, code_fail = asyncio.run(
        run_demo_with_client(
            root=fail_root,
            client=_make_offline_client(_bad_write_transport),
            mode="offline_test",
        )
    )
    assert code_fail != 0
    assert summary_fail["verified"] is False
    assert summary_fail["exercised"] is True
    assert summary_fail["configured"] is True
    assert summary_fail["mode"] == "offline_test"
    assert summary_fail["billed_cost_cny"] is None
    assert "return 999" in (fail_root / "workspace" / "calculator.py").read_text(encoding="utf-8")
    assert (fail_root / "provider-events.jsonl").is_file()
    assert (fail_root / "trace.jsonl").is_file()
    assert (fail_root / "diff.patch").is_file()
    assert "elapsed_seconds" in summary_fail
    assert "junit" in summary_fail
    assert "steps" in summary_fail
    assert "changed_paths" in summary_fail


def _run_clean_demo(root: Path) -> tuple[dict[str, Any], int]:
    return asyncio.run(
        run_demo_with_client(
            root=root,
            client=_make_offline_client(_ScriptedCodingTransport()),
            mode="offline_test",
        )
    )


def _locate_bound_junit(root: Path) -> Path:
    run_result = json.loads((root / "output" / "coding-run-result.json").read_text(encoding="utf-8"))
    results = run_result["acceptance"]["results"]
    assert len(results) == 1
    for arg in results[0]["argv"]:
        if isinstance(arg, str) and arg.startswith("--junitxml="):
            rel_or_abs = Path(arg.split("=", 1)[1])
            return rel_or_abs if rel_or_abs.is_absolute() else (root / "workspace" / rel_or_abs)
    raise AssertionError("bound --junitxml argument missing from coding-run-result.json")


def test_d01_junction_entrypoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    junction_target = tmp_path / "junction-target"
    junction_target.mkdir()
    junction_alias = tmp_path / "junction-alias"
    proc = subprocess.run(
        [
            os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"),
            "/d",
            "/c",
            "mklink",
            "/J",
            str(junction_alias),
            str(junction_target),
        ],
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    attrs = getattr(os.lstat(junction_alias), "st_file_attributes", 0)
    assert (attrs & 0x400) != 0

    for candidate in (junction_alias / "new-demo", junction_alias / ".." / "dotdot-demo"):
        with pytest.raises(CodingApiDemoError, match="reparse"):
            prepare_demo(candidate)
        assert not (junction_target / "new-demo").exists()
        assert not (tmp_path / "dotdot-demo").exists()
        assert not (junction_target.parent / "dotdot-demo").exists()

        transport = _ScriptedCodingTransport()
        with pytest.raises(CodingApiDemoError, match="reparse"):
            asyncio.run(
                run_demo_with_client(
                    root=candidate,
                    client=_make_offline_client(transport),
                    mode="offline_test",
                )
            )
        assert not (junction_target / "new-demo").exists()
        assert not (tmp_path / "dotdot-demo").exists()
        assert not (junction_target.parent / "dotdot-demo").exists()
        assert len(transport.requests) == 0

    monkeypatch.chdir(tmp_path)
    with pytest.raises(CodingApiDemoError, match="reparse"):
        prepare_demo(Path("junction-alias") / ".." / "rel-dotdot-demo")
    assert not (tmp_path / "rel-dotdot-demo").exists()

    existing_root = tmp_path / "ordinary-existing"
    existing_root.mkdir()
    sentinel = existing_root / "sentinel.txt"
    sentinel.write_bytes(b"keep-me\n")
    with pytest.raises(CodingApiDemoError, match="already exists"):
        prepare_demo(existing_root)
    assert sentinel.read_bytes() == b"keep-me\n"

    summary, exit_code = _run_clean_demo(tmp_path / "ordinary-fresh")
    assert exit_code == 0
    assert summary["verified"] is True

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    summary_dotdot, exit_dotdot = _run_clean_demo(plain_dir / ".." / "ordinary-dotdot-fresh")
    assert exit_dotdot == 0
    assert summary_dotdot["verified"] is True


def test_d02_actual_execution_required(tmp_path: Path) -> None:
    root = tmp_path / "d02_demo"
    summary, exit_code = _run_clean_demo(root)
    assert exit_code == 0
    assert summary["verified"] is True
    run_result_path = root / "output" / "coding-run-result.json"
    assert run_result_path.is_file()
    assert validate_demo_evidence(root) == []

    trace_path = root / "trace.jsonl"
    original_trace = trace_path.read_bytes()
    rows = _read_jsonl(trace_path)
    filtered_rows = [
        row
        for row in rows
        if row.get("kind") not in {"attempt_started", "attempt_finished"}
    ]
    _write_jsonl(trace_path, filtered_rows)
    reasons = validate_demo_evidence(root)
    assert "missing:actual_execution_attempts" in reasons
    assert "coding_run_result_attempts_mismatch" in reasons

    trace_path.write_bytes(original_trace)
    assert validate_demo_evidence(root) == []

    original_run_result = run_result_path.read_bytes()
    try:
        run_result_path.unlink()
        missing_reasons = validate_demo_evidence(root)
        assert missing_reasons == ["missing:coding_run_result_json"]
        assert not run_result_path.exists()
    finally:
        run_result_path.write_bytes(original_run_result)
    assert validate_demo_evidence(root) == []


def test_d03_final_run_oracle(tmp_path: Path) -> None:
    root = tmp_path / "d03_demo"
    summary, exit_code = _run_clean_demo(root)
    assert exit_code == 0
    assert summary["verified"] is True
    assert validate_demo_evidence(root) == []

    trace_path = root / "trace.jsonl"
    original_trace = trace_path.read_bytes()
    rows = _read_jsonl(trace_path)
    terminal = next(row for row in reversed(rows) if row.get("kind") == "run_finished")
    target_record_id = terminal["verification_record_id"]
    run_records = [
        row
        for row in rows
        if row.get("record_id") == target_record_id and row.get("scope") == "run"
    ]
    assert len(run_records) == 1

    for row in rows:
        if row.get("record_id") == target_record_id and row.get("scope") == "run":
            row["verification_status"] = "fail"
    _write_jsonl(trace_path, rows)
    reasons = validate_demo_evidence(root)
    assert "failed:run_verification_oracle" in reasons

    trace_path.write_bytes(original_trace)
    assert validate_demo_evidence(root) == []


def test_d04_bound_junit(tmp_path: Path) -> None:
    root = tmp_path / "d04_demo"
    summary, exit_code = _run_clean_demo(root)
    assert exit_code == 0
    assert summary["verified"] is True
    assert validate_demo_evidence(root) == []

    bound_junit = _locate_bound_junit(root)
    passing_bytes = bound_junit.read_bytes()
    bound_junit.write_text(
        '<testsuite tests="2" failures="1" errors="0" skipped="0">'
        '<testcase classname="test_acceptance" name="test_basic_even"><failure/></testcase>'
        '<testcase classname="test_acceptance" name="test_empty_and_negatives"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    assert "invalid:junit_case_not_passed" in validate_demo_evidence(root)

    orphan = root / "output" / "verify" / "not-an-executed-command" / "junit.xml"
    orphan.parent.mkdir(exist_ok=False)
    orphan.write_bytes(passing_bytes)
    future_ts = time.time() + 120.0
    os.utime(orphan, (future_ts, future_ts))
    assert "invalid:junit_case_not_passed" in validate_demo_evidence(root)

    bound_junit.write_bytes(passing_bytes)
    assert validate_demo_evidence(root) == []


def test_d05_exceptional_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert llm_api_smoke.CLOSE_TIMEOUT_SECONDS == 5.0

    # Part A: existing live root refuses before make_client is called.
    existing_root = tmp_path / "d05_existing_live"
    existing_root.mkdir()
    sentinel = existing_root / "sentinel.txt"
    sentinel.write_bytes(b"preserve-me\n")
    created_clients = 0

    def _forbid_make_client(**_: Any) -> Any:
        nonlocal created_clients
        created_clients += 1
        raise AssertionError("make_client must not be called for invalid existing root")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-fake-offline")
    monkeypatch.setattr(coding_api_demo, "make_client", _forbid_make_client)
    code_existing = main(
        [
            "--mode",
            "live",
            "--root",
            str(existing_root),
            "--model",
            "deepseek-flash",
            "--max-requests",
            "8",
            "--budget-cny",
            "1",
        ]
    )
    _ = capsys.readouterr()
    assert code_existing == 2
    assert created_clients == 0
    assert sentinel.read_bytes() == b"preserve-me\n"

    # Part B: HangingClient under bounded close timeout in run_demo_with_client.
    monkeypatch.setattr(llm_api_smoke, "CLOSE_TIMEOUT_SECONDS", 0.15)
    entered_direct = False
    hang_transport_calls = 0

    class _HangingClient:
        async def post(self, *_: Any, **__: Any) -> httpx.Response:
            nonlocal hang_transport_calls
            hang_transport_calls += 1
            raise AssertionError("transport post must not run for invalid root")

        async def aclose(self) -> None:
            nonlocal entered_direct
            entered_direct = True
            await asyncio.sleep(60.0)

    async def _run_part_b() -> tuple[float, int]:
        t0 = time.monotonic()
        with pytest.raises(CodingApiDemoError, match="already exists"):
            await run_demo_with_client(
                root=existing_root,
                client=_HangingClient(),  # type: ignore[arg-type]
                mode="offline_test",
            )
        elapsed = time.monotonic() - t0
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        return elapsed, len(pending)

    elapsed_b, dangling_b = asyncio.run(_run_part_b())
    assert entered_direct is True
    assert elapsed_b < 2.0
    assert dangling_b == 0
    assert hang_transport_calls == 0
    assert sentinel.read_bytes() == b"preserve-me\n"

    # Part C: main live mode failing after make_client returns a HangingClient closes boundedly.
    fresh_live_root = tmp_path / "d05_fresh_live_fail"
    entered_main = False

    class _MainHangingClient:
        async def post(self, *_: Any, **__: Any) -> httpx.Response:
            nonlocal hang_transport_calls
            hang_transport_calls += 1
            raise AssertionError("transport post must not run when prepare_demo fails")

        async def aclose(self) -> None:
            nonlocal entered_main
            entered_main = True
            await asyncio.sleep(60.0)

    def _make_hanging_client(**_: Any) -> _MainHangingClient:
        return _MainHangingClient()

    def _failing_prepare_demo(_: Path) -> dict[str, Path]:
        raise CodingApiDemoError("injected post-client failure")

    monkeypatch.setattr(coding_api_demo, "make_client", _make_hanging_client)
    monkeypatch.setattr(coding_api_demo, "prepare_demo", _failing_prepare_demo)
    t1 = time.monotonic()
    code_fail = main(
        [
            "--mode",
            "live",
            "--root",
            str(fresh_live_root),
            "--model",
            "deepseek-flash",
            "--max-requests",
            "8",
            "--budget-cny",
            "1",
        ]
    )
    elapsed_c = time.monotonic() - t1
    _ = capsys.readouterr()
    assert code_fail != 0
    assert entered_main is True
    assert elapsed_c < 2.0
    assert hang_transport_calls == 0
    assert sentinel.read_bytes() == b"preserve-me\n"


@pytest.mark.parametrize(
    "mutation_kind",
    ["swap_executable", "drop_dash_b", "swap_test_path", "append_collect_only"],
)
def test_d06_bound_full_command(tmp_path: Path, mutation_kind: str) -> None:
    root = tmp_path / f"d06_{mutation_kind}"
    summary, exit_code = _run_clean_demo(root)
    assert exit_code == 0
    assert summary["verified"] is True
    assert validate_demo_evidence(root) == []

    result_path = root / "output" / "coding-run-result.json"
    original_bytes = result_path.read_bytes()
    doc = json.loads(original_bytes.decode("utf-8"))
    argv: list[str] = list(doc["acceptance"]["results"][0]["argv"])
    if mutation_kind == "swap_executable":
        argv[0] = "not-the-approved-python.exe"
    elif mutation_kind == "drop_dash_b":
        argv.remove("-B")
    elif mutation_kind == "swap_test_path":
        argv[-1] = str(Path(argv[-1]).with_name("other_test.py"))
    elif mutation_kind == "append_collect_only":
        argv.append("--collect-only")
    doc["acceptance"]["results"][0]["argv"] = argv
    assert not (root / "external_tests" / "other_test.py").exists()

    try:
        result_path.write_text(json.dumps(doc), encoding="utf-8")
        reasons = validate_demo_evidence(root)
        assert "coding_run_result_cmd_argv_mismatch" in reasons
    finally:
        result_path.write_bytes(original_bytes)
    assert validate_demo_evidence(root) == []


@pytest.mark.parametrize(
    "mutation_kind",
    ["tool_succeeded_false", "read_back_match_false", "both_false", "zero_read_back_sha"],
)
def test_d07_actual_write_feedback(tmp_path: Path, mutation_kind: str) -> None:
    root = tmp_path / f"d07_{mutation_kind}"
    summary, exit_code = _run_clean_demo(root)
    assert exit_code == 0
    assert summary["verified"] is True
    assert validate_demo_evidence(root) == []

    events_path = root / "provider-events.jsonl"
    original_bytes = events_path.read_bytes()
    events = _read_jsonl(events_path)
    write_idx = next(
        idx
        for idx, ev in enumerate(events)
        if json.loads(ev["response"]["choices"][0]["message"]["content"])["proposed_action"]["action_type"]
        == "write_file"
    )
    assert write_idx + 1 < len(events)
    followup_req = json.loads(events[write_idx + 1]["request"]["messages"][1]["content"])
    feedback = followup_req["observation"]["state"]["last_tool_result"]
    assert feedback["action_type"] == "write_file"
    assert feedback["tool_succeeded"] is True
    assert feedback["read_back_match"] is True

    if mutation_kind == "tool_succeeded_false":
        feedback["tool_succeeded"] = False
    elif mutation_kind == "read_back_match_false":
        feedback["read_back_match"] = False
    elif mutation_kind == "both_false":
        feedback["tool_succeeded"] = False
        feedback["read_back_match"] = False
    elif mutation_kind == "zero_read_back_sha":
        feedback["read_back_sha256"] = "0" * 64

    events[write_idx + 1]["request"]["messages"][1]["content"] = json.dumps(followup_req)
    try:
        _write_jsonl(events_path, events)
        reasons = validate_demo_evidence(root)
        assert reasons != []
    finally:
        events_path.write_bytes(original_bytes)
    assert validate_demo_evidence(root) == []
