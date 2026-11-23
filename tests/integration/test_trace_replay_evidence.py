"""Integration tests for offline replay across the eight frozen Web evidence profiles."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
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

PROFILES: tuple[tuple[str, int], ...] = (
    ("normal", 9),
    ("wrong_credentials", 9),
    ("delayed_release", 9),
    ("delayed_deadline", 7),
    ("overlay_blocked", 9),
    ("overlay_released", 9),
    ("commit_unknown", 9),
    ("cancel_after_received", 7),
)


_CURRENT_VERIFICATION_SOURCES: tuple[str, ...] = (
    "packages/core/src/forecontract/memory/derived.py",
    "packages/core/src/forecontract/memory/derived_models.py",
    "packages/core/src/forecontract/verification/__init__.py",
    "packages/core/src/forecontract/verification/composite.py",
    "packages/core/src/forecontract/verification/extension_models.py",
    "packages/core/src/forecontract/verification/model.py",
)


def _fixtures_root() -> Path:
    env_path = os.environ.get("P5_REPLAY_FIXTURES")
    if env_path:
        return Path(env_path)
    return REPO_ROOT / "artifacts/verification/P5-T01/round-001-inputs/fixtures"


def _verifier_sources_expected() -> dict[str, Any]:
    path = REPO_ROOT / "artifacts/verification/P5-T01/round-001-inputs/verifier-sources.json"
    expected: dict[str, Any] = json.loads(path.read_bytes())
    for rel in _CURRENT_VERIFICATION_SOURCES:
        expected["sha256"][rel] = _sha256(REPO_ROOT / rel)
    return expected


def _load_deps() -> tuple[Any, Any, Any, Any]:
    for candidate in _SOURCES:
        entry = str(REPO_ROOT / candidate)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import replay_trace
    from forecontract.runtime.engine import RuntimeEngine
    from forecontract_scripted import ScriptedPlanner
    from forecontract_web.playwright_adapter import PlaywrightWebAdapter

    return replay_trace, RuntimeEngine, PlaywrightWebAdapter, ScriptedPlanner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize(("profile", "expected_count"), PROFILES)
def test_replay_frozen_profile_matches_recorded_verifications(
    profile: str,
    expected_count: int,
    tmp_path: Path,
) -> None:
    replay_trace, _, _, _ = _load_deps()
    root = _fixtures_root() / profile
    before = _tree_inventory(root)

    report = replay_trace.replay_evidence(root)
    assert report["schema_version"] == "0.1"
    assert report["kind"] == "verification_replay_report"
    assert report["profile"] == profile
    assert report["input_index_sha256"] == _sha256(root / "evidence-index.json")
    assert report["replay_status"] == "matched"
    assert report["verification_count"] == expected_count
    assert report["difference_count"] == 0
    assert len(report["verifications"]) == expected_count
    assert report["verifier_sources"] == _verifier_sources_expected()

    idx = json.loads((root / "evidence-index.json").read_bytes())
    assert report["recorded_terminal"] == idx["terminal"]
    assert {row["source_path"] for row in report["verifications"]} == {
        v["path"] for v in idx["verifications"]
    }
    order = [(r["kind"], r["ordinal"], r["source_path"]) for r in report["verifications"]]
    assert order == sorted(order)
    for row in report["verifications"]:
        assert row["equal"] is True
        assert row["differences"] == []
        assert row["recorded"] == row["recomputed"]

    out_file = tmp_path / f"{profile}.json"
    assert replay_trace.main(["--input", str(root), "--output", str(out_file)]) == 0
    assert json.loads(out_file.read_bytes()) == report
    assert _tree_inventory(root) == before


def test_total_68_historical_verifications_and_terminal_preservation() -> None:
    replay_trace, _, _, _ = _load_deps()
    total = 0
    terminals: dict[str, str] = {}
    for profile, count in PROFILES:
        root = _fixtures_root() / profile
        report = replay_trace.replay_evidence(root)
        total += report["verification_count"]
        assert report["verification_count"] == count
        terminals[profile] = report["recorded_terminal"]["terminal_status"]

    assert total == 68
    assert terminals["normal"] == "completed"
    assert terminals["delayed_release"] == "completed"
    assert terminals["overlay_released"] == "completed"
    assert terminals["wrong_credentials"] != "completed"
    assert terminals["delayed_deadline"] == "timeout"
    assert terminals["overlay_blocked"] != "completed"
    assert terminals["commit_unknown"] == "aborted"
    assert terminals["cancel_after_received"] == "cancelled"


def test_commit_unknown_preserves_recovery_reverification_and_independent_criteria() -> None:
    replay_trace, _, _, _ = _load_deps()
    cu_root = _fixtures_root() / "commit_unknown"
    cu_report = replay_trace.replay_evidence(cu_root)
    reverify_rows = [
        row
        for row in cu_report["verifications"]
        if row["kind"] == "step_verification" and row["ordinal"] == 5
    ]
    assert len(reverify_rows) == 1
    rev = reverify_rows[0]
    assert rev["trace_fact"] is None
    assert rev["event_record_id"] is None
    assert rev["attempt_id"] is None
    assert rev["contract_id"] is not None
    assert rev["step_id"] is not None
    assert rev["observation_id"] is not None

    normal_report = replay_trace.replay_evidence(_fixtures_root() / "normal")
    step_statuses = [
        row["recomputed"]["status"]
        for row in normal_report["verifications"]
        if row["kind"] == "step_verification"
    ]
    task_statuses = [
        row["recomputed"]["status"]
        for row in normal_report["verifications"]
        if row["kind"] == "task_verification"
    ]
    assert all(s == "pass" for s in step_statuses)
    assert any(s != "pass" for s in task_statuses)
    assert task_statuses[-1] == "pass"


def test_replay_executes_zero_external_or_runtime_actions() -> None:
    replay_trace, runtime_cls, adapter_cls, planner_cls = _load_deps()
    fix_root = _fixtures_root()
    before = _tree_inventory(fix_root)

    targets = (
        (socket.socket, "connect"),
        (socket, "create_connection"),
        (subprocess, "Popen"),
        (runtime_cls, "run"),
        (adapter_cls, "execute"),
        (planner_cls, "plan"),
    )
    with contextlib.ExitStack() as stack:
        mocks = [
            stack.enter_context(
                patch.object(obj, attr, side_effect=AssertionError(f"forbidden call: {attr}"))
            )
            for obj, attr in targets
        ]
        for profile, expected_count in PROFILES:
            report = replay_trace.replay_evidence(fix_root / profile)
            assert report["verification_count"] == expected_count
            assert report["difference_count"] == 0
        assert all(m.call_count == 0 for m in mocks)

    assert _tree_inventory(fix_root) == before
