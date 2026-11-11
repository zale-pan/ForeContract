"""Real-browser acceptance of the optional single-target observation budget.

The test runs the real ``commit_unknown`` profile over the real loopback site and the real
headless Chromium adapter. It proves that the one declared 500 ms per-target allowance lets the
post-attempt observation return real evidence inside the frozen three-second operation allowance,
that a target group which cannot finish is the existing ``unavailable`` record rather than a
fabricated zero match, that an ordinary page still yields complete collected records, and that
every observation call the Runtime published has a real snapshot. No budget is widened and no
profile is retried.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "examples", REPO / "packages" / "core" / "src",
                  REPO / "packages" / "adapters" / "web" / "src",
                  REPO / "packages" / "adapters" / "fake" / "src",
                  REPO / "packages" / "planners" / "scripted" / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import web_runtime_demo as demo


def _profile_output(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("p3t05-observation-budget")
    output = base / "commit_unknown"
    output.mkdir()
    try:
        return demo.run_profile("commit_unknown", output)
    except demo.DemoFailure as exc:
        if exc.original is not None:
            raise exc.original
        raise


@pytest.fixture(scope="module")
def unknown_profile(tmp_path_factory) -> dict:
    """Run the one real unknown-submit profile once; every node reads its real evidence."""
    return _profile_output(tmp_path_factory)


def _index(profile: dict) -> dict:
    return json.loads(Path(profile["evidence_index_path"]).read_text(encoding="utf-8"))


def _root(profile: dict) -> Path:
    return Path(profile["evidence_index_path"]).parent


def _observations(index: dict) -> list[dict]:
    return [entry for entry in index["snapshots"] if entry["kind"] == "observation"]


def _snapshot_payload(root: Path, entry: dict) -> dict:
    document = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
    return document["payload"]


def _targets(root: Path, entry: dict) -> list[dict]:
    return _snapshot_payload(root, entry)["state"]["targets"]


def test_only_the_unknown_submit_declares_the_optional_target_budget_and_no_budget_moved() -> None:
    """The correction adds one optional value: no frozen allowance is widened or reset."""
    ordinary = {"normal", "wrong_credentials", "delayed_release", "overlay_blocked",
                "overlay_released", "commit_unknown", "cancel_after_received"}
    for name in ordinary:
        profile = demo._profile(name)
        assert (profile.run_ms, profile.adapter_operation_ms, profile.request_ms) == (
            30000, 3000, 15000), name
    deadline = demo._profile("delayed_deadline")
    assert (deadline.run_ms, deadline.adapter_operation_ms, deadline.request_ms) == (
        15000, 25000, 30000)
    for name in ordinary - {"commit_unknown"}:
        assert demo._profile(name).observation_target_timeout_ms is None, name
    assert demo._profile("delayed_deadline").observation_target_timeout_ms is None
    assert demo._profile("commit_unknown").observation_target_timeout_ms == 500.0


def test_the_unknown_submit_keeps_one_submit_and_gets_a_real_post_attempt_snapshot(
        unknown_profile) -> None:
    profile = unknown_profile
    index = _index(profile)
    root = _root(profile)
    assert profile["terminal"] == {"terminal_status": "aborted",
                                   "reason_code": "POLICY_ABORT", "source": "run_result"}
    assert profile["oracle"]["before_stop"]["commits"] == 1
    observations = _observations(index)
    assert observations
    post_attempt = [entry for entry in observations
                    if entry.get("phase") in ("VERIFYING", "RECOVERING")]
    assert post_attempt, [entry.get("phase") for entry in observations]
    for entry in post_attempt:
        records = _targets(root, entry)
        assert len(records) == 4
        for record in records:
            assert record["status"] in ("collected", "unavailable")
            if record["status"] == "unavailable":
                # A bounded local failure is the existing unavailable record, never a zero match
                # and never a fabricated attribute value.
                assert record["match_count"] is None
                assert record["visible"] is None
                assert record["text"] is None
                assert record["value"] is None
                assert record["error_code"] == "TARGET_QUERY_FAILED"
            else:
                assert type(record["match_count"]) is int
                if record["match_count"] == 1:
                    assert record["visible"] is True


def test_an_ordinary_page_still_yields_complete_collected_records_inside_the_same_budget(
        unknown_profile) -> None:
    profile = unknown_profile
    index = _index(profile)
    root = _root(profile)
    ordinary = [entry for entry in _observations(index)
                if entry.get("phase") in ("OBSERVING", "CHECKING_PRECONDITIONS")]
    assert ordinary
    assert any(all(record["status"] == "collected" for record in _targets(root, entry))
               for entry in ordinary)
    login_records = [record for entry in ordinary for record in _targets(root, entry)
                     if record["status"] == "collected" and record["match_count"] == 1]
    assert any(record["visible"] is True for record in login_records)


def test_every_observation_call_of_the_unknown_submit_has_a_real_snapshot(
        unknown_profile) -> None:
    """Core restores every observe call; here every call really returned an observation."""
    profile = unknown_profile
    index = _index(profile)
    calls = index["working_memory"]["observation_calls"]
    observations = _observations(index)
    assert calls
    assert all(call["observation_id"] is not None for call in calls)
    assert len(calls) == len(observations)
    assert [call["ordinal"] for call in calls] == [
        entry["runtime_call_ordinal"] for entry in observations]


def test_all_eight_profiles_declare_the_10000ms_startup_allowance_and_keep_every_other_budget(
        ) -> None:
    """C05-2: one startup-only value, every other frozen budget unchanged."""
    for name in demo.PROFILES:
        profile = demo._profile(name)
        assert profile.startup_timeout_ms == 10000, name
        assert demo.ADAPTER_STARTUP_MS == 10000
        config = demo._adapter_config("http://127.0.0.1:8000", profile)
        assert config.startup_timeout_ms == 10000.0, name
        assert config.effective_startup_timeout_ms == 10000.0, name
        # The startup value never rewrites the operation, per-target or cleanup allowances.
        assert config.operation_timeout_ms == float(profile.adapter_operation_ms), name
        assert config.observation_target_timeout_ms == (
            profile.observation_target_timeout_ms), name
        assert config.cleanup_timeout_ms == 5000.0, name
    # The non-startup budgets stay exactly the historical values.
    assert (demo.ADAPTER_CLEANUP_MS, demo.RUNTIME_CLEANUP_MS, demo.RUNTIME_FINALIZE_MS,
            demo.SITE_CLEANUP_MS, demo.HOST_WAIT_MS) == (5000.0, 5000.0, 5000.0, 3000, 10000)


def test_the_declared_startup_allowance_is_recorded_for_start_only(unknown_profile) -> None:
    """The host diagnostics keep the startup allowance separate from operation and cleanup."""
    index = _index(unknown_profile)
    boundaries = index["diagnostics"]["boundaries"]
    operations = {row["operation"] for row in boundaries}
    assert operations <= {"start", "observe", "execute", "stop"}
    assert "start" in operations
    assert all(row["budget_ms"] == 10000 for row in boundaries
               if row["operation"] == "start")
    assert all(row["budget_ms"] == 3000 for row in boundaries
               if row["operation"] in ("observe", "execute"))
    assert all(row["budget_ms"] == 5000 for row in boundaries if row["operation"] == "stop")
    # ``elapsed_ms`` is the whole host boundary duration. On a failed start it includes the
    # bounded failed-start cleanup, so it is never read as the native start-up duration alone.
    assert all(type(row["elapsed_ms"]) is int and row["elapsed_ms"] >= 0
               for row in boundaries)
