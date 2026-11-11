"""Real eight-profile acceptance of the Web Runtime.

Every profile is executed by the example against the real loopback test site and the real
headless Chromium adapter. The expectations below are stated independently from the example:
literal terminal names, reason codes, step/attempt counts and site-oracle counters derived from
the contract, never from the example's own summary. No profile is skipped and no browser path is
substituted by a mock.
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

import web_runtime_demo


@pytest.fixture(scope="module", autouse=True)
def _unload_playwright_after_module():
    """Leave the process as this test found it: browser imports stay local to the browser phase.

    The older Web capability test asserts that importing the pure Web package does not import
    Playwright. These real-browser profiles necessarily import it in-process, so the module
    removes only the modules it introduced once its own tests have finished.
    """
    yield
    for name in [name for name in sys.modules
                 if name == "playwright" or name.startswith("playwright.")]:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def profiles(tmp_path_factory) -> dict[str, dict]:
    """Run each of the eight profiles once, serially, into its own external directory."""
    base = tmp_path_factory.mktemp("p3t05-web-runtime")
    results: dict[str, dict] = {}
    for name in web_runtime_demo.PROFILES:
        output = base / name
        output.mkdir()
        try:
            results[name] = web_runtime_demo.run_profile(name, output)
        except web_runtime_demo.DemoFailure as exc:
            if exc.original is not None:
                raise exc.original
            raise
    return results


def _result(profile: dict) -> dict:
    assert profile["result"] is not None, profile["terminal"]
    return profile["result"]


def _counts(profile: dict) -> tuple[int, int, int, int, int]:
    oracle = profile["oracle"]
    snapshot = oracle["before_stop"]
    return (snapshot["login_requests"], snapshot["auth_successes"],
            snapshot["auth_failures"], snapshot["commits"],
            snapshot["authenticated_sessions"])


def _index(profile: dict) -> dict:
    return json.loads(Path(profile["evidence_index_path"]).read_text(encoding="utf-8"))


def test_normal_profile_completes_with_exactly_one_login_and_commit(profiles) -> None:
    profile = profiles["normal"]
    result = _result(profile)
    assert profile["terminal"] == {"terminal_status": "completed",
                                   "reason_code": "TASK_ACHIEVED",
                                   "source": "run_result"}
    assert (result["steps_started"], result["attempts_started"]) == (4, 4)
    assert _counts(profile) == (1, 1, 0, 1, 1)
    assert profile["oracle"]["site_stop_complete"] is True


def test_wrong_credentials_aborts_without_a_commit(profiles) -> None:
    profile = profiles["wrong_credentials"]
    result = _result(profile)
    assert profile["terminal"]["terminal_status"] == "aborted"
    assert profile["terminal"]["reason_code"] == "POLICY_ABORT"
    assert result["terminal_status"] != "completed"
    assert (result["steps_started"], result["attempts_started"]) == (4, 4)
    assert _counts(profile) == (1, 0, 1, 0, 0)


def test_delayed_release_waits_for_the_gate_with_zero_commits(profiles) -> None:
    profile = profiles["delayed_release"]
    result = _result(profile)
    facts = profile["oracle"]["phase_facts"]
    assert facts["login_received"] is True
    assert facts["before_release"]["login_requests"] == 1
    assert facts["before_release"]["commits"] == 0
    assert facts["before_release"]["response_blocked"] is True
    assert profile["terminal"]["terminal_status"] == "completed"
    assert (result["steps_started"], result["attempts_started"]) == (4, 4)
    assert _counts(profile) == (1, 1, 0, 1, 1)


def test_delayed_deadline_is_a_real_run_deadline_not_a_prestart_expiry(profiles) -> None:
    profile = profiles["delayed_deadline"]
    result = _result(profile)
    facts = profile["oracle"]["phase_facts"]
    assert facts["login_received"] is True
    assert facts["after_login_received"]["login_requests"] == 1
    assert facts["after_login_received"]["response_blocked"] is True
    assert profile["terminal"] == {"terminal_status": "timeout",
                                   "reason_code": "DEADLINE_EXCEEDED",
                                   "source": "run_result"}
    assert (result["steps_started"], result["attempts_started"]) == (4, 4)
    login, _success, _failure, commits, _sessions = _counts(profile)
    assert (login, commits) == (1, 0)


def test_overlay_blocked_performs_no_request_and_no_commit(profiles) -> None:
    profile = profiles["overlay_blocked"]
    result = _result(profile)
    assert profile["terminal"]["terminal_status"] == "aborted"
    assert profile["terminal"]["reason_code"] == "POLICY_ABORT"
    assert (result["steps_started"], result["attempts_started"]) == (4, 4)
    assert _counts(profile) == (0, 0, 0, 0, 0)


def test_overlay_released_is_an_explicit_new_run_and_new_generation(profiles) -> None:
    profile = profiles["overlay_released"]
    result = _result(profile)
    facts = profile["oracle"]["phase_facts"]
    previous = profile["oracle"]["previous_run"]
    assert profile["terminal"]["terminal_status"] == "completed"
    assert (result["steps_started"], result["attempts_started"]) == (4, 4)
    assert facts["reset_generation"] == 2
    assert previous is not None
    assert previous["terminal"]["terminal_status"] == "aborted"
    assert previous["result"]["steps_started"] == 4
    assert previous["oracle"]["before_reset"]["login_requests"] == 0
    assert previous["oracle"]["before_reset"]["commits"] == 0
    assert _counts(profile) == (1, 1, 0, 1, 1)


def test_commit_unknown_is_aborted_with_the_permitted_policy_abort_and_no_replay(profiles) -> None:
    profile = profiles["commit_unknown"]
    result = _result(profile)
    index = _index(profile)
    assert profile["terminal"]["terminal_status"] == "aborted"
    assert profile["terminal"]["reason_code"] == "POLICY_ABORT"
    assert (result["steps_started"], result["attempts_started"]) == (4, 4)
    login, _success, _failure, commits, sessions = _counts(profile)
    assert (login, commits, sessions) == (1, 1, 1)
    # The unknown submit keeps BOTH facts: the Runtime terminal POLICY_ABORT and the recovery
    # reason EFFECT_UNRESOLVED. The first assessment asks for post-attempt evidence; the second
    # one really aborts on the unknown effect. Neither is a replay.
    outcomes = [entry for entry in index["snapshots"] if entry["kind"] == "actual_outcome"]
    assert [entry["execution_status"] for entry in outcomes] == [
        "succeeded", "succeeded", "succeeded", "unknown"]
    assessments = [entry for entry in index["snapshots"]
                   if entry["kind"] == "recovery_assessment"]
    assert len(assessments) == 2
    assert assessments[0]["assessment_kind"] == "observation_required"
    assert assessments[0]["assessment_reason"] == "OBSERVATION_REQUIRED"
    assert assessments[1]["assessment_decision"] == "abort"
    assert assessments[1]["assessment_reason"] == "EFFECT_UNRESOLVED"
    step_verifications = [entry for entry in index["snapshots"]
                          if entry["kind"] == "step_verification"]
    assert step_verifications[-1]["verifier"] == "web_step_verifier"
    assert [event["kind"] for event in index["events"]].count("attempt_started") == 4
    assert [event["kind"] for event in index["events"]].count("attempt_finished") == 4


def test_every_profile_uses_the_frozen_budget_and_the_standard_recovery_policy() -> None:
    """F01: the budgets are the frozen ones and no profile substitutes an abort policy."""
    ordinary = {"normal", "wrong_credentials", "delayed_release", "overlay_blocked",
                "overlay_released", "commit_unknown", "cancel_after_received"}
    for name in ordinary:
        profile = web_runtime_demo._profile(name)
        # Every ordinary profile keeps the complete frozen triple. No profile may end its site
        # request early to make an observation fit: a real-time conflict is reported, not tuned.
        assert (profile.run_ms, profile.adapter_operation_ms, profile.request_ms) == (
            30000, 3000, 15000), name
    deadline = web_runtime_demo._profile("delayed_deadline")
    assert (deadline.run_ms, deadline.adapter_operation_ms, deadline.request_ms) == (
        15000, 25000, 30000)
    assert web_runtime_demo.STANDARD_RECOVERY.mode == "bounded"
    assert web_runtime_demo.STANDARD_RECOVERY.max_retries == 0
    assert web_runtime_demo.STANDARD_RECOVERY.allow_replan is False
    policies = web_runtime_demo._policies("http://127.0.0.1:1", web_runtime_demo.PASSWORD,
                                          click_timeout_ms=5000.0,
                                          recovery_policy=web_runtime_demo.STANDARD_RECOVERY)
    for policy in policies:
        assert policy.recovery_policy.mode == "bounded"
        assert policy.recovery_policy.max_retries == 0
        assert policy.recovery_policy.allow_replan is False


def test_no_profile_replays_the_non_idempotent_submit(profiles) -> None:
    """The single submit and the bounded attempt counts are independent site-oracle facts."""
    for name, profile in profiles.items():
        oracle = profile["oracle"]
        snapshot = oracle["before_stop"]
        assert snapshot["commits"] <= 1, name
        result = profile["result"]
        if name == "cancel_after_received":
            assert result is None
            assert profile["cancelled"] is True
            assert snapshot["commits"] == 0
        else:
            assert result["attempts_started"] == 4, name
            assert result["retries_used"] == 0, name
            assert result["replans_used"] == 0, name
        previous = oracle.get("previous_run")
        if previous is not None:
            assert previous["oracle"]["before_reset"]["commits"] == 0


def test_the_host_declares_the_real_web_step_verifier_identity() -> None:
    assert web_runtime_demo.WEB_STEP_VERIFIER_NAME == "web_step_verifier"


def test_cancel_after_received_propagates_and_records_one_cancellation(profiles) -> None:
    profile = profiles["cancel_after_received"]
    index = _index(profile)
    assert profile["cancelled"] is True
    assert profile["result"] is None
    assert profile["terminal"] == {"terminal_status": "cancelled",
                                   "reason_code": "USER_CANCELLED",
                                   "source": "last_finalization"}
    finalization = profile["last_finalization"]
    assert finalization["primary_terminal_status"] == "cancelled"
    assert finalization["primary_reason_code"] == "USER_CANCELLED"
    facts = profile["oracle"]["phase_facts"]
    assert facts["login_received"] is True
    assert facts["before_cancel"]["commits"] == 0
    login, _success, _failure, commits, _sessions = _counts(profile)
    assert (login, commits) == (1, 0)
    events = [event["kind"] for event in index["events"]]
    assert events.count("step_started") == 4
    assert events.count("attempt_started") == 4
    assert events.count("attempt_finished") == 3
    assert events.count("run_finished") == 1
    assert index["unfinished_attempt_id"] is not None
    assert index["call_interrupted"] == "CancelledError"
    assert len([entry for entry in index["snapshots"]
                if entry["kind"] == "actual_outcome"]) == 3


def test_the_browser_identity_is_the_verified_environment(profiles) -> None:
    identity = profiles["normal"]["browser"]
    assert identity["playwright"] == "1.64.1"
    assert identity["chromium_revision"] == "1248"
    assert identity["chromium_version"] == "156.0.8078.4"


def test_every_profiles_evidence_index_is_hash_bound(profiles) -> None:
    from web_runtime_evidence import verify_evidence_index

    for name, profile in profiles.items():
        index = verify_evidence_index(Path(profile["evidence_index_path"]).parent)
        assert index["profile"] in (name, "overlay_blocked")  # the prior Run reuses its profile
        assert index["trace"]["records"] >= 1
        assert index["snapshots"]


def test_the_measured_profiles_are_not_prewarmed_by_a_browser_fixture() -> None:
    """Correction 5: no autouse fixture launches a browser before the first measured profile.

    The two measured E2E modules run on the host's real cold-start conditions. A warm-up fixture
    used a setup-only allowance and unconditionally swallowed its ``stop`` failure, which could
    hide the first real failure. This node guards that removal: a cold start is reported as the
    profile's own first failure, never masked by a fixture.
    """
    removed_fixture = "_warm_" + "playwright_browser"
    swallowed_stop = "noqa: BLE001, " + "S110"
    for name in ("test_web_runtime_e2e.py", "test_web_runtime_trace.py"):
        source = (REPO / "tests" / "integration" / name).read_text(encoding="utf-8")
        assert removed_fixture not in source, name
        assert swallowed_stop not in source, name
