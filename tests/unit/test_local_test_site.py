"""Unit tests for the strict records, state machine, pages and HTTP parsers.

These tests need no socket and no browser. They exercise the four scenarios and the five
scenario facts at the public control boundary, prove the immutable snapshot is detached, and
check the canonical form/generation/Content-Length rules that the real HTTP tests then drive over
a socket. A separate isolated subprocess proves that importing ``test_site`` creates no socket,
starts no thread and pulls in no third-party module.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_site
from test_site import (
    LocalSiteError,
    LocalTestSite,
    Scenario,
    SiteConfig,
    SiteEvent,
    SiteOracleSnapshot,
    SiteStopResult,
)
from test_site.pages import (
    FIXTURE_PASSWORD,
    FIXTURE_USERNAME,
    render_dashboard_page,
    render_login_page,
)
from test_site.server import (
    _content_type_ok,
    parse_content_length,
    parse_form_body,
    parse_generation,
)
from test_site.state import GateOutcome, SiteState


def make_config(
    scenario: Scenario = Scenario.NORMAL,
    request_timeout_ms: int = 5000,
    cleanup_timeout_ms: int = 3000,
) -> SiteConfig:
    return SiteConfig(
        scenario=scenario,
        request_timeout_ms=request_timeout_ms,
        cleanup_timeout_ms=cleanup_timeout_ms,
    )


# ---------------------------------------------------------------------------- public records


def test_public_exports_are_exactly_the_approved_types() -> None:
    assert set(test_site.__all__) == {
        "LocalSiteError",
        "LocalTestSite",
        "Scenario",
        "SiteConfig",
        "SiteEvent",
        "SiteOracleSnapshot",
        "SiteStopResult",
    }
    assert {scenario.value for scenario in Scenario} == {
        "normal",
        "delayed_login",
        "blocked_submit",
        "commit_then_stall",
    }
    assert {event.value for event in SiteEvent} == {
        "login_received",
        "response_blocked",
        "login_committed",
    }


def test_import_is_side_effect_free_and_stdlib_only() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import socket
import threading

real_socket = socket.socket
real_start = threading.Thread.start


class ForbiddenSocket:
    def __init__(self, *args, **kwargs):
        raise AssertionError("a socket was created during import")


def forbidden_start(self):
    raise AssertionError("a thread was started during import")


socket.socket = ForbiddenSocket
threading.Thread.start = forbidden_start
import test_site

assert len(threading.enumerate()) == 1
for forbidden in ("pydantic", "playwright", "pytest", "forecontract", "forecontract_web"):
    assert forbidden not in sys.modules, forbidden
assert test_site.__all__
print("import-ok")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-S", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import-ok"


def test_site_config_is_strict_frozen_and_has_no_extra_fields() -> None:
    config = make_config()
    assert config.scenario is Scenario.NORMAL
    with pytest.raises(FrozenInstanceError):
        config.request_timeout_ms = 9  # type: ignore[misc]
    with pytest.raises(TypeError):
        SiteConfig(  # type: ignore[call-arg]
            scenario=Scenario.NORMAL,
            request_timeout_ms=10,
            cleanup_timeout_ms=10,
            extra=True,
        )
    for scenario in (Scenario.NORMAL, Scenario.DELAYED_LOGIN, Scenario.BLOCKED_SUBMIT,
                     Scenario.COMMIT_THEN_STALL):
        assert make_config(scenario).scenario is scenario


@pytest.mark.parametrize(
    ("scenario", "request_timeout_ms", "cleanup_timeout_ms"),
    [
        ("normal", 1000, 1000),
        (Scenario.NORMAL, True, 1000),
        (Scenario.NORMAL, 0, 1000),
        (Scenario.NORMAL, 60001, 1000),
        (Scenario.NORMAL, 1.0, 1000),
        (Scenario.NORMAL, "1000", 1000),
        (Scenario.NORMAL, 1000, False),
        (Scenario.NORMAL, 1000, 0),
        (Scenario.NORMAL, 1000, 60001),
    ],
)
def test_site_config_rejects_every_non_strict_budget(
    scenario: object, request_timeout_ms: object, cleanup_timeout_ms: object
) -> None:
    with pytest.raises(LocalSiteError) as error:
        SiteConfig(
            scenario=scenario,  # type: ignore[arg-type]
            request_timeout_ms=request_timeout_ms,  # type: ignore[arg-type]
            cleanup_timeout_ms=cleanup_timeout_ms,  # type: ignore[arg-type]
        )
    assert error.value.code == "SITE_CONFIG_INVALID"


def test_oracle_and_stop_records_are_frozen_strict_and_detached() -> None:
    snapshot = SiteOracleSnapshot(
        generation=1,
        scenario=Scenario.NORMAL,
        running=True,
        login_requests=0,
        auth_successes=0,
        auth_failures=0,
        commits=0,
        active_requests=0,
        response_blocked=False,
        authenticated_sessions=0,
    )
    assert snapshot.schema_version == "0.1"
    with pytest.raises(FrozenInstanceError):
        snapshot.commits = 1  # type: ignore[misc]
    with pytest.raises(LocalSiteError) as error:
        SiteOracleSnapshot(
            generation=1,
            scenario=Scenario.NORMAL,
            running=True,
            login_requests=0,
            auth_successes=0,
            auth_failures=0,
            commits=0,
            active_requests=True,  # type: ignore[arg-type]
            response_blocked=False,
            authenticated_sessions=0,
        )
    assert error.value.code == "SITE_CONTROL_INVALID"
    result = SiteStopResult(
        complete=True, active_requests=0, open_connections=0, live_threads=0
    )
    with pytest.raises(FrozenInstanceError):
        result.complete = False  # type: ignore[misc]
    with pytest.raises(LocalSiteError):
        SiteStopResult(
            complete=1,  # type: ignore[arg-type]
            active_requests=0,
            open_connections=0,
            live_threads=0,
        )


def test_error_codes_are_closed_and_chain_free() -> None:
    error = LocalSiteError("SITE_BUSY")
    assert error.code == "SITE_BUSY"
    assert "busy" in str(error)
    unknown = LocalSiteError("caller supplied text")
    assert unknown.code == "SITE_CONTROL_INVALID"
    assert "caller supplied text" not in str(unknown)
    assert unknown.__cause__ is None
    assert unknown.__context__ is None


# ---------------------------------------------------------------------------- state machine


def test_state_counts_generation_events_and_session_revocation() -> None:
    state = SiteState(Scenario.NORMAL)
    state.mark_running()
    assert state.begin_request() is True
    assert state.record_login_received() is True
    token = state.commit_session()
    assert isinstance(token, str) and token
    assert state.session_is_valid(token) is True
    assert state.generation_matches(1) is True
    assert state.generation_matches(True) is False
    assert state.generation_matches("1") is False
    state.end_request()

    snapshot = state.snapshot()
    assert snapshot.generation == 1
    assert snapshot.scenario is Scenario.NORMAL
    assert snapshot.login_requests == 1
    assert snapshot.auth_successes == 1
    assert snapshot.auth_failures == 0
    assert snapshot.commits == 1
    assert snapshot.active_requests == 0
    assert snapshot.response_blocked is False
    assert snapshot.authenticated_sessions == 1

    assert state.reset(scenario=Scenario.DELAYED_LOGIN) == 2
    reset = state.snapshot()
    assert reset.generation == 2
    assert reset.scenario is Scenario.DELAYED_LOGIN
    assert (
        reset.login_requests,
        reset.auth_successes,
        reset.auth_failures,
        reset.commits,
        reset.authenticated_sessions,
    ) == (0, 0, 0, 0, 0)
    assert state.session_is_valid(token) is False
    assert state.generation_matches(1) is False
    assert state.generation_matches(2) is True


def test_state_reset_is_atomic_when_busy_and_stop_wakes_waiters() -> None:
    state = SiteState(Scenario.NORMAL)
    state.mark_running()
    assert state.begin_request() is True
    before = state.snapshot()
    with pytest.raises(LocalSiteError) as error:
        state.reset(scenario=Scenario.BLOCKED_SUBMIT)
    assert error.value.code == "SITE_BUSY"
    assert state.snapshot() == before
    state.end_request()

    recorded: list[GateOutcome] = []

    def wait_at_gate() -> None:
        recorded.append(state.gate_wait(time.monotonic() + 5.0))

    state.reset(scenario=Scenario.DELAYED_LOGIN)
    thread = threading.Thread(target=wait_at_gate)
    thread.start()
    assert state.wait_for(SiteEvent.RESPONSE_BLOCKED, 3000) is True
    assert state.snapshot().response_blocked is True
    state.mark_stopping()
    thread.join(3.0)
    assert thread.is_alive() is False
    assert recorded == [GateOutcome.STOPPED]
    assert state.snapshot().commits == 0
    assert state.snapshot().response_blocked is False


def test_wait_for_is_bounded_and_generation_scoped() -> None:
    state = SiteState(Scenario.NORMAL)
    state.mark_running()
    assert state.wait_for(SiteEvent.LOGIN_RECEIVED, 20) is False
    assert state.record_login_received() is True
    assert state.wait_for(SiteEvent.LOGIN_RECEIVED, 10) is True
    assert state.wait_for(SiteEvent.LOGIN_COMMITTED, 10) is False
    state.mark_stopping()
    assert state.wait_for(SiteEvent.LOGIN_RECEIVED, 10) is True
    assert state.wait_for(SiteEvent.LOGIN_COMMITTED, 10) is False
    for invalid in (0, 60001, True, "10", 1.0):
        with pytest.raises(LocalSiteError) as error:
            state.wait_for(SiteEvent.LOGIN_RECEIVED, invalid)  # type: ignore[arg-type]
        assert error.value.code == "SITE_CONTROL_INVALID"


def test_gate_release_timeout_and_stop_are_distinct() -> None:
    state = SiteState(Scenario.DELAYED_LOGIN)
    state.mark_running()
    released: list[GateOutcome] = []

    def wait_at_gate() -> None:
        released.append(state.gate_wait(time.monotonic() + 3.0))

    thread = threading.Thread(target=wait_at_gate)
    thread.start()
    assert state.wait_for(SiteEvent.RESPONSE_BLOCKED, 3000) is True
    assert state.snapshot().commits == 0
    state.release()
    thread.join(3.0)
    assert released == [GateOutcome.RELEASED]
    assert state.snapshot().response_blocked is False

    state.reset(scenario=Scenario.DELAYED_LOGIN)
    timed_out: list[GateOutcome] = []

    def wait_to_timeout() -> None:
        timed_out.append(state.gate_wait(time.monotonic() + 0.05))

    thread = threading.Thread(target=wait_to_timeout)
    thread.start()
    thread.join(3.0)
    assert timed_out == [GateOutcome.TIMED_OUT]
    assert state.snapshot().commits == 0

    state.reset(scenario=Scenario.DELAYED_LOGIN)
    stopped: list[GateOutcome] = []

    def wait_to_stop() -> None:
        stopped.append(state.gate_wait(time.monotonic() + 5.0))

    thread = threading.Thread(target=wait_to_stop)
    thread.start()
    assert state.wait_for(SiteEvent.RESPONSE_BLOCKED, 3000) is True
    state.mark_stopping()
    thread.join(3.0)
    assert stopped == [GateOutcome.STOPPED]
    assert state.snapshot().commits == 0


def test_overlay_and_release_semantics() -> None:
    blocked = SiteState(Scenario.BLOCKED_SUBMIT)
    blocked.mark_running()
    assert blocked.overlay_is_active() is True
    blocked.release()
    assert blocked.overlay_is_active() is False
    blocked.release()
    assert blocked.overlay_is_active() is False

    normal = SiteState(Scenario.NORMAL)
    normal.mark_running()
    normal.release()
    assert normal.overlay_is_active() is False
    assert normal.generation == 1

    stopped = SiteState(Scenario.NORMAL)
    with pytest.raises(LocalSiteError) as error:
        stopped.release()
    assert error.value.code == "SITE_NOT_RUNNING"


# ---------------------------------------------------------------------------- pages


def test_login_page_generation_is_deterministic_and_fixed() -> None:
    first = render_login_page(generation=7)
    second = render_login_page(generation=7)
    assert first == second
    assert first.count('name="username"') == 1
    assert first.count('name="password"') == 1
    assert first.count('name="generation"') == 1
    assert 'value="7"' in first
    assert 'data-testid="username"' in first
    assert 'data-testid="password"' in first
    assert 'data-testid="submit"' in first
    assert 'data-testid="login-error"' in first
    assert 'data-testid="overlay"' in first
    assert "登录" in first
    assert FIXTURE_USERNAME not in first
    assert FIXTURE_PASSWORD not in first
    assert "<script" not in first.lower()
    assert "http://" not in first
    assert "https://" not in first
    assert "src=" not in first
    assert "href=" not in first
    for scenario in Scenario:
        assert scenario.value not in first


def test_login_page_error_and_overlay_states_are_explicit() -> None:
    quiet = render_login_page(generation=3)
    assert 'class="error" hidden>' in quiet
    assert 'aria-hidden="true" hidden>' in quiet
    wrong = render_login_page(generation=3, error=True)
    assert 'class="error">' in wrong
    masked = render_login_page(generation=3, overlay=True)
    assert 'aria-hidden="true">' in masked
    assert render_login_page(generation=3, error=True, overlay=True).count(
        'class="error">'
    ) == 1
    with pytest.raises(ValueError):
        render_login_page(generation=0)


def test_dashboard_page_is_fixed_and_leaks_no_session() -> None:
    body = render_dashboard_page()
    assert body == render_dashboard_page()
    assert 'data-testid="welcome"' in body
    assert "登录成功" in body
    assert "session" not in body.lower()
    assert "<script" not in body.lower()
    assert "http://" not in body
    assert "https://" not in body


# ---------------------------------------------------------------------------- strict parsers


def test_form_parser_requires_exactly_three_canonical_fields() -> None:
    valid = b"username=test-user&password=test-password&generation=1"
    assert parse_form_body(valid) == {
        "username": "test-user",
        "password": "test-password",
        "generation": "1",
    }
    reordered = b"generation=1&password=test-password&username=test-user"
    assert parse_form_body(reordered) == parse_form_body(valid)
    assert parse_form_body(b"username=a+b&password=&generation=1") == {
        "username": "a b",
        "password": "",
        "generation": "1",
    }
    assert parse_form_body(b"username=a%26b&password=x&generation=1") == {
        "username": "a&b",
        "password": "x",
        "generation": "1",
    }
    rejected = [
        b"",
        b"username=a&password=b",
        b"username=a&username=b&password=c&generation=1",
        b"username=a&password=b&generation=1&extra=x",
        b"username=a&password=b&generation=1&",
        b"username=a&password=b&unknown=1",
        b"username=a&password=b&generation=1&generation=1",
        b"username=a&password=b&generation",
        b"username=%FF&password=b&generation=1",
        b"username=%G1&password=b&generation=1",
        b"username=%1&password=b&generation=1",
        b"username=a&password=b&generation=1&&",
    ]
    for body in rejected:
        assert parse_form_body(body) is None, body


def test_generation_parser_is_canonical_ascii_positive_decimal() -> None:
    assert parse_generation("1") == 1
    assert parse_generation("42") == 42
    for rejected in (
        "",
        "0",
        "01",
        "+1",
        "-1",
        " 1",
        "1 ",
        "1.0",
        "١",
        "1\n",
        b"1",
        None,
    ):
        assert parse_generation(rejected) is None, rejected


def test_content_length_parser_requires_one_decimal_value() -> None:
    assert parse_content_length(["0"]) == 0
    assert parse_content_length(["12"]) == 12
    assert parse_content_length(["00012"]) == 12
    for rejected in (
        None,
        [],
        ["1", "1"],
        ["-1"],
        ["+1"],
        [" 1"],
        ["1 "],
        ["abc"],
        ["1.0"],
        [""],
        ["1e3"],
    ):
        assert parse_content_length(rejected) is None, rejected


def test_content_type_is_the_fixed_form_type() -> None:
    assert _content_type_ok("application/x-www-form-urlencoded") is True
    assert (
        _content_type_ok("application/x-www-form-urlencoded; charset=utf-8") is True
    )
    assert _content_type_ok("Application/X-WWW-Form-Urlencoded") is True
    for rejected in (
        None,
        "",
        "text/plain",
        "multipart/form-data; boundary=x",
        "application/x-www-form-urlencoded; charset=latin-1",
        "application/x-www-form-urlencoded; boundary=x",
    ):
        assert _content_type_ok(rejected) is False, rejected


# ---------------------------------------------------------------------------- lifecycle


def test_origin_and_controls_before_start_and_after_stop() -> None:
    site = LocalTestSite(make_config())
    with pytest.raises(LocalSiteError) as error:
        _ = site.origin
    assert error.value.code == "SITE_NOT_RUNNING"
    with pytest.raises(LocalSiteError) as error:
        site.release()
    assert error.value.code == "SITE_NOT_RUNNING"

    before = site.stop()
    assert before.complete is True
    with pytest.raises(LocalSiteError) as error:
        site.start()
    assert error.value.code == "SITE_ALREADY_STOPPED"


def test_start_is_idempotent_and_stop_is_terminal_but_truthful() -> None:
    site = LocalTestSite(make_config())
    assert site.start() is site
    assert site.start() is site
    origin = site.origin
    assert origin.startswith("http://127.0.0.1:")
    port = int(origin.rsplit(":", 1)[1])
    assert 1 <= port <= 65535 and port != 80
    first = site.stop()
    assert first.complete is True
    second = site.stop()
    assert second.complete is True
    assert second == SiteStopResult(
        complete=True, active_requests=0, open_connections=0, live_threads=0
    )
    assert site.origin == origin
    with pytest.raises(LocalSiteError) as error:
        site.reset(scenario=Scenario.NORMAL)
    assert error.value.code == "SITE_NOT_RUNNING"


def test_context_manager_starts_and_releases_completely() -> None:
    with LocalTestSite(make_config()) as site:
        assert site.origin.startswith("http://127.0.0.1:")
        snapshot = site.snapshot_oracle()
        assert snapshot.running is True
    assert site.stop().complete is True
    assert not any(
        thread.name.startswith("test-site") for thread in threading.enumerate()
    )


def test_start_revalidates_a_tampered_stored_configuration() -> None:
    site = LocalTestSite(make_config())
    object.__setattr__(site._config, "request_timeout_ms", 0)
    with pytest.raises(LocalSiteError) as error:
        site.start()
    assert error.value.code == "SITE_CONFIG_INVALID"
    assert site.stop().complete is True


def test_partial_start_failure_releases_the_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = LocalTestSite(make_config())
    real_start = threading.Thread.start

    def fail_listener_start(self: threading.Thread) -> None:
        if self.name == "test-site-listener":
            raise RuntimeError("synthetic partial start failure")
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", fail_listener_start)
    with pytest.raises(LocalSiteError) as error:
        site.start()
    assert error.value.code == "SITE_START_FAILED"
    monkeypatch.setattr(threading.Thread, "start", real_start)
    assert site.stop().complete is True
    assert not any(
        thread.name.startswith("test-site") for thread in threading.enumerate()
    )


# ---------------------------------------------------------------------------- bounded ordinary errors


_MARKER = "AUDIT_SYNTHETIC_READ_FAILURE"


def _no_reachable_chain(error: BaseException) -> None:
    """The fixed diagnostic must not carry a reachable cause, context or raw marker."""
    assert isinstance(error, LocalSiteError)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert _MARKER not in str(error)
    assert _MARKER not in repr(error)


class Foreign:
    """A non-config object whose ``__class__`` property raises an ordinary exception."""

    @property
    def __class__(self) -> object:  # type: ignore[override]
        raise ValueError(_MARKER)


class ForeignGet:
    """A foreign object whose every attribute read raises an ordinary exception."""

    def __getattribute__(self, name: str) -> object:
        raise ValueError(_MARKER)


class HostileConfig(SiteConfig):
    """A genuine subclass whose ``scenario`` read raises an ordinary exception on demand."""

    armed = False

    def __getattribute__(self, name: str) -> object:
        if name == "scenario" and object.__getattribute__(self, "armed"):
            raise ValueError(_MARKER)
        return super().__getattribute__(name)


@pytest.mark.parametrize("target", ["constructor", "start"])
def test_a_hostile_configuration_read_is_a_fixed_error_with_no_reachable_chain(
    target: str,
) -> None:
    config = HostileConfig(
        scenario=Scenario.NORMAL,
        request_timeout_ms=1000,
        cleanup_timeout_ms=1000,
    )
    site: LocalTestSite | None = None
    if target == "start":
        site = LocalTestSite(config)
        site._config = config
    object.__setattr__(config, "armed", True)
    try:
        with pytest.raises(LocalSiteError) as error:
            if site is None:
                LocalTestSite(config)
            else:
                site.start()
    finally:
        if site is not None:
            assert site.stop().complete is True
    assert error.value.code == "SITE_CONFIG_INVALID"
    _no_reachable_chain(error.value)


@pytest.mark.parametrize("variant", [Foreign, ForeignGet], ids=["class", "getattr"])
@pytest.mark.parametrize("boundary", ["constructor", "start", "scenario", "budget"])
def test_a_foreign_configuration_object_is_a_fixed_error_without_a_reachable_chain(
    variant: type, boundary: str
) -> None:
    """Every caller-owned configuration type/classification read is bounded.

    A foreign object that explodes from ``__class__`` (or from every attribute read) must reach
    the fixed ``SITE_CONFIG_INVALID`` diagnostic instead of leaking its ordinary exception, and
    that diagnostic must carry no reachable cause, context or raw marker.
    """
    site: LocalTestSite | None = None
    try:
        with pytest.raises(LocalSiteError) as error:
            if boundary == "constructor":
                LocalTestSite(variant())  # type: ignore[arg-type]
            elif boundary == "start":
                site = LocalTestSite(make_config())
                site._config = variant()  # type: ignore[assignment]
                site.start()
            elif boundary == "scenario":
                SiteConfig(
                    scenario=variant(),  # type: ignore[arg-type]
                    request_timeout_ms=1000,
                    cleanup_timeout_ms=1000,
                )
            else:
                SiteConfig(
                    scenario=Scenario.NORMAL,
                    request_timeout_ms=variant(),  # type: ignore[arg-type]
                    cleanup_timeout_ms=1000,
                )
    finally:
        if site is not None:
            site._config = make_config()
            assert site.stop().complete is True
    assert error.value.code == "SITE_CONFIG_INVALID"
    _no_reachable_chain(error.value)


_SPOOF_HOOKS: list[str] = []


class _SpoofedMroMeta(type):
    """A metaclass whose ``__mro__`` read claims ``SiteConfig`` without a native relationship."""

    def __getattribute__(cls, name: str) -> object:
        if name == "__mro__":
            _SPOOF_HOOKS.append("mro")
            return (SiteConfig,)
        return type.__getattribute__(cls, name)


class _SpoofedMroConfig(metaclass=_SpoofedMroMeta):
    """A real foreign class with valid field values and a forged MRO read."""

    scenario = Scenario.NORMAL
    request_timeout_ms = 1000
    cleanup_timeout_ms = 1000


class _ForgedEqualityMeta(type):
    """A metaclass whose equality claims ``SiteConfig`` without a native relationship."""

    def __eq__(cls, other: object) -> bool:
        _SPOOF_HOOKS.append("class_equality")
        return other is SiteConfig

    __hash__ = type.__hash__


class _ForgedEqualityConfig(metaclass=_ForgedEqualityMeta):
    """A real foreign class with valid field values and truthy class equality."""

    scenario = Scenario.NORMAL
    request_timeout_ms = 1000
    cleanup_timeout_ms = 1000


@pytest.mark.parametrize("target", ["constructor", "start"])
def test_a_spoofed_metaclass_mro_cannot_authorize_a_foreign_configuration(
    target: str,
) -> None:
    """Caller-owned ``__mro__`` metadata is never a native configuration relationship."""
    _SPOOF_HOOKS.clear()
    assert _SpoofedMroConfig.__mro__ == (SiteConfig,)
    _SPOOF_HOOKS.clear()
    config = _SpoofedMroConfig()
    site: LocalTestSite | None = None
    try:
        with pytest.raises(LocalSiteError) as error:
            if target == "constructor":
                LocalTestSite(config)  # type: ignore[arg-type]
            else:
                site = LocalTestSite(make_config())
                site._config = config  # type: ignore[assignment]
                site.start()
    finally:
        if site is not None:
            site._config = make_config()
            assert site.stop().complete is True
    assert issubclass(type(config), SiteConfig) is False
    assert _SPOOF_HOOKS == []
    assert error.value.code == "SITE_CONFIG_INVALID"
    _no_reachable_chain(error.value)


@pytest.mark.parametrize("target", ["constructor", "start"])
def test_a_forged_class_equality_cannot_authorize_a_foreign_configuration(
    target: str,
) -> None:
    """Caller-owned class ``__eq__`` is never a native configuration relationship."""
    _SPOOF_HOOKS.clear()
    assert _ForgedEqualityConfig == SiteConfig
    _SPOOF_HOOKS.clear()
    config = _ForgedEqualityConfig()
    site: LocalTestSite | None = None
    try:
        with pytest.raises(LocalSiteError) as error:
            if target == "constructor":
                LocalTestSite(config)  # type: ignore[arg-type]
            else:
                site = LocalTestSite(make_config())
                site._config = config  # type: ignore[assignment]
                site.start()
    finally:
        if site is not None:
            site._config = make_config()
            assert site.stop().complete is True
    assert issubclass(type(config), SiteConfig) is False
    assert _SPOOF_HOOKS == []
    assert error.value.code == "SITE_CONFIG_INVALID"
    _no_reachable_chain(error.value)


class GenuineConfigSubclass(SiteConfig):
    """A real configuration subclass that keeps the inherited strict validation."""


@pytest.mark.parametrize(
    "config_type", [SiteConfig, GenuineConfigSubclass], ids=["site-config", "subclass"]
)
def test_a_genuine_configuration_and_subclass_are_accepted(
    config_type: type[SiteConfig],
) -> None:
    """The native type relationship keeps the ordinary configuration and subclass controls."""
    site = LocalTestSite(
        config_type(
            scenario=Scenario.DELAYED_LOGIN,
            request_timeout_ms=1000,
            cleanup_timeout_ms=1000,
        )
    )
    assert issubclass(config_type, SiteConfig) is True
    site.start()
    try:
        snapshot = site.snapshot_oracle()
        assert snapshot.running is True
        assert snapshot.scenario is Scenario.DELAYED_LOGIN
    finally:
        assert site.stop().complete is True


@pytest.mark.parametrize(
    "control", ["reset-scenario", "wait-event", "wait-budget"]
)
def test_a_foreign_host_control_is_a_fixed_error_and_changes_nothing(
    control: str,
) -> None:
    """The existing reset/wait controls classify their inputs without touching caller code."""
    site = LocalTestSite(make_config()).start()
    before = site.snapshot_oracle()
    try:
        with pytest.raises(LocalSiteError) as error:
            if control == "reset-scenario":
                site.reset(scenario=Foreign())  # type: ignore[arg-type]
            elif control == "wait-event":
                site.wait_for(event=Foreign(), timeout_ms=100)  # type: ignore[arg-type]
            else:
                site.wait_for(
                    event=SiteEvent.LOGIN_RECEIVED,
                    timeout_ms=Foreign(),  # type: ignore[arg-type]
                )
        assert site.snapshot_oracle() == before
    finally:
        assert site.stop().complete is True
    assert error.value.code == "SITE_CONTROL_INVALID"
    _no_reachable_chain(error.value)


def test_a_partial_start_failure_has_no_reachable_exception_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = LocalTestSite(make_config())
    real_start = threading.Thread.start

    def fail_listener_start(self: threading.Thread) -> None:
        if self.name == "test-site-listener":
            raise RuntimeError(_MARKER)
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", fail_listener_start)
    with pytest.raises(LocalSiteError) as error:
        site.start()
    monkeypatch.setattr(threading.Thread, "start", real_start)
    assert site.stop().complete is True
    assert error.value.code == "SITE_START_FAILED"
    _no_reachable_chain(error.value)


@pytest.mark.parametrize(
    "kind", ["keyboard_interrupt", "system_exit", "cancelled_error"]
)
def test_base_exception_identity_and_cleanup_are_preserved(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind == "keyboard_interrupt":
        sentinel: BaseException = KeyboardInterrupt("synthetic interrupt")
    elif kind == "system_exit":
        sentinel = SystemExit(3)
    else:
        sentinel = asyncio.CancelledError("synthetic cancellation")
    site = LocalTestSite(make_config())
    real_start = threading.Thread.start

    def raise_sentinel(self: threading.Thread) -> None:
        if self.name == "test-site-listener":
            raise sentinel
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", raise_sentinel)
    with pytest.raises(type(sentinel)) as error:
        site.start()
    monkeypatch.setattr(threading.Thread, "start", real_start)
    assert error.value is sentinel
    assert site.stop().complete is True
    assert not any(
        thread.name.startswith("test-site") for thread in threading.enumerate()
    )
