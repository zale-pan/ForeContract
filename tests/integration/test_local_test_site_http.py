"""Real HTTP tests over a real ephemeral loopback socket.

Every test starts the actual site on ``127.0.0.1:0`` and talks to it through ``http.client`` or a
raw socket. The assertions are written from the contract's own literal status codes and the
host-side oracle snapshot: no site helper computes an expected result for the test. The same file
proves the strict request rejections have no business side effects, that stop wakes a blocked
gate without committing, and that the CLI starts and cleans up a real listener.
"""

from __future__ import annotations

import importlib
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_site.server as site_server
from test_site import (
    LocalSiteError,
    LocalTestSite,
    Scenario,
    SiteConfig,
    SiteEvent,
    SiteOracleSnapshot,
    SiteStopResult,
)
from test_site.state import SiteState

USERNAME = "test-user"
PASSWORD = "test-password"
WRONG_USERNAME = "wrong-user"
WRONG_PASSWORD = "wrong-password"


# ---------------------------------------------------------------------------- helpers


def running_site(
    scenario: Scenario,
    *,
    request_timeout_ms: int = 5000,
    cleanup_timeout_ms: int = 3000,
) -> LocalTestSite:
    return LocalTestSite(
        SiteConfig(
            scenario=scenario,
            request_timeout_ms=request_timeout_ms,
            cleanup_timeout_ms=cleanup_timeout_ms,
        )
    ).start()


@pytest.fixture()
def site_factory() -> Iterator[Any]:
    created: list[LocalTestSite] = []

    def factory(
        scenario: Scenario,
        *,
        request_timeout_ms: int = 5000,
        cleanup_timeout_ms: int = 3000,
    ) -> LocalTestSite:
        site = running_site(
            scenario,
            request_timeout_ms=request_timeout_ms,
            cleanup_timeout_ms=cleanup_timeout_ms,
        )
        created.append(site)
        return site

    yield factory
    for site in created:
        result = site.stop()
        assert result.complete, result


def origin_parts(site: LocalTestSite) -> tuple[str, int]:
    parts = urlsplit(site.origin)
    assert parts.hostname == "127.0.0.1"
    assert parts.port is not None and parts.port > 0
    return parts.hostname, parts.port


def header_values(headers: list[tuple[str, str]], name: str) -> list[str]:
    wanted = name.lower()
    return [value for key, value in headers if key.lower() == wanted]


def request(
    site: LocalTestSite,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
) -> tuple[int, list[tuple[str, str]], bytes]:
    import http.client

    host, port = origin_parts(site)
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.getheaders(), response.read()
    finally:
        connection.close()


def raw_exchange(
    site: LocalTestSite, payload: bytes, *, timeout: float = 5.0
) -> bytes:
    host, port = origin_parts(site)
    connection = socket.create_connection((host, port), timeout=timeout)
    connection.settimeout(timeout)
    try:
        connection.sendall(payload)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = connection.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        connection.close()


def raw_status(raw: bytes) -> int:
    if not raw:
        return 0
    line = raw.split(b"\r\n", 1)[0]
    return int(line.split(b" ")[1])


def form_body(
    *, generation: object = 1, username: str = USERNAME, password: str = PASSWORD
) -> bytes:
    return urlencode(
        {
            "username": username,
            "password": password,
            "generation": str(generation),
        }
    ).encode("ascii")


def form_headers(body: bytes) -> dict[str, str]:
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "Content-Length": str(len(body)),
    }


def post_login(
    site: LocalTestSite,
    *,
    generation: object = 1,
    username: str = USERNAME,
    password: str = PASSWORD,
    timeout: float = 5.0,
) -> tuple[int, list[tuple[str, str]], bytes]:
    body = form_body(generation=generation, username=username, password=password)
    return request(
        site,
        "POST",
        "/login",
        body=body,
        headers=form_headers(body),
        timeout=timeout,
    )


def raw_post(
    body: bytes,
    *,
    content_type: str | None = "application/x-www-form-urlencoded",
    content_length: list[str] | None = None,
    transfer_encoding: str | None = None,
    extra: list[str] | None = None,
) -> bytes:
    lines = [b"POST /login HTTP/1.1", b"Host: 127.0.0.1", b"Connection: close"]
    if content_type is not None:
        lines.append(f"Content-Type: {content_type}".encode("ascii"))
    for value in content_length if content_length is not None else [str(len(body))]:
        lines.append(f"Content-Length: {value}".encode("ascii"))
    if transfer_encoding is not None:
        lines.append(f"Transfer-Encoding: {transfer_encoding}".encode("ascii"))
    for line in extra or []:
        lines.append(line.encode("ascii"))
    return b"\r\n".join(lines) + b"\r\n\r\n" + body


def wait_in_thread(
    site: LocalTestSite, event: SiteEvent, timeout_ms: int = 5000
) -> threading.Thread:
    holder: list[bool] = []

    def wait() -> None:
        holder.append(site.wait_for(event=event, timeout_ms=timeout_ms))

    thread = threading.Thread(target=wait)
    thread.start()
    thread.join((timeout_ms / 1000.0) + 1.0)
    assert thread.is_alive() is False
    assert holder == [True], holder
    return thread


def site_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name.startswith("test-site")]


def wait_for_request_retirement(
    site: LocalTestSite, *, timeout_s: float = 2.0
) -> SiteOracleSnapshot:
    """Wait boundedly for every request already counted by the server to retire.

    A complete HTTP response only proves the client received bytes; the handler's own
    ``end_request`` step can still be outstanding. Reading the true ``active_requests`` counter
    until it reaches zero synchronises the host with that actual retirement fact. The counter is
    never forced and the wait has one absolute deadline, so a real timing failure fails loudly
    instead of being hidden by a fixed sleep or a retry until green.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        snapshot = site.snapshot_oracle()
        if snapshot.active_requests == 0:
            return snapshot
        assert time.monotonic() < deadline, snapshot
        time.sleep(0.005)


# ---------------------------------------------------------------------------- routes


def test_fixed_routes_statuses_content_types_and_cookie(site_factory: Any) -> None:
    site = site_factory(Scenario.NORMAL)
    status, headers, body = request(site, "GET", "/")
    assert status == 303
    assert header_values(headers, "location") == ["/login"]
    assert body == b""

    status, headers, body = request(site, "GET", "/login")
    assert status == 200
    assert header_values(headers, "content-type") == ["text/html; charset=utf-8"]
    assert body.count(b'name="username"') == 1
    assert body.count(b'name="password"') == 1
    assert body.count(b'name="generation"') == 1
    assert b'value="1"' in body
    assert b'data-testid="submit"' in body
    assert b"<script" not in body.lower()
    assert b"http://" not in body and b"https://" not in body

    status, headers, body = request(site, "GET", "/dashboard")
    assert status == 303
    assert header_values(headers, "location") == ["/login"]

    status, headers, body = post_login(site)
    assert status == 303
    assert header_values(headers, "location") == ["/dashboard"]
    cookies = header_values(headers, "set-cookie")
    assert len(cookies) == 1
    assert cookies[0].startswith("test_site_session=")
    assert "Path=/" in cookies[0]
    assert "HttpOnly" in cookies[0]
    assert "SameSite=Lax" in cookies[0]
    assert body == b""

    cookie = cookies[0].split(";", 1)[0]
    status, headers, body = request(
        site, "GET", "/dashboard", headers={"Cookie": cookie}
    )
    assert status == 200
    assert header_values(headers, "content-type") == ["text/html; charset=utf-8"]
    assert b'data-testid="welcome"' in body
    assert "登录成功".encode() in body

    for path in ("/oracle", "/reset", "/release", "/admin", "/static/", "/../secret"):
        status, _, body = request(site, "GET", path)
        assert status == 404, path
        assert body == b"not found\n"
    status, _, _ = request(site, "POST", "/release")
    assert status == 404
    status, _, body = request(site, "GET", "/login?generation=1")
    assert status == 404
    assert body == b"not found\n"

    for method, path in (
        ("POST", "/dashboard"),
        ("PUT", "/login"),
        ("OPTIONS", "/login"),
    ):
        status, headers, body = request(site, method, path)
        assert status == 405, (method, path)
        assert header_values(headers, "content-type") == ["text/plain; charset=utf-8"]
        assert body == b"method not allowed\n"
    status, headers, _ = request(site, "HEAD", "/login")
    assert status == 405
    assert header_values(headers, "content-type") == ["text/plain; charset=utf-8"]

    snapshot = wait_for_request_retirement(site)
    assert snapshot.login_requests == 1
    assert snapshot.auth_successes == 1
    assert snapshot.auth_failures == 0
    assert snapshot.commits == 1
    assert snapshot.active_requests == 0
    assert snapshot.authenticated_sessions == 1


def test_wrong_credentials_fail_immediately_in_every_mode(site_factory: Any) -> None:
    for scenario in Scenario:
        site = site_factory(scenario)
        status, headers, body = post_login(
            site, username=WRONG_USERNAME, password=WRONG_PASSWORD
        )
        assert status == 200, scenario
        assert header_values(headers, "content-type") == ["text/html; charset=utf-8"]
        assert b'data-testid="login-error"' in body
        assert b'class="error">' in body
        assert WRONG_PASSWORD.encode("ascii") not in body
        snapshot = site.snapshot_oracle()
        assert snapshot.login_requests == 1, scenario
        assert snapshot.auth_successes == 0, scenario
        assert snapshot.auth_failures == 1, scenario
        assert snapshot.commits == 0, scenario
        assert snapshot.authenticated_sessions == 0, scenario
        assert snapshot.response_blocked is False, scenario
        assert snapshot.active_requests == 0, scenario
        assert site.wait_for(event=SiteEvent.RESPONSE_BLOCKED, timeout_ms=20) is False

        for cookie in (
            "test_site_session=forged",
            "test_site_session=",
            "other=value",
        ):
            status, _, _ = request(
                site, "GET", "/dashboard", headers={"Cookie": cookie}
            )
            assert status == 303, (scenario, cookie)
        assert site.snapshot_oracle() == snapshot


# ---------------------------------------------------------------------------- gates


def test_delayed_login_release_commits_exactly_once(site_factory: Any) -> None:
    site = site_factory(Scenario.DELAYED_LOGIN, request_timeout_ms=8000)
    body = form_body()
    result: list[bytes] = []

    def post() -> None:
        result.append(raw_exchange(site, raw_post(body), timeout=10.0))

    thread = threading.Thread(target=post)
    thread.start()
    wait_in_thread(site, SiteEvent.LOGIN_RECEIVED)
    wait_in_thread(site, SiteEvent.RESPONSE_BLOCKED)
    before = site.snapshot_oracle()
    assert before.login_requests == 1
    assert before.auth_successes == 0
    assert before.commits == 0
    assert before.active_requests == 1
    assert before.response_blocked is True
    assert thread.is_alive() is True

    site.release()
    thread.join(5.0)
    assert thread.is_alive() is False
    assert raw_status(result[0]) == 303
    after = site.snapshot_oracle()
    assert after.login_requests == 1
    assert after.auth_successes == 1
    assert after.commits == 1
    assert after.authenticated_sessions == 1
    assert after.response_blocked is False
    assert after.active_requests == 0


def test_delayed_login_own_deadline_is_503_without_commit(site_factory: Any) -> None:
    site = site_factory(Scenario.DELAYED_LOGIN, request_timeout_ms=250)
    body = form_body()
    result: list[bytes] = []

    def post() -> None:
        result.append(raw_exchange(site, raw_post(body), timeout=5.0))

    thread = threading.Thread(target=post)
    thread.start()
    wait_in_thread(site, SiteEvent.RESPONSE_BLOCKED)
    thread.join(5.0)
    assert thread.is_alive() is False
    assert raw_status(result[0]) == 503
    snapshot = site.snapshot_oracle()
    assert snapshot.login_requests == 1
    assert snapshot.auth_successes == 0
    assert snapshot.commits == 0
    assert snapshot.authenticated_sessions == 0
    assert snapshot.response_blocked is False


def test_commit_then_stall_commits_before_response_and_release_keeps_one(
    site_factory: Any,
) -> None:
    site = site_factory(Scenario.COMMIT_THEN_STALL, request_timeout_ms=8000)
    body = form_body()
    result: list[bytes] = []

    def post() -> None:
        result.append(raw_exchange(site, raw_post(body), timeout=10.0))

    thread = threading.Thread(target=post)
    thread.start()
    wait_in_thread(site, SiteEvent.LOGIN_COMMITTED)
    wait_in_thread(site, SiteEvent.RESPONSE_BLOCKED)
    before = site.snapshot_oracle()
    assert before.login_requests == 1
    assert before.auth_successes == 1
    assert before.commits == 1
    assert before.authenticated_sessions == 1
    assert before.response_blocked is True
    assert before.active_requests == 1
    assert thread.is_alive() is True

    site.release()
    thread.join(5.0)
    assert thread.is_alive() is False
    assert raw_status(result[0]) == 303
    after = site.snapshot_oracle()
    assert after.commits == 1
    assert after.authenticated_sessions == 1
    assert after.response_blocked is False

    expiring = site_factory(Scenario.COMMIT_THEN_STALL, request_timeout_ms=250)
    expired_text = raw_exchange(expiring, raw_post(form_body()), timeout=5.0)
    assert expired_text == b""
    expired = expiring.snapshot_oracle()
    assert expired.commits == 1
    assert expired.authenticated_sessions == 1


def test_stop_wakes_blocked_login_without_commit_and_cleans_up(
    site_factory: Any,
) -> None:
    site = site_factory(Scenario.DELAYED_LOGIN, request_timeout_ms=10000)
    body = form_body()
    result: list[bytes] = []

    def post() -> None:
        result.append(raw_exchange(site, raw_post(body), timeout=8.0))

    thread = threading.Thread(target=post)
    thread.start()
    wait_in_thread(site, SiteEvent.RESPONSE_BLOCKED)
    stopped = site.stop()
    thread.join(8.0)
    assert thread.is_alive() is False
    assert stopped.complete is True, stopped
    assert stopped.active_requests == 0
    assert stopped.open_connections == 0
    assert stopped.live_threads == 0
    snapshot = site.snapshot_oracle()
    assert snapshot.running is False
    assert snapshot.commits == 0
    assert snapshot.authenticated_sessions == 0
    assert raw_status(result[0]) == 503
    assert site_threads() == []
    host, port = urlsplit(site.origin).hostname, urlsplit(site.origin).port
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        assert probe.connect_ex((host, port)) != 0
    finally:
        probe.close()
    assert site.stop().complete is True


# ---------------------------------------------------------------------------- generation


def test_reset_clears_counts_revokes_sessions_and_rejects_stale_forms(
    site_factory: Any,
) -> None:
    site = site_factory(Scenario.NORMAL)
    status, headers, _ = post_login(site)
    assert status == 303
    cookie = header_values(headers, "set-cookie")[0].split(";", 1)[0]
    assert site.snapshot_oracle().commits == 1
    assert site.snapshot_oracle().authenticated_sessions == 1

    generation = site.reset(scenario=Scenario.NORMAL)
    assert generation == 2
    after_reset = site.snapshot_oracle()
    assert after_reset.generation == 2
    assert after_reset.scenario is Scenario.NORMAL
    assert after_reset.login_requests == 0
    assert after_reset.auth_failures == 0
    assert after_reset.commits == 0
    assert after_reset.authenticated_sessions == 0
    assert after_reset.response_blocked is False

    status, _, _ = request(site, "GET", "/dashboard", headers={"Cookie": cookie})
    assert status == 303
    status, _, body = request(site, "GET", "/login")
    assert status == 200
    assert b'value="2"' in body

    status, _, _ = post_login(site, generation=1)
    assert status == 400
    stale = site.snapshot_oracle()
    assert stale.login_requests == 0
    assert stale.commits == 0
    assert stale.authenticated_sessions == 0

    status, _, _ = post_login(site, generation=2, username=WRONG_USERNAME)
    assert status == 200
    wrong = site.snapshot_oracle()
    assert wrong.login_requests == 1
    assert wrong.auth_failures == 1
    assert wrong.commits == 0


def test_busy_reset_is_atomic_and_does_not_change_state(site_factory: Any) -> None:
    site = site_factory(Scenario.DELAYED_LOGIN, request_timeout_ms=8000)
    body = form_body()
    result: list[bytes] = []

    def post() -> None:
        result.append(raw_exchange(site, raw_post(body), timeout=8.0))

    thread = threading.Thread(target=post)
    thread.start()
    wait_in_thread(site, SiteEvent.RESPONSE_BLOCKED)
    before = site.snapshot_oracle()
    with pytest.raises(LocalSiteError) as error:
        site.reset(scenario=Scenario.NORMAL)
    assert error.value.code == "SITE_BUSY"
    assert site.snapshot_oracle() == before
    site.release()
    thread.join(5.0)
    assert raw_status(result[0]) == 303
    assert site.snapshot_oracle().commits == 1


# ---------------------------------------------------------------------------- strict HTTP


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            (
                b"POST /login HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Content-Type: application/x-www-form-urlencoded\r\n\r\n"
            ),
            400,
        ),
        (
            raw_post(
                b"x",
                content_length=["1", "1"],
            ),
            400,
        ),
        (raw_post(b"x", content_length=["abc"]), 400),
        (raw_post(b"", content_length=["-1"]), 400),
        (raw_post(b"", content_length=["+1"]), 400),
        (raw_post(b"", transfer_encoding="chunked"), 400),
        (raw_post(b"", content_type="text/plain"), 415),
        (raw_post(b"", content_type="application/json"), 415),
        (
            raw_post(
                b"",
                content_type="application/x-www-form-urlencoded; charset=latin-1",
            ),
            415,
        ),
        (raw_post(b"", content_length=["4097"]), 413),
        (raw_post(b"username=x&password=y"), 400),
        (
            raw_post(b"username=x&username=y&password=z&generation=1"),
            400,
        ),
        (
            raw_post(b"username=x&password=y&generation=1&extra=z"),
            400,
        ),
        (raw_post(form_body(generation=0)), 400),
        (raw_post(form_body(generation="01")), 400),
        (raw_post(form_body(generation="+1")), 400),
        (raw_post(form_body(generation=2)), 400),
        (raw_post(b"username=%FF&password=y&generation=1"), 400),
        (raw_post(b"username=%G1&password=y&generation=1"), 400),
    ],
)
def test_strict_http_rejections_have_no_business_side_effects(
    site_factory: Any, payload: bytes, expected: int
) -> None:
    site = site_factory(Scenario.NORMAL)
    response = raw_exchange(site, payload)
    assert raw_status(response) == expected, response[:200]
    snapshot = site.snapshot_oracle()
    assert snapshot == SiteOracleSnapshot(
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
    assert PASSWORD.encode("ascii") not in response
    assert b"Traceback" not in response


def test_blocked_submit_overlay_does_not_block_direct_http(site_factory: Any) -> None:
    site = site_factory(Scenario.BLOCKED_SUBMIT)
    status, _, body = request(site, "GET", "/login")
    assert status == 200
    assert b'aria-hidden="true">' in body
    status, headers, _ = post_login(site)
    assert status == 303
    assert header_values(headers, "location") == ["/dashboard"]
    snapshot = site.snapshot_oracle()
    assert snapshot.login_requests == 1
    assert snapshot.auth_successes == 1
    assert snapshot.commits == 1


def test_stop_is_bounded_for_a_short_body_and_reports_real_release(
    site_factory: Any,
) -> None:
    site = site_factory(
        Scenario.NORMAL, request_timeout_ms=10000, cleanup_timeout_ms=2000
    )
    host, port = origin_parts(site)
    incomplete = socket.create_connection((host, port), timeout=5)
    incomplete.sendall(
        b"POST /login HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/x-www-form-urlencoded\r\n"
        b"Content-Length: 100\r\n\r\n"
        b"short"
    )
    deadline = time.monotonic() + 2.0
    while site.snapshot_oracle().active_requests == 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    started = time.monotonic()
    result = site.stop()
    elapsed = time.monotonic() - started
    incomplete.close()
    assert elapsed < 2.5
    assert result.complete is True, result
    assert result.active_requests == 0
    assert result.open_connections == 0
    assert result.live_threads == 0
    assert site_threads() == []


def test_two_instances_have_independent_state_and_events(site_factory: Any) -> None:
    first = site_factory(Scenario.NORMAL)
    second = site_factory(Scenario.NORMAL)
    assert first.origin != second.origin
    status, _, _ = post_login(first)
    assert status == 303
    first_snapshot = first.snapshot_oracle()
    second_snapshot = second.snapshot_oracle()
    assert first_snapshot.login_requests == 1
    assert first_snapshot.commits == 1
    assert second_snapshot.login_requests == 0
    assert second_snapshot.auth_successes == 0
    assert second_snapshot.commits == 0
    assert second_snapshot.authenticated_sessions == 0
    assert second.wait_for(event=SiteEvent.LOGIN_RECEIVED, timeout_ms=20) is False
    assert first.wait_for(event=SiteEvent.LOGIN_RECEIVED, timeout_ms=20) is True


def test_duplicate_post_is_not_deduplicated(site_factory: Any) -> None:
    site = site_factory(Scenario.NORMAL)
    first = post_login(site)
    second = post_login(site)
    assert first[0] == 303 and second[0] == 303
    first_cookie = header_values(first[1], "set-cookie")[0]
    second_cookie = header_values(second[1], "set-cookie")[0]
    assert first_cookie != second_cookie
    snapshot = site.snapshot_oracle()
    assert snapshot.login_requests == 2
    assert snapshot.auth_successes == 2
    assert snapshot.commits == 2
    assert snapshot.authenticated_sessions == 2


# ---------------------------------------------------------------------------- request deadline


def fragmented_exchange(
    site: LocalTestSite,
    first: bytes,
    second: bytes,
    *,
    gap_s: float,
    timeout: float = 5.0,
) -> bytes:
    """Send one request in two pieces around a deliberate pause, then read the response."""
    host, port = origin_parts(site)
    connection = socket.create_connection((host, port), timeout=timeout)
    connection.settimeout(timeout)
    try:
        connection.sendall(first)
        if gap_s:
            time.sleep(gap_s)
        connection.sendall(second)
        chunks: list[bytes] = []
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        connection.close()


@pytest.mark.parametrize("gap_s", [0.0, 0.02, 0.12], ids=["0.0", "0.02", "0.12"])
def test_a_legal_fragmented_body_within_the_budget_still_commits(
    site_factory: Any, gap_s: float
) -> None:
    site = site_factory(Scenario.NORMAL, request_timeout_ms=600)
    body = form_body()
    headers = raw_post(body)[: -len(body)]
    response = fragmented_exchange(
        site, headers + body[:8], body[8:], gap_s=gap_s
    )
    assert raw_status(response) == 303, response[:200]
    assert b"Set-Cookie" in response
    snapshot = site.snapshot_oracle()
    assert snapshot.login_requests == 1
    assert snapshot.auth_successes == 1
    assert snapshot.commits == 1
    assert snapshot.authenticated_sessions == 1


def test_slow_headers_cannot_reset_the_request_deadline(site_factory: Any) -> None:
    site = site_factory(Scenario.NORMAL, request_timeout_ms=200)
    body = form_body()
    host, port = origin_parts(site)
    connection = socket.create_connection((host, port), timeout=5)
    connection.settimeout(5)
    chunks: list[bytes] = []
    try:
        try:
            connection.sendall(b"POST /login HTTP/1.1\r\n")
            time.sleep(0.11)
            connection.sendall(b"Host: 127.0.0.1\r\n")
            time.sleep(0.11)
            connection.sendall(
                b"Content-Type: application/x-www-form-urlencoded\r\n"
            )
            time.sleep(0.11)
            connection.sendall(
                b"Content-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\n\r\n"
                + body
            )
        except OSError:
            pass
        while True:
            try:
                chunk = connection.recv(65536)
            except (TimeoutError, OSError):
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        connection.close()
    response = b"".join(chunks)
    # 330ms of header delays is past the 200ms budget: no later phase may reset the deadline.
    assert raw_status(response) != 303, response[:200]
    assert b"Set-Cookie" not in response
    snapshot = site.snapshot_oracle()
    assert snapshot.login_requests == 0
    assert snapshot.auth_successes == 0
    assert snapshot.auth_failures == 0
    assert snapshot.commits == 0
    assert snapshot.authenticated_sessions == 0
    assert snapshot.active_requests == 0
    assert site.wait_for(event=SiteEvent.RESPONSE_BLOCKED, timeout_ms=20) is False


def test_a_request_that_expires_during_render_cannot_send_a_normal_response(
    site_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed deadline is enforced again before any normal response is written."""
    site = site_factory(Scenario.NORMAL, request_timeout_ms=30)
    original = site_server.render_login_page
    entered = threading.Event()
    release = threading.Event()

    def slow_render(**kwargs: object) -> str:
        entered.set()
        release.wait(2.0)
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(site_server, "render_login_page", slow_render)
    outcome: list[tuple[int, list[tuple[str, str]], bytes]] = []

    def client() -> None:
        outcome.append(request(site, "GET", "/login"))

    thread = threading.Thread(target=client)
    thread.start()
    try:
        assert entered.wait(5.0) is True
        time.sleep(0.09)
        release.set()
        thread.join(5.0)
    finally:
        release.set()
        thread.join(5.0)
    assert thread.is_alive() is False
    assert len(outcome) == 1
    status, headers, body = outcome[0]
    assert status == 503
    assert header_values(headers, "set-cookie") == []
    assert body == b"service unavailable\n"
    snapshot = site.snapshot_oracle()
    assert snapshot.login_requests == 0
    assert snapshot.auth_successes == 0
    assert snapshot.commits == 0
    assert snapshot.authenticated_sessions == 0
    assert snapshot.active_requests == 0


def test_a_commit_entry_delay_cannot_commit_after_expiry(
    site_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deadline is rechecked inside the commit's coordination boundary, so no commit happens."""
    site = site_factory(Scenario.NORMAL, request_timeout_ms=30)
    real_commit = SiteState.commit_session
    entered = threading.Event()
    release = threading.Event()

    def slow_commit(state: SiteState) -> str | None:
        entered.set()
        release.wait(2.0)
        return real_commit(state)

    monkeypatch.setattr(SiteState, "commit_session", slow_commit)
    outcome: list[tuple[int, list[tuple[str, str]], bytes]] = []

    def client() -> None:
        outcome.append(post_login(site))

    thread = threading.Thread(target=client)
    thread.start()
    try:
        assert entered.wait(5.0) is True
        time.sleep(0.09)
        release.set()
        thread.join(5.0)
    finally:
        release.set()
        thread.join(5.0)
    assert thread.is_alive() is False
    assert len(outcome) == 1
    status, headers, body = outcome[0]
    assert status == 503
    assert header_values(headers, "set-cookie") == []
    assert body == b"service unavailable\n"
    snapshot = site.snapshot_oracle()
    assert snapshot.login_requests == 1
    assert snapshot.auth_successes == 0
    assert snapshot.auth_failures == 0
    assert snapshot.commits == 0
    assert snapshot.authenticated_sessions == 0
    assert snapshot.active_requests == 0


# ---------------------------------------------------------------------------- cleanup budget


@pytest.mark.parametrize("budget_ms", [1, 5], ids=["1", "5"])
def test_short_cleanup_budget_returns_within_its_total_budget(
    site_factory: Any, budget_ms: int
) -> None:
    site = site_factory(Scenario.NORMAL, cleanup_timeout_ms=budget_ms)
    started = time.monotonic()
    result = site.stop()
    elapsed = time.monotonic() - started
    assert isinstance(result, SiteStopResult)
    assert elapsed <= budget_ms / 1000.0 + 0.05, elapsed
    if result.complete:
        assert result.active_requests == 0
        assert result.open_connections == 0
        assert result.live_threads == 0
    # Let any still-exiting owned thread finish, then prove the same instance can report the
    # truthful complete release without a new budget being invented for it.
    deadline = time.monotonic() + 5.0
    while site_threads() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert site_threads() == []
    again = site.stop()
    assert again == SiteStopResult(
        complete=True, active_requests=0, open_connections=0, live_threads=0
    )


def test_cleanup_with_a_full_budget_proves_complete_release(site_factory: Any) -> None:
    site = site_factory(Scenario.NORMAL, cleanup_timeout_ms=1000)
    host, port = origin_parts(site)
    started = time.monotonic()
    result = site.stop()
    elapsed = time.monotonic() - started
    assert elapsed <= 1.05, elapsed
    assert result == SiteStopResult(
        complete=True, active_requests=0, open_connections=0, live_threads=0
    )
    assert site_threads() == []
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        assert probe.connect_ex((host, port)) != 0
    finally:
        probe.close()


@pytest.mark.parametrize("fault_site", ["instance", "state", "tracker"])
def test_stop_returns_within_its_budget_under_internal_coordination_contention(
    site_factory: Any, fault_site: str
) -> None:
    """Every internal coordination acquisition is bounded by the one cleanup deadline."""
    site = site_factory(Scenario.NORMAL, cleanup_timeout_ms=5)
    locks = {
        "instance": site._lock,
        "state": site._state._lock,
        "tracker": site._server._track_lock,
    }
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with locks[fault_site]:
            held.set()
            release.wait(2.0)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(5.0) is True
    started = time.monotonic()
    try:
        result = site.stop()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        holder.join(5.0)
    assert holder.is_alive() is False
    # The owner holds the lock for far longer than the 5ms budget; the call must still return.
    assert elapsed < 0.1, elapsed
    assert result.complete is False
    assert result.active_requests == 0
    assert result.open_connections == 0

    # After the owner releases, a subsequent bounded attempt stays safe and becomes truthful.
    deadline = time.monotonic() + 5.0
    while site_threads() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert site_threads() == []
    assert site.stop() == SiteStopResult(
        complete=True, active_requests=0, open_connections=0, live_threads=0
    )


# ---------------------------------------------------------------------------- scope and CLI


def test_pages_have_no_external_resource_or_control_surface(site_factory: Any) -> None:
    site = site_factory(Scenario.NORMAL)
    _, _, login = request(site, "GET", "/login")
    for token in (b"http://", b"https://", b"<script", b"src=", b"href="):
        assert token not in login.lower()
    for path in (
        "/oracle",
        "/reset",
        "/release",
        "/admin",
        "/control",
        "/scenario",
        "/static/app.js",
    ):
        status, _, _ = request(site, "GET", path)
        assert status == 404, path
    status, _, _ = request(site, "POST", "/reset")
    assert status == 404


def test_cli_help_and_argument_errors_never_bind(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    base = [sys.executable, "-B", "-m", "test_site"]

    help_run = subprocess.run(
        [*base, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "scenario" in help_run.stdout
    assert "http://" not in help_run.stdout

    invalid_runs = [
        [*base],
        [*base, "--scenario", "bogus", "--request-timeout-ms", "1000",
         "--cleanup-timeout-ms", "1000"],
        [*base, "--scenario", "normal", "--request-timeout-ms", "0",
         "--cleanup-timeout-ms", "1000"],
        [*base, "--scenario", "normal", "--request-timeout-ms", "1000",
         "--cleanup-timeout-ms", "60001"],
    ]
    for command in invalid_runs:
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode != 0, command
        assert "Traceback" not in result.stderr
        assert "http://" not in result.stdout


def test_cli_real_start_prints_origin_and_exits_after_bounded_cleanup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("test_site.__main__")
    captured: list[str] = []

    def interrupt(site: LocalTestSite) -> None:
        captured.append(site.origin)

    monkeypatch.setattr(module, "_wait_for_interrupt", interrupt)
    code = module.main(
        [
            "--scenario",
            "normal",
            "--request-timeout-ms",
            "5000",
            "--cleanup-timeout-ms",
            "3000",
        ]
    )
    assert code == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.startswith("http://127.0.0.1:")
    assert captured == [output.out.strip()]
    host, port = urlsplit(captured[0]).hostname, urlsplit(captured[0]).port
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        assert probe.connect_ex((host, port)) != 0
    finally:
        probe.close()
    assert site_threads() == []
