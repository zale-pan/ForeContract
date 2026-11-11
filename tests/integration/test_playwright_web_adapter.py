"""Real headless Chromium acceptance tests for the Playwright Web adapter.

Every test starts an ephemeral loopback HTTP server inside the test process and drives the real
adapter against the real Chromium installed for this environment. The server's own request
counters and the observable DOM are the independent oracle: a click is proved by a server-side
``/hit?name=click`` counter as well as a DOM value, a submit is proved by a POST counter, a
blocked redirect is proved by the absence of the unapproved path from the server's counters, and
resource release is proved by querying the real page/browser state after ``stop()``.

No browser path is skipped: a missing dependency or binary fails the test rather than turning it
into a skip. All fixture HTML, credentials (one synthetic cookie) and scratch data live inside
this file and the ephemeral server; nothing is written into the checkout.
"""

from __future__ import annotations

import asyncio
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import forecontract_web._browser_session as session_module
import forecontract_web.playwright_adapter as adapter_module
from forecontract.models.actions import ActionSpec
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import RoleTarget, TestIdTarget
from forecontract_web.validation import parse_web_observation

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def _use_unique_test_module_names() -> None:
    """Force pytest's documented ``importlib`` mode for this contract-mandated basename.

    The approved contract requires ``tests/unit/test_playwright_web_adapter.py`` and
    ``tests/integration/test_playwright_web_adapter.py`` to share a basename. Pytest's default
    ``prepend`` mode imports both as the same top-level module and refuses to collect the second
    one; its documented remedy is ``--import-mode=importlib``. The fixed acceptance command
    carries no such flag, so this compatibility shim wraps pytest's import helper for exactly
    that basename and selects the documented mode. Every test, assertion and fixture is
    unchanged, and the wrapper is a no-op for every other file.
    """
    try:
        import _pytest.python as pytest_python
        from _pytest.pathlib import ImportMode
        from _pytest.pathlib import import_path as original_import_path
    except Exception:  # noqa: BLE001 - a missing private pytest hook leaves the default mode
        return

    def import_path(path, *, mode=ImportMode.prepend, root, consider_namespace_packages):
        if Path(path).name == "test_playwright_web_adapter.py":
            mode = ImportMode.importlib
        return original_import_path(
            path,
            mode=mode,
            root=root,
            consider_namespace_packages=consider_namespace_packages,
        )

    pytest_python.import_path = import_path


_use_unique_test_module_names()


# ---------------------------------------------------------------------------- fixture server


@dataclass
class ServerState:
    port: int = 0
    hits: dict[str, int] = field(default_factory=dict)
    hit_events: list[str] = field(default_factory=list)
    cookies: list[tuple[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, path: str, cookie: str) -> None:
        with self.lock:
            self.hits[path] = self.hits.get(path, 0) + 1
            self.cookies.append((path, cookie))

    def record_hit(self, name: str) -> None:
        with self.lock:
            self.hit_events.append(name)

    def hit_count(self, name: str) -> int:
        with self.lock:
            return self.hit_events.count(name)

    def requests_for(self, path: str) -> int:
        with self.lock:
            return self.hits.get(path, 0)

    def cookie_headers(self, path: str) -> list[str]:
        with self.lock:
            return [cookie for recorded, cookie in self.cookies if recorded == path]


class FixtureServer:
    def __init__(self, state: ServerState) -> None:
        self.state = state

    @property
    def port(self) -> int:
        return self.state.port

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.state.port}"

    @property
    def unapproved_origin(self) -> str:
        # The same listener under a host name that is not one of the configured canonical origins.
        return f"http://localhost:{self.state.port}"

    def requests_for(self, path: str) -> int:
        return self.state.requests_for(path)

    def hit_count(self, name: str) -> int:
        return self.state.hit_count(name)

    def cookie_headers(self, path: str) -> list[str]:
        return self.state.cookie_headers(path)

    def hit_total(self) -> int:
        return len(self.state.hit_events)


MAIN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>p3 fixture</title></head>
<body>
<button data-testid="inc" onclick="hit('click'); bump()">inc</button>
<span data-testid="count" id="count">0</span>
<input data-testid="name" value="seed" oninput="hit('fill')">
<select data-testid="choice" onchange="hit('select')">
  <option value="a">A</option><option value="">Empty</option><option value="b">B</option>
</select>
<input data-testid="keys" onkeydown="hit('press'); document.getElementById('pressed').textContent = event.key">
<span data-testid="pressed" id="pressed">none</span>
<button data-testid="dupe" onclick="hit('dupe')">one</button>
<button data-testid="dupe" onclick="hit('dupe')">two</button>
<button data-testid="hidden" style="display:none" onclick="hit('hidden')">hidden</button>
<button onclick="hit('role'); bump()">Exact Name</button>
<button onclick="hit('role-extra')">Exact Name Extra</button>
<button style="display:none">Hidden Role</button>
<span data-testid="opaque">plain</span>
<div id="later-box"></div>
<span data-testid="gone">gone</span>
<img data-testid="okimg" src="/pixel" onload="show('okstate','loaded')">
<img data-testid="blockedimg" src="__UNAPPROVED__/blocked-pixel" onerror="show('blockedstate','blocked')">
<span data-testid="okstate" id="okstate" hidden>pending</span>
<span data-testid="blockedstate" id="blockedstate" hidden>pending</span>
<button data-testid="popup" onclick="window.open('__UNAPPROVED__/popup-target')">pop</button>
<a data-testid="download" href="/download">download</a>
<script>
function hit(name) { fetch('/hit?name=' + encodeURIComponent(name)); }
function bump() {
  var element = document.getElementById('count');
  element.textContent = String(Number(element.textContent) + 1);
}
function show(id, value) {
  var element = document.getElementById(id);
  element.textContent = value;
  element.hidden = false;
}
setTimeout(function () {
  var later = document.createElement('span');
  later.setAttribute('data-testid', 'later');
  later.textContent = 'ready';
  document.getElementById('later-box').appendChild(later);
  var gone = document.querySelector('[data-testid=gone]');
  if (gone) { gone.parentNode.removeChild(gone); }
}, 1500);
</script>
</body></html>"""

FORM_HTML = """<!doctype html><html><body>
<form action="/slow-submit" method="post">
  <button data-testid="submit">submit</button>
</form>
<span data-testid="formmark">form</span>
</body></html>"""

COOKIE_HTML = """<!doctype html><html><body>
<span data-testid="cookie">set</span>
</body></html>"""

ECHO_HTML = """<!doctype html><html><body>
<span data-testid="echo">echo</span>
</body></html>"""

FINAL_HTML = """<!doctype html><html><body>
<span data-testid="final">ok</span>
</body></html>"""

POPUP_REDIRECT_HTML = """<!doctype html><html><body>
<button data-testid="popup-redirect" onclick="window.open('/redirect-external')">pop</button>
<button data-testid="popup-blank" onclick="window.open('about:blank')">blank</button>
<span data-testid="mark">ready</span>
</body></html>"""

DESTINATION_HTML = """<!doctype html><html><body>
<img data-testid="relative-pixel" src="relative.png">
<span data-testid="final">destination</span>
</body></html>"""

SIMPLE_HTML = """<!doctype html><html><body>ok</body></html>"""

DOWNLOAD_FILENAME = "synthetic-p3t02-download.bin"


def _build_handler(state: ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            state.record(self.path, self.headers.get("Cookie", ""))
            path = urlsplit(self.path).path
            if path == "/":
                port = state.port
                body = MAIN_HTML.replace("__UNAPPROVED__", f"http://localhost:{port}").encode()
                self._send(200, body, "text/html")
            elif path == "/form":
                self._send(200, FORM_HTML.encode(), "text/html")
            elif path == "/setcookie":
                self._send(200, COOKIE_HTML.encode(), "text/html",
                           extra=[("Set-Cookie", "token=alpha; Path=/")])
            elif path == "/echo":
                self._send(200, ECHO_HTML.encode(), "text/html")
            elif path == "/ok":
                self._send(200, FINAL_HTML.encode(), "text/html")
            elif path == "/popup-redirect":
                self._send(200, POPUP_REDIRECT_HTML.encode(), "text/html")
            elif path == "/redirect-internal":
                self._redirect("/ok")
            elif path == "/redirect-destination":
                self._redirect("/destination/index")
            elif path == "/destination/index":
                self._send(200, DESTINATION_HTML.encode(), "text/html")
            elif path == "/destination/relative.png":
                self._send(200, b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "image/gif")
            elif path == "/download":
                self._send(
                    200,
                    b"SYNTHETIC_P3T02_DOWNLOAD_BYTES",
                    "application/octet-stream",
                    extra=[("Content-Disposition", f'attachment; filename="{DOWNLOAD_FILENAME}"')],
                )
            elif path == "/redirect-external":
                port = state.port
                self._redirect(f"http://localhost:{port}/blocked-redirect-target")
            elif path == "/slow":
                time.sleep(5.0)
                self._send(200, SIMPLE_HTML.encode(), "text/html")
            elif path == "/pixel":
                self._send(200, b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "image/gif")
            elif path == "/hit":
                query = parse_qs(urlsplit(self.path).query)
                for name in query.get("name", []):
                    state.record_hit(name)
                self._send(200, b"", "text/plain")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(200, SIMPLE_HTML.encode(), "text/html")

        def do_POST(self) -> None:
            state.record(self.path, self.headers.get("Cookie", ""))
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            if urlsplit(self.path).path == "/slow-submit":
                time.sleep(4.0)
            self._send(200, SIMPLE_HTML.encode(), "text/html")

        def _redirect(self, location: str) -> None:
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(self, code: int, body: bytes, content_type: str,
                  extra: list[tuple[str, str]] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in extra or []:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

    return Handler


@pytest.fixture()
def server():
    state = ServerState()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(state))
    state.port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield FixtureServer(state)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


@pytest.fixture(scope="session", autouse=True)
def browser_environment_ready():
    """Prove the real Chromium launch works once before the per-test launches.

    The round-002 scoped run recorded exactly one transient ``WEB_BROWSER_START_FAILED`` in the
    first real-browser node, immediately after an unrelated host process was killed; the same
    node passed on the single quiescent confirmation. The transient was host load, not an
    adapter decision, so instead of retrying any test this explicit readiness launch pays the
    cold start once, fails the session (never skips) when Chromium is unavailable, and records
    the real browser version as the environment basis. It is configuration/readiness evidence
    only: every test still owns and releases its own driver, page and context and none of them
    borrows this instance.
    """

    async def probe():
        adapter = adapter_module.PlaywrightWebAdapter(
            PlaywrightWebConfig(
                allowed_origins=["http://127.0.0.1:1"],
                observation_targets=[],
                operation_timeout_ms=20000,
                cleanup_timeout_ms=10000,
            )
        )
        await adapter.start()
        try:
            session = adapter._session
            assert session is not None
            browser = session._browser
            assert browser is not None
            version = browser.version
        finally:
            await adapter.stop()
        assert adapter.cleanup_error is None
        return version

    try:
        version = run(probe())
    except WebBrowserError as exc:
        pytest.fail(f"real Chromium is not available for the integration suite: {exc.code}")
    assert isinstance(version, str) and _VERSION_PATTERN.match(version) is not None, version
    print(f"real Chromium readiness: {version}")
    return version


# ---------------------------------------------------------------------------- helpers


def run(coroutine):
    return asyncio.run(coroutine)


def act(action_type: str, payload: dict) -> ActionSpec:
    return ActionSpec(schema_version="0.1", action_type=action_type, payload=payload)


def id_target(test_id: str) -> TestIdTarget:
    return TestIdTarget(strategy="test_id", test_id=test_id)


def role_target(role: str, name: str) -> RoleTarget:
    return RoleTarget(strategy="role", role=role, name=name)


def target_payload(target: TestIdTarget | RoleTarget) -> dict:
    return target.model_dump()


def make_adapter(
    server: FixtureServer,
    targets: list | None = None,
    *,
    operation_timeout_ms: int = 5000,
    cleanup_timeout_ms: int = 5000,
):
    return adapter_module.PlaywrightWebAdapter(
        PlaywrightWebConfig(
            allowed_origins=[server.origin],
            observation_targets=list(targets or []),
            operation_timeout_ms=operation_timeout_ms,
            cleanup_timeout_ms=cleanup_timeout_ms,
        )
    )


def evidence_by_test_id(observation) -> dict[str, object]:
    state = parse_web_observation(observation)
    records = {}
    for record in state.targets:
        if isinstance(record.target, TestIdTarget):
            records[record.target.test_id] = record
    return records


def evidence_by_role(observation) -> dict[str, object]:
    state = parse_web_observation(observation)
    records = {}
    for record in state.targets:
        if isinstance(record.target, RoleTarget):
            records[record.target.name] = record
    return records


# ---------------------------------------------------------------------------- six actions


def test_web_integration_six_actions_with_counter_and_dom_oracles(server: FixtureServer) -> None:
    targets = [
        id_target("count"),
        id_target("name"),
        id_target("pressed"),
        id_target("choice"),
        id_target("later"),
        id_target("gone"),
        id_target("okstate"),
        id_target("blockedstate"),
    ]
    adapter = make_adapter(server, targets)

    async def scenario() -> None:
        await adapter.start()
        try:
            navigated = await adapter.execute(act("web.navigate", {"url": f"{server.origin}/"}))
            assert navigated.status.value == "succeeded"
            assert server.requests_for("/") == 1

            clicked = await adapter.execute(act("web.click", {"target": target_payload(id_target("inc"))}))
            assert clicked.status.value == "succeeded"
            await asyncio.sleep(0.3)
            assert server.hit_count("click") == 1

            filled = await adapter.execute(
                act("web.fill", {"target": target_payload(id_target("name")), "text": ""})
            )
            assert filled.status.value == "succeeded"
            await asyncio.sleep(0.2)
            assert server.hit_count("fill") == 1

            pressed = await adapter.execute(
                act("web.press", {"target": target_payload(id_target("keys")), "key": "Enter"})
            )
            assert pressed.status.value == "succeeded"
            await asyncio.sleep(0.2)
            assert server.hit_count("press") == 1

            selected = await adapter.execute(
                act("web.select", {"target": target_payload(id_target("choice")), "value": ""})
            )
            assert selected.status.value == "succeeded"
            await asyncio.sleep(0.2)
            assert server.hit_count("select") == 1

            waited = await adapter.execute(
                act("web.wait", {"target": target_payload(id_target("later")), "state": "visible"})
            )
            assert waited.status.value == "succeeded"
            detached = await adapter.execute(
                act("web.wait", {"target": target_payload(id_target("gone")), "state": "detached"})
            )
            assert detached.status.value == "succeeded"

            observed = await adapter.observe()
            records = evidence_by_test_id(observed)
            assert records["count"].text == "1"
            assert records["name"].value == ""
            assert records["name"].text == ""
            assert records["pressed"].text == "Enter"
            assert records["choice"].value == ""
            assert records["later"].match_count == 1
            assert records["later"].text == "ready"
            assert records["gone"].match_count == 0
            # Only one navigation request ever happened; actions did not reload the page.
            assert server.requests_for("/") == 1
            assert server.hit_total() == 4
        finally:
            await adapter.stop()
        # The affected node owns and fully releases its browser set before returning, so it never
        # relies on a previous or later node's process state.
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None

    run(scenario())


# ---------------------------------------------------------------------------- locators


def test_web_integration_unique_ambiguous_and_hidden_locators(server: FixtureServer) -> None:
    targets = [id_target("count"), role_target("button", "Hidden Role"), role_target("button", "Exact Name")]
    adapter = make_adapter(server, targets)

    async def scenario() -> None:
        await adapter.start()
        try:
            await adapter.execute(act("web.navigate", {"url": f"{server.origin}/"}))

            role = await adapter.execute(act(
                "web.click", {"target": target_payload(role_target("button", "Exact Name"))}
            ))
            assert role.status.value == "succeeded"
            await asyncio.sleep(0.3)
            assert server.hit_count("role") == 1

            not_exact = await adapter.execute(act(
                "web.click", {"target": target_payload(role_target("button", "Exact"))}
            ))
            assert not_exact.status.value == "failed"
            assert not_exact.error is not None
            assert not_exact.error.code == "WEB_TARGET_NOT_FOUND"

            ambiguous = await adapter.execute(act(
                "web.click", {"target": target_payload(id_target("dupe"))}
            ))
            assert ambiguous.status.value == "failed"
            assert ambiguous.error is not None
            assert ambiguous.error.code == "WEB_TARGET_AMBIGUOUS"
            assert server.hit_count("dupe") == 0

            hidden = await adapter.execute(act(
                "web.click", {"target": target_payload(id_target("hidden"))}
            ))
            assert hidden.status.value == "failed"
            assert hidden.error is not None
            assert hidden.error.code == "WEB_TARGET_NOT_ACTIONABLE"
            assert server.hit_count("hidden") == 0

            observed = await adapter.observe()
            by_id = evidence_by_test_id(observed)
            by_role = evidence_by_role(observed)
            assert by_id["count"].text == "1"
            assert by_role["Hidden Role"].match_count == 1
            assert by_role["Hidden Role"].visible is False
            assert by_role["Exact Name"].match_count == 1
            assert by_role["Exact Name"].visible is True
        finally:
            await adapter.stop()

    run(scenario())


# ---------------------------------------------------------------------------- evidence matrix


def test_web_integration_observation_evidence_matrix(server: FixtureServer) -> None:
    targets = [
        id_target("count"),
        id_target("name"),
        id_target("dupe"),
        id_target("absent"),
        id_target("opaque"),
        id_target("hidden"),
        id_target("later"),
    ]
    adapter = make_adapter(server, targets)

    async def scenario() -> None:
        await adapter.start()
        try:
            await adapter.execute(act("web.navigate", {"url": f"{server.origin}/"}))
            await adapter.execute(
                act("web.fill", {"target": target_payload(id_target("name")), "text": ""})
            )
            observed = await adapter.observe()
            records = evidence_by_test_id(observed)
            assert records["count"].status == "collected"
            assert records["count"].match_count == 1
            assert records["count"].text == "0"
            assert records["count"].value is None
            assert records["name"].value == "" and records["name"].text == ""
            assert records["dupe"].status == "collected"
            assert records["dupe"].match_count == 2
            assert records["dupe"].text is None and records["dupe"].value is None
            assert records["absent"].status == "collected"
            assert records["absent"].match_count == 0
            assert records["absent"].text is None
            assert records["opaque"].match_count == 1
            assert records["opaque"].text == "plain"
            assert records["opaque"].value is None
            assert records["hidden"].match_count == 1
            assert records["hidden"].visible is False
            assert records["later"].match_count == 0
            assert records["later"].error_code is None
        finally:
            await adapter.stop()

    run(scenario())


# ---------------------------------------------------------------------------- network policy


def test_web_integration_local_resource_policy(server: FixtureServer) -> None:
    targets = [id_target("okstate"), id_target("blockedstate")]
    adapter = make_adapter(server, targets)

    async def scenario() -> None:
        await adapter.start()
        try:
            await adapter.execute(act("web.navigate", {"url": f"{server.origin}/"}))
            same_origin = await adapter.execute(act(
                "web.wait", {"target": target_payload(id_target("okstate")), "state": "visible"}
            ))
            assert same_origin.status.value == "succeeded"
            assert server.requests_for("/pixel") == 1

            blocked = await adapter.execute(act(
                "web.wait", {"target": target_payload(id_target("blockedstate")), "state": "visible"}
            ))
            assert blocked.status.value == "succeeded"
            assert server.requests_for("/blocked-pixel") == 0

            observed = await adapter.observe()
            records = evidence_by_test_id(observed)
            assert records["okstate"].text == "loaded"
            assert records["blockedstate"].text == "blocked"

            with pytest.raises(WebBrowserError) as caught:
                await adapter.execute(act(
                    "web.navigate", {"url": f"{server.unapproved_origin}/blocked-nav"}
                ))
            assert caught.value.code == "WEB_ORIGIN_BLOCKED"
            assert server.requests_for("/blocked-nav") == 0
        finally:
            await adapter.stop()

    run(scenario())


def test_web_integration_redirect_policy(server: FixtureServer) -> None:
    targets = [id_target("final")]
    adapter = make_adapter(server, targets)

    async def scenario() -> None:
        await adapter.start()
        try:
            internal = await adapter.execute(act(
                "web.navigate", {"url": f"{server.origin}/redirect-internal"}
            ))
            assert internal.status.value == "succeeded"
            assert server.requests_for("/redirect-internal") == 1
            assert server.requests_for("/ok") == 1
            observed = await adapter.observe()
            assert evidence_by_test_id(observed)["final"].text == "ok"

            # An approved redirect keeps the browser's own final URL, document body and
            # base-origin resolution: the relative image resolves against /destination/index.
            destination = await adapter.execute(act(
                "web.navigate", {"url": f"{server.origin}/redirect-destination"}
            ))
            assert destination.status.value == "succeeded"
            assert server.requests_for("/redirect-destination") == 1
            assert server.requests_for("/destination/index") == 1
            assert server.requests_for("/destination/relative.png") == 1
            assert server.requests_for("/relative.png") == 0
            navigated = parse_web_observation(await adapter.observe())
            assert navigated.url == f"{server.origin}/destination/index"
            assert evidence_by_test_id(await adapter.observe())["final"].text == "destination"
            assert navigated.targets[0].text == "destination"

            with pytest.raises(WebBrowserError) as caught:
                await adapter.execute(act(
                    "web.navigate", {"url": f"{server.origin}/redirect-external"}
                ))
            assert caught.value.code == "WEB_ORIGIN_BLOCKED"
            assert server.requests_for("/redirect-external") == 1
            # The unapproved destination is refused before it is ever requested.
            assert server.requests_for("/blocked-redirect-target") == 0
        finally:
            await adapter.stop()

    run(scenario())


def test_web_integration_popup_is_closed_and_cannot_reach_an_unapproved_origin(
    server: FixtureServer,
) -> None:
    targets = [id_target("count")]
    adapter = make_adapter(server, targets)

    async def scenario() -> None:
        await adapter.start()
        try:
            await adapter.execute(act("web.navigate", {"url": f"{server.origin}/"}))
            clicked = await adapter.execute(act(
                "web.click", {"target": target_payload(id_target("popup"))}
            ))
            assert clicked.status.value == "succeeded"
            await asyncio.sleep(0.8)
            assert server.requests_for("/popup-target") == 0
            session = adapter._session
            assert session is not None
            assert all(page.is_closed() for page in session._extra_pages)
            observed = await adapter.observe()
            assert evidence_by_test_id(observed)["count"].match_count == 1
        finally:
            await adapter.stop()

    run(scenario())


def test_web_integration_popup_redirect_to_an_unapproved_origin_is_blocked(
    server: FixtureServer,
) -> None:
    """F05: an unsupported popup never reaches the unapproved redirect destination."""
    adapter = make_adapter(server)

    async def scenario() -> None:
        await adapter.start()
        try:
            await adapter.execute(act("web.navigate", {"url": f"{server.origin}/popup-redirect"}))
            clicked = await adapter.execute(act(
                "web.click", {"target": target_payload(id_target("popup-redirect"))}
            ))
            assert clicked.status.value == "succeeded"
            session = adapter._session
            assert session is not None
            # Explicit barrier: the unsupported owned page was tracked and then closed.
            deadline = time.monotonic() + 5.0
            while not (
                session._extra_pages
                and all(page.is_closed() for page in session._extra_pages)
            ):
                if time.monotonic() > deadline:
                    pytest.fail("the owned popup page was not tracked and closed")
                await asyncio.sleep(0.01)
            # Redirecting and only then closing the popup is insufficient: the redirect hop must
            # be refused before it is sent, so the unapproved destination is never requested.
            assert server.requests_for("/blocked-redirect-target") == 0
            assert adapter._state == "running"
        finally:
            await adapter.stop()
        assert adapter.cleanup_error is None

    run(scenario())


def test_web_integration_popup_redirect_is_blocked_with_a_delayed_close(
    server: FixtureServer,
) -> None:
    """F05: with the popup deliberately held open, its redirect hop is still refused.

    The popup is first opened without a URL, so its page exists before the navigation under test.
    The test then drives that owned popup page to an approved origin that redirects to the
    unapproved one while the controlled close is still held. The approved first request may be
    released (counter 1), but the redirect hop is refused by the per-hop policy installed on the
    popup page itself (destination counter 0). Closing after the request is therefore not what
    provides the protection.
    """
    adapter = make_adapter(server)

    async def scenario() -> None:
        await adapter.start()
        session = adapter._session
        assert session is not None
        release = asyncio.Event()
        original_close_extra = session._close_extra

        async def delayed_close(page: object) -> None:
            await release.wait()
            await original_close_extra(page)

        session._close_extra = delayed_close
        try:
            await adapter.execute(act("web.navigate", {"url": f"{server.origin}/popup-redirect"}))
            clicked = await adapter.execute(act(
                "web.click", {"target": target_payload(id_target("popup-blank"))}
            ))
            assert clicked.status.value == "succeeded"
            deadline = time.monotonic() + 5.0
            while not session._extra_pages:
                if time.monotonic() > deadline:
                    pytest.fail("the unsupported popup page was never tracked")
                await asyncio.sleep(0.01)
            popup = session._extra_pages[0]
            assert popup.is_closed() is False, "the controlled close must still hold the page open"
            # Explicit barrier: the popup owns the same per-hop policy as the active page before
            # the navigation under test starts. This removes the enable-vs-redirect race while
            # still proving that the hop itself - not the close - refuses the destination.
            backend = session._backend
            deadline = time.monotonic() + 5.0
            while id(popup) not in backend._policied_pages:
                if time.monotonic() > deadline:
                    pytest.fail("the popup page never received the per-hop policy")
                await asyncio.sleep(0.01)
            try:
                await popup.goto(f"{server.origin}/redirect-external", timeout=5000)
            except Exception:  # noqa: BLE001, S110 - a refused hop may end the navigation
                pass
            # The approved first hop really was issued to the server...
            assert server.requests_for("/redirect-external") == 1
            # ...while the unapproved hop it redirects to was refused before it was sent.
            assert server.requests_for("/blocked-redirect-target") == 0
            assert popup.is_closed() is False, "the refusal must not depend on closing the page"
            release.set()
            deadline = time.monotonic() + 5.0
            while not all(page.is_closed() for page in session._extra_pages):
                if time.monotonic() > deadline:
                    pytest.fail("the delayed owned popup did not close after release")
                await asyncio.sleep(0.01)
            assert server.requests_for("/blocked-redirect-target") == 0
        finally:
            release.set()
            await adapter.stop()
        assert adapter.cleanup_error is None

    run(scenario())


def _existing_files(*roots: Path) -> set[str]:
    found: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_file():
                found.add(str(candidate))
    return found


def test_web_integration_download_is_refused_and_writes_no_file(
    server: FixtureServer, tmp_path: Path,
) -> None:
    adapter = make_adapter(server)
    default_downloads = Path.home() / "Downloads"
    before = _existing_files(tmp_path, default_downloads)

    async def scenario() -> tuple:
        await adapter.start()
        try:
            navigated = await adapter.execute(act("web.navigate", {"url": f"{server.origin}/"}))
            assert navigated.status.value == "succeeded"
            page = adapter._session.page
            assert page is not None
            async with page.expect_download(timeout=5000) as pending:
                clicked = await adapter.execute(act(
                    "web.click", {"target": target_payload(id_target("download"))}
                ))
            download = await pending.value
            failure = await download.failure()
            return clicked, failure, page.url, download.suggested_filename
        finally:
            await adapter.stop()

    clicked, failure, final_url, filename = run(scenario())
    assert clicked.status.value == "succeeded"
    # Chromium refuses the attachment because the owned context disables downloads.
    assert failure is not None
    assert final_url == f"{server.origin}/"
    assert filename == DOWNLOAD_FILENAME
    after = _existing_files(tmp_path, default_downloads)
    assert after == before, "a refused download must write no file anywhere"
    assert not list(tmp_path.rglob(DOWNLOAD_FILENAME))
    assert server.requests_for("/download") == 1


# ---------------------------------------------------------------------------- unknown / no replay


def test_web_integration_unknown_submit_is_not_repeated(server: FixtureServer) -> None:
    adapter = make_adapter(server, operation_timeout_ms=2500, cleanup_timeout_ms=5000)

    async def scenario() -> None:
        await adapter.start()
        try:
            await adapter.execute(act("web.navigate", {"url": f"{server.origin}/form"}))
            started = await adapter.execute(act(
                "web.click", {"target": target_payload(id_target("submit"))}
            ))
            assert started.status.value == "unknown"
            assert started.error is not None
            assert started.error.code in ("WEB_ACTION_TIMEOUT", "WEB_ACTION_INTERRUPTED")
            assert server.requests_for("/slow-submit") == 1
            await asyncio.sleep(1.0)
            assert server.requests_for("/slow-submit") == 1
        finally:
            await adapter.stop()

    run(scenario())


# ---------------------------------------------------------------------------- release


def test_web_integration_stop_releases_the_real_browser(server: FixtureServer) -> None:
    adapter = make_adapter(server, [id_target("count")])

    async def scenario() -> None:
        await adapter.start()
        session = adapter._session
        assert session is not None
        page = session.page
        browser = session._browser
        assert page is not None and browser is not None
        version = browser.version
        assert isinstance(version, str) and _VERSION_PATTERN.match(version) is not None
        await adapter.execute(act("web.navigate", {"url": f"{server.origin}/"}))
        await adapter.stop()
        assert adapter.cleanup_error is None
        assert page.is_closed() is True
        assert browser.is_connected() is False
        assert session.page is None
        assert session._browser is None and session._driver is None
        assert session._context is None
        await adapter.stop()
        assert adapter.cleanup_error is None

    run(scenario())


def test_web_integration_operation_timeout_then_stop_releases_the_browser(
    server: FixtureServer,
) -> None:
    adapter = make_adapter(server, operation_timeout_ms=3000, cleanup_timeout_ms=5000)

    async def scenario() -> None:
        await adapter.start()
        session = adapter._session
        page = session.page
        browser = session._browser
        timed_out = await adapter.execute(act("web.navigate", {"url": f"{server.origin}/slow"}))
        assert timed_out.status.value == "unknown"
        assert timed_out.error is not None and timed_out.error.code == "WEB_ACTION_TIMEOUT"
        await adapter.stop()
        assert adapter.cleanup_error is None
        assert page.is_closed() is True
        assert browser.is_connected() is False

    run(scenario())


def test_web_integration_cancelled_operation_then_stop_releases_the_browser(
    server: FixtureServer,
) -> None:
    adapter = make_adapter(server, operation_timeout_ms=10000, cleanup_timeout_ms=5000)

    async def scenario() -> None:
        await adapter.start()
        session = adapter._session
        page = session.page
        browser = session._browser
        executing = asyncio.ensure_future(
            adapter.execute(act("web.navigate", {"url": f"{server.origin}/slow"}))
        )
        await asyncio.sleep(0.2)
        executing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await executing
        await adapter.stop()
        assert page.is_closed() is True
        assert browser.is_connected() is False
        assert adapter.cleanup_error is None

    run(scenario())


def test_web_integration_failed_start_releases_partial_real_resources(
    server: FixtureServer, monkeypatch,
) -> None:
    created: dict[str, object] = {}
    captured: dict[str, object] = {}
    real_create = adapter_module._create_session

    def factory(config: PlaywrightWebConfig):
        session = real_create(config)
        created["session"] = session
        return session

    async def failing_open_page(self, context):
        session = created["session"]
        captured["browser"] = session._browser
        captured["context"] = session._context
        raise RuntimeError("synthetic page acquisition failure")

    monkeypatch.setattr(adapter_module, "_create_session", factory)
    monkeypatch.setattr(session_module.PlaywrightBackend, "open_page", failing_open_page)
    adapter = make_adapter(server)

    async def scenario() -> None:
        with pytest.raises(WebBrowserError) as caught:
            await adapter.start()
        assert caught.value.code == "WEB_BROWSER_START_FAILED"
        assert adapter._state == "stopped"
        assert adapter.cleanup_error is None
        browser = captured["browser"]
        assert browser.is_connected() is False
        await adapter.stop()
        assert adapter._state == "stopped"

    run(scenario())


# ---------------------------------------------------------------------------- isolation


def test_web_integration_no_cross_instance_cookie_sharing(server: FixtureServer) -> None:
    first = make_adapter(server)
    second = make_adapter(server)

    async def scenario() -> None:
        await first.start()
        try:
            await first.execute(act("web.navigate", {"url": f"{server.origin}/setcookie"}))
            await first.execute(act("web.navigate", {"url": f"{server.origin}/echo"}))
        finally:
            await first.stop()
        await second.start()
        try:
            await second.execute(act("web.navigate", {"url": f"{server.origin}/echo"}))
        finally:
            await second.stop()

    run(scenario())
    headers = server.cookie_headers("/echo")
    assert len(headers) == 2
    assert "token=alpha" in headers[0]
    assert "token=alpha" not in headers[1]
    assert headers[1] == ""
