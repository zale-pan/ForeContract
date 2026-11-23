"""Loopback-only HTTP server and the public host control surface of the local test site.

The listener binds exactly ``127.0.0.1:0``; the operating system chooses an ephemeral port and
the public ``origin`` reports it only after a successful start. There is no fixed port, no public
interface, no file mapping, no directory listing and no control route over HTTP: the oracle and
the scenario gates are reachable only through the Python host object.

Cleanup uses one absolute monotonic deadline for the whole attempt. Stop first refuses new work
and wakes every gate, then closes the listener, gives an already-woken request a bounded chance to
deliver its fixed refusal, closes every owned connection and joins every owned thread until that
same deadline. Every internal coordination lock acquisition and every resource count is bounded by
that one deadline too, so a contended lock cannot delay the call past its budget. The returned
:class:`~test_site.models.SiteStopResult` reports the actual resources that remain; a daemon thread
or a mere ``stopped`` flag cannot make it complete.
"""

from __future__ import annotations

import http.server
import select
import socket
import socketserver
import threading
import time
from http import HTTPStatus
from typing import Final, Self
from urllib.parse import urlsplit

from test_site.models import (
    LocalSiteError,
    Scenario,
    SiteConfig,
    SiteEvent,
    SiteOracleSnapshot,
    SiteStopResult,
    _is_site_config,
)
from test_site.pages import (
    FIXTURE_PASSWORD,
    FIXTURE_USERNAME,
    render_dashboard_page,
    render_login_page,
)
from test_site.state import GateOutcome, SiteState, _acquire_within

__all__ = [
    "LocalTestSite",
    "parse_content_length",
    "parse_form_body",
    "parse_generation",
]

_COOKIE_NAME: Final[str] = "test_site_session"
_FORM_CONTENT_TYPE: Final[str] = "application/x-www-form-urlencoded"
_FORM_FIELDS: Final[frozenset[str]] = frozenset({"username", "password", "generation"})
_MAX_BODY_BYTES: Final[int] = 4096
_KNOWN_PATHS: Final[frozenset[str]] = frozenset({"/", "/login", "/dashboard"})
_METHODS_FOR_PATH: Final[dict[str, tuple[str, ...]]] = {
    "/": ("GET",),
    "/login": ("GET", "POST"),
    "/dashboard": ("GET",),
}
_FIXED_TEXT: Final[dict[int, bytes]] = {
    HTTPStatus.BAD_REQUEST: b"bad request\n",
    HTTPStatus.NOT_FOUND: b"not found\n",
    HTTPStatus.METHOD_NOT_ALLOWED: b"method not allowed\n",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: b"payload too large\n",
    HTTPStatus.UNSUPPORTED_MEDIA_TYPE: b"unsupported media type\n",
    HTTPStatus.SERVICE_UNAVAILABLE: b"service unavailable\n",
}
_HEX_DIGITS: Final[bytes] = b"0123456789abcdefABCDEF"

#: How long one poll of a socket or listener waits before re-checking the one absolute deadline
#: and the stop flag. It is a scheduling slice, never a per-phase budget.
_POLL_S: Final[float] = 0.02
#: The largest single raw read; the reader itself owns the buffering.
_READ_CHUNK: Final[int] = 4096
#: After the request deadline has expired, a fixed refusal or close still has to be delivered.
#: This bounded write window is not an extension of the request budget: it can carry only the
#: fixed 503 refusal or the post-commit closed connection, never a new normal commit.
_EXPIRED_RESPONSE_WRITE_S: Final[float] = 0.02
#: Below this remaining cleanup budget, a loopback connect probe cannot be completed reliably
#: (the OS connect attempt costs about a millisecond on this platform). The port proof then
#: falls back to the definitive fact that the only owned listener socket is already closed; it
#: never asks for more time.
_PORT_PROBE_MIN_S: Final[float] = 0.01


def _pause_within(deadline: float) -> bool:
    """Yield until the next short check without ever requesting a wait past ``deadline``.

    Windows rounds a positive ``Condition.wait``/``Thread.join`` timeout up to a ~15ms timer
    tick, which would silently extend a 1ms cleanup budget, and ``time.monotonic()`` itself
    only advances on that tick here. Cleanup deadlines therefore use the high-resolution
    monotonic performance counter and polls never request a blocking wait past the deadline;
    a short remaining budget yields with a zero-time sleep and only a comfortable remaining
    budget uses 1ms slices.
    """
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        return False
    time.sleep(0.001 if remaining >= 0.05 else 0)
    return True


def _wait_for_thread(thread: threading.Thread, deadline: float) -> bool:
    """Wait for one owned thread to finish without a blocking join past the cleanup deadline."""
    while thread.is_alive():
        if not _pause_within(deadline):
            return False
    return True


def parse_content_length(values: object) -> int | None:
    """Parse the one required ``Content-Length`` value, refusing duplicates and non-digits."""
    if not isinstance(values, list) or len(values) != 1:
        return None
    text = values[0]
    if not isinstance(text, str) or not text or len(text) > 12:
        return None
    if not text.isascii() or not text.isdigit():
        return None
    return int(text)


def parse_generation(text: object) -> int | None:
    """Parse the canonical ASCII positive decimal generation, with no sign or leading zero."""
    if not isinstance(text, str) or not text or len(text) > 20:
        return None
    if not text.isascii() or not text.isdigit():
        return None
    if text[0] == "0":
        return None
    return int(text)


def _form_decode(component: bytes) -> bytes | None:
    """Percent-decode one form component; a malformed escape is a rejection, not a repair."""
    decoded = bytearray()
    index = 0
    while index < len(component):
        byte = component[index]
        if byte == 0x2B:
            decoded.append(0x20)
            index += 1
        elif byte == 0x25:
            if index + 2 >= len(component):
                return None
            pair = component[index + 1 : index + 3]
            if any(digit not in _HEX_DIGITS for digit in pair):
                return None
            decoded.append(int(pair, 16))
            index += 3
        else:
            decoded.append(byte)
            index += 1
    return bytes(decoded)


def parse_form_body(body: object) -> dict[str, str] | None:
    """Parse exactly one value for each of the three fixed fields, or return ``None``."""
    if not isinstance(body, (bytes, bytearray)) or not body:
        return None
    parts = bytes(body).split(b"&")
    if len(parts) != len(_FORM_FIELDS):
        return None
    fields: dict[str, str] = {}
    for part in parts:
        name_raw, separator, value_raw = part.partition(b"=")
        if not separator or not name_raw:
            return None
        name_bytes = _form_decode(name_raw)
        value_bytes = _form_decode(value_raw)
        if name_bytes is None or value_bytes is None:
            return None
        try:
            name = name_bytes.decode("utf-8")
            value = value_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if name not in _FORM_FIELDS or name in fields:
            return None
        fields[name] = value
    if set(fields) != _FORM_FIELDS:
        return None
    return fields


def _content_type_ok(value: object) -> bool:
    """Accept only the fixed form media type, with an optional UTF-8 charset parameter."""
    if not isinstance(value, str) or not value:
        return False
    parts = value.split(";")
    if parts[0].strip().lower() != _FORM_CONTENT_TYPE:
        return False
    for parameter in parts[1:]:
        name, separator, parameter_value = parameter.partition("=")
        if not separator or name.strip().lower() != "charset":
            return False
        if parameter_value.strip().lower() not in ("utf-8", "utf8"):
            return False
    return True


def _close_socket(target: socket.socket) -> None:
    """Close one socket without letting a second close failure escape."""
    try:
        target.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        target.close()
    except OSError:
        pass


class _DeadlineReader:
    """A stop-aware, deadline-aware byte reader for one connection.

    ``socket.makefile()`` returns a buffered reader whose internal buffer is documented to become
    inconsistent when a socket timeout interrupts a read. A legal request body split around a
    short pause would then look unreadable even though the client is well inside its budget.
    This reader polls the raw socket with :func:`select.select` instead: each poll is a short
    scheduling slice, every wait is measured against the one absolute request deadline, and
    partial headers or bodies that arrive after a poll simply continue. It keeps its own buffer,
    so no timeout can ever corrupt the stream state.
    """

    def __init__(self, handler: _RequestHandler) -> None:
        self._handler = handler
        self._connection = handler.connection
        self._buffer = bytearray()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether this reader has been closed by the request handler."""
        return self._closed

    def _remaining(self) -> float:
        return self._handler._deadline - time.monotonic()

    def _fill(self) -> bool:
        """Wait for at least one raw byte within the deadline; ``False`` ends the stream."""
        while not self._closed:
            remaining = self._remaining()
            if remaining <= 0:
                raise TimeoutError("request deadline expired")
            if self._handler.server._closing:
                return False
            wait = min(remaining, _POLL_S)
            try:
                readable, _, _ = select.select([self._connection], [], [], wait)
            except (OSError, ValueError):
                return False
            if not readable:
                continue
            try:
                chunk = self._connection.recv(_READ_CHUNK)
            except TimeoutError:
                continue
            except (OSError, ValueError):
                return False
            if not chunk:
                return False
            self._buffer.extend(chunk)
            return True
        return False

    def readline(self, limit: int = -1) -> bytes:
        """Read one line, or ``limit`` bytes, without ever leaving a timed-out buffer behind."""
        if limit <= 0:
            limit = 65537
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                end = newline + 1
                line = bytes(self._buffer[:end])
                del self._buffer[:end]
                return line
            if len(self._buffer) >= limit:
                line = bytes(self._buffer[:limit])
                del self._buffer[:limit]
                return line
            if not self._fill():
                line = bytes(self._buffer)
                self._buffer.clear()
                return line

    def read1(self, size: int = -1) -> bytes:
        """Read up to ``size`` currently or imminently available bytes."""
        if size <= 0:
            size = _READ_CHUNK
        if not self._buffer and not self._fill():
            return b""
        count = min(size, len(self._buffer))
        data = bytes(self._buffer[:count])
        del self._buffer[:count]
        return data

    def close(self) -> None:
        """Release the local buffer; the owning socket is closed by the handler."""
        self._closed = True
        self._buffer.clear()


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    """One request on one owned connection; every response closes the connection."""

    protocol_version = "HTTP/1.1"
    server_version = "ForeContractLocalTestSite"
    sys_version = ""

    # ------------------------------------------------------------------ transport setup

    def setup(self) -> None:
        super().setup()
        # One absolute deadline starts at the beginning of the request read and is never reset by
        # a later phase: headers, body, gate wait and response write all share it.
        self._deadline = time.monotonic() + self.server.request_timeout_s
        # Bind the one absolute request deadline to this request thread. The critical commit
        # rechecks it inside the state's coordination boundary without taking a new argument,
        # so a delay between the last handler check and the commit cannot create a stale commit.
        self.server.site_state.bind_request_deadline(self._deadline)
        self._finished = False
        self.connection.settimeout(self.server.request_timeout_s)
        self.rfile = _DeadlineReader(self)  # type: ignore[assignment]

    def _deadline_expired(self) -> bool:
        """Whether the one absolute request deadline has already passed."""
        return time.monotonic() >= self._deadline

    def log_message(self, format: str, *args: object) -> None:
        """Never write request data, credentials or a stack to a log."""

    def handle_one_request(self) -> None:
        """Run one request and always finish with a closed connection."""
        try:
            super().handle_one_request()
        finally:
            self.close_connection = True

    # ------------------------------------------------------------------ methods

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_HEAD(self) -> None:
        self._handle("HEAD")

    def do_PUT(self) -> None:
        self._handle("PUT")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def do_PATCH(self) -> None:
        self._handle("PATCH")

    def do_OPTIONS(self) -> None:
        self._handle("OPTIONS")

    def do_TRACE(self) -> None:
        self._handle("TRACE")

    def do_CONNECT(self) -> None:
        self._handle("CONNECT")

    def __getattr__(self, name: str) -> object:
        if name.startswith("do_"):
            return lambda: self._handle(name[3:])
        raise AttributeError(name)

    # ------------------------------------------------------------------ routing

    def _handle(self, method: str) -> None:
        if not self.server.site_state.begin_request():
            self.close_connection = True
            return
        try:
            if self._deadline_expired():
                # The budget was already spent while the headers were arriving. No new normal
                # commit may happen, so answer with the fixed pre-commit refusal.
                self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self._route(method)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        finally:
            self.server.site_state.end_request()
            self.close_connection = True

    def _route(self, method: str) -> None:
        target = self.path
        if "?" in target or "#" in target:
            self._send_text(HTTPStatus.NOT_FOUND)
            return
        path = urlsplit(target).path
        if path not in _KNOWN_PATHS:
            self._send_text(HTTPStatus.NOT_FOUND)
            return
        if method not in _METHODS_FOR_PATH[path]:
            self._send_text(
                HTTPStatus.METHOD_NOT_ALLOWED,
                extra=[("Allow", ", ".join(_METHODS_FOR_PATH[path]))],
            )
            return
        if path == "/":
            self._send_redirect("/login")
        elif path == "/login":
            if method == "GET":
                self._send_login_page()
            else:
                self._handle_login()
        else:
            self._send_dashboard()

    # ------------------------------------------------------------------ pages

    def _send_login_page(self, *, error: bool = False) -> None:
        state = self.server.site_state
        body = render_login_page(
            generation=state.generation,
            error=error,
            overlay=state.overlay_is_active(),
        ).encode("utf-8")
        self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def _send_dashboard(self) -> None:
        state = self.server.site_state
        if state.session_is_valid(self._session_token()):
            body = render_dashboard_page().encode("utf-8")
            self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
        else:
            self._send_redirect("/login")

    # ------------------------------------------------------------------ login

    def _session_token(self) -> str | None:
        header = self.headers.get("Cookie")
        if not header:
            return None
        token: str | None = None
        for part in header.split(";"):
            name, separator, value = part.strip().partition("=")
            if separator and name == _COOKIE_NAME:
                token = value
        return token

    def _handle_login(self) -> None:
        if self.headers.get_all("Transfer-Encoding"):
            self._send_text(HTTPStatus.BAD_REQUEST)
            return
        declared = parse_content_length(self.headers.get_all("Content-Length"))
        if declared is None:
            self._send_text(HTTPStatus.BAD_REQUEST)
            return
        if declared > _MAX_BODY_BYTES:
            self._send_text(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        if not _content_type_ok(self.headers.get("Content-Type")):
            self._send_text(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        body = self._read_body(declared)
        if body is None:
            if self._deadline_expired() or not self.server.site_state.running:
                # The request budget (or a stop) ended the read before the body was complete:
                # this is the fixed pre-commit refusal, never a business commit.
                self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self._send_text(HTTPStatus.BAD_REQUEST)
            return
        form = parse_form_body(body)
        if form is None:
            self._send_text(HTTPStatus.BAD_REQUEST)
            return
        generation = parse_generation(form["generation"])
        state = self.server.site_state
        if generation is None or not state.generation_matches(generation):
            self._send_text(HTTPStatus.BAD_REQUEST)
            return
        if self._deadline_expired():
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if not state.record_login_received():
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if (
            form["username"] != FIXTURE_USERNAME
            or form["password"] != FIXTURE_PASSWORD
        ):
            state.record_auth_failure()
            self._send_login_page(error=True)
            return
        scenario = state.scenario
        if scenario is Scenario.DELAYED_LOGIN:
            self._precommit_gate()
            return
        if scenario is Scenario.COMMIT_THEN_STALL:
            self._commit_then_gate()
            return
        if self._deadline_expired():
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        token = state.commit_session()
        if token is None:
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._send_success(token)

    def _read_body(self, declared: int) -> bytes | None:
        """Read the declared body under the one request deadline, polling for a stop.

        Reads are driven by the deadline-aware reader, so a legal partial body that arrives after
        short polls continues normally and the aggregate read is bounded by the one absolute
        deadline. A stop aborts the read instead of waiting for the client.
        """
        chunks: list[bytes] = []
        received = 0
        while received < declared:
            remaining = self._deadline - time.monotonic()
            if (
                remaining <= 0
                or not self.server.site_state.running
                or self.server._closing
            ):
                return None
            try:
                chunk = self.rfile.read1(declared - received)
            except TimeoutError:
                continue
            except (OSError, ValueError):
                return None
            if not chunk:
                return None
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _precommit_gate(self) -> None:
        state = self.server.site_state
        outcome = state.gate_wait(self._deadline)
        if outcome is not GateOutcome.RELEASED or self._deadline_expired():
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        # The gate release can race the budget, so the commit rechecks the same absolute deadline
        # inside its own coordination boundary rather than trusting the pre-gate check.
        token = state.commit_session()
        if token is None:
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._send_success(token)

    def _commit_then_gate(self) -> None:
        state = self.server.site_state
        if self._deadline_expired():
            # The commit must never happen once the one request deadline has expired, even in
            # the post-commit-failure mode.
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        token = state.commit_session()
        if token is None:
            self._close_without_response()
            return
        outcome = state.gate_wait(self._deadline)
        if outcome is GateOutcome.RELEASED and not self._deadline_expired():
            self._send_success(token)
        else:
            # The commit is not undone by an own-deadline close. No success header or cookie is
            # sent before the gate, so the client still has no response confirmation.
            self._close_without_response()

    # ------------------------------------------------------------------ responses

    def _send_success(self, token: str) -> None:
        cookie = f"{_COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Lax"
        self._send_redirect("/dashboard", extra=[("Set-Cookie", cookie)])

    def _send_redirect(
        self, location: str, *, extra: list[tuple[str, str]] | None = None
    ) -> None:
        headers = [("Location", location)] + list(extra or [])
        self._send(HTTPStatus.SEE_OTHER, b"", None, headers)

    def _send_text(
        self, status: int, *, extra: list[tuple[str, str]] | None = None
    ) -> None:
        self._send(status, _FIXED_TEXT[status], "text/plain; charset=utf-8", extra)

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str | None,
        extra: list[tuple[str, str]] | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            remaining = self._deadline - time.monotonic()
            if remaining > 0:
                # The response shares the one absolute request deadline; there is no fixed floor.
                self.connection.settimeout(remaining)
            else:
                # The budget is spent, so only the fixed refusal or a close may be carried. A
                # normal 200/303 (with its page or session cookie) is no longer a legal outcome:
                # replace it before any byte is written. This bounded write window never extends
                # request processing and can never carry a normal success. The post-commit stall
                # still closes without a response instead of reaching this branch.
                if status in (HTTPStatus.OK, HTTPStatus.SEE_OTHER):
                    status = HTTPStatus.SERVICE_UNAVAILABLE
                    body = _FIXED_TEXT[status]
                    content_type = "text/plain; charset=utf-8"
                    extra = None
                self.connection.settimeout(_EXPIRED_RESPONSE_WRITE_S)
            self.send_response_only(status)
            if content_type is not None:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            for name, value in extra or ():
                self.send_header(name, value)
            self.end_headers()
            if body:
                self.wfile.write(body)
                self.wfile.flush()
        except (OSError, ValueError):
            pass
        finally:
            self.close_connection = True

    def _close_without_response(self) -> None:
        """Close the connection without sending a status line, header or cookie."""
        self._finished = True
        self.close_connection = True
        _close_socket(self.connection)


class _SiteHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """A non-daemon thread-per-connection listener that owns its connections and threads."""

    daemon_threads = False
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[_RequestHandler],
        *,
        state: SiteState,
        request_timeout_s: float,
    ) -> None:
        self.site_state = state
        self.request_timeout_s = request_timeout_s
        self._track_lock = threading.Lock()
        self._threads: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._closing = False
        try:
            super().__init__(server_address, handler)
        except BaseException:
            bound = getattr(self, "socket", None)
            if bound is not None:
                _close_socket(bound)
            raise

    def server_bind(self) -> None:
        """Bind without a host-name lookup; the site never needs a DNS or network read."""
        socketserver.TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = int(self.server_address[1])

    def serve_until_stopped(self) -> None:
        """Accept requests until the stop flag is set; never block past the poll slice.

        This replaces ``serve_forever`` so cleanup never has to call the unbounded
        ``shutdown()``: setting ``_closing`` and closing the listener socket is the whole
        coordination step, and the thread leaves on its next bounded poll.
        """
        while not self._closing:
            try:
                readable, _, _ = select.select([self.socket], [], [], _POLL_S)
            except (OSError, ValueError):
                return
            if self._closing:
                return
            if readable:
                self._handle_request_noblock()

    def process_request(self, request: socket.socket, client_address: tuple) -> None:
        with self._track_lock:
            if self._closing:
                _close_socket(request)
                return
            thread = threading.Thread(
                target=self.process_request_thread,
                args=(request, client_address),
                name="test-site-request",
                daemon=False,
            )
            self._threads.add(thread)
            self._connections.add(request)
        try:
            thread.start()
        except BaseException:
            with self._track_lock:
                self._threads.discard(thread)
                self._connections.discard(request)
            _close_socket(request)
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: tuple
    ) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:  # noqa: BLE001,S110 - one handler failure is closed, never logged
            pass
        finally:
            try:
                self.shutdown_request(request)
            except Exception:  # noqa: BLE001 - the connection is already owned and closing
                _close_socket(request)
            with self._track_lock:
                self._threads.discard(threading.current_thread())
                self._connections.discard(request)

    def handle_error(self, request: socket.socket, client_address: tuple) -> None:
        """Keep a handler failure out of stderr; cleanup still owns the connection."""

    def server_close(self) -> None:
        """Close the listening socket only; thread joining is bounded by the caller."""
        self._closing = True
        try:
            self.socket.close()
        except OSError:
            pass

    def _acquire_track_lock(self, deadline: float | None) -> bool:
        """Acquire the tracker lock within the caller's one cleanup deadline, if one is given."""
        if deadline is None:
            self._track_lock.acquire()
            return True
        return _acquire_within(self._track_lock, deadline)

    def close_owned_connections(self, deadline: float | None = None) -> bool:
        """Close every owned connection, bounded by the caller's cleanup deadline.

        Returns ``False`` when the tracker lock is still held by another coordinator when the
        deadline expires; a truthful resource count then reports the connections that remain.
        """
        if not self._acquire_track_lock(deadline):
            return False
        try:
            connections = list(self._connections)
        finally:
            self._track_lock.release()
        for connection in connections:
            _close_socket(connection)
        return True

    def open_connection_count(self, deadline: float | None = None) -> int:
        """Count still-open owned connections without inventing a zero on a contended lock."""
        if not self._acquire_track_lock(deadline):
            # A set length is a single atomic read in CPython; it can over-approximate a socket
            # that was just closed but not yet untracked, and can never hide a live connection.
            return len(self._connections)
        try:
            connections = list(self._connections)
        finally:
            self._track_lock.release()
        return sum(1 for connection in connections if connection.fileno() >= 0)

    def listener_is_closed(self) -> bool:
        """Whether the one owned listening socket has actually been closed."""
        return self.socket.fileno() < 0

    def join_owned_threads(self, deadline: float) -> bool:
        """Join every owned request thread, bounded by the caller's cleanup deadline."""
        if not self._acquire_track_lock(deadline):
            return False
        try:
            threads = list(self._threads)
        finally:
            self._track_lock.release()
        for thread in threads:
            if not _wait_for_thread(thread, deadline):
                return False
        return True

    def live_thread_count(self, deadline: float | None = None) -> int:
        """Count live owned request threads without inventing a zero on a contended lock."""
        if not self._acquire_track_lock(deadline):
            # The tracked set only shrinks when a thread's own ``finally`` discards it, so its
            # length is a truthful non-zero-bound count of threads that were still owned.
            return len(self._threads)
        try:
            threads = list(self._threads)
        finally:
            self._track_lock.release()
        return sum(1 for thread in threads if thread.is_alive())


def _copy_config(config: object) -> SiteConfig:
    """Read a public configuration once, mapping every ordinary read failure to one fixed error.

    A caller may pass a ``SiteConfig`` subclass that raises from an attribute read. Such an
    ordinary failure must not leak its value or its exception chain, so the read is guarded and
    the fixed :class:`LocalSiteError` is raised *after* the handler has ended: the new error then
    has no reachable cause or context. ``BaseException`` (cancellation, interrupt, exit) is never
    converted here.
    """
    if not _is_site_config(config):
        raise LocalSiteError("SITE_CONFIG_INVALID")
    failed = False
    scenario: object = None
    request_timeout_ms: object = None
    cleanup_timeout_ms: object = None
    try:
        scenario = config.scenario
        request_timeout_ms = config.request_timeout_ms
        cleanup_timeout_ms = config.cleanup_timeout_ms
    except Exception:  # noqa: BLE001 - each ordinary read failure maps to one fixed error
        failed = True
    if failed:
        raise LocalSiteError("SITE_CONFIG_INVALID")
    return SiteConfig(
        scenario=scenario,  # type: ignore[arg-type]
        request_timeout_ms=request_timeout_ms,  # type: ignore[arg-type]
        cleanup_timeout_ms=cleanup_timeout_ms,  # type: ignore[arg-type]
    )


class LocalTestSite:
    """One host-controlled local test site bound to an ephemeral loopback port."""

    _NEW: Final[str] = "new"
    _RUNNING: Final[str] = "running"
    _STOPPED: Final[str] = "stopped"

    def __init__(self, config: SiteConfig) -> None:
        self._config = _copy_config(config)
        self._state = SiteState(self._config.scenario)
        self._lock = threading.RLock()
        self._phase = self._NEW
        self._server: _SiteHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._origin: str | None = None
        self._port = 0

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> LocalTestSite:
        """Bind ``127.0.0.1:0`` exactly once; a running instance returns itself."""
        with self._lock:
            if self._phase is self._RUNNING:
                return self
            if self._phase is self._STOPPED:
                raise LocalSiteError("SITE_ALREADY_STOPPED")
            config = _copy_config(self._config)
            self._config = config
            server: _SiteHTTPServer | None = None
            thread: threading.Thread | None = None
            failure: BaseException | None = None
            try:
                server = _SiteHTTPServer(
                    ("127.0.0.1", 0),
                    _RequestHandler,
                    state=self._state,
                    request_timeout_s=config.request_timeout_ms / 1000.0,
                )
                port = int(server.server_address[1])
                thread = threading.Thread(
                    target=server.serve_until_stopped,
                    name="test-site-listener",
                    daemon=False,
                )
                thread.start()
            except BaseException as exc:  # noqa: BLE001 - cleanup, then identity-preserving re-raise
                # Capture the failure, release every resource under one bounded cleanup
                # deadline, and only re-raise after leaving the handler. Raising outside the
                # handler is what keeps the fixed replacement error free of a reachable
                # ``__context__``; ``from None`` alone would not remove it.
                failure = exc
                deadline = time.perf_counter() + config.cleanup_timeout_ms / 1000.0
                self._state.mark_stopping(deadline)
                if server is not None:
                    server._closing = True
                    server.server_close()
                    server.close_owned_connections(deadline)
                    server.join_owned_threads(deadline)
                if thread is not None and thread.is_alive():
                    _wait_for_thread(thread, deadline)
            if failure is not None:
                if isinstance(failure, Exception):
                    raise LocalSiteError("SITE_START_FAILED")
                raise failure
            self._server = server
            self._thread = thread
            self._port = port
            self._origin = f"http://127.0.0.1:{port}"
            self._state.mark_running()
            self._phase = self._RUNNING
            return self

    @property
    def origin(self) -> str:
        """The actual ephemeral origin; refused before a successful start."""
        with self._lock:
            if self._origin is None:
                raise LocalSiteError("SITE_NOT_RUNNING")
            return self._origin

    def snapshot_oracle(self) -> SiteOracleSnapshot:
        """Return one consistent immutable snapshot of the current generation."""
        return self._state.snapshot()

    def wait_for(self, *, event: SiteEvent, timeout_ms: int) -> bool:
        """Wait for one phase fact in the generation current at the call boundary."""
        return self._state.wait_for(event, timeout_ms)

    def release(self) -> None:
        """Release this generation's scenario gate or overlay."""
        self._state.release()

    def reset(self, *, scenario: Scenario) -> int:
        """Reset to a clean generation when idle; return the new generation."""
        return self._state.reset(scenario)

    def stop(self) -> SiteStopResult:
        """Perform one bounded, truthful cleanup attempt over every owned resource.

        The one absolute ``perf_counter`` deadline starts here and bounds every later phase: the
        instance coordination lock, the state lock, listener stop, connection close, thread
        joins, every resource count and the port proof. No phase gets a fresh budget and no
        internal coordination acquisition may wait past that deadline. A contended lock therefore
        cannot delay the call: the attempt returns a truthful incomplete result with the real
        observed counts, and a later bounded attempt stays safe and idempotent.
        """
        started = time.perf_counter()
        budget_s = self._config.cleanup_timeout_ms / 1000.0
        deadline = started + budget_s
        lock_held = _acquire_within(self._lock, deadline)
        phase = self._phase
        if phase is self._NEW:
            if not lock_held:
                # A concurrent start may still own the transition; nothing is claimed complete.
                return SiteStopResult(
                    complete=False,
                    active_requests=0,
                    open_connections=0,
                    live_threads=0,
                )
            try:
                coordinated = self._state.mark_stopping(deadline)
                self._phase = self._STOPPED
            finally:
                self._lock.release()
            return SiteStopResult(
                complete=coordinated,
                active_requests=self._state.active_request_count(deadline),
                open_connections=0,
                live_threads=0,
            )
        server = self._server
        thread = self._thread
        port = self._port
        self._phase = self._STOPPED
        if lock_held:
            self._lock.release()
        # Refusing new work and waking gates is itself a bounded coordination acquisition: a
        # contended state lock reports an incomplete attempt instead of blocking the call.
        coordinated = self._state.mark_stopping(deadline)
        if server is not None:
            server._closing = True
            # Closing the listener releases the port immediately and is non-blocking; there is
            # no unbounded ``server.shutdown()`` wait before the bounded joins.
            server.server_close()
            # Give a request that this stop just woke a bounded chance to deliver its fixed
            # pre-commit 503 before its own socket is closed. The wait shares the deadline.
            self._state.wait_for_idle(deadline)
            server.close_owned_connections(deadline)
            server.join_owned_threads(deadline)
        if thread is not None and thread.is_alive():
            _wait_for_thread(thread, deadline)
        open_connections = 0 if server is None else server.open_connection_count(deadline)
        live_threads = 0 if server is None else server.live_thread_count(deadline)
        if thread is not None and thread.is_alive():
            live_threads += 1
        active_requests = self._state.active_request_count(deadline)
        listener_closed = server is not None and server.listener_is_closed()
        port_released = self._port_is_released(
            port, deadline, listener_closed=listener_closed
        )
        return SiteStopResult(
            complete=(
                coordinated
                and port_released
                and active_requests == 0
                and open_connections == 0
                and live_threads == 0
            ),
            active_requests=active_requests,
            open_connections=open_connections,
            live_threads=live_threads,
        )

    @staticmethod
    def _port_is_released(
        port: int, deadline: float, *, listener_closed: bool = False
    ) -> bool:
        """Prove the listener port is not connectable, bounded by the cleanup deadline.

        A blocking ``connect_ex`` to a closed loopback port waits for the whole socket timeout
        on this platform, so the proof uses one non-blocking connect plus one zero-wait
        readiness check. A live listener completes the loopback handshake immediately and is
        therefore observed; no independent probe timeout is invented and the check cannot
        consume the budget. When less than the cost of one probe remains, an already closed
        owned listening socket is itself definitive: nothing owned can accept on that port any
        more, so the proof returns without asking for extra time. When the cleanup deadline
        has already fully passed, completeness is not claimed.
        """
        if port <= 0:
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        if listener_closed and remaining < _PORT_PROBE_MIN_S:
            return True
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setblocking(False)
            code = probe.connect_ex(("127.0.0.1", port))
            if code == 0:
                return False
            try:
                _, writable, _ = select.select([], [probe], [], 0.0)
            except OSError:
                return True
            if not writable:
                return True
            return probe.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) != 0
        except OSError:
            return True
        finally:
            probe.close()

    # ------------------------------------------------------------------ host context

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        result = self.stop()
        if exc_type is None and not result.complete:
            raise LocalSiteError("SITE_CONTROL_INVALID") from None
        return False
