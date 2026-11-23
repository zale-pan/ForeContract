"""Browser session ownership, local-request policy and bounded resource cleanup.

This module is the one place the optional ``playwright`` dependency is touched. Importing it
never imports Playwright: the dependency is loaded lazily inside :meth:`PlaywrightBackend.
open_driver`, so the Web distribution can be imported, installed and schema-checked without the
optional dependency or any browser binary.

One :class:`BrowserSession` owns exactly one Playwright driver, one browser, one browser context
and one active page. The fixed launch policy is headless Chromium with no CDP attachment to an
existing browser, no existing browser or user profile, no proxy setting and no caller-supplied
launch argument. Downloads are explicitly disabled in the browser context. Every HTTP(S) request
and every redirect hop is checked against the configured canonical loopback origins; the check is
installed on the adapter's own page with a CDP session that belongs to this instance, which is
what lets the browser keep its native redirect semantics (real final document URL, relative
resource and base-origin resolution) while each hop is still validated before it is sent. The
same per-hop policy is installed on every other owned page - an unsupported popup included - as
soon as its page event arrives; a popup request that Playwright reports before the page/frame
exists cannot be attributed to any owned page, so it is refused outright rather than released.
WebSocket connections are refused.

Cleanup is bounded by one absolute monotonic deadline for the whole attempt and proceeds
Page -> Context -> Browser -> Playwright. An ordinary close failure does not stop the remaining
independent closes, a timed-out or cancelled close keeps its handle for an honest later retry, and
a complete cleanup is reported only when every owned handle really is gone. A close whose
cancellation is itself delayed is abandoned at the deadline instead of extending the attempt, and
a handle that a late, cancelled acquisition still produces is either closed inside the remaining
budget or kept with a visible bounded failure for a later attempt. Only owned handles are closed:
no unrelated browser process is enumerated or terminated. The module claims no OS-level release
after a strong kill, a driver loss, a power failure or a storage failure.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from typing import Final, NamedTuple

from forecontract_web.browser_config import Origin, request_origin_allowed
from forecontract_web.browser_errors import WebBrowserError

__all__ = ["BrowserSession", "BudgetExpired", "CleanupResult", "SessionLost"]

#: The abort reason used for every refused request.
_BLOCKED_REASON: Final[str] = "blockedbyclient"
#: The fixed CDP failure reason for a request that the local origin policy refuses.
_BLOCKED_FAILURE: Final[str] = "BlockedByClient"


def _consume_exception(task: asyncio.Task) -> None:
    """Read the exception of an abandoned owned task so the loop does not warn about it."""
    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:  # noqa: BLE001, S110 - the abandoned task's failure is not a public outcome
        pass


class SessionLost(Exception):
    """Internal signal that the whole browser session is no longer usable."""


class BudgetExpired(Exception):
    """Internal signal that the one monotonic operation or cleanup budget is exhausted."""


class BackendTypes(NamedTuple):
    """The Playwright exception classes this adapter classifies by identity, never by text."""

    timeout: type[BaseException]
    error: type[BaseException]


class CleanupResult(NamedTuple):
    """The outcome of one complete bounded cleanup attempt."""

    complete: bool
    error: WebBrowserError | None


def remaining_seconds(deadline: float) -> float:
    """Return the non-resetting time left before one absolute monotonic deadline."""
    return deadline - time.monotonic()


def remaining_ms(deadline: float) -> int:
    """Return the remaining budget as a strictly positive millisecond timeout.

    Playwright treats a zero timeout as "no timeout", so zero is never handed to it: an expired
    budget raises :class:`BudgetExpired` instead, which the public method maps to its own fixed
    timeout diagnostic.
    """
    left = remaining_seconds(deadline)
    if left <= 0:
        raise BudgetExpired()
    milliseconds = int(left * 1000)
    return milliseconds if milliseconds > 0 else 1


class PlaywrightBackend:
    """The lazily imported Playwright backend with the fixed local-only request policy."""

    def __init__(self, origins: frozenset[Origin]) -> None:
        self.origins = origins
        self.types: BackendTypes | None = None
        self.blocked = False
        self.navigation_blocked = False
        self._manager: object | None = None
        self._context: object | None = None
        self._policy_sessions: list[object] = []
        self._policied_pages: set[int] = set()
        self._policy_tasks: dict[int, asyncio.Task] = {}

    # ------------------------------------------------------------------ policy marks

    def clear_marks(self) -> None:
        """Forget the blocked-request marks of a previous action."""
        self.blocked = False
        self.navigation_blocked = False

    # ------------------------------------------------------------------ acquisition

    async def open_driver(self) -> object:
        """Import the optional dependency and start one owned Playwright driver."""
        module = importlib.import_module("playwright.async_api")
        self.types = BackendTypes(module.TimeoutError, module.Error)
        manager = module.async_playwright()
        self._manager = manager
        return await manager.start()

    async def open_browser(self, driver: object) -> object:
        """Launch the fixed headless Chromium with no caller-supplied options."""
        return await driver.chromium.launch(headless=True)  # type: ignore[attr-defined]

    async def open_context(self, browser: object) -> object:
        """Create the isolated context, refuse downloads and install the origin policy.

        The context is created with downloads explicitly disabled, so a synthetic attachment is
        refused by Chromium instead of being written anywhere. The Playwright route callback keeps
        the local-origin policy for every owned page (notably a popup); the per-hop redirect check
        is installed on every owned page - the active page during :meth:`open_page`, an
        unsupported page as soon as its page event arrives, before any of its attributed requests
        is released.
        """
        context = await browser.new_context(  # type: ignore[attr-defined]
            service_workers="block",
            accept_downloads=False,
        )
        self._context = context
        await context.route("**/*", self._handle_route)
        await context.route_web_socket("**/*", self._handle_websocket)
        return context

    async def open_page(self, context: object) -> object:
        """Create the one active page of this session and install its per-hop policy."""
        page = await context.new_page()  # type: ignore[attr-defined]
        await self._install_page_policy(context, page)
        return page

    # ------------------------------------------------------------------ request policy

    async def _install_page_policy(self, context: object, page: object) -> None:
        """Validate every request hop of the owned page, including server-side redirects.

        Playwright's route callback is not invoked for a server-side redirect hop, so the
        origin check is additionally installed on this adapter's own page through a CDP session
        that belongs to this instance. Redirects are then followed natively by the browser: the
        final document URL, its relative resources and its base origin stay the browser's own,
        while each hop is checked before it is sent. This creates no session on an existing
        browser and passes no CDP setting or launch argument from a caller.
        """
        create = getattr(context, "new_cdp_session", None)
        if not callable(create):
            # The fixed Chromium backend always exposes it. Without per-hop interception the
            # policy would fail open on a server-side redirect hop, so the session refuses to
            # start instead of silently weakening the local-only boundary.
            raise TypeError("the browser context does not expose per-hop request interception")
        session = await create(page)
        session.on(  # type: ignore[attr-defined]
            "Fetch.requestPaused",
            lambda params: asyncio.ensure_future(self._handle_paused(session, params)),
        )
        await session.send(  # type: ignore[attr-defined]
            "Fetch.enable",
            {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]},
        )
        self._policy_sessions.append(session)
        self._policied_pages.add(id(page))

    async def _enable_page_policy(self, page: object) -> bool:
        """Install the per-hop policy on one owned page before releasing its requests.

        The active page already carries the policy from :meth:`open_page`; an unsupported page
        (a popup) receives exactly the same redirect/origin policy as soon as its page event
        arrives. Every request that can be attributed to the page is held here until the hop
        interception is actually in place, so a server-side redirect hop can never be released
        before the check. If the policy cannot be installed the request is refused instead of
        falling open.
        """
        key = id(page)
        if key in self._policied_pages:
            return True
        task = self._policy_tasks.get(key)
        if task is None:
            context = self._context
            if context is None:
                return False
            task = asyncio.ensure_future(self._install_page_policy(context, page))
            self._policy_tasks[key] = task
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the request is refused, never allowed, on failure
            self._policy_tasks.pop(key, None)
            return False
        self._policy_tasks.pop(key, None)
        return key in self._policied_pages

    async def _handle_paused(self, session: object, params: object) -> None:
        """Continue one approved paused request or refuse an unapproved one, hop by hop."""
        try:
            request = params.get("request", {})  # type: ignore[union-attr]
            url = request.get("url")
            request_id = params.get("requestId")  # type: ignore[union-attr]
            navigation = params.get("resourceType") == "Document"  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - an unreadable event is refused, never allowed
            return
        if not isinstance(request_id, str):
            return
        allowed = isinstance(url, str) and request_origin_allowed(url, self.origins)
        if not allowed:
            self.blocked = True
            if navigation:
                self.navigation_blocked = True
        action = "Fetch.continueRequest" if allowed else "Fetch.failRequest"
        command: dict[str, object] = {"requestId": request_id}
        if not allowed:
            command["errorReason"] = _BLOCKED_FAILURE
        try:
            await session.send(action, command)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - a closed target needs no further action
            pass

    async def _handle_route(self, route: object) -> None:
        """Keep the local-origin policy on every owned page, including a popup.

        A request that can be attributed to an owned page is held until that page carries the
        per-hop policy, then checked and released unchanged. Playwright reports a popup's first
        navigation before the page/frame exists, so that request cannot be attributed and no
        per-hop interception can be installed for it; it is refused outright instead of being
        guessed or released. No unsupported popup therefore reaches any destination - approved
        or not - through that unattributable entry, and every later request of the page is
        checked by the same policy installed on that page.
        """
        try:
            request = route.request  # type: ignore[attr-defined]
            url = request.url
            navigation = bool(request.is_navigation_request())
        except Exception:  # noqa: BLE001 - an unreadable request is refused, never allowed
            self.blocked = True
            await self._abort(route)
            return
        try:
            page = request.frame.page
        except Exception:  # noqa: BLE001 - a popup navigation is reported before its frame exists
            self.blocked = True
            if navigation:
                self.navigation_blocked = True
            await self._abort(route)
            return
        if not await self._enable_page_policy(page):
            self.blocked = True
            if navigation:
                self.navigation_blocked = True
            await self._abort(route)
            return
        if not request_origin_allowed(url, self.origins):
            self.blocked = True
            if navigation:
                self.navigation_blocked = True
            await self._abort(route)
            return
        try:
            await route.continue_()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - a closed route needs no further action
            pass

    async def _abort(self, route: object) -> None:
        try:
            await route.abort(_BLOCKED_REASON)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - a closed route needs no further action
            pass

    async def _handle_websocket(self, websocket: object) -> None:
        """Refuse every WebSocket connection instead of exposing an unreviewed transport."""
        self.blocked = True
        try:
            await websocket.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - a closed socket needs no further action
            pass


def _page_closed(page: object) -> bool:
    """Ask one owned page whether it is closed, treating an unreadable state as open."""
    try:
        return bool(page.is_closed())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - an unreadable owned handle is kept for a retry
        return False


class BrowserSession:
    """One owned browser resource chain with a bounded, ordered cleanup."""

    def __init__(self, backend: PlaywrightBackend) -> None:
        self._backend = backend
        self._driver: object | None = None
        self._browser: object | None = None
        self._context: object | None = None
        self._page: object | None = None
        self._extra_pages: list[object] = []
        self._released = False
        self._cleanup_deadline: float | None = None
        self._late_handles: list[tuple[object, str]] = []
        self._late_failure: WebBrowserError | None = None

    # ------------------------------------------------------------------ views

    @property
    def page(self) -> object | None:
        """The one active page, or ``None`` when it was released."""
        return self._page

    @property
    def types(self) -> BackendTypes | None:
        """The Playwright exception classes, available once the driver started."""
        return self._backend.types

    @property
    def origins(self) -> frozenset[Origin]:
        """The canonical local origins that may be requested."""
        return self._backend.origins

    @property
    def blocked(self) -> bool:
        """Whether any request was refused since the marks were last cleared."""
        return self._backend.blocked

    @property
    def navigation_blocked(self) -> bool:
        """Whether a navigation request, including a redirect hop, was refused."""
        return self._backend.navigation_blocked

    @property
    def late_failure(self) -> WebBrowserError | None:
        """The bounded cleanup failure of a close that appeared after the attempt had ended."""
        return self._late_failure

    def has_pending_cleanup(self) -> bool:
        """Whether one unconfirmed or failed closed-over handle is still retained."""
        return bool(self._late_handles) or self._late_failure is not None

    def clear_marks(self) -> None:
        """Clear the blocked-request marks before a navigation action."""
        self._backend.clear_marks()

    def is_connected(self) -> bool:
        """Report whether the owned session still looks usable.

        A page that reports closed, a browser that reports disconnected or an unreadable owned
        handle is a dead session. The answer is deliberately conservative: ``False`` means the
        session cannot be trusted, not that a target is absent.
        """
        page = self._page
        if page is None:
            return False
        if _page_closed(page):
            return False
        browser = self._browser
        if browser is not None:
            try:
                if not browser.is_connected():  # type: ignore[attr-defined]
                    return False
            except Exception:  # noqa: BLE001 - an unreadable owned handle is dead
                return False
        return True

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Acquire the owned chain, assigning each handle as soon as it exists.

        A cleanup attempt may begin while this acquisition is still cancelled but not yet
        settled (a backend call is allowed to delay its own cancellation). Each handle obtained
        after that point is closed inside the remaining cleanup budget - or kept in
        ``_late_handles`` for an honest later attempt - and is never retained as if the session
        were still running.
        """
        missing = False
        failed = False
        driver: object | None = None
        try:
            driver = await self._backend.open_driver()
        except asyncio.CancelledError:
            raise
        except ImportError:
            missing = True
        except Exception:  # noqa: BLE001 - a backend launch failure becomes a start error
            failed = True
        if missing:
            raise WebBrowserError("WEB_BROWSER_DEPENDENCY_MISSING", "start")
        if failed:
            raise WebBrowserError("WEB_BROWSER_START_FAILED", "start")
        if driver is not None:
            if self._released:
                await self._close_late(driver, "stop")
                raise SessionLost()
            self._driver = driver
        try:
            browser = await self._backend.open_browser(self._driver)
            if self._released:
                await self._close_late(browser, "close")
                raise SessionLost()
            self._browser = browser
            context = await self._backend.open_context(self._browser)
            if self._released:
                await self._close_late(context, "close")
                raise SessionLost()
            self._context = context
            page = await self._backend.open_page(self._context)
            if self._released:
                await self._close_late(page, "close")
                raise SessionLost()
            self._page = page
            self._watch_pages()
        except asyncio.CancelledError:
            raise
        except (WebBrowserError, SessionLost):
            raise
        except Exception:  # noqa: BLE001 - a backend acquisition failure becomes a start error
            failed = True
        if failed:
            raise WebBrowserError("WEB_BROWSER_START_FAILED", "start")

    def _mark_late_handle(self, handle: object, method: str) -> None:
        """Keep one unconfirmed late close and expose its bounded cleanup failure.

        A close that cannot be confirmed must never be consumed and discarded: the handle stays
        owned for a later attempt and the failure stays visible to the adapter. The failure is
        recorded before any later reader can observe an empty handle list, so a late resource is
        never reported as a complete ``STOPPED`` cleanup.
        """
        if self._late_failure is None:
            self._late_failure = WebBrowserError("WEB_BROWSER_CLEANUP_FAILED", "cleanup")
        self._late_handles.append((handle, method))

    async def _close_late(self, handle: object, method: str) -> None:
        """Release one handle obtained after cleanup started, without exceeding its budget."""
        call = getattr(handle, method, None)
        if not callable(call):
            return
        deadline = self._cleanup_deadline
        left = remaining_seconds(deadline) if deadline is not None else 0.0
        if left <= 0:
            # The one cleanup attempt has already finished. This handle appeared later in an
            # abandoned acquisition, so no public caller is waiting: close it best-effort and
            # keep it - with the bounded failure - for a later attempt only when that close
            # cannot be confirmed.
            try:
                await call()
            except asyncio.CancelledError:
                self._mark_late_handle(handle, method)
                raise
            except Exception:  # noqa: BLE001 - an unclosable late handle stays tracked
                self._mark_late_handle(handle, method)
            return
        try:
            task = asyncio.ensure_future(call())
        except Exception:  # noqa: BLE001 - an unstartable close is retried by a later attempt
            self._mark_late_handle(handle, method)
            return
        try:
            done, _ = await asyncio.wait({task}, timeout=left)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(_consume_exception)
            self._mark_late_handle(handle, method)
            raise
        if task not in done:
            task.cancel()
            task.add_done_callback(_consume_exception)
            self._mark_late_handle(handle, method)
            return
        if task.cancelled():
            # A cancelled close cannot prove the resource is gone; keep it retryable.
            self._mark_late_handle(handle, method)
            return
        try:
            task.result()
        except Exception:  # noqa: BLE001 - the failure is kept visible and retryable
            self._mark_late_handle(handle, method)

    def _watch_pages(self) -> None:
        """Track and close any unsupported page a page or popup opens later."""
        context = self._context
        if context is None or self._page is None:
            return
        register = getattr(context, "on", None)
        if not callable(register):
            return
        register("page", self._track_page)

    def _track_page(self, new_page: object) -> None:
        if new_page is self._page:
            return
        if any(new_page is existing for existing in self._extra_pages):
            return
        self._extra_pages.append(new_page)
        try:
            asyncio.ensure_future(self._secure_unsupported_page(new_page))
        except RuntimeError:
            pass

    async def _secure_unsupported_page(self, page: object) -> None:
        """Install the same per-hop policy on one owned unsupported page, then close it.

        The policy is installed before the close so that any request the popup issues after its
        page event is attributed to a page that already carries the redirect/origin policy. The
        close itself is delegated to :meth:`_close_extra`, which the tests may delay to prove
        that the hop check - not the close - is what refuses the unapproved destination.
        """
        try:
            await self._backend._enable_page_policy(page)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001, S110 - an unsecurable page is still closed below
            pass
        finally:
            await self._close_extra(page)

    async def _close_extra(self, page: object) -> None:
        try:
            await page.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - the owned page is retried by the next cleanup
            pass

    # ------------------------------------------------------------------ cleanup

    async def cleanup(self, deadline: float) -> CleanupResult:
        """Release every owned handle once, in order, inside one absolute deadline.

        Each close is attempted at most once per attempt. An ordinary close failure records a
        bounded failure and does not prevent the remaining independent closes. A close that
        exceeds the budget - including one whose own cancellation is delayed - records a timeout
        and keeps its handle for an honest later retry. The deadline is absolute and is never
        reset by a joining caller.
        """
        self._released = True
        self._cleanup_deadline = deadline
        timeout_seen = False

        for extra in list(self._extra_pages):
            outcome = await self._attempt_close(extra, "close", deadline)
            if outcome == "closed":
                self._extra_pages = [
                    existing for existing in self._extra_pages if existing is not extra
                ]
            elif outcome == "timeout":
                timeout_seen = True

        for handle, method in list(self._late_handles):
            outcome = await self._attempt_close(handle, method, deadline)
            if outcome == "closed":
                self._late_handles = [
                    entry for entry in self._late_handles if entry[0] is not handle
                ]
            elif outcome == "timeout":
                timeout_seen = True

        for attribute, method in (
            ("_page", "close"),
            ("_context", "close"),
            ("_browser", "close"),
            ("_driver", "stop"),
        ):
            handle = getattr(self, attribute)
            if handle is None:
                continue
            outcome = await self._attempt_close(handle, method, deadline)
            if outcome == "closed":
                setattr(self, attribute, None)
            elif outcome == "timeout":
                timeout_seen = True
            if attribute == "_context":
                self._clear_closed_page()

        self._clear_closed_page()
        self._clear_closed_extras()
        complete = (
            self._page is None
            and self._context is None
            and self._browser is None
            and self._driver is None
            and not self._extra_pages
            and not self._late_handles
        )
        if complete:
            self._late_failure = None
            return CleanupResult(True, None)
        code = "WEB_BROWSER_CLEANUP_TIMEOUT" if timeout_seen else "WEB_BROWSER_CLEANUP_FAILED"
        return CleanupResult(False, WebBrowserError(code, "cleanup"))

    async def _attempt_close(self, handle: object, method: str, deadline: float) -> str:
        """Run one bounded close and report ``closed``, ``timeout`` or ``failed``.

        The close runs in its own owned task and is awaited only until the one cleanup deadline:
        a close whose cancellation is itself delayed cannot extend the public attempt. Such a
        close is abandoned at the deadline and its handle is kept for a later retry.
        """
        left = remaining_seconds(deadline)
        if left <= 0:
            return "timeout"
        call = getattr(handle, method, None)
        if not callable(call):
            return "failed"
        try:
            task = asyncio.ensure_future(call())
        except Exception:  # noqa: BLE001 - an unstartable close is reported as a failure
            return "failed"
        try:
            done, _ = await asyncio.wait({task}, timeout=left)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(_consume_exception)
            raise
        if task not in done:
            task.cancel()
            task.add_done_callback(_consume_exception)
            return "timeout"
        if task.cancelled():
            return "timeout"
        try:
            task.result()
        except Exception:  # noqa: BLE001 - one ordinary close failure is reported, not raised
            return "failed"
        return "closed"

    def _clear_closed_page(self) -> None:
        page = self._page
        if page is not None and _page_closed(page):
            self._page = None

    def _clear_closed_extras(self) -> None:
        self._extra_pages = [
            page for page in self._extra_pages if not _page_closed(page)
        ]
