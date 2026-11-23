"""Native execution of the six Web actions against the owned page.

The six payloads are exactly the ones the pure Web contract already declares; this module adds
no action, field, selector strategy or code entry point. A locator is built only through the
native ``get_by_test_id`` or ``get_by_role(name=..., exact=True)`` API. Hidden nodes are included
in the role lookup so the same locator identity is used for evidence and for actions, while the
native actionability checks still decide whether an action may run. There is no ``first``/``nth``
fallback, no ``force`` override, no CSS/XPath/text selector, no frame or shadow option and no
caller-supplied script.

Composition is fixed:

* the whole call has one monotonic deadline, and every native call receives only the remaining
  time (never a reset and never a zero that Playwright would read as unlimited);
* a deterministic pre-interaction observation - zero matches, several matches, or an element
  that is not visible/enabled/editable - is a ``failed`` outcome with its fixed target code and
  performs no action;
* an ordinary exception raised by a native backend call is converted at that call boundary only:
  before any interaction has been entered it is a bounded ``WEB_BROWSER_UNAVAILABLE`` facility
  error, and after the interaction has been entered it is a conservative ``unknown`` outcome;
* a backend timeout or disconnect after the action was entered is a conservative ``unknown``
  outcome; nothing is retried, re-sent or replayed, and a click, submit, fill or press is never
  attempted a second time;
* a successful backend call is ``succeeded`` for execution only; no ``state_before``,
  ``state_after``, artifact or server acknowledgement is fabricated.

Every raised facility error is constructed outside its handler and carries a fixed code, a fixed
message and a fixed operation label only. ``asyncio.CancelledError`` and process-control
exceptions are never caught at a backend boundary, so a host cancellation always propagates.
"""

from __future__ import annotations

import inspect
import time
import uuid
from typing import Final

from forecontract.models.actions import ActionSpec
from forecontract.models.outcomes import ActualOutcome, ErrorInfo

from forecontract_web._browser_session import (
    BackendTypes,
    BrowserSession,
    BudgetExpired,
    remaining_ms,
)
from forecontract_web.browser_config import request_origin_allowed
from forecontract_web.browser_errors import WebBrowserError, outcome_message

__all__ = ["budget_timeout_outcome", "perform"]

_SCHEMA_VERSION: Final[str] = "0.1"


def _locator(page: object, target: dict) -> object:
    """Build the one allowed native locator for a validated payload target."""
    if target["strategy"] == "test_id":
        return page.get_by_test_id(target["test_id"])  # type: ignore[attr-defined]
    return page.get_by_role(  # type: ignore[attr-defined]
        target["role"],
        name=target["name"],
        exact=True,
        include_hidden=True,
    )


def _outcome(status: str, code: str | None, started: float) -> ActualOutcome:
    """Build one execution-fact receipt with a real measured duration."""
    error = None
    if code is not None:
        error = ErrorInfo(
            schema_version=_SCHEMA_VERSION,
            code=code,
            message=outcome_message(code),
        )
    elapsed = (time.monotonic() - started) * 1000.0
    return ActualOutcome(
        schema_version=_SCHEMA_VERSION,
        outcome_id=f"web-outcome-{uuid.uuid4().hex}",
        status=status,
        error=error,
        duration_ms=max(0.0, round(elapsed, 3)),
    )


def budget_timeout_outcome(started: float) -> ActualOutcome:
    """Return the conservative execution fact of an expired whole-operation budget."""
    return _outcome("unknown", "WEB_ACTION_TIMEOUT", started)


async def _call(call, types: BackendTypes) -> tuple[str, object]:
    """Await one native backend call and classify a failure by its actual boundary.

    Only this call - the construction and the await of one native Playwright operation - is
    converted. A native timeout keeps its own ``timeout`` signal; every other ordinary exception
    becomes the ``error`` signal. ``asyncio.CancelledError``, ``KeyboardInterrupt`` and
    ``SystemExit`` are ``BaseException`` and are never swallowed here.
    """
    try:
        result = call()
        if inspect.isawaitable(result):
            result = await result
        return "ok", result
    except types.timeout:
        return "timeout", None
    except types.error:
        return "error", None
    except Exception:  # noqa: BLE001 - converted to a safe, fixed diagnostic by the caller
        return "error", None


def _session_failure(
    session: BrowserSession,
    status: str,
    started: float,
    *,
    entered: bool,
) -> ActualOutcome:
    """Map one failed native call, keeping pre- and post-interaction outcomes distinct.

    Before the interaction is entered an ordinary backend failure means the facility could not
    be used, so it is a bounded ``WEB_BROWSER_UNAVAILABLE`` error with no raw backend text.
    Once the interaction has been entered, the same failure cannot confirm what happened: the
    outcome is conservatively ``unknown`` with its fixed code, never a ``failed``/not-applied
    outcome and never a facility error.
    """
    if not entered and status == "error":
        raise WebBrowserError("WEB_BROWSER_UNAVAILABLE", "execute")
    code = "WEB_ACTION_TIMEOUT" if status == "timeout" else "WEB_ACTION_INTERRUPTED"
    return _outcome("unknown", code, started)


async def _prepare(
    locator: object,
    session: BrowserSession,
    types: BackendTypes,
    deadline: float,
    started: float,
    *,
    editable: bool,
) -> ActualOutcome | None:
    """Run the deterministic pre-interaction checks, or return a fixed refusal receipt."""
    status, count = await _call(locator.count, types)  # type: ignore[attr-defined]
    if status != "ok":
        return _session_failure(session, status, started, entered=False)
    if count == 0:
        return _outcome("failed", "WEB_TARGET_NOT_FOUND", started)
    if count is not None and count > 1:
        return _outcome("failed", "WEB_TARGET_AMBIGUOUS", started)
    checks = ["is_visible", "is_enabled"] + (["is_editable"] if editable else [])
    for check in checks:
        try:
            timeout = remaining_ms(deadline)
        except BudgetExpired:
            return _outcome("unknown", "WEB_ACTION_TIMEOUT", started)
        # Resolve the native method outside the conversion boundary: a missing API must stay an
        # internal defect, not be reported as a backend failure.
        method = getattr(locator, check)  # type: ignore[attr-defined]
        status, value = await _call(
            lambda method=method, timeout=timeout: method(timeout=timeout),
            types,
        )
        if status != "ok":
            return _session_failure(session, status, started, entered=False)
        if value is not True:
            return _outcome("failed", "WEB_TARGET_NOT_ACTIONABLE", started)
    # Recheck the ambiguity immediately before the action; an old count is not DOM identity.
    status, count = await _call(locator.count, types)  # type: ignore[attr-defined]
    if status != "ok":
        return _session_failure(session, status, started, entered=False)
    if count == 0:
        return _outcome("failed", "WEB_TARGET_NOT_FOUND", started)
    if count is not None and count > 1:
        return _outcome("failed", "WEB_TARGET_AMBIGUOUS", started)
    return None


async def _dispatch(
    locator: object,
    call,
    session: BrowserSession,
    types: BackendTypes,
    deadline: float,
    started: float,
) -> ActualOutcome:
    """Run one native action call once and return its execution-fact receipt."""
    try:
        timeout = remaining_ms(deadline)
    except BudgetExpired:
        return _outcome("unknown", "WEB_ACTION_TIMEOUT", started)
    status, _ = await _call(lambda: call(timeout=timeout), types)
    if status == "ok":
        return _outcome("succeeded", None, started)
    return _session_failure(session, status, started, entered=True)


async def _navigate(
    session: BrowserSession,
    page: object,
    payload: dict,
    deadline: float,
    types: BackendTypes,
    started: float,
) -> ActualOutcome:
    url = payload["url"]
    if not request_origin_allowed(url, session.origins):
        raise WebBrowserError("WEB_ORIGIN_BLOCKED", "execute")
    session.clear_marks()
    try:
        timeout = remaining_ms(deadline)
    except BudgetExpired:
        return _outcome("unknown", "WEB_ACTION_TIMEOUT", started)
    goto = page.goto  # type: ignore[attr-defined]
    status, _ = await _call(
        lambda: goto(url, wait_until="load", timeout=timeout),
        types,
    )
    if status == "ok":
        try:
            final = page.url  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - an unreadable URL is evidence of nothing
            final = None
        if isinstance(final, str) and not request_origin_allowed(final, session.origins):
            raise WebBrowserError("WEB_ORIGIN_BLOCKED", "execute")
        return _outcome("succeeded", None, started)
    if session.navigation_blocked:
        raise WebBrowserError("WEB_ORIGIN_BLOCKED", "execute")
    # The navigation was entered; a timeout or disconnect cannot confirm its effect.
    return _session_failure(session, status, started, entered=True)


async def _click(
    session: BrowserSession,
    page: object,
    payload: dict,
    deadline: float,
    types: BackendTypes,
    started: float,
) -> ActualOutcome:
    locator = _locator(page, payload["target"])
    prepared = await _prepare(locator, session, types, deadline, started, editable=False)
    if prepared is not None:
        return prepared
    click = locator.click  # type: ignore[attr-defined]
    return await _dispatch(
        locator,
        click,
        session,
        types,
        deadline,
        started,
    )


async def _fill(
    session: BrowserSession,
    page: object,
    payload: dict,
    deadline: float,
    types: BackendTypes,
    started: float,
) -> ActualOutcome:
    locator = _locator(page, payload["target"])
    prepared = await _prepare(locator, session, types, deadline, started, editable=True)
    if prepared is not None:
        return prepared
    text = payload["text"]
    fill = locator.fill  # type: ignore[attr-defined]
    return await _dispatch(
        locator,
        lambda timeout: fill(text, timeout=timeout),
        session,
        types,
        deadline,
        started,
    )


async def _press(
    session: BrowserSession,
    page: object,
    payload: dict,
    deadline: float,
    types: BackendTypes,
    started: float,
) -> ActualOutcome:
    locator = _locator(page, payload["target"])
    prepared = await _prepare(locator, session, types, deadline, started, editable=False)
    if prepared is not None:
        return prepared
    key = payload["key"]
    press = locator.press  # type: ignore[attr-defined]
    return await _dispatch(
        locator,
        lambda timeout: press(key, timeout=timeout),
        session,
        types,
        deadline,
        started,
    )


async def _select(
    session: BrowserSession,
    page: object,
    payload: dict,
    deadline: float,
    types: BackendTypes,
    started: float,
) -> ActualOutcome:
    locator = _locator(page, payload["target"])
    prepared = await _prepare(locator, session, types, deadline, started, editable=False)
    if prepared is not None:
        return prepared
    value = payload["value"]
    select_option = locator.select_option  # type: ignore[attr-defined]
    return await _dispatch(
        locator,
        lambda timeout: select_option(value=value, timeout=timeout),
        session,
        types,
        deadline,
        started,
    )


async def _wait(
    session: BrowserSession,
    page: object,
    payload: dict,
    deadline: float,
    types: BackendTypes,
    started: float,
) -> ActualOutcome:
    state = payload["state"]
    locator = _locator(page, payload["target"])
    status, count = await _call(locator.count, types)  # type: ignore[attr-defined]
    if status != "ok":
        return _session_failure(session, status, started, entered=False)
    if count is not None and count > 1:
        return _outcome("failed", "WEB_TARGET_AMBIGUOUS", started)
    if state in ("detached", "hidden") and count == 0:
        # A successful zero match is the approved meaning of the detached/hidden wait.
        return _outcome("succeeded", None, started)
    try:
        timeout = remaining_ms(deadline)
    except BudgetExpired:
        return _outcome("unknown", "WEB_ACTION_TIMEOUT", started)
    wait_for = locator.wait_for  # type: ignore[attr-defined]
    status, _ = await _call(
        lambda: wait_for(state=state, timeout=timeout),
        types,
    )
    if status == "ok":
        return _outcome("succeeded", None, started)
    return await _settle_wait(session, locator, state, types, started)


async def _settle_wait(
    session: BrowserSession,
    locator: object,
    state: str,
    types: BackendTypes,
    started: float,
) -> ActualOutcome:
    """Classify a completed wait from a fresh match count, never from the backend text."""
    observed, count = await _call(locator.count, types)  # type: ignore[attr-defined]
    if observed != "ok":
        return _session_failure(session, observed, started, entered=True)
    if count is not None and count > 1:
        return _outcome("failed", "WEB_TARGET_AMBIGUOUS", started)
    if count == 0:
        if state in ("detached", "hidden"):
            return _outcome("succeeded", None, started)
        return _outcome("failed", "WEB_TARGET_NOT_FOUND", started)
    # The element still exists, but it did not reach the requested detached/hidden/visible
    # state inside the budget. Nothing was acted on, so this refusal is deterministic.
    return _outcome("failed", "WEB_TARGET_NOT_ACTIONABLE", started)


async def perform(
    session: BrowserSession,
    action: ActionSpec,
    deadline: float,
    started: float,
) -> ActualOutcome:
    """Execute one already-validated action once against the owned page."""
    page = session.page
    if page is None or not session.is_connected():
        raise WebBrowserError("WEB_BROWSER_UNAVAILABLE", "execute")
    types = session.types
    if types is None:
        raise RuntimeError("the browser backend types are not available")
    action_type = action.action_type
    payload = action.payload
    if action_type == "web.navigate":
        return await _navigate(session, page, payload, deadline, types, started)
    if action_type == "web.click":
        return await _click(session, page, payload, deadline, types, started)
    if action_type == "web.fill":
        return await _fill(session, page, payload, deadline, types, started)
    if action_type == "web.press":
        return await _press(session, page, payload, deadline, types, started)
    if action_type == "web.select":
        return await _select(session, page, payload, deadline, types, started)
    if action_type == "web.wait":
        return await _wait(session, page, payload, deadline, types, started)
    raise RuntimeError("the validated action type is outside the closed Web vocabulary")
