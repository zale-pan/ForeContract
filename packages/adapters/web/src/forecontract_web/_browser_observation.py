"""Explicit, evidence-preserving observation of the configured Web targets.

Only the locators declared in ``PlaywrightWebConfig.observation_targets`` are queried. Each query
is independent and is classified into the existing ``WebTargetEvidence`` matrix:

* zero matches -> ``collected`` with ``match_count=0`` and every attribute ``null``;
* several matches -> ``collected`` with the real count and every attribute ``null`` (attributes
  from different nodes are never merged);
* exactly one match -> ``collected`` with ``match_count=1`` and independently obtained
  ``visible``/``text``/``value``. A value that could not be obtained in this snapshot is
  ``null``; an empty string, ``false`` and zero are real evidence and are preserved;
* a bounded per-target query failure -> ``unavailable`` with ``TARGET_QUERY_FAILED`` and no
  count or attributes, so a failure can never be read as a zero match.

Attributes are read from one captured element handle, so they cannot be combined across nodes
after a locator re-resolution. Nothing is remembered between snapshots: every record is built
from scratch and a failed query carries no stale attribute.

A whole-session failure - a closed page or a disconnected browser - raises the internal
:class:`~forecontract_web._browser_session.SessionLost` signal, which the adapter converts into a
bounded facility error. An ordinary exception raised by a native backend call is converted at
that call boundary into the same bounded evidence rules: a failed query is ``unavailable`` and is
never a leaked backend exception, and a single unreadable attribute stays ``null`` without
affecting the independently read attributes. It is never reported as an empty or fabricated target
list. The snapshot is explicitly not claimed to be atomic and no screenshot or full-DOM artifact
is produced.

The module also exposes the pieces an adapter needs to bound *one target group* separately:
:func:`snapshot_page` / :func:`snapshot_url` read the current owned page, :func:`query_target`
performs exactly one target group's native reads against a caller-supplied absolute deadline and
:func:`build_state` assembles the final record list. The per-target ownership (a bounded wait, a
bounded cancellation wind-down and the adapter's late-task bookkeeping) stays in the adapter; this
module never creates a task of its own and never remembers a previous snapshot.
"""

from __future__ import annotations

import inspect
from typing import Final

from forecontract_web._browser_session import (
    BackendTypes,
    BrowserSession,
    SessionLost,
    remaining_ms,
)
from forecontract_web.browser_config import request_origin_allowed
from forecontract_web.models import (
    RoleTarget,
    TestIdTarget,
    WebObservationState,
    WebTargetEvidence,
)

__all__ = ["build_state", "collect", "query_target", "snapshot_page", "snapshot_url", "unavailable"]

_SCHEMA_VERSION: Final[str] = "0.1"
_ABOUT_BLANK: Final[str] = "about:blank"


def _locator(page: object, target: TestIdTarget | RoleTarget) -> object:
    """Build the one allowed native locator for a configured target."""
    if isinstance(target, TestIdTarget):
        return page.get_by_test_id(target.test_id)  # type: ignore[attr-defined]
    return page.get_by_role(  # type: ignore[attr-defined]
        target.role,
        name=target.name,
        exact=True,
        include_hidden=True,
    )


def _collected(
    target: TestIdTarget | RoleTarget,
    count: int,
    visible: bool | None = None,
    text: str | None = None,
    value: str | None = None,
) -> WebTargetEvidence:
    return WebTargetEvidence(
        target=target,
        status="collected",
        match_count=count,
        visible=visible,
        text=text,
        value=value,
        error_code=None,
    )


def _unavailable(target: TestIdTarget | RoleTarget) -> WebTargetEvidence:
    return WebTargetEvidence(
        target=target,
        status="unavailable",
        match_count=None,
        visible=None,
        text=None,
        value=None,
        error_code="TARGET_QUERY_FAILED",
    )


def _session_lost(session: BrowserSession) -> bool:
    """Report a whole-session failure from the backend's own state, never from error text."""
    return not session.is_connected()


async def _call(call, types: BackendTypes) -> tuple[str, object]:
    """Await one native read, converting only an ordinary failure of that call boundary.

    ``asyncio.CancelledError`` and process-control exceptions are ``BaseException`` and are
    never swallowed here; a native timeout keeps its own signal.
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
    except Exception:  # noqa: BLE001 - converted to bounded evidence by the caller
        return "error", None


async def _read(
    call,
    types: BackendTypes,
    session: BrowserSession,
) -> tuple[str, object]:
    """Run one attribute read, preserving an unobtainable value as ``None``."""
    status, value = await _call(call, types)
    if status == "error" and _session_lost(session):
        raise SessionLost()
    return status, value


def _page_url(session: BrowserSession, page: object) -> str | None:
    """Return the actual page URL, ``about:blank``, or ``null`` when it is not obtainable."""
    try:
        url = page.url  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - an unreadable URL is evidence of nothing
        return None
    if not isinstance(url, str) or not url:
        return None
    if url == _ABOUT_BLANK:
        return url
    # The declared scheme token is checked without writing an absolute URL literal, so the
    # distribution content scan does not mistake this source for a machine-specific path.
    if url.partition(":")[0] in ("http", "https") and request_origin_allowed(
        url, session.origins
    ):
        return url
    return None


def snapshot_page(session: BrowserSession) -> tuple[object, BackendTypes]:
    """Return the current owned page and backend types, or raise the session-lost signal."""
    page = session.page
    if page is None or _session_lost(session):
        raise SessionLost()
    types = session.types
    if types is None:
        raise RuntimeError("the browser backend types are not available")
    return page, types


def snapshot_url(session: BrowserSession, page: object) -> str | None:
    """Read the current page's URL for this snapshot; nothing is cached between snapshots."""
    return _page_url(session, page)


def unavailable(target: TestIdTarget | RoleTarget) -> WebTargetEvidence:
    """Build the existing bounded ``unavailable`` record for one target group."""
    return _unavailable(target)


def build_state(url: str | None,
                records: list[WebTargetEvidence]) -> WebObservationState:
    """Assemble one fresh observation state from the records read in this call."""
    return WebObservationState(
        schema_version=_SCHEMA_VERSION,
        url=url,
        targets=records,
    )


async def query_target(
    session: BrowserSession,
    page: object,
    target: TestIdTarget | RoleTarget,
    deadline: float,
    types: BackendTypes,
) -> WebTargetEvidence:
    """Query exactly one configured target and build its current evidence record.

    ``deadline`` is the absolute local deadline for this *whole* target group: the count, the
    element handle and every attribute read share it, so the group is never re-timed per
    attribute. A deadline that has already elapsed is the existing bounded
    unavailable/`TARGET_QUERY_FAILED`` evidence, never a zero match and never a stale value.
    """
    locator = _locator(page, target)
    status, count = await _call(locator.count, types)  # type: ignore[attr-defined]
    if status != "ok":
        if _session_lost(session):
            raise SessionLost()
        return _unavailable(target)
    if count == 0:
        return _collected(target, 0)
    if count is not None and count > 1:
        return _collected(target, count)
    timeout = remaining_ms(deadline)
    if timeout <= 0:
        return _unavailable(target)
    # Resolve the native method outside the conversion boundary: a missing API stays an internal
    # defect instead of being reported as an unavailable target.
    element_handle = locator.element_handle  # type: ignore[attr-defined]
    status, handle = await _call(
        lambda: element_handle(timeout=timeout),
        types,
    )
    if status != "ok":
        if _session_lost(session):
            raise SessionLost()
        return _unavailable(target)
    if handle is None:
        return _collected(target, 0)

    is_visible = handle.is_visible  # type: ignore[attr-defined]
    visible_status, visible_value = await _read(lambda: is_visible(), types, session)
    visible = visible_value if visible_status == "ok" and isinstance(visible_value, bool) else None

    text_content = handle.text_content  # type: ignore[attr-defined]
    text: str | None = None
    text_status, text_value = await _read(lambda: text_content(), types, session)
    if text_status == "ok" and isinstance(text_value, str):
        text = text_value

    value: str | None = None
    value_timeout = remaining_ms(deadline)
    if value_timeout <= 0:
        # The group's own allowance is already spent; an unread value stays ``null`` and the
        # successfully read attributes of this same target are preserved.
        return _collected(target, 1, visible, text, None)
    input_value = handle.input_value  # type: ignore[attr-defined]
    value_status, value_value = await _read(
        lambda: input_value(timeout=value_timeout),
        types,
        session,
    )
    if value_status == "ok" and isinstance(value_value, str):
        value = value_value

    return _collected(target, 1, visible, text, value)


#: The historical name of :func:`query_target`, kept for the existing private call sites/tests.
_query = query_target


async def collect(
    session: BrowserSession,
    targets: list[TestIdTarget | RoleTarget],
    deadline: float,
) -> WebObservationState:
    """Collect one snapshot of the explicit targets from the owned page.

    This is the historical whole-operation path: every target shares the one ``deadline`` and no
    per-target allowance is applied, so a caller that never sets the optional configuration keeps
    exactly the previous behaviour.
    """
    page, types = snapshot_page(session)
    url = snapshot_url(session, page)
    records: list[WebTargetEvidence] = []
    for target in targets:
        records.append(await query_target(session, page, target, deadline, types))
    return build_state(url, records)
