"""Strict public records, scenarios, events and fixed errors of the local test site.

This module is import-only: it opens no socket, starts no thread and writes no file. Every record
is a frozen :mod:`dataclasses` record, so a successful operation cannot mutate a snapshot another
caller already observed. Budgets are validated strictly: a ``bool`` is not an ``int``, a string
number is never converted, and both budgets must be whole milliseconds in ``1..60000``.

The synthetic fixture values live in :mod:`test_site.pages`, not here: the oracle records never
carry a credential, a request body or a session identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

__all__ = [
    "LocalSiteError",
    "Scenario",
    "SiteConfig",
    "SiteEvent",
    "SiteOracleSnapshot",
    "SiteStopResult",
]

SCHEMA_VERSION: Final[str] = "0.1"

_MIN_BUDGET_MS: Final[int] = 1
_MAX_BUDGET_MS: Final[int] = 60000

#: The closed set of safe diagnostics. No message carries a caller value, a request body or a
#: password, and no error appends a raw chained cause.
_ERROR_MESSAGES: Final[dict[str, str]] = {
    "SITE_CONFIG_INVALID": "the local test site configuration is invalid",
    "SITE_NOT_RUNNING": "the local test site is not running",
    "SITE_ALREADY_STOPPED": "the local test site has already stopped",
    "SITE_BUSY": "the local test site is busy",
    "SITE_CONTROL_INVALID": "the local test site control input is invalid",
    "SITE_START_FAILED": "the local test site could not start",
}


class Scenario(str, Enum):
    """The four fixed server modes of this round."""

    NORMAL = "normal"
    DELAYED_LOGIN = "delayed_login"
    BLOCKED_SUBMIT = "blocked_submit"
    COMMIT_THEN_STALL = "commit_then_stall"


class SiteEvent(str, Enum):
    """The three phase facts a host may wait for in the current generation."""

    LOGIN_RECEIVED = "login_received"
    RESPONSE_BLOCKED = "response_blocked"
    LOGIN_COMMITTED = "login_committed"


def _is_scenario(value: object) -> bool:
    """Classify a scenario without reading a caller-owned ``__class__`` or attribute.

    ``isinstance`` falls back to ``value.__class__`` when the real type is not a subclass, so a
    foreign object whose ``__class__`` is an exploding property (or whose ``__getattribute__``
    raises) would leak an ordinary exception instead of the fixed diagnostic. ``type()`` never
    calls either hook; every :class:`Scenario` member has exactly that type, so the identity
    check is the same strict classification without touching caller code.
    """
    return type(value) is Scenario


def _is_site_config(value: object) -> bool:
    """Whether ``value`` is a :class:`SiteConfig` or a genuine native subclass.

    ``issubclass(type(value), SiteConfig)`` takes the interpreter's real type slot through the
    builtin ``type()`` and then walks the real method resolution order owned by that type. It
    never reads a caller-owned ``__class__``/``__getattribute__``, never reads a metaclass-owned
    ``__mro__`` descriptor, and never falls back to a class ``__eq__``. A foreign class that
    spoofs any of those hooks therefore cannot be authorised, while a genuine
    :class:`SiteConfig` and every real subclass still classify as a configuration.
    """
    try:
        return issubclass(type(value), SiteConfig)
    except Exception:  # noqa: BLE001 - a hostile type relationship is a classification failure
        return False


class LocalSiteError(RuntimeError):
    """A bounded site-control failure carrying one fixed public code.

    The constructor accepts a code, never a message: an unknown code collapses to
    ``SITE_CONTROL_INVALID`` so no caller text can reach the diagnostic. The raised exception is
    always created with ``from None`` at the raising site, so no raw cause or context chain is
    exposed.
    """

    def __init__(self, code: str) -> None:
        if type(code) is not str or code not in _ERROR_MESSAGES:
            code = "SITE_CONTROL_INVALID"
        super().__init__(_ERROR_MESSAGES[code])
        self.code = code


def _error(code: str) -> LocalSiteError:
    """Build one fixed error outside an exception handler."""
    return LocalSiteError(code)


def _strict_milliseconds(name: str, value: object) -> int:
    """Validate one whole-millisecond budget in ``1..60000`` without coercion.

    ``type(value) is int`` already excludes ``bool`` (``type(True) is bool``), and unlike
    ``isinstance(value, bool)`` it never asks a foreign object for its ``__class__``.
    """
    if type(value) is not int:
        raise _error("SITE_CONFIG_INVALID")
    if value < _MIN_BUDGET_MS or value > _MAX_BUDGET_MS:
        raise _error("SITE_CONFIG_INVALID")
    return value


def _non_negative_int(value: object) -> int:
    """Validate an internal counter, refusing a bool and every other type."""
    if type(value) is not int or value < 0:
        raise _error("SITE_CONTROL_INVALID")
    return value


def _strict_bool(value: object) -> bool:
    """Validate an internal flag, refusing truthy substitutes."""
    if type(value) is not bool:
        raise _error("SITE_CONTROL_INVALID")
    return value


@dataclass(frozen=True, kw_only=True)
class SiteConfig:
    """The immutable configuration of one site instance.

    ``scenario`` must be a :class:`Scenario` member; a plain string is not converted. Both
    budgets are strict whole milliseconds in ``1..60000``. The dataclass has no default and no
    extra field, so an unknown keyword is refused at construction time.
    """

    scenario: Scenario
    request_timeout_ms: int
    cleanup_timeout_ms: int

    def __post_init__(self) -> None:
        if not _is_scenario(self.scenario):
            raise _error("SITE_CONFIG_INVALID")
        _strict_milliseconds("request_timeout_ms", self.request_timeout_ms)
        _strict_milliseconds("cleanup_timeout_ms", self.cleanup_timeout_ms)


@dataclass(frozen=True, kw_only=True)
class SiteOracleSnapshot:
    """A consistent, detached host-only view of one generation.

    The record exposes stage facts, never a lock, event, socket, thread, session token or
    password. ``schema_version`` is fixed and cannot be supplied by a caller.
    """

    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    generation: int
    scenario: Scenario
    running: bool
    login_requests: int
    auth_successes: int
    auth_failures: int
    commits: int
    active_requests: int
    response_blocked: bool
    authenticated_sessions: int

    def __post_init__(self) -> None:
        if not _is_scenario(self.scenario):
            raise _error("SITE_CONTROL_INVALID")
        _strict_bool(self.running)
        _non_negative_int(self.generation)
        for name in (
            "login_requests",
            "auth_successes",
            "auth_failures",
            "commits",
            "active_requests",
            "authenticated_sessions",
        ):
            _non_negative_int(getattr(self, name))
        _strict_bool(self.response_blocked)


@dataclass(frozen=True, kw_only=True)
class SiteStopResult:
    """The truthful resource result of one bounded cleanup attempt."""

    complete: bool
    active_requests: int
    open_connections: int
    live_threads: int

    def __post_init__(self) -> None:
        _strict_bool(self.complete)
        _non_negative_int(self.active_requests)
        _non_negative_int(self.open_connections)
        _non_negative_int(self.live_threads)
