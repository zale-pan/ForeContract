"""Bounded facility errors for the Playwright Web adapter.

These are facility failures, not business outcomes. A blocked request, a missing optional
dependency, an unusable browser session or an incomplete cleanup never becomes a ``PASS``, a
``FAIL`` or an ``INCONCLUSIVE`` verdict, and nothing here evaluates a task.

``WebBrowserError`` is defined only in this module. It carries one of the closed facility codes
below, one fixed operation label and one fixed sentence per code. A raw backend exception text,
a URL, a filled value, an unexpected keyword, an object ``repr`` and the original cause/context
are never copied into a public diagnostic: the constructor accepts no free-form detail at all,
and every public boundary raises the error after its handler has finished, so the raised error
carries neither ``__cause__`` nor ``__context__``.

The two outcome vocabularies beside the facility codes are the fixed codes an
``ActualOutcome`` may carry: a deterministic pre-interaction refusal is ``failed`` with one of
the three target codes, and an execution whose effect cannot be confirmed is ``unknown`` with
one of the two unknown codes.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "WEB_BROWSER_ERROR_CODES",
    "WEB_BROWSER_OPERATIONS",
    "WEB_OUTCOME_FAILED_CODES",
    "WEB_OUTCOME_UNKNOWN_CODES",
    "WebBrowserError",
]

#: The closed set of facility failure codes this adapter can raise.
WEB_BROWSER_ERROR_CODES: Final[tuple[str, ...]] = (
    "WEB_BROWSER_CONFIG_INVALID",
    "WEB_BROWSER_DEPENDENCY_MISSING",
    "WEB_ADAPTER_NOT_RUNNING",
    "WEB_ADAPTER_BUSY",
    "WEB_ADAPTER_STOPPED",
    "WEB_ORIGIN_BLOCKED",
    "WEB_BROWSER_START_FAILED",
    "WEB_BROWSER_UNAVAILABLE",
    "WEB_OBSERVE_FAILED",
    "WEB_OBSERVE_TIMEOUT",
    "WEB_BROWSER_CLEANUP_FAILED",
    "WEB_BROWSER_CLEANUP_TIMEOUT",
)

#: The closed set of fixed operation labels a facility error may carry.
WEB_BROWSER_OPERATIONS: Final[tuple[str, ...]] = (
    "config",
    "start",
    "stop",
    "observe",
    "execute",
    "cleanup",
)

#: The fixed codes of a deterministic, pre-interaction execution refusal.
WEB_OUTCOME_FAILED_CODES: Final[tuple[str, ...]] = (
    "WEB_TARGET_NOT_FOUND",
    "WEB_TARGET_AMBIGUOUS",
    "WEB_TARGET_NOT_ACTIONABLE",
)

#: The fixed codes of an execution whose effect could not be confirmed.
WEB_OUTCOME_UNKNOWN_CODES: Final[tuple[str, ...]] = (
    "WEB_ACTION_TIMEOUT",
    "WEB_ACTION_INTERRUPTED",
)

#: One fixed sentence per facility code. No sentence carries a supplied value.
_FACILITY_MESSAGES: Final[dict[str, str]] = {
    "WEB_BROWSER_CONFIG_INVALID": "the browser configuration is not acceptable",
    "WEB_BROWSER_DEPENDENCY_MISSING": "the optional Playwright dependency is not installed",
    "WEB_ADAPTER_NOT_RUNNING": "this operation requires a running adapter",
    "WEB_ADAPTER_BUSY": "another adapter operation is already in progress",
    "WEB_ADAPTER_STOPPED": "a stopped adapter cannot be started again",
    "WEB_ORIGIN_BLOCKED": "the request origin is not allowed by the local network policy",
    "WEB_BROWSER_START_FAILED": "the headless browser could not be started",
    "WEB_BROWSER_UNAVAILABLE": "the browser session is not available",
    "WEB_OBSERVE_FAILED": "the observation query failed",
    "WEB_OBSERVE_TIMEOUT": "the observation operation budget expired",
    "WEB_BROWSER_CLEANUP_FAILED": "the browser resources could not be released",
    "WEB_BROWSER_CLEANUP_TIMEOUT": "the browser cleanup budget expired",
}

#: One fixed sentence per outcome code. No sentence carries a supplied value.
_OUTCOME_MESSAGES: Final[dict[str, str]] = {
    "WEB_TARGET_NOT_FOUND": "the locator matched no element",
    "WEB_TARGET_AMBIGUOUS": "the locator matched more than one element",
    "WEB_TARGET_NOT_ACTIONABLE": "the located element was not actionable",
    "WEB_ACTION_TIMEOUT": "the action budget expired before the effect could be confirmed",
    "WEB_ACTION_INTERRUPTED": "the action result could not be confirmed",
}


class WebBrowserError(RuntimeError):
    """A bounded browser facility failure.

    ``code`` is one of :data:`WEB_BROWSER_ERROR_CODES`, ``operation`` is one of the fixed
    :data:`WEB_BROWSER_OPERATIONS` labels, and ``message`` is the fixed sentence for the code.
    The rendered text is built only from those three closed values.
    """

    def __init__(self, code: str, operation: str) -> None:
        if code not in WEB_BROWSER_ERROR_CODES:
            raise ValueError("unknown Web browser facility code")
        if operation not in WEB_BROWSER_OPERATIONS:
            raise ValueError("unknown Web browser operation label")
        self.code = code
        self.operation = operation
        self.message = _FACILITY_MESSAGES[code]
        super().__init__(f"{self.code} during {self.operation}: {self.message}")


def outcome_message(code: str) -> str:
    """Return the fixed sentence for one outcome code.

    The value is a constant; a caller cannot pass a supplied string through this helper.
    """
    return _OUTCOME_MESSAGES[code]
