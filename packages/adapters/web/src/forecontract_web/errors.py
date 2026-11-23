"""Bounded facility errors for the Web domain contract.

These are facility failures, not business outcomes. A rejected action never becomes a
``PASS``, a ``FAIL`` or an ``INCONCLUSIVE`` verdict, and nothing here evaluates a task.

The public entry points of this package never let a raw dependency error escape. A Pydantic
``ValidationError`` carries the rejected input value in its own representation, so every
public function converts it into one of these errors and drops the chain: the raised error
has no ``__cause__`` and no ``__context__``, and its message, ``path`` and attributes
transcribe only fixed sentences, declared field names and small list indices. A caller
therefore cannot read back a URL, a filled text, a locator value, an unknown key or a
secret from a diagnostic.

``WebActionPayloadError`` is additionally a Core
:class:`~forecontract.protocols.adapter.InvalidActionPayloadError`, so a caller that already
handles Core action failures keeps working: the supported action type was recognised and its
payload was malformed. An unsupported action type keeps raising the plain Core
``UnsupportedActionError`` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from forecontract.protocols.adapter import InvalidActionPayloadError
from pydantic import ValidationError

__all__ = [
    "WEB_ERROR_CODES",
    "WebActionPayloadError",
    "WebValidationError",
]

#: The closed set of facility failure codes this package can raise.
WEB_ERROR_CODES: Final[tuple[str, ...]] = (
    "WEB_ACTION_INVALID",
    "WEB_PAYLOAD_INVALID",
    "WEB_OBSERVATION_INVALID",
    "WEB_CAPABILITIES_INVALID",
)

#: A declared field name that was not recognised is rendered as this constant placeholder, so
#: an undeclared key supplied by a caller can never be read back from a diagnostic.
_FIELD_PLACEHOLDER: Final[str] = "<field>"

#: A list index is rendered only while it stays small; anything else collapses to a placeholder.
_INDEX_PLACEHOLDER: Final[str] = "<index>"
_MAX_INDEX: Final[int] = 99


def _format_path(path: tuple[str | int, ...]) -> str:
    """Render a structural location as ``field[0].nested``, never a supplied value."""
    text = ""
    for part in path:
        if isinstance(part, bool):
            text += (".<field>" if text else "<field>")
        elif isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text or "<root>"


def _safe_location(exc: ValidationError, known: frozenset[str]) -> tuple[str | int, ...]:
    """Convert a schema rejection into a bounded structural path.

    Only a declared field name and a small list index are copied. An undeclared key - which is
    exactly what an extra field produces - becomes a constant placeholder, and every other
    location part is discarded into one, so the returned path can never carry caller input.
    The rejected value itself is never read.
    """
    errors = exc.errors()
    if not errors:
        return ()
    location = errors[0].get("loc") or ()
    path: list[str | int] = []
    for part in location:
        if isinstance(part, bool):
            path.append(_FIELD_PLACEHOLDER)
        elif isinstance(part, int):
            path.append(part if 0 <= part <= _MAX_INDEX else _INDEX_PLACEHOLDER)
        elif isinstance(part, str) and part in known:
            path.append(part)
        else:
            path.append(_FIELD_PLACEHOLDER)
    return tuple(path)


class WebValidationError(ValueError):
    """A supplied Web action envelope, snapshot or capability argument is not acceptable.

    ``code`` is one of :data:`WEB_ERROR_CODES`, ``path`` locates the rejected structure using
    declared field names and small indices only, and ``message`` is a fixed, bounded sentence
    that carries no supplied value.
    """

    code: str = "WEB_ACTION_INVALID"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        path: tuple[str | int, ...] = (),
    ) -> None:
        self.message = message
        self.path = tuple(path)
        if code is not None:
            if code not in WEB_ERROR_CODES:
                raise ValueError("unknown Web error code")
            self.code = code
        super().__init__(f"{self.code} at {_format_path(self.path)}: {message}")


class WebActionPayloadError(WebValidationError, InvalidActionPayloadError):
    """The action type is supported by the Web contract but its payload is malformed.

    The class is also a Core ``InvalidActionPayloadError``, so the established Core meaning of a
    rejected payload is preserved rather than replaced.
    """

    code = "WEB_PAYLOAD_INVALID"


def _bounded_call(action: Callable[[], object], message: str, code: str) -> object:
    """Run one caller-owned read and convert any failure from it into a bounded Web error.

    Reading a caller-supplied ``Mapping`` runs that caller's own ``__iter__``, ``items`` or
    ``__getitem__``; a non-blank string check runs a supplied ``str`` subclass's ``strip``; reading
    an ``artifacts`` container runs whatever ``__iter__`` that object carries. Those are all
    caller-owned calls, and a raw exception from one of them would carry whatever the caller put in
    its message - a URL, a filled text or a synthetic marker - straight out of a public function
    with no agreed code.

    Origin, not class, decides what happens. Only a caller-owned read runs inside this call, so any
    exception raised here originates in the caller and is converted: the class it happens to have
    proves nothing. A caller can define its own ``WebValidationError`` / ``ActionValidationError``
    subclass (or raise one outright) and thereby forge any public code, path or message, so
    re-raising by type would hand it the package's diagnostic authority. This package's own
    rejections are therefore never raised from inside a bounded read - they are decided afterwards,
    from the value the read returned - which is what keeps internally generated
    ``UnsupportedActionError``, ``InvalidActionPayloadError`` and ``WebActionPayloadError`` intact
    with their priority while a forged copy is still refused.

    Only ``Exception`` is caught: a cancellation, a keyboard interrupt or a process exit is not a
    caller read failure and is never swallowed. The replacement error is raised after the handler
    has finished, so it carries neither the original exception as ``__cause__`` nor as
    ``__context__``.
    """
    try:
        return action()
    except Exception:  # noqa: BLE001, S110 - converted below, outside the handler
        pass
    raise WebValidationError(message, code=code)
