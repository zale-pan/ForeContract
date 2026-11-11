"""Explicit Web capability declaration.

``build_web_capabilities`` constructs one detached Core ``AdapterCapabilities`` out of two lists
the caller supplies explicitly. There is no "all open" default and no runnable Web adapter
stand-in in this package: the constructor creates no Adapter, starts no environment, opens no
browser and proves nothing about what is implemented, exercised or verified. Calling it is the
caller's own statement of intent, and the declaration it returns is the caller's responsibility.
"""

from __future__ import annotations

from typing import Final

from forecontract.models.actions import AdapterCapabilities
from forecontract.models.base import SCHEMA_VERSION
from pydantic import ValidationError

from forecontract_web.errors import WebValidationError, _bounded_call, _safe_location
from forecontract_web.models import (
    WEB_ACTION_TYPES,
    WEB_ENVIRONMENT_TYPE,
    WEB_OBSERVATION_TYPES,
)

__all__ = ["build_web_capabilities"]

_CODE: Final[str] = "WEB_CAPABILITIES_INVALID"

#: The nine declared fields of the Core capability record: only these may appear in a path.
_CAPABILITY_FIELDS: Final[frozenset[str]] = frozenset(AdapterCapabilities.model_fields)


_MESSAGE: Final[str] = "the capability declaration does not match the Core capability schema"


def _is_blank(value: str) -> bool:
    """Ask the supplied string whether it is blank, through its own ``strip``.

    That call runs the caller's code, so it happens inside a bounded read: a ``str`` subclass whose
    ``strip`` raises - including one that raises a forged ``WebValidationError`` or
    ``ActionValidationError`` - becomes this entry point's fixed code rather than escaping with the
    caller's own message. The returned answer is then judged by this package, outside the read.
    """
    return _bounded_call(lambda: not value.strip(), _MESSAGE, _CODE)


def _adapter_name(value: object) -> str:
    """Require a non-blank string name without trimming the supplied value."""
    if not isinstance(value, str):
        raise WebValidationError(
            "the adapter name must be a non-blank string",
            code=_CODE,
            path=("adapter_name",),
        )
    if _is_blank(value):
        raise WebValidationError(
            "the adapter name must be a non-blank string",
            code=_CODE,
            path=("adapter_name",),
        )
    return value


def _entry_membership(
    item: str, declared: list[str], allowed: tuple[str, ...]
) -> tuple[bool, bool]:
    """Ask whether an entry is already declared and whether it is in the closed vocabulary.

    Both answers come from comparisons the caller owns: ``item in declared`` and ``item in
    allowed`` run that ``str`` subclass's own ``__eq__``, and a subclass may raise from it. Both
    comparisons therefore run inside one bounded read, so a raising entry becomes this entry
    point's fixed code rather than escaping as a raw ``ValueError``. The two answers are then
    judged here, which is what keeps the genuine rejection messages and paths intact.
    """

    def probe() -> tuple[bool, bool]:
        return item in declared, item in allowed

    return _bounded_call(probe, _MESSAGE, _CODE)


def _declared_types(value: object, field: str, allowed: tuple[str, ...]) -> list[str]:
    """Validate one explicit declaration list and return a fresh copy in the supplied order.

    The declared list must be a real ``list`` - a tuple, a string, a set or any other iterable is
    refused rather than coerced - and every entry must be a non-blank string that belongs to the
    closed vocabulary for that field, with no repetition. The list is copied through a bounded read
    first, so a list subclass whose own iteration raises is refused with this entry point's fixed
    code instead of leaking the caller's exception.
    """
    if not isinstance(value, list):
        raise WebValidationError(
            f"the {field} declaration must be a list", code=_CODE, path=(field,)
        )
    entries = _bounded_call(lambda: list(value), _MESSAGE, _CODE)
    declared: list[str] = []
    for index, item in enumerate(entries):
        if not isinstance(item, str):
            raise WebValidationError(
                f"the {field} declaration must contain non-blank strings",
                code=_CODE,
                path=(field, index),
            )
        if _is_blank(item):
            raise WebValidationError(
                f"the {field} declaration must contain non-blank strings",
                code=_CODE,
                path=(field, index),
            )
        repeated, known = _entry_membership(item, declared, allowed)
        if repeated:
            raise WebValidationError(
                f"the {field} declaration must not repeat an entry",
                code=_CODE,
                path=(field, index),
            )
        if not known:
            raise WebValidationError(
                f"the {field} declaration is outside the closed vocabulary for this round",
                code=_CODE,
                path=(field, index),
            )
        declared.append(item)
    return declared


def build_web_capabilities(
    *,
    adapter_name: str,
    supported_actions: list[str],
    supported_observation_types: list[str],
) -> AdapterCapabilities:
    """Build one explicit Core capability declaration for a Web adapter.

    Both lists are required keyword arguments and are never defaulted: this function cannot
    declare an ability the caller did not name. ``supported_actions`` is restricted to the six
    Web action types and ``supported_observation_types`` to the one Web observation type; the
    condition and outcome lists are fixed empty for this round and the environment type is fixed
    to ``web``. The returned record is a new, detached Core model on every call.
    """
    # Every caller-owned read - the name's own ``strip`` and each declaration list's iteration -
    # happens inside a bounded call, so a forged Web/Action error raised by that code cannot borrow
    # this package's diagnostic authority. The decisions below are this package's own.
    message = _MESSAGE
    name = _adapter_name(adapter_name)
    actions = _declared_types(supported_actions, "supported_actions", WEB_ACTION_TYPES)
    observations = _declared_types(
        supported_observation_types, "supported_observation_types", WEB_OBSERVATION_TYPES
    )
    location: tuple[str | int, ...] = ()
    try:
        capabilities = AdapterCapabilities(
            schema_version=SCHEMA_VERSION,
            adapter_name=name,
            environment_type=WEB_ENVIRONMENT_TYPE,
            supported_actions=actions,
            supported_observation_types=observations,
            supported_condition_types=[],
            supported_outcome_types=[],
        )
    except ValidationError as exc:
        location = _safe_location(exc, _CAPABILITY_FIELDS)
    except Exception:  # noqa: BLE001 - an unexpected dependency failure is converted to a bounded Web error
        location = ()
    else:
        return capabilities
    raise WebValidationError(message, code=_CODE, path=location)
