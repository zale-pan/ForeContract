"""Explicit capability declaration that also names the five Web rule types.

``build_web_rule_capabilities`` constructs one detached Core ``AdapterCapabilities`` out of five
lists the caller supplies explicitly: the closed six-action vocabulary, the one Web observation
type, and the five Web condition and outcome types. There is no "all open" default and no
runnable adapter stand-in here: the constructor opens no browser, starts no environment and
proves nothing about what is implemented, exercised or verified.

Every list is required, must be a real ``list``, may hold only entries of its own closed
vocabulary and may be any subset of it - including empty. Duplicates are rejected, the supplied
order is preserved and the caller's list is never mutated. The existing three-argument
``build_web_capabilities`` helper and the existing capability module are unchanged; this helper
is the one the Playwright adapter uses so its declaration names the five rule types without
changing its action or observation vocabulary.
"""

from __future__ import annotations

from typing import Final

from forecontract.models.actions import AdapterCapabilities
from forecontract.models.base import SCHEMA_VERSION
from pydantic import ValidationError

from forecontract_web.capabilities import (
    _CAPABILITY_FIELDS,
    _adapter_name,
    _declared_types,
)
from forecontract_web.errors import WebValidationError, _safe_location
from forecontract_web.models import (
    WEB_ACTION_TYPES,
    WEB_ENVIRONMENT_TYPE,
    WEB_OBSERVATION_TYPES,
)
from forecontract_web.rule_models import WEB_RULE_TYPES

__all__ = ["build_web_rule_capabilities"]

_CODE: Final[str] = "WEB_CAPABILITIES_INVALID"
_MESSAGE: Final[str] = "the capability declaration does not match the Core capability schema"


def build_web_rule_capabilities(
    *,
    adapter_name: str,
    supported_actions: list[str],
    supported_observation_types: list[str],
    supported_condition_types: list[str],
    supported_outcome_types: list[str],
) -> AdapterCapabilities:
    """Build one explicit Core capability declaration including the five Web rule types.

    All five keywords are required and are never defaulted: this function cannot declare an
    ability the caller did not name. ``supported_actions`` is restricted to the six Web action
    types, ``supported_observation_types`` to the one Web observation type, and the condition
    and outcome lists to the five Web rule types. The environment type is fixed to ``web`` and
    the returned record is a new, detached Core model on every call.
    """
    name = _adapter_name(adapter_name)
    actions = _declared_types(supported_actions, "supported_actions", WEB_ACTION_TYPES)
    observations = _declared_types(
        supported_observation_types, "supported_observation_types", WEB_OBSERVATION_TYPES
    )
    conditions = _declared_types(
        supported_condition_types, "supported_condition_types", WEB_RULE_TYPES
    )
    outcomes = _declared_types(
        supported_outcome_types, "supported_outcome_types", WEB_RULE_TYPES
    )
    location: tuple[str | int, ...] = ()
    try:
        capabilities = AdapterCapabilities(
            schema_version=SCHEMA_VERSION,
            adapter_name=name,
            environment_type=WEB_ENVIRONMENT_TYPE,
            supported_actions=actions,
            supported_observation_types=observations,
            supported_condition_types=conditions,
            supported_outcome_types=outcomes,
        )
    except ValidationError as exc:
        location = _safe_location(exc, _CAPABILITY_FIELDS)
    except Exception:  # noqa: BLE001 - an unexpected dependency failure is a bounded Web error
        location = ()
    else:
        return capabilities
    raise WebValidationError(_MESSAGE, code=_CODE, path=location)
