"""Public Web domain-contract surface for ForeContract.

This distribution depends only on ``forecontract-core`` and Pydantic. The root surface is the
strict, pure Web domain contract - six action payload models selected by ``action_type``, two
locator models, the observation evidence models of a ``web.snapshot`` state, an explicit
capability constructor, and four public functions that revalidate rather than trust a supplied
value. The root export set is unchanged at nineteen names.

The accepted browser adapter, the five verification rules, the Web step/task verifiers, the
Runtime contract-validation hooks, the Web task-criteria prevalidator and the Web Runtime domain
support are explicit submodules. They are imported by name when a host needs them; none of them is
added to the root surface, and importing this root package starts no browser, performs no I/O and
evaluates no verdict.
"""

from __future__ import annotations

from forecontract_web.capabilities import build_web_capabilities
from forecontract_web.errors import WebActionPayloadError, WebValidationError
from forecontract_web.models import (
    WEB_ACTION_TYPES,
    WEB_OBSERVATION_TYPES,
    ClickPayload,
    FillPayload,
    NavigatePayload,
    PressPayload,
    RoleTarget,
    SelectPayload,
    TestIdTarget,
    WaitPayload,
    WebObservationState,
    WebTargetEvidence,
)
from forecontract_web.validation import (
    parse_web_action,
    parse_web_observation,
    validate_web_action,
)

__version__ = "0.1.0"

#: The closed public surface: two vocabularies, the two locators, the six payloads, the two
#: evidence models, the two error classes and the four functions. Nothing else is exported, so a
#: caller cannot come to depend on an internal constant as if it were part of the agreed
#: interface.
__all__ = [
    "WEB_ACTION_TYPES",
    "WEB_OBSERVATION_TYPES",
    "ClickPayload",
    "FillPayload",
    "NavigatePayload",
    "PressPayload",
    "RoleTarget",
    "SelectPayload",
    "TestIdTarget",
    "WaitPayload",
    "WebActionPayloadError",
    "WebObservationState",
    "WebTargetEvidence",
    "WebValidationError",
    "__version__",
    "build_web_capabilities",
    "parse_web_action",
    "parse_web_observation",
    "validate_web_action",
]
