"""Structural Adapter protocol and its action-validation exceptions.

The protocol is a typing contract: it declares what an environment adapter must offer.
Core defines the interface and the exception hierarchy without implementing any domain
parsing, browser behaviour or lifecycle guarantee.
"""

from __future__ import annotations

from typing import Protocol

from forecontract.models.actions import ActionSpec, AdapterCapabilities
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome

__all__ = [
    "ActionValidationError",
    "Adapter",
    "InvalidActionPayloadError",
    "UnsupportedActionError",
]


class ActionValidationError(ValueError):
    """Base class for action validation failures raised by domain adapters."""


class UnsupportedActionError(ActionValidationError):
    """The adapter does not support the requested action type.

    Domain adapters reject an unsupported action type *before* validating its payload
    schema, so an unsupported action never raises a payload error.
    """


class InvalidActionPayloadError(ActionValidationError):
    """The action type is supported but its payload is malformed."""


class Adapter(Protocol):
    """The environment adapter interface.

    ``capabilities`` and ``validate_action`` are synchronous and must not perform I/O.
    The lifecycle and environment interactions are asynchronous. This round defines no
    context, retry, deadline or cancellation parameter; Runtime revisits that later.
    """

    @property
    def name(self) -> str:
        """A stable adapter name."""
        ...

    def capabilities(self) -> AdapterCapabilities:
        """Declare what this adapter can observe and do (synchronous, no I/O)."""
        ...

    def validate_action(self, action: ActionSpec) -> None:
        """Reject an unsupported or malformed action before it is executed."""
        ...

    async def start(self) -> None:
        """Prepare the environment."""
        ...

    async def stop(self) -> None:
        """Release the environment."""
        ...

    async def observe(self) -> Observation:
        """Read the current environment observation."""
        ...

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        """Execute an action and report what actually happened."""
        ...
