"""Public adapter protocol contract.

Import ``Adapter`` and the action-validation exceptions from ``forecontract.protocols``.
"""

from __future__ import annotations

from forecontract.protocols.adapter import (
    ActionValidationError,
    Adapter,
    InvalidActionPayloadError,
    UnsupportedActionError,
)

__all__ = [
    "ActionValidationError",
    "Adapter",
    "InvalidActionPayloadError",
    "UnsupportedActionError",
]
