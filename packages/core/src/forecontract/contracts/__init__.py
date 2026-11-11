"""Public contracts surface for the deterministic Contract Builder.

Import the contracts from ``forecontract.contracts``. The subpackage is additive: it does
not change the ``forecontract`` root exports, ``forecontract.models`` or
``forecontract.protocols``, and it adds no dependency beyond the existing Core runtime
dependency.
"""

from __future__ import annotations

from forecontract.contracts.builder import ContractBuilder
from forecontract.contracts.errors import ContractBuildError
from forecontract.contracts.models import (
    ActionContract,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.contracts.validation import ContractValidationHooks

__all__ = [
    "ActionContract",
    "ContractBuildError",
    "ContractBuilder",
    "ContractValidationHooks",
    "RecoveryPolicy",
    "VerificationBinding",
    "VerificationPolicy",
]
