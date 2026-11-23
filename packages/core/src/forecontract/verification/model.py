"""Offline interpretation and provenance validation for model verification payloads."""

from __future__ import annotations

import copy

from forecontract.models.outcomes import VerificationResult, VerificationStatus
from forecontract.verification.extension_models import validate_model_input


class ModelVerifier:
    """Validate and interpret externally supplied model verification responses offline."""

    def verify(
        self,
        *,
        request: dict[str, object],
        response: dict[str, object],
    ) -> VerificationResult:
        """Validate request/response provenance and map the response to a three-state verdict."""
        validated = validate_model_input(
            request=request,
            response=response,
        )

        if validated.state == "ok":
            assert validated.status is not None
            if validated.status is VerificationStatus.PASS:
                final_status = VerificationStatus.PASS
                reason = "MODEL_PASSED"
            elif validated.status is VerificationStatus.FAIL:
                final_status = VerificationStatus.FAIL
                reason = "MODEL_FAILED"
            else:
                final_status = VerificationStatus.INCONCLUSIVE
                reason = "MODEL_INCONCLUSIVE"
        elif validated.state == "refused":
            final_status = VerificationStatus.INCONCLUSIVE
            reason = "MODEL_REFUSED"
        elif validated.state == "unavailable":
            final_status = VerificationStatus.INCONCLUSIVE
            reason = "MODEL_UNAVAILABLE"
        else:
            final_status = VerificationStatus.INCONCLUSIVE
            reason = "MODEL_TIMEOUT"

        return VerificationResult(
            schema_version="0.1",
            status=final_status,
            verifier="model_offline",
            outcome_results=[],
            reason=reason,
            metadata={
                "scope": "evidence_only",
                "request": copy.deepcopy(request),
                "response": copy.deepcopy(response),
            },
        )
