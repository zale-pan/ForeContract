"""Composite three-state verification over deterministic and model verdicts."""

from __future__ import annotations

import copy
from fractions import Fraction

from forecontract.models.outcomes import VerificationResult, VerificationStatus
from forecontract.verification.extension_models import (
    CompositeItemRecord,
    validate_composite_input,
)


def _compute_weighted_interval(
    records: tuple[CompositeItemRecord, ...],
) -> tuple[Fraction, Fraction]:
    weights = [Fraction(record.weight) for record in records]
    total = sum(weights, Fraction(0))
    pass_total = sum(
        (
            weight
            for weight, record in zip(weights, records, strict=True)
            if record.status is VerificationStatus.PASS
        ),
        Fraction(0),
    )
    upper_total = sum(
        (
            weight
            for weight, record in zip(weights, records, strict=True)
            if record.status in (VerificationStatus.PASS, VerificationStatus.INCONCLUSIVE)
        ),
        Fraction(0),
    )
    return pass_total / total, upper_total / total


class CompositeVerifier:
    """Aggregate deterministic and model verification items with critical gates."""

    def verify(
        self,
        *,
        items: list[dict[str, object]],
        mode: str,
        threshold: float | None = None,
    ) -> VerificationResult:
        """Validate all items and compute the composite three-state verdict."""
        validated = validate_composite_input(
            items=items,
            mode=mode,
            threshold=threshold,
        )
        records = validated.items

        lower: float | None = None
        upper: float | None = None
        lower_exact: Fraction | None = None
        upper_exact: Fraction | None = None
        if validated.mode == "weighted":
            lower_exact, upper_exact = _compute_weighted_interval(records)
            lower = min(1.0, max(0.0, float(lower_exact)))
            upper = min(1.0, max(lower, float(upper_exact)))

        critical_statuses = [record.status for record in records if record.critical]
        if any(status is VerificationStatus.FAIL for status in critical_statuses):
            final_status = VerificationStatus.FAIL
            reason = "CRITICAL_FAILED"
        elif any(status is VerificationStatus.INCONCLUSIVE for status in critical_statuses):
            final_status = VerificationStatus.INCONCLUSIVE
            reason = "CRITICAL_INCONCLUSIVE"
        elif validated.mode == "all":
            if any(record.status is VerificationStatus.FAIL for record in records):
                final_status = VerificationStatus.FAIL
                reason = "COMPOSITE_FAILED"
            elif all(record.status is VerificationStatus.PASS for record in records):
                final_status = VerificationStatus.PASS
                reason = "COMPOSITE_PASSED"
            else:
                final_status = VerificationStatus.INCONCLUSIVE
                reason = "COMPOSITE_INCONCLUSIVE"
        elif validated.mode == "any":
            if any(record.status is VerificationStatus.PASS for record in records):
                final_status = VerificationStatus.PASS
                reason = "COMPOSITE_PASSED"
            elif all(record.status is VerificationStatus.FAIL for record in records):
                final_status = VerificationStatus.FAIL
                reason = "COMPOSITE_FAILED"
            else:
                final_status = VerificationStatus.INCONCLUSIVE
                reason = "COMPOSITE_INCONCLUSIVE"
        else:
            assert (
                lower_exact is not None
                and upper_exact is not None
                and validated.threshold is not None
            )
            threshold_exact = Fraction(validated.threshold)
            if lower_exact >= threshold_exact:
                final_status = VerificationStatus.PASS
                reason = "COMPOSITE_PASSED"
            elif upper_exact < threshold_exact:
                final_status = VerificationStatus.FAIL
                reason = "COMPOSITE_FAILED"
            else:
                final_status = VerificationStatus.INCONCLUSIVE
                reason = "COMPOSITE_INCONCLUSIVE"

        return VerificationResult(
            schema_version="0.1",
            status=final_status,
            verifier="composite",
            outcome_results=[],
            reason=reason,
            metadata={
                "scope": "evidence_only",
                "mode": validated.mode,
                "threshold": threshold,
                "items": copy.deepcopy(items),
                "lower": lower,
                "upper": upper,
            },
        )
