"""Unit tests for CompositeVerifier three-state aggregation and validation."""

from __future__ import annotations

import copy
import itertools

import pytest
from forecontract.models.outcomes import VerificationStatus
from forecontract.verification import CompositeVerifier, VerificationError


def make_item(
    item_id: str = "item-1",
    *,
    status: str = "pass",
    kind: str = "deterministic",
    critical: bool = False,
    weight: float = 1.0,
    source_id: str | None = None,
    reason: str = "rule evaluated",
) -> dict[str, object]:
    return {
        "id": item_id,
        "status": status,
        "kind": kind,
        "critical": critical,
        "weight": weight,
        "source_id": source_id if source_id is not None else f"src-{item_id}",
        "reason": reason,
    }


@pytest.mark.parametrize("mode", ["all", "any", "weighted"])
@pytest.mark.parametrize(
    ("critical_status", "expected_reason"),
    [
        ("fail", "CRITICAL_FAILED"),
        ("inconclusive", "CRITICAL_INCONCLUSIVE"),
    ],
)
def test_critical_deterministic_gate_overrides_all_modes_and_preserves_order(
    mode: str,
    critical_status: str,
    expected_reason: str,
) -> None:
    verifier = CompositeVerifier()
    base_items = [
        make_item("crit", status=critical_status, kind="deterministic", critical=True, weight=0),
        make_item("mod", status="pass", kind="model", critical=False, weight=100),
    ]
    threshold = 0.5 if mode == "weighted" else None
    for ordered in (base_items, list(reversed(base_items))):
        snapshot = copy.deepcopy(ordered)
        result = verifier.verify(items=ordered, mode=mode, threshold=threshold)
        assert result.status == VerificationStatus(critical_status)
        assert result.reason == expected_reason
        assert result.verifier == "composite"
        assert result.outcome_results == []
        assert result.metadata["scope"] == "evidence_only"
        assert result.metadata["mode"] == mode
        assert result.metadata["threshold"] == threshold
        assert result.metadata["items"] == snapshot
        assert ordered == snapshot
        if mode == "weighted":
            assert result.metadata["lower"] == 1.0
            assert result.metadata["upper"] == 1.0
        else:
            assert result.metadata["lower"] is None
            assert result.metadata["upper"] is None


@pytest.mark.parametrize("mode", ["all", "any"])
def test_all_and_any_truth_tables_across_three_items(mode: str) -> None:
    verifier = CompositeVerifier()
    for statuses in itertools.product(("pass", "fail", "inconclusive"), repeat=3):
        items = [make_item(f"i-{idx}", status=st) for idx, st in enumerate(statuses)]
        result = verifier.verify(items=items, mode=mode)
        if mode == "all":
            if "fail" in statuses:
                expected_status, expected_reason = "fail", "COMPOSITE_FAILED"
            elif all(st == "pass" for st in statuses):
                expected_status, expected_reason = "pass", "COMPOSITE_PASSED"
            else:
                expected_status, expected_reason = "inconclusive", "COMPOSITE_INCONCLUSIVE"
        else:
            if "pass" in statuses:
                expected_status, expected_reason = "pass", "COMPOSITE_PASSED"
            elif all(st == "fail" for st in statuses):
                expected_status, expected_reason = "fail", "COMPOSITE_FAILED"
            else:
                expected_status, expected_reason = "inconclusive", "COMPOSITE_INCONCLUSIVE"
        assert result.status.value == expected_status
        assert result.reason == expected_reason
        assert result.metadata["lower"] is None
        assert result.metadata["upper"] is None


@pytest.mark.parametrize(
    ("statuses", "threshold", "expected_status", "expected_reason", "lower", "upper"),
    [
        (("pass", "inconclusive"), 0.75, "inconclusive", "COMPOSITE_INCONCLUSIVE", 0.5, 1.0),
        (("pass", "inconclusive"), 0.5, "pass", "COMPOSITE_PASSED", 0.5, 1.0),
        (("fail", "inconclusive"), 0.75, "fail", "COMPOSITE_FAILED", 0.0, 0.5),
        (("fail", "inconclusive"), 0.5, "inconclusive", "COMPOSITE_INCONCLUSIVE", 0.0, 0.5),
        (("pass", "fail"), 0.5, "pass", "COMPOSITE_PASSED", 0.5, 0.5),
        (("pass", "fail"), 0.51, "fail", "COMPOSITE_FAILED", 0.5, 0.5),
    ],
)
def test_weighted_interval_normalization_and_threshold_boundaries(
    statuses: tuple[str, str],
    threshold: float,
    expected_status: str,
    expected_reason: str,
    lower: float,
    upper: float,
) -> None:
    verifier = CompositeVerifier()
    items = [make_item(f"w-{idx}", status=st, weight=1.0) for idx, st in enumerate(statuses)]
    result = verifier.verify(items=items, mode="weighted", threshold=threshold)
    assert result.status.value == expected_status
    assert result.reason == expected_reason
    assert result.metadata["lower"] == lower
    assert result.metadata["upper"] == upper


def test_weighted_mode_handles_extreme_finite_weights_without_overflow() -> None:
    verifier = CompositeVerifier()
    items = [
        make_item("a", status="pass", weight=1e308),
        make_item("b", status="fail", weight=1e308),
    ]
    result = verifier.verify(items=items, mode="weighted", threshold=0.5)
    assert result.status is VerificationStatus.PASS
    assert result.reason == "COMPOSITE_PASSED"
    assert result.metadata["lower"] == 0.5
    assert result.metadata["upper"] == 0.5


def test_complete_prevalidation_bindings_and_rules_with_non_reflective_errors() -> None:
    verifier = CompositeVerifier()
    secret_marker = "SENSITIVE_TOKEN_VALUE_DO_NOT_REFLECT"

    def assert_err(call: object, expected_code: str) -> None:
        assert callable(call)
        with pytest.raises(VerificationError) as exc_info:
            call()
        err = exc_info.value
        assert err.code == expected_code
        assert len(err.location) >= 1
        assert len(str(err)) <= 240
        assert secret_marker not in str(err)

    assert_err(lambda: verifier.verify(items=[], mode="all"), "VERIFICATION_INPUT_INVALID")
    assert_err(
        lambda: verifier.verify(items=[make_item(f"i-{i}") for i in range(129)], mode="all"),
        "VERIFICATION_INPUT_INVALID",
    )
    assert_err(
        lambda: verifier.verify(
            items=[
                make_item("crit", status="fail", critical=True),
                make_item("bad", status=secret_marker),
            ],
            mode="all",
        ),
        "VERIFICATION_INPUT_INVALID",
    )
    for bad_weight in (True, False, -0.1, float("nan"), float("inf"), float("-inf"), "1"):
        assert_err(
            lambda bw=bad_weight: verifier.verify(
                items=[make_item("a", weight=bw)],  # type: ignore[arg-type]
                mode="all",
            ),
            "VERIFICATION_INPUT_INVALID",
        )
    for key in make_item():
        missing = make_item()
        del missing[key]
        assert_err(
            lambda m=missing: verifier.verify(items=[m], mode="all"),
            "VERIFICATION_INPUT_INVALID",
        )
    for field, bad_val in (
        ("id", ""),
        ("id", "   "),
        ("id", "x" * 129),
        ("kind", secret_marker),
        ("critical", 1),
        ("source_id", ""),
        ("source_id", " "),
        ("source_id", "s" * 129),
        ("reason", ""),
        ("reason", " "),
        ("reason", "r" * 1025),
        ("extra_field", secret_marker),
    ):
        mutated = make_item()
        mutated[field] = bad_val
        assert_err(
            lambda m=mutated: verifier.verify(items=[m], mode="all"),
            "VERIFICATION_INPUT_INVALID",
        )

    assert_err(
        lambda: verifier.verify(items=[make_item("dup"), make_item("dup")], mode="all"),
        "VERIFICATION_BINDING_INVALID",
    )
    assert_err(
        lambda: verifier.verify(
            items=[make_item("m", kind="model", critical=True)],
            mode="all",
        ),
        "VERIFICATION_BINDING_INVALID",
    )

    assert_err(
        lambda: verifier.verify(items=[make_item()], mode=secret_marker),
        "VERIFICATION_RULE_INVALID",
    )
    assert_err(
        lambda: verifier.verify(items=[make_item()], mode="all", threshold=0.5),
        "VERIFICATION_RULE_INVALID",
    )
    assert_err(
        lambda: verifier.verify(items=[make_item()], mode="any", threshold=0.5),
        "VERIFICATION_RULE_INVALID",
    )
    assert_err(
        lambda: verifier.verify(items=[make_item(weight=0)], mode="weighted", threshold=0.5),
        "VERIFICATION_RULE_INVALID",
    )
    for bad_threshold in (True, False, 0, -0.5, 1.01, float("nan"), float("inf"), "0.5", None):
        assert_err(
            lambda bt=bad_threshold: verifier.verify(
                items=[make_item()],
                mode="weighted",
                threshold=bt,  # type: ignore[arg-type]
            ),
            "VERIFICATION_RULE_INVALID",
        )


def test_unknown_item_key_name_is_never_reflected() -> None:
    verifier = CompositeVerifier()
    marker = "UNTRUSTED_FIELD_SENTINEL"
    mutated = make_item()
    mutated[marker] = "ignored"
    with pytest.raises(VerificationError) as exc_info:
        verifier.verify(items=[mutated], mode="all")
    err = exc_info.value
    assert err.code == "VERIFICATION_INPUT_INVALID"
    assert err.location
    assert marker not in str(err.location)
    assert marker not in err.message
    assert marker not in str(err)
    assert len(str(err)) <= 240


@pytest.mark.parametrize("mode", ["all", "any", "weighted"])
def test_finite_large_integer_weights_are_accepted_and_normalized(mode: str) -> None:
    verifier = CompositeVerifier()
    items = [make_item("big-1", status="pass", weight=10**400)]
    threshold = 1 if mode == "weighted" else None
    result = verifier.verify(items=items, mode=mode, threshold=threshold)
    assert result.status is VerificationStatus.PASS
    assert result.metadata["items"] == items
    if mode == "weighted":
        assert result.metadata["lower"] == 1.0
        assert result.metadata["upper"] == 1.0


def test_huge_integer_ratio_and_out_of_range_integer_rejections() -> None:
    verifier = CompositeVerifier()
    pair = [
        make_item("a", status="pass", weight=10**400),
        make_item("b", status="inconclusive", weight=10**400),
    ]
    result = verifier.verify(items=pair, mode="weighted", threshold=0.75)
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.metadata["lower"] == 0.5
    assert result.metadata["upper"] == 1.0
    assert result.metadata["items"] == pair

    with pytest.raises(VerificationError) as neg_weight_err:
        verifier.verify(items=[make_item("neg", weight=-(10**400))], mode="weighted", threshold=1)
    assert neg_weight_err.value.code == "VERIFICATION_INPUT_INVALID"

    for sign in (1, -1):
        with pytest.raises(VerificationError) as thresh_err:
            verifier.verify(items=[make_item()], mode="weighted", threshold=sign * 10**400)
        assert thresh_err.value.code == "VERIFICATION_RULE_INVALID"
        assert thresh_err.value.location and len(str(thresh_err.value)) <= 240
