"""Acceptance tests for the five pure Web rules (W01-W05 and W08).

The matrix is written out independently of the implementation: every rule is driven through the
public ``WebRuleEvaluator.evaluate`` entry point over a structurally valid Web snapshot, and the
expected status, reason and target position are stated literally. The INCONCLUSIVE shapes -
missing URL, missing target record, an unavailable query, a required attribute stored as
``null`` and several matches - each have their own case, and the explicit zero-match evidence is
kept apart from "no record" and from "query failed".

Facility failures are asserted as errors, never as verdicts: an unknown rule or malformed
payload is ``WEB_RULE_INVALID`` and unusable evidence is ``WEB_VERIFICATION_INPUT_INVALID``.
Every diagnostic is checked for the synthetic marker, a reachable cause/context and input
mutation. Valid committed frozen payloads and ordinary mapping/model positives are covered as
controls.
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract.contracts.snapshot import FrozenMapping
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.verification.models import VerificationEvidence
from forecontract_web.errors import WebValidationError
from forecontract_web.models import RoleTarget, TestIdTarget
from forecontract_web.rule_models import (
    ElementHiddenPayload,
    ElementVisiblePayload,
    TextContainsPayload,
    UrlMatchPayload,
    ValueEqualsPayload,
    WebRuleEvaluation,
)
from forecontract_web.rules import (
    WebRuleEvaluator,
    WebRuleInvalid,
    WebVerificationInputInvalid,
)

NOW = datetime.datetime(2026, 10, 29, 12, 0, tzinfo=datetime.timezone.utc)
MARKER = "SYNTHETIC_WEB_RULE_MARKER_7c1"
PAGE = "http://127.0.0.1:8000/checkout?step=2#summary"


def id_target(test_id: str) -> TestIdTarget:
    return TestIdTarget(strategy="test_id", test_id=test_id)


def role_target(role: str, name: str) -> RoleTarget:
    return RoleTarget(strategy="role", role=role, name=name)


def record(target, *, count, visible=None, text=None, value=None) -> dict:
    """One collected target record with an explicit match count."""
    return {
        "target": target.model_dump(),
        "status": "collected",
        "match_count": count,
        "visible": visible,
        "text": text,
        "value": value,
        "error_code": None,
    }


def unavailable(target) -> dict:
    return {
        "target": target.model_dump(),
        "status": "unavailable",
        "match_count": None,
        "visible": None,
        "text": None,
        "value": None,
        "error_code": "TARGET_QUERY_FAILED",
    }


def snapshot(targets: list[dict], *, url: str | None = PAGE) -> dict:
    return {"schema_version": "0.1", "url": url, "targets": targets}


def evidence(state: object | None = None, *, complete: bool = True,
             identifier: str = "obs-1", environment: str = "web") -> VerificationEvidence:
    return VerificationEvidence(
        schema_version="0.1",
        observation=Observation(
            schema_version="0.1",
            observation_id=identifier,
            environment_type=environment,
            state=state if state is not None else snapshot([]),
            timestamp=NOW,
        ),
        state_complete=complete,
    )


def evaluate(rule_type: str, payload: object, piece: VerificationEvidence) -> WebRuleEvaluation:
    return WebRuleEvaluator().evaluate(rule_type=rule_type, payload=payload, evidence=piece)


def target_payload(target) -> dict:
    return {"target": target.model_dump()}


# ------------------------------------------------------------------ record contract (W05)


def test_web_rule_evaluation_has_exactly_the_six_agreed_fields() -> None:
    assert list(WebRuleEvaluation.model_fields) == [
        "schema_version",
        "rule_type",
        "status",
        "reason",
        "observation_id",
        "target_index",
    ]
    evaluation = WebRuleEvaluation(
        schema_version="0.1", rule_type="web.url_match", status=VerificationStatus.PASS,
        reason="RULE_MATCH", observation_id="obs-1", target_index=None,
    )
    assert evaluation.model_dump() == {
        "schema_version": "0.1",
        "rule_type": "web.url_match",
        "status": VerificationStatus.PASS,
        "reason": "RULE_MATCH",
        "observation_id": "obs-1",
        "target_index": None,
    }


@pytest.mark.parametrize(("status", "reason"), [
    (VerificationStatus.PASS, "RULE_MISMATCH"),
    (VerificationStatus.PASS, "TARGET_NOT_FOUND"),
    (VerificationStatus.PASS, "EVIDENCE_INSUFFICIENT"),
    (VerificationStatus.FAIL, "RULE_MATCH"),
    (VerificationStatus.FAIL, "EVIDENCE_INSUFFICIENT"),
    (VerificationStatus.INCONCLUSIVE, "RULE_MATCH"),
    (VerificationStatus.INCONCLUSIVE, "RULE_MISMATCH"),
    (VerificationStatus.INCONCLUSIVE, "TARGET_NOT_FOUND"),
])
def test_a_self_contradictory_rule_verdict_is_rejected(status, reason: str) -> None:
    with pytest.raises(ValidationError):
        WebRuleEvaluation(
            schema_version="0.1", rule_type="web.url_match", status=status, reason=reason,
            observation_id="obs-1", target_index=None,
        )


@pytest.mark.parametrize("target_index", [-1, True, 1.5, "0"])
def test_a_target_index_is_a_strict_non_negative_integer(target_index) -> None:
    with pytest.raises(ValidationError):
        WebRuleEvaluation(
            schema_version="0.1", rule_type="web.element_visible",
            status=VerificationStatus.PASS, reason="RULE_MATCH",
            observation_id="obs-1", target_index=target_index,
        )


def test_the_record_is_frozen_and_rejects_unknown_fields() -> None:
    evaluation = WebRuleEvaluation(
        schema_version="0.1", rule_type="web.url_match", status=VerificationStatus.PASS,
        reason="RULE_MATCH", observation_id="obs-1", target_index=None,
    )
    with pytest.raises(ValidationError):
        evaluation.reason = "RULE_MISMATCH"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        WebRuleEvaluation(
            schema_version="0.1", rule_type="web.url_match", status=VerificationStatus.PASS,
            reason="RULE_MATCH", observation_id="obs-1", target_index=None, value=MARKER,
        )
    with pytest.raises(ValidationError):
        WebRuleEvaluation(
            schema_version="0.1", rule_type="web.url_match", status=VerificationStatus.PASS,
            reason="RULE_MATCH", observation_id=" ", target_index=None,
        )


# ------------------------------------------------------------------ closed payloads (W02)


def test_the_five_payloads_have_exactly_their_declared_fields() -> None:
    assert list(UrlMatchPayload.model_fields) == ["url"]
    assert list(ElementVisiblePayload.model_fields) == ["target"]
    assert list(ElementHiddenPayload.model_fields) == ["target"]
    assert list(TextContainsPayload.model_fields) == ["target", "text"]
    assert list(ValueEqualsPayload.model_fields) == ["target", "value"]
    for model in (
        UrlMatchPayload, ElementVisiblePayload, ElementHiddenPayload,
        TextContainsPayload, ValueEqualsPayload,
    ):
        assert issubclass(model, BaseModel)
        assert model.model_config["extra"] == "forbid"
        assert model.model_config["frozen"] is True


def test_a_payload_never_repeats_the_rule_type_or_the_schema_version() -> None:
    for model, fields in (
        (UrlMatchPayload, {"url": PAGE}),
        (ElementVisiblePayload, {"target": id_target("a").model_dump()}),
        (TextContainsPayload, {"target": id_target("a").model_dump(), "text": "x"}),
    ):
        with pytest.raises(ValidationError):
            model.model_validate({**fields, "rule_type": "web.url_match"})
        with pytest.raises(ValidationError):
            model.model_validate({**fields, "schema_version": "0.1"})
        with pytest.raises(ValidationError):
            model.model_validate({**fields, "required": True})


@pytest.mark.parametrize("payload", [
    {"url": 1},
    {"url": True},
    {"url": b"bytes"},
    {"url": None},
    {},
    {"url": PAGE, "extra": "x"},
])
def test_a_malformed_url_payload_is_a_fixed_rule_error(payload: dict) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"
    assert type(caught.value) is WebRuleInvalid
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize("payload", [
    {},
    {"target": {"strategy": "css", "selector": "#a"}},
    {"target": {"strategy": "test_id"}},
    {"target": {"strategy": "test_id", "test_id": ""}},
    {"target": {"strategy": "test_id", "test_id": "a", "name": "extra"}},
    {"target": {"strategy": "role", "role": "widget", "name": "a"}},
    {"target": "test_id=a"},
    {"target": id_target("a").model_dump(), "extra": 1},
])
def test_a_malformed_target_payload_is_a_fixed_rule_error(payload: dict) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.element_visible", payload, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"


@pytest.mark.parametrize("payload", [
    {"target": id_target("a").model_dump(), "text": ""},
    {"target": id_target("a").model_dump(), "text": 1},
    {"target": id_target("a").model_dump()},
])
def test_text_contains_requires_a_strict_non_empty_expected_text(payload: dict) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.text_contains", payload, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"


def test_text_contains_accepts_a_whitespace_only_expected_text() -> None:
    piece = evidence(snapshot([
        {"target": id_target("a").model_dump(), "status": "collected", "match_count": 1,
         "visible": True, "text": "a b", "value": None, "error_code": None},
    ]))
    result = evaluate(
        "web.text_contains", {"target": id_target("a").model_dump(), "text": " "}, piece
    )
    assert result.status is VerificationStatus.PASS


@pytest.mark.parametrize("payload", [
    {"target": id_target("a").model_dump(), "value": 1},
    {"target": id_target("a").model_dump()},
    {"target": id_target("a").model_dump(), "value": None},
])
def test_value_equals_requires_a_strict_string(payload: dict) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.value_equals", payload, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"


@pytest.mark.parametrize("rule_type", ["web.url_equals", "url_match", "web.text_contains ",
                                       "", "WEB.URL_MATCH", None, 1])
def test_an_unknown_rule_type_is_a_fixed_rule_error(rule_type) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate(rule_type, {"url": PAGE}, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"
    assert caught.value.path == ("rule_type",)


@pytest.mark.parametrize("payload", [None, 1, "url", [("url", PAGE)]])
def test_a_non_mapping_payload_is_a_fixed_rule_error(payload) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"


# ------------------------------------------------------------------ url_match (W03/W04)


def test_web_url_match_passes_on_the_exact_whole_url() -> None:
    result = evaluate("web.url_match", {"url": PAGE}, evidence())
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.PASS, "RULE_MATCH", None,
    )
    assert result.observation_id == "obs-1"


def test_web_url_match_accepts_about_blank_exactly() -> None:
    result = evaluate("web.url_match", {"url": "about:blank"}, evidence(snapshot([], url="about:blank")))
    assert result.status is VerificationStatus.PASS


@pytest.mark.parametrize("expected", [
    PAGE.removesuffix("#summary"),
    PAGE.split("?")[0],
    PAGE.upper(),
    PAGE + "/",
    PAGE.replace("http:", "https:"),
    "http://127.0.0.1:8000/checkout?step=2&extra=1#summary",
])
def test_web_url_match_does_not_normalise_prefix_or_query(expected: str) -> None:
    if expected == PAGE.upper():
        # F02: an upper-case scheme is outside the existing page-URL grammar the observation
        # state enforces, so this variant is an invalid rule definition (the fixed facility
        # error) rather than a mismatch. Every other no-normalisation variant stays a mismatch.
        with pytest.raises(WebRuleInvalid) as caught:
            evaluate("web.url_match", {"url": expected}, evidence())
        assert caught.value.code == "WEB_RULE_INVALID"
        return
    result = evaluate("web.url_match", {"url": expected}, evidence())
    assert (result.status, result.reason) == (VerificationStatus.FAIL, "RULE_MISMATCH")


def test_web_url_match_is_inconclusive_when_the_url_was_not_obtained() -> None:
    result = evaluate("web.url_match", {"url": PAGE}, evidence(snapshot([], url=None)))
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", None,
    )


# ------------------------------------------------------------------ target rules (W03/W04)


def test_web_element_visible_passes_on_a_unique_visible_target() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=True)]))
    result = evaluate("web.element_visible", target_payload(id_target("a")), piece)
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.PASS, "RULE_MATCH", 0,
    )


def test_web_element_visible_fails_on_a_unique_invisible_target() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=False)]))
    result = evaluate("web.element_visible", target_payload(id_target("a")), piece)
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.FAIL, "RULE_MISMATCH", 0,
    )


def test_web_element_visible_is_inconclusive_when_visibility_is_null() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=None)]))
    result = evaluate("web.element_visible", target_payload(id_target("a")), piece)
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", 0,
    )


@pytest.mark.parametrize(("rule_type", "status", "reason"), [
    ("web.element_visible", VerificationStatus.FAIL, "TARGET_NOT_FOUND"),
    ("web.element_hidden", VerificationStatus.PASS, "RULE_MATCH"),
    ("web.text_contains", VerificationStatus.FAIL, "TARGET_NOT_FOUND"),
    ("web.value_equals", VerificationStatus.FAIL, "TARGET_NOT_FOUND"),
])
def test_a_collected_zero_match_is_explicit_negative_evidence(
    rule_type: str, status, reason: str
) -> None:
    piece = evidence(snapshot([record(id_target("a"), count=0)]))
    payload = target_payload(id_target("a"))
    if rule_type == "web.text_contains":
        payload["text"] = "x"
    if rule_type == "web.value_equals":
        payload["value"] = "x"
    result = evaluate(rule_type, payload, piece)
    assert (result.status, result.reason, result.target_index) == (status, reason, 0)


def test_web_element_hidden_passes_on_a_unique_invisible_target() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=False)]))
    result = evaluate("web.element_hidden", target_payload(id_target("a")), piece)
    assert (result.status, result.reason) == (VerificationStatus.PASS, "RULE_MATCH")


def test_web_element_hidden_fails_on_a_unique_visible_target() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=True)]))
    result = evaluate("web.element_hidden", target_payload(id_target("a")), piece)
    assert (result.status, result.reason) == (VerificationStatus.FAIL, "RULE_MISMATCH")


@pytest.mark.parametrize("rule_type", [
    "web.element_visible", "web.element_hidden", "web.text_contains", "web.value_equals",
])
def test_a_missing_target_record_is_inconclusive_not_zero(rule_type: str) -> None:
    piece = evidence(snapshot([record(id_target("other"), count=1, visible=True, text="x", value="x")]))
    payload = target_payload(id_target("a"))
    if rule_type == "web.text_contains":
        payload["text"] = "x"
    if rule_type == "web.value_equals":
        payload["value"] = "x"
    result = evaluate(rule_type, payload, piece)
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", None,
    )


@pytest.mark.parametrize("rule_type", [
    "web.element_visible", "web.element_hidden", "web.text_contains", "web.value_equals",
])
def test_an_unavailable_query_is_inconclusive(rule_type: str) -> None:
    piece = evidence(snapshot([unavailable(id_target("a"))]))
    payload = target_payload(id_target("a"))
    if rule_type == "web.text_contains":
        payload["text"] = "x"
    if rule_type == "web.value_equals":
        payload["value"] = "x"
    result = evaluate(rule_type, payload, piece)
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", 0,
    )


@pytest.mark.parametrize("rule_type", [
    "web.element_visible", "web.element_hidden", "web.text_contains", "web.value_equals",
])
def test_several_matches_are_ambiguity_evidence(rule_type: str) -> None:
    piece = evidence(snapshot([record(id_target("a"), count=3)]))
    payload = target_payload(id_target("a"))
    if rule_type == "web.text_contains":
        payload["text"] = "x"
    if rule_type == "web.value_equals":
        payload["value"] = "x"
    result = evaluate(rule_type, payload, piece)
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", 0,
    )


def test_target_selection_uses_the_exact_locator_fields_not_the_list_position() -> None:
    first = record(id_target("first"), count=1, visible=True)
    second = record(id_target("second"), count=1, visible=False)
    piece = evidence(snapshot([first, second]))
    result = evaluate("web.element_visible", target_payload(id_target("second")), piece)
    assert (result.status, result.target_index) == (VerificationStatus.FAIL, 1)
    result = evaluate("web.element_visible", target_payload(id_target("first")), piece)
    assert (result.status, result.target_index) == (VerificationStatus.PASS, 0)


def test_a_role_locator_is_matched_by_strategy_role_and_exact_name() -> None:
    piece = evidence(snapshot([
        record(role_target("button", "Checkout"), count=1, visible=True),
    ]))
    pass_result = evaluate(
        "web.element_visible", target_payload(role_target("button", "Checkout")), piece
    )
    assert (pass_result.status, pass_result.target_index) == (VerificationStatus.PASS, 0)
    case_result = evaluate(
        "web.element_visible", target_payload(role_target("button", "checkout")), piece
    )
    assert (case_result.status, case_result.target_index) == (
        VerificationStatus.INCONCLUSIVE, None,
    )
    role_result = evaluate(
        "web.element_visible", target_payload(role_target("link", "Checkout")), piece
    )
    assert (role_result.status, role_result.target_index) == (
        VerificationStatus.INCONCLUSIVE, None,
    )


# ------------------------------------------------------------------ text_contains (W03)


def test_web_text_contains_is_an_original_case_sensitive_substring() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, text="Order Success #42")]))
    payload = {"target": id_target("a").model_dump(), "text": "Success #42"}
    assert evaluate("web.text_contains", payload, piece).status is VerificationStatus.PASS
    payload["text"] = "success #42"
    assert evaluate("web.text_contains", payload, piece).status is VerificationStatus.FAIL
    payload["text"] = "Success"
    assert evaluate("web.text_contains", payload, piece).status is VerificationStatus.PASS


@pytest.mark.parametrize(("actual", "expected"), [
    ("a b", "a  b"),
    (" a", "a "),
    ("café", "cafe\u0301"),
    ("line1\nline2", "line1 line2"),
])
def test_web_text_contains_never_trims_or_rewrites(actual: str, expected: str) -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, text=actual)]))
    payload = {"target": id_target("a").model_dump(), "text": expected}
    result = evaluate("web.text_contains", payload, piece)
    assert (result.status, result.reason) == (VerificationStatus.FAIL, "RULE_MISMATCH")


def test_an_obtained_empty_text_is_known_evidence_not_null() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, text="")]))
    result = evaluate(
        "web.text_contains", {"target": id_target("a").model_dump(), "text": "x"}, piece
    )
    assert (result.status, result.reason) == (VerificationStatus.FAIL, "RULE_MISMATCH")


def test_web_text_contains_is_inconclusive_when_the_text_is_null() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, text=None)]))
    result = evaluate(
        "web.text_contains", {"target": id_target("a").model_dump(), "text": "x"}, piece
    )
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", 0,
    )


def test_visibility_is_not_an_implicit_gate_for_text_or_value() -> None:
    piece = evidence(snapshot([
        record(id_target("a"), count=1, visible=False, text="ready", value="seed"),
    ]))
    text = evaluate(
        "web.text_contains", {"target": id_target("a").model_dump(), "text": "ready"}, piece
    )
    assert text.status is VerificationStatus.PASS
    value = evaluate(
        "web.value_equals", {"target": id_target("a").model_dump(), "value": "seed"}, piece
    )
    assert value.status is VerificationStatus.PASS


# ------------------------------------------------------------------ value_equals (W03)


def test_web_value_equals_compares_the_exact_string() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, value="seed-42")]))
    payload = {"target": id_target("a").model_dump(), "value": "seed-42"}
    assert evaluate("web.value_equals", payload, piece).status is VerificationStatus.PASS
    payload["value"] = "seed-43"
    assert evaluate("web.value_equals", payload, piece).status is VerificationStatus.FAIL


def test_web_value_equals_accepts_an_empty_expected_value() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, value="")]))
    payload = {"target": id_target("a").model_dump(), "value": ""}
    result = evaluate("web.value_equals", payload, piece)
    assert (result.status, result.reason) == (VerificationStatus.PASS, "RULE_MATCH")


def test_web_value_equals_is_inconclusive_when_the_value_is_null() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, value=None)]))
    result = evaluate(
        "web.value_equals", {"target": id_target("a").model_dump(), "value": ""}, piece
    )
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT", 0,
    )


def test_the_text_rule_reads_only_text_and_the_value_rule_reads_only_value() -> None:
    piece = evidence(snapshot([record(id_target("a"), count=1, text="alpha", value="beta")]))
    assert evaluate(
        "web.text_contains", {"target": id_target("a").model_dump(), "text": "beta"}, piece
    ).status is VerificationStatus.FAIL
    assert evaluate(
        "web.value_equals", {"target": id_target("a").model_dump(), "value": "alpha"}, piece
    ).status is VerificationStatus.FAIL


# ------------------------------------------------------------------ evidence integrity (W04/W08)


def test_state_complete_never_changes_a_known_local_verdict() -> None:
    complete = evidence(snapshot([record(id_target("a"), count=0)]), complete=True)
    incomplete = evidence(snapshot([record(id_target("a"), count=0)]), complete=False)
    for piece in (complete, incomplete):
        assert evaluate(
            "web.element_hidden", target_payload(id_target("a")), piece
        ).status is VerificationStatus.PASS
        assert evaluate(
            "web.element_visible", target_payload(id_target("a")), piece
        ).status is VerificationStatus.FAIL


def test_state_complete_true_never_synthesises_an_absent_target_record() -> None:
    piece = evidence(snapshot([]), complete=True)
    result = evaluate("web.element_hidden", target_payload(id_target("a")), piece)
    assert (result.status, result.target_index) == (
        VerificationStatus.INCONCLUSIVE, None,
    )


@pytest.mark.parametrize(("state", "label"), [
    ({"schema_version": "0.1", "targets": []}, "missing url field"),
    ({"schema_version": "0.2", "url": None, "targets": []}, "wrong version"),
    ({"schema_version": "0.1", "url": None, "targets": [], "extra": 1}, "extra field"),
    ({"schema_version": "0.1", "url": "ftp://host/x", "targets": []}, "invalid url"),
    ({"schema_version": "0.1", "url": None, "targets": "none"}, "unstrict targets"),
    ({
        "schema_version": "0.1", "url": None, "targets": [
            record(id_target("a"), count=0),
            record(id_target("a"), count=0),
        ],
    }, "duplicate locator"),
    ({
        "schema_version": "0.1", "url": None, "targets": [
            {"target": id_target("a").model_dump(), "status": "collected",
             "match_count": 0, "visible": True, "text": None, "value": None,
             "error_code": None},
        ],
    }, "inconsistent evidence"),
])
def test_unusable_web_evidence_is_a_fixed_input_error(state: dict, label: str) -> None:
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, evidence(state))
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert MARKER not in str(caught.value)


def test_an_observation_outside_the_web_environment_is_rejected() -> None:
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, evidence(environment="fake"))
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.path == ("evidence", "observation", "environment_type")


@pytest.mark.parametrize("piece", [None, 1, "evidence", {}, []])
def test_a_non_evidence_value_is_a_fixed_input_error(piece) -> None:
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, piece)
    assert caught.value.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert caught.value.path == ("evidence",)


def test_constructed_evidence_is_revalidated_from_its_stored_content() -> None:
    state = snapshot([record(id_target("a"), count=0)])
    observation = Observation.model_construct(
        schema_version="0.1", observation_id="obs-1", environment_type="web",
        state=state, timestamp=NOW,
    )
    piece = VerificationEvidence.model_construct(
        schema_version="0.1", observation=observation, state_complete="yes"
    )
    with pytest.raises(WebVerificationInputInvalid):
        evaluate("web.element_hidden", target_payload(id_target("a")), piece)


def test_a_hidden_extra_state_key_cannot_be_laundered_by_serialization() -> None:
    state = {"schema_version": "0.1", "url": None, "targets": []}
    observation = Observation(
        schema_version="0.1", observation_id="obs-1", environment_type="web",
        state=state, timestamp=NOW,
    )
    object.__setattr__(observation, "undeclared", MARKER)
    piece = VerificationEvidence.model_construct(
        schema_version="0.1", observation=observation, state_complete=True
    )
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, piece)
    assert MARKER not in str(caught.value)
    assert MARKER not in repr(caught.value)


# ------------------------------------------------------------------ hostile ingress and controls


class _CallerMapping(Mapping):
    """A legitimate general (non-dict) mapping read through the ordinary ``items`` protocol.

    The F01 dict-rule reads only this protocol for a non-dict mapping; a dict-backed payload is
    read from its real underlying table instead. These fixtures are therefore general mappings,
    so the caller-owned read failure they inject is still the read the product performs.
    """

    def __init__(self, **values: object) -> None:
        self._values = dict(values)

    def __getitem__(self, key: object) -> object:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class _RaisingMapping(_CallerMapping):
    """A mapping whose own item read raises the synthetic marker."""

    def items(self):
        raise ValueError(MARKER)


class _ForgedMapping(_CallerMapping):
    """A mapping whose own item read raises a forged bounded Web error."""

    def items(self):
        raise WebValidationError(MARKER, code="WEB_PAYLOAD_INVALID", path=(MARKER,))


class _InterruptingMapping(_CallerMapping):
    """A mapping whose own item read raises a process-control signal."""

    def items(self):
        raise KeyboardInterrupt


@pytest.mark.parametrize("payload", [_RaisingMapping(url=PAGE), _ForgedMapping(url=PAGE)])
def test_a_failing_caller_mapping_read_becomes_a_fixed_rule_error(payload: dict) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_process_control_signal_from_a_payload_read_is_not_swallowed() -> None:
    with pytest.raises(KeyboardInterrupt):
        evaluate("web.url_match", _InterruptingMapping(url=PAGE), evidence())


def test_a_committed_frozen_payload_is_accepted() -> None:
    payload = FrozenMapping({"url": PAGE})
    result = evaluate("web.url_match", payload, evidence())
    assert result.status is VerificationStatus.PASS


def test_a_committed_payload_model_and_locator_model_are_accepted() -> None:
    result = evaluate("web.url_match", UrlMatchPayload(url=PAGE), evidence())
    assert result.status is VerificationStatus.PASS
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=True)]))
    payload = ElementVisiblePayload(target=id_target("a"))
    assert evaluate("web.element_visible", payload, piece).status is VerificationStatus.PASS


def test_an_ordinary_mapping_and_str_subclass_still_build() -> None:
    class _PlainStr(str):
        pass

    piece = evidence(snapshot([
        record(id_target(_PlainStr("a")), count=1, visible=True, text=_PlainStr("ready")),
    ]))
    result = evaluate(
        "web.text_contains",
        {"target": {"strategy": "test_id", "test_id": _PlainStr("a")}, "text": "ready"},
        piece,
    )
    assert result.status is VerificationStatus.PASS


def test_the_input_payload_and_evidence_are_not_modified() -> None:
    target = id_target("a")
    piece = evidence(snapshot([record(target, count=1, visible=True, text="ready")]))
    before = (piece.model_dump(mode="json"), piece.observation.model_dump(mode="json"))
    payload = {"target": target.model_dump(), "text": "ready"}
    snapshot_of_payload = {key: dict(value) if isinstance(value, dict) else value
                           for key, value in payload.items()}
    evaluate("web.text_contains", payload, piece)
    assert payload == snapshot_of_payload
    assert piece.model_dump(mode="json") == before[0]
    assert piece.observation.model_dump(mode="json") == before[1]


def test_a_rule_error_transcribes_no_supplied_value() -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", {"url": MARKER, "extra": MARKER}, evidence())
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert set(vars(error)) == {"message", "path", "code"}
    for value in vars(error).values():
        assert MARKER not in repr(value)
    assert MARKER not in str(error) and MARKER not in repr(error.args)


def test_the_evaluator_is_stateless_and_takes_no_constructor_argument() -> None:
    evaluator = WebRuleEvaluator()
    assert vars(evaluator) == {}
    first = evaluator.evaluate(rule_type="web.url_match", payload={"url": PAGE}, evidence=evidence())
    second = WebRuleEvaluator().evaluate(
        rule_type="web.url_match", payload={"url": PAGE}, evidence=evidence()
    )
    assert first == second
    with pytest.raises(TypeError):
        WebRuleEvaluator("callback")  # type: ignore[call-arg]


def test_the_evaluator_exposes_no_browser_or_callback_surface() -> None:
    evaluator = WebRuleEvaluator()
    for absent in ("browser", "page", "evaluate_js", "callback", "plugin", "start", "stop"):
        assert not hasattr(evaluator, absent)


# ------------------------------------------------------------------ stored structure (F01)

#: `stored/false-extra`: a non-empty extra store that claims to be empty.
class _FalseExtra(dict):
    def __bool__(self) -> bool:
        return False


#: `stored/raising-extra`: a non-empty extra store whose truth value raises.
class _RaisingExtra(dict):
    def __bool__(self) -> bool:
        raise ValueError(MARKER)


#: `stored/lying-dict`: a real stored mapping that hides an entry from every mapping view.
class _HidingDict(dict):
    def __iter__(self):
        return iter(["url"])

    def keys(self):
        return ["url"]


def _stored_payload() -> UrlMatchPayload:
    return UrlMatchPayload(url=PAGE)


def test_a_non_empty_extra_store_with_a_false_truth_value_is_rejected() -> None:
    payload = _stored_payload()
    object.__setattr__(payload, "__pydantic_extra__", _FalseExtra(hidden=MARKER))
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_an_extra_store_whose_truth_value_raises_is_a_fixed_rule_error() -> None:
    payload = _stored_payload()
    object.__setattr__(payload, "__pydantic_extra__", _RaisingExtra(hidden=MARKER))
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_a_stored_field_dict_cannot_hide_an_undeclared_key() -> None:
    payload = _stored_payload()
    object.__setattr__(payload, "__dict__", _HidingDict(url=PAGE, hidden=MARKER))
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_a_well_behaved_but_non_empty_extra_store_is_rejected() -> None:
    payload = _stored_payload()
    object.__setattr__(payload, "__pydantic_extra__", {"hidden": MARKER})
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    assert caught.value.code == "WEB_RULE_INVALID"
    assert MARKER not in str(caught.value)


def test_an_absent_or_empty_extra_store_still_evaluates() -> None:
    absent = _stored_payload()
    object.__setattr__(absent, "__pydantic_extra__", None)
    empty = _stored_payload()
    object.__setattr__(empty, "__pydantic_extra__", {})
    for payload in (absent, empty):
        assert evaluate("web.url_match", payload, evidence()).status is VerificationStatus.PASS


def test_an_ordinary_payload_subclass_still_evaluates() -> None:
    class _OrdinaryPayload(UrlMatchPayload):
        pass

    result = evaluate("web.url_match", _OrdinaryPayload(url=PAGE), evidence())
    assert (result.status, result.reason) == (VerificationStatus.PASS, "RULE_MATCH")


class _RewritingEvidence(VerificationEvidence):
    """A valid evidence subclass whose serializer rewrites the actual stored URL."""

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["observation"]["state"]["url"] = PAGE
        return data


def _evidence_subclass(recorded: type[VerificationEvidence], url: str) -> VerificationEvidence:
    return recorded(
        schema_version="0.1",
        observation=Observation(
            schema_version="0.1",
            observation_id="obs-1",
            environment_type="web",
            state=snapshot([], url=url),
            timestamp=NOW,
        ),
        state_complete=True,
    )


def test_an_overridden_evidence_dump_cannot_rewrite_the_verdict() -> None:
    actual = PAGE + "/actually-different"
    piece = _evidence_subclass(_RewritingEvidence, actual)
    result = evaluate("web.url_match", {"url": PAGE}, piece)
    assert (result.status, result.reason) == (VerificationStatus.FAIL, "RULE_MISMATCH")
    assert result.observation_id == "obs-1"


def test_an_ordinary_evidence_subclass_still_evaluates() -> None:
    piece = _evidence_subclass(VerificationEvidence, PAGE)
    assert evaluate("web.url_match", {"url": PAGE}, piece).status is VerificationStatus.PASS


def test_a_non_empty_evidence_extra_store_is_a_fixed_input_error() -> None:
    piece = evidence()
    object.__setattr__(piece, "__pydantic_extra__", _FalseExtra(hidden=MARKER))
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, piece)
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert MARKER not in str(error)
    assert error.__cause__ is None and error.__context__ is None


# ------------------------------------------------------------------ expected URL grammar (F02)


@pytest.mark.parametrize("expected", [
    pytest.param("", id="empty"),
    pytest.param("not a url", id="relative"),
    pytest.param("javascript:alert(1)", id="javascript"),
    pytest.param("https://user:pass@example.com/", id="userinfo"),
    pytest.param("http://example.com:bad", id="bad-port"),
    pytest.param("ftp://example.com/", id="other-scheme"),
    pytest.param(" http://example.com/", id="leading-space"),
    pytest.param("http://example.com/a\\b", id="backslash"),
    pytest.param("HTTP://127.0.0.1:8000/CHECKOUT", id="upper-scheme"),
])
def test_an_expected_url_outside_the_page_url_grammar_is_a_fixed_rule_error(expected) -> None:
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", {"url": expected}, evidence())
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


@pytest.mark.parametrize("expected", [
    PAGE.replace("/checkout", "/CHECKOUT"),
    PAGE.replace("step=2", "step=3"),
    PAGE.replace("#summary", "#Summary"),
])
def test_a_case_query_or_fragment_difference_is_a_mismatch_not_a_normalised_match(
    expected: str,
) -> None:
    result = evaluate("web.url_match", {"url": expected}, evidence())
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.FAIL, "RULE_MISMATCH", None,
    )


def test_a_valid_expected_url_is_preserved_verbatim() -> None:
    payload = UrlMatchPayload(url=PAGE)
    assert payload.url == PAGE
    assert evaluate("web.url_match", payload, evidence()).status is VerificationStatus.PASS
    assert UrlMatchPayload(url="about:blank").url == "about:blank"


# ------------------------------------------------------------------ result target index (F03)


@pytest.mark.parametrize(("status", "reason"), [
    (VerificationStatus.PASS, "RULE_MATCH"),
    (VerificationStatus.FAIL, "RULE_MISMATCH"),
    (VerificationStatus.FAIL, "TARGET_NOT_FOUND"),
    (VerificationStatus.INCONCLUSIVE, "EVIDENCE_INSUFFICIENT"),
])
def test_a_url_rule_result_rejects_a_non_null_target_index(status, reason: str) -> None:
    with pytest.raises(ValidationError):
        WebRuleEvaluation(
            schema_version="0.1", rule_type="web.url_match", status=status, reason=reason,
            observation_id="obs-1", target_index=0,
        )


def test_a_url_rule_result_with_a_null_target_index_is_valid() -> None:
    evaluation = WebRuleEvaluation(
        schema_version="0.1", rule_type="web.url_match", status=VerificationStatus.PASS,
        reason="RULE_MATCH", observation_id="obs-1", target_index=None,
    )
    assert evaluation.target_index is None


def test_a_target_rule_result_keeps_a_non_negative_target_index() -> None:
    evaluation = WebRuleEvaluation(
        schema_version="0.1", rule_type="web.element_visible", status=VerificationStatus.PASS,
        reason="RULE_MATCH", observation_id="obs-1", target_index=0,
    )
    assert evaluation.target_index == 0


# ------------------------------------------------------------------ remaining F01 (round-003)


class _OrdinaryDict(dict):
    """A plain dict subclass with no overridden mapping view (audit control)."""


class _HiddenPayloadDict(dict):
    """A dict-backed payload that really stores an unknown entry but hides it from ``items``."""

    def items(self):
        return [("url", PAGE)]


class _HiddenTargetDict(dict):
    """A dict-backed locator that really stores an unknown entry but hides it from ``items``."""

    def items(self):
        return [("strategy", "test_id"), ("test_id", "a")]


def test_a_real_dict_payload_cannot_hide_an_unknown_stored_entry() -> None:
    """F01 payload/hidden-items-real-unknown: the actual dict table is inspected."""
    payload = _HiddenPayloadDict(url=PAGE, hidden=MARKER)
    # The fixture really stores two entries; only the overridden view reports one.
    assert len(dict.items(payload)) == 2
    assert list(payload.items()) == [("url", PAGE)]
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", payload, evidence())
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert error.path == ("<field>",)
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_a_real_dict_locator_cannot_hide_an_unknown_stored_entry() -> None:
    """F01 target/hidden-items-real-unknown: the actual locator table is inspected."""
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=True)]))
    locator = _HiddenTargetDict(strategy="test_id", test_id="a", hidden=MARKER)
    assert len(dict.items(locator)) == 3
    assert list(locator.items()) == [("strategy", "test_id"), ("test_id", "a")]
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.element_visible", {"target": locator}, piece)
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert error.path[-1:] == ("<field>",)
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_an_ordinary_dict_payload_subclass_still_evaluates() -> None:
    """F01 payload/plain-dict-subclass control."""
    result = evaluate("web.url_match", _OrdinaryDict(url=PAGE), evidence())
    assert (result.status, result.reason) == (VerificationStatus.PASS, "RULE_MATCH")


def test_a_plain_unknown_payload_field_is_still_a_fixed_rule_error() -> None:
    """F01 payload/plain-extra-control: an ordinary dict with an extra key is refused."""
    with pytest.raises(WebRuleInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE, "hidden": MARKER}, evidence())
    error = caught.value
    assert error.code == "WEB_RULE_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)


def test_an_ordinary_dict_locator_subclass_still_evaluates() -> None:
    """F01 target/plain-dict-subclass control."""
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=True)]))
    result = evaluate(
        "web.element_visible",
        {"target": _OrdinaryDict(strategy="test_id", test_id="a")},
        piece,
    )
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.PASS, "RULE_MATCH", 0,
    )


class _SecondHashKey(str):
    """A stored ``str`` key whose second hash raises the synthetic marker."""

    hashes = 0

    def __hash__(self):
        type(self).hashes += 1
        if type(self).hashes >= 2:
            raise ValueError(MARKER)
        return str.__hash__(self)


class _PlainKey(str):
    """A ``str`` subclass with no behavioural override (provably hook-free key)."""


class _CountingHashKey(str):
    """A stored ``str`` key that counts its hashes and returns a legal hash."""

    hashes = 0

    def __hash__(self):
        type(self).hashes += 1
        return str.__hash__(self)


class _ExplodingItemsDict(dict):
    """A dict-backed value whose caller ``items`` view must never be consulted."""

    def items(self):
        raise AssertionError("the caller items view must not be consulted")


def test_a_dict_backed_payload_never_calls_the_caller_items_view() -> None:
    """F01: only the real dict table is read, so a raising ``items`` view is irrelevant."""
    result = evaluate("web.url_match", _ExplodingItemsDict(url=PAGE), evidence())
    assert (result.status, result.reason) == (VerificationStatus.PASS, "RULE_MATCH")


def test_a_dict_backed_locator_never_calls_the_caller_items_view() -> None:
    """F01: only the real locator table is read, so a raising ``items`` view is irrelevant."""
    piece = evidence(snapshot([record(id_target("a"), count=1, visible=True)]))
    locator = _ExplodingItemsDict(strategy="test_id", test_id="a")
    result = evaluate("web.element_visible", {"target": locator}, piece)
    assert (result.status, result.reason, result.target_index) == (
        VerificationStatus.PASS, "RULE_MATCH", 0,
    )


def _evidence_with_stored_state(state: dict) -> VerificationEvidence:
    """Return valid evidence whose observation really stores the supplied state mapping."""
    piece = evidence()
    object.__getattribute__(piece.observation, "__dict__")["state"] = state
    return piece


def test_a_caller_key_hash_hook_that_runs_is_a_fixed_input_error() -> None:
    """F01 evidence/nested-mapping-key-hash-error and the faulting-key policy."""
    _SecondHashKey.hashes = 0
    key = _SecondHashKey("schema_version")
    piece = _evidence_with_stored_state({key: "0.1", "url": PAGE, "targets": []})
    # The dict was built with the hook's first (successful) hash; the rebuild hashes again.
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, piece)
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None
    assert _SecondHashKey.hashes >= 2, "the caller hash hook must really run in the product"


def test_a_provably_hook_free_stored_key_reconstructs_exactly() -> None:
    """F01 evidence/ordinary-stored-key-subclass and the safe-reconstruction branch.

    ``_PlainKey`` overrides nothing, so the product provably executes no caller-defined hash or
    comparison hook. The exact stored key object survives, the input mapping is not mutated and
    no field is lost.
    """
    assert _PlainKey.__hash__ is str.__hash__
    assert _PlainKey.__eq__ is str.__eq__
    key = _PlainKey("schema_version")
    piece = _evidence_with_stored_state({key: "0.1", "url": PAGE, "targets": []})
    stored_before = object.__getattribute__(piece.observation, "__dict__")["state"]
    result = evaluate("web.url_match", {"url": PAGE}, piece)
    assert (result.status, result.reason, result.observation_id) == (
        VerificationStatus.PASS, "RULE_MATCH", "obs-1",
    )
    stored_after = object.__getattribute__(piece.observation, "__dict__")["state"]
    assert stored_after is stored_before
    assert list(stored_after) == ["schema_version", "url", "targets"]
    assert next(iter(stored_after)) is key


def test_a_caller_key_hook_that_runs_but_returns_is_still_a_fixed_input_error() -> None:
    """Faulting-key policy: a PASS is only for a provably hook-free reconstruction.

    This key defines a legal, non-raising ``__hash__``; the reconstruction must still refuse it
    with the fixed facility error instead of running the hook and returning a verdict.
    """
    assert _CountingHashKey.__hash__ is not str.__hash__
    _CountingHashKey.hashes = 0
    piece = _evidence_with_stored_state(
        {_CountingHashKey("schema_version"): "0.1", "url": PAGE, "targets": []}
    )
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, piece)
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None
    assert _CountingHashKey.hashes >= 1, "the caller key really defines a hook"


# ------------------------------------------------------------------ remaining F01 (round-004)


class _RaisingGetattributeObject:
    """``outer/untrusted-getattribute``: a foreign object whose every attribute read raises."""

    def __getattribute__(self, name: str) -> object:
        raise ValueError(MARKER)


class _RaisingClassPropertyObject:
    """``outer/untrusted-class-property``: a Pydantic-lookalike whose ``__class__`` raises."""

    __pydantic_decorators__ = True

    @property
    def __class__(self):
        raise ValueError(MARKER)


def test_an_outer_getattribute_read_is_a_fixed_input_error() -> None:
    """F01 outer/untrusted-getattribute: the evidence is classified by its real type only."""
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, _RaisingGetattributeObject())
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert error.path == ("evidence",)
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


def test_an_outer_class_property_read_is_a_fixed_input_error() -> None:
    """F01 outer/untrusted-class-property: a foreign ``__class__`` is never called."""
    with pytest.raises(WebVerificationInputInvalid) as caught:
        evaluate("web.url_match", {"url": PAGE}, _RaisingClassPropertyObject())
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert error.path == ("evidence",)
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


#: Independently observed hostile ``__class__`` property calls, per nested record kind.
_CLASS_READS = {"evidence": 0, "observation": 0}


class _HostileEvidenceClass(VerificationEvidence):
    """``stored/evidence-class-property``: a real subclass with a raising ``__class__``."""

    @property
    def __class__(self):
        _CLASS_READS["evidence"] += 1
        raise ValueError(MARKER)


class _HostileObservationClass(Observation):
    """``stored/observation-class-property``: a nested real subclass with a raising ``__class__``."""

    @property
    def __class__(self):
        _CLASS_READS["observation"] += 1
        raise ValueError(MARKER)


def _constructed_evidence(recorded, observation_cls=Observation, *,
                          identifier: str = "obs-1") -> VerificationEvidence:
    """Build evidence through ``model_construct`` so a hostile subclass instance is possible."""
    observation = observation_cls.model_construct(
        schema_version="0.1",
        observation_id=identifier,
        environment_type="web",
        timestamp=NOW,
        state=snapshot([], url=PAGE),
        artifacts=[],
        metadata={},
    )
    return recorded.model_construct(
        schema_version="0.1", observation=observation, state_complete=True
    )


def test_a_stored_evidence_class_property_is_never_read_and_the_evidence_passes() -> None:
    """F01 stored/evidence-class-property: exact stored content and zero hostile property calls."""
    _CLASS_READS["evidence"] = 0
    piece = _constructed_evidence(_HostileEvidenceClass)
    before = object.__getattribute__(piece.observation, "__dict__")["state"]
    result = evaluate("web.url_match", {"url": PAGE}, piece)
    assert (result.status, result.reason, result.observation_id) == (
        VerificationStatus.PASS, "RULE_MATCH", "obs-1",
    )
    assert _CLASS_READS["evidence"] == 0, "the nested policy check must not run __class__"
    after = object.__getattribute__(piece.observation, "__dict__")["state"]
    assert after is before
    assert after == snapshot([], url=PAGE)


def test_a_stored_observation_class_property_is_never_read_and_the_evidence_passes() -> None:
    """F01 stored/observation-class-property: exact stored content and zero hostile property calls."""
    _CLASS_READS["observation"] = 0
    piece = _constructed_evidence(VerificationEvidence, _HostileObservationClass)
    before = object.__getattribute__(piece.observation, "__dict__")["state"]
    result = evaluate("web.url_match", {"url": PAGE}, piece)
    assert (result.status, result.reason, result.observation_id) == (
        VerificationStatus.PASS, "RULE_MATCH", "obs-1",
    )
    assert _CLASS_READS["observation"] == 0, "the nested policy check must not run __class__"
    after = object.__getattribute__(piece.observation, "__dict__")["state"]
    assert after is before
    assert after == snapshot([], url=PAGE)


class _SilentItemsDict(dict):
    """``control/real-dict-doesnotcall-view``: a real dict whose caller view must never run."""

    calls = 0

    def items(self):
        type(self).calls += 1
        raise KeyboardInterrupt


def test_a_real_dict_payload_never_calls_the_caller_items_view() -> None:
    """control/real-dict-doesnotcall-view: the actual dict table is read, the view is not."""
    _SilentItemsDict.calls = 0
    result = evaluate("web.url_match", _SilentItemsDict(url=PAGE), evidence())
    assert result.status is VerificationStatus.PASS
    assert _SilentItemsDict.calls == 0


class _ExplodingKey(str):
    """``key/raises-if-any-hook-is-executed``: a stored key whose hash and equality raise."""

    executed = 0
    armed = False

    def __hash__(self):
        type(self).executed += 1
        if type(self).armed:
            raise ValueError(MARKER)
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        type(self).executed += 1
        if type(self).armed:
            raise ValueError(MARKER)
        return str.__eq__(self, other)


def test_a_key_that_raises_if_any_hook_executes_is_refused_without_running_it() -> None:
    """key/raises-if-any-hook-is-executed: a raising key hook never leaks a raw error.

    The stored key owns a hash/equality hook that raises when armed. Whether the Core stored-shape
    check runs that hook or the reconstruction refuses it first, the caller exception must become
    the fixed facility error with no marker, cause or context.
    """
    _ExplodingKey.executed = 0
    key = _ExplodingKey("schema_version")
    piece = _evidence_with_stored_state({key: "0.1", "url": PAGE, "targets": []})
    assert _ExplodingKey.executed == 1, "the key's own hook built the stored state"
    _ExplodingKey.armed = True
    try:
        with pytest.raises(WebVerificationInputInvalid) as caught:
            evaluate("web.url_match", {"url": PAGE}, piece)
    finally:
        _ExplodingKey.armed = False
    error = caught.value
    assert error.code == "WEB_VERIFICATION_INPUT_INVALID"
    assert MARKER not in str(error) and MARKER not in repr(error)
    assert error.__cause__ is None and error.__context__ is None


#: The caller metaclass hook of the ``type-dispatch`` controls.
_METACLASS_HOOKS = {"armed": False, "calls": 0}


class _CountingMeta(type(Observation)):
    """A real Pydantic metaclass subclass whose special-class read can be counted."""

    def __getattribute__(cls, name: str) -> object:
        if _METACLASS_HOOKS["armed"] and name == "__pydantic_decorators__":
            _METACLASS_HOOKS["calls"] += 1
            raise ValueError(MARKER)
        return super().__getattribute__(name)


class _MetaObservation(Observation, metaclass=_CountingMeta):
    """``type-dispatch/metaclass-read``: a real observation subclass with a custom metaclass."""


def _metaclass_evidence() -> VerificationEvidence:
    observation = _MetaObservation.model_construct(
        schema_version="0.1",
        observation_id="obs-1",
        environment_type="web",
        timestamp=NOW,
        state=snapshot([], url=PAGE),
        artifacts=[],
        metadata={},
    )
    return VerificationEvidence.model_construct(
        schema_version="0.1", observation=observation, state_complete=True
    )


def test_real_type_dispatch_never_reads_a_caller_metaclass_hook() -> None:
    """type-dispatch/metaclass-read: real-type dispatch stays clear of the metaclass hook."""
    _METACLASS_HOOKS.update(armed=True, calls=0)
    try:
        result = evaluate("web.url_match", {"url": PAGE}, _metaclass_evidence())
    finally:
        _METACLASS_HOOKS["armed"] = False
    assert result.status is VerificationStatus.PASS
    assert _METACLASS_HOOKS["calls"] == 0


def test_an_ordinary_custom_metaclass_is_accepted() -> None:
    """control/ordinary-custom-metaclass: a real subclass with a custom metaclass still passes."""
    _METACLASS_HOOKS.update(armed=False, calls=0)
    result = evaluate("web.url_match", {"url": PAGE}, _metaclass_evidence())
    assert result.status is VerificationStatus.PASS
    assert _METACLASS_HOOKS["calls"] == 0


def test_ordinary_evidence_is_a_positive_control() -> None:
    """control/ordinary-evidence: the ordinary Core evidence still evaluates."""
    result = evaluate("web.url_match", {"url": PAGE}, evidence())
    assert (result.status, result.reason, result.observation_id) == (
        VerificationStatus.PASS, "RULE_MATCH", "obs-1",
    )


def test_an_ordinary_evidence_subclass_is_a_positive_control() -> None:
    """control/ordinary-evidence-subclass: a genuine subclass without hooks still evaluates."""
    class _OrdinaryEvidence(VerificationEvidence):
        pass

    piece = _evidence_subclass(_OrdinaryEvidence, PAGE)
    result = evaluate("web.url_match", {"url": PAGE}, piece)
    assert (result.status, result.reason, result.observation_id) == (
        VerificationStatus.PASS, "RULE_MATCH", "obs-1",
    )
