"""Acceptance tests for the strict Web domain models (W01 and W04).

Every fixture is synthetic and embedded here. The tests import the real package from its source
directory, build the real Pydantic models and assert exact field sets, exact accepted values and
exact rejections. No assertion is satisfied by a set-membership shortcut, by a version of the
implementation's own `__all__`, or by a blind `except Exception`.

W01 covers the six action payloads and the two locators; W04 covers observation evidence and the
whole consistency matrix of a ``web.snapshot`` state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from forecontract_web import (
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

TEST_ID = {"strategy": "test_id", "test_id": "checkout-button"}
ROLE_TARGET = {"strategy": "role", "role": "button", "name": "Checkout"}

#: One canonical, complete payload per action type, in the agreed vocabulary order.
PAYLOADS = [
    ("web.navigate", NavigatePayload, {"url": "http://example.test/checkout"}),
    ("web.click", ClickPayload, {"target": TEST_ID}),
    ("web.fill", FillPayload, {"target": ROLE_TARGET, "text": "buyer@example.test"}),
    ("web.press", PressPayload, {"target": TEST_ID, "key": "Enter"}),
    ("web.select", SelectPayload, {"target": ROLE_TARGET, "value": "express"}),
    ("web.wait", WaitPayload, {"target": TEST_ID, "state": "visible"}),
]

PAYLOAD_MODELS = [
    (NavigatePayload, {"url": "http://example.test/checkout"}),
    (ClickPayload, {"target": TEST_ID}),
    (FillPayload, {"target": ROLE_TARGET, "text": "buyer@example.test"}),
    (PressPayload, {"target": TEST_ID, "key": "Enter"}),
    (SelectPayload, {"target": ROLE_TARGET, "value": "express"}),
    (WaitPayload, {"target": TEST_ID, "state": "visible"}),
]


def rejection(model, payload) -> ValidationError:
    """Build through the ordinary low-level schema and return the rejection it raises."""
    with pytest.raises(ValidationError) as caught:
        model.model_validate(payload)
    return caught.value


def evidence(**overrides) -> WebTargetEvidence:
    """A valid single-match record with the named fields replaced."""
    document = {
        "target": TEST_ID,
        "status": "collected",
        "match_count": 1,
        "visible": True,
        "text": "Checkout",
        "value": None,
        "error_code": None,
    }
    document.update(overrides)
    return WebTargetEvidence.model_validate(document)


def state(**overrides) -> WebObservationState:
    """A valid minimal snapshot state with the named fields replaced."""
    document = {"schema_version": "0.1", "url": "about:blank", "targets": []}
    document.update(overrides)
    return WebObservationState.model_validate(document)


# ---------------------------------------------------------------- W01 vocabulary


def test_action_type_vocabulary_is_exactly_the_six_agreed_types() -> None:
    assert WEB_ACTION_TYPES == (
        "web.navigate",
        "web.click",
        "web.fill",
        "web.press",
        "web.select",
        "web.wait",
    )
    assert len(WEB_ACTION_TYPES) == 6
    assert len(set(WEB_ACTION_TYPES)) == 6


def test_observation_type_vocabulary_is_exactly_the_single_snapshot_type() -> None:
    assert WEB_OBSERVATION_TYPES == ("web.snapshot",)


def test_payload_and_locator_models_declare_exactly_the_agreed_fields() -> None:
    assert set(NavigatePayload.model_fields) == {"url"}
    assert set(ClickPayload.model_fields) == {"target"}
    assert set(FillPayload.model_fields) == {"target", "text"}
    assert set(PressPayload.model_fields) == {"target", "key"}
    assert set(SelectPayload.model_fields) == {"target", "value"}
    assert set(WaitPayload.model_fields) == {"target", "state"}
    assert set(TestIdTarget.model_fields) == {"strategy", "test_id"}
    assert set(RoleTarget.model_fields) == {"strategy", "role", "name"}


def test_evidence_models_declare_exactly_the_agreed_fields() -> None:
    assert set(WebTargetEvidence.model_fields) == {
        "target",
        "status",
        "match_count",
        "visible",
        "text",
        "value",
        "error_code",
    }
    assert set(WebObservationState.model_fields) == {"schema_version", "url", "targets"}


@pytest.mark.parametrize(("action_type", "model", "payload"), PAYLOADS)
def test_each_action_type_selects_one_complete_payload_model(
    action_type: str, model, payload: dict
) -> None:
    validated = model.model_validate(payload)
    assert validated.model_dump(mode="python") == payload
    assert model.model_validate_json(validated.model_dump_json()).model_dump(
        mode="python"
    ) == payload


def test_the_six_action_types_map_one_to_one_onto_the_six_payload_models() -> None:
    assert tuple(action_type for action_type, _, _ in PAYLOADS) == WEB_ACTION_TYPES
    assert len({model for _, model, _ in PAYLOADS}) == 6


@pytest.mark.parametrize(("model", "payload"), PAYLOAD_MODELS)
def test_a_payload_never_repeats_the_action_type(model, payload: dict) -> None:
    assert rejection(model, {**payload, "action_type": "web.click"}).errors()[0]["loc"] == (
        "action_type",
    )


@pytest.mark.parametrize(
    "field",
    [
        "timeout",
        "timeout_ms",
        "sleep",
        "evaluate",
        "script",
        "javascript",
        "shell",
        "python",
        "retry_safe",
        "expression",
    ],
)
@pytest.mark.parametrize(("model", "payload"), PAYLOAD_MODELS)
def test_no_payload_accepts_a_timing_script_shell_or_retry_entry(
    model, payload: dict, field: str
) -> None:
    assert rejection(model, {**payload, field: 1}).errors()[0]["loc"] == (field,)


def test_every_required_payload_field_must_be_present() -> None:
    assert rejection(NavigatePayload, {}).errors()[0]["loc"] == ("url",)
    assert rejection(ClickPayload, {}).errors()[0]["loc"] == ("target",)
    assert rejection(FillPayload, {"target": TEST_ID}).errors()[0]["loc"] == ("text",)
    assert rejection(FillPayload, {"text": "x"}).errors()[0]["loc"] == ("target",)
    assert rejection(PressPayload, {"target": TEST_ID}).errors()[0]["loc"] == ("key",)
    assert rejection(SelectPayload, {"target": TEST_ID}).errors()[0]["loc"] == ("value",)
    assert rejection(WaitPayload, {"target": TEST_ID}).errors()[0]["loc"] == ("state",)


# ---------------------------------------------------------------- W01 navigation


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test",
        "https://example.test",
        "http://example.test/",
        "http://example.test/checkout/step-2?cart=1#pay",
        "https://example.test:443/a",
        "http://127.0.0.1:8080/health",
        "http://[::1]:8080/a",
        "https://sub.domain.example.test/a%2Fb?q=%2f",
    ],
)
def test_navigation_accepts_an_absolute_page_url_and_keeps_it_verbatim(url: str) -> None:
    assert NavigatePayload(url=url).url == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        " ",
        "\t",
        "http://example.test\n/next",
        "http://example.test/a\x00b",
        "http://example.test/a\x7fb",
        "/checkout",
        "checkout",
        "example.test",
        "./relative",
        "//example.test/a",
        "http:example.test",
        "http:/example.test/a",
        "http://",
        "http:///checkout",
        "http://user@example.test",
        "http://user:secret@example.test",
        "http://example.test:",
        "http://example.test:0",
        "http://example.test:65536",
        "http://example.test:8o",
        "http://example.test:1_0",
        "http://example.test:0000000001",
        "https://exam ple.test",
        "http://example.test/a b",
        "http://example.test\\evil",
        "javascript:alert(1)",
        "data:text/html,checkout",
        "file:///synthetic/checkout",
        "ftp://example.test/a",
        "about:blank",
        "HTTP://example.test",
        "HTTPS://example.test",
    ],
)
def test_navigation_rejects_anything_that_is_not_an_absolute_http_page(url: str) -> None:
    assert rejection(NavigatePayload, {"url": url}).errors()[0]["loc"] == ("url",)


def test_navigation_rejects_a_non_string_url() -> None:
    for value in (None, 1, 1.5, True, b"http://example.test", ["http://example.test"]):
        assert rejection(NavigatePayload, {"url": value}).errors()[0]["loc"] == ("url",)


# ---------------------------------------------------------------- W01 locators


def test_test_id_target_accepts_an_exact_non_blank_string() -> None:
    assert TestIdTarget(strategy="test_id", test_id="checkout-button").test_id == (
        "checkout-button"
    )
    assert TestIdTarget(strategy="test_id", test_id=" padded ").test_id == " padded "


@pytest.mark.parametrize("value", ["", " ", "\t\n", "\u00a0", None, 1, True, 1.0, b"id", ["id"]])
def test_test_id_target_rejects_a_blank_or_non_string_value(value) -> None:
    assert rejection(TestIdTarget, {"strategy": "test_id", "test_id": value}).errors()[0][
        "loc"
    ] == ("test_id",)


@pytest.mark.parametrize(
    "role",
    [
        "button",
        "link",
        "textbox",
        "combobox",
        "checkbox",
        "radio",
        "option",
        "listbox",
        "heading",
        "dialog",
        "status",
        "alert",
    ],
)
def test_role_target_accepts_each_of_the_twelve_closed_roles(role: str) -> None:
    assert RoleTarget(strategy="role", role=role, name="Checkout").role == role


@pytest.mark.parametrize(
    "role",
    ["", " ", "Button", "BUTTON", "text", "image", "frame", "paragraph", " button", "button "],
)
def test_role_target_rejects_a_role_outside_the_closed_set(role: str) -> None:
    assert rejection(RoleTarget, {"strategy": "role", "role": role, "name": "X"}).errors()[0][
        "loc"
    ] == ("role",)


@pytest.mark.parametrize("name", ["", " ", "\t", None, 1, True, b"Checkout"])
def test_role_target_rejects_a_blank_or_non_string_name(name) -> None:
    assert rejection(RoleTarget, {"strategy": "role", "role": "button", "name": name}).errors()[
        0
    ]["loc"] == ("name",)


@pytest.mark.parametrize(
    "target",
    [
        {"strategy": "css", "selector": "#checkout"},
        {"strategy": "xpath", "path": "//button"},
        {"strategy": "text", "text": "Checkout"},
        {"strategy": "testid", "test_id": "checkout"},
        {"strategy": "role", "role": "button"},
        {"strategy": "test_id"},
        {"strategy": "role", "role": "button", "name": "Checkout", "test_id": "checkout"},
        {"strategy": "test_id", "test_id": "checkout", "role": "button", "name": "Checkout"},
        {"test_id": "checkout"},
        "checkout-button",
        1,
        None,
    ],
)
def test_locator_rejects_an_unknown_strategy_or_a_mixed_and_missing_field(target) -> None:
    with pytest.raises(ValidationError):
        ClickPayload(target=target)


@pytest.mark.parametrize(
    "extra",
    [
        {"selector": "#checkout"},
        {"css": "#checkout"},
        {"xpath": "//button"},
        {"text": "Checkout"},
        {"nth": 0},
        {"first": True},
        {"index": 1},
        {"frame": "main"},
        {"shadow": True},
        {"strict": False},
        {"wait": 100},
    ],
)
def test_locator_rejects_a_position_container_or_selector_field(extra: dict) -> None:
    with pytest.raises(ValidationError):
        ClickPayload(target={**TEST_ID, **extra})
    with pytest.raises(ValidationError):
        ClickPayload(target={**ROLE_TARGET, **extra})


# ---------------------------------------------------------------- W01 fill/select/press/wait


def test_fill_and_select_accept_an_empty_string() -> None:
    assert FillPayload(target=TEST_ID, text="").text == ""
    assert SelectPayload(target=TEST_ID, value="").value == ""
    assert FillPayload(target=TEST_ID, text=" ").text == " "
    assert SelectPayload(target=TEST_ID, value=" padded ").value == " padded "


@pytest.mark.parametrize("text", [None, 1, True, 1.5, b"text", ["text"]])
def test_fill_rejects_a_non_string_text(text) -> None:
    assert rejection(FillPayload, {"target": TEST_ID, "text": text}).errors()[0]["loc"] == (
        "text",
    )


@pytest.mark.parametrize("value", [None, 1, True, 1.5, b"value", ["value"]])
def test_select_rejects_a_non_string_value(value) -> None:
    assert rejection(SelectPayload, {"target": TEST_ID, "value": value}).errors()[0]["loc"] == (
        "value",
    )


@pytest.mark.parametrize(
    "key",
    [
        "Enter",
        "Tab",
        "Escape",
        "Space",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "Backspace",
        "Delete",
        "Home",
        "End",
    ],
)
def test_press_accepts_each_of_the_twelve_closed_keys(key: str) -> None:
    assert PressPayload(target=TEST_ID, key=key).key == key


@pytest.mark.parametrize(
    "key",
    [
        "",
        " ",
        "enter",
        "ENTER",
        "Control+A",
        "Shift+Tab",
        "Enter+Tab",
        "F5",
        "f5",
        "Enter ",
        " Enter",
        "Return",
        "Ctrl",
        "Meta",
        "Enter\n",
        1,
        None,
    ],
)
def test_press_rejects_a_combination_script_or_unknown_key(key) -> None:
    assert rejection(PressPayload, {"target": TEST_ID, "key": key}).errors()[0]["loc"] == (
        "key",
    )


@pytest.mark.parametrize("wait_state", ["attached", "detached", "visible", "hidden"])
def test_wait_accepts_each_of_the_four_states(wait_state: str) -> None:
    assert WaitPayload(target=TEST_ID, state=wait_state).state == wait_state


@pytest.mark.parametrize(
    "wait_state", ["", " ", "VISIBLE", "Visible", "visible ", "enabled", "loading", "present", 1, None]
)
def test_wait_rejects_an_unknown_state(wait_state) -> None:
    assert rejection(WaitPayload, {"target": TEST_ID, "state": wait_state}).errors()[0][
        "loc"
    ] == ("state",)


# ---------------------------------------------------------------- frozen models


@pytest.mark.parametrize(("model", "payload"), PAYLOAD_MODELS)
def test_payload_models_are_frozen(model, payload: dict) -> None:
    validated = model.model_validate(payload)
    name = next(iter(model.model_fields))
    with pytest.raises(ValidationError):
        setattr(validated, name, getattr(validated, name))
    assert validated.model_dump(mode="python") == payload


def test_locator_and_evidence_models_are_frozen() -> None:
    target = TestIdTarget(strategy="test_id", test_id="checkout")
    with pytest.raises(ValidationError):
        target.test_id = "other"
    record = evidence()
    with pytest.raises(ValidationError):
        record.status = "unavailable"
    snapshot = state()
    with pytest.raises(ValidationError):
        snapshot.url = "http://example.test"


# ---------------------------------------------------------------- W04 evidence


@pytest.mark.parametrize(
    "field",
    [
        "target",
        "status",
        "match_count",
        "visible",
        "text",
        "value",
        "error_code",
    ],
)
def test_every_evidence_field_is_required(field: str) -> None:
    document = evidence().model_dump(mode="python")
    document.pop(field)
    assert rejection(WebTargetEvidence, document).errors()[0]["loc"] == (field,)


def test_collected_zero_matches_is_existence_evidence() -> None:
    record = evidence(match_count=0, visible=None, text=None, value=None)
    assert record.status == "collected"
    assert record.match_count == 0
    assert record.visible is None
    assert record.text is None
    assert record.value is None
    assert record.error_code is None
    assert not hasattr(record, "exists")


def test_collected_single_match_carries_each_attribute_independently() -> None:
    assert evidence(visible=None, text=None, value=None).match_count == 1
    assert evidence(visible=False, text=None, value=None).visible is False
    assert evidence(visible=None, text="", value=None).text == ""
    assert evidence(visible=None, text=None, value="").value == ""
    complete = evidence(visible=True, text="Checkout", value="checkout")
    assert (complete.visible, complete.text, complete.value) == (True, "Checkout", "checkout")


@pytest.mark.parametrize("field", ["visible", "text", "value"])
@pytest.mark.parametrize("count", [0, 2, 7])
def test_collected_zero_or_many_matches_must_not_merge_an_attribute(
    field: str, count: int
) -> None:
    document = {
        "target": TEST_ID,
        "status": "collected",
        "match_count": count,
        "visible": None,
        "text": None,
        "value": None,
        "error_code": None,
    }
    assert WebTargetEvidence.model_validate(document).match_count == count
    document[field] = {"visible": True, "text": "Checkout", "value": "checkout"}[field]
    with pytest.raises(ValidationError):
        WebTargetEvidence.model_validate(document)


def test_collected_record_needs_a_match_count_and_forbids_an_error_code() -> None:
    with pytest.raises(ValidationError):
        evidence(match_count=None, visible=None, text=None, value=None)
    with pytest.raises(ValidationError):
        evidence(error_code="TARGET_QUERY_FAILED")


def test_unavailable_record_requires_the_one_query_failure_code() -> None:
    record = evidence(
        status="unavailable",
        match_count=None,
        visible=None,
        text=None,
        value=None,
        error_code="TARGET_QUERY_FAILED",
    )
    assert record.error_code == "TARGET_QUERY_FAILED"
    assert record.match_count is None
    for code in (None, "OTHER", "target_query_failed", "", 1, True):
        with pytest.raises(ValidationError):
            evidence(
                status="unavailable",
                match_count=None,
                visible=None,
                text=None,
                value=None,
                error_code=code,
            )


def test_unavailable_record_must_not_forge_a_zero_match_count() -> None:
    for count in (0, 1, 2):
        with pytest.raises(ValidationError):
            evidence(
                status="unavailable",
                match_count=count,
                visible=None,
                text=None,
                value=None,
                error_code="TARGET_QUERY_FAILED",
            )


@pytest.mark.parametrize("field", ["visible", "text", "value"])
def test_unavailable_record_must_not_carry_an_attribute(field: str) -> None:
    document = {
        "target": TEST_ID,
        "status": "unavailable",
        "match_count": None,
        "visible": None,
        "text": None,
        "value": None,
        "error_code": "TARGET_QUERY_FAILED",
    }
    assert WebTargetEvidence.model_validate(document).status == "unavailable"
    document[field] = {"visible": True, "text": "", "value": ""}[field]
    with pytest.raises(ValidationError):
        WebTargetEvidence.model_validate(document)


def test_status_is_a_closed_vocabulary() -> None:
    for status in ("", "Collected", "UNKNOWN", "missing", None, 1):
        with pytest.raises(ValidationError):
            evidence(status=status)


@pytest.mark.parametrize("count", [True, False, 1.0, 0.5, "1", "0", -1, -0.0])
def test_match_count_rejects_a_bool_float_or_string(count) -> None:
    with pytest.raises(ValidationError):
        evidence(match_count=count)


@pytest.mark.parametrize("visible", [1, 0, "true", "false", "True", "", [], {}])
def test_visible_rejects_a_non_boolean(visible) -> None:
    with pytest.raises(ValidationError):
        evidence(visible=visible)


@pytest.mark.parametrize("value", [0, 1, True, 1.5, b"x", ["x"], {"x": 1}])
def test_text_and_value_reject_a_non_string(value) -> None:
    with pytest.raises(ValidationError):
        evidence(text=value)
    with pytest.raises(ValidationError):
        evidence(value=value)


# ---------------------------------------------------------------- W04 snapshot state


@pytest.mark.parametrize("field", ["schema_version", "url", "targets"])
def test_every_state_field_is_required(field: str) -> None:
    document = {"schema_version": "0.1", "url": None, "targets": []}
    document.pop(field)
    assert rejection(WebObservationState, document).errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("version", ["0.2", "1.0", "", 1, 0.1, None, True])
def test_state_rejects_an_unknown_schema_version(version) -> None:
    assert rejection(
        WebObservationState, {"schema_version": version, "url": None, "targets": []}
    ).errors()[0]["loc"] == ("schema_version",)


def test_state_url_may_be_explicitly_null() -> None:
    snapshot = state(url=None)
    assert snapshot.url is None
    assert snapshot.url != ""


@pytest.mark.parametrize(
    "url",
    ["about:blank", "http://example.test", "https://example.test/checkout?cart=1#pay"],
)
def test_state_url_accepts_about_blank_and_absolute_pages(url: str) -> None:
    assert state(url=url).url == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        " ",
        "/checkout",
        "checkout",
        "about:blank#x",
        "about:Blank",
        "javascript:alert(1)",
        "data:text/html,checkout",
        "file:///synthetic/checkout",
        "http://",
        "http://user@example.test",
        "http://example.test:70000",
        "http://example.test\\evil",
        "http://exam ple.test",
        1,
        True,
    ],
)
def test_state_url_rejects_anything_but_null_about_blank_or_a_page(url) -> None:
    assert rejection(
        WebObservationState, {"schema_version": "0.1", "url": url, "targets": []}
    ).errors()[0]["loc"] == ("url",)


def test_state_accepts_an_empty_target_list() -> None:
    assert state(targets=[]).targets == []


@pytest.mark.parametrize("targets", [(), ("x",), None, "targets", {"a": 1}, 1])
def test_state_rejects_a_non_list_targets_value(targets) -> None:
    assert rejection(
        WebObservationState, {"schema_version": "0.1", "url": None, "targets": targets}
    ).errors()[0]["loc"] == ("targets",)


def test_state_rejects_a_duplicate_locator() -> None:
    first = evidence()
    second = evidence(visible=False, text=None, value=None)
    with pytest.raises(ValidationError):
        state(targets=[first, second])
    with pytest.raises(ValidationError):
        state(
            targets=[
                {
                    "target": {"strategy": "role", "role": "link", "name": "Checkout"},
                    "status": "collected",
                    "match_count": 0,
                    "visible": None,
                    "text": None,
                    "value": None,
                    "error_code": None,
                },
                {
                    "target": {"name": "Checkout", "role": "link", "strategy": "role"},
                    "status": "collected",
                    "match_count": 0,
                    "visible": None,
                    "text": None,
                    "value": None,
                    "error_code": None,
                },
            ]
        )


def test_state_accepts_distinct_locators() -> None:
    first = evidence(target={"strategy": "test_id", "test_id": "a"})
    second = evidence(target={"strategy": "test_id", "test_id": "b"})
    third = evidence(target={"strategy": "role", "role": "button", "name": "a"})
    assert len(state(targets=[first, second, third]).targets) == 3


def test_state_rejects_an_unknown_state_field() -> None:
    assert rejection(
        WebObservationState,
        {"schema_version": "0.1", "url": None, "targets": [], "verdict": "pass"},
    ).errors()[0]["loc"] == ("verdict",)


def test_state_and_evidence_expose_no_verdict_or_completeness_rule() -> None:
    for model in (WebObservationState, WebTargetEvidence):
        assert not hasattr(model, "verdict")
        assert not hasattr(model, "evaluate")
        assert not hasattr(model, "is_complete")
        assert not hasattr(model, "status_of")
