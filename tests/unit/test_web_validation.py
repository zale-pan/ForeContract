"""Acceptance tests for the four public Web entry points (W02, W03 and W06).

The tests import the real package from its source directory and drive only the public surface.
Every accepted value and every rejection is asserted exactly: the exception type, its ``code``,
its bounded ``path`` and its fixed message. No test derives an expectation from the code under
test, and no test hides a failure behind a blind ``except Exception``.
"""

from __future__ import annotations

import copy
import json
import socket
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import forecontract_web
from forecontract.contracts.snapshot import (
    FrozenActionSpec,
    FrozenMapping,
    freeze_action,
    freeze_json,
)
from forecontract.models.actions import ActionSpec
from forecontract.models.observations import ArtifactRef, Observation
from forecontract.protocols.adapter import (
    ActionValidationError,
    InvalidActionPayloadError,
    UnsupportedActionError,
)

NOW = datetime(2026, 10, 22, 9, 30, tzinfo=timezone.utc)
MARKER = "SYNTHETIC_PRIVATE_VALUE_9f3a"
TARGET = {"strategy": "test_id", "test_id": "checkout-button"}
EMPTY_STATE = {"schema_version": "0.1", "url": None, "targets": []}


def action(action_type: str = "web.click", payload=None, metadata=None) -> ActionSpec:
    """A real Core action candidate carrying a default click payload."""
    return ActionSpec(
        schema_version="0.1",
        action_type=action_type,
        payload={"target": TARGET} if payload is None else payload,
        metadata={} if metadata is None else metadata,
    )


def observation_document(*, drop: tuple[str, ...] = (), **overrides) -> dict:
    """A valid outer Core observation document with the named fields replaced or removed."""
    document = {
        "schema_version": "0.1",
        "observation_id": "obs-1",
        "environment_type": "web",
        "state": copy.deepcopy(EMPTY_STATE),
        "timestamp": NOW,
    }
    document.update(overrides)
    for name in drop:
        document.pop(name)
    return document


def observation(state=None, *, environment_type: str = "web", **overrides) -> Observation:
    """A real outer Core observation whose state is the supplied Web snapshot document."""
    return Observation.model_validate(
        observation_document(
            environment_type=environment_type,
            state=copy.deepcopy(EMPTY_STATE) if state is None else state,
            **overrides,
        )
    )


def evidence_document(**overrides) -> dict:
    """A valid single-match evidence document with the named fields replaced."""
    document = {
        "target": TARGET,
        "status": "collected",
        "match_count": 1,
        "visible": True,
        "text": "Checkout",
        "value": None,
        "error_code": None,
    }
    document.update(overrides)
    return document


def exception_chain(error: BaseException) -> list[BaseException]:
    """Every exception reachable from ``error`` along ``__cause__`` and unsuppressed context."""
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__
    return chain


def assert_no_marker(error: BaseException) -> None:
    """Assert that no reachable diagnostic transcribes the synthetic marker."""
    assert error.__cause__ is None
    assert error.__context__ is None
    for link in exception_chain(error):
        assert MARKER not in str(link)
        assert MARKER not in repr(link)
        assert MARKER not in repr(link.args)
        for value in vars(link).values():
            assert MARKER not in repr(value)


# ---------------------------------------------------------------- W03 revalidation


def test_parse_web_action_returns_a_detached_core_action_spec() -> None:
    supplied = action("web.fill", {"target": TARGET, "text": "buyer"}, {"note": ["a"]})
    parsed = forecontract_web.parse_web_action(supplied)
    assert type(parsed) is ActionSpec
    assert parsed is not supplied
    assert parsed.payload is not supplied.payload
    assert parsed.metadata is not supplied.metadata
    assert parsed.model_dump(mode="python") == supplied.model_dump(mode="python")


def test_mapping_and_instance_ingress_keep_the_same_semantics() -> None:
    supplied = action("web.select", {"target": TARGET, "value": "express"})
    from_instance = forecontract_web.parse_web_action(supplied)
    from_mapping = forecontract_web.parse_web_action(supplied.model_dump(mode="python"))
    assert from_mapping.model_dump(mode="python") == from_instance.model_dump(mode="python")
    assert from_mapping.payload is not from_instance.payload


def test_python_and_json_round_trip_keep_the_same_semantics() -> None:
    supplied = action(
        "web.fill",
        {"target": {"strategy": "role", "role": "textbox", "name": "Email"}, "text": ""},
    )
    python_side = forecontract_web.parse_web_action(supplied.model_dump(mode="python"))
    json_side = forecontract_web.parse_web_action(
        ActionSpec.model_validate_json(supplied.model_dump_json())
    )
    assert json_side.model_dump(mode="python") == python_side.model_dump(mode="python")
    assert json.loads(json_side.model_dump_json()) == json.loads(python_side.model_dump_json())


def test_a_builder_frozen_action_spec_is_accepted() -> None:
    plain = action(
        "web.fill",
        {"target": TARGET, "text": "buyer"},
        {"tag": ["a", {"b": [1, None]}]},
    )
    frozen = freeze_action(plain)
    assert isinstance(frozen, FrozenActionSpec)
    parsed = forecontract_web.parse_web_action(frozen)
    assert type(parsed) is ActionSpec
    assert type(parsed.payload) is dict
    assert type(parsed.metadata) is dict
    assert parsed.model_dump(mode="python") == plain.model_dump(mode="python")
    direct = FrozenActionSpec(
        schema_version="0.1", action_type="web.click", payload={"target": TARGET}
    )
    assert forecontract_web.parse_web_action(direct).payload == {"target": TARGET}


def test_only_the_known_core_frozen_containers_are_unwrapped() -> None:
    base = {
        "schema_version": "0.1",
        "action_type": "web.fill",
        "payload": {"target": TARGET, "text": "buyer"},
    }
    for value in ((1, 2), b"bytes", frozenset({1, 2}), object()):
        document = copy.deepcopy(base)
        document["payload"]["text"] = value
        with pytest.raises(forecontract_web.WebValidationError) as caught:
            forecontract_web.parse_web_action(document)
        assert caught.value.code == "WEB_ACTION_INVALID"
        assert caught.value.path == ("payload", "text", "<field>")


def test_a_tuple_is_never_coerced_into_a_locator_document() -> None:
    document = {
        "schema_version": "0.1",
        "action_type": "web.click",
        "payload": {"target": (("strategy", "test_id"), ("test_id", "checkout"))},
    }
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(document)
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("payload", "target", "<field>")


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), b"bytes", object()])
def test_a_non_finite_or_non_json_payload_value_is_rejected(value: object) -> None:
    document = {
        "schema_version": "0.1",
        "action_type": "web.fill",
        "payload": {"target": TARGET, "text": value},
    }
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(document)
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("payload", "text", "<field>")


@pytest.mark.parametrize(
    "field", ["script", "javascript", "evaluate", "shell", "python", "timeout", "retry_safe"]
)
def test_a_code_timing_or_retry_entry_is_rejected_by_the_public_entry_point(field: str) -> None:
    document = {
        "schema_version": "0.1",
        "action_type": "web.click",
        "payload": {"target": TARGET, field: 1},
    }
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(document)
    assert caught.value.code == "WEB_PAYLOAD_INVALID"
    assert caught.value.path == ("<field>",)


def test_model_copy_and_hidden_keys_cannot_bypass_the_entry_points() -> None:
    copied = action().model_copy(update={"undeclared": 1})
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(copied)
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("<field>",)

    injected = action()
    object.__setattr__(injected, "undeclared", MARKER)
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(injected)
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("<field>",)

    constructed = ActionSpec.model_construct(
        schema_version="0.1",
        action_type="web.click",
        payload={"target": {"strategy": "css", "selector": "#checkout"}},
    )
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(constructed)
    assert caught.value.code == "WEB_PAYLOAD_INVALID"


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": "9.9"},
        {"action_type": "  "},
        {"action_type": None},
        {"payload": []},
        {"payload": "not-an-object"},
        {"metadata": "not-an-object"},
    ],
)
def test_model_copy_with_an_invalid_envelope_field_is_rejected(update: dict) -> None:
    supplied = action().model_copy(update=update)
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(supplied)
    assert caught.value.code == "WEB_ACTION_INVALID"


def test_in_place_nested_contamination_is_rejected() -> None:
    supplied = action("web.fill", {"target": TARGET, "text": "buyer"})
    supplied.payload["target"]["test_id"] = "  "
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(supplied)
    assert caught.value.code == "WEB_PAYLOAD_INVALID"
    assert caught.value.path == ("target", "test_id", "test_id")

    contaminated = action("web.fill", {"target": TARGET, "text": "buyer"})
    contaminated.payload["text"] = b"bytes"
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(contaminated)
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("payload", "text", "<field>")


def test_the_supplied_object_is_not_mutated_and_the_result_is_isolated() -> None:
    supplied = action("web.fill", {"target": TARGET, "text": "buyer"}, {"note": ["a"]})
    before = copy.deepcopy(supplied.model_dump(mode="python"))
    parsed = forecontract_web.parse_web_action(supplied)
    assert supplied.model_dump(mode="python") == before
    parsed.payload["target"]["test_id"] = "mutated"
    parsed.payload["text"] = "mutated"
    parsed.metadata["note"].append("b")
    assert supplied.payload["target"]["test_id"] == "checkout-button"
    assert supplied.payload["text"] == "buyer"
    assert supplied.metadata == {"note": ["a"]}


def test_validate_web_action_returns_none_and_changes_nothing() -> None:
    supplied = action()
    before = copy.deepcopy(supplied.model_dump(mode="python"))
    assert forecontract_web.validate_web_action(supplied) is None
    assert supplied.model_dump(mode="python") == before
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.validate_web_action(
            action("web.wait", {"target": TARGET, "state": "enabled"})
        )
    assert caught.value.code == "WEB_PAYLOAD_INVALID"
    with pytest.raises(UnsupportedActionError) as caught:
        forecontract_web.validate_web_action(action("web.drag", {"target": TARGET}))
    assert type(caught.value) is UnsupportedActionError


# ---------------------------------------------------------------- W02 URLs


LEGAL_URLS = [
    "http://example.test",
    "https://example.test",
    "http://example.test/",
    "https://example.test/checkout/step-2?cart=1#pay",
    "http://Example.TEST:8080/A%2Fb/../c?q=%2f&x=1#Frag",
    "https://example.test:443/a",
    "http://[::1]:8080/health",
    "http://127.0.0.1:9/",
    "https://example.test/path?",
]


@pytest.mark.parametrize("url", LEGAL_URLS)
def test_navigation_keeps_a_legal_url_verbatim(url: str) -> None:
    from_mapping = forecontract_web.parse_web_action(
        {"schema_version": "0.1", "action_type": "web.navigate", "payload": {"url": url}}
    )
    from_instance = forecontract_web.parse_web_action(action("web.navigate", {"url": url}))
    assert from_mapping.payload == {"url": url}
    assert from_instance.payload == {"url": url}
    assert set(from_mapping.payload) == {"url"}


@pytest.mark.parametrize(
    "url",
    [
        "",
        " ",
        "http://example.test\n/next",
        "http://example.test/a\x00b",
        "http://example.test/a\x7fb",
        "/checkout",
        "checkout",
        "example.test",
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
        "https://exam ple.test",
        "http://example.test/a b",
        "http://example.test\\evil",
        "javascript:alert(1)",
        "data:text/html,checkout",
        "file:///synthetic/checkout",
        "ftp://example.test/a",
        "about:blank",
        "HTTP://example.test",
    ],
)
def test_navigation_rejects_an_illegal_url_with_the_payload_code(url: str) -> None:
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(
            {"schema_version": "0.1", "action_type": "web.navigate", "payload": {"url": url}}
        )
    assert caught.value.code == "WEB_PAYLOAD_INVALID"
    assert caught.value.path == ("url",)


def test_url_validation_performs_no_network_or_dns_access(monkeypatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access was attempted")

    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "getaddrinfo", explode)
    monkeypatch.setattr(socket, "create_connection", explode)
    for url in LEGAL_URLS:
        parsed = forecontract_web.parse_web_action(action("web.navigate", {"url": url}))
        assert parsed.payload == {"url": url}


def test_no_authorization_is_inferred_from_a_url() -> None:
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(
            action("web.navigate", {"url": "http://user:secret@example.test/checkout"})
        )
    assert caught.value.code == "WEB_PAYLOAD_INVALID"
    parsed = forecontract_web.parse_web_action(
        action("web.navigate", {"url": "https://example.test/checkout"})
    )
    assert set(parsed.payload) == {"url"}
    assert set(parsed.model_dump(mode="python")) == {
        "schema_version",
        "action_type",
        "payload",
        "metadata",
    }


# ---------------------------------------------------------------- W06 exceptions


def test_an_unknown_action_type_raises_the_core_unsupported_error() -> None:
    with pytest.raises(UnsupportedActionError) as caught:
        forecontract_web.parse_web_action(action("web.drag", {"target": TARGET}))
    error = caught.value
    assert type(error) is UnsupportedActionError
    assert isinstance(error, ActionValidationError)
    assert not isinstance(error, InvalidActionPayloadError)
    assert not isinstance(error, forecontract_web.WebValidationError)
    assert str(error) == "the Web contract does not support this action type"
    assert error.__cause__ is None
    assert error.__context__ is None


def test_an_unsupported_action_wins_over_a_malformed_payload() -> None:
    with pytest.raises(UnsupportedActionError) as caught:
        forecontract_web.parse_web_action(
            {
                "schema_version": "0.1",
                "action_type": "web.drag",
                "payload": {"target": {"strategy": "css", "selector": "#checkout"}},
            }
        )
    assert type(caught.value) is UnsupportedActionError


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": "9.9", "action_type": "web.drag", "payload": {}},
        {"action_type": "web.drag", "payload": {}},
        {"schema_version": "0.1", "payload": {}},
        {"schema_version": "0.1", "action_type": "   ", "payload": {}},
        {"schema_version": "0.1", "action_type": "web.drag", "payload": {}, "extra": 1},
        {"schema_version": "0.1", "action_type": "web.drag", "payload": "not-an-object"},
        {"schema_version": "0.1", "action_type": "web.drag", "payload": {}, "metadata": []},
    ],
)
def test_an_invalid_envelope_wins_over_an_unsupported_action(document: dict) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(document)
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert not isinstance(caught.value, InvalidActionPayloadError)


@pytest.mark.parametrize("value", [None, 1, "action", b"x", [("schema_version", "0.1")], 1.5])
def test_a_non_mapping_non_action_candidate_is_rejected(value: object) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(value)
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ()


def test_a_payload_error_is_also_a_core_invalid_action_payload_error() -> None:
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(
            action("web.press", {"target": TARGET, "key": "Control+A"})
        )
    error = caught.value
    assert issubclass(forecontract_web.WebActionPayloadError, InvalidActionPayloadError)
    assert isinstance(error, InvalidActionPayloadError)
    assert isinstance(error, ActionValidationError)
    assert isinstance(error, forecontract_web.WebValidationError)
    assert isinstance(error, ValueError)
    assert error.code == "WEB_PAYLOAD_INVALID"
    assert error.path == ("key",)


def test_diagnostics_never_transcribe_a_supplied_value() -> None:
    documents = [
        {
            "schema_version": "0.1",
            "action_type": "web.fill",
            "payload": {"target": {"strategy": "test_id", "test_id": MARKER}, "text": 123},
        },
        {
            "schema_version": "0.1",
            "action_type": "web.navigate",
            "payload": {"url": "javascript:" + MARKER},
        },
        {
            "schema_version": "0.1",
            "action_type": "web.wait",
            "payload": {"target": TARGET, "state": MARKER},
        },
        {
            "schema_version": "0.1",
            "action_type": "web.click",
            "payload": {"target": TARGET, MARKER: MARKER},
        },
        {"schema_version": "0.1", "action_type": "web.click", "payload": MARKER},
    ]
    for document in documents:
        with pytest.raises(forecontract_web.WebValidationError) as caught:
            forecontract_web.parse_web_action(document)
        assert_no_marker(caught.value)
    with pytest.raises(UnsupportedActionError) as caught:
        forecontract_web.parse_web_action(
            {"schema_version": "0.1", "action_type": MARKER, "payload": {}}
        )
    assert_no_marker(caught.value)
    assert str(caught.value) == "the Web contract does not support this action type"


def test_an_unknown_payload_key_becomes_a_constant_placeholder() -> None:
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(
            {
                "schema_version": "0.1",
                "action_type": "web.click",
                "payload": {"target": TARGET, MARKER: 1},
            }
        )
    assert caught.value.code == "WEB_PAYLOAD_INVALID"
    assert caught.value.path == ("<field>",)
    assert MARKER not in str(caught.value)
    assert MARKER not in repr(caught.value)


def test_public_error_attributes_hold_only_bounded_values() -> None:
    with pytest.raises(forecontract_web.WebActionPayloadError) as caught:
        forecontract_web.parse_web_action(action("web.press", {"target": TARGET, "key": MARKER}))
    payload_error = caught.value
    assert set(vars(payload_error)) == {"message", "path"}
    assert payload_error.message == "the action payload does not match the Web action schema"
    assert payload_error.path == ("key",)
    assert payload_error.args == (
        "WEB_PAYLOAD_INVALID at key: the action payload does not match the Web action schema",
    )

    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(action("web.navigate", {"url": "javascript:" + MARKER}))
    navigation_error = caught.value
    assert set(vars(navigation_error)) == {"message", "path"}
    assert navigation_error.code == "WEB_PAYLOAD_INVALID"

    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(
            observation({"schema_version": "0.1", "url": "javascript:" + MARKER, "targets": []})
        )
    observation_error = caught.value
    assert set(vars(observation_error)) == {"message", "path", "code"}
    assert observation_error.code == "WEB_OBSERVATION_INVALID"
    assert observation_error.path == ("url",)
    assert_no_marker(observation_error)


# ---------------------------------------------------------------- W04 observation ingress


def test_parse_web_observation_returns_the_domain_snapshot() -> None:
    supplied = observation(
        {
            "schema_version": "0.1",
            "url": "about:blank",
            "targets": [evidence_document(match_count=0, visible=None, text=None, value=None)],
        }
    )
    snapshot = forecontract_web.parse_web_observation(supplied)
    assert type(snapshot) is forecontract_web.WebObservationState
    assert snapshot.schema_version == "0.1"
    assert snapshot.url == "about:blank"
    assert len(snapshot.targets) == 1
    assert snapshot.targets[0].status == "collected"
    assert snapshot.targets[0].match_count == 0
    assert snapshot.targets[0].visible is None


def test_parse_web_observation_does_not_forge_or_replace_the_outer_identity() -> None:
    snapshot = forecontract_web.parse_web_observation(
        observation(observation_id="obs-42", summary="outer summary")
    )
    assert not hasattr(snapshot, "observation_id")
    assert not hasattr(snapshot, "timestamp")
    assert not hasattr(snapshot, "artifacts")
    assert not hasattr(snapshot, "metadata")
    assert not hasattr(snapshot, "environment_type")


@pytest.mark.parametrize("environment_type", ["fake", "core", "Web", "web ", "WEB"])
def test_parse_web_observation_rejects_a_non_web_environment_type(
    environment_type: str,
) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(observation(environment_type=environment_type))
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("environment_type",)


DAMAGED_ENVELOPES = [
    observation_document(drop=("observation_id",)),
    observation_document(drop=("timestamp",)),
    observation_document(drop=("state",)),
    observation_document(drop=("schema_version",)),
    observation_document(schema_version="9.9"),
    observation_document(schema_version=1),
    observation_document(environment_type=1),
    observation_document(artifacts="not-a-list"),
    observation_document(state=[]),
    observation_document(state="not-an-object"),
    observation_document(state={"schema_version": "0.1", "url": float("nan"), "targets": []}),
    observation_document(state={"schema_version": "0.1", "url": None, "targets": [], "x": 1.0}),
    observation_document(unknown_outer=MARKER),
]


@pytest.mark.parametrize("document", DAMAGED_ENVELOPES)
def test_parse_web_observation_rejects_a_damaged_outer_envelope(document: dict) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(document)
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_parse_web_observation_rejects_a_non_string_outer_key() -> None:
    document = observation_document()
    document[1] = "value"
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(document)
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("<field>",)


@pytest.mark.parametrize("value", [None, 1, "observation", b"x", [1, 2], 1.5])
def test_parse_web_observation_rejects_a_non_mapping_non_observation(value: object) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(value)
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ()


def test_a_snapshot_state_is_not_accepted_as_an_outer_observation() -> None:
    snapshot = forecontract_web.parse_web_observation(observation())
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(snapshot)
    assert caught.value.code == "WEB_OBSERVATION_INVALID"


def test_a_copied_observation_cannot_hide_an_undeclared_key() -> None:
    supplied = observation().model_copy(update={"undeclared": MARKER})
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(supplied)
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("<field>",)
    assert_no_marker(caught.value)


def test_an_in_place_damaged_state_is_rejected() -> None:
    supplied = observation()
    supplied.state["verdict"] = "pass"
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(supplied)
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("<field>",)
    assert "verdict" not in str(caught.value)


DAMAGED_STATES = [
    ({"schema_version": "0.1", "targets": []}, ("url",)),
    ({"schema_version": "0.1", "url": None}, ("targets",)),
    ({"url": None, "targets": []}, ("schema_version",)),
    ({"schema_version": "9.9", "url": None, "targets": []}, ("schema_version",)),
    ({"schema_version": "0.1", "url": "/checkout", "targets": []}, ("url",)),
    ({"schema_version": "0.1", "url": "about:blank#x", "targets": []}, ("url",)),
    ({"schema_version": "0.1", "url": None, "targets": {}}, ("targets",)),
    ({"schema_version": "0.1", "url": None, "targets": [{"target": TARGET}]}, ("targets", 0, "status")),
    (
        {
            "schema_version": "0.1",
            "url": None,
            "targets": [evidence_document(), evidence_document(visible=None, text=None)],
        },
        ("targets",),
    ),
    (
        {
            "schema_version": "0.1",
            "url": None,
            "targets": [evidence_document(match_count=None, visible=None, text=None, value=None)],
        },
        ("targets", 0),
    ),
    (
        {
            "schema_version": "0.1",
            "url": None,
            "targets": [
                evidence_document(
                    status="unavailable",
                    match_count=0,
                    visible=None,
                    text=None,
                    value=None,
                    error_code="TARGET_QUERY_FAILED",
                )
            ],
        },
        ("targets", 0),
    ),
    (
        {
            "schema_version": "0.1",
            "url": None,
            "targets": [
                evidence_document(
                    status="unavailable",
                    match_count=None,
                    visible=None,
                    text=None,
                    value=None,
                    error_code=None,
                )
            ],
        },
        ("targets", 0),
    ),
    (
        {
            "schema_version": "0.1",
            "url": None,
            "targets": [evidence_document(error_code="TARGET_QUERY_FAILED")],
        },
        ("targets", 0),
    ),
    (
        {
            "schema_version": "0.1",
            "url": None,
            "targets": [evidence_document(match_count=True)],
        },
        ("targets", 0, "match_count"),
    ),
    (
        {
            "schema_version": "0.1",
            "url": None,
            "targets": [evidence_document(visible=1)],
        },
        ("targets", 0, "visible"),
    ),
]


@pytest.mark.parametrize(("document", "location"), DAMAGED_STATES)
def test_parse_web_observation_rejects_a_damaged_state(
    document: dict, location: tuple
) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(observation(document))
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == location


def test_parse_web_observation_accepts_null_url_and_empty_targets() -> None:
    snapshot = forecontract_web.parse_web_observation(
        observation({"schema_version": "0.1", "url": None, "targets": []})
    )
    assert snapshot.url is None
    assert snapshot.targets == []
    assert snapshot.model_dump(mode="python") == {
        "schema_version": "0.1",
        "url": None,
        "targets": [],
    }


def test_parse_web_observation_isolates_the_returned_snapshot() -> None:
    supplied = observation(
        {
            "schema_version": "0.1",
            "url": "http://example.test",
            "targets": [evidence_document()],
        }
    )
    before = copy.deepcopy(supplied.model_dump(mode="python"))
    snapshot = forecontract_web.parse_web_observation(supplied)
    assert supplied.model_dump(mode="python") == before
    supplied.state["targets"].clear()
    supplied.state["url"] = "http://other.test"
    assert snapshot.url == "http://example.test"
    assert len(snapshot.targets) == 1
    assert snapshot.targets[0].text == "Checkout"
    assert type(snapshot.targets[0].target) is forecontract_web.TestIdTarget


# ---------------------------------------------------------------- F01/F02/F03 ingress hardening


def _extra_slot_record(record, schema_name: str):
    """Build a derived schema that allows an extra key and store one on a real instance."""
    derived = type(
        schema_name,
        (type(record),),
        {"model_config": {**type(record).model_config, "extra": "allow"}},
    )
    dirty = derived.model_validate({**record.model_dump(mode="json"), "hidden_slot": MARKER})
    assert dirty.__pydantic_extra__, "the derived schema must really store the unknown key"
    assert "hidden_slot" not in dirty.__dict__, "the key must live in the extra store, not __dict__"
    return dirty


@pytest.mark.parametrize("entry", ["parse", "validate"])
def test_a_stored_extra_slot_on_an_action_is_rejected_instead_of_dropped(entry: str) -> None:
    """F01: a key held in the extra store must be rejected, never silently discarded."""
    dirty = _extra_slot_record(action(), "_ExtraActionSpec")
    call = forecontract_web.parse_web_action if entry == "parse" else forecontract_web.validate_web_action
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(dirty)
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("<field>",)
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_stored_extra_slot_on_an_observation_is_rejected_instead_of_dropped() -> None:
    """F01: the observation ingress rejects an unknown key held in the extra store."""
    clean = Observation.model_validate(observation_document())
    dirty = _extra_slot_record(clean, "_ExtraObservation")
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(dirty)
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("<field>",)
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def _artifact(**overrides) -> ArtifactRef:
    document = {
        "schema_version": "0.1",
        "artifact_id": "art-1",
        "artifact_type": "screenshot",
        "uri": "https://example.test/a.png",
    }
    document.update(overrides)
    return ArtifactRef.model_validate(document)


def _observation_with_nested(artifact: ArtifactRef) -> Observation:
    """A real observation whose nested artifact reference has been replaced in place."""
    observation = Observation.model_validate(
        {**observation_document(), "artifacts": [_artifact().model_dump(mode="python")]}
    )
    object.__setattr__(observation, "artifacts", [artifact])
    return observation


@pytest.mark.parametrize(
    ("field", "value", "tail"),
    [
        ("artifact_id", 7, "<field>"),
        ("artifact_type", b"screenshot", "<field>"),
        ("uri", None, "<field>"),
        ("uri", 123, "<field>"),
        ("schema_version", "0.2", "schema_version"),
    ],
)
def test_a_contaminated_nested_artifact_reference_is_rejected(
    field: str, value: object, tail: str
) -> None:
    """F02: a nested Core model is revalidated from its stored content, not shallow-checked."""
    dirty = _artifact().model_copy(update={field: value})
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(_observation_with_nested(dirty))
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("artifacts", 0, tail)
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_valid_nested_artifact_reference_still_passes_and_keeps_the_outer_identity() -> None:
    """F02 positive control: an honest nested reference is accepted and nothing is rewritten."""
    observation = _observation_with_nested(_artifact())
    state = forecontract_web.parse_web_observation(observation)
    assert state.url is None
    assert observation.observation_id == "obs-1"
    assert observation.environment_type == "web"
    assert [artifact.artifact_id for artifact in observation.artifacts] == ["art-1"]


class _ExplodingMapping(dict):
    """A legal Mapping whose reads raise the synthetic marker instead of returning data."""

    def __init__(self, *, where: str = "items") -> None:
        super().__init__()
        self._where = where

    def items(self):
        if self._where == "items":
            raise ValueError(MARKER)
        return super().items()

    def __iter__(self):
        raise ValueError(MARKER)

    def __getitem__(self, key):
        raise ValueError(MARKER)


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        ("parse", "WEB_ACTION_INVALID"),
        ("validate", "WEB_ACTION_INVALID"),
        ("observe", "WEB_OBSERVATION_INVALID"),
    ],
)
def test_a_failing_mapping_read_becomes_a_bounded_web_error(entry: str, code: str) -> None:
    """F03: a dependency failure while reading a Mapping is converted, not propagated raw."""
    candidate = _ExplodingMapping()
    call = {
        "parse": forecontract_web.parse_web_action,
        "validate": forecontract_web.validate_web_action,
        "observe": forecontract_web.parse_web_observation,
    }[entry]
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(candidate)
    assert caught.value.code == code
    assert caught.value.path == ()
    assert MARKER not in str(caught.value)
    assert MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_process_control_interrupt_from_a_mapping_is_not_swallowed() -> None:
    """F03: only ordinary dependency failures are converted, never a process-control signal."""

    class _Interrupting(dict):
        def items(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        forecontract_web.parse_web_action(_Interrupting())


def test_the_bounded_dependency_path_does_not_mutate_the_supplied_mapping() -> None:
    """F03 positive control: a normal mapping is read without being changed."""
    candidate = {
        "schema_version": "0.1",
        "action_type": "web.click",
        "payload": {"target": TARGET},
    }
    before = copy.deepcopy(candidate)
    parsed = forecontract_web.parse_web_action(candidate)
    assert candidate == before
    assert parsed.action_type == "web.click"


# ---------------------------------------------------------------- F03 caller-origin and artifact reads


def _forged_web_error() -> forecontract_web.WebValidationError:
    """A forged bounded error whose code, path and message all carry the marker."""
    return forecontract_web.WebValidationError(
        MARKER, code="WEB_PAYLOAD_INVALID", path=(MARKER,)
    )


class _ForgedWebMapping(dict):
    """A legal Mapping whose reads raise a forged Web error of the package's own classes."""

    def items(self):
        raise _forged_web_error()

    def __iter__(self):
        raise _forged_web_error()

    def __getitem__(self, key):
        raise _forged_web_error()


class _ForgedCoreMapping(dict):
    """A legal Mapping whose reads raise a Core action validation error."""

    def items(self):
        raise ActionValidationError(MARKER)

    def __iter__(self):
        raise ActionValidationError(MARKER)

    def __getitem__(self, key):
        raise ActionValidationError(MARKER)


@pytest.mark.parametrize("entry", ["parse", "validate", "observe"])
@pytest.mark.parametrize("forged", ["web_error", "core_error"])
def test_a_caller_raised_public_error_cannot_borrow_the_diagnostic(
    entry: str, forged: str
) -> None:
    """F03: origin, not exception class, decides - a forged error from a read is converted."""
    candidate = _ForgedWebMapping() if forged == "web_error" else _ForgedCoreMapping()
    call = {
        "parse": forecontract_web.parse_web_action,
        "validate": forecontract_web.validate_web_action,
        "observe": forecontract_web.parse_web_observation,
    }[entry]
    expected = "WEB_OBSERVATION_INVALID" if entry == "observe" else "WEB_ACTION_INVALID"
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(candidate)
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == expected
    assert caught.value.path == ()
    assert not isinstance(caught.value, InvalidActionPayloadError)
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_caller_raised_unsupported_error_is_not_re_raised_as_itself() -> None:
    """F03: an UnsupportedActionError thrown by caller code cannot claim the internal verdict."""

    class _Forging(dict):
        def items(self):
            raise UnsupportedActionError(MARKER)

    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action(_Forging())
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert MARKER not in str(caught.value)


class _ExplodingArtifacts(list):
    """A list subclass whose own iteration raises the marker."""

    def __iter__(self):
        raise ValueError(MARKER)


def _observation_with_artifacts(container) -> Observation:
    observation = Observation.model_validate(observation_document())
    object.__setattr__(observation, "artifacts", container)
    return observation


def test_an_artifacts_container_iteration_failure_is_bounded() -> None:
    """F03: the artifacts container read is inside the conversion boundary."""
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(
            _observation_with_artifacts(_ExplodingArtifacts())
        )
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_the_artifacts_control_does_not_weaken_the_round_two_nested_checks() -> None:
    """F03 control: a contaminated nested artifact still refuses with its own path."""
    dirty = _artifact().model_copy(update={"uri": 123})
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(_observation_with_artifacts([dirty]))
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("artifacts", 0, "<field>")


# ---------------------------------------------------------------- F03-C key hashing and F03-D entry equality


class _HashingKey(str):
    """A str subclass that spells a legal key but raises when hashed."""

    def __hash__(self):
        raise ValueError(MARKER)


class _KeyedMapping(Mapping):
    """A legal Mapping that yields caller-owned key objects without hashing them itself."""

    def __init__(self, pairs) -> None:
        self._pairs = list(pairs)

    def items(self):
        return list(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self):
        return (key for key, _ in self._pairs)

    def __getitem__(self, key):
        for candidate, value in self._pairs:
            if candidate is key:
                return value
        raise KeyError(key)


def _hashing_key_document():
    """A structurally complete action document whose keys raise on hashing."""
    return _KeyedMapping([
        (_HashingKey("schema_version"), "0.1"),
        (_HashingKey("action_type"), "web.click"),
        (_HashingKey("payload"), {"target": {"strategy": "test_id", "test_id": "checkout-button"}}),
    ])


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        ("parse", "WEB_ACTION_INVALID"),
        ("validate", "WEB_ACTION_INVALID"),
        ("observe", "WEB_OBSERVATION_INVALID"),
    ],
)
def test_a_mapping_key_that_raises_on_hashing_is_bounded(entry: str, code: str) -> None:
    """F03-C: materialising a mapping does not make its keys trusted."""
    call = {
        "parse": forecontract_web.parse_web_action,
        "validate": forecontract_web.validate_web_action,
        "observe": forecontract_web.parse_web_observation,
    }[entry]
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(_hashing_key_document())
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == code
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_plain_str_derived_key_still_parses() -> None:
    """F03-C positive: an ordinary str subclass is still a usable key."""

    class _PlainKey(str):
        pass

    parsed = forecontract_web.parse_web_action(_KeyedMapping([
        (_PlainKey("schema_version"), "0.1"),
        (_PlainKey("action_type"), "web.click"),
        (_PlainKey("payload"), {"target": {"strategy": "test_id", "test_id": "checkout-button"}}),
    ]))
    assert parsed.action_type == "web.click"


def test_an_unknown_key_is_still_rejected_with_its_safe_path() -> None:
    """F03-C positive: the unknown-field rejection is unchanged and still bounded."""
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action({
            "schema_version": "0.1",
            "action_type": "web.click",
            "payload": {"target": {"strategy": "test_id", "test_id": "checkout-button"}},
            "hidden_field": "x",
        })
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("<field>",)
    assert "hidden_field" not in str(caught.value)


def test_a_normal_mapping_still_parses_after_the_key_boundary() -> None:
    """F03-C positive: the plain mapping path is unchanged."""
    parsed = forecontract_web.parse_web_action({
        "schema_version": "0.1",
        "action_type": "web.fill",
        "payload": {"target": {"strategy": "role", "role": "textbox", "name": "Search"}, "text": ""},
    })
    assert parsed.payload["text"] == ""


# ---------------------------------------------------------------- F03-C full key chain: copy and lookup

_ACTION_PAIRS = (
    ("schema_version", "0.1"),
    ("action_type", "web.click"),
    ("payload", {"target": {"strategy": "test_id", "test_id": "checkout-button"}}),
)
_OBSERVATION_PAIRS = (
    ("schema_version", "0.1"),
    ("observation_id", "obs-1"),
    ("environment_type", "web"),
    ("state", {"schema_version": "0.1", "url": None, "targets": []}),
    ("timestamp", "2026-10-21T00:00:00+00:00"),
)


class _SecondHashKey(str):
    """A legal key whose first hash succeeds and whose second hash raises."""

    hashes = 0

    def __hash__(self):
        type(self).hashes += 1
        if type(self).hashes >= 2:
            raise ValueError(MARKER)
        return str.__hash__(self)

    def __eq__(self, other):
        return str.__eq__(self, other)


class _SecondCompareKey(str):
    """A legal key whose first comparison succeeds and whose second comparison raises."""

    compares = 0

    def __hash__(self):
        return str.__hash__(self)

    def __eq__(self, other):
        type(self).compares += 1
        if type(self).compares >= 2:
            raise ValueError(MARKER)
        return str.__eq__(self, other)


def _keyed(pairs, key_class):
    return _KeyedMapping([(key_class(name), value) for name, value in pairs])


_ENTRIES = (
    ("parse", forecontract_web.parse_web_action, _ACTION_PAIRS, "WEB_ACTION_INVALID"),
    ("validate", forecontract_web.validate_web_action, _ACTION_PAIRS, "WEB_ACTION_INVALID"),
    ("observe", forecontract_web.parse_web_observation, _OBSERVATION_PAIRS, "WEB_OBSERVATION_INVALID"),
)


@pytest.mark.parametrize(("entry", "call", "pairs", "code"), _ENTRIES)
def test_a_second_hash_of_a_retained_key_is_bounded(entry, call, pairs, code) -> None:
    """F03-C: the store after the membership test hashes the same caller key again."""
    _SecondHashKey.hashes = 0
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(_keyed(pairs, _SecondHashKey))
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == code
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert _SecondHashKey.hashes >= 2, "the second hash must really be attempted inside the product"


@pytest.mark.parametrize(("entry", "call", "pairs", "code"), _ENTRIES)
def test_a_second_comparison_of_a_retained_key_is_bounded(entry, call, pairs, code) -> None:
    """F03-C: a retained caller key still compares itself when the field is looked up later."""
    _SecondCompareKey.compares = 0
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(_keyed(pairs, _SecondCompareKey))
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == code
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert _SecondCompareKey.compares >= 2, "the second comparison must really be attempted"


def test_the_full_key_chain_leaves_the_ordinary_key_path_alone() -> None:
    """F03-C control: an ordinary str subclass key still parses both envelopes."""

    class _PlainKey(str):
        pass

    parsed = forecontract_web.parse_web_action(_keyed(_ACTION_PAIRS, _PlainKey))
    assert parsed.action_type == "web.click"
    state = forecontract_web.parse_web_observation(_keyed(_OBSERVATION_PAIRS, _PlainKey))
    assert state.url is None


def test_an_unknown_key_is_still_rejected_after_the_full_key_chain() -> None:
    """F03-C control: the unknown-key rejection keeps its bounded path and echoes nothing."""
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_action({
            "schema_version": "0.1",
            "action_type": "web.click",
            "payload": {"target": {"strategy": "test_id", "test_id": "checkout-button"}},
            "hidden_field": "x",
        })
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ("<field>",)
    assert "hidden_field" not in str(caught.value)


# ---------------------------------------------------------------- F03-C writeback: the fourth comparison

_WRITEBACK_BASE = (
    ("schema_version", "0.1"),
    ("action_type", "web.click"),
)
_OBSERVATION_WRITEBACK_BASE = (
    ("schema_version", "0.1"),
    ("observation_id", "obs-1"),
    ("environment_type", "web"),
    ("timestamp", "2026-10-21T00:00:00+00:00"),
)
_WRITEBACK_VALUES = {
    "payload": {"target": {"strategy": "test_id", "test_id": "checkout-button"}},
    "metadata": {"note": "x"},
    "state": {"schema_version": "0.1", "url": None, "targets": []},
    "artifacts": [],
}


class _FourthCompareKey(str):
    """A legal key whose first three comparisons succeed and whose fourth raises."""

    compares = 0
    threshold = 4

    def __hash__(self):
        return str.__hash__(self)

    def __eq__(self, other):
        type(self).compares += 1
        if type(self).compares >= type(self).threshold:
            raise ValueError(MARKER)
        return str.__eq__(self, other)


def _writeback_action(field: str):
    pairs = _WRITEBACK_BASE + ((field, _WRITEBACK_VALUES[field]),)
    return _KeyedMapping([(_FourthCompareKey(name), value) for name, value in pairs])


def _writeback_observation(field: str):
    pairs = _OBSERVATION_WRITEBACK_BASE + ((field, _WRITEBACK_VALUES[field]),)
    return _KeyedMapping([(_FourthCompareKey(name), value) for name, value in pairs])


@pytest.mark.parametrize(("entry", "call"), (
    ("parse", forecontract_web.parse_web_action),
    ("validate", forecontract_web.validate_web_action),
))
@pytest.mark.parametrize("field", ("payload", "metadata"))
def test_a_fourth_comparison_during_writeback_is_bounded(entry, call, field) -> None:
    """F03-C: writing the rewritten value back returns to the retained caller key."""
    _FourthCompareKey.compares = 0
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(_writeback_action(field))
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == "WEB_ACTION_INVALID"
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert _FourthCompareKey.compares >= 4, "the fourth comparison must really be attempted"


@pytest.mark.parametrize("field", ("state", "artifacts"))
def test_a_fourth_comparison_during_observation_writeback_is_bounded(field) -> None:
    """F03-C: the observation writeback is bounded for both rewritten fields."""
    _FourthCompareKey.compares = 0
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(_writeback_observation(field))
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert _FourthCompareKey.compares >= 4


def test_the_writeback_boundary_leaves_ordinary_keys_alone() -> None:
    """F03-C control: ordinary action and observation keys still parse with every field."""

    class _PlainKey(str):
        pass

    parsed = forecontract_web.parse_web_action(_KeyedMapping([
        (_PlainKey(name), value) for name, value in _WRITEBACK_BASE + (
            ("payload", _WRITEBACK_VALUES["payload"]),
            ("metadata", _WRITEBACK_VALUES["metadata"]),
        )
    ]))
    assert parsed.action_type == "web.click"
    assert parsed.metadata == {"note": "x"}
    state = forecontract_web.parse_web_observation(_KeyedMapping([
        (_PlainKey(name), value) for name, value in _OBSERVATION_WRITEBACK_BASE + (
            ("state", _WRITEBACK_VALUES["state"]),
            ("artifacts", _WRITEBACK_VALUES["artifacts"]),
        )
    ]))
    assert state.url is None
    assert state.targets == []


# ---------------------------------------------------------------- F03 frozen argument read

_FROZEN_PAYLOAD = {"target": {"strategy": "test_id", "test_id": "checkout-button"}}
_FROZEN_METADATA = {"note": "x"}
_FROZEN_STATE = {"schema_version": "0.1", "url": None, "targets": []}


class _ExplodingBacking(dict):
    """The mapping a FrozenMapping wraps, whose own reads raise the marker."""

    def __init__(self, payload, mode: str) -> None:
        super().__init__(payload)
        self._mode = mode

    def __iter__(self):
        if self._mode == "iter":
            raise ValueError(MARKER)
        return super().__iter__()

    def __getitem__(self, key):
        if self._mode == "getitem":
            raise ValueError(MARKER)
        return super().__getitem__(key)


def _poisoned_frozen(payload, mode: str) -> FrozenMapping:
    return FrozenMapping(_ExplodingBacking(payload, mode))


def _frozen_action_document(field: str, frozen) -> dict:
    document = {"schema_version": "0.1", "action_type": "web.click", "payload": _FROZEN_PAYLOAD}
    if field == "metadata":
        document["metadata"] = _FROZEN_METADATA
    document[field] = frozen
    return document


def _frozen_observation_document(frozen) -> dict:
    return {
        "schema_version": "0.1",
        "observation_id": "obs-1",
        "environment_type": "web",
        "state": frozen,
        "timestamp": "2026-10-21T00:00:00+00:00",
    }


_FROZEN_ENTRIES = (
    ("parse", forecontract_web.parse_web_action, "payload", "WEB_ACTION_INVALID"),
    ("validate", forecontract_web.validate_web_action, "payload", "WEB_ACTION_INVALID"),
    ("parse", forecontract_web.parse_web_action, "metadata", "WEB_ACTION_INVALID"),
    ("validate", forecontract_web.validate_web_action, "metadata", "WEB_ACTION_INVALID"),
    ("observe", forecontract_web.parse_web_observation, "state", "WEB_OBSERVATION_INVALID"),
)


@pytest.mark.parametrize(("entry", "call", "field", "code"), _FROZEN_ENTRIES)
@pytest.mark.parametrize("mode", ("iter", "getitem"))
def test_a_frozen_argument_read_that_raises_is_bounded(entry, call, field, code, mode) -> None:
    """F03: normalising a frozen container is its own caller-owned read, not a store argument."""
    payload = (_FROZEN_PAYLOAD if field == "payload"
               else _FROZEN_METADATA if field == "metadata" else _FROZEN_STATE)
    frozen = _poisoned_frozen(payload, mode)
    document = (_frozen_observation_document(frozen) if entry == "observe"
                else _frozen_action_document(field, frozen))
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        call(document)
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == code
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_the_frozen_boundary_leaves_the_artifacts_controls_intact() -> None:
    """F03 control: the round-002 artifacts path and its iteration boundary are unchanged."""
    dirty = _artifact().model_copy(update={"uri": 123})
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.parse_web_observation(_observation_with_artifacts([dirty]))
    assert caught.value.code == "WEB_OBSERVATION_INVALID"
    assert caught.value.path == ("artifacts", 0, "<field>")

    with pytest.raises(forecontract_web.WebValidationError) as exploding:
        forecontract_web.parse_web_observation(
            _observation_with_artifacts(_ExplodingArtifacts())
        )
    assert exploding.value.code == "WEB_OBSERVATION_INVALID"
    assert exploding.value.path == ()


def test_exact_frozen_containers_still_parse() -> None:
    """F03 control: legitimate FrozenMapping and FrozenActionSpec inputs keep working."""
    payload = freeze_json(_FROZEN_PAYLOAD)
    metadata = freeze_json(_FROZEN_METADATA)
    state = freeze_json(_FROZEN_STATE)

    parsed = forecontract_web.parse_web_action({
        "schema_version": "0.1",
        "action_type": "web.click",
        "payload": payload,
        "metadata": metadata,
    })
    assert parsed.payload["target"]["test_id"] == "checkout-button"
    assert parsed.metadata["note"] == "x"

    observed = forecontract_web.parse_web_observation(_frozen_observation_document(state))
    assert observed.url is None
    assert list(observed.targets) == []

    frozen_action = freeze_action(forecontract_web.parse_web_action({
        "schema_version": "0.1",
        "action_type": "web.click",
        "payload": _FROZEN_PAYLOAD,
    }))
    reparsed = forecontract_web.parse_web_action(frozen_action)
    assert reparsed.action_type == "web.click"
    assert reparsed.payload["target"]["test_id"] == "checkout-button"


def test_a_process_control_signal_from_a_frozen_read_still_propagates() -> None:
    """F03 control: only ordinary exceptions are converted, never KeyboardInterrupt."""

    class _Interrupting(dict):
        def __iter__(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        forecontract_web.parse_web_action({
            "schema_version": "0.1",
            "action_type": "web.click",
            "payload": FrozenMapping(_Interrupting(_FROZEN_PAYLOAD)),
        })
