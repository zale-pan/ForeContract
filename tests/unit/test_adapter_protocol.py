"""Tests for the structural Adapter protocol.

These tests prove the declared interface and one test-local sample flow only. They do not
prove conformance of every future domain adapter, and they make no lifecycle, ordering,
retry, deadline or cancellation guarantee.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.models import (
    ActionSpec,
    ActualOutcome,
    AdapterCapabilities,
    ExecutionStatus,
    JsonObject,
    Observation,
)
from forecontract.protocols import (
    ActionValidationError,
    Adapter,
    InvalidActionPayloadError,
    UnsupportedActionError,
)

UTC = dt.timezone.utc
TIMESTAMP = dt.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
SYNC_METHODS = ("capabilities", "validate_action")
ASYNC_METHODS = ("start", "stop", "observe", "execute")
ALL_MEMBERS = ("name", *SYNC_METHODS, *ASYNC_METHODS)


class LocalTestAdapter:
    """A test-local adapter: it demonstrates the interface, nothing more."""

    SUPPORTED_ACTIONS = ("navigate", "read_text")

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._started = False

    @property
    def name(self) -> str:
        return "local-test-adapter"

    def capabilities(self) -> AdapterCapabilities:
        self.calls.append("capabilities")
        return AdapterCapabilities(
            schema_version="0.1", adapter_name=self.name, environment_type="local_test_site",
            supported_actions=list(self.SUPPORTED_ACTIONS), supported_observation_types=["page"],
            supported_condition_types=["element_visible"], supported_outcome_types=["page_contains"],
        )

    def validate_action(self, action: ActionSpec) -> None:
        self.calls.append("validate_action")
        if action.action_type not in self.SUPPORTED_ACTIONS:
            # Unsupported action types are rejected before any payload inspection.
            raise UnsupportedActionError(f"unsupported action type: {action.action_type}")
        if action.action_type == "navigate":
            url = action.payload.get("url")
            if not isinstance(url, str) or not url.strip():
                raise InvalidActionPayloadError("navigate requires a non-empty 'url' payload")

    async def start(self) -> None:
        self.calls.append("start")
        self._started = True

    async def stop(self) -> None:
        self.calls.append("stop")
        self._started = False

    async def observe(self) -> Observation:
        self.calls.append("observe")
        return Observation(
            schema_version="0.1", observation_id="local-1", environment_type="local_test_site",
            state={"url": "https://example.invalid"}, timestamp=TIMESTAMP,
        )

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        self.calls.append("execute")
        return ActualOutcome(
            schema_version="0.1", outcome_id="local-outcome-1", status=ExecutionStatus.SUCCEEDED,
            duration_ms=1.0,
        )


def action(action_type: str, payload: JsonObject | None = None) -> ActionSpec:
    return ActionSpec(schema_version="0.1", action_type=action_type, payload=payload or {})


# --- declared interface ------------------------------------------------------

def test_protocol_declares_the_agreed_interface() -> None:
    assert getattr(Adapter, "_is_protocol", False) is True
    assert isinstance(Adapter.name, property)
    for member in ALL_MEMBERS:
        assert member in dir(Adapter), member


@pytest.mark.parametrize("member", SYNC_METHODS)
def test_synchronous_members_are_not_coroutines(member: str) -> None:
    function = getattr(Adapter, member)
    assert callable(function)
    assert inspect.iscoroutinefunction(function) is False


@pytest.mark.parametrize("member", ASYNC_METHODS)
def test_asynchronous_members_are_coroutines(member: str) -> None:
    assert inspect.iscoroutinefunction(getattr(Adapter, member)) is True


@pytest.mark.parametrize("member,parameters", [
    ("capabilities", ["self"]),
    ("validate_action", ["self", "action"]),
    ("start", ["self"]),
    ("stop", ["self"]),
    ("observe", ["self"]),
    ("execute", ["self", "action"]),
])
def test_member_parameter_lists_are_exact(member: str, parameters: list[str]) -> None:
    assert list(inspect.signature(getattr(Adapter, member)).parameters) == parameters


@pytest.mark.parametrize("member", (*SYNC_METHODS, *ASYNC_METHODS))
def test_no_member_introduces_a_context_parameter(member: str) -> None:
    signature = inspect.signature(getattr(Adapter, member))
    assert "context" not in signature.parameters
    assert "ctx" not in signature.parameters


def test_name_property_introduces_no_context_parameter() -> None:
    assert list(inspect.signature(Adapter.name.fget).parameters) == ["self"]


def test_validate_action_declares_a_none_return_and_execute_returns_an_outcome() -> None:
    assert inspect.signature(Adapter.validate_action, eval_str=True).return_annotation is None
    assert inspect.signature(Adapter.execute, eval_str=True).return_annotation is ActualOutcome
    assert inspect.signature(Adapter.observe, eval_str=True).return_annotation is Observation
    assert inspect.signature(Adapter.capabilities, eval_str=True).return_annotation is AdapterCapabilities
    assert inspect.signature(Adapter.start, eval_str=True).return_annotation is None
    assert inspect.signature(Adapter.stop, eval_str=True).return_annotation is None


# --- validation exceptions ---------------------------------------------------

def test_action_validation_exception_hierarchy() -> None:
    assert issubclass(ActionValidationError, ValueError)
    assert issubclass(UnsupportedActionError, ActionValidationError)
    assert issubclass(InvalidActionPayloadError, ActionValidationError)
    assert not issubclass(UnsupportedActionError, InvalidActionPayloadError)
    assert not issubclass(InvalidActionPayloadError, UnsupportedActionError)


def test_validation_exceptions_carry_their_message() -> None:
    with pytest.raises(UnsupportedActionError, match="nope"):
        raise UnsupportedActionError("nope")
    with pytest.raises(InvalidActionPayloadError, match="bad payload"):
        raise InvalidActionPayloadError("bad payload")
    with pytest.raises(ActionValidationError):
        raise InvalidActionPayloadError("caught as the base class")


# --- test-local adapter flow -------------------------------------------------

def test_local_adapter_satisfies_every_protocol_member() -> None:
    adapter = LocalTestAdapter()
    for member in ALL_MEMBERS:
        assert hasattr(adapter, member), member
    assert isinstance(adapter.name, str) and adapter.name


def test_local_adapter_threads_the_protocol_flow() -> None:
    adapter = LocalTestAdapter()

    async def flow() -> tuple[AdapterCapabilities, Observation, ActualOutcome]:
        await adapter.start()
        declared = adapter.capabilities()
        adapter.validate_action(action("navigate", {"url": "https://example.invalid"}))
        return declared, await adapter.observe(), await adapter.execute(action("read_text"))

    capabilities, observation, outcome = asyncio.run(flow())
    asyncio.run(adapter.stop())
    assert isinstance(capabilities, AdapterCapabilities)
    assert capabilities.supported_actions == ["navigate", "read_text"]
    assert isinstance(observation, Observation) and observation.environment_type == "local_test_site"
    assert isinstance(outcome, ActualOutcome) and outcome.status is ExecutionStatus.SUCCEEDED
    assert adapter.calls == ["start", "capabilities", "validate_action", "observe", "execute", "stop"]
    assert adapter._started is False


def test_unsupported_action_is_rejected_before_payload_validation() -> None:
    adapter = LocalTestAdapter()
    # Both payloads would be malformed for a supported action, but the action type is
    # unsupported, so the unsupported-action error must win.
    with pytest.raises(UnsupportedActionError):
        adapter.validate_action(action("teleport", {"url": ""}))
    with pytest.raises(UnsupportedActionError):
        adapter.validate_action(action("teleport", {"url": 5}))
    assert adapter.calls == ["validate_action", "validate_action"]


def test_supported_action_with_malformed_payload_raises_the_payload_error() -> None:
    adapter = LocalTestAdapter()
    for payload in ({}, {"url": ""}, {"url": "   "}, {"url": 5}):
        with pytest.raises(InvalidActionPayloadError):
            adapter.validate_action(action("navigate", payload))


def test_supported_action_with_valid_payload_validates_silently() -> None:
    adapter = LocalTestAdapter()
    assert adapter.validate_action(action("navigate", {"url": "https://example.invalid"})) is None
    assert adapter.validate_action(action("read_text", {"selector": "#main"})) is None


def test_validate_action_performs_no_environment_io() -> None:
    adapter = LocalTestAdapter()
    adapter.validate_action(action("read_text"))
    assert adapter.calls == ["validate_action"]
    assert adapter._started is False
