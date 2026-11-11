"""Acceptance tests for the explicit Web capability constructor and the root surface (W05/W07).

The capability declaration is the caller's statement, never a default: both lists are required,
each entry must belong to the closed Web vocabulary, and the constructor creates no adapter, opens
no browser and proves nothing about what is implemented, exercised or verified. The root export
set is asserted literally, never derived from the module's own ``__all__``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
WEB_SRC = Path(__file__).resolve().parents[2] / "packages" / "adapters" / "web" / "src"
for candidate in (CORE_SRC, WEB_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import forecontract_web
import forecontract_web.capabilities
from forecontract.models.actions import AdapterCapabilities
from pydantic import BaseModel

MARKER = "SYNTHETIC_PRIVATE_VALUE_9f3a"

#: The exact agreed public surface of ``forecontract_web``, written out rather than imported.
EXPECTED_PUBLIC_NAMES = {
    "__version__",
    "WEB_ACTION_TYPES",
    "WEB_OBSERVATION_TYPES",
    "TestIdTarget",
    "RoleTarget",
    "NavigatePayload",
    "ClickPayload",
    "FillPayload",
    "PressPayload",
    "SelectPayload",
    "WaitPayload",
    "WebTargetEvidence",
    "WebObservationState",
    "WebValidationError",
    "WebActionPayloadError",
    "parse_web_action",
    "validate_web_action",
    "parse_web_observation",
    "build_web_capabilities",
}


def capabilities(**overrides) -> AdapterCapabilities:
    """Declare the default one-action capability set with the named arguments replaced."""
    arguments = {
        "adapter_name": "web-1",
        "supported_actions": ["web.click"],
        "supported_observation_types": ["web.snapshot"],
    }
    arguments.update(overrides)
    return forecontract_web.build_web_capabilities(**arguments)


# ---------------------------------------------------------------- W05 construction


def test_both_lists_are_required_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        forecontract_web.build_web_capabilities("web-1", ["web.click"], ["web.snapshot"])
    with pytest.raises(TypeError):
        forecontract_web.build_web_capabilities(
            adapter_name="web-1", supported_actions=["web.click"]
        )
    with pytest.raises(TypeError):
        forecontract_web.build_web_capabilities(
            adapter_name="web-1", supported_observation_types=["web.snapshot"]
        )
    with pytest.raises(TypeError):
        forecontract_web.build_web_capabilities()


def test_an_explicit_empty_declaration_is_valid() -> None:
    declared = capabilities(supported_actions=[], supported_observation_types=[])
    assert declared.model_dump() == {
        "schema_version": "0.1",
        "adapter_name": "web-1",
        "environment_type": "web",
        "supported_actions": [],
        "supported_observation_types": [],
        "supported_condition_types": [],
        "supported_outcome_types": [],
        "metadata": {},
    }


def test_a_subset_declaration_is_valid_and_keeps_the_supplied_order() -> None:
    declared = capabilities(supported_actions=["web.wait", "web.navigate"])
    assert declared.supported_actions == ["web.wait", "web.navigate"]
    assert declared.supported_observation_types == ["web.snapshot"]


def test_the_full_six_action_vocabulary_is_valid() -> None:
    declared = capabilities(
        supported_actions=[
            "web.navigate",
            "web.click",
            "web.fill",
            "web.press",
            "web.select",
            "web.wait",
        ]
    )
    assert declared.supported_actions == [
        "web.navigate",
        "web.click",
        "web.fill",
        "web.press",
        "web.select",
        "web.wait",
    ]


def test_the_declaration_fixes_the_environment_and_leaves_conditions_empty() -> None:
    declared = capabilities()
    assert declared.environment_type == "web"
    assert declared.supported_condition_types == []
    assert declared.supported_outcome_types == []
    assert type(declared) is AdapterCapabilities


@pytest.mark.parametrize(
    "action_type",
    ["web.drag", "click", "web.click ", "WEB.CLICK", "", " ", "web.snapshot"],
)
def test_an_action_type_outside_the_closed_vocabulary_is_rejected(action_type: str) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_actions=["web.click", action_type])
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_actions", 1)


@pytest.mark.parametrize("observation_type", ["web.state", "web.snapshot ", "", "fake.state"])
def test_an_observation_type_outside_the_closed_vocabulary_is_rejected(
    observation_type: str,
) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_observation_types=[observation_type])
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_observation_types", 0)


def test_a_duplicate_entry_is_rejected_in_both_lists() -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_actions=["web.click", "web.click"])
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_actions", 1)
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(
            supported_observation_types=["web.snapshot", "web.snapshot"]
        )
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_observation_types", 1)


@pytest.mark.parametrize(
    "value",
    [("web.click",), "web.click", {"web.click"}, {"actions": ["web.click"]}, None, 1, {"web.click": 1}],
)
def test_a_non_list_action_declaration_is_rejected(value) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_actions=value)
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_actions",)


@pytest.mark.parametrize("value", [("web.snapshot",), "web.snapshot", {"web.snapshot"}, None, 1])
def test_a_non_list_observation_declaration_is_rejected(value) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_observation_types=value)
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_observation_types",)


@pytest.mark.parametrize("name", ["", " ", "\t", "\n", "\u00a0"])
def test_a_blank_adapter_name_is_rejected(name: str) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(adapter_name=name)
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("adapter_name",)


@pytest.mark.parametrize("name", [None, 1, True, 1.5, b"web-1", ["web-1"], {"name": "web-1"}])
def test_a_non_string_adapter_name_is_rejected(name) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(adapter_name=name)
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("adapter_name",)


@pytest.mark.parametrize("entry", ["", " ", "\t", None, 1, True, ["web.click"], {"web.click": 1}])
def test_a_blank_or_non_string_action_entry_is_rejected(entry) -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_actions=["web.click", entry])
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_actions", 1)


def test_the_inputs_are_not_mutated_and_the_model_is_detached() -> None:
    actions = ["web.click"]
    observations = ["web.snapshot"]
    declared = capabilities(
        supported_actions=actions, supported_observation_types=observations
    )
    assert actions == ["web.click"]
    assert observations == ["web.snapshot"]
    actions.append("web.wait")
    actions[0] = "web.press"
    observations.clear()
    assert declared.supported_actions == ["web.click"]
    assert declared.supported_observation_types == ["web.snapshot"]


def test_each_call_returns_an_independent_model() -> None:
    first = capabilities()
    second = capabilities()
    assert first is not second
    assert first.supported_actions is not second.supported_actions
    first.supported_actions.append("web.wait")
    first.adapter_name = "web-2"
    assert second.supported_actions == ["web.click"]
    assert second.adapter_name == "web-1"


# ---------------------------------------------------------------- non-goals


def test_the_constructor_creates_no_adapter_and_claims_no_verification() -> None:
    declared = capabilities()
    assert type(declared) is AdapterCapabilities
    assert not hasattr(forecontract_web, "WebAdapter")
    assert not hasattr(forecontract_web, "BrowserSession")
    assert not hasattr(forecontract_web, "RuntimeDomainSupport")
    assert not hasattr(forecontract_web, "JsonRuleEvaluator")
    assert not hasattr(forecontract_web.capabilities, "WebAdapter")
    assert not hasattr(forecontract_web.capabilities, "capabilities")
    assert "playwright" not in sys.modules
    assert "selenium" not in sys.modules


def test_a_capability_error_transcribes_no_supplied_value() -> None:
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_actions=[MARKER])
    error = caught.value
    assert error.code == "WEB_CAPABILITIES_INVALID"
    assert error.path == ("supported_actions", 0)
    assert set(vars(error)) == {"message", "path", "code"}
    assert MARKER not in str(error)
    assert MARKER not in repr(error)
    assert MARKER not in repr(error.args)
    for value in vars(error).values():
        assert MARKER not in repr(value)
    assert error.__cause__ is None
    assert error.__context__ is None


# ---------------------------------------------------------------- W07 root surface


def test_the_root_export_set_is_exactly_the_agreed_nineteen_names() -> None:
    assert len(forecontract_web.__all__) == 19
    assert len(set(forecontract_web.__all__)) == 19
    assert set(forecontract_web.__all__) == EXPECTED_PUBLIC_NAMES
    for name in sorted(EXPECTED_PUBLIC_NAMES):
        assert hasattr(forecontract_web, name), name


def test_the_root_version_is_the_unchanged_package_version() -> None:
    assert forecontract_web.__version__ == "0.1.0"


def test_the_root_surface_exposes_no_runnable_browser_or_verifier() -> None:
    for absent in (
        "WebAdapter",
        "Browser",
        "BrowserSession",
        "Playwright",
        "Selenium",
        "RuntimeDomainSupport",
        "JsonRuleEvaluator",
        "RuleEvaluation",
        "RuleValidationHooks",
    ):
        assert not hasattr(forecontract_web, absent)


def test_the_exported_names_have_the_agreed_kinds() -> None:
    for name in (
        "TestIdTarget",
        "RoleTarget",
        "NavigatePayload",
        "ClickPayload",
        "FillPayload",
        "PressPayload",
        "SelectPayload",
        "WaitPayload",
        "WebTargetEvidence",
        "WebObservationState",
    ):
        assert issubclass(getattr(forecontract_web, name), BaseModel)
    for name in (
        "parse_web_action",
        "validate_web_action",
        "parse_web_observation",
        "build_web_capabilities",
    ):
        assert callable(getattr(forecontract_web, name))
    assert isinstance(forecontract_web.WEB_ACTION_TYPES, tuple)
    assert isinstance(forecontract_web.WEB_OBSERVATION_TYPES, tuple)
    assert issubclass(forecontract_web.WebActionPayloadError, forecontract_web.WebValidationError)
    assert issubclass(forecontract_web.WebValidationError, ValueError)


# ---------------------------------------------------------------- F03 string-operation hardening


class _ExplodingName(str):
    """A str subclass whose own ``strip`` raises the synthetic marker."""

    def strip(self, *args, **kwargs):
        raise ValueError(MARKER)


def test_an_adapter_name_string_operation_failure_becomes_a_bounded_error() -> None:
    """F03: a failing ``strip`` on the supplied name is converted to the capability code."""
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(adapter_name=_ExplodingName("probe"))
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ()
    assert MARKER not in str(caught.value)
    assert MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_declaration_entry_string_operation_failure_becomes_a_bounded_error() -> None:
    """F03: the same conversion applies to an entry of a declaration list."""
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        capabilities(supported_actions=[_ExplodingName("web.click")])
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ()
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_a_process_control_interrupt_from_a_name_is_not_swallowed() -> None:
    """F03: a process-control signal raised by the dependency is never converted."""

    class _InterruptingName(str):
        def strip(self, *args, **kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        capabilities(adapter_name=_InterruptingName("probe"))


def test_a_normal_name_and_entries_still_build_a_detached_record() -> None:
    """F03 positive control: ordinary strings keep working and nothing is mutated."""
    actions = ["web.click"]
    observations = ["web.snapshot"]
    record = forecontract_web.build_web_capabilities(
        adapter_name="probe",
        supported_actions=actions,
        supported_observation_types=observations,
    )
    assert record.adapter_name == "probe"
    assert record.environment_type == "web"
    assert record.supported_actions == ["web.click"]
    assert actions == ["web.click"] and observations == ["web.snapshot"]


# ---------------------------------------------------------------- F03 caller-origin string operations


def _forged_web_error() -> forecontract_web.WebValidationError:
    """A forged bounded error whose code, path and message all carry the marker."""
    return forecontract_web.WebValidationError(
        MARKER, code="WEB_PAYLOAD_INVALID", path=(MARKER,)
    )


class _ForgedName(str):
    """A str subclass whose own ``strip`` raises a forged Web error."""

    def strip(self, *args, **kwargs):
        raise _forged_web_error()


@pytest.mark.parametrize(
    ("kwargs", "expected_path"),
    [
        ({"adapter_name": _ForgedName("probe")}, ()),
        ({"supported_actions": [_ForgedName("web.click")]}, ()),
        ({"supported_observation_types": [_ForgedName("web.snapshot")]}, ()),
    ],
)
def test_a_forged_public_error_from_a_string_operation_is_converted(
    kwargs: dict, expected_path: tuple
) -> None:
    """F03: a caller's own error class cannot supply the capability diagnostic."""
    arguments = {
        "adapter_name": "probe",
        "supported_actions": ["web.click"],
        "supported_observation_types": ["web.snapshot"],
    }
    arguments.update(kwargs)
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.build_web_capabilities(**arguments)
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == expected_path
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_the_capability_diagnostic_keeps_its_own_paths_after_the_origin_rule() -> None:
    """F03 control: genuine rejections still carry the specific field path."""
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.build_web_capabilities(
            adapter_name="probe",
            supported_actions=["web.click", "web.click"],
            supported_observation_types=[],
        )
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ("supported_actions", 1)


# ---------------------------------------------------------------- F03-D entry equality and closed vocabulary


class _EqualityEntry(str):
    """A str subclass that spells a legal entry but raises when compared."""

    def __eq__(self, other):
        raise ValueError(MARKER)

    def __hash__(self) -> int:
        return str.__hash__(self)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"supported_actions": [_EqualityEntry("web.click")]},
        {"supported_observation_types": [_EqualityEntry("web.snapshot")]},
    ],
)
def test_a_declaration_entry_that_raises_on_equality_is_bounded(kwargs: dict) -> None:
    """F03-D: materialising and stripping a list does not make its entries trusted."""
    arguments = {
        "adapter_name": "probe",
        "supported_actions": ["web.click"],
        "supported_observation_types": ["web.snapshot"],
    }
    arguments.update(kwargs)
    with pytest.raises(forecontract_web.WebValidationError) as caught:
        forecontract_web.build_web_capabilities(**arguments)
    assert type(caught.value) is forecontract_web.WebValidationError
    assert caught.value.code == "WEB_CAPABILITIES_INVALID"
    assert caught.value.path == ()
    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_an_ordinary_str_derived_declaration_still_builds() -> None:
    """F03-D positive: an ordinary str subclass declaration is still accepted."""

    class _PlainName(str):
        pass

    class _PlainEntry(str):
        pass

    record = forecontract_web.build_web_capabilities(
        adapter_name=_PlainName("probe"),
        supported_actions=[_PlainEntry("web.click"), "web.fill"],
        supported_observation_types=[_PlainEntry("web.snapshot")],
    )
    assert record.environment_type == "web"
    assert list(record.supported_actions) == ["web.click", "web.fill"]
    assert list(record.supported_observation_types) == ["web.snapshot"]


def test_the_genuine_entry_rejections_keep_their_specific_paths() -> None:
    """F03-D positive: duplicate and out-of-vocabulary entries keep their precise paths."""
    with pytest.raises(forecontract_web.WebValidationError) as duplicate:
        forecontract_web.build_web_capabilities(
            adapter_name="probe",
            supported_actions=["web.click", "web.click"],
            supported_observation_types=[],
        )
    assert duplicate.value.code == "WEB_CAPABILITIES_INVALID"
    assert duplicate.value.path == ("supported_actions", 1)

    with pytest.raises(forecontract_web.WebValidationError) as outside:
        forecontract_web.build_web_capabilities(
            adapter_name="probe",
            supported_actions=["web.drag"],
            supported_observation_types=[],
        )
    assert outside.value.code == "WEB_CAPABILITIES_INVALID"
    assert outside.value.path == ("supported_actions", 0)
