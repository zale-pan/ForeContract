"""Acceptance tests for in-process Working Memory.

Working Memory is process-local and file-free, so these tests are pure in-memory: they check the
declared behaviour (two-Run isolation, full replacement, clear, missing, empty state) and the
strict JSON ownership rules (detached copies, string keys, plain containers, finite numbers).

Every assertion checks a value, an identity or a state transition rather than the mere absence
of an exception.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.memory import (
    MemoryNotFoundError,
    MemoryValidationError,
    WorkingMemory,
)


@pytest.fixture
def working() -> WorkingMemory:
    return WorkingMemory()


# --- two-run isolation, replacement and lifecycle -----------------------------

def test_two_runs_are_isolated_from_each_other(working: WorkingMemory) -> None:
    working.set("run-1", {"step": 1})
    working.set("run-2", {"step": 2})
    assert working.get("run-1") == {"step": 1}
    assert working.get("run-2") == {"step": 2}
    assert working.clear("run-1") is True
    assert working.get("run-2") == {"step": 2}
    with pytest.raises(MemoryNotFoundError):
        working.get("run-1")


def test_set_fully_replaces_the_previous_state(working: WorkingMemory) -> None:
    working.set("run-1", {"kept": 1, "dropped": 2})
    working.set("run-1", {"kept": 9})
    assert working.get("run-1") == {"kept": 9}


def test_an_empty_state_is_a_legal_state(working: WorkingMemory) -> None:
    working.set("run-1", {})
    assert working.get("run-1") == {}


def test_clear_reports_whether_the_run_existed(working: WorkingMemory) -> None:
    assert working.clear("run-absent") is False
    working.set("run-1", {"a": 1})
    assert working.clear("run-1") is True
    assert working.clear("run-1") is False


def test_a_missing_run_is_not_found(working: WorkingMemory) -> None:
    with pytest.raises(MemoryNotFoundError) as info:
        working.get("run-absent")
    assert info.value.code == "memory_not_found"


def test_clear_does_not_touch_another_run(working: WorkingMemory) -> None:
    working.set("run-1", {"a": 1})
    working.set("run-2", {"b": 2})
    working.clear("run-1")
    assert working.get("run-2") == {"b": 2}


# --- detachment of inputs and outputs -----------------------------------------

def test_the_caller_cannot_mutate_stored_state_through_its_own_object(working: WorkingMemory) -> None:
    state = {"nested": {"values": [1, 2]}}
    working.set("run-1", state)
    state["nested"]["values"].append(3)
    state["added"] = True
    assert working.get("run-1") == {"nested": {"values": [1, 2]}}


def test_a_returned_state_is_detached_from_the_store(working: WorkingMemory) -> None:
    working.set("run-1", {"nested": {"values": [1]}})
    first = working.get("run-1")
    first["nested"]["values"].append(2)
    first["extra"] = "value"
    assert working.get("run-1") == {"nested": {"values": [1]}}


def test_two_reads_return_equal_but_distinct_objects(working: WorkingMemory) -> None:
    working.set("run-1", {"nested": [1]})
    first = working.get("run-1")
    second = working.get("run-1")
    assert first == second
    assert first is not second
    assert first["nested"] is not second["nested"]


def test_every_json_scalar_kind_round_trips(working: WorkingMemory) -> None:
    state = {"null": None, "true": True, "false": False, "int": 7, "float": 1.5, "text": "value"}
    working.set("run-1", state)
    assert working.get("run-1") == state


# --- strict ownership rejection ------------------------------------------------

@pytest.mark.parametrize("bad", [
    (1, 2),
    {1, 2},
    {"a": (1, 2)},
    {"a": {1, 2}},
    {"a": b"bytes"},
    {"a": object()},
    {"a": math.nan},
    {"a": math.inf},
    {"a": -math.inf},
    {"a": complex(1, 2)},
])
def test_invalid_values_are_rejected(working: WorkingMemory, bad: object) -> None:
    state = bad if isinstance(bad, dict) else {"value": bad}
    with pytest.raises(MemoryValidationError) as info:
        working.set("run-1", state)
    assert info.value.code == "memory_validation"


@pytest.mark.parametrize("key", [1, True, None, (1, 2)])
def test_a_non_string_object_key_is_rejected(working: WorkingMemory, key: object) -> None:
    with pytest.raises(MemoryValidationError):
        working.set("run-1", {key: "value"})


def test_a_nested_non_string_key_is_rejected(working: WorkingMemory) -> None:
    with pytest.raises(MemoryValidationError):
        working.set("run-1", {"outer": [{"inner": {3: "value"}}]})


def test_a_cyclic_container_is_rejected(working: WorkingMemory) -> None:
    cyclic: dict = {"name": "root"}
    cyclic["self"] = cyclic
    with pytest.raises(MemoryValidationError):
        working.set("run-1", cyclic)


def test_a_cyclic_list_is_rejected(working: WorkingMemory) -> None:
    cyclic: list = [1]
    cyclic.append(cyclic)
    with pytest.raises(MemoryValidationError):
        working.set("run-1", {"values": cyclic})


def test_deep_nesting_is_bounded(working: WorkingMemory) -> None:
    deep: dict = {}
    cursor = deep
    for _ in range(200):
        cursor["next"] = {}
        cursor = cursor["next"]
    with pytest.raises(MemoryValidationError):
        working.set("run-1", deep)


@pytest.mark.parametrize("run_id", ["", "   ", 1, True, None])
def test_a_malformed_run_identifier_is_rejected(working: WorkingMemory, run_id: object) -> None:
    with pytest.raises(MemoryValidationError):
        working.set(run_id, {"a": 1})
    with pytest.raises(MemoryValidationError):
        working.get(run_id)
    with pytest.raises(MemoryValidationError):
        working.clear(run_id)


def test_a_non_mapping_state_is_rejected(working: WorkingMemory) -> None:
    with pytest.raises(MemoryValidationError):
        working.set("run-1", ["a", "list"])


# --- failure does not damage the previous state --------------------------------

def test_a_rejected_set_leaves_the_previous_state_untouched(working: WorkingMemory) -> None:
    working.set("run-1", {"kept": [1, 2]})
    with pytest.raises(MemoryValidationError):
        working.set("run-1", {"kept": (1, 2)})
    assert working.get("run-1") == {"kept": [1, 2]}


def test_a_rejected_set_does_not_create_the_run(working: WorkingMemory) -> None:
    with pytest.raises(MemoryValidationError):
        working.set("run-1", {"bad": math.nan})
    with pytest.raises(MemoryNotFoundError):
        working.get("run-1")


def test_working_memory_never_touches_the_filesystem(working: WorkingMemory) -> None:
    """Working Memory is process-local: no path, no handle and no stored file object."""
    working.set("run-1", {"a": 1})
    assert working.get("run-1") == {"a": 1}
    assert not hasattr(working, "path")
    assert [name for name in vars(working) if not name.startswith("_states")] == []


# --- round-002 correction: F02 exact builtin JSON types and plain keys ---------

class _CallerString(str):
    """A caller-defined string subclass, which is not one of the approved JSON types."""


class _CallerInt(int):
    """A caller-defined integer subclass."""


class _CallerFloat(float):
    """A caller-defined float subclass."""


class _CallerList(list):
    """A caller-defined list subclass."""


class _CallerDict(dict):
    """A caller-defined mapping subclass."""


@pytest.mark.parametrize("bad,label", [
    (_CallerString("safe"), "string subclass"),
    (_CallerInt(3), "integer subclass"),
    (_CallerFloat(1.5), "float subclass"),
    (_CallerList([1, 2]), "list subclass"),
])
def test_a_caller_defined_scalar_or_list_subclass_is_rejected(
    working: WorkingMemory, bad: object, label: str,
) -> None:
    """F02: only the exact builtin JSON types are approved, never a subclass of one."""
    with pytest.raises(MemoryValidationError) as info:
        working.set("run-1", {"value": bad})
    assert info.value.code == "memory_validation"


def test_a_caller_defined_mapping_subclass_is_rejected(working: WorkingMemory) -> None:
    """F02: the state itself must be a plain dict, not a mapping subclass."""
    with pytest.raises(MemoryValidationError):
        working.set("run-1", _CallerDict(value=1))


def test_a_caller_defined_mapping_subclass_nested_in_the_state_is_rejected(
    working: WorkingMemory,
) -> None:
    """F02: a nested mapping subclass is rejected just like a nested key or value."""
    with pytest.raises(MemoryValidationError):
        working.set("run-1", {"nested": _CallerDict(value=1)})


def test_a_caller_defined_string_subclass_key_is_rejected(working: WorkingMemory) -> None:
    """F02: an object key must be a plain string, so a string subclass is not a key."""
    with pytest.raises(MemoryValidationError) as info:
        working.set("run-1", {_CallerString("key"): 1})
    assert "plain string" in str(info.value)


def test_plain_nested_json_still_round_trips_with_exact_builtin_types(
    working: WorkingMemory,
) -> None:
    """F02 control: ordinary nested JSON is accepted and comes back as plain builtins."""
    state = {"a": [1, {"b": True}], "text": "value", "number": 2.5, "nothing": None}
    working.set("run-1", state)
    returned = working.get("run-1")
    assert returned == state
    assert type(returned) is dict
    assert type(returned["a"]) is list
    assert type(returned["a"][1]) is dict
    assert type(returned["a"][1]["b"]) is bool
    assert type(returned["text"]) is str
    assert type(returned["number"]) is float
    assert type(returned["a"][0]) is int
    assert returned["nothing"] is None


def test_a_rejected_subclass_leaves_the_previous_state_untouched(working: WorkingMemory) -> None:
    """F02: the rejection happens before the replacement, so old state survives."""
    working.set("run-1", {"kept": [1, 2]})
    with pytest.raises(MemoryValidationError):
        working.set("run-1", {"kept": _CallerList([9])})
    assert working.get("run-1") == {"kept": [1, 2]}


def test_a_rejected_subclass_does_not_create_the_run(working: WorkingMemory) -> None:
    """F02: a rejected first write creates nothing at all."""
    with pytest.raises(MemoryValidationError):
        working.set("run-1", _CallerDict(value=1))
    with pytest.raises(MemoryNotFoundError):
        working.get("run-1")
