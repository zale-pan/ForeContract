"""In-process Working Memory: one fully replaced state per Run.

Working Memory is a small, strict process-local store. It holds ordinary JSON data only, never
trace facts, and it never touches the filesystem. ``set`` replaces one Run's state wholesale
after validating and copying the whole input, so a rejected value cannot damage the previous
state and a caller cannot alias the stored one. Every read returns a detached plain dictionary.
"""

from __future__ import annotations

import math
from typing import Any, Final

from forecontract.memory.errors import MemoryNotFoundError, MemoryValidationError

__all__ = ["WorkingMemory"]

#: The largest container nesting a Working state may use, so a pathological input cannot turn a
#: validation into an unbounded walk.
_MAX_DEPTH: Final[int] = 64


def _is_json_scalar(value: object) -> bool:
    """Report whether one value is an exact JSON scalar builtin, never a subclass."""
    the_type = type(value)
    if value is None or the_type is bool or the_type is str or the_type is int:
        return True
    return the_type is float and math.isfinite(value)


def _require_identifier(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise MemoryValidationError(f"the {field_name} must be a non-blank string")
    return value


def _copy_json(value: object, *, depth: int, seen: set[int]) -> Any:
    """Validate one JSON value and return a plain, detached copy of it.

    The accepted value types are exactly the builtins: ``None``, ``bool``, ``str``, ``int``,
    ``float``, ``list`` and ``dict``. A subclass of any of them is a different type and is
    rejected rather than copied or coerced, so the stored state can never alias a caller object
    of a custom type or carry a value this module would not itself produce. A tuple, a set, a
    non-string key, a non-finite number, an arbitrary object and a cycle are all rejected too.
    """
    the_type = type(value)
    if value is None or the_type is bool:
        return value
    if the_type is str:
        return str(value)
    if the_type is int:
        return int(value)
    if the_type is float:
        if not math.isfinite(value):
            raise MemoryValidationError("the working state accepts only finite numbers")
        return float(value)
    if the_type is dict:
        if depth >= _MAX_DEPTH:
            raise MemoryValidationError("the working state is nested too deeply")
        marker = id(value)
        if marker in seen:
            raise MemoryValidationError("the working state contains a cycle")
        seen.add(marker)
        try:
            copied: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise MemoryValidationError("a working state object key must be a plain string")
                copied[str(key)] = _copy_json(item, depth=depth + 1, seen=seen)
            return copied
        finally:
            seen.discard(marker)
    if the_type is list:
        if depth >= _MAX_DEPTH:
            raise MemoryValidationError("the working state is nested too deeply")
        marker = id(value)
        if marker in seen:
            raise MemoryValidationError("the working state contains a cycle")
        seen.add(marker)
        try:
            return [_copy_json(item, depth=depth + 1, seen=seen) for item in value]
        finally:
            seen.discard(marker)
    raise MemoryValidationError("the working state accepts only plain JSON values")


class WorkingMemory:
    """A process-local map from a Run identifier to one replaced JSON state."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def set(self, run_id: str, state: dict[str, Any]) -> None:
        """Validate and copy ``state``, then atomically replace this Run's entry.

        Nothing is stored when the value is rejected, so a failed call leaves the previous
        state exactly as it was.
        """
        key = _require_identifier(run_id, "run_id")
        if type(state) is not dict:
            raise MemoryValidationError("the working state must be a plain object")
        detached = _copy_json(state, depth=0, seen=set())
        self._states[key] = detached

    def get(self, run_id: str) -> dict[str, Any]:
        """Return a detached plain copy of this Run's state, or raise ``not found``."""
        key = _require_identifier(run_id, "run_id")
        stored = self._states.get(key)
        if stored is None:
            raise MemoryNotFoundError("the requested run has no working state")
        return _copy_json(stored, depth=0, seen=set())

    def clear(self, run_id: str) -> bool:
        """Remove only this Run's entry and report whether it existed."""
        key = _require_identifier(run_id, "run_id")
        return self._states.pop(key, None) is not None
