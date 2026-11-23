"""Single-writer local JSONL Episodic Memory.

:class:`JsonlEpisodicMemory` stores versioned Memory facts, one JSON object per line. It is
deliberately unglamorous: one file, one writer, no lock, no background index, no cache and no
automatic recovery or truncation.

Two invariants shape every operation:

* The **whole file** is validated on every read and every append, and so is **every** stored
  version's real trace source, including superseded versions. A corrupt line, a stored version
  whose source has disappeared or a graph that no longer holds is an explicit failure; nothing
  is skipped, filtered or repaired, and a query with ``limit=0`` still performs the full check.
* A **candidate is validated and detached before** any byte is written, and the acknowledgement
  must be a real integer character count equal to the length of the encoded line. A rejected
  candidate leaves the file untouched; a partial operating-system write may leave a bad tail,
  which the next read reports as corruption.

One logical experience has exactly one root chain. A first record without ``supersedes_id`` is
allowed only while that experience has no chain; every later record must name the current head.
The chain key is the logical experience derived from the resolved real source, never the source
locator: two Attempts are always two experiences, and no Attempt can correct another one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from forecontract.memory.errors import (
    MemoryCorruptionError,
    MemoryNotFoundError,
    MemoryStorageError,
    MemoryValidationError,
)
from forecontract.memory.models import MEMORY_RESULTS, MemoryItem, validated_instance
from forecontract.memory.sources import TraceSourceResolver, experience_key

__all__ = ["JsonlEpisodicMemory"]


@dataclass
class _Experience:
    """The validated chain of one logical experience."""

    version_ids: list[str] = field(default_factory=list)

    @property
    def head(self) -> str | None:
        return self.version_ids[-1] if self.version_ids else None


@dataclass
class _Head:
    """The validated state of one Memory file."""

    experiences: dict[tuple[str, ...], _Experience] = field(default_factory=dict)
    known_ids: set[str] = field(default_factory=set)
    owner: dict[str, str] = field(default_factory=dict)
    by_id: dict[str, MemoryItem] = field(default_factory=dict)
    items: list[MemoryItem] = field(default_factory=list)


def _require_identifier(value: object, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"the {field_name} must be a non-blank string")
    return value


def _require_limit(value: object) -> int | None:
    """Accept only ``None`` or a strict non-negative integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MemoryValidationError("the limit must be a non-negative integer or absent")
    if value < 0:
        raise MemoryValidationError("the limit must be a non-negative integer or absent")
    return value


class JsonlEpisodicMemory:
    """Versioned, append-only Episodic Memory stored as JSONL."""

    def __init__(self, path: str | Path, *, source_resolver: TraceSourceResolver) -> None:
        if not isinstance(source_resolver, TraceSourceResolver):
            raise MemoryValidationError("a trace source resolver is required")
        try:
            self._path = Path(path)
        except (TypeError, ValueError) as exc:
            raise MemoryStorageError("the memory path is not a usable filesystem path", cause=exc) from exc
        self._resolver = source_resolver

    @property
    def path(self) -> Path:
        """The configured Memory path. Never an open handle."""
        return self._path

    # ------------------------------------------------------------------ writing

    def append(self, item: MemoryItem) -> None:
        """Validate the whole history, the candidate and its source, then write one line."""
        head = self._validated_head()
        candidate = validated_instance(MemoryItem, item)
        if candidate.memory_id in head.known_ids:
            raise MemoryValidationError("the memory identifier is already used in this file")
        resolved = self._resolver.resolve_for_item(candidate)
        self._check_supersession(candidate, resolved, head)
        try:
            line = json.dumps(
                candidate.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise MemoryValidationError("the memory record could not be encoded", cause=exc) from exc
        self._write_line(line + "\n")

    @staticmethod
    def _check_supersession(item: MemoryItem, resolved: Any, head: _Head) -> None:
        experience = experience_key(resolved)
        chain = head.experiences.get(experience)
        current = chain.head if chain is not None else None
        if item.supersedes_id is None:
            if current is not None:
                raise MemoryValidationError("this experience already has a chain head")
            return
        if item.supersedes_id == item.memory_id:
            raise MemoryValidationError("a memory version cannot supersede itself")
        if current is None:
            raise MemoryValidationError("the superseded memory version does not exist")
        if item.supersedes_id != current:
            raise MemoryValidationError("the superseded memory version is not the current head")
        if head.owner.get(item.supersedes_id) != item.run_id:
            raise MemoryValidationError("the superseded memory version belongs to another run")

    def _write_line(self, text: str) -> None:
        """Write one complete line and close it, mapping every failure to a bounded error."""
        try:
            with open(self._path, "a", encoding="utf-8", newline="", closefd=True) as handle:
                written = handle.write(text)
                if not self._is_complete_write(written, text):
                    raise MemoryStorageError("the memory record was only partly written")
                handle.flush()
        except MemoryStorageError:
            raise
        except OSError as exc:
            raise MemoryStorageError("the memory file could not be written", cause=exc) from exc
        except ValueError as exc:
            raise MemoryStorageError("the memory record could not be written", cause=exc) from exc

    @staticmethod
    def _is_complete_write(acknowledged: object, text: str) -> bool:
        """Accept only a real integer character count equal to the supplied length."""
        if isinstance(acknowledged, bool) or not isinstance(acknowledged, int):
            return False
        return acknowledged == len(text)

    # ------------------------------------------------------------------ reading

    def get(self, memory_id: str) -> MemoryItem:
        """Return one exact version, including a superseded one."""
        wanted = _require_identifier(memory_id, "memory_id")
        head = self._validated_head()
        found = head.by_id.get(wanted)
        if found is None:
            raise MemoryNotFoundError("the requested memory record does not exist")
        return found.model_copy(deep=True)

    def query(
        self,
        *,
        run_id: str | None = None,
        source_level: str | None = None,
        result: str | None = None,
        limit: int | None = None,
    ) -> tuple[MemoryItem, ...]:
        """Return the current chain heads that match every supplied exact filter."""
        wanted_run = None if run_id is None else _require_identifier(run_id, "run_id")
        wanted_level = None if source_level is None else _require_identifier(source_level, "source_level")
        if wanted_level is not None and wanted_level not in ("run", "attempt"):
            raise MemoryValidationError("the source level must be run or attempt")
        wanted_result = None if result is None else _require_identifier(result, "result")
        if wanted_result is not None and wanted_result not in MEMORY_RESULTS:
            raise MemoryValidationError("the result is not one of the closed memory results")
        ceiling = _require_limit(limit)

        head = self._validated_head()
        selected = [head.by_id[chain.head] for chain in head.experiences.values() if chain.head is not None]
        if wanted_run is not None:
            selected = [item for item in selected if item.run_id == wanted_run]
        if wanted_level is not None:
            selected = [item for item in selected if item.source_level == wanted_level]
        if wanted_result is not None:
            selected = [item for item in selected if item.result == wanted_result]
        selected.sort(key=lambda item: item.utc_key())
        if ceiling is not None:
            selected = selected[:ceiling]
        return tuple(item.model_copy(deep=True) for item in selected)

    # ------------------------------------------------------------------ internals

    def _read_lines(self) -> list[str]:
        """Read and decode the file. A missing file is the empty history.

        A decode failure is located at the physical 1-based line that actually failed, computed
        from the byte or character offset the decoder reports. The file is never partially
        accepted, skipped or repaired (F03).
        """
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise MemoryStorageError("the memory file could not be read", cause=exc) from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryCorruptionError(
                "the memory file is not valid UTF-8", line_number=_utf8_line(raw, exc.start)
            ) from None
        if not text:
            return []
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        else:
            raise MemoryCorruptionError(
                "the memory file ends with an incomplete record", line_number=len(lines)
            )
        return lines

    @staticmethod
    def _decode_line(line: str, number: int) -> Any:
        try:
            return json.loads(
                line, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys
            )
        except MemoryCorruptionError as exc:
            raise MemoryCorruptionError(exc.message, line_number=number) from None
        except json.JSONDecodeError:
            raise MemoryCorruptionError("the memory line is not valid JSON", line_number=number) from None
        except ValueError:
            raise MemoryCorruptionError(
                "the memory line contains a number the decoder refuses", line_number=number
            ) from None

    def _validated_head(self) -> _Head:
        """Validate the whole file, every source and the whole graph, then return the state."""
        lines = self._read_lines()
        head = _Head()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                raise MemoryCorruptionError("the memory file contains a blank line", line_number=number)
            decoded = self._decode_line(line, number)
            try:
                item = MemoryItem.model_validate(decoded)
            except (ValidationError, MemoryValidationError) as exc:
                raise MemoryCorruptionError(
                    f"the memory line is not a valid record: {_corruption_category(exc)}",
                    line_number=number,
                ) from None
            if item.memory_id in head.known_ids:
                raise MemoryCorruptionError(
                    "the memory identifier is already used in this file", line_number=number
                )
            resolved = self._resolver.resolve_for_item(item, line_number=number)
            self._attach(item, resolved, head, number)
        return head

    def _attach(self, item: MemoryItem, resolved: Any, head: _Head, number: int) -> None:
        """Add one stored version to the graph, rejecting any illegal relation."""
        experience = experience_key(resolved)
        chain = head.experiences.get(experience)
        if item.supersedes_id is None:
            if chain is not None and chain.version_ids:
                raise MemoryCorruptionError(
                    "this experience already has a chain head", line_number=number
                )
            if chain is None:
                chain = _Experience()
                head.experiences[experience] = chain
        else:
            if item.supersedes_id == item.memory_id:
                raise MemoryCorruptionError("a memory version supersedes itself", line_number=number)
            owner = head.owner.get(item.supersedes_id)
            if owner is None:
                raise MemoryCorruptionError(
                    "the superseded memory version does not exist", line_number=number
                )
            if owner != item.run_id:
                raise MemoryCorruptionError(
                    "the superseded memory version belongs to another run", line_number=number
                )
            if chain is None or chain.head != item.supersedes_id:
                raise MemoryCorruptionError(
                    "the superseded memory version is not the current head", line_number=number
                )
        if chain is None:  # pragma: no cover - defensive; a chain exists on both paths
            raise MemoryCorruptionError("the memory version has no chain", line_number=number)
        chain.version_ids.append(item.memory_id)
        head.known_ids.add(item.memory_id)
        head.owner[item.memory_id] = item.run_id
        head.by_id[item.memory_id] = item
        head.items.append(item)


def _reject_constant(name: str) -> Any:
    raise MemoryCorruptionError("the memory line contains a non-finite number")


def _reject_duplicate_keys(pairs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MemoryCorruptionError("the memory line repeats a JSON object key")
        result[key] = value
    return result


def _corruption_category(exc: object) -> str:
    """Name the schema error type of a rejected stored line, without transcribing it."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        found = errors()
        if found:
            return str(found[0].get("type") or "invalid")
    return "invalid"


def _utf8_line(raw: bytes, offset: int) -> int:
    """Return the 1-based physical line that contains one byte offset.

    Only the line break count is used, so no payload byte is ever read into a message.
    """
    return raw[:offset].count(b"\n") + 1
