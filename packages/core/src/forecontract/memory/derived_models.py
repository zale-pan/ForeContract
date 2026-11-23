"""Internal strict validation models and helpers for read-only derived memory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from forecontract.memory.errors import MemoryValidationError

__all__: list[str] = []

DERIVED_SCHEMA_VERSION: Final[str] = "0.1"
MAX_ENTRIES: Final[int] = 64
MAX_TAGS: Final[int] = 32
MAX_SEMANTIC_TEXT_LEN: Final[int] = 2048
MAX_PROCEDURAL_STEPS: Final[int] = 16
MAX_STEP_LEN: Final[int] = 256
MAX_EXPIRES_AT_LEN: Final[int] = 40
MAX_QUERY_LIMIT: Final[int] = 20

ENTRY_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "entry_id",
    "kind",
    "memory_id",
    "run_id",
    "source_record_id",
    "source_level",
    "tags",
    "text",
    "steps",
    "expires_at",
)
_ENTRY_KEY_SET: Final[frozenset[str]] = frozenset(ENTRY_KEYS)
_VALID_KINDS: Final[frozenset[str]] = frozenset({"semantic", "procedural"})
_VALID_SOURCE_LEVELS: Final[frozenset[str]] = frozenset({"run", "attempt"})
_ID_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]{1,128}")
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")


@dataclass(frozen=True, slots=True)
class DerivedEntryRecord:
    """Immutable, detached internal representation of one derived index entry."""

    schema_version: str
    entry_id: str
    kind: str
    memory_id: str
    run_id: str
    source_record_id: str
    source_level: str
    tags: tuple[str, ...]
    tag_set: frozenset[str]
    text: str | None
    steps: tuple[str, ...]
    expires_at: str | None
    expires_at_utc_us: int | None

    def to_dict(self) -> dict[str, object]:
        """Return a fresh detached plain dictionary copy of this entry."""
        return {
            "schema_version": self.schema_version,
            "entry_id": self.entry_id,
            "kind": self.kind,
            "memory_id": self.memory_id,
            "run_id": self.run_id,
            "source_record_id": self.source_record_id,
            "source_level": self.source_level,
            "tags": list(self.tags),
            "text": self.text,
            "steps": list(self.steps),
            "expires_at": self.expires_at,
        }


def _require_identifier(value: object, message: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise MemoryValidationError(message)
    return value


def _validate_tags(value: object, *, message: str) -> tuple[tuple[str, ...], frozenset[str]]:
    if type(value) is not list or not (1 <= len(value) <= MAX_TAGS):
        raise MemoryValidationError(message)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in value:
        if type(item) is not str or _TAG_RE.fullmatch(item) is None:
            raise MemoryValidationError(message)
        if item in seen:
            raise MemoryValidationError(message)
        seen.add(item)
        ordered.append(item)
    return tuple(ordered), frozenset(seen)


_MICROSECONDS_PER_SECOND: Final[int] = 1_000_000
_SECONDS_PER_DAY: Final[int] = 86_400


def aware_datetime_to_utc_micros(
    value: datetime,
    *,
    message: str = "timestamp must be a timezone-aware datetime",
) -> int:
    """Convert an aware datetime to exact integer UTC microseconds without astimezone overflow."""
    if value.tzinfo is None:
        raise MemoryValidationError(message)
    try:
        offset = value.utcoffset()
    except (TypeError, ValueError, OverflowError):
        raise MemoryValidationError(message) from None
    if not isinstance(offset, timedelta):
        raise MemoryValidationError(message)
    # Use integer ordinal/time arithmetic so valid local years near 1 or 9999 never overflow UTC bounds.
    local_seconds = (
        value.toordinal() * _SECONDS_PER_DAY
        + value.hour * 3_600
        + value.minute * 60
        + value.second
    )
    local_us = local_seconds * _MICROSECONDS_PER_SECOND + value.microsecond
    offset_us = (
        (offset.days * _SECONDS_PER_DAY + offset.seconds) * _MICROSECONDS_PER_SECOND
        + offset.microseconds
    )
    return local_us - offset_us


def _parse_aware_datetime(value: object, *, message: str) -> int:
    if type(value) is not datetime:
        raise MemoryValidationError(message)
    return aware_datetime_to_utc_micros(value, message=message)


def _parse_expires_at(value: object) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    msg = "derived entry expires_at must be None or a timezone-aware ISO8601 string"
    if type(value) is not str:
        raise MemoryValidationError(msg)
    if not (1 <= len(value) <= MAX_EXPIRES_AT_LEN) or value != value.strip():
        raise MemoryValidationError(msg)
    if "T" not in value and "t" not in value and " " not in value:
        raise MemoryValidationError(msg)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise MemoryValidationError(msg) from None
    utc_us = _parse_aware_datetime(parsed, message=msg)
    return value, utc_us


def _validate_kind_payload(kind: str, raw_text: object, raw_steps: object) -> tuple[str | None, tuple[str, ...]]:
    if kind == "semantic":
        if (
            type(raw_text) is not str
            or not (1 <= len(raw_text) <= MAX_SEMANTIC_TEXT_LEN)
            or not raw_text.strip()
        ):
            raise MemoryValidationError(
                "semantic entry text must be a non-blank string of 1 to 2048 characters"
            )
        if type(raw_steps) is not list or len(raw_steps) != 0:
            raise MemoryValidationError("semantic entry steps must be an empty list")
        return raw_text, ()

    if raw_text is not None:
        raise MemoryValidationError("procedural entry text must be None")
    if type(raw_steps) is not list or not (1 <= len(raw_steps) <= MAX_PROCEDURAL_STEPS):
        raise MemoryValidationError(
            "procedural entry steps must be a list of 1 to 16 non-blank strings"
        )
    validated_steps: list[str] = []
    for step in raw_steps:
        if type(step) is not str or not (1 <= len(step) <= MAX_STEP_LEN) or not step.strip():
            raise MemoryValidationError(
                "each procedural step must be a non-blank string of 1 to 256 characters"
            )
        validated_steps.append(step)
    return None, tuple(validated_steps)


def validate_entries(entries: object) -> tuple[DerivedEntryRecord, ...]:
    """Validate and detach the caller-supplied derived catalog entries."""
    if type(entries) is not list or len(entries) > MAX_ENTRIES:
        raise MemoryValidationError("derived entries must be a list of at most 64 entries")
    seen_entry_ids: set[str] = set()
    validated: list[DerivedEntryRecord] = []
    for raw in entries:
        if type(raw) is not dict:
            raise MemoryValidationError("each derived entry must be a plain dictionary")
        for key in raw:
            if type(key) is not str:
                raise MemoryValidationError("derived entry keys must be plain strings")
        for required_key in ENTRY_KEYS:
            if required_key not in raw:
                raise MemoryValidationError("derived entry is missing a required field")
        for key in raw:
            if key not in _ENTRY_KEY_SET:
                raise MemoryValidationError("derived entry contains an unexpected field")

        schema_version = raw["schema_version"]
        if type(schema_version) is not str or schema_version != DERIVED_SCHEMA_VERSION:
            raise MemoryValidationError("derived entry schema_version must be 0.1")

        entry_id = _require_identifier(raw["entry_id"], "derived entry_id is invalid")
        if entry_id in seen_entry_ids:
            raise MemoryValidationError("derived entry_id values must be unique")
        seen_entry_ids.add(entry_id)

        kind = raw["kind"]
        if type(kind) is not str or kind not in _VALID_KINDS:
            raise MemoryValidationError("derived entry kind must be semantic or procedural")

        memory_id = _require_identifier(raw["memory_id"], "derived entry memory_id is invalid")
        run_id = _require_identifier(raw["run_id"], "derived entry run_id is invalid")
        source_record_id = _require_identifier(
            raw["source_record_id"], "derived entry source_record_id is invalid"
        )

        source_level = raw["source_level"]
        if type(source_level) is not str or source_level not in _VALID_SOURCE_LEVELS:
            raise MemoryValidationError("derived entry source_level must be run or attempt")

        tags, tag_set = _validate_tags(
            raw["tags"],
            message="derived entry tags must be 1 to 32 unique valid tags",
        )
        text, steps = _validate_kind_payload(kind, raw["text"], raw["steps"])
        expires_at_raw, expires_at_utc_us = _parse_expires_at(raw["expires_at"])

        validated.append(
            DerivedEntryRecord(
                schema_version=schema_version,
                entry_id=entry_id,
                kind=kind,
                memory_id=memory_id,
                run_id=run_id,
                source_record_id=source_record_id,
                source_level=source_level,
                tags=tags,
                tag_set=tag_set,
                text=text,
                steps=steps,
                expires_at=expires_at_raw,
                expires_at_utc_us=expires_at_utc_us,
            )
        )
    return tuple(validated)


def validate_query_args(
    *,
    tags: object,
    now: object,
    kind: object,
    limit: object,
) -> tuple[frozenset[str], int, str | None, int]:
    """Validate derived memory query arguments before reading episodic sources."""
    _, tag_set = _validate_tags(
        tags,
        message="query tags must be 1 to 32 unique valid tags",
    )
    now_utc_us = _parse_aware_datetime(
        now,
        message="query now must be a timezone-aware datetime",
    )
    if kind is not None and (type(kind) is not str or kind not in _VALID_KINDS):
        raise MemoryValidationError("query kind must be None, semantic, or procedural")
    if type(limit) is not int or not (0 <= limit <= MAX_QUERY_LIMIT):
        raise MemoryValidationError("query limit must be an integer between 0 and 20")
    return tag_set, now_utc_us, kind, limit
