"""Read-only source-verified Semantic and Procedural derived tag index."""

from __future__ import annotations

from datetime import datetime

from forecontract.memory.derived_models import (
    DERIVED_SCHEMA_VERSION,
    DerivedEntryRecord,
    aware_datetime_to_utc_micros,
    validate_entries,
    validate_query_args,
)
from forecontract.memory.episodic import JsonlEpisodicMemory
from forecontract.memory.errors import (
    MemoryNotFoundError,
    MemorySourceError,
    MemoryValidationError,
)
from forecontract.memory.models import MemoryItem

__all__ = ["DerivedMemoryIndex"]


class DerivedMemoryIndex:
    """Read-only tag index over host-supplied semantic notes and high-level procedures.

    The index never writes files, never caches source results across queries and never
    interprets or executes procedural steps. Every query validates the live episodic store
    and returns detached copies of matching entries paired with their real ``MemoryItem``.
    """

    def __init__(
        self,
        *,
        episodic_memory: JsonlEpisodicMemory,
        entries: list[dict[str, object]],
    ) -> None:
        if not isinstance(episodic_memory, JsonlEpisodicMemory):
            raise MemoryValidationError("an episodic memory store instance is required")
        self._episodic_memory = episodic_memory
        self._entries: tuple[DerivedEntryRecord, ...] = validate_entries(entries)

    def query(
        self,
        *,
        tags: list[str],
        now: datetime,
        kind: str | None = None,
        limit: int = 5,
    ) -> dict[str, object]:
        """Verify referenced episodic sources and return ranked, detached tag hits."""
        query_tag_set, now_utc_us, wanted_kind, wanted_limit = validate_query_args(
            tags=tags,
            now=now,
            kind=kind,
            limit=limit,
        )

        heads = self._read_heads()
        head_ids = {item.memory_id for item in heads}
        items_by_id = self._resolve_distinct_items()
        self._verify_entry_identities(items_by_id)

        excluded: list[dict[str, object]] = []
        candidates: list[tuple[float, DerivedEntryRecord, MemoryItem]] = []
        query_tag_count = len(query_tag_set)

        for entry in self._entries:
            source_item = items_by_id[entry.memory_id]
            reason = self._exclusion_reason(
                entry=entry,
                source_item=source_item,
                head_ids=head_ids,
                now_utc_us=now_utc_us,
                wanted_kind=wanted_kind,
                query_tag_set=query_tag_set,
            )
            if reason is not None:
                excluded.append({"entry_id": entry.entry_id, "reason": reason})
                continue
            overlap = len(entry.tag_set & query_tag_set)
            score = overlap / query_tag_count
            candidates.append((score, entry, source_item))

        candidates.sort(key=lambda row: (-row[0], row[1].entry_id))
        selected = candidates[:wanted_limit]
        for _, overflow_entry, _ in candidates[wanted_limit:]:
            excluded.append({"entry_id": overflow_entry.entry_id, "reason": "limit"})

        excluded.sort(key=lambda item: str(item["entry_id"]))
        hits: list[dict[str, object]] = [
            {
                "entry": entry.to_dict(),
                "source": source_item.model_dump(mode="json"),
                "score": score,
            }
            for score, entry, source_item in selected
        ]
        return {
            "schema_version": DERIVED_SCHEMA_VERSION,
            "hits": hits,
            "excluded": excluded,
        }

    def _read_heads(self) -> tuple[MemoryItem, ...]:
        try:
            return self._episodic_memory.query()
        except MemoryNotFoundError as exc:
            raise MemorySourceError(
                "the referenced episodic memory record does not exist",
                cause=exc,
            ) from exc

    def _resolve_distinct_items(self) -> dict[str, MemoryItem]:
        resolved: dict[str, MemoryItem] = {}
        for memory_id in dict.fromkeys(entry.memory_id for entry in self._entries):
            try:
                resolved[memory_id] = self._episodic_memory.get(memory_id)
            except MemoryNotFoundError as exc:
                raise MemorySourceError(
                    "the referenced episodic memory record does not exist",
                    cause=exc,
                ) from exc
        return resolved

    def _verify_entry_identities(self, items_by_id: dict[str, MemoryItem]) -> None:
        for entry in self._entries:
            item = items_by_id[entry.memory_id]
            if (
                item.memory_id != entry.memory_id
                or item.run_id != entry.run_id
                or item.source_record_id != entry.source_record_id
                or item.source_level != entry.source_level
            ):
                raise MemorySourceError(
                    "the derived entry does not match the referenced episodic memory identity"
                )

    @staticmethod
    def _exclusion_reason(
        *,
        entry: DerivedEntryRecord,
        source_item: MemoryItem,
        head_ids: set[str],
        now_utc_us: int,
        wanted_kind: str | None,
        query_tag_set: frozenset[str],
    ) -> str | None:
        if entry.memory_id not in head_ids:
            return "superseded"
        if source_item.result != "success":
            return "source_not_success"
        if aware_datetime_to_utc_micros(source_item.created_at) > now_utc_us:
            return "source_in_future"
        if entry.expires_at_utc_us is not None and now_utc_us >= entry.expires_at_utc_us:
            return "expired"
        if wanted_kind is not None and entry.kind != wanted_kind:
            return "kind_mismatch"
        if not (entry.tag_set & query_tag_set):
            return "no_match"
        return None
