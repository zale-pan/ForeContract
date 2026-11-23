"""Read and write protocols for the structured tracing store.

The two protocols are deliberately small and synchronous. They describe *persistence*, not
scheduling: no Runtime, no retry policy, no deadline handling and no crash recovery lives
behind them.

``get_record`` and ``read_run`` return detached snapshots. A caller may keep, compare or
discard the result without any way to reach back into the store's own objects.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from forecontract.tracing.models import TraceRecord

__all__ = ["TraceReader", "TraceWriter"]


@runtime_checkable
class TraceWriter(Protocol):
    """Append validated facts to one trace."""

    def append(self, record: TraceRecord) -> None:
        """Validate history and candidate, then write one complete JSONL line.

        Success means one encode, write, flush and close completed without error. It is a
        durability claim for that one write only: no ``fsync`` guarantee is made. The
        sequence number is supplied by the caller and the store never invents one.
        """
        ...


@runtime_checkable
class TraceReader(Protocol):
    """Read recorded facts back as detached snapshots."""

    def get_record(self, run_id: str, record_id: str) -> TraceRecord:
        """Return one record of ``run_id``.

        Raises ``TraceNotFoundError`` when the run or the record does not exist. A record
        identifier that belongs to a different run is not returned as a source of this run.
        """
        ...

    def read_run(self, run_id: str) -> tuple[TraceRecord, ...]:
        """Return every record of ``run_id`` ordered by ascending ``sequence``.

        Raises ``TraceNotFoundError`` when the run does not exist.
        """
        ...
