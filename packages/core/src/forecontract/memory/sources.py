"""Resolution of real trace records into Memory sources.

:class:`TraceSourceResolver` wraps the verified read-only tracing protocol. It reads one record
through the real ``TraceReader.get_record``, re-validates the returned record as a member of the
closed tracing union, checks that the record really belongs to the requested run and identifier,
and only then maps it to a Memory result.

Mapping is deliberately narrow:

======================  =========================  ==========================================
real tracing record     Memory source level        Memory result
======================  =========================  ==========================================
``RunStarted``          ``run``                    ``unknown``
``RunFinished``         ``run``                    ``terminal_label(terminal_status)``
``AttemptStarted``      ``attempt``                ``unknown``
``AttemptFinished``     ``attempt``                ``attempt_label(execution_status)``
======================  =========================  ==========================================

``StepStarted``, ``ContractCommitted`` and ``VerificationRecorded`` are not Memory sources and
are rejected. A verification PASS is never a success source on its own, and a later successful
Attempt never rewrites an earlier failed one: each Attempt is its own experience, and only an
explicit new Memory version that references a new real fact changes a chain head.

The logical experience key is kept apart from the source locator. A run-level fact is located by
``(run_id, record_id)`` but belongs to the logical experience ``("run", run_id)``; an
attempt-level fact belongs to ``("attempt", run_id, step_id, contract_id, attempt_id)``.
"""

from __future__ import annotations

from typing import Final

from forecontract.memory.errors import MemorySourceError, MemoryValidationError
from forecontract.memory.models import MemoryItem, ResolvedSource
from forecontract.models.base import SCHEMA_VERSION
from forecontract.tracing import (
    AttemptFinished,
    AttemptStarted,
    ContractCommitted,
    RunFinished,
    RunStarted,
    StepStarted,
    VerificationRecorded,
    attempt_label,
    attempt_ref,
    parse_record,
    terminal_label,
)
from forecontract.tracing.errors import (
    TraceCorruptionError,
    TraceError,
    TraceNotFoundError,
)

__all__ = ["TraceSourceResolver"]

#: The logical experience key of a run-level fact is ``("run", run_id)`` and of an attempt-level
#: fact ``("attempt", run_id, step_id, contract_id, attempt_id)``.
ExperienceKey = tuple[str, ...]

_SOURCE_RECORDS: Final[tuple[type, ...]] = (RunStarted, RunFinished, AttemptStarted, AttemptFinished)


def _require_identifier(value: object, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"the {field_name} must be a non-blank string")
    return value


def experience_key(record: ResolvedSource) -> ExperienceKey:
    """Return the logical experience key of one resolved source."""
    if record.source_level == "run":
        return ("run", record.run_id)
    return ("attempt", record.run_id, record.step_id or "", record.contract_id or "", record.attempt_id or "")


class TraceSourceResolver:
    """Resolve real trace records through the verified read-only tracing reader."""

    def __init__(self, reader: object) -> None:
        getter = getattr(reader, "get_record", None)
        if not callable(getter):
            raise MemoryValidationError("the source reader does not provide get_record")
        self._reader = reader

    def resolve(self, run_id: str, source_record_id: str) -> ResolvedSource:
        """Return the mapped source of one real record, or raise a bounded source error."""
        wanted_run = _require_identifier(run_id, "run_id")
        wanted_record = _require_identifier(source_record_id, "source_record_id")
        try:
            raw = self._reader.get_record(wanted_run, wanted_record)
        except TraceNotFoundError as exc:
            raise MemorySourceError("the requested trace source does not exist", cause=exc) from exc
        except (TraceCorruptionError, TraceError) as exc:
            raise MemorySourceError("the requested trace source could not be read", cause=exc) from exc
        except OSError as exc:
            raise MemorySourceError("the requested trace source could not be read", cause=exc) from exc
        try:
            record = parse_record(raw)
        except TraceError as exc:
            raise MemorySourceError("the trace source is not a valid record", cause=exc) from exc
        except ValueError as exc:
            raise MemorySourceError("the trace source is not a valid record", cause=exc) from exc
        if not isinstance(record, _SOURCE_RECORDS):
            raise MemorySourceError("this trace record is not a memory source")
        if record.run_id != wanted_run or record.record_id != wanted_record:
            raise MemorySourceError("the trace source does not match the requested identity")
        if isinstance(record, RunStarted):
            return ResolvedSource(
                schema_version=SCHEMA_VERSION, run_id=record.run_id, record_id=record.record_id,
                source_level="run", result="unknown",
            )
        if isinstance(record, RunFinished):
            return ResolvedSource(
                schema_version=SCHEMA_VERSION, run_id=record.run_id, record_id=record.record_id,
                source_level="run", result=terminal_label(record.terminal_status),
            )
        reference = attempt_ref(record)
        if isinstance(record, AttemptStarted):
            result = "unknown"
        else:
            result = attempt_label(record.execution_status)
        return ResolvedSource(
            schema_version=SCHEMA_VERSION, run_id=record.run_id, record_id=record.record_id,
            source_level="attempt", step_id=reference.step_id, contract_id=reference.contract_id,
            attempt_id=reference.attempt_id, result=result,
        )

    def resolve_for_item(self, item: MemoryItem, *, line_number: int | None = None) -> ResolvedSource:
        """Resolve the source of a stored version and check its mapped result.

        A stored version whose source has disappeared, no longer parses, or maps to a different
        result fails explicitly. The failure names the Memory line so the stored version can be
        located without echoing any stored value.
        """
        try:
            resolved = self.resolve(item.run_id, item.source_record_id)
        except MemorySourceError as exc:
            raise MemorySourceError(
                "a stored memory version could not resolve its trace source",
                cause=exc.__cause__ if exc.__cause__ is not None else exc,
                line_number=line_number,
            ) from exc
        if resolved.result != item.result:
            raise MemorySourceError(
                "a stored memory version disagrees with its trace source", line_number=line_number
            )
        if resolved.source_level != item.source_level:
            raise MemorySourceError(
                "a stored memory version uses a different source level", line_number=line_number
            )
        return resolved


def is_memory_source(record: object) -> bool:
    """Report whether one tracing record class is an allowed Memory source."""
    return isinstance(record, (RunStarted, RunFinished, AttemptStarted, AttemptFinished))


# The rejected kinds are named here so the module documents exactly what is refused.
REJECTED_SOURCE_CLASSES: Final[tuple[type, ...]] = (
    StepStarted, ContractCommitted, VerificationRecorded,
)
