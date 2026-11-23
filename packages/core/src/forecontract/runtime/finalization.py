"""The bounded finalization snapshot: what actually happened at the end of a Run.

A Run can end in more than one way at once. The business outcome may be a success, a facility
failure, a user cancellation, a deadline expiry or a policy abort; the cleanup may or may not have
run, and may or may not have finished inside its allowance; the single terminal fact may have been
persisted, may have failed, or may be genuinely unknown. Collapsing all of that into one status would
lose the distinction a host needs, so this record states each part separately and lets the
cross-field rules keep them consistent.

Nothing here is a channel for data. There is no observation, no payload, no exception text and no
free-form string: every field is a boolean, a closed-vocabulary name or an opaque identifier.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, model_validator

from forecontract.models.base import (
    SCHEMA_VERSION,
    NonBlankStr,
    SchemaVersion,
    StrictBool,
)
from forecontract.runtime.models import (
    PHASES,
    TERMINAL_PHASES,
    RuntimeReason,
    RuntimeRecord,
    RuntimeTerminalStatus,
)

__all__ = [
    "CLEANUP_STATUSES",
    "TRACE_STATUSES",
    "RuntimeFinalization",
]

#: The closed set of cleanup outcomes.
CLEANUP_STATUSES: tuple[str, ...] = (
    "not_required",
    "succeeded",
    "failed",
    "timed_out",
    "skipped_pending_operation",
)

#: The closed set of terminal-persistence outcomes. ``unknown`` means the write was attempted and
#: its result is genuinely undetermined - it is never a synonym for "not written".
TRACE_STATUSES: tuple[str, ...] = ("not_attempted", "persisted", "unknown")


def _closed_trace(value: object) -> object:
    """Accept only a member of the closed trace-status vocabulary."""
    if isinstance(value, str) and value in TRACE_STATUSES:
        return value
    raise ValueError("the trace status is outside the closed vocabulary")


#: A closed cleanup-status name. The literal itself is the whole declaration: an unknown name is not
#: a member of it, so no additional strictness is required and none can be applied to a literal.
CleanupField = Literal[CLEANUP_STATUSES]

#: A closed trace-status name.
TraceField = Literal[TRACE_STATUSES]


class RuntimeFinalization(RuntimeRecord):
    """A read-only diagnostic snapshot of how one Run actually ended.

    ``primary_*`` states the first non-success diagnosis the Run accepted, or the candidate success
    if none arrived. The cleanup, trace and terminal fields then state what happened *after* that, so
    a late cancellation, a failed cleanup or an uncertain write is visible without overwriting the
    primary diagnosis.

    The record is frozen. A snapshot is a statement about something that already happened, so it is
    not a surface a caller may edit, and the engine hands out isolated copies as well: neither route
    can change the diagnosis the Runtime recorded.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              revalidate_instances="always", frozen=True)

    schema_version: SchemaVersion
    run_id: NonBlankStr
    primary_terminal_status: RuntimeTerminalStatus
    primary_reason_code: RuntimeReason
    pending_operation_at_return: StrictBool
    cleanup_status: CleanupField
    cleanup_error_code: RuntimeReason | None = None
    trace_status: TraceField
    trace_error_code: RuntimeReason | None = None
    terminal_record_id: NonBlankStr | None = None
    finalize_deadline_exceeded: StrictBool
    quarantined: StrictBool

    @model_validator(mode="after")
    def _the_parts_agree(self) -> RuntimeFinalization:
        """Refuse a snapshot whose parts contradict each other.

        Each rule below is a statement the snapshot would otherwise be making falsely: that a
        terminal fact exists although no successful write was seen, that a cleanup problem is absent
        although cleanup failed, or that a resource is safe although an operation was still pending.
        """
        if self.primary_terminal_status not in (RuntimeTerminalStatus.COMPLETED,
                                                RuntimeTerminalStatus.FAILED,
                                                RuntimeTerminalStatus.ABORTED,
                                                RuntimeTerminalStatus.CANCELLED,
                                                RuntimeTerminalStatus.TIMEOUT):
            raise ValueError("the primary status is outside the closed terminal vocabulary")
        # A persisted terminal fact is only claimed when the real append returned an identifier, and
        # an identifier may only be named when the write really was confirmed.
        if self.trace_status == "persisted":
            if self.terminal_record_id is None:
                raise ValueError("a persisted terminal fact must name the record that was written")
        elif self.terminal_record_id is not None:
            raise ValueError("only a persisted terminal fact may name a record identifier")
        if self.cleanup_status in ("failed", "timed_out", "skipped_pending_operation"):
            if self.cleanup_error_code is not RuntimeReason.CLEANUP_FAILURE:
                raise ValueError("an unsuccessful cleanup must report the cleanup failure code")
        elif self.cleanup_error_code is not None:
            raise ValueError("a successful or unneeded cleanup must not report a failure code")
        if self.pending_operation_at_return and not self.quarantined:
            raise ValueError("a pending operation at return must be quarantined")
        if self.trace_error_code is RuntimeReason.TRACE_FAILURE:
            # A trace failure means an attempted write whose result is undetermined, an attempt that
            # never started, or a write that really landed *after* its allowance elapsed. The overdue
            # shape is the approved one: it is a persisted fact that is also reported as a failure,
            # because the allowance was exceeded and the caller must see that.
            if self.trace_status == "persisted":
                if not self.finalize_deadline_exceeded:
                    raise ValueError("a persisted trace failure must be the overdue-write shape")
            elif self.trace_status not in ("unknown", "not_attempted"):
                raise ValueError("a trace failure is an uncertain or unstarted write")
        elif self.trace_error_code is not None:
            raise ValueError("the trace error code is outside the closed vocabulary")
        if (self.primary_terminal_status is RuntimeTerminalStatus.COMPLETED
                and self.cleanup_status not in ("not_required", "succeeded")):
            raise ValueError("a completed run must have succeeded or needed no cleanup")
        if (self.primary_terminal_status is RuntimeTerminalStatus.TIMEOUT
                and self.primary_reason_code not in (RuntimeReason.DEADLINE_EXCEEDED,
                                                     RuntimeReason.OPERATION_TIMEOUT)):
            raise ValueError("a timeout must report a timeout reason")
        if (self.primary_terminal_status is RuntimeTerminalStatus.CANCELLED
                and self.primary_reason_code is not RuntimeReason.USER_CANCELLED):
            raise ValueError("a cancelled run must report the user cancellation reason")
        return self

    @property
    def is_success(self) -> bool:
        """Whether the primary outcome was a genuine task success."""
        return self.primary_terminal_status is RuntimeTerminalStatus.COMPLETED

    @property
    def phases_considered(self) -> frozenset[str]:
        """The declared phase vocabulary this snapshot was built against, for diagnostics."""
        return frozenset(PHASES) | frozenset(TERMINAL_PHASES)

    def isolation_copy(self) -> RuntimeFinalization:
        """A detached copy, so a reader cannot mutate the snapshot the engine keeps."""
        from forecontract.runtime.models import isolated_record, stored_payload

        return isolated_record(RuntimeFinalization, stored_payload(self), validate_fully=True)


def build_finalization(*, run_id: str,
                       primary_terminal_status: RuntimeTerminalStatus,
                       primary_reason_code: RuntimeReason,
                       pending_operation_at_return: bool,
                       cleanup_status: str,
                       trace_status: str,
                       finalize_deadline_exceeded: bool,
                       quarantined: bool,
                       cleanup_error_code: RuntimeReason | None = None,
                       trace_error_code: RuntimeReason | None = None,
                       terminal_record_id: str | None = None) -> RuntimeFinalization:
    """Construct a finalization snapshot through its declared schema.

    The single construction point keeps every caller on the same cross-field rules, so a snapshot
    cannot be assembled that claims a persisted write it did not make or a clean shutdown it did not
    finish.
    """
    return RuntimeFinalization(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        primary_terminal_status=primary_terminal_status,
        primary_reason_code=primary_reason_code,
        pending_operation_at_return=pending_operation_at_return,
        cleanup_status=cleanup_status,
        cleanup_error_code=cleanup_error_code,
        trace_status=trace_status,
        trace_error_code=trace_error_code,
        terminal_record_id=terminal_record_id,
        finalize_deadline_exceeded=finalize_deadline_exceeded,
        quarantined=quarantined,
    )
