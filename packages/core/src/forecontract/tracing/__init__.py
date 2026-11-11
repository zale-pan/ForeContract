"""Structured trajectory and explicit terminal persistence.

Import the tracing surface from ``forecontract.tracing``. The subpackage is additive: it does
not change the ``forecontract`` root exports, ``forecontract.models``, ``forecontract.protocols``
or ``forecontract.contracts``, and it adds no dependency beyond the existing Core runtime
dependency.

What this package is: a versioned record contract for one run's trajectory, two narrow
persistence protocols, one single-writer local JSONL store and explicit bounded errors.

What it is not: a Runtime, a scheduler, a verification algorithm, a retry or replan policy, a
Memory schema, a crash-recovery mechanism or a secret scanner. It also claims nothing about
whether a task succeeded; it records the facts a later consumer uses to decide that.
"""

from __future__ import annotations

from forecontract.tracing.errors import (
    TRACE_ERROR_CODES,
    TraceCorruptionError,
    TraceError,
    TraceNotFoundError,
    TraceStorageError,
    TraceValidationError,
)
from forecontract.tracing.jsonl import JsonlTraceStore
from forecontract.tracing.models import (
    ATTEMPT_RESULT_LABELS,
    ATTEMPT_SOURCE_LEVEL,
    CLOSED_CODES,
    EVENT_KINDS,
    EVENT_MODELS,
    RUN_SOURCE_LEVEL,
    SOURCE_LEVELS,
    TERMINAL_RESULT_LABELS,
    TERMINAL_STATUSES,
    AttemptFinished,
    AttemptRef,
    AttemptStarted,
    ContractCommitted,
    Event,
    RunFinished,
    RunStarted,
    StepStarted,
    TraceRecord,
    VerificationRecorded,
    attempt_label,
    attempt_ref,
    parse_record,
    source_location,
    terminal_label,
)
from forecontract.tracing.protocols import TraceReader, TraceWriter

__all__ = [
    "ATTEMPT_RESULT_LABELS",
    "ATTEMPT_SOURCE_LEVEL",
    "CLOSED_CODES",
    "EVENT_KINDS",
    "EVENT_MODELS",
    "RUN_SOURCE_LEVEL",
    "SOURCE_LEVELS",
    "TERMINAL_RESULT_LABELS",
    "TERMINAL_STATUSES",
    "TRACE_ERROR_CODES",
    "AttemptFinished",
    "AttemptRef",
    "AttemptStarted",
    "ContractCommitted",
    "Event",
    "JsonlTraceStore",
    "RunFinished",
    "RunStarted",
    "StepStarted",
    "TraceCorruptionError",
    "TraceError",
    "TraceNotFoundError",
    "TraceReader",
    "TraceRecord",
    "TraceStorageError",
    "TraceValidationError",
    "TraceWriter",
    "VerificationRecorded",
    "attempt_label",
    "attempt_ref",
    "parse_record",
    "source_location",
    "terminal_label",
]
