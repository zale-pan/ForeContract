"""Public Memory surface.

Import the Memory contract from ``forecontract.memory``. The subpackage is additive: it does not
change the ``forecontract`` root exports, ``forecontract.models``, ``forecontract.protocols``,
``forecontract.contracts`` or ``forecontract.tracing``, and it adds no dependency beyond the
existing Core runtime dependency.

What it is: one strict in-process Working state per Run, a closed versioned Memory record, a
resolver that maps real trace records to Memory results, and a single-writer local JSONL
Episodic store with bounded errors.

What it is not: an embedding store, a vector index, an LLM summariser, a semantic search, a
scheduler, an automatic lifecycle, a crash-recovery mechanism or a secret scanner. It records
and indexes facts that already exist in the trace; it never decides that a task succeeded.
"""

from __future__ import annotations

from forecontract.memory.episodic import JsonlEpisodicMemory
from forecontract.memory.errors import (
    MEMORY_ERROR_CODES,
    MemoryCorruptionError,
    MemoryError,
    MemoryNotFoundError,
    MemorySourceError,
    MemoryStorageError,
    MemoryValidationError,
)
from forecontract.memory.models import (
    MEMORY_RESULTS,
    SOURCE_LEVELS,
    MemoryItem,
    ResolvedSource,
)
from forecontract.memory.sources import TraceSourceResolver
from forecontract.memory.working import WorkingMemory

__all__ = [
    "MEMORY_ERROR_CODES",
    "MEMORY_RESULTS",
    "SOURCE_LEVELS",
    "JsonlEpisodicMemory",
    "MemoryCorruptionError",
    "MemoryError",
    "MemoryItem",
    "MemoryNotFoundError",
    "MemorySourceError",
    "MemoryStorageError",
    "MemoryValidationError",
    "ResolvedSource",
    "TraceSourceResolver",
    "WorkingMemory",
]
