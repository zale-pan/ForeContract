"""The three injected interfaces of controlled evidence retrieval.

The planner depends on these structural protocols only. It never imports a concrete provider,
reader or sink, and no implementation here performs I/O. Each protocol states the calling
convention the planner relies on; it does not certify that an implementation is honest.
"""

from __future__ import annotations

from typing import Protocol

from forecontract_llm.models import ModelResponse

from .audit import RetrievalAuditEvent
from .models import (
    EvidenceReadRequest,
    EvidenceReadResult,
    EvidenceScope,
    RetrievalModelRequest,
)

__all__ = ["EvidenceReader", "RetrievalAuditSink", "RetrievalProvider"]


class RetrievalProvider(Protocol):
    """The injected retrieval transport: one complete request in, one response out.

    A concrete provider must be a cooperative, non-blocking coroutine and must bind its
    response to ``request.request_id``. The planner owns the single call task and the bounded
    deadline; the provider owns no retry, no JSON repair and no hidden cache.
    """

    async def complete(self, request: RetrievalModelRequest) -> ModelResponse:
        """Return one response bound to this request, or raise."""
        ...


class EvidenceReader(Protocol):
    """The host-side evidence boundary: a frozen scope, bounded reads and idempotent close.

    ``open_scope`` is a synchronous snapshot of what the host registered for the current Run
    and Observation. ``read`` is asynchronous and returns one measured result for exactly the
    requested binding. ``close_scope`` is synchronous, idempotent and performs no I/O.
    """

    def open_scope(self, run_id: str, observation_id: str) -> EvidenceScope:
        """Freeze and return the directory for one Run and one current Observation."""
        ...

    async def read(self, request: EvidenceReadRequest) -> EvidenceReadResult:
        """Return one measured read result, or refuse the read."""
        ...

    def close_scope(self, scope_id: str) -> None:
        """Stop publishing results for this scope; never await and never perform I/O."""
        ...


class RetrievalAuditSink(Protocol):
    """The host-injected persistence boundary for the bypass audit trail.

    A successful ``append`` means the host has committed to keeping the record. An in-memory
    list does not satisfy the formal demo/acceptance persistence requirement.
    """

    async def append(self, event: RetrievalAuditEvent) -> None:
        """Persist exactly one audit event, or raise."""
        ...
