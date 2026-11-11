"""Public surface of the controlled evidence retrieval subpackage.

The package exposes exactly the thirteen agreed entry points: the planner, its configuration and
request record, the three injected protocols, the evidence records, the host-side registered
reader and the audit record. Every record and error is strict and value-free; nothing here opens
a network connection, reads a credential or writes a file.
"""

from __future__ import annotations

from .audit import RetrievalAuditEvent
from .errors import RetrievalError
from .models import (
    EvidenceGrant,
    EvidenceReadRequest,
    EvidenceReadResult,
    EvidenceScope,
    RetrievalConfig,
    RetrievalModelRequest,
)
from .planner import EvidenceLLMPlanner
from .protocols import EvidenceReader, RetrievalAuditSink, RetrievalProvider
from .reader import RegisteredEvidenceReader

__all__ = [
    "EvidenceGrant",
    "EvidenceLLMPlanner",
    "EvidenceReadRequest",
    "EvidenceReadResult",
    "EvidenceReader",
    "EvidenceScope",
    "RegisteredEvidenceReader",
    "RetrievalAuditEvent",
    "RetrievalAuditSink",
    "RetrievalConfig",
    "RetrievalError",
    "RetrievalModelRequest",
    "RetrievalProvider",
]
