"""The bypass audit record and its canonical encoding.

The audit sink is a host-injected persistence boundary. Every event this module builds carries
its own canonical payload bytes and SHA-256, and the record re-measures that binding on
construction, so a caller cannot claim one payload while persisting another. Event kinds are a
closed vocabulary and the sequence is strictly increasing within one planner instance.

A payload never carries an API key, a credential, a local filesystem path or a facility
exception string; the planner only stores the protocol records it really built or received.
"""

from __future__ import annotations

import hashlib
from typing import Final, Literal

from forecontract.models.json_types import JsonObject
from pydantic import model_validator

from .errors import RetrievalError
from .models import (
    SCHEMA_VERSION,
    HexDigest,
    NonBlankStrict,
    NonNegativeInt,
    PositiveInt,
    SchemaVersion,
    _StrictRecord,
    canonical_json_bytes,
)

__all__ = [
    "AUDIT_EVENT_KINDS",
    "RetrievalAuditEvent",
    "build_event",
]

#: The closed event-kind vocabulary (C07).
AUDIT_EVENT_KINDS: Final[tuple[str, ...]] = (
    "plan_started",
    "request_prepared",
    "provider_returned",
    "read_requested",
    "read_returned",
    "forecast_ready",
    "plan_failed",
    "plan_cancelled",
    "plan_closed",
)

#: The kinds that must name the request they belong to.
_REQUEST_KINDS: Final[frozenset[str]] = frozenset(
    {"request_prepared", "provider_returned", "read_requested", "read_returned",
     "forecast_ready"}
)


class RetrievalAuditEvent(_StrictRecord):
    """One persisted audit fact, bound to its own canonical payload bytes."""

    model_config = _StrictRecord.model_config

    schema_version: SchemaVersion
    run_id: NonBlankStrict
    plan_id: NonBlankStrict
    sequence: PositiveInt
    kind: Literal[
        "plan_started",
        "request_prepared",
        "provider_returned",
        "read_requested",
        "read_returned",
        "forecast_ready",
        "plan_failed",
        "plan_cancelled",
        "plan_closed",
    ]
    request_id: NonBlankStrict | None = None
    turn: PositiveInt | None = None
    scope_id: NonBlankStrict | None = None
    payload: JsonObject
    payload_sha256: HexDigest
    payload_bytes: NonNegativeInt

    @model_validator(mode="after")
    def _payload_is_bound(self) -> RetrievalAuditEvent:
        raw = canonical_json_bytes(self.payload)
        if len(raw) != self.payload_bytes or _digest(raw) != self.payload_sha256:
            raise ValueError("an audit event must carry its own canonical payload bytes")
        if self.kind in _REQUEST_KINDS and self.request_id is None:
            raise ValueError("this audit kind requires a request identifier")
        if self.turn is not None and self.request_id is None:
            raise ValueError("an audit turn requires a request identifier")
        return self


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_event(
    *,
    run_id: str,
    plan_id: str,
    sequence: int,
    kind: str,
    payload: object,
    request_id: str | None = None,
    turn: int | None = None,
    scope_id: str | None = None,
) -> RetrievalAuditEvent:
    """Build one event whose declared digest and byte count cover exactly its payload."""
    if not isinstance(payload, dict):
        raise RetrievalError("invalid_input", "audit")
    try:
        raw = canonical_json_bytes(payload)
    except RetrievalError:
        raise RetrievalError("invalid_input", "audit") from None
    return RetrievalAuditEvent(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        plan_id=plan_id,
        sequence=sequence,
        kind=kind,
        request_id=request_id,
        turn=turn,
        scope_id=scope_id,
        payload=payload,
        payload_sha256=_digest(raw),
        payload_bytes=len(raw),
    )
