"""Host-side registration and bounded reads of real observation snapshots.

Only the host registers evidence; the model can never call :meth:`register_observation`. A
registration validates one real ``EvidenceRecorder`` entry against the same bounded raw bytes:
length, SHA-256, strict JSON, the snapshot schema, kind, Run, Observation, ordinal/call ordinal,
the association fields and the payload's equality with the actual Observation. The validated
bytes are kept as an independent copy, and every later read selects from that copy - the file is
never opened again. A change to the file after registration therefore does not change the
registered evidence, and the audit states that this is the registration-time snapshot rather
than live disk state.

Path handling is confined to an explicitly supplied root: an absolute path, a ``..`` escape, a
symlink or a Windows reparse component is refused, and ``ArtifactRef.uri`` is never used as a
read location. The reader opens each file at most once, reading at most the declared limit plus
one byte.

This mechanism does not claim to defeat a malicious concurrent OS replacement of a file while
it is being read, and it provides no OS sandbox.
"""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Final

from forecontract.models.observations import Observation
from pydantic import ValidationError

from .errors import RetrievalError
from .models import (
    REGISTERED_MAX_BYTES,
    REGISTERED_MAX_COUNT,
    RESULT_MAX_BYTES,
    SCHEMA_VERSION,
    SCOPE_MAX_BYTES,
    SNAPSHOT_MAX_BYTES,
    SNAPSHOT_MAX_VIEWS,
    EvidenceGrant,
    EvidenceReadRequest,
    EvidenceReadResult,
    EvidenceScope,
    EvidenceScopeEntry,
    RetrievalConfig,
    ScopeView,
    canonical_json_bytes,
    copy_record,
    effective_base_config,
    measure_record,
    parse_strict_json,
)

__all__ = ["RegisteredEvidenceReader"]

#: The snapshot subdirectory and file naming the existing recorder uses.
SNAPSHOT_DIR: Final[str] = "snapshots"
_REQUIRED_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {"kind", "ordinal", "path", "sha256", "bytes", "call_ordinal", "observation_id"}
)
_DOCUMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "kind", "run_id", "ordinal", "recorded_at", "associations", "payload"}
)
_ASSOCIATION_FIELDS: Final[frozenset[str]] = frozenset({"call_ordinal", "observation_id"})
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

_MISSING = object()
_INVALID = object()


def _is_strict_int(value: object) -> bool:
    return type(value) is int


def _hex_digest(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _bounded_name(value: object) -> bool:
    return type(value) is str and 0 < len(value.strip()) <= 128


def _timestamp_ok(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


@dataclass(frozen=True, slots=True)
class _View:
    view_id: str
    selector: tuple[object, ...]
    format: str


@dataclass(frozen=True, slots=True)
class _Registered:
    evidence_id: str
    observation_id: str
    raw: bytes
    payload: bytes
    sha256: str
    raw_bytes: int
    view: _View


@dataclass(frozen=True, slots=True)
class _Snapshot:
    scope_id: str
    run_id: str
    observation_id: str
    entries: tuple[_Registered, ...]


class RegisteredEvidenceReader:
    """One host-bound reader for one Run's registered observation snapshots."""

    def __init__(self, root: object, run_id: str, config: RetrievalConfig) -> None:
        if not isinstance(root, (str, os.PathLike)):
            raise RetrievalError("invalid_input", "input")
        try:
            self._root = Path(root).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            raise RetrievalError("invalid_input", "input") from None
        if not _bounded_name(run_id):
            raise RetrievalError("invalid_input", "input")
        self._run_id = run_id
        self._config = copy_record(config, RetrievalConfig, "invalid_input", "input")
        self._base = effective_base_config(self._config.base)
        self._registered: dict[str, _Registered] = {}
        self._order: list[str] = []
        self._observation_ids: set[str] = set()
        self._scopes: dict[str, _Snapshot] = {}
        self._snapshot_count = 0
        self._snapshot_bytes = 0

    # ------------------------------------------------------------------ declaration

    @property
    def root(self) -> Path:
        """The explicitly supplied containment root (no filesystem read)."""
        return self._root

    @property
    def snapshot_count(self) -> int:
        """How many snapshots this reader has registered."""
        return self._snapshot_count

    @property
    def snapshot_bytes(self) -> int:
        """How many raw bytes this reader's registered snapshots occupy."""
        return self._snapshot_bytes

    # ------------------------------------------------------------------ registration

    def register_observation(
        self,
        observation: object,
        entry: object,
        grants: object,
    ) -> None:
        """Validate one real snapshot and register its granted views atomically."""
        current = copy_record(observation, Observation, "invalid_input", "input")
        if not isinstance(grants, (list, tuple)):
            raise RetrievalError("invalid_input", "input")
        if len(grants) > SNAPSHOT_MAX_VIEWS:
            raise RetrievalError("budget_exceeded", "budget")
        declared = self._validated_grants(current, grants)
        payload, raw, sha256 = self._validate_entry(current, entry)
        views = {
            grant.view_id: _View(
                view_id=grant.view_id,
                selector=tuple(grant.selector),
                format=grant.format,
            )
            for grant in declared
        }
        committed: list[_Registered] = []
        for grant in declared:
            committed.append(
                _Registered(
                    evidence_id=grant.evidence_id,
                    observation_id=current.observation_id,
                    raw=raw,
                    payload=payload,
                    sha256=sha256,
                    raw_bytes=len(raw),
                    view=views[grant.view_id],
                )
            )
        # Commit only after every check has passed: a failed registration leaves no entry.
        for item in committed:
            self._registered[item.evidence_id] = item
            self._order.append(item.evidence_id)
        self._observation_ids.add(current.observation_id)
        self._snapshot_count += 1
        self._snapshot_bytes += len(raw)

    def _validated_grants(
        self,
        observation: Observation,
        grants: list[object] | tuple[object, ...],
    ) -> list[EvidenceGrant]:
        declared: list[EvidenceGrant] = []
        evidence_ids: set[str] = set()
        view_ids: set[str] = set()
        for item in grants:
            grant = copy_record(item, EvidenceGrant, "invalid_input", "input")
            if grant.run_id != self._run_id:
                raise RetrievalError("binding_mismatch", "binding")
            if grant.observation_id != observation.observation_id:
                raise RetrievalError("binding_mismatch", "binding")
            if grant.evidence_id in evidence_ids or grant.evidence_id in self._registered:
                raise RetrievalError("invalid_input", "input")
            if grant.view_id in view_ids:
                raise RetrievalError("invalid_input", "input")
            evidence_ids.add(grant.evidence_id)
            view_ids.add(grant.view_id)
            declared.append(grant)
        return declared

    def _validate_entry(
        self,
        observation: Observation,
        entry: object,
    ) -> tuple[bytes, bytes, str]:
        """Return ``(payload_json, raw, sha256)`` for one validated snapshot entry."""
        if not isinstance(entry, dict):
            raise RetrievalError("invalid_input", "input")
        if not _REQUIRED_ENTRY_FIELDS <= set(entry):
            raise RetrievalError("invalid_input", "input")
        if entry.get("kind") != "observation":
            raise RetrievalError("source_invalid", "source")
        ordinal = entry.get("ordinal")
        call_ordinal = entry.get("call_ordinal")
        size = entry.get("bytes")
        if not _is_strict_int(ordinal) or ordinal < 1:
            raise RetrievalError("source_invalid", "source")
        if not _is_strict_int(call_ordinal) or call_ordinal < 1:
            raise RetrievalError("source_invalid", "source")
        if not _is_strict_int(size) or size < 1:
            raise RetrievalError("source_invalid", "source")
        if size > SNAPSHOT_MAX_BYTES:
            raise RetrievalError("budget_exceeded", "source")
        if (self._snapshot_count + 1 > REGISTERED_MAX_COUNT
                or self._snapshot_bytes + size > REGISTERED_MAX_BYTES):
            raise RetrievalError("budget_exceeded", "budget")
        relative = self._relative_member(entry.get("path"), ordinal)
        entry_observation_id = entry.get("observation_id")
        if not _bounded_name(entry_observation_id):
            raise RetrievalError("source_invalid", "source")
        if entry_observation_id != observation.observation_id:
            raise RetrievalError("source_invalid", "source")
        target = self._resolve_member(relative)
        raw = self._read_bounded(target)
        declared_sha = entry.get("sha256")
        if not _hex_digest(declared_sha):
            raise RetrievalError("source_invalid", "source")
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != declared_sha:
            raise RetrievalError("source_invalid", "source")
        document = parse_strict_json(
            raw,
            max_depth=self._base.max_json_depth,
            max_nodes=self._base.max_json_nodes,
            code="source_invalid",
            stage="source",
        )
        if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
            raise RetrievalError("source_invalid", "source")
        if document.get("schema_version") != SCHEMA_VERSION:
            raise RetrievalError("source_invalid", "source")
        if document.get("kind") != "observation":
            raise RetrievalError("source_invalid", "source")
        if document.get("run_id") != self._run_id:
            raise RetrievalError("source_invalid", "source")
        if document.get("ordinal") != ordinal or not _is_strict_int(document.get("ordinal")):
            raise RetrievalError("source_invalid", "source")
        if not _timestamp_ok(document.get("recorded_at")):
            raise RetrievalError("source_invalid", "source")
        associations = document.get("associations")
        if not isinstance(associations, dict) or set(associations) != _ASSOCIATION_FIELDS:
            raise RetrievalError("source_invalid", "source")
        if (associations.get("call_ordinal") != call_ordinal
                or not _is_strict_int(associations.get("call_ordinal"))):
            raise RetrievalError("source_invalid", "source")
        if associations.get("observation_id") != entry_observation_id:
            raise RetrievalError("source_invalid", "source")
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise RetrievalError("source_invalid", "source")
        try:
            snapshot_observation = Observation.model_validate(payload)
        except (ValidationError, TypeError, ValueError):
            raise RetrievalError("source_invalid", "source") from None
        try:
            expected = observation.model_dump(mode="json")
        except (TypeError, ValueError, RecursionError):
            raise RetrievalError("invalid_input", "input") from None
        if snapshot_observation.model_dump(mode="json") != expected or payload != expected:
            raise RetrievalError("source_invalid", "source")
        try:
            payload_json = canonical_json_bytes(payload)
        except RetrievalError:
            raise RetrievalError("source_invalid", "source") from None
        return payload_json, raw, hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------------ scopes

    def open_scope(self, run_id: object, observation_id: object) -> EvidenceScope:
        """Freeze and return the directory for one Run and its current Observation."""
        if not _bounded_name(run_id) or not _bounded_name(observation_id):
            raise RetrievalError("invalid_input", "input")
        if run_id != self._run_id:
            raise RetrievalError("binding_mismatch", "binding")
        if observation_id not in self._observation_ids:
            raise RetrievalError("access_denied", "scope")
        entries: list[EvidenceScopeEntry] = []
        for evidence_id in self._order:
            registered = self._registered[evidence_id]
            entries.append(
                EvidenceScopeEntry(
                    schema_version=SCHEMA_VERSION,
                    evidence_id=registered.evidence_id,
                    source_observation_id=registered.observation_id,
                    kind="observation",
                    raw_sha256=registered.sha256,
                    raw_bytes=registered.raw_bytes,
                    views=[
                        ScopeView(
                            schema_version=SCHEMA_VERSION,
                            view_id=registered.view.view_id,
                            selector=list(registered.view.selector),
                            format=registered.view.format,
                        )
                    ],
                )
            )
        scope = EvidenceScope(
            schema_version=SCHEMA_VERSION,
            scope_id=str(uuid.uuid4()),
            run_id=self._run_id,
            observation_id=observation_id,
            entries=entries,
        )
        measure_record(
            scope,
            max_bytes=SCOPE_MAX_BYTES,
            max_depth=self._base.max_json_depth,
            max_nodes=self._base.max_json_nodes,
            code="budget_exceeded",
            stage="scope",
        )
        self._scopes[scope.scope_id] = _Snapshot(
            scope_id=scope.scope_id,
            run_id=self._run_id,
            observation_id=observation_id,
            entries=tuple(
                self._registered[evidence_id] for evidence_id in self._order
            ),
        )
        return scope

    def close_scope(self, scope_id: object) -> None:
        """Stop publishing results for this scope; idempotent, synchronous, no I/O."""
        if type(scope_id) is not str:
            return
        self._scopes.pop(scope_id, None)

    # ------------------------------------------------------------------ reads

    async def read(self, request: object) -> EvidenceReadResult:
        """Return one measured read of one authorized view from the registered bytes."""
        current = copy_record(request, EvidenceReadRequest, "invalid_input", "input")
        scope = self._scopes.get(current.scope_id)
        if scope is None:
            raise RetrievalError("invalid_state", "scope", request_id=current.request_id)
        if current.run_id != self._run_id or current.run_id != scope.run_id:
            raise RetrievalError("binding_mismatch", "binding", request_id=current.request_id)
        registered = next(
            (item for item in scope.entries if item.evidence_id == current.evidence_id), None)
        if registered is None:
            raise RetrievalError("access_denied", "grant",
                                 request_id=current.request_id)
        view = registered.view
        if view.view_id != current.view_id:
            raise RetrievalError("access_denied", "grant",
                                 request_id=current.request_id)
        if current.text_range is not None and view.format == "json":
            raise RetrievalError("selection_invalid", "selection",
                                 request_id=current.request_id)
        payload = parse_strict_json(
            registered.payload,
            max_depth=self._base.max_json_depth,
            max_nodes=self._base.max_json_nodes,
            code="source_invalid",
            stage="source",
        )
        if not isinstance(payload, dict):
            raise RetrievalError("source_invalid", "source", request_id=current.request_id)
        selected = _walk(payload, view.selector)
        if selected is _INVALID:
            raise RetrievalError("selection_invalid", "selection",
                                 request_id=current.request_id)
        if selected is _MISSING:
            return self._not_found(current, registered)
        if view.format == "text":
            if not isinstance(selected, str):
                raise RetrievalError("selection_invalid", "selection",
                                     request_id=current.request_id)
            if current.text_range is None:
                start, end = 0, len(selected)
                partial = False
            else:
                start, end = current.text_range.start, current.text_range.end
                if end > len(selected):
                    raise RetrievalError("selection_invalid", "selection",
                                         request_id=current.request_id)
                partial = (start, end) != (0, len(selected))
            value: object = selected[start:end]
        else:
            value = selected
            partial = False
        measured = canonical_json_bytes(value)
        result = EvidenceReadResult(
            schema_version=SCHEMA_VERSION,
            scope_id=scope.scope_id,
            run_id=current.run_id,
            plan_id=current.plan_id,
            turn=current.turn,
            request_id=current.request_id,
            evidence_id=current.evidence_id,
            view_id=current.view_id,
            status="found",
            source_observation_id=registered.observation_id,
            raw_sha256=registered.sha256,
            raw_bytes=registered.raw_bytes,
            selector=list(view.selector),
            format=view.format,
            text_range=current.text_range,
            partial=partial,
            value=value,
            value_sha256=hashlib.sha256(measured).hexdigest(),
            value_bytes=len(measured),
        )
        return self._measured_result(result, current)

    def _not_found(
        self,
        request: EvidenceReadRequest,
        registered: _Registered,
    ) -> EvidenceReadResult:
        result = EvidenceReadResult(
            schema_version=SCHEMA_VERSION,
            scope_id=request.scope_id,
            run_id=request.run_id,
            plan_id=request.plan_id,
            turn=request.turn,
            request_id=request.request_id,
            evidence_id=request.evidence_id,
            view_id=request.view_id,
            status="not_found",
            source_observation_id=registered.observation_id,
            raw_sha256=registered.sha256,
            raw_bytes=registered.raw_bytes,
            selector=list(registered.view.selector),
            format=registered.view.format,
            text_range=request.text_range,
            partial=False,
            value=None,
            value_sha256=None,
            value_bytes=0,
        )
        return self._measured_result(result, request)

    def _measured_result(
        self,
        result: EvidenceReadResult,
        request: EvidenceReadRequest,
    ) -> EvidenceReadResult:
        """Enforce the single-result budget and refuse to publish a late result."""
        measure_record(
            result,
            max_bytes=RESULT_MAX_BYTES,
            max_depth=self._base.max_json_depth,
            max_nodes=self._base.max_json_nodes,
            code="budget_exceeded",
            stage="budget",
        )
        if request.scope_id not in self._scopes:
            raise RetrievalError("invalid_state", "scope", request_id=request.request_id)
        return result

    # ------------------------------------------------------------------ path containment

    def _relative_member(self, raw: object, ordinal: int) -> PurePosixPath:
        """Refuse anything but the declared snapshot member path of this entry."""
        if type(raw) is not str or not raw or "\\" in raw:
            raise RetrievalError("source_invalid", "source")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise RetrievalError("source_invalid", "source")
        parts = relative.parts
        if len(parts) != 2 or parts[0] != SNAPSHOT_DIR:
            raise RetrievalError("source_invalid", "source")
        if parts[1] != f"observation-{ordinal:04d}.json":
            raise RetrievalError("source_invalid", "source")
        return relative

    def _resolve_member(self, relative: PurePosixPath) -> Path:
        """Resolve the member under the explicit root, refusing an escape or a reparse."""
        candidate = self._root / Path(*relative.parts)
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError):
            raise RetrievalError("source_invalid", "source") from None
        if resolved != self._root and self._root not in resolved.parents:
            raise RetrievalError("source_invalid", "source")
        current = self._root
        for part in relative.parts:
            current = current / part
            try:
                info = os.lstat(current)
            except OSError:
                raise RetrievalError("source_invalid", "source") from None
            if stat.S_ISLNK(info.st_mode):
                raise RetrievalError("source_invalid", "source")
            if getattr(info, "st_file_attributes", 0) & _REPARSE:
                raise RetrievalError("source_invalid", "source")
        return resolved

    @staticmethod
    def _read_bounded(path: Path) -> bytes:
        """Read at most the declared per-snapshot limit plus one byte."""
        try:
            with path.open("rb") as handle:
                raw = handle.read(SNAPSHOT_MAX_BYTES + 1)
        except OSError:
            raise RetrievalError("source_invalid", "source") from None
        if len(raw) > SNAPSHOT_MAX_BYTES:
            raise RetrievalError("budget_exceeded", "source")
        return raw


def _walk(payload: object, selector: tuple[object, ...]) -> object:
    """Walk one selector from the snapshot payload; report a missing or an illegal path."""
    node = payload
    for part in selector:
        if isinstance(part, str):
            if not isinstance(node, dict):
                return _INVALID
            if part not in node:
                return _MISSING
            node = node[part]
        else:
            if not isinstance(node, list):
                return _INVALID
            index = int(part)
            if index >= len(node):
                return _MISSING
            node = node[index]
    return node
