"""Strict, versioned records and shared encoding rules for controlled evidence retrieval.

Every record requires an explicit ``schema_version="0.1"``, forbids unknown fields and
validates assignment transactionally: a refused assignment restores the previous stored state,
so no half-valid record is left behind. Integers are strict (a boolean is never an integer, a
float is never an integer, a numeric string is never a number). Records own their nested state:
construction revalidates every nested record from its actual ``__dict__`` storage and replaces
it with a detached copy, and every public boundary revalidates a supplied record from that same
declared storage - never from an overridable ``model_dump`` or a caller-supplied ``__dict__``
view.

The canonical JSON rule of this round is UTF-8, ``ensure_ascii=False``, ``sort_keys=True``,
compact separators, ``allow_nan=False`` and no trailing newline. It is the measurement rule of
this protocol only; it does not claim to be the event-protocol RFC 8785 form.

This module performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, ClassVar, Final, Literal, Self, TypeVar

from forecontract.models.json_types import JsonObject, JsonValue
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from forecontract_llm.errors import PlannerError
from forecontract_llm.models import (
    ModelRequest,
    PlannerConfig,
    measure_json,
    read_declared_copy,
)

from .errors import RetrievalError

__all__ = [
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_MAX_READS",
    "JSON_DEPTH_MAX",
    "JSON_NODES_MAX",
    "OUTPUT_TOKENS_MAX",
    "PLAN_MAX_MODEL_CALLS",
    "PLAN_MAX_READS",
    "PLAN_REQUEST_MAX_BYTES",
    "PLAN_RESPONSE_MAX_BYTES",
    "PLAN_RESULT_MAX_BYTES",
    "REGISTERED_MAX_BYTES",
    "REGISTERED_MAX_COUNT",
    "REQUEST_MAX_BYTES",
    "RESPONSE_MAX_BYTES",
    "RESULT_MAX_BYTES",
    "SCOPE_MAX_BYTES",
    "SNAPSHOT_MAX_BYTES",
    "SNAPSHOT_MAX_VIEWS",
    "EvidenceGrant",
    "EvidenceReadRequest",
    "EvidenceReadResult",
    "EvidenceScope",
    "EvidenceScopeEntry",
    "RetrievalConfig",
    "RetrievalModelRequest",
    "ScopeView",
    "TextRange",
    "canonical_json_bytes",
    "canonical_record_bytes",
    "copy_record",
    "effective_base_config",
    "measure_record",
    "parse_strict_json",
]

SCHEMA_VERSION: Final[str] = "0.1"
SchemaVersion = Literal["0.1"]

#: C05 fixed measurement limits. The configurable retrieval fields can only narrow them.
DEFAULT_MAX_READS: Final[int] = 2
DEFAULT_MAX_MODEL_CALLS: Final[int] = 3
PLAN_MAX_READS: Final[int] = 2
PLAN_MAX_MODEL_CALLS: Final[int] = 3
SNAPSHOT_MAX_BYTES: Final[int] = 1_048_576
REGISTERED_MAX_BYTES: Final[int] = 8_388_608
REGISTERED_MAX_COUNT: Final[int] = 64
SNAPSHOT_MAX_VIEWS: Final[int] = 8
SCOPE_MAX_BYTES: Final[int] = 32_768
RESULT_MAX_BYTES: Final[int] = 65_536
PLAN_RESULT_MAX_BYTES: Final[int] = 131_072
REQUEST_MAX_BYTES: Final[int] = 262_144
PLAN_REQUEST_MAX_BYTES: Final[int] = 786_432
RESPONSE_MAX_BYTES: Final[int] = 65_536
PLAN_RESPONSE_MAX_BYTES: Final[int] = 196_608
OUTPUT_TOKENS_MAX: Final[int] = 2_048
JSON_DEPTH_MAX: Final[int] = 32
JSON_NODES_MAX: Final[int] = 10_000

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SELECTOR_SEGMENTS_MAX: Final[int] = 32
_SELECTOR_TEXT_MAX: Final[int] = 128
_VIEW_FORMATS: Final[frozenset[str]] = frozenset({"json", "text"})


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain a non-whitespace character")
    return value


def _identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("must be 1-128 ASCII letters, digits, underscores or hyphens")
    return value


StrictStr = Annotated[str, Field(strict=True)]
NonBlankStrict = Annotated[str, Field(strict=True), AfterValidator(_non_blank)]
StrictBool = Annotated[bool, Field(strict=True)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
HexDigest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(strict=True), AfterValidator(_identifier)]
SelectorSegment = NonNegativeInt | StrictStr

_MODEL_CONFIG = ConfigDict(
    extra="forbid", validate_assignment=True, hide_input_in_errors=True
)

RecordT = TypeVar("RecordT", bound=BaseModel)

#: The declared base-class ``__dict__`` descriptor, so a subclass property cannot hide storage.
_DECLARED_STORAGE = BaseModel.__dict__.get("__dict__")


def _actual_storage(record: BaseModel) -> dict[str, object]:
    """Read the record's real instance mapping, never an overridable view."""
    if _DECLARED_STORAGE is None:
        raise ValueError("the supplied record cannot be read as declared fields")
    try:
        stored = _DECLARED_STORAGE.__get__(record, type(record))
    except (AttributeError, TypeError):
        raise ValueError("the supplied record cannot be read as declared fields") from None
    if not isinstance(stored, dict):
        raise TypeError("the supplied record cannot be read as declared fields")
    return stored


def _owned(value: object, declared: type[BaseModel] | None,
           declared_list: type[BaseModel] | None) -> object:
    """Return a detached copy of one stored value; revalidate a declared nested record."""
    if isinstance(value, BaseModel):
        if declared is None:
            raise ValueError("the field carries an undeclared nested record")
        try:
            return read_declared_copy(value, declared, "INPUT_INVALID")
        except PlannerError:
            raise ValueError("the nested record does not match its declared type") from None
    if isinstance(value, dict):
        return {key: _owned(item, None, None) for key, item in value.items()}
    if isinstance(value, list):
        if declared_list is not None:
            return [_owned(item, declared_list, None) for item in value]
        return [_owned(item, None, None) for item in value]
    return value


class _StrictRecord(BaseModel):
    """The shared strict envelope: explicit schema version, no extra field, owned state."""

    model_config = _MODEL_CONFIG

    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {}
    _nested_list_types: ClassVar[dict[str, type[BaseModel]]] = {}

    @classmethod
    def model_validate(cls, obj: object, **kwargs: object) -> Self:
        if isinstance(obj, cls):
            return read_declared_copy(obj, cls, "INPUT_INVALID")
        return super().model_validate(obj, **kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        if name not in type(self).model_fields:
            super().__setattr__(name, value)
            return
        stored = _actual_storage(self)
        snapshot = dict(stored)
        fields_set = set(object.__getattribute__(self, "__pydantic_fields_set__"))
        try:
            super().__setattr__(name, value)
        except BaseException:
            retained = _actual_storage(self)
            retained.clear()
            retained.update(snapshot)
            retained_fields = object.__getattribute__(self, "__pydantic_fields_set__")
            retained_fields.clear()
            retained_fields.update(fields_set)
            raise

    @model_validator(mode="after")
    def _own_nested_state(self) -> Self:
        nested = type(self)._nested_types
        nested_list = type(self)._nested_list_types
        stored = _actual_storage(self)
        for name in type(self).model_fields:
            value = stored[name]
            owned = _owned(value, nested.get(name), nested_list.get(name))
            if owned is not value:
                object.__setattr__(self, name, owned)
        return self


class TextRange(_StrictRecord):
    """A half-open Unicode code-point range ``[start, end)`` in a text view."""

    schema_version: SchemaVersion
    start: NonNegativeInt
    end: NonNegativeInt

    @model_validator(mode="after")
    def _ordered(self) -> TextRange:
        if self.start > self.end:
            raise ValueError("a text range must not start after it ends")
        return self


class ScopeView(_StrictRecord):
    """One authorized projection of one registered observation snapshot."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    view_id: Identifier
    selector: Annotated[list[SelectorSegment], Field(strict=True)]
    format: Literal["json", "text"]

    @model_validator(mode="after")
    def _legal_selector(self) -> ScopeView:
        _check_selector(self.selector)
        return self


class EvidenceScopeEntry(_StrictRecord):
    """One directory entry: an evidence identifier and its authorized views.

    It carries no local path, no raw payload and no credential. The entry names the snapshot
    bytes only by their SHA-256 and length.
    """

    model_config = _MODEL_CONFIG

    _nested_list_types: ClassVar[dict[str, type[BaseModel]]] = {"views": ScopeView}

    schema_version: SchemaVersion
    evidence_id: Identifier
    source_observation_id: NonBlankStrict
    kind: Literal["observation"]
    raw_sha256: HexDigest
    raw_bytes: PositiveInt
    views: Annotated[list[ScopeView], Field(strict=True)]

    @model_validator(mode="after")
    def _declared_directory(self) -> EvidenceScopeEntry:
        if not self.views:
            raise ValueError("a directory entry must declare at least one view")
        view_ids = [view.view_id for view in self.views]
        if len(set(view_ids)) != len(view_ids):
            raise ValueError("a directory entry must not repeat a view identifier")
        return self


class EvidenceScope(_StrictRecord):
    """The frozen, read-only directory handed to one planning attempt.

    ``scope_id`` names an active reader-side scope; it is not by itself a grant
    token. The directory carries no local path.
    """

    model_config = _MODEL_CONFIG

    _nested_list_types: ClassVar[dict[str, type[BaseModel]]] = {"entries": EvidenceScopeEntry}

    schema_version: SchemaVersion
    scope_id: NonBlankStrict
    run_id: NonBlankStrict
    observation_id: NonBlankStrict
    entries: Annotated[list[EvidenceScopeEntry], Field(strict=True)]

    @model_validator(mode="after")
    def _unique_evidence(self) -> EvidenceScope:
        evidence_ids = [entry.evidence_id for entry in self.entries]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("a scope must not repeat an evidence identifier")
        return self


class EvidenceGrant(_StrictRecord):
    """The host's grant of one view over one registered evidence snapshot."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    run_id: Identifier
    evidence_id: Identifier
    observation_id: Identifier
    view_id: Identifier
    selector: Annotated[list[SelectorSegment], Field(strict=True)]
    format: Literal["json", "text"]

    @model_validator(mode="after")
    def _legal_grant(self) -> EvidenceGrant:
        _check_selector(self.selector)
        return self


class RetrievalConfig(_StrictRecord):
    """The caller's bounded retrieval configuration.

    ``base`` is the already-verified planner configuration. The two retrieval fields can only
    narrow the fixed plan limits of C05: at most two reads per plan (zero is legal) and at most
    three model calls.
    """

    model_config = _MODEL_CONFIG

    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {"base": PlannerConfig}

    schema_version: SchemaVersion
    base: PlannerConfig
    max_reads: Annotated[int, Field(strict=True, ge=0, le=PLAN_MAX_READS)] = DEFAULT_MAX_READS
    max_model_calls: Annotated[
        int, Field(strict=True, ge=1, le=PLAN_MAX_MODEL_CALLS)
    ] = DEFAULT_MAX_MODEL_CALLS


class EvidenceReadRequest(_StrictRecord):
    """One bounded read of one authorized view, with host-supplied bindings.

    The model never supplies the run, plan, turn, request or scope identity; the planner fills
    them from its own trusted state.
    """

    model_config = _MODEL_CONFIG

    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {"text_range": TextRange}

    schema_version: SchemaVersion
    scope_id: NonBlankStrict
    run_id: NonBlankStrict
    plan_id: NonBlankStrict
    turn: PositiveInt
    request_id: NonBlankStrict
    evidence_id: Identifier
    view_id: Identifier
    text_range: TextRange | None = None


class EvidenceReadResult(_StrictRecord):
    """The complete, measured result of one authorized read.

    ``not_found`` carries ``value=None`` and ``value_sha256=None`` with ``value_bytes=0``; a
    found JSON ``null`` is distinguishable because its status is ``found`` and its value digest
    is the digest of the canonical bytes ``null``.
    """

    model_config = _MODEL_CONFIG

    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {"text_range": TextRange}

    schema_version: SchemaVersion
    scope_id: NonBlankStrict
    run_id: NonBlankStrict
    plan_id: NonBlankStrict
    turn: PositiveInt
    request_id: NonBlankStrict
    evidence_id: Identifier
    view_id: Identifier
    status: Literal["found", "not_found"]
    source_observation_id: NonBlankStrict
    raw_sha256: HexDigest
    raw_bytes: PositiveInt
    selector: Annotated[list[SelectorSegment], Field(strict=True)]
    format: Literal["json", "text"]
    text_range: TextRange | None = None
    partial: StrictBool
    value: JsonValue | None = None
    value_sha256: HexDigest | None = None
    value_bytes: NonNegativeInt

    @model_validator(mode="after")
    def _measured_result(self) -> EvidenceReadResult:
        _check_selector(self.selector)
        if self.status == "not_found":
            if self.value is not None or self.value_sha256 is not None:
                raise ValueError("a not_found result must not carry a value or a value digest")
            if self.value_bytes != 0:
                raise ValueError("a not_found result must measure zero value bytes")
            if self.partial:
                raise ValueError("a not_found result is not a partial value")
            return self
        if self.value_sha256 is None:
            raise ValueError("a found result must carry its value digest")
        try:
            measured = canonical_json_bytes(self.value)
        except RetrievalError:
            raise ValueError("a found result must carry a JSON value") from None
        if self.value_bytes != len(measured) or self.value_sha256 != _digest(measured):
            raise ValueError("a found result must carry its own measured value")
        if self.format == "json":
            if self.text_range is not None or self.partial:
                raise ValueError("a json view has no text range and is never partial")
        else:
            if not isinstance(self.value, str):
                raise ValueError("a text view must carry a string value")
            # `partial` is a statement about the *source* string, which a result cannot
            # re-derive from the slice alone; the reader that owns the snapshot sets it and the
            # planner only checks that the declared interval was echoed.
        return self


class RetrievalModelRequest(_StrictRecord):
    """One complete, measured request handed to exactly one retrieval provider call.

    It repeats the original Core request (context and observation), the frozen evidence
    directory and every read result already obtained in this plan. ``base_request`` carries the
    same request identifier, the same Run and the same response schema as this envelope.
    """

    model_config = _MODEL_CONFIG

    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {
        "base_request": ModelRequest,
        "scope": EvidenceScope,
    }
    _nested_list_types: ClassVar[dict[str, type[BaseModel]]] = {
        "exchanges": EvidenceReadResult,
    }

    schema_version: SchemaVersion
    run_id: NonBlankStrict
    plan_id: NonBlankStrict
    turn: PositiveInt
    request_id: NonBlankStrict
    base_request: ModelRequest
    scope: EvidenceScope
    exchanges: Annotated[list[EvidenceReadResult], Field(strict=True)]
    response_schema: JsonObject

    @model_validator(mode="after")
    def _bound_request(self) -> RetrievalModelRequest:
        base = self.base_request
        if base.request_id != self.request_id:
            raise ValueError("the base request must carry this request identifier")
        if base.model_context.run_id != self.run_id:
            raise ValueError("the base request must carry this run identifier")
        if base.response_schema != self.response_schema:
            raise ValueError("the base request must carry this response schema")
        if self.scope.run_id != self.run_id:
            raise ValueError("the frozen scope must belong to this run")
        for exchange in self.exchanges:
            if exchange.run_id != self.run_id or exchange.plan_id != self.plan_id:
                raise ValueError("an exchange must belong to this plan")
            if exchange.scope_id != self.scope.scope_id:
                raise ValueError("an exchange must belong to this scope")
            if exchange.turn >= self.turn:
                raise ValueError("an exchange must precede this turn")
        return self


def _check_selector(selector: list[object]) -> None:
    """The closed selector grammar: payload-rooted, at most 32 bounded segments."""
    if not 1 <= len(selector) <= _SELECTOR_SEGMENTS_MAX:
        raise ValueError("a selector must carry 1 to 32 segments")
    first = selector[0]
    if first not in ("state", "summary"):
        raise ValueError("a selector must start at the observation state or summary")
    for part in selector:
        if isinstance(part, str):
            if len(part) > _SELECTOR_TEXT_MAX:
                raise ValueError("a selector segment must not exceed 128 characters")
        elif isinstance(part, bool) or not isinstance(part, int) or part < 0:
            raise ValueError("a selector segment must be a non-negative integer or a string")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_plain_json(value: object) -> None:
    """Refuse a non-JSON container or key before the canonical encoder sees it."""
    stack: list[tuple[object, bool]] = [(value, False)]
    active: set[int] = set()
    while stack:
        item, leaving = stack.pop()
        if isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if leaving:
                active.discard(identity)
                continue
            if identity in active:
                raise RetrievalError("invalid_input", "input")
            if isinstance(item, tuple):
                raise RetrievalError("invalid_input", "input")
            active.add(identity)
            stack.append((item, True))
            if isinstance(item, dict):
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise RetrievalError("invalid_input", "input")
                    stack.append((child, False))
            else:
                for child in item:
                    stack.append((child, False))
        elif isinstance(item, bool) or item is None or isinstance(item, (int, float, str)):
            continue
        else:
            raise RetrievalError("invalid_input", "input")


def canonical_json_bytes(value: object) -> bytes:
    """The protocol's canonical JSON UTF-8 bytes of one already-validated JSON value."""
    _require_plain_json(value)
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        raise RetrievalError("invalid_input", "input") from None


def canonical_record_bytes(record: BaseModel) -> bytes:
    """The canonical JSON bytes of one record's declared JSON projection."""
    try:
        return canonical_json_bytes(record.model_dump(mode="json"))
    except RetrievalError:
        raise RetrievalError("invalid_input", "input") from None


def measure_record(
    record: BaseModel,
    *,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
    code: str,
    stage: str,
) -> int:
    """Measure one record's canonical JSON bytes, depth and nodes; refuse an oversize value."""
    payload = record.model_dump(mode="json")
    try:
        measure_json(payload, max_depth=max_depth, max_nodes=max_nodes,
                     code="INPUT_INVALID", location=("json",))
    except PlannerError:
        raise RetrievalError(code, stage) from None
    raw = canonical_json_bytes(payload)
    if len(raw) > max_bytes:
        raise RetrievalError(code, stage)
    return len(raw)


def copy_record(
    value: object,
    model: type[RecordT],
    code: str,
    stage: str,
) -> RecordT:
    """Revalidate one supplied record from its own storage and return a detached copy."""
    try:
        return read_declared_copy(value, model, "INPUT_INVALID")
    except RetrievalError:
        raise
    except PlannerError:
        raise RetrievalError(code, stage) from None
    except (ValidationError, TypeError, ValueError, RecursionError):
        raise RetrievalError(code, stage) from None


def effective_base_config(base: PlannerConfig) -> PlannerConfig:
    """The stricter of the caller's base configuration and the fixed C05 retrieval caps."""
    return PlannerConfig(
        schema_version=SCHEMA_VERSION,
        call_timeout_ms=base.call_timeout_ms,
        cleanup_timeout_ms=base.cleanup_timeout_ms,
        max_calls=base.max_calls,
        max_request_bytes=min(base.max_request_bytes, REQUEST_MAX_BYTES),
        max_response_bytes=min(base.max_response_bytes, RESPONSE_MAX_BYTES),
        max_output_tokens=min(base.max_output_tokens, OUTPUT_TOKENS_MAX),
        max_json_depth=min(base.max_json_depth, JSON_DEPTH_MAX),
        max_json_nodes=min(base.max_json_nodes, JSON_NODES_MAX),
    )


class _RejectedJSON(ValueError):
    """An internal marker for content the strict parser must refuse."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _RejectedJSON("duplicate key")
        result[key] = value
    return result


def _reject_constant(_text: str) -> Any:
    raise _RejectedJSON("non-finite number")


def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in text)


def check_no_surrogates(value: object, code: str, stage: str) -> None:
    """Refuse a lone surrogate anywhere in a parsed JSON value."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _has_surrogate(item):
                raise RetrievalError(code, stage)
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or _has_surrogate(key):
                    raise RetrievalError(code, stage)
                stack.append(child)
        elif isinstance(item, list):
            stack.extend(item)


def _raw_depth_within(text: str, limit: int) -> bool:
    """Bound the nesting depth before the recursive JSON parser runs."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > limit:
                return False
        elif character in "}]":
            depth -= 1
    return True


def parse_strict_json(
    data: bytes | str,
    *,
    max_depth: int,
    max_nodes: int,
    code: str,
    stage: str,
) -> object:
    """Parse exactly one strict JSON value: no duplicate key, no NaN, no trailing text."""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise RetrievalError(code, stage) from None
    elif isinstance(data, str):
        text = data
    else:
        raise RetrievalError(code, stage)
    if not _raw_depth_within(text, max_depth):
        raise RetrievalError(code, stage)
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (_RejectedJSON, ValueError, RecursionError):
        raise RetrievalError(code, stage) from None
    check_no_surrogates(parsed, code, stage)
    try:
        measure_json(parsed, max_depth=max_depth, max_nodes=max_nodes,
                     code="INPUT_INVALID", location=("json",))
    except PlannerError:
        raise RetrievalError(code, stage) from None
    return parsed
