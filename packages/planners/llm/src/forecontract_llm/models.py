"""Strict, versioned records for the offline LLM planner.

Every record requires an explicit ``schema_version="0.1"``, forbids unknown fields and
validates assignment. Scalars are strict: a boolean is never an integer, a numeric string is
never a number, and a tuple is never a list. One field assignment is transactional: when a
cross-field verdict refuses the new value, the previous stored state is restored, so a refused
assignment cannot leave a half-valid record behind.

Records own their nested state. Construction revalidates every nested record from its actual
``__dict__`` storage and replaces it with a detached copy, so a caller-supplied capability
record cannot be aliased into the planner. A record handed to a public entry point is read
back from that same declared storage - never through an overridable ``model_dump``, and never
through an overridable ``__getattribute__``/``__dict__`` view - so an object created by
``model_construct``/``model_copy`` or mutated through ``__dict__`` cannot pass a public
boundary as if it were valid. The storage mapping is fetched by calling the declared
``BaseModel.__dict__`` descriptor directly, which performs no descriptor lookup on the
instance's type; even ``object.__getattribute__`` would still run a subclass ``__dict__``
property, so it is not used for this read.

This module performs no I/O. The JSON helpers here are the single place where the request
size, depth and node limits are measured, so the codec cannot accidentally measure something
different from what the planner sends.
"""

from __future__ import annotations

import json
from typing import Annotated, ClassVar, Literal, Self, TypeVar

from forecontract.models.actions import AdapterCapabilities
from forecontract.models.json_types import JsonObject
from forecontract.models.observations import Observation
from forecontract.verification.models import TaskCompletionCriteria
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from forecontract_llm.errors import PlannerError

__all__ = [
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
    "PlannerCloseResult",
    "PlannerConfig",
    "PlannerContext",
    "measure_json",
    "read_declared_copy",
    "request_byte_length",
]

SCHEMA_VERSION = "0.1"
SchemaVersion = Literal["0.1"]


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain a non-whitespace character")
    return value


#: A strict string that keeps the supplied bytes; nothing is trimmed or repaired.
NonBlankStr = Annotated[str, Field(strict=True), AfterValidator(_non_blank)]
StrictStr = Annotated[str, Field(strict=True)]
StrictBool = Annotated[bool, Field(strict=True)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
#: A finite, strictly positive number. Pydantic's strict float accepts an integer input but
#: rejects a numeric string and a boolean, which is exactly the agreed timeout rule.
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]

_MODEL_CONFIG = ConfigDict(extra="forbid", validate_assignment=True, hide_input_in_errors=True)

ModelT = TypeVar("ModelT", bound=BaseModel)

#: The declared base-class ``__dict__`` descriptor.
#:
#: Calling this descriptor directly reads the instance dictionary the object really stores its
#: fields in. It is bound to ``BaseModel`` and never looked up on the instance, so a subclass
#: that defines its own ``__dict__`` property - with a setter that writes the real storage and a
#: getter that returns a sanitised mapping - cannot substitute its view here. A plain
#: ``object.__getattribute__(record, "__dict__")`` is not enough: that is the ordinary attribute
#: protocol and still performs the descriptor lookup that finds the subclass property. Every
#: pydantic 2.x ``BaseModel`` declares ``__dict__`` in ``__slots__``, so this descriptor exists
#: on the declared base class; if a runtime ever lacked it, the record is refused instead of
#: being read through an overridable view.
_DECLARED_STORAGE = BaseModel.__dict__.get("__dict__")


def _match_guidance(guidance: dict, allowed: list[str], field: str) -> None:
    """Reject a guidance map whose key set is not exactly the declared capability names."""
    if set(guidance) != set(allowed):
        raise ValueError(f"{field} must describe exactly the declared capability names")
    for value in guidance.values():
        if not isinstance(value, dict) or not value:
            raise ValueError(f"every {field} entry must be a non-empty JSON object")


def _actual_storage(record: BaseModel, code: str) -> dict[str, object]:
    """Return the record's real ``__dict__`` mapping, never a caller-supplied view.

    The mapping is read by calling the declared ``BaseModel.__dict__`` descriptor directly, so
    no descriptor lookup on the instance's type happens and a subclass ``__dict__`` property has
    no authority here. Reading ``object.__getattribute__(record, "__dict__")`` would still find
    that property, because the ordinary attribute protocol performs the lookup; that is the gap
    this function closes. The returned mapping is the object's own stored state, so a value that
    a virtual view hides is still seen and refused by the declared schema.
    """
    if _DECLARED_STORAGE is None:
        raise PlannerError(code, "the supplied record cannot be read as declared fields",
                           location=("record",))
    try:
        stored = _DECLARED_STORAGE.__get__(record, type(record))
    except (AttributeError, TypeError):
        raise PlannerError(code, "the supplied record cannot be read as declared fields",
                           location=("record",)) from None
    if not isinstance(stored, dict):
        raise PlannerError(code, "the supplied record cannot be read as declared fields",
                           location=("record",))
    return stored


class _StrictRecord(BaseModel):
    """The shared strict envelope.

    One field assignment is transactional. Pydantic stores the new value before a
    ``mode="after"`` verdict can refuse it, which would leave the record half-valid; this
    envelope snapshots the stored fields, and restores them when the assignment is refused.

    Every nested record is revalidated from its own stored fields and replaced with a detached
    copy, so a caller cannot keep an alias into the record and a forged nested value is refused
    at construction instead of being trusted.

    The overridden Python ingress refuses an existing instance that was produced by
    ``model_construct`` or ``model_copy``: it is read back from its own storage and revalidated
    instead of being trusted. A plain mapping still goes through the ordinary schema.
    """

    #: The facility code a rejected nested value is reported with by this record.
    _error_code: ClassVar[str] = "INPUT_INVALID"
    #: The declared record type of each nested-model field, used to revalidate real storage.
    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {}

    @classmethod
    def model_validate(cls, obj: object, **kwargs: object) -> Self:
        if isinstance(obj, cls):
            code = "RESPONSE_INVALID" if isinstance(obj, ModelResponse) else "INPUT_INVALID"
            return read_declared_copy(obj, cls, code)
        return super().model_validate(obj, **kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        if name not in type(self).model_fields:
            super().__setattr__(name, value)
            return
        stored = _actual_storage(self, type(self)._error_code)
        snapshot = dict(stored)
        fields_set = set(object.__getattribute__(self, "__pydantic_fields_set__"))
        try:
            super().__setattr__(name, value)
        except BaseException:
            # A refused assignment may already have replaced ``__dict__`` with the new value,
            # so restore the mapping that is current now instead of the pre-assignment object.
            retained_storage = _actual_storage(self, type(self)._error_code)
            retained_storage.clear()
            retained_storage.update(snapshot)
            retained_fields = object.__getattribute__(self, "__pydantic_fields_set__")
            retained_fields.clear()
            retained_fields.update(fields_set)
            raise

    @model_validator(mode="after")
    def _own_nested_state(self) -> Self:
        code = self._error_code
        nested = type(self)._nested_types
        stored = _actual_storage(self, code)
        for name in type(self).model_fields:
            value = stored[name]
            owned = _own_value(value, code, nested.get(name))
            if owned is not value:
                object.__setattr__(self, name, owned)
        return self


class PlannerContext(_StrictRecord):
    """The trusted, host-supplied planning context for exactly one Run.

    ``action_guidance`` and ``outcome_guidance`` are domain-supplied payload descriptions.
    Their key sets must be exactly the adapter's declared capability names and every value
    must be a non-empty JSON object. They are format documentation only: this package does not
    execute an arbitrary JSON Schema, and a name being declared does not make an action
    executable.
    """

    model_config = _MODEL_CONFIG

    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {
        "task_criteria": TaskCompletionCriteria,
        "capabilities": AdapterCapabilities,
    }

    schema_version: SchemaVersion
    run_id: NonBlankStr
    goal_id: NonBlankStr
    goal_text: NonBlankStr
    task_criteria: TaskCompletionCriteria
    capabilities: AdapterCapabilities
    action_guidance: JsonObject
    outcome_guidance: JsonObject

    @model_validator(mode="after")
    def _guidance_matches_capabilities(self) -> PlannerContext:
        _match_guidance(
            self.action_guidance,
            self.capabilities.supported_actions,
            "action_guidance",
        )
        _match_guidance(
            self.outcome_guidance,
            self.capabilities.supported_outcome_types,
            "outcome_guidance",
        )
        return self


class PlannerConfig(_StrictRecord):
    """The caller's bounded planning limits.

    Every value is a configurable upper bound; none of them widens the Run's own budget. The
    timeout fields accept a finite positive integer or float and reject a boolean; every other
    field is a strict positive integer.
    """

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    call_timeout_ms: PositiveNumber = 30000
    cleanup_timeout_ms: PositiveNumber = 100
    max_calls: PositiveInt = 20
    max_request_bytes: PositiveInt = 262144
    max_response_bytes: PositiveInt = 65536
    max_output_tokens: PositiveInt = 2048
    max_json_depth: PositiveInt = 32
    max_json_nodes: PositiveInt = 10000


class ModelUsage(_StrictRecord):
    """A provider-declared token usage. A missing usage is not a zero cost."""

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt


class ModelRequest(_StrictRecord):
    """The structured request handed to exactly one provider call.

    It carries the trusted context, the complete current observation, the response schema and
    the output-token constraint. It contains no API key, provider URL or tool permission.
    """

    model_config = _MODEL_CONFIG

    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {
        "model_context": PlannerContext,
        "observation": Observation,
    }

    schema_version: SchemaVersion
    request_id: NonBlankStr
    model_context: PlannerContext
    observation: Observation
    response_schema: JsonObject
    max_output_tokens: PositiveInt


class ModelResponse(_StrictRecord):
    """The provider's answer to exactly one request.

    ``ok`` requires a string ``content``; ``refused`` requires ``content=None``. ``usage`` may
    be absent. The response identifier must name the current request; a replayed response is
    refused by the codec.
    """

    model_config = _MODEL_CONFIG

    _error_code: ClassVar[str] = "RESPONSE_INVALID"
    _nested_types: ClassVar[dict[str, type[BaseModel]]] = {"usage": ModelUsage}

    schema_version: SchemaVersion
    request_id: NonBlankStr
    status: Literal["ok", "refused"]
    content: StrictStr | None = None
    usage: ModelUsage | None = None

    @model_validator(mode="after")
    def _content_matches_status(self) -> ModelResponse:
        if self.status == "ok" and not isinstance(self.content, str):
            raise ValueError("an ok response must carry string content")
        if self.status == "refused" and self.content is not None:
            raise ValueError("a refused response must not carry content")
        return self


class PlannerCloseResult(_StrictRecord):
    """The honest outcome of one ``aclose`` call.

    ``drained`` is true exactly when no provider call task remains pending.
    """

    model_config = _MODEL_CONFIG

    schema_version: SchemaVersion
    drained: StrictBool
    pending_calls: NonNegativeInt

    @model_validator(mode="after")
    def _drained_matches_pending(self) -> PlannerCloseResult:
        if self.drained != (self.pending_calls == 0):
            raise ValueError("drained must agree with pending_calls")
        return self


def _check_declared_tree(record: BaseModel, code: str) -> None:
    """Reject a cycle or an undeclared stored key anywhere below one record instance.

    Only real `BaseModel` instances are followed. A plain mapping or list is walked for
    cycles as well; a scalar is ignored because the schema revalidation decides its validity.
    """
    stack: list[tuple[object, bool]] = [(record, False)]
    active: set[int] = set()
    while stack:
        value, leaving = stack.pop()
        if isinstance(value, BaseModel):
            identity = id(value)
            if leaving:
                active.discard(identity)
                continue
            if identity in active:
                raise PlannerError(code, "the supplied record contains a cycle",
                                   location=("record",))
            stored = _actual_storage(value, code)
            allowed = frozenset(type(value).model_fields)
            for key in stored:
                if not (isinstance(key, str) and key in allowed):
                    raise PlannerError(
                        code,
                        "the supplied record carries a stored key that is not a declared field",
                        location=("record",),
                    )
            active.add(identity)
            stack.append((value, True))
            for key in stored:
                stack.append((stored[key], False))
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if leaving:
                active.discard(identity)
                continue
            if identity in active:
                raise PlannerError(code, "the supplied record contains a cycle",
                                   location=("record",))
            active.add(identity)
            stack.append((value, True))
            for item in value:
                stack.append((item, False))
        elif isinstance(value, dict):
            identity = id(value)
            if leaving:
                active.discard(identity)
                continue
            if identity in active:
                raise PlannerError(code, "the supplied record contains a cycle",
                                   location=("record",))
            active.add(identity)
            stack.append((value, True))
            for key in value:
                if not isinstance(key, str):
                    raise PlannerError(
                        code,
                        "the supplied record carries an object key that is not a string",
                        location=("record",),
                    )
                stack.append((value[key], False))


def _plain_stored(value: object, code: str) -> object:
    """Rebuild one stored value as plain data, reading only declared storage.

    A real record is expanded from its own ``__dict__`` - never from ``model_dump`` or any
    other overridable serialiser - and every nested record is expanded in the same way. The
    rebuilt plain tree is then validated by the declared schema, so a subclass method and a
    forged stored field have no authority over the result. A tuple is preserved so that a
    strict list field still refuses it instead of silently coercing it.
    """
    if isinstance(value, BaseModel):
        stored = _actual_storage(value, code)
        allowed = frozenset(type(value).model_fields)
        rebuilt: dict[str, object] = {}
        for key, item in stored.items():
            if not (isinstance(key, str) and key in allowed):
                raise PlannerError(
                    code,
                    "the supplied record carries a stored key that is not a declared field",
                    location=("record",),
                )
            rebuilt[key] = _plain_stored(item, code)
        return rebuilt
    if isinstance(value, dict):
        rebuilt_object: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlannerError(
                    code,
                    "the supplied record carries an object key that is not a string",
                    location=("record",),
                )
            rebuilt_object[key] = _plain_stored(item, code)
        return rebuilt_object
    if isinstance(value, list):
        return [_plain_stored(item, code) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_stored(item, code) for item in value)
    return value


def _own_value(value: object, code: str, declared: type[BaseModel] | None) -> object:
    """Return a detached copy of one stored field, revalidating a nested record in place.

    A nested record is rebuilt from its declared type - not from the type the caller stored -
    so an unsupported subclass is normalised to the declared record and its stored fields are
    revalidated. A JSON object or list is copied recursively, and a scalar is returned as is.
    """
    if isinstance(value, BaseModel):
        if declared is None:
            raise PlannerError(code, "the supplied record carries an undeclared nested record",
                               location=("record",))
        return read_declared_copy(value, declared, code)
    if isinstance(value, dict):
        return {key: _own_value(item, code, None) for key, item in value.items()}
    if isinstance(value, list):
        return [_own_value(item, code, None) for item in value]
    if isinstance(value, tuple):
        return tuple(_own_value(item, code, None) for item in value)
    return value


def read_declared_copy(value: object, model: type[ModelT], code: str) -> ModelT:
    """Revalidate one supplied value from its own storage and return a detached copy.

    A real instance is first walked for an undeclared stored key or a cycle, then rebuilt as
    plain data from its actual ``__dict__`` storage and validated through the declared schema.
    A plain mapping is revalidated directly. Either way the caller keeps no shared mutable
    container, a forged stored field is refused instead of being trusted, and neither an
    overridable serialiser nor an overridable ``__getattribute__``/``__dict__`` view is
    consulted.
    """
    if isinstance(value, model):
        try:
            _check_declared_tree(value, code)
        except PlannerError:
            raise
        except RecursionError:
            raise PlannerError(code, "the supplied record is too deeply nested",
                               location=("record",)) from None
        try:
            raw = _plain_stored(value, code)
        except PlannerError:
            raise
        except RecursionError:
            raise PlannerError(code, "the supplied record is too deeply nested",
                               location=("record",)) from None
    else:
        raw = value
    try:
        return model.model_validate(raw)
    except (AttributeError, ValidationError, RecursionError, TypeError, ValueError):
        raise PlannerError(code, "the supplied value does not match the declared schema",
                           location=("record",)) from None
    except PlannerError:
        raise


def measure_json(
    value: object,
    *,
    max_depth: int,
    max_nodes: int,
    code: str,
    location: tuple[str | int, ...] = ("json",),
) -> tuple[int, int]:
    """Measure one JSON tree iteratively and refuse an oversize depth or node count.

    The root container is depth 1 and every nested container adds 1. The node count includes
    the root, every container and every scalar value; object keys are not counted as nodes.
    The walk is iterative, so a deep or cyclic Python structure becomes a bounded refusal
    instead of a recursion failure.
    """
    depth = 0
    nodes = 0
    stack: list[tuple[object, int, bool]] = [(value, 1, False)]
    active: set[int] = set()
    while stack:
        item, level, leaving = stack.pop()
        if isinstance(item, dict):
            identity = id(item)
            if leaving:
                active.discard(identity)
                continue
            if identity in active:
                raise PlannerError(code, "the value contains a cycle", location=location)
            active.add(identity)
            nodes += 1
            depth = max(depth, level)
            if level > max_depth or nodes > max_nodes:
                raise PlannerError(code, "the value exceeds the configured JSON limit",
                                   location=location)
            stack.append((item, level, True))
            for child in item.values():
                stack.append((child, level + 1, False))
        elif isinstance(item, (list, tuple)):
            identity = id(item)
            if leaving:
                active.discard(identity)
                continue
            if identity in active:
                raise PlannerError(code, "the value contains a cycle", location=location)
            active.add(identity)
            nodes += 1
            depth = max(depth, level)
            if level > max_depth or nodes > max_nodes:
                raise PlannerError(code, "the value exceeds the configured JSON limit",
                                   location=location)
            stack.append((item, level, True))
            for child in item:
                stack.append((child, level + 1, False))
        else:
            nodes += 1
            if nodes > max_nodes:
                raise PlannerError(code, "the value exceeds the configured JSON limit",
                                   location=location)
    return depth, nodes


def request_byte_length(request: ModelRequest) -> int:
    """The exact UTF-8 length of the compact JSON form of one request.

    The measurement covers the complete request including its response schema. It uses the
    agreed ``model_dump(mode="json")`` plus ``json.dumps`` representation.
    """
    try:
        payload = request.model_dump(mode="json")
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":"))
        return len(text.encode("utf-8"))
    except RecursionError:
        raise PlannerError("REQUEST_TOO_LARGE", "the request is too deeply nested",
                           location=("request",)) from None
    except (UnicodeEncodeError, ValueError):
        raise PlannerError("INPUT_INVALID", "the request cannot be represented as JSON",
                           location=("request",)) from None
