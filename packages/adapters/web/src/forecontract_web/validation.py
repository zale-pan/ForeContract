"""Public Web validation entry points.

``parse_web_action`` and ``validate_web_action`` revalidate the Core action envelope and then the
domain payload selected by ``action_type``; ``parse_web_observation`` revalidates the Core
observation envelope and returns the isolated Web snapshot state. None of them performs I/O,
submits or executes an action, evaluates a verdict, edits a supplied object or replaces an outer
Core identity.

Revalidation never trusts a supplied instance. Pydantic returns an existing instance from
``model_validate`` after only a shallow check, ``model_copy`` and ``model_construct`` can store an
invalid field or an undeclared key, and ``model_dump`` silently drops a key Pydantic does not know
about. Both envelopes are therefore read from the actual stored fields, an undeclared stored key
is rejected, and the record is rebuilt from those raw fields through the declared schema before
any domain rule runs. A known Core contract-snapshot container is unwrapped because the Builder
stores JSON payloads in immutable containers that the Core JSON schema cannot read; nothing else
is coerced, so an arbitrary tuple, bytes, set or wrong object still fails the strict schema.

The order of decisions is fixed: the Core envelope is confirmed first, the action type is decided
second and the selected domain payload is validated last. An unsupported action type therefore
always raises the Core ``UnsupportedActionError`` before any payload rule can reject the payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from forecontract.contracts.snapshot import FrozenList, FrozenMapping
from forecontract.models.actions import ActionSpec
from forecontract.models.observations import ArtifactRef, Observation
from forecontract.protocols.adapter import UnsupportedActionError
from pydantic import BaseModel, ValidationError

from forecontract_web.errors import (
    _FIELD_PLACEHOLDER,
    WebActionPayloadError,
    WebValidationError,
    _bounded_call,
    _safe_location,
)
from forecontract_web.models import (
    _PAYLOAD_MODELS,
    WEB_ACTION_TYPES,
    WEB_ENVIRONMENT_TYPE,
    WebObservationState,
)

__all__ = [
    "parse_web_action",
    "parse_web_observation",
    "validate_web_action",
]

#: The four declared fields of the Core action envelope.
_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(ActionSpec.model_fields)

#: The eight declared fields of the Core observation envelope.
_OBSERVATION_FIELDS: Final[frozenset[str]] = frozenset(Observation.model_fields)

#: Every declared field name of the Web payload, locator and evidence models. Only these names
#: may appear in a public diagnostic; anything else becomes a constant placeholder.
_DOMAIN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "error_code",
        "key",
        "match_count",
        "name",
        "role",
        "schema_version",
        "state",
        "status",
        "strategy",
        "target",
        "targets",
        "test_id",
        "text",
        "url",
        "value",
        "visible",
    }
)


def _thaw_frozen(value: object, seen: frozenset[int] = frozenset()) -> object:
    """Unwrap Core's two known immutable JSON containers, and nothing else.

    ``FrozenMapping`` and ``FrozenList`` are the containers the Core contract snapshot stores a
    committed JSON payload in; the strict Core JSON schema cannot read them, so leaving them in
    place would reject a legitimate Builder-submitted action. Only those two types are rewritten
    into the plain ``dict`` and ``list`` the schema declares, and the result is revalidated by
    that same strict schema. A tuple, a set, bytes or any other object is returned unchanged and
    is rejected by the schema instead of being coerced into legal JSON.
    """
    if isinstance(value, (FrozenMapping, FrozenList)):
        if id(value) in seen:
            return value
        nested = seen | {id(value)}
        if isinstance(value, FrozenMapping):
            return {key: _thaw_frozen(value[key], nested) for key in value}
        return [_thaw_frozen(item, nested) for item in value]
    return value


def _thaw(value: object, message: str, code: str) -> object:
    """Unwrap the known frozen containers with their own reads inside the boundary.

    A ``FrozenMapping`` is only a wrapper: iterating it and reading ``value[key]`` runs the backing
    mapping's code, which a caller can supply. Computing the normalised value as an *argument* to
    the store evaluates that code before the bounded store is entered, so the frozen read sat outside
    the boundary even after the store itself was bounded. The normalisation is therefore its own
    bounded read, and the store that follows is a second one: a frozen container whose backing read
    raises becomes the entry point's fixed code instead of a raw ``ValueError``, while a legitimate
    frozen payload still unwraps exactly as before.
    """
    return _bounded_call(lambda: _thaw_frozen(value), message, code)


#: The five declared fields of a Core artifact reference.
_ARTIFACT_FIELDS: Final[frozenset[str]] = frozenset(ArtifactRef.model_fields)


def _key_is_declared(key: object, allowed: frozenset[str], message: str, code: str) -> bool:
    """Ask whether a supplied key is one this envelope declares, through the key's own code.

    Materialising a mapping does not make its keys trusted: a supported ``str`` subclass may
    override ``__hash__`` or ``__eq__``, and the membership test itself then runs that caller code.
    The test therefore runs inside a bounded read, so a raising key becomes this entry point's
    fixed code instead of leaking a raw ``ValueError``. The answer it returns is judged here.
    """

    def probe() -> bool:
        return isinstance(key, str) and key in allowed

    return _bounded_call(probe, message, code)


def _store_field(fields: dict, key: object, item: object, message: str, code: str) -> None:
    """Store one field under the caller's own key, through that key's own hashing.

    The membership test is only the first use of a key. ``fields[key] = item`` hashes the same
    caller object again, and a key whose first hash succeeds but whose second raises is still a
    caller-owned step on the same chain. It therefore runs inside its own bounded read, so the
    failure is converted instead of escaping after a successful membership check.
    """

    def store() -> None:
        fields[key] = item

    _bounded_call(store, message, code)


_MISSING: Final[object] = object()


def _take(fields: dict, name: str, message: str, code: str) -> object:
    """Look one declared field up, through any retained caller key's own comparison.

    Rebuilding a dictionary does not remove caller behaviour from it: a retained caller key still
    hashes and compares itself when this package later looks the field up. The lookup therefore
    runs inside a bounded read too, and a sentinel distinguishes absence from a stored ``None``.
    The value itself is judged afterwards.
    """

    def take() -> object:
        if name not in fields:
            return _MISSING
        return fields[name]

    return _bounded_call(take, message, code)


def _stored_fields(
    record: BaseModel,
    allowed: frozenset[str],
    message: str,
    code: str,
) -> dict:
    """Read a record's actual stored fields, rejecting a key it does not declare.

    ``__dict__`` holds the declared fields a model instance carries, but it is not the whole
    store: a record built with ``model_construct`` from an ``extra="allow"`` schema keeps an
    undeclared key in ``__pydantic_extra__`` instead. Reading only ``__dict__`` would therefore
    accept that record and silently drop the key on the way out, so the extra store is inspected
    too and any entry in it is a rejection rather than something to discard.

    The copy is taken inside the bounded read as well, because copying a dictionary hashes every
    key it contains - the same caller objects this function is already checking.
    """
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        raise WebValidationError(message, code=code)

    def read() -> tuple[list, object, dict]:
        return list(stored), getattr(record, "__pydantic_extra__", None), dict(stored)

    keys, extra, fields = _bounded_call(read, message, code)
    if extra:
        raise WebValidationError(message, code=code, path=(_FIELD_PLACEHOLDER,))
    for key in keys:
        if not _key_is_declared(key, allowed, message, code):
            raise WebValidationError(message, code=code, path=(_FIELD_PLACEHOLDER,))
    return fields


def _mapping_fields(
    value: Mapping,
    allowed: frozenset[str],
    message: str,
    code: str,
) -> dict:
    """Read a plain mapping, rejecting a non-string or undeclared key with a safe path.

    The caller owns this mapping's iteration, so the items are materialised inside a bounded call:
    an exception from ``items`` or ``__getitem__`` - including a forged ``WebValidationError`` or
    ``ActionValidationError`` - becomes this entry point's fixed code instead of escaping with the
    caller's own code, path or message.

    Materialising is not the end of the caller's code. Each key is still a caller object, so both
    the declared-key membership test and the store that follows it run through bounded reads; only
    the answers are decided here.
    """

    def read() -> list:
        return list(value.items())

    items = _bounded_call(read, message, code)
    fields: dict = {}
    for key, item in items:
        if not _key_is_declared(key, allowed, message, code):
            raise WebValidationError(message, code=code, path=(_FIELD_PLACEHOLDER,))
        _store_field(fields, key, item, message, code)
    return fields


def _artifact_items(value: object, message: str, code: str) -> object:
    """Rewrite a nested Core artifact reference into its raw stored fields.

    ``Observation.model_validate`` returns an existing ``ArtifactRef`` instance after a shallow
    check, so a nested record built by ``model_copy`` or ``model_construct`` with an ``int``
    identifier, ``bytes`` type or ``None`` uri reaches the returned envelope unchanged. Each
    supported nested model is therefore detached into its own stored fields first, which makes the
    outer schema revalidate every nested value instead of trusting the instance, and an undeclared
    stored field inside the nested record is a rejection like any other.

    The container itself is caller-owned: an ``artifacts`` value may be a list subclass whose own
    ``__iter__`` raises, and that iteration used to run outside the conversion boundary so its raw
    exception escaped. The container is therefore materialised into a plain list inside a bounded
    call first; the per-item rewriting then runs on that plain list.
    """

    def read() -> object:
        thawed = _thaw_frozen(value)
        return list(thawed) if isinstance(thawed, list) else thawed

    thawed = _bounded_call(read, message, code)
    if not isinstance(thawed, list):
        return thawed
    items: list[object] = []
    for item in thawed:
        if isinstance(item, ArtifactRef):
            items.append(_stored_fields(item, _ARTIFACT_FIELDS, message, code))
        elif isinstance(item, Mapping):
            items.append(_mapping_fields(item, _ARTIFACT_FIELDS, message, code))
        else:
            items.append(item)
    return items


def _action_envelope(action: object) -> ActionSpec:
    """Revalidate the Core action envelope, whatever shape the candidate arrives in."""
    code = "WEB_ACTION_INVALID"
    message = "the action does not match the Core action envelope"
    if isinstance(action, ActionSpec):
        raw = _stored_fields(action, _ENVELOPE_FIELDS, message, code)
    elif isinstance(action, Mapping):
        raw = _mapping_fields(action, _ENVELOPE_FIELDS, message, code)
    else:
        raise WebValidationError(message, code=code)
    # The rebuilt document still holds the caller's own key objects, so each later lookup runs
    # through the key's own comparison and is therefore bounded too. A value that is absent is
    # distinguished by a sentinel rather than by a second lookup.
    # The rebuilt document keeps the caller's own key objects, so every later operation on them is
    # bounded: the lookup, and then the writeback that stores the rewritten value under the same
    # field. Writing back re-hashes the key and compares it with the retained one, which is another
    # caller-owned step - bounded reads do not make later writes safe on their own.
    for name in ("payload", "metadata"):
        value = _take(raw, name, message, code)
        if value is not _MISSING:
            _store_field(raw, name, _thaw(value, message, code), message, code)
    location: tuple[str | int, ...] = ()
    try:
        envelope = ActionSpec.model_validate(raw)
    except ValidationError as exc:
        location = _safe_location(exc, _DOMAIN_FIELDS | _ENVELOPE_FIELDS)
    except Exception:  # noqa: BLE001 - an unexpected dependency failure is converted to a bounded Web error
        location = ()
    else:
        return envelope
    raise WebValidationError(message, code=code, path=location)


def _action_payload(envelope: ActionSpec) -> dict:
    """Validate the payload the action type selects and return its isolated plain document."""
    model = _PAYLOAD_MODELS[envelope.action_type]
    message = "the action payload does not match the Web action schema"
    location: tuple[str | int, ...] = ()
    try:
        validated = model.model_validate(envelope.payload)
    except ValidationError as exc:
        location = _safe_location(exc, _DOMAIN_FIELDS)
    except Exception:  # noqa: BLE001 - an unexpected dependency failure is converted to a bounded Web error
        location = ()
    else:
        return validated.model_dump(mode="python")
    raise WebActionPayloadError(message, path=location)


def _revalidated_action(action: object) -> ActionSpec:
    """Revalidate an action and return a detached Core ``ActionSpec``."""
    envelope = _action_envelope(action)
    if envelope.action_type not in WEB_ACTION_TYPES:
        raise UnsupportedActionError("the Web contract does not support this action type")
    payload = _action_payload(envelope)
    return ActionSpec(
        schema_version=envelope.schema_version,
        action_type=envelope.action_type,
        payload=payload,
        metadata=envelope.metadata,
    )


def _observation_envelope(observation: object) -> Observation:
    """Revalidate the Core observation envelope, whatever shape it arrives in."""
    code = "WEB_OBSERVATION_INVALID"
    message = "the observation does not match the Core observation envelope"
    if isinstance(observation, Observation):
        raw = _stored_fields(observation, _OBSERVATION_FIELDS, message, code)
    elif isinstance(observation, Mapping):
        raw = _mapping_fields(observation, _OBSERVATION_FIELDS, message, code)
    else:
        raise WebValidationError(message, code=code)
    # Same chain as the action envelope: the rebuilt document keeps the caller's key objects, so
    # each later lookup and each writeback is bounded, and the sentinel separates absence from a
    # stored ``None``.
    state = _take(raw, "state", message, code)
    if state is not _MISSING:
        _store_field(raw, "state", _thaw(state, message, code), message, code)
    artifacts = _take(raw, "artifacts", message, code)
    if artifacts is not _MISSING:
        _store_field(raw, "artifacts", _artifact_items(artifacts, message, code), message, code)
    location: tuple[str | int, ...] = ()
    try:
        envelope = Observation.model_validate(raw)
    except ValidationError as exc:
        location = _safe_location(exc, _OBSERVATION_FIELDS)
    except Exception:  # noqa: BLE001 - an unexpected dependency failure is converted to a bounded Web error
        location = ()
    else:
        return envelope
    raise WebValidationError(message, code=code, path=location)


def parse_web_action(action: ActionSpec | Mapping[str, object]) -> ActionSpec:
    """Revalidate a Web action and return a detached Core ``ActionSpec``.

    ``action`` may be a Core ``ActionSpec`` - including the frozen Builder snapshot - or a plain
    mapping. The returned record is a new Core ``ActionSpec`` whose payload and metadata are
    detached from the supplied object, so editing either side afterwards cannot reach the other.
    This is not a new contract and it neither submits nor executes anything.
    """
    return _revalidated_action(action)


def validate_web_action(action: ActionSpec | Mapping[str, object]) -> None:
    """Reject an unsupported action type or a malformed Web payload.

    The decision order and the raised errors are exactly those of :func:`parse_web_action`. The
    call performs no I/O, changes no state and does not modify the supplied object.
    """
    _revalidated_action(action)


def parse_web_observation(
    observation: Observation | Mapping[str, object],
) -> WebObservationState:
    """Revalidate a Web observation and return an isolated snapshot of its ``state``.

    The outer Core identity - ``observation_id``, ``timestamp``, ``artifacts``, ``metadata`` - is
    validated but never forged, copied or replaced: only the Web snapshot state is returned.
    """
    envelope = _observation_envelope(observation)
    if envelope.environment_type != WEB_ENVIRONMENT_TYPE:
        raise WebValidationError(
            "the observation does not declare the web environment type",
            code="WEB_OBSERVATION_INVALID",
            path=("environment_type",),
        )
    location: tuple[str | int, ...] = ()
    try:
        snapshot = WebObservationState.model_validate(envelope.state)
    except ValidationError as exc:
        location = _safe_location(exc, _DOMAIN_FIELDS)
    except Exception:  # noqa: BLE001 - an unexpected dependency failure is converted to a bounded Web error
        location = ()
    else:
        return snapshot
    raise WebValidationError(
        "the observation state does not match the Web snapshot schema",
        code="WEB_OBSERVATION_INVALID",
        path=location,
    )
