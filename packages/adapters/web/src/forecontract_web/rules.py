"""The five pure Web verification rules over already-collected observation evidence.

:class:`WebRuleEvaluator` is synchronous, deterministic and stateless. It never opens a browser,
queries live DOM, waits for a page, repairs evidence or calls a model. Its only input is one
piece of the existing Core ``VerificationEvidence`` whose observation carries a ``web.snapshot``
state, plus one closed rule payload.

The order of work is fixed: revalidate the whole evidence, decide the rule type, revalidate the
closed payload, select the exact target record, then apply the truth table. Damaged evidence or
a malformed rule raises one of the two fixed facility errors of this round and is never reported
as a verdict:

* ``WEB_RULE_INVALID`` for an unsupported rule type or a payload that does not match its schema;
* ``WEB_VERIFICATION_INPUT_INVALID`` for an outer contract, criteria or evidence value that is
  not usable.

Both errors are carried by subclasses of the existing
:class:`~forecontract_web.errors.WebValidationError`; they add the two fixed codes of this round
without changing the existing Web error module.

The truth table is the approved one. ``web.url_match`` compares the whole original page URL
string exactly (``about:blank`` included) and never trims, normalises, ignores a query/fragment
or treats a prefix as a match. ``web.text_contains`` is an original case-sensitive substring
test with a non-empty expected text. ``web.value_equals`` compares the exact string and allows
an empty expected value. An obtained empty text/value and an explicit ``visible=false`` are
known evidence, not ``null``, and visibility is never used as an implicit gate for the text or
value rules.

Selection is by the locator's exact declared fields, never by list position. A structurally
valid snapshot with no URL, no record for the target, an unavailable query, a required
attribute that is ``null`` or several matches is ``INCONCLUSIVE``. A successful query that
matched zero nodes is explicit negative evidence: ``web.element_hidden`` passes, while
``web.element_visible``, ``web.text_contains`` and ``web.value_equals`` fail with
``TARGET_NOT_FOUND``. A unique match decides normally from the known property.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Final

from forecontract.contracts.models import RecoveryPolicy
from forecontract.models.observations import Observation
from forecontract.models.outcomes import VerificationStatus
from forecontract.verification.models import (
    VerificationEvidence,
    validate_declared_instance,
)
from pydantic import BaseModel, ValidationError

from forecontract_web.errors import (
    _FIELD_PLACEHOLDER,
    WebValidationError,
    _safe_location,
)
from forecontract_web.models import (
    RoleTarget,
    TestIdTarget,
    WebObservationState,
    WebTargetEvidence,
)
from forecontract_web.rule_models import (
    EVIDENCE_INSUFFICIENT,
    PAYLOAD_FIELDS,
    RULE_MATCH,
    RULE_MISMATCH,
    TARGET_FIELDS,
    TARGET_NOT_FOUND,
    WEB_RULE_TYPES,
    ElementHiddenPayload,
    ElementVisiblePayload,
    TextContainsPayload,
    UrlMatchPayload,
    ValueEqualsPayload,
    WebRuleEvaluation,
)

__all__ = ["WebRuleEvaluator", "WebRuleInvalid", "WebVerificationInputInvalid"]

#: The five rule names, in their declared order.
_URL_MATCH: Final[str] = "web.url_match"
_ELEMENT_VISIBLE: Final[str] = "web.element_visible"
_ELEMENT_HIDDEN: Final[str] = "web.element_hidden"
_TEXT_CONTAINS: Final[str] = "web.text_contains"
_VALUE_EQUALS: Final[str] = "web.value_equals"

#: The closed payload model each rule type selects.
_PAYLOAD_MODELS: Final[dict[str, type[BaseModel]]] = {
    _URL_MATCH: UrlMatchPayload,
    _ELEMENT_VISIBLE: ElementVisiblePayload,
    _ELEMENT_HIDDEN: ElementHiddenPayload,
    _TEXT_CONTAINS: TextContainsPayload,
    _VALUE_EQUALS: ValueEqualsPayload,
}

#: The declared payload field names, as a fixed tuple for deterministic key probing.
_PAYLOAD_FIELDS: Final[tuple[str, ...]] = tuple(sorted(PAYLOAD_FIELDS))
#: The declared locator field names, as a fixed tuple for deterministic key probing.
_TARGET_FIELDS: Final[tuple[str, ...]] = tuple(sorted(TARGET_FIELDS))

#: The closed evidence and envelope field names.
_EVIDENCE_FIELDS: Final[tuple[str, ...]] = tuple(sorted(VerificationEvidence.model_fields))
_OBSERVATION_FIELDS: Final[tuple[str, ...]] = tuple(sorted(Observation.model_fields))

#: The exact environment type a Web observation must declare.
_WEB_ENVIRONMENT: Final[str] = "web"
_SCHEMA_VERSION: Final[str] = "0.1"

_RULE_MESSAGE: Final[str] = "the web rule payload does not match the selected rule schema"
_EVIDENCE_MESSAGE: Final[str] = "the verification evidence is not usable"


class WebRuleInvalid(WebValidationError):
    """A rule type or rule payload rejected by this round's fixed ``WEB_RULE_INVALID`` code."""

    code = "WEB_RULE_INVALID"

    def __init__(self, message: str, *, path: tuple[str | int, ...] = ()) -> None:
        super().__init__(message, path=path)
        # Keep the fixed code visible as an instance value too, like the existing Web errors.
        self.code = type(self).code


class WebVerificationInputInvalid(WebValidationError):
    """Unusable contract, criteria, evidence or binding input.

    The fixed code is ``WEB_VERIFICATION_INPUT_INVALID``; the error is still an existing
    :class:`~forecontract_web.errors.WebValidationError`.
    """

    code = "WEB_VERIFICATION_INPUT_INVALID"

    def __init__(self, message: str, *, path: tuple[str | int, ...] = ()) -> None:
        super().__init__(message, path=path)
        self.code = type(self).code


def _bounded(read, message: str, error: type[WebValidationError],
             path: tuple[str | int, ...] = ()):
    """Run one caller-owned read and convert any ordinary failure into a fixed facility error.

    Process-control exceptions are ``BaseException`` and are never caught here. The replacement
    is raised after the handler has finished, so it carries neither the original as ``__cause__``
    nor as ``__context__``.
    """
    try:
        return read()
    except Exception:  # noqa: BLE001, S110 - converted below, outside the handler
        pass
    raise error(message, path=path)


def _real_instance(record: object, model: type[BaseModel]) -> bool:
    """Whether ``record`` really is an instance of ``model``, without reading a caller attribute.

    ``isinstance`` falls back to the instance's ``__class__`` attribute whenever the real type is
    not a subtype of the model, and that lookup executes a caller ``__class__`` property or
    ``__getattribute__``. A foreign object can therefore raise there, and a genuine model subclass
    can carry a hostile ``__class__`` property that the classification itself would run. The
    built-in ``type`` reads the real ``Py_TYPE`` slot only and ``issubclass`` compares that real
    type against the trusted model class, so neither operation calls a caller attribute while a
    real subclass is still accepted.
    """
    return issubclass(type(record), model)


def _declared_name(key: object, declared: tuple[str, ...]) -> str | None:
    """Return the declared field name a caller key spells, or ``None``.

    The test starts from the object's real ``type`` rather than ``isinstance`` so a caller
    ``__class__`` property cannot run, and content comparison uses ``str.__eq__`` directly so a
    ``str`` subclass cannot change which declared field its text denotes. Only a plain built-in
    name is ever returned and no caller-owned read, comparison or hash hook executes here.
    """
    if not issubclass(type(key), str):
        return None
    for name in declared:
        if str.__eq__(key, name):
            return name
    return None


def _plain_fields(items: object, declared: tuple[str, ...], message: str,
                  error: type[WebValidationError],
                  path: tuple[str | int, ...] = ()) -> dict:
    """Build one plain document from caller items, rejecting unknown or repeated field names.

    The caller items are already materialised into a list or tuple by the call site, so this
    loop reads only built-in containers. Iterating or unpacking an entry is still a caller-owned
    conversion (a general ``Mapping`` may yield any objects), so any ordinary failure here is
    converted into the fixed facility error as well. The error is raised after the handler has
    finished, so it carries neither the original as ``__cause__`` nor as ``__context__``.
    """
    fields: dict = {}
    failure: tuple[str | int, ...] | None = None
    try:
        for entry in items:
            key, value = entry
            name = _declared_name(key, declared)
            if name is None or name in fields:
                failure = path + (_FIELD_PLACEHOLDER,)
                break
            fields[name] = value
    except Exception:  # noqa: BLE001 - a caller-owned iteration/conversion failure
        failure = path + (_FIELD_PLACEHOLDER,)
    if failure is not None:
        raise error(message, path=failure)
    return fields


def _base_storage_descriptor(name: str) -> object | None:
    """Return the base ``BaseModel`` descriptor of one storage slot, or ``None``.

    Pydantic stores the declared fields in the instance ``__dict__`` and an allowed extra
    payload in ``__pydantic_extra__``; both slots are described by a C-level getset/member
    descriptor on ``BaseModel``. Reading the slot through that descriptor returns the real
    stored object and bypasses any property a caller subclass declares for the same name.
    """
    for klass in BaseModel.__mro__:
        candidate = vars(klass).get(name)
        if isinstance(candidate, (types.GetSetDescriptorType, types.MemberDescriptorType)):
            return candidate
    return None


_INSTANCE_FIELDS_DESCRIPTOR: Final[object | None] = _base_storage_descriptor("__dict__")
_EXTRA_STORE_DESCRIPTOR: Final[object | None] = _base_storage_descriptor("__pydantic_extra__")


def _stored_slot(record: BaseModel, name: str, descriptor: object | None) -> object:
    """Read one real stored slot, falling back to the ordinary read when no descriptor exists."""
    if descriptor is None:
        return object.__getattribute__(record, name)
    return descriptor.__get__(record, type(record))  # type: ignore[attr-defined]


def _stored_entries(store: dict) -> tuple[tuple[object, object], ...]:
    """Enumerate the entries a stored dict *really* holds, never a caller mapping view.

    ``dict.items`` reads the underlying table of the stored dict itself, so an override of
    ``__iter__``/``keys``/``items``/``__getitem__`` - or a lying ``__bool__``/``__len__`` - on
    a dict subclass cannot hide an entry that is actually stored. The read always executes
    inside the caller-owned read boundary of its call site.
    """
    return tuple(dict.items(store))


def _extra_entries(extra: object) -> tuple[tuple[object, object], ...]:
    """Return the actual entries of a stored extra store; ``None`` and ``{}`` mean 'none'."""
    if extra is None:
        return ()
    if not isinstance(extra, dict):
        raise TypeError("an extra store must be a dict or None")
    return tuple(dict.items(extra))


def _stored_fields(record: BaseModel, declared: tuple[str, ...], message: str,
                   error: type[WebValidationError],
                   path: tuple[str | int, ...] = ()) -> dict:
    """Read a record's actual stored fields, rejecting an undeclared or extra stored key.

    The instance ``__dict__`` and ``__pydantic_extra__`` are read through the base storage
    descriptors and the field mapping is enumerated from its real underlying table. A caller
    truthiness answer, an overridable mapping view and a detached copy that can omit an entry
    are never trusted: a non-empty extra store whose ``__bool__`` returns false, an extra store
    whose ``__bool__`` raises and a field mapping whose ``__iter__`` hides a key are all read
    as what is actually stored and refused with the fixed facility error.
    """
    def read() -> tuple[object, object] | None:
        stored = _stored_slot(record, "__dict__", _INSTANCE_FIELDS_DESCRIPTOR)
        if not isinstance(stored, dict):
            return None
        extra = _stored_slot(record, "__pydantic_extra__", _EXTRA_STORE_DESCRIPTOR)
        return stored, extra

    result = _bounded(read, message, error, path)
    if result is None:
        raise error(message, path=path)
    stored, extra = result
    extra_entries = _bounded(
        lambda: _extra_entries(extra), message, error, path + (_FIELD_PLACEHOLDER,)
    )
    if extra_entries:
        raise error(message, path=path + (_FIELD_PLACEHOLDER,))
    entries = _bounded(lambda: _stored_entries(stored), message, error, path)
    fields: dict = {}
    for key, value in entries:
        name = _declared_name(key, declared)
        if name is None or name in fields:
            raise error(message, path=path + (_FIELD_PLACEHOLDER,))
        fields[name] = value
    return fields


def _caller_key_hook(key: object) -> bool:
    """Whether inserting a caller key would execute a caller-defined hash or equality hook.

    A plain built-in ``str`` - or a ``str`` subclass that inherits ``str.__hash__`` and
    ``str.__eq__`` unchanged - has no caller hook to run, so it is a provably hook-free key.
    Any other key would execute caller code when the rebuilt document hashes or compares it;
    the reconstruction refuses such a key instead of executing that unsafe hook.
    """
    key_type = type(key)
    if key_type is str:
        return False
    if not issubclass(key_type, str):
        return True
    try:
        hasher = type.__getattribute__(key_type, "__hash__")
        comparer = type.__getattribute__(key_type, "__eq__")
    except Exception:  # noqa: BLE001 - an unreadable key type is a caller-owned failure
        return True
    return hasher is not str.__hash__ or comparer is not str.__eq__


def _plain_mapping(value: Mapping, message: str, error: type[WebValidationError],
                   path: tuple[str | int, ...], active: set[int]) -> dict:
    """Build a plain dict from a stored mapping, never from a caller-owned view.

    A dict-backed mapping is enumerated from its real underlying table; any other general
    ``Mapping`` is read through its ordinary ``items`` protocol inside a caller-owned read
    boundary. Appending each entry is a caller-owned hash/equality/insertion operation too: a
    key type that defines its own hash or equality hook is refused *without* executing it, and
    a key whose ``__hash__``/``__eq__`` raises, or an entry that cannot be unpacked, becomes
    the fixed facility error with no reachable cause/context and no raw value.
    """
    if issubclass(type(value), dict):
        entries = _bounded(lambda: _stored_entries(value), message, error, path)
    else:
        entries = _bounded(lambda: tuple(value.items()), message, error, path)
    document: dict = {}
    for entry in entries:
        blocked = False
        try:
            key, item = entry
        except Exception:  # noqa: BLE001 - a caller-owned entry conversion failure
            blocked = True
        if blocked:
            raise error(message, path=path)
        if _caller_key_hook(key):
            raise error(message, path=path)
        converted = _plain_value(item, message, error, path, active)
        try:
            document[key] = converted
        except Exception:  # noqa: BLE001 - a caller-owned hash/equality/insertion failure
            blocked = True
        if blocked:
            raise error(message, path=path)
    return document


def _plain_sequence(value: object, message: str, error: type[WebValidationError],
                    path: tuple[str | int, ...], active: set[int]) -> list:
    """Build a plain list from a stored sequence, enumerating its real items."""
    identity = id(value)
    if identity in active:
        raise error(message, path=path)
    active.add(identity)
    try:
        if issubclass(type(value), list):
            iterator = list.__iter__(value)
        else:
            iterator = tuple.__iter__(value)  # type: ignore[arg-type]
        items = _bounded(lambda: list(iterator), message, error, path)
        return [_plain_value(item, message, error, path, active) for item in items]
    finally:
        active.discard(identity)


def _plain_value(value: object, message: str, error: type[WebValidationError],
                 path: tuple[str | int, ...], active: set[int]) -> object:
    """Convert one stored field value into plain data without calling a caller serializer.

    The dispatch starts from the object's real ``type`` instead of ``isinstance`` so a caller
    ``__class__`` property cannot run, and therefore cannot raise or redirect the conversion.
    """
    value_type = type(value)
    if issubclass(value_type, BaseModel):
        declared = _bounded(
            lambda: tuple(sorted(value_type.model_fields)), message, error, path
        )
        return _plain_document(value, declared, message, error, path, active)
    if issubclass(value_type, Mapping):
        return _plain_mapping(value, message, error, path, active)  # type: ignore[arg-type]
    if issubclass(value_type, (list, tuple)):
        return _plain_sequence(value, message, error, path, active)
    return value


def _plain_document(record: BaseModel, declared: tuple[str, ...], message: str,
                    error: type[WebValidationError], path: tuple[str | int, ...],
                    active: set[int]) -> dict:
    """Build one plain document from a record's actual stored fields.

    This replaces the caller-owned ``model_dump`` that used to build the revalidation input:
    the document is assembled from the real storage descriptors and the real entries of every
    nested mapping, so an overridden serializer cannot rewrite a stored URL or goal, and it
    cannot drop an undeclared key either. ``active`` holds only the identities on the current
    descent, so a genuine cycle is a bounded error while a value legally referenced twice is
    still accepted.

    A nested ``RecoveryPolicy`` is identified by its real type - never by ``isinstance`` on the
    caller-supplied record, whose ``__class__`` property could run - and is checked from its
    actual stored fields *before* the legal abort projection: an abort policy whose real
    ``max_retries``/``allow_replan`` slot stores a non-null value is refused with the fixed
    facility error instead of being silently rewritten as the two-field abort shape. Only an
    actually null/absent control is projected away, which keeps the ordinary default abort record
    and the legal bounded policy accepted.
    """
    identity = id(record)
    if identity in active:
        raise error(message, path=path)
    active.add(identity)
    try:
        fields = _stored_fields(record, declared, message, error, path)
        document: dict = {}
        for name, value in fields.items():
            document[name] = _plain_value(
                value, message, error, path + (name,), active
            )
        if _real_instance(record, RecoveryPolicy):
            mode = fields.get("mode")
            if issubclass(type(mode), str) and str.__eq__(mode, "abort"):
                for control in ("max_retries", "allow_replan"):
                    if fields.get(control) is not None:
                        raise error(message, path=path + (control,))
                # The declared RecoveryPolicy serializer emits only ``schema_version`` and
                # ``mode`` for an aborting policy and omits its absent bounded controls, so
                # rebuilding that policy through its own schema needs the same two-field
                # shape rather than two explicit nulls it would reject as supplied controls.
                document.pop("max_retries", None)
                document.pop("allow_replan", None)
        return document
    finally:
        active.discard(identity)


def _revalidated_model(record: object, model: type[BaseModel], message: str,
                       error: type[WebValidationError],
                       path: tuple[str | int, ...] = ()) -> BaseModel:
    """Revalidate one Core record from its actual stored content and return a detached rebuild.

    The caller-owned serializer is never consulted, so a valid subclass that overrides
    ``model_dump`` cannot rewrite the actual stored URL or goal. The record is classified by its
    real type only, so a foreign object whose ``__class__``/``__getattribute__`` raises is refused
    with the fixed error instead of leaking the caller exception, while a genuine subclass is
    still accepted. The whole record is then checked by the Core stored-shape validator, and a
    plain document is built from the real storage descriptors and validated as a fresh instance:
    an undeclared stored key, an illegal stored shape, an unreadable store or a schema rejection
    becomes the fixed error.
    """
    if not _real_instance(record, model):
        raise error(message, path=path)
    declared = tuple(sorted(model.model_fields))
    _bounded(
        lambda: validate_declared_instance(record, declared), message, error, path
    )
    document = _plain_document(record, declared, message, error, path, set())

    def rebuild() -> BaseModel:
        return model.model_validate(document)

    return _bounded(rebuild, message, error, path)


def _payload_fields(payload: object, path: tuple[str | int, ...]) -> dict:
    """Read one closed rule payload from a mapping or from a committed payload model.

    A dict-backed payload is enumerated from its real underlying table (``dict.items``), never
    from the caller's ``items``/``__iter__``/``keys`` view, so a ``dict`` subclass that really
    stores an undeclared entry cannot hide it. A general non-dict ``Mapping`` keeps using its
    ordinary ``items`` protocol inside the caller-owned read boundary.
    """
    if issubclass(type(payload), tuple(_PAYLOAD_MODELS.values())):
        return _stored_fields(
            payload, _PAYLOAD_FIELDS, _RULE_MESSAGE, WebRuleInvalid, path
        )
    if issubclass(type(payload), Mapping):
        if issubclass(type(payload), dict):
            items = _bounded(
                lambda: _stored_entries(payload), _RULE_MESSAGE, WebRuleInvalid, path
            )
        else:
            items = _bounded(
                lambda: list(payload.items()), _RULE_MESSAGE, WebRuleInvalid, path
            )
        return _plain_fields(items, _PAYLOAD_FIELDS, _RULE_MESSAGE, WebRuleInvalid, path)
    raise WebRuleInvalid(_RULE_MESSAGE, path=path)


def _target_fields(value: object, path: tuple[str | int, ...]) -> object:
    """Return a plain locator document, rejecting an undeclared stored locator key.

    As with the payload, a dict-backed locator is read from its real table so an overridden
    mapping view cannot hide an unknown entry; an ordinary locator model is read through its
    storage descriptors.
    """
    if issubclass(type(value), (TestIdTarget, RoleTarget)):
        return _stored_fields(value, _TARGET_FIELDS, _RULE_MESSAGE, WebRuleInvalid, path)
    if issubclass(type(value), Mapping):
        if issubclass(type(value), dict):
            items = _bounded(
                lambda: _stored_entries(value), _RULE_MESSAGE, WebRuleInvalid, path
            )
        else:
            items = _bounded(
                lambda: list(value.items()), _RULE_MESSAGE, WebRuleInvalid, path
            )
        return _plain_fields(items, _TARGET_FIELDS, _RULE_MESSAGE, WebRuleInvalid, path)
    return value


def _prepare_rule(rule_type: object, payload: object, *,
                  location: tuple[str | int, ...] = ()) -> tuple[str, BaseModel]:
    """Decide the rule type and revalidate its closed payload before any verdict is produced."""
    selected = _bounded(
        lambda: next(
            (name for name in WEB_RULE_TYPES
             if issubclass(type(rule_type), str) and str.__eq__(rule_type, name)),
            None,
        ),
        _RULE_MESSAGE,
        WebRuleInvalid,
        location + ("rule_type",),
    )
    if selected is None:
        raise WebRuleInvalid(
            "the web rule type is not supported", path=location + ("rule_type",)
        )
    model = _PAYLOAD_MODELS[selected]
    fields = _payload_fields(payload, location)
    if "target" in fields:
        fields["target"] = _target_fields(fields["target"], location)
    try:
        prepared = model.model_validate(fields)
    except ValidationError as exc:
        spot = _safe_location(exc, PAYLOAD_FIELDS | TARGET_FIELDS)
    except Exception:  # noqa: BLE001 - an unexpected dependency failure is a bounded rule error
        spot = ()
    else:
        return selected, prepared
    raise WebRuleInvalid(_RULE_MESSAGE, path=location + spot)


def _revalidate_evidence(evidence: object) -> tuple[Observation, WebObservationState, bool]:
    """Revalidate the whole evidence and return its detached observation and Web snapshot."""
    rebuilt = _revalidated_model(
        evidence, VerificationEvidence, _EVIDENCE_MESSAGE,
        WebVerificationInputInvalid, ("evidence",),
    )
    observation = rebuilt.observation
    if not _same_text(observation.environment_type, _WEB_ENVIRONMENT):
        raise WebVerificationInputInvalid(
            "the observation does not declare the web environment type",
            path=("evidence", "observation", "environment_type"),
        )
    state = _bounded(
        lambda: WebObservationState.model_validate(observation.state),
        _EVIDENCE_MESSAGE,
        WebVerificationInputInvalid,
        ("evidence", "observation", "state"),
    )
    return observation, state, rebuilt.state_complete


def _same_text(left: object, right: object) -> bool:
    """Exact content equality of two strings, independent of a subclass's own comparison."""
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and bool(str.__eq__(left, right))
    )


def _same_target(left: object, right: object) -> bool:
    """Exact declared-field identity of two locator records, independent of key order."""
    if type(left) is not type(right):
        return False
    if isinstance(left, TestIdTarget):
        return _same_text(left.strategy, right.strategy) and _same_text(
            left.test_id, right.test_id
        )
    return (
        _same_text(left.strategy, right.strategy)
        and _same_text(left.role, right.role)
        and _same_text(left.name, right.name)
    )


def _find_target(state: WebObservationState,
                 target: object) -> tuple[int, WebTargetEvidence] | None:
    """Select the one record whose locator fields exactly equal the payload target."""
    for index, record in enumerate(state.targets):
        if _same_target(record.target, target):
            return index, record
    return None


def _result(rule_type: str, status: VerificationStatus, reason: str,
            observation_id: str, target_index: int | None) -> WebRuleEvaluation:
    """Build the bounded verdict record for one evaluated rule."""
    return WebRuleEvaluation(
        schema_version=_SCHEMA_VERSION,
        rule_type=rule_type,
        status=status,
        reason=reason,
        observation_id=observation_id,
        target_index=target_index,
    )


def _evaluate_prepared(prepared: tuple[str, BaseModel], state: WebObservationState,
                       observation_id: str) -> WebRuleEvaluation:
    """Apply the approved truth table to one already validated rule and snapshot."""
    rule_type, payload = prepared
    if rule_type == _URL_MATCH:
        url = state.url
        if url is None:
            return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                           EVIDENCE_INSUFFICIENT, observation_id, None)
        if _same_text(url, payload.url):
            return _result(rule_type, VerificationStatus.PASS, RULE_MATCH,
                           observation_id, None)
        return _result(rule_type, VerificationStatus.FAIL, RULE_MISMATCH,
                       observation_id, None)

    found = _find_target(state, payload.target)
    if found is None:
        # The snapshot did not provide this target; a missing record is not a zero match.
        return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                       EVIDENCE_INSUFFICIENT, observation_id, None)
    index, record = found
    if _same_text(record.status, "unavailable"):
        return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                       EVIDENCE_INSUFFICIENT, observation_id, index)
    count = record.match_count
    if count == 1:
        return _decide_unique(rule_type, payload, record, observation_id, index)
    if count == 0:
        if rule_type == _ELEMENT_HIDDEN:
            return _result(rule_type, VerificationStatus.PASS, RULE_MATCH,
                           observation_id, index)
        return _result(rule_type, VerificationStatus.FAIL, TARGET_NOT_FOUND,
                       observation_id, index)
    # Several matched nodes are explicit ambiguity evidence, never "pick the first".
    return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                   EVIDENCE_INSUFFICIENT, observation_id, index)


def _decide_unique(rule_type: str, payload: BaseModel, record: WebTargetEvidence,
                   observation_id: str, index: int) -> WebRuleEvaluation:
    """Decide one rule from the known properties of its single matched node."""
    if rule_type == _ELEMENT_VISIBLE:
        if record.visible is None:
            return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                           EVIDENCE_INSUFFICIENT, observation_id, index)
        if record.visible:
            return _result(rule_type, VerificationStatus.PASS, RULE_MATCH,
                           observation_id, index)
        return _result(rule_type, VerificationStatus.FAIL, RULE_MISMATCH,
                       observation_id, index)
    if rule_type == _ELEMENT_HIDDEN:
        if record.visible is None:
            return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                           EVIDENCE_INSUFFICIENT, observation_id, index)
        if not record.visible:
            return _result(rule_type, VerificationStatus.PASS, RULE_MATCH,
                           observation_id, index)
        return _result(rule_type, VerificationStatus.FAIL, RULE_MISMATCH,
                       observation_id, index)
    if rule_type == _TEXT_CONTAINS:
        text = record.text
        if text is None:
            return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                           EVIDENCE_INSUFFICIENT, observation_id, index)
        if bool(str.__contains__(text, payload.text)):
            return _result(rule_type, VerificationStatus.PASS, RULE_MATCH,
                           observation_id, index)
        return _result(rule_type, VerificationStatus.FAIL, RULE_MISMATCH,
                       observation_id, index)
    value = record.value
    if value is None:
        return _result(rule_type, VerificationStatus.INCONCLUSIVE,
                       EVIDENCE_INSUFFICIENT, observation_id, index)
    if _same_text(value, payload.value):
        return _result(rule_type, VerificationStatus.PASS, RULE_MATCH,
                       observation_id, index)
    return _result(rule_type, VerificationStatus.FAIL, RULE_MISMATCH,
                   observation_id, index)


class WebRuleEvaluator:
    """Evaluate one of the five closed Web rules against one piece of supplied evidence.

    The class holds no state and takes no constructor argument, so repeated calls with equal
    arguments always produce equal results and no callback, plugin or live page can be injected.
    """

    def evaluate(self, *, rule_type: str, payload: object,
                 evidence: VerificationEvidence) -> WebRuleEvaluation:
        """Return the bounded verdict of one Web rule for the supplied evidence."""
        observation, state, _complete = _revalidate_evidence(evidence)
        prepared = _prepare_rule(rule_type, payload)
        return _evaluate_prepared(prepared, state, observation.observation_id)
