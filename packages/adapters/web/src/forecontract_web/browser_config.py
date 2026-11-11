"""Strict configuration and local-origin policy for the Playwright Web adapter.

``PlaywrightWebConfig`` carries four required fields plus two optional, both explicitly absent by
default:

* ``allowed_origins`` - a non-empty list of absolute ``http``/``https`` origins whose host is a
  loopback name (``localhost``, ``127.0.0.1`` or ``::1``) and whose canonical
  ``(scheme, host, port)`` tuple is unique. A path is not accepted (one trailing slash is), and
  user information, a query, a fragment, another scheme and a remote host are rejected. The
  supplied text is preserved as evidence; permission checks compare the canonical origin tuples,
  never a string prefix or substring.
* ``observation_targets`` - the explicit, unique locators (``TestIdTarget`` or ``RoleTarget``)
  an observation may query. The list may be empty.
* ``operation_timeout_ms`` / ``cleanup_timeout_ms`` - strict, finite, positive JSON numbers.
  Booleans, strings, ``NaN``, infinities, zero and negative numbers are rejected, and there is
  no hidden default.
* ``observation_target_timeout_ms`` - an optional per-target read allowance for one observation.
  ``None`` (the default) keeps the historical behaviour: each target is bounded only by the whole
  operation deadline. A supplied value must be a strict, finite, positive millisecond number
  smaller than ``operation_timeout_ms``; it shortens only a single target's read opportunity and
  never extends the whole ``observe`` call.
* ``startup_timeout_ms`` - an optional allowance for the browser resource acquisition of one
  ``start``. ``None`` (the default) keeps the historical behaviour: ``start`` is bounded by the
  existing ``operation_timeout_ms`` value. A supplied value obeys the same strict, finite,
  positive JSON-number rule (booleans, strings, ``NaN``, infinities, zero and negative numbers are
  rejected) and may legitimately be larger than ``operation_timeout_ms``. It is a *startup*
  allowance only: it bounds the one browser acquisition attempt with a single absolute deadline
  and never widens or shortens ``observe``, ``execute``, the per-target read allowance, the
  cleanup allowance, the Core Run deadline or any other budget.

Revalidation never trusts a supplied instance. ``PlaywrightWebConfig`` instances are rebuilt
from their stored fields, and a target supplied as an existing model instance is rebuilt from
its own stored fields, so ``model_construct``, ``model_copy`` or a mutated backing list cannot
smuggle an invalid value past the adapter constructor. Supplied strings and mapping keys are
read through their original stored characters, so a caller ``__str__`` override can neither
launder an origin nor rename an unknown key; ordinary str subclasses, mappings and locator
models remain accepted. A stored locator's ``__dict__`` and ``__pydantic_extra__`` are read
through their real stored structure - the instance's actual field mapping and the actual entries
of its extra store - never through a caller truthiness answer, an overridable mapping view or a
detached copy that can differ from the real store; every such read executes inside the same
bounded read, so an attribute, slot, store or key read that raises becomes the fixed
configuration error instead of a raw caller exception. The stored state is an immutable tuple;
every public accessor returns a detached copy, so mutating the caller's list or model after
construction cannot change later behaviour.

Every rejection is a :class:`~forecontract_web.browser_errors.WebBrowserError` with the fixed
``WEB_BROWSER_CONFIG_INVALID`` code and no supplied value in its text or chain. A converted
caller-owned failure is raised only after its handler has finished, so the public error carries
neither ``__cause__`` nor ``__context__``.
"""

from __future__ import annotations

import math
import types
from collections.abc import Callable, Mapping
from typing import Final
from urllib.parse import urlsplit

from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.models import RoleTarget, TestIdTarget

__all__ = ["PlaywrightWebConfig"]

#: A canonical origin: the lower-case scheme, the loopback host and the effective port.
Origin = tuple[str, str, int]
#: Either supported locator model.
SupportedTarget = TestIdTarget | RoleTarget

_SCHEME_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1"})
_UNSET: Final[object] = object()
_CONFIG_CODE: Final[str] = "WEB_BROWSER_CONFIG_INVALID"
_CONFIG_OPERATION: Final[str] = "config"


def _invalid() -> WebBrowserError:
    """A fresh bounded configuration error with no supplied value."""
    return WebBrowserError(_CONFIG_CODE, _CONFIG_OPERATION)


def _attempt(call: Callable[[], object]) -> tuple[bool, object]:
    """Run one caller-owned read, converting an ordinary failure outside the handler.

    Only ``Exception`` is caught: a cancellation, a keyboard interrupt or a process exit is not
    a configuration read failure and is never swallowed.
    """
    try:
        return True, call()
    except Exception:  # noqa: BLE001 - converted to a bounded error by the caller
        return False, None


def _original_text(value: object) -> str:
    """Return a detached plain copy of a supplied string, or reject it.

    The characters actually stored in the supplied string are read through the base ``str``
    implementation, so a caller override of ``__str__`` (or of ``__getitem__``) can neither
    rename nor launder the value: the original identity is validated. Ordinary string subclasses
    remain accepted, and the returned value is always a plain detached ``str``.
    """
    if not isinstance(value, str):
        raise _invalid()
    ok, text = _attempt(lambda: str.__getitem__(value, slice(None)))
    if not ok or not isinstance(text, str) or type(text) is not str:
        raise _invalid()
    return text


def _port_number(text: str) -> int:
    """Parse one explicit decimal port literal, without whitespace or a sign."""
    if not text or len(text) > 5:
        raise _invalid()
    if not all("0" <= character <= "9" for character in text):
        raise _invalid()
    number = int(text)
    if number < 1 or number > 65535:
        raise _invalid()
    return number


def _authority_host_and_port(authority: str) -> tuple[str, int | None]:
    """Split one origin authority into its loopback host and optional explicit port."""
    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            raise _invalid()
        literal = authority[1:closing]
        remainder = authority[closing + 1 :]
        if literal.lower() != "::1":
            raise _invalid()
        if not remainder:
            return "::1", None
        if not remainder.startswith(":"):
            raise _invalid()
        return "::1", _port_number(remainder[1:])
    host, separator, port_text = authority.partition(":")
    if not host or ":" in port_text or "[" in host or "]" in host or "%" in host:
        raise _invalid()
    lowered = host.lower()
    if lowered not in _LOOPBACK_HOSTS:
        raise _invalid()
    if not separator:
        return lowered, None
    return lowered, _port_number(port_text)


def canonical_origin(text: str) -> Origin:
    """Return the canonical ``(scheme, host, effective port)`` tuple of one origin literal."""
    if not text:
        raise _invalid()
    for character in text:
        if character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F:
            raise _invalid()
    if "\\" in text:
        raise _invalid()
    scheme, separator, remainder = text.partition("://")
    if separator != "://" or scheme not in _SCHEME_PORTS:
        raise _invalid()
    if not remainder or "#" in remainder or "?" in remainder:
        raise _invalid()
    authority, slash, path = remainder.partition("/")
    if slash and path:
        raise _invalid()
    if not authority or "@" in authority:
        raise _invalid()
    host, explicit_port = _authority_host_and_port(authority)
    effective = explicit_port if explicit_port is not None else _SCHEME_PORTS[scheme]
    return scheme, host, effective


def request_origin_allowed(url: object, origins: frozenset[Origin]) -> bool:
    """Report whether one request URL belongs to an explicitly allowed local origin.

    The comparison is semantic: the URL is parsed and its scheme, user information, host and
    effective port are checked against the canonical tuples. ``about:blank`` is allowed because
    it is internal browser state, never a navigation payload. A non-``http``/``https`` scheme, a
    remote host, user information, a malformed authority or any control/whitespace/backslash
    text is refused.
    """
    if not isinstance(url, str) or not url:
        return False
    if url == "about:blank":
        return True
    for character in url:
        if character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F:
            return False
    if "\\" in url:
        return False
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
        userinfo_username = parts.username
        userinfo_secret = parts.password
    except ValueError:
        return False
    scheme = parts.scheme.lower()
    if scheme not in _SCHEME_PORTS:
        return False
    if userinfo_username is not None or userinfo_secret is not None:
        return False
    if not host:
        return False
    effective = port if port is not None else _SCHEME_PORTS[scheme]
    return (scheme, host.lower(), effective) in origins


def _validated_origins(value: object) -> tuple[tuple[str, ...], frozenset[Origin]]:
    """Validate the origin list and return its detached text and canonical tuple set."""
    if not isinstance(value, list):
        raise _invalid()
    ok, supplied = _attempt(lambda: list(value))
    if not ok:
        raise _invalid()
    if not supplied:  # type: ignore[truthy-bool]
        raise _invalid()
    texts: list[str] = []
    tuples: list[Origin] = []
    for item in supplied:  # type: ignore[union-attr]
        text = _original_text(item)
        origin = canonical_origin(text)
        if origin in tuples:
            raise _invalid()
        texts.append(text)
        tuples.append(origin)
    return tuple(texts), frozenset(tuples)


def _trusted_descriptor(name: str) -> object | None:
    """Return the base storage descriptor of one locator attribute, ignoring subclass views.

    A supported locator stores its declared fields in its instance ``__dict__`` and its extra
    keys in the ``__pydantic_extra__`` slot. The first matching C-level storage descriptor in the
    model MRO - a getset or a slot member descriptor, never a Python property - reads that real
    stored object, so a subclass-provided view cannot stand in for the instance's actual store.
    """
    for klass in TestIdTarget.__mro__:
        candidate = vars(klass).get(name)
        if isinstance(candidate, (types.GetSetDescriptorType, types.MemberDescriptorType)):
            return candidate
    return None


_INSTANCE_FIELDS_DESCRIPTOR: Final[object | None] = _trusted_descriptor("__dict__")
_EXTRA_STORE_DESCRIPTOR: Final[object | None] = _trusted_descriptor("__pydantic_extra__")


def _stored_slot(record: object, name: str, descriptor: object | None) -> object:
    """Read one real stored object through its trusted descriptor, or by an ordinary read.

    The descriptor (when the current Pydantic base exposes one) returns the instance's actual
    stored mapping or slot value, bypassing any subclass property that claims a different view.
    Without one, the ordinary attribute read is used unchanged. A missing or unset slot raises
    exactly like an ordinary attribute read; both paths stay caller-read bounded at the call
    site.
    """
    if descriptor is None:
        return object.__getattribute__(record, name)
    return descriptor.__get__(record, type(record))  # type: ignore[attr-defined]


def _stored_entries(store: dict) -> list[tuple[object, object]] | None:
    """Read the entries a stored dict *actually* holds, ignoring caller mapping views.

    A caller can assign a ``dict`` subclass as a model's ``__dict__`` or ``__pydantic_extra__``
    and override ``keys``/``items``/``__iter__`` (or ``__bool__``/``__len__``) so that the
    ordinary mapping protocol hides entries that are really stored. ``dict.items`` reads the
    underlying table of the stored dict itself, so the returned entries come from the real store
    rather than from a detached copy or an announced view. The read stays inside the
    caller-owned read boundary.
    """
    ok, entries = _attempt(lambda: list(dict.items(store)))
    if not ok or not isinstance(entries, list):
        return None
    return entries  # type: ignore[return-value]


def _stored_extra_has_entries(extra: object) -> bool | None:
    """Report whether a stored extra store actually holds entries; ``None`` means unreadable.

    The two shapes a supported locator model stores are ``None`` (no extra store at all) and a
    ``dict`` of additional keys. Presence is decided structurally: ``None`` is absent, a dict is
    read through its real underlying entries - never through ``bool(extra)``, ``len(extra)`` or
    an overridable ``keys``/``items`` view - and any other object is not a legal stored extra
    store. An unreadable or illegal store is reported as ``None`` so the caller raises the fixed
    error after this handler has finished.
    """
    if extra is None:
        return False
    if not isinstance(extra, dict):
        return None
    entries = _stored_entries(extra)
    if entries is None:
        return None
    return bool(entries)


def _stored_target_fields(record: TestIdTarget | RoleTarget) -> dict | None:
    """Read one locator model's actual stored fields, rejecting extra stored keys.

    ``__dict__`` holds the declared fields, while ``__pydantic_extra__`` holds keys a record
    built from an ``extra="allow"`` schema would keep. Both stores are revalidated by their real
    stored structure: the instance's actual field mapping and the actual entries of its extra
    store. A detached ``dict(stored)`` copy is never trusted, because a caller mapping subclass
    can make that copy omit a stored entry, and ``bool(extra)`` is never trusted either, because
    a caller store can announce emptiness while still holding a key. Every read executes inside
    :func:`_attempt`: a bypass-constructed record can neither hide an undeclared key nor make an
    attribute, slot, store or key read raise a raw caller exception outside the boundary. Every
    failure is reported as ``None`` and becomes the fixed ``WEB_BROWSER_CONFIG_INVALID`` error
    at the caller.
    """
    ok, stored = _attempt(
        lambda: _stored_slot(record, "__dict__", _INSTANCE_FIELDS_DESCRIPTOR)
    )
    if not ok or not isinstance(stored, dict):
        return None
    ok, extra = _attempt(
        lambda: _stored_slot(record, "__pydantic_extra__", _EXTRA_STORE_DESCRIPTOR)
    )
    if not ok:
        return None
    ok, has_extra = _attempt(lambda: _stored_extra_has_entries(extra))
    if not ok or has_extra is not False:
        return None
    entries = _stored_entries(stored)
    if entries is None:
        return None

    def build() -> dict:
        fields: dict = {}
        for key, item in entries:
            fields[_original_text(key)] = item
        return fields

    ok, fields = _attempt(build)
    if not ok:
        return None
    return fields


def _mapping_fields(value: Mapping) -> dict | None:
    """Materialise a plain mapping into a detached dictionary, or reject it.

    Every key is read through its original stored characters - never through a caller
    ``__str__`` - so an unknown key cannot be renamed into a declared field. An ordinary
    caller-owned read failure is reported as ``None`` and becomes the bounded configuration
    error at the caller, with no receiver from the original exception.
    """
    ok, items = _attempt(lambda: list(value.items()))
    if not ok:
        return None

    def build() -> dict:
        fields: dict = {}
        for key, item in items:  # type: ignore[union-attr]
            fields[_original_text(key)] = item
        return fields

    ok, fields = _attempt(build)
    if not ok:
        return None
    return fields  # type: ignore[return-value]


def _strategy_model(fields: dict) -> type[TestIdTarget | RoleTarget] | None:
    """Select the locator model from the original strategy value, or ``None`` when unknown.

    This call executes a caller-owned equality method, so it is invoked only inside
    :func:`_attempt`: an ordinary failure becomes the bounded configuration error instead of
    escaping as a raw exception.
    """
    strategy = fields.get("strategy")
    if strategy == "test_id":
        return TestIdTarget
    if strategy == "role":
        return RoleTarget
    return None


def _validated_target(value: object) -> SupportedTarget:
    """Rebuild one locator from its stored or supplied fields, revalidating every field."""
    if isinstance(value, (TestIdTarget, RoleTarget)):
        model: type[TestIdTarget | RoleTarget] = (
            TestIdTarget if isinstance(value, TestIdTarget) else RoleTarget
        )
        raw = _stored_target_fields(value)
        if raw is None:
            raise _invalid()
    elif isinstance(value, Mapping):
        raw = _mapping_fields(value)
        if raw is None:
            raise _invalid()
        ok, selected = _attempt(lambda: _strategy_model(raw))
        if not ok or selected is None:
            raise _invalid()
        model = selected  # type: ignore[assignment]
    else:
        raise _invalid()
    try:
        target = model.model_validate(raw)
    except Exception:  # noqa: BLE001 - converted to a bounded error below
        target = None
    if target is None:
        raise _invalid()
    return target


def _target_key(target: SupportedTarget) -> tuple[str, ...]:
    """The exact declared-field identity of a locator, independent of mapping key order."""
    return tuple(getattr(target, name) for name in sorted(type(target).model_fields))


def _validated_targets(value: object) -> tuple[SupportedTarget, ...]:
    """Validate the explicit target list and return a detached immutable copy."""
    if not isinstance(value, list):
        raise _invalid()
    ok, supplied = _attempt(lambda: list(value))
    if not ok:
        raise _invalid()
    targets: list[SupportedTarget] = []
    seen: set[tuple[str, ...]] = set()
    for item in supplied:  # type: ignore[union-attr]
        target = _validated_target(item)
        # The rebuilt identity is read and hashed inside the same boundary as every other
        # caller-owned read: a stored field object with a raising ``__hash__`` (or an unreadable
        # declared field) becomes the bounded configuration error, never a raw exception.
        ok, key = _attempt(lambda target=target: _target_key(target))
        if not ok or not isinstance(key, tuple):
            raise _invalid()
        ok, duplicate = _attempt(lambda key=key, seen=seen: key in seen)
        if not ok or duplicate:
            raise _invalid()
        ok, _added = _attempt(lambda key=key, seen=seen: seen.add(key))
        if not ok:
            raise _invalid()
        targets.append(target)
    return tuple(targets)


def _positive_timeout_ms(value: object) -> float:
    """Return a strict, finite, positive millisecond value."""
    if type(value) is not int and type(value) is not float:
        raise _invalid()
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number) or number <= 0:
        raise _invalid()
    return number


def _target_timeout_ms(value: object, operation_timeout_ms: float) -> float | None:
    """Return the optional per-target allowance, if one was declared.

    ``None`` is the only absent value and keeps the historical whole-operation behaviour. A
    supplied value obeys the same strict numeric rule as the other timeouts and must be smaller
    than the operation allowance, because it may only shorten one target's read opportunity. A
    per-target allowance is never a second, larger budget for the whole observation.
    """
    if value is None:
        return None
    number = _positive_timeout_ms(value)
    if number >= operation_timeout_ms:
        raise _invalid()
    return number


def _startup_timeout_ms(value: object) -> float | None:
    """Return the optional start-acquisition allowance, if one was declared.

    ``None`` is the only absent value: it keeps the historical behaviour, where ``start`` is
    bounded by ``operation_timeout_ms``. A supplied value obeys exactly the same strict numeric
    rule as the other timeouts. It is not required to be smaller or larger than the operation
    allowance: the two budgets govern different calls, and a startup value only ever bounds the
    one browser acquisition attempt.
    """
    if value is None:
        return None
    return _positive_timeout_ms(value)


class PlaywrightWebConfig:
    """The immutable, validated configuration of one Playwright Web adapter instance.

    Construction validates every field, copies every supplied list and rebuilds every supplied
    locator model, then stores only immutable tuples. Accessors return detached copies, so no
    caller-held object can change later behaviour. The four original fields are required; the
    optional per-target read allowance and the optional startup allowance both default to
    ``None``, and only ``startup_timeout_ms`` is keyword-only.

    ``__repr__`` stays the fixed, value-free representation it has always been: adding the
    optional fields must not change the existing observable default behaviour.
    """

    __slots__ = (
        "_allowed_origins",
        "_cleanup_timeout_ms",
        "_observation_target_timeout_ms",
        "_observation_targets",
        "_operation_timeout_ms",
        "_origin_set",
        "_startup_timeout_ms",
    )

    def __init__(
        self,
        allowed_origins: object = _UNSET,
        observation_targets: object = _UNSET,
        operation_timeout_ms: object = _UNSET,
        cleanup_timeout_ms: object = _UNSET,
        observation_target_timeout_ms: object = None,
        *,
        startup_timeout_ms: object = None,
        **extra: object,
    ) -> None:
        if extra or (
            allowed_origins is _UNSET
            or observation_targets is _UNSET
            or operation_timeout_ms is _UNSET
            or cleanup_timeout_ms is _UNSET
        ):
            raise _invalid()
        origin_texts, origin_set = _validated_origins(allowed_origins)
        targets = _validated_targets(observation_targets)
        operation = _positive_timeout_ms(operation_timeout_ms)
        cleanup = _positive_timeout_ms(cleanup_timeout_ms)
        target_timeout = _target_timeout_ms(observation_target_timeout_ms, operation)
        startup = _startup_timeout_ms(startup_timeout_ms)
        object.__setattr__(self, "_allowed_origins", origin_texts)
        object.__setattr__(self, "_origin_set", origin_set)
        object.__setattr__(self, "_observation_targets", targets)
        object.__setattr__(self, "_operation_timeout_ms", operation)
        object.__setattr__(self, "_cleanup_timeout_ms", cleanup)
        object.__setattr__(self, "_observation_target_timeout_ms", target_timeout)
        object.__setattr__(self, "_startup_timeout_ms", startup)

    # ------------------------------------------------------------------ public fields

    @property
    def allowed_origins(self) -> list[str]:
        """The configured origin literals, detached from the stored state."""
        return list(self._allowed_origins)

    @property
    def observation_targets(self) -> list[TestIdTarget | RoleTarget]:
        """The configured locators, detached from the stored state."""
        return [target.model_copy(deep=True) for target in self._observation_targets]

    @property
    def operation_timeout_ms(self) -> float:
        """The finite per-operation budget in milliseconds."""
        return self._operation_timeout_ms

    @property
    def cleanup_timeout_ms(self) -> float:
        """The finite per-cleanup budget in milliseconds."""
        return self._cleanup_timeout_ms

    @property
    def observation_target_timeout_ms(self) -> float | None:
        """The optional per-target read allowance, or ``None`` for the historical behaviour."""
        return self._observation_target_timeout_ms

    @property
    def startup_timeout_ms(self) -> float | None:
        """The optional start-acquisition allowance, or ``None`` for the operation fallback."""
        return self._startup_timeout_ms

    @property
    def effective_startup_timeout_ms(self) -> float:
        """The allowance ``start`` really applies: the declared startup value or the operation one."""
        declared = self._startup_timeout_ms
        return self._operation_timeout_ms if declared is None else declared

    # ------------------------------------------------------------------ internal view

    @property
    def _canonical_origins(self) -> frozenset[Origin]:
        """The immutable canonical origin set used for every permission comparison."""
        return self._origin_set

    # ------------------------------------------------------------------ immutability

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PlaywrightWebConfig is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("PlaywrightWebConfig is immutable")

    def __repr__(self) -> str:
        """A fixed representation that carries no configured value."""
        return "PlaywrightWebConfig(<4 validated fields>)"


def revalidated_config(value: object) -> PlaywrightWebConfig:
    """Strictly revalidate a supplied configuration and return a detached copy.

    An adapter constructor never trusts the instance it was handed. The six stored fields are
    read back through the class slots and the configuration is rebuilt through the public
    constructor, so a ``model_construct``-style forgery or a mutated backing value is rejected
    with the same fixed code as any other malformed configuration. The optional per-target
    allowance and the optional startup allowance are separated from the operation allowance and
    revalidated on their own.
    """
    if not isinstance(value, PlaywrightWebConfig):
        raise _invalid()

    def read() -> tuple[object, object, object, object, object, object]:
        return (
            object.__getattribute__(value, "_allowed_origins"),
            object.__getattribute__(value, "_observation_targets"),
            object.__getattribute__(value, "_operation_timeout_ms"),
            object.__getattribute__(value, "_cleanup_timeout_ms"),
            object.__getattribute__(value, "_observation_target_timeout_ms"),
            object.__getattribute__(value, "_startup_timeout_ms"),
        )

    ok, fields = _attempt(read)
    if not ok:
        raise _invalid()

    def detach() -> tuple[list, list]:
        origins, targets = fields[0], fields[1]  # type: ignore[index]
        return list(origins), list(targets)  # type: ignore[arg-type]

    ok, copies = _attempt(detach)
    if not ok:
        raise _invalid()
    supplied_origins, supplied_targets = copies  # type: ignore[misc]
    return PlaywrightWebConfig(
        allowed_origins=supplied_origins,
        observation_targets=supplied_targets,
        operation_timeout_ms=fields[2],  # type: ignore[index]
        cleanup_timeout_ms=fields[3],  # type: ignore[index]
        observation_target_timeout_ms=fields[4],  # type: ignore[index]
        startup_timeout_ms=fields[5],  # type: ignore[index]
    )
