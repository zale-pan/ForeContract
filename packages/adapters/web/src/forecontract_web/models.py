"""Strict Web domain models for action payloads, locators and observation evidence.

Six fixed action types select one strict payload model each, and a payload never repeats the
action type: ``action_type`` is the discriminator that chooses the model. A locator is a closed
discriminated union on ``strategy`` with no selector passthrough - ``test_id`` or ``role`` only,
with no CSS, XPath, text, ``nth``, frame or shadow option and no "pick any match" fallback. A
locator equality check compares the declared fields exactly, so no trimming, case folding,
Unicode normalisation or fuzzy matching can make two different locators look the same.

Observation evidence keeps "not obtained" and "obtained as an empty value" apart. ``null`` means
the value was not obtained in this snapshot; an empty string is a real obtained empty value and
``visible=false`` is an explicit invisible result. A successful query that matched nothing is
existence evidence with ``match_count=0``; a successful query that matched several nodes is
ambiguity evidence and carries no attributes at all; a failed query is ``unavailable`` with an
explicit ``TARGET_QUERY_FAILED`` code and no count, so a failure can never forge a zero match.

Every model rejects unknown fields, refuses implicit coercion, requires an explicit schema
version where the record is versioned, and is frozen so that an assignment can never leave a
model in a state its own cross-field rule would reject. Freezing is not deep immutability and it
is not a validation-bypass guard: the public entry points in ``forecontract_web.validation``
revalidate the stored fields of a supplied instance instead of trusting it.
"""

from __future__ import annotations

import ipaddress
from typing import Annotated, Final, Literal

from forecontract.models.base import (
    MODEL_CONFIG,
    NonBlankStr,
    NonNegativeInt,
    SchemaVersion,
    StrictBool,
    StrictStr,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "WEB_ACTION_TYPES",
    "WEB_OBSERVATION_TYPES",
    "ClickPayload",
    "FillPayload",
    "NavigatePayload",
    "PressPayload",
    "RoleTarget",
    "SelectPayload",
    "TestIdTarget",
    "WaitPayload",
    "WebObservationState",
    "WebTargetEvidence",
]

#: The one environment type a Web observation may declare. Internal: not a public export.
WEB_ENVIRONMENT_TYPE: Final[str] = "web"

#: The closed action vocabulary. Order is part of the declaration and is preserved.
WEB_ACTION_TYPES: Final[tuple[str, ...]] = (
    "web.navigate",
    "web.click",
    "web.fill",
    "web.press",
    "web.select",
    "web.wait",
)

#: The closed observation-type vocabulary of this round.
WEB_OBSERVATION_TYPES: Final[tuple[str, ...]] = ("web.snapshot",)

#: The closed scheme vocabulary for a page URL. Only the literal lowercase token is accepted.
WEB_URL_SCHEMES: Final[tuple[str, ...]] = ("http", "https")

#: The one non-network page URL a snapshot may report verbatim.
ABOUT_BLANK: Final[str] = "about:blank"

RoleName = Literal[
    "alert",
    "button",
    "checkbox",
    "combobox",
    "dialog",
    "heading",
    "link",
    "listbox",
    "option",
    "radio",
    "status",
    "textbox",
]
PressKey = Literal[
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "ArrowUp",
    "Backspace",
    "Delete",
    "End",
    "Enter",
    "Escape",
    "Home",
    "Space",
    "Tab",
]
WaitState = Literal["attached", "detached", "hidden", "visible"]
TargetStatus = Literal["collected", "unavailable"]
TargetQueryErrorCode = Literal["TARGET_QUERY_FAILED"]

#: The shared strict configuration. ``validate_assignment`` is kept from the Core base and is
#: superseded by ``frozen``; ``hide_input_in_errors`` keeps a rejected value out of even the
#: ordinary low-level schema error text.
_WEB_MODEL_CONFIG = ConfigDict(**MODEL_CONFIG, frozen=True, hide_input_in_errors=True)


def _port_problem(port: str) -> str | None:
    """Return a fixed reason when a port component is not a decimal port number."""
    if not port or len(port) > 5:
        return "must carry a valid port"
    if not all("0" <= character <= "9" for character in port):
        return "must carry a valid port"
    if int(port) < 1 or int(port) > 65535:
        return "must carry a valid port"
    return None


def _authority_of(remainder: str) -> str:
    """The authority component: everything before the first path, query or fragment mark."""
    for index, character in enumerate(remainder):
        if character in "/?#":
            return remainder[:index]
    return remainder


def _authority_problem(authority: str) -> str | None:
    """Reject an empty host, user information, an invalid port or a broken IPv6 literal."""
    if not authority:
        return "must carry a non-empty host"
    if "@" in authority:
        return "must not carry user information"
    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            return "must carry a valid IPv6 host"
        literal = authority[1:closing]
        if not literal:
            return "must carry a non-empty host"
        try:
            ipaddress.IPv6Address(literal)
        except ValueError:
            return "must carry a valid IPv6 host"
        remainder = authority[closing + 1 :]
        if not remainder:
            return None
        if not remainder.startswith(":"):
            return "must carry a valid port"
        return _port_problem(remainder[1:])
    host, separator, port = authority.partition(":")
    if not host:
        return "must carry a non-empty host"
    if "[" in host or "]" in host:
        return "must carry a valid IPv6 host"
    if separator:
        problem = _port_problem(port)
        if problem is not None:
            return problem
    if not any(character.isalnum() for character in host):
        return "must carry a non-empty host"
    return None


def _url_problem(value: str, *, allow_about_blank: bool) -> str | None:
    """Return a fixed reason when a URL is not an absolute ``http`` or ``https`` URL.

    Only syntax is checked: no network request, no DNS lookup and no URL rewriting. The supplied
    text is never normalised, so a caller-supplied authority, port, escape, path, query and
    fragment reach the caller unchanged.
    """
    if allow_about_blank and value == ABOUT_BLANK:
        return None
    for character in value:
        if character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F:
            return "must not contain whitespace or control characters"
    if "\\" in value:
        return "must not contain a backslash"
    scheme, separator, remainder = value.partition(":")
    if separator != ":" or scheme not in WEB_URL_SCHEMES:
        return "must use the http or https scheme"
    if not remainder.startswith("//"):
        return "must be an absolute URL"
    return _authority_problem(_authority_of(remainder[2:]))


def _target_key(target: TestIdTarget | RoleTarget) -> tuple[str, ...]:
    """The exact declared-field identity of a locator, independent of mapping key order."""
    return tuple(getattr(target, name) for name in sorted(type(target).model_fields))


class TestIdTarget(BaseModel):
    """A locator that names one ``data-testid`` value exactly."""

    model_config = _WEB_MODEL_CONFIG

    #: A public model whose name begins with "Test" would otherwise be collected as a test class
    #: by pytest. The marker is a class attribute only; it is not a field and is never dumped.
    __test__ = False

    strategy: Literal["test_id"]
    test_id: NonBlankStr


class RoleTarget(BaseModel):
    """A locator that names one accessibility role and its exact accessible name."""

    model_config = _WEB_MODEL_CONFIG

    strategy: Literal["role"]
    role: RoleName
    name: NonBlankStr


#: A locator is exactly one of the two declared strategies, selected by its own ``strategy``.
WebTarget = Annotated[TestIdTarget | RoleTarget, Field(discriminator="strategy")]


class NavigatePayload(BaseModel):
    """The payload of ``web.navigate``: one absolute ``http`` or ``https`` URL."""

    model_config = _WEB_MODEL_CONFIG

    url: StrictStr

    @field_validator("url")
    @classmethod
    def _absolute_page_url(cls, value: str) -> str:
        problem = _url_problem(value, allow_about_blank=False)
        if problem is not None:
            raise ValueError(problem)
        return value


class ClickPayload(BaseModel):
    """The payload of ``web.click``: exactly one locator."""

    model_config = _WEB_MODEL_CONFIG

    target: WebTarget


class FillPayload(BaseModel):
    """The payload of ``web.fill``: one locator and the text to enter.

    An empty ``text`` is legal: clearing a field is a real fill request, and a carried empty
    string is not the same thing as an omitted one.
    """

    model_config = _WEB_MODEL_CONFIG

    target: WebTarget
    text: StrictStr


class PressPayload(BaseModel):
    """The payload of ``web.press``: one locator and one closed-vocabulary key."""

    model_config = _WEB_MODEL_CONFIG

    target: WebTarget
    key: PressKey


class SelectPayload(BaseModel):
    """The payload of ``web.select``: one locator and the option value to select.

    An empty ``value`` is legal and is carried verbatim; it is never treated as a missing value.
    """

    model_config = _WEB_MODEL_CONFIG

    target: WebTarget
    value: StrictStr


class WaitPayload(BaseModel):
    """The payload of ``web.wait``: one locator and the closed state to wait for."""

    model_config = _WEB_MODEL_CONFIG

    target: WebTarget
    state: WaitState


class WebTargetEvidence(BaseModel):
    """What one successful or failed target query obtained in this snapshot.

    ``status`` decides the whole record. A ``collected`` query must report a non-negative
    ``match_count`` and no error code; attributes may be carried only when exactly one node
    matched, and each of them may independently be an obtained value or ``null``. An
    ``unavailable`` query must carry a ``null`` count, the ``TARGET_QUERY_FAILED`` code and no
    attributes, so a failure can never be read as "zero matches" and can never merge stale
    attributes into a failed query.
    """

    model_config = _WEB_MODEL_CONFIG

    target: WebTarget
    status: TargetStatus
    match_count: NonNegativeInt | None
    visible: StrictBool | None
    text: StrictStr | None
    value: StrictStr | None
    error_code: TargetQueryErrorCode | None

    @model_validator(mode="after")
    def _status_decides_the_record(self) -> WebTargetEvidence:
        """Reject every combination the consistency table does not allow."""
        if self.status == "collected":
            if self.error_code is not None:
                raise ValueError("a collected target must not carry an error code")
            if self.match_count is None:
                raise ValueError("a collected target must carry a match count")
            if self.match_count == 1:
                return self
            if self.visible is not None or self.text is not None or self.value is not None:
                raise ValueError("only a single match may carry an attribute")
            return self
        if self.error_code != "TARGET_QUERY_FAILED":
            raise ValueError("an unavailable target must carry the query failure code")
        if self.match_count is not None:
            raise ValueError("an unavailable target must not carry a match count")
        if self.visible is not None or self.text is not None or self.value is not None:
            raise ValueError("an unavailable target must not carry an attribute")
        return self


class WebObservationState(BaseModel):
    """The Web part of one Core observation ``state``.

    ``url`` is either an absolute page URL, exactly ``about:blank``, or ``null``. ``null`` means
    the URL was not obtained in this snapshot and must never be read as an empty URL; a missing
    ``url`` field is a schema error instead, so the two cases cannot be confused. ``targets`` may
    be empty - an absent record means the evidence was not provided, never that the target does
    not exist - and one locator may appear at most once.
    """

    model_config = _WEB_MODEL_CONFIG

    schema_version: SchemaVersion
    url: StrictStr | None
    targets: Annotated[list[WebTargetEvidence], Field(strict=True)]

    @field_validator("url")
    @classmethod
    def _page_url_or_null(cls, value: str | None) -> str | None:
        if value is None:
            return None
        problem = _url_problem(value, allow_about_blank=True)
        if problem is not None:
            raise ValueError(problem)
        return value

    @field_validator("targets")
    @classmethod
    def _targets_are_unique(
        cls, value: list[WebTargetEvidence]
    ) -> list[WebTargetEvidence]:
        """Reject two records for the same locator; distinct locators keep the supplied order."""
        seen: set[tuple[str, ...]] = set()
        for evidence in value:
            key = _target_key(evidence.target)
            if key in seen:
                raise ValueError("a locator may appear at most once in a snapshot")
            seen.add(key)
        return value


#: The payload model each action type selects. The mapping is closed and total: both the action
#: vocabulary and this mapping are declared once, and the public functions assert that they agree.
_PAYLOAD_MODELS: Final[dict[str, type[BaseModel]]] = {
    "web.navigate": NavigatePayload,
    "web.click": ClickPayload,
    "web.fill": FillPayload,
    "web.press": PressPayload,
    "web.select": SelectPayload,
    "web.wait": WaitPayload,
}
