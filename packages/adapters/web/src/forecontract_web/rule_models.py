"""Strict closed payloads and the bounded result record of the five pure Web rules.

Five deterministic rule types exist and nothing else:

* ``web.url_match`` with the single field ``url``;
* ``web.element_visible`` and ``web.element_hidden`` with the single field ``target``;
* ``web.text_contains`` with the fields ``target`` and ``text``;
* ``web.value_equals`` with the fields ``target`` and ``value``.

A payload never repeats the rule type and never carries a schema version or a ``required``
flag: the rule type is the discriminator its caller already named, the Core
``ExpectedOutcome``/``Precondition`` envelope owns ``schema_version`` and ``required``, and the
evidence keeps the observation version. Every model rejects unknown fields, refuses implicit
coercion and is frozen.

``WebRuleEvaluation`` is the single result record. Its ``target_index`` locates the exact
record in the supplied snapshot and is ``null`` when no record matched; the URL rule has no
target, so its ``target_index`` is always ``null``. The record deliberately carries no copied
URL, text or value: a caller reads the original snapshot through ``observation_id`` and the
target position. Its ``status``/``reason`` pair is one closed verdict - ``PASS`` only with
``RULE_MATCH``, ``FAIL`` only with ``RULE_MISMATCH`` or ``TARGET_NOT_FOUND``, and
``INCONCLUSIVE`` only with ``EVIDENCE_INSUFFICIENT`` - so a self-contradictory record cannot be
built through any ingress.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal

from forecontract.models.base import (
    NonBlankStr,
    NonNegativeInt,
    SchemaVersion,
    StrictStr,
)
from forecontract.models.outcomes import VerificationStatus
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from forecontract_web.models import WebTarget, _url_problem

__all__ = [
    "WEB_RULE_TYPES",
    "ElementHiddenPayload",
    "ElementVisiblePayload",
    "TextContainsPayload",
    "UrlMatchPayload",
    "ValueEqualsPayload",
    "WebRuleEvaluation",
]

#: The closed rule vocabulary. Order is part of the declaration and is preserved.
WEB_RULE_TYPES: Final[tuple[str, ...]] = (
    "web.url_match",
    "web.element_visible",
    "web.element_hidden",
    "web.text_contains",
    "web.value_equals",
)

#: The closed reason vocabulary of one rule verdict.
RULE_MATCH: Final[str] = "RULE_MATCH"
RULE_MISMATCH: Final[str] = "RULE_MISMATCH"
TARGET_NOT_FOUND: Final[str] = "TARGET_NOT_FOUND"
EVIDENCE_INSUFFICIENT: Final[str] = "EVIDENCE_INSUFFICIENT"

#: The five declared payload fields. Only these names may appear in a safe diagnostic path.
PAYLOAD_FIELDS: Final[frozenset[str]] = frozenset({"url", "target", "text", "value"})

#: The declared fields of the two existing locator records.
TARGET_FIELDS: Final[frozenset[str]] = frozenset(
    {"strategy", "test_id", "role", "name"}
)

#: Strict, closed, immutable configuration shared by the payload and result records.
_RULE_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)

#: A non-empty expected substring: an empty value would make ``text_contains`` vacuously true.
_NonEmptyStr = Annotated[str, Field(strict=True, min_length=1)]


class UrlMatchPayload(BaseModel):
    """The payload of ``web.url_match``: one exact expected page URL string.

    The expected URL must already obey the existing page-URL grammar the observation state
    enforces: an absolute ``http``/``https`` URL, or exactly ``about:blank``. The check reads
    the characters actually stored in the supplied string and returns that same object
    untouched, so nothing is trimmed, canonicalised or case-folded and the later whole-string
    comparison still sees the caller's exact text.
    """

    model_config = _RULE_CONFIG

    url: StrictStr

    @field_validator("url")
    @classmethod
    def _expected_page_url(cls, value: str) -> str:
        """Reject an expected URL outside the existing observation-state page-URL grammar."""
        text = str.__getitem__(value, slice(None))
        problem = _url_problem(text, allow_about_blank=True)
        if problem is not None:
            raise ValueError(problem)
        return value


class ElementVisiblePayload(BaseModel):
    """The payload of ``web.element_visible``: exactly one existing locator."""

    model_config = _RULE_CONFIG

    target: WebTarget


class ElementHiddenPayload(BaseModel):
    """The payload of ``web.element_hidden``: exactly one existing locator."""

    model_config = _RULE_CONFIG

    target: WebTarget


class TextContainsPayload(BaseModel):
    """The payload of ``web.text_contains``: one locator and one non-empty expected text."""

    model_config = _RULE_CONFIG

    target: WebTarget
    text: _NonEmptyStr


class ValueEqualsPayload(BaseModel):
    """The payload of ``web.value_equals``: one locator and one exact expected value.

    An empty expected value is legal: clearing a field is a real observed value, and the empty
    string is evidence rather than a missing measurement.
    """

    model_config = _RULE_CONFIG

    target: WebTarget
    value: StrictStr


class WebRuleEvaluation(BaseModel):
    """The bounded verdict of one Web rule against one piece of supplied evidence.

    The status, the reason, the observation identity and the target position are one verdict,
    not independent fields; the truth table is enforced here as well as by the evaluator.
    """

    model_config = _RULE_CONFIG

    schema_version: SchemaVersion
    rule_type: Literal[
        "web.url_match",
        "web.element_visible",
        "web.element_hidden",
        "web.text_contains",
        "web.value_equals",
    ]
    status: VerificationStatus
    reason: Literal[
        "RULE_MATCH",
        "RULE_MISMATCH",
        "TARGET_NOT_FOUND",
        "EVIDENCE_INSUFFICIENT",
    ]
    observation_id: NonBlankStr
    target_index: NonNegativeInt | None

    @model_validator(mode="after")
    def _verdict_is_consistent(self) -> WebRuleEvaluation:
        """Reject every status/reason pair the frozen truth table does not contain.

        A ``web.url_match`` verdict has no target record at all, so it must carry a null
        ``target_index`` at this ordinary result-schema ingress too, not only from the
        evaluator.
        """
        if self.rule_type == "web.url_match" and self.target_index is not None:
            raise ValueError("a URL verdict has no target and must carry a null target index")
        if self.status is VerificationStatus.PASS:
            if self.reason != RULE_MATCH:
                raise ValueError("a PASS verdict reports RULE_MATCH only")
        elif self.status is VerificationStatus.FAIL:
            if self.reason not in (RULE_MISMATCH, TARGET_NOT_FOUND):
                raise ValueError("a FAIL verdict reports a mismatch or a missing target only")
        elif self.reason != EVIDENCE_INSUFFICIENT:
            raise ValueError("an INCONCLUSIVE verdict reports EVIDENCE_INSUFFICIENT only")
        return self
