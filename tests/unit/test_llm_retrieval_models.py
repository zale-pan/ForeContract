"""Acceptance tests for the strict retrieval records, encoding rules and error taxonomy.

C02/C06: every record requires the explicit schema version, forbids unknown fields, rejects
integer coercion, validates assignment transactionally, and revalidates a supplied instance from
its own storage. The shared canonical-JSON and strict-parser rules are the measurement basis of
the whole protocol, so they are tested here directly.
"""

from __future__ import annotations

import hashlib

import pytest
from forecontract_llm.models import PlannerConfig
from forecontract_llm.retrieval import (
    EvidenceGrant,
    EvidenceReadRequest,
    EvidenceReadResult,
    EvidenceScope,
    RetrievalConfig,
    RetrievalError,
    RetrievalModelRequest,
)
from forecontract_llm.retrieval.audit import (
    AUDIT_EVENT_KINDS,
    RetrievalAuditEvent,
    build_event,
)
from forecontract_llm.retrieval.errors import (
    RETRIEVAL_ERROR_CODES,
    RETRIEVAL_ERROR_STAGES,
)
from forecontract_llm.retrieval.models import (
    JSON_DEPTH_MAX,
    JSON_NODES_MAX,
    OUTPUT_TOKENS_MAX,
    PLAN_MAX_MODEL_CALLS,
    PLAN_MAX_READS,
    PLAN_REQUEST_MAX_BYTES,
    PLAN_RESPONSE_MAX_BYTES,
    PLAN_RESULT_MAX_BYTES,
    REGISTERED_MAX_BYTES,
    REGISTERED_MAX_COUNT,
    REQUEST_MAX_BYTES,
    RESPONSE_MAX_BYTES,
    RESULT_MAX_BYTES,
    SCOPE_MAX_BYTES,
    SNAPSHOT_MAX_BYTES,
    SNAPSHOT_MAX_VIEWS,
    EvidenceScopeEntry,
    ScopeView,
    TextRange,
    canonical_json_bytes,
    copy_record,
    parse_strict_json,
)
from pydantic import ValidationError

SCHEMA = "0.1"
HEX = "a" * 64


def _base_config(**overrides: object) -> PlannerConfig:
    return PlannerConfig(schema_version=SCHEMA, **overrides)


def _grant(**overrides: object) -> EvidenceGrant:
    values = {
        "schema_version": SCHEMA,
        "run_id": "run-1",
        "evidence_id": "evidence-1",
        "observation_id": "obs-1",
        "view_id": "status",
        "selector": ["state", "order", "status"],
        "format": "text",
    }
    values.update(overrides)
    return EvidenceGrant(**values)


def _view(**overrides: object) -> ScopeView:
    values = {"schema_version": SCHEMA, "view_id": "status",
              "selector": ["state", "order", "status"], "format": "text"}
    values.update(overrides)
    return ScopeView(**values)


def _entry(**overrides: object) -> EvidenceScopeEntry:
    values = {"schema_version": SCHEMA, "evidence_id": "evidence-1",
              "source_observation_id": "obs-1", "kind": "observation",
              "raw_sha256": HEX, "raw_bytes": 10, "views": [_view()]}
    values.update(overrides)
    return EvidenceScopeEntry(**values)


def _scope(**overrides: object) -> EvidenceScope:
    values = {"schema_version": SCHEMA, "scope_id": "scope-1", "run_id": "run-1",
              "observation_id": "obs-1", "entries": [_entry()]}
    values.update(overrides)
    return EvidenceScope(**values)


def _request(**overrides: object) -> EvidenceReadRequest:
    values = {"schema_version": SCHEMA, "scope_id": "scope-1", "run_id": "run-1",
              "plan_id": "plan-1", "turn": 1, "request_id": "request-1",
              "evidence_id": "evidence-1", "view_id": "status", "text_range": None}
    values.update(overrides)
    return EvidenceReadRequest(**values)


def _result(**overrides: object) -> EvidenceReadResult:
    value = "pending"
    raw = canonical_json_bytes(value)
    values = {
        "schema_version": SCHEMA,
        "scope_id": "scope-1",
        "run_id": "run-1",
        "plan_id": "plan-1",
        "turn": 1,
        "request_id": "request-1",
        "evidence_id": "evidence-1",
        "view_id": "status",
        "status": "found",
        "source_observation_id": "obs-1",
        "raw_sha256": HEX,
        "raw_bytes": 10,
        "selector": ["state", "order", "status"],
        "format": "text",
        "text_range": None,
        "partial": False,
        "value": value,
        "value_sha256": hashlib.sha256(raw).hexdigest(),
        "value_bytes": len(raw),
    }
    values.update(overrides)
    return EvidenceReadResult(**values)


def test_the_subpackage_exports_exactly_the_thirteen_agreed_entries() -> None:
    """C01: the public entry points are a fixed set, not whatever the modules define."""
    from forecontract_llm import retrieval

    assert set(retrieval.__all__) == {
        "EvidenceLLMPlanner",
        "RetrievalConfig",
        "RetrievalModelRequest",
        "RetrievalProvider",
        "EvidenceReader",
        "EvidenceScope",
        "EvidenceReadRequest",
        "EvidenceReadResult",
        "EvidenceGrant",
        "RegisteredEvidenceReader",
        "RetrievalAuditSink",
        "RetrievalAuditEvent",
        "RetrievalError",
    }
    assert len(retrieval.__all__) == 13


def test_the_budget_constants_are_the_contract_values() -> None:
    """C05: the fixed caps are the declared ones; only narrowing is configurable."""
    assert (PLAN_MAX_READS, PLAN_MAX_MODEL_CALLS) == (2, 3)
    assert SNAPSHOT_MAX_BYTES == 1_048_576
    assert (REGISTERED_MAX_BYTES, REGISTERED_MAX_COUNT) == (8_388_608, 64)
    assert SNAPSHOT_MAX_VIEWS == 8
    assert SCOPE_MAX_BYTES == 32_768
    assert (RESULT_MAX_BYTES, PLAN_RESULT_MAX_BYTES) == (65_536, 131_072)
    assert (REQUEST_MAX_BYTES, PLAN_REQUEST_MAX_BYTES) == (262_144, 786_432)
    assert (RESPONSE_MAX_BYTES, PLAN_RESPONSE_MAX_BYTES) == (65_536, 196_608)
    assert (OUTPUT_TOKENS_MAX, JSON_DEPTH_MAX, JSON_NODES_MAX) == (2_048, 32, 10_000)
    config = RetrievalConfig(schema_version=SCHEMA, base=_base_config())
    assert (config.max_reads, config.max_model_calls) == (2, 3)


_RECORDS = {
    "grant": _grant,
    "view": _view,
    "entry": _entry,
    "scope": _scope,
    "request": _request,
    "result": _result,
    "text_range": lambda **kw: TextRange(schema_version=SCHEMA, start=0, end=1, **kw),
    "config": lambda **kw: RetrievalConfig(schema_version=SCHEMA, base=_base_config(), **kw),
}


@pytest.mark.parametrize("name", sorted(_RECORDS))
def test_every_record_requires_the_explicit_schema_version(name: str) -> None:
    """C02: an omitted schema version is refused on every record."""
    builder = _RECORDS[name]
    record = builder()
    values = {field: getattr(record, field) for field in type(record).model_fields}
    values.pop("schema_version")
    with pytest.raises(ValidationError):
        type(record)(**values)


@pytest.mark.parametrize("name", sorted(_RECORDS))
def test_every_record_refuses_an_unknown_field(name: str) -> None:
    """C02: extra fields are forbidden on every record."""
    builder = _RECORDS[name]
    record = builder()
    values = {field: getattr(record, field) for field in type(record).model_fields}
    values["injected"] = 1
    with pytest.raises(ValidationError):
        type(record)(**values)


@pytest.mark.parametrize("value", [3, 4, True, "2", 1.0, -1])
def test_the_read_limit_is_strict_and_capped_at_two(value: object) -> None:
    """C02/C05: the retrieval fields are strict and can only narrow the plan caps."""
    with pytest.raises(ValidationError):
        RetrievalConfig(schema_version=SCHEMA, base=_base_config(), max_reads=value)
    assert RetrievalConfig(schema_version=SCHEMA, base=_base_config(),
                           max_reads=0).max_reads == 0
    assert RetrievalConfig(schema_version=SCHEMA, base=_base_config(),
                           max_reads=2).max_reads == 2


@pytest.mark.parametrize("value", [4, True, "2", 1.0, 0, -1])
def test_the_model_call_limit_is_strict_and_capped_at_three(value: object) -> None:
    """C05: at most three model calls per plan, never a widening value."""
    with pytest.raises(ValidationError):
        RetrievalConfig(schema_version=SCHEMA, base=_base_config(), max_model_calls=value)
    assert RetrievalConfig(schema_version=SCHEMA, base=_base_config(),
                           max_model_calls=3).max_model_calls == 3


@pytest.mark.parametrize("value", [True, "2", 2.0, 0, -1])
def test_a_turn_is_a_strict_positive_integer(value: object) -> None:
    """C02: no boolean, numeric string or float is coerced into a turn."""
    with pytest.raises(ValidationError):
        _request(turn=value)
    with pytest.raises(ValidationError):
        _result(turn=value)


@pytest.mark.parametrize("value", [True, 1.0, "1", -1])
def test_a_text_range_bound_is_a_strict_non_negative_integer(value: object) -> None:
    """C02: a range bound is a real integer, and start never follows end."""
    with pytest.raises(ValidationError):
        TextRange(schema_version=SCHEMA, start=value, end=3)
    with pytest.raises(ValidationError):
        TextRange(schema_version=SCHEMA, start=0, end=value)
    with pytest.raises(ValidationError):
        TextRange(schema_version=SCHEMA, start=3, end=2)
    assert (TextRange(schema_version=SCHEMA, start=2, end=2).start,
            TextRange(schema_version=SCHEMA, start=2, end=2).end) == (2, 2)


@pytest.mark.parametrize("value", ["", "a" * 129, "a b", "a.b", "a/b", "茅"])
def test_a_grant_identifier_is_bounded_ascii(value: str) -> None:
    """C03: identifiers are 1-128 ASCII letters, digits, underscores or hyphens."""
    with pytest.raises(ValidationError):
        _grant(evidence_id=value)
    with pytest.raises(ValidationError):
        _grant(run_id=value)
    with pytest.raises(ValidationError):
        _grant(observation_id=value)
    with pytest.raises(ValidationError):
        _grant(view_id=value)


def test_a_grant_identifier_accepts_every_legal_character() -> None:
    """C03: the legal alphabet is accepted in a full-length identifier."""
    value = ("aZ0_-" * 26)[:128]
    grant = _grant(evidence_id=value)
    assert grant.evidence_id == value


@pytest.mark.parametrize("selector", [
    ["other"],
    ["state"] * 33,
    ["state", "a" * 129],
    ["state", -1],
    ["state", True],
    ["state", 1.5],
    [],
])
def test_a_selector_obeys_the_closed_grammar(selector: list[object]) -> None:
    """C03: the selector starts at state/summary, is bounded and has no coercion."""
    with pytest.raises(ValidationError):
        _grant(selector=selector)
    with pytest.raises(ValidationError):
        _view(selector=selector)


@pytest.mark.parametrize("selector", [["state"], ["summary"], ["state", 0],
                                  ["state", 12]])
def test_a_selector_accepts_the_declared_segments(selector: list[object]) -> None:
    """C03: string roots and non-negative integer segments are legal."""
    grant = _grant(selector=selector)
    assert grant.selector == selector


def test_a_grant_owns_a_detached_copy_of_its_selector() -> None:
    """C02: a caller cannot keep an alias into a record's nested list."""
    selector = ["state", "order", "status"]
    grant = _grant(selector=selector)
    selector.append("injected")
    selector[0] = "summary"
    assert grant.selector == ["state", "order", "status"]


def test_assignment_is_transactional_on_a_refused_value() -> None:
    """C02: a refused assignment restores the previous stored state."""
    grant = _grant()
    with pytest.raises(ValidationError):
        grant.format = "xml"
    assert grant.format == "text"
    with pytest.raises(ValidationError):
        grant.selector = ["other"]
    assert grant.selector == ["state", "order", "status"]
    grant.format = "json"
    assert grant.format == "json"


def test_a_forged_stored_key_is_refused_at_the_boundary() -> None:
    """C02: a model_construct forgery is read from storage and refused."""
    forged = EvidenceGrant.model_construct(
        schema_version=SCHEMA, run_id="run-1", evidence_id="e1", observation_id="obs-1",
        view_id="v1", selector=["state"], format="text")
    object.__getattribute__(forged, "__dict__")["injected"] = "x"
    with pytest.raises(RetrievalError) as info:
        copy_record(forged, EvidenceGrant, "invalid_input", "input")
    assert (info.value.code, info.value.stage) == ("invalid_input", "input")


def test_a_virtual_dict_view_cannot_hide_a_forged_stored_value() -> None:
    """C02: the declared base descriptor is read, not an overridable view."""
    class VirtualGrant(EvidenceGrant):
        def __getattribute__(self, name: str) -> object:
            if name == "__dict__":
                return dict(object.__getattribute__(self, "__dict__"), format="text")
            return super().__getattribute__(name)

    forged = VirtualGrant(
        schema_version=SCHEMA, run_id="run-1", evidence_id="e1", observation_id="obs-1",
        view_id="v1", selector=["state"], format="text")
    object.__getattribute__(forged, "__dict__")["format"] = "xml"
    with pytest.raises(RetrievalError):
        copy_record(forged, EvidenceGrant, "invalid_input", "input")


def test_a_detached_copy_is_independent_of_the_original() -> None:
    """C02: a boundary copy shares no container with the record it came from."""
    original = _scope()
    copy = copy_record(original, EvidenceScope, "invalid_input", "input")
    original.entries[0].views[0].selector.append("injected")
    assert copy.entries[0].views[0].selector == ["state", "order", "status"]


def test_a_found_json_null_is_not_a_missing_value() -> None:
    """C03: found-null and not_found are distinguishable by status and digest."""
    null = _result(format="json", selector=["state", "value"], value=None, partial=False,
                   value_sha256=hashlib.sha256(b"null").hexdigest(), value_bytes=4)
    assert (null.status, null.value, null.value_bytes) == ("found", None, 4)
    missing = _result(status="not_found", value=None, value_sha256=None, value_bytes=0)
    assert (missing.status, missing.value, missing.value_sha256,
            missing.value_bytes) == ("not_found", None, None, 0)


def test_a_found_result_must_carry_its_own_value_digest() -> None:
    """C03: the digest and the byte count are recomputed from the value."""
    raw = canonical_json_bytes("pending")
    with pytest.raises(ValidationError):
        _result(value_sha256="b" * 64)
    with pytest.raises(ValidationError):
        _result(value_bytes=len(raw) + 1)
    with pytest.raises(ValidationError):
        _result(status="not_found")
    with pytest.raises(ValidationError):
        _result(status="not_found", value_sha256=hashlib.sha256(raw).hexdigest(),
                value_bytes=len(raw))


def test_a_text_view_may_state_a_partial_interval() -> None:
    """C03: the reader states partiality from the source string it owns; the record echoes it."""
    whole = _result(partial=False)
    partial = _result(text_range=TextRange(schema_version=SCHEMA, start=0, end=3),
                      partial=True)
    assert (whole.partial, partial.partial) == (False, True)


def test_a_json_view_forbids_a_text_range_and_partiality() -> None:
    """C03: a JSON selector is a projection, not a text interval."""
    with pytest.raises(ValidationError):
        _result(format="json", text_range=TextRange(schema_version=SCHEMA, start=0, end=1))
    with pytest.raises(ValidationError):
        _result(format="json", partial=True)


def test_a_scope_entry_is_non_empty_and_unique() -> None:
    """C03: a directory entry has at least one view and no repeated view identifier."""
    with pytest.raises(ValidationError):
        _entry(views=[])
    with pytest.raises(ValidationError):
        _entry(views=[_view(), _view()])
    with pytest.raises(ValidationError):
        _scope(entries=[_entry(), _entry()])


def test_a_scope_entry_names_no_local_path() -> None:
    """C03: the directory carries identifiers and digests, never a filesystem path."""
    assert set(EvidenceScopeEntry.model_fields) == {
        "schema_version", "evidence_id", "source_observation_id", "kind",
        "raw_sha256", "raw_bytes", "views"}
    assert "path" not in _entry().model_dump(mode="json")


def test_a_retrieval_request_is_bound_to_its_base_and_scope() -> None:
    """C02: the envelope and the base request carry the same identity and schema."""
    from forecontract_llm.models import ModelRequest

    schema = {"type": "object"}
    base = ModelRequest(
        schema_version=SCHEMA, request_id="request-1",
        model_context=_planner_context(), observation=_observation(),
        response_schema=schema, max_output_tokens=128)
    request = RetrievalModelRequest(
        schema_version=SCHEMA, run_id="run-1", plan_id="plan-1", turn=2,
        request_id="request-1", base_request=base, scope=_scope(),
        exchanges=[], response_schema=schema)
    assert (request.base_request.request_id, request.base_request.model_context.run_id) == (
        "request-1", "run-1")
    with pytest.raises(ValidationError):
        RetrievalModelRequest(
            schema_version=SCHEMA, run_id="run-2", plan_id="plan-1", turn=2,
            request_id="request-1", base_request=base, scope=_scope(),
            exchanges=[], response_schema=schema)
    with pytest.raises(ValidationError):
        RetrievalModelRequest(
            schema_version=SCHEMA, run_id="run-1", plan_id="plan-1", turn=1,
            request_id="request-1", base_request=base, scope=_scope(),
            exchanges=[_result()], response_schema=schema)


def test_the_canonical_json_rule_is_compact_sorted_and_utf8() -> None:
    """C05: the measurement rule is declared and stable."""
    assert canonical_json_bytes({"b": 1, "a": "\u4e2d"}) == b'{"a":"\xe4\xb8\xad","b":1}'
    assert canonical_json_bytes(None) == b"null"
    with pytest.raises(RetrievalError):
        canonical_json_bytes({"a": float("nan")})
    with pytest.raises(RetrievalError):
        canonical_json_bytes({1: "a"})


def test_the_strict_parser_refuses_duplicates_nan_surrogates_and_trailing_text() -> None:
    """C02/C05: one strict JSON value only, with the agreed refusals."""
    parse = lambda text: parse_strict_json(
        text, max_depth=32, max_nodes=10_000, code="invalid_response", stage="response")
    assert parse('{"a": [1, null]}') == {"a": [1, None]}
    assert parse("[]") == []
    for text in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": "\ud800"}',
                 '{"a": 1} trailing', "```json\n{}\n```"):
        try:
            parsed = parse(text)
        except RetrievalError:
            continue
        raise AssertionError(f"accepted {text!r}: {parsed!r}")
    with pytest.raises(RetrievalError):
        parse_strict_json(b"\xff", max_depth=32, max_nodes=10, code="invalid_response",
                          stage="response")


def test_the_error_taxonomy_is_closed_and_value_free() -> None:
    """C06: fixed codes and stages; no supplied value is echoed."""
    assert len(RETRIEVAL_ERROR_CODES) == 14
    assert RETRIEVAL_ERROR_STAGES
    error = RetrievalError("access_denied", "grant", request_id="request-1")
    assert (error.code, error.stage, error.request_id) == (
        "access_denied", "grant", "request-1")
    assert "request-1" in str(error)
    for code in RETRIEVAL_ERROR_CODES:
        assert RetrievalError(code, "input").code == code
    with pytest.raises(ValueError):
        RetrievalError("invented", "input")
    with pytest.raises(ValueError):
        RetrievalError("invalid_input", "invented")
    with pytest.raises(TypeError):
        RetrievalError("invalid_input", "input", request_id="x" * 129)


def test_an_audit_event_binds_its_own_canonical_payload() -> None:
    """C07: an event cannot claim one payload and persist another."""
    payload = {"request_id": "request-1", "status": "ok"}
    raw = canonical_json_bytes(payload)
    event = build_event(run_id="run-1", plan_id="plan-1", sequence=1,
                        kind="provider_returned", request_id="request-1", turn=1,
                        payload=payload)
    assert (event.payload_sha256, event.payload_bytes) == (
        hashlib.sha256(raw).hexdigest(), len(raw))
    with pytest.raises(ValidationError):
        RetrievalAuditEvent(
            schema_version=SCHEMA, run_id="run-1", plan_id="plan-1", sequence=1,
            kind="provider_returned", request_id="request-1", turn=1, payload=payload,
            payload_sha256="b" * 64, payload_bytes=len(raw))
    with pytest.raises(ValidationError):
        RetrievalAuditEvent(
            schema_version=SCHEMA, run_id="run-1", plan_id="plan-1", sequence=1,
            kind="provider_returned", request_id="request-1", turn=1, payload=payload,
            payload_sha256=hashlib.sha256(raw).hexdigest(), payload_bytes=len(raw) + 1)


def test_the_audit_event_kinds_are_the_closed_vocabulary() -> None:
    """C07: the nine declared kinds, no more and no fewer."""
    assert set(AUDIT_EVENT_KINDS) == {
        "plan_started", "request_prepared", "provider_returned", "read_requested",
        "read_returned", "forecast_ready", "plan_failed", "plan_cancelled", "plan_closed"}
    with pytest.raises(ValidationError):
        build_event(run_id="run-1", plan_id="plan-1", sequence=1, kind="invented",
                    payload={})
    with pytest.raises(ValidationError):
        build_event(run_id="run-1", plan_id="plan-1", sequence=1, kind="request_prepared",
                    payload={})
    with pytest.raises(ValidationError):
        build_event(run_id="run-1", plan_id="plan-1", sequence=0, kind="plan_started",
                    payload={})


# --------------------------------------------------------------------------- shared fixtures


def _planner_context():
    from forecontract.models import AdapterCapabilities, ExpectedOutcome
    from forecontract.verification.models import TaskCompletionCriteria
    from forecontract_llm.models import PlannerContext

    capabilities = AdapterCapabilities(
        schema_version=SCHEMA, adapter_name="synthetic", environment_type="offline",
        supported_actions=["mark_done"], supported_outcome_types=["json_rule"])
    criteria = TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id="goal-1",
        conditions=[ExpectedOutcome(schema_version=SCHEMA, outcome_type="json_rule")])
    return PlannerContext(
        schema_version=SCHEMA, run_id="run-1", goal_id="goal-1", goal_text="finish",
        task_criteria=criteria, capabilities=capabilities,
        action_guidance={"mark_done": {"payload": {"type": "object"}}},
        outcome_guidance={"json_rule": {"payload": {"type": "object"}}})


def _observation():
    from datetime import datetime, timezone

    from forecontract.models import Observation

    return Observation(schema_version=SCHEMA, observation_id="obs-1",
                       environment_type="offline", state={"done": False},
                       timestamp=datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc))
