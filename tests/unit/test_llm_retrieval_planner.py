"""Acceptance tests for the bounded retrieval round trip.

C02/C04/C05/C06/C07: at most two authorized reads and three provider calls per plan, one
monotonic deadline, intent-before-call audit records, a strict message protocol, and a
Forecast that still passes the original Core codec. Nothing here may fabricate a forecast after
a refusal, a missing audit record or an unauthorized read.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone

import pytest
from forecontract.models import AdapterCapabilities, ExpectedOutcome, Observation
from forecontract.verification.models import TaskCompletionCriteria
from forecontract_llm.models import (
    ModelResponse,
    PlannerConfig,
    PlannerContext,
)
from forecontract_llm.retrieval import (
    EvidenceReadResult,
    EvidenceScope,
    RetrievalConfig,
    RetrievalError,
)
from forecontract_llm.retrieval import planner as planner_module
from forecontract_llm.retrieval.models import (
    SCOPE_MAX_BYTES,
    EvidenceScopeEntry,
    ScopeView,
    canonical_json_bytes,
)
from forecontract_llm.retrieval.planner import EvidenceLLMPlanner

SCHEMA = "0.1"
RUN = "run-1"
OBS = "obs-1"
HEX = "a" * 64


# --------------------------------------------------------------------------- host fixtures


def _context() -> PlannerContext:
    capabilities = AdapterCapabilities(
        schema_version=SCHEMA, adapter_name="synthetic", environment_type="offline",
        supported_actions=["mark_done"], supported_outcome_types=["json_rule"])
    criteria = TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id="goal-1",
        conditions=[ExpectedOutcome(schema_version=SCHEMA, outcome_type="json_rule")])
    return PlannerContext(
        schema_version=SCHEMA, run_id=RUN, goal_id="goal-1", goal_text="finish",
        task_criteria=criteria, capabilities=capabilities,
        action_guidance={"mark_done": {"payload": {"type": "object"}}},
        outcome_guidance={"json_rule": {"payload": {"type": "object"}}})


def _config(*, max_reads: int = 2, max_model_calls: int = 3,
            call_timeout_ms: float = 5_000, cleanup_timeout_ms: float = 100,
            max_calls: int = 20) -> RetrievalConfig:
    return RetrievalConfig(
        schema_version=SCHEMA,
        base=PlannerConfig(schema_version=SCHEMA, call_timeout_ms=call_timeout_ms,
                           cleanup_timeout_ms=cleanup_timeout_ms, max_calls=max_calls),
        max_reads=max_reads, max_model_calls=max_model_calls)


def _observation(state: dict | None = None, observation_id: str = OBS) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="offline",
                       state=state if state is not None else
                       {"done": False, "order": {"status": "pending"}, "note": "hello"},
                       timestamp=datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc))


def _entry(evidence_id: str = "evidence-1") -> EvidenceScopeEntry:
    return EvidenceScopeEntry(
        schema_version=SCHEMA, evidence_id=evidence_id,
        source_observation_id=OBS, kind="observation", raw_sha256=HEX, raw_bytes=10,
        views=[
            ScopeView(schema_version=SCHEMA, view_id="status",
                      selector=["state", "order", "status"], format="text"),
            ScopeView(schema_version=SCHEMA, view_id="note",
                      selector=["state", "note"], format="text"),
            ScopeView(schema_version=SCHEMA, view_id="flag",
                      selector=["state", "done"], format="json"),
        ])


def _scope(*, entries: list[EvidenceScopeEntry] | None = None,
           observation_id: str = OBS, run_id: str = RUN) -> EvidenceScope:
    return EvidenceScope(schema_version=SCHEMA, scope_id="scope-1", run_id=run_id,
                         observation_id=observation_id,
                         entries=entries if entries is not None else [_entry()])


class StubReader:
    """A host reader double whose results are built from its own frozen scope."""

    def __init__(self, scope: EvidenceScope | None = None, *,
                 missing: set[tuple[str, str]] | None = None,
                 fail: bool = False) -> None:
        self.scope = scope or _scope()
        self.missing = missing or set()
        self.fail = fail
        self.reads = []
        self.closed: list[str] = []
        self.overrides: dict[tuple, EvidenceReadResult] = {}
        self.observed: list[tuple[str, str]] = []

    def open_scope(self, run_id: str, observation_id: str) -> EvidenceScope:
        self.observed.append((run_id, observation_id))
        if run_id != self.scope.run_id or observation_id != self.scope.observation_id:
            raise RetrievalError("access_denied", "scope")
        return self.scope

    def _result(self, request) -> EvidenceReadResult:
        entry = next(item for item in self.scope.entries
                     if item.evidence_id == request.evidence_id)
        view = next(item for item in entry.views if item.view_id == request.view_id)
        if (request.evidence_id, request.view_id) in self.missing:
            return EvidenceReadResult(
                schema_version=SCHEMA, scope_id=request.scope_id, run_id=request.run_id,
                plan_id=request.plan_id, turn=request.turn,
                request_id=request.request_id, evidence_id=request.evidence_id,
                view_id=request.view_id, status="not_found",
                source_observation_id=entry.source_observation_id,
                raw_sha256=entry.raw_sha256, raw_bytes=entry.raw_bytes,
                selector=list(view.selector), format=view.format,
                text_range=request.text_range, partial=False, value=None,
                value_sha256=None, value_bytes=0)
        if view.format == "text":
            if request.view_id == "status":
                full = "pending"
            else:
                full = "hello"
            if request.text_range is None:
                value = full
                partial = False
            else:
                start, end = request.text_range.start, request.text_range.end
                value = full[start:end]
                partial = (start, end) != (0, len(full))
        else:
            value = bool(self.scope.entries[0].evidence_id)
            partial = False
        raw = canonical_json_bytes(value)
        return EvidenceReadResult(
            schema_version=SCHEMA, scope_id=request.scope_id, run_id=request.run_id,
            plan_id=request.plan_id, turn=request.turn, request_id=request.request_id,
            evidence_id=request.evidence_id, view_id=request.view_id, status="found",
            source_observation_id=entry.source_observation_id,
            raw_sha256=entry.raw_sha256, raw_bytes=entry.raw_bytes,
            selector=list(view.selector), format=view.format,
            text_range=request.text_range, partial=partial, value=value,
            value_sha256=hashlib.sha256(raw).hexdigest(), value_bytes=len(raw))

    async def read(self, request):
        self.reads.append(request)
        if self.fail:
            raise RuntimeError("the reader is not cooperating")
        override = self.overrides.get(
            (request.evidence_id, request.view_id,
             None if request.text_range is None else
             (request.text_range.start, request.text_range.end)))
        if override is not None:
            return override
        return self._result(request)

    def close_scope(self, scope_id: str) -> None:
        self.closed.append(scope_id)


class MemorySink:
    """A deterministic in-memory audit sink for unit-level round trips."""

    def __init__(self) -> None:
        self.events = []
        self.fail_on: set[str] = set()
        self.attempts: list[str] = []

    async def append(self, event) -> None:
        self.attempts.append(event.kind)
        if event.kind in self.fail_on:
            raise RuntimeError("the audit sink is unavailable")
        self.events.append(event)


def _read_content(request, evidence_id: str = "evidence-1", view_id: str = "status",
                  text_range: dict | None = None) -> str:
    return json.dumps({
        "schema_version": SCHEMA, "run_id": request.run_id, "plan_id": request.plan_id,
        "turn": request.turn, "request_id": request.request_id, "kind": "read_evidence",
        "read": {"evidence_id": evidence_id, "view_id": view_id,
                 "text_range": text_range}})


def _forecast_content(request, *, payload: dict | None = None,
                      action_type: str = "mark_done") -> str:
    return json.dumps({
        "schema_version": SCHEMA, "run_id": request.run_id, "plan_id": request.plan_id,
        "turn": request.turn, "request_id": request.request_id, "kind": "forecast",
        "forecast": {
            "schema_version": SCHEMA,
            "proposed_action": {"schema_version": SCHEMA, "action_type": action_type,
                                "payload": payload or {}},
            "expected_outcomes": [
                {"schema_version": SCHEMA, "outcome_type": "json_rule",
                 "payload": {"schema_version": SCHEMA, "rule_type": "value_equals",
                             "path": ["done"], "expected": True}}],
            "task_completion_hint": True,
        },
    })


class ScriptedProvider:
    """One answer per call, built from the request the planner really sent."""

    def __init__(self, answers: list) -> None:
        self._answers = list(answers)
        self.requests: list = []
        self.calls = 0
        self.sink: MemorySink | None = None
        self.reader: StubReader | None = None

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        self.requests.append(request)
        if self.sink is not None:
            assert self.sink.events and self.sink.events[-1].kind == "request_prepared"
        if self.reader is not None:
            for exchange in request.exchanges:
                assert exchange.status in ("found", "not_found")
        if not self._answers:
            raise RuntimeError("no scripted answer left")
        answer = self._answers.pop(0)
        if callable(answer):
            answer = answer(request)
        if isinstance(answer, ModelResponse):
            return answer
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="ok", content=answer)


def _planner(provider, *, reader: StubReader | None = None, sink: MemorySink | None = None,
             config: RetrievalConfig | None = None) -> tuple[EvidenceLLMPlanner, StubReader,
                                                             MemorySink]:
    current_reader = reader or StubReader()
    current_sink = sink or MemorySink()
    planner = EvidenceLLMPlanner(
        provider=provider, context=_context(), config=config or _config(),
        reader=current_reader, audit=current_sink)
    return planner, current_reader, current_sink


def _run(planner, observation: Observation | None = None):
    async def main():
        try:
            return await planner.plan(observation or _observation())
        finally:
            await planner.aclose()

    return asyncio.run(main())


def _error(call) -> RetrievalError:
    with pytest.raises(RetrievalError) as info:
        call()
    return info.value


def _kinds(sink: MemorySink) -> list[str]:
    return [event.kind for event in sink.events]


# --------------------------------------------------------------------------- happy paths


def test_a_direct_forecast_needs_no_read_and_one_call() -> None:
    """C05: with zero reads the first answer may already be a forecast."""
    provider = ScriptedProvider([_forecast_content])
    planner, reader, sink = _planner(provider, config=_config(max_reads=0))
    forecast = _run(planner)
    assert forecast.proposed_action.action_type == "mark_done"
    assert provider.calls == 1
    assert reader.reads == []
    assert _kinds(sink) == ["plan_started", "request_prepared", "provider_returned",
                            "forecast_ready", "plan_closed"]
    assert (planner.calls_used, planner.state) == (1, "closed")


def test_one_read_then_a_forecast_repeats_the_result_in_the_next_request() -> None:
    """C02/C04: the second request carries the complete read result, and the action uses it."""
    def forecast_from_status(request):
        value = request.exchanges[-1].value
        return _forecast_content(request, payload={"observed": value})

    provider = ScriptedProvider([_read_content, forecast_from_status])
    planner, reader, sink = _planner(provider)
    forecast = _run(planner)
    assert forecast.proposed_action.payload == {"observed": "pending"}
    assert provider.calls == 2
    assert len(provider.requests[1].exchanges) == 1
    exchange = provider.requests[1].exchanges[0]
    assert (exchange.status, exchange.value) == ("found", "pending")
    assert _kinds(sink) == [
        "plan_started", "request_prepared", "provider_returned", "read_requested",
        "read_returned", "request_prepared", "provider_returned", "forecast_ready",
        "plan_closed"]
    assert [event.sequence for event in sink.events] == list(range(1, 10))
    assert reader.closed == ["scope-1"]


def test_the_forecast_metadata_comes_from_the_original_core_codec() -> None:
    """C02: evidence content never replaces the run, goal or observation identity."""
    provider = ScriptedProvider([_read_content,
                                 lambda request: _forecast_content(
                                     request, payload={"observed": "pending"})])
    planner, _reader, _sink = _planner(provider)
    forecast = _run(planner)
    assert forecast.metadata == {
        "planner": "llm", "run_id": RUN, "goal_id": "goal-1",
        "observation_id": OBS, "request_id": forecast.forecast_id,
    }


def test_validated_result_used_for_audit() -> None:
    """F03: measurement, audit and the next request use the detached validated result.

    A reader may return a subclass whose ``model_dump`` lies. The planner must copy the
    result from its real storage, validate that copy and use only that copy afterwards.
    """
    class ForgedResult(EvidenceReadResult):
        def model_dump(self, *args, **kwargs):
            document = super().model_dump(*args, **kwargs)
            document["value"] = "AUDIT-FORGED"
            return document

    class ForgingReader(StubReader):
        async def read(self, request):
            result = await super().read(request)
            return ForgedResult.model_validate(result.model_dump(mode="python"))

    provider = ScriptedProvider([_read_content, _forecast_content])
    planner, _reader, sink = _planner(provider, reader=ForgingReader())
    forecast = _run(planner)
    returned = [event for event in sink.events if event.kind == "read_returned"]
    assert len(returned) == 1
    assert returned[0].payload["result"]["value"] == "pending"
    assert returned[0].payload["result"]["value_sha256"] == hashlib.sha256(
        canonical_json_bytes("pending")).hexdigest()
    assert provider.requests[1].exchanges[0].value == "pending"
    assert forecast.proposed_action.action_type == "mark_done"


def test_two_reads_are_allowed_and_a_third_is_refused_before_the_reader() -> None:
    """C05: the third read terminates the plan instead of asking the model to repair it."""
    provider = ScriptedProvider([
        _read_content,
        lambda request: _read_content(request, view_id="note"),
        lambda request: _read_content(request, text_range={"start": 0, "end": 3}),
        _forecast_content,
    ])
    planner, reader, sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("budget_exceeded", "budget")
    assert len(reader.reads) == 2
    assert provider.calls == 3
    assert _kinds(sink)[-2:] == ["plan_failed", "plan_closed"]


def test_a_not_found_read_still_consumes_the_read_budget() -> None:
    """C05: a legal miss is charged like any other read."""
    provider = ScriptedProvider([
        _read_content,
        lambda request: _read_content(request, view_id="note"),
    ])
    reader = StubReader(missing={("evidence-1", "status")})
    planner, reader, _sink = _planner(provider, reader=reader, config=_config(max_reads=1))
    error = _error(lambda: _run(planner))
    assert error.code == "budget_exceeded"
    assert len(reader.reads) == 1
    assert provider.calls == 2


def test_a_duplicate_read_is_refused_even_after_not_found() -> None:
    """C05: the same evidence, view and range may be requested only once per plan."""
    provider = ScriptedProvider([_read_content, _read_content])
    reader = StubReader(missing={("evidence-1", "status")})
    planner, reader, _sink = _planner(provider, reader=reader)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("duplicate_read", "selection")
    assert len(reader.reads) == 1


def test_max_reads_zero_refuses_a_read_before_the_reader_is_called() -> None:
    """C05: a zero read budget still permits a direct forecast and refuses any read."""
    provider = ScriptedProvider([_read_content, _forecast_content])
    planner, reader, _sink = _planner(provider, config=_config(max_reads=0))
    error = _error(lambda: _run(planner))
    assert error.code == "budget_exceeded"
    assert reader.reads == []
    assert provider.calls == 1


def test_a_read_on_the_last_allowed_call_ends_in_a_budget_failure() -> None:
    """C05: the model-call allowance is bounded by both the cap and max_reads + 1."""
    provider = ScriptedProvider([_read_content, _forecast_content])
    planner, reader, _sink = _planner(provider, config=_config(max_model_calls=1))
    error = _error(lambda: _run(planner))
    assert error.code == "budget_exceeded"
    assert (provider.calls, len(reader.reads)) == (1, 1)


# --------------------------------------------------------------------------- instance budget


def test_the_instance_call_allowance_is_cumulative_across_plans() -> None:
    """C05: base.max_calls counts every provider attempt of the instance."""
    provider = ScriptedProvider([_forecast_content, _forecast_content])
    planner, _reader, _sink = _planner(provider, config=_config(max_reads=0, max_calls=1))

    async def main():
        assert (await planner.plan(_observation())).proposed_action.action_type == "mark_done"
        with pytest.raises(RetrievalError) as info:
            await planner.plan(_observation())
        await planner.aclose()
        return info.value

    error = asyncio.run(main())
    assert (error.code, error.stage) == ("budget_exceeded", "budget")
    assert (planner.calls_used, provider.calls) == (1, 1)


def test_the_plan_request_budget_is_charged_per_complete_request(monkeypatch) -> None:
    """C05: every fully sent request is charged again, and the exact boundary is honoured."""
    provider = ScriptedProvider([_read_content,
                                 lambda request: _read_content(request, view_id="note")])
    planner, _reader, _sink = _planner(provider)
    _error(lambda: _run(planner))
    first = canonical_json_bytes(provider.requests[0].model_dump(mode="json"))
    monkeypatch.setattr(planner_module, "PLAN_REQUEST_MAX_BYTES", len(first))
    provider = ScriptedProvider([_read_content,
                                 lambda request: _read_content(request, view_id="note")])
    planner, reader, _sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert error.code == "budget_exceeded"
    assert len(reader.reads) == 1
    assert provider.calls == 1


def test_the_plan_response_budget_is_charged_per_complete_response(monkeypatch) -> None:
    """C05: the second response pushes the cumulative content bytes over the cap."""
    first_run = ScriptedProvider([_read_content,
                                  lambda request: _forecast_content(request)])
    planner, _reader, _sink = _planner(first_run)
    _run(planner)
    size = len(_read_content(first_run.requests[0]).encode("utf-8"))
    monkeypatch.setattr(planner_module, "PLAN_RESPONSE_MAX_BYTES", size)
    provider = ScriptedProvider([_read_content,
                                 lambda request: _forecast_content(request)])
    planner, reader, _sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert error.code == "budget_exceeded"
    assert len(reader.reads) == 1
    assert provider.calls == 2


def test_the_plan_result_budget_is_charged_per_complete_result(monkeypatch) -> None:
    """C05: the returned results are charged together and refused over their plan cap."""
    monkeypatch.setattr(planner_module, "PLAN_RESULT_MAX_BYTES", 10)
    provider = ScriptedProvider([_read_content, _forecast_content])
    planner, reader, _sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert error.code == "budget_exceeded"
    assert len(reader.reads) == 1
    assert provider.calls == 1


# --------------------------------------------------------------------------- scope trust boundary


def _sized_scope(total_bytes: int) -> EvidenceScope:
    """A scope whose canonical directory is exactly ``total_bytes`` (padding the scope id)."""
    base = _scope()
    base_size = len(canonical_json_bytes(base.model_dump(mode="json")))
    assert total_bytes >= base_size
    return EvidenceScope(
        schema_version=SCHEMA, scope_id="scope-1" + "x" * (total_bytes - base_size),
        run_id=RUN, observation_id=OBS, entries=[_entry()])


def test_scope_directory_limit_at_planner() -> None:
    """F02: the planner enforces the 32,768-byte directory cap before the provider call.

    The reader's own check is not the planner's trust boundary: an injected reader that
    returns exactly the cap is accepted, one byte more is refused with no provider call.
    """
    exact = _sized_scope(SCOPE_MAX_BYTES)
    assert len(canonical_json_bytes(exact.model_dump(mode="json"))) == SCOPE_MAX_BYTES
    provider = ScriptedProvider([_forecast_content])
    planner, _reader, _sink = _planner(
        provider, reader=StubReader(exact), config=_config(max_reads=0))
    forecast = _run(planner)
    assert forecast.proposed_action.action_type == "mark_done"
    assert provider.calls == 1

    over = _sized_scope(SCOPE_MAX_BYTES + 1)
    assert len(canonical_json_bytes(over.model_dump(mode="json"))) == SCOPE_MAX_BYTES + 1
    provider = ScriptedProvider([_forecast_content])
    planner, reader, sink = _planner(
        provider, reader=StubReader(over), config=_config(max_reads=0))
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("budget_exceeded", "scope")
    assert provider.calls == 0
    assert reader.reads == []
    assert _kinds(sink)[-2:] == ["plan_failed", "plan_closed"]


# --------------------------------------------------------------------------- audit protocol


def test_every_intent_is_persisted_before_its_external_call() -> None:
    """C07: no provider or reader call happens before its intent record."""
    sink = MemorySink()
    provider = ScriptedProvider([_read_content, _forecast_content])
    reader = StubReader()
    provider.sink = sink
    provider.reader = reader
    planner, _reader, _sink = _planner(provider, reader=reader, sink=sink)
    _run(planner)
    kinds = _kinds(sink)
    for index, kind in enumerate(kinds):
        if kind == "provider_returned":
            assert kinds[index - 1] == "request_prepared"
        if kind == "read_returned":
            assert kinds[index - 1] == "read_requested"


def test_an_audit_failure_before_the_provider_stops_the_call() -> None:
    """C07: a result record that cannot be persisted forbids the external call."""
    sink = MemorySink()
    sink.fail_on = {"request_prepared"}
    provider = ScriptedProvider([_forecast_content])
    planner, _reader, _sink = _planner(provider, sink=sink)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("audit_failed", "audit")
    assert provider.calls == 0
    assert planner.audit_unconverged is True


def test_an_audit_failure_after_the_provider_forbids_a_forecast() -> None:
    """C07: a failed result record forbids returning a forecast."""
    sink = MemorySink()
    sink.fail_on = {"forecast_ready"}
    provider = ScriptedProvider([_forecast_content])
    planner, _reader, _sink = _planner(provider, sink=sink, config=_config(max_reads=0))
    error = _error(lambda: _run(planner))
    assert error.code == "audit_failed"
    assert provider.calls == 1
    assert planner.audit_unconverged is True


def test_an_audit_failure_before_a_read_forbids_the_read() -> None:
    """C07: the read intent is persisted before the reader is called."""
    sink = MemorySink()
    sink.fail_on = {"read_requested"}
    provider = ScriptedProvider([_read_content])
    reader = StubReader()
    planner, _reader, _sink = _planner(provider, reader=reader, sink=sink)
    error = _error(lambda: _run(planner))
    assert error.code == "audit_failed"
    assert reader.reads == []
    assert provider.calls == 1


# --------------------------------------------------------------------------- refusals


def test_a_provider_refusal_ends_the_plan_without_another_call() -> None:
    """C02: refused terminates; it is not a forecast and not a repair prompt."""
    def refused(request):
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="refused")

    provider = ScriptedProvider([refused])
    planner, _reader, sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("provider_refused", "provider")
    assert provider.calls == 1
    assert _kinds(sink)[-2:] == ["plan_failed", "plan_closed"]


def test_a_provider_failure_is_one_bounded_code() -> None:
    """C06: a provider exception is normalised, never echoed."""
    provider = ScriptedProvider([lambda request: 1 / 0])
    planner, _reader, sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("provider_failed", "provider")
    assert "ZeroDivision" not in str(error)
    assert _kinds(sink)[-2:] == ["plan_failed", "plan_closed"]


def test_a_reader_failure_is_one_bounded_code() -> None:
    """C06: a reader exception is normalised, never echoed."""
    provider = ScriptedProvider([_read_content])
    reader = StubReader(fail=True)
    planner, _reader, _sink = _planner(provider, reader=reader)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("reader_failed", "reader")


def test_a_provider_response_for_another_request_is_refused() -> None:
    """C02/C06: the response must name the current request."""
    class ForeignProvider:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            return ModelResponse(schema_version=SCHEMA, request_id="someone-else",
                                 status="ok", content=_forecast_content(request))

    provider = ForeignProvider()
    planner, _reader, sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("binding_mismatch", "binding")
    assert provider.calls == 1
    assert _kinds(sink)[-2:] == ["plan_failed", "plan_closed"]


def test_invalid_response_preserved() -> None:
    """F05: a real but wrongly bound provider return is persisted before the plan ends.

    The response shell fails its request binding, yet the audit must record the actual
    returned content and identity instead of pretending the provider never returned.
    """
    def foreign(request):
        return ModelResponse(
            schema_version=SCHEMA, request_id="someone-else", status="ok",
            content=_forecast_content(request))

    provider = ScriptedProvider([foreign])
    planner, _reader, sink = _planner(provider, config=_config(max_reads=0))
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("binding_mismatch", "binding")
    returned = [event for event in sink.events if event.kind == "provider_returned"]
    assert len(returned) == 1
    payload = returned[0].payload
    assert payload["request_id"] == "someone-else"
    assert payload["status"] == "ok"
    assert payload["usage_known"] is False
    assert payload["content"] is not None
    assert '"kind": "forecast"' in payload["content"]
    assert provider.calls == 1


def test_an_unauthorized_read_is_refused_without_calling_the_reader() -> None:
    """C03/C06: identity is refused before existence and before any external call."""
    provider = ScriptedProvider([
        lambda request: _read_content(request, evidence_id="not-granted"),
    ])
    planner, reader, _sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("access_denied", "grant")
    assert reader.reads == []
    provider = ScriptedProvider([lambda request: _read_content(request, view_id="not-granted")])
    planner, reader, _sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert error.code == "access_denied"
    assert reader.reads == []


def test_a_text_range_on_a_json_view_is_refused_before_the_reader() -> None:
    """C03: the JSON projection takes no interval; the reader is never called."""
    provider = ScriptedProvider([
        lambda request: _read_content(request, view_id="flag",
                                      text_range={"start": 0, "end": 1}),
    ])
    planner, reader, _sink = _planner(provider)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("selection_invalid", "selection")
    assert reader.reads == []


def test_a_reader_result_with_a_foreign_binding_is_refused() -> None:
    """C06: a returned result is revalidated against the request and the frozen scope."""
    provider = ScriptedProvider([_read_content])
    reader = StubReader()
    original = reader.read

    async def wrong_turn(request):
        result = await original(request)
        return result.model_copy(update={"turn": result.turn + 1})

    reader.read = wrong_turn
    planner, _reader, _sink = _planner(provider, reader=reader)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("binding_mismatch", "binding")


def test_a_tampered_reader_result_is_refused() -> None:
    """C06: a changed snapshot hash is a binding failure, not accepted evidence."""
    provider = ScriptedProvider([_read_content])
    reader = StubReader()
    original = reader.read

    async def tampered(request):
        result = await original(request)
        return result.model_copy(update={"raw_sha256": "b" * 64})

    reader.read = tampered
    planner, _reader, _sink = _planner(provider, reader=reader)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("binding_mismatch", "binding")


def test_the_planner_state_and_counters_are_honest() -> None:
    """C06: ready/busy/closed and the cumulative call count are observable facts."""
    provider = ScriptedProvider([_forecast_content])
    planner, _reader, _sink = _planner(provider, config=_config(max_reads=0))
    assert (planner.calls_used, planner.state) == (0, "ready")
    _run(planner)
    assert (planner.calls_used, planner.state) == (1, "closed")


def test_a_refused_plan_still_counts_its_attempt() -> None:
    """C05: a refused provider answer is charged and never refunded."""
    def refused(request):
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="refused")

    provider = ScriptedProvider([refused])
    planner, _reader, _sink = _planner(provider, config=_config(max_reads=0, max_calls=1))
    _error(lambda: _run(planner))
    assert planner.calls_used == 1


def test_a_second_plan_is_refused_after_close() -> None:
    """C06: a closed planner accepts no new plan."""
    planner, _reader, _sink = _planner(ScriptedProvider([_forecast_content]),
                                       config=_config(max_reads=0))
    _run(planner)
    error = _error(lambda: _run(planner))
    assert (error.code, error.stage) == ("invalid_state", "lifecycle")


def test_a_plan_while_busy_is_refused() -> None:
    """C02: one instance supports no concurrent plan."""
    class SlowProvider:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()

        async def complete(self, request):
            self.calls += 1
            self.started.set()
            await asyncio.sleep(0.2)
            return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                                 status="ok", content=_forecast_content(request))

    provider = SlowProvider()
    planner, _reader, _sink = _planner(provider, config=_config(max_reads=0))

    async def main():
        first = asyncio.create_task(planner.plan(_observation()))
        await provider.started.wait()
        with pytest.raises(RetrievalError) as info:
            await planner.plan(_observation())
        assert (info.value.code, info.value.stage) == ("invalid_state", "lifecycle")
        await first
        await planner.aclose()

    asyncio.run(main())
