"""Acceptance tests for the retrieval planner through the real Core Runtime.

C01/C04/C09: the real Runtime, Contract Builder, StepVerifier, TaskCompletionOracle, Recovery
Engine and Fake adapter run with :class:`EvidenceLLMPlanner` and a real registered snapshot
reader. The retrieval proposal is still only a proposal: Core keeps the contract, execution,
verification and terminal authority, and a Core deadline really cancels the retrieval plan.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_SRC = REPO_ROOT / "packages" / "planners" / "llm" / "src"
INTEGRATION_DIR = Path(__file__).resolve().parent
for candidate in (LLM_SRC, INTEGRATION_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from conftest import (
    FLAG,
    DeclaredAdapter,
    HostDomain,
    action,
    build_engine,
    fake_scenario,
    request_for,
)
from forecontract_fake import FakeEnvironmentAdapter
from forecontract_llm.models import ModelResponse, PlannerConfig, PlannerContext
from forecontract_llm.retrieval import (
    EvidenceGrant,
    EvidenceLLMPlanner,
    RegisteredEvidenceReader,
    RetrievalConfig,
)
from forecontract_llm.retrieval.models import EvidenceReadResult, canonical_json_bytes

SCHEMA = "0.1"
FIXED_TIME = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _context(adapter, request) -> PlannerContext:
    capabilities = adapter.capabilities()
    return PlannerContext(
        schema_version=SCHEMA, run_id=request.run_id, goal_id=request.goal_id,
        goal_text="finish the retrieval task", task_criteria=request.task_criteria,
        capabilities=capabilities,
        action_guidance={name: {"payload": {"type": "object"}}
                         for name in capabilities.supported_actions},
        outcome_guidance={name: {"payload": {"type": "object"}}
                          for name in capabilities.supported_outcome_types})


def _config(*, call_timeout_ms: float = 5000) -> RetrievalConfig:
    return RetrievalConfig(
        schema_version=SCHEMA,
        base=PlannerConfig(schema_version=SCHEMA, call_timeout_ms=call_timeout_ms,
                           cleanup_timeout_ms=100),
    )


class HostRecorder:
    """The host-side snapshot writer and registration step, one per observe call."""

    def __init__(self, root: Path, run_id: str, *, grant: bool = True,
                 selector: list[object] | None = None) -> None:
        self.root = root
        self.run_id = run_id
        self.grant = grant
        self.selector = selector or ["state", FLAG]
        self.ordinal = 0
        self.evidence_ids: list[str] = []
        self.reader = RegisteredEvidenceReader(root, run_id, _config())

    def write(self, observation) -> dict:
        self.ordinal += 1
        document = {
            "schema_version": SCHEMA, "kind": "observation", "run_id": self.run_id,
            "ordinal": self.ordinal,
            "recorded_at": FIXED_TIME.isoformat(),
            "associations": {"call_ordinal": self.ordinal,
                             "observation_id": observation.observation_id},
            "payload": observation.model_dump(mode="json"),
        }
        raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
        relative = f"snapshots/observation-{self.ordinal:04d}.json"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "kind": "observation", "ordinal": self.ordinal, "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "call_ordinal": self.ordinal,
            "observation_id": observation.observation_id,
        }

    def register(self, observation) -> None:
        entry = self.write(observation)
        grants = []
        if self.grant:
            evidence_id = f"evidence-{self.ordinal:04d}"
            self.evidence_ids.append(evidence_id)
            grants.append(EvidenceGrant(
                schema_version=SCHEMA, run_id=self.run_id, evidence_id=evidence_id,
                observation_id=observation.observation_id, view_id="status",
                selector=list(self.selector), format="json"))
        self.reader.register_observation(observation, entry, grants)


class ObservingAdapter:
    """Forward every adapter call unchanged, registering each returned observation."""

    def __init__(self, inner, recorder: HostRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    @property
    def name(self):
        return self._inner.name

    def capabilities(self):
        return self._inner.capabilities()

    def validate_action(self, action_spec):
        return self._inner.validate_action(action_spec)

    async def start(self):
        await self._inner.start()

    async def stop(self):
        await self._inner.stop()

    async def observe(self):
        observation = await self._inner.observe()
        self._recorder.register(observation)
        return observation

    async def execute(self, action_spec):
        return await self._inner.execute(action_spec)


class FieldProvider:
    """Ask for one granted view, then build the action payload from the returned value."""

    def __init__(self, *, direct: bool = False, reject: bool = False) -> None:
        self.direct = direct
        self.reject = reject
        self.calls = 0
        self.requests: list = []

    async def complete(self, request) -> ModelResponse:
        self.calls += 1
        self.requests.append(request)
        if not self.direct and not request.exchanges:
            entry = request.scope.entries[0]
            view = entry.views[0]
            content = json.dumps({
                "schema_version": SCHEMA, "run_id": request.run_id,
                "plan_id": request.plan_id, "turn": request.turn,
                "request_id": request.request_id, "kind": "read_evidence",
                "read": {"evidence_id": entry.evidence_id, "view_id": view.view_id,
                         "text_range": None}})
        else:
            if self.direct:
                observed = request.base_request.observation.state["mode"]
            else:
                observed = request.exchanges[-1].value
            payload = {"observed": "outside-the-catalog" if self.reject else observed}
            content = json.dumps({
                "schema_version": SCHEMA, "run_id": request.run_id,
                "plan_id": request.plan_id, "turn": request.turn,
                "request_id": request.request_id, "kind": "forecast",
                "forecast": {
                    "schema_version": SCHEMA,
                    "proposed_action": {"schema_version": SCHEMA,
                                        "action_type": "mark_done", "payload": payload},
                    "expected_outcomes": [
                        {"schema_version": SCHEMA, "outcome_type": "json_rule",
                         "payload": {"schema_version": SCHEMA,
                                     "rule_type": "value_equals",
                                     "path": [FLAG], "expected": True}}],
                    "task_completion_hint": True}})
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="ok", content=content)


def _compose(tmp_path: Path, *, state: dict, mode_value: str = "a", direct: bool = False,
             reject: bool = False, call_timeout_ms: float = 5000,
             run_timeout_ms: float | None = 20000.0):
    scenario_action = action(payload={"observed": mode_value})
    scenario = fake_scenario(action_spec=scenario_action, start_state=state,
                             end_state={FLAG: True, "mode": mode_value})
    declared = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for()
    recorder = HostRecorder(tmp_path, request.run_id, grant=not direct,
                            selector=["state", "mode"])
    adapter = ObservingAdapter(declared, recorder)
    provider = FieldProvider(direct=direct, reject=reject)
    planner = EvidenceLLMPlanner(
        provider=provider, context=_context(declared, request),
        config=_config(call_timeout_ms=call_timeout_ms), reader=recorder.reader,
        audit=_MemorySink())
    engine, trace, _memory = build_engine(
        tmp_path=tmp_path, adapter=adapter, planner=planner, domain=HostDomain(),
        run_timeout_ms=run_timeout_ms)
    return engine, trace, planner, provider, declared, recorder, request


class _MemorySink:
    def __init__(self) -> None:
        self.events = []

    async def append(self, event) -> None:
        self.events.append(event)


def _run(engine, planner, request):
    async def main():
        try:
            return await engine.run(request=request)
        finally:
            return_value = await planner.aclose()
            assert return_value.drained is True

    return asyncio.run(main())


def test_the_real_core_engine_runs_a_retrieval_forecast(tmp_path: Path) -> None:
    """C01/C04: a real Core Run completes through plan -> build -> execute -> verify."""
    engine, trace, planner, provider, declared, recorder, request = _compose(
        tmp_path, state={FLAG: False, "mode": "a"})
    result = _run(engine, planner, request)
    assert (result.terminal_status.value, result.reason_code.value) == (
        "completed", "TASK_ACHIEVED")
    assert [item.payload for item in declared.actions] == [{"observed": "a"}]
    assert (result.steps_started, result.planning_calls, result.attempts_started) == (1, 1, 1)
    assert provider.calls == 2
    assert planner.calls_used == 2
    assert len(recorder.evidence_ids) >= 1
    reads = [exchange for exchange in provider.requests[-1].exchanges]
    assert len(reads) == 1 and reads[0].value == "a"
    assert planner.state == "closed"
    records = trace.read_run(request.run_id)
    assert [record.kind for record in records if record.kind == "run_finished"]
    assert any(record.kind == "contract_committed" for record in records)


def test_a_direct_forecast_is_legal_with_no_granted_view(tmp_path: Path) -> None:
    """C03/C05: a zero-grant scope is empty and a direct forecast still completes."""
    engine, _trace, planner, provider, _declared, recorder, request = _compose(
        tmp_path, state={FLAG: False, "mode": "a"}, direct=True)
    result = _run(engine, planner, request)
    assert (result.terminal_status.value, result.reason_code.value) == (
        "completed", "TASK_ACHIEVED")
    assert provider.calls == 1
    assert recorder.evidence_ids == []
    assert provider.requests[0].scope.entries == []


def test_two_real_snapshots_drive_two_different_actions(tmp_path: Path) -> None:
    """C04: the really returned field decides the really executed action payload."""
    executed: list[dict] = []
    for name, mode in (("run-a", "alpha"), ("run-b", "beta")):
        root = tmp_path / name
        root.mkdir()
        engine, _trace, planner, _provider, declared, _recorder, request = _compose(
            root, state={FLAG: False, "mode": mode}, mode_value=mode)
        result = _run(engine, planner, request)
        assert result.terminal_status.value == "completed"
        executed.append(declared.actions[0].model_dump(mode="json")["payload"])
    assert executed == [{"observed": "alpha"}, {"observed": "beta"}]
    assert executed[0] != executed[1]


def test_the_core_still_rejects_an_out_of_catalog_action(tmp_path: Path) -> None:
    """C01: the retrieval forecast is only a proposal; the Builder still refuses it."""
    engine, trace, planner, provider, _declared, _recorder, request = _compose(
        tmp_path, state={FLAG: False, "mode": "a"}, reject=True)
    result = _run(engine, planner, request)
    assert (result.terminal_status.value, result.reason_code.value) == (
        "failed", "EVIDENCE_INVALID")
    assert (result.steps_started, result.attempts_started) == (0, 0)
    assert provider.calls == 2
    assert [record.kind for record in trace.read_run(request.run_id)] == [
        "run_started", "run_finished"]


def test_a_core_deadline_really_cancels_the_retrieval_plan(tmp_path: Path) -> None:
    """C06/C09: a shorter Core deadline cancels the provider wait and publishes a timeout."""

    class NeverAnswering:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            await asyncio.sleep(5)
            return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                                 status="ok", content="{}")

    scenario_action = action(payload={})
    scenario = fake_scenario(action_spec=scenario_action, start_state={FLAG: False},
                             end_state={FLAG: True})
    declared = DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = request_for()
    recorder = HostRecorder(tmp_path, request.run_id, grant=True)
    adapter = ObservingAdapter(declared, recorder)
    provider = NeverAnswering()
    planner = EvidenceLLMPlanner(
        provider=provider, context=_context(declared, request), config=_config(),
        reader=recorder.reader, audit=_MemorySink())
    engine, _trace, _memory = build_engine(
        tmp_path=tmp_path, adapter=adapter, planner=planner, domain=HostDomain(),
        run_timeout_ms=250.0)
    result = _run(engine, planner, request)
    assert (result.terminal_status.value, result.reason_code.value) == (
        "timeout", "DEADLINE_EXCEEDED")
    assert provider.calls == 1
    assert planner.state == "closed"


def test_the_registered_view_is_the_read_value_the_model_receives(tmp_path: Path) -> None:
    """C03/C04: the text in the provider request is the registered snapshot value."""
    engine, _trace, planner, provider, _declared, _recorder, request = _compose(
        tmp_path, state={FLAG: False, "mode": "a"})
    _run(engine, planner, request)
    first = provider.requests[0]
    entry = first.scope.entries[0]
    assert entry.views[0].selector == ["state", "mode"]
    assert entry.views[0].format == "json"
    exchange = provider.requests[1].exchanges[0]
    assert isinstance(exchange, EvidenceReadResult)
    assert exchange.status == "found"
    assert exchange.value == "a"
    assert exchange.value_sha256 == hashlib.sha256(
        canonical_json_bytes("a")).hexdigest()
