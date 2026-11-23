"""Host-side DeepSeek smoke: the real Core loop with the existing recorders.

The assembly reuses the already-verified components: ``RuntimeEngine``, ``ContractBuilder``,
``StepVerifier``, ``TaskCompletionOracle``, ``RecoveryEngine``, the Fake adapter, the retrieval
planner, ``FileAuditSink``, ``RegisteringEvidenceRecorder`` and the ``Recording*`` forwarders.
It introduces no second recorder and no acceptance platform.

Evidence is written before any assertion and read back from the files: ``validate_case_evidence``
never trusts a summary's own claim. The default CLI is offline and never reads a credential.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from deepseek_provider import DeepSeekProvider, make_client
from forecontract.contracts import ContractBuilder
from forecontract.memory import WorkingMemory
from forecontract.models import (
    ActionSpec,
    ActualOutcome,
    ExpectedOutcome,
    Forecast,
    Observation,
)
from forecontract.recovery import ActionSafetyDeclaration, RecoveryEngine
from forecontract.runtime import (
    ContractPreparation,
    MonotonicBudgetSource,
    RunRequest,
    RuntimeEngine,
    RuntimeTimingPolicy,
)
from forecontract.tracing import JsonlTraceStore
from forecontract.verification import TaskCompletionCriteria, VerificationEvidence
from forecontract.verification.oracle import TaskCompletionOracle
from forecontract.verification.step import StepVerifier
from forecontract_fake import FakeEnvironmentAdapter, FakeScenario, FakeStage, FakeStep
from forecontract_llm.models import PlannerConfig, PlannerContext
from forecontract_llm.planner import LLMPlanner
from forecontract_llm.retrieval import (
    EvidenceGrant,
    EvidenceLLMPlanner,
    RegisteredEvidenceReader,
    RetrievalConfig,
)
from llm_retrieval_demo import FileAuditSink
from web_retrieval_evidence import RegisteringEvidenceRecorder
from web_runtime_evidence import (
    EvidenceRecorder,
    RecordingAdapter,
    RecordingBuilder,
    RecordingPlanner,
    RecordingRecoveryEngine,
    RecordingStepVerifier,
    RecordingTaskOracle,
)

__all__ = ["CASE_IDS", "load_cases", "main", "make_client", "run_case", "validate_case_evidence"]

SCHEMA: Final[str] = "0.1"
FIXED_TIME: Final[datetime] = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)
FIXTURE_ID: Final[str] = "p4-t02-smoke-v1"
CASE_IDS: Final[tuple[str, str, str]] = ("direct", "retrieval-pending", "retrieval-ready")
ACTION_TYPE: Final[str] = "mark_done"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "examples/deepseek_provider.py",
    "examples/llm_api_smoke.py",
    "examples/llm_api_smoke_cases.v1.json",
    "examples/llm_api_smoke.requirements.txt",
)
_LIVE_CASES: Final[tuple[str, ...]] = ("direct", "retrieval-pending")
MAX_TOKENS_PER_CALL: Final[int] = 1024
MAX_BODY_BYTES: Final[int] = 32768
SINGLE_DEADLINE: Final[float] = 30.0
GROUP_DEADLINE: Final[float] = 90.0


class SmokeError(Exception):
    """A bounded host failure; it never carries a provider or reader sentence."""


def _fail(message: str) -> None:
    raise SmokeError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- fixture loading


def load_cases(path: str | Path) -> list[dict]:
    """Load the versioned public fixture and refuse anything but the exact three cases."""

    document = json.loads(Path(path).read_bytes())
    if not isinstance(document, dict):
        _fail("the fixture must be an object")
    if document.get("schema_version") != "1":
        _fail("unsupported fixture schema version")
    if document.get("fixture_id") != FIXTURE_ID:
        _fail("unexpected fixture identifier")
    cases = document.get("cases")
    if not isinstance(cases, list):
        _fail("the fixture cases must be a list")
    seen: list[str] = []
    for item in cases:
        if not isinstance(item, dict):
            _fail("a case must be an object")
        case_id = item.get("case_id")
        if case_id not in CASE_IDS:
            _fail("unknown case identifier")
        seen.append(case_id)
    if len(seen) != len(set(seen)):
        _fail("duplicate case identifier")
    if set(seen) != set(CASE_IDS):
        _fail("the fixture must contain exactly the declared cases")
    return [dict(item) for item in cases]


# --------------------------------------------------------------------------- Core assembly


class _Domain:
    """The explicit host domain the Runtime requires, with no hidden defaults."""

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        return VerificationEvidence(schema_version=SCHEMA, observation=observation,
                                    state_complete=True)

    def prepare_contract(self, *, forecast: Forecast,
                         evidence: VerificationEvidence) -> ContractPreparation:
        from forecontract.contracts import (
            RecoveryPolicy,
            VerificationBinding,
            VerificationPolicy,
        )

        return ContractPreparation(
            schema_version=SCHEMA,
            preconditions=[],
            verification_policy=VerificationPolicy(
                schema_version=SCHEMA,
                bindings=[VerificationBinding(schema_version=SCHEMA, outcome_index=index,
                                              verifier_type="json_rule")
                          for index in range(len(forecast.expected_outcomes))]),
            recovery_policy=RecoveryPolicy(schema_version=SCHEMA, mode="bounded",
                                           max_retries=0, allow_replan=False),
            timeout_ms=15000.0,
        )

    def declare_action_safety(self, *, contract: object) -> ActionSafetyDeclaration:
        return ActionSafetyDeclaration(schema_version=SCHEMA, contract_id=contract.contract_id,
                                       adapter_name=contract.adapter_name,
                                       action_type=contract.action.action_type,
                                       retry_safety="idempotent")

    def assess_effect(self, *, contract: object, actual_outcome: ActualOutcome,
                      evidence: VerificationEvidence) -> str:
        if actual_outcome.status == "succeeded":
            return "applied"
        observed = evidence.observation.state
        if isinstance(observed, dict) and observed.get("done") is True:
            return "applied"
        return "not_applied"


def _declared(adapter: object) -> object:
    """Declare the host's json_rule outcome capability over the real Fake adapter."""

    original = adapter.capabilities

    class _Declared:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        @property
        def name(self) -> object:
            return self._inner.capabilities().adapter_name

        def capabilities(self) -> object:
            return original().model_copy(update={"supported_outcome_types": ["json_rule"]})

        def validate_action(self, action: object) -> object:
            return self._inner.validate_action(action)

        async def start(self) -> object:
            return await self._inner.start()

        async def stop(self) -> object:
            return await self._inner.stop()

        async def observe(self) -> object:
            return await self._inner.observe()

        async def execute(self, action: object) -> object:
            return await self._inner.execute(action)

    return _Declared(adapter)


def _observation(state: dict, observation_id: str) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="fake", state=state, timestamp=FIXED_TIME)


def _scenario(case: dict) -> tuple[FakeScenario, ActionSpec]:
    state = dict(case["initial_state"])
    expected = case["expected_action"]
    action = ActionSpec(schema_version=SCHEMA, action_type=ACTION_TYPE,
                        payload=dict(expected["payload"]))
    start = {**state}
    end = {**state, "done": True}
    scenario = FakeScenario(
        schema_version=SCHEMA,
        scenario_id="smoke-" + str(case["case_id"]),
        adapter_name="fake",
        allowed_actions=[action],
        stages=[
            FakeStage(schema_version=SCHEMA, actual_state=start,
                      observation=_observation(start, "obs-before")),
            FakeStage(schema_version=SCHEMA, actual_state=end,
                      observation=_observation(end, "obs-after")),
        ],
        steps=[FakeStep(schema_version=SCHEMA, step_id="step-1", action=action,
                        outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="outcome-1",
                                              status="succeeded", state_after=end))],
    )
    return scenario, action


def _criteria(goal_id: str) -> TaskCompletionCriteria:
    return TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id=goal_id,
        conditions=[ExpectedOutcome(
            schema_version=SCHEMA, outcome_type="json_rule",
            payload={"schema_version": SCHEMA, "rule_type": "value_equals",
                     "path": ["done"], "expected": True})])


def _context(run_id: str, goal_id: str, capabilities: object,
             criteria: TaskCompletionCriteria) -> PlannerContext:
    return PlannerContext(
        schema_version=SCHEMA, run_id=run_id, goal_id=goal_id,
        goal_text="complete the synthetic smoke task", task_criteria=criteria,
        capabilities=capabilities,
        action_guidance={name: {"payload": {"type": "object"}}
                         for name in capabilities.supported_actions},
        outcome_guidance={name: {"payload": {"type": "object"}}
                          for name in capabilities.supported_outcome_types})


def _granter(run_id: str):
    """One authorization per real snapshot, minted before observe returns."""

    counter = {"ordinal": 0}

    def grants_for(observation: Observation) -> list[EvidenceGrant]:
        counter["ordinal"] += 1
        return [EvidenceGrant(
            schema_version=SCHEMA, run_id=run_id,
            observation_id=observation.observation_id,
            evidence_id=f"order-status-{counter['ordinal']:04d}",
            view_id="order-status-view",
            selector=["state", "order", "status"], format="text")]

    return grants_for


class _ProviderRecorder:
    """Append one redacted event per real provider call before returning it."""

    def __init__(self, inner: object, path: Path, *, mode: str, case_id: str,
                 guard: object | None = None) -> None:
        self._inner = inner
        self._path = path
        self.mode = mode
        self.case_id = case_id
        self.count = 0
        self.last_error: str | None = None
        self._guard = guard
        if guard is not None:
            guard._inner = inner

    async def complete(self, request: object) -> object:
        if self._guard is not None:
            self._guard.before_send()
        payload = request.model_dump(mode="json")
        body = json.dumps(payload, ensure_ascii=False)
        started = datetime.now(timezone.utc)
        tic = time.monotonic()
        try:
            response = await self._inner.complete(request)
        except BaseException as error:
            self._append({"mode": self.mode, "case_id": self.case_id, "endpoint": _endpoint(),
                          "request": payload, "request_bytes": len(body.encode("utf-8")),
                          "response": None, "http_status": 0, "started_at": started.isoformat(),
                          "ended_at": datetime.now(timezone.utc).isoformat(),
                          "elapsed_seconds": round(time.monotonic() - tic, 6),
                          "error_code": type(error).__name__,
                          "sent": bool(getattr(self._inner, "last_sent", False))})
            self.last_error = type(error).__name__
            raise
        raw = getattr(self._inner, "last_payload", None)
        wire = getattr(self._inner, "last_wire", None) or payload
        sent = getattr(self._inner, "last_wire_bytes", None) or len(body.encode("utf-8"))
        self._append({"mode": self.mode, "case_id": self.case_id, "endpoint": _endpoint(),
                      "request": wire, "request_bytes": sent,
                      "response": raw,
                      "http_status": (getattr(self._inner, "last_http_status", None)
                                       or (200 if response.status == "ok" else 0)),
                      "started_at": started.isoformat(),
                      "ended_at": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": round(time.monotonic() - tic, 6),
                      "model": getattr(self._inner, "last_model", None),
                      "completion_id": getattr(self._inner, "last_completion_id", None),
                      "error_code": (None if getattr(response, "status", None) == "ok"
                                     else "provider_refused"),
                      "sent": bool(getattr(self._inner, "last_sent", False))})
        if self._guard is not None:
            self._guard.after_response(response)
        return response

    def _append(self, event: dict) -> None:
        self.count += 1
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def _endpoint() -> str:
    from deepseek_provider import ENDPOINT

    return ENDPOINT


async def run_case(case: dict, output_dir: str | Path, *, provider: object,
                   mode: str = "offline", guard: object | None = None) -> dict:
    """Run one case through the real Core loop and persist the complete evidence closure."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        # Any existing case directory - empty or not - is refused before provider/execute.
        _fail("the case output directory already exists")
    target.mkdir(parents=True, exist_ok=False)

    run_id = "run-" + uuid.uuid4().hex[:12]
    goal_id = "goal-" + uuid.uuid4().hex[:12]
    scenario, _action = _scenario(case)
    base = FakeEnvironmentAdapter(scenario)
    adapter = _declared(base)
    request = RunRequest(schema_version=SCHEMA, run_id=run_id, goal_id=goal_id,
                         task_criteria=_criteria(goal_id), max_steps=2, max_planning_calls=2,
                         max_retries=0, max_replans=0)
    config = RetrievalConfig(schema_version=SCHEMA,
                             base=PlannerConfig(schema_version=SCHEMA, call_timeout_ms=15000,
                                                cleanup_timeout_ms=1000))
    context = _context(run_id, goal_id, adapter.capabilities(), request.task_criteria)

    audit = FileAuditSink(target / "audit")
    events_path = target / "provider-events.jsonl"
    events_path.touch(exist_ok=False)
    recording_provider = _ProviderRecorder(provider, events_path, mode=mode,
                                           case_id=case["case_id"], guard=guard)
    retrieval = case["mode"] == "retrieval"
    if retrieval:
        reader = RegisteredEvidenceReader(target, run_id, config)
        recorder = RegisteringEvidenceRecorder(target, run_id=run_id, profile="p4-t02-offline",
                                               reader=reader, grants_for=_granter(run_id))
        planner = EvidenceLLMPlanner(provider=recording_provider, context=context, config=config,
                                     reader=reader, audit=audit)
    else:
        reader = None
        recorder = EvidenceRecorder(target, run_id=run_id, profile="p4-t02-offline")
        planner = LLMPlanner(provider=recording_provider, context=context, config=config.base)

    trace_path = target / "trace.jsonl"
    trace = JsonlTraceStore(trace_path)
    trace_path.touch(exist_ok=False)
    engine = RuntimeEngine(
        adapter=RecordingAdapter(adapter, recorder),
        planner=RecordingPlanner(planner, recorder),
        domain_support=_Domain(),
        budget_source=MonotonicBudgetSource(total_budget_ms=30000.0),
        trace_writer=trace,
        working_memory=WorkingMemory(),
        builder=RecordingBuilder(ContractBuilder(), recorder),
        step_verifier=RecordingStepVerifier(StepVerifier(), recorder),
        task_oracle=RecordingTaskOracle(TaskCompletionOracle(), recorder),
        recovery_engine=RecordingRecoveryEngine(RecoveryEngine(step_verifier_name="json_rule"),
                                                recorder),
        timing_policy=RuntimeTimingPolicy(schema_version=SCHEMA, run_timeout_ms=20000.0,
                                          cleanup_timeout_ms=2000.0, finalize_timeout_ms=2000.0),
    )
    result = await engine.run(request=request)
    close = await planner.aclose()

    events = [json.loads(line) for line in events_path.read_bytes().splitlines() if line.strip()]
    executed = _executed_action(target, case)
    summary = {
        "case_id": case["case_id"],
        "mode": mode,
        "run_id": run_id,
        "goal_id": goal_id,
        "terminal_status": result.terminal_status.value,
        "reason_code": result.reason_code.value,
        "executed_action": executed,
        "read_values": _read_values(events),
        "provider_calls": len(events),
        "actual_http_requests": len(events),
        "close": close.model_dump(mode="json"),
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": _fixture_sha256(),
        "source_fingerprint": _source_fingerprint(),
        "evidence_root": str(target),
        "trace_path": str(trace_path),
        "provider_events_path": str(events_path),
    }
    # ``files`` is computed before summary.json exists, so it never contains itself.
    summary["files"] = _files_map(target)
    (target / "summary.json").write_bytes(_canonical(summary) + b"\n")
    return summary


def _executed_action(target: Path, case: dict) -> dict:
    """Read the actual execution snapshot the recorder wrote."""

    action = None
    for path in sorted((target / "snapshots").glob("*.json")):
        document = json.loads(path.read_bytes())
        if document.get("kind") == "actual_outcome":
            action = document.get("payload", {}).get("action")
    if action is None:
        return ActionSpec.model_validate(case["expected_action"]).model_dump(mode="json")
    return ActionSpec.model_validate(action).model_dump(mode="json")


def _read_values(events: list[dict]) -> list[object]:
    """The read values the last sent wire really carried."""

    if not events:
        return []
    messages = (events[-1].get("request") or {}).get("messages") or []
    if len(messages) < 2:
        return []
    payload = json.loads(messages[1]["content"])
    return [item.get("value") for item in payload.get("exchanges", [])]


def _fixture_sha256() -> str:
    return _digest(_repo_root() / "examples/llm_api_smoke_cases.v1.json")


def _source_fingerprint() -> dict[str, str]:
    return {rel: _digest(_repo_root() / rel) for rel in SOURCE_FILES}


def _files_map(target: Path) -> dict[str, str]:
    return {path.relative_to(target).as_posix(): _digest(path)
            for path in sorted(target.rglob("*"))
            if path.is_file() and path.name != "summary.json"}


def validate_case_evidence(path: str | Path) -> list[str]:
    """Read the persisted evidence back; [] means consistent, never a self-reported pass."""

    target = Path(path)
    issues: list[str] = []
    if not (target / "summary.json").is_file():
        return ["missing:summary"]
    try:
        summary = json.loads((target / "summary.json").read_bytes())
    except ValueError:
        return ["unreadable:summary"]
    if not (target / "trace.jsonl").is_file():
        return ["missing:trace"]
    if not (target / "provider-events.jsonl").is_file():
        return ["missing:provider_events"]
    records = [json.loads(line) for line in (target / "trace.jsonl").read_bytes().splitlines()
               if line.strip()]
    if not records:
        issues.append("empty:trace")
    else:
        last = records[-1]
        if last.get("kind") != "run_finished":
            issues.append("trace_not_finished")
        elif last.get("terminal_status") != "success":
            issues.append("trace_terminal_not_success")
        if last.get("run_id") != summary.get("run_id"):
            issues.append("trace_run_mismatch")
    events = [json.loads(line) for line in
              (target / "provider-events.jsonl").read_bytes().splitlines() if line.strip()]
    if {event.get("mode") for event in events} - {summary.get("mode")}:
        issues.append("mode_mismatch")
    if events and any(event.get("case_id") != summary.get("case_id") for event in events):
        issues.append("case_mismatch")
    if summary.get("actual_http_requests") != len(events):
        issues.append("http_count_mismatch")
    actual = _files_map(target)
    if summary.get("files") != actual:
        issues.append("file_closure_mismatch")
    snapshot_dir = target / "snapshots"
    outcomes = []
    if not snapshot_dir.is_dir():
        issues.append("missing:snapshots")
    else:
        for item in sorted(snapshot_dir.glob("*.json")):
            document = json.loads(item.read_bytes())
            if document.get("run_id") != summary.get("run_id"):
                issues.append("snapshot_run_mismatch")
            if document.get("kind") == "actual_outcome":
                outcomes.append(document)
    if len(outcomes) != 1:
        issues.append("actual_outcome_count")
    else:
        action = outcomes[0].get("payload", {}).get("action")
        if ActionSpec.model_validate(action).model_dump(mode="json") != summary.get(
                "executed_action"):
            issues.append("executed_action_mismatch")
    audit_dir = target / "audit"
    if summary.get("case_id") != "direct":
        if not audit_dir.is_dir():
            issues.append("missing:audit")
        else:
            names = {item.name for item in audit_dir.glob("*.json")}
            if len(names) != 9:
                issues.append("audit_count_mismatch")
            for item in sorted(audit_dir.glob("*.json")):
                document = json.loads(item.read_bytes())
                if document.get("run_id") != summary.get("run_id"):
                    issues.append("audit_run_mismatch")
                name = f"{document.get('sequence', 0):04d}-{document.get('kind')}.json"
                if name != item.name:
                    issues.append("audit_name_mismatch")
                payload = _canonical(document.get("payload"))
                if hashlib.sha256(payload).hexdigest() != document.get("payload_sha256"):
                    issues.append("audit_payload_mismatch")
    return issues


# --------------------------------------------------------------------------- CLI


def _offline_client() -> tuple[object, object]:
    """One local client and the real transport over a MockTransport; no socket, no credential."""

    import httpx

    def handle(incoming: httpx.Request) -> httpx.Response:
        wire = json.loads(incoming.content)
        envelope = {"id": "offline-local", "model": "deepseek-flash",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant",
                                               "content": json.dumps(_offline_answer(wire))}}],
                    "usage": {"prompt_tokens": 21, "completion_tokens": 5}}
        return httpx.Response(200, json=envelope)

    client = make_client(api_key="offline-local-transport",
                         transport=httpx.MockTransport(handle))
    provider = DeepSeekProvider(client=client, model="deepseek-flash",
                                timeout_seconds=SINGLE_DEADLINE,
                                max_output_tokens_cap=MAX_TOKENS_PER_CALL)
    return client, provider


def _offline_main(cases: list[dict], target: Path) -> int:
    started = time.monotonic()
    client, provider = _offline_client()
    summaries = []
    try:
        for case in cases:
            summaries.append(asyncio.run(run_case(case, target / case["case_id"],
                                                  provider=provider, mode="offline")))
    except SmokeError as error:
        print(f"offline run failed: {error}", file=sys.stderr)
        return 2
    finally:
        asyncio.run(client.aclose())
    root = {
        "mode": "offline",
        "status": "completed",
        "requested_model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "max_requests": 0,
        "budget_cny": 0,
        "actual_http_requests": 0,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": _fixture_sha256(),
        "source_fingerprint": _source_fingerprint(),
        "cases": {summary["case_id"]: summary for summary in summaries},
    }
    (target / "offline-summary.json").write_bytes(_canonical(root) + b"\n")
    print(json.dumps({"mode": "offline", "status": "completed",
                      "cases": sorted(root["cases"])}, ensure_ascii=False))
    return 0


class _LiveGuard:
    """Per-call bounds for one authorized live group: request cap, deadline, known usage."""

    def __init__(self, *, max_requests: int, deadline_at: float, budget_cny: float) -> None:
        self._max_requests = max_requests
        self._deadline_at = deadline_at
        self._budget_cny = budget_cny
        self.sent = 0
        self.cost_cny = 0.0
        self.reserved_cny = 0.0

    def check(self) -> None:
        if self.sent >= self._max_requests:
            raise SmokeError("the per-run request cap was reached")
        if time.monotonic() >= self._deadline_at:
            raise SmokeError("the group deadline elapsed before sending")

    def before_send(self) -> None:
        self.check()
        self.sent += 1

    def after_response(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if getattr(response, "status", None) != "ok" or usage is None:
            raise SmokeError("the provider usage is unknown")
        sent_bytes = getattr(self._inner, "last_wire_bytes", None) or 0
        max_tokens = getattr(self._inner, "last_max_tokens", None) or 0
        call_reserved = ((sent_bytes + 4096) * 2 + max_tokens * 8) / 1_000_000
        call_cost = (usage.input_tokens * 2 + usage.output_tokens * 8) / 1_000_000
        self.reserved_cny += call_reserved
        self.cost_cny += call_cost
        if call_cost > call_reserved:
            raise SmokeError("the observed usage exceeded this call's reservation")
        if time.monotonic() >= self._deadline_at:
            raise SmokeError("the group deadline elapsed before settlement")

    def remaining(self) -> float:
        return self._deadline_at - time.monotonic()


CLOSE_TIMEOUT_SECONDS: Final[float] = 5.0


async def _bounded_close(client: object) -> object | None:
    """Close the client under one explicit finite bound; return a bounded failure or None.

    `asyncio.wait_for` cancels the close attempt at the bound, so a transport that never
    finishes closing cannot hold the group open. Cancellation of the caller is not a close
    failure: it is re-raised unchanged so the caller can record it as a cancellation.
    """

    try:
        await asyncio.wait_for(client.aclose(), CLOSE_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except (asyncio.TimeoutError, TimeoutError):
        return SmokeError("the client close exceeded its finite bound")
    except Exception as error:  # noqa: BLE001 - a close failure is bounded, never fatal
        return error
    return None


def _sent_total(target: Path) -> int:
    """Count only events that really reached the transport, never a pre-send refusal."""

    total = 0
    for case_dir in target.iterdir():
        path = case_dir / "provider-events.jsonl"
        if case_dir.is_dir() and path.is_file():
            for line in path.read_bytes().splitlines():
                if not line.strip():
                    continue
                if json.loads(line).get("sent"):
                    total += 1
    return total


def _reservation_cny() -> float:
    per_call = (MAX_BODY_BYTES + 4096) * 2 / 1_000_000 + MAX_TOKENS_PER_CALL * 8 / 1_000_000
    return round(per_call * 3, 6)


def _live_main(arguments: argparse.Namespace, cases: list[dict], target: Path) -> int:
    started = time.monotonic()
    base = {"mode": "live", "requested_model": arguments.model,
            "thinking": {"type": "disabled"}, "max_requests": arguments.max_requests,
            "budget_cny": arguments.budget_cny, "fixture_id": FIXTURE_ID,
            "fixture_sha256": _fixture_sha256(), "source_fingerprint": _source_fingerprint()}

    def write(**changes: object) -> None:
        document = {**base, "elapsed_seconds": round(time.monotonic() - started, 6), "cases": {},
                    **changes}
        (target / "live-summary.json").write_bytes(_canonical(document) + b"\n")

    if arguments.max_requests <= 0 or arguments.budget_cny <= 0:
        write(status="refused", reason="non_positive_limit", actual_http_requests=0)
        print("live limits must be positive", file=sys.stderr)
        return 5
    if arguments.budget_cny < _reservation_cny():
        # The whole group is reserved before the first send; a smaller budget never sends.
        write(status="refused", reason="budget_below_group_reservation",
              actual_http_requests=0, reservation_cny=_reservation_cny())
        print("live budget is below the reserved group cost", file=sys.stderr)
        return 5
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        write(status="not_run", reason="missing_local_credentials", actual_http_requests=0)
        print("live not_run: missing_local_credentials", file=sys.stderr)
        return 4

    guard = _LiveGuard(max_requests=arguments.max_requests,
                       deadline_at=started + GROUP_DEADLINE,
                       budget_cny=arguments.budget_cny)
    summaries: dict[str, object] = {}
    close_failure: list[object] = [None]
    close_cancelled: list[bool] = [False]

    async def group() -> int:
        client = make_client(api_key=api_key)
        provider = DeepSeekProvider(client=client, model=arguments.model,
                                    timeout_seconds=SINGLE_DEADLINE,
                                    max_request_bytes=MAX_BODY_BYTES,
                                    max_output_tokens_cap=MAX_TOKENS_PER_CALL,
                                    remaining_seconds=guard.remaining)
        sent = 0
        try:
            for case in cases:
                if case["case_id"] not in _LIVE_CASES:
                    continue
                try:
                    guard.check()
                except SmokeError as error:
                    write(status="failed", reason=str(error), actual_http_requests=_sent_total(target),
                          cases=summaries)
                    print(f"live stopped: {error}", file=sys.stderr)
                    return 6
                summary = await run_case(case, target / case["case_id"], provider=provider,
                                         mode="live", guard=guard)
                summaries[case["case_id"]] = summary
                sent += summary["actual_http_requests"]
                if summary["terminal_status"] != "completed":
                    write(status="failed", reason="case_failed", actual_http_requests=_sent_total(target),
                          cases=summaries)
                    print("live stopped after a failed case", file=sys.stderr)
                    return 6
        except asyncio.CancelledError:
            write(status="failed", reason="cancelled_during_live_group",
                  actual_http_requests=_sent_total(target), cases=summaries)
            raise
        except Exception as error:  # noqa: BLE001 - any bounded failure becomes a failed root status
            write(status="failed", reason=type(error).__name__ + ": " + str(error)[:120],
                  actual_http_requests=_sent_total(target), cases=summaries)
            print(f"live stopped: {error}", file=sys.stderr)
            return 6
        finally:
            # A cancellation raised by the close itself still has to settle real failure evidence.
            try:
                close_failure[0] = await _bounded_close(client)
            except asyncio.CancelledError:
                close_cancelled[0] = True
                close_failure[0] = SmokeError("cancelled while closing the client")
        if close_failure[0] is not None:
            write(status="failed",
                  reason=("cancelled_during_client_close" if close_cancelled[0]
                          else "client_close_failed: " + str(close_failure[0])[:100]),
                  actual_http_requests=_sent_total(target), cases=summaries)
            print("live failed while closing the client", file=sys.stderr)
            return 6
        if time.monotonic() >= guard._deadline_at:
            write(status="failed", reason="group_deadline_elapsed_at_close",
                  actual_http_requests=_sent_total(target), cases=summaries)
            print("live expired during close", file=sys.stderr)
            return 6
        write(status="completed", actual_http_requests=_sent_total(target), cases=summaries)
        print(json.dumps({"mode": "live", "status": "completed", "requests": sent}))
        return 0

    try:
        return asyncio.run(group())
    except asyncio.CancelledError:
        # Last resort: the close-path cancellation must still leave a real failed root summary.
        write(status="failed", reason="cancelled_during_cleanup",
              actual_http_requests=_sent_total(target), cases=summaries)
        print("live failed during cleanup", file=sys.stderr)
        return 6


def _usage_missing(case_dir: Path) -> bool:
    path = case_dir / "provider-events.jsonl"
    if not path.is_file():
        return True
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        response = event.get("response") or {}
        if event.get("http_status") != 200 or not response.get("usage"):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument("--budget-cny", type=float, default=1.0)
    arguments = parser.parse_args(argv)

    try:
        cases = load_cases(arguments.cases)
    except (SmokeError, ValueError, OSError) as error:
        print(f"fixture refused: {error}", file=sys.stderr)
        return 2

    target = Path(arguments.output_dir).expanduser().resolve()
    if target.exists():
        print("the output directory must not exist", file=sys.stderr)
        return 3
    target.mkdir(parents=True, exist_ok=False)
    if arguments.mode == "live":
        if not arguments.model:
            (target / "live-summary.json").write_bytes(
                _canonical({"mode": "live", "status": "refused",
                            "reason": "missing_model", "actual_http_requests": 0}) + b"\n")
            print("live mode requires an explicit --model", file=sys.stderr)
            return 2
        return _live_main(arguments, cases, target)
    return _offline_main(cases, target)


def _offline_answer(wire: dict) -> dict:
    """Build the local answer from the request the wire really carries."""

    request = json.loads(wire["messages"][1]["content"])
    retrieval = "base_request" in request
    if retrieval and not request["exchanges"]:
        entry = request["scope"]["entries"][0]
        value = {"kind": "read_evidence",
                 "read": {"evidence_id": entry["evidence_id"],
                           "view_id": entry["views"][0]["view_id"], "text_range": None}}
    else:
        observed = (request["exchanges"][-1]["value"] if retrieval
                    else request["observation"]["state"]["order"]["status"])
        forecast = {"schema_version": SCHEMA,
                    "proposed_action": {"schema_version": SCHEMA, "action_type": ACTION_TYPE,
                                         "payload": {"observed": observed}},
                    "expected_outcomes": [{"schema_version": SCHEMA,
                                            "outcome_type": "json_rule",
                                            "payload": {"schema_version": SCHEMA,
                                                         "rule_type": "value_equals",
                                                         "path": ["done"], "expected": True}}],
                    "task_completion_hint": True}
        value = {"kind": "forecast", "forecast": forecast} if retrieval else forecast
    if retrieval:
        value.update({key: request[key] for key in
                      ("schema_version", "run_id", "plan_id", "turn", "request_id")})
    return value


class _OfflineProvider:
    """Deprecated local stub kept out of the default path."""

    async def complete(self, request: object) -> object:
        from forecontract_llm.models import ModelResponse

        payload = request.model_dump(mode="json")
        retrieval = "base_request" in payload
        if retrieval and not payload.get("exchanges"):
            entries = (payload.get("scope") or {}).get("entries") or []
            if entries:
                entry = entries[0]
                body = {"kind": "read_evidence",
                        "read": {"evidence_id": entry.get("evidence_id"),
                                 "view_id": (entry.get("views") or [{}])[0].get("view_id"),
                                 "text_range": None}}
                body.update({key: payload[key] for key in
                             ("schema_version", "run_id", "plan_id", "turn", "request_id")})
                return ModelResponse(schema_version=SCHEMA, request_id=payload["request_id"],
                                     status="ok", content=json.dumps(body))
        if retrieval and payload.get("exchanges"):
            observed = payload["exchanges"][-1].get("value")
        else:
            state = (payload.get("observation") or {}).get("state") or {}
            observed = (state.get("order") or {}).get("status")
        forecast = {"schema_version": SCHEMA,
                    "proposed_action": {"schema_version": SCHEMA, "action_type": ACTION_TYPE,
                                        "payload": {"observed": observed}},
                    "expected_outcomes": [{"schema_version": SCHEMA, "outcome_type": "json_rule",
                                           "payload": {"schema_version": SCHEMA,
                                                       "rule_type": "value_equals",
                                                       "path": ["done"], "expected": True}}]}
        value = {"kind": "forecast", "forecast": forecast} if retrieval else forecast
        if retrieval:
            value.update({key: payload[key] for key in
                          ("schema_version", "run_id", "plan_id", "turn", "request_id")})
        return ModelResponse(schema_version=SCHEMA, request_id=payload["request_id"],
                             status="ok", content=json.dumps(value))


if __name__ == "__main__":
    raise SystemExit(main())
