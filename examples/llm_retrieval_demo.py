#!/usr/bin/env python
"""Offline retrieval demo: real host snapshots, the registration bridge, the planner and Core.

This example composes the already-verified components the way a host does: the real
``EvidenceRecorder`` through :class:`RegisteringEvidenceRecorder`, the real Core
``RuntimeEngine``, the real ``ContractBuilder``, step verifier and task oracle, the real
``FakeEnvironmentAdapter``, and :class:`EvidenceLLMPlanner` with an offline, field-driven
provider. Nothing here opens a socket, reads an API key or touches a production system.

The provider is driven by the data it receives, not by a fixed call count: its first answer
asks for one authorized evidence read, and its second answer builds the action payload from the
value the read really returned. Changing the synthetic snapshot field therefore changes the
action that is really executed. The host grants exactly one text view over the synthetic state
field ``order.status``.

Every retrieval audit event is persisted by :class:`FileAuditSink` as one atomic file under a
host-supplied directory outside the repository; an existing event file is never overwritten.
The completed-state evidence index remains the original recorder's business and is deliberately
not produced here: this demo writes real host snapshots and reads them through the registration
bridge, and it never invents a simpler "verified" index.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from forecontract.contracts import (
    ContractBuilder,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
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
from forecontract.verification import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract_fake import (
    FakeEnvironmentAdapter,
    FakeScenario,
    FakeStage,
    FakeStep,
)
from forecontract_llm.models import PlannerConfig, PlannerContext
from forecontract_llm.retrieval import (
    EvidenceGrant,
    EvidenceLLMPlanner,
    RegisteredEvidenceReader,
    RetrievalConfig,
)
from web_retrieval_evidence import RegisteringEvidenceRecorder
from web_runtime_evidence import (
    RecordingAdapter,
    RecordingBuilder,
    RecordingPlanner,
    RecordingRecoveryEngine,
    RecordingStepVerifier,
    RecordingTaskOracle,
)

__all__ = ["DEFAULT_STATE", "FileAuditSink", "main", "run_demo"]

SCHEMA: Final[str] = "0.1"
AUDIT_DIR: Final[str] = "audit"
EVIDENCE_DIR: Final[str] = "evidence"
DEFAULT_STATE: Final[dict] = {
    "done": False,
    "order": {"status": "pending", "item": "widget"},
}
VIEW_ID: Final[str] = "order_status"
EVIDENCE_ID: Final[str] = "order-status"
FIXED_TIME: Final[datetime] = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


class DemoError(Exception):
    """A bounded demo failure; it never carries a raw provider or reader sentence."""


def _fail(message: str) -> None:
    raise DemoError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class FileAuditSink:
    """One atomic, exclusive file per audit event under a host-supplied directory.

    A successful ``append`` returns only after the event bytes are flushed and fsynced. An
    existing file is never overwritten, so a retry of the same sequence cannot silently replace
    an earlier fact.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise DemoError("the audit directory could not be created") from None

    @property
    def directory(self) -> Path:
        return self._directory

    async def append(self, event: object) -> None:
        name = f"{event.sequence:04d}-{event.kind}.json"
        document = event.model_dump(mode="json")
        try:
            raw = _canonical(document)
        except (TypeError, ValueError, UnicodeEncodeError):
            raise DemoError("the audit event is not canonical JSON") from None
        if hashlib.sha256(_canonical(document["payload"])).hexdigest() != document["payload_sha256"]:
            raise DemoError("the audit event does not bind its declared payload digest")
        target = self._directory / name
        # F05: write and flush a private temporary file first, then publish the final name
        # atomically and without overwriting. A short/interrupted write therefore leaves no
        # partially committed ``*.json`` under the committed name.
        temporary = self._directory / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                raise DemoError("an audit event file already exists") from None
            except OSError:
                # Hard links are available on the supported NTFS host. On a filesystem
                # without them, a no-overwrite rename is still atomic on Windows; never
                # fall back to an overwriting replace.
                if os.name != "nt" or target.exists():
                    raise DemoError("an audit event could not be persisted") from None
                try:
                    os.rename(temporary, target)
                except FileExistsError:
                    raise DemoError("an audit event file already exists") from None
                except OSError:
                    raise DemoError("an audit event could not be persisted") from None
        except DemoError:
            raise
        except FileExistsError:
            raise DemoError("an audit event file already exists") from None
        except OSError:
            raise DemoError("an audit event could not be persisted") from None
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


class _DeclaredAdapter:
    """Forward every adapter call to the real Fake and declare the host's ``json_rule``."""

    def __init__(self, inner: FakeEnvironmentAdapter) -> None:
        self._inner = inner
        self.actions: list[ActionSpec] = []

    @property
    def name(self) -> str:
        return self._inner.capabilities().adapter_name

    def capabilities(self):
        declared = self._inner.capabilities()
        return declared.model_copy(update={"supported_outcome_types": ["json_rule"]})

    def validate_action(self, action: ActionSpec) -> None:
        self._inner.validate_action(action)

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()

    async def observe(self) -> Observation:
        return await self._inner.observe()

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        self.actions.append(action.model_copy(deep=True))
        return await self._inner.execute(action)


class _Domain:
    """The explicit host domain binding Core requires, with no hidden default."""

    def to_evidence(self, observation: Observation) -> VerificationEvidence:
        return VerificationEvidence(schema_version=SCHEMA, observation=observation,
                                    state_complete=True)

    def prepare_contract(self, *, forecast: Forecast,
                         evidence: VerificationEvidence) -> ContractPreparation:
        return ContractPreparation(
            schema_version=SCHEMA,
            preconditions=[],
            verification_policy=VerificationPolicy(
                schema_version=SCHEMA,
                bindings=[
                    VerificationBinding(schema_version=SCHEMA, outcome_index=index,
                                        verifier_type="json_rule")
                    for index in range(len(forecast.expected_outcomes))
                ],
            ),
            recovery_policy=RecoveryPolicy(schema_version=SCHEMA, mode="bounded",
                                           max_retries=0, allow_replan=False),
            timeout_ms=15000,
        )

    def declare_action_safety(self, *, contract: object) -> ActionSafetyDeclaration:
        return ActionSafetyDeclaration(
            schema_version=SCHEMA, contract_id=contract.contract_id,
            adapter_name=contract.adapter_name, action_type=contract.action.action_type,
            retry_safety="idempotent")

    def assess_effect(self, *, contract: object, actual_outcome: ActualOutcome,
                      evidence: VerificationEvidence) -> str:
        if actual_outcome.status == "succeeded":
            return "applied"
        return "unknown"


class FieldDrivenProvider:
    """An offline provider whose answers are built from the fields it really received."""

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.last_forecast: dict | None = None

    async def complete(self, request: object):
        from forecontract_llm.models import ModelResponse

        self.requests.append(request)
        if not request.exchanges:
            # First answer: ask for exactly one view that the frozen scope really grants.
            entry = request.scope.entries[0]
            view = entry.views[0]
            read = {
                "schema_version": SCHEMA,
                "run_id": request.run_id,
                "plan_id": request.plan_id,
                "turn": request.turn,
                "request_id": request.request_id,
                "kind": "read_evidence",
                "read": {"evidence_id": entry.evidence_id, "view_id": view.view_id,
                         "text_range": None},
            }
            content = json.dumps(read)
        else:
            exchange = request.exchanges[-1]
            if exchange.status != "found":
                raise DemoError("the authorized view was unexpectedly missing")
            observed = exchange.value
            body = {
                "schema_version": SCHEMA,
                "proposed_action": {
                    "schema_version": SCHEMA,
                    "action_type": "mark_done",
                    "payload": {"observed": observed},
                },
                "expected_outcomes": [
                    {"schema_version": SCHEMA, "outcome_type": "json_rule",
                     "payload": {"schema_version": SCHEMA, "rule_type": "value_equals",
                                 "path": ["done"], "expected": True}}
                ],
                "task_completion_hint": True,
            }
            self.last_forecast = body["proposed_action"]
            content = json.dumps({
                "schema_version": SCHEMA,
                "run_id": request.run_id,
                "plan_id": request.plan_id,
                "turn": request.turn,
                "request_id": request.request_id,
                "kind": "forecast",
                "forecast": body,
            })
        return ModelResponse(schema_version=SCHEMA, request_id=request.request_id,
                             status="ok", content=content)


def _observation(observation_id: str, state: dict) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="fake", state=state, timestamp=FIXED_TIME)


def _scenario(state: dict) -> tuple[FakeScenario, ActionSpec]:
    status = state["order"]["status"]
    action = ActionSpec(schema_version=SCHEMA, action_type="mark_done",
                        payload={"observed": status})
    start = {key: value for key, value in state.items()}
    end = {**state, "done": True}
    scenario = FakeScenario(
        schema_version=SCHEMA,
        scenario_id="retrieval-demo",
        adapter_name="fake",
        allowed_actions=[action],
        stages=[
            FakeStage(schema_version=SCHEMA, actual_state=start,
                      observation=_observation("obs-before", start)),
            FakeStage(schema_version=SCHEMA, actual_state=end,
                      observation=_observation("obs-after", end)),
        ],
        steps=[
            FakeStep(
                schema_version=SCHEMA,
                step_id="step-1",
                action=action,
                outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="outcome-1",
                                      status="succeeded", state_after=end),
            )
        ],
    )
    return scenario, action


def _criteria() -> TaskCompletionCriteria:
    return TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id="goal-retrieval",
        conditions=[ExpectedOutcome(
            schema_version=SCHEMA, outcome_type="json_rule",
            payload={"schema_version": SCHEMA, "rule_type": "value_equals",
                     "path": ["done"], "expected": True})])


def _context(capabilities, criteria: TaskCompletionCriteria) -> PlannerContext:
    return PlannerContext(
        schema_version=SCHEMA, run_id="run-retrieval-demo", goal_id="goal-retrieval",
        goal_text="finish the offline retrieval task", task_criteria=criteria,
        capabilities=capabilities,
        action_guidance={name: {"payload": {"type": "object"}}
                         for name in capabilities.supported_actions},
        outcome_guidance={name: {"payload": {"type": "object"}}
                          for name in capabilities.supported_outcome_types})


def _request() -> RunRequest:
    return RunRequest(schema_version=SCHEMA, run_id="run-retrieval-demo",
                      goal_id="goal-retrieval", task_criteria=_criteria(),
                      max_steps=2, max_planning_calls=2, max_retries=0, max_replans=0)


def _granter():
    """One authorization per real snapshot; the host callback runs before observe returns."""
    counter = {"ordinal": 0}

    def grants_for(observation: Observation) -> list[EvidenceGrant]:
        counter["ordinal"] += 1
        return [
            EvidenceGrant(
                schema_version=SCHEMA,
                run_id="run-retrieval-demo",
                evidence_id=f"{EVIDENCE_ID}-{counter['ordinal']:04d}",
                observation_id=observation.observation_id,
                view_id=VIEW_ID,
                selector=["state", "order", "status"],
                format="text",
            )
        ]

    return grants_for


def _summarise_audit(directory: Path) -> dict:
    """Re-read every persisted event and prove the pairings from the files themselves."""
    files = sorted(directory.glob("*.json"))
    events: list[dict] = []
    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _fail("an audit event file is unreadable")
        payload = document.get("payload")
        try:
            raw = _canonical(payload)
        except (TypeError, ValueError, UnicodeEncodeError):
            _fail("an audit payload is not canonical JSON")
        if (hashlib.sha256(raw).hexdigest() != document.get("payload_sha256")
                or len(raw) != document.get("payload_bytes")):
            _fail("an audit event does not match its payload digest")
        events.append(document)
    sequences = [event["sequence"] for event in events]
    if sequences != list(range(1, len(events) + 1)):
        _fail("the audit sequence is not strictly increasing from one")
    kinds = [event["kind"] for event in events]
    prepared = [event for event in events if event["kind"] == "request_prepared"]
    returned = [event for event in events if event["kind"] == "provider_returned"]
    requested = [event for event in events if event["kind"] == "read_requested"]
    delivered = [event for event in events if event["kind"] == "read_returned"]
    if len(prepared) != len(returned):
        _fail("a provider intent has no persisted result")
    if len(requested) != len(delivered):
        _fail("a read intent has no persisted result")
    if kinds.count("forecast_ready") != 1:
        _fail("the audit does not record exactly one ready forecast")
    if not kinds or kinds[-1] != "plan_closed":
        _fail("the audit does not close the plan")
    for event in returned:
        if event["payload"].get("content") is None:
            _fail("a provider result carries no content")
    for event in delivered:
        if event["payload"].get("source", {}).get("registration_time_snapshot") is not True:
            _fail("a read result does not name its registration-time snapshot")
    return {
        "directory": str(directory),
        "event_count": len(events),
        "kinds": kinds,
        "sha256": hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest(),
        "paired": {"provider_intents": len(prepared), "read_intents": len(requested)},
    }


async def run_demo(output_dir: Path, *, state: dict | None = None) -> dict:
    """Run one complete offline retrieval loop and return its real facts."""
    repository = Path(__file__).resolve().parents[1]
    target = Path(output_dir).expanduser().resolve()
    if target == repository or repository in target.parents:
        _fail("the demo output directory must be outside the repository")
    if target.exists() and any(target.iterdir()):
        _fail("the demo output directory must be new or empty")
    target.mkdir(parents=True, exist_ok=True)
    synthetic = {key: value for key, value in (state or DEFAULT_STATE).items()}
    evidence_root = target
    audit = FileAuditSink(target / AUDIT_DIR)
    scenario, _expected_action = _scenario(synthetic)
    adapter = _DeclaredAdapter(FakeEnvironmentAdapter(scenario))
    request = _request()
    config = RetrievalConfig(
        schema_version=SCHEMA,
        base=PlannerConfig(schema_version=SCHEMA, call_timeout_ms=15000,
                           cleanup_timeout_ms=1000),
    )
    reader = RegisteredEvidenceReader(evidence_root, request.run_id, config)
    recorder = RegisteringEvidenceRecorder(
        evidence_root, run_id=request.run_id, profile="offline-retrieval",
        reader=reader, grants_for=_granter())
    recording_adapter = RecordingAdapter(adapter, recorder)
    provider = FieldDrivenProvider()
    context = _context(adapter.capabilities(), request.task_criteria)
    planner = EvidenceLLMPlanner(provider=provider, context=context, config=config,
                                 reader=reader, audit=audit)
    trace_path = target / "trace.jsonl"
    trace = JsonlTraceStore(trace_path)
    trace_path.touch(exist_ok=False)
    memory = WorkingMemory()
    engine = RuntimeEngine(
        adapter=recording_adapter,
        planner=RecordingPlanner(planner, recorder),
        domain_support=_Domain(),
        budget_source=MonotonicBudgetSource(total_budget_ms=30000.0),
        trace_writer=trace,
        working_memory=memory,
        builder=RecordingBuilder(ContractBuilder(), recorder),
        step_verifier=RecordingStepVerifier(
            _default_step_verifier(), recorder),
        task_oracle=RecordingTaskOracle(_default_task_oracle(), recorder),
        recovery_engine=RecordingRecoveryEngine(
            RecoveryEngine(step_verifier_name="json_rule"), recorder),
        timing_policy=RuntimeTimingPolicy(
            schema_version=SCHEMA, run_timeout_ms=20000.0,
            cleanup_timeout_ms=2000.0, finalize_timeout_ms=2000.0),
    )
    try:
        result = await engine.run(request=request)
    finally:
        close = await planner.aclose()
    audit_summary = _summarise_audit(target / AUDIT_DIR)
    return {
        "run_id": request.run_id,
        "terminal_status": result.terminal_status.value,
        "reason_code": result.reason_code.value,
        "planned_action": provider.last_forecast,
        "executed_action": adapter.actions[0].model_dump(mode="json"),
        "read_values": [
            exchange.value for exchange in provider.requests[-1].exchanges
        ],
        "provider_calls": len(provider.requests),
        "calls_used": planner.calls_used,
        "close": close.model_dump(mode="json"),
        "audit": audit_summary,
        "evidence": {"root": str(evidence_root),
                     "snapshot_count": reader.snapshot_count},
        "trace": str(trace_path),
    }


def _default_step_verifier():
    from forecontract.verification.step import StepVerifier

    return StepVerifier()


def _default_task_oracle():
    from forecontract.verification.oracle import TaskCompletionOracle

    return TaskCompletionOracle()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="new or empty directory outside the repository")
    arguments = parser.parse_args(argv)
    try:
        summary = asyncio.run(run_demo(Path(arguments.output_dir)))
    except (DemoError, ValueError) as error:
        print(f"retrieval demo failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
