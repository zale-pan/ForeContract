"""Host-only, pure-forwarding evidence recording for the Web runtime example.

The Core JSONL trace keeps its existing strict schema: it records identifiers, statuses,
verifications and the terminal fact, but it deliberately carries no complete ``Observation``,
``Forecast``, committed ``Contract`` or ``ActualOutcome`` payload. This module does not change
that schema. It supplies host-side forwarding proxies that capture the real inputs and returns of
the already-verified components, write them to an external snapshot directory and publish an
``evidence-index.json`` that binds every snapshot to its raw bytes, its SHA-256, its run/step/
contract/attempt/event/observation associations and the real trace and Working Memory.

The proxies have no decision authority. Each one calls the real component exactly once with the
very arguments it received and returns that component's object unchanged; an exception from the
real component is propagated untouched. Snapshots are written synchronously after the real call
returns and before the proxy returns, so the write cannot escape the Runtime's original allowance.
A serialization or write failure is raised as :class:`EvidenceError`; it is never swallowed.

The same recorder carries the host-boundary diagnostics the acceptance requires: every real
adapter boundary (start/observe/execute/stop) records its measured duration and, when it fails, the
fixed facility code *before* the Core Runtime wraps the exception; the host also records its
LOGIN_RECEIVED wait, its request/submit counts at the declared phases and the cleanup facts. Only
the closed fixed code vocabularies are accepted - an arbitrary exception sentence is never stored -
and the Core JSONL trace schema is not touched. A replay of the index revalidates this complete
diagnostics structure from the same bytes.

The pure domain support and the contract-validation hooks are deliberately *not* wrapped: they are
no-I/O components, and adding file writes there would violate their contract. This recorder only
observes the adapter, planner, builder, verifier, oracle and recovery engine.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from forecontract.runtime.finalization import CLEANUP_STATUSES
from forecontract_web.browser_errors import WEB_BROWSER_ERROR_CODES
from pydantic import BaseModel

__all__ = [
    "EvidenceError",
    "EvidenceRecorder",
    "RecordingAdapter",
    "RecordingBuilder",
    "RecordingPlanner",
    "RecordingRecoveryEngine",
    "RecordingStepVerifier",
    "RecordingTaskOracle",
    "verify_evidence_index",
]

SCHEMA: Final[str] = "0.1"
_INDEX_NAME: Final[str] = "evidence-index.json"
_SNAPSHOT_DIR: Final[str] = "snapshots"


class EvidenceError(Exception):
    """A bounded evidence-recording failure, never a substitute for the real run outcome."""


def _jsonable(value: object) -> object:
    """Convert a real component value into strict JSON without inventing or dropping content."""
    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="json", warnings=False)
        except Exception:  # noqa: BLE001 - reported as one bounded evidence failure
            raise EvidenceError("a returned record could not be serialized") from None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        document: dict = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError("a snapshot mapping carries a non-string key")
            document[key] = _jsonable(item)
        return document
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    raise EvidenceError("a snapshot value is not serializable")


def _digest(raw: bytes) -> str:
    """The SHA-256 of exactly the bytes that were written."""
    return hashlib.sha256(raw).hexdigest()


class EvidenceRecorder:
    """Write real component snapshots and a hash-bound index into one external directory."""

    def __init__(self, root: Path, *, run_id: str, profile: str) -> None:
        self._root = Path(root)
        self._snapshots = self._root / _SNAPSHOT_DIR
        self._run_id = run_id
        self._profile = profile
        self._entries: list[dict[str, Any]] = []
        self._counters: dict[str, int] = defaultdict(int)
        self._observation_ordinal = 0
        self._last_contract_id: str | None = None
        self._attempts: dict[str, int] = defaultdict(int)
        self._contract_entries: dict[str, dict[str, Any]] = {}
        self._outcome_entries: list[dict[str, Any]] = []
        self._verification_entries: list[dict[str, Any]] = []
        self._interrupted: str | None = None
        self._record_failure: str | None = None
        self._unfinished_attempt: str | None = None
        self._boundaries: list[dict[str, Any]] = []
        self._host_waits: list[dict[str, Any]] = []
        self._host_counts: dict[str, dict[str, Any]] = {}
        self._host_started: dict[str, Any] | None = None
        self._host_cleanup: dict[str, Any] = {
            "adapter_error_code": None,
            "runtime_cleanup_status": None,
            "runtime_cleanup_error_code": None,
        }
        try:
            self._snapshots.mkdir(parents=True, exist_ok=False)
        except OSError:
            raise EvidenceError("the evidence snapshot directory could not be created") from None

    # ------------------------------------------------------------------ snapshots

    @property
    def root(self) -> Path:
        """The external evidence root."""
        return self._root

    def _write(self, kind: str, payload: object, associations: dict[str, Any]) -> dict[str, Any]:
        """Write one snapshot synchronously and return its index entry."""
        self._counters[kind] += 1
        ordinal = self._counters[kind]
        name = f"{kind}-{ordinal:04d}.json"
        document = {
            "schema_version": SCHEMA,
            "kind": kind,
            "run_id": self._run_id,
            "ordinal": ordinal,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "associations": associations,
            "payload": _jsonable(payload),
        }
        try:
            raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode("utf-8")
        except (TypeError, ValueError):
            raise EvidenceError("a snapshot could not be serialized as strict JSON") from None
        target = self._snapshots / name
        try:
            target.write_bytes(raw)
        except OSError:
            raise EvidenceError("a snapshot could not be written") from None
        entry = {
            "kind": kind,
            "ordinal": ordinal,
            "path": f"{_SNAPSHOT_DIR}/{name}",
            "sha256": _digest(raw),
            "bytes": len(raw),
            **associations,
        }
        self._entries.append(entry)
        return entry

    def record_observation(self, observation: object) -> dict[str, Any]:
        """Record one real observation, distinguished by this Run's observe call ordinal."""
        self._observation_ordinal += 1
        observation_id = getattr(observation, "observation_id", None)
        return self._write(
            "observation", observation,
            {"call_ordinal": self._observation_ordinal, "observation_id": observation_id},
        )

    def record_forecast(self, observation: object, forecast: object) -> dict[str, Any]:
        """Record one real forecast and the observation it was made from."""
        return self._write(
            "forecast", forecast,
            {"forecast_id": getattr(forecast, "forecast_id", None),
             "observation_id": getattr(observation, "observation_id", None),
             "action_type": getattr(getattr(forecast, "proposed_action", None),
                                    "action_type", None)},
        )

    def record_contract(self, contract: object) -> dict[str, Any]:
        """Record one really committed contract."""
        contract_id = getattr(contract, "contract_id", None)
        entry = self._write(
            "contract", contract,
            {"contract_id": contract_id,
             "forecast_id": getattr(contract, "forecast_id", None),
             "goal_id": getattr(contract, "goal_id", None),
             "observation_id": getattr(contract, "observation_id", None),
             "action_type": getattr(getattr(contract, "action", None), "action_type", None)},
        )
        if isinstance(contract_id, str):
            self._contract_entries[contract_id] = entry
            self._last_contract_id = contract_id
        return entry

    def record_execution(self, action: object, outcome: object) -> dict[str, Any]:
        """Record one real execute input and its actual outcome."""
        contract_id = self._last_contract_id
        attempt_ordinal = None
        if contract_id is not None:
            self._attempts[contract_id] += 1
            attempt_ordinal = self._attempts[contract_id]
        entry = self._write(
            "actual_outcome", {"action": action, "actual_outcome": outcome},
            {"contract_id": contract_id,
             "attempt_ordinal": attempt_ordinal,
             "outcome_id": getattr(outcome, "outcome_id", None),
             "execution_status": getattr(getattr(outcome, "status", None), "value", None)},
        )
        self._outcome_entries.append(entry)
        return entry

    def record_step_verification(self, contract: object, evidence: object,
                                 result: object) -> dict[str, Any]:
        """Record one real step verdict and the complete evidence it was judged on."""
        entry = self._write(
            "step_verification",
            {"evidence": evidence, "result": result},
            {"contract_id": getattr(contract, "contract_id", None),
             "observation_id": getattr(getattr(evidence, "observation", None),
                                       "observation_id", None),
             "verifier": getattr(result, "verifier", None),
             "status": getattr(getattr(result, "status", None), "value", None)},
        )
        self._verification_entries.append(entry)
        return entry

    def record_task_verification(self, criteria: object, evidence: object,
                                 result: object) -> dict[str, Any]:
        """Record one real task-oracle verdict and its independently supplied criteria."""
        entry = self._write(
            "task_verification",
            {"criteria": criteria, "evidence": evidence, "result": result},
            {"goal_id": getattr(criteria, "goal_id", None),
             "observation_id": getattr(getattr(evidence, "observation", None),
                                       "observation_id", None),
             "verifier": getattr(result, "verifier", None),
             "status": getattr(getattr(result, "status", None), "value", None)},
        )
        self._verification_entries.append(entry)
        return entry

    def record_recovery_input(self, contract: object, context: object, safety: object,
                              evidence: object | None) -> dict[str, Any]:
        """Record exactly the inputs one real recovery assessment was asked to judge."""
        return self._write(
            "recovery_input",
            {"contract": contract, "context": context, "safety": safety,
             "evidence": evidence},
            {"contract_id": getattr(context, "contract_id", None),
             "attempt_id": getattr(context, "attempt_id", None),
             "outcome_id": getattr(getattr(context, "actual_outcome", None),
                                   "outcome_id", None)},
        )

    def record_recovery_assessment(self, assessment: object) -> dict[str, Any]:
        """Record one real recovery assessment."""
        return self._write(
            "recovery_assessment", assessment,
            {"contract_id": getattr(assessment, "contract_id", None),
             "attempt_id": getattr(assessment, "attempt_id", None),
             "outcome_id": getattr(assessment, "outcome_id", None),
             "assessment_kind": getattr(assessment, "kind", None),
             "assessment_decision": getattr(assessment, "decision", None),
             "assessment_reason": getattr(assessment, "reason", None)},
        )

    def mark_interrupted(self, name: str) -> None:
        """Record that the caller's run ended without returning a result."""
        self._interrupted = str(name)

    # ------------------------------------------------------------------ diagnostics

    def record_boundary(self, operation: str, *, elapsed_ms: int,
                        error: BaseException | None = None,
                        budget_ms: int | None = None) -> None:
        """Record one real adapter boundary and its permitted fixed failure code.

        ``operation`` is one of the four existing adapter entry points. A failure records the
        closed facility ``code`` carried by the adapter error - never the exception text, its
        arguments or its cause - and an exception without a permitted fixed code records no code
        at all. The record is written before the exception is re-raised into the Core Runtime.
        """
        if operation not in _DIAGNOSTIC_OPERATIONS:
            raise EvidenceError("a host boundary names no declared adapter operation") from None
        code = None
        if error is not None:
            candidate = getattr(error, "code", None)
            if type(candidate) is str and candidate in WEB_BROWSER_ERROR_CODES:
                code = candidate
        self._boundaries.append({
            "operation": operation,
            "outcome": "failed" if error is not None else "ok",
            "error_code": code,
            "elapsed_ms": _elapsed_ms(elapsed_ms),
            "budget_ms": None if budget_ms is None else _positive_ms(budget_ms),
        })

    def record_host_started(self, counts: object) -> None:
        """Record the real site counts observed when the host started the Run."""
        self._host_started = _host_counts(counts)

    def record_host_wait(self, phase: str, *, received: bool, elapsed_ms: int) -> None:
        """Record one bounded host LOGIN_RECEIVED wait of a declared profile phase."""
        if phase not in _DIAGNOSTIC_WAIT_PHASES:
            raise EvidenceError("a host wait names no declared profile phase") from None
        if type(received) is not bool:
            raise EvidenceError("a host wait carries no real boolean receipt") from None
        self._host_waits.append({
            "phase": phase,
            "received": received,
            "elapsed_ms": _elapsed_ms(elapsed_ms),
        })

    def record_host_counts(self, phase: str, counts: object) -> None:
        """Record the real site counts at one declared phase of the profile."""
        if phase not in _DIAGNOSTIC_COUNT_PHASES:
            raise EvidenceError("a host count names no declared profile phase") from None
        self._host_counts[phase] = _host_counts(counts)

    def record_host_cleanup(self, *, adapter_error_code: object,
                            runtime_cleanup_status: object,
                            runtime_cleanup_error_code: object) -> None:
        """Record the real resource-cleanup facts of this Run, in the closed vocabularies."""
        if adapter_error_code is not None and (
                type(adapter_error_code) is not str
                or adapter_error_code not in WEB_BROWSER_ERROR_CODES):
            raise EvidenceError("a cleanup fact carries no permitted adapter code") from None
        if runtime_cleanup_status is not None and (
                type(runtime_cleanup_status) is not str
                or runtime_cleanup_status not in CLEANUP_STATUSES):
            raise EvidenceError("a cleanup fact carries no declared Runtime status") from None
        if runtime_cleanup_error_code is not None and (
                type(runtime_cleanup_error_code) is not str
                or runtime_cleanup_error_code not in _RUNTIME_CLEANUP_CODES):
            raise EvidenceError("a cleanup fact carries no declared Runtime code") from None
        self._host_cleanup = {
            "adapter_error_code": adapter_error_code,
            "runtime_cleanup_status": runtime_cleanup_status,
            "runtime_cleanup_error_code": runtime_cleanup_error_code,
        }

    def _diagnostics(self) -> dict[str, Any]:
        """Return the complete host diagnostics of this Run as a detached strict record."""
        return {
            "boundaries": [dict(record) for record in self._boundaries],
            "host": {
                "started": None if self._host_started is None else dict(self._host_started),
                "waits": [dict(record) for record in self._host_waits],
                "counts": {phase: dict(counts) for phase, counts in self._host_counts.items()},
                "cleanup": dict(self._host_cleanup),
            },
        }

    # ------------------------------------------------------------------ finalisation

    def finalize(self, *, result: object | None, last_finalization: object | None,
                 terminal: dict[str, Any], trace_path: Path, working_memory_path: Path,
                 working_state: object, trace_records: object,
                 cleanup: dict[str, Any]) -> Path:
        """Cross-check the real trace and Working Memory, then write the evidence index."""
        trace_records = list(trace_records)
        events = [self._event(record) for record in trace_records]
        working = _jsonable(working_state) if working_state is not None else None
        observation_calls = (working or {}).get("observation_calls", [])
        self._cross_check_observations(observation_calls)
        contracts = self._cross_check_contracts(events)
        attempts = self._cross_check_attempts(events)
        verifications = self._cross_check_verifications(events)
        trace_raw = self._read_bytes(trace_path)
        memory_raw = self._read_bytes(working_memory_path)
        index = {
            "schema_version": SCHEMA,
            "kind": "web_runtime_evidence_index",
            "run_id": self._run_id,
            "profile": self._profile,
            "call_interrupted": self._interrupted,
            "record_failure": self._record_failure,
            "result": _jsonable(result) if result is not None else None,
            "last_finalization": (_jsonable(last_finalization)
                                  if last_finalization is not None else None),
            "terminal": terminal,
            "cleanup": cleanup,
            "trace": {
                "path": str(trace_path),
                "sha256": _digest(trace_raw),
                "bytes": len(trace_raw),
                "records": len(events),
            },
            "working_memory": {
                "path": str(working_memory_path),
                "sha256": _digest(memory_raw),
                "bytes": len(memory_raw),
                "observation_calls": observation_calls,
            },
            "events": events,
            "contracts": contracts,
            "attempts": attempts,
            "unfinished_attempt_id": self._unfinished_attempt,
            "verifications": verifications,
            "snapshots": self._entries,
            "diagnostics": self._diagnostics(),
        }
        raw = (json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
        target = self._root / _INDEX_NAME
        try:
            target.write_bytes(raw)
        except OSError:
            raise EvidenceError("the evidence index could not be written") from None
        return target

    # ------------------------------------------------------------------ cross-checks

    def _cross_check_observations(self, observation_calls: object) -> None:
        observations = [entry for entry in self._entries
                        if entry["kind"] == "observation"]
        if not isinstance(observation_calls, list):
            if observations:
                raise EvidenceError("the published working memory carries no observation calls")
            return
        # The Runtime publishes every observe call, including one whose observation never
        # returned. Only a call that really returned an observation can have a snapshot; an
        # unreturned call stays in the published record with no snapshot and is never deleted just
        # to make the two counts agree.
        returned: list[dict] = []
        for call in observation_calls:
            if not isinstance(call, dict):
                raise EvidenceError("a published observation call is not a record")
            observation_id = call.get("observation_id")
            if observation_id is None:
                continue
            if type(observation_id) is not str or not observation_id:
                raise EvidenceError("a published observation call is not a usable record")
            returned.append(call)
        if len(returned) != len(observations):
            raise EvidenceError(
                "the returned observation calls do not match the recorded observations")
        # The Runtime's own ordinal counts every local call, so an observation's ordinal is not a
        # dense 1..N sequence. The recorder associates observations by call order and records the
        # Runtime's ordinal separately; a repeated observation_id is therefore still distinguished.
        for entry, call in zip(observations, returned):
            if call.get("observation_id") != entry.get("observation_id"):
                raise EvidenceError("the observation identifier does not match the call provenance")
            entry["runtime_call_ordinal"] = call.get("ordinal")
            entry["phase"] = call.get("phase")
            entry["step_id"] = call.get("step_id")
            entry["attempt_id"] = call.get("attempt_id")

    def _cross_check_contracts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        committed: dict[str, dict[str, Any]] = {}
        for event in events:
            if event.get("kind") == "contract_committed" and event.get("contract_id"):
                committed[event["contract_id"]] = event
        contracts: list[dict[str, Any]] = []
        for contract_id, entry in self._contract_entries.items():
            event = committed.get(contract_id)
            if event is None:
                raise EvidenceError("a recorded contract is missing from the real trace")
            entry["step_id"] = event.get("step_id")
            entry["event_record_id"] = event.get("record_id")
            contracts.append(entry)
        for contract_id in committed:
            if contract_id not in self._contract_entries:
                raise EvidenceError("the real trace committed a contract with no snapshot")
        return sorted(contracts, key=lambda item: item["ordinal"])

    def _cross_check_attempts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        started = [event for event in events if event.get("kind") == "attempt_started"]
        finished = [event for event in events if event.get("kind") == "attempt_finished"]
        per_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in started:
            per_contract[event.get("contract_id")].append(event)
        attempts: list[dict[str, Any]] = []
        for entry in self._outcome_entries:
            contract_id = entry.get("contract_id")
            order = entry.get("attempt_ordinal")
            candidates = per_contract.get(contract_id, [])
            if not isinstance(order, int) or order < 1 or order > len(candidates):
                raise EvidenceError("a recorded outcome has no matching attempt in the trace")
            event = candidates[order - 1]
            entry["attempt_id"] = event.get("attempt_id")
            entry["event_record_id"] = event.get("record_id")
            entry["step_id"] = event.get("step_id")
            attempts.append(entry)
        if len(finished) != len(self._outcome_entries):
            raise EvidenceError("the real attempt finish facts do not match the outcome snapshots")
        if len(started) not in (len(finished), len(finished) + 1):
            raise EvidenceError("the attempt facts and outcome snapshots do not agree")
        if len(started) == len(finished) + 1:
            # A cancellation or a deadline may leave exactly the last Attempt started with no
            # finish fact and no invented ActualOutcome. It is reported, never fabricated.
            unfinished = started[-1]
            if any(event.get("attempt_id") == unfinished.get("attempt_id")
                   for event in finished):
                raise EvidenceError("an attempt without a snapshot was reported as finished")
            self._unfinished_attempt = unfinished.get("attempt_id")
        return sorted(attempts, key=lambda item: item["ordinal"])

    def _cross_check_verifications(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recorded = [event for event in events if event.get("kind") == "verification_recorded"]
        attempt_events = [event for event in recorded if event.get("scope") == "attempt"]
        run_events = [event for event in recorded if event.get("scope") == "run"]
        verifications: list[dict[str, Any]] = []
        attempt_index = 0
        used_run = 0
        step_entries = [entry for entry in self._verification_entries
                        if entry["kind"] == "step_verification"]
        task_entries = [entry for entry in self._verification_entries
                        if entry["kind"] == "task_verification"]
        # The Runtime records one attempt-scope fact for the normal step verdict. A bounded
        # recovery re-checks the same step against a fresh observation; the host records that
        # real call too, but it has no second trace fact, so it is marked as a re-verification.
        for entry in step_entries:
            if attempt_index < len(attempt_events):
                event = attempt_events[attempt_index]
                attempt_index += 1
                entry["trace_fact"] = "attempt_verification"
                entry["event_record_id"] = event.get("record_id")
                entry["attempt_id"] = event.get("attempt_id")
            else:
                entry["trace_fact"] = None
                entry["recovery_reverification"] = True
            verifications.append(entry)
        if attempt_index != len(attempt_events):
            raise EvidenceError("the real trace carries a step verification with no snapshot")
        for entry in task_entries:
            # The Runtime only writes a run-scope verification fact for a real PASS; a failed
            # task verdict is recorded by the host but has no trace fact to reference.
            if entry.get("status") != "pass":
                entry["trace_fact"] = None
                verifications.append(entry)
                continue
            if used_run >= len(run_events):
                raise EvidenceError("a passing task verdict has no run-scope trace fact")
            event = run_events[used_run]
            used_run += 1
            entry["trace_fact"] = "run_verification"
            entry["event_record_id"] = event.get("record_id")
            entry["attempt_id"] = event.get("attempt_id")
            verifications.append(entry)
        if used_run != len(run_events):
            raise EvidenceError("the real trace carries a task verification with no snapshot")
        return sorted(verifications, key=lambda item: item["ordinal"])

    @staticmethod
    def _event(record: object) -> dict[str, Any]:
        """Project one strict trace record into a small association summary."""
        return {
            "record_id": getattr(record, "record_id", None),
            "kind": getattr(record, "kind", None),
            "sequence": getattr(record, "sequence", None),
            "step_id": getattr(record, "step_id", None),
            "contract_id": getattr(record, "contract_id", None),
            "attempt_id": getattr(record, "attempt_id", None),
            "scope": getattr(record, "scope", None),
            "verification_status": getattr(
                getattr(record, "verification_status", None), "value", None),
        }

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        try:
            return Path(path).read_bytes()
        except OSError:
            raise EvidenceError("a referenced evidence file could not be read") from None


class RecordingAdapter:
    """A pure-forwarding adapter proxy: one real call, one unchanged return.

    Every real boundary is timed and, when it fails, its permitted fixed facility code is recorded
    before the exception continues into the Core Runtime. The forwarding behaviour itself is
    unchanged: one real call, the real object returned unchanged, the real exception re-raised
    with its type and identity intact.

    The declared budgets are recorded per operation and never merged: ``start`` records the
    effective startup allowance (``startup_timeout_ms`` when declared, otherwise the operation
    value so the historical diagnostics stay identical), ``observe``/``execute`` record the
    operation allowance, and ``stop`` records the cleanup allowance. A budget is diagnostic
    metadata only; it never changes what the real adapter enforces.
    """

    def __init__(self, inner: object, recorder: EvidenceRecorder, *,
                 operation_timeout_ms: float | None = None,
                 cleanup_timeout_ms: float | None = None,
                 startup_timeout_ms: float | None = None) -> None:
        self._inner = inner
        self._recorder = recorder
        self._operation_budget_ms = operation_timeout_ms
        self._cleanup_budget_ms = cleanup_timeout_ms
        self._startup_budget_ms = (
            operation_timeout_ms if startup_timeout_ms is None else startup_timeout_ms)

    @property
    def name(self) -> object:
        return self._inner.name  # type: ignore[attr-defined]

    def capabilities(self) -> object:
        return self._inner.capabilities()  # type: ignore[attr-defined]

    def validate_action(self, action: object) -> object:
        return self._inner.validate_action(action)  # type: ignore[attr-defined]

    def _budget_ms(self, operation: str) -> float | None:
        if operation == "stop":
            return self._cleanup_budget_ms
        if operation == "start":
            return self._startup_budget_ms
        return self._operation_budget_ms

    async def _record(self, operation: str, call) -> object:
        """Run one real boundary, record its duration and fixed code, then forward the result."""
        started = time.monotonic()
        try:
            result = await call()
        except BaseException as error:
            self._recorder.record_boundary(
                operation,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                error=error,
                budget_ms=self._budget_ms(operation),
            )
            raise
        self._recorder.record_boundary(
            operation,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            budget_ms=self._budget_ms(operation),
        )
        return result

    async def start(self) -> object:
        return await self._record("start", self._inner.start)  # type: ignore[attr-defined]

    async def stop(self) -> object:
        return await self._record("stop", self._inner.stop)  # type: ignore[attr-defined]

    async def observe(self) -> object:
        observation = await self._record("observe", self._inner.observe)  # type: ignore[attr-defined]
        self._recorder.record_observation(observation)
        return observation

    async def execute(self, action: object) -> object:
        outcome = await self._record(
            "execute", lambda: self._inner.execute(action))  # type: ignore[attr-defined]
        self._recorder.record_execution(action, outcome)
        return outcome


class RecordingPlanner:
    """A pure-forwarding planner proxy."""

    def __init__(self, inner: object, recorder: EvidenceRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def plan(self, observation: object) -> object:
        forecast = await self._inner.plan(observation)  # type: ignore[attr-defined]
        self._recorder.record_forecast(observation, forecast)
        return forecast


class RecordingBuilder:
    """A pure-forwarding Contract Builder proxy."""

    def __init__(self, inner: object, recorder: EvidenceRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def build(self, *args: object, **kwargs: object) -> object:
        contract = self._inner.build(*args, **kwargs)  # type: ignore[attr-defined]
        self._recorder.record_contract(contract)
        return contract


class RecordingStepVerifier:
    """A pure-forwarding step-verifier proxy."""

    def __init__(self, inner: object, recorder: EvidenceRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def verify_step(self, *, contract: object, evidence: object) -> object:
        result = self._inner.verify_step(contract=contract, evidence=evidence)  # type: ignore[attr-defined]
        self._recorder.record_step_verification(contract, evidence, result)
        return result


class RecordingTaskOracle:
    """A pure-forwarding task-oracle proxy."""

    def __init__(self, inner: object, recorder: EvidenceRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def evaluate_completion(self, *, criteria: object, evidence: object) -> object:
        result = self._inner.evaluate_completion(criteria=criteria, evidence=evidence)  # type: ignore[attr-defined]
        self._recorder.record_task_verification(criteria, evidence, result)
        return result


class RecordingRecoveryEngine:
    """A pure-forwarding recovery-engine proxy that records its real inputs and verdict."""

    def __init__(self, inner: object, recorder: EvidenceRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def assess(self, *, contract: object, context: object, safety: object,
               evidence: object | None = None) -> object:
        self._recorder.record_recovery_input(contract, context, safety, evidence)
        assessment = self._inner.assess(contract=contract, context=context, safety=safety,
                                        evidence=evidence)  # type: ignore[attr-defined]
        self._recorder.record_recovery_assessment(assessment)
        return assessment


_SNAPSHOT_KINDS: Final[tuple[str, ...]] = (
    "observation", "forecast", "contract", "actual_outcome",
    "step_verification", "task_verification", "recovery_input", "recovery_assessment",
)
_INDEX_KEYS: Final[frozenset[str]] = frozenset({
    "schema_version", "kind", "run_id", "profile", "call_interrupted", "record_failure",
    "result", "last_finalization", "terminal", "cleanup", "trace", "working_memory",
    "events", "contracts", "attempts", "unfinished_attempt_id", "verifications", "snapshots",
    "diagnostics",
})
_EVENT_KEYS: Final[frozenset[str]] = frozenset({
    "record_id", "kind", "sequence", "step_id", "contract_id", "attempt_id", "scope",
    "verification_status",
})
_TRACE_KINDS: Final[frozenset[str]] = frozenset({
    "run_started", "step_started", "contract_committed", "attempt_started",
    "attempt_finished", "verification_recorded", "run_finished",
})
_SNAPSHOT_REQUIRED: Final[tuple[str, ...]] = ("kind", "ordinal", "path", "sha256", "bytes")

#: The host boundaries whose real latency and fixed failure code are recorded. Only these four
#: adapter entry points exist, so no caller-supplied operation name can become a record.
_DIAGNOSTIC_OPERATIONS: Final[tuple[str, ...]] = ("start", "observe", "execute", "stop")
_DIAGNOSTIC_OUTCOMES: Final[tuple[str, ...]] = ("ok", "failed")
_DIAGNOSTIC_BOUNDARY_KEYS: Final[frozenset[str]] = frozenset({
    "operation", "outcome", "error_code", "elapsed_ms", "budget_ms",
})
#: The host wait phases whose LOGIN_RECEIVED fact is recorded. Every one is a declared profile
#: phase, never a caller-invented name.
_DIAGNOSTIC_WAIT_PHASES: Final[tuple[str, ...]] = ("release", "deadline", "cancel")
_DIAGNOSTIC_WAIT_KEYS: Final[frozenset[str]] = frozenset({"phase", "received", "elapsed_ms"})
#: The declared count phases of a profile. A profile only records the phases it really reached.
_DIAGNOSTIC_COUNT_PHASES: Final[frozenset[str]] = frozenset({
    "before_release", "before_cancel", "after_run",
})
#: The fixed fields of one real site count fact. No free text is accepted except the closed
#: scenario name, so an arbitrary exception sentence can never be stored as a count.
_DIAGNOSTIC_COUNT_KEYS: Final[frozenset[str]] = frozenset({
    "generation", "scenario", "running", "login_requests", "auth_successes", "auth_failures",
    "commits", "active_requests", "response_blocked", "authenticated_sessions",
})
_DIAGNOSTIC_CLEANUP_KEYS: Final[frozenset[str]] = frozenset({
    "adapter_error_code", "runtime_cleanup_status", "runtime_cleanup_error_code",
})
#: The only Runtime cleanup failure code a finalization may carry.
_RUNTIME_CLEANUP_CODES: Final[tuple[str, ...]] = ("CLEANUP_FAILURE",)

#: The Runtime's public terminal vocabulary mapped onto the stricter Trace vocabulary.
_RUNTIME_TERMINAL_TO_TRACE: Final[dict[str, str]] = {
    "completed": "success",
    "failed": "failure",
    "cancelled": "cancelled",
    "timeout": "timeout",
    "aborted": "aborted",
}
#: The Runtime's closed reason vocabulary mapped onto the Trace's closed reason vocabulary.
_RUNTIME_REASON_TO_TRACE: Final[dict[str, str | None]] = {
    "TASK_ACHIEVED": None,
    "BUDGET_EXHAUSTED": "policy_abort",
    "POLICY_ABORT": "policy_abort",
    "MAX_STEPS_EXHAUSTED": "policy_abort",
    "MAX_PLANNING_EXHAUSTED": "policy_abort",
    "MAX_REPLANS_EXHAUSTED": "policy_abort",
    "DEADLINE_EXCEEDED": "deadline_exceeded",
    "OPERATION_TIMEOUT": "deadline_exceeded",
    "USER_CANCELLED": "user_cancelled",
    "DEPENDENCY_FAILURE": "internal_error",
    "EVIDENCE_INVALID": "internal_error",
    "IDENTITY_CONFLICT": "internal_error",
    "INVALID_TRANSITION": "internal_error",
    "TRACE_FAILURE": "internal_error",
    "CLEANUP_FAILURE": "internal_error",
}


def _fail(message: str) -> None:
    """Refuse one invariant, always with the fixed bounded evidence error."""
    raise EvidenceError(message)


def _require(condition: object, message: str) -> None:
    """Require one structural invariant."""
    if not condition:
        _fail(message)


def _strict_int(value: object, minimum: int) -> bool:
    """Whether a value is a real ``int`` (never a bool) not below ``minimum``."""
    return type(value) is int and value >= minimum


def _hex_digest(value: object) -> bool:
    """Whether a value is a lowercase 64-character SHA-256 hex digest."""
    if type(value) is not str or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _timestamp_ok(value: object) -> bool:
    """Whether a value is an ISO-8601 timestamp with an explicit timezone."""
    if type(value) is not str or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _resolve_member(root: Path, raw: object) -> Path:
    """Resolve a declared member path against the evidence root, refusing an escape."""
    _require(type(raw) is str and bool(raw), "an evidence member path is not usable")
    candidate = Path(raw)
    base = root.resolve()
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    _require(resolved == base or base in resolved.parents,
             "an evidence member path escapes the evidence root")
    return resolved


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        _fail("a referenced evidence file could not be read")
        raise


def _mapping(value: object, what: str) -> dict:
    """Require one JSON object and return it."""
    _require(isinstance(value, dict), f"{what} is not a record")
    return value


def _elapsed_ms(value: object) -> int:
    """Return one non-negative integer millisecond duration."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError("a host diagnostic timing is not a usable duration") from None
    if not (0 <= value < float("inf")):
        raise EvidenceError("a host diagnostic timing is not a usable duration") from None
    return int(value)


def _positive_ms(value: object) -> int:
    """Return one strictly positive integer millisecond allowance."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError("a host boundary budget is not a usable duration") from None
    if not (0 < value < float("inf")):
        raise EvidenceError("a host boundary budget is not a usable duration") from None
    return int(value)


def _host_counts(value: object) -> dict[str, Any]:
    """Return one detached, strictly typed site count record."""
    document = _mapping(value, "the host site counts")
    _require(frozenset(document) == _DIAGNOSTIC_COUNT_KEYS,
             "the host site counts do not declare their structure")
    for name, item in document.items():
        if name == "scenario":
            _require(type(item) is str and 0 < len(item) <= 64,
                     "the host site counts carry no scenario name")
        elif name in ("running", "response_blocked"):
            _require(type(item) is bool, "a host site count is not a real boolean")
        else:
            _require(_strict_int(item, 0), "a host site count is not a non-negative integer")
    return dict(document)


def _list(value: object, what: str) -> list:
    """Require one JSON array and return it."""
    _require(isinstance(value, list), f"{what} is not a list")
    return value


def _association(mapping: object, key: str, what: str) -> object:
    """Require one declared association key."""
    _require(isinstance(mapping, dict) and key in mapping, f"{what} is missing")
    return mapping[key]


def _bind_association(record: dict, snapshot: dict[str, Any], what: str) -> None:
    """Bind one association row to the exact snapshot entry it claims to reference.

    The row and the snapshot entry are written by the same recorder call, so a genuine index
    duplicates the very same fields. Requiring the two to carry exactly the same key set and the
    same value for every key is what makes a snapshot swap, a changed digest, a changed identifier
    or an invented (or dropped) association visible: a hash on the inventory alone would not bind
    the copied row to that exact entry.
    """
    _require(frozenset(record) == frozenset(snapshot),
             f"{what} does not match its snapshot entry")
    for name, value in record.items():
        _require(snapshot[name] == value,
                 f"{what} carries another {name}")


def _read_index(root: Path) -> dict[str, Any]:
    """Read the index and require its complete strict top-level structure."""
    index_path = root / _INDEX_NAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _fail("the evidence index is missing or unreadable")
    document = _mapping(index, "the evidence index")
    _require(frozenset(document) == _INDEX_KEYS, "the evidence index does not declare its structure")
    _require(document.get("schema_version") == SCHEMA, "the evidence index does not declare its schema")
    _require(document.get("kind") == "web_runtime_evidence_index",
             "the evidence index does not declare its kind")
    _require(type(document.get("run_id")) is str and bool(document["run_id"]),
             "the evidence index carries no run identifier")
    _require(type(document.get("profile")) is str and bool(document["profile"]),
             "the evidence index carries no profile")
    _require(document.get("call_interrupted") is None
             or type(document["call_interrupted"]) is str,
             "the evidence index carries an unusable interruption")
    _require(document.get("record_failure") is None
             or type(document["record_failure"]) is str,
             "the evidence index carries an unusable recording failure")
    _mapping(document.get("cleanup"), "the evidence cleanup")
    return document


def _snapshot_entries(root: Path, index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate the complete snapshot inventory, its files and every snapshot's own content."""
    run_id = index["run_id"]
    entries = _list(index.get("snapshots"), "the snapshot inventory")
    by_path: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, list[int]] = {kind: [] for kind in _SNAPSHOT_KINDS}
    for entry in entries:
        record = _mapping(entry, "a snapshot entry")
        for name in _SNAPSHOT_REQUIRED:
            _require(name in record, "a snapshot entry is missing a required field")
        kind = record["kind"]
        _require(type(kind) is str and kind in _SNAPSHOT_KINDS,
                 "a snapshot entry declares an unknown kind")
        _require(_strict_int(record["ordinal"], 1), "a snapshot ordinal is not a positive integer")
        _require(_strict_int(record["bytes"], 1), "a snapshot size is not a positive integer")
        _require(_hex_digest(record["sha256"]), "a snapshot digest is not a SHA-256 value")
        relative = record["path"]
        _require(type(relative) is str and "\\" not in relative, "a snapshot path is not portable")
        parts = PurePosixPath(relative).parts
        _require(len(parts) == 2 and parts[0] == _SNAPSHOT_DIR,
                 "a snapshot path is not a direct child of the snapshot directory")
        expected_name = f"{kind}-{record['ordinal']:04d}.json"
        _require(parts[1] == expected_name, "a snapshot path does not match its declared identity")
        _require(relative not in by_path, "a snapshot path is listed more than once")
        by_path[relative] = record
        by_kind[kind].append(record["ordinal"])

    for kind, ordinals in by_kind.items():
        _require(sorted(ordinals) == list(range(1, len(ordinals) + 1)),
                 f"the {kind} snapshot ordinals are not unique and contiguous")

    snapshot_dir = root / _SNAPSHOT_DIR
    on_disk: list[str] = []
    if snapshot_dir.exists():
        _require(snapshot_dir.is_dir(), "the snapshot path is not a directory")
        for child in sorted(snapshot_dir.iterdir()):
            if child.is_file():
                on_disk.append(child.name)
            else:
                _fail("the snapshot directory carries an undeclared entry")
    _require(sorted(on_disk) == sorted(Path(relative).name for relative in by_path),
             "the snapshot directory does not match the complete inventory")
    if not by_path:
        _require(not on_disk, "an empty snapshot inventory hides files on disk")

    for relative, entry in by_path.items():
        target = _resolve_member(root, relative)
        _require(target.is_file(), "a referenced snapshot is missing")
        raw = _read_bytes(target)
        _require(len(raw) == entry["bytes"] and _digest(raw) == entry["sha256"],
                 "a referenced snapshot does not match its recorded bytes")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            _fail("a referenced snapshot is not valid JSON")
        document = _mapping(document, "a snapshot")
        _require(document.get("schema_version") == SCHEMA,
                 "a snapshot does not declare its schema")
        _require(document.get("kind") == entry["kind"], "a snapshot's kind does not match its entry")
        _require(document.get("ordinal") == entry["ordinal"],
                 "a snapshot's ordinal does not match its entry")
        _require(document.get("run_id") == run_id, "a snapshot belongs to another run")
        _require(_timestamp_ok(document.get("recorded_at")),
                 "a snapshot carries no usable recording time")
        associations = _mapping(document.get("associations"), "a snapshot's associations")
        for name, value in associations.items():
            _require(name in entry and entry[name] == value,
                     "a snapshot association does not match its index entry")
        _require("payload" in document, "a snapshot carries no payload")
        _check_snapshot_payload(entry["kind"], document["payload"], entry)
    return by_path


def _check_snapshot_payload(kind: str, payload: object, entry: dict[str, Any]) -> None:
    """Cross-check the real identifiers carried inside one snapshot payload."""
    body = _mapping(payload, "a snapshot payload")
    if kind == "observation":
        _require(_association(body, "observation_id", "an observation snapshot")
                 == entry.get("observation_id"), "an observation snapshot carries another identifier")
    elif kind == "forecast":
        _require(_association(body, "forecast_id", "a forecast snapshot")
                 == entry.get("forecast_id"), "a forecast snapshot carries another identifier")
        action = _mapping(body.get("proposed_action"), "a forecast action")
        _require(action.get("action_type") == entry.get("action_type"),
                 "a forecast snapshot carries another action type")
    elif kind == "contract":
        _require(_association(body, "contract_id", "a contract snapshot")
                 == entry.get("contract_id"), "a contract snapshot carries another identifier")
        _require(_association(body, "goal_id", "a contract snapshot") == entry.get("goal_id"),
                 "a contract snapshot carries another goal")
        _require(_association(body, "forecast_id", "a contract snapshot")
                 == entry.get("forecast_id"), "a contract snapshot carries another forecast")
        _require(_association(body, "observation_id", "a contract snapshot")
                 == entry.get("observation_id"), "a contract snapshot carries another observation")
        action = _mapping(body.get("action"), "a contract action")
        _require(action.get("action_type") == entry.get("action_type"),
                 "a contract snapshot carries another action type")
    elif kind == "actual_outcome":
        outcome = _mapping(_association(body, "actual_outcome", "an outcome snapshot"),
                           "an actual outcome")
        _require(_association(outcome, "outcome_id", "an outcome snapshot")
                 == entry.get("outcome_id"), "an outcome snapshot carries another identifier")
        status = outcome.get("status")
        _require(status == entry.get("execution_status"),
                 "an outcome snapshot carries another execution status")
    elif kind == "step_verification":
        result = _mapping(_association(body, "result", "a step verification snapshot"),
                          "a step result")
        _require(_association(result, "verifier", "a step result") == entry.get("verifier"),
                 "a step verification snapshot carries another verifier")
        _require(result.get("status") == entry.get("status"),
                 "a step verification snapshot carries another status")
        metadata = _mapping(result.get("metadata"), "a step result's metadata")
        _require(metadata.get("contract_id") == entry.get("contract_id"),
                 "a step verification snapshot is bound to another contract")
        _require(metadata.get("observation_id") == entry.get("observation_id"),
                 "a step verification snapshot is bound to another observation")
    elif kind == "task_verification":
        result = _mapping(_association(body, "result", "a task verification snapshot"),
                          "a task result")
        _require(_association(result, "verifier", "a task result") == entry.get("verifier"),
                 "a task verification snapshot carries another verifier")
        _require(result.get("status") == entry.get("status"),
                 "a task verification snapshot carries another status")
        metadata = _mapping(result.get("metadata"), "a task result's metadata")
        _require(metadata.get("goal_id") == entry.get("goal_id"),
                 "a task verification snapshot is bound to another goal")
        _require(metadata.get("observation_id") == entry.get("observation_id"),
                 "a task verification snapshot is bound to another observation")
    elif kind == "recovery_input":
        context = _mapping(_association(body, "context", "a recovery input snapshot"),
                           "a recovery context")
        _require(context.get("contract_id") == entry.get("contract_id"),
                 "a recovery input snapshot carries another contract")
        _require(context.get("attempt_id") == entry.get("attempt_id"),
                 "a recovery input snapshot carries another attempt")
        outcome = _mapping(context.get("actual_outcome"), "a recovery outcome")
        _require(outcome.get("outcome_id") == entry.get("outcome_id"),
                 "a recovery input snapshot carries another outcome")
    elif kind == "recovery_assessment":
        _require(_association(body, "contract_id", "a recovery assessment")
                 == entry.get("contract_id"), "a recovery assessment carries another contract")
        _require(_association(body, "attempt_id", "a recovery assessment")
                 == entry.get("attempt_id"), "a recovery assessment carries another attempt")
        _require(_association(body, "outcome_id", "a recovery assessment")
                 == entry.get("outcome_id"), "a recovery assessment carries another outcome")
        _require(body.get("kind") == entry.get("assessment_kind"),
                 "a recovery assessment carries another kind")
        _require(body.get("decision") == entry.get("assessment_decision"),
                 "a recovery assessment carries another decision")
        _require(body.get("reason") == entry.get("assessment_reason"),
                 "a recovery assessment carries another reason")


def _read_trace(root: Path, index: dict[str, Any]) -> tuple[list[dict], dict, dict]:
    """Re-read the real trace bytes and prove the declared record history."""
    trace = _mapping(index.get("trace"), "the trace declaration")
    for name in ("path", "sha256", "bytes", "records"):
        _require(name in trace, "the trace declaration is incomplete")
    _require(_hex_digest(trace["sha256"]), "the trace declaration carries no SHA-256 value")
    _require(_strict_int(trace["bytes"], 0), "the trace size is not a non-negative integer")
    _require(_strict_int(trace["records"], 0), "the trace record count is not a non-negative integer")
    path = _resolve_member(root, trace["path"])
    _require(path.is_file(), "the declared trace file is missing")
    raw = _read_bytes(path)
    _require(len(raw) == trace["bytes"] and _digest(raw) == trace["sha256"],
             "the trace file does not match its declared bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail("the trace file is not valid UTF-8")
    lines = [line for line in text.splitlines() if line.strip()]
    _require(len(lines) == trace["records"], "the trace record count does not match its file")
    records: list[dict] = []
    seen: set[str] = set()
    for position, line in enumerate(lines):
        try:
            record = json.loads(line)
        except ValueError:
            _fail("the trace carries a record that is not valid JSON")
        record = _mapping(record, "a trace record")
        _require(record.get("schema_version") == SCHEMA,
                 "a trace record does not declare its schema")
        _require(record.get("run_id") == index["run_id"], "a trace record belongs to another run")
        _require(record.get("sequence") == position, "the trace sequence is not contiguous")
        record_id = record.get("record_id")
        _require(type(record_id) is str and bool(record_id),
                 "a trace record carries no usable identity")
        _require(record_id not in seen, "the trace repeats a record identity")
        seen.add(record_id)
        _require(record.get("kind") in _TRACE_KINDS, "a trace record declares an unknown kind")
        records.append(record)
    _require(bool(records) and records[0].get("kind") == "run_started",
             "the trace does not begin with a Run")
    finished = [record for record in records if record.get("kind") == "run_finished"]
    _require(len(finished) == 1 and records[-1] is finished[0],
             "the trace does not carry exactly one final terminal fact")
    return records, trace, finished[0]


def _verify_events(index: dict[str, Any], records: list[dict]) -> list[dict]:
    """Prove every projected event is a faithful summary of a real trace record."""
    events = _list(index.get("events"), "the event projection")
    _require(len(events) == len(records), "the event projection does not match the trace")
    projected: list[dict] = []
    for position, event in enumerate(events):
        item = _mapping(event, "an event projection")
        _require(frozenset(item) == _EVENT_KEYS, "an event projection has an unexpected shape")
        record = records[position]
        _require(item["record_id"] == record.get("record_id"),
                 "an event projection carries another record identity")
        _require(item["kind"] == record.get("kind"),
                 "an event projection carries another kind")
        _require(item["sequence"] == record.get("sequence"),
                 "an event projection carries another sequence")
        for name in ("step_id", "contract_id", "attempt_id", "scope", "verification_status"):
            _require(item[name] == record.get(name),
                     f"an event projection carries another {name}")
        projected.append(item)
    return projected


def _verify_working_memory(root: Path, index: dict[str, Any],
                           observations: list[dict]) -> None:
    """Re-read Working Memory bytes and prove the observation call provenance."""
    memory = _mapping(index.get("working_memory"), "the working memory declaration")
    for name in ("path", "sha256", "bytes", "observation_calls"):
        _require(name in memory, "the working memory declaration is incomplete")
    _require(_hex_digest(memory["sha256"]), "the working memory carries no SHA-256 value")
    _require(_strict_int(memory["bytes"], 0), "the working memory size is not a non-negative integer")
    _list(memory["observation_calls"], "the working memory observation calls")
    path = _resolve_member(root, memory["path"])
    _require(path.is_file(), "the declared working memory file is missing")
    raw = _read_bytes(path)
    _require(len(raw) == memory["bytes"] and _digest(raw) == memory["sha256"],
             "the working memory file does not match its declared bytes")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        _fail("the working memory file is not valid JSON")
    document = _mapping(document, "the working memory")
    _require(document.get("published_by") == "runtime",
             "the working memory was not published by the Runtime")
    state = _mapping(document.get("state"), "the working memory state")
    _require(state.get("run_id") == index["run_id"], "the working memory belongs to another run")
    _require(state.get("schema_version") == SCHEMA, "the working memory does not declare its schema")
    calls = state.get("observation_calls")
    _require(calls == memory["observation_calls"],
             "the working memory call provenance does not match its declaration")
    by_ordinal = {}
    returned: set[int] = set()
    for call in calls:
        call = _mapping(call, "an observation call")
        ordinal = call.get("ordinal")
        _require(_strict_int(ordinal, 1), "an observation call ordinal is not a positive integer")
        _require(ordinal not in by_ordinal, "an observation call ordinal is repeated")
        by_ordinal[ordinal] = call
        observation_id = call.get("observation_id")
        if observation_id is not None:
            _require(type(observation_id) is str and bool(observation_id),
                     "a returned observation call carries no usable identifier")
            returned.add(ordinal)
        # A call whose observation never returned keeps its real ``None`` and has no snapshot;
        # deleting it to make the counts agree would erase the Runtime's provenance.
    observed: set[int] = set()
    for entry in observations:
        ordinal = entry.get("runtime_call_ordinal")
        _require(ordinal in by_ordinal, "an observation snapshot has no published call")
        _require(ordinal not in observed, "two observation snapshots claim one call")
        observed.add(ordinal)
        call = by_ordinal[ordinal]
        _require(call.get("observation_id") is not None,
                 "an observation snapshot claims a call that never returned")
        for name in ("observation_id", "phase", "step_id", "attempt_id"):
            _require(entry.get(name) == call.get(name),
                     f"an observation snapshot carries another {name} call association")
    _require(observed == returned,
             "the observation snapshots do not cover exactly the returned calls")
    terminal = _mapping(index.get("terminal"), "the terminal declaration")
    for name in ("terminal_status", "reason_code"):
        _require(state.get(name) == terminal.get(name),
                 "the working memory terminal does not match the published terminal")
    # The result is a summary of the very Run whose working memory is published here. Its
    # counters are cross-checked against that independent Runtime record; a trace-derived count
    # is checked in ``_verify_terminal``.
    result = index.get("result")
    if result is not None:
        result = _mapping(result, "the run result")
        for name in ("steps_started", "planning_calls", "attempts_started",
                     "retries_used", "replans_used"):
            _require(result.get(name) == state.get(name),
                     f"the run result {name} does not match the published working memory")


def _verify_terminal(index: dict[str, Any], completed: dict,
                     records: list[dict]) -> dict[str, Any]:
    """Prove the published terminal is the one the real trace ended with."""
    terminal = _mapping(index.get("terminal"), "the terminal declaration")
    _require(frozenset(terminal) == {"terminal_status", "reason_code", "source"},
             "the terminal declaration has an unexpected shape")
    status = terminal["terminal_status"]
    reason = terminal["reason_code"]
    source = terminal["source"]
    _require(type(status) is str and status in _RUNTIME_TERMINAL_TO_TRACE,
             "the terminal status is not a declared Runtime status")
    _require(reason is None or (type(reason) is str and reason in _RUNTIME_REASON_TO_TRACE),
             "the terminal reason is not a declared Runtime reason")
    _require(source in ("run_result", "last_finalization"),
             "the terminal declaration names no real source")
    _require(completed.get("terminal_status") == _RUNTIME_TERMINAL_TO_TRACE[status],
             "the terminal status contradicts the real trace")
    _require(completed.get("reason_code") == _RUNTIME_REASON_TO_TRACE.get(reason),
             "the terminal reason contradicts the real trace")
    result = index.get("result")
    last_finalization = index.get("last_finalization")
    if result is not None:
        result = _mapping(result, "the run result")
        _require(result.get("run_id") == index["run_id"], "the run result belongs to another run")
        _require(result.get("terminal_status") == status,
                 "the run result terminal status does not match the terminal declaration")
        _require(result.get("reason_code") == reason,
                 "the run result reason does not match the terminal declaration")
        for name in ("steps_started", "planning_calls", "attempts_started",
                     "retries_used", "replans_used"):
            _require(_strict_int(result.get(name), 0), f"the run result carries no usable {name}")
        # The two counts the trace itself publishes are cross-checked against it, not trusted
        # from the summary. The remaining counters are cross-checked against the published
        # working memory in ``_verify_working_memory``.
        steps = sum(1 for record in records if record.get("kind") == "step_started")
        started = sum(1 for record in records if record.get("kind") == "attempt_started")
        _require(result["steps_started"] == steps,
                 "the run result step count contradicts the real trace")
        _require(result["attempts_started"] == started,
                 "the run result attempt count contradicts the real trace")
    _require((source == "run_result") == (result is not None),
             "the terminal source does not match the published result")
    if last_finalization is not None:
        last_finalization = _mapping(last_finalization, "the finalization snapshot")
        _require(last_finalization.get("run_id") == index["run_id"],
                 "the finalization snapshot belongs to another run")
        # The primary diagnosis is the same fact in every source mode, so it must agree with the
        # published terminal. A cleanup error is a *secondary* fact and is deliberately never
        # required to equal the primary one: a completed Run may legitimately carry a cleanup
        # failure, and a failed Run may legitimately carry none.
        _require(last_finalization.get("primary_terminal_status") == status,
                 "the finalization terminal status contradicts the terminal declaration")
        _require(last_finalization.get("primary_reason_code") == reason,
                 "the finalization reason contradicts the terminal declaration")
    elif source == "last_finalization":
        _fail("the terminal claims a finalization that was never published")
    return terminal


def _event_by_id(records: list[dict]) -> dict[str, dict]:
    """Index the real trace records by their identity."""
    return {record["record_id"]: record for record in records}


def _snapshot_values(snapshots: dict[str, dict[str, Any]], kind: str,
                     name: str) -> set:
    """The declared values of one association field across snapshots of one kind."""
    return {entry.get(name) for entry in snapshots.values() if entry["kind"] == kind}


def _verify_contracts(index: dict[str, Any], records: list[dict],
                      snapshots: dict[str, dict[str, Any]]) -> None:
    """Prove every committed-contract snapshot is bound to a real commit event."""
    contracts = _list(index.get("contracts"), "the contract associations")
    events = _event_by_id(records)
    committed = [record for record in records
                 if record.get("kind") == "contract_committed"]
    _require(len(contracts) == len(committed),
             "the contract associations do not cover the real commits")
    _require(len(contracts) == sum(1 for entry in snapshots.values()
                                   if entry["kind"] == "contract"),
             "the contract associations do not cover the contract snapshots")
    observation_ids = _snapshot_values(snapshots, "observation", "observation_id")
    seen: set[str] = set()
    seen_paths: set[str] = set()
    for entry in contracts:
        record = _mapping(entry, "a contract association")
        contract_id = record.get("contract_id")
        _require(type(contract_id) is str and bool(contract_id),
                 "a contract association carries no identifier")
        _require(contract_id not in seen, "a contract association repeats a contract")
        seen.add(contract_id)
        _require(record.get("path") in snapshots,
                 "a contract association does not name a contract snapshot")
        snapshot = snapshots[record["path"]]
        _require(snapshot["kind"] == "contract",
                 "a contract association does not name a contract snapshot")
        _bind_association(record, snapshot, "a contract association")
        _require(record["path"] not in seen_paths,
                 "a contract association repeats a snapshot")
        seen_paths.add(record["path"])
        event = events.get(record.get("event_record_id"))
        _require(event is not None and event.get("kind") == "contract_committed"
                 and event.get("contract_id") == contract_id,
                 "a contract association names no real commit event")
        _require(record.get("step_id") == event.get("step_id"),
                 "a contract association carries another step")
        result = index.get("result")
        if result is not None:
            _require(record.get("goal_id") == result.get("goal_id"),
                     "a contract association carries another goal")
        _require(record.get("observation_id") in observation_ids,
                 "a contract association names no real observation")


def _verify_attempts(index: dict[str, Any], records: list[dict],
                     snapshots: dict[str, dict[str, Any]]) -> None:
    """Prove every outcome snapshot is bound to a real attempt and the attempt facts agree."""
    attempts = _list(index.get("attempts"), "the attempt associations")
    events = _event_by_id(records)
    started = [record for record in records if record.get("kind") == "attempt_started"]
    finished = [record for record in records if record.get("kind") == "attempt_finished"]
    _require(len(attempts) == len(finished),
             "the attempt associations do not cover the finished attempts")
    _require(len(attempts) == sum(1 for entry in snapshots.values()
                                  if entry["kind"] == "actual_outcome"),
             "the attempt associations do not cover the outcome snapshots")
    seen: set[str] = set()
    ordinals: dict[str, list[int]] = {}
    for entry in attempts:
        record = _mapping(entry, "an attempt association")
        attempt_id = record.get("attempt_id")
        _require(type(attempt_id) is str and bool(attempt_id),
                 "an attempt association carries no attempt identifier")
        _require(attempt_id not in seen, "an attempt association repeats an attempt")
        seen.add(attempt_id)
        _require(record.get("path") in snapshots,
                 "an attempt association does not name an outcome snapshot")
        snapshot = snapshots[record["path"]]
        _require(snapshot["kind"] == "actual_outcome",
                 "an attempt association does not name an outcome snapshot")
        _bind_association(record, snapshot, "an attempt association")
        event = events.get(record.get("event_record_id"))
        _require(event is not None and event.get("kind") == "attempt_started"
                 and event.get("attempt_id") == attempt_id,
                 "an attempt association names no real attempt start")
        _require(event.get("contract_id") == record.get("contract_id"),
                 "an attempt association carries another contract")
        _require(event.get("step_id") == record.get("step_id"),
                 "an attempt association carries another step")
        order = record.get("attempt_ordinal")
        _require(_strict_int(order, 1), "an attempt ordinal is not a positive integer")
        for_contract = [fact for fact in started
                        if fact.get("contract_id") == record.get("contract_id")]
        _require(order <= len(for_contract)
                 and for_contract[order - 1].get("attempt_id") == attempt_id,
                 "an attempt association does not name the attempt at its ordinal")
        ordinals.setdefault(record["contract_id"], []).append(order)
        finish = [fact for fact in finished if fact.get("attempt_id") == attempt_id]
        _require(len(finish) == 1, "an attempt association has no unique finish fact")
        _require(finish[0].get("execution_status") == record.get("execution_status"),
                 "an attempt association carries another execution status")
    for contract_id, values in ordinals.items():
        _require(sorted(values) == list(range(1, len(values) + 1)),
                 f"the attempt ordinals of {contract_id} are not unique and contiguous")
    unfinished = [record for record in started
                  if record.get("attempt_id") not in {fact.get("attempt_id") for fact in finished}]
    unfinished_id = index.get("unfinished_attempt_id")
    if not unfinished:
        _require(unfinished_id is None, "an unfinished attempt is declared without a real start")
    else:
        _require(len(unfinished) == 1
                 and unfinished[0].get("attempt_id") == unfinished_id,
                 "the unfinished attempt does not match the real trace")


def _verify_verifications(index: dict[str, Any], records: list[dict],
                          snapshots: dict[str, dict[str, Any]]) -> None:
    """Prove every verification snapshot is bound to a real verification fact."""
    verifications = _list(index.get("verifications"), "the verification associations")
    events = _event_by_id(records)
    step_unique = {record["attempt_id"]: record for record in records
                   if record.get("kind") == "verification_recorded"
                   and record.get("scope") == "attempt"}
    run_facts = [record for record in records
                 if record.get("kind") == "verification_recorded"
                 and record.get("scope") == "run"]
    _require(len(verifications) == sum(1 for entry in snapshots.values()
                                       if entry["kind"] in ("step_verification",
                                                            "task_verification")),
             "the verification associations do not cover the verification snapshots")
    used_run = 0
    seen: set[tuple[str, int]] = set()
    seen_paths: set[str] = set()
    observation_ids = _snapshot_values(snapshots, "observation", "observation_id")
    contract_ids = _snapshot_values(snapshots, "contract", "contract_id")
    for entry in verifications:
        record = _mapping(entry, "a verification association")
        kind = record.get("kind")
        _require(kind in ("step_verification", "task_verification"),
                 "a verification association declares an unknown kind")
        _require(record.get("path") in snapshots,
                 "a verification association does not name its snapshot")
        snapshot = snapshots[record["path"]]
        _require(snapshot["kind"] == kind,
                 "a verification association does not name its snapshot")
        _bind_association(record, snapshot, "a verification association")
        _require(record["path"] not in seen_paths,
                 "a verification association repeats a snapshot")
        seen_paths.add(record["path"])
        key = (kind, record.get("ordinal"))
        _require(key not in seen, "a verification association repeats an ordinal")
        seen.add(key)
        status = record.get("status")
        _require(status in ("pass", "fail", "inconclusive"),
                 "a verification association carries no declared status")
        _require(record.get("verifier") in ("web_step_verifier",
                                             "web_task_completion_oracle"),
                 "a verification association carries no declared verifier")
        _require(record.get("observation_id") in observation_ids,
                 "a verification association names no real observation")
        trace_fact = record.get("trace_fact")
        if kind == "step_verification":
            _require(record.get("contract_id") in contract_ids,
                     "a step verification names no real contract")
            if trace_fact == "attempt_verification":
                event = events.get(record.get("event_record_id"))
                _require(event is not None and event.get("kind") == "verification_recorded"
                         and event.get("scope") == "attempt"
                         and event.get("attempt_id") == record.get("attempt_id"),
                         "a step verification names no real attempt verification")
                _require(event.get("verification_status") == status,
                         "a step verification carries another status")
                _require(step_unique.get(record.get("attempt_id")) == event,
                         "a step verification is not the attempt's unique fact")
            else:
                _require(trace_fact is None and record.get("recovery_reverification") is True,
                         "a step verification names no real trace fact and no re-verification")
        else:
            _require(record.get("goal_id") is not None,
                     "a task verification carries no goal")
            if trace_fact == "run_verification":
                _require(used_run < len(run_facts),
                         "a task verification names no real run verification")
                event = run_facts[used_run]
                used_run += 1
                _require(event.get("record_id") == record.get("event_record_id"),
                         "a task verification names another run verification")
                _require(event.get("verification_status") == status,
                         "a task verification carries another status")
            else:
                _require(trace_fact is None, "a task verification declares no real supporting fact")
    _require(used_run == len(run_facts),
             "the real trace carries a run verification with no snapshot")
    referenced = {record.get("attempt_id") for record in verifications
                  if record.get("kind") == "step_verification"
                  and record.get("trace_fact") == "attempt_verification"}
    _require(referenced == set(step_unique),
             "the real trace carries an attempt verification with no snapshot")


def _verify_diagnostics(index: dict[str, Any]) -> None:
    """Prove the host diagnostics are the complete closed structure this recorder writes.

    The boundaries name only the four real adapter entry points, their outcome vocabulary is
    closed, a failure may carry only one of the closed facility codes and a success carries none.
    The host facts keep the LOGIN_RECEIVED waits, the declared count phases and the cleanup facts
    in their own closed vocabularies. No arbitrary exception sentence is reachable from here.
    """
    diagnostics = _mapping(index.get("diagnostics"), "the host diagnostics")
    _require(frozenset(diagnostics) == {"boundaries", "host"},
             "the host diagnostics do not declare their structure")
    for row in _list(diagnostics["boundaries"], "the host boundaries"):
        record = _mapping(row, "a host boundary")
        _require(frozenset(record) == _DIAGNOSTIC_BOUNDARY_KEYS,
                 "a host boundary does not declare its structure")
        _require(type(record["operation"]) is str
                 and record["operation"] in _DIAGNOSTIC_OPERATIONS,
                 "a host boundary names no declared adapter operation")
        _require(type(record["outcome"]) is str
                 and record["outcome"] in _DIAGNOSTIC_OUTCOMES,
                 "a host boundary declares no declared outcome")
        code = record["error_code"]
        _require(code is None
                 or (type(code) is str and code in WEB_BROWSER_ERROR_CODES),
                 "a host boundary carries no permitted facility code")
        _require(_strict_int(record["elapsed_ms"], 0),
                 "a host boundary timing is not a non-negative integer")
        budget = record["budget_ms"]
        _require(budget is None or _strict_int(budget, 1),
                 "a host boundary budget is not a positive integer")
        _require(code is None or record["outcome"] == "failed",
                 "a successful host boundary carries a failure code")
    host = _mapping(diagnostics["host"], "the host diagnostic facts")
    _require(frozenset(host) == {"started", "waits", "counts", "cleanup"},
             "the host diagnostic facts do not declare their structure")
    if host["started"] is not None:
        _host_counts(host["started"])
    for row in _list(host["waits"], "the host waits"):
        record = _mapping(row, "a host wait")
        _require(frozenset(record) == _DIAGNOSTIC_WAIT_KEYS,
                 "a host wait does not declare its structure")
        _require(type(record["phase"]) is str
                 and record["phase"] in _DIAGNOSTIC_WAIT_PHASES,
                 "a host wait names no declared profile phase")
        _require(type(record["received"]) is bool,
                 "a host wait carries no real boolean receipt")
        _require(_strict_int(record["elapsed_ms"], 0),
                 "a host wait timing is not a non-negative integer")
    counts = _mapping(host["counts"], "the host counts")
    _require(frozenset(counts) <= _DIAGNOSTIC_COUNT_PHASES,
             "the host counts name no declared profile phase")
    for value in counts.values():
        _host_counts(value)
    cleanup = _mapping(host["cleanup"], "the host cleanup facts")
    _require(frozenset(cleanup) == _DIAGNOSTIC_CLEANUP_KEYS,
             "the host cleanup facts do not declare their structure")
    adapter_code = cleanup["adapter_error_code"]
    _require(adapter_code is None
             or (type(adapter_code) is str and adapter_code in WEB_BROWSER_ERROR_CODES),
             "the host cleanup carries no permitted adapter code")
    status = cleanup["runtime_cleanup_status"]
    _require(status is None or (type(status) is str and status in CLEANUP_STATUSES),
             "the host cleanup carries no declared Runtime status")
    runtime_code = cleanup["runtime_cleanup_error_code"]
    _require(runtime_code is None
             or (type(runtime_code) is str and runtime_code in _RUNTIME_CLEANUP_CODES),
             "the host cleanup carries no declared Runtime code")


def verify_evidence_index(root: Path) -> dict[str, Any]:
    """Re-read an index and prove every referenced record and association is real.

    The check does not trust the index's own summary: it re-reads the complete snapshot directory,
    proves the inventory is exactly the files on disk, re-hashes every snapshot byte, re-reads the
    trace and Working Memory raw bytes, and re-derives every run/step/contract/attempt/observation
    and terminal association from those files. Every contract, attempt and verification
    association row must match the exact snapshot entry it references, field by field, and that
    entry is already bound to its payload identifiers; the row's event reference is then checked
    against the real trace. The result's step and attempt counts are cross-checked against the
    trace counts and its remaining counters against the published Working Memory. The terminal,
    the result and the published finalization must agree on the primary diagnosis in every source
    mode, while a cleanup error stays a secondary fact. A returned observation must map to its
    published call, a call whose observation never returned keeps its record and has no snapshot,
    and a genuinely pre-start Run may have zero snapshots only when the real trace carries its
    terminal fact.
    """
    root = Path(root)
    index = _read_index(root)
    snapshots = _snapshot_entries(root, index)
    records, _trace, completed = _read_trace(root, index)
    _verify_events(index, records)
    observations = [entry for entry in snapshots.values() if entry["kind"] == "observation"]
    _verify_working_memory(root, index, observations)
    terminal = _verify_terminal(index, completed, records)
    _verify_contracts(index, records, snapshots)
    _verify_attempts(index, records, snapshots)
    _verify_verifications(index, records, snapshots)
    _verify_diagnostics(index)
    if not snapshots:
        _require(not index["contracts"] and not index["attempts"]
                 and not index["verifications"],
                 "an empty snapshot inventory still declares associations")
        _require(terminal["terminal_status"] is not None,
                 "an empty snapshot inventory has no real terminal basis")
    return index
