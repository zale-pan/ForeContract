"""Acceptance tests for the JSONL Episodic Memory store.

Every case drives the real :class:`JsonlEpisodicMemory` against a real :class:`JsonlTraceStore`
in an external scratch directory: sources are genuine tracing records, not stand-ins. Assertions
check mapped values, chain identities, ordering and on-disk bytes; a rejection is checked through
its error class and through the unchanged file.

Storage failures are injected by replacing the module's ``open`` with a controlled stand-in. All
fixtures are synthetic and contain no real credential, path or governance content.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.memory import (
    JsonlEpisodicMemory,
    MemoryCorruptionError,
    MemoryItem,
    MemoryNotFoundError,
    MemorySourceError,
    MemoryStorageError,
    MemoryValidationError,
    ResolvedSource,
    TraceSourceResolver,
    WorkingMemory,
)
from forecontract.tracing import (
    AttemptFinished,
    AttemptStarted,
    ContractCommitted,
    JsonlTraceStore,
    RunFinished,
    RunStarted,
    StepStarted,
    TraceNotFoundError,
    VerificationRecorded,
)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
DECLARED_SCRATCH = Path(os.environ.get("FORECONTRACT_TEST_SCRATCH", ""))
REAL_OPEN = builtins.open
_MARKER = "SYNTHETIC_PRIVATE_VALUE"


def _scratch_base() -> Path:
    """The explicit external test scratch; there is no fallback directory."""

    if not os.environ.get("FORECONTRACT_TEST_SCRATCH"):
        raise RuntimeError("FORECONTRACT_TEST_SCRATCH must be set to an external writable directory")
    candidate = Path(DECLARED_SCRATCH)
    source_root = Path(__file__).resolve().parents[2]
    resolved = candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
    if resolved == source_root or source_root in resolved.parents:
        raise RuntimeError("FORECONTRACT_TEST_SCRATCH must not be the source tree or a subdirectory")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / f".probe-{uuid.uuid4().hex[:8]}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return candidate
    except OSError:
        raise RuntimeError(
            "FORECONTRACT_TEST_SCRATCH must name a writable external directory") from None


@pytest.fixture
def scratch() -> Path:
    base = _scratch_base() / f"memory-{uuid.uuid4().hex[:10]}"
    base.mkdir(parents=True, exist_ok=True)
    return base


# --- real trace fixtures -------------------------------------------------------

def _run_started(run_id: str, sequence: int, record_id: str | None = None) -> RunStarted:
    return RunStarted(schema_version="0.1", record_id=record_id or f"{run_id}-start", run_id=run_id,
                      sequence=sequence, occurred_at=NOW)


def _step(run_id: str, sequence: int, step_id: str = "step-1") -> StepStarted:
    return StepStarted(schema_version="0.1", record_id=f"{run_id}-{step_id}", run_id=run_id,
                       sequence=sequence, occurred_at=NOW, step_id=step_id)


def _contract(run_id: str, sequence: int, contract_id: str = "contract-1",
              step_id: str = "step-1") -> ContractCommitted:
    return ContractCommitted(schema_version="0.1", record_id=f"{run_id}-{contract_id}", run_id=run_id,
                             sequence=sequence, occurred_at=NOW, step_id=step_id,
                             contract_id=contract_id)


def _attempt(run_id: str, sequence: int, attempt_id: str = "attempt-1",
             contract_id: str = "contract-1", step_id: str = "step-1") -> AttemptStarted:
    return AttemptStarted(schema_version="0.1", record_id=f"{run_id}-{attempt_id}", run_id=run_id,
                          sequence=sequence, occurred_at=NOW, step_id=step_id,
                          contract_id=contract_id, attempt_id=attempt_id)


def _finished(run_id: str, sequence: int, status: str = "succeeded", attempt_id: str = "attempt-1",
              contract_id: str = "contract-1", step_id: str = "step-1") -> AttemptFinished:
    return AttemptFinished(schema_version="0.1", record_id=f"{run_id}-{attempt_id}-finished",
                           run_id=run_id, sequence=sequence, occurred_at=NOW, step_id=step_id,
                           contract_id=contract_id, attempt_id=attempt_id, execution_status=status)


def _verification(run_id: str, sequence: int, scope: str = "run", status: str = "pass",
                  record_id: str | None = None) -> VerificationRecorded:
    fields = {
        "schema_version": "0.1",
        "record_id": record_id or f"{run_id}-verification-{sequence}",
        "run_id": run_id,
        "sequence": sequence,
        "occurred_at": NOW,
        "scope": scope,
        "verification_status": status,
    }
    if scope == "attempt":
        fields.update(step_id="step-1", contract_id="contract-1", attempt_id="attempt-1")
    return VerificationRecorded(**fields)


def _terminal(run_id: str, sequence: int, status: str = "failure",
              verification_record_id: str | None = None,
              record_id: str | None = None) -> RunFinished:
    return RunFinished(schema_version="0.1", record_id=record_id or f"{run_id}-terminal",
                       run_id=run_id, sequence=sequence, occurred_at=NOW, terminal_status=status,
                       verification_record_id=verification_record_id)


def build_trace(scratch: Path, records) -> JsonlTraceStore:
    path = scratch / f"trace-{uuid.uuid4().hex[:8]}.jsonl"
    store = JsonlTraceStore(path)
    for record in records:
        store.append(record)
    return store


def episodic(scratch: Path, trace: JsonlTraceStore) -> JsonlEpisodicMemory:
    return JsonlEpisodicMemory(scratch / f"memory-{uuid.uuid4().hex[:8]}.jsonl",
                               source_resolver=TraceSourceResolver(trace))


def seed_bytes(store: JsonlEpisodicMemory, payload: bytes) -> None:
    """Write raw seed bytes at the configured Memory path, creating the file the store creates."""
    store.path.write_bytes(payload)


def append_raw(store: JsonlEpisodicMemory, text: str) -> None:
    """Append raw text at the configured Memory path, creating it when absent."""
    with open(store.path, "a", encoding="utf-8") as handle:
        handle.write(text)


def item(memory_id: str, run_id: str, source_record_id: str, result: str, *,
         level: str = "run", created_at: datetime = NOW, supersedes: str | None = None) -> MemoryItem:
    return MemoryItem(schema_version="0.1", memory_id=memory_id, run_id=run_id, source_level=level,
                      source_record_id=source_record_id, result=result, created_at=created_at,
                      supersedes_id=supersedes)


# --- the closed record ---------------------------------------------------------

def test_the_record_declares_exactly_the_eight_agreed_fields() -> None:
    assert list(MemoryItem.model_fields) == [
        "schema_version", "memory_id", "run_id", "source_level", "source_record_id", "result",
        "created_at", "supersedes_id",
    ]


@pytest.mark.parametrize("field", [
    "summary", "metadata", "body", "observation", "actual_outcome", "url", "error_message",
    "reasoning", "payload", "confidence",
])
def test_a_raw_payload_field_is_rejected(field: str) -> None:
    fields = {
        "schema_version": "0.1", "memory_id": "m1", "run_id": "run-1", "source_level": "run",
        "source_record_id": "r1", "result": "unknown", "created_at": NOW, field: _MARKER,
    }
    with pytest.raises(MemoryValidationError) as info:
        MemoryItem(**fields)
    message = str(info.value)
    assert _MARKER not in message
    assert "unexpected field" in message


@pytest.mark.parametrize("field,value", [
    ("schema_version", "0.2"),
    ("schema_version", "1"),
    ("memory_id", 1),
    ("memory_id", True),
    ("memory_id", ""),
    ("memory_id", "   "),
    ("run_id", 7),
    ("source_level", "step"),
    ("source_level", "ATTEMPT"),
    ("source_record_id", None),
    ("result", "pass"),
    ("result", "succeeded"),
    ("result", "maybe"),
    ("created_at", "not-a-time"),
    ("created_at", "2026-10-08T12:00:00"),
    ("supersedes_id", 5),
    ("supersedes_id", "   "),
])
def test_a_malformed_field_is_rejected(field: str, value: object) -> None:
    fields = {
        "schema_version": "0.1", "memory_id": "m1", "run_id": "run-1", "source_level": "run",
        "source_record_id": "r1", "result": "unknown", "created_at": NOW,
    }
    fields[field] = value
    with pytest.raises(MemoryValidationError):
        MemoryItem(**fields)


def test_a_record_is_frozen() -> None:
    record = item("m1", "run-1", "r1", "unknown")
    with pytest.raises(ValidationError):
        record.memory_id = "m2"  # type: ignore[misc]


def test_a_record_round_trips_through_json_text() -> None:
    record = item("m1", "run-1", "r1", "unknown", created_at=NOW + timedelta(hours=2))
    reloaded = MemoryItem.model_validate_json(record.model_dump_json())
    assert reloaded == record
    assert reloaded.created_at == NOW + timedelta(hours=2)
    assert reloaded.supersedes_id is None


# --- ordinary copy/construct ingress ------------------------------------------

@pytest.mark.parametrize("key", [
    "__metadata__", "__pydantic_synthetic_payload", 17, "__pydantic_fields_set__",
])
def test_an_injected_storage_key_is_rejected_before_any_write(
    scratch: Path, key: object,
) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1)])
    store = episodic(scratch, trace)
    forged = item("m1", "run-1", "run-1-start", "unknown").model_copy(update={key: _MARKER})
    assert key in forged.__dict__
    with pytest.raises(MemoryValidationError) as info:
        store.append(forged)
    assert _MARKER not in str(info.value)
    assert not store.path.exists()


@pytest.mark.parametrize("forgery", [
    {"schema_version": "9.9"},
    {"result": "succeeded"},
    {"source_level": "step"},
    {"memory_id": 5},
])
def test_an_invalid_copied_record_is_rejected_before_any_write(
    scratch: Path, forgery: dict,
) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1)])
    store = episodic(scratch, trace)
    forged = item("m1", "run-1", "run-1-start", "unknown").model_copy(update=forgery)
    with pytest.raises(MemoryValidationError):
        store.append(forged)
    assert not store.path.exists()


def test_a_validated_instance_is_accepted(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert store.get("m1").memory_id == "m1"
    assert store.path.read_bytes().endswith(b"\n")


def test_a_foreign_object_is_rejected(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    with pytest.raises(MemoryValidationError):
        store.append({"memory_id": "m1"})
    with pytest.raises(MemoryValidationError):
        store.append(WorkingMemory())


# --- source mapping: four allowed classes, six results ------------------------

def test_a_run_start_source_maps_to_unknown(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1)])
    resolver = TraceSourceResolver(trace)
    resolved = resolver.resolve("run-1", "run-1-start")
    assert resolved.source_level == "run"
    assert resolved.result == "unknown"
    assert (resolved.step_id, resolved.contract_id, resolved.attempt_id) == (None, None, None)
    assert resolved.record_id == "run-1-start"


@pytest.mark.parametrize("status,expected", [
    ("success", "success"),
    ("failure", "failure"),
    ("cancelled", "cancelled"),
    ("timeout", "timeout"),
    ("aborted", "aborted"),
])
def test_a_run_terminal_source_maps_exactly(scratch: Path, status: str, expected: str) -> None:
    reference = None
    if status == "success":
        reference = "run-1-verification-1"
    trace = build_trace(scratch, [
        _run_started("run-1", 0),
        _verification("run-1", 1, scope="run", status="pass"),
        _terminal("run-1", 2, status=status, verification_record_id=reference),
    ])
    resolver = TraceSourceResolver(trace)
    assert resolver.resolve("run-1", "run-1-terminal").result == expected


def test_an_attempt_start_source_maps_to_unknown(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _step("run-1", 1), _contract("run-1", 2), _attempt("run-1", 3),
    ])
    resolved = TraceSourceResolver(trace).resolve("run-1", "run-1-attempt-1")
    assert resolved.source_level == "attempt"
    assert resolved.result == "unknown"
    assert (resolved.step_id, resolved.contract_id, resolved.attempt_id) == (
        "step-1", "contract-1", "attempt-1",
    )


@pytest.mark.parametrize("status,expected", [
    ("succeeded", "success"),
    ("failed", "failure"),
    ("unknown", "unknown"),
])
def test_an_attempt_completion_source_maps_exactly(scratch: Path, status: str, expected: str) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _step("run-1", 1), _contract("run-1", 2), _attempt("run-1", 3),
        _finished("run-1", 4, status=status),
    ])
    resolved = TraceSourceResolver(trace).resolve("run-1", "run-1-attempt-1-finished")
    assert resolved.result == expected
    assert resolved.source_level == "attempt"


@pytest.mark.parametrize("kind", ["step", "contract", "verification"])
def test_a_non_source_record_kind_is_rejected(scratch: Path, kind: str) -> None:
    records = [_run_started("run-1", 0), _step("run-1", 1), _contract("run-1", 2)]
    if kind == "verification":
        records.append(_verification("run-1", 3, scope="run", status="pass"))
    trace = build_trace(scratch, records)
    identifier = {"step": "run-1-step-1", "contract": "run-1-contract-1",
                  "verification": "run-1-verification-3"}[kind]
    with pytest.raises(MemorySourceError) as info:
        TraceSourceResolver(trace).resolve("run-1", identifier)
    assert info.value.code == "memory_source"


def test_a_verification_pass_record_cannot_be_a_memory_source(scratch: Path) -> None:
    """PASS is a verification verdict, never a Run success and never a Memory source."""
    trace = build_trace(scratch, [_run_started("run-1", 0),
                                  _verification("run-1", 1, scope="run", status="pass")])
    with pytest.raises(MemorySourceError):
        TraceSourceResolver(trace).resolve("run-1", "run-1-verification-1")
    store = episodic(scratch, trace)
    with pytest.raises(MemorySourceError):
        store.append(item("m1", "run-1", "run-1-verification-1", "success"))


def test_a_missing_source_record_is_a_source_error(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    with pytest.raises(MemorySourceError) as info:
        TraceSourceResolver(trace).resolve("run-1", "record-absent")
    assert isinstance(info.value.__cause__, TraceNotFoundError)


def test_a_missing_run_is_a_source_error(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    with pytest.raises(MemorySourceError):
        TraceSourceResolver(trace).resolve("run-absent", "run-1-start")


def test_a_corrupt_trace_is_a_source_error(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    with open(trace.path, "a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(MemorySourceError):
        TraceSourceResolver(trace).resolve("run-1", "run-1-start")


# --- request identity ----------------------------------------------------------

def test_a_stored_version_must_match_its_source_result(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1, status="failure")])
    store = episodic(scratch, trace)
    with pytest.raises(MemorySourceError):
        store.append(item("m1", "run-1", "run-1-terminal", "success"))


def test_a_stored_version_must_match_its_source_level(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _step("run-1", 1), _contract("run-1", 2), _attempt("run-1", 3),
    ])
    store = episodic(scratch, trace)
    with pytest.raises(MemorySourceError):
        store.append(item("m1", "run-1", "run-1-attempt-1", "unknown", level="run"))


def test_a_record_for_another_run_cannot_use_this_runs_source(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _run_started("run-2", 0)])
    store = episodic(scratch, trace)
    with pytest.raises(MemorySourceError):
        store.append(item("m1", "run-2", "run-1-start", "unknown"))


def test_a_stored_record_whose_source_disappears_fails_explicitly(scratch: Path) -> None:
    trace_path = scratch / "trace-gone.jsonl"
    store_trace = JsonlTraceStore(trace_path)
    store_trace.append(_run_started("run-1", 0))
    memory = episodic(scratch, store_trace)
    memory.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert memory.get("m1").memory_id == "m1"
    trace_path.unlink()
    with pytest.raises(MemorySourceError):
        memory.get("m1")
    with pytest.raises(MemorySourceError):
        memory.query()


def test_a_stored_record_whose_source_is_corrupt_fails_explicitly(scratch: Path) -> None:
    trace_path = scratch / "trace-broken.jsonl"
    store_trace = JsonlTraceStore(trace_path)
    store_trace.append(_run_started("run-1", 0))
    memory = episodic(scratch, store_trace)
    memory.append(item("m1", "run-1", "run-1-start", "unknown"))
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write("{broken\n")
    with pytest.raises(MemorySourceError) as info:
        memory.get("m1")
    assert info.value.line_number == 1


# --- chain rules ---------------------------------------------------------------

def test_one_experience_accepts_exactly_one_root(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    with pytest.raises(MemoryValidationError):
        store.append(item("m2", "run-1", "run-1-start", "unknown"))


def test_two_runs_have_independent_chains(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _terminal("run-1", 1),
        _run_started("run-2", 0), _terminal("run-2", 1, status="cancelled"),
    ])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    store.append(item("m2", "run-2", "run-2-start", "unknown"))
    heads = store.query()
    assert [(entry.memory_id, entry.run_id) for entry in heads] == [("m1", "run-1"), ("m2", "run-2")]


def test_two_attempts_are_always_two_experiences(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _step("run-1", 1), _contract("run-1", 2),
        _attempt("run-1", 3, attempt_id="attempt-1"),
        _finished("run-1", 4, status="failed", attempt_id="attempt-1"),
        _attempt("run-1", 5, attempt_id="attempt-2"),
        _finished("run-1", 6, status="succeeded", attempt_id="attempt-2"),
    ])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-attempt-1-finished", "failure", level="attempt"))
    store.append(item("m2", "run-1", "run-1-attempt-2-finished", "success", level="attempt"))
    heads = store.query()
    assert [(entry.memory_id, entry.result) for entry in heads] == [("m1", "failure"), ("m2", "success")]
    assert store.get("m1").result == "failure"


def test_an_attempt_success_cannot_correct_another_attempt(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _step("run-1", 1), _contract("run-1", 2),
        _attempt("run-1", 3, attempt_id="attempt-1"),
        _finished("run-1", 4, status="failed", attempt_id="attempt-1"),
        _attempt("run-1", 5, attempt_id="attempt-2"),
        _finished("run-1", 6, status="succeeded", attempt_id="attempt-2"),
    ])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-attempt-1-finished", "failure", level="attempt"))
    with pytest.raises(MemoryValidationError):
        store.append(item("m2", "run-1", "run-1-attempt-2-finished", "success", level="attempt",
                          supersedes="m1"))


def test_a_new_version_may_supersede_the_current_head(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _verification("run-1", 1, scope="run", status="pass"),
        _terminal("run-1", 2, status="success", verification_record_id="run-1-verification-1"),
    ])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    store.append(item("m2", "run-1", "run-1-terminal", "success", supersedes="m1"))
    assert [entry.memory_id for entry in store.query()] == ["m2"]
    assert store.get("m1").result == "unknown"


def test_an_explicit_correction_replaces_the_head_but_keeps_the_old_version(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1, status="failure")])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    store.append(item("m2", "run-1", "run-1-terminal", "failure", supersedes="m1"))
    heads = store.query()
    assert [(entry.memory_id, entry.result) for entry in heads] == [("m2", "failure")]
    assert store.get("m1").result == "unknown"
    assert store.get("m2").supersedes_id == "m1"


def test_superseding_a_non_current_head_is_rejected(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _terminal("run-1", 1)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    store.append(item("m2", "run-1", "run-1-terminal", "failure", supersedes="m1"))
    with pytest.raises(MemoryValidationError):
        store.append(item("m3", "run-1", "run-1-terminal", "failure", supersedes="m1"))


def test_a_version_cannot_supersede_itself(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    with pytest.raises(MemoryValidationError):
        store.append(item("m1", "run-1", "run-1-start", "unknown", supersedes="m1"))


def test_superseding_a_missing_version_is_rejected(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    with pytest.raises(MemoryValidationError):
        store.append(item("m1", "run-1", "run-1-start", "unknown", supersedes="absent"))


def test_superseding_a_version_in_another_experience_is_rejected(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _run_started("run-2", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    with pytest.raises(MemoryValidationError):
        store.append(item("m2", "run-2", "run-2-start", "unknown", supersedes="m1"))


def test_a_duplicate_memory_identifier_is_rejected(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _run_started("run-2", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    with pytest.raises(MemoryValidationError):
        store.append(item("m1", "run-2", "run-2-start", "unknown"))


# --- reads and queries ---------------------------------------------------------

def test_a_missing_file_queries_empty_and_get_is_not_found(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    assert store.query() == ()
    with pytest.raises(MemoryNotFoundError):
        store.get("m1")
    assert not store.path.exists()


def test_an_empty_file_queries_empty(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, b"")
    assert store.query() == ()
    with pytest.raises(MemoryNotFoundError):
        store.get("m1")


def test_filters_are_combined_with_and(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _terminal("run-1", 1, status="failure"),
        _run_started("run-2", 0), _step("run-2", 1), _contract("run-2", 2),
        _attempt("run-2", 3), _finished("run-2", 4, status="failed"),
    ])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-terminal", "failure"))
    store.append(item("m2", "run-2", "run-2-attempt-1-finished", "failure", level="attempt"))
    assert [entry.memory_id for entry in store.query(result="failure")] == ["m1", "m2"]
    assert [entry.memory_id for entry in store.query(run_id="run-2")] == ["m2"]
    assert [entry.memory_id for entry in store.query(source_level="attempt")] == ["m2"]
    assert [entry.memory_id for entry in store.query(run_id="run-2", source_level="attempt",
                                                    result="failure")] == ["m2"]
    assert store.query(run_id="run-2", result="success") == ()
    assert store.query(run_id="run-absent") == ()


def test_query_returns_heads_and_orders_by_utc_then_identifier(scratch: Path) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _terminal("run-1", 1),
        _run_started("run-2", 0), _terminal("run-2", 1),
        _run_started("run-3", 0), _terminal("run-3", 1),
    ])
    store = episodic(scratch, trace)
    later = NOW + timedelta(hours=1)
    store.append(item("m2", "run-2", "run-2-start", "unknown", created_at=NOW))
    store.append(item("m1", "run-1", "run-1-start", "unknown", created_at=NOW))
    store.append(item("m3", "run-3", "run-3-start", "unknown", created_at=later))
    assert [entry.memory_id for entry in store.query()] == ["m1", "m2", "m3"]


def test_two_timezones_are_ordered_by_their_utc_instant(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _run_started("run-2", 0)])
    store = episodic(scratch, trace)
    east = timezone(timedelta(hours=8))
    # 20:00+08:00 is 12:00 UTC, which is earlier than 13:00 UTC, so the eastern record sorts
    # first even though its local clock reading is the larger one.
    store.append(item("m-eastern", "run-1", "run-1-start", "unknown",
                      created_at=datetime(2026, 10, 8, 20, 0, tzinfo=east)))
    store.append(item("m-utc", "run-2", "run-2-start", "unknown",
                      created_at=datetime(2026, 10, 8, 13, 0, tzinfo=timezone.utc)))
    assert [entry.memory_id for entry in store.query()] == ["m-eastern", "m-utc"]


@pytest.mark.parametrize("limit,expected", [(None, 3), (0, 0), (1, 1), (2, 2), (3, 3), (9, 3)])
def test_the_limit_is_applied_after_ordering(scratch: Path, limit: object, expected: int) -> None:
    trace = build_trace(scratch, [
        _run_started("run-1", 0), _run_started("run-2", 0), _run_started("run-3", 0),
    ])
    store = episodic(scratch, trace)
    for index in (1, 2, 3):
        store.append(item(f"m{index}", f"run-{index}", f"run-{index}-start", "unknown"))
    assert len(store.query(limit=limit)) == expected


@pytest.mark.parametrize("bad_limit", [True, False, -1, 1.5, "2", -0.5])
def test_a_malformed_limit_is_rejected(scratch: Path, bad_limit: object) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    with pytest.raises(MemoryValidationError):
        store.query(limit=bad_limit)


@pytest.mark.parametrize("kwargs", [
    {"run_id": 1}, {"run_id": ""}, {"source_level": "step"}, {"result": "pass"},
    {"result": "maybe"}, {"result": 3},
])
def test_a_malformed_filter_is_rejected(scratch: Path, kwargs: dict) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    with pytest.raises(MemoryValidationError):
        store.query(**kwargs)


def test_query_returns_detached_records(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    first = store.query()
    second = store.query()
    assert first == second
    assert first[0] is not second[0]
    assert store.get("m1") is not store.get("m1")


def test_a_limit_of_zero_still_validates_the_whole_file(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert store.query(limit=0) == ()
    with open(store.path, "a", encoding="utf-8") as handle:
        handle.write("{broken\n")
    with pytest.raises(MemoryCorruptionError):
        store.query(limit=0)


def test_a_limit_of_zero_still_validates_every_source(scratch: Path) -> None:
    trace_path = scratch / "trace-limit0.jsonl"
    store_trace = JsonlTraceStore(trace_path)
    store_trace.append(_run_started("run-1", 0))
    memory = episodic(scratch, store_trace)
    memory.append(item("m1", "run-1", "run-1-start", "unknown"))
    trace_path.unlink()
    with pytest.raises(MemorySourceError):
        memory.query(limit=0)


# --- corruption ----------------------------------------------------------------

@pytest.mark.parametrize("payload,line", [
    (b'{"schema_version":"0.1"}\n', 1),
    (b"".join([
        b'{"schema_version":"0.2","memory_id":"m","run_id":"r","source_level":"run",',
        b'"source_record_id":"s","result":"unknown","created_at":"2026-10-08T12:00:00Z",',
        b'"supersedes_id":null}\n',
    ]), 1),
    (b"".join([
        b'{"schema_version":"0.1","memory_id":"m","run_id":"r","source_level":"run",',
        b'"source_record_id":"s","result":"pass","created_at":"2026-10-08T12:00:00Z"}\n',
    ]), 1),
    (b'{"not":"a record"}\n', 1),
])
def test_a_line_that_violates_the_schema_is_corruption(scratch: Path, payload: bytes, line: int) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, payload)
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == line


def test_a_blank_line_is_corruption(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    append_raw(store, "\n")
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 2


def test_a_duplicate_json_key_is_corruption(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, b"".join([
        b'{"schema_version":"0.1","memory_id":"m","memory_id":"n","run_id":"run-1",',
        b'"source_level":"run","source_record_id":"run-1-start","result":"unknown",',
        b'"created_at":"2026-10-08T12:00:00Z","supersedes_id":null}\n',
    ]))
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 1


def test_a_non_finite_number_is_corruption(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, b"".join([
        b'{"schema_version":"0.1","memory_id":"m","run_id":"run-1","source_level":"run",',
        b'"source_record_id":"run-1-start","result":"unknown","created_at":NaN,',
        b'"supersedes_id":null}\n',
    ]))
    with pytest.raises(MemoryCorruptionError):
        store.query()


def test_a_truncated_tail_is_corruption(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    with open(store.path, "a", encoding="utf-8") as handle:
        handle.write('{"schema_version":"0.1","memory_id":"m2"')
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 2


def test_a_missing_final_newline_is_corruption(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    seed_bytes(store, store.path.read_text(encoding="utf-8").rstrip("\n").encode())
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 1


def test_a_bad_middle_line_cannot_hide_behind_a_good_prefix(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _run_started("run-2", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    append_raw(store, "{broken\n")
    with pytest.raises(MemoryCorruptionError) as info:
        store.get("m1")
    assert info.value.line_number == 2


def test_an_oversized_integer_is_a_bounded_located_corruption(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, (
        '{"schema_version":"0.1","memory_id":"m","run_id":"run-1","source_level":"run",'
        '"source_record_id":"run-1-start","result":"unknown","created_at":'
        + "9" * 5000 + ',"supersedes_id":null}\n'
    ).encode())
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 1
    assert "9" * 20 not in str(info.value)


def test_an_invalid_utf8_file_is_corruption(scratch: Path) -> None:
    """F03: a decode failure is located at its physical 1-based line, not left unlocated."""
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, b"\xff\xfe\x00\x00")
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 1


def test_an_invalid_utf8_byte_on_the_first_line_is_line_one(scratch: Path) -> None:
    """F03: the location comes from the decode offset, so the first line reports line one."""
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    valid = store.path.read_bytes()
    seed_bytes(store, b"\xff" + valid)
    with pytest.raises(MemoryCorruptionError) as info:
        store.query(limit=0)
    assert info.value.line_number == 1


def test_an_invalid_utf8_byte_on_a_later_line_is_located_there(scratch: Path) -> None:
    """F03: a bad byte after a complete first line is located on the second line."""
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    valid = store.path.read_bytes()
    seed_bytes(store, valid + b"\xff\n")
    with pytest.raises(MemoryCorruptionError) as info:
        store.query(limit=0)
    assert info.value.line_number == 2
    with pytest.raises(MemoryCorruptionError) as get_info:
        store.get("m1")
    assert get_info.value.line_number == 2


def test_a_trailing_invalid_utf8_byte_is_located_on_its_own_line(scratch: Path) -> None:
    """F03: a bad byte after two complete lines is located on the third line."""
    trace = build_trace(scratch, [_run_started("run-1", 0), _run_started("run-2", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    store.append(item("m2", "run-2", "run-2-start", "unknown"))
    valid = store.path.read_bytes()
    assert valid.count(b"\n") == 2
    seed_bytes(store, valid + b"\xff")
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 3


def test_the_utf8_error_never_echoes_the_payload_or_the_path(scratch: Path) -> None:
    """F03: the located decode message names a line number and nothing else."""
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, b'{"memory_id":"SYNTHETIC_PRIVATE_VALUE"}\n\xff\n')
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    message = str(info.value)
    assert _MARKER not in message and "SYNTHETIC" not in message
    assert str(store.path) not in message
    assert info.value.line_number == 2


# --- injected storage failures --------------------------------------------------

class _StandInHandle:
    """A file stand-in that supports the context-manager protocol and delegates by default."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def write(self, text):
        return self._handle.write(text)

    def flush(self):
        return self._handle.flush()

    def close(self):
        return self._handle.close()


def test_an_open_failure_is_a_storage_error(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)

    def refuse(*args, **kwargs):
        raise PermissionError("synthetic refusal")

    monkeypatch.setattr(builtins, "open", refuse, raising=True)
    with pytest.raises(MemoryStorageError) as info:
        store.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert isinstance(info.value.__cause__, PermissionError)
    assert "synthetic refusal" not in str(info.value)
    assert not store.path.exists()


def test_a_write_failure_leaves_the_file_unchanged(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0), _run_started("run-2", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    before = store.path.read_bytes()

    class FailingWrite(_StandInHandle):
        def write(self, text):
            raise OSError("synthetic write refusal")

    monkeypatch.setattr(builtins, "open",
                        lambda path, *a, **k: FailingWrite(REAL_OPEN(path, *a, **k)), raising=True)
    with pytest.raises(MemoryStorageError) as info:
        store.append(item("m2", "run-2", "run-2-start", "unknown"))
    monkeypatch.undo()
    assert isinstance(info.value.__cause__, OSError)
    assert store.path.read_bytes() == before


def test_a_flush_failure_is_a_storage_error(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)

    class FailingFlush(_StandInHandle):
        def flush(self):
            raise OSError("synthetic flush refusal")

    monkeypatch.setattr(builtins, "open",
                        lambda path, *a, **k: FailingFlush(REAL_OPEN(path, *a, **k)), raising=True)
    with pytest.raises(MemoryStorageError) as info:
        store.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert isinstance(info.value.__cause__, OSError)


def test_a_close_failure_is_a_storage_error(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)

    class FailingClose(_StandInHandle):
        def close(self):
            self._handle.close()
            raise OSError("synthetic close refusal")

    monkeypatch.setattr(builtins, "open",
                        lambda path, *a, **k: FailingClose(REAL_OPEN(path, *a, **k)), raising=True)
    with pytest.raises(MemoryStorageError) as info:
        store.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert isinstance(info.value.__cause__, OSError)


@pytest.mark.parametrize("acknowledged", [None, True, False, 3.5, "12", -1, 10**9])
def test_a_bad_write_acknowledgement_is_a_storage_error(
    scratch: Path, monkeypatch: pytest.MonkeyPatch, acknowledged: object,
) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)

    class BadAck(_StandInHandle):
        def write(self, text):
            self._handle.write(text)
            return acknowledged

    monkeypatch.setattr(builtins, "open",
                        lambda path, *a, **k: BadAck(REAL_OPEN(path, *a, **k)), raising=True)
    with pytest.raises(MemoryStorageError) as info:
        store.append(item("m1", "run-1", "run-1-start", "unknown"))
    monkeypatch.undo()
    assert "partly written" in str(info.value)
    assert str(acknowledged) not in str(info.value)


def test_a_short_write_is_detected_and_its_tail_is_reported(scratch: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)

    class ShortWriter(_StandInHandle):
        def write(self, text):
            half = max(1, len(text) // 3)
            self._handle.write(text[:half])
            return half

    monkeypatch.setattr(builtins, "open",
                        lambda path, *a, **k: ShortWriter(REAL_OPEN(path, *a, **k)), raising=True)
    with pytest.raises(MemoryStorageError):
        store.append(item("m1", "run-1", "run-1-start", "unknown"))
    monkeypatch.undo()
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert info.value.line_number == 1


def test_a_complete_write_is_still_accepted(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert store.path.read_bytes().endswith(b"\n")
    assert len(store.path.read_text(encoding="utf-8").splitlines()) == 1


def test_an_unreadable_path_is_a_storage_error(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = JsonlEpisodicMemory(scratch, source_resolver=TraceSourceResolver(trace))
    with pytest.raises(MemoryStorageError) as info:
        store.query()
    assert isinstance(info.value.__cause__, OSError)


def test_an_unusable_path_is_a_storage_error(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    with pytest.raises(MemoryStorageError):
        JsonlEpisodicMemory(7, source_resolver=TraceSourceResolver(trace))


def test_a_missing_resolver_is_rejected(scratch: Path) -> None:
    with pytest.raises(MemoryValidationError):
        JsonlEpisodicMemory(scratch / "memory.jsonl", source_resolver=object())


# --- bounded, value-free public surface ----------------------------------------

def test_the_stored_line_carries_only_the_closed_fields(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    store.append(item("m1", "run-1", "run-1-start", "unknown"))
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version", "memory_id", "run_id", "source_level", "source_record_id", "result",
        "created_at", "supersedes_id",
    }
    for token in ("summary", "metadata", "observation", "actual_outcome", "url", "traceback"):
        assert token not in store.path.read_text(encoding="utf-8")


def test_a_rejected_record_never_reaches_the_file_or_a_message(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    forged = item("m1", "run-1", "run-1-start", "unknown").model_copy(
        update={f"__{_MARKER}__": _MARKER}
    )
    with pytest.raises(MemoryValidationError) as info:
        store.append(forged)
    assert _MARKER not in str(info.value)
    assert not store.path.exists()


def test_a_storage_error_message_does_not_include_the_path(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    missing = scratch / "absent-directory" / "memory.jsonl"
    store = JsonlEpisodicMemory(missing, source_resolver=TraceSourceResolver(trace))
    with pytest.raises(MemoryStorageError) as info:
        store.append(item("m1", "run-1", "run-1-start", "unknown"))
    assert str(missing) not in str(info.value)


def test_a_source_error_keeps_its_cause_separately_from_a_memory_io_cause(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    resolver = TraceSourceResolver(trace)
    with pytest.raises(MemorySourceError) as info:
        resolver.resolve("run-1", "absent")
    assert isinstance(info.value.__cause__, TraceNotFoundError)
    assert info.value.code == "memory_source"


def test_the_corruption_message_never_transcribes_the_line(scratch: Path) -> None:
    trace = build_trace(scratch, [_run_started("run-1", 0)])
    store = episodic(scratch, trace)
    seed_bytes(store, (
        '{"schema_version":"0.1","memory_id":"m","run_id":"run-1","source_level":"run",'
        '"source_record_id":"run-1-start","result":"unknown","created_at":"2026-10-08T12:00:00Z",'
        f'"supersedes_id":null,"unknown_field":"{_MARKER}"}}\n'
    ).encode())
    with pytest.raises(MemoryCorruptionError) as info:
        store.query()
    assert _MARKER not in str(info.value)
    assert info.value.line_number == 1


# --- round-002 correction: F01 bounded diagnostics, F02 exact builtins, F04 ingress ---

_INGRESS_BASES = (
    ("MemoryItem", MemoryItem, None),
    ("ResolvedSource", ResolvedSource, {
        "schema_version": "0.1", "run_id": "run-1", "record_id": "record-1",
        "source_level": "run", "result": "unknown",
    }),
)


def _ingress_base(payload: dict | None) -> dict:
    if payload is not None:
        return dict(payload)
    return {
        "schema_version": "0.1", "memory_id": "m1", "run_id": "run-1", "source_level": "run",
        "source_record_id": "run-1-start", "result": "unknown", "created_at": NOW,
        "supersedes_id": None,
    }


@pytest.mark.parametrize("model_name,model,payload", _INGRESS_BASES,
                         ids=[case[0] for case in _INGRESS_BASES])
@pytest.mark.parametrize("route", ["constructor", "model_validate", "model_validate_json"])
def test_an_unknown_key_is_rejected_with_a_bounded_value_free_message(
    model_name: str, model: type, payload: dict | None, route: str,
) -> None:
    """F01: no public ingress echoes an unknown key, a value or a path, and the text is short."""
    long_key = _MARKER + "X" * 2000
    full = _ingress_base(payload)
    full[long_key] = _MARKER + "_VALUE"
    if route == "constructor":
        action = lambda m=model, p=full: m(**p)
    elif route == "model_validate":
        action = lambda m=model, p=full: m.model_validate(p)
    else:
        action = lambda m=model, p=full: m.model_validate_json(json.dumps(p, default=str))
    with pytest.raises(MemoryValidationError) as info:
        action()
    message = str(info.value)
    assert _MARKER not in message
    assert long_key not in message
    assert len(message) < 400
    assert "unexpected field" in message or "declared field" in message
    assert model_name in (model.__name__,)


@pytest.mark.parametrize("model_name,model,payload", _INGRESS_BASES,
                         ids=[case[0] for case in _INGRESS_BASES])
def test_a_declared_field_location_is_still_reported(
    model_name: str, model: type, payload: dict | None,
) -> None:
    """F01 control: a declared field keeps its name in the bounded diagnostic."""
    full = _ingress_base(payload)
    full["schema_version"] = "9.9"
    with pytest.raises(MemoryValidationError) as info:
        model(**full)
    message = str(info.value)
    assert "schema_version" in message
    assert "9.9" not in message
    assert len(message) < 400
    assert model_name in (model.__name__,)


@pytest.mark.parametrize("value", ["success", "failure", "unknown", "cancelled", "timeout", "aborted"])
def test_a_valid_result_enum_still_constructs(value: str) -> None:
    """F01/F04 control: every closed result value still builds a valid record."""
    record = MemoryItem(schema_version="0.1", memory_id="m1", run_id="run-1", source_level="run",
                        source_record_id="run-1-start", result=value, created_at=NOW)
    assert record.result == value
    assert MemoryItem.model_validate(record) == record


@pytest.mark.parametrize("key", [
    "__metadata__", "__pydantic_fields_set__", "__pydantic_extra__", "__pydantic_private__", 17,
])
def test_an_undeclared_stored_key_is_rejected_through_public_instance_ingress(key: object) -> None:
    """F04: a copied record with an undeclared raw key is refused by model_validate too."""
    forged = item("m1", "run-1", "run-1-start", "unknown").model_copy(update={key: _MARKER})
    assert key in forged.__dict__
    with pytest.raises(MemoryValidationError) as info:
        MemoryItem.model_validate(forged)
    assert _MARKER not in str(info.value)


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"result": "succeeded"},
    {"source_level": "step"},
    {"run_id": 17},
    {"memory_id": ""},
])
def test_a_copied_invalid_field_is_rejected_through_public_instance_ingress(mutation: dict) -> None:
    """F04: a copied record whose stored field is invalid is refused, never half-accepted."""
    forged = item("m1", "run-1", "run-1-start", "unknown").model_copy(update=mutation)
    with pytest.raises(MemoryValidationError):
        MemoryItem.model_validate(forged)


@pytest.mark.parametrize("mutation", [
    {"schema_version": "9.9"},
    {"result": "succeeded"},
    {"source_level": "step"},
    {"run_id": 17},
])
def test_a_constructed_invalid_field_is_rejected_through_public_instance_ingress(
    mutation: dict,
) -> None:
    """F04: a constructed record whose stored field is invalid is refused as well."""
    fields = dict(item("m1", "run-1", "run-1-start", "unknown").__dict__)
    fields.update(mutation)
    forged = MemoryItem.model_construct(**fields)
    with pytest.raises(MemoryValidationError):
        MemoryItem.model_validate(forged)


def test_a_constructed_extra_input_is_discarded_and_never_inferred() -> None:
    """F04: pydantic discards an unknown construct argument, so there is nothing to infer.

    The guidance is explicit that a discarded ``model_construct`` extra must not be inferred
    from a valid remaining state, so this records the real behaviour: the extra never reaches
    storage, the record stays valid, and nothing about it can be read back.
    """
    fields = dict(item("m1", "run-1", "run-1-start", "unknown").__dict__)
    fields[_MARKER] = _MARKER + "_VALUE"
    forged = MemoryItem.model_construct(**fields)
    assert _MARKER not in forged.__dict__
    validated = MemoryItem.model_validate(forged)
    assert validated.memory_id == "m1"
    assert _MARKER not in validated.model_dump_json()


def test_a_valid_instance_passes_public_instance_ingress_unchanged() -> None:
    """F04 control: an ordinary valid instance is accepted and yields equal content."""
    record = item("m1", "run-1", "run-1-start", "unknown")
    validated = MemoryItem.model_validate(record)
    assert validated == record
    assert validated.memory_id == record.memory_id
    assert validated.created_at == record.created_at


def test_a_valid_mapping_and_json_still_load() -> None:
    """F04 control: the mapping and JSON ingress routes still work for valid content."""
    record = item("m1", "run-1", "run-1-start", "unknown")
    from_mapping = MemoryItem.model_validate(record.model_dump())
    from_json = MemoryItem.model_validate_json(record.model_dump_json())
    assert from_mapping == record == from_json


def test_an_ordinary_copied_instance_is_still_accepted() -> None:
    """F04 control: model_copy without a forgery is still a valid record."""
    copy = item("m1", "run-1", "run-1-start", "unknown").model_copy()
    assert MemoryItem.model_validate(copy) == copy


# --- round-003 correction: F01 complete public parser boundaries ---------------

_INGRESS_MODELS = (
    ("MemoryItem", MemoryItem, MemoryItem(
        schema_version="0.1", memory_id="m1", run_id="run-1", source_level="run",
        source_record_id="run-1-start", result="unknown", created_at=NOW)),
    ("ResolvedSource", ResolvedSource, ResolvedSource(
        schema_version="0.1", run_id="run-1", record_id="record-1", source_level="run",
        result="unknown")),
)
_INGRESS_MODEL_IDS = [case[0] for case in _INGRESS_MODELS]


def _assert_bounded_rejection(action, *, expect: str | None = None) -> None:
    """Run one ingress action and require a bounded, value-free MemoryValidationError.

    Every public surface is inspected, including the ones a pydantic error would expose. A
    Memory error has no ``errors``/``json`` of its own, so an absent surface is simply empty
    rather than a leak.
    """
    with pytest.raises(MemoryValidationError) as info:
        action()
    error = info.value
    errors_surface = getattr(error, "errors", None)
    json_surface = getattr(error, "json", None)
    surfaces = (
        str(error), repr(error),
        str(errors_surface() if callable(errors_surface) else ""),
        str(json_surface() if callable(json_surface) else ""),
    )
    for surface in surfaces:
        assert _MARKER not in surface, surface
    assert max(len(surface) for surface in surfaces) < 400
    if expect is not None:
        assert expect in str(error), str(error)


@pytest.mark.parametrize("model_name,model,record", _INGRESS_MODELS, ids=_INGRESS_MODEL_IDS)
@pytest.mark.parametrize("non_record", [1, 1.5, [(_MARKER,)], _MARKER, None, True, object()])
def test_a_non_record_python_input_is_a_bounded_rejection(
    model_name: str, model: type, record: object, non_record: object,
) -> None:
    """F01: a top-level type failure is wrapped, never a raw ValidationError."""
    _assert_bounded_rejection(lambda m=model, v=non_record: m.model_validate(v),
                              expect="not an object this record can be built from")


@pytest.mark.parametrize("model_name,model,record", _INGRESS_MODELS, ids=_INGRESS_MODEL_IDS)
@pytest.mark.parametrize("document", [
    '{"' + _MARKER + '":',
    "[" + _MARKER + "]",
    '"' + _MARKER + '"',
    "null",
    "7",
    "not-json-" + _MARKER,
    "",
    "{",
])
def test_a_malformed_or_non_object_json_document_is_a_bounded_rejection(
    model_name: str, model: type, record: object, document: str,
) -> None:
    """F01: JSON syntax failures and non-object documents are wrapped too."""
    _assert_bounded_rejection(lambda m=model, d=document: m.model_validate_json(d))


@pytest.mark.parametrize("model_name,model,record", _INGRESS_MODELS, ids=_INGRESS_MODEL_IDS)
def test_a_json_syntax_failure_is_described_as_an_invalid_document(
    model_name: str, model: type, record: object,
) -> None:
    """F01: a document that never parsed is described by its category alone."""
    _assert_bounded_rejection(
        lambda m=model: m.model_validate_json('{"' + _MARKER + '":'),
        expect="not a valid JSON document",
    )


@pytest.mark.parametrize("model_name,model,record", _INGRESS_MODELS, ids=_INGRESS_MODEL_IDS)
def test_a_field_level_failure_through_both_entry_points_stays_located(
    model_name: str, model: type, record: object,
) -> None:
    """F01: a field-level rejection keeps its declared location through both boundaries."""
    payload = record.model_dump(mode="json")
    payload["schema_version"] = "9.9"
    for action in (
        lambda p=payload, m=model: m.model_validate(p),
        lambda p=payload, m=model: m.model_validate_json(json.dumps(p)),
        lambda p=payload, m=model: m(**p),
    ):
        _assert_bounded_rejection(action, expect="schema_version")


@pytest.mark.parametrize("model_name,model,record", _INGRESS_MODELS, ids=_INGRESS_MODEL_IDS)
def test_a_long_synthetic_key_stays_bounded_through_both_entry_points(
    model_name: str, model: type, record: object,
) -> None:
    """F01: a 2000-character key is neither echoed nor allowed to grow the message."""
    payload = record.model_dump(mode="json")
    payload[_MARKER + "X" * 2000] = _MARKER + "_VALUE"
    for action in (
        lambda p=payload, m=model: m.model_validate(p),
        lambda p=payload, m=model: m.model_validate_json(json.dumps(p)),
        lambda p=payload, m=model: m(**p),
    ):
        _assert_bounded_rejection(action, expect="unexpected field")


def test_no_public_parser_surface_returns_a_raw_pydantic_error() -> None:
    """F01: both entry points on both records always raise the bounded Memory error."""
    for model_name, model, record in _INGRESS_MODELS:
        for action in (
            lambda m=model: m.model_validate(1),
            lambda m=model: m.model_validate(None),
            lambda m=model: m.model_validate_json("{"),
            lambda m=model: m.model_validate_json("null"),
        ):
            with pytest.raises(MemoryValidationError):
                action()


@pytest.mark.parametrize("model_name,model,record", _INGRESS_MODELS, ids=_INGRESS_MODEL_IDS)
def test_an_unparsable_json_document_keeps_its_cause_in_process_only(
    model_name: str, model: type, record: object,
) -> None:
    """F01: the pydantic cause stays available in-process and is never serialized."""
    with pytest.raises(MemoryValidationError) as info:
        model.model_validate_json('{"' + _MARKER + '":')
    cause = info.value.__cause__
    assert cause is not None
    assert _MARKER not in str(info.value)
    # The bounded public surface has no structured rendering of its own to leak through.
    assert not hasattr(info.value, "errors")
    assert not hasattr(info.value, "json")


@pytest.mark.parametrize("model_name,model,record", _INGRESS_MODELS, ids=_INGRESS_MODEL_IDS)
def test_the_supported_validation_parameters_still_work(
    model_name: str, model: type, record: object,
) -> None:
    """F01 control: the supported parameters keep working through the wrapper."""
    payload = record.model_dump(mode="json")
    assert model.model_validate(payload, strict=False).model_dump(mode="json") == payload
    assert model.model_validate_json(record.model_dump_json()).model_dump(mode="json") == payload
    assert model.model_validate(record).model_dump(mode="json") == payload


def test_strict_json_timestamp_semantics_are_preserved() -> None:
    """F01 control: the wrapper must not become a lossy JSON-to-Python coercion."""
    valid = item("m1", "run-1", "run-1-start", "unknown").model_dump_json()
    assert MemoryItem.model_validate_json(valid).created_at == NOW
    naive = json.dumps({
        "schema_version": "0.1", "memory_id": "m1", "run_id": "run-1", "source_level": "run",
        "source_record_id": "run-1-start", "result": "unknown",
        "created_at": "2026-10-08T12:00:00", "supersedes_id": None,
    })
    with pytest.raises(MemoryValidationError) as info:
        MemoryItem.model_validate_json(naive)
    assert "created_at" in str(info.value)
    with pytest.raises(MemoryValidationError):
        MemoryItem.model_validate_json(json.dumps({
            "schema_version": "0.1", "memory_id": "m1", "run_id": "run-1", "source_level": "run",
            "source_record_id": "run-1-start", "result": "unknown", "created_at": 7,
            "supersedes_id": None,
        }))
