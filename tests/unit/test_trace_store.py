"""Acceptance tests for the JSONL tracing store.

Every case drives the real :class:`JsonlTraceStore` against a file under an external scratch
directory. Assertions check values, identities, order and on-disk bytes; a rejection is
checked through its error class and through the unchanged file.

Storage failures are injected by replacing the module's ``open`` with a controlled stand-in or
by pointing the store at a path whose parent or target the operating system itself refuses.
Nothing is monkeypatched inside the store's own logic beyond that single seam.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.models.outcomes import ExecutionStatus, VerificationStatus
from forecontract.tracing import (
    JsonlTraceStore,
    RunFinished,
    RunStarted,
    TraceCorruptionError,
    TraceNotFoundError,
    TraceReader,
    TraceStorageError,
    TraceValidationError,
    TraceWriter,
    parse_record,
)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
DECLARED_SCRATCH = Path(os.environ.get("FORECONTRACT_TEST_SCRATCH", ""))

#: The genuine ``open`` captured before any test replaces it, so a wrapper can still reach the
#: real file object while the module under test sees the stand-in.
REAL_OPEN = builtins.open


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
def trace_path() -> Path:
    base = _scratch_base() / f"trace-session-{uuid.uuid4().hex[:10]}"
    base.mkdir(parents=True, exist_ok=True)
    return base / "trace.jsonl"


@pytest.fixture
def store(trace_path: Path) -> JsonlTraceStore:
    return JsonlTraceStore(trace_path)


def _record(model, **overrides):
    values = {
        "schema_version": "0.1",
        "record_id": "record-1",
        "run_id": "run-1",
        "sequence": 0,
        "occurred_at": NOW,
        "kind": model.model_fields["kind"].default,
    }
    values.update(overrides)
    return model(**values)


def started(run_id: str = "run-1", sequence: int = 0, record_id: str | None = None) -> RunStarted:
    return _record(
        RunStarted, run_id=run_id, sequence=sequence,
        record_id=record_id or f"{run_id}-start",
    )


def step(run_id: str = "run-1", sequence: int = 1, step_id: str = "step-1"):
    from forecontract.tracing import StepStarted

    return _record(StepStarted, run_id=run_id, sequence=sequence, record_id=f"{run_id}-step-{step_id}",
                   step_id=step_id)


def contract(run_id: str = "run-1", sequence: int = 2, step_id: str = "step-1",
             contract_id: str = "contract-1", replaces: str | None = None):
    from forecontract.tracing import ContractCommitted

    return _record(
        ContractCommitted, run_id=run_id, sequence=sequence,
        record_id=f"{run_id}-{contract_id}", step_id=step_id, contract_id=contract_id,
        replaces_contract_id=replaces,
    )


def attempt(run_id: str = "run-1", sequence: int = 3, step_id: str = "step-1",
            contract_id: str = "contract-1", attempt_id: str = "attempt-1"):
    from forecontract.tracing import AttemptStarted

    return _record(
        AttemptStarted, run_id=run_id, sequence=sequence, record_id=f"{run_id}-{attempt_id}",
        step_id=step_id, contract_id=contract_id, attempt_id=attempt_id,
    )


def finished(run_id: str = "run-1", sequence: int = 4, step_id: str = "step-1",
             contract_id: str = "contract-1", attempt_id: str = "attempt-1",
             status: str = "succeeded", error_code: str | None = None):
    from forecontract.tracing import AttemptFinished

    return _record(
        AttemptFinished, run_id=run_id, sequence=sequence,
        record_id=f"{run_id}-{attempt_id}-finished", step_id=step_id, contract_id=contract_id,
        attempt_id=attempt_id, execution_status=status, error_code=error_code,
    )


def verification(run_id: str = "run-1", sequence: int = 5, scope: str = "run",
                 status: str = "pass", record_id: str | None = None,
                 step_id: str | None = None, contract_id: str | None = None,
                 attempt_id: str | None = None):
    from forecontract.tracing import VerificationRecorded

    return _record(
        VerificationRecorded, run_id=run_id, sequence=sequence,
        record_id=record_id or f"{run_id}-verification-{sequence}", scope=scope,
        verification_status=status, step_id=step_id, contract_id=contract_id, attempt_id=attempt_id,
    )


def terminal(run_id: str = "run-1", sequence: int = 6, status: str = "failure",
             reason_code: str | None = None, verification_record_id: str | None = None,
             record_id: str | None = None):
    return _record(
        RunFinished, run_id=run_id, sequence=sequence, record_id=record_id or f"{run_id}-terminal",
        terminal_status=status, reason_code=reason_code, verification_record_id=verification_record_id,
    )


def append_all(store: JsonlTraceStore, *records) -> None:
    for record in records:
        store.append(record)


def _json_line(record) -> str:
    """The exact one-line JSON encoding the store writes for one record."""
    return json.dumps(record.model_dump(mode="json"), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def complete_run(store: JsonlTraceStore, run_id: str = "run-1", *, status: str = "failure") -> None:
    """Append the shortest legal complete run: start, one terminal fact."""
    append_all(store, started(run_id, 0), terminal(run_id, 1, status))


# --- protocol conformance and side-effect-free construction -------------------

def test_the_store_implements_both_tracing_protocols(store: JsonlTraceStore) -> None:
    assert isinstance(store, TraceWriter)
    assert isinstance(store, TraceReader)


def test_construction_and_queries_never_create_or_touch_the_file(trace_path: Path) -> None:
    store = JsonlTraceStore(trace_path)
    assert store.path == trace_path
    assert not trace_path.exists()
    with pytest.raises(TraceNotFoundError):
        store.read_run("run-1")
    with pytest.raises(TraceNotFoundError):
        store.get_record("run-1", "record-1")
    assert not trace_path.exists()


def test_the_first_append_creates_the_file_and_later_appends_extend_it(
    store: JsonlTraceStore, trace_path: Path
) -> None:
    append_all(store, started(), terminal(sequence=1))
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["kind"] for line in lines] == ["run_started", "run_finished"]
    assert trace_path.read_bytes().endswith(b"\n")


def test_a_stored_line_is_one_complete_json_record(
    store: JsonlTraceStore, trace_path: Path
) -> None:
    record = started()
    store.append(record)
    text = trace_path.read_text(encoding="utf-8")
    assert text.count("\n") == 1
    payload = json.loads(text)
    assert payload["record_id"] == record.record_id == "run-1-start"
    assert payload["run_id"] == "run-1"
    assert payload["sequence"] == 0
    assert payload["occurred_at"] == "2026-10-08T12:00:00Z"


# --- reading and source identity ---------------------------------------------

def test_read_run_returns_records_in_ascending_sequence_order(
    store: JsonlTraceStore
) -> None:
    append_all(
        store,
        started(),
        step(),
        contract(),
        attempt(),
        finished(),
        verification(scope="attempt", step_id="step-1", contract_id="contract-1", attempt_id="attempt-1"),
        terminal(sequence=6),
    )
    records = store.read_run("run-1")
    assert [item.sequence for item in records] == [0, 1, 2, 3, 4, 5, 6]
    assert [item.kind for item in records] == [
        "run_started", "step_started", "contract_committed", "attempt_started",
        "attempt_finished", "verification_recorded", "run_finished",
    ]
    assert records[4].execution_status is ExecutionStatus.SUCCEEDED
    assert records[5].verification_status is VerificationStatus.PASS


def test_source_identity_and_occurred_at_survive_a_reload(trace_path: Path) -> None:
    store = JsonlTraceStore(trace_path)
    record = attempt(sequence=3, attempt_id="attempt-42")
    append_all(store, started(), step(), contract(), record)
    reloaded = JsonlTraceStore(trace_path).get_record("run-1", "run-1-attempt-42")
    assert reloaded.record_id == "run-1-attempt-42"
    assert reloaded.attempt_id == "attempt-42"
    assert reloaded.sequence == record.sequence == 3
    assert reloaded.occurred_at == NOW
    assert reloaded.occurred_at.tzinfo is not None
    assert reloaded == record


def test_get_record_rejects_a_record_identifier_of_another_run(store: JsonlTraceStore) -> None:
    append_all(store, started("run-1", 0), terminal("run-1", 1), started("run-2", 0, "run-2-start"))
    with pytest.raises(TraceNotFoundError):
        store.get_record("run-2", "run-1-start")


def test_a_missing_run_and_a_missing_record_are_not_found(store: JsonlTraceStore) -> None:
    complete_run(store)
    with pytest.raises(TraceNotFoundError):
        store.read_run("run-absent")
    with pytest.raises(TraceNotFoundError):
        store.get_record("run-1", "record-absent")


@pytest.mark.parametrize("query,arguments", [
    ("read_run", (1,)),
    ("read_run", (True,)),
    ("read_run", ("",)),
    ("get_record", ("run-1", None)),
    ("get_record", ("run-1", "  ")),
])
def test_a_malformed_query_argument_is_a_validation_error(
    store: JsonlTraceStore, query: str, arguments: tuple
) -> None:
    with pytest.raises(TraceValidationError):
        getattr(store, query)(*arguments)


def test_an_empty_file_has_no_runs(trace_path: Path) -> None:
    trace_path.write_bytes(b"")
    store = JsonlTraceStore(trace_path)
    with pytest.raises(TraceNotFoundError):
        store.read_run("run-1")


def test_reading_never_mutates_the_caller_or_the_stored_file(
    store: JsonlTraceStore, trace_path: Path
) -> None:
    complete_run(store)
    before = trace_path.read_bytes()
    first = store.read_run("run-1")
    second = store.read_run("run-1")
    assert first == second
    assert first is not second
    assert first[0] is not second[0]
    assert first[0].record_id == "run-1-start"
    assert trace_path.read_bytes() == before


# --- multi-run interleaving and continuity -----------------------------------

def test_two_runs_interleave_and_keep_separate_sequence_chains(store: JsonlTraceStore) -> None:
    append_all(
        store,
        started("run-a", 0),
        started("run-b", 0),
        terminal("run-a", 1),
        terminal("run-b", 1, "cancelled"),
    )
    run_a = store.read_run("run-a")
    run_b = store.read_run("run-b")
    assert [item.record_id for item in run_a] == ["run-a-start", "run-a-terminal"]
    assert [item.record_id for item in run_b] == ["run-b-start", "run-b-terminal"]
    assert [item.sequence for item in run_a] == [0, 1]
    assert [item.sequence for item in run_b] == [0, 1]
    assert run_a[1].terminal_status == "failure"
    assert run_b[1].terminal_status == "cancelled"


@pytest.mark.parametrize("sequence", [2, 5])
def test_a_sequence_gap_is_rejected(store: JsonlTraceStore, sequence: int) -> None:
    store.append(started())
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=sequence))


def test_a_repeated_sequence_number_is_rejected(store: JsonlTraceStore) -> None:
    store.append(started())
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=0))


def test_a_record_before_its_run_start_is_rejected(store: JsonlTraceStore) -> None:
    with pytest.raises(TraceValidationError):
        store.append(step())


def test_a_second_run_start_for_one_run_is_rejected(store: JsonlTraceStore) -> None:
    store.append(started())
    with pytest.raises(TraceValidationError):
        store.append(started("run-1", 1))


def test_a_duplicate_record_identifier_is_rejected_across_runs(store: JsonlTraceStore) -> None:
    append_all(store, started("run-1", 0), terminal("run-1", 1))
    duplicate = started("run-2", 0, record_id="run-1-start")
    with pytest.raises(TraceValidationError):
        store.append(duplicate)


def test_a_rejected_append_leaves_the_file_and_the_next_sequence_untouched(
    store: JsonlTraceStore, trace_path: Path
) -> None:
    store.append(started())
    before = trace_path.read_bytes()
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=7))
    assert trace_path.read_bytes() == before
    store.append(terminal(sequence=1))
    assert [json.loads(line)["sequence"] for line in trace_path.read_text(encoding="utf-8").splitlines()] == [0, 1]


def test_an_object_that_is_not_a_record_is_rejected(store: JsonlTraceStore) -> None:
    with pytest.raises(TraceValidationError):
        store.append({"kind": "run_started"})
    with pytest.raises(TraceValidationError):
        store.append(None)


# --- entity association -------------------------------------------------------

def test_a_contract_for_an_unknown_step_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step())
    with pytest.raises(TraceValidationError):
        store.append(contract(step_id="step-unknown"))


def test_an_attempt_for_an_unknown_contract_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step())
    with pytest.raises(TraceValidationError):
        store.append(attempt(contract_id="contract-unknown"))


def test_an_attempt_for_a_step_it_does_not_belong_to_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(step_id="step-1"), step(sequence=2, step_id="step-2"),
               contract(sequence=3, step_id="step-2", contract_id="contract-2"))
    with pytest.raises(TraceValidationError):
        store.append(attempt(sequence=4, step_id="step-1", contract_id="contract-2"))


def test_a_duplicate_step_identifier_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step())
    with pytest.raises(TraceValidationError):
        store.append(step(sequence=2, step_id="step-1"))


def test_a_duplicate_contract_identifier_in_one_step_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(replaces=None))
    with pytest.raises(TraceValidationError):
        store.append(contract(sequence=3, contract_id="contract-1", replaces="contract-1"))


def test_a_duplicate_attempt_identifier_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt())
    with pytest.raises(TraceValidationError):
        store.append(attempt(sequence=4, attempt_id="attempt-1"))


# --- replan, retry and contract versions -------------------------------------

def test_a_replan_replaces_the_current_contract_and_starts_a_new_attempt(
    store: JsonlTraceStore
) -> None:
    append_all(
        store,
        started(),
        step(),
        contract(contract_id="contract-1"),
        attempt(sequence=3, attempt_id="attempt-1"),
        finished(sequence=4, attempt_id="attempt-1", status="failed", error_code="execution_failed"),
        contract(sequence=5, contract_id="contract-2", replaces="contract-1"),
        attempt(sequence=6, contract_id="contract-2", attempt_id="attempt-2"),
    )
    records = store.read_run("run-1")
    assert [item.sequence for item in records] == list(range(7))
    assert records[5].replaces_contract_id == "contract-1"
    assert records[6].attempt_id == "attempt-2"
    assert records[6].contract_id == "contract-2"


def test_a_retry_is_always_a_new_attempt_of_the_same_contract(store: JsonlTraceStore) -> None:
    append_all(
        store,
        started(),
        step(),
        contract(),
        attempt(attempt_id="attempt-1"),
        finished(sequence=4, attempt_id="attempt-1", status="failed"),
        attempt(sequence=5, attempt_id="attempt-2"),
        finished(sequence=6, attempt_id="attempt-2", status="succeeded"),
    )
    attempts = [item for item in store.read_run("run-1") if item.kind in {"attempt_started", "attempt_finished"}]
    assert [item.attempt_id for item in attempts] == ["attempt-1", "attempt-1", "attempt-2", "attempt-2"]
    assert {item.contract_id for item in attempts} == {"contract-1"}


def test_a_contract_that_replaces_a_stale_version_is_rejected(store: JsonlTraceStore) -> None:
    append_all(
        store,
        started(),
        step(),
        contract(contract_id="contract-1"),
        contract(sequence=3, contract_id="contract-2", replaces="contract-1"),
    )
    with pytest.raises(TraceValidationError):
        store.append(contract(sequence=4, contract_id="contract-3", replaces="contract-1"))


def test_a_second_contract_may_not_be_a_silent_replacement(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(contract_id="contract-1"))
    with pytest.raises(TraceValidationError):
        store.append(contract(sequence=3, contract_id="contract-2", replaces=None))


def test_a_replacement_from_another_step_or_run_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(step_id="step-1"), step(sequence=2, step_id="step-2"),
               contract(sequence=3, step_id="step-1", contract_id="contract-1"))
    with pytest.raises(TraceValidationError):
        store.append(contract(sequence=4, step_id="step-2", contract_id="contract-9",
                              replaces="contract-1"))


def test_a_contract_cannot_replace_itself(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(contract_id="contract-1"))
    with pytest.raises(TraceValidationError):
        store.append(contract(sequence=3, contract_id="contract-1", replaces="contract-1"))


def test_a_second_attempt_on_the_current_contract_is_the_retry_experience(
    store: JsonlTraceStore
) -> None:
    """Retry keeps the Contract and opens a new Attempt; a replan adds the version instead."""
    append_all(
        store,
        started(),
        step(),
        contract(contract_id="contract-1"),
        attempt(sequence=3, attempt_id="attempt-1"),
        finished(sequence=4, attempt_id="attempt-1", status="failed", error_code="execution_failed"),
        attempt(sequence=5, attempt_id="attempt-2"),
        finished(sequence=6, attempt_id="attempt-2", status="succeeded"),
    )
    starts = [(item.attempt_id, item.contract_id) for item in store.read_run("run-1")
              if item.kind == "attempt_started"]
    assert starts == [("attempt-1", "contract-1"), ("attempt-2", "contract-1")]
    failures = [item.execution_status for item in store.read_run("run-1")
                if item.kind == "attempt_finished"]
    assert [status.value for status in failures] == ["failed", "succeeded"]


def test_a_new_attempt_after_a_replan_uses_the_current_contract(store: JsonlTraceStore) -> None:
    append_all(
        store,
        started(),
        step(),
        contract(contract_id="contract-1"),
        attempt(sequence=3, attempt_id="attempt-1"),
        finished(sequence=4, attempt_id="attempt-1", status="failed"),
        contract(sequence=5, contract_id="contract-2", replaces="contract-1"),
        attempt(sequence=6, contract_id="contract-2", attempt_id="attempt-2"),
    )
    starts = [(item.attempt_id, item.contract_id) for item in store.read_run("run-1")
              if item.kind == "attempt_started"]
    assert starts == [("attempt-1", "contract-1"), ("attempt-2", "contract-2")]


# --- attempt completion facts -------------------------------------------------

def test_a_second_completion_for_one_attempt_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(), finished())
    with pytest.raises(TraceValidationError):
        store.append(finished(sequence=5, status="failed"))


def test_a_completion_of_an_unknown_attempt_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract())
    with pytest.raises(TraceValidationError):
        store.append(finished(sequence=3, attempt_id="attempt-unknown"))


def test_a_completion_with_a_mismatched_contract_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(attempt_id="attempt-1"))
    with pytest.raises(TraceValidationError):
        store.append(finished(sequence=4, attempt_id="attempt-1", contract_id="contract-other"))


def test_a_failed_attempt_may_carry_only_a_closed_code(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(),
               finished(status="failed", error_code="execution_failed"))
    record = store.get_record("run-1", "run-1-attempt-1-finished")
    assert record.execution_status is ExecutionStatus.FAILED
    assert record.error_code == "execution_failed"


def test_an_unknown_attempt_status_is_rejected_before_it_can_be_stored(
    store: JsonlTraceStore
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        finished(status="maybe")


# --- verification facts and the terminal rule ---------------------------------

def test_an_attempt_scoped_verification_requires_a_finished_attempt(
    store: JsonlTraceStore
) -> None:
    append_all(store, started(), step(), contract(), attempt())
    with pytest.raises(TraceValidationError):
        store.append(verification(sequence=4, scope="attempt", step_id="step-1",
                                  contract_id="contract-1", attempt_id="attempt-1"))


def test_a_matching_attempt_scoped_verification_is_accepted(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(), finished(),
               verification(scope="attempt", status="fail", step_id="step-1",
                            contract_id="contract-1", attempt_id="attempt-1"))
    record = store.get_record("run-1", "run-1-verification-5")
    assert record.scope == "attempt" and record.verification_status is VerificationStatus.FAIL


def test_a_verification_for_an_unknown_attempt_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(), finished())
    with pytest.raises(TraceValidationError):
        store.append(verification(scope="attempt", step_id="step-1", contract_id="contract-1",
                                  attempt_id="attempt-unknown"))


def test_several_verification_facts_keep_distinct_identities(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(), finished(),
               verification(sequence=5, scope="attempt", status="fail", step_id="step-1",
                            contract_id="contract-1", attempt_id="attempt-1"),
               verification(sequence=6, scope="attempt", status="pass", step_id="step-1",
                            contract_id="contract-1", attempt_id="attempt-1"),
               verification(sequence=7, scope="run", status="pass"),
               terminal(sequence=8, status="success", verification_record_id="run-1-verification-7"))
    records = store.read_run("run-1")
    identity = [item.record_id for item in records if item.kind == "verification_recorded"]
    assert identity == ["run-1-verification-5", "run-1-verification-6", "run-1-verification-7"]
    assert records[-1].verification_record_id == "run-1-verification-7"


def test_a_success_without_a_reference_is_rejected_by_the_model() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        terminal(status="success")


def test_a_success_referencing_an_attempt_scoped_pass_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(), finished(),
               verification(scope="attempt", status="pass", step_id="step-1",
                            contract_id="contract-1", attempt_id="attempt-1"))
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=6, status="success", verification_record_id="run-1-verification-5"))


def test_a_success_referencing_a_run_scoped_fail_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started(),
               verification(sequence=1, scope="run", status="fail", record_id="verification-fail"))
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=2, status="success", verification_record_id="verification-fail"))


def test_a_success_referencing_a_missing_verification_is_rejected(store: JsonlTraceStore) -> None:
    store.append(started())
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=1, status="success", verification_record_id="verification-absent"))


def test_a_success_referencing_a_run_scoped_pass_is_accepted(store: JsonlTraceStore) -> None:
    append_all(store, started(), verification(sequence=1, scope="run", status="pass",
                                              record_id="verification-pass"),
               terminal(sequence=2, status="success", verification_record_id="verification-pass"))
    assert store.read_run("run-1")[-1].terminal_status == "success"


def test_a_success_referencing_another_runs_pass_is_rejected(store: JsonlTraceStore) -> None:
    append_all(store, started("run-2", 0, "run-2-start"),
               verification("run-2", 1, scope="run", status="pass", record_id="run-2-pass"),
               started("run-1", 0))
    with pytest.raises(TraceValidationError):
        store.append(terminal("run-1", 1, status="success", verification_record_id="run-2-pass"))


def test_a_run_scoped_pass_is_not_reachable_from_another_record_identity(store: JsonlTraceStore) -> None:
    """A PASS record is only a source through its exact identifier, never as the latest one."""
    append_all(store, started(),
               verification(sequence=1, scope="run", status="pass", record_id="verification-pass"))
    records = store.read_run("run-1")
    assert [item.record_id for item in records] == ["run-1-start", "verification-pass"]
    assert records[1].verification_status is VerificationStatus.PASS
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=2, status="success", verification_record_id="run-1-start"))


@pytest.mark.parametrize("status", ["failure", "cancelled", "timeout", "aborted"])
def test_every_non_success_terminal_status_is_accepted(store: JsonlTraceStore, status: str) -> None:
    complete_run(store, status=status)
    assert store.read_run("run-1")[-1].terminal_status == status


def test_an_early_failure_may_leave_an_attempt_unfinished(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(),
               terminal(sequence=4, status="failure", reason_code="internal_error"))
    records = store.read_run("run-1")
    assert [item.kind for item in records][-1] == "run_finished"
    assert not [item for item in records if item.kind == "attempt_finished"]
    unknown = [item for item in records if item.kind == "attempt_started"]
    assert unknown[0].attempt_id == "attempt-1"


def test_an_abort_may_leave_an_attempt_unfinished(store: JsonlTraceStore) -> None:
    append_all(store, started(), step(), contract(), attempt(),
               terminal(sequence=4, status="aborted", reason_code="policy_abort"))
    assert store.read_run("run-1")[-1].reason_code == "policy_abort"


def test_a_run_may_finish_without_any_step(store: JsonlTraceStore) -> None:
    complete_run(store)
    assert [item.kind for item in store.read_run("run-1")] == ["run_started", "run_finished"]


def test_a_second_terminal_fact_is_rejected(store: JsonlTraceStore) -> None:
    complete_run(store)
    with pytest.raises(TraceValidationError):
        store.append(terminal(sequence=2, status="cancelled"))


@pytest.mark.parametrize("fact", ["step", "verification", "attempt"])
def test_every_record_after_the_terminal_fact_is_rejected(store: JsonlTraceStore, fact: str) -> None:
    append_all(store, started(), step(), contract())
    store.append(terminal(sequence=3, status="cancelled"))
    with pytest.raises(TraceValidationError):
        if fact == "step":
            store.append(step(sequence=4, step_id="step-2"))
        elif fact == "verification":
            store.append(verification(sequence=4, scope="run", status="pass"))
        else:
            store.append(attempt(sequence=4, attempt_id="attempt-1"))
    assert len(store.read_run("run-1")) == 4


# --- corruption detection -----------------------------------------------------

@pytest.mark.parametrize("payload,label", [
    (b'{"schema_version":"0.1"}\n', "missing fields"),
    (b"".join([
        b'{"schema_version":"0.2","record_id":"a","run_id":"r","sequence":0,',
        b'"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n',
    ]), "unknown schema version"),
    (b"".join([
        b'{"schema_version":"0.1","record_id":"a","run_id":"r","sequence":0,',
        b'"occurred_at":"2026-10-08T12:00:00Z","kind":"run_ended"}\n',
    ]), "unknown kind"),
    (b"".join([
        b'{"schema_version":"0.1","record_id":"a","run_id":"r","sequence":true,',
        b'"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n',
    ]), "boolean sequence"),
])
def test_a_line_that_violates_the_schema_is_corruption(trace_path: Path, payload: bytes, label: str) -> None:
    trace_path.write_bytes(payload)
    store = JsonlTraceStore(trace_path)
    with pytest.raises(TraceCorruptionError) as info:
        store.read_run("r")
    assert info.value.line_number == 1


def test_invalid_utf8_is_corruption(trace_path: Path) -> None:
    trace_path.write_bytes(b"\xff\xfe\x00\x00")
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("run-1")
    assert info.value.line_number is None


def test_a_blank_line_is_corruption(trace_path: Path) -> None:
    store = JsonlTraceStore(trace_path)
    store.append(started())
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(TraceCorruptionError) as info:
        store.read_run("run-1")
    assert info.value.line_number == 2


def test_a_duplicate_json_key_is_corruption(trace_path: Path) -> None:
    trace_path.write_text(
        '{"schema_version":"0.1","record_id":"a","record_id":"b","run_id":"r","sequence":0,'
        '"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("r")
    assert info.value.line_number == 1


def test_a_non_finite_number_is_corruption(trace_path: Path) -> None:
    trace_path.write_text(
        '{"schema_version":"0.1","record_id":"a","run_id":"r","sequence":NaN,'
        '"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TraceCorruptionError):
        JsonlTraceStore(trace_path).read_run("r")


def test_a_truncated_tail_is_corruption(trace_path: Path) -> None:
    store = JsonlTraceStore(trace_path)
    complete_run(store)
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write('{"schema_version":"0.1","record_id":"tail"')
    with pytest.raises(TraceCorruptionError) as info:
        store.read_run("run-1")
    assert info.value.line_number == 3


def test_a_final_line_without_a_newline_is_corruption(trace_path: Path) -> None:
    store = JsonlTraceStore(trace_path)
    store.append(started())
    text = trace_path.read_text(encoding="utf-8")
    trace_path.write_text(text.rstrip("\n"), encoding="utf-8")
    with pytest.raises(TraceCorruptionError) as info:
        store.read_run("run-1")
    assert info.value.line_number == 1


def test_corruption_after_the_requested_record_is_still_reported(trace_path: Path) -> None:
    store = JsonlTraceStore(trace_path)
    complete_run(store)
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write('{"not":"a record"}\n')
    with pytest.raises(TraceCorruptionError):
        store.get_record("run-1", "run-1-start")
    with pytest.raises(TraceCorruptionError):
        store.read_run("run-1")


def test_an_appended_line_that_would_extend_corruption_is_rejected(trace_path: Path) -> None:
    trace_path.write_text('{"not":"a record"}\n', encoding="utf-8")
    store = JsonlTraceStore(trace_path)
    with pytest.raises(TraceCorruptionError):
        store.append(started())
    assert trace_path.read_text(encoding="utf-8") == '{"not":"a record"}\n'


def test_a_duplicate_record_identifier_on_disk_is_corruption(trace_path: Path) -> None:
    line = (b'{"schema_version":"0.1","record_id":"same","run_id":"run-1","sequence":0,'
            b'"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n')
    trace_path.write_bytes(line + line.replace(b'"sequence":0', b'"sequence":1'))
    with pytest.raises(TraceCorruptionError):
        JsonlTraceStore(trace_path).read_run("run-1")


def test_an_out_of_order_sequence_on_disk_is_corruption(trace_path: Path) -> None:
    first = (b'{"schema_version":"0.1","record_id":"a","run_id":"run-1","sequence":0,'
             b'"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n')
    second = (b'{"schema_version":"0.1","record_id":"b","run_id":"run-1","sequence":4,'
              b'"occurred_at":"2026-10-08T12:00:00Z","kind":"run_finished",'
              b'"terminal_status":"failure","reason_code":null,"verification_record_id":null}\n')
    trace_path.write_bytes(first + second)
    with pytest.raises(TraceCorruptionError):
        JsonlTraceStore(trace_path).read_run("run-1")


def test_an_invalid_relation_on_disk_is_corruption(trace_path: Path) -> None:
    first = (b'{"schema_version":"0.1","record_id":"a","run_id":"run-1","sequence":0,'
             b'"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n')
    second = (b'{"schema_version":"0.1","record_id":"b","run_id":"run-1","sequence":1,'
              b'"occurred_at":"2026-10-08T12:00:00Z","kind":"step_started","step_id":"step-1"}\n')
    third = (b'{"schema_version":"0.1","record_id":"c","run_id":"run-1","sequence":2,'
             b'"occurred_at":"2026-10-08T12:00:00Z","kind":"contract_committed","step_id":"step-9",'
             b'"contract_id":"contract-1","replaces_contract_id":null}\n')
    trace_path.write_bytes(first + second + third)
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("run-1")
    assert info.value.line_number == 3


# --- injected storage failures -----------------------------------------------

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
        # A real full write acknowledges the real character count, which is what the store now
        # requires before it reports success (F06).
        return self._handle.write(text)

    def flush(self):
        return self._handle.flush()

    def close(self):
        return self._handle.close()


def test_an_open_failure_is_a_storage_error_that_preserves_its_cause(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlTraceStore(trace_path)

    def refuse(*args, **kwargs):
        raise PermissionError("synthetic open refusal")

    monkeypatch.setattr(builtins, "open", refuse, raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    assert isinstance(info.value.__cause__, PermissionError)
    assert info.value.code == "TRACE_STORAGE_FAILURE"
    assert not trace_path.exists()


def test_a_write_failure_does_not_claim_persistence_or_leave_a_partial_record(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlTraceStore(trace_path)
    store.append(started())
    before = trace_path.read_bytes()

    class FailingWrite(_StandInHandle):
        def write(self, text):
            raise OSError("synthetic write refusal")

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: FailingWrite(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(terminal(sequence=1))
    assert isinstance(info.value.__cause__, OSError)
    monkeypatch.undo()
    assert trace_path.read_bytes() == before
    store = JsonlTraceStore(trace_path)
    store.append(terminal(sequence=1))
    payloads = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [item["sequence"] for item in payloads] == [0, 1]
    assert [item["kind"] for item in payloads] == ["run_started", "run_finished"]


def test_a_flush_failure_is_a_storage_error(trace_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlTraceStore(trace_path)

    class FailingFlush(_StandInHandle):
        def flush(self):
            raise OSError("synthetic flush refusal")

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: FailingFlush(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    assert isinstance(info.value.__cause__, OSError)


def test_a_close_failure_is_a_storage_error(trace_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonlTraceStore(trace_path)

    class FailingClose(_StandInHandle):
        def close(self):
            self._handle.close()
            raise OSError("synthetic close refusal")

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: FailingClose(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    assert isinstance(info.value.__cause__, OSError)


def test_appending_under_a_missing_parent_is_a_storage_error(trace_path: Path) -> None:
    missing = trace_path.parent / "absent-directory" / "trace.jsonl"
    store = JsonlTraceStore(missing)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    assert isinstance(info.value.__cause__, OSError)
    assert not missing.exists()


def test_a_read_failure_is_a_storage_error(trace_path: Path) -> None:
    store = JsonlTraceStore(trace_path.parent)
    with pytest.raises(TraceStorageError) as info:
        store.read_run("run-1")
    assert isinstance(info.value.__cause__, OSError)


def test_an_unusable_path_is_a_storage_error() -> None:
    with pytest.raises(TraceStorageError):
        JsonlTraceStore(7)


# --- safe payload boundary ----------------------------------------------------

def test_only_safe_codes_are_persisted_and_no_raw_text_reaches_the_file(
    store: JsonlTraceStore, trace_path: Path
) -> None:
    append_all(store, started(), step(), contract(), attempt(),
               finished(status="failed", error_code="execution_failed"),
               terminal(sequence=5, status="failure", reason_code="internal_error"))
    text = trace_path.read_text(encoding="utf-8")
    for token in ("message", "metadata", "artifact", "http", "prompt", "traceback", "observations"):
        assert token not in text
    payloads = [json.loads(line) for line in text.splitlines()]
    assert payloads[4]["error_code"] == "execution_failed"
    assert payloads[5]["reason_code"] == "internal_error"
    assert set(payloads[5]) == {
        "schema_version", "record_id", "run_id", "sequence", "occurred_at", "kind",
        "terminal_status", "reason_code", "verification_record_id",
    }


def test_a_rejected_extra_field_is_absent_from_disk_and_from_the_message(
    store: JsonlTraceStore, trace_path: Path
) -> None:
    marker = "SYNTHETIC-MARKER-NEVER-PERSISTED"
    store.append(started())
    before = trace_path.read_bytes()
    try:
        from forecontract.tracing import parse_record

        parse_record({
            "schema_version": "0.1", "record_id": "record-x", "run_id": "run-1", "sequence": 1,
            "occurred_at": "2026-10-08T12:00:00Z", "kind": "run_finished",
            "terminal_status": "failure", "metadata": marker,
        })
    except TraceValidationError as exc:
        assert marker not in str(exc)
    else:  # pragma: no cover - the record must be rejected
        raise AssertionError("a raw metadata field must not be accepted")
    assert marker.encode() not in trace_path.read_bytes()
    assert trace_path.read_bytes() == before


def test_a_storage_error_message_does_not_include_the_absolute_path(trace_path: Path) -> None:
    missing = trace_path.parent / "absent-directory" / "trace.jsonl"
    store = JsonlTraceStore(missing)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    assert str(missing) not in str(info.value)


def test_the_stored_metadata_never_gains_the_raw_cause(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonlTraceStore(trace_path)

    def refuse(*args, **kwargs):
        raise OSError("synthetic raw detail that must not be serialised")

    monkeypatch.setattr(builtins, "open", refuse, raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    assert "synthetic raw detail" not in str(info.value)
    assert not trace_path.exists()


# --- round-003 correction: F01, F02, F03, F05, F06 ---------------------------

def test_a_new_attempt_may_not_start_against_a_replaced_contract(store: JsonlTraceStore) -> None:
    """F01: after a replan the replaced version is historical, so it cannot open an Attempt."""
    append_all(
        store,
        started(),
        step(),
        contract(contract_id="contract-1"),
        contract(sequence=3, contract_id="contract-2", replaces="contract-1"),
    )
    with pytest.raises(TraceValidationError):
        store.append(attempt(sequence=4, contract_id="contract-1", attempt_id="attempt-1"))


def test_the_replaced_contract_rule_also_applies_on_reload(trace_path: Path) -> None:
    """F01: a file that already contains a stale-contract start is corruption, not accepted."""
    records = [
        _json_line(started()),
        _json_line(step()),
        _json_line(contract(contract_id="contract-1")),
        _json_line(contract(sequence=3, contract_id="contract-2", replaces="contract-1")),
        _json_line(attempt(sequence=4, contract_id="contract-1", attempt_id="attempt-1")),
    ]
    trace_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("run-1")
    assert info.value.line_number == 5


def test_an_attempt_that_started_before_a_replan_may_still_finish(store: JsonlTraceStore) -> None:
    """F01: the fix forbids a stale start, it does not invalidate an already-started Attempt."""
    append_all(
        store,
        started(),
        step(),
        contract(contract_id="contract-1"),
        attempt(sequence=3, contract_id="contract-1", attempt_id="attempt-1"),
        contract(sequence=4, contract_id="contract-2", replaces="contract-1"),
        finished(sequence=5, contract_id="contract-1", attempt_id="attempt-1", status="succeeded"),
    )
    records = store.read_run("run-1")
    assert [item.sequence for item in records] == [0, 1, 2, 3, 4, 5]
    completion = records[-1]
    assert completion.kind == "attempt_finished"
    assert completion.contract_id == "contract-1"
    assert completion.execution_status is ExecutionStatus.SUCCEEDED


def test_a_stale_contract_start_is_corruption_on_reload_after_a_valid_completion(
    trace_path: Path,
) -> None:
    """F01: the allowed historical completion stays valid while the stale start does not."""
    store = JsonlTraceStore(trace_path)
    append_all(
        store,
        started(),
        step(),
        contract(contract_id="contract-1"),
        attempt(sequence=3, contract_id="contract-1", attempt_id="attempt-1"),
        contract(sequence=4, contract_id="contract-2", replaces="contract-1"),
        finished(sequence=5, contract_id="contract-1", attempt_id="attempt-1", status="failed"),
    )
    reloaded = JsonlTraceStore(trace_path).read_run("run-1")
    assert len(reloaded) == 6
    with pytest.raises(TraceValidationError):
        store.append(attempt(sequence=6, contract_id="contract-1", attempt_id="attempt-2"))
    assert len(JsonlTraceStore(trace_path).read_run("run-1")) == 6


def test_a_contract_identifier_is_unique_within_one_run(store: JsonlTraceStore) -> None:
    """F02: the same Contract ID may not be committed twice in one Run, even across steps."""
    append_all(store, started(), step(step_id="step-1"), step(sequence=2, step_id="step-2"),
               contract(sequence=3, step_id="step-1", contract_id="contract-1"))
    with pytest.raises(TraceValidationError):
        store.append(contract(sequence=4, step_id="step-2", contract_id="contract-1"))


def test_a_duplicate_contract_identifier_on_disk_is_corruption(trace_path: Path) -> None:
    """F02: the uniqueness rule is enforced on reload as well as on append."""
    records = [
        _json_line(started()),
        _json_line(step(step_id="step-1")),
        _json_line(step(sequence=2, step_id="step-2")),
        _json_line(contract(sequence=3, step_id="step-1", contract_id="contract-1")),
        _json_line(contract(sequence=4, step_id="step-2", contract_id="contract-1")),
    ]
    trace_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("run-1")
    assert info.value.line_number == 5


def test_the_same_contract_identifier_may_be_used_in_distinct_runs(store: JsonlTraceStore) -> None:
    """F02: uniqueness is per Run, so two Runs may each commit their own contract-1."""
    append_all(
        store,
        started("run-1", 0),
        step("run-1", 1, "step-1"),
        contract("run-1", 2, "step-1", "contract-1"),
        started("run-2", 0, "run-2-start"),
        step("run-2", 1, "step-1"),
        contract("run-2", 2, "step-1", "contract-1"),
    )
    for run_id in ("run-1", "run-2"):
        identifiers = [item.contract_id for item in store.read_run(run_id)
                       if item.kind == "contract_committed"]
        assert identifiers == ["contract-1"]
    assert store.get_record("run-2", "run-2-contract-1").contract_id == "contract-1"


@pytest.mark.parametrize("forgery", [
    {"sequence": True},
    {"schema_version": "9.9"},
    {"kind": "run_ended"},
    {"terminal_status": "success"},
])
def test_an_invalid_copied_record_is_rejected_before_any_write(
    store: JsonlTraceStore, trace_path: Path, forgery: dict
) -> None:
    """F03: class identity is not proof; a copied record is revalidated from raw state."""
    record = started()
    if "terminal_status" in forgery:
        record = _record(RunFinished, record_id="terminal", kind="run_finished",
                         terminal_status="failure")
    forged = record.model_copy(update=forgery)
    assert not trace_path.exists()
    with pytest.raises(TraceValidationError):
        store.append(forged)
    assert not trace_path.exists()


def test_a_forged_record_cannot_start_a_file_and_leave_no_bytes(store: JsonlTraceStore,
                                                                trace_path: Path) -> None:
    """F03: a rejected copy leaves zero bytes, so nothing is persisted."""
    store.append(started())
    before = trace_path.read_bytes()
    forged = step(sequence=1).model_copy(update={"sequence": True})
    with pytest.raises(TraceValidationError) as info:
        store.append(forged)
    assert "sequence" in str(info.value)
    assert trace_path.read_bytes() == before


def test_a_constructed_record_is_revalidated_from_its_raw_fields(store: JsonlTraceStore,
                                                                 trace_path: Path) -> None:
    """F03: model_construct bypasses validation, so ingress must not trust it."""
    forged = RunStarted.model_construct(
        schema_version="0.1", record_id="r", run_id="run-1", sequence=0,
        occurred_at="not-a-timestamp", kind="run_started",
    )
    with pytest.raises(TraceValidationError):
        store.append(forged)
    assert not trace_path.exists()


def test_a_constructed_record_with_an_extra_field_never_persists_it(
    store: JsonlTraceStore, trace_path: Path
) -> None:
    """F03: an unvalidated construction may not smuggle a raw field onto disk."""
    payload = {
        "schema_version": "0.1", "record_id": "r", "run_id": "run-1", "sequence": 0,
        "occurred_at": NOW, "kind": "run_started", "metadata": "SYNTHETIC_PRIVATE_VALUE",
    }
    forged = RunStarted.model_construct(**payload)
    store.append(forged)
    text = trace_path.read_text(encoding="utf-8")
    assert "SYNTHETIC_PRIVATE_VALUE" not in text
    assert "metadata" not in text
    assert json.loads(text)["schema_version"] == "0.1"


def test_a_corrupt_record_error_never_echoes_an_attacker_controlled_key(
    trace_path: Path,
) -> None:
    """F04: the located corruption message keeps a safe location, never the raw key."""
    marker = "SYNTHETIC_PRIVATE_FIELD"
    trace_path.write_text(
        '{"schema_version":"0.1","record_id":"r","run_id":"run-1","sequence":0,'
        '"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started","' + marker + '":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("run-1")
    message = str(info.value)
    assert marker not in message, message
    assert info.value.line_number == 1
    assert message.startswith("invalid record at run_started.<field>"), message


def test_a_public_store_error_never_echoes_a_rejected_value(store: JsonlTraceStore) -> None:
    """F04: a rejected copied value must not be transcribed by the store error either."""
    marker = "SYNTHETIC_PRIVATE_VALUE"
    forged = started().model_copy(update={"schema_version": marker})
    with pytest.raises(TraceValidationError) as info:
        store.append(forged)
    assert marker not in str(info.value)
    assert "schema_version" in str(info.value)


def test_an_oversized_json_integer_is_a_bounded_located_corruption(trace_path: Path) -> None:
    """F05: the decoder's integer limit becomes a located corruption error, not a crash."""
    digits = "1" * 5000
    trace_path.write_text(
        '{"schema_version":"0.1","record_id":"r","run_id":"run-1","sequence":' + digits +
        ',"occurred_at":"2026-10-08T12:00:00Z","kind":"run_started"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("run-1")
    assert info.value.line_number == 1
    assert digits[:20] not in str(info.value)
    assert "number the decoder refuses" in str(info.value)


def test_an_oversized_integer_mid_file_is_located_and_does_not_change_global_limits(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F05: the failure is located, the value is not accepted and no global limit moves."""
    import sys as _sys

    limit_before = _sys.get_int_max_str_digits()
    store.append(started())
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write('{"schema_version":"0.1","record_id":"big","run_id":"run-1","sequence":'
                     + "9" * 6000 + ',"occurred_at":"2026-10-08T12:00:00Z","kind":"step_started",'
                     '"step_id":"step-1"}\n')
    with pytest.raises(TraceCorruptionError) as info:
        store.read_run("run-1")
    assert info.value.line_number == 2
    assert _sys.get_int_max_str_digits() == limit_before


def test_an_incomplete_write_acknowledgement_is_a_storage_error(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F06: a short write is an explicit storage failure, never a successful append."""
    store = JsonlTraceStore(trace_path)

    class ShortWriter(_StandInHandle):
        def write(self, text):
            half = max(1, len(text) // 2)
            self._handle.write(text[:half])
            return half

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: ShortWriter(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    monkeypatch.undo()
    assert info.value.code == "TRACE_STORAGE_FAILURE"
    assert trace_path.exists()
    # Whatever bytes did reach the file stay there: a partial line, never a complete record.
    tail = trace_path.read_bytes()
    assert tail and not tail.endswith(b"\n")
    assert b'"record_id":"run-1-start"' in tail
    with pytest.raises(TraceCorruptionError):
        JsonlTraceStore(trace_path).read_run("run-1")


def test_a_partial_tail_after_an_incomplete_write_is_detected_as_corruption(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F06: whatever bytes did reach the file are reported as corruption, not repaired."""
    store = JsonlTraceStore(trace_path)

    class ShortWriter(_StandInHandle):
        def write(self, text):
            half = max(1, len(text) // 4)
            self._handle.write(text[:half])
            return half

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: ShortWriter(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError):
        store.append(started())
    monkeypatch.undo()
    with pytest.raises(TraceCorruptionError) as info:
        JsonlTraceStore(trace_path).read_run("run-1")
    assert info.value.line_number == 1


def test_a_normal_write_still_reports_success_after_the_short_write_check(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F06 control: an ordinary complete write is still accepted and stays complete."""
    append_all(store, started(), terminal(sequence=1))
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["sequence"] for line in lines] == [0, 1]
    assert store.read_run("run-1")[-1].kind == "run_finished"


# --- round-004 correction: remaining F03, F04 and F06 ------------------------

def test_an_unknown_attribute_smuggled_into_a_record_is_rejected(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03: an attribute the schema does not declare must fail ingress, not be discarded."""
    record = started()
    object.__setattr__(record, "__dict__",
                       {**record.__dict__, "metadata": "SYNTHETIC_PRIVATE_VALUE"})
    with pytest.raises(TraceValidationError) as info:
        store.append(record)
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(info.value)
    assert not trace_path.exists()


def test_a_rejected_unknown_attribute_leaves_the_existing_bytes_unchanged(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03: the rejection happens before the write, so the file keeps its previous bytes."""
    store.append(started())
    before = trace_path.read_bytes()
    record = step(sequence=1)
    object.__setattr__(record, "__dict__",
                       {**record.__dict__, "raw_outcome": "SYNTHETIC_PRIVATE_VALUE"})
    with pytest.raises(TraceValidationError):
        store.append(record)
    assert trace_path.read_bytes() == before
    assert len(store.read_run("run-1")) == 1


def test_an_unknown_attribute_makes_an_instance_unacceptable_to_parse_record() -> None:
    """F03: parse_record revalidates an instance from its detached state as well."""
    record = started()
    object.__setattr__(record, "__dict__",
                       {**record.__dict__, "metadata": "SYNTHETIC_PRIVATE_VALUE"})
    with pytest.raises(TraceValidationError) as info:
        parse_record(record)
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(info.value)


@pytest.mark.parametrize("bad_kind", [[], {}, 7, True, None, 3.5])
def test_a_copied_non_string_kind_is_a_bounded_rejection(
    store: JsonlTraceStore, trace_path: Path, bad_kind: object,
) -> None:
    """F03: a non-string or unhashable kind is rejected, never a raw lookup error."""
    forged = started().model_copy(update={"kind": bad_kind})
    with pytest.raises(TraceValidationError) as info:
        store.append(forged)
    assert "kind" in str(info.value)
    assert not trace_path.exists()


def test_a_kind_that_disagrees_with_its_own_class_is_rejected(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03: the declared kind and the instance class must agree before validation."""
    forged = started().model_copy(update={"kind": "step_started"})
    with pytest.raises(TraceValidationError):
        store.append(forged)
    assert not trace_path.exists()


def test_an_ordinary_validated_copy_is_still_accepted(store: JsonlTraceStore) -> None:
    """F03 control: revalidating from detached state must not reject a legitimate snapshot."""
    append_all(store, started(), step())
    record = store.get_record("run-1", "run-1-step-step-1")
    assert record.kind == "step_started"
    assert record.step_id == "step-1"


@pytest.mark.parametrize("acknowledged", [None, True, False, 3.5, "12", -1, 10**9])
def test_a_bad_write_acknowledgement_is_a_storage_error(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch, acknowledged: object,
) -> None:
    """F06: only a complete integer character count may confirm a successful write."""
    store = JsonlTraceStore(trace_path)

    class BadAck(_StandInHandle):
        def write(self, text):
            self._handle.write(text)
            return acknowledged

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: BadAck(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    monkeypatch.undo()
    assert info.value.code == "TRACE_STORAGE_FAILURE"
    assert str(acknowledged) not in str(info.value)
    assert "partly written" in str(info.value)


def test_an_incorrect_integer_acknowledgement_is_a_storage_error(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F06: a count that is not the supplied length is an incomplete acknowledgement."""
    store = JsonlTraceStore(trace_path)

    class OffsetAck(_StandInHandle):
        def write(self, text):
            self._handle.write(text)
            return len(text) + 1

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: OffsetAck(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    monkeypatch.undo()
    assert "partly written" in str(info.value)
    # The acknowledgement was wrong, but the bytes that reached the file are a complete line:
    # the store never claims success and never repairs, so the next read reports what is there.
    assert trace_path.read_bytes().endswith(b"\n")
    reloaded = JsonlTraceStore(trace_path).read_run("run-1")
    assert [item.record_id for item in reloaded] == ["run-1-start"]


def test_a_bad_acknowledgement_preserves_the_original_io_cause(
    trace_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F06: an acknowledgement failure keeps its I/O context instead of a bare error."""
    store = JsonlTraceStore(trace_path)

    class ExplodingAck(_StandInHandle):
        def write(self, text):
            raise OSError("synthetic acknowledgement refusal")

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: ExplodingAck(REAL_OPEN(path, *a, **k)),
                        raising=True)
    with pytest.raises(TraceStorageError) as info:
        store.append(started())
    assert isinstance(info.value.__cause__, OSError)
    assert "synthetic acknowledgement refusal" not in str(info.value)


def test_a_complete_real_character_count_is_still_accepted(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F06 control: the ordinary text-mode write returns the full count and stays successful."""
    append_all(store, started(), terminal(sequence=1))
    payloads = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert [item["kind"] for item in payloads] == ["run_started", "run_finished"]
    assert trace_path.read_bytes().endswith(b"\n")


# --- round-005 correction: remaining F03 -------------------------------------

def test_a_copied_string_status_is_consumed_as_the_fresh_validated_candidate(
    store: JsonlTraceStore,
) -> None:
    """F03: a copied record is used only after revalidation, never as a raw string carrier.

    The declared contract accepts the JSON string form of a status, so the copied value is not
    rejected here: what matters is that the freshly validated record is what the association
    checks and the write consume, so no raw ``AttributeError`` can escape.
    """
    append_all(store, started())
    record = verification(sequence=1, scope="run", status="pass", record_id="verification-pass")
    forged = record.model_copy(update={"verification_status": "pass"})
    store.append(forged)
    stored = store.get_record("run-1", "verification-pass")
    assert stored.verification_status is VerificationStatus.PASS
    assert isinstance(stored.verification_status, VerificationStatus)


@pytest.mark.parametrize("name", ["__metadata", "_metadata", "__SYNTHETIC_PRIVATE"])
def test_a_copied_private_or_dunder_attribute_is_rejected_before_disk_changes(
    store: JsonlTraceStore, trace_path: Path, name: str,
) -> None:
    """F03: an unknown copied attribute is rejected with no blanket private-name exemption."""
    forged = started().model_copy(update={name: "SYNTHETIC_PRIVATE_VALUE"})
    assert name in forged.__dict__
    with pytest.raises(TraceValidationError) as info:
        store.append(forged)
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(info.value)
    assert not trace_path.exists()


def test_a_copied_private_attribute_leaves_existing_bytes_unchanged(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03: the rejection happens before the write, so previously recorded bytes survive."""
    store.append(started())
    before = trace_path.read_bytes()
    forged = step(sequence=1).model_copy(update={"__raw_outcome": "SYNTHETIC_PRIVATE_VALUE"})
    with pytest.raises(TraceValidationError):
        store.append(forged)
    assert trace_path.read_bytes() == before


def test_parse_record_rejects_a_foreign_model_instance() -> None:
    """F03: an arbitrary BaseModel must not come back as a member of the closed union."""
    from forecontract.models.actions import ActionSpec

    foreign = ActionSpec(schema_version="0.1", action_type="click", payload={"a": 1})
    with pytest.raises(TraceValidationError) as info:
        parse_record(foreign)
    assert "closed union" in str(info.value)


def test_parse_record_still_accepts_a_real_record_instance(store: JsonlTraceStore) -> None:
    """F03 control: a legitimate record instance is still accepted through the union."""
    record = started()
    reloaded = parse_record(record)
    assert type(reloaded) is RunStarted
    assert reloaded.record_id == record.record_id
    assert reloaded.sequence == record.sequence
    store.append(reloaded)
    assert store.read_run("run-1")[0].record_id == "run-1-start"


def test_a_normal_snapshot_and_enum_still_survive_the_round_005_ingress(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03 control: an ordinary snapshot and enum member are accepted and preserved."""
    append_all(store, started(), step(), contract(), attempt(),
               finished(status="succeeded"),
               verification(sequence=5, scope="run", status="inconclusive",
                            record_id="verification-run"))
    records = store.read_run("run-1")
    assert [item.sequence for item in records] == [0, 1, 2, 3, 4, 5]
    assert records[4].execution_status is ExecutionStatus.SUCCEEDED
    assert records[5].verification_status is VerificationStatus.INCONCLUSIVE
    assert trace_path.read_bytes().endswith(b"\n")


# --- round-006 correction: remaining F03 raw stored-key whitelist -------------

#: The three raw key classes the approved contract names explicitly, as ordinary copies.
RAW_KEY_CASES = [
    ("dunder-shaped name", "__metadata__"),
    ("invented internal name", "__pydantic_synthetic_payload"),
    ("non-string key", 17),
]

#: The three real pydantic bookkeeping names, injected as ordinary ``__dict__`` entries.
BOOKKEEPING_NAME_CASES = [
    ("fields_set entry", "__pydantic_fields_set__"),
    ("extra entry", "__pydantic_extra__"),
    ("private entry", "__pydantic_private__"),
]

ALL_RAW_KEY_CASES = RAW_KEY_CASES + BOOKKEEPING_NAME_CASES
ALL_RAW_KEY_IDS = [case[0] for case in ALL_RAW_KEY_CASES]
ALL_RAW_KEY_VALUES = [case[1] for case in ALL_RAW_KEY_CASES]


@pytest.mark.parametrize("key", ALL_RAW_KEY_VALUES, ids=ALL_RAW_KEY_IDS)
def test_an_injected_dictionary_entry_is_rejected_before_any_disk_change(
    store: JsonlTraceStore, trace_path: Path, key: object,
) -> None:
    """F03: every ordinary-copy injected __dict__ entry is rejected, not skipped."""
    store.append(started())
    before = trace_path.read_bytes()
    forged = step(sequence=1).model_copy(update={key: "SYNTHETIC_PRIVATE_VALUE"})
    assert key in forged.__dict__
    with pytest.raises(TraceValidationError) as info:
        store.append(forged)
    message = str(info.value)
    assert "SYNTHETIC_PRIVATE_VALUE" not in message
    assert trace_path.read_bytes() == before


@pytest.mark.parametrize("key", ALL_RAW_KEY_VALUES, ids=ALL_RAW_KEY_IDS)
def test_an_injected_dictionary_entry_makes_parse_record_reject_without_writing(key: object) -> None:
    """F03: the same strict gate applies to parse_record, and it never writes bytes."""
    forged = started().model_copy(update={key: "SYNTHETIC_PRIVATE_VALUE"})
    with pytest.raises(TraceValidationError) as info:
        parse_record(forged)
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(info.value)
    assert "declared field" in str(info.value)


@pytest.mark.parametrize("key", ALL_RAW_KEY_VALUES, ids=ALL_RAW_KEY_IDS)
def test_an_injected_dictionary_entry_is_rejected_without_echoing_it(
    store: JsonlTraceStore, key: object,
) -> None:
    """F03/F04: the bounded rejection never transcribes the rejected key itself."""
    name = str(key)
    forged = started().model_copy(update={key: name})
    with pytest.raises(TraceValidationError) as info:
        store.append(forged)
    message = str(info.value)
    assert name not in message, message
    assert message == "the record carries a stored key that is not a declared field"


def test_genuine_pydantic_bookkeeping_is_recognised_by_its_exact_name(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03 control: a real instance keeps working because its bookkeeping keys are exact."""
    plain = started()
    assert set(plain.__dict__) == {name for name in type(plain).model_fields}
    assert "__pydantic_fields_set__" not in plain.__dict__
    fields_set = plain.__pydantic_fields_set__
    assert "record_id" in fields_set and "sequence" in fields_set
    store.append(plain)
    stored = store.get_record("run-1", "run-1-start")
    assert stored.record_id == "run-1-start"
    assert stored.sequence == 0
    assert stored.occurred_at == NOW
    assert stored.kind == "run_started"
    assert trace_path.read_bytes().endswith(b"\n")


def test_an_invented_internal_sibling_is_not_treated_as_bookkeeping(
    store: JsonlTraceStore,
) -> None:
    """F03: an exact-name allowlist rejects an invented internal name that only looks like one."""
    forged = started().model_copy(update={"__pydantic_synthetic_payload": {"a": 1}})
    assert "__pydantic_synthetic_payload" in forged.__dict__
    with pytest.raises(TraceValidationError):
        store.append(forged)


def test_a_bookkeeping_name_used_as_a_dictionary_entry_is_rejected(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03: a real bookkeeping name is only bookkeeping in its slot, never as a __dict__ entry.

    The previous expectation here accepted a manually replaced ``__dict__`` that carried
    ``__pydantic_fields_set__``; that state is not produced by pydantic, so accepting it was
    itself the defect.
    """
    forged = started()
    object.__setattr__(
        forged, "__dict__", {**forged.__dict__, "__pydantic_fields_set__": {"record_id"}}
    )
    assert "__pydantic_fields_set__" in forged.__dict__
    with pytest.raises(TraceValidationError) as info:
        store.append(forged)
    assert "declared field" in str(info.value)
    assert not trace_path.exists()


def test_genuine_bookkeeping_keeps_working_from_its_real_slot(
    store: JsonlTraceStore, trace_path: Path,
) -> None:
    """F03 control: the gate reads __dict__, so pydantic's real slot state is untouched."""
    plain = started()
    assert set(plain.__dict__) == {name for name in type(plain).model_fields}
    assert "__pydantic_fields_set__" not in plain.__dict__
    assert plain.__pydantic_fields_set__ == {
        "schema_version", "record_id", "run_id", "sequence", "occurred_at", "kind",
    }
    assert plain.__pydantic_extra__ is None
    assert plain.__pydantic_private__ is None
    store.append(plain)
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version", "record_id", "run_id", "sequence", "occurred_at", "kind",
    }
    assert "pydantic" not in json.dumps(payload)


def test_the_raw_key_gate_still_accepts_every_event_variant(store: JsonlTraceStore) -> None:
    """F03 control: the strict key gate keeps every legitimate event snapshot acceptable."""
    append_all(
        store,
        started(),
        step(),
        contract(),
        attempt(),
        finished(),
        verification(sequence=5, scope="attempt", status="pass", step_id="step-1",
                     contract_id="contract-1", attempt_id="attempt-1"),
        terminal(sequence=6, status="failure"),
    )
    records = store.read_run("run-1")
    assert [item.kind for item in records] == [
        "run_started", "step_started", "contract_committed", "attempt_started",
        "attempt_finished", "verification_recorded", "run_finished",
    ]
    assert records[5].verification_status is VerificationStatus.PASS
    assert records[6].terminal_status == "failure"
