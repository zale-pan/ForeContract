"""Single-writer local JSONL storage for structured traces.

:class:`JsonlTraceStore` is the only implementation of the tracing protocols in this round. It
is deliberately unglamorous: one file, one line per record, no background worker, no lock, no
cache and no automatic recovery.

Reading validates the whole file, never just the prefix up to a match, so a corrupt line after
the requested record still fails. A missing file reads as *not found*; an empty file simply has
no runs; a blank line, a duplicate JSON key, a non-finite number, a truncated line or a final
line without a newline is corruption and is reported with its line number instead of being
skipped, truncated or repaired.

Writing validates the entire existing file and the candidate first, then writes one complete
line, flushes and closes. A failed operating-system write may leave a partial tail; that tail
is never hidden, never repaired and never advances any in-process state, and the next read
reports it as corruption. Errors keep the original exception as ``__cause__`` while the public
message stays fixed and bounded.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from forecontract.tracing.errors import (
    TraceCorruptionError,
    TraceNotFoundError,
    TraceStorageError,
    TraceValidationError,
)
from forecontract.tracing.models import (
    EVENT_MODELS,
    AttemptFinished,
    AttemptStarted,
    ContractCommitted,
    RunFinished,
    RunStarted,
    StepStarted,
    TraceRecord,
    VerificationRecorded,
    bounded_schema_message,
    detached_state,
    parse_record,
)

__all__ = ["JsonlTraceStore"]


@dataclass
class _AttemptState:
    """What the store knows about one started attempt."""

    step_id: str
    contract_id: str
    finished: bool = False


@dataclass
class _RunState:
    """The validated state of one run in one file."""

    next_sequence: int = 0
    started: bool = False
    steps: set[str] = field(default_factory=set)
    contracts: dict[str, list[str]] = field(default_factory=dict)
    contract_ids: set[str] = field(default_factory=set)
    attempts: dict[str, _AttemptState] = field(default_factory=dict)
    verification_scope: dict[str, str] = field(default_factory=dict)
    run_scope_passes: set[str] = field(default_factory=set)
    terminal: str | None = None
    records: list[TraceRecord] = field(default_factory=list)

    def current_contract(self, step_id: str) -> str | None:
        versions = self.contracts.get(step_id)
        return versions[-1] if versions else None

    def step_of_contract(self, contract_id: str) -> str | None:
        for step_id, versions in self.contracts.items():
            if contract_id in versions:
                return step_id
        return None


def _reject(message: str, line_number: int | None = None, *, corrupt: bool = False) -> None:
    """Raise the failure class that matches where the rule was checked."""
    if corrupt:
        raise TraceCorruptionError(message, line_number=line_number)
    raise TraceValidationError(message)


class JsonlTraceStore:
    """A single-writer JSONL trace store.

    Construction and reading are side-effect free: the file is never created or touched by
    the constructor or by a query. The parent directory must already exist and is never
    created; the first successful append may create the file.
    """

    def __init__(self, path: str | Path) -> None:
        try:
            self._path = Path(path)
        except (TypeError, ValueError) as exc:
            raise TraceStorageError("the trace path is not a usable filesystem path", cause=exc) from exc

    @property
    def path(self) -> Path:
        """The configured trace path. Returned as a plain path, never as an open handle."""
        return self._path

    # ------------------------------------------------------------------ writing

    def append(self, record: TraceRecord) -> None:
        """Validate the recorded history and ``record``, then write one complete line.

        The candidate must be an already validated record of the closed union. A record that
        would break sequence continuity, entity association, the single terminal rule or the
        success-reference rule is rejected before anything is written.
        """
        self._assert_known_record(record)
        # A validated instance of the closed union is not proof on its own: ``model_copy`` and
        # ``model_construct`` can produce an object of the right class whose raw fields were
        # never checked, or whose raw representation is not the declared one at all. The
        # candidate is therefore re-validated from its detached raw state before any association
        # check, normalisation or write, and the freshly validated record is the one used from
        # here on (F03).
        candidate = self._validated_candidate(record)
        # Validating the whole existing file first is what makes a bad tail a rejection
        # instead of an append onto corruption. The candidate is checked as if it came last,
        # so the history it is checked against is exactly the history already on disk.
        self._validated_states(extra=candidate)
        payload = candidate.model_dump(mode="json")
        try:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise TraceValidationError("the record could not be encoded as one JSON line", cause=exc) from exc
        self._write_line(encoded + "\n")

    @staticmethod
    def _assert_known_record(record: object) -> None:
        if not isinstance(
            record,
            (
                RunStarted,
                StepStarted,
                ContractCommitted,
                AttemptStarted,
                AttemptFinished,
                VerificationRecorded,
                RunFinished,
            ),
        ):
            raise TraceValidationError("the appended value is not a validated trace record")

    @staticmethod
    def _validated_candidate(record: TraceRecord) -> TraceRecord:
        """Re-validate one candidate from its detached raw fields and return the fresh record.

        A copied or constructed instance of the correct class can carry a boolean sequence, an
        unknown schema version, an unknown kind, an invalid cross-field combination, an enum
        value that is only a plain string, or an attribute the declared schema does not know.
        The state is read from the instance's own storage rather than from a normalized dump,
        so an unknown attribute is part of what has to pass validation instead of being
        silently discarded, and the check rejects exactly what the reload path would call
        corrupt. No bytes are touched (F03).
        """
        kind = record.kind
        if not isinstance(kind, str) or kind not in EVENT_MODELS:
            # A non-string or unhashable kind must be a bounded rejection, never a raw lookup
            # error escaping from the store (F03).
            raise TraceValidationError("the record kind is not a known event variant")
        model = EVENT_MODELS[kind]
        if not isinstance(record, model):
            # The declared kind and the instance class must agree, otherwise the record would
            # be validated against a schema it does not claim.
            raise TraceValidationError("the record kind does not match its own record type")
        state = detached_state(record)
        try:
            return model.model_validate(state)
        except ValidationError as exc:
            raise TraceValidationError(bounded_schema_message(exc), cause=exc) from exc
        except (TypeError, ValueError) as exc:
            raise TraceValidationError(
                "the record could not be validated from its own fields", cause=exc
            ) from exc

    def _write_line(self, text: str) -> None:
        """Write one complete record and close it, mapping every failure to a safe error.

        A context manager is used so the file is closed on every path, including the failure
        paths; the surrounding ``except`` only translates the operating-system error, it never
        retries or repairs. A write that reports fewer characters than were supplied is an
        incomplete acknowledgement, so it is an explicit storage failure and never a success
        (F06); whatever bytes did reach the file stay there and the next read reports the
        partial tail as corruption.
        """
        try:
            with open(self._path, "a", encoding="utf-8", newline="", closefd=True) as handle:
                written = handle.write(text)
                if not self._is_complete_write(written, text):
                    raise TraceStorageError("the trace record was only partly written")
                handle.flush()
        except TraceStorageError:
            raise
        except OSError as exc:
            raise TraceStorageError("the trace file could not be written", cause=exc) from exc
        except ValueError as exc:
            # Text-mode encoding failures surface as ValueError; they are storage failures too.
            raise TraceStorageError("the trace record could not be written", cause=exc) from exc

    @staticmethod
    def _is_complete_write(acknowledged: object, text: str) -> bool:
        """Accept only a real integer character count equal to the supplied length.

        A write that acknowledges ``None``, a boolean, a float, a string or the wrong count has
        not confirmed that the whole record reached the file, so it is an incomplete
        acknowledgement rather than a success (F06).
        """
        if isinstance(acknowledged, bool) or not isinstance(acknowledged, int):
            return False
        return acknowledged == len(text)

    # ------------------------------------------------------------------ reading

    def get_record(self, run_id: str, record_id: str) -> TraceRecord:
        """Return one detached record of ``run_id``.

        A missing run or a missing record is ``TraceNotFoundError``. A record identifier that
        belongs to another run is not returned as a source of this run.
        """
        wanted_run = self._require_identifier(run_id, "run_id")
        wanted_record = self._require_identifier(record_id, "record_id")
        states, _ = self._validated_states()
        state = states.get(wanted_run)
        if state is None or not state.started:
            raise TraceNotFoundError("the requested run is not present in this trace")
        for record in state.records:
            if record.record_id == wanted_record:
                return record.model_copy(deep=True)
        raise TraceNotFoundError("the requested record is not present in this run")

    def read_run(self, run_id: str) -> tuple[TraceRecord, ...]:
        """Return every record of ``run_id`` in ascending sequence order."""
        wanted_run = self._require_identifier(run_id, "run_id")
        states, _ = self._validated_states()
        state = states.get(wanted_run)
        if state is None or not state.started:
            raise TraceNotFoundError("the requested run is not present in this trace")
        ordered = sorted(state.records, key=lambda item: item.sequence)
        return tuple(record.model_copy(deep=True) for record in ordered)

    @staticmethod
    def _require_identifier(value: object, field_name: str) -> str:
        if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
            raise TraceValidationError(f"the {field_name} query argument must be a non-blank string")
        return value

    # ------------------------------------------------------------------ internals

    def _read_lines(self) -> list[str]:
        """Read and decode the file, mapping every operating-system failure safely.

        A missing file is not an error here: it is the empty history the first append is
        allowed to create and the empty state a query turns into ``TraceNotFoundError``.
        """
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise TraceStorageError("the trace file could not be read", cause=exc) from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise TraceCorruptionError("the trace file is not valid UTF-8") from None
        if not text:
            return []
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        else:
            raise TraceCorruptionError(
                "the trace file ends with an incomplete record", line_number=len(lines)
            )
        return lines

    def _validated_states(self, extra: TraceRecord | None = None) -> tuple[dict[str, _RunState], list[TraceRecord]]:
        """Validate the whole file, optionally as if ``extra`` had been appended last."""
        lines = self._read_lines()
        records: list[TraceRecord] = []
        states: dict[str, _RunState] = {}
        global_ids: dict[str, str] = {}
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                _reject("the trace file contains a blank line", number, corrupt=True)
            parsed = self._decode_line(line, number)
            try:
                record = parse_record(parsed)
            except TraceValidationError as exc:
                _reject(exc.message, number, corrupt=True)
                raise  # pragma: no cover - _reject always raises
            self._check_record(record, states, global_ids, number, corrupt=True)
            records.append(record)
        if extra is not None:
            self._check_record(extra, states, global_ids, None, corrupt=False)
        return states, records

    @staticmethod
    def _decode_line(line: str, number: int) -> Any:
        """Decode one line, rejecting duplicate keys and oversized integers as corruption."""
        try:
            return json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except TraceCorruptionError as exc:
            # A hook rejection cannot know the physical line, so the reader adds it here.
            raise TraceCorruptionError(exc.message, line_number=number) from None
        except json.JSONDecodeError:
            raise TraceCorruptionError("the trace line is not valid JSON", line_number=number) from None
        except ValueError:
            # ``json`` also raises a plain ValueError for an integer literal the interpreter's
            # digit limit refuses. That is a malformed line, not a crash: it becomes a bounded
            # located corruption error, the value is never accepted and no global limit is
            # changed (F05).
            raise TraceCorruptionError(
                "the trace line contains a number the decoder refuses", line_number=number
            ) from None

    def _check_record(
        self,
        record: TraceRecord,
        states: dict[str, _RunState],
        global_ids: dict[str, str],
        number: int | None,
        *,
        corrupt: bool,
    ) -> None:
        """Apply every association and continuity rule to one record, in order."""
        state = states.get(record.run_id)
        if state is None:
            state = _RunState()
            states[record.run_id] = state

        if state.terminal is not None:
            _reject("the run already has a terminal fact and accepts no further records", number, corrupt=corrupt)
        if record.record_id in global_ids:
            _reject("the record identifier is already used in this trace", number, corrupt=corrupt)
        if record.sequence != state.next_sequence:
            _reject("the record sequence is not the next number for this run", number, corrupt=corrupt)
        if isinstance(record, RunStarted):
            if state.started:
                _reject("the run has already started", number, corrupt=corrupt)
        elif not state.started:
            _reject("the run has no start record before this fact", number, corrupt=corrupt)

        if isinstance(record, StepStarted):
            if record.step_id in state.steps:
                _reject("the step identifier is already used in this run", number, corrupt=corrupt)
            state.steps.add(record.step_id)
            state.contracts.setdefault(record.step_id, [])
        elif isinstance(record, ContractCommitted):
            if record.step_id not in state.steps:
                _reject("the contract refers to a step this run has not started", number, corrupt=corrupt)
            if record.contract_id in state.contract_ids:
                _reject("the contract identifier is already used in this run", number, corrupt=corrupt)
            versions = state.contracts.setdefault(record.step_id, [])
            current = versions[-1] if versions else None
            if record.contract_id == record.replaces_contract_id:
                _reject("a contract cannot replace itself", number, corrupt=corrupt)
            if record.replaces_contract_id is None:
                if current is not None:
                    _reject("the first contract of a step cannot be a silent replacement", number, corrupt=corrupt)
            elif record.replaces_contract_id != current:
                _reject("the replaced contract is not the current version of this step", number, corrupt=corrupt)
            versions.append(record.contract_id)
            state.contract_ids.add(record.contract_id)
        elif isinstance(record, AttemptStarted):
            self._check_attempt_start(record, state, number, corrupt=corrupt)
        elif isinstance(record, AttemptFinished):
            attempt = state.attempts.get(record.attempt_id)
            if attempt is None:
                _reject("the attempt was never started in this contract", number, corrupt=corrupt)
            elif attempt.step_id != record.step_id or attempt.contract_id != record.contract_id:
                _reject("the finished attempt does not match the started attempt", number, corrupt=corrupt)
            elif attempt.finished:
                _reject("the attempt already has a completion fact", number, corrupt=corrupt)
            else:
                attempt.finished = True
        elif isinstance(record, VerificationRecorded):
            self._check_verification(record, state, number, corrupt=corrupt)
        elif isinstance(record, RunFinished):
            self._check_terminal(record, state, number, corrupt=corrupt)

        global_ids[record.record_id] = record.run_id
        state.records.append(record)
        state.started = state.started or isinstance(record, RunStarted)
        state.next_sequence += 1

    @staticmethod
    def _check_attempt_start(
        record: AttemptStarted, state: _RunState, number: int | None, *, corrupt: bool
    ) -> None:
        if record.step_id not in state.steps:
            _reject("the attempt refers to a step this run has not started", number, corrupt=corrupt)
        step_of_contract = state.step_of_contract(record.contract_id)
        if step_of_contract is None:
            _reject("the attempt refers to a contract this run has not committed", number, corrupt=corrupt)
        if step_of_contract != record.step_id:
            _reject("the attempt contract belongs to a different step", number, corrupt=corrupt)
        if state.current_contract(record.step_id) != record.contract_id:
            # A retry keeps the current Contract and opens a new Attempt. After a replan the
            # replaced version is historical: a new Attempt must not start against it (F01).
            _reject("the attempt does not start against the current contract of this step",
                    number, corrupt=corrupt)
        if record.attempt_id in state.attempts:
            _reject("the attempt identifier is already used in this run", number, corrupt=corrupt)
        state.attempts[record.attempt_id] = _AttemptState(record.step_id, record.contract_id)

    @staticmethod
    def _check_verification(
        record: VerificationRecorded, state: _RunState, number: int | None, *, corrupt: bool
    ) -> None:
        if record.scope == "attempt":
            attempt = state.attempts.get(record.attempt_id)
            if attempt is None:
                _reject("the verification refers to an attempt this run never started", number, corrupt=corrupt)
            elif not attempt.finished:
                _reject("the verification refers to an attempt that has not finished", number, corrupt=corrupt)
            elif attempt.step_id != record.step_id or attempt.contract_id != record.contract_id:
                _reject("the verification does not match the attempt it refers to", number, corrupt=corrupt)
        state.verification_scope[record.record_id] = record.scope
        if record.scope == "run" and record.verification_status.value == "pass":
            state.run_scope_passes.add(record.record_id)

    @staticmethod
    def _check_terminal(
        record: RunFinished, state: _RunState, number: int | None, *, corrupt: bool
    ) -> None:
        if record.verification_record_id is not None:
            if state.verification_scope.get(record.verification_record_id) != "run":
                _reject("the terminal reference is not a run-scoped verification of this run", number, corrupt=corrupt)
            if record.terminal_status == "success" and record.verification_record_id not in state.run_scope_passes:
                _reject("a successful terminal fact must reference a run-scoped pass record", number, corrupt=corrupt)
        state.terminal = record.terminal_status


def _reject_constant(name: str) -> Any:
    """Reject ``NaN``, ``Infinity`` and ``-Infinity`` before they can enter a mapping."""
    raise TraceCorruptionError("the trace line contains a non-finite number")


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while refusing a key that appears twice in the same object."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceCorruptionError("the trace line repeats a JSON object key")
        result[key] = value
    return result
