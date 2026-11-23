"""Unit tests for the read-only source-verified DerivedMemoryIndex."""

from __future__ import annotations

import copy
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

import forecontract.memory as memory_pkg
from forecontract.memory import (
    JsonlEpisodicMemory,
    MemoryCorruptionError,
    MemoryItem,
    MemorySourceError,
    MemoryValidationError,
    TraceSourceResolver,
)
from forecontract.memory.derived import DerivedMemoryIndex
from forecontract.tracing import (
    AttemptFinished,
    AttemptStarted,
    ContractCommitted,
    JsonlTraceStore,
    RunFinished,
    RunStarted,
    StepStarted,
    VerificationRecorded,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_TIME = datetime(2026, 11, 13, 0, 0, tzinfo=timezone.utc)
FUTURE_TIME = BASE_TIME + timedelta(days=3)
QUERY_NOW = BASE_TIME + timedelta(days=1)


class _StrSub(str):
    pass


class _ListSub(list):
    pass


class _DictSub(dict):
    pass


class _IntSub(int):
    pass


class _DatetimeSub(datetime):
    pass


@pytest.fixture
def real_store(tmp_path: Path) -> tuple[JsonlEpisodicMemory, JsonlTraceStore]:
    """Create a real trace store and episodic memory with run and attempt sources."""
    resolved = tmp_path.resolve()
    assert REPO_ROOT not in resolved.parents and resolved != REPO_ROOT
    trace = JsonlTraceStore(tmp_path / f"trace-{uuid.uuid4().hex[:8]}.jsonl")
    memory = JsonlEpisodicMemory(
        tmp_path / f"memory-{uuid.uuid4().hex[:8]}.jsonl",
        source_resolver=TraceSourceResolver(trace),
    )

    run_cases = [
        ("run-good", "success", BASE_TIME),
        ("run-other", "success", BASE_TIME),
        ("run-bad", "failure", BASE_TIME),
        ("run-unknown", "unknown", BASE_TIME),
        ("run-cancel", "cancelled", BASE_TIME),
        ("run-timeout", "timeout", BASE_TIME),
        ("run-abort", "aborted", BASE_TIME),
        ("run-future", "success", FUTURE_TIME),
    ]
    for run_id, outcome, ts in run_cases:
        trace.append(
            RunStarted(
                schema_version="0.1",
                record_id=f"{run_id}-start",
                run_id=run_id,
                sequence=0,
                occurred_at=ts,
            )
        )
        if outcome == "unknown":
            rec_id = f"{run_id}-start"
        else:
            seq = 1
            ver_id: str | None = None
            if outcome == "success":
                ver_id = f"{run_id}-verify"
                trace.append(
                    VerificationRecorded(
                        schema_version="0.1",
                        record_id=ver_id,
                        run_id=run_id,
                        sequence=seq,
                        occurred_at=ts,
                        scope="run",
                        verification_status="pass",
                    )
                )
                seq += 1
            rec_id = f"{run_id}-end"
            trace.append(
                RunFinished(
                    schema_version="0.1",
                    record_id=rec_id,
                    run_id=run_id,
                    sequence=seq,
                    occurred_at=ts,
                    terminal_status=outcome,
                    verification_record_id=ver_id,
                )
            )
        memory.append(
            MemoryItem(
                schema_version="0.1",
                memory_id=f"mem-{run_id}",
                run_id=run_id,
                source_level="run",
                source_record_id=rec_id,
                result=outcome,
                created_at=ts,
            )
        )

    attempt_cases = [
        ("attempt-good", "succeeded", "success", BASE_TIME),
        ("attempt-fail", "failed", "failure", BASE_TIME),
        ("attempt-unknown", "unknown", "unknown", BASE_TIME),
    ]
    for idx, (tag, exec_status, mem_result, ts) in enumerate(attempt_cases, start=1):
        run_id = f"run-att-{idx}"
        step_id = "step-1"
        contract_id = f"contract-{idx}"
        attempt_id = f"att-{idx}"
        trace.append(
            RunStarted(
                schema_version="0.1",
                record_id=f"{run_id}-start",
                run_id=run_id,
                sequence=0,
                occurred_at=ts,
            )
        )
        trace.append(
            StepStarted(
                schema_version="0.1",
                record_id=f"{run_id}-step",
                run_id=run_id,
                sequence=1,
                occurred_at=ts,
                step_id=step_id,
            )
        )
        trace.append(
            ContractCommitted(
                schema_version="0.1",
                record_id=f"{run_id}-contract",
                run_id=run_id,
                sequence=2,
                occurred_at=ts,
                step_id=step_id,
                contract_id=contract_id,
            )
        )
        trace.append(
            AttemptStarted(
                schema_version="0.1",
                record_id=f"{run_id}-att-start",
                run_id=run_id,
                sequence=3,
                occurred_at=ts,
                step_id=step_id,
                contract_id=contract_id,
                attempt_id=attempt_id,
            )
        )
        end_id = f"{run_id}-att-end"
        trace.append(
            AttemptFinished(
                schema_version="0.1",
                record_id=end_id,
                run_id=run_id,
                sequence=4,
                occurred_at=ts,
                step_id=step_id,
                contract_id=contract_id,
                attempt_id=attempt_id,
                execution_status=exec_status,
            )
        )
        memory.append(
            MemoryItem(
                schema_version="0.1",
                memory_id=f"mem-{tag}",
                run_id=run_id,
                source_level="attempt",
                source_record_id=end_id,
                result=mem_result,
                created_at=ts,
            )
        )

    return memory, trace


def _run_entry(
    entry_id: str = "entry-a",
    *,
    run_id: str = "run-good",
    kind: str = "semantic",
    tags: list[str] | None = None,
    expires_at: str | None = None,
) -> dict[str, object]:
    rec_id = f"{run_id}-start" if run_id == "run-unknown" else f"{run_id}-end"
    return {
        "schema_version": "0.1",
        "entry_id": entry_id,
        "kind": kind,
        "memory_id": f"mem-{run_id}",
        "run_id": run_id,
        "source_record_id": rec_id,
        "source_level": "run",
        "tags": ["login"] if tags is None else tags,
        "text": "Host semantic note" if kind == "semantic" else None,
        "steps": [] if kind == "semantic" else ["Inspect dialog", "Submit credential"],
        "expires_at": expires_at,
    }


def test_root_exports_unchanged_and_index_has_no_mutation_methods(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
) -> None:
    memory, _ = real_store
    assert len(memory_pkg.__all__) == 14
    assert "DerivedMemoryIndex" not in memory_pkg.__all__
    idx = DerivedMemoryIndex(episodic_memory=memory, entries=[])
    for forbidden in ("append", "delete", "save", "execute"):
        assert not hasattr(idx, forbidden)
    res = idx.query(tags=["login"], now=QUERY_NOW)
    assert res == {"schema_version": "0.1", "hits": [], "excluded": []}


def test_accepts_run_and_attempt_real_sources(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
) -> None:
    memory, _ = real_store
    entries = [
        _run_entry("run-sem", run_id="run-good", tags=["login", "form"]),
        {
            "schema_version": "0.1",
            "entry_id": "att-proc",
            "kind": "procedural",
            "memory_id": "mem-attempt-good",
            "run_id": "run-att-1",
            "source_record_id": "run-att-1-att-end",
            "source_level": "attempt",
            "tags": ["login"],
            "text": None,
            "steps": ["Wait for form ready", "Fill password"],
            "expires_at": None,
        },
        {
            "schema_version": "0.1",
            "entry_id": "att-bad",
            "kind": "procedural",
            "memory_id": "mem-attempt-fail",
            "run_id": "run-att-2",
            "source_record_id": "run-att-2-att-end",
            "source_level": "attempt",
            "tags": ["login", "form"],
            "text": None,
            "steps": ["Should be excluded"],
            "expires_at": None,
        },
    ]
    idx = DerivedMemoryIndex(episodic_memory=memory, entries=entries)
    res = idx.query(tags=["login", "form"], now=QUERY_NOW)
    assert [hit["entry"]["entry_id"] for hit in res["hits"]] == ["run-sem", "att-proc"]
    assert [hit["score"] for hit in res["hits"]] == [1.0, 0.5]
    assert res["hits"][0]["source"] == memory.get("mem-run-good").model_dump(mode="json")
    assert res["hits"][1]["source"] == memory.get("mem-attempt-good").model_dump(mode="json")
    assert res["excluded"] == [{"entry_id": "att-bad", "reason": "source_not_success"}]


@pytest.mark.parametrize(
    "bad_run",
    ["run-bad", "run-unknown", "run-cancel", "run-timeout", "run-abort"],
)
def test_non_success_sources_always_excluded(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
    bad_run: str,
) -> None:
    memory, _ = real_store
    idx = DerivedMemoryIndex(
        episodic_memory=memory,
        entries=[_run_entry("reject", run_id=bad_run), _run_entry("keep", run_id="run-good")],
    )
    res = idx.query(tags=["login"], now=QUERY_NOW)
    assert [h["entry"]["entry_id"] for h in res["hits"]] == ["keep"]
    assert res["excluded"] == [{"entry_id": "reject", "reason": "source_not_success"}]


def test_exclusion_reason_precedence_and_expiry_boundary(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
) -> None:
    memory, _ = real_store
    prior = memory.get("mem-run-other")
    memory.append(
        MemoryItem(
            **{
                **prior.model_dump(),
                "memory_id": "mem-run-other-v2",
                "supersedes_id": "mem-run-other",
                "created_at": BASE_TIME + timedelta(hours=1),
            }
        )
    )
    # entry-superseded: superseded AND non-matching tag AND wrong kind AND expired -> superseded wins
    # entry-not-succ: source_not_success AND wrong kind AND expired -> source_not_success wins
    # entry-future: source_in_future AND expired -> source_in_future wins
    # entry-exact-exp: now == expires_at -> expired (even with offset timezone string)
    # entry-almost-exp: expires 1 microsecond after now -> hit
    entries = [
        _run_entry(
            "entry-superseded",
            run_id="run-other",
            kind="semantic",
            tags=["other"],
            expires_at="2026-11-13T00:00:00Z",
        ),
        _run_entry(
            "entry-not-succ",
            run_id="run-bad",
            kind="semantic",
            tags=["other"],
            expires_at="2026-11-13T00:00:00Z",
        ),
        _run_entry(
            "entry-future",
            run_id="run-future",
            kind="semantic",
            tags=["other"],
            expires_at="2026-11-13T00:00:00Z",
        ),
        _run_entry(
            "entry-exact-exp",
            run_id="run-good",
            kind="semantic",
            tags=["login"],
            expires_at="2026-11-14T08:00:00+08:00",
        ),
        _run_entry(
            "entry-kind-miss",
            run_id="run-good",
            kind="semantic",
            tags=["other"],
            expires_at=None,
        ),
        _run_entry(
            "entry-no-match",
            run_id="run-good",
            kind="procedural",
            tags=["checkout"],
            expires_at=None,
        ),
        _run_entry(
            "entry-hit",
            run_id="run-good",
            kind="procedural",
            tags=["login"],
            expires_at="2026-11-14T00:00:00.000001Z",
        ),
    ]
    idx = DerivedMemoryIndex(episodic_memory=memory, entries=entries)
    res = idx.query(tags=["login"], now=QUERY_NOW, kind="procedural", limit=5)
    assert [h["entry"]["entry_id"] for h in res["hits"]] == ["entry-hit"]
    assert res["excluded"] == [
        {"entry_id": "entry-exact-exp", "reason": "expired"},
        {"entry_id": "entry-future", "reason": "source_in_future"},
        {"entry_id": "entry-kind-miss", "reason": "kind_mismatch"},
        {"entry_id": "entry-no-match", "reason": "no_match"},
        {"entry_id": "entry-not-succ", "reason": "source_not_success"},
        {"entry_id": "entry-superseded", "reason": "superseded"},
    ]


def test_limit_zero_and_overflow_exclusion_still_verify_sources(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
) -> None:
    memory, _ = real_store
    entries = [
        _run_entry("c", run_id="run-good", tags=["login"]),
        _run_entry("a", run_id="run-good", tags=["login", "form"]),
        _run_entry("b", run_id="run-good", tags=["login", "form"]),
    ]
    idx = DerivedMemoryIndex(episodic_memory=memory, entries=entries)
    res_one = idx.query(tags=["login", "form"], now=QUERY_NOW, limit=1)
    assert [h["entry"]["entry_id"] for h in res_one["hits"]] == ["a"]
    assert res_one["excluded"] == [
        {"entry_id": "b", "reason": "limit"},
        {"entry_id": "c", "reason": "limit"},
    ]
    res_zero = idx.query(tags=["login", "form"], now=QUERY_NOW, limit=0)
    assert res_zero["hits"] == []
    assert res_zero["excluded"] == [
        {"entry_id": "a", "reason": "limit"},
        {"entry_id": "b", "reason": "limit"},
        {"entry_id": "c", "reason": "limit"},
    ]


def test_source_identity_mismatch_and_missing_memory_cannot_be_bypassed(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
) -> None:
    memory, _ = real_store
    for field, wrong_val in (
        ("memory_id", "mem-missing"),
        ("run_id", "run-other"),
        ("source_record_id", "run-other-end"),
        ("source_level", "attempt"),
    ):
        bad = _run_entry("x", tags=["other"], expires_at="2026-11-13T00:00:00Z")
        bad[field] = wrong_val
        idx = DerivedMemoryIndex(episodic_memory=memory, entries=[bad])
        with pytest.raises(MemorySourceError) as caught:
            idx.query(tags=["no-match"], now=QUERY_NOW, limit=0)
        assert len(str(caught.value)) <= 240
        assert str(wrong_val) not in str(caught.value)


def test_does_not_touch_store_during_init_and_queries_distinct_ids_once(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, trace = real_store
    calls = {"query": 0, "get": 0}
    orig_query = memory.query
    orig_get = memory.get

    def spy_query(*args: object, **kwargs: object) -> tuple[MemoryItem, ...]:
        calls["query"] += 1
        return orig_query(*args, **kwargs)  # type: ignore[arg-type]

    def spy_get(memory_id: str) -> MemoryItem:
        calls["get"] += 1
        return orig_get(memory_id)

    monkeypatch.setattr(memory, "query", spy_query)
    monkeypatch.setattr(memory, "get", spy_get)

    # Two entries share the same memory_id -> get must be called at most once for mem-run-good
    idx = DerivedMemoryIndex(
        episodic_memory=memory,
        entries=[_run_entry("e1", run_id="run-good"), _run_entry("e2", run_id="run-good")],
    )
    assert calls == {"query": 0, "get": 0}

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("forbidden network or process call")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    before = (memory.path.read_bytes(), trace.path.read_bytes())
    res = idx.query(tags=["login"], now=QUERY_NOW)
    assert [h["entry"]["entry_id"] for h in res["hits"]] == ["e1", "e2"]
    assert calls == {"query": 1, "get": 1}
    assert (memory.path.read_bytes(), trace.path.read_bytes()) == before


def test_constructor_and_query_reject_subclasses_boundary_values_and_no_reflection(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
) -> None:
    memory, _ = real_store

    with pytest.raises(MemoryValidationError):
        DerivedMemoryIndex(episodic_memory="not-a-store", entries=[])  # type: ignore[arg-type]

    for bad_entries in ((), [_DictSub(_run_entry())], [_run_entry(str(i)) for i in range(65)]):
        with pytest.raises(MemoryValidationError) as caught:
            DerivedMemoryIndex(episodic_memory=memory, entries=bad_entries)  # type: ignore[arg-type]
        assert len(str(caught.value)) <= 240

    invalid_fields: list[tuple[str, object]] = [
        ("schema_version", "0.2"),
        ("entry_id", _StrSub("a")),
        ("entry_id", "a" * 129),
        ("entry_id", "bad space"),
        ("kind", "episodic"),
        ("memory_id", ""),
        ("run_id", "bad:id"),
        ("source_record_id", "x" * 129),
        ("source_level", "step"),
        ("tags", _ListSub(["login"])),
        ("tags", []),
        ("tags", [f"t{i}" for i in range(33)]),
        ("tags", ["_bad"]),
        ("tags", ["UPPER"]),
        ("tags", ["login", "login"]),
        ("tags", [" login"]),
        ("text", ""),
        ("text", "   "),
        ("text", "x" * 2049),
        ("steps", ["not-allowed-for-semantic"]),
        ("expires_at", "2026-11-14"),
        ("expires_at", "2026-11-14T00:00:00"),
        ("expires_at", "2026-11-14T00:00:00+00:00" + ("0" * 20)),
        ("UNTRUSTED_EXTRA_KEY", "SECRET_SENTINEL"),
    ]
    for field, val in invalid_fields:
        candidate = _run_entry()
        candidate[field] = val
        with pytest.raises(MemoryValidationError) as caught:
            DerivedMemoryIndex(episodic_memory=memory, entries=[candidate])
        msg = str(caught.value)
        assert len(msg) <= 240
        assert "UNTRUSTED_EXTRA_KEY" not in msg
        assert "SECRET_SENTINEL" not in msg

    # Procedural validation errors
    proc = _run_entry(kind="procedural")
    proc["text"] = "forbidden"
    with pytest.raises(MemoryValidationError):
        DerivedMemoryIndex(episodic_memory=memory, entries=[proc])

    for bad_steps in ([], ["   "], ["x" * 257], [f"s{i}" for i in range(17)]):
        proc2 = _run_entry(kind="procedural")
        proc2["steps"] = bad_steps
        with pytest.raises(MemoryValidationError):
            DerivedMemoryIndex(episodic_memory=memory, entries=[proc2])

    # Missing required field
    for req_key in list(_run_entry().keys()):
        missing = _run_entry()
        del missing[req_key]
        with pytest.raises(MemoryValidationError):
            DerivedMemoryIndex(episodic_memory=memory, entries=[missing])

    # Query argument strictness
    idx = DerivedMemoryIndex(episodic_memory=memory, entries=[_run_entry()])
    for bad_tags in ([], ["login", "login"], ["BAD"], _ListSub(["login"]), [_StrSub("login")]):
        with pytest.raises(MemoryValidationError):
            idx.query(tags=bad_tags, now=QUERY_NOW)  # type: ignore[arg-type]
    for bad_now in (
        datetime(2026, 11, 14),  # noqa: DTZ001 - deliberate naive datetime rejection check
        _DatetimeSub(2026, 11, 14, tzinfo=timezone.utc),
        "2026-11-14T00:00:00Z",
    ):
        with pytest.raises(MemoryValidationError):
            idx.query(tags=["login"], now=bad_now)  # type: ignore[arg-type]
    for bad_limit in (True, -1, 21, _IntSub(1), "1"):
        with pytest.raises(MemoryValidationError):
            idx.query(tags=["login"], now=QUERY_NOW, limit=bad_limit)  # type: ignore[arg-type]
    with pytest.raises(MemoryValidationError):
        idx.query(tags=["login"], now=QUERY_NOW, kind="other")


def test_detached_snapshot_and_corruption_never_silently_empty(
    real_store: tuple[JsonlEpisodicMemory, JsonlTraceStore],
) -> None:
    memory, trace = real_store
    data = [_run_entry("detached", tags=["login", "form"])]
    original = copy.deepcopy(data)
    idx = DerivedMemoryIndex(episodic_memory=memory, entries=data)
    data[0]["tags"].append("mutated")  # type: ignore[union-attr]
    data.clear()

    first = idx.query(tags=["login"], now=QUERY_NOW)
    assert first["hits"][0]["entry"] == original[0]
    first["hits"][0]["entry"]["tags"].clear()
    second = idx.query(tags=["login"], now=QUERY_NOW)
    assert second["hits"][0]["entry"] == original[0]

    # Corruption preserves MemoryCorruptionError without modifying memory or trace bytes
    corrupt_bytes = memory.path.read_bytes() + b"{broken\n"
    memory.path.write_bytes(corrupt_bytes)
    trace_before = trace.path.read_bytes()
    with pytest.raises(MemoryCorruptionError):
        idx.query(tags=["missing"], now=QUERY_NOW, limit=0)
    assert memory.path.read_bytes() == corrupt_bytes
    assert trace.path.read_bytes() == trace_before


class _NullOffsetTz(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        return "null"


def test_aware_time_boundaries_never_overflow_utc_and_preserve_microseconds(
    tmp_path: Path,
) -> None:
    """UTC-overflowing aware timestamps preserve domain, offset equality and 1-us precision."""
    low_created = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    high_created = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    trace = JsonlTraceStore(tmp_path / f"trace-{uuid.uuid4().hex[:8]}.jsonl")
    memory = JsonlEpisodicMemory(
        tmp_path / f"memory-{uuid.uuid4().hex[:8]}.jsonl",
        source_resolver=TraceSourceResolver(trace),
    )
    for run_id, mem_id, ts in (
        ("run-low", "mem-run-low", low_created),
        ("run-high", "mem-run-high", high_created),
    ):
        trace.append(
            RunStarted(
                schema_version="0.1",
                record_id=f"{run_id}-start",
                run_id=run_id,
                sequence=0,
                occurred_at=ts,
            )
        )
        trace.append(
            VerificationRecorded(
                schema_version="0.1",
                record_id=f"{run_id}-verify",
                run_id=run_id,
                sequence=1,
                occurred_at=ts,
                scope="run",
                verification_status="pass",
            )
        )
        trace.append(
            RunFinished(
                schema_version="0.1",
                record_id=f"{run_id}-end",
                run_id=run_id,
                sequence=2,
                occurred_at=ts,
                terminal_status="success",
                verification_record_id=f"{run_id}-verify",
            )
        )
        memory.append(
            MemoryItem(
                schema_version="0.1",
                memory_id=mem_id,
                run_id=run_id,
                source_level="run",
                source_record_id=f"{run_id}-end",
                result="success",
                created_at=ts,
            )
        )

    low_exp = "0001-01-01T00:00:00.500000+01:00"
    high_exp = "9999-12-31T23:59:59.500000-01:00"
    entries = [
        _run_entry("low-entry", run_id="run-low", expires_at=low_exp),
        _run_entry("high-entry", run_id="run-high", expires_at=high_exp),
    ]
    idx = DerivedMemoryIndex(episodic_memory=memory, entries=entries)

    # Lower-bound UTC-overflowing query instant never overflows and preserves ordering
    before_low = datetime(1, 1, 1, 0, 0, 0, 499999, tzinfo=timezone(timedelta(hours=1)))
    low_future = idx.query(tags=["login"], now=before_low)
    assert low_future["hits"] == []
    assert low_future["excluded"] == [
        {"entry_id": "high-entry", "reason": "source_in_future"},
        {"entry_id": "low-entry", "reason": "source_in_future"},
    ]

    # At 0001-01-01T00:00:00Z, source is valid and +01:00 entry is recognized as expired
    at_low_utc = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert idx.query(tags=["login"], now=at_low_utc)["excluded"] == [
        {"entry_id": "high-entry", "reason": "source_in_future"},
        {"entry_id": "low-entry", "reason": "expired"},
    ]

    # Upper bound with equivalent -02:00 offset: 1 us before -> hit; at exact instant -> expired
    before_high_eq = datetime(9999, 12, 31, 22, 59, 59, 499999, tzinfo=timezone(timedelta(hours=-2)))
    high_hit = idx.query(tags=["login"], now=before_high_eq)
    assert [hit["entry"]["entry_id"] for hit in high_hit["hits"]] == ["high-entry"]
    assert high_hit["hits"][0]["entry"]["expires_at"] == high_exp

    at_high_eq = datetime(9999, 12, 31, 22, 59, 59, 500000, tzinfo=timezone(timedelta(hours=-2)))
    assert idx.query(tags=["login"], now=at_high_eq)["excluded"] == [
        {"entry_id": "high-entry", "reason": "expired"},
        {"entry_id": "low-entry", "reason": "expired"},
    ]

    # Invalid null-offset tzinfo still raises bounded MemoryValidationError
    with pytest.raises(MemoryValidationError):
        idx.query(tags=["login"], now=datetime(2026, 11, 14, 0, 0, tzinfo=_NullOffsetTz()))
