"""Integration tests for host-supplied derived memory over real trace and episodic stores."""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from forecontract.memory import (
    JsonlEpisodicMemory,
    MemoryCorruptionError,
    MemoryItem,
    MemorySourceError,
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
T0 = datetime(2026, 11, 13, 0, 0, tzinfo=timezone.utc)


def _external_scratch(tmp_path: Path) -> Path:
    env_scratch = os.environ.get("FORECONTRACT_TEST_SCRATCH")
    if env_scratch:
        base = Path(env_scratch) / f"derived-int-{uuid.uuid4().hex[:8]}"
        resolved = base.resolve()
        assert REPO_ROOT not in resolved.parents and resolved != REPO_ROOT
        base.mkdir(parents=True, exist_ok=True)
        return base
    resolved = tmp_path.resolve()
    assert REPO_ROOT not in resolved.parents and resolved != REPO_ROOT
    return tmp_path


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    return _external_scratch(tmp_path)


def _build_real_history(root: Path) -> tuple[JsonlTraceStore, JsonlEpisodicMemory]:
    trace = JsonlTraceStore(root / "trace.jsonl")
    memory = JsonlEpisodicMemory(root / "memory.jsonl", source_resolver=TraceSourceResolver(trace))

    # Run-level success experiences
    for run_id in ("run-auth", "run-form"):
        trace.append(
            RunStarted(
                schema_version="0.1",
                record_id=f"{run_id}-start",
                run_id=run_id,
                sequence=0,
                occurred_at=T0,
            )
        )
        trace.append(
            VerificationRecorded(
                schema_version="0.1",
                record_id=f"{run_id}-v",
                run_id=run_id,
                sequence=1,
                occurred_at=T0,
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
                occurred_at=T0,
                terminal_status="success",
                verification_record_id=f"{run_id}-v",
            )
        )
        memory.append(
            MemoryItem(
                schema_version="0.1",
                memory_id=f"m-{run_id}",
                run_id=run_id,
                source_level="run",
                source_record_id=f"{run_id}-end",
                result="success",
                created_at=T0,
            )
        )

    # Run-level failed experience
    trace.append(
        RunStarted(
            schema_version="0.1",
            record_id="run-bad-start",
            run_id="run-bad",
            sequence=0,
            occurred_at=T0,
        )
    )
    trace.append(
        RunFinished(
            schema_version="0.1",
            record_id="run-bad-end",
            run_id="run-bad",
            sequence=1,
            occurred_at=T0,
            terminal_status="failure",
            verification_record_id=None,
        )
    )
    memory.append(
        MemoryItem(
            schema_version="0.1",
            memory_id="m-run-bad",
            run_id="run-bad",
            source_level="run",
            source_record_id="run-bad-end",
            result="failure",
            created_at=T0,
        )
    )

    # Attempt-level completed experience inside one run
    trace.append(
        RunStarted(
            schema_version="0.1",
            record_id="run-mix-start",
            run_id="run-mix",
            sequence=0,
            occurred_at=T0,
        )
    )
    trace.append(
        StepStarted(
            schema_version="0.1",
            record_id="run-mix-step",
            run_id="run-mix",
            sequence=1,
            occurred_at=T0,
            step_id="step-checkout",
        )
    )
    trace.append(
        ContractCommitted(
            schema_version="0.1",
            record_id="run-mix-contract",
            run_id="run-mix",
            sequence=2,
            occurred_at=T0,
            step_id="step-checkout",
            contract_id="c-1",
        )
    )
    trace.append(
        AttemptStarted(
            schema_version="0.1",
            record_id="run-mix-att1-start",
            run_id="run-mix",
            sequence=3,
            occurred_at=T0,
            step_id="step-checkout",
            contract_id="c-1",
            attempt_id="att-1",
        )
    )
    trace.append(
        AttemptFinished(
            schema_version="0.1",
            record_id="run-mix-att1-end",
            run_id="run-mix",
            sequence=4,
            occurred_at=T0,
            step_id="step-checkout",
            contract_id="c-1",
            attempt_id="att-1",
            execution_status="succeeded",
        )
    )
    memory.append(
        MemoryItem(
            schema_version="0.1",
            memory_id="m-att-checkout",
            run_id="run-mix",
            source_level="attempt",
            source_record_id="run-mix-att1-end",
            result="success",
            created_at=T0,
        )
    )
    return trace, memory


def test_host_json_catalog_reload_and_fixed_offline_evaluation(scratch_dir: Path) -> None:
    """Host reloads plain JSON catalog into a fresh read-only index with fixed evaluation metrics."""
    trace, memory = _build_real_history(scratch_dir)
    catalog_path = scratch_dir / "host_catalog.json"
    host_catalog = [
        {
            "schema_version": "0.1",
            "entry_id": "a",
            "kind": "semantic",
            "memory_id": "m-run-auth",
            "run_id": "run-auth",
            "source_record_id": "run-auth-end",
            "source_level": "run",
            "tags": ["login"],
            "text": "Login route requires CSRF cookie",
            "steps": [],
            "expires_at": None,
        },
        {
            "schema_version": "0.1",
            "entry_id": "b",
            "kind": "semantic",
            "memory_id": "m-run-form",
            "run_id": "run-form",
            "source_record_id": "run-form-end",
            "source_level": "run",
            "tags": ["login", "form"],
            "text": "Login form submits via POST /session",
            "steps": [],
            "expires_at": None,
        },
        {
            "schema_version": "0.1",
            "entry_id": "c",
            "kind": "procedural",
            "memory_id": "m-att-checkout",
            "run_id": "run-mix",
            "source_record_id": "run-mix-att1-end",
            "source_level": "attempt",
            "tags": ["checkout"],
            "text": None,
            "steps": ["Confirm total", "Click submit"],
            "expires_at": "2026-11-20T00:00:00Z",
        },
        {
            "schema_version": "0.1",
            "entry_id": "bad",
            "kind": "semantic",
            "memory_id": "m-run-bad",
            "run_id": "run-bad",
            "source_record_id": "run-bad-end",
            "source_level": "run",
            "tags": ["login", "form"],
            "text": "Should not be surfaced because source run failed",
            "steps": [],
            "expires_at": None,
        },
    ]
    catalog_path.write_text(json.dumps(host_catalog, ensure_ascii=False), encoding="utf-8")

    loaded_entries = json.loads(catalog_path.read_text(encoding="utf-8"))
    index = DerivedMemoryIndex(episodic_memory=memory, entries=loaded_entries)

    now = T0 + timedelta(days=1)
    oracle = [
        (["login", "form"], ["b", "a"]),
        (["checkout"], ["c"]),
        (["missing"], []),
        (["login"], ["a", "b"]),
    ]
    exact_matches = []
    baseline_matches = []
    top1_hits = []
    baseline_top1_hits = []
    invalid_source_hits = 0

    for tags, expected in oracle:
        result = index.query(tags=tags, now=now)
        actual = [hit["entry"]["entry_id"] for hit in result["hits"]]
        invalid_source_hits += sum(1 for eid in actual if eid == "bad")
        exact_matches.append(actual == expected)
        id_only = sorted(
            item["entry_id"]
            for item in loaded_entries
            if item["entry_id"] != "bad" and (set(item["tags"]) & set(tags))
        )
        baseline_matches.append(id_only == expected)
        if expected:
            top1_hits.append(actual[:1] == expected[:1])
            baseline_top1_hits.append(id_only[:1] == expected[:1])

    assert sum(exact_matches) == 4
    assert sum(baseline_matches) == 3
    assert sum(top1_hits) == 3
    assert sum(baseline_top1_hits) == 2
    assert invalid_source_hits == 0
    assert (trace.path.exists(), memory.path.exists()) == (True, True)


def test_supersession_and_source_failure_boundaries_in_integration(scratch_dir: Path) -> None:
    """Superseding an attempt item invalidates old catalog references; disconnected/corrupt fails explicitly."""
    trace, memory = _build_real_history(scratch_dir)
    catalog = [
        {
            "schema_version": "0.1",
            "entry_id": "checkout-proc",
            "kind": "procedural",
            "memory_id": "m-att-checkout",
            "run_id": "run-mix",
            "source_record_id": "run-mix-att1-end",
            "source_level": "attempt",
            "tags": ["checkout"],
            "text": None,
            "steps": ["Review items"],
            "expires_at": None,
        }
    ]
    index = DerivedMemoryIndex(episodic_memory=memory, entries=catalog)
    now = T0 + timedelta(days=1)

    initial = index.query(tags=["checkout"], now=now)
    assert [hit["entry"]["entry_id"] for hit in initial["hits"]] == ["checkout-proc"]

    # Append a newer memory version on the same attempt chain -> old entry becomes superseded
    prior = memory.get("m-att-checkout")
    memory.append(
        MemoryItem(
            **{
                **prior.model_dump(),
                "memory_id": "m-att-checkout-v2",
                "supersedes_id": "m-att-checkout",
                "created_at": T0 + timedelta(hours=2),
            }
        )
    )
    after_supersede = index.query(tags=["checkout"], now=now)
    assert after_supersede["hits"] == []
    assert after_supersede["excluded"] == [{"entry_id": "checkout-proc", "reason": "superseded"}]

    # Disconnecting trace file produces MemorySourceError even when entry is superseded and limit=0
    backup = trace.path.with_suffix(".offline")
    trace.path.rename(backup)
    with pytest.raises(MemorySourceError):
        index.query(tags=["no-match"], now=now, limit=0)

    backup.rename(trace.path)
    raw_mem = memory.path.read_bytes()
    memory.path.write_bytes(raw_mem + b"corrupt-tail\n")
    with pytest.raises(MemoryCorruptionError):
        index.query(tags=["no-match"], now=now, limit=0)
    assert memory.path.read_bytes() == raw_mem + b"corrupt-tail\n"


def test_extreme_aware_timestamps_and_cross_offset_equivalence_on_real_store(
    scratch_dir: Path,
) -> None:
    """Real stores accept and compare UTC-overflowing aware timestamps across offsets cleanly."""
    trace, memory = _build_real_history(scratch_dir)
    base_item = memory.get("m-run-auth")

    entries = [
        {
            "schema_version": "0.1",
            "entry_id": "low-window",
            "kind": "semantic",
            "memory_id": "m-run-auth",
            "run_id": base_item.run_id,
            "source_record_id": base_item.source_record_id,
            "source_level": base_item.source_level,
            "tags": ["login"],
            "text": "Lower UTC boundary semantic note",
            "steps": [],
            "expires_at": "0001-01-01T00:00:00+01:00",
        },
        {
            "schema_version": "0.1",
            "entry_id": "high-window",
            "kind": "procedural",
            "memory_id": "m-att-checkout",
            "run_id": "run-mix",
            "source_record_id": "run-mix-att1-end",
            "source_level": "attempt",
            "tags": ["checkout"],
            "text": None,
            "steps": ["Verify upper bound"],
            "expires_at": "9999-12-31T23:59:59-01:00",
        },
    ]
    index = DerivedMemoryIndex(episodic_memory=memory, entries=entries)
    trace_before = trace.path.read_bytes()
    memory_before = memory.path.read_bytes()

    low_now = datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone(timedelta(hours=1)))
    low_res = index.query(tags=["login"], now=low_now)
    assert low_res["hits"] == []
    assert low_res["excluded"] == [
        {"entry_id": "high-window", "reason": "source_in_future"},
        {"entry_id": "low-window", "reason": "source_in_future"},
    ]
    assert index.query(tags=["login"], now=T0)["excluded"] == [
        {"entry_id": "high-window", "reason": "no_match"},
        {"entry_id": "low-window", "reason": "expired"},
    ]

    high_before = datetime(9999, 12, 31, 22, 59, 58, 999999, tzinfo=timezone(timedelta(hours=-2)))
    high_hit = index.query(tags=["checkout"], now=high_before)
    assert [hit["entry"]["entry_id"] for hit in high_hit["hits"]] == ["high-window"]
    assert high_hit["hits"][0]["entry"]["expires_at"] == "9999-12-31T23:59:59-01:00"

    high_equal_offset = datetime(9999, 12, 31, 22, 59, 59, tzinfo=timezone(timedelta(hours=-2)))
    high_expired = index.query(tags=["checkout"], now=high_equal_offset)
    assert high_expired["hits"] == []
    assert high_expired["excluded"] == [
        {"entry_id": "high-window", "reason": "expired"},
        {"entry_id": "low-window", "reason": "expired"},
    ]
    assert trace.path.read_bytes() == trace_before
    assert memory.path.read_bytes() == memory_before
