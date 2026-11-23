"""Acceptance tests for real host evidence registration, the bridge and the offline demo.

C04/C07: the new bridge subclasses the existing recorder without changing it, registers each
real snapshot before ``observe`` returns, leaves the raw bytes on disk when registration fails,
and the demo's host-persisted audit directory is re-read from its own files: every event is
canonically encoded, every payload digest is real, every intent has its result and the ready
forecast is written only after final validation. The demo never touches the governance event log.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
for candidate in (
    REPO_ROOT,
    EXAMPLES,
    REPO_ROOT / "packages" / "core" / "src",
    REPO_ROOT / "packages" / "adapters" / "fake" / "src",
    REPO_ROOT / "packages" / "adapters" / "web" / "src",
    REPO_ROOT / "packages" / "planners" / "scripted" / "src",
    REPO_ROOT / "packages" / "planners" / "llm" / "src",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import llm_retrieval_demo as demo
import web_retrieval_evidence as bridge_module
from forecontract.models import Observation
from forecontract_llm.models import PlannerConfig
from forecontract_llm.retrieval import (
    EvidenceGrant,
    RegisteredEvidenceReader,
    RetrievalConfig,
    RetrievalError,
)
from forecontract_llm.retrieval.audit import build_event
from forecontract_llm.retrieval.models import canonical_json_bytes
from web_retrieval_evidence import RegisteringEvidenceRecorder
from web_runtime_evidence import (
    EvidenceError,
    EvidenceRecorder,
    RecordingAdapter,
    verify_evidence_index,
)

SCHEMA = "0.1"
RUN = "run-1"
OBS = "obs-1"
FIXED_TIME = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def _config() -> RetrievalConfig:
    return RetrievalConfig(
        schema_version=SCHEMA,
        base=PlannerConfig(schema_version=SCHEMA, call_timeout_ms=5000,
                           cleanup_timeout_ms=100))


def _observation(state: dict | None = None, observation_id: str = OBS) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="offline",
                       state=state if state is not None else {"done": False},
                       timestamp=FIXED_TIME)


def _grant(*, observation_id: str = OBS, evidence_id: str = "evidence-1",
           view_id: str = "status", run_id: str = RUN) -> EvidenceGrant:
    return EvidenceGrant(
        schema_version=SCHEMA, run_id=run_id, evidence_id=evidence_id,
        observation_id=observation_id, view_id=view_id,
        selector=["state", "done"], format="json")


def _bridge(tmp_path: Path, *, grant: EvidenceGrant | None = None):
    root = tmp_path / "evidence"
    root.mkdir()
    reader = RegisteredEvidenceReader(root, RUN, _config())
    recorder = RegisteringEvidenceRecorder(
        root, run_id=RUN, profile="offline-retrieval", reader=reader,
        grants_for=lambda observation: [grant or _grant(
            observation_id=observation.observation_id)])
    return recorder, reader, root


class _AdapterDouble:
    """The smallest declarative adapter a RecordingAdapter can forward to."""

    name = "fake-double"

    def capabilities(self):
        from forecontract.models import AdapterCapabilities

        return AdapterCapabilities(
            schema_version=SCHEMA, adapter_name=self.name, environment_type="offline",
            supported_actions=["mark_done"], supported_outcome_types=[])

    def validate_action(self, action) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def observe(self) -> Observation:
        return _observation()

    async def execute(self, action):
        raise AssertionError("the bridge test does not execute")


# --------------------------------------------------------------------------- bridge


def test_the_bridge_registers_the_real_snapshot_before_observe_returns(
        tmp_path: Path) -> None:
    """C04: RecordingAdapter.observe returns only after the snapshot is registered."""
    recorder, reader, root = _bridge(tmp_path)
    adapter = RecordingAdapter(_AdapterDouble(), recorder)
    observation = asyncio.run(adapter.observe())
    entry = recorder._entries[-1]
    assert entry["kind"] == "observation"
    recorded = (root / entry["path"]).read_bytes()
    assert hashlib.sha256(recorded).hexdigest() == entry["sha256"]
    scope = reader.open_scope(RUN, observation.observation_id)
    assert [item.evidence_id for item in scope.entries] == ["evidence-1"]
    assert reader.snapshot_count == 1


def test_the_bridge_is_the_existing_recorder_plus_registration(tmp_path: Path) -> None:
    """C04: the bridge subclasses the frozen recorder; it does not replace it."""
    assert issubclass(RegisteringEvidenceRecorder, EvidenceRecorder)
    recorder, _reader, _root = _bridge(tmp_path)
    assert isinstance(recorder, EvidenceRecorder)
    source = (EXAMPLES / "web_runtime_evidence.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == \
        "8d68b4211e578a66639139263d8d1c16e65189bb5ec57b26f779129f8037e6e9"


def test_a_failed_registration_leaves_the_raw_snapshot_but_no_read_entry(
        tmp_path: Path) -> None:
    """C04: the parent may have written the bytes; they are not readable evidence."""
    recorder, reader, root = _bridge(tmp_path, grant=_grant(observation_id="obs-other"))
    with pytest.raises(RetrievalError) as info:
        recorder.record_observation(_observation())
    assert (info.value.code, info.value.stage) == ("binding_mismatch", "binding")
    assert len(recorder._entries) == 1
    assert (root / recorder._entries[0]["path"]).is_file()
    assert reader.snapshot_count == 0
    with pytest.raises(RetrievalError) as info:
        reader.open_scope(RUN, OBS)
    assert info.value.code == "access_denied"


def test_the_bridge_keeps_the_parent_index_entry_identity(tmp_path: Path) -> None:
    """C04: the returned entry is exactly the parent's entry, not a rebuilt one."""
    recorder, _reader, root = _bridge(tmp_path)
    entry = recorder.record_observation(_observation())
    assert entry == recorder._entries[-1]
    assert set(entry) == {"kind", "ordinal", "path", "sha256", "bytes", "call_ordinal",
                          "observation_id"}
    raw = (root / entry["path"]).read_bytes()
    assert len(raw) == entry["bytes"]
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert entry["ordinal"] == 1 and entry["call_ordinal"] == 1


def test_a_non_grant_callback_is_refused_by_the_bridge(tmp_path: Path) -> None:
    """C04: only real grants are accepted; the raw snapshot still exists afterwards."""
    root = tmp_path / "evidence"
    root.mkdir()
    reader = RegisteredEvidenceReader(root, RUN, _config())
    recorder = RegisteringEvidenceRecorder(
        root, run_id=RUN, profile="offline-retrieval", reader=reader,
        grants_for=lambda observation: ["not-a-grant"])
    with pytest.raises(EvidenceError):
        recorder.record_observation(_observation())
    with pytest.raises(RetrievalError):
        reader.open_scope(RUN, OBS)


# --------------------------------------------------------------------------- demo


def _run_demo(tmp_path: Path, name: str = "demo", state: dict | None = None) -> tuple[dict,
                                                                                     Path]:
    target = tmp_path / name
    summary = asyncio.run(demo.run_demo(target, state=state))
    return summary, target


def test_the_demo_runs_offline_and_persists_a_hash_bound_audit_trail(
        tmp_path: Path) -> None:
    """C04/C07: the demo completes through Core and its audit files verify from disk."""
    summary, target = _run_demo(tmp_path)
    assert (summary["terminal_status"], summary["reason_code"]) == (
        "completed", "TASK_ACHIEVED")
    assert summary["provider_calls"] == 2
    assert summary["calls_used"] == 2
    assert summary["close"] == {"schema_version": SCHEMA, "drained": True,
                                "pending_calls": 0}
    audit_dir = target / "audit"
    files = sorted(audit_dir.glob("*.json"))
    assert [path.name for path in files] == [
        "0001-plan_started.json", "0002-request_prepared.json",
        "0003-provider_returned.json", "0004-read_requested.json",
        "0005-read_returned.json", "0006-request_prepared.json",
        "0007-provider_returned.json", "0008-forecast_ready.json",
        "0009-plan_closed.json"]
    sequences = []
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        payload = document["payload"]
        assert document["payload_sha256"] == hashlib.sha256(
            canonical_json_bytes(payload)).hexdigest()
        assert document["payload_bytes"] == len(canonical_json_bytes(payload))
        sequences.append(document["sequence"])
    assert sequences == list(range(1, 10))
    forecast_ready = json.loads(files[7].read_text(encoding="utf-8"))
    assert forecast_ready["payload"]["forecast"]["proposed_action"]["action_type"] == \
        "mark_done"
    returned = json.loads(files[2].read_text(encoding="utf-8"))
    assert returned["payload"]["status"] == "ok"
    assert returned["payload"]["usage_known"] is False
    trace_records = [
        json.loads(line) for line in
        (target / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert trace_records[-1]["kind"] == "run_finished"
    assert trace_records[-1]["terminal_status"] == "success"


def test_two_real_demo_snapshots_drive_two_different_actions(tmp_path: Path) -> None:
    """C04: changing the synthetic snapshot field changes the executed action payload."""
    first, _target = _run_demo(tmp_path, "first",
                               {"done": False, "order": {"status": "pending",
                                                         "item": "widget"}})
    second, _target = _run_demo(tmp_path, "second",
                                {"done": False, "order": {"status": "ready",
                                                          "item": "widget"}})
    assert first["read_values"] == ["pending"]
    assert second["read_values"] == ["ready"]
    assert first["executed_action"]["payload"] == {"observed": "pending"}
    assert second["executed_action"]["payload"] == {"observed": "ready"}
    assert first["executed_action"] != second["executed_action"]


def test_the_demo_audit_sink_never_overwrites_an_existing_event(tmp_path: Path) -> None:
    """C07: one exclusive file per event; an existing record is never replaced."""
    sink = demo.FileAuditSink(tmp_path / "audit")
    event = build_event(run_id=RUN, plan_id="plan-1", sequence=1, kind="plan_started",
                        payload={"started": True})
    asyncio.run(sink.append(event))
    first = (tmp_path / "audit" / "0001-plan_started.json").read_bytes()
    with pytest.raises(demo.DemoError):
        asyncio.run(sink.append(event))
    assert (tmp_path / "audit" / "0001-plan_started.json").read_bytes() == first


def test_atomic_audit_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F05: a short write leaves no partially committed event under the final name.

    The sink writes into a private temporary file and publishes only after a successful
    flush. An injected OSError while writing must leave the audit directory without a
    ``*.json`` file, and the temporary file must be cleaned up.
    """
    directory = tmp_path / "audit"
    sink = demo.FileAuditSink(directory)
    event = build_event(run_id=RUN, plan_id="plan-1", sequence=1, kind="plan_started",
                        payload={"started": True})
    original = Path.open

    class Partial:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            self.handle.close()

        def write(self, raw: bytes) -> None:
            self.handle.write(raw[:12])
            self.handle.flush()
            raise OSError("injected-short-write")

    def patched(path: Path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if path.parent == directory and args and args[0] == "xb":
            return Partial(handle)
        return handle

    monkeypatch.setattr(Path, "open", patched)
    with pytest.raises(demo.DemoError):
        asyncio.run(sink.append(event))
    assert list(directory.glob("*.json")) == []
    assert list(directory.iterdir()) == []


def test_the_demo_refuses_a_repository_output_directory() -> None:
    """C04: demo artefacts stay outside the repository."""
    with pytest.raises(demo.DemoError):
        asyncio.run(demo.run_demo(REPO_ROOT / "tests" / "demo-must-not-exist"))
    assert not (REPO_ROOT / "tests" / "demo-must-not-exist").exists()


def test_the_demo_never_touches_the_governance_event_log(tmp_path: Path) -> None:
    """C07: a retrieval audit sink must not write the governance event log."""
    governance = REPO_ROOT / "artifacts" / "events" / "governance-events.jsonl"
    before = governance.read_bytes()
    _run_demo(tmp_path)
    assert governance.read_bytes() == before


def test_the_demo_summary_reports_the_registration_time_snapshot(tmp_path: Path) -> None:
    """C03/C04: the demo's real snapshots are read as registration-time copies."""
    summary, target = _run_demo(tmp_path)
    assert summary["evidence"]["snapshot_count"] >= 2
    snapshots = sorted((target / "snapshots").glob("observation-*.json"))
    assert len(snapshots) == summary["evidence"]["snapshot_count"]
    for path in snapshots:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["kind"] == "observation"
        assert document["run_id"] == "run-retrieval-demo"
    reads = json.loads(
        (target / "audit" / "0005-read_returned.json").read_text(encoding="utf-8"))
    assert reads["payload"]["source"]["registration_time_snapshot"] is True
    assert reads["payload"]["source"]["live_disk_state"] is False


def test_the_bridge_and_the_demo_keep_the_completed_state_verifier_untouched(
        tmp_path: Path) -> None:
    """C04: the original completed-state verifier still imports and its source is unchanged."""
    assert callable(verify_evidence_index)
    source = (EXAMPLES / "web_runtime_evidence.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == \
        "8d68b4211e578a66639139263d8d1c16e65189bb5ec57b26f779129f8037e6e9"
    assert bridge_module.EvidenceRecorder is EvidenceRecorder
