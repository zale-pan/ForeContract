"""Trace and complete-evidence acceptance for the Web Runtime.

The test proves that the host evidence sidecar is genuinely bound to the real trace and Working
Memory: every snapshot is re-read and hashed, each observation is tied to the Runtime's own call
ordinal, repeated observation identifiers are distinguished by call order, a missing or altered
snapshot is refused and a recording serialization failure is explicit. It also proves that the
recording proxies do not change the Core trace's shape by running the same profile once with and
once without them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "examples", REPO / "packages" / "core" / "src",
                  REPO / "packages" / "adapters" / "web" / "src",
                  REPO / "packages" / "adapters" / "fake" / "src",
                  REPO / "packages" / "planners" / "scripted" / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import web_runtime_demo as demo
from forecontract.contracts import ContractBuilder
from forecontract.memory import WorkingMemory
from forecontract.recovery import RecoveryEngine
from forecontract.runtime import (
    MonotonicBudgetSource,
    RuntimeEngine,
    RuntimeTimingPolicy,
)
from forecontract.tracing import JsonlTraceStore
from forecontract_scripted import ScriptedPlanner
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.browser_errors import WebBrowserError
from forecontract_web.playwright_adapter import PlaywrightWebAdapter
from forecontract_web.runtime_hooks import (
    WebContractHooksFactory,
    WebTaskCriteriaValidator,
)
from forecontract_web.runtime_support import WebRuntimeDomainSupport
from forecontract_web.verification import WebStepVerifier, WebTaskCompletionOracle
from web_runtime_evidence import (
    EvidenceError,
    EvidenceRecorder,
    RecordingAdapter,
    verify_evidence_index,
)

from test_site import LocalTestSite, Scenario, SiteConfig


@pytest.fixture(scope="module", autouse=True)
def _unload_playwright_after_module():
    """Leave the process as this test found it: browser imports stay local to the browser phase."""
    yield
    for name in [name for name in sys.modules
                 if name == "playwright" or name.startswith("playwright.")]:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("p3t05-web-runtime-trace")
    output = base / "normal"
    output.mkdir()
    try:
        return demo.run_profile("normal", output)
    except demo.DemoFailure as exc:
        if exc.original is not None:
            raise exc.original
        raise


@pytest.fixture(scope="module")
def plain_normal(tmp_path_factory) -> dict:
    """The same real profile without any recording proxy, for the trace-shape comparison."""
    base = tmp_path_factory.mktemp("p3t05-web-runtime-plain")
    output = base / "normal"
    output.mkdir()
    site = LocalTestSite(SiteConfig(scenario=Scenario.NORMAL, request_timeout_ms=15000,
                                    cleanup_timeout_ms=3000)).start()
    try:
        origin = site.origin
        policies = demo._policies(origin, demo.PASSWORD, click_timeout_ms=5000.0,
                                  recovery_policy=demo.STANDARD_RECOVERY)
        script = demo._script(origin, demo.PASSWORD)
        domain = WebRuntimeDomainSupport(
            adapter_name="playwright_web", required_targets=list(demo.TARGETS),
            require_url=True, action_policies=policies)
        adapter = PlaywrightWebAdapter(PlaywrightWebConfig(
            allowed_origins=[origin], observation_targets=list(demo.TARGETS),
            operation_timeout_ms=3000, cleanup_timeout_ms=demo.ADAPTER_CLEANUP_MS))
        trace = JsonlTraceStore(output / "trace.jsonl")
        engine = RuntimeEngine(
            adapter=adapter,
            planner=ScriptedPlanner(script),
            domain_support=domain,
            budget_source=MonotonicBudgetSource(total_budget_ms=30000.0),
            trace_writer=trace,
            working_memory=WorkingMemory(),
            builder=ContractBuilder(),
            step_verifier=WebStepVerifier(),
            task_oracle=WebTaskCompletionOracle(),
            recovery_engine=RecoveryEngine(),
            timing_policy=RuntimeTimingPolicy(
                schema_version="0.1", run_timeout_ms=30000.0,
                cleanup_timeout_ms=demo.RUNTIME_CLEANUP_MS,
                finalize_timeout_ms=demo.RUNTIME_FINALIZE_MS),
            contract_hooks_factory=WebContractHooksFactory(),
            task_criteria_validator=WebTaskCriteriaValidator(),
        )
        request = demo._request("plain-run", origin)
        result = asyncio.run(engine.run(request=request))
        assert adapter.cleanup_error is None
        records = list(trace.read_run("plain-run"))
        return {"result": result, "records": records, "trace_path": output / "trace.jsonl"}
    finally:
        assert site.stop().complete


def _project(records) -> list[tuple]:
    """Every durable trace field that the recording proxies must not change."""
    return [
        (record.kind, record.sequence, getattr(record, "scope", None),
         getattr(getattr(record, "verification_status", None), "value", None),
         getattr(getattr(record, "execution_status", None), "value", None),
         getattr(record, "error_code", None))
        for record in records
    ]


def test_the_recording_proxies_do_not_change_the_core_trace(recorded, plain_normal) -> None:
    recorded_records = list(JsonlTraceStore(Path(recorded["trace_path"])).read_run(
        recorded["run_id"]))
    assert _project(recorded_records) == _project(plain_normal["records"])
    assert recorded["result"]["terminal_status"] == plain_normal["result"].terminal_status.value
    assert recorded["result"]["steps_started"] == plain_normal["result"].steps_started
    assert recorded["result"]["attempts_started"] == plain_normal["result"].attempts_started
    # The sidecar is a separate file; the trace keeps its strict existing schema.
    trace_document = json.loads(Path(recorded["trace_path"]).read_text(encoding="utf-8")
                               .splitlines()[0])
    assert set(trace_document) == {
        "schema_version", "record_id", "run_id", "sequence", "occurred_at", "kind"}


def test_every_snapshot_is_hash_bound_and_the_index_revalidates(recorded) -> None:
    root = Path(recorded["evidence_index_path"]).parent
    index = verify_evidence_index(root)
    assert index["run_id"] == recorded["run_id"]
    assert index["trace"]["records"] >= 1
    paths = {entry["path"] for entry in index["snapshots"]}
    assert len(paths) == len(index["snapshots"])
    for entry in index["snapshots"]:
        assert (root / entry["path"]).is_file()
        assert entry["sha256"] and entry["bytes"] > 0


def test_contract_attempt_verification_and_event_associations_are_real(recorded) -> None:
    root = Path(recorded["evidence_index_path"]).parent
    index = verify_evidence_index(root)
    event_ids = {event["record_id"] for event in index["events"]}
    assert index["contracts"]
    for contract in index["contracts"]:
        assert contract["event_record_id"] in event_ids
        assert contract["contract_id"]
        assert contract["step_id"]
    for attempt in index["attempts"]:
        assert attempt["event_record_id"] in event_ids
        assert attempt["attempt_id"]
        assert attempt["contract_id"]
    for verification in index["verifications"]:
        if verification["trace_fact"] is not None:
            assert verification["event_record_id"] in event_ids
    observation_ordinals = [entry["runtime_call_ordinal"]
                            for entry in index["snapshots"]
                            if entry["kind"] == "observation"]
    assert len(observation_ordinals) == len(set(observation_ordinals))
    assert index["events"][-1]["kind"] == "run_finished"
    assert [event["kind"] for event in index["events"]].count("run_finished") == 1


def test_repeated_observation_identifiers_are_distinguished_by_call_order(
        tmp_path: Path) -> None:
    recorder = EvidenceRecorder(tmp_path, run_id="run-repeat", profile="normal")

    from datetime import datetime, timezone

    from forecontract.models.observations import Observation

    repeated = Observation(
        schema_version="0.1", observation_id="same-observation",
        environment_type="web", timestamp=datetime(2026, 1, 2, 3, 4, 5,
                                                  tzinfo=timezone.utc),
        state={"schema_version": "0.1", "url": "about:blank", "targets": []})
    recorder.record_observation(repeated)
    recorder.record_observation(repeated)
    trace_path = tmp_path / "trace.jsonl"
    memory_path = tmp_path / "working-memory.json"
    trace_path.write_text("", encoding="utf-8")
    memory_state = {"run_id": "run-repeat", "observation_calls": [
        {"ordinal": 1, "phase": "observing", "observation_id": "same-observation",
         "step_id": None, "attempt_id": None},
        {"ordinal": 7, "phase": "verifying", "observation_id": "same-observation",
         "step_id": "step-1", "attempt_id": "attempt-1"},
    ]}
    memory_path.write_text(json.dumps({"published_by": "runtime", "state": memory_state}),
                           encoding="utf-8")
    index_path = recorder.finalize(
        result=None, last_finalization=None,
        terminal={"terminal_status": None, "reason_code": None, "source": "last_finalization"},
        trace_path=trace_path, working_memory_path=memory_path,
        working_state=memory_state, trace_records=[],
        cleanup={"adapter_cleanup_error": None, "browser": {}})
    index = json.loads(index_path.read_text(encoding="utf-8"))
    observations = [entry for entry in index["snapshots"]
                    if entry["kind"] == "observation"]
    assert [entry["observation_id"] for entry in observations] == [
        "same-observation", "same-observation"]
    assert [entry["runtime_call_ordinal"] for entry in observations] == [1, 7]
    assert [entry["path"] for entry in observations] == [
        "snapshots/observation-0001.json", "snapshots/observation-0002.json"]


def test_a_missing_or_altered_snapshot_is_refused(recorded, tmp_path: Path) -> None:
    copy = _copy_index_rebased(recorded, tmp_path, "tampered")
    assert verify_evidence_index(copy)
    index = json.loads((copy / "evidence-index.json").read_text(encoding="utf-8"))
    first = index["snapshots"][0]
    (copy / first["path"]).write_bytes(b"{}")
    with pytest.raises(EvidenceError):
        verify_evidence_index(copy)
    shutil.rmtree(copy)
    _copy_index_rebased(recorded, tmp_path, "tampered")
    (copy / first["path"]).unlink()
    with pytest.raises(EvidenceError):
        verify_evidence_index(copy)


def test_a_recording_serialization_failure_is_explicit(tmp_path: Path) -> None:
    recorder = EvidenceRecorder(tmp_path, run_id="run-bad", profile="normal")
    with pytest.raises(EvidenceError):
        recorder.record_observation({"not": "a record", "payload": {1, 2, 3}})
    assert not (tmp_path / "evidence-index.json").exists()


def test_a_real_zero_snapshot_facility_failure_remains_verifiable(tmp_path: Path) -> None:
    """A Run that failed before any observation still has a real, verifiable index.

    The trace carries its genuine terminal fact, the snapshot inventory is empty and the recorded
    Working Memory declares no observation call. The stricter association/result checks must not
    turn that honest empty inventory into a fabricated snapshot or a refusal.
    """
    from types import SimpleNamespace

    run_id = "run-zero"
    trace_path = tmp_path / "trace.jsonl"
    records = [
        SimpleNamespace(record_id="rt-record-0", kind="run_started", sequence=0,
                        step_id=None, contract_id=None, attempt_id=None, scope=None,
                        verification_status=None),
        SimpleNamespace(record_id="rt-record-1", kind="run_finished", sequence=1,
                        step_id=None, contract_id=None, attempt_id=None, scope=None,
                        verification_status=None),
    ]
    trace_path.write_text("\n".join(json.dumps({
        "schema_version": "0.1", "record_id": record.record_id, "run_id": run_id,
        "sequence": record.sequence, "kind": record.kind,
        "terminal_status": "failure" if record.kind == "run_finished" else None,
        "reason_code": "internal_error" if record.kind == "run_finished" else None,
    }) for record in records) + "\n", encoding="utf-8")
    memory_path = tmp_path / "working-memory.json"
    state = {
        "schema_version": "0.1", "run_id": run_id, "goal_id": "goal-zero",
        "phase": "FAILED", "terminal_status": "failed",
        "reason_code": "DEPENDENCY_FAILURE", "steps_started": 0, "planning_calls": 0,
        "attempts_started": 0, "retries_used": 0, "replans_used": 0,
        "verification_record_id": None, "terminal_record_id": "rt-record-1",
        "observation_calls": [],
    }
    memory_path.write_text(json.dumps({"published_by": "runtime", "state": state},
                                      ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    recorder = EvidenceRecorder(tmp_path, run_id=run_id, profile="normal")
    index_path = recorder.finalize(
        result=None,
        last_finalization={"run_id": run_id, "primary_terminal_status": "failed",
                           "primary_reason_code": "DEPENDENCY_FAILURE"},
        terminal={"terminal_status": "failed", "reason_code": "DEPENDENCY_FAILURE",
                  "source": "last_finalization"},
        trace_path=trace_path, working_memory_path=memory_path,
        working_state=state, trace_records=records,
        cleanup={"adapter_cleanup_error": None, "browser": {}})
    index = verify_evidence_index(index_path.parent)
    assert index["snapshots"] == []
    assert index["working_memory"]["observation_calls"] == []
    assert index["terminal"]["terminal_status"] == "failed"


# --- F04: the index is a strict structure, not a caller-offered file list ---------------------


def _copy_index(recorded, tmp_path: Path, name: str) -> Path:
    source = Path(recorded["evidence_index_path"]).parent
    copy = tmp_path / name
    shutil.copytree(source, copy)
    return copy


def _copy_index_rebased(recorded, tmp_path: Path, name: str) -> Path:
    """Copy one recorded index and relocate its trace/memory declarations into the copy.

    Without the relocation a copied index would still point at the original absolute files, so a
    rejection could merely mean the path escaped the copy rather than that the mutation was
    detected. Every tamper test starts from this self-contained, verified copy.
    """
    root = _copy_index(recorded, tmp_path, name)
    index_path = root / "evidence-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["trace"]["path"] = str(root / "trace.jsonl")
    index["working_memory"]["path"] = str(root / "working-memory.json")
    index_path.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True),
                          encoding="utf-8")
    return root


def _mutate_index(root: Path, mutation) -> None:
    index_path = root / "evidence-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mutation(index)
    index_path.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True),
                          encoding="utf-8")


def _assert_mutation_refused(recorded, tmp_path: Path, name: str, mutation) -> None:
    """Prove the unmutated relocated copy first, then prove the one mutation is refused."""
    root = _copy_index_rebased(recorded, tmp_path, name)
    assert verify_evidence_index(root)["run_id"] == recorded["run_id"]
    _mutate_index(root, mutation)
    with pytest.raises(EvidenceError):
        verify_evidence_index(root)


def test_a_valid_relocated_copy_passes_before_any_mutation(recorded, tmp_path: Path) -> None:
    root = _copy_index_rebased(recorded, tmp_path, "positive-control")
    index = verify_evidence_index(root)
    assert index["run_id"] == recorded["run_id"]
    assert index["snapshots"]


def test_an_index_without_its_complete_structure_is_refused(recorded, tmp_path: Path) -> None:
    root = _copy_index_rebased(recorded, tmp_path, "minimal-index")
    assert verify_evidence_index(root)
    (root / "evidence-index.json").write_text(
        json.dumps({"kind": "web_runtime_evidence_index"}), encoding="utf-8")
    with pytest.raises(EvidenceError):
        verify_evidence_index(root)


def test_a_forged_or_incomplete_snapshot_inventory_is_refused(
        recorded, tmp_path: Path) -> None:
    _assert_mutation_refused(recorded, tmp_path, "removed-inventory",
                             lambda index: index.update(snapshots=[]))
    _assert_mutation_refused(recorded, tmp_path, "dropped-one",
                             lambda index: index["snapshots"].pop())


def test_an_extra_snapshot_on_disk_is_refused(recorded, tmp_path: Path) -> None:
    root = _copy_index_rebased(recorded, tmp_path, "extra-file")
    assert verify_evidence_index(root)
    (root / "snapshots" / "observation-9999.json").write_bytes(b"{}")
    with pytest.raises(EvidenceError):
        verify_evidence_index(root)


def test_a_changed_terminal_is_refused(recorded, tmp_path: Path) -> None:
    _assert_mutation_refused(recorded, tmp_path, "wrong-terminal",
                             lambda index: index["terminal"].update(
                                 terminal_status="cancelled", reason_code="USER_CANCELLED"))


def test_a_changed_trace_or_memory_digest_is_refused(recorded, tmp_path: Path) -> None:
    _assert_mutation_refused(recorded, tmp_path, "wrong-trace",
                             lambda index: index["trace"].update(sha256="0" * 64))
    _assert_mutation_refused(recorded, tmp_path, "wrong-memory",
                             lambda index: index["working_memory"].update(sha256="0" * 64))


def test_a_changed_event_association_is_refused(recorded, tmp_path: Path) -> None:
    _assert_mutation_refused(recorded, tmp_path, "wrong-association",
                             lambda index: index["contracts"][0].update(
                                 event_record_id="invented"))
    _assert_mutation_refused(recorded, tmp_path, "wrong-attempt",
                             lambda index: index["attempts"][0].update(
                                 event_record_id="invented"))
    _assert_mutation_refused(recorded, tmp_path, "wrong-verification",
                             lambda index: index["verifications"][0].update(
                                 event_record_id="invented"))


def test_a_missing_snapshot_directory_is_not_a_valid_empty_inventory(
        recorded, tmp_path: Path) -> None:
    root = _copy_index_rebased(recorded, tmp_path, "deleted-inventory")
    assert verify_evidence_index(root)
    _mutate_index(root, lambda index: index.update(snapshots=[]))
    shutil.rmtree(root / "snapshots")
    with pytest.raises(EvidenceError):
        verify_evidence_index(root)


# --- F04: the association rows are bound to their exact snapshot entry -------------------------


def test_a_swapped_contract_association_is_refused(recorded, tmp_path: Path) -> None:
    def mutation(index) -> None:
        rows = index["contracts"]
        assert len(rows) >= 2, rows
        rows[0]["path"] = rows[1]["path"]

    _assert_mutation_refused(recorded, tmp_path, "swapped-contract", mutation)


def test_a_changed_contract_bytes_or_digest_is_refused(recorded, tmp_path: Path) -> None:
    def mutation(index) -> None:
        rows = index["contracts"]
        rows[0]["bytes"] = rows[0]["bytes"] + 1

    _assert_mutation_refused(recorded, tmp_path, "contract-bytes", mutation)

    def digest_mutation(index) -> None:
        rows = index["contracts"]
        rows[0]["sha256"] = "0" * 64

    _assert_mutation_refused(recorded, tmp_path, "contract-digest", digest_mutation)


def test_a_swapped_attempt_association_is_refused(recorded, tmp_path: Path) -> None:
    def mutation(index) -> None:
        rows = index["attempts"]
        assert len(rows) >= 2, rows
        rows[0]["path"] = rows[1]["path"]

    _assert_mutation_refused(recorded, tmp_path, "swapped-attempt", mutation)


def test_a_swapped_verification_association_is_refused(recorded, tmp_path: Path) -> None:
    def mutation(index) -> None:
        rows = index["verifications"]
        assert len(rows) >= 2, rows
        rows[0]["path"] = rows[1]["path"]

    _assert_mutation_refused(recorded, tmp_path, "swapped-verification", mutation)


def test_an_invented_attempt_step_association_is_refused(recorded, tmp_path: Path) -> None:
    def mutation(index) -> None:
        index["attempts"][0]["step_id"] = "invented-step"

    _assert_mutation_refused(recorded, tmp_path, "invented-attempt-step", mutation)


def test_result_counters_that_contradict_the_trace_are_refused(recorded, tmp_path: Path) -> None:
    def mutation(index) -> None:
        index["result"]["steps_started"] = 999
        index["result"]["attempts_started"] = 999

    _assert_mutation_refused(recorded, tmp_path, "result-counters", mutation)


def test_a_finalization_that_contradicts_the_submitted_result_is_refused(
        recorded, tmp_path: Path) -> None:
    def mutation(index) -> None:
        index["last_finalization"]["primary_terminal_status"] = "failed"

    _assert_mutation_refused(recorded, tmp_path, "finalization-conflict", mutation)


# --- F04: a returned observation and an unreturned call are different facts --------------------


def _with_unreturned_call(root: Path) -> int:
    """Add one real-shaped observe call whose observation never returned.

    The call is appended to both the published Working Memory file and its index declaration, and
    the declaration's digest is recomputed from the new bytes. It has no snapshot, exactly like a
    real call interrupted before the adapter answered.
    """
    memory_path = root / "working-memory.json"
    document = json.loads(memory_path.read_text(encoding="utf-8"))
    calls = document["state"]["observation_calls"]
    ordinal = max(call["ordinal"] for call in calls) + 1
    calls.append({"ordinal": ordinal, "phase": "VERIFYING", "observation_id": None,
                  "step_id": None, "attempt_id": None})
    raw = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")
    memory_path.write_bytes(raw)
    _mutate_index(root, lambda index: index["working_memory"].update(
        observation_calls=calls, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    return ordinal


def test_a_published_call_that_never_returned_stays_without_a_snapshot(
        recorded, tmp_path: Path) -> None:
    root = _copy_index_rebased(recorded, tmp_path, "unreturned-call")
    assert verify_evidence_index(root)
    ordinal = _with_unreturned_call(root)
    index = verify_evidence_index(root)
    calls = index["working_memory"]["observation_calls"]
    assert calls[-1]["ordinal"] == ordinal
    assert calls[-1]["observation_id"] is None
    # The record is kept, not deleted to make the snapshot and call counts equal.
    assert len(calls) == len([entry for entry in index["snapshots"]
                              if entry["kind"] == "observation"]) + 1


def test_a_snapshot_cannot_claim_a_call_that_never_returned(recorded, tmp_path: Path) -> None:
    root = _copy_index_rebased(recorded, tmp_path, "bad-call-claim")
    assert verify_evidence_index(root)
    ordinal = _with_unreturned_call(root)
    assert verify_evidence_index(root)

    def mutation(index) -> None:
        entry = next(entry for entry in index["snapshots"]
                     if entry["kind"] == "observation")
        entry["runtime_call_ordinal"] = ordinal

    _mutate_index(root, mutation)
    with pytest.raises(EvidenceError):
        verify_evidence_index(root)


# --- F07: attributable host diagnostics before the Core Runtime wraps an exception -------------


def _finalize_zero_snapshot(root: Path, run_id: str, recorder: EvidenceRecorder) -> dict:
    """Write a real zero-snapshot terminal history, then finalize and fully verify the index."""
    from types import SimpleNamespace

    trace_path = root / "trace.jsonl"
    facts = (
        ("rt-record-0", 0, "run_started", None, None),
        ("rt-record-1", 1, "run_finished", "failure", "internal_error"),
    )
    trace_path.write_text("\n".join(json.dumps({
        "schema_version": "0.1", "record_id": record_id, "run_id": run_id,
        "sequence": sequence, "kind": kind,
        "terminal_status": terminal_status, "reason_code": reason_code,
    }) for record_id, sequence, kind, terminal_status, reason_code in facts) + "\n",
        encoding="utf-8")
    memory_path = root / "working-memory.json"
    state = {
        "schema_version": "0.1", "run_id": run_id, "goal_id": "goal-zero",
        "phase": "FAILED", "terminal_status": "failed",
        "reason_code": "DEPENDENCY_FAILURE", "steps_started": 0, "planning_calls": 0,
        "attempts_started": 0, "retries_used": 0, "replans_used": 0,
        "verification_record_id": None, "terminal_record_id": "rt-record-1",
        "observation_calls": [],
    }
    memory_path.write_text(json.dumps({"published_by": "runtime", "state": state},
                                      ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    projected = [
        SimpleNamespace(record_id=record_id, kind=kind, sequence=sequence,
                        step_id=None, contract_id=None, attempt_id=None, scope=None,
                        verification_status=None)
        for record_id, sequence, kind, _terminal, _reason in facts
    ]
    recorder.finalize(
        result=None,
        last_finalization={"run_id": run_id, "primary_terminal_status": "failed",
                           "primary_reason_code": "DEPENDENCY_FAILURE"},
        terminal={"terminal_status": "failed", "reason_code": "DEPENDENCY_FAILURE",
                  "source": "last_finalization"},
        trace_path=trace_path, working_memory_path=memory_path,
        working_state=state, trace_records=projected,
        cleanup={"adapter_cleanup_error": None, "browser": {}})
    return verify_evidence_index(root)


def _site_counts(**overrides: object) -> dict:
    counts = {
        "generation": 1, "scenario": "normal", "running": True,
        "login_requests": 0, "auth_successes": 0, "auth_failures": 0,
        "commits": 0, "active_requests": 0, "response_blocked": False,
        "authenticated_sessions": 0,
    }
    counts.update(overrides)
    return counts


def test_each_host_boundary_records_its_fixed_code_before_core_wraps_it(
        tmp_path: Path) -> None:
    class FailingInner:
        name = "playwright_web"

        def capabilities(self) -> object:
            return None

        def validate_action(self, action: object) -> object:
            return None

        async def start(self) -> object:
            raise WebBrowserError("WEB_BROWSER_START_FAILED", "start")

        async def stop(self) -> object:
            raise WebBrowserError("WEB_BROWSER_CLEANUP_TIMEOUT", "stop")

        async def observe(self) -> object:
            raise WebBrowserError("WEB_OBSERVE_TIMEOUT", "observe")

        async def execute(self, action: object) -> object:
            raise WebBrowserError("WEB_BROWSER_UNAVAILABLE", "execute")

    recorder = EvidenceRecorder(tmp_path, run_id="run-boundary", profile="normal")
    proxy = RecordingAdapter(FailingInner(), recorder, operation_timeout_ms=3000.0,
                             cleanup_timeout_ms=5000.0)

    async def scenario() -> None:
        for call in (proxy.start, proxy.observe):
            with pytest.raises(WebBrowserError):
                await call()
        with pytest.raises(WebBrowserError):
            await proxy.execute(object())
        with pytest.raises(WebBrowserError):
            await proxy.stop()

    asyncio.run(scenario())
    index = _finalize_zero_snapshot(tmp_path, "run-boundary", recorder)
    boundaries = index["diagnostics"]["boundaries"]
    assert [(row["operation"], row["outcome"], row["error_code"], row["budget_ms"])
            for row in boundaries] == [
        ("start", "failed", "WEB_BROWSER_START_FAILED", 3000),
        ("observe", "failed", "WEB_OBSERVE_TIMEOUT", 3000),
        ("execute", "failed", "WEB_BROWSER_UNAVAILABLE", 3000),
        ("stop", "failed", "WEB_BROWSER_CLEANUP_TIMEOUT", 5000),
    ]
    assert all(type(row["elapsed_ms"]) is int and row["elapsed_ms"] >= 0
               for row in boundaries)
    raw = (tmp_path / "evidence-index.json").read_text(encoding="utf-8")
    # The actual fixed code is attributable in the completed index.
    assert "WEB_BROWSER_START_FAILED" in raw
    # Only the closed code is stored; the fixed sentence itself is never copied.
    assert "the headless browser could not be started" not in raw


def test_the_startup_allowance_is_recorded_separately_from_operation_and_cleanup(
        tmp_path: Path) -> None:
    """C05-2: start records the startup allowance; observe/execute keep the operation value."""

    class FailingStartInner:
        name = "playwright_web"

        def capabilities(self) -> object:
            return None

        def validate_action(self, action: object) -> object:
            return None

        async def start(self) -> object:
            raise WebBrowserError("WEB_BROWSER_START_FAILED", "start")

        async def stop(self) -> object:
            raise WebBrowserError("WEB_BROWSER_CLEANUP_TIMEOUT", "stop")

        async def observe(self) -> object:
            raise WebBrowserError("WEB_OBSERVE_TIMEOUT", "observe")

        async def execute(self, action: object) -> object:
            raise WebBrowserError("WEB_BROWSER_UNAVAILABLE", "execute")

    recorder = EvidenceRecorder(tmp_path, run_id="run-startup", profile="normal")
    proxy = RecordingAdapter(FailingStartInner(), recorder, operation_timeout_ms=3000.0,
                             cleanup_timeout_ms=5000.0, startup_timeout_ms=10000.0)

    async def scenario() -> None:
        for call in (proxy.start, proxy.observe):
            with pytest.raises(WebBrowserError):
                await call()
        with pytest.raises(WebBrowserError):
            await proxy.execute(object())
        with pytest.raises(WebBrowserError):
            await proxy.stop()

    asyncio.run(scenario())
    index = _finalize_zero_snapshot(tmp_path, "run-startup", recorder)
    boundaries = index["diagnostics"]["boundaries"]
    assert [(row["operation"], row["outcome"], row["error_code"], row["budget_ms"])
            for row in boundaries] == [
        ("start", "failed", "WEB_BROWSER_START_FAILED", 10000),
        ("observe", "failed", "WEB_OBSERVE_TIMEOUT", 3000),
        ("execute", "failed", "WEB_BROWSER_UNAVAILABLE", 3000),
        ("stop", "failed", "WEB_BROWSER_CLEANUP_TIMEOUT", 5000),
    ]


def test_the_host_diagnostics_keep_login_received_counts_and_cleanup_facts(
        tmp_path: Path) -> None:
    class PassingInner:
        name = "playwright_web"

        def capabilities(self) -> object:
            return None

        def validate_action(self, action: object) -> object:
            return None

        async def start(self) -> object:
            return None

        async def stop(self) -> object:
            return None

        async def observe(self) -> object:
            raise AssertionError("the synthetic run must not observe")

        async def execute(self, action: object) -> object:
            return None

    recorder = EvidenceRecorder(tmp_path, run_id="run-host", profile="delayed_release")
    proxy = RecordingAdapter(PassingInner(), recorder, operation_timeout_ms=3000.0,
                             cleanup_timeout_ms=5000.0)
    asyncio.run(proxy.start())
    recorder.record_host_started(_site_counts())
    recorder.record_host_wait("release", received=True, elapsed_ms=7)
    recorder.record_host_counts("before_release", _site_counts(login_requests=1))
    recorder.record_host_cleanup(adapter_error_code=None,
                                 runtime_cleanup_status="succeeded",
                                 runtime_cleanup_error_code=None)
    index = _finalize_zero_snapshot(tmp_path, "run-host", recorder)

    assert index["diagnostics"]["boundaries"] == [{
        "operation": "start", "outcome": "ok", "error_code": None,
        "elapsed_ms": index["diagnostics"]["boundaries"][0]["elapsed_ms"],
        "budget_ms": 3000,
    }]
    host = index["diagnostics"]["host"]
    assert host["started"] == _site_counts()
    assert host["waits"] == [{"phase": "release", "received": True, "elapsed_ms": 7}]
    assert host["counts"]["before_release"]["login_requests"] == 1
    assert host["cleanup"] == {
        "adapter_error_code": None,
        "runtime_cleanup_status": "succeeded",
        "runtime_cleanup_error_code": None,
    }


def test_an_arbitrary_exception_sentence_is_never_recorded(tmp_path: Path) -> None:
    class Arbitrary(Exception):
        pass

    class RudeInner:
        name = "playwright_web"

        def capabilities(self) -> object:
            return None

        def validate_action(self, action: object) -> object:
            return None

        async def start(self) -> object:
            raise Arbitrary("SYNTHETIC-ARBITRARY-EXCEPTION-TEXT")

        async def stop(self) -> object:
            return None

        async def observe(self) -> object:
            raise AssertionError("unused")

        async def execute(self, action: object) -> object:
            return None

    recorder = EvidenceRecorder(tmp_path, run_id="run-rude", profile="normal")
    proxy = RecordingAdapter(RudeInner(), recorder, operation_timeout_ms=3000.0,
                             cleanup_timeout_ms=5000.0)
    with pytest.raises(Arbitrary):
        asyncio.run(proxy.start())
    index = _finalize_zero_snapshot(tmp_path, "run-rude", recorder)
    record = index["diagnostics"]["boundaries"][0]
    assert record["operation"] == "start"
    assert record["outcome"] == "failed"
    # No permitted fixed code exists for an arbitrary exception, so none is invented.
    assert record["error_code"] is None
    raw = (tmp_path / "evidence-index.json").read_text(encoding="utf-8")
    assert "SYNTHETIC-ARBITRARY-EXCEPTION-TEXT" not in raw


def test_an_undeclared_host_count_phase_is_refused(tmp_path: Path) -> None:
    recorder = EvidenceRecorder(tmp_path, run_id="run-phase", profile="normal")
    with pytest.raises(EvidenceError):
        recorder.record_host_counts("invented", _site_counts())


def test_a_tampered_host_diagnostic_is_refused(tmp_path: Path) -> None:
    recorder = EvidenceRecorder(tmp_path, run_id="run-tamper", profile="normal")
    recorder.record_host_started(_site_counts())
    assert _finalize_zero_snapshot(tmp_path, "run-tamper", recorder)

    def mutation(index) -> None:
        index["diagnostics"]["boundaries"] = [{
            "operation": "start", "outcome": "failed", "error_code": "ARBITRARY_CODE",
            "elapsed_ms": 0, "budget_ms": None,
        }]

    _mutate_index(tmp_path, mutation)
    with pytest.raises(EvidenceError):
        verify_evidence_index(tmp_path)


def test_an_undeclared_host_boundary_operation_is_refused(tmp_path: Path) -> None:
    recorder = EvidenceRecorder(tmp_path, run_id="run-op", profile="normal")
    assert _finalize_zero_snapshot(tmp_path, "run-op", recorder)

    def mutation(index) -> None:
        index["diagnostics"]["boundaries"].append({
            "operation": "invented", "outcome": "ok", "error_code": None,
            "elapsed_ms": 0, "budget_ms": None,
        })

    _mutate_index(tmp_path, mutation)
    with pytest.raises(EvidenceError):
        verify_evidence_index(tmp_path)
