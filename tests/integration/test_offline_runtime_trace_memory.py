"""C11 and C12: Runs over one Working Memory, and the host-side Episodic projection.

The Runtime consumes Working Memory directly and never writes Episodic Memory. This file therefore states the
two responsibilities separately: the Runtime's own published state is read back per run, and the Episodic
projection is written by the host only after a trace source has really been resolved. A source that cannot be
resolved produces no item, and an item is never allowed to turn a failed or cancelled result into a success.

All five terminal endings a Run can really reach - completed, failed, cancelled, timeout and abort - are
exercised here, because C11 requires one shared store to hold every one of them under its own Run identifier
and each terminal source label to stay exactly what the real trace fact says. The Episodic store's own source
check is exercised directly as well: a cited source that does not exist, belongs to another Run, is not a
Memory source at all, or disagrees with the result the item claims is refused before any byte is written.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import (
    FIXED_TIME,
    FLAG,
    SENSITIVE_MARKER,
    DeclaredAdapter,
    HostDomain,
    action,
    build_engine,
    failed,
    fake_scenario,
    forecast_for,
    json_rule_condition,
    kinds,
    no_step_scenario,
    planner_script,
    project_run,
    read_jsonl,
    read_records,
    request_for,
    terminal_records,
)
from forecontract.memory import (
    JsonlEpisodicMemory,
    MemoryItem,
    MemorySourceError,
    MemoryStorageError,
    MemoryValidationError,
    TraceSourceResolver,
    WorkingMemory,
)
from forecontract.runtime import RuntimeError as RuntimeRefusal
from forecontract.tracing import (
    JsonlTraceStore,
    RunStarted,
    TraceNotFoundError,
    TraceStorageError,
)
from forecontract_fake import FakeEnvironmentAdapter
from forecontract_scripted import ScriptedPlanner

#: The key-like tokens C12 forbids from every artefact. The first one is the suite's synthetic marker and is
#: deliberately not a secret; the rest are the shapes a real credential would take, never a real credential.
FORBIDDEN_TOKENS = (SENSITIVE_MARKER, "api_key", "apikey", "secret", "password", "bearer", "sk-")


def assert_no_forbidden_token(text: str, label: str) -> None:
    """Fail when one really-read artefact carries the synthetic marker or a key-like token."""
    lowered = text.lower()
    for token in FORBIDDEN_TOKENS:
        assert token.lower() not in lowered, f"the {label} artefact contains a forbidden token"


class RefusingTraceStore:
    """A trace writer that refuses every append and delegates every read to the real store.

    The Run still runs: the Runtime reports the store's refusal as its own bounded trace failure, and the
    proxy records how often the real target was invoked so a test can prove the refusal was really reached.
    """

    def __init__(self, inner: JsonlTraceStore) -> None:
        self._inner = inner
        self.append_calls = 0

    def append(self, record: object) -> None:
        self.append_calls += 1
        raise TraceStorageError("the trace file could not be written")

    def get_record(self, run_id: str, record_id: str):
        return self._inner.get_record(run_id, record_id)

    def read_run(self, run_id: str):
        return self._inner.read_run(run_id)


class RefusingObserver(DeclaredAdapter):
    """The real Fake with one single-point fault: the observation refuses at the adapter boundary."""

    async def observe(self):
        raise OSError("the environment refused the observation")


class BarrierObserver(DeclaredAdapter):
    """The real Fake with one declared barrier: the first observation waits until the host cancels the Run."""

    def __init__(self, inner: FakeEnvironmentAdapter, *, entered: asyncio.Event) -> None:
        super().__init__(inner)
        self._entered = entered
        self._first = True

    async def observe(self):
        if self._first:
            self._first = False
            self._entered.set()
            await asyncio.Event().wait()
        return await super().observe()


def test_two_runs_keep_their_published_state_separate_in_one_working_memory(workdir):
    """C11: the same Working Memory holds two runs' states without either overwriting the other."""
    done = action()
    shared = WorkingMemory()

    def run_one(run_id: str, *, satisfied: bool, scenario_id: str):
        scenario = (no_step_scenario(action_spec=done, state={FLAG: True}, scenario_id=scenario_id)
                    if satisfied else
                    fake_scenario(action_spec=done, start_state={FLAG: False},
                                  end_state={FLAG: True}, scenario_id=scenario_id))
        script = (planner_script([]) if satisfied else planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))]))
        engine, trace, _ = build_engine(
            tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
            planner=ScriptedPlanner(script), domain=HostDomain(), memory=shared,
            trace_name=f"{scenario_id}.jsonl")
        result = asyncio.run(engine.run(request=request_for(run_id=run_id,
                                                           conditions=[json_rule_condition()])))
        return result, trace

    first, first_trace = run_one("run-a", satisfied=True, scenario_id="scenario-a")
    second, second_trace = run_one("run-b", satisfied=False, scenario_id="scenario-b")

    assert first.run_id == "run-a"
    assert second.run_id == "run-b"
    # Both runs published into the same store, and each is still readable under its own identifier.
    state_a = shared.get("run-a")
    state_b = shared.get("run-b")
    assert state_a["run_id"] == "run-a"
    assert state_b["run_id"] == "run-b"
    assert state_a is not state_b
    # The two traces are separate files with one terminal fact each.
    assert len(terminal_records(read_records(first_trace, "run-a"))) == 1
    assert len(terminal_records(read_records(second_trace, "run-b"))) == 1


def test_a_successful_run_projects_one_success_item_from_its_confirmed_terminal_fact(workdir):
    """C11: the host projects a success experience only from a really resolved terminal source."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())

    result = asyncio.run(engine.run(request=request_for()))
    assert result.terminal_status.value == "completed"

    # The host resolves the source itself, then writes the item. The product is asked to do neither.
    resolver = TraceSourceResolver(trace)
    resolved = resolver.resolve("run-1", result.terminal_record_id)
    assert resolved.source_level == "run"
    assert resolved.result == "success"
    assert resolved.record_id == result.terminal_record_id

    episodic = JsonlEpisodicMemory(workdir / "episodic.jsonl", source_resolver=resolver)
    episodic.append(MemoryItem(schema_version="0.1", memory_id="memory-run-1", run_id="run-1",
                               source_level=resolved.source_level,
                               source_record_id=resolved.record_id, result=resolved.result,
                               created_at=FIXED_TIME))

    items = episodic.query(run_id="run-1")
    assert [item.memory_id for item in items] == ["memory-run-1"]
    assert items[0].result == "success"
    # The projection is queryable from the file after a fresh open, not only from the live object.
    reopened = JsonlEpisodicMemory(workdir / "episodic.jsonl", source_resolver=resolver)
    assert [item.memory_id for item in reopened.query(run_id="run-1")] == ["memory-run-1"]
    # And the cited source record really exists in the trace it names.
    records = read_records(trace, "run-1")
    assert items[0].source_record_id in {record.record_id for record in records}


def test_a_failed_run_is_not_projected_as_a_success_experience(workdir):
    """C11: a failed run's projection keeps the failure label rather than becoming a success."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: False},
                             outcome=failed("ao-1"))
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain(effect="applied"))

    result = asyncio.run(engine.run(request=request_for(max_steps=1, max_planning_calls=1)))
    assert result.terminal_status.value != "completed"

    resolver = TraceSourceResolver(trace)
    resolved = resolver.resolve("run-1", result.terminal_record_id)
    assert resolved.result != "success"
    episodic = JsonlEpisodicMemory(workdir / "episodic.jsonl", source_resolver=resolver)
    episodic.append(MemoryItem(schema_version="0.1", memory_id="memory-failed", run_id="run-1",
                               source_level=resolved.source_level,
                               source_record_id=resolved.record_id, result=resolved.result,
                               created_at=FIXED_TIME))
    assert episodic.get("memory-failed").result != "success"


def test_an_unresolvable_source_produces_no_projection(workdir):
    """C12: a source that cannot be resolved raises, and no item is invented for it."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())
    asyncio.run(engine.run(request=request_for()))

    # The identifier below really is absent from this Run's trace: the premise is read back, not assumed.
    present = {record.record_id for record in read_records(trace, "run-1")}
    assert "rt-record-absent" not in present

    resolver = TraceSourceResolver(trace)
    with pytest.raises(MemorySourceError, match="does not exist") as missing:
        resolver.resolve("run-1", "rt-record-absent")
    # The failure is the resolver's own bounded source error, never a silently invented terminal source.
    assert missing.value.code == "memory_source"

    episodic = JsonlEpisodicMemory(workdir / "episodic.jsonl", source_resolver=resolver)
    assert episodic.query(run_id="run-1") == ()
    assert read_jsonl(workdir / "episodic.jsonl") == []
    assert not (workdir / "episodic.jsonl").exists()


def test_an_episodic_write_refusal_is_reported_and_the_run_result_is_unchanged(workdir):
    """C12: a refused Episodic write is reported and never rewrites the real Run result."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())
    result = asyncio.run(engine.run(request=request_for()))
    before = result.model_dump(mode="json")

    resolver = TraceSourceResolver(trace)
    resolved = resolver.resolve("run-1", result.terminal_record_id)
    assert resolved.result == "success"

    # The host asks the store to write into a path that cannot be a file: the directory itself. The refusal
    # is the host's storage problem, and the product is not involved in it at all.
    episodic = JsonlEpisodicMemory(workdir, source_resolver=resolver)
    with pytest.raises(MemoryStorageError) as refusal:
        episodic.append(MemoryItem(schema_version="0.1", memory_id="memory-refused", run_id="run-1",
                                   source_level=resolved.source_level,
                                   source_record_id=resolved.record_id, result=resolved.result,
                                   created_at=FIXED_TIME))
    # The refusal is the store's own bounded storage diagnosis, not a silent success.
    assert refusal.value.code == "memory_storage"
    # The real Run result is byte-for-byte what it was before the host's write was refused.
    assert result.model_dump(mode="json") == before
    assert result.terminal_status.value == "completed"
    assert result.reason_code.value == "TASK_ACHIEVED"
    assert result.steps_started == 1
    assert result.attempts_started == 1
    assert result.terminal_record_id == resolved.record_id
    # No partial item reached the memory path either: the refused target is still the untouched directory,
    # and nothing else was written beside the Run's own trace file.
    assert workdir.is_dir()
    assert sorted(entry.name for entry in workdir.iterdir()) == ["trace.jsonl"]
    assert read_jsonl(workdir / "episodic.jsonl") == []


def test_a_clean_projection_round_trips_without_a_sensitive_marker(workdir):
    """C12: a normal projection is written and readable, and carries no sensitive marker."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())
    result = asyncio.run(engine.run(request=request_for()))

    resolver = TraceSourceResolver(trace)
    resolved = resolver.resolve("run-1", result.terminal_record_id)
    episodic = JsonlEpisodicMemory(workdir / "episodic.jsonl", source_resolver=resolver)
    episodic.append(MemoryItem(schema_version="0.1", memory_id="memory-clean", run_id="run-1",
                               source_level=resolved.source_level,
                               source_record_id=resolved.record_id, result=resolved.result,
                               created_at=FIXED_TIME))

    written = (workdir / "episodic.jsonl").read_text(encoding="utf-8")
    assert "memory-clean" in written
    assert SENSITIVE_MARKER not in written
    assert "api_key" not in written.lower()
    assert episodic.get("memory-clean").result == "success"


def test_one_working_memory_keeps_five_terminal_runs_and_their_labels_apart(workdir):
    """C11: five real terminal endings share one Working Memory and keep five exact terminal source labels."""
    done = action()
    shared = WorkingMemory()
    traces: dict[str, object] = {}
    terminals: dict[str, str] = {}

    def scenario_for(name: str):
        return fake_scenario(action_spec=done, start_state={FLAG: False},
                             end_state={FLAG: True}, scenario_id=name)

    def engine_for(name: str, adapter, **options):
        return build_engine(tmp_path=workdir, adapter=adapter, memory=shared,
                            planner=ScriptedPlanner(planner_script(
                                [("plan-1", {FLAG: False}, forecast_for(done))])),
                            domain=HostDomain(), trace_name=f"{name}.jsonl", **options)

    # 1. completed: one real action, Oracle PASS, the whole declared successful record sequence.
    success, success_trace, _ = engine_for(
        "success", DeclaredAdapter(FakeEnvironmentAdapter(scenario_for("success"))))
    success_result = asyncio.run(success.run(request=request_for(run_id="run-success")))
    assert success_result.terminal_status.value == "completed"
    assert success_result.reason_code.value == "TASK_ACHIEVED"
    assert kinds(read_records(success_trace, "run-success")) == [
        "run_started", "step_started", "contract_committed", "attempt_started", "attempt_finished",
        "verification_recorded", "verification_recorded", "run_finished"]
    traces["run-success"] = success_trace

    # 2. failed: a real dependency refusal at the observation boundary is published as a failure terminal
    #    fact and then propagated to the caller.
    failure, failure_trace, _ = engine_for(
        "failure", RefusingObserver(FakeEnvironmentAdapter(scenario_for("failure"))))
    with pytest.raises(RuntimeRefusal) as refusal:
        asyncio.run(failure.run(request=request_for(run_id="run-failure")))
    assert refusal.value.code == "DEPENDENCY_FAILURE"
    assert refusal.value.stage == "observe"
    assert kinds(read_records(failure_trace, "run-failure")) == ["run_started", "run_finished"]
    traces["run-failure"] = failure_trace

    # 3. cancelled: the host really cancels the in-flight Run once the observation has been entered.
    entered = asyncio.Event()
    cancelled, cancelled_trace, _ = engine_for(
        "cancelled", BarrierObserver(FakeEnvironmentAdapter(scenario_for("cancelled")), entered=entered))

    async def cancel_in_flight() -> None:
        task = asyncio.create_task(cancelled.run(request=request_for(run_id="run-cancelled")))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_in_flight())
    assert kinds(read_records(cancelled_trace, "run-cancelled")) == ["run_started", "run_finished"]
    traces["run-cancelled"] = cancelled_trace

    # 4. timeout: an immediately elapsed Run allowance, so no step and no attempt is ever started.
    timed_out, timeout_trace, _ = engine_for(
        "timeout", DeclaredAdapter(FakeEnvironmentAdapter(scenario_for("timeout"))), run_timeout_ms=0.0)
    timeout_result = asyncio.run(timed_out.run(request=request_for(run_id="run-timeout")))
    assert timeout_result.terminal_status.value == "timeout"
    assert timeout_result.reason_code.value == "DEADLINE_EXCEEDED"
    assert timeout_result.steps_started == 0
    assert timeout_result.attempts_started == 0
    assert kinds(read_records(timeout_trace, "run-timeout")) == ["run_started", "run_finished"]
    traces["run-timeout"] = timeout_trace

    # 5. aborted: the host declares no budget at all, which is an abort and never a timeout.
    aborted, abort_trace, _ = engine_for(
        "aborted", DeclaredAdapter(FakeEnvironmentAdapter(scenario_for("aborted"))), budget_ms=0.0)
    abort_result = asyncio.run(aborted.run(request=request_for(run_id="run-aborted")))
    assert abort_result.terminal_status.value == "aborted"
    assert abort_result.reason_code.value == "BUDGET_EXHAUSTED"
    assert abort_result.steps_started == 0
    assert abort_result.attempts_started == 0
    assert kinds(read_records(abort_trace, "run-aborted")) == ["run_started", "run_finished"]
    traces["run-aborted"] = abort_trace

    # C11: all five Runs published into the one shared store, each under its own identifier and each still
    # carrying its own exact terminal pair. An overwrite would surface here as another Run's pair.
    expected_pairs = {
        "run-success": ("completed", "TASK_ACHIEVED"),
        "run-failure": ("failed", "DEPENDENCY_FAILURE"),
        "run-cancelled": ("cancelled", "USER_CANCELLED"),
        "run-timeout": ("timeout", "DEADLINE_EXCEEDED"),
        "run-aborted": ("aborted", "BUDGET_EXHAUSTED"),
    }
    published = {run_id: shared.get(run_id) for run_id in expected_pairs}
    for run_id, state in published.items():
        assert state["run_id"] == run_id
        assert (state["terminal_status"], state["reason_code"]) == expected_pairs[run_id]

    # C11/C12: every published state names a terminal fact that really resolves in its own trace, and the
    # label is exactly what that fact maps to.
    labels = {}
    for run_id, trace in traces.items():
        finished = terminal_records(read_records(trace, run_id))
        assert len(finished) == 1
        terminals[run_id] = finished[0].record_id
        assert published[run_id]["terminal_record_id"] == finished[0].record_id
        resolved = TraceSourceResolver(trace).resolve(run_id, finished[0].record_id)
        assert resolved.source_level == "run"
        labels[run_id] = resolved.result
    assert labels == {"run-success": "success", "run-failure": "failure", "run-cancelled": "cancelled",
                      "run-timeout": "timeout", "run-aborted": "aborted"}

    # Each label survives a real store write and a fresh read of the file. The failed, cancelled, timeout and
    # aborted experiences stay readable as themselves and are never rewritten into a success.
    for run_id, label in labels.items():
        name = run_id.removeprefix("run-")
        path = workdir / f"episodic-{name}.jsonl"
        source = TraceSourceResolver(traces[run_id])
        JsonlEpisodicMemory(path, source_resolver=source).append(
            MemoryItem(schema_version="0.1", memory_id=f"memory-{name}", run_id=run_id,
                       source_level="run", source_record_id=terminals[run_id],
                       result=label, created_at=FIXED_TIME))
        reopened = JsonlEpisodicMemory(path, source_resolver=source)
        assert [stored.result for stored in reopened.query(run_id=run_id)] == [label]


def test_a_refusing_trace_store_leaves_no_confirmed_terminal_source_to_project(workdir):
    """C11: a Run whose trace store refused every write has no terminal fact, so nothing is projected."""
    done = action()
    shared = WorkingMemory()
    refusing = RefusingTraceStore(JsonlTraceStore(workdir / "refused.jsonl"))
    engine, _, _ = build_engine(
        tmp_path=workdir, memory=shared, trace_writer=refusing,
        adapter=DeclaredAdapter(FakeEnvironmentAdapter(
            fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True}))),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())

    with pytest.raises(RuntimeRefusal) as refusal:
        asyncio.run(engine.run(request=request_for()))
    # The real store target was invoked and refused both writes, and the refusal is a bounded facility error.
    assert refusing.append_calls == 2
    assert refusal.value.code == "TRACE_FAILURE"
    assert refusal.value.stage == "finalize"

    # The Run published its own state, and that state names no terminal record at all.
    published = shared.get("run-1")
    assert published["terminal_status"] == "failed"
    assert published["reason_code"] == "TRACE_FAILURE"
    assert published["terminal_record_id"] is None

    # There is no trace fact to resolve: the store holds nothing for this Run, so the real reader refuses and
    # the resolver refuses too, instead of the host inventing a terminal source.
    assert read_jsonl(workdir / "refused.jsonl") == []
    assert not (workdir / "refused.jsonl").exists()
    with pytest.raises(TraceNotFoundError):
        read_records(refusing, "run-1")
    resolver = TraceSourceResolver(refusing)
    with pytest.raises(MemoryValidationError):
        resolver.resolve("run-1", published["terminal_record_id"])
    with pytest.raises(MemorySourceError, match="does not exist"):
        resolver.resolve("run-1", "rt-record-no-such-fact")

    episodic = JsonlEpisodicMemory(workdir / "episodic.jsonl", source_resolver=resolver)
    assert episodic.query(run_id="run-1") == ()
    assert read_jsonl(workdir / "episodic.jsonl") == []
    assert not (workdir / "episodic.jsonl").exists()


def test_a_trace_without_a_terminal_fact_projects_no_item(workdir):
    """C11: a resolvable Run fact that is not a terminal fact is still not a terminal source."""
    store = JsonlTraceStore(workdir / "unterminated.jsonl")
    store.append(RunStarted(schema_version="0.1", record_id="rt-record-0",
                            run_id="run-unterminated", sequence=0, occurred_at=FIXED_TIME))
    records = read_records(store, "run-unterminated")
    assert kinds(records) == ["run_started"]
    assert terminal_records(records) == []

    # The Run fact is a real Memory source, but it maps to `unknown` and never to a terminal label: the host
    # therefore has no confirmed terminal source and must write no item.
    started = TraceSourceResolver(store).resolve("run-unterminated", "rt-record-0")
    assert started.source_level == "run"
    assert started.result == "unknown"

    episodic, resolver = project_run(store, workdir, "run-unterminated")
    assert episodic is None
    assert read_jsonl(workdir / "episodic.jsonl") == []
    assert not (workdir / "episodic.jsonl").exists()

    # The trace file really holds exactly the one fact that was appended to it.
    written = read_jsonl(workdir / "unterminated.jsonl")
    assert [record["kind"] for record in written] == ["run_started"]
    assert [record["run_id"] for record in written] == ["run-unterminated"]

    # A Run that was never written is refused as well: the real reader raises and the resolver raises.
    with pytest.raises(TraceNotFoundError):
        read_records(store, "run-never-written")
    with pytest.raises(MemorySourceError, match="does not exist"):
        resolver.resolve("run-never-written", "rt-record-0")


def test_a_cited_source_that_disagrees_with_the_item_is_rejected_without_an_append(workdir):
    """C12: the store's own source check refuses an item whose cited record cannot carry that result."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    engine, trace, _ = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())
    result = asyncio.run(engine.run(request=request_for()))
    assert result.terminal_status.value == "completed"

    records = read_records(trace, "run-1")
    terminal = terminal_records(records)[0]
    run_started = next(record for record in records if record.kind == "run_started")
    step_started = next(record for record in records if record.kind == "step_started")
    assert terminal.record_id == result.terminal_record_id
    assert len({terminal.record_id, run_started.record_id, step_started.record_id}) == 3

    episodic = JsonlEpisodicMemory(workdir / "episodic.jsonl",
                                  source_resolver=TraceSourceResolver(trace))

    def memory_item(**overrides):
        declared = {"schema_version": "0.1", "memory_id": "memory-1", "run_id": "run-1",
                    "source_level": "run", "source_record_id": terminal.record_id,
                    "result": "success", "created_at": FIXED_TIME}
        declared.update(overrides)
        return MemoryItem(**declared)

    # (a) the cited terminal fact really maps to `success`; claiming `failure` for it is a disagreement.
    with pytest.raises(MemorySourceError, match="disagrees with its trace source") as disagreement:
        episodic.append(memory_item(memory_id="memory-mislabelled", result="failure"))
    assert disagreement.value.code == "memory_source"
    assert disagreement.value.line_number is None
    # (b) a run-level fact is not an attempt-level source.
    with pytest.raises(MemorySourceError, match="uses a different source level"):
        episodic.append(memory_item(memory_id="memory-wrong-level", source_level="attempt"))
    # (c) a real record that is not a Memory source, and one that really does not exist, fail to resolve.
    with pytest.raises(MemorySourceError, match="could not resolve its trace source") as not_a_source:
        episodic.append(memory_item(memory_id="memory-step-cited",
                                    source_record_id=step_started.record_id))
    assert not_a_source.value.code == "memory_source"
    assert not_a_source.value.cause_name == "MemorySourceError"
    with pytest.raises(MemorySourceError, match="could not resolve its trace source"):
        episodic.append(memory_item(memory_id="memory-absent", source_record_id="rt-record-absent",
                                    result="unknown"))
    # (d) a real fact of this Run is not a source of another Run.
    with pytest.raises(MemorySourceError, match="could not resolve its trace source"):
        episodic.append(memory_item(memory_id="memory-other-run", run_id="run-other"))
    # (e) this Run's own start fact maps to `unknown`, so it cannot be stored as a success experience.
    with pytest.raises(MemorySourceError, match="disagrees with its trace source"):
        episodic.append(memory_item(memory_id="memory-start-as-success",
                                    source_record_id=run_started.record_id))

    # None of the refusals appended anything: the Memory file was never created at all.
    assert episodic.query(run_id="run-1") == ()
    assert read_jsonl(workdir / "episodic.jsonl") == []
    assert not (workdir / "episodic.jsonl").exists()

    # Positive control: the same cited terminal fact with the result it really maps to is accepted, so the
    # refusals above are about the disagreement and not about a store that rejects everything.
    episodic.append(memory_item(memory_id="memory-accepted"))
    assert [(record["memory_id"], record["result"])
            for record in read_jsonl(workdir / "episodic.jsonl")] == [("memory-accepted", "success")]


def test_a_normal_run_leaves_no_forbidden_marker_in_trace_memory_or_projection(workdir):
    """C12: a normal Run's trace, published Memory read-back and projection carry no forbidden marker."""
    done = action()
    scenario = fake_scenario(action_spec=done, start_state={FLAG: False}, end_state={FLAG: True})
    engine, trace, memory = build_engine(
        tmp_path=workdir, adapter=DeclaredAdapter(FakeEnvironmentAdapter(scenario)),
        planner=ScriptedPlanner(planner_script(
            [("plan-1", {FLAG: False}, forecast_for(done))])), domain=HostDomain())
    result = asyncio.run(engine.run(request=request_for()))
    assert result.terminal_status.value == "completed"

    # The trace really holds the declared successful sequence, and every record belongs to this Run.
    trace_records = read_jsonl(workdir / "trace.jsonl")
    assert [record["kind"] for record in trace_records] == [
        "run_started", "step_started", "contract_committed", "attempt_started", "attempt_finished",
        "verification_recorded", "verification_recorded", "run_finished"]
    assert {record["run_id"] for record in trace_records} == {"run-1"}

    # The Working Memory read-back is a real detached read of what the Run published.
    published = memory.get("run-1")
    assert published["run_id"] == "run-1"
    assert published["terminal_status"] == "completed"

    # The projection is a real host write from the confirmed terminal source.
    resolver = TraceSourceResolver(trace)
    resolved = resolver.resolve("run-1", result.terminal_record_id)
    episodic = JsonlEpisodicMemory(workdir / "episodic.jsonl", source_resolver=resolver)
    episodic.append(MemoryItem(schema_version="0.1", memory_id="memory-sweep", run_id="run-1",
                               source_level=resolved.source_level,
                               source_record_id=resolved.record_id, result=resolved.result,
                               created_at=FIXED_TIME))

    artefacts = {
        "trace": (workdir / "trace.jsonl").read_text(encoding="utf-8"),
        "working memory": json.dumps(published, ensure_ascii=False, sort_keys=True),
        "episodic": (workdir / "episodic.jsonl").read_text(encoding="utf-8"),
    }
    for label, text in artefacts.items():
        assert text, f"the {label} artefact is empty"
        assert_no_forbidden_token(text, label)
    # The sweep read real content: the declared memory identifier reached the file it was written to.
    assert "memory-sweep" in artefacts["episodic"]
