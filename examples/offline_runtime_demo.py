#!/usr/bin/env python
"""P2-T08 offline runtime demo: one real action through the real installed components.

This example runs one closed offline success loop with no network, no credentials and no model. It drives the
installed Core Runtime with the installed Fake environment and the installed ScriptedPlanner, writes a real
JSONL Trace and a real Working Memory state, and then projects the confirmed terminal fact into an Episodic
Memory record from the host side.

Two responsibilities that the product deliberately leaves to the host are spelled out here:

* the **domain binding** - the Runtime calls ``to_evidence``, ``prepare_contract``,
  ``declare_action_safety`` and ``assess_effect`` on the object the host passes as ``domain_support``;
* the **capability declaration** - the Fake declares its own supported outcome types, so a thin proxy adds
  the ``json_rule`` binding this host is asking for while every lifecycle, observation, validation and
  execution call still goes to the real Fake.

The example never reads the Fake's private actual state, never decides task completion itself, and never
installs a dependency or edits ``sys.path``: it imports the installed distributions as a normal consumer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from forecontract.contracts import (
    ContractBuilder,
    RecoveryPolicy,
    VerificationBinding,
    VerificationPolicy,
)
from forecontract.memory import (
    JsonlEpisodicMemory,
    MemoryItem,
    TraceSourceResolver,
    WorkingMemory,
)
from forecontract.models import (
    ActionSpec,
    ActualOutcome,
    ExpectedOutcome,
    Forecast,
    Observation,
)
from forecontract.recovery import ActionSafetyDeclaration
from forecontract.runtime import (
    ContractPreparation,
    MonotonicBudgetSource,
    RunRequest,
    RuntimeEngine,
    RuntimeTimingPolicy,
)
from forecontract.tracing import JsonlTraceStore
from forecontract.verification import TaskCompletionCriteria, VerificationEvidence
from forecontract_fake import (
    FakeEnvironmentAdapter,
    FakeScenario,
    FakeStage,
    FakeStep,
)
from forecontract_scripted import PlannerScript, PlannerStep, ScriptedPlanner

SCHEMA = "0.1"
FLAG = "done"
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

#: The synthetic goal this demo solves. Exactly one real action is required.
GOAL_ID = "demo-goal"
RUN_ID = "demo-run"
ACTION_TYPE = "mark_done"


def completion_condition() -> ExpectedOutcome:
    """The one completion condition: the observation's own state must report the flag as set."""
    return ExpectedOutcome(
        schema_version=SCHEMA, outcome_type="json_rule",
        payload={"schema_version": SCHEMA, "rule_type": "value_equals",
                 "path": [FLAG], "expected": True})


def observation(observation_id: str, state: dict) -> Observation:
    return Observation(schema_version=SCHEMA, observation_id=observation_id,
                       environment_type="fake", state=state, timestamp=NOW)


class DemoDomain:
    """The host's explicit domain binding.

    ``to_evidence`` states whether the observation carries complete evidence. ``prepare_contract`` declares
    the verification binding and the recovery policy. ``declare_action_safety`` states the retry safety of
    the action. ``assess_effect`` reports whether the observed outcome applied the action's effect, using only
    the evidence it was handed.
    """

    def to_evidence(self, seen: Observation) -> VerificationEvidence:
        return VerificationEvidence(schema_version=SCHEMA, observation=seen, state_complete=True)

    def prepare_contract(self, *, forecast: Forecast,
                         evidence: VerificationEvidence) -> ContractPreparation:
        return ContractPreparation(
            schema_version=SCHEMA, preconditions=[],
            verification_policy=VerificationPolicy(
                schema_version=SCHEMA,
                bindings=[VerificationBinding(schema_version=SCHEMA, outcome_index=index,
                                              verifier_type="json_rule")
                          for index in range(len(forecast.expected_outcomes))]),
            recovery_policy=RecoveryPolicy(schema_version=SCHEMA, mode="bounded",
                                           max_retries=1, allow_replan=False),
            timeout_ms=5000.0)

    def declare_action_safety(self, *, contract: object) -> ActionSafetyDeclaration:
        return ActionSafetyDeclaration(
            schema_version=SCHEMA, contract_id=contract.contract_id,
            adapter_name=contract.adapter_name, action_type=contract.action.action_type,
            retry_safety="idempotent")

    def assess_effect(self, *, contract: object, actual_outcome: ActualOutcome,
                      evidence: VerificationEvidence) -> str:
        """Judge the effect from the evidence the host actually received, never from a receipt alone.

        The declaration is decided in this order, and each step says what it really knows:

        * without complete evidence there is nothing to judge against, so the effect is `unknown`;
        * the outcome's own `state_after`, when the adapter reported one, is the environment's word about
          the effect, and it decides `applied` or `not_applied` on its own;
        * otherwise a succeeded execution is evidence that the effect applied, and a failed or unknown one
          is evidence that it did not - but a failed or unknown receipt is never allowed to *claim* the
          effect applied, and the goal's own truth is still decided only by the task oracle.
        """
        if not evidence.state_complete:
            return "unknown"
        # The receipt's own reported state is the environment's direct word about the effect.
        reported = actual_outcome.state_after
        if isinstance(reported, dict) and FLAG in reported:
            return "applied" if reported.get(FLAG) is True else "not_applied"
        # The latest valid observation is equally decisive: if it carries the effect field, that field - not
        # the receipt's status - decides. A failed or unknown receipt therefore cannot downgrade evidence that
        # already shows the effect applied.
        observed = evidence.observation.state
        if FLAG not in observed:
            return "unknown"
        return "applied" if observed.get(FLAG) is True else "not_applied"


class DeclaredAdapter:
    """A thin proxy that keeps the real Fake's behaviour and declares the host's outcome binding."""

    def __init__(self, inner: FakeEnvironmentAdapter) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.capabilities().adapter_name

    def capabilities(self):
        real = self._inner.capabilities()
        return real.model_copy(update={"supported_outcome_types": ["json_rule"]})

    def validate_action(self, action: ActionSpec) -> None:
        self._inner.validate_action(action)

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()

    async def observe(self) -> Observation:
        return await self._inner.observe()

    async def execute(self, action: ActionSpec) -> ActualOutcome:
        return await self._inner.execute(action)


def build_components():
    """Assemble the synthetic task: one action, one observation before, one after."""
    done = ActionSpec(schema_version=SCHEMA, action_type=ACTION_TYPE, payload={})
    scenario = FakeScenario(
        schema_version=SCHEMA, scenario_id="demo-scenario", adapter_name="fake",
        allowed_actions=[done],
        stages=[
            FakeStage(schema_version=SCHEMA, actual_state={FLAG: False},
                      observation=observation("demo-obs-before", {FLAG: False})),
            FakeStage(schema_version=SCHEMA, actual_state={FLAG: True},
                      observation=observation("demo-obs-after", {FLAG: True})),
        ],
        steps=[FakeStep(schema_version=SCHEMA, step_id="demo-step", action=done,
                        outcome=ActualOutcome(schema_version=SCHEMA, outcome_id="demo-outcome",
                                              status="succeeded", state_after={FLAG: True}))])
    forecast = Forecast(schema_version=SCHEMA, forecast_id="demo-forecast", proposed_action=done,
                        expected_outcomes=[completion_condition()])
    script = PlannerScript(
        schema_version=SCHEMA, script_id="demo-script",
        steps=[PlannerStep(schema_version=SCHEMA, step_id="demo-plan",
                           expected_environment_type="fake",
                           expected_state={FLAG: False}, forecast=forecast)])
    request = RunRequest(
        schema_version=SCHEMA, run_id=RUN_ID, goal_id=GOAL_ID,
        task_criteria=TaskCompletionCriteria(schema_version=SCHEMA, goal_id=GOAL_ID,
                                             conditions=[completion_condition()]),
        max_steps=3, max_planning_calls=2, max_retries=1, max_replans=0)
    return DeclaredAdapter(FakeEnvironmentAdapter(scenario)), ScriptedPlanner(script), request


#: The fixed, value-free diagnostic the example prints when a stage fails. It names the stage and nothing
#: about the failure itself, so no dependency text, class name, repr or cause can reach the output.
_FAILURE_DIAGNOSTICS = {
    "runtime": "the offline runtime run did not complete",
    "projection": "the host episodic projection did not complete",
    "output": "the requested output directory could not be prepared",
    "summary": "the example produced an unexpected summary",
    "unknown": "the offline runtime demo did not complete",
}


class DemoFailure(Exception):
    """A bounded example failure: the stage that failed, and one fixed message.

    The example deliberately does not carry the original exception, its type name, its text or its cause
    chain. A demo that echoed any of those would turn its own stdout and stderr into a channel for
    dependency text, which is exactly what the bounded diagnostic contract exists to prevent.
    """

    def __init__(self, stage: str, detail: str | None = None) -> None:
        self.stage = stage
        # ``detail`` is only ever one of the fixed sentences below - never a path, never an exception text.
        message = detail or _FAILURE_DIAGNOSTICS.get(
            stage, "the offline runtime demo did not complete")
        super().__init__(message)


#: The fixed sentences for the output-preparation refusals. One is chosen by rule, never formatted from input.
_OUTPUT_REFUSALS = {
    "repository": "refusing to write inside the repository",
    "not_a_directory": "refusing to overwrite a path that is not a directory",
    "not_empty": "refusing to write into a non-empty directory",
    "unavailable": "the requested output directory could not be prepared",
}


def resolve_output_dir(raw: str | None) -> Path:
    """Resolve the output directory, refusing unsafe targets and bounded on every filesystem failure.

    Preparation is part of the example, so it is part of the same bounded failure contract: a refusal of the
    policy rules, and any ``OSError`` the filesystem raises while the target is inspected or created, is
    reported as the ``output`` stage. The refused path and the operating-system text are never echoed.
    """
    repository = Path(__file__).resolve().parents[1]
    if raw is None:
        try:
            return Path(tempfile.mkdtemp(prefix="offline-runtime-demo-"))
        except OSError:
            raise DemoFailure("output") from None
    try:
        target = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        raise DemoFailure("output") from None
    if target == repository or repository in target.parents:
        # The refused path is not echoed: the rule that refused it is what the caller needs to know.
        raise DemoFailure("output", _OUTPUT_REFUSALS["repository"])
    try:
        if target.exists():
            if not target.is_dir():
                raise DemoFailure("output", _OUTPUT_REFUSALS["not_a_directory"])
            if any(target.iterdir()):
                raise DemoFailure("output", _OUTPUT_REFUSALS["not_empty"])
        else:
            target.mkdir(parents=True)
    except DemoFailure:
        raise
    except OSError:
        # A denied ACL, a read-only parent or any other filesystem refusal lands here rather than escaping as
        # a raw traceback.
        raise DemoFailure("output") from None
    return target


def run_demo(output_dir: Path) -> dict:
    """Run the closed loop and return the summary object the CLI prints."""
    adapter, planner, request = build_components()
    trace_path = output_dir / "trace.jsonl"
    episodic_path = output_dir / "episodic.jsonl"
    memory_path = output_dir / "working-memory.json"
    trace = JsonlTraceStore(trace_path)
    working = WorkingMemory()
    engine = RuntimeEngine(
        adapter=adapter, planner=planner, domain_support=DemoDomain(),
        budget_source=MonotonicBudgetSource(total_budget_ms=30000.0),
        trace_writer=trace, working_memory=working, builder=ContractBuilder(),
        timing_policy=RuntimeTimingPolicy(schema_version=SCHEMA, run_timeout_ms=20000.0,
                                          cleanup_timeout_ms=2000.0, finalize_timeout_ms=2000.0))
    try:
        result = asyncio.run(engine.run(request=request))
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise DemoFailure("runtime") from None

    # The host reads back the safe Working Memory state the Runtime published - a detached read, not the
    # Runtime's own object - and then projects the confirmed terminal fact into Episodic Memory itself. The
    # Runtime never writes Episodic Memory and this example never asks it to.
    try:
        published = working.get(RUN_ID)
        memory_path.write_text(json.dumps({"published_by": "runtime", "state": published},
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        resolver = TraceSourceResolver(trace)
        resolved = resolver.resolve(RUN_ID, result.terminal_record_id)
        episodic = JsonlEpisodicMemory(episodic_path, source_resolver=resolver)
        episodic.append(MemoryItem(schema_version=SCHEMA, memory_id="demo-memory", run_id=RUN_ID,
                                   source_level=resolved.source_level,
                                   source_record_id=resolved.record_id,
                                   result=resolved.result, created_at=NOW))
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise DemoFailure("projection") from None
    return {
        "result": result.model_dump(mode="json"),
        "trace_path": str(trace_path),
        "episodic_memory_path": str(episodic_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one offline Fake-backed Runtime success loop and project its terminal fact.")
    parser.add_argument("--output-dir", default=None,
                        help="an external new-or-empty directory; defaults to a fresh temporary directory")
    args = parser.parse_args(argv)
    # The whole invocation is one boundary: preparation, the run, the projection and the summary check. Every
    # refusal - a policy refusal, a filesystem error while the target is prepared, a run or projection failure -
    # ends in one fixed message chosen from the declared table, and never in a traceback, an exception name, a
    # path or a cause. Control flow is never converted: an interrupt or a process exit is re-raised untouched.
    try:
        output_dir = resolve_output_dir(args.output_dir)
        summary = run_demo(output_dir)
        if set(summary) != {"result", "trace_path", "episodic_memory_path"}:
            raise DemoFailure("summary")
    except DemoFailure as failure:
        print(f"offline runtime demo failed: {failure}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 - the message is a fixed sentence, never the exception
        print(f"offline runtime demo failed: {_FAILURE_DIAGNOSTICS['unknown']}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
