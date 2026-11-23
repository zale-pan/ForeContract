#!/usr/bin/env python
"""Eight real Web Runtime profiles over the fixed local test site.

This example composes the already-verified components the way a host does: the real
``RuntimeEngine``, ``ContractBuilder``, ``ScriptedPlanner``, ``PlaywrightWebAdapter``,
``WebRuntimeDomainSupport``, ``WebContractHooksFactory``, Web step/task verifiers, the real
``RecoveryEngine``, ``MonotonicBudgetSource``, ``WorkingMemory`` and ``JsonlTraceStore``. The
only test fixture is the loopback-only ``test_site``; every profile starts a real browser, uses
its own run identifier and writes to an external output directory.

The script has exactly four actions: navigate to ``/login``, fill the username, fill the
password and click submit. Each action gets at most one Attempt and no retry or replan. Every
profile declares the frozen standard ``bounded`` policy with ``max_retries=0`` and
``allow_replan=False``. The host composes the generic Core ``RecoveryEngine`` with the explicitly
declared ``web_step_verifier`` identity, so the real Web step verdict is accepted natively: the
engine is never relabelled, never fed a rewritten verifier field and never bypassed with an
``abort`` policy. The unknown-submit profile therefore keeps both facts at once - the Runtime
terminal ``POLICY_ABORT`` and the Recovery reason ``EFFECT_UNRESOLVED`` - with exactly one submit
and zero replay.

The task criteria are supplied independently of the script and state the same final goal (the
dashboard URL and the welcome heading). The host also injects ``WebTaskCriteriaValidator``, so the
Web criteria grammar is prevalidated before any trace write, ``start`` or ``observe``.

The eight profiles are the concrete contract's eight:

``normal``, ``wrong_credentials``, ``delayed_release``, ``delayed_deadline``,
``overlay_blocked``, ``overlay_released``, ``commit_unknown`` and ``cancel_after_received``.

Every profile declares the same explicit ``startup_timeout_ms=10000``. That is a *startup*
allowance: it bounds only the one browser resource acquisition of ``start``, with no retry, warm-up
or phase reset. The frozen operation, per-target, Run, request and cleanup budgets are unchanged,
and the host-observed whole-call elapsed time of a failed start includes the bounded failed-start
cleanup, so it is not evidence of the native browser start-up duration alone.

Two host responsibilities are deliberately explicit:

* **the recovery route** - the generic ``RecoveryEngine(step_verifier_name="web_step_verifier")``
  checks the true Web verifier label through its constructor-declared dependency identity. The
  decision order, bounded policy and the single-submit rule are unchanged;
* **the evidence sidecar** - the Core JSONL trace schema is unchanged. Complete Observation,
  Forecast, Contract, ActualOutcome and verification snapshots are written by the host-only
  forwarding proxies in ``web_runtime_evidence``.

No API key, production site, remote publication or arbitrary script execution is involved. The
local fixture credentials are public synthetic data.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Final

from forecontract.contracts import (
    ContractBuilder,
)
from forecontract.contracts.models import RecoveryPolicy
from forecontract.memory import WorkingMemory
from forecontract.models.actions import (
    ActionSpec,
    ExpectedOutcome,
    Forecast,
    Precondition,
)
from forecontract.models.base import SCHEMA_VERSION
from forecontract.recovery import RecoveryEngine
from forecontract.runtime import (
    MonotonicBudgetSource,
    RunRequest,
    RuntimeEngine,
    RuntimeTimingPolicy,
)
from forecontract.tracing import JsonlTraceStore
from forecontract.verification import TaskCompletionCriteria
from forecontract_scripted import ScriptedPlanner
from forecontract_web.browser_config import PlaywrightWebConfig
from forecontract_web.models import RoleTarget, TestIdTarget
from forecontract_web.playwright_adapter import PlaywrightWebAdapter
from forecontract_web.runtime_hooks import (
    WebContractHooksFactory,
    WebTaskCriteriaValidator,
)
from forecontract_web.runtime_support import (
    WebActionPolicy,
    WebRuntimeDomainSupport,
)
from forecontract_web.verification import WebStepVerifier, WebTaskCompletionOracle
from web_runtime_evidence import (
    EvidenceError,
    EvidenceRecorder,
    RecordingAdapter,
    RecordingBuilder,
    RecordingPlanner,
    RecordingRecoveryEngine,
    RecordingStepVerifier,
    RecordingTaskOracle,
    verify_evidence_index,
)

from test_site import LocalTestSite, Scenario, SiteConfig, SiteEvent
from test_site.pages import FIXTURE_PASSWORD, FIXTURE_USERNAME

SCHEMA: Final[str] = SCHEMA_VERSION
PROFILES: Final[tuple[str, ...]] = (
    "normal",
    "wrong_credentials",
    "delayed_release",
    "delayed_deadline",
    "overlay_blocked",
    "overlay_released",
    "commit_unknown",
    "cancel_after_received",
)

#: The public synthetic credentials of the local fixture. They are not secrets.
USERNAME: Final[str] = FIXTURE_USERNAME
PASSWORD: Final[str] = FIXTURE_PASSWORD
WRONG_PASSWORD: Final[str] = "wrong-synthetic-password"
SUBMIT_TEXT: Final[str] = "登录"
WRONG_ERROR_TEXT: Final[str] = "用户名或密码错误"

#: The three targets observed for every snapshot. The login page also has an error marker and an
#: overlay, but those are not needed by this four-action script and are not observed.
TARGET_NAMES: Final[tuple[str, ...]] = ("username", "password", "submit", "welcome")

#: The frozen standard recovery policy: bounded, zero retries, no replanning. Every profile uses
#: it, including the failing ones: the generic Recovery identity makes the real Web verdict usable
#: without relabelling it and without bypassing the policy.
STANDARD_RECOVERY = RecoveryPolicy(
    schema_version=SCHEMA, mode="bounded", max_retries=0, allow_replan=False)

#: The real Web step-verifier identity the host composes. It is declared to the generic Core
#: recovery engine at construction; nothing is rewritten and no whitelist grows at runtime.
WEB_STEP_VERIFIER_NAME: Final[str] = "web_step_verifier"

#: The two cleanup/finalisation allowances of the Runtime and the browser/site allowances.
RUNTIME_CLEANUP_MS: Final[float] = 5000.0
RUNTIME_FINALIZE_MS: Final[float] = 5000.0
ADAPTER_CLEANUP_MS: Final[float] = 5000.0
SITE_CLEANUP_MS: Final[int] = 3000
HOST_WAIT_MS: Final[int] = 10000
#: The explicit startup allowance declared by every profile. It bounds only the browser resource
#: acquisition of ``start``; it is a startup-only allowance and is not derived from any observed
#: first-failure duration. It does not replace or widen the frozen operation/target/Run/cleanup
#: budgets, and the first real cold start must still be measured on its own.
ADAPTER_STARTUP_MS: Final[int] = 10000


class DemoFailure(Exception):
    """A bounded example failure: a stage name, a fixed message and the original in-process cause."""

    def __init__(self, stage: str, message: str, *, original: BaseException | None = None) -> None:
        self.stage = stage
        self.original = original
        super().__init__(message)


_FIXED_FAILURES: Final[dict[str, str]] = {
    "arguments": "the requested profile or output directory is not usable",
    "site_start": "the local test site could not start",
    "runtime": "the web runtime run did not complete",
    "evidence": "the evidence snapshots could not be completed",
    "cleanup": "the browser or site resources were not completely released",
    "summary": "the example produced an unexpected summary",
}


def failure(stage: str, original: BaseException | None = None) -> DemoFailure:
    """Build one fixed failure outside a handler."""
    return DemoFailure(stage, _FIXED_FAILURES.get(stage, _FIXED_FAILURES["runtime"]),
                       original=original)


# --------------------------------------------------------------------------- fixtures


def _by_test_id(name: str) -> TestIdTarget:
    return TestIdTarget(strategy="test_id", test_id=name)


TARGETS: Final[tuple[TestIdTarget, ...]] = tuple(_by_test_id(name) for name in TARGET_NAMES)
SUBMIT_TARGET: Final[TestIdTarget] = _by_test_id("submit")
WELCOME_TARGET: Final[TestIdTarget] = _by_test_id("welcome")


def _target_payload(target: TestIdTarget | RoleTarget) -> dict:
    return target.model_dump()


def _precondition(rule_type: str, payload: dict) -> Precondition:
    return Precondition(schema_version=SCHEMA, condition_type=rule_type,
                        payload=payload, required=True)


def _outcome(rule_type: str, payload: dict) -> ExpectedOutcome:
    return ExpectedOutcome(schema_version=SCHEMA, outcome_type=rule_type,
                           payload=payload, required=True)


def _url_precondition(url: str) -> Precondition:
    return _precondition("web.url_match", {"url": url})


def _visible_precondition(target: TestIdTarget | RoleTarget) -> Precondition:
    return _precondition("web.element_visible", {"target": _target_payload(target)})


def _url_outcome(url: str) -> ExpectedOutcome:
    return _outcome("web.url_match", {"url": url})


def _visible_outcome(target: TestIdTarget | RoleTarget) -> ExpectedOutcome:
    return _outcome("web.element_visible", {"target": _target_payload(target)})


def _value_outcome(target: TestIdTarget | RoleTarget, text: str) -> ExpectedOutcome:
    return _outcome("web.value_equals", {"target": _target_payload(target), "value": text})


def _target_record(target: TestIdTarget, count: int, *, visible: bool | None = None,
                   text: str | None = None, value: str | None = None) -> dict:
    return {
        "target": _target_payload(target),
        "status": "collected",
        "match_count": count,
        "visible": visible,
        "text": text,
        "value": value,
        "error_code": None,
    }


def _state(url: str, records: list[dict]) -> dict:
    return {"schema_version": SCHEMA, "url": url, "targets": records}


def _blank_state() -> dict:
    return _state("about:blank", [_target_record(target, 0) for target in TARGETS])


def _login_state(origin: str, username_value: str, password_value: str) -> dict:
    return _state(
        f"{origin}/login",
        [
            _target_record(TARGETS[0], 1, visible=True, text="", value=username_value),
            _target_record(TARGETS[1], 1, visible=True, text="", value=password_value),
            _target_record(TARGETS[2], 1, visible=True, text=SUBMIT_TEXT, value=None),
            _target_record(TARGETS[3], 0),
        ],
    )


def _actions(origin: str, password: str) -> dict[str, ActionSpec]:
    login = f"{origin}/login"
    return {
        "login": login,
        "dashboard": f"{origin}/dashboard",
        "navigate": ActionSpec(schema_version=SCHEMA, action_type="web.navigate",
                               payload={"url": login}),
        "fill_username": ActionSpec(
            schema_version=SCHEMA, action_type="web.fill",
            payload={"target": _target_payload(TARGETS[0]), "text": USERNAME}),
        "fill_password": ActionSpec(
            schema_version=SCHEMA, action_type="web.fill",
            payload={"target": _target_payload(TARGETS[1]), "text": password}),
        "click": ActionSpec(
            schema_version=SCHEMA, action_type="web.click",
            payload={"target": _target_payload(SUBMIT_TARGET)}),
    }


def _policies(origin: str, password: str, *, click_timeout_ms: float,
              recovery_policy: RecoveryPolicy) -> list[WebActionPolicy]:
    actions = _actions(origin, password)
    login = actions["login"]
    dashboard = actions["dashboard"]
    return [
        WebActionPolicy(
            action=actions["navigate"],
            preconditions=[],
            retry_safety="read_only",
            applied_when=[],
            not_applied_when=[],
            recovery_policy=recovery_policy,
            timeout_ms=5000.0,
        ),
        WebActionPolicy(
            action=actions["fill_username"],
            preconditions=[_url_precondition(login), _visible_precondition(TARGETS[0])],
            retry_safety="idempotent",
            applied_when=[_value_outcome(TARGETS[0], USERNAME)],
            not_applied_when=[],
            recovery_policy=recovery_policy,
            timeout_ms=5000.0,
        ),
        WebActionPolicy(
            action=actions["fill_password"],
            preconditions=[_url_precondition(login), _visible_precondition(TARGETS[1])],
            retry_safety="idempotent",
            applied_when=[_value_outcome(TARGETS[1], password)],
            not_applied_when=[],
            recovery_policy=recovery_policy,
            timeout_ms=5000.0,
        ),
        WebActionPolicy(
            action=actions["click"],
            preconditions=[
                _url_precondition(login),
                _visible_precondition(SUBMIT_TARGET),
                _visible_precondition(TARGETS[0]),
                _visible_precondition(TARGETS[1]),
            ],
            retry_safety="non_idempotent",
            applied_when=[_url_outcome(dashboard), _visible_outcome(WELCOME_TARGET)],
            # The login page does not prove that the POST was not applied: an unknown execution
            # must stay unknown rather than being downgraded from a page the host happens to see.
            not_applied_when=[],
            recovery_policy=recovery_policy,
            timeout_ms=click_timeout_ms,
        ),
    ]


def _forecast(forecast_id: str, action: ActionSpec,
              outcomes: list[ExpectedOutcome]) -> Forecast:
    return Forecast(schema_version=SCHEMA, forecast_id=forecast_id, proposed_action=action,
                    expected_outcomes=outcomes, task_completion_hint=None)


def _script(origin: str, password: str):
    from forecontract_scripted import PlannerScript, PlannerStep

    actions = _actions(origin, password)
    initial = _blank_state()
    login = _login_state(origin, "", "")
    after_username = _login_state(origin, USERNAME, "")
    after_password = _login_state(origin, USERNAME, password)
    steps = [
        PlannerStep(
            schema_version=SCHEMA, step_id="web-plan-1",
            expected_environment_type="web", expected_state=initial,
            forecast=_forecast("web-forecast-1", actions["navigate"],
                               [_url_outcome(actions["login"])])),
        PlannerStep(
            schema_version=SCHEMA, step_id="web-plan-2",
            expected_environment_type="web", expected_state=login,
            forecast=_forecast("web-forecast-2", actions["fill_username"],
                               [_value_outcome(TARGETS[0], USERNAME)])),
        PlannerStep(
            schema_version=SCHEMA, step_id="web-plan-3",
            expected_environment_type="web", expected_state=after_username,
            forecast=_forecast("web-forecast-3", actions["fill_password"],
                               [_value_outcome(TARGETS[1], password)])),
        PlannerStep(
            schema_version=SCHEMA, step_id="web-plan-4",
            expected_environment_type="web", expected_state=after_password,
            forecast=_forecast("web-forecast-4", actions["click"],
                               [_url_outcome(actions["dashboard"]),
                                _visible_outcome(WELCOME_TARGET)])),
    ]
    return PlannerScript(schema_version=SCHEMA, script_id="web-runtime-script", steps=steps)


def _criteria(origin: str, goal_id: str) -> TaskCompletionCriteria:
    return TaskCompletionCriteria(
        schema_version=SCHEMA, goal_id=goal_id,
        conditions=[_url_outcome(f"{origin}/dashboard"),
                    _visible_outcome(WELCOME_TARGET)])


def _request(run_id: str, origin: str) -> RunRequest:
    goal_id = f"web-goal-{run_id}"
    return RunRequest(
        schema_version=SCHEMA, run_id=run_id, goal_id=goal_id,
        task_criteria=_criteria(origin, goal_id),
        max_steps=4, max_planning_calls=4, max_retries=0, max_replans=0)


# --------------------------------------------------------------------------- profile settings


class _Profile:
    """The fixed site mode and timing declaration of one profile."""

    __slots__ = ("adapter_operation_ms", "click_timeout_ms", "mode",
                 "observation_target_timeout_ms", "password", "request_ms", "run_ms",
                 "scenario", "startup_timeout_ms")

    def __init__(self, *, scenario: Scenario, mode: str, password: str, run_ms: int,
                 adapter_operation_ms: int, request_ms: int, click_timeout_ms: float,
                 startup_timeout_ms: int,
                 observation_target_timeout_ms: float | None = None) -> None:
        self.scenario = scenario
        self.mode = mode
        self.password = password
        self.run_ms = run_ms
        self.adapter_operation_ms = adapter_operation_ms
        self.request_ms = request_ms
        self.click_timeout_ms = click_timeout_ms
        #: The explicit startup allowance of this profile. Every profile declares the same 10000 ms;
        #: the value is a startup-only allowance and never changes the operation/target budgets.
        self.startup_timeout_ms = startup_timeout_ms
        #: The optional per-target read allowance. Only ``commit_unknown`` declares one: after the
        #: submit times out, the page may still be waiting on the stalled response, so a single
        #: target group must not be allowed to consume the entire observation allowance. Every
        #: other profile keeps the historical ``None`` behaviour.
        self.observation_target_timeout_ms = observation_target_timeout_ms


def _profile(name: str) -> _Profile:
    """The declared site mode and frozen timing of one profile.

    Every profile uses the frozen set: ordinary Run 30000 / operation 3000 / site 15000, and
    ``delayed_deadline`` uses 15000/25000/30000. The operation allowance is never widened to hide
    a first failure, and a real-time budget conflict is reported rather than tuned away.

    Every profile also declares the same explicit startup allowance (``10000`` ms). It bounds the
    one browser acquisition of ``start`` only; it does not change the operation, per-target, Run,
    request or cleanup budgets declared here, and it is not proof of the native startup duration.
    """
    if name == "normal":
        return _Profile(scenario=Scenario.NORMAL, mode="normal", password=PASSWORD,
                        run_ms=30000, adapter_operation_ms=3000, request_ms=15000,
                        click_timeout_ms=5000.0, startup_timeout_ms=ADAPTER_STARTUP_MS)
    if name == "wrong_credentials":
        return _Profile(scenario=Scenario.NORMAL, mode="normal", password=WRONG_PASSWORD,
                        run_ms=30000, adapter_operation_ms=3000, request_ms=15000,
                        click_timeout_ms=5000.0, startup_timeout_ms=ADAPTER_STARTUP_MS)
    if name == "delayed_release":
        return _Profile(scenario=Scenario.DELAYED_LOGIN, mode="release", password=PASSWORD,
                        run_ms=30000, adapter_operation_ms=3000, request_ms=15000,
                        click_timeout_ms=5000.0, startup_timeout_ms=ADAPTER_STARTUP_MS)
    if name == "delayed_deadline":
        return _Profile(scenario=Scenario.DELAYED_LOGIN, mode="deadline", password=PASSWORD,
                        run_ms=15000, adapter_operation_ms=25000, request_ms=30000,
                        click_timeout_ms=28000.0, startup_timeout_ms=ADAPTER_STARTUP_MS)
    if name == "overlay_blocked":
        return _Profile(scenario=Scenario.BLOCKED_SUBMIT, mode="normal", password=PASSWORD,
                        run_ms=30000, adapter_operation_ms=3000, request_ms=15000,
                        click_timeout_ms=5000.0, startup_timeout_ms=ADAPTER_STARTUP_MS)
    if name == "overlay_released":
        return _Profile(scenario=Scenario.BLOCKED_SUBMIT, mode="normal", password=PASSWORD,
                        run_ms=30000, adapter_operation_ms=3000, request_ms=15000,
                        click_timeout_ms=5000.0, startup_timeout_ms=ADAPTER_STARTUP_MS)
    if name == "commit_unknown":
        # The only profile that declares the optional per-target read allowance. It shortens one
        # target group's read opportunity; it does not extend the observation or the Run, and the
        # frozen operation allowance stays 3000 ms. The startup allowance stays the shared 10000 ms.
        return _Profile(scenario=Scenario.COMMIT_THEN_STALL, mode="normal", password=PASSWORD,
                        run_ms=30000, adapter_operation_ms=3000, request_ms=15000,
                        click_timeout_ms=5000.0, startup_timeout_ms=ADAPTER_STARTUP_MS,
                        observation_target_timeout_ms=500.0)
    if name == "cancel_after_received":
        return _Profile(scenario=Scenario.DELAYED_LOGIN, mode="cancel", password=PASSWORD,
                        run_ms=30000, adapter_operation_ms=3000, request_ms=15000,
                        click_timeout_ms=5000.0, startup_timeout_ms=ADAPTER_STARTUP_MS)
    raise failure("arguments")


# --------------------------------------------------------------------------- one run


class _RunEvidence:
    """The real records of one run, kept in process for the host's independent assertions."""

    __slots__ = ("cancelled", "evidence_root", "index_path", "last_finalization", "memory_path",
                 "phase_facts", "recorder", "result", "run_id", "site_stop", "trace_path",
                 "trace_records")

    def __init__(self) -> None:
        self.run_id = ""
        self.result = None
        self.cancelled = False
        self.last_finalization = None
        self.trace_path: Path | None = None
        self.memory_path: Path | None = None
        self.index_path: Path | None = None
        self.evidence_root: Path | None = None
        self.recorder: EvidenceRecorder | None = None
        self.trace_records: list = []
        self.phase_facts: dict = {}
        self.site_stop = None


def _browser_identity() -> dict:
    """Report the installed Playwright and Chromium identities from the installed metadata."""
    try:
        playwright_version = importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError:
        playwright_version = None
    chromium_version = None
    revision = None
    try:
        spec = importlib.util.find_spec("playwright")
        if spec is not None and spec.origin:
            definitions = Path(spec.origin).parent / "driver" / "package" / "browsers.json"
            payload = json.loads(definitions.read_text(encoding="utf-8"))
            for browser in payload.get("browsers", []):
                if browser.get("name") == "chromium":
                    revision = browser.get("revision")
                    chromium_version = browser.get("browserVersion")
                    break
    except (OSError, ValueError):
        pass
    return {"playwright": playwright_version, "chromium_revision": revision,
            "chromium_version": chromium_version}


def _adapter_config(origin: str, profile: _Profile) -> PlaywrightWebConfig:
    """Build the one explicit adapter configuration of a profile.

    Every budget is stated separately and none is derived from another: the operation allowance,
    the optional per-target read allowance, the cleanup allowance and the explicit startup
    allowance. Only ``start`` uses the startup value.
    """
    return PlaywrightWebConfig(
        allowed_origins=[origin],
        observation_targets=list(TARGETS),
        operation_timeout_ms=profile.adapter_operation_ms,
        cleanup_timeout_ms=ADAPTER_CLEANUP_MS,
        observation_target_timeout_ms=profile.observation_target_timeout_ms,
        startup_timeout_ms=profile.startup_timeout_ms,
    )


async def _run_once(*, site: LocalTestSite, output_dir: Path, profile_name: str,
                    profile: _Profile, run_id: str, selected_scenario: Scenario | None = None
                    ) -> _RunEvidence:
    """Run one real profile and write its complete evidence sidecar."""
    run = _RunEvidence()
    run.run_id = run_id
    run.evidence_root = output_dir
    run.trace_path = output_dir / "trace.jsonl"
    run.memory_path = output_dir / "working-memory.json"
    try:
        recorder = EvidenceRecorder(output_dir, run_id=run_id, profile=profile_name)
    except EvidenceError as exc:
        raise failure("evidence", exc) from None
    run.recorder = recorder
    origin = site.origin
    # The host-boundary diagnostics start from the real site state before the Run; the proxy
    # records every adapter boundary and its fixed code as the Run proceeds.
    recorder.record_host_started(_snapshot(site))
    # Every profile keeps the frozen bounded policy. The real Web verifier identity is declared to
    # the generic recovery engine, so the post-attempt step verdict is accepted natively; no label
    # is rewritten and no abort policy is used to bypass the real evidence.
    recovery_policy = STANDARD_RECOVERY
    policies = _policies(origin, profile.password,
                         click_timeout_ms=profile.click_timeout_ms,
                         recovery_policy=recovery_policy)
    script = _script(origin, profile.password)
    domain = WebRuntimeDomainSupport(
        adapter_name="playwright_web",
        required_targets=list(TARGETS),
        require_url=True,
        action_policies=policies,
    )
    adapter = PlaywrightWebAdapter(_adapter_config(origin, profile))
    recording_adapter = RecordingAdapter(
        adapter, recorder, operation_timeout_ms=profile.adapter_operation_ms,
        cleanup_timeout_ms=ADAPTER_CLEANUP_MS,
        startup_timeout_ms=profile.startup_timeout_ms)
    trace = JsonlTraceStore(run.trace_path)
    try:
        run.trace_path.touch(exist_ok=True)
    except OSError as exc:
        raise failure("evidence", exc) from None
    memory = WorkingMemory()
    engine = RuntimeEngine(
        adapter=recording_adapter,
        planner=RecordingPlanner(ScriptedPlanner(script), recorder),
        domain_support=domain,
        budget_source=MonotonicBudgetSource(total_budget_ms=30000.0),
        trace_writer=trace,
        working_memory=memory,
        builder=RecordingBuilder(ContractBuilder(), recorder),
        step_verifier=RecordingStepVerifier(WebStepVerifier(), recorder),
        task_oracle=RecordingTaskOracle(WebTaskCompletionOracle(), recorder),
        recovery_engine=RecordingRecoveryEngine(
            RecoveryEngine(step_verifier_name=WEB_STEP_VERIFIER_NAME), recorder),
        timing_policy=RuntimeTimingPolicy(
            schema_version=SCHEMA, run_timeout_ms=float(profile.run_ms),
            cleanup_timeout_ms=RUNTIME_CLEANUP_MS,
            finalize_timeout_ms=RUNTIME_FINALIZE_MS),
        contract_hooks_factory=WebContractHooksFactory(),
        task_criteria_validator=WebTaskCriteriaValidator(),
    )
    request = _request(run_id, origin)
    failure_exc: BaseException | None = None
    phase_facts: dict = {}

    async def wait_for(event: SiteEvent) -> bool:
        return await asyncio.to_thread(site.wait_for, event=event, timeout_ms=HOST_WAIT_MS)

    async def wait_for_login(phase: str) -> bool:
        """Record one bounded host LOGIN_RECEIVED wait and its real receipt."""
        started = time.monotonic()
        received = await wait_for(SiteEvent.LOGIN_RECEIVED)
        recorder.record_host_wait(
            phase, received=received,
            elapsed_ms=(time.monotonic() - started) * 1000.0)
        return received

    async def release_host() -> None:
        received = await wait_for_login("release")
        phase_facts["login_received"] = received
        if received:
            snapshot = _snapshot(site)
            phase_facts["before_release"] = snapshot
            recorder.record_host_counts("before_release", snapshot)
            try:
                await asyncio.to_thread(site.release)
            except Exception:  # noqa: BLE001 - a host release failure is reported by the run
                phase_facts["release_failed"] = True

    async def deadline_host() -> None:
        received = await wait_for_login("deadline")
        phase_facts["login_received"] = received
        if received:
            phase_facts["after_login_received"] = _snapshot(site)

    async def cancel_host(run_task: asyncio.Task) -> None:
        received = await wait_for_login("cancel")
        phase_facts["login_received"] = received
        if received:
            snapshot = _snapshot(site)
            phase_facts["before_cancel"] = snapshot
            recorder.record_host_counts("before_cancel", snapshot)
            run_task.cancel()

    try:
        if profile.mode == "cancel":
            run_task = asyncio.create_task(engine.run(request=request))
            host_task = asyncio.create_task(cancel_host(run_task))
            try:
                run.result = await run_task
            except asyncio.CancelledError:
                run.cancelled = True
                recorder.mark_interrupted("CancelledError")
            await host_task
        elif profile.mode == "release":
            host_task = asyncio.create_task(release_host())
            try:
                run.result = await engine.run(request=request)
            finally:
                await host_task
        elif profile.mode == "deadline":
            host_task = asyncio.create_task(deadline_host())
            try:
                run.result = await engine.run(request=request)
            finally:
                await host_task
        else:
            run.result = await engine.run(request=request)
    except asyncio.CancelledError:
        run.cancelled = True
        recorder.mark_interrupted("CancelledError")
    except DemoFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - finalisation must still happen, then it is reported
        failure_exc = exc
    finally:
        run.last_finalization = engine.last_finalization
        run.phase_facts = phase_facts
        published = None
        try:
            published = memory.get(run_id)
        except Exception:  # noqa: BLE001 - a Memory read failure is an explicit evidence failure
            published = None
        try:
            run.memory_path.write_text(
                json.dumps({"published_by": "runtime", "state": published},
                           ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        except OSError as exc:
            raise failure("evidence", exc) from None
        if adapter.cleanup_error is not None:
            phase_facts["adapter_cleanup_error"] = adapter.cleanup_error.code
        # The host records the real resource-cleanup facts and the post-Run counts in the same
        # evidence index. A start/observe/execute failure therefore still leaves its actual
        # boundary, fixed code, LOGIN_RECEIVED fact, counts and cleanup facts attributable.
        cleanup_status = None
        cleanup_code = None
        if run.last_finalization is not None:
            cleanup_status = run.last_finalization.cleanup_status
            if run.last_finalization.cleanup_error_code is not None:
                cleanup_code = run.last_finalization.cleanup_error_code.value
        try:
            recorder.record_host_counts("after_run", _snapshot(site))
            recorder.record_host_cleanup(
                adapter_error_code=(adapter.cleanup_error.code
                                    if adapter.cleanup_error is not None else None),
                runtime_cleanup_status=cleanup_status,
                runtime_cleanup_error_code=cleanup_code,
            )
        except EvidenceError as exc:
            raise failure("evidence", exc) from None
        try:
            terminal = _terminal(run)
            run.index_path = recorder.finalize(
                result=run.result,
                last_finalization=run.last_finalization,
                terminal=terminal,
                trace_path=run.trace_path,
                working_memory_path=run.memory_path,
                working_state=published,
                trace_records=_read_trace(trace, run_id),
                cleanup={"adapter_cleanup_error": (
                             adapter.cleanup_error.code
                             if adapter.cleanup_error is not None else None),
                         "browser": _browser_identity()},
            )
            verify_evidence_index(output_dir)
            run.trace_records = _read_trace(trace, run_id)
        except EvidenceError as exc:
            raise failure("evidence", exc) from None
        if failure_exc is not None:
            raise failure("runtime", failure_exc) from None
    return run


def _read_trace(trace: JsonlTraceStore, run_id: str) -> list:
    """Read one run's facts, treating a pre-trace refusal as an empty real history."""
    try:
        return list(trace.read_run(run_id))
    except Exception:  # noqa: BLE001 - a Run refused before any trace write has no facts
        return []


def _snapshot(site: LocalTestSite) -> dict:
    snapshot = site.snapshot_oracle()
    return {
        "generation": snapshot.generation,
        "scenario": snapshot.scenario.value,
        "running": snapshot.running,
        "login_requests": snapshot.login_requests,
        "auth_successes": snapshot.auth_successes,
        "auth_failures": snapshot.auth_failures,
        "commits": snapshot.commits,
        "active_requests": snapshot.active_requests,
        "response_blocked": snapshot.response_blocked,
        "authenticated_sessions": snapshot.authenticated_sessions,
    }


def _terminal(run: _RunEvidence) -> dict:
    """The terminal record, strictly sourced from a real result or the public finalization."""
    if run.result is not None:
        return {
            "terminal_status": run.result.terminal_status.value,
            "reason_code": run.result.reason_code.value,
            "source": "run_result",
        }
    finalization = run.last_finalization
    if finalization is None:
        return {"terminal_status": None, "reason_code": None, "source": "last_finalization"}
    return {
        "terminal_status": finalization.primary_terminal_status.value,
        "reason_code": finalization.primary_reason_code.value,
        "source": "last_finalization",
    }


# --------------------------------------------------------------------------- output directory


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_output_dir(raw: object) -> Path:
    """Resolve an external new-or-empty output directory, refusing the repository."""
    try:
        target = Path(str(raw)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        raise failure("arguments") from None
    repository = _repository_root()
    if target == repository or repository in target.parents:
        raise failure("arguments")
    try:
        if target.exists():
            if not target.is_dir() or any(target.iterdir()):
                raise failure("arguments")
        else:
            target.mkdir(parents=True)
    except DemoFailure:
        raise
    except OSError:
        raise failure("arguments") from None
    return target


# --------------------------------------------------------------------------- profiles


def run_profile(name: str, output_dir: object) -> dict:
    """Run one named profile and return its summary plus host-only oracle records."""
    if name not in PROFILES:
        raise failure("arguments")
    root = resolve_output_dir(output_dir)
    profile = _profile(name)
    site = LocalTestSite(SiteConfig(
        scenario=profile.scenario,
        request_timeout_ms=profile.request_ms,
        cleanup_timeout_ms=SITE_CLEANUP_MS,
    ))
    try:
        try:
            site.start()
        except Exception as exc:  # noqa: BLE001 - one bounded site-start failure
            raise failure("site_start", exc) from None
        started = _snapshot(site)
        previous = None
        try:
            if name == "overlay_released":
                # The blocking Run is a complete, explicit Run of its own: it finishes its
                # cleanup before the host resets the idle site and starts a genuinely new Run.
                previous_dir = root / "previous-run"
                previous_dir.mkdir()
                previous_run = asyncio.run(_run_once(
                    site=site, output_dir=previous_dir, profile_name="overlay_blocked",
                    profile=_profile("overlay_blocked"),
                    run_id=f"web-run-{uuid.uuid4().hex}"))
                previous = _run_summary(previous_run)
                previous["oracle"] = {"before_reset": _snapshot(site),
                                      "previous_run_completed": True}
                try:
                    generation = site.reset(scenario=Scenario.BLOCKED_SUBMIT)
                    site.release()
                except Exception as exc:  # noqa: BLE001
                    raise failure("cleanup", exc) from None
                phase = {"reset_generation": generation}
            else:
                phase = {}
            run = asyncio.run(_run_once(
                site=site, output_dir=root, profile_name=name, profile=profile,
                run_id=f"web-run-{uuid.uuid4().hex}"))
        finally:
            before_stop = _snapshot(site)
            stop = site.stop()
            after_stop = _snapshot(site)
        if not stop.complete:
            raise failure("cleanup")
        summary = _run_summary(run)
        if run.phase_facts.get("adapter_cleanup_error"):
            raise failure("cleanup")
        summary.update({
            "profile": name,
            "oracle": {
                "started": started,
                "before_stop": before_stop,
                "after_stop": after_stop,
                "site_stop_complete": stop.complete,
                "site_stop": {"active_requests": stop.active_requests,
                              "open_connections": stop.open_connections,
                              "live_threads": stop.live_threads},
                "phase_facts": dict(run.phase_facts, **phase) if name == "overlay_released"
                else run.phase_facts,
                "previous_run": previous,
            },
            "browser": _browser_identity(),
            "evidence_verified": True,
        })
        return summary
    finally:
        # The site is owned by the host and is always stopped with one bounded attempt, including
        # on cancellation or an earlier failure.
        try:
            site.stop()
        except Exception:  # noqa: BLE001, S110 - the primary failure is already reported
            pass


def _run_summary(run: _RunEvidence) -> dict:
    terminal = _terminal(run)
    return {
        "run_id": run.run_id,
        "result": run.result.model_dump(mode="json") if run.result is not None else None,
        "terminal": terminal,
        "cancelled": run.cancelled,
        "trace_path": str(run.trace_path),
        "evidence_index_path": str(run.index_path),
        "working_memory_path": str(run.memory_path),
        "last_finalization": (
            run.last_finalization.model_dump(mode="json")
            if run.last_finalization is not None else None),
        "cleanup": (run.index_path and json.loads(run.index_path.read_text(encoding="utf-8"))
                    .get("cleanup")),
        "trace_records": [
            {"kind": getattr(record, "kind", None),
             "sequence": getattr(record, "sequence", None),
             "scope": getattr(record, "scope", None),
             "verification_status": getattr(
                 getattr(record, "verification_status", None), "value", None),
             "execution_status": getattr(
                 getattr(record, "execution_status", None), "value", None),
             "error_code": getattr(record, "error_code", None)}
            for record in run.trace_records
        ],
    }


def _cli_record(summary: dict) -> dict:
    """The exact five-field object the CLI prints."""
    return {
        "result": summary["result"],
        "terminal": summary["terminal"],
        "trace_path": summary["trace_path"],
        "evidence_index_path": summary["evidence_index_path"],
        "working_memory_path": summary["working_memory_path"],
    }


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one of the eight real Web Runtime profiles against the local test site.")
    parser.add_argument("--profile", choices=PROFILES, default="normal",
                        help="the profile to run (default: normal)")
    parser.add_argument("--output-dir", required=True,
                        help="an external new-or-empty directory for trace, evidence and memory")
    args = parser.parse_args(argv)
    try:
        summary = run_profile(args.profile, args.output_dir)
        record = _cli_record(summary)
        if set(record) != {"result", "terminal", "trace_path", "evidence_index_path",
                           "working_memory_path"}:
            raise failure("summary")
    except DemoFailure as exc:
        print(f"web runtime demo failed: {exc}", file=sys.stderr)
        return 1
    except EvidenceError:
        print(f"web runtime demo failed: {_FIXED_FAILURES['evidence']}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 - a fixed sentence, never the exception
        print(f"web runtime demo failed: {_FIXED_FAILURES['runtime']}", file=sys.stderr)
        return 1
    print(json.dumps(record, ensure_ascii=False))
    if summary["cancelled"]:
        return 130
    if summary["result"] is None:
        return 1
    if summary["terminal"]["terminal_status"] == "completed":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
