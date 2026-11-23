"""The minimal sequential single-Run Runtime.

This engine *composes* already-verified components. It does not re-implement the Contract Builder,
the JSON-rule verifier, the task oracle or the recovery algorithm: it calls them, and it owns only
what none of them can own - the order of operations, the provenance of every observation, the
host's limits, the trace and the terminal state of one Run.

Four separations are load-bearing here and are worth stating plainly, because collapsing any one of
them would silently turn a fact into a claim:

* **A step result is not task success.** A passing step only says the contracted outcomes matched.
  Only the task oracle, evaluated on a real observation, decides that the goal is reached.
* **An execution status is not an effect state.** Whether an action changed the world is a domain
  judgement; the engine never derives it from ``succeeded``/``failed``/``unknown``.
* **A planner hint is not evidence.** ``task_completion_hint`` is advisory: it can neither make a
  Run succeed nor let a proposed action skip the Builder.
* **A declaration is not a measurement.** The host budget snapshot is read, checked and never
  increased or reset, but nothing here proves that wall-clock time was measured. Real deadlines and
  cancellation belong to a later task.

An observation is a *call*, not an identifier. Every ``observe`` gets a Runtime-local call ordinal
and the phase it happened in, so an execution-time and a recovery-time observation can never be
confused even when the adapter returns the same ``observation_id`` twice. That provenance is kept
in this Run's Working Memory and in the local call log, and is deliberately *not* written into the
trace records, whose strict schemas have no field for it.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json as json_module
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

from pydantic import BaseModel

from forecontract.contracts import ContractBuilder, ContractBuildError
from forecontract.contracts.builder import ContractBuilder as _ContractBuilder
from forecontract.memory.working import WorkingMemory
from forecontract.models.actions import Forecast
from forecontract.models.base import SCHEMA_VERSION
from forecontract.models.observations import Observation
from forecontract.models.outcomes import (
    ActualOutcome,
    VerificationResult,
    VerificationStatus,
)
from forecontract.protocols.adapter import ActionValidationError
from forecontract.recovery import (
    ActionSafetyDeclaration,
    RecoveryAssessment,
    RecoveryContext,
    RecoveryEngine,
    RecoveryError,
    RecoveryEvidence,
)
from forecontract.runtime.errors import (
    RUNTIME_DIAGNOSTICS,
    RUNTIME_STAGES,
    RuntimeError,
    sanitized,
)
from forecontract.runtime.finalization import RuntimeFinalization, build_finalization
from forecontract.runtime.models import (
    ContractPreparation,
    RunRequest,
    RunResult,
    RuntimeBudgetSnapshot,
    RuntimePhase,
    RuntimeReason,
    RuntimeTerminalStatus,
    RuntimeTransition,
    detached_state,
    isolated_record,
    record_payload,
    require_acyclic,
    stored_payload,
    validated_budget_value,
)
from forecontract.runtime.protocols import action_is_declared
from forecontract.runtime.state import RuntimeStateMachine
from forecontract.runtime.timing import (
    MonotonicReading,
    RuntimeTimingPolicy,
    SystemMonotonicClock,
    milliseconds_to_nanoseconds,
    timing_policy_from,
)
from forecontract.tracing import (
    AttemptFinished,
    AttemptStarted,
    ContractCommitted,
    RunFinished,
    RunStarted,
    StepStarted,
    TraceError,
    VerificationRecorded,
)
from forecontract.verification import (
    RuleValidationHooks,
    StepVerifier,
    TaskCompletionOracle,
)
from forecontract.verification.errors import VerificationError
from forecontract.verification.models import (
    TaskCompletionCriteria,
    VerificationEvidence,
)
from forecontract.verification.oracle import prevalidate_criteria
from forecontract.verification.step import required_all_aggregate

__all__ = ["RuntimeEngine"]

#: The existing trace vocabulary the Runtime maps its own terminal diagnosis onto. A custom Runtime
#: reason is never forced into the older closed schema; a facility failure becomes ``internal_error``
#: and a policy or limit stop becomes ``policy_abort``.
#: The control-flow exceptions this boundary never converts into a facility verdict: a caller
#: cancellation, an interrupt and a process exit are not diagnoses about a Run's terminal trace.
_CONTROL_FLOW: tuple[type[BaseException], ...] = (asyncio.CancelledError, KeyboardInterrupt,
                                                  SystemExit)

#: The codes a refused terminal preparation may carry to the caller. A refusal carrying one of these is
#: already a diagnosis about this Run's facility boundary: a dependency that failed, the Runtime's own input
#: contract, or a trace that could not be persisted. Every other refusal - a plain exception, a custom
#: subclass, a builtin one, an unpublished proposal code, or a spoofed attribute - is normalized into this
#: segment's bounded failure. The code is only ever accepted together with a declared stage, so an arbitrary
#: ``code`` attribute alone can never make a foreign exception trusted.
_PREPARATION_FAILURE_CODES = frozenset((
    "DEPENDENCY_FAILURE", "TRACE_FAILURE", "INPUT_INVALID",
))

#: The stage names the Runtime declares, used to validate a refusal's own stage before any of its text or
#: identity is reused.
_DECLARED_STAGES = frozenset(RUNTIME_STAGES)

#: The longest structural location this boundary will reuse from a refusal. A location is a diagnostic, not
#: a data channel, so a refusal that carries an enormous one cannot make the published surface unbounded.
_MAX_REFUSAL_LOCATION = 8

_TRACE_CODE: dict[str, str] = {
    RuntimeReason.DEPENDENCY_FAILURE.value: "internal_error",
    RuntimeReason.EVIDENCE_INVALID.value: "internal_error",
    RuntimeReason.IDENTITY_CONFLICT.value: "internal_error",
    RuntimeReason.INVALID_TRANSITION.value: "internal_error",
    RuntimeReason.TRACE_FAILURE.value: "internal_error",
    RuntimeReason.CLEANUP_FAILURE.value: "internal_error",
    RuntimeReason.BUDGET_EXHAUSTED.value: "policy_abort",
    RuntimeReason.POLICY_ABORT.value: "policy_abort",
    RuntimeReason.MAX_STEPS_EXHAUSTED.value: "policy_abort",
    RuntimeReason.MAX_PLANNING_EXHAUSTED.value: "policy_abort",
    RuntimeReason.MAX_REPLANS_EXHAUSTED.value: "policy_abort",
    # A real deadline expiry and a user cancellation are *not* facility failures and must not be
    # reported as one. The trace record carries the existing closed ``reason_code`` vocabulary - the
    # terminal *status* is the separate ``timeout`` or ``cancelled`` label - so neither is spelled
    # ``abort`` and no new code is invented.
    RuntimeReason.DEADLINE_EXCEEDED.value: "deadline_exceeded",
    RuntimeReason.OPERATION_TIMEOUT.value: "deadline_exceeded",
    RuntimeReason.USER_CANCELLED.value: "user_cancelled",
}


#: The literal event kind of each trace variant. The envelope's ``kind`` field has no usable
#: subclass default for an explicit construction, so the declared kind is passed by name.
_EVENT_KIND: dict[type, str] = {
    RunStarted: "run_started",
    StepStarted: "step_started",
    ContractCommitted: "contract_committed",
    AttemptStarted: "attempt_started",
    AttemptFinished: "attempt_finished",
    VerificationRecorded: "verification_recorded",
    RunFinished: "run_finished",
}


#: The process-global opaque identifier sequence. It is monotonic and never reused, so records,
#: Steps, Contracts and Attempts generated by *different* Engines cannot collide when their Runs
#: share one real trace store. It carries no caller content and is deliberately not persisted: this
#: round guarantees uniqueness within the process, not across a restart.
_OPAQUE_SEQUENCE = itertools.count(1)
_OPAQUE_LOCK = threading.Lock()


def _next_opaque() -> int:
    """The next process-global opaque identifier number."""
    with _OPAQUE_LOCK:
        return next(_OPAQUE_SEQUENCE)


def _deep_criteria(criteria: TaskCompletionCriteria) -> TaskCompletionCriteria:
    """Rebuild a task criteria as a completely fresh object graph.

    A JSON round trip is used deliberately: it is the only way to guarantee that no nested
    condition, payload or mapping is still shared with the caller's object, so a later mutation of
    the host's criteria cannot reach into the snapshot this Run judges against.
    """
    try:
        payload = json_module.loads(criteria.model_dump_json())
        return TaskCompletionCriteria.model_validate(payload)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("INPUT_INVALID", "validate", ("task_criteria",),
                           "the task criteria could not be detached", cause=exc) from None


def _deep_observation(observation: Observation) -> Observation:
    """Rebuild an observation as a completely fresh object graph.

    The same round trip as the criteria snapshot is used, so the adapter's returned object and the
    copy handed to the domain support share no nested container at all.
    """
    try:
        payload = json_module.loads(observation.model_dump_json())
        return Observation.model_validate(payload)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("EVIDENCE_INVALID", "observe", ("observation",),
                           "the adapter returned an observation that cannot be detached",
                           cause=exc) from None


def _is_plan_unavailable(exc: BaseException) -> bool:
    """Whether a planner refusal means no plan exists for the observation it was given.

    Answered by the refusal's own declared code rather than by any supplied value: a mismatch against
    the scripted step or an exhausted script both mean the declared plan does not cover this
    observation, which is a bounded outcome of the Run. Every other planner error is a facility failure.
    """
    return getattr(exc, "code", None) in ("planner_observation_mismatch", "planner_script_exhausted")


class _BudgetExhausted(Exception):
    """Internal signal: the host declared no budget, so a new business operation is forbidden.

    It is raised only where the Runtime is about to start a *new* business operation, and the caller
    turns it into a recorded budget abort rather than a facility failure. It never escapes the
    engine, and it is not part of the public error surface.
    """


class _TerminalReached(Exception):
    """Internal signal: the Run reached a terminal phase, so the phase loop stops.

    It carries the phase move the caller must apply once the teardown conclusion is known. A terminal
    phase is a *conclusion*, so it is deliberately not entered in the middle of the run: a candidate
    success in particular must not be recorded as permanent before the cleanup that the contract makes
    part of the conclusion.
    """

    def __init__(self, move: object) -> None:
        super().__init__("terminal")
        self.move = move


@dataclass
class _PhaseState:
    """The working state one Run's phase loop carries between phases."""

    evidence: VerificationEvidence | None = None
    observation: Observation | None = None
    contract: object | None = None
    actual_outcome: ActualOutcome | None = None
    step_result: VerificationResult | None = None
    pending_forecast: Forecast | None = None
    pending_replaces: str | None = None
    done: bool = False


@dataclass
class _RunOutcome:
    """The terminal facts one Run accumulates, filled in as the phases decide them."""

    terminal_status: RuntimeTerminalStatus = RuntimeTerminalStatus.COMPLETED
    terminal_reason: RuntimeReason = RuntimeReason.TASK_ACHIEVED
    verification_record_id: str | None = None
    run_pass_record: str | None = None
    primary_failure: BaseException | None = None
    #: The unique terminal trace record this Run wrote, once it has been written.
    terminal_record_id: str | None = None
    #: The terminal phase move that is still to be applied. A terminal phase is a conclusion, so it is
    #: entered after the teardown rather than beside the candidate outcome that produced it.
    deferred_terminal: object | None = None
    #: Whether the primary failure must propagate as a bounded error rather than being published in a
    #: RunResult. It is set for a facility failure the Runtime translated from a raw dependency
    #: exception: the contract keeps those propagating, while a bounded business refusal is reported in
    #: the result. A cleanup that leads for its own reason also propagates, which the caller decides.
    propagate_failure: bool = False


@dataclass
class _LocalCall:
    """One Runtime-local dependency call: what it was, and where it happened."""

    ordinal: int
    operation: str
    phase: str
    step_id: str | None
    contract_id: str | None
    attempt_id: str | None
    observation_id: str | None


@dataclass
class _Progress:
    """The counters this Run owns. Every counter is global to the Run except the contract-local
    retry count, which is reset only when a genuinely new Contract is committed."""

    steps_started: int = 0
    planning_calls: int = 0
    attempts_started: int = 0
    retries_used: int = 0
    replans_used: int = 0
    contract_retry_count: int = 0
    remaining_retries: int = 0
    remaining_replans: int = 0
    remaining_planning: int = 0
    remaining_steps: int = 0
    budget_ms: float = 0.0
    next_ordinal: int = 0
    next_sequence: int = 0
    calls: list[_LocalCall] = field(default_factory=list)
    transitions: list[RuntimeTransition] = field(default_factory=list)
    current_step: str | None = None
    current_contract: str | None = None
    current_attempt: str | None = None
    previous_contract: str | None = None
    execution_anomaly: bool = False
    #: The Run's business start, taken on entry, and the deadline derived from the first legal host
    #: budget. Both are monotonic nanosecond readings; the deadline is fixed for the whole Run.
    business_start_ns: int = 0
    deadline_ns: int | None = None
    effective_remaining_ns: int = 0
    #: The monotonic span of the operation currently being charged. A synchronous dependency cannot
    #: be preempted in this event loop, so its real elapsed time is charged when it returns.
    operation_started_ns: int | None = None
    sync_elapsed_ms: int = 0
    sync_spans: list[int] = field(default_factory=list)
    #: An asynchronous operation that did not finish inside the allowance. It is *pending*, not
    #: stopped, and a strong reference is kept so it can never be collected mid-flight.
    pending_task: Any | None = None
    pending_operation_at_return: bool = False
    #: Whether the cancellation currently in flight came from the host that owns this Run, or from the
    #: dependency itself. Only a host cancellation is a user cancellation; a dependency that cancels
    #: its own task is a facility failure, and the two must never be reported as each other.
    cancelled_by_caller: bool = False
    dependency_self_cancel: bool = False
    #: Whether the single lifecycle stop was really confirmed, which is what allows the resource
    #: occupancy to be released. Anything else keeps it occupied.
    stop_confirmed: bool = False
    #: Which bound the last effective deadline actually came from, so the diagnosis can distinguish a
    #: committed contract's own local allowance from the Run's remaining business budget.
    local_bound_first: bool = False
    local_bound_elapsed: bool = False
    #: Whether the failure currently leading came from a real facility failure rather than a bounded
    #: business refusal. A facility failure is what the contract keeps propagating to the caller once
    #: the finalisation is complete; a business refusal is published in the RunResult instead.
    facility_failure: bool = False
    #: Whether the leading failure is one the Runtime had to *translate* from a raw dependency
    #: exception. That is the facility failure the contract keeps propagating; a refusal the
    #: dependency raised as its own bounded error is published in the result instead.
    translated_failure: bool = False
    #: Whether the Run's own allowance was declared as zero. That is an immediate
    #: deadline rather than an absent one, so it is reported as a timeout.
    run_limit_zero: bool = False
    #: The event timestamp the terminal record will carry, and the complete record built before the
    #: pre-append check so that check tests the record the Runtime would really write.
    terminal_occurred_at: datetime | None = None
    terminal_record: dict[str, Any] | None = None
    #: The bounded diagnosis of the refusal the finalising segment met while preparing the terminal record.
    #: The snapshot and the resource verdict are written either way; the caller receives this diagnosis
    #: instead of the segment's own persistence failure when it is set.
    finalization_refusal: BaseException | None = None
    #: The safe state the preparation published, so the confirmed terminal identifier can be added to it
    #: inside the same allowance once the append has really returned.
    memory_state: dict[str, Any] | None = None
    #: Whether the safe Memory diagnosis refused during finalising. The Memory record and the terminal
    #: trace fact are different facts at different boundaries, so a Memory refusal does not by itself
    #: mean the terminal write was never attempted: the remaining allowance still belongs to it.
    memory_failed: bool = False



class RuntimeEngine:
    """A sequential, dependency-injected Runtime for one Run at a time.

    The engine is reusable across Runs but refuses to run one concurrently with itself or to reuse
    a ``run_id`` it has already started, because either would make two histories share one identity.
    It also refuses a resource another Run left unsettled. That occupancy is keyed by the real adapter
    object rather than by this Engine, because swapping the Engine does not make a resource whose
    operation never confirmed it stopped safe to reuse.
    """

    #: The process-wide occupancy registry. An entry is keyed by the adapter object's own identity, so
    #: the same real resource is recognised across Engines; an entry is removed only when a cleanup
    #: really succeeded, and every other outcome quarantines it. The stored object keeps the address
    #: valid, so a recycled ``id()`` cannot be mistaken for the resource it replaced.
    _OCCUPANCY: ClassVar[dict[int, tuple[object, str]]] = {}
    _OCCUPANCY_LOCK: ClassVar[threading.Lock] = threading.Lock()
    #: Strong owners for operations that never confirmed they stopped. A quarantined resource may not
    #: have its attempt collected mid-flight, so the Engine and this registry hold it until an explicit
    #: release proves the operation ended.
    _PENDING_OWNERS: ClassVar[dict[object, bool]] = {}

    def __init__(self, *, adapter: object, planner: object, domain_support: object,
                 budget_source: object, trace_writer: object,
                 working_memory: object | None = None,
                 builder: object | None = None,
                 step_verifier: object | None = None,
                 task_oracle: object | None = None,
                 recovery_engine: object | None = None,
                 clock: object | None = None,
                 timing_policy: object | None = None,
                 monotonic_clock: object | None = None,
                 contract_hooks_factory: object | None = None,
                 task_criteria_validator: object | None = None) -> None:
        if adapter is None or planner is None or domain_support is None:
            raise RuntimeError("INPUT_INVALID", "validate", ("dependencies",),
                               "the adapter, planner and domain support are required")
        if budget_source is None or trace_writer is None:
            raise RuntimeError("INPUT_INVALID", "validate", ("dependencies",),
                               "the budget source and trace writer are required")
        self._adapter = adapter
        self._planner = planner
        self._domain = domain_support
        self._budget_source = budget_source
        self._trace = trace_writer
        self._memory = working_memory if working_memory is not None else WorkingMemory()
        # The four already-verified components. They are injectable so a host can supply a
        # differently configured instance, and every default is the real implementation.
        self._builder = builder if builder is not None else ContractBuilder()
        self._verifier = step_verifier if step_verifier is not None else StepVerifier()
        self._oracle = task_oracle if task_oracle is not None else TaskCompletionOracle()
        self._recovery = recovery_engine if recovery_engine is not None else RecoveryEngine()
        # The existing ``clock`` keeps its meaning: it is the UTC event-time source for trace
        # timestamps. The monotonic clock is a separate dependency, so a duration is never measured
        # with an event-time clock and the original parameter is never reinterpreted.
        self._clock = clock
        self._timing = (RuntimeTimingPolicy(schema_version=SCHEMA_VERSION)
                        if timing_policy is None else timing_policy_from(timing_policy))
        # One reader enforces the declared non-decreasing rule on every business reading, and one real
        # clock - the production monotonic clock, never the injected business clock - measures the two
        # teardown segments. Keeping them apart is what lets a failed business clock still finalise.
        self._business_clock = MonotonicReading(SystemMonotonicClock() if monotonic_clock is None
                                                else monotonic_clock, stage="facility")
        self._real_clock = SystemMonotonicClock()
        #: The optional domain hooks factory. ``None`` keeps the original JSON-rule path exactly as
        #: it was; an object is only ever used through its ``create`` method, at the two declared
        #: boundaries, and never cached across phases.
        self._contract_hooks_factory = contract_hooks_factory
        #: The optional, independent task-criteria prevalidator. ``None`` keeps the original JSON
        #: ``prevalidate_criteria`` path exactly as it was. Supplying one never disables the
        #: prevalidation and never depends on whether a contract-hooks factory was also supplied.
        self._task_criteria_validator = task_criteria_validator
        self._active_run: str | None = None
        self._started: set[str] = set()
        #: Whether the Run currently in flight really entered the adapter's ``start``. A Run that never
        #: did acquired no resource, so its occupancy is released; every other ending is judged by the
        #: teardown's verdict instead.
        self._start_attempted_in_flight = False
        # The bounded finalization snapshot of the most recent Run, and whether this engine is
        # currently holding a quarantined resource. Both are diagnostics: the quarantine is never
        # lifted by reading it or by mutating the returned snapshot.
        self._last_finalization: RuntimeFinalization | None = None
        self._quarantined = False
        #: Set when a business-clock reading was refused, which marks the leading
        #: failure as a facility failure rather than a bounded business refusal.
        self._facility_clock_failed = False

    @property
    def last_finalization(self) -> RuntimeFinalization | None:
        """The bounded finalization snapshot of the most recent Run, as an isolated copy.

        The copy is deliberate: a host may inspect and even mutate what it receives without being
        able to change the diagnosis the engine holds, and the quarantine state cannot be cleared
        through this property either.
        """
        if self._last_finalization is None:
            return None
        return self._last_finalization.isolation_copy()

    @property
    def is_quarantined(self) -> bool:
        """Whether this engine is holding a resource whose operations did not confirm they stopped."""
        return self._quarantined

    @property
    def timing_policy(self) -> RuntimeTimingPolicy:
        """The validated timing allowance this engine was constructed with, as an isolated copy."""
        return self._timing.isolation_copy() if hasattr(self._timing, "isolation_copy") \
            else self._timing.model_copy(deep=True)

    # -- public entry point ---------------------------------------------------------

    async def run(self, *, request: RunRequest) -> RunResult:
        """Execute one sequential Run and return its safe terminal summary.

        Every failure that damages the facility is raised as a bounded
        :class:`~forecontract.runtime.errors.RuntimeError`; a business failure is reported inside
        the returned :class:`RunResult` instead. A caller cancellation is never swallowed: it is
        propagated after the bounded teardown has finished.
        """
        if self._active_run is not None:
            raise RuntimeError("IDENTITY_CONFLICT", "validate", ("run_id",),
                               "this engine is already running")
        # The business start is taken before validation, so the prevalidation work is genuinely
        # charged to the deadline rather than starting the clock once the Run is already running.
        business_start_ns = self._now_ns()
        validated = self._validate_request(request)
        if validated.run_id in self._started:
            raise RuntimeError("IDENTITY_CONFLICT", "validate", ("run_id",),
                               "this run identifier was already started by this engine")
        if self._quarantined:
            # A resource whose earlier cleanup did not confirm it stopped is not reused, and there is
            # deliberately no public way to clear that decision.
            raise RuntimeError("DEPENDENCY_FAILURE", "validate", ("adapter",),
                               "this engine is holding a quarantined resource")
        # The resource identity is checked too, not only this Engine's own flag: another Engine may
        # have left the very same adapter object occupied, and swapping the Engine must not make that
        # resource safe to reuse. The refusal happens before ``start``, so nothing is touched.
        holder = self._occupancy_holder(self._adapter)
        if holder is not None:
            self._quarantined = True
            raise RuntimeError("DEPENDENCY_FAILURE", "validate", ("adapter",),
                               "the adapter resource is occupied by an unsettled Run")
        self._active_run = validated.run_id
        self._started.add(validated.run_id)
        # A genuinely accepted new Run clears the previous Run's diagnostic. A refused Run never
        # reaches this point, so a rejection cannot erase an existing diagnosis.
        self._reset_for_new_run()
        self._start_attempted_in_flight = False
        # The resource is marked occupied for as long as this Run holds it. It is released only by a
        # cleanup that really succeeded, so a crash or an unsettled ending cannot silently free it.
        self._occupy(self._adapter, "running")
        try:
            return await self._drive(validated, business_start_ns)
        finally:
            self._active_run = None
            # A Run that never entered ``start`` acquired no resource, so occupying it would wrongly
            # block a later Run; every other ending leaves the occupancy to the teardown's verdict.
            if not self._start_attempted_in_flight:
                self._release(self._adapter)

    # -- timing, bounded waiting and the finalization snapshot ----------------------

    def _now_ns(self) -> int:
        """Read the *business* clock, refusing an unusable or non-monotonic reading.

        The reader enforces the declared non-decreasing rule, so a business clock that goes backwards
        is one bounded refusal rather than a deadline that moves outwards and a budget that grows. The
        refusal is marked as a facility failure because it is one: the host's clock is damaged, not
        the business input, so it propagates to the caller rather than being published as a business
        outcome.
        """
        try:
            return self._business_clock.read()
        except RuntimeError:
            self._facility_clock_failed = True
            raise

    def _real_now_ns(self) -> int:
        """Read the *real* clock that the bounded teardown is measured against.

        The cleanup and finalization allowances are deliberately measured against this clock and never
        against the business clock: a business clock that has failed must not be able to prevent the
        Run from stopping its resource and recording its own end. This is the single place the two
        timelines are kept apart.
        """
        return self._real_clock.now_ns()

    def _progress_trace(self, progress: _Progress) -> None:
        """Record one operation's actual monotonic span, without measuring anything it cannot."""
        now = self._now_ns()
        started = progress.operation_started_ns
        if started is None or now < started:
            return
        elapsed = (now - started) // 1_000_000
        if elapsed > 0:
            progress.sync_elapsed_ms += elapsed
        del progress.sync_spans[-1:]

    def _check_deadline(self, progress: _Progress, stage: str) -> None:
        """Refuse to start or accept further business work once the effective budget is spent.

        The effective remaining budget is the smaller of what the *declared* host budget allows and
        what the monotonic deadline allows. The host's own snapshot is still validated separately by
        :meth:`_budget_refresh`, so taking the smaller value can never hide a host that increased its
        declaration. Nothing here measures wall-clock time or claims a hard real-time guarantee.
        """
        self._budget_refresh(progress, stage)
        host_ns = milliseconds_to_nanoseconds(progress.budget_ms, (stage, "remaining_budget_ms"))
        if progress.deadline_ns is None:
            remaining = host_ns
        else:
            remaining = min(progress.deadline_ns - self._now_ns(), host_ns)
        if remaining <= 0:
            # A host that declares zero *budget* keeps its existing abort semantics: that is a host
            # declaration, not a measured expiry. A Run allowance that declares zero is different - it
            # is an immediate deadline - so it is reported as the timeout it really is, and the Run
            # performs no lifecycle call at all.
            if progress.run_limit_zero:
                raise RuntimeError("EVIDENCE_INVALID", stage, ("deadline",),
                                   "the business deadline has elapsed")
            if host_ns <= 0:
                raise _BudgetExhausted
            raise RuntimeError("EVIDENCE_INVALID", stage, ("deadline",),
                               "the business deadline has elapsed")
        progress.effective_remaining_ns = remaining

    def _establish_deadline(self, progress: _Progress, host_ms: float) -> None:
        """Derive the business deadline once, from the first legal host budget.

        The Run's start is taken on entry, so validation, isolation, rule prevalidation and the first
        host budget reading are all charged to the deadline. An explicit policy timeout may only make
        the deadline *earlier*: it is combined with the host budget by taking the smaller value, so a
        policy cannot extend what the host declared. The deadline is then fixed for the Run; a retry,
        a replan, a new Step or a new Contract never moves it.
        """
        host_ns = milliseconds_to_nanoseconds(host_ms, ("validate", "remaining_budget_ms"))
        policy_ms = self._timing.run_timeout_ms
        candidates = [host_ns]
        if policy_ms is not None:
            candidates.append(milliseconds_to_nanoseconds(policy_ms, ("run_timeout_ms",)))
        # ``<= 0`` rather than ``< 0``: a zero declaration is an *immediate* deadline, so the Run
        # performs no lifecycle call at all. Treating it as positive would let one operation start
        # before the expiry is noticed, which is the unrequested adapter I/O this forbids.
        progress.run_limit_zero = min(candidates) <= 0
        progress.deadline_ns = progress.business_start_ns + min(candidates)
        progress.effective_remaining_ns = min(progress.deadline_ns - self._now_ns(), host_ns)

    async def _await_bounded(self, progress: _Progress, operation: object, stage: str,
                             phase: RuntimePhase, *, operation_ns: int | None = None,
                             on_accepted: object | None = None,
                             marks_start: bool = False) -> object:
        """Await one asynchronous business operation within the effective remaining budget.

        ``operation`` is a zero-argument callable that produces the awaitable. It is called only
        after the bounds are known, so a refused operation never even creates a coroutine and nothing
        is left un-awaited.

        ``on_accepted`` is an optional zero-argument callable run the instant a *synchronous* result is
        taken, before any later business step. It exists so a mutation check on a controlled argument
        is performed against the value the dependency actually returned, rather than after further
        work that legitimately rewrites the argument.

        The operation runs in its own task so the Runtime can stop *waiting* for it at the bound
        without pretending it stopped. ``asyncio.wait`` is used rather than a bare ``wait_for`` for
        exactly that reason: waiting for the cancellation to complete is what makes ``wait_for``
        unbounded, so the Runtime asks for the cancellation and returns, keeping a strong reference to
        the unfinished task so it is never collected mid-flight and consuming its late result or
        exception silently. A late result can therefore never write a trace record, move a counter,
        commit a contract or start another Attempt.
        """
        self._check_deadline(progress, stage)
        remaining = progress.effective_remaining_ns
        if operation_ns is not None:
            local_ns = operation_ns - self._now_ns()
            # The *earliest fixed deadline* decides the diagnosis, and a tie belongs to the Run's own
            # deadline: the committed contract's allowance is only a local bound, so it is attributed
            # to the operation solely when it is strictly earlier than the Run's remaining budget.
            # Treating a tie, or a later local bound, as local would misstate which limit the host
            # actually declared.
            progress.local_bound_first = local_ns < remaining
            remaining = min(remaining, local_ns)
        else:
            progress.local_bound_first = False
        if remaining <= 0:
            raise RuntimeError("EVIDENCE_INVALID", stage, ("deadline",),
                               "the business deadline has elapsed")
        if marks_start:
            # This is the boundary where the lifecycle start is really invoked: every check above - the
            # Run's own, this wait's, and any check a later version adds - has already passed, so no
            # refusing path can leave the Run claiming a start it never entered.
            self._start_attempted_in_flight = True
        task = asyncio.ensure_future(operation())  # type: ignore[operator]
        # The task is owned from the moment it exists: an operation that never confirms it stopped keeps a
        # quarantined resource, so it must not be collectable while the Engine still holds that resource.
        self._PENDING_OWNERS[task] = True
        try:
            done, _ = await asyncio.wait({task}, timeout=remaining / 1_000_000_000)
        except _BudgetExhausted:
            # The internal "no budget remains" signal is not a dependency failure and is never
            # relabelled: it is re-raised so the phase loop records the abort it really is.
            raise
        except asyncio.CancelledError:
            # The caller cancelled this Run while the operation was in flight. The operation is not
            # killed: it is asked to cancel, remembered as pending and its late outcome is consumed
            # silently by :meth:`_discard_late`.
            task.cancel()
            self._discard_late(task)
            progress.pending_task = task
            progress.cancelled_by_caller = True
            raise
        if not done:
            # The bound elapsed. Stop waiting, request the cancellation and keep the reference: the
            # operation is *pending*, not stopped, and the snapshot must say so.
            task.cancel()
            self._discard_late(task)
            progress.pending_task = task
            progress.local_bound_elapsed = progress.local_bound_first
            raise RuntimeError("EVIDENCE_INVALID", stage, ("deadline",),
                               "the operation did not finish within its allowance")
        progress.pending_task = None
        # A business result is only accepted while its bound still holds. An operation that returned
        # *after* its allowance is a late result: it is discarded rather than committed as a PASS, a
        # receipt, a trace fact, a counter or further business work. Which bound was crossed decides
        # the reason, and the Run's own deadline is what a tie or an equal crossing reports.
        if progress.deadline_ns is not None and self._now_ns() >= progress.deadline_ns:
            progress.local_bound_elapsed = False
            raise RuntimeError("EVIDENCE_INVALID", stage, ("deadline",),
                               "the business deadline elapsed before the result was accepted")
        if operation_ns is not None and self._now_ns() >= operation_ns:
            progress.local_bound_elapsed = True
            raise RuntimeError("EVIDENCE_INVALID", stage, ("deadline",),
                               "the local allowance elapsed before the result was accepted")
        try:
            result = task.result()
        except asyncio.CancelledError:
            # The dependency cancelled *its own* task. This is not the host cancelling the Run, and
            # reporting it as a user cancellation would invent a decision nobody made: it is a
            # facility failure, and the flag records which of the two really happened. It is marked to
            # propagate because it is the facility failure the contract keeps raising.
            progress.dependency_self_cancel = True
            progress.facility_failure = True
            # A self-cancellation is one of the facility failures the contract keeps propagating, so
            # the translated mark is set here as well as on the adapter-observe translation.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", stage,
                               ("dependency",), "the dependency cancelled its own operation") from None
        if on_accepted is not None:
            on_accepted()  # type: ignore[operator]
        return result

    @classmethod
    def _discard_late(cls, task: asyncio.Future) -> None:
        """Consume a task's late outcome, and keep a strong reference while it is still running.

        Consuming the outcome is what stops a late result or exception from escaping. The strong
        reference is what keeps an operation that never confirmed it stopped alive: the resource it still
        holds is quarantined, so the attempt that holds it may not be collected mid-flight. The reference
        is held by the Engine and by the process-wide owner registry, and it is released only through an
        explicit release path.
        """

        def _consume(finished: asyncio.Future) -> None:
            try:
                finished.exception()
            except BaseException:  # noqa: BLE001 - a late outcome is consumed, never re-raised
                return

        task.add_done_callback(_consume)
        if not task.done():
            cls._PENDING_OWNERS[task] = True

    @classmethod
    def _release_pending_owner(cls, task: asyncio.Future | None) -> None:
        """Release the strong reference once the operation really confirmed it stopped."""
        if task is not None:
            cls._PENDING_OWNERS.pop(task, None)

    def _note_pending_operation(self, progress: _Progress) -> None:
        """Record that an asynchronous operation was still running when the Run returned."""
        pending = progress.pending_task
        if pending is not None and not pending.done():
            progress.pending_operation_at_return = True
            # A resource whose operation never confirmed it stopped is quarantined. The decision is
            # never lifted by reading it, and no public reset exists.
            self._quarantined = True

    def _write_finalization(self, progress: _Progress, request: RunRequest,
                            primary_status: RuntimeTerminalStatus,
                            primary_reason: RuntimeReason, *, cleanup_status: str,
                            trace_status: str,
                            terminal_record_id: str | None = None,
                            cleanup_error_code: RuntimeReason | None = None,
                            trace_error_code: RuntimeReason | None = None,
                            finalize_deadline_exceeded: bool = False) -> None:
        """Store the bounded finalization snapshot, through its own declared cross-field rules."""
        self._note_pending_operation(progress)
        safe_trace_status = trace_status
        safe_terminal_id = terminal_record_id
        if trace_status == "persisted" and terminal_record_id is None:
            # A persisted fact without an identifier would be a claim the Runtime cannot support.
            safe_trace_status = "unknown"
        if trace_status != "persisted":
            safe_terminal_id = None
        try:
            self._last_finalization = build_finalization(
                run_id=request.run_id,
                primary_terminal_status=primary_status,
                primary_reason_code=primary_reason,
                pending_operation_at_return=progress.pending_operation_at_return,
                cleanup_status=cleanup_status,
                cleanup_error_code=cleanup_error_code,
                trace_status=safe_trace_status,
                trace_error_code=trace_error_code,
                terminal_record_id=safe_terminal_id,
                finalize_deadline_exceeded=finalize_deadline_exceeded,
                quarantined=self._quarantined)
        except ValueError:
            # A snapshot the declared rules exclude must never replace a good diagnosis with a bad
            # one, so the previous snapshot is kept and the quarantine decision stands.
            return

    def _reset_for_new_run(self) -> None:
        """Clear the previous Run's snapshot when a genuinely new Run is accepted."""
        self._last_finalization = None

    # -- resource occupancy ----------------------------------------------------------

    @classmethod
    def _occupancy_holder(cls, adapter: object) -> str | None:
        """Return the quarantine reason if this adapter object's resource is still occupied."""
        with cls._OCCUPANCY_LOCK:
            entry = cls._OCCUPANCY.get(id(adapter))
            if entry is None or entry[0] is not adapter:
                # A recycled address belongs to a different object, so it is not this resource.
                return None
            return entry[1]

    @classmethod
    def _occupy(cls, adapter: object, reason: str) -> None:
        """Mark this adapter object's resource occupied or quarantined, keeping a reference to it."""
        with cls._OCCUPANCY_LOCK:
            cls._OCCUPANCY[id(adapter)] = (adapter, reason)

    @classmethod
    def _release(cls, adapter: object) -> None:
        """Release the occupancy of this adapter object's resource after a real successful cleanup."""
        with cls._OCCUPANCY_LOCK:
            entry = cls._OCCUPANCY.get(id(adapter))
            if entry is not None and entry[0] is adapter:
                del cls._OCCUPANCY[id(adapter)]

    def _release_finished_owners(self) -> None:
        """Drop the strong owners of operations that really finished.

        A running operation keeps its owner: the resource it holds is quarantined, so it may not be
        collected mid-flight. Only an operation that has really finished stops needing that protection.
        """
        finished = [task for task in self._PENDING_OWNERS if task.done()]
        for task in finished:
            self._PENDING_OWNERS.pop(task, None)

    def _settle_occupancy(self, progress: _Progress, cleanup_status: str,
                          trace_status: str) -> None:
        """Release the resource only when the whole teardown really succeeded.

        Every other ending keeps the resource occupied: a pending operation, a failed or timed-out
        stop, a skipped stop and an uncertain terminal write all mean the resource's state is not
        settled, and a later Run must not treat it as free. The release is the only path that ever
        clears an entry, and it is never reachable through a public property or a returned value.
        """
        if cleanup_status == "succeeded" and progress.stop_confirmed and trace_status == "persisted":
            self._release(self._adapter)
            return
        if cleanup_status == "not_required" and trace_status == "persisted":
            # Nothing was ever started, so no resource was acquired and there is nothing to release.
            self._release(self._adapter)
            return
        reason = "pending_operation" if progress.pending_operation_at_return else cleanup_status
        if trace_status != "persisted":
            reason = "uncertain_terminal_write"
        self._occupy(self._adapter, reason)
        self._quarantined = True

    def _seal_machine(self, machine: RuntimeStateMachine, progress: _Progress,
                      request: RunRequest, outcome: _RunOutcome) -> None:
        """Apply the deferred terminal move, now that the teardown conclusion is known.

        The move is the one the phase body decided, so a Run that was refused a terminal edge still
        records the phase it really reached; it is applied here rather than in the loop precisely so
        that a candidate success is not recorded as permanent before its cleanup has succeeded.
        """
        move = outcome.deferred_terminal
        target = {
            RuntimeTerminalStatus.COMPLETED: RuntimePhase.COMPLETED,
            RuntimeTerminalStatus.FAILED: RuntimePhase.FAILED,
            RuntimeTerminalStatus.ABORTED: RuntimePhase.ABORTED,
            RuntimeTerminalStatus.CANCELLED: RuntimePhase.CANCELLED,
            RuntimeTerminalStatus.TIMEOUT: RuntimePhase.TIMED_OUT,
        }[outcome.terminal_status]
        # A decision that agrees with the final outcome is simply applied, and the Run ends in the
        # phase it really reached. A candidate success whose cleanup then failed is different: the
        # cleanup conclusion decides, so the machine is sealed at the terminal the Run really has
        # without ever recording the phase it only appeared to reach.
        if move is not None and move[0] is target:
            self._force_terminal(machine, progress, request, move[0], move[1])
            return
        if machine.phase is not target:
            self._force_terminal(machine, progress, request, target, outcome.terminal_reason)

    # -- request validation ---------------------------------------------------------

    def _validate_request(self, request: object) -> RunRequest:
        """Check the complete declared shape of the request before anything is started or written.

        The task criteria, their goal binding, the JSON-rule payload of every condition and the
        declared field shapes are all validated here, before the first ``start``, ``observe`` and
        trace write, so a malformed request can never be discovered halfway through a Run that has
        already touched the world. The Run then consumes its *own* detached snapshot, so a later
        mutation of the host's object cannot change the goal this Run is judged against.
        """
        if not isinstance(request, RunRequest):
            raise RuntimeError("INPUT_INVALID", "validate", ("request",),
                               "the request is not a RunRequest record")
        try:
            # Re-derive the record from its own stored fields through its own schema, so a copied
            # or constructed instance is checked rather than trusted on class identity. The active
            # path is proved cycle-free *before* the recursive validation runs, so a self-referential
            # graph is one bounded INPUT_INVALID rather than an interpreter recursion failure.
            state = detached_state(request)
            require_acyclic(state, ("request",))
            detached = RunRequest.model_validate(state)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("INPUT_INVALID", "validate", ("request",),
                               "the request does not match its declared schema",
                               cause=exc) from None
        criteria = detached.task_criteria
        if not isinstance(criteria, TaskCompletionCriteria):
            raise RuntimeError("INPUT_INVALID", "validate", ("task_criteria",),
                               "the task criteria are not the declared record type")
        if criteria.goal_id != detached.goal_id:
            raise RuntimeError("INPUT_INVALID", "validate", ("task_criteria", "goal_id"),
                               "the task criteria belong to a different goal")
        # The existing oracle publishes the criteria prevalidation, so the Runtime reuses that
        # check rather than reimplementing the JSON-rule algorithm. A malformed rule is refused
        # here, before any trace write, start or observation. This prevalidation is independent of
        # the contract-hooks factory: supplying a factory never bypasses it. A host that owns a
        # different domain rule grammar supplies its own validator explicitly and the same boundary
        # runs that validator instead, still before any resource is acquired.
        self._prevalidate_criteria(criteria)
        # The Run consumes its *own* deep snapshot of the criteria. Pydantic validation keeps an
        # already-valid nested model as it found it, so the host's conditions and payload objects
        # would otherwise still be shared: a dependency that rewrites the host's goal through the
        # object it was handed could change what this Run is judged against. A JSON round trip
        # therefore rebuilds the whole structure as fresh objects.
        return detached.model_copy(update={"task_criteria": _deep_criteria(criteria)})

    def _prevalidate_criteria(self, criteria: TaskCompletionCriteria) -> None:
        """Prevalidate the task criteria through the default or the injected validator.

        This boundary runs before the first trace write, ``start`` or ``observe``, and its cost is
        already charged because the Run's business start was taken on entry. The default path is
        the original JSON prevalidation, byte-for-byte. With an injected validator the Runtime
        hands over a detached copy, refuses anything other than ``None``, checks both the copy and
        the saved record for mutation and maps every failure onto its existing bounded code.

        Every converted ordinary failure is raised after its handler has finished, so the public
        error has no reachable original exception chain and no supplied text. ``BaseException`` is
        never caught, so a cancellation or interrupt keeps its identity.
        """
        validator = self._task_criteria_validator
        if validator is None:
            try:
                prevalidate_criteria(criteria)
            except VerificationError as exc:
                raise RuntimeError("INPUT_INVALID", "validate",
                                   ("task_criteria", "conditions"), exc.message,
                                   cause=exc) from None
            return
        failure: RuntimeError | None = None
        detached: object | None = None
        saved_payload: object | None = None
        detached_payload: object | None = None
        try:
            saved_payload = self._call_signature(record_payload(criteria))
            detached = self._isolate(criteria, TaskCompletionCriteria)
            detached_payload = self._call_signature(record_payload(detached))
        except (ValueError, TypeError):
            failure = RuntimeError("INPUT_INVALID", "validate", ("task_criteria",),
                                   "the request does not match its declared schema")
        if failure is not None:
            raise failure
        try:
            validate = validator.validate
        except Exception:  # noqa: BLE001 - a dependency read failure is a facility failure
            failure = RuntimeError("DEPENDENCY_FAILURE", "validate", ("task_criteria",),
                                   "the dependency did not answer as declared")
        if failure is not None:
            raise failure
        if not callable(validate):
            raise RuntimeError("DEPENDENCY_FAILURE", "validate", ("task_criteria",),
                               "the dependency did not answer as declared")
        outcome: object = None
        rejection: VerificationError | None = None
        try:
            outcome = validate(criteria=detached)
        except VerificationError as exc:
            rejection = exc
        except Exception:  # noqa: BLE001 - an ordinary dependency failure is bounded here
            failure = RuntimeError("DEPENDENCY_FAILURE", "validate", ("task_criteria",),
                                   "the dependency did not answer as declared")
        if rejection is not None:
            # The validator's own structural location and fixed diagnostic are reused; the
            # exception object itself is not attached, so no original chain is reachable.
            raise RuntimeError("INPUT_INVALID", "validate", tuple(rejection.location),
                               rejection.message)
        if failure is not None:
            raise failure
        if inspect.isawaitable(outcome):
            self._close_awaitable(outcome)
            raise RuntimeError("DEPENDENCY_FAILURE", "validate", ("task_criteria",),
                               "the dependency did not answer as declared")
        if outcome is not None:
            raise RuntimeError("DEPENDENCY_FAILURE", "validate", ("task_criteria",),
                               "the dependency did not answer as declared")
        assert detached_payload is not None
        self._check_inputs_unchanged("validate", "task_criteria", detached, detached_payload)
        self._check_inputs_unchanged("validate", "task_criteria", criteria, saved_payload)

    # -- the sequential loop --------------------------------------------------------

    async def _drive(self, request: RunRequest, business_start_ns: int) -> RunResult:
        progress = _Progress(
            remaining_steps=request.max_steps,
            remaining_planning=request.max_planning_calls,
            remaining_retries=request.max_retries,
            remaining_replans=request.max_replans,
            business_start_ns=business_start_ns,
        )
        # A fresh Run starts with no clock refusal recorded.
        self._facility_clock_failed = False
        machine = RuntimeStateMachine()
        outcome = _RunOutcome()
        start_attempted = False
        cleanup_status = "not_required"
        cleanup_error_code: RuntimeReason | None = None
        trace_status = "not_attempted"
        trace_error_code: RuntimeReason | None = None
        cleanup_failure: RuntimeError | None = None

        try:
            # The host budget is revalidated in its raw declared shape and consumed before any
            # side effect. The first legal reading also fixes the Run's business deadline, so
            # validation, isolation, rule prevalidation and this reading are all charged to it.
            self._budget_refresh(progress, "validate")
            self._establish_deadline(progress, progress.budget_ms)
            self._trace_append(request, progress, RunStarted)
            if progress.budget_ms > 0:
                # The flag is set *before* the call, so an ordinary exception raised part-way
                # through start still triggers the normal stop: a partially started resource is
                # exactly the case that most needs cleanup, and only the never-attempted
                # zero-budget or already-expired path performs no business I/O at all.
                #
                # The allowance is re-checked immediately *before* the flag is set, so a Run whose own
                # deadline is already spent performs no lifecycle call at all: claiming a start was
                # attempted would issue a ``stop`` for a resource that was never acquired.
                self._check_deadline(progress, "observe")
                # The deadline is re-checked here, immediately before the real call: a crossing between
                # the decision to start and the call itself performs no lifecycle I/O at all, so there
                # is neither a ``start`` to enter nor a resource that would need a ``stop``. The flag
                # below is set only once both checks have passed, so the refusing path leaves it clear
                # and the cleanup correctly reports that nothing was acquired.
                self._check_deadline(progress, "observe")
                start_attempted = True
                # ``start_attempted`` records the Run's *decision*. Whether a resource may have been
                # acquired is a different fact, and ``_start_adapter`` sets its own flag at the single
                # boundary where the start operation is really invoked. Every pre-start refusal above
                # leaves that flag clear, however many budget reads it performed, so nothing stops a
                # resource that was never acquired.
                await self._start_adapter(progress)
                self._move(machine, progress, request, RuntimePhase.OBSERVING,
                           RuntimeReason.TASK_ACHIEVED)
                await self._run_phases(request, progress, machine, outcome)
            else:
                outcome.terminal_reason = RuntimeReason.BUDGET_EXHAUSTED
                outcome.terminal_status = RuntimeTerminalStatus.ABORTED
                self._force_terminal(machine, progress, request, RuntimePhase.ABORTED,
                                     outcome.terminal_reason)

        except _BudgetExhausted:
            # A host that declares no budget keeps its existing abort semantics. This is a host
            # declaration, not a measured deadline, and the two are never conflated.
            outcome.terminal_reason = RuntimeReason.BUDGET_EXHAUSTED
            outcome.terminal_status = RuntimeTerminalStatus.ABORTED
        except _TerminalReached:
            # The phase loop already recorded the terminal outcome and stashed the phase move; the
            # move is applied once the teardown conclusion is known, below.
            pass
        except asyncio.CancelledError:
            # A delivered caller cancellation is recorded as the first non-success diagnosis and is
            # re-raised only after the bounded teardown has finished.
            progress.cancelled_by_caller = True
            outcome.terminal_reason = RuntimeReason.USER_CANCELLED
            outcome.terminal_status = RuntimeTerminalStatus.CANCELLED
            outcome.deferred_terminal = (RuntimePhase.CANCELLED,
                                         RuntimeReason.USER_CANCELLED)
        except RuntimeError as exc:
            outcome.primary_failure = exc
            if exc.code == "EVIDENCE_INVALID" and exc.location == ("deadline",):
                # A real elapsed deadline is a timeout, never a facility failure and never an abort.
                # Which bound elapsed decides the reason: a committed contract's own local allowance
                # is an operation timeout, the Run's remaining budget is a deadline expiry.
                if progress.local_bound_elapsed or progress.local_bound_first:
                    outcome.terminal_reason = RuntimeReason.OPERATION_TIMEOUT
                else:
                    outcome.terminal_reason = RuntimeReason.DEADLINE_EXCEEDED
                outcome.terminal_status = RuntimeTerminalStatus.TIMEOUT
                outcome.deferred_terminal = (RuntimePhase.TIMED_OUT,
                                             outcome.terminal_reason)
            else:
                outcome.terminal_reason = (RuntimeReason(exc.code)
                                           if exc.code in {item.value for item in RuntimeReason}
                                           else RuntimeReason.DEPENDENCY_FAILURE)
                outcome.terminal_status = RuntimeTerminalStatus.FAILED
                progress.facility_failure = progress.facility_failure or \
                    self._facility_clock_failed
                progress.translated_failure = progress.translated_failure \
                    or self._facility_clock_failed
                # A refusal the *engine itself* produced from a facility failure - a dependency that
                # cancelled its own operation, or a real clock that went backwards - is a facility
                # failure the contract keeps propagating, not a business refusal to publish.
                outcome.propagate_failure = progress.translated_failure
                outcome.deferred_terminal = (RuntimePhase.FAILED, outcome.terminal_reason)
        except (RecoveryError, VerificationError, ContractBuildError, ActionValidationError,
                TraceError, ValueError, TypeError, OSError, ArithmeticError, KeyError) as exc:
            # An ordinary dependency exception is a bounded facility failure: it is translated
            # rather than escaping raw, and the Run still attempts its normal finalisation. It is also
            # the one kind of business failure the contract keeps *propagating* to the caller once the
            # finalisation is complete, which the flag records for the tail below.
            outcome.primary_failure = exc
            outcome.terminal_reason = RuntimeReason.DEPENDENCY_FAILURE
            outcome.terminal_status = RuntimeTerminalStatus.FAILED
            outcome.propagate_failure = True
            outcome.deferred_terminal = (RuntimePhase.FAILED,
                                         RuntimeReason.DEPENDENCY_FAILURE)

        # The bounded teardown runs in two independent segments with their own allowances: the first
        # covers confirming and cancelling the current operation together with the single ``stop``, and
        # the second covers the safe Memory write, the terminal-fact preparation and the terminal-fact
        # append. Neither segment shares its balance with the other or with the business deadline, and
        # both are timed with the real monotonic clock, so a damaged business clock cannot extend them.
        #
        # A Run that never entered ``start`` acquired no resource, so its teardown is a no-op: issuing
        # ``stop`` on a resource nobody started would be an unrequested lifecycle call, and the Run
        # still records its own terminal fact because an explicit zero declaration is a recorded
        # timeout, not a silent one.
        if start_attempted:
            cleanup_status, cleanup_failure = await self._bounded_cleanup(
                request, progress, outcome, start_attempted)
        else:
            cleanup_status, cleanup_failure = "not_required", None
        if cleanup_failure is not None:
            cleanup_error_code = RuntimeReason.CLEANUP_FAILURE
            if outcome.primary_failure is None and not progress.cancelled_by_caller \
                    and outcome.terminal_status is RuntimeTerminalStatus.COMPLETED:
                # The candidate success was only a *candidate*: the final cleanup conclusion decides.
                # Nothing more important happened, so the cleanup failure is the leading diagnosis and
                # the Run is published as a failure, not as a success that was later retracted.
                outcome.primary_failure = cleanup_failure
                outcome.terminal_reason = RuntimeReason.CLEANUP_FAILURE
                outcome.terminal_status = RuntimeTerminalStatus.FAILED
            elif isinstance(outcome.primary_failure, RuntimeError):
                outcome.primary_failure.with_cleanup_failure(cleanup_failure)
            elif outcome.primary_failure is not None:
                outcome.primary_failure = self._bounded_failure(
                    outcome.primary_failure).with_cleanup_failure(cleanup_failure)

        if outcome.terminal_status is not RuntimeTerminalStatus.COMPLETED:
            outcome.verification_record_id = None
        if progress.cancelled_by_caller and outcome.terminal_status is RuntimeTerminalStatus.COMPLETED:
            # A cancellation the host really delivered decides the terminal only while the Run is still
            # a success candidate. Once a timeout or an abort has really been observed, that is the
            # primary diagnosis and the cancellation cannot rewrite it: the caller still receives the
            # ``CancelledError``, because the propagation and the Run's own outcome are different things.
            outcome.primary_failure = None
            outcome.terminal_reason = RuntimeReason.USER_CANCELLED
            outcome.terminal_status = RuntimeTerminalStatus.CANCELLED
            outcome.verification_record_id = None
            outcome.deferred_terminal = (RuntimePhase.CANCELLED, RuntimeReason.USER_CANCELLED)
            self._seal_machine(machine, progress, request, outcome)
        else:
            # The terminal phase is sealed only now, once the cleanup conclusion is known, so the Run
            # is never permanently recorded as completed and then contradicted by a failed stop.
            self._seal_machine(machine, progress, request, outcome)
        # The safe Memory diagnosis is written inside the finalization allowance, and the terminal
        # append is attempted only while that same allowance still holds.
        trace_status, terminal_record_id, trace_error_code = self._persist_terminal(
            request, progress, outcome)

        # The resource verdict is settled *before* the snapshot is written, so the read-only record
        # and the Engine's own state are decided together rather than one describing the other.
        self._settle_occupancy(progress, cleanup_status, trace_status)
        self._release_finished_owners()
        self._write_finalization(
            progress, request, outcome.terminal_status, outcome.terminal_reason,
            cleanup_status=cleanup_status, cleanup_error_code=cleanup_error_code,
            trace_status=trace_status, trace_error_code=trace_error_code,
            terminal_record_id=terminal_record_id,
            finalize_deadline_exceeded=progress.finalize_deadline_exceeded)
        if progress.cancelled_by_caller:
            # The caller's cancellation is never swallowed. A trace failure during teardown is
            # attached as the explicit bounded cause rather than replacing the cancellation.
            error = asyncio.CancelledError()
            if trace_error_code is not None:
                error.__cause__ = RuntimeError("TRACE_FAILURE", "finalize", ("append",),
                                               "the terminal fact could not be persisted")
            raise error
        if progress.finalization_refusal is not None:
            # The finalising segment met a refusal while preparing the terminal record - a Memory write, an
            # unusable event-time source, a record that cannot be constructed, or any other finite refusal.
            # The snapshot and the resource verdict are already written, so the caller now receives that
            # bounded diagnosis rather than having it relabelled as this segment's persistence problem.
            raise progress.finalization_refusal
        if trace_error_code is not None:
            # The failure is reported on the finalization stage, which is where the unique terminal
            # fact is persisted; the original bounded location and message are preserved rather than
            # replaced by a generic one.
            raise RuntimeError("TRACE_FAILURE", "finalize", ("append",),
                               "the terminal fact could not be persisted")
        if progress.memory_failed and trace_status == "persisted":
            # Memory refused, and the terminal fact was still written inside the remaining allowance. The
            # explicit bounded diagnosis is what the caller receives: a Run whose own Memory refused is not
            # reported as a clean success.
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the terminal fact could not be persisted")
        if progress.memory_failed and trace_status == "persisted":
            # Memory refused and the terminal fact was still written. The explicit bounded diagnosis is
            # what the caller receives: a Run whose own Memory refused is not reported as a clean success,
            # and the snapshot keeps stating the Trace fact it really confirmed.
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the terminal fact could not be persisted")
        if cleanup_failure is not None:
            if outcome.terminal_status in (RuntimeTerminalStatus.TIMEOUT,
                                           RuntimeTerminalStatus.ABORTED):
                # A timeout or an abort whose terminal fact was persisted keeps its primary status and
                # returns the RunResult: the cleanup verdict is already carried as the secondary
                # diagnostic on the snapshot, so the caller is not handed a different outcome than the
                # one the Run recorded.
                return self._finish(request, progress, machine, outcome.terminal_status,
                                    outcome.terminal_reason, outcome.verification_record_id,
                                    terminal_record_id or "")
            # The leading diagnosis wins, and the cleanup fact is preserved on it as the bounded
            # secondary fact rather than replacing it. Either way the caller receives the reason the
            # Run really failed, so a Run whose shutdown did not succeed is never mistaken for a clean
            # one.
            if isinstance(outcome.primary_failure, RuntimeError):
                raise outcome.primary_failure
            if outcome.primary_failure is not None and not outcome.propagate_failure:
                raise self._bounded_failure(outcome.primary_failure)
            raise cleanup_failure
        if outcome.propagate_failure and outcome.terminal_status is RuntimeTerminalStatus.FAILED \
                and outcome.terminal_reason is RuntimeReason.DEPENDENCY_FAILURE \
                and isinstance(outcome.primary_failure, BaseException) \
                and getattr(outcome.primary_failure, "code", "DEPENDENCY_FAILURE") \
                == "DEPENDENCY_FAILURE":
            # An ordinary facility failure keeps its bounded propagation: the finalisation is complete
            # and the caller still receives the reason the Run really failed. A translated refusal
            # whose own bounded diagnosis is different - an evidence refusal the Runtime refined - is a
            # business outcome and is published instead.
            raise outcome.primary_failure

        return self._finish(request, progress, machine, outcome.terminal_status,
                            outcome.terminal_reason, outcome.verification_record_id,
                            terminal_record_id or "")

    async def _bounded_cleanup(self, request: RunRequest, progress: _Progress,
                               outcome: _RunOutcome, start_attempted: bool
                               ) -> tuple[str, RuntimeError | None]:
        """Run the first bounded teardown segment and describe what really happened.

        The segment owns one allowance, entered once, and shares it between two things: waiting for the
        current operation to *confirm* it stopped, and issuing the single ``stop``. Only when that
        allowance is exhausted while the operation is still pending is the ``stop`` skipped - a
        cooperative operation that confirms its cancellation within the allowance is still stopped, and
        the Run returns an ordinary timeout result rather than quarantining a resource it cleaned up.
        A Run that never entered ``start`` performs no lifecycle I/O at all.

        The allowance is measured against the real clock, never the business clock, so a failed
        business clock cannot prevent the resource from being stopped and the Run from being recorded.
        """
        if not start_attempted or not self._start_attempted_in_flight:
            # No ``start`` was really entered, so no resource may have been acquired and there is nothing
            # to stop. A Run that only *decided* to start, and was refused before the call, performs no
            # lifecycle I/O at all.
            return "not_required", None
        allowance_ns = milliseconds_to_nanoseconds(
            self._timing.cleanup_timeout_ms, ("cleanup_timeout_ms",))
        # The declared deadline is what the business clock answers, and the real deadline is the safety
        # bound that a damaged business clock cannot disable.
        deadline_ns = self._real_now_ns() + allowance_ns
        # The cooperative confirmation is part of this same allowance: wait for the pending operation
        # to acknowledge its cancellation before deciding whether the stop can still be issued.
        status, failure, pending_now = await self._confirm_stopped(progress, deadline_ns)
        if pending_now:
            self._note_pending_operation(progress)
            return status, failure
        stop = getattr(self._adapter, "stop", None)
        if stop is None:
            return "failed", RuntimeError("CLEANUP_FAILURE", "cleanup", ("adapter",),
                                          "the adapter does not provide a stop operation")
        remaining_ns = deadline_ns - self._real_now_ns()
        if remaining_ns <= 0:
            return "timed_out", RuntimeError("CLEANUP_FAILURE", "cleanup", ("adapter",),
                                             "the cleanup allowance elapsed before the stop")
        try:
            awaitable = stop()
        except Exception:  # noqa: BLE001 - a synchronous refusal is a cleanup failure
            return "failed", RuntimeError("CLEANUP_FAILURE", "cleanup", ("adapter",),
                                          "the adapter stop operation failed")
        if not hasattr(awaitable, "__await__"):
            progress.stop_confirmed = True
            return "succeeded", None
        task = asyncio.ensure_future(awaitable)
        # The stop is the one lifecycle action that must not be abandoned mid-flight by a caller
        # cancellation: the resource is already acquired, so the teardown has to reach its conclusion.
        # The *wait* is shielded for exactly that reason, and the cancellation is remembered so it can
        # be delivered once the Run has recorded its own end. The allowance is unchanged, no second stop
        # is issued, and nothing waits for an uncancellable task: the bound below still applies.
        waiter = asyncio.ensure_future(asyncio.shield(task))
        # The shield future fails whenever the inner task does. Its failure is consumed here so it can
        # never surface as an unconsumed asynchronous exception, and the inner task is what carries the
        # real verdict.
        self._discard_late(waiter)
        cancellation_seen = False
        while True:
            remaining_ns = deadline_ns - self._real_now_ns()
            pending = {waiter} if not waiter.done() else set()
            if task.done():
                pending = set()
            if remaining_ns <= 0 or not pending:
                break
            try:
                await asyncio.wait(pending, timeout=remaining_ns / 1_000_000_000)
            except asyncio.CancelledError:
                # A caller cancellation while the stop is in flight is remembered, and the stop is left
                # to finish inside the *same* allowance rather than being torn down here. The deadline
                # is not extended and no second stop is issued.
                progress.cancelled_by_caller = True
                cancellation_seen = True
                continue
        _ = cancellation_seen
        if not task.done():
            # The stop did not confirm it finished inside the allowance.
            task.cancel()
            self._discard_late(task)
            progress.pending_task = task
            self._note_stop_pending(progress)
            return "timed_out", RuntimeError("CLEANUP_FAILURE", "cleanup", ("adapter",),
                                             "the adapter stop operation did not finish in time")
        try:
            task.result()
        except asyncio.CancelledError:
            # The stop cancelled *itself*. That is not the host cancelling the Run: it is a cleanup
            # failure, and the teardown still records its terminal fact and its snapshot.
            return "failed", RuntimeError("CLEANUP_FAILURE", "cleanup", ("adapter",),
                                          "the adapter stop operation cancelled itself")
        except Exception:  # noqa: BLE001 - a cleanup failure is reported, never raised raw
            return "failed", RuntimeError("CLEANUP_FAILURE", "cleanup", ("adapter",),
                                          "the adapter stop operation failed")
        progress.pending_task = None
        progress.stop_confirmed = True
        _ = (request, outcome)
        return "succeeded", None

    def _note_stop_pending(self, progress: _Progress) -> None:
        """Record that the stop itself never confirmed it finished.

        Only the lifecycle action can produce this: the Run asked the resource to stop, the allowance
        elapsed, and nothing confirmed the resource actually stopped. The snapshot must say both that
        the cleanup did not succeed and that the Run holds a resource it cannot vouch for.
        """
        progress.pending_operation_at_return = True
        progress.stop_confirmed = False
        self._quarantined = True

    async def _confirm_stopped(self, progress: _Progress, deadline_ns: int
                               ) -> tuple[str, RuntimeError | None, bool]:
        """Wait, inside the cleanup allowance, for the pending operation to confirm it stopped.

        A pending task is *asked* to cancel when the business bound elapses; whether it acknowledged is
        something only the task can settle. Waiting for that acknowledgement inside the same allowance
        is what distinguishes a cooperative dependency, which really stopped and whose resource can be
        reused, from one that is still running, whose resource must be quarantined. Returns the cleanup
        status, the bounded failure and whether the operation is still pending.
        """
        pending = progress.pending_task
        if pending is None or pending.done():
            return "succeeded", None, False
        if not pending.cancelled():
            pending.cancel()
        while True:
            remaining_ns = deadline_ns - self._real_now_ns()
            if remaining_ns <= 0:
                break
            try:
                done, _ = await asyncio.wait({pending}, timeout=remaining_ns / 1_000_000_000)
            except asyncio.CancelledError:
                # A caller cancellation during the confirmation is remembered, and the confirmation
                # continues: a repeat cancellation neither extends the allowance nor lets the teardown
                # skip the cooperative check, so an operation that really acknowledged its cancellation
                # is still stopped exactly once and the caller still receives the cancellation after the
                # Run has recorded its own end. The allowance is *not* extended by the interruption, and
                # the deadline is re-evaluated at the top of the loop, so a repeated cancellation cannot
                # spin here.
                progress.cancelled_by_caller = True
                done = set()
            if done or pending.done():
                break
        if pending.done():
            # The operation confirmed it stopped inside the allowance: its late outcome is consumed
            # and, because it really stopped, the resource is not quarantined.
            self._discard_late(pending)
            progress.pending_task = None
            return "succeeded", None, False
        self._note_pending_operation(progress)
        return ("skipped_pending_operation",
                RuntimeError("CLEANUP_FAILURE", "cleanup", ("adapter",),
                             "the current operation did not confirm it stopped"),
                True)

    def _persist_terminal(self, request: RunRequest, progress: _Progress,
                          outcome: _RunOutcome) -> tuple[str, str | None, RuntimeReason | None]:
        """Attempt the unique terminal fact inside the second bounded segment.

        The finalization allowance is independent of the cleanup allowance and of the business
        deadline, and it is measured on the *real* clock, so a frozen or damaged business clock cannot
        make it unbounded. The segment covers three things, because all three are part of finalising:
        the safe Memory diagnosis, the preparation of the unique terminal fact, and the append itself.
        The allowance is re-checked immediately before the append, so a preparation that already spent
        it does not start a write. A successful synchronous append is a persisted fact; an append that
        raised may have written before it failed, so the outcome is genuinely *unknown* and is never
        retried or assumed to be zero writes; and an append that returns only after the allowance
        elapsed is the approved overdue shape: it really was written, and it is reported as persisted
        *and* as a failure rather than rewritten or duplicated.
        """
        finalize_deadline_ns = self._real_now_ns() + milliseconds_to_nanoseconds(
            self._timing.finalize_timeout_ms, ("finalize_timeout_ms",))
        progress.finalize_deadline_exceeded = False
        progress.memory_failed = False
        progress.finalization_refusal = None
        # The preparation reads the same allowance it is charged against, so a write that spends F is
        # visible to the preparation itself rather than only to the append that follows.
        self._finalize_deadline_ns = finalize_deadline_ns
        # The safe Memory diagnosis and the terminal preparation are charged to this same allowance.
        try:
            self._finalize_prepare(request, progress, outcome)
        except asyncio.CancelledError:
            # Control flow, never a facility verdict: the Memory preparation was cancelled and the
            # cancellation keeps its own semantics.
            raise
        except Exception as exc:  # noqa: BLE001 - every finite ordinary refusal is one boundary
            progress.finalize_deadline_exceeded = self._real_now_ns() >= finalize_deadline_ns
            bounded = self._bounded_preparation_refusal(exc)
            if bounded.code != "TRACE_FAILURE":
                # A bounded refusal the Run itself raised keeps its own meaning; it is never re-labelled as
                # this segment's persistence failure. It was normalized first, so only a refusal that really
                # satisfies the bounded contract reaches the caller.
                progress.finalization_refusal = bounded
                return "not_attempted", None, RuntimeReason.TRACE_FAILURE
            if not progress.memory_failed:
                return self._preparation_refused(progress, exc)
            # Memory refused, but the *terminal* record is a different fact: the unique terminal write is
            # the Run's remaining chance to record its end, so it is attempted - once - inside whatever
            # allowance is left, and its refusal is reported explicitly.
        # The *complete* terminal record is built here - its timestamp, status, reason and verification
        # reference - so the check that follows is a test on the record the append would really write,
        # rather than on a placeholder for it. Preparing the record is a facility boundary like any other,
        # so a refusal there is bounded: the Run reports what it could not do, keeps its honest snapshot and
        # settles its resource, instead of letting a raw exception escape the teardown.
        try:
            occurred_at = self._read_event_time()
            progress.terminal_occurred_at = occurred_at
            self._build_terminal_record(request, progress, outcome, occurred_at)
        except asyncio.CancelledError:
            # Control flow, never a facility verdict: the cancellation keeps its own semantics.
            raise
        except Exception as exc:  # noqa: BLE001 - every finite ordinary refusal is one boundary
            progress.finalize_deadline_exceeded = self._real_now_ns() >= finalize_deadline_ns
            return self._preparation_refused(progress, exc)
        # The allowance is re-checked immediately before the append: a construction that spent F produced
        # no record to write, so nothing is appended and the trace is honestly reported as never attempted.
        preparation_spent_f = self._real_now_ns() >= finalize_deadline_ns
        progress.finalize_deadline_exceeded = preparation_spent_f
        if preparation_spent_f or not self._finalization_has_room(finalize_deadline_ns):
            # F is gone, so no new Memory or Trace write may start at all - not even the terminal attempt
            # a Memory refusal would otherwise justify. A Memory refusal is a reason to *try* the terminal
            # write, never a licence to write after the allowance is spent.
            return "not_attempted", None, RuntimeReason.TRACE_FAILURE
        try:
            record_id = self._finalize(progress.terminal_record)
        except RuntimeError as exc:
            progress.finalize_deadline_exceeded = self._real_now_ns() >= finalize_deadline_ns
            if exc.code != "TRACE_FAILURE":
                # Anything other than a store failure keeps its own bounded diagnosis, so a damaged
                # clock or an invalid record is never relabelled as a persistence problem.
                raise
            # The store may have written before it failed, so the result is undetermined. It is
            # never retried, and the single terminal fact is never duplicated.
            return "unknown", None, RuntimeReason.TRACE_FAILURE
        # The append returned, and its own cost - the timestamp read and the record building - is part
        # of this allowance, so the allowance is re-checked here. A write that really returned is a
        # persisted fact: it is reported as persisted, never rewritten and never duplicated, and it is
        # additionally reported as overdue when the allowance had already elapsed.
        if self._real_now_ns() >= finalize_deadline_ns:
            progress.finalize_deadline_exceeded = True
            return "persisted", record_id, RuntimeReason.TRACE_FAILURE
        # The append really returned, so the terminal identifier is confirmed. Publishing it keeps the
        # stored Memory state consistent with the Trace fact the Run actually wrote. The publication is a
        # retained Memory write, so it is bounded by the same allowance: the allowance is checked before it
        # starts, and again after it returns, and a refusal is propagated explicitly rather than being
        # dressed up as elapsed time.
        if not self._finalization_has_room(finalize_deadline_ns):
            # No room to start the publication: it is not started, and the confirmed fact is still
            # reported as persisted because it really was written.
            progress.finalize_deadline_exceeded = True
            return "persisted", record_id, RuntimeReason.TRACE_FAILURE
        try:
            self._publish_confirmed_terminal(progress, record_id)
        except RuntimeError as exc:
            if exc.code != "TRACE_FAILURE":
                raise
            # The publication refused, but the terminal fact is confirmed and unique. A refusal charges the
            # allowance exactly as a return does: if the call's own real cost already spent F, the overdue
            # shape is the honest report; otherwise no elapsed-deadline fact is invented to fit an error
            # shape.
            progress.memory_failed = True
            if self._real_now_ns() >= finalize_deadline_ns:
                progress.finalize_deadline_exceeded = True
                return "persisted", record_id, RuntimeReason.TRACE_FAILURE
            return "persisted", record_id, None
        if self._real_now_ns() >= finalize_deadline_ns:
            # The publication itself consumed the allowance. The fact is persisted and the overdue shape
            # is the honest report: the allowance really had elapsed.
            progress.finalize_deadline_exceeded = True
            return "persisted", record_id, RuntimeReason.TRACE_FAILURE
        return "persisted", record_id, None

    @staticmethod
    def _safe_diagnostic_attribute(refusal: BaseException, name: str) -> object:
        """Read one untrusted diagnostic attribute, treating any finite refusal to read it as absence.

        The attribute belongs to an exception the Runtime did not raise, so reading it can fail or raise in
        any way at all. A refusal that cannot be read is simply not carrying that diagnostic.
        """
        try:
            return getattr(refusal, name, None)
        except _CONTROL_FLOW:
            raise
        except BaseException:  # noqa: BLE001 - an unreadable attribute is not a diagnostic
            return None

    @staticmethod
    def _exact_declared_text(value: object, allowed: frozenset[str]) -> str | None:
        """Return the plain builtin text of a declared name, or ``None``.

        The type test is an exact identity check rather than an ``isinstance`` test, because a ``str``
        subclass can report any characters it likes to a membership test and then raise or substitute its own
        text when it is finally formatted into a message. Only a value whose type really is ``str`` and whose
        content really is one of the declared names is reused, and what is returned is a fresh plain ``str``,
        so nothing about the original object survives into the outward diagnostic.
        """
        if type(value) is not str or value not in allowed:
            return None
        return str.__str__(value)

    @staticmethod
    def _canonical_location(value: object) -> tuple[str | int, ...] | None:
        """Return a plain structural location built from exact builtin elements, or ``None``.

        Every element is copied into a plain builtin, so an element subclass cannot smuggle its own rendering
        into the location the bounded error publishes.
        """
        if type(value) is not tuple:
            return None
        try:
            length = value.__len__()
        except _CONTROL_FLOW:
            raise
        except BaseException:  # noqa: BLE001 - an unreadable structure is not a location
            return None
        if not 0 < length <= _MAX_REFUSAL_LOCATION:
            return None
        canonical: list[str | int] = []
        try:
            elements = tuple(value)
        except _CONTROL_FLOW:
            raise
        except BaseException:  # noqa: BLE001 - an unreadable structure is not a location
            return None
        for element in elements:
            if type(element) is str:
                canonical.append(str.__str__(element))
            elif type(element) is int:
                canonical.append(int.__int__(element))
            else:
                return None
        return tuple(canonical)

    def _reusable_preparation_refusal(self, refusal: BaseException) -> RuntimeError | None:
        """Return a bounded diagnosis for a refusal that really carries one, or ``None``.

        Reading the refusal is itself part of the boundary: the code, the stage, the location and the message
        are all untrusted, so every read is contained, every membership test is performed only on a value
        proven to be an exact builtin, and everything that is reused is canonicalized to a plain builtin. The
        code alone is never sufficient: it is accepted only together with a declared stage and a structural
        location of bounded length, and the message is taken only from the declared diagnostic vocabulary.
        """
        code = self._exact_declared_text(
            self._safe_diagnostic_attribute(refusal, "code"), _PREPARATION_FAILURE_CODES)
        if code is None:
            return None
        stage = self._exact_declared_text(
            self._safe_diagnostic_attribute(refusal, "stage"), _DECLARED_STAGES)
        if stage is None:
            return None
        location = self._canonical_location(self._safe_diagnostic_attribute(refusal, "location"))
        if location is None:
            return None
        message = self._exact_declared_text(
            self._safe_diagnostic_attribute(refusal, "message"), RUNTIME_DIAGNOSTICS)
        return RuntimeError(
            code, stage, location,
            message if message is not None else "an internal facility invariant was violated")

    @staticmethod
    def _unreadable_preparation_refusal() -> RuntimeError:
        """The bounded failure reported when a refusal carries no usable bounded diagnosis."""
        return RuntimeError("TRACE_FAILURE", "finalize", ("record",),
                            "an internal facility invariant was violated")

    def _bounded_preparation_refusal(self, refusal: BaseException) -> RuntimeError:
        """Normalize a finite ordinary preparation refusal onto this Runtime's bounded diagnostic contract.

        A refusal is reused only when it really satisfies the contract, as decided by
        :meth:`_reusable_preparation_refusal`. Everything else - a plain exception, a custom subclass, a
        builtin one, an unpublished proposal code, a spoofed attribute, a list-valued attribute or an
        attribute getter that refuses to be read - is replaced by a freshly constructed bounded failure. The
        message always comes from the declared diagnostic vocabulary and the cause always passes through the
        existing bounded stand-in, so no foreign text and no foreign instance is ever republished.

        The inspection is wrapped, so even a finite failure raised anywhere inside it still produces this
        segment's bounded result rather than escaping the preparation boundary that called it. Control flow is
        never converted: cancellation, an interrupt and a process exit pass straight through.
        """
        try:
            reusable = self._reusable_preparation_refusal(refusal)
        except _CONTROL_FLOW:
            raise
        except BaseException:  # noqa: BLE001 - a refusal whose shape cannot be read is not a diagnosis
            return self._unreadable_preparation_refusal()
        return reusable if reusable is not None else self._unreadable_preparation_refusal()

    def _preparation_refused(self, progress: _Progress, refusal: BaseException,
                             ) -> tuple[str, None, RuntimeReason]:
        """Report a refused terminal preparation as this segment's bounded, honest outcome.

        A preparation refusal means one thing for the Run however it was raised: there is no terminal record
        to write, so the trace was never attempted. The refusal is normalized first, so the caller receives a
        bounded diagnosis that really satisfies this Runtime's contract, or this segment's own bounded
        persistence failure when the refusal has no bounded meaning for the Run. Control-flow exceptions never
        reach here - the callers re-raise them - so a cancellation keeps its own semantics.

        The snapshot and the resource verdict are written by the finalisation from this outcome, and the
        diagnosis is raised after that snapshot exists, so neither is ever lost to the refusal.
        """
        progress.finalization_refusal = self._bounded_preparation_refusal(refusal)
        return "not_attempted", None, RuntimeReason.TRACE_FAILURE

    def _build_terminal_record(self, request: RunRequest, progress: _Progress,
                               outcome: _RunOutcome, occurred_at: datetime) -> None:
        """Construct the complete terminal record the append will write.

        The whole record - its real identifier, its sequence number, the closed terminal status and reason
        code, the verification reference and the timestamp that was really read - is constructed here,
        inside the finalization allowance and before the check that decides whether the append may start.
        That way the check tests the record the Runtime would really write, and the construction's own cost
        is part of what the allowance has to cover.
        """
        if outcome.terminal_status is RuntimeTerminalStatus.COMPLETED \
                and outcome.run_pass_record is None:
            raise RuntimeError("TRACE_FAILURE", "finalize", ("verification_record_id",),
                               "a successful terminal fact requires a recorded run-scope pass")
        trace_code = _TRACE_CODE.get(outcome.terminal_reason, "internal_error")
        if outcome.terminal_status is RuntimeTerminalStatus.COMPLETED:
            fields = {"terminal_status": "success", "reason_code": None,
                      "verification_record_id": outcome.verification_record_id}
        else:
            terminal = {
                RuntimeTerminalStatus.FAILED: "failure",
                RuntimeTerminalStatus.CANCELLED: "cancelled",
                RuntimeTerminalStatus.TIMEOUT: "timeout",
            }.get(outcome.terminal_status, "aborted")
            fields = {"terminal_status": terminal, "reason_code": trace_code,
                      "verification_record_id": None}
        progress.terminal_record = self._new_trace_record(
            request, progress, RunFinished, occurred_at=occurred_at, **fields)

    def _publish_confirmed_terminal(self, progress: _Progress, record_id: str) -> None:
        """Record the confirmed terminal identifier in the published Memory state.

        The stored state carries the identifier the Trace really assigned, so a reader of the Memory state
        and a reader of the Trace fact see the same terminal record. The write belongs to the finalization
        allowance, and a refusal here is absorbed: the terminal fact itself was already persisted, and a
        damaged Memory source must not be able to unrecord it.
        """
        state = progress.memory_state
        if state is None:
            return
        state["terminal_record_id"] = record_id
        try:
            self._memory.set(state.get("run_id", ""), state)
        except Exception as exc:  # noqa: BLE001 - a Memory refusal is a bounded facility failure
            # The publication is a retained Memory write, so its refusal is propagated explicitly rather
            # than swallowed. It does not unrecord the confirmed terminal fact: it only means the stored
            # state could not be refreshed, which is what the caller is told.
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the terminal fact could not be persisted", cause=exc) from None
        if progress.memory_failed:
            # Memory refused earlier in this same segment. The publish really happened, but the explicit
            # bounded diagnosis stands: the caller is not told the Run succeeded when its own diagnosis
            # says the terminal fact could not be persisted.
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the terminal fact could not be persisted")

    def _finalization_has_room(self, finalize_deadline_ns: int) -> bool:
        """Whether the finalization allowance still has room to start another write."""
        return self._real_now_ns() < finalize_deadline_ns

    def _read_event_time(self, *, tolerant: bool = False) -> datetime:
        """Read the Run's UTC event-time source and return the timestamp.

        ``tolerant`` is set only by the teardown, which must record the Run's end even when the timestamp
        source is damaged: a damaged source is absorbed there and an aware timestamp is substituted. The
        ordinary path never sets it, so a clock that cannot produce an aware timestamp is refused with the
        bounded ``INPUT_INVALID`` diagnosis instead of being silently accepted.
        """
        if self._clock is None:
            return datetime.now(timezone.utc)
        try:
            value = self._clock()
        except Exception as exc:
            if not tolerant:
                raise RuntimeError("INPUT_INVALID", "facility", ("clock",),
                                   "the clock did not return an aware timestamp") from exc
            return datetime.now(timezone.utc)
        if not isinstance(value, datetime) or value.tzinfo is None:
            if not tolerant:
                raise RuntimeError("INPUT_INVALID", "facility", ("clock",),
                                   "the clock did not return an aware timestamp")
            return datetime.now(timezone.utc)
        return value

    def _finalize_prepare(self, request: RunRequest, progress: _Progress,
                          outcome: _RunOutcome) -> None:
        """Write the safe Memory diagnosis and prepare the unique terminal fact.

        Both belong to finalising, so both are charged to the finalization allowance: a slow Memory
        write must not be able to leave the allowance intact for the append that follows it. The
        Memory record carries only the safe, already-bounded diagnosis and never raw observation data.
        """
        # Both Memory writes happen here, inside the same allowance: the safe diagnosis and the complete
        # published state. Writing the complete state after the terminal fact existed would put a Memory
        # write outside F where its cost could never be charged, and a slow one would be invisible to
        # the allowance that is supposed to bound finalising.
        phase_value = (outcome.deferred_terminal[0].value
                       if outcome.deferred_terminal is not None
                       else outcome.terminal_status.value)
        state = {
            "schema_version": SCHEMA_VERSION,
            "run_id": request.run_id,
            "goal_id": request.goal_id,
            "phase": phase_value,
            "terminal_status": outcome.terminal_status.value,
            "reason_code": outcome.terminal_reason.value,
            "steps_started": progress.steps_started,
            "planning_calls": progress.planning_calls,
            "attempts_started": progress.attempts_started,
            "retries_used": progress.retries_used,
            "replans_used": progress.replans_used,
            "verification_record_id": outcome.verification_record_id,
            "terminal_record_id": None,
            # Every observe call the Runtime really made is published, including a call whose
            # observation never returned. Dropping such a call would erase real call provenance;
            # the index that consumes this record distinguishes a returned observation (which has
            # a snapshot) from an unreturned call (which has none).
            "observation_calls": [
                {"ordinal": call.ordinal, "phase": call.phase,
                 "observation_id": call.observation_id,
                 "step_id": call.step_id, "attempt_id": call.attempt_id}
                for call in progress.calls if call.operation == "observe"
            ],
        }
        progress.memory_state = state
        try:
            self._memory.set(request.run_id, state)
        except Exception as exc:  # noqa: BLE001 - a Memory failure is a bounded trace failure
            # The Memory refusal is an explicit, bounded persistence failure of its own: it is marked so
            # the *terminal* record - a different fact - is still attempted once inside the remaining
            # allowance, and it is reported rather than swallowed.
            progress.memory_failed = True
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the terminal fact could not be persisted", cause=exc) from None
        if self._real_now_ns() >= self._finalize_deadline_ns:
            # The write succeeded but spent the allowance, so no further Memory or Trace write may start.
            # This is the preparation spending F, not a Memory refusal: Memory did not refuse. The spend is
            # recorded so the snapshot reports the allowance the preparation really consumed.
            progress.memory_failed = False
            progress.finalize_deadline_exceeded = True
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the local allowance elapsed before the result was accepted")
        # The complete published state is the second Memory write, and it belongs to this same allowance:
        # publishing the whole record here is what keeps the *entire* stored state inside F, and the check
        # after it records the allowance the publication really consumed.
        try:
            self._memory.set(request.run_id, state)
        except Exception as exc:  # noqa: BLE001 - a Memory failure is a bounded trace failure
            progress.memory_failed = True
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the terminal fact could not be persisted", cause=exc) from None
        if self._real_now_ns() >= self._finalize_deadline_ns:
            # The publication spent the allowance, so no further Memory or Trace write may start, and the
            # spend is recorded rather than hidden.
            progress.memory_failed = False
            progress.finalize_deadline_exceeded = True
            raise RuntimeError("TRACE_FAILURE", "finalize", ("memory",),
                               "the local allowance elapsed before the result was accepted")
        # The terminal record's own event timestamp is taken here, as the last step of preparing it: the
        # read belongs to building the record the append would write, so its real cost is charged before
        # the freshness check decides whether that append may start. A damaged clock is absorbed.
        progress.terminal_occurred_at = self._read_event_time(tolerant=True)

    async def _start_adapter(self, progress: _Progress) -> None:
        """Call the public adapter lifecycle start, before the first observation.

        The Runtime owns the lifecycle: a host is not required to start the adapter itself. A
        missing or failing start is a bounded dependency failure, and the Run still attempts its
        normal finalisation so the failure is reported rather than escaping raw. The start is bounded
        by the business deadline like every other asynchronous business entry point.
        """
        start = getattr(self._adapter, "start", None)
        if start is None:
            raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("adapter",),
                               "the adapter does not provide a start operation")
        # The boundary that decides to start re-checks the allowance before anything is created. This is a
        # decision, not a claim: the shared wait reports the actual invocation separately, so neither this
        # check nor the wait's own pre-check can leave the Run claiming a start it never entered.
        self._check_deadline(progress, "observe")
        try:
            await self._await_bounded(progress, start, "observe", RuntimePhase.OBSERVING,
                                      marks_start=True)
        except _BudgetExhausted:
            # The internal "no budget remains" signal is not a dependency failure: it is re-raised so
            # the phase loop records the abort it really is.
            raise
        except asyncio.CancelledError:
            if progress.dependency_self_cancel:
                raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("dependency",),
                                   "the dependency cancelled its own operation") from None
            raise
        except RuntimeError as exc:
            if exc.location == ("deadline",):
                raise
            raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("adapter",),
                               sanitized(exc), cause=exc) from None
        except Exception as exc:  # noqa: BLE001 - a dependency failure is a facility failure
            # An ordinary dependency exception is the facility failure the contract keeps propagating
            # once the finalisation is complete, so the mark is set here too: the rule is uniform
            # across every business boundary rather than special to the observation.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("adapter",),
                               sanitized(exc), cause=exc) from None

    async def _run_phases(self, request: RunRequest, progress: _Progress,
                          machine: RuntimeStateMachine, outcome: _RunOutcome) -> None:
        """Drive the declared phases of one Run until it reaches a terminal phase."""
        state = _PhaseState()

        while True:
            phase = machine.phase
            try:
                # The effective remaining budget is the smaller of the host declaration and the
                # monotonic deadline, checked before every new phase. A zero host declaration keeps
                # its existing abort semantics; an elapsed deadline is a timeout and is reported as
                # such, never as a facility failure or an abort.
                self._check_deadline(progress, "validate")
                await self._step_phase(request, progress, machine, outcome, phase, state)
            except _TerminalReached:
                # The Run reached a terminal conclusion. The phase move is deliberately *not* applied
                # here: it is applied once the teardown conclusion is known, so a candidate success
                # cannot become permanent before its cleanup has succeeded.
                return
            except _BudgetExhausted:
                outcome.terminal_reason = RuntimeReason.BUDGET_EXHAUSTED
                outcome.terminal_status = RuntimeTerminalStatus.ABORTED
                outcome.deferred_terminal = (RuntimePhase.ABORTED,
                                              outcome.terminal_reason)
                return
            if state.done:
                return

    async def _step_phase(self, request: RunRequest, progress: _Progress,
                    machine: RuntimeStateMachine, outcome: _RunOutcome, phase: RuntimePhase,
                    state: _PhaseState) -> None:
        """Run one phase body. ``state.done`` is set when the Run reached a terminal phase."""
        evidence = state.evidence
        observation = state.observation
        contract = state.contract
        step_result = state.step_result

        if phase is RuntimePhase.OBSERVING:
            state.observation, state.evidence = await self._observe(request, progress, phase)
            self._move(machine, progress, request, RuntimePhase.CHECKING_TASK,
                       RuntimeReason.TASK_ACHIEVED)
            return

        if phase is RuntimePhase.CHECKING_TASK:
            assert evidence is not None and observation is not None
            verdict = self._oracle_verdict(request, progress, evidence, observation)
            if verdict.status is VerificationStatus.PASS:
                # A PASS is a *business result*, and the oracle is a synchronous call whose real cost
                # is charged when it returns. A verdict that arrived after the Run's deadline must not
                # be accepted as success: the work was not finished inside the allowance the host
                # declared, so no task PASS is recorded and no success fact is written.
                self._check_deadline(progress, "check_task")
                record_id = self._record_run_verification(
                    request, progress, observation, VerificationStatus.PASS)
                outcome.run_pass_record = record_id
                outcome.verification_record_id = record_id
                outcome.terminal_reason = RuntimeReason.TASK_ACHIEVED
                outcome.terminal_status = RuntimeTerminalStatus.COMPLETED
                outcome.deferred_terminal = (RuntimePhase.COMPLETED,
                                              RuntimeReason.TASK_ACHIEVED)
                raise _TerminalReached(outcome.deferred_terminal)
                return
            if progress.remaining_steps <= 0:
                outcome.terminal_reason = RuntimeReason.MAX_STEPS_EXHAUSTED
                outcome.terminal_status = RuntimeTerminalStatus.ABORTED
                outcome.deferred_terminal = (RuntimePhase.ABORTED,
                                              outcome.terminal_reason)
                raise _TerminalReached(outcome.deferred_terminal)
                return
            if progress.remaining_planning <= 0:
                outcome.terminal_reason = RuntimeReason.MAX_PLANNING_EXHAUSTED
                outcome.terminal_status = RuntimeTerminalStatus.ABORTED
                outcome.deferred_terminal = (RuntimePhase.ABORTED,
                                              outcome.terminal_reason)
                raise _TerminalReached(outcome.deferred_terminal)
                return
            self._move(machine, progress, request, RuntimePhase.PLANNING,
                       RuntimeReason.TASK_ACHIEVED)
            return

        if phase is RuntimePhase.PLANNING:
            assert evidence is not None and observation is not None
            state.pending_forecast = await self._plan(request, progress, observation)
            self._move(machine, progress, request, RuntimePhase.BUILDING,
                       RuntimeReason.TASK_ACHIEVED)
            return

        if phase is RuntimePhase.BUILDING:
            assert evidence is not None and observation is not None
            assert state.pending_forecast is not None
            previous = progress.current_contract
            contract = self._build(request, progress, state.pending_forecast, observation,
                                   evidence, state.pending_replaces)
            replaces = (state.pending_replaces if state.pending_replaces
                        and state.pending_replaces != contract.contract_id else None)
            self._commit_contract(request, progress, contract, previous, replaces)
            state.pending_replaces = None
            state.contract = contract
            progress.current_contract = contract.contract_id
            progress.contract_retry_count = 0
            self._move(machine, progress, request, RuntimePhase.CHECKING_PRECONDITIONS,
                       RuntimeReason.TASK_ACHIEVED)
            return

        if phase is RuntimePhase.CHECKING_PRECONDITIONS:
            assert contract is not None
            # A real pre-execution observation is required after every Builder submission and
            # before every retry, even when the committed precondition list is empty: the contract
            # requires an actual new observation at this boundary, and an empty list is PASS by the
            # approved required-all rule but is still judged on this call's own evidence.
            state.observation, state.evidence = await self._observe(request, progress, phase)
            status = self._check_preconditions(contract, state.observation, state.evidence,
                                               progress)
            if status is VerificationStatus.PASS:
                self._move(machine, progress, request, RuntimePhase.EXECUTING,
                           RuntimeReason.TASK_ACHIEVED)
                return
            # A stale precondition forbids execution: no Attempt is created and no ActualOutcome is
            # invented. Replanning needs both allowances.
            if progress.remaining_replans <= 0 or progress.remaining_planning <= 0:
                outcome.terminal_reason = (RuntimeReason.MAX_REPLANS_EXHAUSTED
                                           if progress.remaining_replans <= 0
                                           else RuntimeReason.MAX_PLANNING_EXHAUSTED)
                outcome.terminal_status = RuntimeTerminalStatus.ABORTED
                outcome.deferred_terminal = (RuntimePhase.ABORTED,
                                              outcome.terminal_reason)
                raise _TerminalReached(outcome.deferred_terminal)
                return
            progress.remaining_replans -= 1
            progress.replans_used += 1
            state.pending_replaces = contract.contract_id
            self._move(machine, progress, request, RuntimePhase.PLANNING,
                       RuntimeReason.MAX_REPLANS_EXHAUSTED)
            return

        if phase is RuntimePhase.EXECUTING:
            assert contract is not None
            state.actual_outcome = await self._execute(request, progress, contract)
            state.step_result = None
            self._move(machine, progress, request, RuntimePhase.VERIFYING,
                       RuntimeReason.TASK_ACHIEVED)
            return

        if phase is RuntimePhase.VERIFYING:
            assert contract is not None and state.actual_outcome is not None
            # The step is judged on a real post-attempt observation, even when the execution's
            # effect is already unknown. A failed observation is a facility failure, not an
            # invitation to judge the step on the evidence that authorised the attempt: stale
            # evidence could report a condition that no longer holds.
            state.observation, state.evidence = await self._observe(request, progress, phase)
            state.step_result = self._verify(request, progress, contract, state.evidence,
                                             state.actual_outcome)
            if state.step_result.status is VerificationStatus.PASS:
                self._move(machine, progress, request, RuntimePhase.CHECKING_TASK,
                           RuntimeReason.TASK_ACHIEVED)
            else:
                self._move(machine, progress, request, RuntimePhase.RECOVERING,
                           RuntimeReason.TASK_ACHIEVED)
            return

        if phase is RuntimePhase.RECOVERING:
            assert contract is not None and state.actual_outcome is not None
            kind, latest = await self._recover(request, progress, contract, state.actual_outcome,
                                               evidence, step_result)
            if latest is not None:
                # The recovery-time observation and its evidence become the current ones, so the
                # task oracle judges the latest real state rather than an old snapshot.
                state.observation, state.evidence = latest
            if kind == "retry":
                self._move(machine, progress, request,
                           RuntimePhase.CHECKING_PRECONDITIONS,
                           RuntimeReason.TASK_ACHIEVED)
            elif kind == "replan":
                state.pending_replaces = contract.contract_id
                self._move(machine, progress, request, RuntimePhase.PLANNING,
                           RuntimeReason.TASK_ACHIEVED)
            elif kind == "no_recovery":
                # The verdict reports a verified step (possibly after an execution anomaly).
                # Recovery makes no retry or replan claim, so the run goes back to the task oracle:
                # a verified step is still not task success.
                self._move(machine, progress, request, RuntimePhase.CHECKING_TASK,
                           RuntimeReason.TASK_ACHIEVED)
            elif kind == "abort":
                outcome.terminal_reason = RuntimeReason.POLICY_ABORT
                outcome.terminal_status = RuntimeTerminalStatus.ABORTED
                outcome.deferred_terminal = (RuntimePhase.ABORTED,
                                              outcome.terminal_reason)
                raise _TerminalReached(outcome.deferred_terminal)
            else:  # pragma: no cover - the recovery engine only yields the kinds above
                raise RuntimeError("INPUT_INVALID", "recover", ("decision",),
                                   "the recovery decision is not a declared decision")
            return

        # pragma: no cover - every declared active phase is handled above
        raise RuntimeError("INVALID_TRANSITION", "facility", (machine.phase.value,),
                           "the run reached an unhandled phase")

    def _finish(self, request: RunRequest, progress: _Progress, machine: RuntimeStateMachine,
                terminal_status: RuntimeTerminalStatus, terminal_reason: RuntimeReason,
                verification_record_id: str | None, terminal_record_id: str) -> RunResult:
        """Build the safe result summary and publish this Run's final diagnosis."""
        result = RunResult(
            schema_version=SCHEMA_VERSION,
            run_id=request.run_id,
            goal_id=request.goal_id,
            terminal_status=terminal_status,
            reason_code=terminal_reason,
            steps_started=progress.steps_started,
            planning_calls=progress.planning_calls,
            attempts_started=progress.attempts_started,
            retries_used=progress.retries_used,
            replans_used=progress.replans_used,
            verification_record_id=verification_record_id,
            terminal_record_id=terminal_record_id,
        )
        # No Memory write happens here. The complete diagnosis was published inside the finalization
        # allowance, and a write after the terminal fact existed would be a write outside F.
        return result

    # -- individual operations ------------------------------------------------------

    async def _observe(self, request: RunRequest, progress: _Progress,
                       phase: RuntimePhase) -> tuple[Observation, VerificationEvidence]:
        """Perform one real observation and ask the domain to state its completeness.

        The returned evidence must carry this very observation's *content*, not merely its
        identifier, and the domain support receives an isolated copy, so it can neither substitute a
        same-identifier observation with different state nor rewrite the adapter's object.
        """
        self._check_deadline(progress, "observe")
        call = self._begin_call("observe", phase, progress)
        try:
            observation = await self._await_bounded(progress, self._adapter.observe, "observe", phase)
        except _BudgetExhausted:
            # The internal "no budget remains" signal is not a dependency failure: it is re-raised so
            # the phase loop records the abort it really is.
            raise
        except asyncio.CancelledError:
            if progress.dependency_self_cancel:
                # The dependency cancelled its own operation: a facility failure, never a user
                # cancellation. The flag keeps the two sources apart.
                raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("dependency",),
                                   "the dependency cancelled its own operation") from None
            raise
        except RuntimeError as exc:
            # A real elapsed bound keeps its own diagnosis so the caller reports a timeout rather
            # than a facility failure. The clause order matters: the generic handler below would
            # otherwise relabel the deadline refusal as a damaged dependency.
            if exc.location == ("deadline",):
                raise
            if exc.code == "DEPENDENCY_FAILURE":
                # The dependency failed: that is a facility failure, so it is recorded as one and
                # keeps propagating to the caller once the finalisation is complete.
                progress.facility_failure = True
                raise
            if self._facility_clock_failed:
                # The business clock itself was refused. That is a damaged facility rather than a
                # bounded business refusal, so it keeps propagating with the facility diagnosis.
                progress.translated_failure = True
                raise RuntimeError("DEPENDENCY_FAILURE", "facility", ("monotonic_clock",),
                                   "an internal facility invariant was violated") from None
            # A bounded refusal at this boundary is translated like any other facility failure there, so
            # the leading diagnosis is uniform and the cleanup fact attached to it is never replaced.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("adapter",),
                               sanitized(exc), cause=exc) from None
        except Exception as exc:  # noqa: BLE001 - a dependency failure is a facility failure
            # The adapter failed while producing the observation. That is the facility failure the
            # contract keeps propagating once the finalisation is complete, so it is marked as a
            # translated failure here and nowhere else on this boundary: a failure the *domain
            # callback* raises is published in the RunResult instead.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("adapter",),
                               sanitized(exc), cause=exc) from None
        if not isinstance(observation, Observation):
            raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("adapter",),
                               "the adapter did not return an observation record")
        # The observation this call really returned is detached into its own isolated copy, so the
        # domain support cannot mutate the object the adapter returned. Two raw type-sensitive
        # signatures are taken: the state the adapter declared, and the state of the copy after the
        # callback, so rewriting ``1`` into ``True`` or a payload value is detected.
        declared_state = self._observation_signature(observation)
        isolated = _deep_observation(observation)
        call.observation_id = observation.observation_id
        try:
            evidence = self._domain.to_evidence(isolated)
        except Exception as exc:  # noqa: BLE001
            # A domain callback that raises an ordinary exception is a facility failure too: the
            # propagation rule is uniform across every business boundary.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "observe", ("domain_support",),
                               sanitized(exc), cause=exc) from None
        if not isinstance(evidence, VerificationEvidence):
            raise RuntimeError("EVIDENCE_INVALID", "observe", ("domain_support",),
                               "the domain support did not return verification evidence")
        # The domain support must not have rewritten the value it was handed.
        if self._observation_signature(isolated) != declared_state:
            raise RuntimeError("EVIDENCE_INVALID", "observe", ("observation",),
                               "the domain support mutated the observation it was given")
        # The evidence must carry this call's observation *content*, not merely its identifier. A
        # substitute that repeats the identifier while carrying a different state is refused.
        carried = evidence.observation
        if not isinstance(carried, Observation):
            raise RuntimeError("EVIDENCE_INVALID", "observe", ("observation",),
                               "the evidence does not carry an observation record")
        if self._observation_signature(carried) != declared_state:
            raise RuntimeError("EVIDENCE_INVALID", "observe", ("observation",),
                               "the evidence does not carry the content of this observation")
        self._budget_refresh(progress, "observe")
        _ = request
        return isolated, evidence

    def _observation_signature(self, observation: Observation) -> object:
        """A raw type-sensitive signature of an observation's real content.

        The signature distinguishes ``1`` from ``True`` and ``1`` from ``1.0`` and keeps the original
        key type, so a callback that rewrites a scalar's type or a mapping's key cannot pass as if
        nothing changed. It is cycle-safe, so a pathological container is a bounded rejection rather
        than an interpreter failure.
        """
        try:
            state = detached_state(observation)
        except ValueError:
            raise RuntimeError("EVIDENCE_INVALID", "observe", ("observation",),
                               "the observation does not expose its declared fields") from None
        return _signature(state, set(), set())

    async def _plan(self, request: RunRequest, progress: _Progress,
                    observation: Observation) -> Forecast:
        """Consume one planning allowance and ask the real planner for a candidate."""
        if progress.remaining_planning <= 0:
            raise RuntimeError("INPUT_INVALID", "plan", ("max_planning_calls",),
                               "no planning allowance remains")
        # The allowance is consumed before the call, so a failed call still costs one.
        progress.remaining_planning -= 1
        progress.planning_calls += 1
        self._check_deadline(progress, "plan")
        self._begin_call("plan", RuntimePhase.PLANNING, progress)
        # The observation is a controlled argument too: it is handed over as a detached copy and
        # signed before and after the call, so a planner that rewrites what it was given is refused
        # instead of having a contaminated observation accepted as the basis of a forecast.
        observation_payload = self._call_signature(record_payload(observation))
        detached_observation = self._isolate(observation, Observation)
        try:
            forecast = await self._await_bounded(progress, lambda: self._planner.plan(detached_observation),
                                                 "plan", RuntimePhase.PLANNING)
        except _BudgetExhausted:
            # The internal "no budget remains" signal is not a dependency failure: it is re-raised so
            # the phase loop records the abort it really is.
            raise
        except asyncio.CancelledError:
            if progress.dependency_self_cancel:
                raise RuntimeError("DEPENDENCY_FAILURE", "plan", ("dependency",),
                                   "the dependency cancelled its own operation") from None
            raise
        except RuntimeError as exc:
            # The clause order matters: the generic handler below would relabel a real elapsed bound
            # as a damaged planner.
            if exc.location == ("deadline",):
                raise
            if exc.code == "DEPENDENCY_FAILURE":
                progress.facility_failure = True
                raise
            # A bounded refusal at this boundary is translated like any other facility failure there, so
            # the leading diagnosis is uniform and the cleanup fact attached to it is never replaced.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "plan", ("planner",),
                               sanitized(exc), cause=exc) from None
        except Exception as exc:  # noqa: BLE001
            if _is_plan_unavailable(exc):
                # No plan can be made for this observation: the declared script does not cover the
                # state the Run really reached. That is a bounded business outcome of this Run - the
                # Run cannot proceed, and the caller receives a failure result - rather than a damaged
                # facility, so it is published instead of propagated.
                raise RuntimeError("DEPENDENCY_FAILURE", "plan", ("planner",),
                                   "the planner cannot plan for the current observation") from None
            # Any other planner error is a facility failure, not a plan result: the ScriptedPlanner
            # contract states that a planner error is never returned to a caller as data, so it keeps
            # the bounded propagation like every other facility failure.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "plan", ("planner",),
                               sanitized(exc), cause=exc) from None
        if not self._signature_unchanged(record_payload(detached_observation), observation_payload):
            raise RuntimeError("EVIDENCE_INVALID", "plan", ("observation",),
                               "the planner mutated the observation it was given")
        if not isinstance(forecast, Forecast):
            raise RuntimeError("DEPENDENCY_FAILURE", "plan", ("planner",),
                               "the planner did not return a forecast record")
        try:
            forecast = self._isolate(forecast, Forecast)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "plan", ("forecast",),
                               "the planner returned a damaged forecast record",
                               cause=exc) from None
        self._budget_refresh(progress, "plan")
        _ = request
        return forecast

    def _build(self, request: RunRequest, progress: _Progress, forecast: Forecast,
               observation: Observation, evidence: VerificationEvidence,
               replaces: str | None) -> object:
        """Ask the domain for a candidate configuration and submit it to the real Builder.

        The Builder is the only component allowed to commit a Contract. A rejected build executes
        nothing, creates no Attempt and fabricates no committed Contract.
        """
        self._budget_refresh(progress, "build")
        self._begin_call("prepare_contract", RuntimePhase.BUILDING, progress)
        # Both controlled arguments are detached before the call and signed from their own raw
        # stored shapes, so a callback that rewrites the evidence it was handed - not only the
        # forecast - is a refusal rather than a contaminated snapshot the Runtime then accepts. The
        # signature of the Runtime's *saved* evidence is taken as well, because a callback that
        # reaches the object the Runtime still holds must be caught even though it saw only a copy.
        forecast_payload = self._call_signature(record_payload(forecast))
        evidence_payload = self._call_signature(record_payload(evidence))
        saved_evidence = self._call_signature(record_payload(evidence))
        detached_forecast = self._isolate(forecast, Forecast)
        detached_evidence = self._isolate(evidence, VerificationEvidence)

        def check_arguments_unchanged() -> None:
            """Compare the controlled arguments against their pre-call shapes immediately.

            This runs the instant the dependency returned, before any later business step: resolving
            the verification rules or the contract's own timeout necessarily rewrites these same
            objects, so a later comparison would report the Runtime's own work as a mutation and
            would miss a real one.
            """
            if not self._signature_unchanged(record_payload(detached_forecast), forecast_payload):
                raise RuntimeError("EVIDENCE_INVALID", "build", ("forecast",),
                                   "the domain support mutated the forecast it was given")
            if not self._signature_unchanged(record_payload(detached_evidence), evidence_payload):
                raise RuntimeError("EVIDENCE_INVALID", "build", ("evidence",),
                                   "the domain support mutated the evidence it was given")

        try:
            preparation = self._domain.prepare_contract(forecast=detached_forecast,
                                                        evidence=detached_evidence)
        except Exception as exc:  # noqa: BLE001
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "build", ("domain_support",),
                               sanitized(exc), cause=exc) from None
        check_arguments_unchanged()
        if not isinstance(preparation, ContractPreparation):
            raise RuntimeError("DEPENDENCY_FAILURE", "build", ("domain_support",),
                               "the domain support did not return a contract preparation")
        if not self._signature_unchanged(record_payload(detached_evidence), evidence_payload):
            raise RuntimeError("EVIDENCE_INVALID", "build", ("evidence",),
                               "the domain support mutated the evidence it was given")
        if not self._signature_unchanged(record_payload(evidence), saved_evidence):
            raise RuntimeError("EVIDENCE_INVALID", "build", ("evidence",),
                               "the domain support mutated the evidence this run holds")
        # The returned candidate is a dependency return like any other, so its own raw stored shape is
        # validated through its declared schema before the Builder sees it. A candidate carrying a
        # wrong schema version, an undeclared stored key or an ordinary tuple where a declared list
        # belongs is a facility error, and the Builder is never asked to commit it.
        try:
            preparation = self._validate_return(preparation, ContractPreparation)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "build", ("preparation",),
                               "the domain support returned a damaged contract preparation",
                               cause=exc) from None
        # The preparation's own return is charged here, before the Builder is asked to commit: a
        # preparation that crossed the Run's fixed deadline leaves no contract behind, and the
        # diagnosis is a deadline expiry rather than a damaged domain support.
        self._check_deadline(progress, "build")
        self._budget_refresh(progress, "build")
        contract_id = self._next_id("contract")
        hooks = self._contract_hooks(evidence, progress, "build")
        try:
            contract = self._builder.build(
                forecast=forecast,
                observation=observation,
                adapter=self._adapter,
                hooks=hooks,
                preconditions=list(preparation.preconditions),
                verification_policy=preparation.verification_policy,
                timeout_ms=preparation.timeout_ms,
                remaining_budget_ms=progress.budget_ms,
                contract_id=contract_id,
                goal_id=request.goal_id,
                created_at=self._now(),
                recovery_policy=preparation.recovery_policy,
            )
        except ContractBuildError as exc:
            raise RuntimeError(self._build_code(exc.code), "build", tuple(exc.location),
                               exc.message, cause=exc) from None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", "build", ("builder",),
                               sanitized(exc), cause=exc) from None
        if self._contract_hooks_factory is not None:
            # An injected hooks callback runs inside the Builder. Its real cost is charged when the
            # build returns, and a build that crossed the Run's fixed deadline commits nothing: the
            # contract is only recorded by the caller, so refusing here leaves no committed contract
            # and no execution.
            self._check_deadline(progress, "build")
        _ = replaces
        return contract

    def _check_preconditions(self, contract: object, observation: Observation,
                             evidence: VerificationEvidence | None = None,
                             progress: _Progress | None = None) -> VerificationStatus:
        """Check every committed precondition against this call's observation and evidence.

        The completeness used is the one this call's domain evidence actually declared, never a
        forced ``True``: a provider that states its observation is incomplete keeps that statement,
        so an absent path becomes INCONCLUSIVE rather than a FAIL the state does not support. The
        aggregation is the approved required-all rule: any FAIL makes the whole check FAIL, a
        FAIL-free run with an INCONCLUSIVE is INCONCLUSIVE, and an empty precondition list is PASS.

        Without an injected factory this is exactly the original path: one ``RuleValidationHooks``
        instance built from the declared completeness. With a factory, every hook call is treated
        as an injectable dependency boundary of its own - detached arguments, a before/after
        integrity check on both the copy and the saved input, an explicit verdict check and the
        Run's fixed monotonic deadline enforced after the callback returns.
        """
        state_complete = bool(evidence.state_complete) if evidence is not None else None
        if state_complete is None:
            raise RuntimeError("EVIDENCE_INVALID", "check_preconditions", ("evidence",),
                               "the preconditions need the declared evidence of this observation")
        if self._contract_hooks_factory is not None:
            return self._injected_preconditions(contract, observation, evidence, progress)
        hooks = RuleValidationHooks(state_complete=state_complete)
        statuses: list[VerificationStatus] = []
        for condition in contract.preconditions:
            try:
                statuses.append(hooks.check_precondition(condition, observation))
            except (VerificationError, ActionValidationError, ValueError, TypeError) as exc:
                raise RuntimeError("EVIDENCE_INVALID", "check_preconditions", ("preconditions",),
                                   "a committed precondition cannot be checked",
                                   cause=exc) from None
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("DEPENDENCY_FAILURE", "check_preconditions",
                                   ("preconditions",), sanitized(exc), cause=exc) from None
        if not statuses:
            return VerificationStatus.PASS
        if any(status is VerificationStatus.FAIL for status in statuses):
            return VerificationStatus.FAIL
        if any(status is VerificationStatus.INCONCLUSIVE for status in statuses):
            return VerificationStatus.INCONCLUSIVE
        return VerificationStatus.PASS

    # -- injected contract-hooks boundary ---------------------------------------------

    def _contract_hooks(self, evidence: VerificationEvidence, progress: _Progress,
                        stage: str) -> object:
        """Return the hooks for one boundary: the default JSON hooks or the factory's result.

        The factory is only consulted when one was supplied, so the original JSON path keeps its
        exact construction. The current evidence is detached and signed before the call, and both
        the detached copy and the evidence this Run still holds are checked afterwards.
        """
        if self._contract_hooks_factory is None:
            return RuleValidationHooks(state_complete=evidence.state_complete)
        return self._factory_hooks(evidence, progress, stage)

    def _factory_hooks(self, evidence: VerificationEvidence, progress: _Progress | None,
                       stage: str) -> object:
        """Create hooks through the injected factory and police the dependency boundary."""
        try:
            saved_payload = self._call_signature(record_payload(evidence))
            detached_evidence = self._isolate(evidence, VerificationEvidence)
            detached_payload = self._call_signature(record_payload(detached_evidence))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", stage, ("evidence",),
                               "the current evidence cannot be isolated for the hooks factory",
                               cause=exc) from None
        try:
            create = self._contract_hooks_factory.create
        except Exception:  # noqa: BLE001 - a dependency read failure is a facility failure
            failure = RuntimeError("DEPENDENCY_FAILURE", stage, ("contract_hooks_factory",),
                                   "the dependency did not answer as declared")
        else:
            failure = None
        if failure is not None:
            # Raised after the handler has finished, so the public error has no reachable
            # ``__context__`` (``from None`` would only hide it from the rendered traceback).
            raise failure
        if not callable(create):
            raise RuntimeError("DEPENDENCY_FAILURE", stage, ("contract_hooks_factory",),
                               "the dependency did not answer as declared")
        try:
            hooks = create(evidence=detached_evidence)
        except Exception:  # noqa: BLE001 - an ordinary callback failure is bounded here
            failure = RuntimeError("DEPENDENCY_FAILURE", stage, ("contract_hooks_factory",),
                                   "the dependency did not answer as declared")
        else:
            failure = None
        if failure is not None:
            raise failure
        # The factory must not rewrite either the copy it was given or the evidence this Run holds.
        self._check_inputs_unchanged(stage, "evidence", detached_evidence, detached_payload)
        self._check_inputs_unchanged(stage, "evidence", evidence, saved_payload)
        if inspect.isawaitable(hooks):
            self._close_awaitable(hooks)
            raise RuntimeError("DEPENDENCY_FAILURE", stage, ("contract_hooks_factory",),
                               "the contract hooks factory returned an awaitable")
        self._require_hook_methods(hooks, stage)
        # A synchronous callback cannot be pre-empted, so its real cost is charged when it returns.
        # Returning after the fixed Run deadline must not lead to a commit or an execution.
        if progress is not None:
            self._check_deadline(progress, stage)
        return hooks

    @staticmethod
    def _close_awaitable(value: object) -> None:
        """Close an unexpected coroutine so a refusal does not leak an un-awaited warning."""
        close = getattr(value, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001, S110 - the refusal is what matters
                pass

    @staticmethod
    def _require_hook_methods(hooks: object, stage: str) -> None:
        """Require the four existing synchronous hooks methods, without calling them."""
        for name in ("validate_precondition", "validate_expected_outcome",
                     "validate_verification_binding", "check_precondition"):
            try:
                method = getattr(hooks, name)
            except Exception:  # noqa: BLE001 - a dependency read failure is a facility failure
                failure = RuntimeError("DEPENDENCY_FAILURE", stage, ("contract_hooks", name),
                                       "the dependency did not answer as declared")
            else:
                failure = None
            if failure is not None:
                raise failure
            if not callable(method):
                raise RuntimeError("DEPENDENCY_FAILURE", stage, ("contract_hooks", name),
                                   "the dependency did not answer as declared")

    def _injected_preconditions(self, contract: object, observation: Observation,
                                evidence: VerificationEvidence,
                                progress: _Progress | None) -> VerificationStatus:
        """Evaluate committed preconditions through injected hooks, policing every call.

        Each call receives detached copies, and both the copies and the values this Run still holds
        are compared before and after the call. A hook that mutates its argument, raises an ordinary
        exception, returns an awaitable or returns anything but a ``VerificationStatus`` is refused
        rather than trusted. The aggregation is the same approved required-all rule.
        """
        hooks = self._factory_hooks(evidence, progress, "check_preconditions")
        statuses: list[VerificationStatus] = []
        saved_evidence_payload = self._call_signature(record_payload(evidence))
        for index, condition in enumerate(contract.preconditions):
            location = ("preconditions", index)
            try:
                condition_payload = self._call_signature(record_payload(condition))
                observation_payload = self._call_signature(record_payload(observation))
                detached_condition = self._isolate(condition, type(condition))
                detached_observation = self._isolate(observation, Observation)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("EVIDENCE_INVALID", "check_preconditions", location,
                                   "a committed precondition cannot be checked",
                                   cause=exc) from None
            try:
                method = hooks.check_precondition
            except Exception:  # noqa: BLE001
                failure = RuntimeError("DEPENDENCY_FAILURE", "check_preconditions",
                                       ("contract_hooks", "check_precondition"),
                                       "the dependency did not answer as declared")
            else:
                failure = None
            if failure is not None:
                raise failure
            if not callable(method):
                raise RuntimeError("DEPENDENCY_FAILURE", "check_preconditions",
                                   ("contract_hooks", "check_precondition"),
                                   "the dependency did not answer as declared")
            try:
                verdict = method(detached_condition, detached_observation)
            except (VerificationError, ActionValidationError, ValueError, TypeError):
                failure = RuntimeError("EVIDENCE_INVALID", "check_preconditions", location,
                                       "a committed precondition cannot be checked")
            except Exception:  # noqa: BLE001 - an ordinary callback failure is a facility failure
                failure = RuntimeError("DEPENDENCY_FAILURE", "check_preconditions", location,
                                       "the dependency did not answer as declared")
            else:
                failure = None
            if failure is not None:
                raise failure
            self._check_inputs_unchanged("check_preconditions", "condition",
                                         detached_condition, condition_payload)
            self._check_inputs_unchanged("check_preconditions", "observation",
                                         detached_observation, observation_payload)
            # The callback only ever received copies, so a change to the saved values would mean it
            # reached shared storage; both are checked rather than assuming the copy was enough.
            self._check_inputs_unchanged("check_preconditions", "condition",
                                         condition, condition_payload)
            self._check_inputs_unchanged("check_preconditions", "observation",
                                         observation, observation_payload)
            if inspect.isawaitable(verdict):
                self._close_awaitable(verdict)
                raise RuntimeError("DEPENDENCY_FAILURE", "check_preconditions", location,
                                   "the contract hook returned an awaitable")
            if not isinstance(verdict, VerificationStatus):
                raise RuntimeError("DEPENDENCY_FAILURE", "check_preconditions", location,
                                   "the contract hook did not return a VerificationStatus")
            statuses.append(verdict)
            if progress is not None:
                # The callback's real cost is charged when it returns; a precondition that only
                # became PASS after the fixed deadline must not lead to an execution.
                self._check_deadline(progress, "check_preconditions")
        self._check_inputs_unchanged("check_preconditions", "evidence",
                                     evidence, saved_evidence_payload)
        if not statuses:
            return VerificationStatus.PASS
        if any(status is VerificationStatus.FAIL for status in statuses):
            return VerificationStatus.FAIL
        if any(status is VerificationStatus.INCONCLUSIVE for status in statuses):
            return VerificationStatus.INCONCLUSIVE
        return VerificationStatus.PASS

    async def _execute(self, request: RunRequest, progress: _Progress,
                       contract: object) -> ActualOutcome:
        """Start one Attempt and really execute the committed action.

        The wait is bounded by the *smaller* of the Run's remaining effective budget and the
        committed contract's own ``timeout_ms``, both measured against the same monotonic clock. A
        timeout is not a business outcome: no ``ActualOutcome`` is invented, the Attempt keeps the
        ``AttemptStarted`` fact it really wrote, and recovery is never entered or retried
        automatically for it.
        """
        self._check_deadline(progress, "execute")
        step_id = progress.current_step
        assert step_id is not None
        attempt_id = self._next_id("attempt")
        self._trace_append(request, progress, AttemptStarted, step_id=step_id,
                           contract_id=contract.contract_id, attempt_id=attempt_id)
        progress.attempts_started += 1
        progress.current_attempt = attempt_id
        self._begin_call("execute", RuntimePhase.EXECUTING, progress)
        contract_deadline_ns = self._now_ns() + milliseconds_to_nanoseconds(
            contract.timeout_ms, ("execute", "timeout_ms"))
        try:
            outcome = await self._await_bounded(progress,
                                                lambda: self._adapter.execute(contract.action),
                                                "execute", RuntimePhase.EXECUTING,
                                                operation_ns=contract_deadline_ns)
        except _BudgetExhausted:
            # The internal "no budget remains" signal is not a dependency failure: it is re-raised so
            # the phase loop records the abort it really is.
            raise
        except asyncio.CancelledError:
            if progress.dependency_self_cancel:
                raise RuntimeError("DEPENDENCY_FAILURE", "execute", ("dependency",),
                                   "the dependency cancelled its own operation") from None
            raise
        except RuntimeError as exc:
            if exc.location == ("deadline",):
                # A real elapsed bound is a timeout, never a facility failure. It is raised as its
                # own diagnosis so the caller reports a timeout rather than a damaged dependency.
                raise RuntimeError("EVIDENCE_INVALID", "execute", ("deadline",),
                                   "the execution did not finish within its allowance") from None
            if exc.code == "DEPENDENCY_FAILURE":
                raise
            # A bounded refusal at this boundary is translated like any other facility failure there, so
            # the leading diagnosis is uniform and the cleanup fact attached to it is never replaced.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "execute", ("adapter",),
                               sanitized(exc), cause=exc) from None
        except Exception as exc:  # noqa: BLE001
            # An exception is a facility failure. No ActualOutcome is invented here and no
            # retryable receipt is fabricated: the Attempt simply has no real ending yet.
            progress.translated_failure = True
            raise RuntimeError("DEPENDENCY_FAILURE", "execute", ("adapter",),
                               sanitized(exc), cause=exc) from None
        if not isinstance(outcome, ActualOutcome):
            raise RuntimeError("DEPENDENCY_FAILURE", "execute", ("adapter",),
                               "the adapter did not return an actual outcome record")
        # The receipt is validated through its complete declared schema before any finish fact is
        # written. The record's own field validators and cross-field invariants run here, so a receipt
        # that claims success while carrying an error, or that misstates its own status, is a facility
        # failure rather than a real ending: the Attempt keeps its honest unfinished AttemptStarted
        # fact, no finish fact is fabricated and the task cannot complete.
        try:
            receipt = self._isolate(outcome, ActualOutcome, validate_fully=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "execute", ("actual_outcome",),
                               "the adapter returned a damaged actual outcome record",
                               cause=exc) from None
        self._budget_refresh(progress, "execute")
        error_code = None
        if receipt.status.value == "failed":
            error_code = "execution_failed"
        elif receipt.status.value == "unknown":
            error_code = "execution_unknown"
        self._trace_append(request, progress, AttemptFinished, step_id=step_id,
                           contract_id=contract.contract_id, attempt_id=attempt_id,
                           execution_status=receipt.status, error_code=error_code)
        progress.execution_anomaly = receipt.status.value != "succeeded"
        return receipt

    def _verify(self, request: RunRequest, progress: _Progress, contract: object,
                evidence: VerificationEvidence, outcome: ActualOutcome) -> VerificationResult:
        """Record and return the step verification, after validating the declared verdict record.

        The real verifier stays the only component that judges the step. Its return is checked as a
        declared ``VerificationResult`` whose declared fields revalidate, whose metadata names this
        Run's scope, contract, goal, observation and declared completeness, whose outcome results
        cover every expected outcome exactly once and whose aggregate agrees with them. A malformed
        or contextually foreign return is a facility failure rather than an accepted verdict.
        """
        step_id = progress.current_step
        attempt_id = progress.current_attempt
        assert step_id is not None and attempt_id is not None
        self._begin_call("verify_step", RuntimePhase.VERIFYING, progress)
        detached_contract = self._isolate(contract, type(contract))
        detached_evidence = self._isolate(evidence, VerificationEvidence)
        contract_payload = self._call_signature(record_payload(detached_contract))
        evidence_payload = self._call_signature(record_payload(detached_evidence))
        try:
            result = self._verifier.verify_step(contract=detached_contract,
                                                evidence=detached_evidence)
        except VerificationError as exc:
            raise RuntimeError("EVIDENCE_INVALID", "verify", tuple(exc.location), exc.message,
                               cause=exc) from None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", "verify", ("verifier",),
                               sanitized(exc), cause=exc) from None
        if not self._signature_unchanged(record_payload(detached_contract), contract_payload):
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("contract",),
                               "the verifier mutated the contract it was given")
        if not self._signature_unchanged(record_payload(detached_evidence), evidence_payload):
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("evidence",),
                               "the verifier mutated the evidence it was given")
        if not isinstance(result, VerificationResult):
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("result",),
                               "the step verifier did not return a declared verification result")
        try:
            type(result).model_validate(detached_state(result))
        except (ValueError, TypeError, RecursionError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("result",),
                               "the step verdict does not match its declared schema",
                               cause=exc) from None
        if result.status not in (VerificationStatus.PASS, VerificationStatus.FAIL,
                                 VerificationStatus.INCONCLUSIVE):
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("status",),
                               "the verifier returned a verdict outside the closed set")
        metadata = result.metadata
        if metadata.get("scope") != "step":
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("metadata", "scope"),
                               "the step verdict does not report the step scope")
        if metadata.get("contract_id") != contract.contract_id:
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("metadata", "contract_id"),
                               "the step verdict was made for a different contract")
        if metadata.get("goal_id") != contract.goal_id:
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("metadata", "goal_id"),
                               "the step verdict was made for a different goal")
        if metadata.get("observation_id") != evidence.observation.observation_id:
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("metadata", "observation_id"),
                               "the step verdict was made on another observation")
        if metadata.get("state_complete") is not evidence.state_complete:
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("metadata", "state_complete"),
                               "the step verdict reports a different completeness")
        required = [item.required for item in contract.expected_outcomes]
        if len(result.outcome_results) != len(required):
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("outcome_results",),
                               "the step verdict does not report every expected outcome")
        indices = [item.outcome_index for item in result.outcome_results]
        if sorted(indices) != list(range(len(required))):
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("outcome_results",),
                               "the step verdict does not report every outcome exactly once")
        if result.status is not required_all_aggregate(list(result.outcome_results), required):
            raise RuntimeError("EVIDENCE_INVALID", "verify", ("status",),
                               "the step verdict does not agree with its own outcome results")
        self._trace_append(request, progress, VerificationRecorded, scope="attempt",
                           verification_status=result.status, step_id=step_id,
                           contract_id=contract.contract_id, attempt_id=attempt_id)
        self._budget_refresh(progress, "verify")
        _ = outcome
        return result

    def _enter_recovery(self, machine: RuntimeStateMachine, progress: _Progress,
                        request: RunRequest) -> None:
        """Move into recovery for a failed or inconclusive step."""
        self._move(machine, progress, request, RuntimePhase.RECOVERING,
                   RuntimeReason.TASK_ACHIEVED)

    async def _recover(self, request: RunRequest, progress: _Progress, contract: object,
                       outcome: ActualOutcome, evidence: VerificationEvidence | None,
                       step_result: VerificationResult | None
                       ) -> tuple[str, str | None]:
        """Ask the real RecoveryEngine what to do, with at most one extra observation.

        The engine is asked first without evidence, exactly as its contract allows. Only an
        ``observation_required`` request triggers a further ``observe`` - and only one: a second
        request is a facility failure rather than an unbounded loop. The observation identifier may
        legitimately repeat; provenance is the Runtime's call ordinal, not the identifier.
        """
        step_id = progress.current_step
        attempt_id = progress.current_attempt
        assert step_id is not None and attempt_id is not None
        safety = self._declare_safety(contract, progress)
        context = RecoveryContext(
            schema_version=SCHEMA_VERSION,
            run_id=request.run_id,
            step_id=step_id,
            contract_id=contract.contract_id,
            attempt_id=attempt_id,
            actual_outcome=outcome,
            remaining_budget_ms=progress.budget_ms,
            remaining_retries=progress.remaining_retries,
            remaining_replans=progress.remaining_replans,
            contract_retry_count=progress.contract_retry_count,
            stop_requested=False,
        )
        self._begin_call("assess", RuntimePhase.RECOVERING, progress)
        assessment = self._assess(contract, context, safety, None, progress)
        latest: tuple[Observation, VerificationEvidence] | None = None
        if assessment.kind == "observation_required":
            if progress.budget_ms <= 0:
                raise RuntimeError("INPUT_INVALID", "recover", ("budget",),
                                   "no budget remains for the required observation")
            observation, fresh = await self._observe(request, progress,
                                                     RuntimePhase.RECOVERING)
            # The recovery evidence must carry a verdict judged on *this* observation, so the
            # step is re-verified against it. The decision itself is still driven by the
            # Attempt's own execution status, its outcome and the domain's effect judgement,
            # never by a re-judged verdict that could quietly turn a failed step into a
            # verified one. The completeness is this call's declared one, never a forced value.
            verdict = self._verify_recovery_step(contract, fresh, progress)
            effect = self._assess_effect(contract, outcome, fresh, progress)
            recovery_evidence = RecoveryEvidence(
                schema_version=SCHEMA_VERSION,
                run_id=request.run_id,
                step_id=step_id,
                contract_id=contract.contract_id,
                attempt_id=attempt_id,
                outcome_id=outcome.outcome_id,
                verification_evidence=fresh,
                step_result=verdict,
                effect_state=effect,
                preconditions_status=self._check_preconditions(contract, observation, fresh,
                                                               progress),
            )
            assert step_result is not None  # the failing verdict this Attempt was decided on
            self._begin_call("reassess", RuntimePhase.RECOVERING, progress)
            assessment = self._assess(contract, context, safety, recovery_evidence, progress)
            if assessment.kind == "observation_required":
                raise RuntimeError("EVIDENCE_INVALID", "recover", ("assessment",),
                                   "the recovery engine asked for a second observation")
            # The latest real observation and its evidence travel back with the decision, so
            # a subsequent task-oracle evaluation judges the newest state, not a stale one.
            latest = (observation, fresh)
        kind = assessment.kind
        if kind == "no_recovery":
            return "no_recovery", latest
        if kind != "decision":
            raise RuntimeError("INPUT_INVALID", "recover", ("assessment",),
                               "the recovery assessment kind is not declared")
        decision = assessment.decision
        if decision == "retry":
            progress.retries_used += 1
            progress.remaining_retries -= 1
            progress.contract_retry_count += 1
            return "retry", latest
        if decision == "replan":
            progress.replans_used += 1
            progress.remaining_replans -= 1
            return "replan", latest
        return "abort", latest

    def _verify_recovery_step(self, contract: object, evidence: VerificationEvidence,
                              progress: _Progress) -> VerificationResult:
        """Re-verify a step against a supplied observation, for callers that need a fresh verdict.

        The recovery path itself deliberately does not use this: it must judge the Attempt on the
        verdict that actually failed. This helper exists for a caller that wants the step re-judged
        on a *new* observation, and it records the call so the provenance stays explicit.
        """
        self._begin_call("verify_step", RuntimePhase.RECOVERING, progress)
        detached_contract = self._isolate(contract, type(contract))
        detached_evidence = self._isolate(evidence, VerificationEvidence)
        contract_payload = self._call_signature(record_payload(detached_contract))
        evidence_payload = self._call_signature(record_payload(detached_evidence))
        try:
            result = self._verifier.verify_step(contract=detached_contract,
                                                evidence=detached_evidence)
        except VerificationError as exc:
            raise RuntimeError("EVIDENCE_INVALID", "recover", tuple(exc.location), exc.message,
                               cause=exc) from None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", "recover", ("verifier",),
                               sanitized(exc), cause=exc) from None
        if not self._signature_unchanged(record_payload(detached_contract), contract_payload):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("result", "contract"),
                               "the verifier mutated the contract it was given")
        if not self._signature_unchanged(record_payload(detached_evidence), evidence_payload):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("result", "evidence"),
                               "the verifier mutated the evidence it was given")
        if not isinstance(result, VerificationResult):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("result",),
                               "the step verifier did not return a declared verification result")
        return result

    def _assess_effect(self, contract: object, outcome: ActualOutcome,
                       evidence: VerificationEvidence, progress: _Progress) -> str:
        """Ask the domain for the effect state, and refuse anything outside the closed set.

        All three controlled arguments - the contract, the real outcome and the evidence - are
        handed over as detached copies and signed before and after the call, so a callback that
        rewrites the outcome or the evidence it was given cannot have its contaminated view accepted
        as the effect judgement for this Attempt.
        """
        self._begin_call("assess_effect", RuntimePhase.RECOVERING, progress)
        contract_payload = self._call_signature(record_payload(contract))
        outcome_payload = self._call_signature(record_payload(outcome))
        evidence_payload = self._call_signature(record_payload(evidence))
        detached_contract = self._isolate(contract, type(contract))
        detached_outcome = self._isolate(outcome, ActualOutcome)
        detached_evidence = self._isolate(evidence, VerificationEvidence)
        try:
            effect = self._domain.assess_effect(contract=detached_contract,
                                                actual_outcome=detached_outcome,
                                                evidence=detached_evidence)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", "recover", ("domain_support",),
                               sanitized(exc), cause=exc) from None
        for name, value, before in (("contract", detached_contract, contract_payload),
                                    ("actual_outcome", detached_outcome, outcome_payload),
                                    ("evidence", detached_evidence, evidence_payload)):
            if not self._signature_unchanged(record_payload(value), before):
                raise RuntimeError("EVIDENCE_INVALID", "recover", ("effect_state", name),
                                   "the domain support mutated the value it was given")
        if effect not in ("not_applied", "applied", "unknown"):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("effect_state",),
                               "the domain support returned an undeclared effect state")
        return effect

    def _declare_safety(self, contract: object, progress: _Progress) -> ActionSafetyDeclaration:
        """Ask the domain for the repeat-safety declaration of the contracted action.

        The contract is a controlled argument: it is handed over as a detached copy and signed
        before and after the call, so a declaration produced from a rewritten contract cannot be
        accepted as a declaration about this one.
        """
        self._begin_call("declare_action_safety", RuntimePhase.RECOVERING, progress)
        contract_payload = self._call_signature(record_payload(contract))
        detached_contract = self._isolate(contract, type(contract))
        try:
            declaration = self._domain.declare_action_safety(contract=detached_contract)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", "recover", ("domain_support",),
                               sanitized(exc), cause=exc) from None
        if not self._signature_unchanged(record_payload(detached_contract), contract_payload):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("safety", "contract"),
                               "the domain support mutated the contract it was given")
        if not isinstance(declaration, ActionSafetyDeclaration):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("safety",),
                               "the domain support did not return a safety declaration")
        try:
            declaration = self._isolate(declaration, ActionSafetyDeclaration)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("safety",),
                               "the safety declaration does not match its declared schema",
                               cause=exc) from None
        if not action_is_declared(declaration, contract.contract_id, contract.adapter_name,
                                  contract.action):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("safety",),
                               "the safety declaration is bound to a different action")
        _ = progress
        return declaration

    def _assess(self, contract: object, context: RecoveryContext,
                safety: ActionSafetyDeclaration, evidence: RecoveryEvidence | None,
                progress: _Progress) -> RecoveryAssessment:
        """Call the already-verified recovery engine and validate both its inputs and its verdict.

        Every controlled argument - the contract, the context, the safety declaration and the
        evidence - is handed over as a detached copy whose raw structure is signed before and after
        the call, so a recovery implementation that rewrites, adds a key to or retypes anything it
        was given is refused instead of having its contaminated view accepted. The returned verdict is
        then validated through its complete declared schema, which re-runs the nested declarations,
        the closed vocabularies and the cross-field consistency rule, and it is checked against *this*
        Run, Step, Contract, Attempt, outcome and observation before its ``kind`` or ``decision`` is
        read at all. Because every refusal happens here, a malformed assessment or a mutated argument
        can never grant a retry, a replan or an observation request, and no counter is consumed and
        no next Attempt is created.
        """
        contract_payload = self._call_signature(record_payload(contract))
        context_payload = self._call_signature(record_payload(context))
        safety_payload = self._call_signature(record_payload(safety))
        evidence_payload = (self._call_signature(record_payload(evidence))
                            if evidence is not None else None)
        # Detaching the arguments is itself a validation step: the records are rebuilt through their
        # declared schemas, so a damaged record is refused here. The detachment is therefore inside
        # the protected region below, otherwise a refusal would escape as a raw engine-level error
        # instead of the Runtime's own bounded code.
        try:
            detached_contract = self._isolate(contract, type(contract))
            detached_context = self._isolate(context, RecoveryContext)
            detached_safety = self._isolate(safety, ActionSafetyDeclaration)
            detached_evidence = (self._isolate(evidence, RecoveryEvidence)
                                 if evidence is not None else None)
            assessment = self._recovery.assess(contract=detached_contract,
                                               context=detached_context,
                                               safety=detached_safety,
                                               evidence=detached_evidence)
        except RecoveryError as exc:
            # A recovery refusal - whether it came from detaching an argument or from the engine's own
            # judgement - is an evidence/input problem on the recovery surface, never a dependency
            # crash: it is translated to the Runtime's own closed ``EVIDENCE_INVALID``.
            raise RuntimeError("EVIDENCE_INVALID", "recover", tuple(exc.location), exc.message,
                               cause=exc) from None
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("recovery",),
                               "a recovery input does not match its declared schema",
                               cause=exc) from None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", "recover", ("recovery",),
                               sanitized(exc), cause=exc) from None
        # The mutation check runs before the returned value is examined, so a contaminated call is a
        # refusal rather than an answer produced from an input the recovery engine rewrote.
        for name, value, before in (("contract", detached_contract, contract_payload),
                                    ("context", detached_context, context_payload),
                                    ("safety", detached_safety, safety_payload)):
            self._check_inputs_unchanged("recover", f"recovery.{name}", value, before)
        if detached_evidence is not None:
            self._check_inputs_unchanged("recover", "recovery.evidence", detached_evidence,
                                         evidence_payload)
        if not isinstance(assessment, RecoveryAssessment):
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("assessment",),
                               "the recovery engine did not return a declared assessment")
        try:
            # ``validate_fully`` is required here rather than optional: a recovery verdict is one
            # record whose *cross-field* consistency is part of the contract, so a contradictory
            # return such as a ``no_recovery`` kind carrying a reason that demands a decision, or an
            # observation association that does not match its reason, has to be refused at ingress.
            # Per-field validation alone would accept it, and the Runtime would then act on a verdict
            # the declared schema itself excludes.
            checked = self._isolate(assessment, RecoveryAssessment, validate_fully=True)
        except RecoveryError as exc:
            # The recovery models translate their own schema refusal into their bounded error type,
            # so a contradictory verdict arrives as a RecoveryError rather than a pydantic
            # ValidationError. Both spellings mean the same thing here.
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("assessment",),
                               "the recovery assessment does not match its declared schema",
                               cause=exc) from None
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "recover", ("assessment",),
                               "the recovery assessment does not match its declared schema",
                               cause=exc) from None
        bound = (
            ("run_id", checked.run_id, context.run_id),
            ("step_id", checked.step_id, context.step_id),
            ("contract_id", checked.contract_id, context.contract_id),
            ("attempt_id", checked.attempt_id, context.attempt_id),
            ("outcome_id", checked.outcome_id, context.actual_outcome.outcome_id),
        )
        for name, reported, actual in bound:
            if reported != actual:
                raise RuntimeError("EVIDENCE_INVALID", "recover", ("assessment", name),
                                   "the recovery assessment reports another run, step, contract "
                                   "or attempt")
        expected_observation = (evidence.verification_evidence.observation.observation_id
                                if evidence is not None else None)
        if checked.observation_id != expected_observation:
            raise RuntimeError("EVIDENCE_INVALID", "recover",
                               ("assessment", "observation_id"),
                               "the recovery assessment reports another observation")
        # The kind and decision are read only from the validated record, never from the raw return.
        if checked.kind == "decision" and checked.decision is not None:
            if checked.decision == "retry" and (progress.remaining_retries <= 0
                                                or progress.current_contract
                                                != context.contract_id):
                raise RuntimeError("EVIDENCE_INVALID", "recover", ("assessment", "decision"),
                                   "the recovery assessment granted a retry without allowance")
            if checked.decision == "replan" and progress.remaining_replans <= 0:
                raise RuntimeError("EVIDENCE_INVALID", "recover", ("assessment", "decision"),
                                   "the recovery assessment granted a replan without allowance")
        return checked

    def _bounded_failure(self, exc: BaseException) -> RuntimeError:
        """Translate a raw dependency exception into the bounded surface, keeping where it happened.

        A ``ValueError`` or ``TypeError`` raised while inspecting a record is an input-shaped failure,
        and any other ordinary exception is a dependency failure - but either way the *stage* and
        *location* recorded on the original bounded failure are preserved, so a caller still learns
        which operation and which field failed instead of a generic facility message. No raw
        exception text is carried.
        """
        if isinstance(exc, RuntimeError):
            return exc
        code = "INPUT_INVALID" if isinstance(exc, (ValueError, TypeError)) else "DEPENDENCY_FAILURE"
        stage = getattr(exc, "stage", None)
        if stage not in RUNTIME_STAGES:
            stage = "validate" if code == "INPUT_INVALID" else "facility"
        location = tuple(getattr(exc, "location", ()) or ())
        message = (getattr(exc, "message", None)
                   or "the dependency raised a bounded exception")
        return RuntimeError(code, stage, location, message)

    def _check_inputs_unchanged(self, stage: str, name: str, value: object,
                                before: object) -> None:
        """Refuse a callback that changed a controlled argument, including by adding a stored key.

        Reading the argument's raw storage is itself part of the check: a callback that injects an
        undeclared stored key makes that read fail, and a failure to prove the input unchanged is a
        refusal rather than an accepted answer. Both outcomes are reported the same way, so a
        mutation is never laundered into a dependency failure.
        """
        try:
            after = self._call_signature(record_payload(value))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", stage, (name,),
                               "the dependency mutated the input it was given",
                               cause=exc) from None
        if after != before:
            raise RuntimeError("EVIDENCE_INVALID", stage, (name,),
                               "the dependency mutated the input it was given")

    def _trace_failure(self, request: RunRequest, progress: _Progress,
                       machine: RuntimeStateMachine, exc: BaseException) -> None:
        """Keep the bounded diagnosis of a failed Run, and nothing else.

        An earlier draft also wrote a full traceback to a path taken from an environment variable.
        That was an undeclared effect: the Runtime has no approved file-writing surface beyond the
        injected trace writer, and an environment-controlled path is not part of its contract. The
        bypass is removed, so a failure is reported through the bounded code, stage, message and
        sanitised location only. The parameters are kept so the call sites stay explicit about which
        Run and phase failed.
        """
        _ = (request, progress, machine, exc)

    # -- trace, budget, memory and cleanup ------------------------------------------

    def _new_trace_record(self, request: RunRequest, progress: _Progress, model: type,
                          *, occurred_at: datetime | None = None, **fields: Any) -> tuple[str, object]:
        """Construct one trace record with the next sequence number for this Run.

        The variant's ``kind`` literal is passed explicitly. The shared record envelope declares a plain
        ``kind``, so relying on the subclass default would silently construct the base kind and write a
        record of the wrong variant - a mistake the store would then reject as a missing start fact.

        Construction is separated from writing so a caller can build the record *first*, measure what
        building it really cost against its allowance, and only then decide whether to write it.
        """
        record_id = self._next_id("record")
        kind = _EVENT_KIND[model]
        record = model(schema_version=SCHEMA_VERSION, record_id=record_id,
                       run_id=request.run_id, sequence=progress.next_sequence,
                       occurred_at=self._now() if occurred_at is None else occurred_at,
                       kind=kind, **fields)
        progress.next_sequence += 1
        return record_id, record

    def _write_trace_record(self, record_id: str, record: object) -> str:
        """Write an already-built trace record, reporting a store refusal as a bounded failure."""
        try:
            self._trace.append(record)
        except TraceError as exc:
            raise RuntimeError("TRACE_FAILURE", "trace", ("append",),
                               "the trace store refused the record", cause=exc) from None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("TRACE_FAILURE", "trace", ("append",),
                               sanitized(exc), cause=exc) from None
        return record_id

    def _trace_append(self, request: RunRequest, progress: _Progress, model: type,
                      *, occurred_at: datetime | None = None, **fields: Any) -> str:
        """Construct and write one trace record in a single step."""
        record_id, record = self._new_trace_record(
            request, progress, model, occurred_at=occurred_at, **fields)
        return self._write_trace_record(record_id, record)

    def _budget_refresh(self, progress: _Progress, stage: str) -> None:
        """Revalidate the host budget in its raw declared shape, then consume it.

        The snapshot is re-derived from its own stored fields through its declared schema before its
        value is used, so a bypass construction carrying a NaN, an infinity, a boolean, a numeric
        string, an unknown field or a cycle is refused here rather than laundered through a float
        coercion downstream. The value is read at the agreed boundaries and may not increase within
        a Run, because an increasing declaration would silently undo a stop this Run already acted
        on. Nothing here measures wall-clock time.
        """
        try:
            snapshot = self._budget_source.snapshot()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", stage, ("budget_source",),
                               sanitized(exc), cause=exc) from None
        if not isinstance(snapshot, RuntimeBudgetSnapshot):
            raise RuntimeError("INPUT_INVALID", stage, ("budget",),
                               "the budget source did not return a budget snapshot record")
        try:
            value = validated_budget_value(snapshot, ("remaining_budget_ms",))
        except ValueError as exc:
            raise RuntimeError("INPUT_INVALID", stage, ("remaining_budget_ms",),
                               "the budget snapshot is not a finite non-negative number",
                               cause=exc) from None
        # The first read establishes the baseline; every later read must not exceed it.
        if (progress.calls or progress.budget_ms) and value > progress.budget_ms:
            raise RuntimeError("INPUT_INVALID", stage, ("remaining_budget_ms",),
                               "the declared budget must not increase within a run")
        progress.budget_ms = value

    def _finalize(self, built_record: tuple[str, object] | None) -> str:
        """Write the unique terminal fact that was already constructed and approved.

        The record is built before the pre-append allowance check, so this method only writes it: the fact
        is unique, it is never constructed a second time, and a store refusal keeps its own bounded
        diagnosis. The five terminal statuses map onto the existing closed trace vocabulary - a completed
        Run is ``success``, a cancellation is ``cancelled``, a deadline expiry is ``timeout``, and a
        failure is ``failure``. None of them is spelled ``abort``.
        """
        if built_record is None:
            raise RuntimeError("TRACE_FAILURE", "finalize", ("append",),
                               "the terminal fact was never constructed")
        record_id, record = built_record
        return self._write_trace_record(record_id, record)

    # -- helpers --------------------------------------------------------------------

    def _record_run_verification(self, request: RunRequest, progress: _Progress,
                                 observation: Observation,
                                 status: VerificationStatus) -> str:
        """Record the run-scoped verification fact that a successful terminal may reference.

        The append is a business result, so its own real cost is charged against the Run's fixed deadline
        before the PASS is accepted: the write really happened and is kept, but a PASS recorded only after
        the deadline elapsed is a timeout rather than a success.
        """
        record_id = self._trace_append(request, progress, VerificationRecorded, scope="run",
                                       verification_status=status)
        if progress.deadline_ns is not None and self._now_ns() >= progress.deadline_ns:
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("deadline",),
                               "the business deadline elapsed before the result was accepted")
        return record_id

    def _oracle_verdict(self, request: RunRequest, progress: _Progress,
                        evidence: VerificationEvidence,
                        observation: Observation) -> VerificationResult:
        """Ask the real task oracle for a verdict and validate the record it returns.

        The oracle is the only component that decides task success. Its return is a
        ``VerificationResult`` produced by the real implementation, and it is checked as a declared
        record rather than read through one attribute: the class identity, the declared field shape,
        the run scope, the goal and observation the verdict was made on and the required-all
        aggregate all have to agree with this Run. A fault-injected stand-in that returns a
        look-alike object cannot create a success, and a real verdict that does not correspond to
        this observation is a facility failure rather than an accepted answer.
        """
        self._begin_call("evaluate_completion", RuntimePhase.CHECKING_TASK, progress)
        criteria_payload = self._call_signature(record_payload(request.task_criteria))
        evidence_payload = self._call_signature(record_payload(evidence))
        detached_criteria = self._isolate(request.task_criteria, type(request.task_criteria))
        detached_evidence = self._isolate(evidence, VerificationEvidence)
        try:
            verdict = self._oracle.evaluate_completion(criteria=detached_criteria,
                                                       evidence=detached_evidence)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("DEPENDENCY_FAILURE", "check_task", ("oracle",),
                               sanitized(exc), cause=exc) from None
        if not self._signature_unchanged(record_payload(detached_criteria), criteria_payload):
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("task_criteria",),
                               "the task oracle mutated the criteria it was given")
        if not self._signature_unchanged(record_payload(detached_evidence), evidence_payload):
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("evidence",),
                               "the task oracle mutated the evidence it was given")
        if not isinstance(verdict, VerificationResult):
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("verdict",),
                               "the task oracle did not return a declared verification result")
        try:
            type(verdict).model_validate(detached_state(verdict))
        except (ValueError, TypeError, RecursionError) as exc:
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("verdict",),
                               "the task oracle verdict does not match its declared schema",
                               cause=exc) from None
        if verdict.status not in (VerificationStatus.PASS, VerificationStatus.FAIL,
                                  VerificationStatus.INCONCLUSIVE):
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("status",),
                               "the task oracle returned a verdict outside the closed set")
        metadata = verdict.metadata
        if metadata.get("scope") != "task":
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("metadata", "scope"),
                               "the task verdict does not report the task scope")
        if metadata.get("goal_id") != request.goal_id:
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("metadata", "goal_id"),
                               "the task verdict was made for a different goal")
        if metadata.get("observation_id") != observation.observation_id:
            raise RuntimeError("EVIDENCE_INVALID", "check_task",
                               ("metadata", "observation_id"),
                               "the task verdict was made on another observation")
        if metadata.get("state_complete") is not evidence.state_complete:
            raise RuntimeError("EVIDENCE_INVALID", "check_task",
                               ("metadata", "state_complete"),
                               "the task verdict reports a different completeness")
        # The aggregate has to agree with the required outcome results it reports.
        required = [condition.required for condition in request.task_criteria.conditions]
        if len(verdict.outcome_results) != len(required):
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("outcome_results",),
                               "the task verdict does not report every condition")
        indices = [item.outcome_index for item in verdict.outcome_results]
        if sorted(indices) != list(range(len(required))):
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("outcome_results",),
                               "the task verdict does not report every condition exactly once")
        if verdict.status is not required_all_aggregate(list(verdict.outcome_results), required):
            raise RuntimeError("EVIDENCE_INVALID", "check_task", ("status",),
                               "the task verdict does not agree with its own outcome results")
        return verdict

    def _commit_contract(self, request: RunRequest, progress: _Progress, contract: object,
                         previous: str | None, replaces: str | None) -> None:
        """Record the committed Contract, starting a new logical Step when one really begins.

        A contract that replaces an earlier one belongs to the *same* logical Step, so no new step
        is created and the run's global step allowance is not consumed again. A contract created
        without a replacement is a new logical Step: it gets its own identifier and its own trace
        fact, because the contract shares its identifier with the step's first contract otherwise
        and two successive normal steps would be indistinguishable.
        """
        if progress.current_step is None or replaces is None:
            if progress.current_step is not None:
                # A genuinely new step: the previous step's contract is deliberately not a
                # replacement of this one, so the step identity has to move on.
                progress.current_step = None
            step_id = self._next_id("step")
            progress.current_step = step_id
            progress.steps_started += 1
            progress.remaining_steps -= 1
            self._trace_append(request, progress, StepStarted, step_id=step_id)
        self._trace_append(request, progress, ContractCommitted,
                           step_id=progress.current_step,
                           contract_id=contract.contract_id,
                           replaces_contract_id=replaces)

    def _move(self, machine: RuntimeStateMachine, progress: _Progress, request: RunRequest,
              target: RuntimePhase, reason: RuntimeReason) -> None:
        """Move the machine and record the transition as local diagnostic data."""
        source = machine.phase
        machine.move_to(target)
        progress.transitions.append(RuntimeTransition(
            schema_version=SCHEMA_VERSION, sequence=len(progress.transitions),
            run_id=request.run_id, from_phase=source, to_phase=target, reason=reason))

    def _force_terminal(self, machine: RuntimeStateMachine, progress: _Progress,
                        request: RunRequest, target: RuntimePhase,
                        reason: RuntimeReason) -> None:
        """Move to a terminal phase while reporting a failure, without masking it.

        The move is best-effort by design: if the transition itself is refused, the primary failure
        is what the caller must see, so the refusal is swallowed here and nowhere else.
        """
        try:
            self._move(machine, progress, request, target, reason)
        except RuntimeError:
            _ = machine

    def _begin_call(self, operation: str, phase: RuntimePhase | None,
                    progress: _Progress | None) -> _LocalCall:
        """Register one local dependency call and return its provenance record.

        The ordinal is Runtime-local and monotonic. It is what distinguishes an execution-time from
        a recovery-time observation; the adapter's own identifier is not used as a call proof
        because it may legitimately repeat.
        """
        if progress is None:
            raise RuntimeError("INPUT_INVALID", "facility", ("call",),
                               "a local call needs the run progress record")
        progress.next_ordinal += 1
        call = _LocalCall(ordinal=progress.next_ordinal, operation=operation,
                          phase=(phase or RuntimePhase.NEW).value,
                          step_id=progress.current_step,
                          contract_id=progress.current_contract,
                          attempt_id=progress.current_attempt,
                          observation_id=None)
        progress.calls.append(call)
        return call

    def _next_id(self, kind: str) -> str:
        """An opaque identifier that contains no caller content and never collides.

        The counter is process-global and monotonic, not Run-scoped and not Engine-scoped: two
        distinct Engines whose Runs share one real trace store must not both start at
        ``rt-record-0001``. The identifier is a kind name plus a counter, so it carries no caller
        content at all, and it is generated outside any stored record, so nothing here extends the
        existing trace schema or claims a persistent restart guarantee.
        """
        return f"rt-{kind}-{_next_opaque()}"

    def _now(self) -> datetime:
        """An event timestamp. It is never used as a duration or a deadline control."""
        if self._clock is not None:
            value = self._clock()
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise RuntimeError("INPUT_INVALID", "facility", ("clock",),
                                   "the clock did not return an aware timestamp")
            return value
        return datetime.now(timezone.utc)

    def _signature(self, value: object) -> object:
        """A structural signature of a dependency argument, used to detect a mutation.

        The signature is built from the value's own declared fields and JSON containers, so a
        dependency that rewrites, adds or retypes anything it was handed is detected rather than
        having its contaminated result accepted.
        """
        return _signature(value, set(), set())

    def _call_signature(self, value: object) -> object:
        """The signature used to compare a controlled argument before and after a callback.

        The comparison is lossless: it records the real container types, every actual key and value
        and each nested record's own storage, so a callback that swaps a list for a tuple, retypes a
        scalar, adds a nested key or rewrites an opaque value is detected. Nothing is normalised or
        rebuilt first, because doing so would erase exactly the differences this check exists to see.
        """
        return _signature(value, set(), set())

    @staticmethod
    def _signature_unchanged(value: object, before: object) -> bool:
        """Whether a controlled argument still has the structure it had before the call.

        Reading the argument's storage is part of the check, so a callback that injected an
        undeclared stored key makes the read itself fail. That is reported as the same refusal as a
        changed signature: a failure to *prove* the input unchanged is not evidence that it was,
        and letting the read error escape would misreport a detected mutation as a data problem.
        """
        try:
            after = _signature(value, set(), set())
        except (ValueError, TypeError):
            return False
        return after == before

    @staticmethod
    def _isolate(record: object, declared: type, *, validate_fully: bool = False) -> object:
        """Detach a record for a controlled call, keeping its own supplied-field distinction.

        The validation covers every stored field including a defaulted one; the supplied set only
        limits what the rebuilt copy replays, so a record whose schema distinguishes an omitted field
        from an explicitly supplied one keeps that distinction in the copy the callback receives.
        """
        return isolated_record(declared, record_payload(record), validate_fully=validate_fully,
                               supplied=getattr(record, "model_fields_set", None))

    @staticmethod
    def _validate_return(record: object, declared: type) -> object:
        """Validate a dependency's returned record through its declared schema and isolate it.

        The return is read from its own storage, so an undeclared stored key is a refusal rather than
        something a copy drops, and an ordinary tuple in a strict list position is refused before any
        conversion could accept it. A legal return is rebuilt as a detached copy.
        """
        if not isinstance(record, declared):
            raise TypeError("the dependency did not return the declared record type")
        return isolated_record(declared, record_payload(record), validate_fully=True)

    @staticmethod
    def _build_code(code: str) -> str:
        """Map a Builder failure category onto the Runtime's own bounded category."""
        if code == "DEPENDENCY_FAILURE":
            return "DEPENDENCY_FAILURE"
        if code == "BUDGET_INVALID":
            return "INPUT_INVALID"
        return "EVIDENCE_INVALID"


def _signature(value: object, active: set[int], done: set[int]) -> object:
    """A comparable, cycle-safe, lossless structural signature of a value.

    The signature records what a value really *is*: each scalar keeps its own type, a mapping keeps
    its real type, its keys and its values, a sequence keeps its real type, and a record is followed
    through its own stored fields - including a field filled from a default - so nothing is dropped
    or rebuilt before the comparison. Python's ``1 == True`` and ``1 == 1.0`` would otherwise make a
    rewritten scalar indistinguishable, and a mapping key rewritten from an integer to its string
    form would collapse onto the same signature.

    An object that is neither a container nor a record contributes its real type identity. That is
    what lets a genuine opaque value be compared meaningfully while an unrelated tuple cannot pass
    itself off as the record's own immutable sequence: the two have different types, so their
    signatures differ.

    Scalar branches follow the same rule rather than emitting a type *name*: each returns the actual
    ``type`` object together with a value that is safe to compare. A name tag would let an ordinary
    subclass of ``int``, ``str`` or ``float`` carrying the same value pass as the builtin, because
    Python compares a subclass equal to its base; the real type object cannot be forged by a value.
    ``bool`` is tested before ``int`` so the two stay distinct, and a float is carried as its
    ``repr`` so a NaN does not make the signature compare unequal to itself.
    """
    if value is None:
        return ("none",)
    if isinstance(value, (bool, int, str)):
        return (type(value), value)
    if isinstance(value, float):
        return (type(value), repr(value))
    if isinstance(value, (bytes, bytearray)):
        return (type(value), bytes(value).hex())
    if isinstance(value, BaseModel):
        marker = id(value)
        if marker in active:
            return ("cycle",)
        active.add(marker)
        try:
            return (type(value),
                    tuple((name, _signature(item, active, done))
                          for name, item in stored_payload(value).items()))
        finally:
            active.discard(marker)
    items = getattr(value, "items", None)
    if callable(items):
        marker = id(value)
        if marker in active:
            return ("cycle",)
        active.add(marker)
        try:
            pairs = [(_signature(key, active, done), _signature(item, active, done))
                     for key, item in items()]
            return (type(value), tuple(sorted(pairs, key=repr)))
        finally:
            active.discard(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            return ("cycle",)
        active.add(marker)
        try:
            return (type(value),
                    tuple(_signature(item, active, done) for item in value))
        finally:
            active.discard(marker)
    return ("opaque", type(value), repr(value))


# ``_ContractBuilder`` is imported so the default component is provably the real one; the name is
# otherwise unused and is deliberately kept out of the public surface.
_ = _ContractBuilder
