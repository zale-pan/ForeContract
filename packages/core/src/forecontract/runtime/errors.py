"""Bounded, value-free facility errors for the sequential Runtime.

A Runtime failure is reported as a closed :data:`RUNTIME_ERROR_CODES` category, the declared
:data:`RUNTIME_STAGES` stage it happened in and a structural location. The rendered message is
deliberately short and never transcribes a rejected value, an arbitrary exception text, a repr or a
secret: the whole point of the Runtime boundary is that a facility failure is diagnosable without
becoming a channel that echoes caller input back out.

A correctly shaped business failure (a failed step, an inconclusive verification, a recovery
decision) is not a Runtime error at all. It travels as a normal result; only damage to the facility
is raised.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "RUNTIME_DIAGNOSTICS",
    "RUNTIME_ERROR_CODES",
    "RUNTIME_STAGES",
    "RuntimeError",
    "sanitised_location",
]

#: The closed set of Runtime failure categories. No free-text category is accepted.
RUNTIME_ERROR_CODES: Final[tuple[str, ...]] = (
    "INPUT_INVALID",
    "INVALID_TRANSITION",
    "DEPENDENCY_FAILURE",
    "EVIDENCE_INVALID",
    "IDENTITY_CONFLICT",
    "TRACE_FAILURE",
    "CLEANUP_FAILURE",
)

#: The declared pipeline stages a failure can be attributed to. ``facility`` covers work that is
#: not one of the numbered pipeline steps, such as finalisation or cleanup.
RUNTIME_STAGES: Final[tuple[str, ...]] = (
    "facility",
    "validate",
    "observe",
    "check_task",
    "plan",
    "build",
    "check_preconditions",
    "execute",
    "verify",
    "recover",
    "finalize",
    "cleanup",
    "trace",
)

#: A location part is a declared field name or a list index, never an arbitrary key. The allowlist
#: is the union of the declared field names of every Runtime record, so a caller-supplied key can
#: never be rendered: it is not a declared name and therefore becomes the fixed placeholder.
_MAX_MESSAGE = 240

#: The longest public diagnostic text the Runtime ever exposes. Every public attribute is bounded,
#: not only the rendered prefix, because ``.message`` is part of the surface a consumer may print.
_MAX_DIAGNOSTIC = 120

#: The longest structural location the Runtime ever stores. A location describes *where* in a
#: declared shape a failure happened; it is a diagnostic, not a data channel, so a caller that
#: supplies an enormous segment list cannot make the stored attribute unbounded. Only the rendered
#: string being truncated would leave the attribute itself as an unbounded public surface.
_MAX_LOCATION = 64

#: The largest index the Runtime renders as an index. A huge integer is still an index-shaped value
#: rather than caller text, but storing it verbatim would let an arbitrary magnitude leak through the
#: public attribute, so anything beyond this is reported as an out-of-range index.
_MAX_INDEX = 1_000_000

#: The closed set of public diagnostic texts. A diagnostic describes the *kind* of facility damage
#: in the Runtime's own words. Caller text, arbitrary exception text and dynamic class names are
#: never part of it, so an unrecognised message is replaced rather than echoed.
RUNTIME_DIAGNOSTICS: Final[tuple[str, ...]] = (
    "the request is not a RunRequest record",
    "the request does not match its declared schema",
    "the request carries an undeclared nested field or shape",
    "the task criteria are not the declared record type",
    "the task criteria belong to a different goal",
    "the task criteria could not be detached",
    "the task criteria are not a usable json_rule definition",
    "the record does not expose its stored fields",
    "the record carries a stored key that is not a declared field",
    "a declared field is missing from the stored record",
    "the record does not match its declared schema",
    "the candidate preparation is not the declared record type",
    "the candidate preparation does not match its declared schema",
    "the dependency raised a bounded exception",
    "the dependency did not answer as declared",
    "the dependency mutated the input it was given",
    "the dependency changed the value it was handed",
    "the dependency asked for a second observation",
    "the dependency returned an undeclared record",
    "the dependency returned a record for another run",
    "the declared budget is not a finite non-negative number",
    "the declared budget must not increase within a run",
    "the declared budget is exhausted",
    "the adapter did not return an observation record",
    "the adapter did not return an actual outcome record",
    "the adapter did not return a forecast record",
    "the adapter does not provide a start operation",
    "the adapter does not provide a stop operation",
    "the adapter start operation failed",
    "the adapter stop operation failed",
    "the evidence does not carry an observation record",
    "the evidence does not carry the content of this observation",
    "the evidence does not carry the declared completeness of this observation",
    "the domain support did not return verification evidence",
    "the domain support declared an undeclared effect state",
    "the verifier did not return a declared verification result",
    "the task oracle did not return a declared verification result",
    "the recovery engine did not return a declared assessment",
    "the recovery assessment is not consistent with this run",
    "the recovery assessment reports another run, step, contract or attempt",
    "the recovery assessment reports another outcome or observation",
    "the recovery decision is not a declared decision",
    "the recovery assessment kind is not declared",
    "the recovery engine asked for a second observation",
    "the step verdict does not match its declared schema",
    "the step verdict does not agree with its own outcome results",
    "the task verdict does not match its declared schema",
    "the task verdict does not agree with its own outcome results",
    "the verifier returned a verdict outside the closed set",
    "the task oracle returned a verdict outside the closed set",
    "the step verdict does not report the step scope",
    "the task verdict does not report the task scope",
    "the precondition evaluation was refused",
    "the recovery evaluation was refused",
    "the builder refused the candidate configuration",
    "the trace writer refused the record",
    "a successful terminal fact requires a recorded run-scope pass",
    "the run reached an unhandled phase",
    "the transition is not legal",
    "the run identifier is already started",
    "the engine is already running",
    "no planning allowance remains",
    "no budget remains for the required observation",
    "the clock did not return an aware timestamp",
    "the budget source did not return a budget snapshot record",
    # P2-T07 round-002: the bounded diagnostics the correction introduced. Each is a fixed statement
    # about the Runtime's own decision, so the vocabulary stays closed to caller text.
    "the monotonic clock reading went backwards",
    "the dependency cancelled its own operation",
    "the operation did not finish within its allowance",
    "the local allowance elapsed before the result was accepted",
    "the business deadline elapsed before the result was accepted",
    "the domain support mutated the forecast it was given",
    "the domain support mutated the evidence it was given",
    "the current operation did not confirm it stopped",
    "the cleanup allowance elapsed before the stop",
    "the adapter stop operation did not finish in time",
    "the adapter resource is occupied by an unsettled Run",
    "this engine is holding a quarantined resource",
    "the terminal fact could not be persisted",
    "an internal facility invariant was violated",
)


def _bounded_diagnostic(message: object) -> str:
    """Map a raw message onto the closed, bounded, value-free diagnostic vocabulary.

    A diagnostic is a public surface, so it must not become a channel that echoes a caller value, an
    arbitrary dependency message or a dynamic class name. A message the Runtime itself declared is
    kept; anything else - including a long synthetic caller marker - is replaced by one fixed
    bounded text, so the length and the vocabulary of the whole surface stay closed.
    """
    if isinstance(message, str) and message in RUNTIME_DIAGNOSTICS:
        return message
    return "an internal facility invariant was violated"


def _declared_location_names() -> frozenset[str]:
    """The declared field names the diagnostic allowlist accepts.

    The set is the union of the declared field names of the Runtime records and of the Core records
    the Runtime handles, so a nested location such as a contract's expected-outcome field stays
    readable while an arbitrary caller key can never be rendered. It is derived from the models
    themselves rather than maintained by hand, so it cannot drift from the schema. The import is
    local to keep this module free of an import cycle: the models module raises this error type.
    """
    from pydantic import BaseModel

    from forecontract.contracts.models import ActionContract
    from forecontract.models.actions import Forecast, Precondition
    from forecontract.models.observations import Observation
    from forecontract.models.outcomes import ActualOutcome, VerificationResult
    from forecontract.runtime import models as runtime_models
    from forecontract.verification.models import (
        TaskCompletionCriteria,
        VerificationEvidence,
    )

    roots = (
        ActionContract, ActualOutcome, Forecast, Observation, Precondition,
        TaskCompletionCriteria, VerificationEvidence, VerificationResult,
    )
    names: set[str] = set()
    for root in roots:
        names.update(root.model_fields)
    for value in vars(runtime_models).values():
        if isinstance(value, type) and issubclass(value, runtime_models.RuntimeRecord):
            names.update(value.model_fields)
    # The nested structures the Runtime itself builds are named by the models above; the base
    # record names are added so a nested runtime record field is readable too. ``deadline``,
    # ``dependency`` and ``memory`` are the Runtime's own fixed structural labels - the time bound, a
    # dependency that failed or cancelled its own operation, and the safe-diagnosis write during
    # finalisation. None is a field of any record: adding them keeps those refusals distinguishable by
    # location while leaving the allowlist closed to anything a caller supplies.
    names.update({"record", "field", "value", "deadline", "dependency", "memory"})
    _ = BaseModel
    return frozenset(names)


def _format_location(location: tuple[object, ...]) -> str:
    """Render a structural location from declared names and indices only.

    A segment that is a declared Runtime record field or an index is kept because it is useful and
    cannot carry caller content. Anything else - including an arbitrary identifier that happens to
    look like a name - becomes a fixed placeholder, so no caller-supplied key can be echoed back
    through a diagnostic.
    """
    return _render(sanitised_location(location))


def _render(location: tuple[object, ...]) -> str:
    """Render an already-normalised location."""
    parts: list[str] = []
    for part in location:
        if isinstance(part, bool):
            parts.append("<value>")
        elif isinstance(part, int):
            parts.append(f"[{part}]")
        elif isinstance(part, str):
            parts.append(("." if parts else "") + part)
        else:
            parts.append(".<field>" if parts else "<record>")
    return "".join(parts) or "<root>"


def sanitised_location(location: object) -> tuple[object, ...]:
    """Normalise a raw location onto declared Runtime field names, bounded in length and magnitude.

    This is the single redaction point for the whole diagnostic surface. The stored ``location``
    attribute is normalised as well as the rendered message, because a public attribute is part of
    the diagnostic surface: leaving a caller-supplied key in the tuple would expose it to any
    consumer that formats the attribute itself. A declared field name or a canonical index is kept
    (it is useful and cannot carry caller content); anything else is replaced by a fixed placeholder.

    The tuple is also *structurally* bounded. Truncating only the rendered string would still leave
    an arbitrarily long tuple reachable through the public attribute, and an arbitrarily large
    integer index would leak its magnitude the same way, so both are limited here rather than at
    print time.
    """
    allowed = _declared_location_names()
    parts = list(location) if isinstance(location, (tuple, list)) else [location]
    truncated = len(parts) > _MAX_LOCATION
    # The elision marker is one of the segments, so the tuple is bounded *including* it: a public
    # attribute that grows to say it was truncated would not be bounded at all.
    kept = parts[:_MAX_LOCATION - 1] if truncated else parts
    safe: list[object] = []
    for part in kept:
        if isinstance(part, bool):
            # A boolean is not an index and not a declared name: it is rendered as a value.
            safe.append("<value>")
        elif isinstance(part, int):
            safe.append(part if -_MAX_INDEX <= part <= _MAX_INDEX else "<index>")
        elif isinstance(part, str) and part in allowed:
            safe.append(part)
        else:
            safe.append("<field>")
    if truncated:
        # The marker is itself a fixed, declared value-free segment, so a truncated location is
        # visibly truncated instead of silently claiming to be the complete path.
        safe.append("<truncated>")
    return tuple(safe)


class RuntimeError(Exception):
    """A bounded Runtime facility failure.

    ``code`` must be one of :data:`RUNTIME_ERROR_CODES` and ``stage`` one of
    :data:`RUNTIME_STAGES`; an undocumented name is a programming error and is refused loudly so a
    typo cannot silently widen the closed vocabulary.
    """

    def __init__(self, code: str, stage: str, location: tuple[str | int, ...],
                 message: str, *, cause: BaseException | None = None) -> None:
        if code not in RUNTIME_ERROR_CODES:
            raise ValueError("an unknown runtime error code is not allowed")
        if stage not in RUNTIME_STAGES:
            raise ValueError("an unknown runtime stage is not allowed")
        self.code = code
        self.stage = stage
        self.location = sanitised_location(location)
        # Every public attribute is bounded and value-free, not only the rendered prefix: a consumer
        # that prints ``.message`` directly must not become the place where caller text escapes.
        self.message = _bounded_diagnostic(message)
        rendered = f"{code} at {stage}:{_render(self.location)}: {self.message}"
        super().__init__(rendered[:_MAX_MESSAGE])
        #: The bounded secondary cleanup fact, set only by :meth:`with_cleanup_failure` when a Run
        #: really failed to stop. It is deliberately *not* derived from ``cause``: an attached cause
        #: is the primary chain, and labelling any bounded cause as a cleanup failure would both
        #: claim a cleanup problem that never happened and occupy the slot that the real later stop
        #: failure has to fill. The two facts stay distinct.
        self.cleanup_failure: RuntimeError | None = None
        if cause is not None:
            self.__cause__ = _bounded_cause(cause)

    def with_cleanup_failure(self, cleanup: RuntimeError) -> RuntimeError:
        """Attach a bounded secondary cleanup fact to the leading diagnosis and return it.

        The leading failure keeps its own code, stage, location and message: a cleanup problem is
        never promoted over an earlier, more important reason. It is recorded here rather than
        discarded, and it is recorded on the approved diagnostic surface - a declared attribute of
        the bounded error - so no new trace variant is introduced and no raw exception text is
        involved.
        """
        if self.cleanup_failure is None:
            self.cleanup_failure = cleanup
        return self


def _bounded_cause(cause: BaseException) -> BaseException:
    """A bounded stand-in for an attached cause, so no arbitrary text is reachable.

    ``__cause__`` is a public attribute and Python renders it in a traceback. Attaching the original
    dependency exception would therefore re-expose exactly the arbitrary message this module exists
    to withhold, so only a bounded Runtime error is attached verbatim; anything else is replaced by
    a fixed value-free Runtime error that names nothing but the fact of a dependency failure.
    """
    if isinstance(cause, RuntimeError):
        return cause
    if isinstance(cause, (ValueError, TypeError)):
        code, stage = "INPUT_INVALID", "validate"
    else:
        code, stage = "DEPENDENCY_FAILURE", "facility"
    return RuntimeError(code, stage, (),
                        "the dependency raised a bounded exception")


def sanitized(cause: BaseException) -> str:
    """A bounded, value-free description of a dependency failure.

    The dependency's class name is *not* used. A dynamic class name is caller-controlled text - a
    class can be created with any name at all, including a payload - so echoing it would reopen the
    channel this module closes. The returned text is one fixed declared diagnostic.
    """
    _ = cause
    return "the dependency raised a bounded exception"
