"""Versioned, deterministic Fake scenario records.

Three closed records describe one reproducible environment script:

* :class:`FakeStage` - one *actual* environment state plus the observation the scenario author
  chose to expose at that point. The two are independent: an observation may be stale or may
  omit a change, and nothing copies one from the other.
* :class:`FakeStep` - one scripted action and the outcome it produces.
* :class:`FakeScenario` - the finite action catalog, the ordered stages and the ordered steps.

The records reuse the existing Core :class:`ActionSpec`, :class:`Observation` and
:class:`ActualOutcome` semantics without adding implicit coercion. Every identifier and
timestamp comes from the script: this package never reads a clock and never generates a random
identifier.

Construction re-validates the supplied record from its own stored fields, so an ordinary
``model_copy``/``model_construct`` cannot smuggle an invalid field or an undeclared key into a
scenario, and no JSON round-trip is used to launder invalid Python content into valid data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, ClassVar, Final, TypeVar

from forecontract.models.actions import ActionSpec
from forecontract.models.base import SchemaVersion
from forecontract.models.json_types import JsonObject
from forecontract.models.observations import Observation
from forecontract.models.outcomes import ActualOutcome
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from forecontract_fake.errors import FakeInputError, FakeScenarioError

__all__ = [
    "FakeScenario",
    "FakeStage",
    "FakeStep",
]

#: The one environment type a Fake scenario may declare. Internal: not a public export.
FAKE_ENVIRONMENT_TYPE: Final[str] = "fake"

#: The closed Fake-boundary mapping from a non-succeeded status to its permitted error codes.
FAILURE_CODES: Final[dict[str, tuple[str, ...]]] = {
    "failed": ("ENVIRONMENT_REJECTED", "EXECUTION_FAILED"),
    "unknown": ("RESPONSE_LOST", "TIMEOUT"),
}

NonBlankIdentifier = Annotated[str, Field(strict=True, min_length=1)]
ModelT = TypeVar("ModelT", bound=BaseModel)


def _reject_blank_identifier(value: object) -> object:
    """Reject a blank or whitespace-only identifier without rewriting the caller value (F03).

    The check inspects the value but never trims it, so a supplied whitespace string is refused
    rather than quietly normalized into a different, apparently valid identifier.
    """
    if isinstance(value, str) and not value.strip():
        raise ValueError("an identifier must not be blank")
    return value


def _bounded(
    message: str, path: tuple[str | int, ...], allowed: frozenset[str] = frozenset()
) -> str:
    """Describe a rejection with a bounded sentence and a safe structural path.

    A location segment is rendered only when it is a declared field name of the record; any
    other key becomes a placeholder, so an undeclared key can never be read back from a message
    (F01-style boundary for this fixture).
    """
    text = ""
    for part in path:
        if isinstance(part, int) and not isinstance(part, bool):
            text += f"[{part}]"
        elif isinstance(part, str) and part in allowed:
            text += ("." if text else "") + part
        else:
            text += (".<field>" if text else "<field>")
    return f"{message} at {text or '<root>'}"[:240]


def _location(exc: ValidationError) -> tuple[str | int, ...]:
    """The first reported error location of a schema rejection, or an empty tuple."""
    errors = exc.errors()
    return tuple(errors[0].get("loc") or ()) if errors else ()


def _reject_nested(record: BaseModel, error: type[Exception]) -> None:
    """Recursively reject an unknown stored key or an invalid stored field anywhere below.

    ``model_dump`` drops an attribute pydantic does not know about, so a copied nested record
    could otherwise hide an injected key inside an otherwise valid scenario. Every reachable
    model is visited through its declared fields, through lists and tuples of models, and
    through plain containers of models - which is how a Core record can sit several levels down
    inside an ``artifacts`` list or a ``payload`` object - and each one is revalidated from its
    own storage before any dump can lose the evidence (F04).
    """
    _walk_value(record, error, frozenset())


def _walk_value(value: object, error: type[Exception], visited: frozenset[int]) -> None:
    if isinstance(value, BaseModel):
        if id(value) in visited:
            return
        visited = visited | {id(value)}
        _check_stored_keys(value, error)
        for name in type(value).model_fields:
            _walk_value(getattr(value, name, None), error, visited)
        _revalidate(value, error)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk_value(item, error, visited)
        return
    if isinstance(value, dict):
        for key in value:
            if not (isinstance(key, str) and (type(key) is str or isinstance(key, str))):
                raise error("a record carries an object key that is not a string")
            _walk_value(value[key], error, visited)


def _check_stored_keys(nested: BaseModel, error: type[Exception]) -> None:
    """Reject a stored key the model does not declare, at any depth."""
    allowed = frozenset(type(nested).model_fields)
    stored = getattr(nested, "__dict__", None)
    if not isinstance(stored, dict):
        return
    for key in stored:
        if not (isinstance(key, str) and key in allowed):
            raise error("a nested record carries a stored key that is not a declared field")


def _revalidate(nested: BaseModel, error: type[Exception]) -> None:
    """Revalidate one nested model from its own storage, so no invalid field is trusted."""
    allowed = frozenset(type(nested).model_fields)
    try:
        type(nested).model_validate(nested.model_dump(mode="python", warnings=False))
    except ValidationError as exc:
        raise error(
            _bounded("a nested record does not match the declared schema", _location(exc), allowed),
            cause=exc,
        ) from exc


def validate_existing_instance(record: object, error: type[Exception]) -> None:
    """Revalidate a real record instance from its declared fields and actual stored keys (F04).

    ``model_validate(instance)`` returns the same instance after only a shallow check, so an
    object produced by ``model_copy`` or ``model_construct`` can carry an invalid stored field
    or an undeclared key and still be accepted. Public ingress therefore reads the instance
    dictionary itself, rejects an undeclared key, walks every reachable nested model and finally
    revalidates the whole record from those raw fields before any dependent validator runs.
    """
    if not isinstance(record, BaseModel):
        return
    allowed = frozenset(type(record).model_fields)
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        raise error("the supplied value is not a readable record instance")
    for key in stored:
        if not (isinstance(key, str) and key in allowed):
            raise error("the record carries a stored key that is not a declared field")
    _walk_value(record, error, frozenset())
    try:
        raw = dict(stored)
    except Exception as exc:
        raise error("the record could not be read back as declared fields", cause=exc) from exc
    try:
        type(record).model_validate(raw)
    except ValidationError as exc:
        raise error(
            _bounded("the record does not match the declared schema", _location(exc), allowed),
            cause=exc,
        ) from exc


def validated(record: object, model: type[ModelT], *, scenario: bool = False) -> ModelT:
    """Return a freshly validated, deeply isolated copy of one record.

    A real pydantic instance keeps its bookkeeping in class slots, not in ``__dict__``, so an
    ordinary instance passes while an unknown key, a non-string key or an injected
    bookkeeping-looking key is rejected. The declared raw state is then re-validated through the
    schema, which both catches an in-place nested mutation and builds brand-new nested objects,
    so the caller's own models and containers are never shared with the fixture.
    """
    error = FakeScenarioError if scenario else FakeInputError
    if not isinstance(record, model):
        raise error("the supplied value is not a validated record")
    allowed = frozenset(model.model_fields)
    stored = getattr(record, "__dict__", None)
    if not isinstance(stored, dict):
        raise error("the supplied value is not a validated record")
    for key in stored:
        if not (isinstance(key, str) and key in allowed):
            raise error("the record carries a stored key that is not a declared field")
    # The nested walk has to run before the dump, which would otherwise drop the very attribute
    # this gate exists to find.
    _reject_nested(record, error)
    try:
        raw = record.model_dump(mode="python", warnings=False)
    except Exception as exc:
        raise error("the record could not be read back as declared fields", cause=exc) from exc
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        errors = exc.errors()
        location = tuple(errors[0].get("loc") or ()) if errors else ()
        raise error(
            _bounded("the record does not match the declared schema", location,
                     frozenset(model.model_fields)),
            cause=exc,
        ) from exc


def _is_declared_list(annotation: object) -> bool:
    """Whether a declared field annotation is a list-shaped annotation."""
    if isinstance(annotation, str):
        return annotation.startswith("list[")
    return getattr(annotation, "__origin__", None) is list


def _element_types(annotation: object) -> tuple[type, ...]:
    """The declared element classes of a list annotation, ignoring non-class arguments."""
    arguments = getattr(annotation, "__args__", None) or ()
    return tuple(argument for argument in arguments if isinstance(argument, type))


def _element_fields(candidate: type | None) -> frozenset[str]:
    """The declared field names of a declared element class, when it is a model."""
    if isinstance(candidate, type) and issubclass(candidate, BaseModel):
        return frozenset(candidate.model_fields)
    return frozenset()


def _is_instance_of(value: object, candidate: object) -> bool:
    """``isinstance`` against a declared annotation that may not be a class at all."""
    return isinstance(candidate, type) and isinstance(value, candidate)


def _lint_raw(field: str, annotation: object, value: object, error: type[Exception]) -> None:
    """Reject structurally invalid raw nested content before any dependent validator runs (F04).

    A field validator that reads a nested attribute - ``outcome.status.value`` for the Fake
    error mapping, a set of identifier strings for uniqueness - runs *before* the nested model
    is revalidated, so a raw instance built by ``model_copy`` or ``model_construct`` or a plain
    mapping that never produced a valid record could otherwise surface as a raw ``AttributeError``
    or ``TypeError``.

    The declared shape is checked before anything is rebuilt. A field whose annotation is a list
    must actually carry a list, and every element must be an object the declared element class
    accepts - either an instance of that class or a plain mapping the schema will build. An
    element of the wrong model class is refused here rather than being rebuilt, which is what
    stops a record that holds itself from walking its own constructor. A field that is not
    declared as a list is passed straight to the schema and never inspected for records, so a
    legal nested JSON object and a legal shared reference stay untouched. Nothing is swallowed:
    only a damaged nested record is reported, the message stays bounded and free of supplied
    values, and a genuine internal failure keeps its cause in-process.
    """
    if not _is_declared_list(annotation):
        _lint_record(field, value, error)
        return
    if not isinstance(value, list):
        raise error(
            _bounded(f"the declared list field {field} is not a list", (field,)),
            path=(field,),
        )
    declared = _element_types(annotation)
    for item in value:
        if isinstance(item, dict):
            continue
        if declared and not any(_is_instance_of(item, candidate) for candidate in declared):
            raise error(
                _bounded(f"the declared list field {field} carries an element of the wrong type",
                         (field,), _element_fields(declared[0])),
                path=(field,),
            )
        _lint_record(field, item, error)


def _lint_record(field: str, value: object, error: type[Exception]) -> None:
    """Revalidate one nested record from its own stored fields, if it is a record at all.

    An element that is not a model instance - a plain nested mapping, a string, a number or a
    cyclic container - is left to the schema, which refuses an element the declared element type
    does not accept. The revalidated record is then walked along its declared nested models only,
    so a cycle is detected by identity and refused as this bounded facility error while a legal
    shared reference is simply visited once.
    """
    _lint_declared(field, value, error, ())


def _lint_declared(field: str, value: object, error: type[Exception],
                   seen: tuple[tuple[int, type], ...]) -> None:
    """Revalidate one nested model, then walk only the model classes its fields declare."""
    if not isinstance(value, BaseModel):
        return
    identity = (id(value), type(value))
    if identity in seen:
        raise error(
            _bounded(f"the {field} record contains a cyclic reference", (field,)),
            path=(field,),
        )
    stored = getattr(value, "__dict__", None)
    if not isinstance(stored, dict):
        return
    _revalidate_record(field, value, stored, error)
    nested_seen = seen + (identity,)
    for name, declared in type(value).model_fields.items():
        if name not in stored:
            continue
        child = getattr(value, name, None)
        _lint_nested_field(field, name, declared.annotation, child, error, nested_seen)


def _lint_nested_field(field: str, name: str, annotation: object, value: object,
                       error: type[Exception], seen: tuple[tuple[int, type], ...]) -> None:
    """Walk one declared nested model field of an already revalidated record."""
    if _is_declared_list(annotation):
        if not isinstance(value, list):
            return
        declared = _element_types(annotation)
        for item in value:
            if isinstance(item, dict):
                continue
            if declared and not any(_is_instance_of(item, candidate) for candidate in declared):
                raise error(
                    _bounded(
                        f"the declared list field {name} carries an element of the wrong type",
                        (field, name), _element_fields(declared[0])),
                    path=(field, name),
                )
            _lint_declared(field, item, error, seen)
        return
    if (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and (isinstance(value, dict) or _is_instance_of(value, annotation))
    ):
        _lint_declared(field, value, error, seen)


def _revalidate_record(field: str, value: BaseModel, stored: dict,
                       error: type[Exception]) -> None:
    """Revalidate one nested record from its own storage."""
    allowed = frozenset(type(value).model_fields)
    try:
        type(value).model_validate(dict(stored))
        return
    except ValidationError as exc:
        inner = _location(exc)
    except Exception as exc:
        raise error(
            _bounded(f"the {field} record could not be revalidated from its stored fields",
                     (field,), allowed),
            cause=exc,
        ) from exc
    detail = "".join(
        f"[{part}]" if isinstance(part, int) and not isinstance(part, bool) else f".{part}"
        for part in inner
        if isinstance(part, str) and part in allowed
    )
    message = _bounded(f"the stored {field} does not match the declared schema", (field,), allowed)
    if detail:
        message = f"{message} (rejected field {detail.lstrip('.')})"[:240]
    raise error(message, path=(field,))


class _FakeRecord(BaseModel):
    """The shared strict envelope: extra fields are rejected and no value is echoed."""

    model_config = {"extra": "forbid", "validate_assignment": True, "frozen": False,
                    "hide_input_in_errors": True}

    #: Guards the constructor's read-back so that the read-back's own rebuild terminates (F04).
    _reading_back: ClassVar[bool] = False

    __parse_raw__: ClassVar[Callable[[dict], object] | None] = None

    def __init__(self, /, **_declared: object) -> None:
        # ``self`` is positional-only, exactly as pydantic's own constructor declares it, so a
        # caller mapping that legally carries an unknown key named ``self`` is forwarded intact
        # instead of colliding with the parameter and raising a raw TypeError before the closed
        # schema can reject the key (F06). ``extra="forbid"`` then decides, so the rejection is the
        # same bounded facility error as for any other unknown field. The raw pre-check runs in
        # the model's own before validator, which every public parser path enters with the mapping
        # still whole.
        try:
            super().__init__(**_declared)
        except ValidationError as exc:
            raise FakeScenarioError(
                _bounded("the record does not match the declared schema",
                         _location(exc), frozenset(type(self).model_fields)),
                cause=exc,
            ) from None
        if type(self)._reading_back:
            return
        type(self)._reading_back = True
        try:
            self.__post_init_check()
        finally:
            type(self)._reading_back = False

    @model_validator(mode="before")
    @classmethod
    def __raw_precheck(cls, data: object) -> object:
        """Reject invalid raw content before any field validator runs, for every parser path.

        A ``mode="before"`` validator receives the caller's mapping unchanged - it is part of the
        schema rather than of the keyword-expanding constructor - so it also sees a key the
        constructor signature would otherwise collide with, and it runs before every dependent
        field validator.
        """
        if isinstance(data, dict):
            for name in cls.model_fields:
                if name in data:
                    _lint_raw(name, cls.model_fields[name].annotation, data[name],
                              FakeScenarioError)
        return data

    def __post_init_check(self) -> None:
        """Revalidate every declared field from the freshly stored values.

        The constructor is the public Python ingress for keyword arguments, and pydantic does
        not rerun a cross-field rule once a nested model instance is inserted directly, so the
        record is read back from its own storage before it is handed to the caller.
        """
        validate_existing_instance(self, FakeScenarioError)

    @classmethod
    def model_validate(cls, obj: object, **kwargs: object) -> object:
        """Wrap public Python ingress, including an existing instance, as a facility error."""
        if isinstance(obj, cls):
            validate_existing_instance(obj, FakeScenarioError)
        try:
            return super().model_validate(obj, **kwargs)
        except FakeScenarioError:
            raise
        except ValidationError as exc:
            raise FakeScenarioError(
                _bounded("the record does not match the declared schema",
                         _location(exc), frozenset(cls.model_fields)),
                cause=exc,
            ) from None

    @classmethod
    def model_validate_json(cls, json_data: object, **kwargs: object) -> object:
        """Wrap the public JSON ingress, including a document that never parsed."""
        try:
            return super().model_validate_json(json_data, **kwargs)
        except FakeScenarioError:
            raise
        except ValidationError as exc:
            raise FakeScenarioError(
                _bounded("the record does not match the declared schema",
                         _location(exc), frozenset(cls.model_fields)),
                cause=exc,
            ) from None


class FakeStage(_FakeRecord):
    """One actual environment state and the observation exposed at that point."""

    schema_version: SchemaVersion
    actual_state: JsonObject
    observation: Observation


class FakeStep(_FakeRecord):
    """One scripted action and the outcome it produces.

    Two rules beyond the Core schema are enforced here, because this is the Fake scenario
    boundary rather than the Core model (F02):

    * a ``failed`` or ``unknown`` outcome must carry an explicit ``ErrorInfo``;
    * its code must belong to the closed set that status permits - ``ENVIRONMENT_REJECTED`` or
      ``EXECUTION_FAILED`` for ``failed`` and ``RESPONSE_LOST`` or ``TIMEOUT`` for ``unknown``.

    Core itself stays wider: this package does not change :class:`ActualOutcome`, and partial
    effects on a failed or unknown step remain legal.
    """

    schema_version: SchemaVersion
    step_id: NonBlankIdentifier
    action: ActionSpec
    outcome: ActualOutcome

    _step_identifier = field_validator("step_id")(_reject_blank_identifier)

    @model_validator(mode="after")
    def _fault_mapping(self) -> FakeStep:
        status = self.outcome.status.value
        permitted = FAILURE_CODES.get(status)
        if permitted is None:
            return self
        error = self.outcome.error
        if error is None:
            raise ValueError(f"a {status} outcome must carry an explicit ErrorInfo")
        if error.code not in permitted:
            raise ValueError(f"a {status} outcome must use one of its declared error codes")
        return self


class FakeScenario(_FakeRecord):
    """A finite deterministic environment script.

    ``stages`` is exactly one longer than ``steps``, so a zero-step scenario has one stage and
    a scenario may also be exhausted. Identifiers and outcome identifiers are unique within
    the scenario, every observation declares the fake environment type, and every step action
    must belong to the finite action catalog.
    """

    schema_version: SchemaVersion
    scenario_id: NonBlankIdentifier
    adapter_name: NonBlankIdentifier
    allowed_actions: Annotated[list[ActionSpec], Field(strict=True)]
    stages: Annotated[list[FakeStage], Field(strict=True)]
    steps: Annotated[list[FakeStep], Field(strict=True)]

    _scenario_identifier = field_validator("scenario_id", "adapter_name")(_reject_blank_identifier)

    @model_validator(mode="after")
    def _structure(self) -> FakeScenario:
        if len(self.stages) != len(self.steps) + 1:
            raise ValueError("a scenario needs exactly one more stage than steps")
        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step identifiers must be unique in a scenario")
        outcome_ids = [step.outcome.outcome_id for step in self.steps]
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("outcome identifiers must be unique in a scenario")
        observation_ids = [stage.observation.observation_id for stage in self.stages]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation identifiers must be unique in a scenario")
        for index, stage in enumerate(self.stages):
            if stage.observation.environment_type != FAKE_ENVIRONMENT_TYPE:
                raise ValueError(f"stage {index} must declare the fake environment type")
        keys = [_action_key(action) for action in self.allowed_actions]
        if len(set(keys)) != len(keys):
            raise ValueError("the action catalog must not repeat an action type and payload")
        if not self.allowed_actions and self.steps:
            raise ValueError("a scenario without an action catalog cannot script a step")
        for index, step in enumerate(self.steps):
            if _action_key(step.action) not in set(keys):
                raise ValueError(f"step {index} uses an action outside the catalog")
        self._snapshots_match_stages()
        return self

    def _snapshots_match_stages(self) -> None:
        """Require a supplied receipt snapshot to equal the truth of its stage (F01).

        Step *i* moves stage *i* to stage *i + 1*, so a supplied ``state_before`` must equal the
        actual state of stage *i* and a supplied ``state_after`` must equal the actual state of
        stage *i + 1*. The comparison is the same type-aware recursive JSON equality used for
        action matching, so ``1``, ``1.0`` and ``True`` are distinct. An omitted snapshot stays
        legal: this fixture never synthesises hidden state, and it reports only the stage index,
        never the states themselves.
        """
        for index, step in enumerate(self.steps):
            before = step.outcome.state_before
            after = step.outcome.state_after
            if before is not None and not json_equal(before, self.stages[index].actual_state):
                raise ValueError(f"step {index} state_before does not match its stage")
            if after is not None and not json_equal(after, self.stages[index + 1].actual_state):
                raise ValueError(f"step {index} state_after does not match its stage")


def _action_key(action: ActionSpec) -> tuple[str, str]:
    """The catalog identity of one action: its type and its payload, never its metadata."""
    return (action.action_type, _canonical(action.payload))


def _canonical(value: object) -> str:
    """A deterministic, type-aware rendering used only as a set key, never as a message."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{value!r}"
    if isinstance(value, int):
        return f"int:{value!r}"
    if isinstance(value, float):
        return f"float:{value!r}"
    if isinstance(value, str):
        return f"str:{value!r}"
    if isinstance(value, list):
        return "list:[" + ",".join(_canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        return "dict:{" + ",".join(f"{_canonical(key)}={_canonical(value[key])}" for key in sorted(value, key=str)) + "}"
    return f"other:{type(value).__name__}"


def json_equal(left: object, right: object) -> bool:
    """Type-aware recursive JSON equality.

    ``True`` is not ``1`` and ``1`` is not ``1.0``. Object key order is irrelevant and array
    order is significant. No wildcard, subset, conditional or callback matching exists.
    """
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, bool) and isinstance(right, bool):
        return left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, float) and isinstance(right, float):
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_equal(item, other) for item, other in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(json_equal(left[key], right[key]) for key in left)
    return False
