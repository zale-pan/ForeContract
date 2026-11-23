"""Versioned, deterministic planner script records.

Two closed records describe one reproducible planning script:

* :class:`PlannerStep` - the environment type and complete state the planner expects to see,
  plus the :class:`Forecast` it returns for that observation.
* :class:`PlannerScript` - the ordered steps of one named script.

Matching compares the expected environment type and the complete expected state only;
``observation_id``, ``timestamp``, ``summary``, ``artifacts`` and ``metadata`` are ignored. A
returned forecast only has to satisfy the existing Core structure: an unsupported action, an
empty ``expected_outcomes`` list, a wrong expectation and a ``task_completion_hint=True`` are
all legitimately producible, because whether a candidate may be committed is a Contract Builder
decision and never this fixture's.

Construction re-validates the supplied record from its own stored fields, so an ordinary
``model_copy``/``model_construct`` cannot smuggle an invalid field or an undeclared key into a
script, and no JSON round-trip launders invalid Python content into valid data.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, TypeVar

from forecontract.models.actions import Forecast
from forecontract.models.base import SchemaVersion
from forecontract.models.json_types import JsonObject
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from forecontract_scripted.errors import PlannerInputError, PlannerScriptError

__all__ = ["PlannerScript", "PlannerStep"]

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
    other key becomes a placeholder, so an undeclared key can never be read back from a message.
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
    could otherwise hide an injected key inside an otherwise valid script. Every reachable model
    is visited through its declared fields, through lists and tuples of models, and through
    plain containers of models - which is how a Core record can sit several levels down inside a
    forecast - and each one is revalidated from its own storage before any dump can lose the
    evidence (F04).
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
            if not isinstance(key, str):
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
    """Revalidate one nested model from its own storage, so no invalid field is trusted.

    A nested record that this same package declares reports its own facility error; a nested
    Core record reports the error of the boundary that is validating it.
    """
    allowed = frozenset(type(nested).model_fields)
    raised: type[Exception] = error
    if isinstance(nested, _ScriptRecord):
        raised = PlannerScriptError
    try:
        type(nested).model_validate(nested.model_dump(mode="python", warnings=False))
    except ValidationError as exc:
        raise raised(
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


def validated(record: object, model: type[ModelT], *, script: bool = False) -> ModelT:
    """Return a freshly validated, deeply isolated copy of one record.

    A real pydantic instance keeps its bookkeeping in class slots, not in ``__dict__``, so an
    ordinary instance passes while an unknown key, a non-string key or an injected
    bookkeeping-looking key is rejected. The declared raw state is then re-validated through the
    schema, which both catches an in-place nested mutation and builds brand-new nested objects,
    so the caller's own models and containers are never shared with the fixture.
    """
    error = PlannerScriptError if script else PlannerInputError
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
        raise error(
            _bounded("the record does not match the declared schema", _location(exc),
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

    A dependent validator that reads a nested attribute or a set of nested identifiers runs
    before the nested model is revalidated, so a raw instance built by ``model_copy`` or
    ``model_construct`` could otherwise surface as a raw ``AttributeError`` or ``TypeError``.

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


class _ScriptRecord(BaseModel):
    """The shared strict envelope: extra fields are rejected and no value is echoed."""

    model_config = {"extra": "forbid", "validate_assignment": True, "frozen": False,
                    "hide_input_in_errors": True}

    #: Guards the constructor's read-back so that the read-back's own rebuild terminates (F04).
    _reading_back: ClassVar[bool] = False

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
            raise PlannerScriptError(
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
                              PlannerScriptError)
        return data

    def __post_init_check(self) -> None:
        """Revalidate every declared field from the freshly stored values.

        The constructor is the public Python ingress for keyword arguments, and pydantic does
        not rerun a cross-field rule once a nested model instance is inserted directly, so the
        record is read back from its own storage before it is handed to the caller.
        """
        validate_existing_instance(self, PlannerScriptError)

    @classmethod
    def model_validate(cls, obj: object, **kwargs: object) -> object:
        """Wrap public Python ingress, including an existing instance, as a facility error."""
        if isinstance(obj, cls):
            validate_existing_instance(obj, PlannerScriptError)
        try:
            return super().model_validate(obj, **kwargs)
        except PlannerScriptError:
            raise
        except ValidationError as exc:
            raise PlannerScriptError(
                _bounded("the record does not match the declared schema",
                         _location(exc), frozenset(cls.model_fields)),
                cause=exc,
            ) from None

    @classmethod
    def model_validate_json(cls, json_data: object, **kwargs: object) -> object:
        """Wrap the public JSON ingress, including a document that never parsed."""
        try:
            return super().model_validate_json(json_data, **kwargs)
        except PlannerScriptError:
            raise
        except ValidationError as exc:
            raise PlannerScriptError(
                _bounded("the record does not match the declared schema",
                         _location(exc), frozenset(cls.model_fields)),
                cause=exc,
            ) from None


class PlannerStep(_ScriptRecord):
    """One expected observation and the forecast returned for it."""

    schema_version: SchemaVersion
    step_id: NonBlankIdentifier
    expected_environment_type: NonBlankIdentifier
    expected_state: JsonObject
    forecast: Forecast

    _step_identifier = field_validator("step_id", "expected_environment_type")(
        _reject_blank_identifier
    )


class PlannerScript(_ScriptRecord):
    """A finite deterministic planning script.

    ``steps`` may be empty, which makes the planner immediately exhausted. Step identifiers and
    forecast identifiers are unique within one script.
    """

    schema_version: SchemaVersion
    script_id: NonBlankIdentifier
    steps: Annotated[list[PlannerStep], Field(strict=True)]

    _script_identifier = field_validator("script_id")(_reject_blank_identifier)

    @model_validator(mode="after")
    def _unique_identifiers(self) -> PlannerScript:
        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step identifiers must be unique in a script")
        forecast_ids = [step.forecast.forecast_id for step in self.steps]
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("forecast identifiers must be unique in a script")
        return self
