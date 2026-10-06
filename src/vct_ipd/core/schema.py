"""Column definitions and declarative constraints; no row evaluator."""

from typing import Self

from pydantic import Field, StrictBool, model_validator

from .base import AnnotatedContract, Artifact, Condition, Contract, NonNegativeInt, Number, NumericRange, Reference, Scalar, Text, unique
from .enums import MissingnessPolicy, VariableRole, VariableType


class Constraint(AnnotatedContract):
    description: Text
    variables: list[Text] = Field(min_length=1)
    expression: Text | None = None  # Documentation only; never eval/exec this string.


def same_value(a: Scalar, b: Scalar) -> bool:
    """Numeric equality is allowed, but Boolean categories are not integer codes."""
    return a == b and (type(a) is bool) == (type(b) is bool)


class CategoryEncoding(Contract):
    value: Scalar
    score: Number


class VariableSpec(AnnotatedContract):
    name: Text
    label: Text
    type: VariableType
    role: VariableRole
    unit: Text | None = None
    allowed_values: list[Scalar] | None = Field(default=None, min_length=1)
    range: NumericRange | None = None
    required: StrictBool = True
    missingness: MissingnessPolicy = MissingnessPolicy.FORBIDDEN
    generation_order: NonNegativeInt
    derived_from: list[Text] = Field(default_factory=list)
    derivation: Text | None = None
    constraints: list[Constraint] = Field(default_factory=list)
    numeric_encoding: list[CategoryEncoding] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        unique(self.derived_from, "derived_from")
        if self.name in self.derived_from:
            raise ValueError("a variable cannot derive from itself")
        if self.derived_from and not self.derivation:
            raise ValueError("derived variables need a derivation description")
        if self.range and self.type not in (VariableType.CONTINUOUS, VariableType.INTEGER):
            raise ValueError("numeric ranges require numeric variables")
        if self.allowed_values is not None:
            unique(self.allowed_values, "allowed values")
            for value in self.allowed_values:
                valid = {
                    VariableType.CONTINUOUS: type(value) in (int, float),
                    VariableType.INTEGER: type(value) is int,
                    VariableType.BOOLEAN: type(value) is bool,
                    VariableType.STRING: type(value) is str,
                    VariableType.CATEGORICAL: True,
                    VariableType.ORDINAL: True,
                }[self.type]
                if not valid:
                    raise ValueError("allowed value does not match variable type")
                if self.range and ((self.range.lower is not None and value < self.range.lower)
                                   or (self.range.upper is not None and value > self.range.upper)):
                    raise ValueError("allowed value outside range")
        if self.numeric_encoding is not None:
            if self.type not in (VariableType.CATEGORICAL, VariableType.ORDINAL, VariableType.STRING):
                raise ValueError("explicit encoding is for categorical, ordinal or string columns")
            if self.allowed_values is None:
                raise ValueError("an encoding needs a declared finite category domain")
            values = [entry.value for entry in self.numeric_encoding]
            unique(values, "encoded categories")
            if len(values) != len(self.allowed_values) or any(
                not any(same_value(value, allowed) for allowed in self.allowed_values) for value in values
            ):
                raise ValueError("encoding must cover exactly the declared categories")
        return self

    @property
    def supports_numeric_summary(self) -> bool:
        # Boolean summaries use the explicit built-in convention False=0, True=1.
        return (self.type in (VariableType.CONTINUOUS, VariableType.INTEGER, VariableType.BOOLEAN)
                or self.numeric_encoding is not None
                or (self.type in (VariableType.CATEGORICAL, VariableType.ORDINAL)
                    and self.allowed_values is not None
                    and all(type(v) in (int, float) for v in self.allowed_values)))

    def validate_value(self, value: Scalar, *, check_range: bool = True) -> None:
        valid = {
            VariableType.CONTINUOUS: type(value) in (int, float),
            VariableType.INTEGER: type(value) is int,
            VariableType.BOOLEAN: type(value) is bool,
            VariableType.STRING: type(value) is str,
            VariableType.CATEGORICAL: True,
            VariableType.ORDINAL: True,
        }[self.type]
        if not valid or (self.allowed_values is not None and not any(
            same_value(value, allowed) for allowed in self.allowed_values
        )):
            raise ValueError(f"value does not match the declared domain of {self.name}")
        if check_range and self.range and (
            (self.range.lower is not None and value < self.range.lower)
            or (self.range.upper is not None and value > self.range.upper)
        ):
            raise ValueError(f"value outside the range of {self.name}")

    def validate_condition(self, condition: Condition, *, canonical: bool) -> None:
        if condition.values is not None:
            for value in condition.values:
                self.validate_value(value, check_range=canonical)
        else:
            # Conditions concern stored values, never implicit encoded scores.
            if self.type not in (VariableType.CONTINUOUS, VariableType.INTEGER):
                raise ValueError("range conditions require a numeric stored column")
            lower, upper = condition.range.lower, condition.range.upper
            if canonical and self.range and (
                (lower is not None and self.range.upper is not None and lower > self.range.upper)
                or (upper is not None and self.range.lower is not None and upper < self.range.lower)
            ):
                raise ValueError("condition range does not intersect the declared domain")
            if self.allowed_values is not None and not any(
                (lower is None or value >= lower) and (upper is None or value <= upper)
                for value in self.allowed_values
            ):
                raise ValueError("condition range contains no allowed value")


class DataSchema(Artifact):
    task: Reference
    variables: list[VariableSpec] = Field(min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self) -> Self:
        unique([v.name for v in self.variables], "variable names")
        by_name = {v.name: v for v in self.variables}
        for variable in self.variables:
            for parent in variable.derived_from:
                if parent not in by_name:
                    raise ValueError(f"unknown derivation variable: {parent}")
                if by_name[parent].generation_order >= variable.generation_order:
                    raise ValueError("derived variables must follow their dependencies")
        for constraint in self.constraints + [c for v in self.variables for c in v.constraints]:
            if set(constraint.variables) - by_name.keys():
                raise ValueError("constraint references unknown variables")
        return self

    def survival_columns(self) -> tuple[VariableSpec, VariableSpec]:
        """Single right-censored endpoint, bound by roles (never guessed by names).

        The pair inherits the endpoint name and origin from the referenced TaskSpec.
        True/1 denotes an observed event, False/0 denotes right censoring.
        """
        times = [v for v in self.variables if v.role == VariableRole.OBSERVED_TIME]
        events = [v for v in self.variables if v.role == VariableRole.EVENT_INDICATOR]
        if len(times) != 1 or len(events) != 1:
            raise ValueError("survival requires exactly one observed-time and one event-indicator column")
        time, event = times[0], events[0]
        if time.type not in (VariableType.CONTINUOUS, VariableType.INTEGER) or time.unit is None:
            raise ValueError("observed time must be numeric with an explicit unit")
        binary_integer = (event.type == VariableType.INTEGER and event.allowed_values is not None
                          and set(event.allowed_values) == {0, 1})
        if event.type != VariableType.BOOLEAN and not binary_integer:
            raise ValueError("event indicator must be Boolean or integer with domain [0, 1]")
        return time, event
