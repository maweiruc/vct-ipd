"""Shared, JSON-safe primitives. No storage or execution dependencies."""

import math
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictFloat, StrictInt, StrictStr, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Identifier = Text
Version = Text
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Number = StrictInt | StrictFloat
NonNegative = Annotated[Number, Field(ge=0)]
Probability = Annotated[Number, Field(ge=0, le=1)]
Scalar = StrictBool | StrictInt | StrictFloat | StrictStr


def unique(values: list, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)

    @model_validator(mode="before")
    @classmethod
    def finite_json(cls, value):
        # Covers arbitrary JsonValue payloads too; JSON NaN/Infinity are not portable.
        def check(item):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("non-finite numbers are not allowed")
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise ValueError("object keys must be strings")
                for child in item.values():
                    check(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    check(child)
        check(value)
        return value


class Reference(Contract):
    id: Identifier
    version: Version


class Assumption(Contract):
    id: Identifier
    statement: Text
    rationale: Text


class Uncertainty(Contract):
    id: Identifier
    question: Text
    impact: Text | None = None


class AnnotatedContract(Contract):
    assumptions: list[Assumption] = Field(default_factory=list)
    uncertainties: list[Uncertainty] = Field(default_factory=list)
    extensions: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def annotation_ids(self) -> Self:
        unique([a.id for a in self.assumptions], "assumption IDs")
        unique([u.id for u in self.uncertainties], "uncertainty IDs")
        return self


class Artifact(AnnotatedContract):
    id: Identifier
    version: Version = "1"

    @property
    def ref(self) -> Reference:
        return Reference(id=self.id, version=self.version)


class NumericRange(Contract):
    lower: Number | None = None
    upper: Number | None = None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.lower is None and self.upper is None:
            raise ValueError("a range needs at least one bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower must not exceed upper")
        return self


class Condition(Contract):
    """A subset: values means membership; range means inclusive numeric bounds."""

    variable: Text
    values: list[Scalar] | None = Field(default=None, min_length=1)
    range: NumericRange | None = None

    @model_validator(mode="after")
    def one_form(self) -> Self:
        if (self.values is None) == (self.range is None):
            raise ValueError("provide exactly one of values or range")
        return self
