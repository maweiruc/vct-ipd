"""Serializable generator descriptions and lineage, with no generator implementation."""

from typing import Self

from pydantic import Field, JsonValue, model_validator

from .base import Artifact, Identifier, NonNegativeInt, Reference, Text, unique
from .schema import DataSchema
from .targets import CalibrationTargetSet
from .task import TaskSpec


class GeneratorState(Artifact):
    run_id: Identifier
    task: Reference
    schema_ref: Reference
    target_set: Reference
    generator_type: Text
    iteration: NonNegativeInt
    model_specification: dict[str, JsonValue] = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    code_version: Text
    config_version: Text
    parent_state: Reference | None = None
    modification_reason: Text

    @model_validator(mode="after")
    def local_lineage(self) -> Self:
        if (self.iteration == 0) != (self.parent_state is None):
            raise ValueError("only iteration zero has no parent")
        if self.parent_state == self.ref:
            raise ValueError("a state cannot be its own parent")
        return self

    def validate_parent(self, parent: "GeneratorState") -> None:
        if self.parent_state != parent.ref:
            raise ValueError("parent reference does not match supplied parent")
        if self.run_id != parent.run_id or self.task != parent.task:
            raise ValueError("parent must belong to the same run and task snapshot")
        if self.iteration != parent.iteration + 1:
            raise ValueError("child iteration must immediately follow parent")

    def validate_references(self, task: TaskSpec, schema: DataSchema, targets: CalibrationTargetSet) -> None:
        if (self.task != task.ref or schema.task != task.ref or targets.task != task.ref
                or self.schema_ref != schema.ref or targets.schema_ref != schema.ref
                or self.target_set != targets.ref):
            raise ValueError("generator input snapshots do not match")
        targets.validate_task(task, schema)


def validate_lineage(states: list[GeneratorState]) -> None:
    """Validate a complete ordered chain, beginning with the root state."""
    if not states or states[0].iteration != 0:
        raise ValueError("lineage must begin at iteration zero")
    unique([(s.id, s.version) for s in states], "state snapshots")
    for parent, child in zip(states, states[1:]):
        child.validate_parent(parent)
