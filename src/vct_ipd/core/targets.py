"""Calibration goals remain separate from source evidence."""

from typing import Self

from pydantic import Field, StrictBool, model_validator

from .base import Artifact, Contract, Identifier, NonNegative, NumericRange, Reference, Text, unique
from .enums import Metric, Priority
from .evidence import Estimate, EvidenceSet, SummarySpec, SURVIVAL_METRICS
from .schema import DataSchema
from .task import TaskSpec


class DesiredValue(Contract):
    center: Estimate | None = None
    range: NumericRange | None = None

    @model_validator(mode="after")
    def meaningful(self) -> Self:
        if self.center is None and self.range is None:
            raise ValueError("a target needs a center, range, distribution, or curve")
        if self.range and self.center is not None:
            if type(self.center) not in (float, int):
                raise ValueError("a range can accompany only a numeric center")
            if ((self.range.lower is not None and self.center < self.range.lower)
                    or (self.range.upper is not None and self.center > self.range.upper)):
                raise ValueError("target center must lie within its range")
        return self


class Tolerance(Contract):
    absolute: NonNegative | None = None
    relative: NonNegative | None = None

    @model_validator(mode="after")
    def one_mode(self) -> Self:
        if (self.absolute is None) == (self.relative is None):
            raise ValueError("specify exactly one tolerance mode")
        return self


class CalibrationTarget(Artifact):
    summary: SummarySpec
    desired: DesiredValue
    tolerance: Tolerance | None = None
    priority: Priority = Priority.MEDIUM
    hard_constraint: StrictBool = False
    evidence_ids: list[Identifier] = Field(default_factory=list)
    evidence_limited: StrictBool = False
    evidence_limitation: Text | None = None
    conflict_notes: list[Text] = Field(default_factory=list)
    rationale: Text  # Synthesis/design rationale; no silent pooling of evidence.
    adoption_notes: dict[Identifier, Text] = Field(default_factory=dict)

    @model_validator(mode="after")
    def supported(self) -> Self:
        unique(self.evidence_ids, "supporting evidence IDs")
        if self.adoption_notes.keys() - set(self.evidence_ids):
            raise ValueError("adoption notes must reference supporting evidence IDs")
        if self.evidence_limited != (self.evidence_limitation is not None):
            raise ValueError("evidence-limited flag and explanation must be supplied together")
        if not self.evidence_ids and not self.hard_constraint:
            if not self.evidence_limited or not self.assumptions:
                raise ValueError("unsupported soft targets require an explicit assumption and evidence limitation")
        if self.desired.center is not None:
            self.summary.validate_estimate(self.desired.center)
        if self.desired.range:
            if self.summary.metric in (Metric.DISTRIBUTION, Metric.SURVIVAL_CURVE):
                raise ValueError("distribution/curve targets require a structured center")
            for bound in (self.desired.range.lower, self.desired.range.upper):
                if bound is not None:
                    self.summary.validate_estimate(bound)
        return self


class CalibrationTargetSet(Artifact):
    task: Reference
    schema_ref: Reference
    evidence_set: Reference
    items: list[CalibrationTarget] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_targets(self) -> Self:
        unique([t.id for t in self.items], "target IDs")
        return self

    def validate_references(self, evidence: EvidenceSet, schema: DataSchema) -> None:
        evidence.validate_references(schema)
        if self.evidence_set != evidence.ref or self.schema_ref != schema.ref or self.task != schema.task:
            raise ValueError("target set references do not match the evidence/schema/task snapshots")
        ids = {e.id for e in evidence.items}
        for target in self.items:
            target.summary.validate_schema(schema, canonical=True)
            if set(target.evidence_ids) - ids:
                raise ValueError(f"target {target.id} references unknown evidence")
            if target.summary.metric in SURVIVAL_METRICS:
                for source in evidence.items:
                    if source.id in target.evidence_ids and source.endpoint is not None:
                        if (source.endpoint.name != target.summary.endpoint
                                or source.endpoint.time_origin != target.summary.time_origin):
                            if source.id not in target.adoption_notes:
                                raise ValueError("adopting a different source endpoint/origin requires explicit adoption notes")

    def validate_task(self, task: TaskSpec, schema: DataSchema) -> None:
        """Finalize canonical meaning against the actual task, not just its ID."""
        if self.task != task.ref or schema.task != task.ref or self.schema_ref != schema.ref:
            raise ValueError("canonical target task/schema snapshots do not match")
        for target in self.items:
            summary = target.summary
            summary.validate_schema(schema, canonical=True)
            if summary.endpoint is not None and summary.endpoint != task.endpoint.name:
                raise ValueError("canonical target endpoint must match the task")
            if summary.time_origin is not None and summary.time_origin != task.endpoint.time_origin:
                raise ValueError("canonical target time origin must match the task")
            if summary.metric in SURVIVAL_METRICS and summary.unit != task.endpoint.unit:
                raise ValueError("canonical survival target time unit must match the task")
