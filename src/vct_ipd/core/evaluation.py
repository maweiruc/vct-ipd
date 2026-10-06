"""Evaluation outputs, without metric computation or stopping logic."""

from typing import Self

from pydantic import Field, JsonValue, StrictBool, model_validator

from .base import AnnotatedContract, Artifact, Contract, Identifier, NonNegative, NonNegativeInt, NumericRange, Reference, Text, unique
from .dataset import SyntheticDataset
from .enums import EstimateStatus, EvaluationStatus, Metric
from .evidence import Estimate, NUMERIC_METRICS, validate_availability
from .generator import GeneratorState
from .targets import CalibrationTarget, CalibrationTargetSet


class CheckResult(Contract):
    """Explicit qualitative/custom result, not a substitute for a numeric estimate."""

    check: Text
    passed: StrictBool
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TargetResult(AnnotatedContract):
    target: Reference
    status: EvaluationStatus
    observed: Estimate | None = None
    discrepancy: NonNegative | None = None
    diagnostic: Text
    estimate_status: EstimateStatus = EstimateStatus.AVAILABLE
    estimate_reason: Text | None = None
    estimate_bound: NumericRange | None = None
    structured_result: CheckResult | None = None

    @model_validator(mode="after")
    def completeness(self) -> Self:
        if self.observed is not None and self.structured_result is not None:
            raise ValueError("supply an observed estimate or a structured check, not both")
        value = self.structured_result if self.structured_result is not None else self.observed
        validate_availability(self.estimate_status, value, self.estimate_reason, self.estimate_bound)
        if self.estimate_status != EstimateStatus.AVAILABLE:
            if self.status == EvaluationStatus.PASS or self.discrepancy is not None:
                raise ValueError("unavailable results cannot PASS or carry a computed discrepancy")
        if self.status == EvaluationStatus.PASS:
            if type(self.observed) in (int, float) and self.discrepancy is None:
                raise ValueError("numeric PASS requires an observed value and discrepancy")
            if self.structured_result is not None and not self.structured_result.passed:
                raise ValueError("a failed structured check cannot PASS")
        return self

    def validate_target(self, target: CalibrationTarget) -> None:
        if self.target != target.ref:
            raise ValueError("target result version does not match")
        if self.structured_result is not None:
            if target.summary.metric not in (Metric.VALIDITY, Metric.OTHER, Metric.DEPENDENCE):
                raise ValueError("structured checks cannot replace built-in numerical/distribution results")
        else:
            target.summary.validate_estimate_status(self.estimate_status, self.observed,
                                                   self.estimate_reason, self.estimate_bound)
        if target.summary.metric in NUMERIC_METRICS and self.status == EvaluationStatus.PASS:
            if self.observed is None or self.discrepancy is None:
                raise ValueError("numeric PASS requires an observed value and discrepancy")


class EvaluationReport(Artifact):
    generator_state: Reference
    target_set: Reference
    schema_ref: Reference
    datasets: list[Reference] = Field(min_length=1)
    run_id: Identifier
    iteration: NonNegativeInt
    status: EvaluationStatus
    results: list[TargetResult] = Field(min_length=1)
    summary: Text
    overall_metrics: dict[str, JsonValue] = Field(default_factory=dict)
    evaluator_version: Text

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        unique([(r.id, r.version) for r in self.datasets], "dataset snapshots")
        unique([r.target.id for r in self.results], "target result IDs")
        if self.status == EvaluationStatus.PASS and any(r.status != EvaluationStatus.PASS for r in self.results):
            raise ValueError("PASS requires all reported targets to pass")
        return self

    def validate_references(self, state: GeneratorState, targets: CalibrationTargetSet,
                            datasets: list[SyntheticDataset]) -> None:
        if (self.generator_state != state.ref or self.target_set != targets.ref
                or self.target_set != state.target_set or self.schema_ref != state.schema_ref
                or self.schema_ref != targets.schema_ref or state.task != targets.task
                or self.run_id != state.run_id or self.iteration != state.iteration):
            raise ValueError("report input snapshots do not match")
        unique([(d.id, d.version) for d in datasets], "supplied dataset snapshots")
        if {(r.id, r.version) for r in self.datasets} != {(d.id, d.version) for d in datasets}:
            raise ValueError("report dataset references do not match supplied datasets")
        for dataset in datasets:
            if (dataset.generator_state != state.ref or dataset.schema_ref != self.schema_ref
                    or dataset.run_id != self.run_id or dataset.iteration != self.iteration):
                raise ValueError("report dataset lineage does not match")
        expected = {t.id: t for t in targets.items}
        if {r.target.id for r in self.results} != expected.keys():
            raise ValueError("report must cover every target exactly once")
        for result in self.results:
            target = expected[result.target.id]
            result.validate_target(target)
            if target.hard_constraint and result.status != EvaluationStatus.PASS and self.status != EvaluationStatus.FAIL:
                raise ValueError("unpassed hard constraints require overall FAIL")
