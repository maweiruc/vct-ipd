"""Evidence records, source provenance, and exact-match selection."""

from typing import Self

from pydantic import Field, JsonValue, model_validator

from .base import AnnotatedContract, Artifact, Condition, Contract, Identifier, NonNegative, Number, NumericRange, PositiveInt, Probability, Reference, Scalar, Text, unique
from .enums import EstimateStatus, EvidenceNature, Metric, SourceType
from .schema import DataSchema
from .task import Endpoint, Population, Treatment


class Provenance(Contract):
    source_type: SourceType
    reference: Text
    location: Text
    nature: EvidenceNature  # Deliberate, mandatory observed/demo declaration.
    uri: Text | None = None
    source_version: Text | None = None
    content_hash: Text | None = None
    extraction_method: Text | None = None
    extracted_by: Text | None = None
    notes: Text | None = None


class DistributionSpec(Contract):
    family: Text
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class CurvePoint(Contract):
    x: Number
    y: Number


class Curve(Contract):
    points: list[CurvePoint] = Field(min_length=1)
    x_unit: Text

    @model_validator(mode="after")
    def increasing(self) -> Self:
        if any(a.x >= b.x for a, b in zip(self.points, self.points[1:])):
            raise ValueError("curve x values must be strictly increasing")
        return self


Estimate = Scalar | DistributionSpec | Curve

NUMERIC_METRICS = frozenset({Metric.MEAN, Metric.SD, Metric.QUANTILE, Metric.PROPORTION,
                           Metric.CORRELATION, Metric.SURVIVAL_PROBABILITY, Metric.MEDIAN_SURVIVAL})
SURVIVAL_METRICS = frozenset({Metric.SURVIVAL_PROBABILITY, Metric.MEDIAN_SURVIVAL, Metric.SURVIVAL_CURVE})


def validate_availability(status: EstimateStatus, value: object | None,
                          reason: str | None, bound: NumericRange | None) -> None:
    """Presence checks shared by source evidence and generated-sample results."""
    if status == EstimateStatus.AVAILABLE:
        if value is None:
            raise ValueError("available requires an actual estimate or structured check")
        if reason is not None or bound is not None:
            raise ValueError("unavailability reason/bound cannot accompany an available point estimate")
    elif value is not None or reason is None:
        raise ValueError("unavailable estimates require a reason and must not supply a point estimate")


class SummarySpec(AnnotatedContract):
    """Metric plus its estimand context. Conditions define a subgroup denominator."""

    metric: Metric
    variables: list[Text] = Field(min_length=1)
    conditions: list[Condition] = Field(default_factory=list)
    category: Scalar | None = None  # Numerator category for a proportion.
    endpoint: Text | None = None
    time_origin: Text | None = None
    unit: Text | None = None
    quantile: Probability | None = None
    at_time: NonNegative | None = None
    method: Text | None = None
    estimand: Text | None = None
    custom_metric: Text | None = None

    @model_validator(mode="after")
    def metric_arguments(self) -> Self:
        unique(self.variables, "summary variables")
        if self.metric in (Metric.MEAN, Metric.SD, Metric.QUANTILE, Metric.PROPORTION) and len(self.variables) != 1:
            raise ValueError("scalar mean/SD/quantile/proportion requires exactly one variable")
        if (self.metric == Metric.QUANTILE) != (self.quantile is not None):
            raise ValueError("quantile level is required only for quantile metrics")
        if (self.metric == Metric.PROPORTION) != (self.category is not None):
            raise ValueError("category is required only for proportion metrics")
        if (self.metric == Metric.SURVIVAL_PROBABILITY) != (self.at_time is not None):
            raise ValueError("at_time is required only for survival probability")
        if self.metric in (Metric.SURVIVAL_PROBABILITY, Metric.MEDIAN_SURVIVAL, Metric.SURVIVAL_CURVE):
            if self.endpoint is None or self.unit is None:
                raise ValueError("survival metrics require endpoint and time unit")
            if len(self.variables) != 2:
                raise ValueError("survival metrics require observed-time and event-indicator variables")
        if self.time_origin is not None and self.endpoint is None:
            raise ValueError("time_origin requires an endpoint context")
        if self.metric == Metric.CORRELATION and len(self.variables) != 2:
            raise ValueError("correlation requires exactly two variables")
        if (self.metric == Metric.OTHER) != (self.custom_metric is not None):
            raise ValueError("custom_metric is required only for OTHER")
        return self

    def validate_schema(self, schema: DataSchema, *, canonical: bool = False) -> None:
        by_name = {v.name: v for v in schema.variables}
        names = by_name.keys()
        referenced = set(self.variables) | {c.variable for c in self.conditions}
        if referenced - names:
            raise ValueError(f"unknown summary variables: {sorted(referenced - names)}")
        for condition in self.conditions:
            by_name[condition.variable].validate_condition(condition, canonical=canonical)
        if self.metric in (Metric.MEAN, Metric.SD, Metric.QUANTILE, Metric.CORRELATION):
            if any(not by_name[name].supports_numeric_summary for name in self.variables):
                raise ValueError("numeric summary requires numeric columns or a declared numerical encoding")
        if self.metric == Metric.PROPORTION:
            by_name[self.variables[0]].validate_value(self.category, check_range=canonical)
        if self.metric in SURVIVAL_METRICS:
            time, event = schema.survival_columns()
            if set(self.variables) != {time.name, event.name}:
                raise ValueError("survival variables must be the schema's observed-time/event pair")
            if canonical and (self.unit != time.unit or self.time_origin is None):
                raise ValueError("canonical survival summaries require the time column unit and an explicit origin")
        if canonical:
            if self.metric in (Metric.MEAN, Metric.SD, Metric.QUANTILE, Metric.DISTRIBUTION):
                # Omitted units inherit the schema; explicit units must already be normalized.
                if self.unit is not None and any(self.unit != by_name[name].unit for name in self.variables):
                    raise ValueError("canonical summary units must match the schema; normalize source values first")
            if self.metric in (Metric.PROPORTION, Metric.CORRELATION) and self.unit not in (None, "1"):
                raise ValueError("canonical proportions/correlations are dimensionless fractions")

    def validate_estimate(self, value: Estimate) -> None:
        if self.metric in NUMERIC_METRICS and type(value) not in (float, int):
            raise ValueError("this metric requires a numeric estimate")
        if self.metric in (Metric.PROPORTION, Metric.SURVIVAL_PROBABILITY) and not 0 <= value <= 1:
            raise ValueError("probabilities must be between zero and one")
        if self.metric == Metric.CORRELATION and not -1 <= value <= 1:
            raise ValueError("correlation must be between -1 and one")
        if self.metric in (Metric.SD, Metric.MEDIAN_SURVIVAL) and value < 0:
            raise ValueError("this metric must be nonnegative")
        if self.metric == Metric.DISTRIBUTION and not isinstance(value, DistributionSpec):
            raise ValueError("distribution metric requires DistributionSpec")
        if self.metric == Metric.SURVIVAL_CURVE:
            if not isinstance(value, Curve) or value.x_unit != self.unit:
                raise ValueError("survival curve requires a Curve with matching time unit")
            if any(p.x < 0 or not 0 <= p.y <= 1 for p in value.points):
                raise ValueError("invalid survival curve coordinates")
            if any(a.y < b.y for a, b in zip(value.points, value.points[1:])):
                raise ValueError("survival probabilities must not increase")

    def validate_estimate_status(self, status: EstimateStatus, value: Estimate | None,
                                 reason: str | None, bound: NumericRange | None) -> None:
        validate_availability(status, value, reason, bound)
        if value is not None:
            self.validate_estimate(value)
        if status == EstimateStatus.NOT_REACHED and self.metric != Metric.MEDIAN_SURVIVAL:
            raise ValueError("not_reached is supported only for median survival")
        if bound is not None:
            if self.metric != Metric.MEDIAN_SURVIVAL:
                raise ValueError("unavailable-estimate bounds currently support median survival only")
            for number in (bound.lower, bound.upper):
                if number is not None:
                    self.validate_estimate(number)
            if status == EstimateStatus.NOT_REACHED and (bound.lower is None or bound.upper is not None):
                raise ValueError("not_reached may supply only a supported lower bound")


class EstimateUncertainty(Contract):
    interval: NumericRange | None = None
    confidence_level: Probability | None = None
    standard_error: NonNegative | None = None
    description: Text | None = None

    @model_validator(mode="after")
    def meaningful(self) -> Self:
        if self.interval is None and self.standard_error is None and self.description is None:
            raise ValueError("uncertainty needs an interval, standard error, or description")
        if self.confidence_level is not None:
            if self.interval is None or not 0 < self.confidence_level < 1:
                raise ValueError("confidence level must lie in (0, 1) and accompany an interval")
        return self


class Evidence(Artifact):
    provenance: Provenance
    population: Population
    treatment: Treatment | None = None
    endpoint: Endpoint | None = None
    summary: SummarySpec
    estimate: Estimate | None = None
    estimate_status: EstimateStatus = EstimateStatus.AVAILABLE
    estimate_reason: Text | None = None
    estimate_bound: NumericRange | None = None
    uncertainty: EstimateUncertainty | None = None
    sample_size: PositiveInt | None = None
    relevance: Probability | None = None
    reliability: Probability | None = None

    @model_validator(mode="after")
    def estimate_context(self) -> Self:
        self.summary.validate_estimate_status(self.estimate_status, self.estimate,
                                            self.estimate_reason, self.estimate_bound)
        if self.summary.endpoint is not None:
            if self.endpoint is None or self.endpoint.name != self.summary.endpoint:
                raise ValueError("summary endpoint must match the evidence endpoint context")
            if self.summary.metric in (Metric.SURVIVAL_PROBABILITY, Metric.MEDIAN_SURVIVAL, Metric.SURVIVAL_CURVE) and self.endpoint.unit != self.summary.unit:
                raise ValueError("evidence endpoint and summary time units must match")
            if self.summary.time_origin is not None and self.summary.time_origin != self.endpoint.time_origin:
                raise ValueError("evidence summary origin must match its source endpoint context")
        return self


class EvidenceSet(Artifact):
    task: Reference
    schema_ref: Reference
    items: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_evidence(self) -> Self:
        unique([e.id for e in self.items], "evidence IDs within a snapshot")
        return self

    def select(self, *, variable: str | None = None, endpoint: str | None = None,
               source_type: SourceType | None = None, population: Population | None = None) -> list[Evidence]:
        """AND filters; population matches supplied non-null fields exactly. No ranking."""
        criteria = population.model_dump(exclude_none=True) if population else {}
        return [e for e in self.items
                if (variable is None or variable in e.summary.variables or variable in [c.variable for c in e.summary.conditions])
                and (endpoint is None or (e.endpoint is not None and e.endpoint.name == endpoint))
                and (source_type is None or e.provenance.source_type == source_type)
                and all(getattr(e.population, key) == value for key, value in criteria.items())]

    def validate_references(self, schema: DataSchema) -> None:
        if self.schema_ref != schema.ref or self.task != schema.task:
            raise ValueError("evidence set and schema must reference the same task and schema snapshot")
        for item in self.items:
            item.summary.validate_schema(schema)
