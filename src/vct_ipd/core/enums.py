"""Closed vocabularies for the initial contracts, not plugin registries."""

from enum import StrEnum


class VariableType(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BOOLEAN = "boolean"
    STRING = "string"


class VariableRole(StrEnum):
    IDENTIFIER = "identifier"
    BASELINE_COVARIATE = "baseline_covariate"
    TREATMENT = "treatment"
    EVENT_TIME = "event_time"
    CENSORING_TIME = "censoring_time"
    OBSERVED_TIME = "observed_time"
    EVENT_INDICATOR = "event_indicator"
    DERIVED = "derived"
    OTHER = "other"


class MissingnessPolicy(StrEnum):
    FORBIDDEN = "forbidden"
    ALLOWED = "allowed"
    MODELED = "modeled"


class SourceType(StrEnum):
    LITERATURE = "literature"
    USER_SUMMARY = "user_summary"
    REFERENCE_IPD = "reference_ipd"
    DATABASE = "database"
    OTHER = "other"


class EvidenceNature(StrEnum):
    OBSERVED = "observed"
    DEMO = "demo"


class EstimateStatus(StrEnum):
    AVAILABLE = "available"
    NOT_REACHED = "not_reached"
    NOT_REPORTED = "not_reported"
    NOT_ESTIMABLE = "not_estimable"


class Metric(StrEnum):
    VALIDITY = "validity"
    MEAN = "mean"
    SD = "sd"
    QUANTILE = "quantile"
    PROPORTION = "proportion"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    DEPENDENCE = "dependence"
    SURVIVAL_PROBABILITY = "survival_probability"
    MEDIAN_SURVIVAL = "median_survival"
    SURVIVAL_CURVE = "survival_curve"
    OTHER = "other"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvaluationStatus(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    EVIDENCE_LIMITED = "EVIDENCE_LIMITED"


class ActionType(StrEnum):
    TUNE_PARAMETERS = "TUNE_PARAMETERS"
    MODIFY_GENERATOR = "MODIFY_GENERATOR"
    RETRIEVE_EVIDENCE = "RETRIEVE_EVIDENCE"
    REVISE_TARGET = "REVISE_TARGET"
    REVISE_SCHEMA = "REVISE_SCHEMA"
    STOP = "STOP"


class StopReason(StrEnum):
    CALIBRATED = "calibrated"
    MAX_ITERATIONS = "max_iterations"
    NO_IMPROVEMENT = "no_improvement"
    EVIDENCE_LIMITED = "evidence_limited"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STRUCTURAL_FAILURE = "structural_failure"
