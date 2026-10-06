"""Clinical intent and declarative run settings."""

from typing import Self

from pydantic import Field, model_validator

from .base import AnnotatedContract, Artifact, Contract, NonNegative, NonNegativeInt, PositiveInt, Text, unique


class Population(Contract):
    disease: Text
    stage: Text | None = None
    geography: Text | None = None
    line_of_therapy: Text | None = None
    description: Text | None = None


class Treatment(Contract):
    setting: Text | None = None
    regimen: Text | None = None


class Endpoint(Contract):
    name: Text
    time_origin: Text
    unit: Text


class FollowUp(Contract):
    duration: NonNegative | None = None
    unit: Text
    description: Text | None = None


class GenerationSettings(AnnotatedContract):
    comparison_seeds: list[NonNegativeInt] = Field(default_factory=list)
    validation_seeds: list[NonNegativeInt] = Field(default_factory=list)

    @model_validator(mode="after")
    def distinct_seeds(self) -> Self:
        unique(self.comparison_seeds, "comparison seeds")
        unique(self.validation_seeds, "validation seeds")
        if set(self.comparison_seeds) & set(self.validation_seeds):
            raise ValueError("validation seeds must be fresh")
        return self


class CalibrationSettings(AnnotatedContract):
    max_iterations: PositiveInt = 10
    no_improvement_patience: PositiveInt | None = None
    initial_search_budget: NonNegativeInt = 0
    searches_per_iteration: NonNegativeInt = 0
    total_search_budget: NonNegativeInt = 0

    @model_validator(mode="after")
    def search_budget(self) -> Self:
        if self.initial_search_budget > self.total_search_budget or self.searches_per_iteration > self.total_search_budget:
            raise ValueError("individual search budgets cannot exceed the total")
        return self


class TaskSpec(Artifact):
    population: Population
    treatment: Treatment
    endpoint: Endpoint
    sample_size: PositiveInt
    intended_use: Text
    original_request: Text | None = None
    follow_up: FollowUp | None = None
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
