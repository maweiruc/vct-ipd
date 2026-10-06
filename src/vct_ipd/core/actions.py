"""Validated proposals only. No action execution, controller, or code evaluation."""

from typing import Self

from pydantic import Field, JsonValue, model_validator

from .base import Artifact, Contract, Identifier, NonNegativeInt, PositiveInt, Reference, Text
from .enums import ActionType, EvaluationStatus, StopReason
from .evaluation import EvaluationReport
from .generator import GeneratorState
from .schema import DataSchema
from .targets import CalibrationTarget


class TuneParameters(Contract):
    component: Text
    parameters: dict[str, JsonValue] = Field(min_length=1)


class ModifyGenerator(Contract):
    component: Text
    proposal: Text
    model_specification: dict[str, JsonValue] = Field(min_length=1)


class RetrieveEvidence(Contract):
    query: Text
    variables: list[Text] = Field(min_length=1)
    max_sources: PositiveInt
    endpoint: Text | None = None


class ReviseTarget(Contract):
    previous: Reference
    replacement: CalibrationTarget

    @model_validator(mode="after")
    def new_snapshot(self) -> Self:
        if self.previous.id != self.replacement.id or self.previous.version == self.replacement.version:
            raise ValueError("revision must keep the target ID and change its version")
        return self


class ReviseSchema(Contract):
    previous: Reference
    replacement: DataSchema

    @model_validator(mode="after")
    def new_snapshot(self) -> Self:
        if self.previous.id != self.replacement.id or self.previous.version == self.replacement.version:
            raise ValueError("revision must keep the schema ID and change its version")
        return self


class Stop(Contract):
    reason: StopReason


ActionPayload = TuneParameters | ModifyGenerator | RetrieveEvidence | ReviseTarget | ReviseSchema | Stop


class Action(Artifact):
    type: ActionType
    payload: ActionPayload
    reason: Text
    evaluation_report: Reference
    generator_state: Reference
    run_id: Identifier
    iteration: NonNegativeInt

    @model_validator(mode="after")
    def payload_matches_type(self) -> Self:
        expected = {
            ActionType.TUNE_PARAMETERS: TuneParameters,
            ActionType.MODIFY_GENERATOR: ModifyGenerator,
            ActionType.RETRIEVE_EVIDENCE: RetrieveEvidence,
            ActionType.REVISE_TARGET: ReviseTarget,
            ActionType.REVISE_SCHEMA: ReviseSchema,
            ActionType.STOP: Stop,
        }[self.type]
        if not isinstance(self.payload, expected):
            raise ValueError(f"{self.type} requires {expected.__name__} payload")
        return self

    def validate_references(self, report: EvaluationReport, state: GeneratorState) -> None:
        if (self.evaluation_report != report.ref or self.generator_state != state.ref
                or report.generator_state != state.ref or self.run_id != state.run_id
                or report.run_id != self.run_id or self.iteration != state.iteration
                or report.iteration != self.iteration or report.target_set != state.target_set
                or report.schema_ref != state.schema_ref):
            raise ValueError("action report/state/run/iteration references do not match")
        if isinstance(self.payload, Stop) and self.payload.reason == StopReason.CALIBRATED:
            if report.status != EvaluationStatus.PASS:
                raise ValueError("calibrated STOP requires a passing evaluation report")
        if isinstance(self.payload, ReviseTarget):
            if self.payload.previous not in [r.target for r in report.results]:
                raise ValueError("revised target must occur in the triggering report")
        if isinstance(self.payload, ReviseSchema):
            if self.payload.previous != state.schema_ref or self.payload.replacement.task != state.task:
                raise ValueError("schema revision must match the current schema and task")
