"""Dataset metadata references an artifact; it does not load or generate data."""

from .base import Artifact, Contract, Identifier, NonNegativeInt, PositiveInt, Reference, Text
from .generator import GeneratorState
from .schema import DataSchema


class DataArtifact(Contract):
    uri: Text
    format: Text
    content_hash: Text | None = None


class SyntheticDataset(Artifact):
    generator_state: Reference
    schema_ref: Reference
    run_id: Identifier
    iteration: NonNegativeInt
    seed: NonNegativeInt
    sample_size: PositiveInt
    data: DataArtifact

    def validate_references(self, state: GeneratorState, schema: DataSchema) -> None:
        if (self.generator_state != state.ref or self.schema_ref != schema.ref
                or state.schema_ref != schema.ref or state.task != schema.task
                or self.run_id != state.run_id or self.iteration != state.iteration):
            raise ValueError("dataset state/schema/run/iteration references do not match")
