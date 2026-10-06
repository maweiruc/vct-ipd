"""Small extraction schema plus richer deterministic planning artifacts."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from vct_ipd.core import TaskSpec, DataSchema
from vct_ipd.generators.statistical import GeneratorConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Claim(StrictModel):
    field: str
    value: str
    quote: str


class Requirement(StrictModel):
    description: str
    quote: str


class Draft(StrictModel):
    # All fields required; open strings preserve endpoints/variables outside M2.
    claims: list[Claim]
    additional_requirements: list[Requirement]
    ambiguities: list[str]


class PlanningRequest(StrictModel):
    text: str = Field(min_length=1, max_length=8000)
    profile: str


class Issue(StrictModel):
    code: str
    boundary: Literal['extraction', 'capability', 'evidence', 'resource', 'input']
    message: str


class Provenance(StrictModel):
    field: str
    source: Literal['user', 'profile', 'normalization', 'assumption', 'unresolved']
    value: JsonValue
    quote: str | None = None
    rule: str | None = None


class PlanningResult(StrictModel):
    version: Literal['m3-v1'] = 'm3-v1'
    request: PlanningRequest
    draft: Draft | None = None
    status: Literal['READY', 'NEEDS_CLARIFICATION', 'UNSUPPORTED', 'ERROR']
    issues: list[Issue] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    normalized: dict[str, JsonValue] = Field(default_factory=dict)
    profile_hash: str
    context_hash: str
    input_manifest: dict[str, JsonValue] = Field(default_factory=dict)
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    task: TaskSpec | None = None
    data_schema: DataSchema | None = None
    execution: GeneratorConfig | None = None
