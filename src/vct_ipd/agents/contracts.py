"""M5 sidecars, leaving accepted core contracts unchanged."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Bound(Model):
    lower: float
    upper: float
    @model_validator(mode='after')
    def ordered(self):
        if self.lower >= self.upper: raise ValueError('Parameter bounds must be ordered')
        return self


class Limits(Model):
    states: int = Field(default=80, ge=1, le=201)
    decisions: int = Field(default=100, ge=1, le=250)
    provider_calls: int = Field(default=100, ge=0, le=250)
    queries: int = Field(default=2, ge=0, le=10)
    pages: int = Field(default=2, ge=0, le=10)
    documents: int = Field(default=5, ge=0, le=20)
    bytes: int = Field(default=200000, ge=0, le=1000000)
    extractions: int = Field(default=1, ge=0, le=5)
    transitions: int = Field(default=1, ge=0, le=3)


class SlotPolicy(Model):
    binding: str
    target_ids: list[str] = Field(min_length=1)
    required: bool = True
    adoption: Literal['hypothetical_preferred', 'review_only'] = 'review_only'


class ControllerPolicy(Model):
    version: str = 'm5-controller-v1'
    allowed_actions: list[str] = Field(default_factory=lambda: ['TUNE_PARAMETERS','RETRIEVE_EVIDENCE','STOP'])
    parameters: dict[str, Bound] = Field(default_factory=lambda: {
        k: Bound(lower=a,upper=b) for k,a,b in [('age_mean',30.,85.),('male_probability',.05,.95),
        ('ecog_intercept',-4.,4.),('ecog_male',-4.,4.),('weibull_scale',2.,80.)]})
    limits: Limits = Field(default_factory=Limits)
    max_sample_size: int = Field(default=5000, ge=1, le=5000)
    patience: int = Field(default=15, ge=1, le=100)
    max_rejections: int = Field(default=3, ge=1, le=10)
    provider_failure: Literal['stop'] = 'stop'
    slots: dict[str, SlotPolicy] = Field(default_factory=dict)


class Change(Model):
    name: str
    value: float


class DecisionDraft(Model):
    diagnosis: str = Field(min_length=1, max_length=1200)
    context_hash: str
    base_state: str
    report: str
    target_set: str
    affected_targets: list[str]
    action: Literal['TUNE_PARAMETERS','RETRIEVE_EVIDENCE','STOP','MODIFY_GENERATOR','REVISE_SCHEMA','REVISE_TARGET']
    changes: list[Change]
    slot: str | None
    stop_reason: str | None


class DiagnosticContext(Model):
    epoch: int
    binding: dict
    state: dict
    report: dict
    targets: list[dict]
    evidence: list[dict]
    assumptions: list[dict]
    remaining: dict[str,int]
    slots: dict
    used_slots: list[str]
    prior_actions: list[dict]


class DecisionRecord(Model):
    step: int
    epoch: int
    context_hash: str
    context: DiagnosticContext
    proposed: dict | None
    accepted: bool
    reason: str
    applied_action: dict | None
    provider: dict
    budget: dict
    resulting_state: dict | None = None
    resulting_epoch: int | None = None
