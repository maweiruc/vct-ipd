"""M4 sidecar contracts. Incomplete sources stay candidates, not fake core Evidence."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from vct_ipd.core import Evidence


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class EvidenceRequest(Model):
    target_description: str = Field(min_length=1, max_length=8000)
    statistics: list[str] = Field(min_length=1, max_length=10)
    queries: list[str] = Field(default_factory=list, max_length=2)
    allowed_sources: list[Literal['local', 'europe_pmc']] = Field(default_factory=lambda: ['local'])
    max_results: int = Field(default=5, ge=1, le=10)
    max_pages: int = Field(default=1, ge=1, le=2)
    max_documents: int = Field(default=5, ge=1, le=10)
    max_bytes: int = Field(default=200000, ge=1000, le=500000)
    max_extraction_calls: int = Field(default=1, ge=0, le=1)


class Scope(Model):
    disease: str | None
    stage: str | None
    geography: str | None
    line_of_therapy: str | None
    regimen: str | None
    setting: str | None
    endpoint: str | None
    origin: str | None
    subgroup: str | None


class DocumentRecord(Model):
    id: str
    source: Literal['local', 'europe_pmc']
    source_key: str
    title: str
    year: str | None = None
    doi: str | None = None
    uri: str
    nature: Literal['demo', 'observed']
    access_scope: Literal['metadata', 'abstract', 'local_text']
    retrieved_at: str
    text: str
    content_hash: str
    raw_hash: str
    text_version: str = 'utf8-lf-v1'
    rights: str
    cache_text: bool = False
    registered_scope: Scope | None = None


class Support(Model):
    start: int
    end: int
    quote: str


class Proposal(Model):
    metric: str
    value: float | None
    unit: str | None
    category: str | None
    at_time: float | None
    endpoint: str | None
    estimate_status: str
    lower_bound: float | None
    ci_lower: float | None
    ci_upper: float | None
    confidence_level: float | None
    standard_error: float | None
    sample_size: int | None


class Candidate(Model):
    document_id: str
    support: Support
    scope: Scope
    statistic: Proposal
    unknowns: list[str]


class ExtractionDraft(Model):
    candidates: list[Candidate]


class CandidateResult(Model):
    index: int
    status: Literal['accepted', 'needs_review', 'unsupported', 'rejected', 'duplicate']
    reasons: list[str]
    evidence_id: str | None = None


class EvidenceBatch(Model):
    validated: list[Evidence] = Field(default_factory=list)
    validation: list[CandidateResult] = Field(default_factory=list)
    conflicts: list[list[str]] = Field(default_factory=list)


class RetrievalResult(Model):
    status: str
    documents: list[DocumentRecord] = Field(default_factory=list)
    searches: list[dict] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    calls: int = 0
    mode: str
