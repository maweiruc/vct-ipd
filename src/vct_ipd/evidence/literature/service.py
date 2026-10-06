"""Collect evidence once. No sampling, target adoption or iterative search here."""
from dataclasses import dataclass, field
from .contracts import EvidenceRequest, RetrievalResult, ExtractionDraft, EvidenceBatch
from .extraction import extraction_context
from .validation import validate
from vct_ipd.llm.providers import ProviderError, safe_metadata


@dataclass
class Collection:
    request: EvidenceRequest
    retrieval: RetrievalResult
    draft: ExtractionDraft
    batch: EvidenceBatch
    status: str
    extraction_metadata: dict = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)


def collect(request, retriever, extractor=None):
    request = EvidenceRequest.model_validate_json(request.model_dump_json())
    retrieval = retriever.search(request)
    result = Collection(request, retrieval, ExtractionDraft(candidates=[]), EvidenceBatch(), retrieval.status)
    if retrieval.status not in ('documents',):
        return result
    if extractor is None:
        result.status = 'retrieved_only'; return result
    try:
        result.draft = extractor.extract(request, retrieval.documents)
        result.batch = validate(result.draft, retrieval.documents)
        statuses = {v.status for v in result.batch.validation}
        result.status = ('conflicting_evidence' if result.batch.conflicts else 'needs_review' if 'needs_review' in statuses
                         else 'validated_evidence' if result.batch.validated
                         else 'unsupported' if 'unsupported' in statuses
                         else 'rejected' if 'rejected' in statuses else 'no_statistics')
    except ProviderError as exc:
        result.status = 'extraction_error'; result.diagnostics.append(exc.code + ': ' + str(exc))
    result.extraction_metadata = {**safe_metadata(extractor), 'context': extraction_context(request, retrieval.documents)}
    return result
