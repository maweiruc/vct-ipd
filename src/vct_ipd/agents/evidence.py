"""Application-configured M4 slots. A decision selects an ID, never a path."""
from dataclasses import dataclass
from pathlib import Path
from vct_ipd.evidence.literature.contracts import EvidenceRequest
from vct_ipd.evidence.literature.retrieval import LocalDocuments, bounded_json
from vct_ipd.evidence.literature.extraction import ReplayExtractor
from vct_ipd.evidence.literature.service import collect
from vct_ipd.evidence.literature.adoption import adopt_demo
from vct_ipd.planners.profile import digest
from vct_ipd.core import EvidenceSet, CalibrationTargetSet


@dataclass
class EvidenceSlot:
    request: EvidenceRequest
    retriever: object
    extractor_factory: object
    binding: str
    selection: Path | None = None
    inputs: Path | None = None
    check_binding: object = None

    def costs(self):
        r=self.request
        return {'queries':len(r.queries), 'pages':len(r.queries)*r.max_pages,
                'documents':min(r.max_documents,r.max_results), 'bytes':r.max_bytes,
                'extractions':r.max_extraction_calls, 'provider_calls':r.max_extraction_calls}

    def collect(self):
        if self.check_binding and self.check_binding()!=self.binding:
            raise ValueError('Evidence slot files changed after policy binding')
        return collect(self.request,self.retriever,self.extractor_factory())


def local_slot(directory, inputs, selection):
    directory,selection=Path(directory),Path(selection)
    def binding():
        return digest({name:bounded_json(directory/name) for name in ('request.json','sources.json','extraction.json')} | {'selection':bounded_json(selection)})
    return EvidenceSlot(EvidenceRequest.model_validate(bounded_json(directory/'request.json')),
        LocalDocuments(directory/'sources.json'),lambda:ReplayExtractor(directory/'extraction.json'),
        binding(),selection,Path(inputs),binding)


def adopt(slot, collection, task, schema, old_evidence, old_targets, policy, epoch):
    if policy.adoption!='hypothetical_preferred':return None
    assembled, record=adopt_demo(collection,slot.selection,slot.inputs)
    _, _, new_evidence, proposed, _=assembled
    old={e.id:e for e in old_evidence.items}
    for item in new_evidence.items:
        if item.id in old and item!=old[item.id]:raise ValueError('Source identity collision during enrichment')
        old[item.id]=item
    evidence=EvidenceSet.model_validate({**old_evidence.model_dump(mode='json'),
        'version':str(epoch+1),'items':[e.model_dump(mode='json') for e in old.values()]})
    replacements={t.id:t for t in proposed.items}
    items=[]
    for original in old_targets.items:
        chosen=replacements[original.id] if original.id in policy.target_ids else original
        for attr in ('summary','tolerance','hard_constraint','priority'):
            if getattr(chosen,attr)!=getattr(original,attr):raise ValueError('Adoption cannot alter requirements or tolerance')
        items.append({**chosen.model_dump(mode='json'),'version':str(epoch+1)})
    targets=CalibrationTargetSet.model_validate({**old_targets.model_dump(mode='json'),
        'version':str(epoch+1),'evidence_set':evidence.ref.model_dump(),'items':items})
    evidence.validate_references(schema);targets.validate_references(evidence,schema);targets.validate_task(task,schema)
    return evidence, targets, record
