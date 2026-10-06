"""Explicit configured selection for a separate fictional literature demo only."""
from pydantic import Field
from .contracts import Model
from .retrieval import bounded_json
from .validation import validate, find_conflicts
from vct_ipd.core import EvidenceSet, SummarySpec
from vct_ipd.demo import demo_context
from vct_ipd.evidence.local import SummaryProvider, IPDProvider, combine
from vct_ipd.generators.statistical import StatisticalGenerator, Parameters
from vct_ipd.planners.profile import profile, input_manifest, digest
from vct_ipd.targets.builder import TargetBuilder, TargetRule
from vct_ipd.pipeline.runner import ready_to_run

DEMO_DESCRIPTION = 'Hypothetical advanced NSCLC, single DEMO-A cohort, treatment-start OS; software demonstration only.'


class AdoptionRecord(Model):
    policy: str
    authority: str
    target_description: str
    literature_ids: list[str]
    rules: list[dict]
    conflicts: list[list[str]] = Field(default_factory=list)
    unadopted_ids: list[str] = Field(default_factory=list)
    notes: str


def adopt_demo(collection, selection_path, local_inputs):
    if validate(collection.draft, collection.retrieval.documents) != collection.batch:
        raise ValueError('Extraction validation snapshot changed; collect and review again')
    selection = bounded_json(selection_path)
    required = {'policy', 'authority', 'target_description', 'document_hashes', 'rules', 'conflict_notes'}
    if not isinstance(selection, dict) or set(selection) != required or selection['policy'] != 'hypothetical-literature-v1' or selection['authority'] != 'application_configuration':
        raise ValueError('Explicit registered demo selection policy is required')
    if collection.request.target_description != DEMO_DESCRIPTION or selection['target_description'] != DEMO_DESCRIPTION:
        raise ValueError('Evidence search for a different population cannot activate the hypothetical generator')
    if collection.status in ('extraction_error', 'rejected', 'no_statistics') or not collection.batch.validated:
        raise ValueError('No validated literature statistics to adopt')
    documents = {d.id: d for d in collection.retrieval.documents}
    if selection['document_hashes'] != {key: d.content_hash for key, d in documents.items()}:
        raise ValueError('Adoption selection is bound to another source snapshot')
    rules = [TargetRule.model_validate(r) for r in selection['rules']]
    selected = {r.evidence_id for r in rules}
    allowed = {r.evidence_id for r in collection.batch.validation if r.status == 'accepted'}
    literature = {e.id: e for e in collection.batch.validated}
    chosen = selected & literature.keys()
    if not chosen or not chosen <= allowed:
        raise ValueError('Unreviewed or unsupported literature cannot auto-activate the demo')
    for eid in chosen:
        source = literature[eid]
        doc = documents[source.extensions['literature']['document_id']]
        if doc.source != 'local' or doc.nature != 'demo' or doc.registered_scope is None:
            raise ValueError('Real/unregistered literature is collect-only for this demo')
    if collection.batch.conflicts and (not isinstance(selection['conflict_notes'], str) or not selection['conflict_notes'].strip()):
        raise ValueError('Conflicting candidates require an explicit selection rationale')
    p = profile('hypothetical-nsclc')
    manifest = input_manifest(p, local_inputs)  # Existing M3 source identities stay strict.
    task, schema, config = demo_context()
    task = type(task).model_validate({**task.model_dump(mode='json'), 'original_request': collection.request.target_description})
    for eid in chosen:
        scope = literature[eid].extensions['literature']['scope']
        if scope['endpoint'] != task.endpoint.name or scope['origin'] != task.endpoint.time_origin:
            raise ValueError('Literature source endpoint/origin transport is unsupported, including baseline summaries')
    litset = EvidenceSet(id='literature-evidence', task=task.ref, schema_ref=schema.ref,
                         items=list(literature.values()), extensions={'selection_hash': digest(selection)})
    summary = SummaryProvider(manifest['summary.json']['path']).collect(task, schema)
    ipd = IPDProvider(manifest['reference.csv']['path'], columns=p['columns'],
        summaries={f'e-ecog-{sex}': SummarySpec(metric='proportion', variables=['ecog'], category=1,
                     conditions=[{'variable': 'sex', 'values': [sex]}]) for sex in ('female', 'male')},
        **p['source_metadata']).collect(task, schema)
    evidence = combine(task, schema, [litset, summary, ipd])
    conflicts = find_conflicts(evidence.items)
    if conflicts and (not isinstance(selection['conflict_notes'], str) or not selection['conflict_notes'].strip()):
        raise ValueError('Cross-provider conflicts require an explicit selection rationale')
    # Selection tolerances are configured policy, not extractor fields. Restrict the
    # demo to its six established target definitions/tolerances; only source IDs vary.
    if {r.target_id: r.tolerance.model_dump() for r in rules} != {r['target_id']: r['tolerance'] for r in p['rules']} or len(rules) != len(p['rules']):
        raise ValueError('Demo target tolerances/IDs must match the fixed application policy')
    targets = TargetBuilder().build(task, schema, evidence, rules).targets
    # Confirm adopted summaries match each established target's intended statistic.
    baseline = TargetBuilder().build(task, schema, evidence, [TargetRule.model_validate(r) for r in p['rules']]).targets
    if [t.summary for t in targets.items] != [t.summary for t in baseline.items]:
        raise ValueError('Selected source does not represent the configured target statistic')
    state = StatisticalGenerator().build(task, schema, targets, config=config,
        parameters=Parameters.model_validate(p['parameters']), run_id='hypothetical-literature')
    inputs = tuple(ready_to_run(task, schema, evidence, targets, state, StatisticalGenerator()))
    adoption = AdoptionRecord(policy=selection['policy'], authority=selection['authority'],
        target_description=collection.request.target_description, literature_ids=sorted(chosen),
        rules=[r.model_dump(mode='json') for r in rules], conflicts=conflicts,
        unadopted_ids=[e.id for e in evidence.items if e.id not in selected],
        notes='No pooling or transport. ' + selection['conflict_notes'])
    return inputs, adoption
