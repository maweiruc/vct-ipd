"""Controller audit sidecars plus the unchanged canonical-export writer."""
import json
from pathlib import Path
from vct_ipd.reporting.writer import write_run
from vct_ipd.evidence.literature.storage import save_collection
from vct_ipd.generators.statistical import csv_text


def save_session(session,destination):
    if session.result is not None:
        root=write_run(session.result,destination)
    else:
        root=Path(destination);root.mkdir(parents=True,exist_ok=False)
        manifest={'success':False,'validation_status':'ERROR','exported_dataset':None,
                  'export_blocked_reason':session.process_reason+': '+(session.final_error or 'No completed objective'),
                  'best_state':session.epochs[-1].states[session.epochs[-1].best].ref.model_dump() if session.epochs[-1].states else None}
        (root/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    folder=root/'controller';folder.mkdir()
    def save(path,value):
        p=folder/path;p.parent.mkdir(parents=True,exist_ok=True)
        if hasattr(value,'model_dump'):value=value.model_dump(mode='json')
        p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save(Path('policy.json'),session.policy)
    save(Path('budget.json'),{'charged':session.budget,'limits':session.policy.limits.model_dump(),
        'retrieval_accounting':'Reserve worst-case query/page/document/byte/extraction allowances before dispatch; no refunds on failure. Actual calls recorded in each M4 package.',
        'provider_accounting':'Attempt admission reserved before proposal; provider metadata records actual calls. Replay preserves original admission charges without live calls.'})
    save(Path('provider.json'),session.provider)
    for name,obj in zip(('task','schema','evidence','targets','state'),session.initial_inputs):save(Path('initial')/(name+'.json'),obj)
    (folder/'decisions.jsonl').write_text(''.join(r.model_dump_json()+'\n' for r in session.decisions))
    epoch_index=[]
    for ep in session.epochs:
        prefix=Path('epochs')/str(ep.number)
        epoch_index.append({'epoch':ep.number,'parent':ep.parent,'targets':ep.targets.ref.model_dump(),
            'evidence':ep.evidence.ref.model_dump(),'best_state':ep.states[ep.best].ref.model_dump() if ep.states else None,
            'best_report':ep.reports[ep.best].ref.model_dump() if ep.reports else None,'error':ep.error,'evaluated_states':len(ep.states),
            'final_objective':ep is session.epochs[-1]})
        for name,objects in [('evidence',ep.evidence),('targets',ep.targets),
                             ('states',[s.model_dump(mode='json') for s in ep.states]),
                             ('reports',[r.model_dump(mode='json') for r in ep.reports]),
                             ('actions',[a.model_dump(mode='json') for a in ep.actions])]:save(prefix/(name+'.json'),objects)
        if ep.attempted_state is not None:save(prefix/'attempted_state.json',ep.attempted_state)
        # Prior epochs retain raw diagnostics; only the final epoch uses write_run.
        if ep is not session.epochs[-1] or session.result is None:
            for sample in ep.samples:
                p=folder/prefix/sample.metadata.data.uri;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(csv_text(sample.rows,[v.name for v in session.initial_inputs[1].variables]))
                save(prefix/'latent'/(Path(sample.metadata.data.uri).stem+'.json'),sample.latent)
            save(prefix/'datasets.json',[s.metadata.model_dump(mode='json') for s in ep.samples])
    save(Path('epochs.json'),epoch_index)
    for index,request in enumerate(session.evidence_requests):
        path=folder/'evidence_requests'/str(index)
        save_collection(request['collection'],path,adoption=request['adoption'])
        (path/'outcome.json').write_text(json.dumps({'slot':request['slot'],'outcome':request['outcome']},indent=2)+'\n')
    manifest=json.loads((root/'run.json').read_text())
    report=['# Bounded diagnostic controller','',f'Process stop: **{session.process_reason}**.',
        f'Final objective: epoch {session.epochs[-1].number}; selected state: {(manifest.get("best_state") or {}).get("id", "none")}.',
        f'Comparison: {session.epochs[-1].reports[session.epochs[-1].best].status if session.epochs[-1].reports else "ERROR"}; fresh validation: {manifest["validation_status"]}; success: {manifest["success"]}.',
        f'Canonical export: {manifest["exported_dataset"]}; blocked reason: {manifest["export_blocked_reason"]}.',
        f'Provider: {session.provider}.',f'Global charged resources: {session.budget}.','',
        'Hypothetical sources only. Model assumptions remain in generator states. Matching summaries and fresh-seed stability are not clinical validation.',
        'Fresh validation occurs only after a completed comparison objective and never feeds diagnostic contexts; a failed new objective skips it.','',
        '| Step | Epoch | Proposal | Accepted | Reason |','| --- | --- | --- | --- | --- |']
    report += [f'| {r.step} | {r.epoch} | {r.proposed["action"] if r.proposed else "provider error"} | {r.accepted} | {r.reason} |' for r in session.decisions]
    (folder/'report.md').write_text('\n'.join(report)+'\n')
    return root
