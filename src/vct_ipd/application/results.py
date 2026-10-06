"""Validated local artifact view and fixed-allowlist downloads; no URI dereferencing."""
import csv
from dataclasses import dataclass
import hashlib
from html import escape
import io
import json
import re
import zipfile
from vct_ipd.core import TaskSpec,DataSchema,EvidenceSet,CalibrationTargetSet,GeneratorState,EvaluationReport,SyntheticDataset
from vct_ipd.calibration.search import feasible
from vct_ipd.planners.contracts import PlanningResult
from .service import safe_path,RunConfig

BASE=('application.json','error.json','run.json','task.json','schema.json','evidence.json','targets.json',
      'generator/best.json','generator/states.json','history/actions.json','datasets/manifest.json',
      'evaluation/initial.json','evaluation/best.json','evaluation/final_validation.json',
      'controller/policy.json','controller/budget.json','controller/provider.json','controller/epochs.json','controller/decisions.jsonl')
PLAN=tuple('planning/'+p for p in ('request.txt','draft.json','result.json','provenance.json','provider_metadata.json','input_manifest.json','integrity.json'))
LITERATURE=('request.json','retrieval/searches.json','retrieval/documents.json','extraction/drafts.json',
            'extraction/provider.json','extraction/validation.json','evidence.json','adoption.json','outcome.json')


def allowed(name):
    return (name in BASE+PLAN or name in ('synthetic_ipd.csv',) or
        name in tuple('literature/'+s for s in LITERATURE) or
        re.fullmatch(r'controller/epochs/[0-9]{1,3}/(evidence|targets|states|reports|actions|attempted_state)\.json',name) is not None or
        any(re.fullmatch(r'controller/evidence_requests/[0-9]{1,3}/'+re.escape(s),name) for s in LITERATURE) or
        re.fullmatch(r'controller/initial/(task|schema|evidence|targets|state)\.json',name) is not None)


def payload(path):
    if path.stat().st_size>10_000_000:raise ValueError('Oversized artifact')
    return path.read_bytes()


def seal(root):
    """Hash only known output artifacts. This detects changes, not malicious re-signing."""
    files={}
    for p in root.rglob('*'):
        name=p.relative_to(root).as_posix()
        if allowed(name) and p.is_file():files[name]=hashlib.sha256(payload(safe_path(root,name))).hexdigest()
    safe_path(root,'application-integrity.json').write_text(json.dumps(files,indent=2)+'\n')


@dataclass
class ResultView:
    token: str
    status: str
    sections: dict
    files: dict
    cohort: bytes | None
    preview: list

    def target_rows(self):
        rows=[]
        for item in self.sections.get('Target results',[]):
            rows.append({'Target':item['target'],'Sources':', '.join(item['evidence_ids']),
                'Desired':json.dumps(item['desired']),'Unit':item['summary']['unit'] or 'dimensionless/check',
                'Best':str(item['comparison']['observed']),'Fresh':str(item['fresh_validation']['observed']),
                'Availability':item['fresh_validation']['estimate_status'],'Status':item['fresh_validation']['status']})
        return rows

    def html(self):
        blocks=['<!doctype html><html lang="en"><meta charset="utf-8"><title>VCT-IPD local research report</title>',
                '<style>body{max-width:1000px;margin:2em auto;font-family:system-ui;padding:1em}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{border-top:1px solid #ccc;padding-top:1em}</style>',
                '<h1>VCT-IPD · hypothetical research prototype</h1>',
                '<p>No clinical validation. Holdout seeds measure Monte Carlo stability. A CSV is one fresh replica; reported statistics aggregate per-replica estimates.</p>',
                '<p><strong>'+escape(self.status)+'</strong></p>']
        rows=self.target_rows()
        if rows:
            blocks+=['<h2>Target comparison / fresh validation</h2><table><tr>'+''.join('<th>'+escape(k)+'</th>' for k in rows[0])+'</tr>']
            blocks+=['<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row.values())+'</tr>' for row in rows]
            blocks+=['</table>']
        for name,value in self.sections.items():
            blocks+=['<h2>'+escape(name)+'</h2><pre>'+escape(json.dumps(value,indent=2,ensure_ascii=False))+'</pre>']
        return '\n'.join(blocks)+ '</html>'

    def bundle(self):
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,content in self.files.items():
                if name=='synthetic_ipd.csv' and self.cohort is None:continue
                z.writestr(name,content)
            z.writestr('report.html',self.html())
        return buffer.getvalue()


def load_result(output,token):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',token):raise ValueError('Invalid run identity')
    root=safe_path(output,token)
    hashes=json.loads(payload(safe_path(root,'application-integrity.json')))
    if not isinstance(hashes,dict) or not hashes or len(hashes)>1000:raise ValueError('Malformed package integrity')
    files={}
    for name,digest in hashes.items():
        if not allowed(name) or not isinstance(digest,str):raise ValueError('Unapproved package member')
        data=payload(safe_path(root,name))
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Changed artifact; downloads blocked')
        files[name]=data
    if sum(map(len,files.values()))>50_000_000:raise ValueError('Package too large')
    def read(name):return json.loads(files[name])
    app=read('application.json');RunConfig.model_validate(app['config'])
    if app['version']!='m6-v1' or app['token']!=token:raise ValueError('Run identity mismatch')
    if app['status'] not in ('completed','partial','error','blocked','review'):raise ValueError('Incomplete application package')
    if app['status'] in ('completed','partial'):
        required={'run.json','task.json','schema.json','evidence.json','targets.json','generator/best.json','evaluation/best.json','evaluation/final_validation.json','planning/result.json','datasets/manifest.json'}
        if not required<=files.keys():raise ValueError('Incomplete execution package')
    sections={'Request and workflow':app};cohort=None;preview=[]
    if 'planning/result.json' in files:
        plan=PlanningResult.model_validate_json(files['planning/result.json'])
        sections['Resolved plan, defaults and issues']=plan.model_dump(mode='json')
    if 'run.json' in files:
        manifest=read('run.json');sections['Selection and export']=manifest
        if (app['status']=='completed') != bool(manifest['success']):raise ValueError('Application success mismatch')
        if manifest.get('exported_dataset') is not None and app['status'] not in ('completed','partial'):
            raise ValueError('Failed/blocked workflow cannot advertise a cohort')
        if 'task.json' in files:
            task=TaskSpec.model_validate_json(files['task.json']);schema=DataSchema.model_validate_json(files['schema.json'])
            evidence=EvidenceSet.model_validate_json(files['evidence.json']);targets=CalibrationTargetSet.model_validate_json(files['targets.json'])
            state=GeneratorState.model_validate_json(files['generator/best.json'])
            if plan.status!='READY' or plan.task!=task or plan.data_schema!=schema:raise ValueError('Runtime task differs from resolved plan')
            evidence.validate_references(schema);targets.validate_references(evidence,schema);targets.validate_task(task,schema)
            state.validate_references(task,schema,targets)
            if manifest['best_state']!=state.ref.model_dump() or manifest['run_id']!=state.run_id:raise ValueError('Selected state mismatch')
            datasets=[SyntheticDataset.model_validate(d) for d in read('datasets/manifest.json')]
            best=EvaluationReport.model_validate_json(files['evaluation/best.json'])
            final=EvaluationReport.model_validate_json(files['evaluation/final_validation.json'])
            for report in (best,final):
                selected=[d for d in datasets if d.ref in report.datasets]
                for d in selected:d.validate_references(state,schema)
                report.validate_references(state,targets,selected)
            if manifest['validation_status']!=final.status:raise ValueError('Validation status mismatch')
            if manifest['success'] and (best.status!='PASS' or final.status!='PASS' or manifest['calibration_stop_reason']!='calibrated'):
                raise ValueError('False success manifest')
            rows=[];b={r.target.id:r for r in best.results};f={r.target.id:r for r in final.results}
            for t in targets.items:
                rows.append({'target':t.id,'summary':t.summary.model_dump(mode='json'),'desired':t.desired.model_dump(mode='json'),
                    'tolerance':t.tolerance.model_dump(mode='json') if t.tolerance else None,'evidence_ids':t.evidence_ids,
                    'comparison':b[t.id].model_dump(mode='json'),'fresh_validation':f[t.id].model_dump(mode='json')})
            sections.update({'Target results':rows,'Data dictionary':[v.model_dump(mode='json') for v in schema.variables],
                'Source Evidence and adoption rationale':evidence.model_dump(mode='json'),'Canonical targets':targets.model_dump(mode='json'),
                'Generator settings and assumptions':state.model_dump(mode='json')})
            if manifest['exported_dataset'] is not None:
                if not manifest['best_feasible'] or not feasible(best,targets) or not feasible(final,targets) or manifest['export_blocked_reason'] is not None:
                    raise ValueError('Canonical export gate is closed')
                matches=[d for d in datasets if d.ref.model_dump()==manifest['exported_dataset'] and d.ref in final.datasets]
                if len(matches)!=1:raise ValueError('Exported dataset reference mismatch')
                d=matches[0];cohort=files['synthetic_ipd.csv']
                if hashlib.sha256(cohort).hexdigest()!=d.data.content_hash:raise ValueError('Canonical data hash mismatch')
                reader=csv.DictReader(io.StringIO(cohort.decode()));actual=list(reader)
                if reader.fieldnames!=[v.name for v in schema.variables] or len(actual)!=task.sample_size or len(actual)!=d.sample_size:
                    raise ValueError('Canonical dimensions mismatch')
                preview=actual[:8];sections['Exported replica']={'rows':len(actual),'columns':reader.fieldnames,'seed':d.seed,'preview':preview}
        elif manifest.get('exported_dataset') is not None:raise ValueError('Cohort lacks validated contracts')
    for name in files:
        if name.startswith('controller/') and name.endswith('.json') and ('/initial/' not in name):sections[name]=read(name)
        elif name=='controller/decisions.jsonl':sections['Controller decisions']=[json.loads(s) for s in files[name].decode().splitlines()]
        elif name.startswith('literature/'):sections[name]=read(name)
    if 'error.json' in files:sections['Error']=read('error.json')
    status=app['status']
    if cohort is not None and not sections['Selection and export']['success']:status='PARTIAL — permitted hard-valid cohort; NOT FULLY CALIBRATED; success=false'
    return ResultView(token,status,sections,files,cohort,preview)
