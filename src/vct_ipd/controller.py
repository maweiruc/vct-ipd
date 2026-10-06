"""Explicit hypothetical M5 controller CLI. No arbitrary clinical-plan substitution."""
import argparse
import json
import os
from pathlib import Path
from vct_ipd.agents.contracts import ControllerPolicy, SlotPolicy
from vct_ipd.agents.diagnosis import RuleDiagnoser, ReplayDiagnoser, LiveDiagnoser, ScriptedDiagnoser, proposal
from vct_ipd.agents.controller import execute
from vct_ipd.agents.evidence import local_slot
from vct_ipd.agents.storage import save_session
from vct_ipd.demo import demo_context
from vct_ipd.planners.profile import profile,input_manifest
from vct_ipd.planners.bridge import assemble
from vct_ipd.core import CalibrationTargetSet
from vct_ipd.evidence.literature.retrieval import bounded_json
from vct_ipd.llm.providers import ProviderError


def prepare_controller(inputs, mode='tuning', literature=None, selection=None):
    p=profile('hypothetical-nsclc');task,schema,config=demo_context()
    objects=list(assemble(task,schema,config,p,input_manifest(p,inputs)))
    return configure_controller(objects,inputs,mode,literature,selection)


def configure_controller(objects, inputs, mode='tuning', literature=None, selection=None):
    """Apply a predeclared policy to caller-resolved objects without resetting task/config."""
    objects=list(objects)
    policy=ControllerPolicy();slots={}
    if mode=='evidence-gap':
        if literature is None or selection is None:raise ValueError('Evidence-gap demo requires explicit literature directory and selection')
        slot=local_slot(literature,inputs,selection);slots={'preferred-age':slot}
        policy=ControllerPolicy(slots={'preferred-age':SlotPolicy(binding=slot.binding,target_ids=['age'],adoption='hypothetical_preferred')})
        targets=objects[3].model_dump(mode='json')
        for t in targets['items']:
            if t['id']=='age':
                t.update(evidence_limited=True,evidence_limitation='Predeclared preferred fictional source has not yet been adopted')
        objects[3]=CalibrationTargetSet.model_validate(targets)
    elif mode=='budget-stop':policy=ControllerPolicy(limits={'states':2})
    elif mode!='tuning':raise ValueError('Unknown explicitly hypothetical controller mode')
    return objects,policy,slots


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--mode',choices=['tuning','evidence-gap','budget-stop'],default='tuning')
    p.add_argument('--literature',type=Path);p.add_argument('--selection',type=Path)
    p.add_argument('--provider',choices=['rule','mock','replay','openai'],default='rule')
    p.add_argument('--decisions',type=Path);p.add_argument('--model',default=os.environ.get('VCT_IPD_MODEL'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():p.error('Choose a fresh output directory')
    try:
        objects,policy,slots=prepare_controller(a.inputs,a.mode,a.literature,a.selection)
        if a.provider=='replay':
            if not a.decisions:raise ValueError('Replay requires --decisions')
            provider=ReplayDiagnoser(a.decisions)
        elif a.provider=='openai':provider=LiveDiagnoser(a.model)
        elif a.provider=='mock':
            # A human-authored first age patch, then explicitly rule-driven replies.
            rule=RuleDiagnoser()
            provider=ScriptedDiagnoser(lambda c: (proposal(c,'TUNE_PARAMETERS',changes=[{'name':'age_mean','value':65.0}],
                targets=['age'],diagnosis='Human-authored mock: set age mean to 65') if not c.prior_actions and a.mode=='tuning' else rule.propose(c)).model_dump(mode='json'))
        else:provider=RuleDiagnoser()
        session=execute(*objects,policy=policy,diagnoser=provider,slots=slots)
        save_session(session,a.output)
        if session.result is None:
            print(json.dumps({'process_stop':session.process_reason,'success':False,'error':session.final_error}))
            return 3
        print(json.dumps({'process_stop':session.process_reason,'epoch':session.epochs[-1].number,
            'selected':session.result.states[session.result.best_index].id,'fresh_validation':session.result.validation_report.status,
            'success':session.result.success,'budget':session.budget}))
        return 0 if session.result.success else 4
    except (ValueError,OSError,ProviderError) as exc:
        a.output.mkdir(parents=True,exist_ok=False)
        (a.output/'error.json').write_text(json.dumps({'status':'ERROR','reason':str(exc)},indent=2)+'\n')
        print('ERROR: '+str(exc));return 3


if __name__=='__main__':raise SystemExit(main())
