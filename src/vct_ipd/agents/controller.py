"""Comparison-only decision loop; finalization is beyond its one-way boundary."""
from dataclasses import dataclass, field
import warnings
from .contracts import ControllerPolicy, DiagnosticContext, DecisionDraft, DecisionRecord
from .diagnosis import RuleDiagnoser, PROMPT
from .evidence import adopt
from vct_ipd.core import Action, GeneratorState
from vct_ipd.generators.statistical import StatisticalGenerator, Parameters, snapshot
from vct_ipd.evaluation.runner import Evaluator
from vct_ipd.calibration.search import better, feasible
from vct_ipd.pipeline.runner import ready_to_run, evaluate_state, finalize
from vct_ipd.planners.profile import digest
from vct_ipd.llm.providers import ProviderError, safe_metadata


@dataclass
class Epoch:
    number: int
    evidence: object
    targets: object
    parent: dict | None = None
    states: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    best: int = 0
    error: str | None = None
    attempted_state: object | None = None


@dataclass
class Session:
    policy: ControllerPolicy
    epochs: list
    decisions: list
    budget: dict
    evidence_requests: list
    process_reason: str
    result: object
    provider: dict
    initial_inputs: list
    final_error: str | None = None


def context(task,schema,ep,policy,used,slots_used,records,limits=None):
    state,report=ep.states[ep.best],ep.reports[ep.best]
    return DiagnosticContext(epoch=ep.number,
        binding={'policy':digest(policy),'task':digest(task),'schema':digest(schema),'evidence':digest(ep.evidence),
                 'targets':digest(ep.targets),'state':digest(state),'report':digest(report),
                 'prompt':digest(PROMPT),'decision_schema':digest(DecisionDraft.model_json_schema()),
                 'allowed_parameters':{k:v.model_dump() for k,v in policy.parameters.items()}},
        state={'id':state.id,'ref':state.ref.model_dump(),'parameters':dict(state.parameters)},
        report={'id':report.id,'status':report.status,'loss':report.overall_metrics['loss'],
                'results':[r.model_dump(mode='json') for r in report.results]},
        targets=[t.model_dump(mode='json') for t in ep.targets.items],
        evidence=[{'id':e.id,'nature':e.provenance.nature,'source_type':e.provenance.source_type,
                   'estimate_status':e.estimate_status} for e in ep.evidence.items],
        assumptions=[a.model_dump() for a in state.assumptions],
        remaining={k:v-used[k] for k,v in (limits or policy.limits.model_dump()).items()},
        slots={k:v.model_dump() for k,v in policy.slots.items()},used_slots=sorted(slots_used),
        prior_actions=[{'action':r.proposed['action'] if r.proposed else None,'accepted':r.accepted,'reason':r.reason} for r in records[-8:]])


def validate_decision(d,c,policy):
    d=DecisionDraft.model_validate_json(d.model_dump_json())
    if (d.context_hash!=digest(c) or d.base_state!=c.state['id'] or d.report!=c.report['id'] or d.target_set!=c.binding['targets']):
        raise ValueError('Fabricated or stale context/state/report/target reference')
    if d.action not in policy.allowed_actions or d.action not in ('TUNE_PARAMETERS','RETRIEVE_EVIDENCE','STOP'):
        raise ValueError('Deferred/unsupported action; no schema/model/target editing executor')
    if not set(d.affected_targets)<= {t['id'] for t in c.targets}:raise ValueError('Unknown affected target')
    if d.action=='TUNE_PARAMETERS':
        if not d.changes or d.slot is not None or d.stop_reason is not None:raise ValueError('Invalid tune payload')
        if len({p.name for p in d.changes})!=len(d.changes):raise ValueError('Duplicate parameter patches')
        values=dict(c.state['parameters'])
        for change in d.changes:
            bound=policy.parameters.get(change.name)
            if bound is None or not bound.lower<=change.value<=bound.upper:raise ValueError('Parameter unknown or outside policy bounds')
            values[change.name]=change.value
        values=Parameters.model_validate(values).model_dump(mode='json')
        if values==c.state['parameters']:raise ValueError('No-op parameter proposal')
        return values
    if d.changes:raise ValueError('Only tuning may contain parameters')
    if d.action=='RETRIEVE_EVIDENCE':
        if d.slot not in policy.slots or d.slot in c.used_slots or d.stop_reason is not None:raise ValueError('Unknown or already used evidence slot')
        if set(d.affected_targets)!=set(policy.slots[d.slot].target_ids):raise ValueError('Evidence request must identify its configured gap')
    elif d.slot is not None or d.stop_reason not in ('calibrated','no_improvement','unresolved_evidence','unavailable_metric','unsupported_metric','unsupported_parameter','structural_failure','budget_exhausted'):
        raise ValueError('Unsupported STOP payload')
    elif d.stop_reason=='calibrated' and (c.report['status']!='PASS' or any(s.required and k not in c.used_slots for k,s in policy.slots.items())):
        raise ValueError('Calibrated STOP requires all comparison PASS and resolved slots')
    return None


def execute(task,schema,evidence,targets,state,*,policy=None,diagnoser=None,slots=None,generator=None,evaluator=None,on_event=None):
    def emit(stage, **detail):
        if on_event is not None:
            try:on_event({'stage':stage,**detail})
            except Exception:
                # Observability must never alter selection or trigger a holdout retry.
                warnings.warn('Progress observer failed; execution continues',RuntimeWarning)
    policy=snapshot(policy or ControllerPolicy());diagnoser=diagnoser or RuleDiagnoser();slots=slots or {}
    generator=generator or StatisticalGenerator();evaluator=evaluator or Evaluator()
    task,schema,evidence,targets,state=ready_to_run(task,schema,evidence,targets,state,generator)
    if task.sample_size>policy.max_sample_size:raise ValueError('Sample-size budget exceeded')
    if set(policy.slots)!=set(slots):raise ValueError('Configured evidence slots do not match policy')
    for name,spec in policy.slots.items():
        if slots[name].binding!=spec.binding or not set(spec.target_ids)<={t.id for t in targets.items}:
            raise ValueError('Slot identity/targets not bound to policy')
    for name,bound in policy.parameters.items():
        if name not in state.parameters or not bound.lower<=state.parameters[name]<=bound.upper:raise ValueError('Initial parameter outside policy')
        for value in (bound.lower,bound.upper):Parameters.model_validate({**state.parameters,name:value})
    initial=[snapshot(x) for x in (task,schema,evidence,targets,state)]
    used={k:0 for k in policy.limits.model_dump()};limits=policy.limits.model_dump()
    limits['states']=min(limits['states'],task.calibration.max_iterations)
    records=[];epochs=[Epoch(0,evidence,targets)];ep=epochs[0];requests=[];slots_used=set()
    stale=rejects=0;reason=None
    def charge(costs):
        if any(type(v) is not int or v<0 for v in costs.values()):raise ValueError('Invalid resource charge')
        if any(used[k]+v>limits[k] for k,v in costs.items()):raise ValueError('Global resource budget exhausted')
        for k,v in costs.items():used[k]+=v
    def compare(candidate):
        nonlocal stale
        charge({'states':1})
        emit('comparison',epoch=ep.number,state=candidate.id,charged_states=used['states'])
        report,replicas=evaluate_state(task,schema,ep.evidence,ep.targets,candidate,generator,evaluator)
        ep.states.append(snapshot(candidate));ep.reports.append(snapshot(report));ep.samples.extend(replicas)
        if len(ep.states)==1 or better(report,ep.reports[ep.best],ep.targets):ep.best=len(ep.states)-1;stale=0
        else:stale+=1
        emit('comparison_complete',epoch=ep.number,state=candidate.id,status=str(report.status),loss=report.overall_metrics['loss'])
    compare(state)
    while reason is None:
        best,best_report=ep.states[ep.best],ep.reports[ep.best]
        pending=any(s.required and name not in slots_used for name,s in policy.slots.items())
        if best_report.status=='PASS' and not pending:reason='calibrated';break
        if used['states']>=limits['states'] or used['decisions']>=limits['decisions']:reason='budget_exhausted';break
        if stale>=policy.patience:reason='no_improvement';break
        call_charge=getattr(diagnoser,'call_cost',0 if isinstance(diagnoser,RuleDiagnoser) else 1)
        if used['provider_calls']+call_charge>limits['provider_calls']:reason='budget_exhausted';break
        charge({'decisions':1})
        c=context(task,schema,ep,policy,used,slots_used,records,limits)
        raw=None;action=None;accepted=False;message='';next_state=None;next_epoch=None
        try:
            charge({'provider_calls':call_charge})
            raw=diagnoser.propose(snapshot(c))
            parameters=validate_decision(raw,c,policy)
            if raw.action=='TUNE_PARAMETERS':
                action=Action(id=f'{best.run_id}-decision-{used["decisions"]}',type='TUNE_PARAMETERS',
                    payload={'component':'statistical','parameters':parameters},reason=raw.diagnosis,
                    evaluation_report=best_report.ref,generator_state=best.ref,run_id=best.run_id,iteration=best.iteration,
                    extensions={'proposal_base':best.ref.model_dump(),'context_hash':digest(c)})
                action.validate_references(best_report,best)
                candidate=generator.update(snapshot(best),action,snapshot(best_report))
                # Chronological lineage is distinct from the explicit proposal base.
                latest=ep.states[-1]
                candidate=GeneratorState.model_validate({**candidate.model_dump(mode='json'),
                    'id':f'{best.run_id}-g{len(ep.states)}','iteration':latest.iteration+1,'parent_state':latest.ref.model_dump()})
                candidate.validate_parent(latest)
                if any(s.parameters==candidate.parameters for s in ep.states):raise ValueError('Repeated parameter state in this epoch')
                compare(candidate);next_state=candidate.ref.model_dump()
            elif raw.action=='RETRIEVE_EVIDENCE':
                slot=slots[raw.slot];spec=policy.slots[raw.slot]
                if spec.adoption=='hypothetical_preferred' and used['transitions']>=limits['transitions']:raise ValueError('Global epoch transition budget exhausted')
                charge(slot.costs());slots_used.add(raw.slot)
                variables=sorted({v for t in ep.targets.items if t.id in raw.affected_targets for v in t.summary.variables})
                action=Action(id=f'{best.run_id}-evidence-{used["decisions"]}',type='RETRIEVE_EVIDENCE',
                    payload={'query':slot.request.target_description,'variables':variables,'max_sources':slot.request.max_documents,'endpoint':'OS'},
                    reason=raw.diagnosis,evaluation_report=best_report.ref,generator_state=best.ref,run_id=best.run_id,iteration=best.iteration)
                action.validate_references(best_report,best)
                emit('evidence_collection',epoch=ep.number,slot=raw.slot)
                collected=slot.collect()
                request_record={'slot':raw.slot,'collection':collected,'adoption':None,'outcome':collected.status}
                requests.append(request_record)
                adopted=adopt(slot,collected,task,schema,ep.evidence,ep.targets,spec,len(epochs)) if spec.adoption=='hypothetical_preferred' else None
                if adopted:
                    ev,ts,adoption_record=adopted
                    charge({'transitions':1})
                    request_record.update(adoption=adoption_record,outcome='adopted')
                    ep.actions.append(snapshot(action))
                    parent={'epoch':ep.number,'state':best.ref.model_dump(),'report':best_report.ref.model_dump(),'decision':used['decisions']}
                    emit('target_transition',from_epoch=ep.number,to_epoch=len(epochs))
                    ep=Epoch(len(epochs),snapshot(ev),snapshot(ts),parent);epochs.append(ep)
                    data=best.model_dump(mode='json');run_id=initial[4].run_id+f'-epoch{ep.number}'
                    data.update(id=run_id+'-g0',run_id=run_id,iteration=0,parent_state=None,target_set=ts.ref.model_dump(),modification_reason='Reevaluate previous best under newly adopted fixed targets')
                    candidate=GeneratorState.model_validate(data)
                    ep.attempted_state=snapshot(candidate)
                    ready_to_run(task,schema,ev,ts,candidate,generator)
                    compare(candidate);next_state=candidate.ref.model_dump();next_epoch=ep.number
                    action=None  # Retrieval action belongs to the parent epoch.
                elif spec.required:reason='unresolved_evidence'
            else:reason=raw.stop_reason
            accepted=True;message='Applied' if reason is None else reason;rejects=0
            if action:ep.actions.append(snapshot(action))
        except ProviderError as exc:
            message=exc.code+': '+str(exc);reason='provider_error'
        except (ValueError, OSError) as exc:
            message=str(exc);rejects+=1
            if not ep.states:
                ep.error=message;reason='comparison_error'
            elif 'budget' in message.lower():reason='budget_exhausted'
            elif raw is not None and raw.action=='RETRIEVE_EVIDENCE':reason='unresolved_evidence'
            elif rejects>=policy.max_rejections:reason='invalid_actions'
        records.append(DecisionRecord(step=used['decisions'],epoch=c.epoch,context_hash=digest(c),context=c,
            proposed=raw.model_dump(mode='json') if raw else None,accepted=accepted,reason=message,
            applied_action=(action.model_dump(mode='json') if action else epochs[c.epoch].actions[-1].model_dump(mode='json') if next_epoch is not None else None) if accepted else None,
            provider={'budget_call_charge':call_charge,'metadata':safe_metadata(diagnoser)},budget=dict(used),
            resulting_state=next_state,resulting_epoch=next_epoch))
    if not ep.states:
        emit('error',epoch=ep.number,reason=reason)
        return Session(policy,epochs,records,used,requests,reason,None,safe_metadata(diagnoser),initial,ep.error)
    # The decision loop ends permanently here. Never expose this output to a proposer.
    best,best_report=ep.states[ep.best],ep.reports[ep.best]
    core_reason=reason if reason in ('calibrated','no_improvement','budget_exhausted','structural_failure') else 'evidence_limited' if reason=='unresolved_evidence' else 'budget_exhausted'
    if not feasible(best_report,ep.targets):core_reason='structural_failure'
    stop=Action(id=best.run_id+'-stop',type='STOP',payload={'reason':core_reason},reason='Controller termination: '+reason,
        evaluation_report=best_report.ref,generator_state=best.ref,run_id=best.run_id,iteration=best.iteration)
    stop.validate_references(best_report,best);ep.actions.append(snapshot(stop))
    if records and records[-1].accepted and records[-1].proposed['action']=='STOP':
        records[-1]=DecisionRecord.model_validate({**records[-1].model_dump(mode='json'),
                                                  'applied_action':stop.model_dump(mode='json')})
    error=None
    try:
        emit('final_validation',epoch=ep.number,state=best.id)
        result=finalize(task,schema,ep.evidence,ep.targets,ep.states,ep.reports,ep.actions,ep.samples,ep.best,core_reason,
            {'implementation':'vct_ipd.agents.controller','policy':digest(policy),'global_evaluated_states':used['states']},generator,evaluator)
    except (ValueError,OSError) as exc:
        # No fabricated final report and no retry or alternative state selection.
        result=None;error=str(exc);reason='finalization_error'
    emit('error' if result is None else 'completed' if result.success else 'partial',epoch=ep.number,reason=reason)
    return Session(policy,epochs,records,used,requests,reason,result,safe_metadata(diagnoser),initial,error)
