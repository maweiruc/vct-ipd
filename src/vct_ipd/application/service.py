"""One local run per frozen token, using the accepted M3 -> M5 boundaries."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Literal
from pydantic import Field
from vct_ipd.planners.contracts import StrictModel, PlanningRequest
from vct_ipd.planners.planner import plan
from vct_ipd.planners.artifacts import revalidate, save_planning
from vct_ipd.planners.bridge import assemble
from vct_ipd.controller import configure_controller
from vct_ipd.agents.controller import execute
from vct_ipd.agents.storage import save_session
from vct_ipd.agents.diagnosis import LiveDiagnoser
from vct_ipd.llm.providers import ReplayProvider, ProviderError
from vct_ipd.llm.openai_adapter import OpenAIProvider
from vct_ipd.evidence.literature.contracts import EvidenceRequest
from vct_ipd.evidence.literature.retrieval import LocalDocuments, LocalManifestError, EuropePMC, bounded_json
from vct_ipd.evidence.literature.extraction import ReplayExtractor, LiveExtractor
from vct_ipd.evidence.literature.service import collect
from vct_ipd.evidence.literature.storage import save_collection

class ApplicationError(ValueError):
    """Safe, fixed application messages (never raw transport errors)."""


EXAMPLES = ('english','chinese','defaults','blocked-population','blocked-backend','blocked-followup','small')


# These are operation outcomes, not judgments of scientific/evidence support.
# Only fixed messages enter the visible error; original diagnostics stay in M4.
COLLECTION_ERRORS = {
    'transport_error': 'Collection transport failed.',
    'server_error': 'Collection service returned a server error.',
    'timeout': 'Collection request timed out.',
    'http_error': 'Collection request failed with an HTTP error.',
    'not_found': 'Collection service endpoint was not found (HTTP 404).',
    'rate_limited': 'Collection service rate limit reached; no automatic retry.',
    'malformed_response': 'Collection service returned a malformed response.',
    'extraction_error': 'Evidence extraction failed; retrieved material remains diagnostic only.',
    'byte_limit': 'Collection response exceeded the byte budget; retrieval is incomplete.',
}
COLLECTION_REVIEW_STATUSES = frozenset({
    'zero_hits', 'unavailable_text', 'retrieved_only', 'validated_evidence',
    'needs_review', 'conflicting_evidence', 'unsupported', 'rejected', 'no_statistics',
})


def safe_path(root, relative):
    """Only local, non-symlink descendants; no model-selected paths."""
    root=Path(root).absolute(); part=Path(relative)
    if root.is_symlink():raise ValueError('Symlink roots are not supported')
    root=root.resolve()
    if part.is_absolute() or '..' in part.parts:raise ValueError('Unsafe artifact path')
    path=root/part
    for parent in (path,*path.parents):
        if parent==root:break
        if parent.is_symlink():raise ValueError('Symlink paths are not supported')
    if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Path outside approved root')
    return path


class RunConfig(StrictModel):
    workflow: Literal['generate','collect'] = 'generate'
    text: str = Field(min_length=1,max_length=8000)
    example: Literal['english','chinese','defaults','blocked-population','blocked-backend','blocked-followup','small'] = 'english'
    profile: Literal['hypothetical-nsclc'] = 'hypothetical-nsclc'
    provider: Literal['offline','live'] = 'offline'
    policy: Literal['tuning','evidence-gap'] = 'tuning'
    collection: Literal['fictional','ambiguous','public'] = 'fictional'
    live_extraction: bool = False
    live_diagnosis: bool = False


@dataclass
class Outcome:
    token: str
    config: RunConfig
    root: Path
    status: str = 'planning'
    events: list = field(default_factory=list)
    plan: object = None
    session: object = None
    error: str | None = None


class WorkflowService:
    def __init__(self, fixtures, output, *, model=None, allow_live=False):
        self.fixtures=Path(fixtures).absolute(); self.output=Path(output).absolute()
        self.fixtures=safe_path(self.fixtures,'.');self.output=safe_path(self.output,'.')
        if self.output.is_relative_to(self.fixtures):raise ValueError('Outputs must be outside the input fixture root')
        self.model=model; self.allow_live=allow_live
        self.runs={};self.active=None

    @property
    def inputs(self):return safe_path(self.fixtures,'nsclc_os')

    def example_path(self,name,suffix):
        if name not in EXAMPLES:raise ValueError('Unknown configured example')
        return safe_path(self.fixtures,('ui' if name=='small' else 'planning')+'/'+name+suffix)

    def example_text(self,name):return self.example_path(name,'.txt').read_text().strip()

    def collection_request(self,mode):
        name='real-scope-request.json' if mode=='ambiguous' else 'request.json'
        return EvidenceRequest.model_validate(bounded_json(safe_path(self.fixtures,'literature/'+name)))

    def make_plan(self,config):
        if config.provider=='live':
            self.require_live();provider=OpenAIProvider(self.model)
        else:provider=ReplayProvider(self.example_path(config.example,'.json'))
        return plan(PlanningRequest(text=config.text,profile=config.profile),provider,self.inputs)

    def require_live(self):
        if not self.allow_live or not self.model:raise ApplicationError('Live provider unavailable: configure explicit live permission and VCT_IPD_MODEL; no canned fallback')

    def execute_plan(self,planned,config,*,on_event=None,diagnoser=None,evaluator=None):
        checked,p=revalidate(planned,self.inputs)
        objects=assemble(checked.task,checked.data_schema,checked.execution,p,checked.input_manifest)
        objects,policy,slots=configure_controller(objects,self.inputs,config.policy,
            safe_path(self.fixtures,'literature'),safe_path(self.fixtures,'controller/preferred-age.json'))
        if on_event:on_event({'stage':'ready','normalized':checked.normalized,
            'requested_states':checked.task.calibration.max_iterations,
            'effective_states':min(policy.limits.states,checked.task.calibration.max_iterations),
            'sample_size':checked.task.sample_size,'observation':checked.execution.observation.model_dump(mode='json'),
            'defaults_and_assumptions':[p.model_dump(mode='json') for p in checked.provenance]})
        if config.live_diagnosis:
            self.require_live();diagnoser=LiveDiagnoser(self.model)
        return execute(*objects,policy=policy,slots=slots,diagnoser=diagnoser,evaluator=evaluator,on_event=on_event)

    def run_once(self,token,config,*,on_event=None,diagnoser=None,evaluator=None):
        config=RunConfig.model_validate_json(config.model_dump_json())
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',token):raise ValueError('Invalid run token')
        if self.active is not None:raise ValueError('A local run is already active')
        if token in self.runs:
            old=self.runs[token]
            if old.config!=config:raise ValueError('Run configuration is frozen; explicitly start a new request')
            return old
        root=safe_path(self.output,token)
        if root.exists():raise ValueError('Output exists; choose a new run token')
        outcome=Outcome(token,config,root);self.runs[token]=outcome;self.active=token
        def emit(event):
            event=json.loads(json.dumps(event))
            outcome.events.append(event);outcome.status=event['stage']
            if on_event:
                try:on_event(json.loads(json.dumps(event)))
                except Exception:outcome.events.append({'stage':'observer_error','reason':'Display observer failed; scientific execution unchanged'})
        try:
            if config.workflow=='generate':
                emit({'stage':'planning'})
                outcome.plan=self.make_plan(config)
                if outcome.plan.status!='READY':
                    root.mkdir(parents=True);emit({'stage':'error' if outcome.plan.status=='ERROR' else 'blocked'})
                else:
                    outcome.session=self.execute_plan(outcome.plan,config,on_event=emit,diagnoser=diagnoser,evaluator=evaluator)
                    save_session(outcome.session,root)
                save_planning(outcome.plan,root)
            else:
                emit({'stage':'evidence_collection'})
                request=self.collection_request(config.collection)
                if config.collection=='public':
                    if not self.allow_live:raise ApplicationError('Public retrieval is not enabled in this local application')
                    request=EvidenceRequest(target_description=config.text,statistics=['median OS'],queries=[config.text],
                        allowed_sources=['europe_pmc'],max_extraction_calls=int(config.live_extraction))
                    retriever=EuropePMC(cache_text=False);extractor=None
                    if config.live_extraction:self.require_live();extractor=LiveExtractor(self.model)
                else:
                    if config.text!=request.target_description:raise ApplicationError('Offline collection is bound to the displayed example request; edited text requires configured public retrieval')
                    prefix='ambiguous-' if config.collection=='ambiguous' else ''
                    retriever=LocalDocuments(safe_path(self.fixtures,'literature/'+prefix+'sources.json'))
                    extractor=ReplayExtractor(safe_path(self.fixtures,'literature/'+prefix+'extraction.json'))
                collection=collect(request,retriever,extractor)
                save_collection(collection,root/'literature')
                if collection.status in COLLECTION_ERRORS:
                    raise ApplicationError(COLLECTION_ERRORS[collection.status])
                if collection.status not in COLLECTION_REVIEW_STATUSES:
                    raise ApplicationError('Unrecognized collection outcome; inspect the saved diagnostics.')
                emit({'stage':'review','collection_status':collection.status,'adoption':'none; collect-only'})
        except (ValueError,OSError,ProviderError) as exc:
            # Preserve unexpected programming exceptions for debugging, not false success.
            # External error text is not exposed: it can contain URLs/credentials.
            outcome.error=(str(exc) if isinstance(exc,(ApplicationError,LocalManifestError)) else
                (exc.code if isinstance(exc,ProviderError) else type(exc).__name__)+': input/provider/workflow failed; inspect validated diagnostics')
            root.mkdir(parents=True,exist_ok=True)
            (root/'error.json').write_text(json.dumps({'status':'ERROR','reason':outcome.error}))
            emit({'stage':'error','reason':outcome.error})
            if outcome.plan is not None and not (root/'planning').exists():save_planning(outcome.plan,root)
        finally:self.active=None
        sidecar={'version':'m6-v1','token':token,'config':config.model_dump(mode='json'),'status':outcome.status,
                 'events':outcome.events,'error':outcome.error,'live_enabled':self.allow_live,
                 'verification':'Live calls are opt-in; provider metadata records actual mode. Offline examples are human-authored mocks.'}
        (root/'application.json').write_text(json.dumps(sidecar,indent=2,ensure_ascii=False)+'\n')
        from .results import seal, load_result
        seal(root)
        view=load_result(self.output,token)
        (root/'report.html').write_text(view.html())
        return outcome
