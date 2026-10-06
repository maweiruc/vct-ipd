"""Offline integrated demo / explicit live smoke using the same UI service."""
import argparse
import json
import os
from pathlib import Path
from .service import WorkflowService,RunConfig,EXAMPLES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--token',required=True);p.add_argument('--example',choices=EXAMPLES,default='english')
    p.add_argument('--policy',choices=['tuning','evidence-gap'],default='tuning')
    p.add_argument('--collect',choices=['fictional','ambiguous','public'])
    p.add_argument('--live',action='store_true',help='Explicitly authorize configured live planning/public retrieval')
    p.add_argument('--live-extraction',action='store_true');p.add_argument('--live-diagnosis',action='store_true')
    p.add_argument('--text',help='Free-form request; offline replay remains bound')
    a=p.parse_args();s=WorkflowService(a.fixtures,a.output,model=os.environ.get('VCT_IPD_MODEL'),allow_live=a.live)
    text=a.text or (s.collection_request(a.collect).target_description if a.collect else s.example_text(a.example))
    r=s.run_once(a.token,RunConfig(text=text,example=a.example,policy=a.policy,
        workflow='collect' if a.collect else 'generate',collection=a.collect or 'fictional',
        provider='live' if a.live else 'offline',live_extraction=a.live_extraction,live_diagnosis=a.live_diagnosis))
    print(json.dumps({'token':r.token,'status':r.status,'output':str(r.root),
        'stop':r.session.process_reason if r.session else None,
        'validation':str(r.session.result.validation_report.status) if r.session and r.session.result else None}))
    return 0 if r.status in ('completed','review') else 4

if __name__=='__main__':raise SystemExit(main())
