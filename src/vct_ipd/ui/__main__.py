"""Start the optional UI on loopback; explicit server-owned fixture/output paths."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from vct_ipd.application.service import WorkflowService


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=8501);p.add_argument('--allow-live',action='store_true')
    a=p.parse_args();WorkflowService(a.fixtures,a.output)
    try:import streamlit  # optional; no import in offline paths
    except ImportError:p.error('Install vct-ipd[ui] to start the local interface')
    env={**os.environ,'VCT_IPD_FIXTURES':str(a.fixtures.absolute()),'VCT_IPD_OUTPUT':str(a.output.absolute()),
         'VCT_IPD_ALLOW_LIVE':'1' if a.allow_live else '0'}
    return subprocess.call([sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('app.py')),
        '--server.address=127.0.0.1','--server.port='+str(a.port),'--server.headless=true',
        '--browser.gatherUsageStats=false'],env=env)

if __name__=='__main__':raise SystemExit(main())
