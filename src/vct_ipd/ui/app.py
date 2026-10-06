"""Streamlit presentation only; all numerical work stays in accepted components."""
import os
import uuid
import streamlit as st
from vct_ipd.application.service import WorkflowService,RunConfig,EXAMPLES
from vct_ipd.application.results import load_result

st.set_page_config(page_title='VCT-IPD · local research',layout='wide')
st.title('VCT-IPD · local research prototype')
st.warning('One hypothetical NSCLC / DEMO-A cohort · OS only. Matching mock summaries is not clinical validation.')
st.caption('Offline examples are bound human-authored mock extractions, not general language understanding. Local single-user mode; no cancellation or background jobs.')
if 'service' not in st.session_state:
    try:
        st.session_state.service=WorkflowService(os.environ['VCT_IPD_FIXTURES'],os.environ['VCT_IPD_OUTPUT'],
            model=os.environ.get('VCT_IPD_MODEL'),allow_live=os.environ.get('VCT_IPD_ALLOW_LIVE')=='1')
    except (KeyError,ValueError,OSError):
        st.error('Configure approved fixture/output roots with python -m vct_ipd.ui.');st.stop()
service=st.session_state.service
if st.button('New request',key='new'):
    st.session_state.pop('token',None);st.session_state.pop('outcome',None);st.session_state.pop('message',None)
workflow=st.radio('Workflow',['generate','collect'],horizontal=True)
if workflow=='generate':
    st.text('Execution profile: hypothetical-nsclc (explicitly fictional evidence)')
    example=st.selectbox('Bound offline example',EXAMPLES)
    try:default=service.example_text(example)
    except (ValueError,OSError):
        st.session_state.pop('outcome',None);st.error('Configured example files are unavailable. Check the approved fixture root.');st.stop()
else:
    collection=st.selectbox('Collection configuration',['fictional','ambiguous','public'])
    try:default=service.collection_request(collection).target_description
    except (ValueError,OSError):
        st.session_state.pop('outcome',None);st.error('Configured literature inputs are unavailable.');st.stop()
with st.form('request',enter_to_submit=False):
    text=st.text_area('Request / 研究问题',value=default,key='text-'+workflow+'-'+(example if workflow=='generate' else collection),height=100)
    provider=st.selectbox('Planning provider',['offline','live']) if workflow=='generate' else 'offline'
    policy=st.selectbox('Predeclared evidence policy',['tuning','evidence-gap']) if workflow=='generate' else 'tuning'
    live_diagnosis=st.checkbox('Use configured live diagnostic proposals (billable opt-in)',value=False) if workflow=='generate' else False
    live_extraction=st.checkbox('Use configured live extraction (billable opt-in)',value=False) if workflow=='collect' and collection=='public' else False
    submitted=st.form_submit_button('Execute',key='execute')
progress=st.empty()
if submitted:
    st.session_state.pop('outcome',None);st.session_state.pop('message',None)
    token=st.session_state.setdefault('token','run-'+uuid.uuid4().hex)
    try:
        config=RunConfig(workflow=workflow,text=text,example=example if workflow=='generate' else 'english',
            provider=provider,policy=policy,collection=collection if workflow=='collect' else 'fictional',
            live_extraction=live_extraction,live_diagnosis=live_diagnosis)
    except ValueError:
        st.error('Supply a nonempty request of at most 8,000 characters. No execution or downloads.');st.stop()
    def event(e):
        progress.text('Stage: '+e['stage'])
        if e['stage']=='ready':st.json(e,expanded=True)
    try:st.session_state.outcome=service.run_once(token,config,on_event=event)
    except (ValueError,OSError):st.session_state.message='Run could not start. The configuration is frozen; select New request. No previous downloads are displayed.'
if st.session_state.get('message'):st.error(st.session_state.message)
if st.session_state.get('outcome'):
    outcome=st.session_state.outcome
    try:
        view=load_result(service.output,outcome.token)
    except (ValueError,OSError,KeyError,TypeError):
        st.error('Invalid or changed result package. All downloads blocked.');st.stop()
    st.subheader('Saved result')
    st.text('Run: '+outcome.token)
    if view.status=='completed':st.success('Completed · comparison and fresh validation passed')
    elif view.status=='error':
        st.error('Failed run · no cohort export')
        reason=view.sections['Request and workflow'].get('error')
        if reason:st.text(reason)  # Fixed application/input diagnosis; never executable HTML.
    else:st.warning(view.status)
    st.caption('These are the frozen run settings and actual artifacts. Re-rendering never executes the pipeline.')
    if view.target_rows():st.dataframe(view.target_rows(),hide_index=True)
    for title,value in view.sections.items():
        with st.expander(title,expanded=title in ('Selection and export','Target results','Exported replica')):st.json(value)
    if view.preview:st.dataframe(view.preview,hide_index=True)
    if view.cohort is not None:
        st.download_button('Download synthetic cohort' if view.status=='completed' else 'Download uncalibrated / partial cohort',
            view.cohort,'synthetic_ipd.csv','text/csv',key='cohort',on_click='ignore')
    st.download_button('Download readable report',view.html(),'report.html','text/html',key='report',on_click='ignore')
    st.download_button('Download controlled results bundle',view.bundle(),'results.zip','application/zip',key='bundle',on_click='ignore')
