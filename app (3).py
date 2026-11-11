"""ASTRAIA deployment set 5.0.1. Run: python -m streamlit run app.py"""
from pathlib import Path
from dataclasses import asdict
import json, hashlib, sys, importlib.util
import streamlit as st
import pandas as pd

st.set_page_config(page_title='ASTRAIA Campaign Lab',layout='wide')
BASE=Path(__file__).resolve().parent
# Load the sibling file explicitly: do not accidentally import another installed engine.
engine_path=BASE/'engine.py'
if not engine_path.is_file():
    st.error('Missing engine.py. Upload it beside app.py, from the SAME deployment set.'); st.stop()
try:
    spec=importlib.util.spec_from_file_location('astraia_campaign_engine',engine_path)
    engine=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=engine
    spec.loader.exec_module(engine)
except Exception as e:
    st.error('Cannot load sibling engine.py: '+type(e).__name__+': '+str(e)); st.stop()
required=('VERSION','scenario_from_document','candidates','run_study','simulate','Policy','validate')
missing=[x for x in required if not hasattr(engine,x)]
if missing:
    st.error('Wrong engine.py; missing symbols: '+', '.join(missing)); st.stop()
if engine.VERSION!='5.0.1-research':
    st.error('Mismatched deployment files. Expected engine 5.0.1-research.'); st.stop()
VERSION=engine.VERSION
scenario_from_document=engine.scenario_from_document
candidates=engine.candidates
run_study=engine.run_study
simulate=engine.simulate
Policy=engine.Policy
validate=engine.validate

st.title('ASTRAIA — Campaign Logistics Lab')
st.caption(VERSION+' | Research screening, not flight assurance')
st.warning('Both examples use synthetic inputs. No prediction of spacecraft loss or actual mission reliability. External transport durations and prices are assumptions.')
examples={name:BASE/'examples'/name for name in ('synthetic_campaign.json','analytic_benchmark.json')}
absent=[str(p.relative_to(BASE)) for p in examples.values() if not p.is_file()]
if absent:
    st.error('Missing files: '+', '.join(absent)); st.stop()
choice=st.sidebar.selectbox('Example',list(examples))
upload=st.sidebar.file_uploader('Import campaign v1 JSON',type=['json'])
if upload is not None and upload.size>500000:
    st.error('Maximum upload size: 500 kB'); st.stop()
try:
    raw=upload.getvalue().decode('utf-8-sig') if upload else examples[choice].read_text(encoding='utf-8')
except Exception as e:
    st.error(str(e)); st.stop()
text=st.sidebar.text_area('Editable scenario JSON',raw,height=300,key=hashlib.sha256(raw.encode()).hexdigest())
try:
    doc=json.loads(text,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Non-finite JSON value')))
    s=scenario_from_document(doc)
except Exception as e:
    st.error('Invalid scenario: '+str(e)); st.stop()
train=st.sidebar.number_input('Selection samples',100,2000,256,100)
test=st.sidebar.number_input('Independent holdout samples',100,4000,512,100)
seed=st.sidebar.number_input('Random seed',0,1000000,42)
target=st.sidebar.slider('Target Wilson lower bound',.50,.99,.90,.01)
tabs=st.tabs(['Scenario & evidence','Policies & comparison','Event audit','Methods & limits'])
with tabs[0]:
    st.subheader(s.name)
    st.write('Endpoint:',s.endpoint,'| Commodity:',s.commodity)
    st.caption('Provenance is declared by the operator, not independently verified by this software.')
    st.dataframe(pd.DataFrame([{'field':k,'value':str(v),**doc['provenance'][k]} for k,v in doc['scenario'].items()]),hide_index=True,use_container_width=True)
    st.dataframe(pd.DataFrame([asdict(d) for d in s.demands]),hide_index=True)
    for note in doc['limitations']: st.write('- '+str(note))
    st.download_button('Export scenario JSON',json.dumps(doc,indent=2,ensure_ascii=False),'campaign.json','application/json')
policies=candidates(s)
with tabs[1]:
    st.caption('Open-loop finite grid: trip count, cadence, first departure and retained reserve. Not adaptive control or global optimisation.')
    st.dataframe(pd.DataFrame([asdict(p) for p in policies]),hide_index=True,use_container_width=True)
    enabled=st.checkbox('Include custom policy')
    name=st.text_input('Custom policy name','custom')
    times=st.text_input('Custom dispatch days (comma separated)','0,8,16')
    load=st.number_input('Custom load per trip (kg)',min_value=0.0,max_value=float(s.max_load_kg),value=float(s.max_load_kg))
    reserve=st.number_input('Custom retained reserve (kg)',min_value=0.0,max_value=float(s.capacity_kg),value=0.0)
    if enabled:
        try:
            p=Policy(name,tuple(float(x.strip()) for x in times.split(',') if x.strip()),load,reserve)
            validate(s,p)
            if name in {p.name for p in policies}: raise ValueError('Duplicate policy name')
            policies.append(p)
        except Exception as e:
            st.error('Invalid custom policy: '+str(e)); st.stop()
    fingerprint=hashlib.sha256(json.dumps({'doc':doc,'policies':[asdict(p) for p in policies],'train':train,'test':test,'seed':seed,'target':target},sort_keys=True).encode()).hexdigest()
    if st.button('Compare policies',type='primary'):
        try:
            with st.spinner('Evaluating declared model...'):
                report=run_study(s,policies,int(train),int(test),int(seed),float(target))
            st.session_state['study']=(fingerprint,report)
        except Exception as e: st.error('Study failed: '+str(e))
    saved=st.session_state.get('study')
    if saved and saved[0]!=fingerprint: st.info('Inputs changed. Re-run comparison. Previous results are hidden.')
    if saved and saved[0]==fingerprint:
        report=saved[1]
        frame=pd.DataFrame([{k:v for k,v in row.items() if k!='missed_counts_by_demand'} for row in report['selection_rows']])
        st.subheader('Selection-sample comparison')
        st.dataframe(frame,hide_index=True,use_container_width=True)
        st.scatter_chart(frame,x='mean_attempt_cost_usd',y='modeled_service_fraction')
        st.write('Observed Pareto policies:',report['pareto_on_selection_samples'])
        if report['selected_policy']:
            st.write('Selected policy:',report['selected_policy'])
            h=report['holdout']; a,b,c=st.columns(3)
            a.metric('Holdout modeled service',f"{h['modeled_service_fraction']:.2%}")
            b.metric('95% Wilson interval',f"{h['wilson_low_95']:.2%}–{h['wilson_high_95']:.2%}")
            c.metric('Mean attempted cost',f"${h['mean_attempt_cost_usd']:,.0f}")
            st.write('Holdout lower-bound target met:',report['holdout_target_met'])
            st.write('Holdout demand misses:',h['missed_counts_by_demand'])
            st.subheader('One-at-a-time stresses (not physical redesign advice)')
            st.json(report['stress_tests'])
        else: st.warning('No candidate meets the selection criterion. No recommendation.')
        evidence={'scenario_document':doc,'policies':[asdict(p) for p in policies],'report':report}
        st.download_button('Export evidence JSON',json.dumps(evidence,indent=2,allow_nan=False),'astraia_evidence.json','application/json')
        st.download_button('Export comparison CSV',frame.to_csv(index=False),'policy_comparison.csv','text/csv')
with tabs[2]:
    if policies:
        selected=st.selectbox('Nominal audit: no sampled delay',[p.name for p in policies])
        p=next(p for p in policies if p.name==selected)
        audit=simulate(s,p,trace=True)
        st.json({k:v for k,v in audit.items() if k not in ('events','trips')})
        st.dataframe(pd.DataFrame(audit['events']),hide_index=True,use_container_width=True)
        st.dataframe(pd.DataFrame(audit['trips']),hide_index=True,use_container_width=True)
        st.download_button('Export event ledger',pd.DataFrame(audit['events']).to_csv(index=False),'events.csv','text/csv')
        st.download_button('Export trip ledger',pd.DataFrame(audit['trips']).to_csv(index=False),'trips.csv','text/csv')
    else: st.info('No valid grid candidates. Add a custom policy.')
with tabs[3]:
    path=BASE/'MODEL_CARD.md'
    if path.is_file(): st.markdown(path.read_text(encoding='utf-8'))
    else: st.warning('Upload MODEL_CARD.md beside app.py to display the full model limitations.')
