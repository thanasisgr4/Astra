"""Run with: streamlit run app.py"""
import json, csv, io, hashlib
from pathlib import Path
from dataclasses import asdict
import streamlit as st
import pandas as pd
from engine import VERSION, scenario_from_document, candidates, run_study, simulate, Policy, validate

st.set_page_config(page_title='ASTRAIA | Campaign Lab',layout='wide')
st.title('ASTRAIA — Campaign Logistics Lab')
st.caption(VERSION+' | Independent research prototype; not a certified mission planner')
st.warning('Synthetic examples. No flight-reliability or cargo-loss predictions. Transfer durations and delivery costs are external assumptions, not computed trajectories.')
base=Path(__file__).parent
upload=st.sidebar.file_uploader('Import campaign JSON v1',type=['json'])
example=st.sidebar.selectbox('Example', ['synthetic_campaign.json','analytic_benchmark.json'])
if upload is not None:
    if upload.size>500000:
        st.error('Maximum JSON size: 500 kB'); st.stop()
    raw=upload.getvalue().decode('utf-8-sig')
else: raw=(base/'examples'/example).read_text()
text=st.sidebar.text_area('Editable scenario JSON',value=raw,height=350,key=hashlib.sha256(raw.encode()).hexdigest())
try:
    doc=json.loads(text,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Non-finite JSON value')))
    s=scenario_from_document(doc)
except Exception as e:
    st.error('Invalid input: '+str(e)); st.stop()
st.sidebar.caption('No legacy 4.1 import. No code is executed from uploaded JSON.')
train=st.sidebar.number_input('Selection samples',min_value=100,max_value=2000,value=256,step=100)
test=st.sidebar.number_input('Independent holdout samples',min_value=100,max_value=4000,value=512,step=100)
seed=st.sidebar.number_input('Seed',min_value=0,max_value=1000000,value=42)
target=st.sidebar.slider('Target: Wilson lower bound',0.50,0.99,0.90,0.01)
tabs=st.tabs(['Scenario & evidence','Policies & comparison','Event audit','Methods & limits'])
with tabs[0]:
    st.caption('Provenance labels are operator declarations, not automatically verified evidence.')
    st.subheader(s.name)
    st.write('Endpoint:',s.endpoint,'| Commodity:',s.commodity)
    st.dataframe(pd.DataFrame([{'field':k,'value':str(v),'status':doc['provenance'][k]['status'],'basis':doc['provenance'][k]['basis']} for k,v in doc['scenario'].items()]),hide_index=True,use_container_width=True)
    st.subheader('Demand windows')
    st.dataframe(pd.DataFrame([asdict(d) for d in s.demands]),hide_index=True)
    for note in doc['limitations']: st.write('- '+str(note))
    st.download_button('Download scenario with provenance',json.dumps(doc,indent=2,ensure_ascii=False),'campaign.json','application/json')
policies=candidates(s)
with tabs[1]:
    st.write('Finite policy grid: dispatch count, cadence, first departure and reserve floor. These are open-loop schedules, not adaptive replanning.')
    st.dataframe(pd.DataFrame([asdict(p) for p in policies]),hide_index=True,use_container_width=True)
    st.caption('Optional additional policy. Times are campaign days, not calendar dates.')
    custom_name=st.text_input('Custom name','custom')
    custom_times=st.text_input('Custom dispatch days (comma separated)','0,8,16')
    custom_load=st.number_input('Custom load per trip (kg)',min_value=0.0,max_value=float(s.max_load_kg),value=float(s.max_load_kg))
    custom_reserve=st.number_input('Custom retained reserve floor (kg)',min_value=0.0,max_value=float(s.capacity_kg),value=0.0)
    include_custom=st.checkbox('Include custom policy')
    if include_custom:
        try:
            custom=Policy(custom_name,tuple(float(x.strip()) for x in custom_times.split(',') if x.strip()),custom_load,custom_reserve)
            validate(s,custom)
            if custom.name in {p.name for p in policies}: raise ValueError('Duplicate name')
            policies.append(custom)
        except Exception as e:
            st.error(str(e)); st.stop()
    fingerprint=hashlib.sha256(json.dumps({'doc':doc,'policies':[asdict(p) for p in policies],'train':train,'test':test,'seed':seed,'target':target},sort_keys=True).encode()).hexdigest()
    if st.button('Compare policies / independent holdout',type='primary'):
        with st.spinner('Evaluating the declared model, not actual flight reliability...'):
            try:
                report=run_study(s,policies,int(train),int(test),int(seed),target)
                st.session_state['result']=(fingerprint,report)
            except Exception as e: st.error('Study failed: '+str(e))
    saved=st.session_state.get('result')
    if saved and saved[0]!=fingerprint: st.info('Inputs changed. Re-run comparison; previous results are hidden.')
    if saved and saved[0]==fingerprint:
        report=saved[1]
        rows=report['selection_rows']
        frame=pd.DataFrame([{k:v for k,v in r.items() if k!='missed_counts_by_demand'} for r in rows])
        st.subheader('Selection sample — exploratory comparison')
        st.dataframe(frame.sort_values(['modeled_service_fraction','mean_attempt_cost_usd'],ascending=[False,True]),hide_index=True,use_container_width=True)
        st.scatter_chart(frame,x='mean_attempt_cost_usd',y='modeled_service_fraction')
        st.write('Observed nondominated policies:',', '.join(report['pareto_on_selection_samples']))
        st.caption('Pareto membership uses sample estimates; it is not proof of statistical dominance.')
        if report['selected_policy']:
            st.write('Selected on selection sample:',report['selected_policy'])
            h=report['holdout']; a,b,c=st.columns(3)
            a.metric('Holdout modeled campaign service',f"{h['modeled_service_fraction']:.2%}")
            b.metric('95% Wilson interval',f"{h['wilson_low_95']:.2%}–{h['wilson_high_95']:.2%}")
            c.metric('Mean attempted campaign cost',f"${h['mean_attempt_cost_usd']:,.0f}")
            st.write('Independent holdout lower-bound target met:',report['holdout_target_met'])
            st.write('Demand misses on holdout:',h['missed_counts_by_demand'])
            st.subheader('One-at-a-time stress tests; not redesign recommendations')
            st.dataframe(pd.DataFrame([{'case':r['case'],**{k:v for k,v in r.get('result',{}).items() if k not in ('missed_counts_by_demand',)},'error':r.get('error','')} for r in report['stress_tests']]),hide_index=True,use_container_width=True)
        else: st.warning('No candidate meets the selection-sample lower-bound target. No recommendation is issued.')
        bundle={'scenario_document':doc,'policies':[asdict(p) for p in policies],'report':report}
        st.download_button('Export full evidence JSON',json.dumps(bundle,indent=2,allow_nan=False),'astraia_evidence.json','application/json')
        st.download_button('Export selection CSV',frame.to_csv(index=False),'policy_comparison.csv','text/csv')
with tabs[2]:
    if not policies: st.warning('No valid candidates for this horizon/cadence; supply a custom policy.')
    else:
        name=st.selectbox('Policy to audit (nominal, no sampled delay)',[p.name for p in policies])
        p=next(p for p in policies if p.name==name)
        audit=simulate(s,p,trace=True)
        st.json({k:v for k,v in audit.items() if k not in ('events','trips')})
        st.caption('Deposits are booked at unloading completion. Same-time order: deposit, demand release, deadline. Service is all-or-nothing, earliest deadline first, no bypass of a blocked head.')
        st.dataframe(pd.DataFrame(audit['events']),hide_index=True,use_container_width=True)
        st.dataframe(pd.DataFrame(audit['trips']),hide_index=True,use_container_width=True)
        st.download_button('Export nominal event ledger',pd.DataFrame(audit['events']).to_csv(index=False),'nominal_events.csv','text/csv')
        st.download_button('Export nominal trip ledger',pd.DataFrame(audit['trips']).to_csv(index=False),'nominal_trips.csv','text/csv')
with tabs[3]:
    st.markdown((base/'MODEL_CARD.md').read_text())
