# ASTRAIA 5.1.1-research: frozen single-file campaign-screening architecture.
import json, math, random, hashlib, csv, io
from itertools import groupby
from copy import deepcopy
VERSION='5.1.1-research'
SCHEMA='astraia.singledepot.v2'
LIMITS='''Research prototype, not flight operations, hardware reliability or cargo-loss prediction.
One compatible aggregate commodity, one depot and one unloading port.
Transport times and prices are external inputs, not solved trajectories.
Instantaneous all-or-nothing handover at depot; no downstream vehicle or consumer storage.
Existing infrastructure and initial-stock acquisition are outside costs.
All built-in numerical values and uncertainty distributions are synthetic and uncalibrated.'''
PARAMS=dict(horizon_days=45.,capacity_kg=10000.,initial_kg=0.,storage_loss_daily=.001,
 transit_loss_daily=.0002,transfer_efficiency=.99,lead_days=4.,unload_kg_per_day=6000.,
 tanker_max_load_kg=6000.,minimum_dispatch_gap_days=2.,individual_delay_probability=.25,
 individual_delay_max_days=5.,common_delay_probability=.1,common_delay_max_days=4.,
 delivery_service_usd_per_trip=2000000.,commodity_usd_per_kg=50.,fixed_campaign_usd=500000.,
 cost_relative_half_range=.15)

def example(analytic=False):
    p=dict(PARAMS)
    demands=[dict(id='D1',ready_day=20.,deadline_day=23.,kg=8000.),dict(id='D2',ready_day=35.,deadline_day=38.,kg=8000.)]
    if analytic:
        p.update(horizon_days=25.,storage_loss_daily=0.,transit_loss_daily=0.,transfer_efficiency=1.,lead_days=1.,individual_delay_probability=0.,individual_delay_max_days=0.,common_delay_probability=0.,common_delay_max_days=0.,cost_relative_half_range=0.)
        demands=[dict(id='D1',ready_day=10.,deadline_day=10.,kg=4000.),dict(id='D2',ready_day=20.,deadline_day=20.,kg=4000.)]
    return dict(schema=SCHEMA,name='Analytic test fixture' if analytic else 'Synthetic campaign',endpoint='Generic depot handover; NOT Gateway or a named flight vehicle',commodity='One aggregate compatible commodity',parameters=p,demands=demands,provenance={k:dict(status='synthetic',basis='Mathematical demonstration; not flight measurements or supplier quotes.') for k in ('name','endpoint','commodity','demands',*p)},limitations=LIMITS.splitlines())

def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def validate(s,policy=None):
    if not isinstance(s,dict) or set(s)!={'schema','name','endpoint','commodity','parameters','demands','provenance','limitations'} or s['schema']!=SCHEMA: raise ValueError('Expected ASTRAIA singledepot.v2 scenario; legacy schemas are incompatible.')
    for k in ('name','endpoint','commodity'):
        if not isinstance(s[k],str) or not s[k].strip(): raise ValueError('Required text: '+k)
    p=s['parameters']
    if not isinstance(p,dict) or set(p)!=set(PARAMS): raise ValueError('Missing or unknown numerical parameter')
    for k,v in p.items():
        if not finite(v) or v<0: raise ValueError('Finite nonnegative value required: '+k)
    for k in ('horizon_days','capacity_kg','unload_kg_per_day','tanker_max_load_kg'):
        if p[k]<=0: raise ValueError('Must be positive: '+k)
    if p['initial_kg']>p['capacity_kg']: raise ValueError('Initial stock exceeds capacity')
    for k in ('storage_loss_daily','transit_loss_daily','cost_relative_half_range'):
        if p[k]>=1: raise ValueError('Must be < 1: '+k)
    for k in ('transfer_efficiency','individual_delay_probability','common_delay_probability'):
        if p[k]>1: raise ValueError('Must be <= 1: '+k)
    if not isinstance(s['demands'],list) or not 1<=len(s['demands'])<=100: raise ValueError('Use 1..100 demands')
    ids=[]
    for d in s['demands']:
        if not isinstance(d,dict) or set(d)!={'id','ready_day','deadline_day','kg'}: raise ValueError('Invalid demand fields')
        if not isinstance(d['id'],str) or not d['id'].strip(): raise ValueError('Demand ID required')
        if not all(finite(d[k]) for k in ('ready_day','deadline_day','kg')) or not 0<=d['ready_day']<=d['deadline_day']<=p['horizon_days'] or d['kg']<=0: raise ValueError('Invalid demand window or mass')
        ids.append(d['id'])
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate demand IDs')
    needed={'name','endpoint','commodity','demands',*p}
    if not isinstance(s['provenance'],dict) or not needed.issubset(s['provenance']): raise ValueError('Every input requires provenance')
    for k in needed:
        a=s['provenance'][k]
        if not isinstance(a,dict) or a.get('status') not in ('synthetic','assumption','published','measured') or not isinstance(a.get('basis'),str) or not a['basis'].strip(): raise ValueError('Invalid provenance: '+k)
    if not isinstance(s['limitations'],list) or not s['limitations'] or any(not isinstance(x,str) for x in s['limitations']): raise ValueError('List of limitations required')
    if policy is None: return
    if not isinstance(policy,dict) or set(policy)!={'name','dispatch_days','load_kg','reserve_kg'}: raise ValueError('Invalid policy fields')
    if not isinstance(policy['name'],str) or not policy['name'].strip(): raise ValueError('Policy name required')
    days=policy['dispatch_days']
    if not isinstance(days,list) or len(days)>32 or any(not finite(t) or not 0<=t<=p['horizon_days'] for t in days): raise ValueError('Use up to 32 valid dispatch times')
    if days!=sorted(days) or any(b-a<p['minimum_dispatch_gap_days']-1e-10 for a,b in zip(days,days[1:])): raise ValueError('Invalid departure order/cadence')
    if not finite(policy['load_kg']) or not 0<policy['load_kg']<=p['tanker_max_load_kg']: raise ValueError('Invalid trip load')
    if not finite(policy['reserve_kg']) or not 0<=policy['reserve_kg']<=p['capacity_kg']: raise ValueError('Invalid reserve floor')

def make_policy(name,days,load=6000.,reserve=0.): return dict(name=name,dispatch_days=list(days),load_kg=load,reserve_kg=reserve)

def draw_shocks(s,n,seed):
    validate(s)
    if not isinstance(n,int) or not 1<=n<=10000: raise ValueError('Use 1..10000 draws')
    p=s['parameters']; rng=random.Random(seed); result=[]
    for _ in range(n):
        u,v=rng.random(),rng.random(); common=v*p['common_delay_max_days'] if u<p['common_delay_probability'] else 0.
        delays=[]
        for i in range(32):
            u,v=rng.random(),rng.random(); delays.append(v*p['individual_delay_max_days'] if u<p['individual_delay_probability'] else 0.)
        result.append((common,delays,1+p['cost_relative_half_range']*(2*rng.random()-1)))
    return result

def simulate(s,policy,shock=None,trace=False):
    validate(s,policy); p=s['parameters']; load=policy['load_kg']
    common,delays,multiplier=shock if shock is not None else (0.,[0.]*32,1.)
    if len(delays)<len(policy['dispatch_days']) or any(not finite(x) or x<0 for x in (common,multiplier,*delays)): raise ValueError('Invalid shock')
    arrivals=sorted((t+p['lead_days']+common+delays[i],i,t) for i,t in enumerate(policy['dispatch_days']))
    events=[]; trips=[]; free=0.
    for arrival,i,departure in arrivals:
        start=max(arrival,free); end=start+load/p['unload_kg_per_day']; free=end
        retained=load*(1-p['transit_loss_daily'])**(end-departure); offered=retained*p['transfer_efficiency']
        trips.append(dict(trip=i,departure_day=departure,arrival_day=arrival,unload_start_day=start,deposit_day=end,loaded_kg=load,offered_kg=offered,transit_loss_kg=load-retained,transfer_loss_kg=retained-offered,completed_in_horizon=end<=p['horizon_days']))
        if end<=p['horizon_days']: events.append((end,0,str(i),offered))
    for d in s['demands']: events.extend([(d['ready_day'],1,d['id'],d['kg']),(d['deadline_day'],2,d['id'],d['kg'])])
    events.append((p['horizon_days'],3,'horizon',0.)); events.sort(key=lambda e:(e[0],e[1],e[2]))
    deadlines={d['id']:d['deadline_day'] for d in s['demands']}; inv=p['initial_kg']
    last=received=boiled=overflow=served=0.; pending={}; delivery_days={}; missed=[]; history=[]
    for t,g in groupby(events,key=lambda e:e[0]):
        batch=list(g); after=inv*(1-p['storage_loss_daily'])**(t-last)
        loss=inv-after; boiled+=loss; inv=after; last=t; accepted=spilled=0.; handed=[]
        for _,kind,key,q in batch:
            if kind==0:
                received+=q; take=min(q,max(0.,p['capacity_kg']-inv)); inv+=take; accepted+=take; spilled+=q-take
        overflow+=spilled
        for _,kind,key,q in batch:
            if kind==1: pending[key]=q
        def allocate():
            nonlocal inv,served
            for key in sorted(list(pending),key=lambda k:(deadlines[k],k)):
                q=pending[key]
                if inv+1e-8<q+policy['reserve_kg']: break
                inv=max(0.,inv-q); served+=q; delivery_days[key]=t; handed.append(key); del pending[key]
        allocate()
        for _,kind,key,q in batch:
            if kind==2 and key in pending:
                missed.append(key); del pending[key]; allocate()
        if trace: history.append(dict(day=t,event='+'.join(sorted(set(('deposit','ready','deadline','horizon')[e[1]] for e in batch))),inventory_kg=inv,accepted_kg=accepted,overflow_kg=spilled,storage_loss_kg=loss,served_ids=','.join(handed)))
    balance=p['initial_kg']+received-served-boiled-overflow-inv
    if abs(balance)>1e-6*max(1.,p['initial_kg']+received): raise ArithmeticError('Depot mass balance failed')
    cost=(p['fixed_campaign_usd']+len(policy['dispatch_days'])*(p['delivery_service_usd_per_trip']+load*p['commodity_usd_per_kg']))*multiplier
    return dict(policy=policy['name'],campaign_service=not missed,delivered_kg=served,fill_fraction=served/sum(d['kg'] for d in s['demands']),missed_ids=missed,delivery_days=delivery_days,attempt_cost_usd=cost,offered_to_depot_kg=received,storage_loss_kg=boiled,overflow_kg=overflow,final_inventory_kg=inv,balance_error_kg=balance,unfinished_trips=sum(not t['completed_in_horizon'] for t in trips),events=history,trips=trips if trace else [])

def wilson(k,n):
    if n<=0 or not 0<=k<=n: raise ValueError('Invalid counts')
    z=1.959963984540054; ph=k/n; den=1+z*z/n
    centre=(ph+z*z/(2*n))/den; half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return max(0.,centre-half),min(1.,centre+half)

def evaluate(s,p,shocks):
    if not shocks: raise ValueError('No samples')
    rr=[simulate(s,p,z) for z in shocks]; n=len(rr); k=sum(r['campaign_service'] for r in rr)
    lo,hi=wilson(k,n); costs=sorted(r['attempt_cost_usd'] for r in rr); count=max(1,math.ceil(.05*n))
    x=.95*(n-1); i=int(x); p95=costs[i]+(costs[min(i+1,n-1)]-costs[i])*(x-i)
    return dict(policy=p['name'],samples=n,served_campaigns=k,modeled_service=k/n,wilson_low_95=lo,wilson_high_95=hi,mean_attempt_cost_usd=sum(costs)/n,p95_attempt_cost_usd=p95,upper_5pct_mean_attempt_cost_usd=sum(costs[-count:])/count,mean_fill_fraction=sum(r['fill_fraction'] for r in rr)/n,mean_storage_loss_kg=sum(r['storage_loss_kg'] for r in rr)/n,mean_overflow_kg=sum(r['overflow_kg'] for r in rr)/n,max_abs_balance_error_kg=max(abs(r['balance_error_kg']) for r in rr),missed_counts={d['id']:sum(d['id'] in r['missed_ids'] for r in rr) for d in s['demands']})

def policy_grid(s):
    result=[]
    for n in (2,3,4):
        for gap in (4.,8.):
            for first in (0.,3.):
                for reserve in (0.,500.):
                    p=make_policy(f'n{n}_gap{gap:g}_first{first:g}_reserve{reserve:g}',[first+i*gap for i in range(n)],s['parameters']['tanker_max_load_kg'],reserve)
                    try: validate(s,p)
                    except ValueError: continue
                    result.append(p)
    return result

def study(s,policies,selection_n=256,holdout_n=512,seed=42,target=.90):
    validate(s)
    if not policies or len(policies)>100 or len({p['name'] for p in policies})!=len(policies): raise ValueError('Use 1..100 uniquely named policies')
    if not finite(target) or not 0<target<1: raise ValueError('Target must be in (0,1)')
    shared=draw_shocks(s,selection_n,seed); rows=[evaluate(s,p,shared) for p in policies]
    acceptable=[r for r in rows if r['wilson_low_95']>=target]
    chosen=min(acceptable,key=lambda r:(r['mean_attempt_cost_usd'],r['policy']))['policy'] if acceptable else None
    holdout=None; stress=[]
    if chosen:
        p=next(p for p in policies if p['name']==chosen); holdout=evaluate(s,p,draw_shocks(s,holdout_n,seed+1000003))
        for key,value in [('capacity_kg',s['parameters']['capacity_kg']*.8),('lead_days',s['parameters']['lead_days']+3),('storage_loss_daily',min(.999,s['parameters']['storage_loss_daily']*2))]:
            changed=deepcopy(s); changed['parameters'][key]=value
            try: stress.append(dict(changed_parameter=key,new_value=value,result=evaluate(changed,p,draw_shocks(changed,holdout_n,seed+1000003))))
            except ValueError as e: stress.append(dict(changed_parameter=key,new_value=value,error=str(e)))
    pareto=[a['policy'] for a in rows if not any(b['mean_attempt_cost_usd']<=a['mean_attempt_cost_usd'] and b['modeled_service']>=a['modeled_service'] and (b['mean_attempt_cost_usd']<a['mean_attempt_cost_usd'] or b['modeled_service']>a['modeled_service']) for b in rows)]
    return dict(version=VERSION,scenario_sha256=hashlib.sha256(json.dumps(s,sort_keys=True,allow_nan=False).encode()).hexdigest(),selection_seed=seed,holdout_seed=seed+1000003,target_lower_bound=target,selection=rows,selected_policy=chosen,holdout=holdout,holdout_target_met=holdout['wilson_low_95']>=target if holdout else None,pareto_on_selection_samples=pareto,stress=stress,interpretation='Conditional model service frequency; NOT mission reliability or spacecraft-loss probability.')

def to_csv(rows):
    if not rows: return ''
    b=io.StringIO(); w=csv.DictWriter(b,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); return b.getvalue()

METHODS='''### Boundary and accounting
One commodity, one depot, one non-preemptive unloading port. External times and delivery-service prices.
No physical mission validation, ESA endorsement, propulsion or orbital solution.
Daily retention (1-b)**dt is a declared loss convention, not a thermal model.
Transit loss includes waiting and unloading. Duration uses loaded mass/rate. Credit supply at completion.
At a timestamp: decay, all deposits, all demand releases, allocation, expiration and retry.
All-or-nothing, earliest deadline first; no bypass of a blocked head. Reserve stays at depot.
Depot balance: initial+offered=handovers+storage loss+overflow+final. Offered includes overflow.
Trip balance: load=transit loss+transfer loss+offered; excludes propulsion. No double-counting.
Unfinished trips are charged, not received; their eventual offered mass is a projection, not horizon inventory.
No complete system in-transit mass balance is claimed. No consumer storage/downstream model.

### Economics
Attempt cost = fixed campaign charge + committed trips*(service quote+loaded commodity price), times a shared multiplier.
No revenue, NPV, salvage, refunds, automatic capital/reuse amortisation or cost-per-success claim.
Initial stock acquisition and existing infrastructure excluded; include a justified external allocation if needed.
Do not double-count launch/propulsion already included in a service quote.

### Uncertainty and selection
Bernoulli delay occurrence followed by uniform magnitude; campaign-common plus individual delay.
Uniform common cost multiplier within relative half-range. Built-in distributions uncalibrated.
Shared draws correspond to trip index, not physical flight IDs. No hardware-loss failure model.
Cheapest finite-grid policy with selection-sample two-sided 95% Wilson LOWER bound >= target is selected.
Only that policy is checked on independent model draws, seed+1000003. No qualifying candidate: no recommendation.
Intervals quantify model-sampling uncertainty, not physical model validity. Repeated tuning after viewing holdout
requires a new predeclared confirmation case. Pareto is sample-based, not proven statistical dominance.
Stresses: capacity -20%, lead +3 days, loss x2 separately. No redesign cost included; not engineering advice.
Provenance is operator-declared. SHA-256 identifies the full scenario document; it is not certification.
Finite open-loop schedules, not globally optimal or adaptive control. Research use only.'''

def render_app():
    import streamlit as st
    st.set_page_config(page_title='ASTRAIA Campaign Lab',layout='wide')
    st.title('ASTRAIA — Campaign Logistics Lab'); st.caption(VERSION+' | Single-file research prototype')
    st.warning('Built-in examples are SYNTHETIC. No actual flight reliability or cargo-loss predictions.')
    mode=st.sidebar.selectbox('Built-in example',['Synthetic campaign','Analytic benchmark'])
    upload=st.sidebar.file_uploader('Import v2 scenario JSON',type=['json'])
    try:
        if upload is not None and upload.size>250000: raise ValueError('Maximum JSON upload: 250 kB')
        raw=upload.getvalue().decode('utf-8-sig') if upload is not None else json.dumps(example(mode=='Analytic benchmark'),indent=2)
    except Exception as e: st.error(str(e)); st.stop()
    text=st.sidebar.text_area('Editable scenario JSON',raw,height=260,max_chars=250000,key=hashlib.sha256(raw.encode()).hexdigest())
    try:
        s=json.loads(text,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Non-finite JSON'))); validate(s)
    except Exception as e: st.error('Invalid scenario: '+str(e)); st.stop()
    s=deepcopy(s); input_key=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    with st.sidebar.expander('Edit numerical parameters'):
        for key,original in list(s['parameters'].items()):
            value=st.number_input(key,min_value=0.,value=float(original),format='%.6f',key=input_key+'_'+key)
            if value!=original:
                s['parameters'][key]=value; s['provenance'][key]={'status':'assumption','basis':'User numeric-control override; not independently verified.'}
    try: validate(s)
    except Exception as e: st.error('Invalid parameter override: '+str(e)); st.stop()
    train=st.sidebar.number_input('Selection samples',100,2000,256,100)
    test=st.sidebar.number_input('Independent holdout samples',100,4000,512,100)
    seed=st.sidebar.number_input('Seed',0,1000000,42)
    target=st.sidebar.slider('Target Wilson lower bound',.50,.99,.90,.01)
    tabs=st.tabs(['Scenario','Policies & comparison','Event audit','Methods'])
    with tabs[0]:
        st.subheader(s['name']); st.write('Endpoint:',s['endpoint']); st.write('Commodity:',s['commodity'])
        st.dataframe([dict(parameter=k,value=v,**s['provenance'][k]) for k,v in s['parameters'].items()],hide_index=True,use_container_width=True)
        st.dataframe(s['demands'],hide_index=True); st.caption('Provenance is declared, not independently verified.')
        for note in s['limitations']: st.write('- '+note)
        st.download_button('Export scenario JSON',json.dumps(s,indent=2,ensure_ascii=False),'scenario_v2.json','application/json')
    policies=policy_grid(s)
    with tabs[1]:
        st.write('Finite open-loop policy grid plus optional custom schedule.')
        st.dataframe([dict(p,dispatch_days=str(p['dispatch_days'])) for p in policies],hide_index=True,use_container_width=True)
        enabled=st.checkbox('Include a custom policy'); name=st.text_input('Policy name','reference')
        times=st.text_input('Departure days, comma separated','0,8')
        load=st.number_input('Load per trip (kg)',min_value=0.,max_value=float(s['parameters']['tanker_max_load_kg']),value=float(s['parameters']['tanker_max_load_kg']))
        reserve=st.number_input('Retained reserve floor (kg)',min_value=0.,max_value=float(s['parameters']['capacity_kg']),value=0.)
        if enabled:
            try:
                custom=make_policy(name,[float(t.strip()) for t in times.split(',') if t.strip()],load,reserve); validate(s,custom)
                if name in {p['name'] for p in policies}: raise ValueError('Duplicate policy name')
                policies.append(custom)
            except Exception as e: st.error('Invalid custom policy: '+str(e)); st.stop()
        settings=dict(scenario=s,policies=policies,selection_n=train,holdout_n=test,seed=seed,target=target)
        signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
        if st.button('Compare policies',type='primary'):
            try:
                with st.spinner('Evaluating declared logistics model...'): report=study(s,policies,int(train),int(test),int(seed),target)
                st.session_state['astraia_result']=(signature,report)
            except Exception as e: st.error('Study failed: '+str(e))
        stored=st.session_state.get('astraia_result')
        if stored and stored[0]!=signature: st.info('Inputs changed. Re-run comparison; old results hidden.')
        if stored and stored[0]==signature:
            r=stored[1]; rows=[{k:v for k,v in row.items() if k!='missed_counts'} for row in r['selection']]
            st.subheader('Selection-sample comparison'); st.dataframe(rows,hide_index=True,use_container_width=True)
            st.scatter_chart(rows,x='mean_attempt_cost_usd',y='modeled_service')
            st.write('Sample-based Pareto set:',r['pareto_on_selection_samples'])
            if r['selected_policy']:
                h=r['holdout']; st.write('Selected policy:',r['selected_policy']); a,b,c=st.columns(3)
                a.metric('Holdout modeled service',f"{h['modeled_service']:.2%}")
                b.metric('95% Wilson interval',f"{h['wilson_low_95']:.2%}–{h['wilson_high_95']:.2%}")
                c.metric('Mean attempted cost',f"${h['mean_attempt_cost_usd']:,.0f}")
                st.write('Independent holdout target met:',r['holdout_target_met']); st.write('Demand misses:',h['missed_counts'])
                st.subheader('Separate stress diagnostics'); st.json(r['stress'])
            else: st.warning('No qualifying candidate; no recommendation issued.')
            st.download_button('Export evidence JSON',json.dumps(dict(settings=settings,report=r),indent=2,allow_nan=False),'evidence.json','application/json')
            st.download_button('Export comparison CSV',to_csv(rows),'comparison.csv','text/csv')
    with tabs[2]:
        if not policies: st.info('No grid candidate; add a custom policy.')
        else:
            name=st.selectbox('Nominal audit (no sampled delays)',[p['name'] for p in policies]); p=next(p for p in policies if p['name']==name)
            audit=simulate(s,p,trace=True); st.json({k:v for k,v in audit.items() if k not in ('events','trips')})
            st.line_chart(audit['events'],x='day',y='inventory_kg')
            st.dataframe(audit['events'],hide_index=True,use_container_width=True); st.dataframe(audit['trips'],hide_index=True,use_container_width=True)
            st.download_button('Export events CSV',to_csv(audit['events']),'events.csv','text/csv')
            st.download_button('Export trips CSV',to_csv(audit['trips']),'trips.csv','text/csv')
    with tabs[3]: st.markdown(METHODS); st.code(LIMITS)

if __name__=='__main__': render_app()
