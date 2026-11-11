"""ASTRAIA 5.0.1 research: single-depot campaign logistics, not flight assurance."""
from dataclasses import dataclass, asdict, fields, replace
from itertools import groupby
import random, math, hashlib, json
VERSION='5.0.1-research'
SCHEMA='astraia.campaign.v1'

@dataclass(frozen=True)
class Demand:
    id: str
    ready_day: float
    deadline_day: float
    kg: float

@dataclass(frozen=True)
class Scenario:
    name: str
    endpoint: str
    commodity: str
    horizon_days: float
    capacity_kg: float
    initial_kg: float
    storage_loss_fraction_per_day: float
    transit_loss_fraction_per_day: float
    transfer_efficiency: float
    lead_days: float
    unloading_kg_per_day: float
    max_load_kg: float
    min_dispatch_gap_days: float
    independent_delay_probability: float
    independent_delay_max_days: float
    common_delay_probability: float
    common_delay_max_days: float
    delivery_service_cost_per_dispatch: float
    commodity_cost_per_kg_loaded: float
    campaign_fixed_cost: float
    cost_relative_half_range: float
    demands: tuple

@dataclass(frozen=True)
class Policy:
    name: str
    dispatch_days: tuple
    load_kg: float
    reserve_floor_kg: float=0.0

@dataclass(frozen=True)
class Shock:
    common_delay: float
    delays: tuple
    cost_multiplier: float

def finite(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def validate(s,p=None):
    for f in fields(s):
        if f.name not in ('name','endpoint','commodity','demands'):
            x=getattr(s,f.name)
            if not finite(x) or x<0: raise ValueError('Nonnegative finite value required: '+f.name)
    for k in ('name','endpoint','commodity'):
        if not isinstance(getattr(s,k),str) or not getattr(s,k).strip(): raise ValueError('Required: '+k)
    for k in ('horizon_days','capacity_kg','unloading_kg_per_day','max_load_kg'):
        if getattr(s,k)<=0: raise ValueError('Must be positive: '+k)
    if s.initial_kg>s.capacity_kg: raise ValueError('Initial inventory exceeds capacity')
    for k in ('storage_loss_fraction_per_day','transit_loss_fraction_per_day','cost_relative_half_range'):
        if getattr(s,k)>=1: raise ValueError('Must be less than one: '+k)
    for k in ('transfer_efficiency','independent_delay_probability','common_delay_probability'):
        if getattr(s,k)>1: raise ValueError('Must not exceed one: '+k)
    if not 1<=len(s.demands)<=100: raise ValueError('Use 1..100 demands')
    ids=[]
    for d in s.demands:
        if not isinstance(d.id,str) or not d.id.strip(): raise ValueError('Demand ID required')
        if not all(finite(x) for x in (d.ready_day,d.deadline_day,d.kg)): raise ValueError('Invalid demand number')
        if not 0<=d.ready_day<=d.deadline_day<=s.horizon_days or d.kg<=0: raise ValueError('Invalid demand window/mass')
        ids.append(d.id)
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate demand ID')
    if p is None: return
    if not isinstance(p.name,str) or not p.name.strip(): raise ValueError('Policy name required')
    if not finite(p.load_kg) or not 0<p.load_kg<=s.max_load_kg: raise ValueError('Invalid trip load')
    if not finite(p.reserve_floor_kg) or not 0<=p.reserve_floor_kg<=s.capacity_kg: raise ValueError('Invalid reserve floor')
    if len(p.dispatch_days)>32: raise ValueError('At most 32 dispatches')
    if any(not finite(t) or not 0<=t<=s.horizon_days for t in p.dispatch_days): raise ValueError('Dispatch outside horizon')
    if list(p.dispatch_days)!=sorted(p.dispatch_days): raise ValueError('Dispatches must be sorted')
    if any(b-a<s.min_dispatch_gap_days-1e-10 for a,b in zip(p.dispatch_days,p.dispatch_days[1:])): raise ValueError('Dispatch gap too short')

def scenario_from_document(doc):
    if not isinstance(doc,dict) or set(doc)!={'schema','scenario','provenance','limitations'} or doc['schema']!=SCHEMA:
        raise ValueError('Expected campaign v1 document; legacy v4.1 JSON is incompatible')
    data=dict(doc['scenario'])
    if set(data)!={f.name for f in fields(Scenario)}: raise ValueError('Unknown or missing scenario fields')
    if not isinstance(doc['provenance'],dict) or not set(data).issubset(doc['provenance']): raise ValueError('Provenance required for every field')
    for k in data:
        item=doc['provenance'][k]
        if not isinstance(item,dict) or item.get('status') not in ('synthetic','assumption','published','measured') or not isinstance(item.get('basis'),str) or not item['basis'].strip(): raise ValueError('Invalid provenance: '+k)
    if not isinstance(doc['limitations'],list) or not doc['limitations']: raise ValueError('Limitations required')
    data['demands']=tuple(Demand(**d) for d in data['demands'])
    s=Scenario(**data); validate(s); return s

def sample_shocks(s,n,seed):
    validate(s)
    if not isinstance(n,int) or not 1<=n<=10000: raise ValueError('Use 1..10000 samples')
    rng=random.Random(seed); result=[]
    for _ in range(n):
        u,v=rng.random(),rng.random()
        common=v*s.common_delay_max_days if u<s.common_delay_probability else 0.0
        delays=[]
        for i in range(32):
            u,v=rng.random(),rng.random()
            delays.append(v*s.independent_delay_max_days if u<s.independent_delay_probability else 0.0)
        result.append(Shock(common,tuple(delays),1+s.cost_relative_half_range*(2*rng.random()-1)))
    return result

def simulate(s,p,shock=None,trace=False):
    validate(s,p)
    shock=shock or Shock(0,(0,)*32,1)
    if len(shock.delays)<len(p.dispatch_days) or any(not finite(v) or v<0 for v in (shock.common_delay,shock.cost_multiplier,*shock.delays)): raise ValueError('Invalid shock')
    events=[]; trips=[]; free=0.0
    arrivals=sorted((t+s.lead_days+shock.common_delay+shock.delays[i],i,t) for i,t in enumerate(p.dispatch_days))
    for arrival,i,departure in arrivals:
        start=max(arrival,free); end=start+p.load_kg/s.unloading_kg_per_day; free=end
        retained=p.load_kg*(1-s.transit_loss_fraction_per_day)**(end-departure)
        offered=retained*s.transfer_efficiency
        trips.append(dict(trip_index=i,departure_day=departure,arrival_day=arrival,unload_start_day=start,deposit_day=end,loaded_kg=p.load_kg,offered_kg=offered,transit_loss_kg=p.load_kg-retained,transfer_loss_kg=retained-offered,completed_in_horizon=end<=s.horizon_days))
        if end<=s.horizon_days: events.append((end,0,str(i),offered))
    for d in s.demands:
        events.extend([(d.ready_day,1,d.id,d.kg),(d.deadline_day,2,d.id,d.kg)])
    events.append((s.horizon_days,3,'horizon',0))
    events.sort(key=lambda x:(x[0],x[1],x[2]))
    inv=s.initial_kg; last=received=boiled=overflow=served=0.0
    pending={}; delivery_days={}; missed=[]; history=[]
    deadlines={d.id:d.deadline_day for d in s.demands}
    for t,group in groupby(events,key=lambda e:e[0]):
        batch=list(group)
        after=inv*(1-s.storage_loss_fraction_per_day)**(t-last)
        step_loss=inv-after; boiled+=step_loss; inv=after; last=t
        accepted=spilled=0.0; handed=[]
        for _,kind,key,q in batch:
            if kind==0:
                received+=q; take=min(q,max(0,s.capacity_kg-inv))
                accepted+=take; spilled+=q-take; inv+=take
        overflow+=spilled
        for _,kind,key,q in batch:
            if kind==1: pending[key]=q
        def allocate():
            nonlocal inv,served
            for key in sorted(list(pending),key=lambda k:(deadlines[k],k)):
                q=pending[key]
                if inv+1e-8<q+p.reserve_floor_kg: break
                inv=max(0,inv-q); served+=q; delivery_days[key]=t; del pending[key]; handed.append(key)
        allocate()
        for _,kind,key,q in batch:
            if kind==2 and key in pending:
                missed.append(key); del pending[key]; allocate()
        if trace: history.append(dict(day=t,event='+'.join(sorted(set(('deposit','demand_ready','deadline','horizon')[e[1]] for e in batch))),id=','.join(e[2] for e in batch),inventory_kg=inv,accepted_kg=accepted,overflow_kg=spilled,storage_loss_since_last_kg=step_loss,served_ids=','.join(handed)))
    error=s.initial_kg+received-served-boiled-overflow-inv
    if abs(error)>1e-6*max(1,s.initial_kg+received): raise ArithmeticError('Depot mass balance failed')
    total=sum(d.kg for d in s.demands)
    cost=(s.campaign_fixed_cost+len(p.dispatch_days)*(s.delivery_service_cost_per_dispatch+p.load_kg*s.commodity_cost_per_kg_loaded))*shock.cost_multiplier
    return dict(policy=p.name,campaign_service=len(missed)==0,served_demands=len(delivery_days),total_demands=len(s.demands),delivered_kg=served,demanded_kg=total,fill_fraction=served/total,missed_ids=missed,cost_usd=cost,received_at_depot_kg=received,storage_loss_kg=boiled,overflow_kg=overflow,final_inventory_kg=inv,balance_error_kg=error,unfinished_trips=sum(not x['completed_in_horizon'] for x in trips),delivery_days=delivery_days,trips=trips if trace else [],events=history)

def wilson(k,n,z=1.959963984540054):
    if n<=0 or not 0<=k<=n: raise ValueError('Invalid counts')
    p=k/n; den=1+z*z/n; centre=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,centre-half),min(1,centre+half)

def quantile(xs,q):
    if not xs or not 0<=q<=1: raise ValueError('Invalid quantile')
    xs=sorted(xs); t=q*(len(xs)-1); i=int(t)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(t-i)

def evaluate(s,p,shocks):
    if not shocks: raise ValueError('No samples')
    rr=[simulate(s,p,z) for z in shocks]; n=len(rr)
    k=sum(r['campaign_service'] for r in rr); low,high=wilson(k,n)
    costs=[r['cost_usd'] for r in rr]; tail=sorted(costs)[-max(1,math.ceil(.05*n)):]
    return dict(policy=p.name,samples=n,served_campaigns=k,modeled_service_fraction=k/n,wilson_low_95=low,wilson_high_95=high,mean_attempt_cost_usd=sum(costs)/n,p95_attempt_cost_usd=quantile(costs,.95),upper_5pct_mean_attempt_cost_usd=sum(tail)/len(tail),mean_fill_fraction=sum(r['fill_fraction'] for r in rr)/n,mean_overflow_kg=sum(r['overflow_kg'] for r in rr)/n,mean_storage_loss_kg=sum(r['storage_loss_kg'] for r in rr)/n,max_abs_balance_error_kg=max(abs(r['balance_error_kg']) for r in rr),missed_counts_by_demand={d.id:sum(d.id in r['missed_ids'] for r in rr) for d in s.demands})

def candidates(s):
    pp=[]
    for n in (2,3,4):
        for gap in (4.,8.):
            for first in (0.,3.):
                for reserve in (0.,500.):
                    p=Policy(f'n{n}_gap{gap:g}_first{first:g}_reserve{reserve:g}',tuple(first+i*gap for i in range(n)),s.max_load_kg,reserve)
                    try: validate(s,p)
                    except ValueError: continue
                    pp.append(p)
    return pp

def pareto_names(rows):
    return [a['policy'] for a in rows if not any(b['mean_attempt_cost_usd']<=a['mean_attempt_cost_usd'] and b['modeled_service_fraction']>=a['modeled_service_fraction'] and (b['mean_attempt_cost_usd']<a['mean_attempt_cost_usd'] or b['modeled_service_fraction']>a['modeled_service_fraction']) for b in rows)]

def run_study(s,policies,train_n=256,test_n=512,seed=42,target=.90):
    validate(s)
    if not policies or len(policies)>100 or len({p.name for p in policies})!=len(policies): raise ValueError('Use 1..100 unique policies')
    if not finite(target) or not 0<target<1: raise ValueError('Target must be in (0,1)')
    shocks=sample_shocks(s,train_n,seed)
    rows=[evaluate(s,p,shocks) for p in policies]
    eligible=[r for r in rows if r['wilson_low_95']>=target]
    chosen=min(eligible,key=lambda r:(r['mean_attempt_cost_usd'],r['policy']))['policy'] if eligible else None
    holdout=None; stress=[]
    if chosen:
        p=next(p for p in policies if p.name==chosen)
        holdout=evaluate(s,p,sample_shocks(s,test_n,seed+1000003))
        for label,changed in [('capacity_minus_20pct',replace(s,capacity_kg=s.capacity_kg*.8)),('lead_plus_3days',replace(s,lead_days=s.lead_days+3)),('storage_loss_double',replace(s,storage_loss_fraction_per_day=min(.999,s.storage_loss_fraction_per_day*2)))]:
            try: stress.append({'case':label,'result':evaluate(changed,p,sample_shocks(changed,test_n,seed+1000003))})
            except ValueError as e: stress.append({'case':label,'error':str(e)})
    return dict(version=VERSION,scenario_sha256=hashlib.sha256(json.dumps(asdict(s),sort_keys=True,allow_nan=False).encode()).hexdigest(),seed=seed,holdout_seed=seed+1000003,selection_target_wilson_lower=target,selection_rows=rows,pareto_on_selection_samples=pareto_names(rows),selected_policy=chosen,holdout=holdout,holdout_target_met=holdout['wilson_low_95']>=target if holdout else None,stress_tests=stress,selection_rule='Cheapest finite-grid policy with selection-sample 95% Wilson lower bound >= target; evaluated once on independent model-sample holdout.',interpretation='Conditional model service frequency, NOT spacecraft reliability or cargo-loss risk.')
