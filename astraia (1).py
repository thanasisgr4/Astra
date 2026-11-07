"""Astraia Assurance Lab 4.1 — single-file research application.
Install: python -m pip install -r requirements.txt
Run: python -m streamlit run astraia.py
Tests: python astraia.py --test
No flight qualification, validated trajectory or funding guarantee.
"""
from dataclasses import dataclass, asdict, replace
import math
import json
import hashlib
import sys
import io
import unittest
import numpy as np
import pandas as pd

VERSION='Astraia Assurance Lab 4.1'
MODEL_ID='astraia-refuel-screening-4.1'
G0=9.80665

@dataclass(frozen=True)
class Scenario:
    payload_kg: float=2500.0
    cargo_dry_kg: float=2000.0
    chemical_isp_s: float=450.0
    electric_isp_s: float=2000.0
    direct_dv: float=5900.0
    cargo_upstream_dv: float=3100.0
    cargo_downstream_dv: float=2800.0
    chemical_tanker_dv: float=3100.0
    electric_tanker_dv: float=4000.0
    chemical_tanker_dry_kg: float=1500.0
    electric_tanker_dry_kg: float=2500.0
    tanker_capacity_kg: float=5000.0
    depot_capacity_kg: float=10000.0
    chemical_cargo_prop_capacity_kg: float=15000.0
    chemical_tanker_prop_capacity_kg: float=15000.0
    electric_tanker_prop_capacity_kg: float=3000.0
    direct_days: float=5.0
    cargo_upstream_days: float=4.0
    cargo_downstream_days: float=3.0
    chemical_tanker_days: float=4.0
    electric_tanker_days: float=90.0
    electric_thrust_n: float=5.0
    electric_power_kw: float=120.0
    electric_efficiency: float=0.65
    electric_duty_fraction: float=0.8
    depot_boil_daily: float=0.001
    transit_cargo_boil_daily: float=0.0005
    transfer_efficiency: float=0.99
    deadline_days: float=20.0
    max_extra_delay_days: float=5.0
    delay_probability: float=0.15
    loss_cv: float=0.25
    launch_cv: float=0.15
    launch_per_kg: float=4000.0
    chemical_propellant_per_kg: float=10.0
    electric_propellant_per_kg: float=1000.0
    cargo_vehicle_service: float=10e6
    chemical_tanker_service: float=3e6
    electric_tanker_service: float=6e6
    depot_service_per_mission: float=2e6
    depot_capital_allocation: float=0.0
    thermal_service_per_kg_day: float=0.0
    delay_cost_per_day: float=80000.0
    seed: int=42
    draws: int=2000

@dataclass(frozen=True)
class Policy:
    route: str='Hub'
    tanker_mode: str='Chemical'
    dispatch_advance_days: float=5.0
    reserve_fraction: float=0.10

@dataclass(frozen=True)
class Requirements:
    max_failure_probability: float=0.05
    risk_weight: float=0.25

PROVENANCE=[
 {'Group':'Destination, delta-v and durations','Status':'Unreviewed','Source / evidence':'','Reviewer':''},
 {'Group':'Vehicle dry masses, propulsion and capacities','Status':'Unreviewed','Source / evidence':'','Reviewer':''},
 {'Group':'Thermal losses and transfer efficiency','Status':'Unreviewed','Source / evidence':'','Reviewer':''},
 {'Group':'Costs and service definitions','Status':'Unreviewed','Source / evidence':'','Reviewer':''},
 {'Group':'Delay and price distributions','Status':'Unreviewed','Source / evidence':'','Reviewer':''},
]

LIMITS=[
 'Research screening of ONE cargo mission; not a trajectory solver, annual inventory scheduler, fleet optimizer or flight-control system.',
 'Both options share the supplied final endpoint, payload and cargo dry mass. User must validate that their delta-v and durations actually represent that endpoint.',
 'The same chemical cargo vehicle flies both hub legs, arrives depleted of upstream propellant and refuels before the downstream leg. No free downstream vehicle.',
 'Tanker batch may be chemical or electric; integer vehicles, fixed dry mass per vehicle and separate cargo/propulsion capacities are counted.',
 'Ideal constant-Isp mass equation. Tanker propulsion is conservatively sized as if all loaded fuel cargo is carried through the full effective delta-v; cargo evaporation during flight does not reduce sized propulsion. This is a screening approximation, not finite-burn integration.',
 'Transit boil-off applies to delivered chemical fuel carried as tanker cargo. Depot loss applies after tanker arrival. Separate own propulsion fuel is assumed thermally managed with negligible storage loss. This needs design validation and costs.',
 'Thermal service price is a user commercial proxy, NOT a physical cooling-power model or evidence of zero-boil-off capability.',
 'Electric dry mass includes power/propulsion hardware. Power and burn-time checks are necessary screening conditions; effective electric delta-v, duration and duty cycle require independent low-thrust analysis.',
 'Chemical thrust, pressure, tank volume, mixture ratios, gravity/steering losses, eclipse power, radiation exposure and launch windows are not solved.',
 'All tankers in a batch share its delay. Price and loss multipliers are independent; no calibrated mission reliability, launch/docking failure or weather model.',
 'Depot capacity is checked on arrival BEFORE storage losses. Invalid capacity does not become a usable service merely because material might later boil off.',
 'Service met means modeled fuel sufficiency, deadline, capacity and electric screening only. Fuel-shortfall or other failure produces no claimed completed delivery; arrival in failed cases is nominal.',
 'Cost is attempted-service expenditure, including failed samples. No imagined recovery mission or penalty purchases missing fuel. Savings headlines require both alternatives to pass the selected independent statistical screen.',
 'No reusable vehicle returns, salvage, financing model, detailed depot deployment or chemical production model. Include relevant service/capital allocation through explicit input prices.',
 'Vehicle/service prices exclude the separately counted LEO launch and commodity propellant prices. Check supplier boundaries to avoid double counting.',
 'Original 37.87% is preserved only as an isolated reproduction of older partial-route equations, not a target, guarantee or SaaS benefit.',
 'No certified TRL, independent validation, patent novelty or ESA/ NASA endorsement is claimed. Default numerical assumptions are illustrative.',
 'No authentication, tenancy isolation or secure persistent database. Do not use confidential customer data in an unsecured public deployment.',
]

def validate(s,p=None,q=None):
    for key,value in asdict(s).items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError(f'{key}: finite numeric value required.')
        if value<0: raise ValueError(f'{key}: cannot be negative.')
    positive=('payload_kg','cargo_dry_kg','chemical_isp_s','electric_isp_s','chemical_tanker_dry_kg','electric_tanker_dry_kg','tanker_capacity_kg','depot_capacity_kg','chemical_cargo_prop_capacity_kg','chemical_tanker_prop_capacity_kg','electric_tanker_prop_capacity_kg','direct_days','cargo_upstream_days','cargo_downstream_days','chemical_tanker_days','electric_tanker_days','electric_thrust_n','electric_power_kw')
    for k in positive:
        if getattr(s,k)<=0:raise ValueError(f'{k}: must be positive.')
    for k in ('depot_boil_daily','transit_cargo_boil_daily'):
        if not 0<=getattr(s,k)<1:raise ValueError(f'{k}: must be in [0,1).')
    for k in ('transfer_efficiency','electric_efficiency','electric_duty_fraction'):
        if not 0<getattr(s,k)<=1:raise ValueError(f'{k}: must be in (0,1].')
    if s.delay_probability>1:raise ValueError('Delay probability exceeds 1.')
    if s.draws!=int(s.draws) or not 100<=s.draws<=20000:raise ValueError('Use 100–20,000 integer samples.')
    if s.seed!=int(s.seed) or not 0<=s.seed<2**32:raise ValueError('Seed must be an integer in [0, 2^32).')
    if max(s.direct_days,s.cargo_upstream_days,s.cargo_downstream_days,s.chemical_tanker_days,s.electric_tanker_days,s.max_extra_delay_days)>3650:raise ValueError('Transfer-time inputs exceed screening bounds.')
    for dv,isp in ((s.direct_dv,s.chemical_isp_s),(s.cargo_upstream_dv,s.chemical_isp_s),(s.cargo_downstream_dv,s.chemical_isp_s),(s.chemical_tanker_dv,s.chemical_isp_s),(s.electric_tanker_dv,s.electric_isp_s)):
        if dv/(G0*isp)>20:raise ValueError('Extreme delta-v/Isp combination.')
    if p:
        if p.route not in ('Direct','Hub') or p.tanker_mode not in ('Chemical','Electric'):raise ValueError('Invalid policy type.')
        for v in (p.dispatch_advance_days,p.reserve_fraction):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('Policy values must be finite and nonnegative.')
        if p.dispatch_advance_days>3650 or p.reserve_fraction>10:raise ValueError('Policy outside screening bounds.')
    if q:
        if not math.isfinite(q.max_failure_probability) or not 0<=q.max_failure_probability<=1:raise ValueError('Invalid service threshold.')
        if not math.isfinite(q.risk_weight) or not 0<=q.risk_weight<=1:raise ValueError('Invalid risk weight.')

def wet_mass(final,dv,isp):
    if not all(math.isfinite(x) for x in (final,dv,isp)) or min(final,dv)<0 or isp<=0:raise ValueError('Invalid burn parameters.')
    exponent=dv/(G0*isp)
    if exponent>20:raise ValueError('Mass ratio exceeds screening limit.')
    return final*math.exp(exponent)

def retention(rate,days):return np.exp(np.log1p(-np.asarray(rate))*np.asarray(days))

def legacy_reference():
    p=100000.;v=450*9.81
    d=p*(1+(math.exp(5900/v)-1)*1.1)
    h=p+p*(math.exp(3800/v)-1)*(2-.995**5)*1.1
    return {'direct_t':d/1000,'hub_t':h/1000,'savings_pct':100*(1-h/d)}

def samples(s):
    validate(s);rng=np.random.default_rng(int(s.seed));n=int(s.draws)
    def lognormal(cv):
        sig=math.sqrt(math.log1p(cv*cv));return rng.lognormal(-sig*sig/2,sig,n)
    def delay():return np.where(rng.random(n)<s.delay_probability,rng.uniform(0,s.max_extra_delay_days,n),0.)
    return {'price':lognormal(s.launch_cv),'loss':lognormal(s.loss_cv),'cargo_delay':delay(),'tanker_delay':delay(),'direct_delay':delay()}

def manifest(s,p):
    validate(s,p);final=s.payload_kg+s.cargo_dry_kg
    base={'Payload':s.payload_kg,'Cargo dry mass':s.cargo_dry_kg}
    if p.route=='Direct':
        mass=wet_mass(final,s.direct_dv,s.chemical_isp_s);prop=mass-final
        base['Direct propulsion fuel']=prop
        out={'leo_mass':mass,'chemical_procured':prop,'electric_procured':0.,'fuel_loaded':0.,'downstream_fuel':0.,'tankers':0,'transit_days':0.,'nominal_storage_days':0.,'electric_burn_days':0.,'required_power_kw':0.,'tank_ok':prop<=s.chemical_cargo_prop_capacity_kg,'power_ok':True,'burn_ok':True,'cargo_tank_ok':prop<=s.chemical_cargo_prop_capacity_kg,'tanker_tank_ok':True}
    else:
        upstream=wet_mass(final,s.cargo_upstream_dv,s.chemical_isp_s)-final
        downstream=wet_mass(final,s.cargo_downstream_dv,s.chemical_isp_s)-final
        electric=p.tanker_mode=='Electric'
        duration=s.electric_tanker_days if electric else s.chemical_tanker_days
        hold=max(s.cargo_upstream_days-(-p.dispatch_advance_days+duration),0.)
        keep=float(retention(s.transit_cargo_boil_daily,duration)*retention(s.depot_boil_daily,hold)*s.transfer_efficiency)
        if keep<1e-10:raise ValueError('Near-total nominal loss: revise timing or thermal assumptions.')
        loaded=downstream*(1+p.reserve_fraction)/keep
        n=math.ceil(loaded/s.tanker_capacity_kg) if loaded>0 else 0
        dry=s.electric_tanker_dry_kg if electric else s.chemical_tanker_dry_kg
        isp=s.electric_isp_s if electric else s.chemical_isp_s
        dv=s.electric_tanker_dv if electric else s.chemical_tanker_dv
        twet=wet_mass(loaded+n*dry,dv,isp);tprop=twet-loaded-n*dry
        mdot=s.electric_thrust_n/(G0*s.electric_isp_s)
        burn=(tprop/n)/mdot/86400 if electric and n else 0.
        power=s.electric_thrust_n*(G0*s.electric_isp_s)/(2*s.electric_efficiency)/1000 if electric and n else 0.
        cargo_ok=max(upstream,downstream)<=s.chemical_cargo_prop_capacity_kg
        capacity=s.electric_tanker_prop_capacity_kg if electric else s.chemical_tanker_prop_capacity_kg
        tanker_ok=(tprop/n<=capacity) if n else True
        power_ok=power<=s.electric_power_kw
        burn_ok=burn<=duration*s.electric_duty_fraction if electric else True
        base.update({'Cargo upstream fuel':upstream,'Depot fuel loaded on tankers':loaded,'Tanker dry mass':n*dry,'Tanker propulsion fuel':tprop})
        out={'leo_mass':final+upstream+twet,'chemical_procured':upstream+loaded+(0. if electric else tprop),'electric_procured':tprop if electric else 0.,'fuel_loaded':loaded,'downstream_fuel':downstream,'tankers':n,'transit_days':duration,'nominal_storage_days':hold,'electric_burn_days':burn,'required_power_kw':power,'tank_ok':cargo_ok and tanker_ok,'power_ok':power_ok,'burn_ok':burn_ok,'cargo_tank_ok':cargo_ok,'tanker_tank_ok':tanker_ok}
    out['components']=base
    out['mass_balance_error']=abs(sum(base.values())-out['leo_mass'])
    if not math.isclose(sum(base.values()),out['leo_mass'],rel_tol=1e-10,abs_tol=1e-5):raise ValueError('LEO manifest conservation failure.')
    return out

def failure_upper(failures,n):
    z=1.6448536269514722;p=failures/n
    return (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def evaluate(s,p,rd=None):
    b=manifest(s,p);rd=samples(s) if rd is None else rd;n=int(s.draws)
    zero=np.zeros(n);reasons={}
    if p.route=='Direct':
        arrival=s.direct_days+rd['direct_delay'];transit=zero;storage=zero
        missing=zero;transit_loss=zero;depot_loss=zero;transfer_loss=zero;residual=zero
        fuel_ok=np.ones(n,dtype=bool);cap_ok=np.ones(n,dtype=bool);thermal=zero;fuel_balance=0.
        fixed=s.cargo_vehicle_service
    else:
        transit=b['transit_days']+rd['tanker_delay']
        cargo_arrival=s.cargo_upstream_days+rd['cargo_delay']
        tanker_arrival=-p.dispatch_advance_days+transit
        if b['tankers']==0:tanker_arrival=cargo_arrival.copy();transit=zero.copy()
        depart=np.maximum(cargo_arrival,tanker_arrival);storage=depart-tanker_arrival
        tr=np.clip(s.transit_cargo_boil_daily*rd['loss'],0,.95)
        dr=np.clip(s.depot_boil_daily*rd['loss'],0,.95)
        on_arrival=b['fuel_loaded']*retention(tr,transit)
        available=on_arrival*retention(dr,storage)
        transit_loss=b['fuel_loaded']-on_arrival;depot_loss=on_arrival-available
        possible=available*s.transfer_efficiency
        allocated=np.minimum(possible,b['downstream_fuel'])
        withdrawn=allocated/s.transfer_efficiency
        transfer_loss=withdrawn-allocated;residual=available-withdrawn
        missing=np.maximum(b['downstream_fuel']-possible,0.)
        fuel_ok=missing<=1e-7;cap_ok=on_arrival<=s.depot_capacity_kg+1e-7
        balance=b['fuel_loaded']-transit_loss-depot_loss-transfer_loss-residual-allocated
        fuel_balance=float(np.max(np.abs(balance)))
        if fuel_balance>1e-5:raise ValueError('Fuel flow conservation failure.')
        arrival=depart+s.cargo_downstream_days
        thermal=b['fuel_loaded']*(transit+storage)*s.thermal_service_per_kg_day
        fixed=(s.cargo_vehicle_service+b['tankers']*(s.electric_tanker_service if p.tanker_mode=='Electric' else s.chemical_tanker_service)+s.depot_service_per_mission+s.depot_capital_allocation)
    late=np.maximum(arrival-s.deadline_days,0.)
    reasons={'Fuel shortage':~fuel_ok,'Deadline missed':late>1e-9,'Depot capacity':~cap_ok,
             'Cargo propulsion tank':np.full(n,not b['cargo_tank_ok']),
             'Tanker propulsion tank':np.full(n,not b['tanker_tank_ok']),
             'Electric power':np.full(n,not b['power_ok']),
             'Electric burn duration':np.full(n,not b['burn_ok'])}
    fail=np.logical_or.reduce(list(reasons.values()));success=~fail
    procurement=b['chemical_procured']*s.chemical_propellant_per_kg+b['electric_procured']*s.electric_propellant_per_kg
    launch=b['leo_mass']*s.launch_per_kg*rd['price'];delay=late*s.delay_cost_per_day
    cost=launch+fixed+procurement+thermal+delay
    p95=float(np.quantile(cost,.95));cvar=float(cost[cost>=p95].mean())
    data=pd.DataFrame({'Cost USD':cost,'Service met':success,'Nominal arrival day':arrival,'Missing fuel kg':missing,
                       'Tanker transit days':transit,'Depot storage days':storage,'Transit fuel loss kg':transit_loss,
                       'Depot fuel loss kg':depot_loss,'Transfer fuel loss kg':transfer_loss,'Depot residual kg':residual,
                       'Launch USD':launch,'Delay cost USD':delay,'Thermal service USD':thermal})
    for name,mask in reasons.items():data['Failure: '+name]=mask
    return {'policy':asdict(p),'manifest':b,'mean_cost':float(cost.mean()),'median_cost':float(np.median(cost)),
            'p95_cost':p95,'cvar95':cvar,'service_probability':float(success.mean()),
            'failure_upper95':failure_upper(int(fail.sum()),n),'p95_arrival':float(np.quantile(arrival,.95)),
            'fuel_balance_error':fuel_balance,'failure_reasons':{name:float(mask.mean()) for name,mask in reasons.items()},
            'fixed_cost':fixed,'commodity_cost':procurement,'data':data}

def candidates():
    return [Policy('Direct','Chemical',0.,0.)]+[Policy('Hub',mode,lead,reserve) for mode in ('Chemical','Electric') for lead in (0.,5.,15.,60.,100.) for reserve in (0.,.05,.1,.15,.25,.35,.5,.75)]

def is_pareto(table):
    answer=np.zeros(len(table),dtype=bool)
    ids=np.flatnonzero(table['Training feasible'].to_numpy())
    vals=table[['Mean cost USD','Failure probability','LEO mass kg']].to_numpy()
    for i in ids:
        competitors=vals[ids]
        dominated=np.any(np.all(competitors<=vals[i],axis=1)&np.any(competitors<vals[i],axis=1))
        answer[i]=not dominated
    return answer

def optimize(s,q):
    validate(s,q=q);rd=samples(s);rows=[]
    for p in candidates():
        try:
            r=evaluate(s,p,rd);failure=1-r['service_probability']
            issues='; '.join(k for k,v in r['failure_reasons'].items() if v>q.max_failure_probability)
            rows.append({**asdict(p),'Mean cost USD':r['mean_cost'],'P95 cost USD':r['p95_cost'],'CVaR95 USD':r['cvar95'],
                         'Failure probability':failure,'LEO mass kg':r['manifest']['leo_mass'],
                         'Score USD':r['mean_cost']+q.risk_weight*(r['cvar95']-r['mean_cost']),
                         'Training feasible':failure<=q.max_failure_probability+1e-12,'Rejection / warning':issues})
        except (ValueError,OverflowError) as exc:
            rows.append({**asdict(p),'Mean cost USD':np.nan,'P95 cost USD':np.nan,'CVaR95 USD':np.nan,'Failure probability':1.,'LEO mass kg':np.nan,'Score USD':np.nan,'Training feasible':False,'Rejection / warning':str(exc)})
    table=pd.DataFrame(rows).sort_values(['Training feasible','Score USD'],ascending=[False,True],na_position='last').reset_index(drop=True)
    table['Pareto among training-feasible']=is_pareto(table)
    feasible=table[table['Training feasible']]
    if feasible.empty:return table,None,None
    row=feasible.iloc[0];p=Policy(row['route'],row['tanker_mode'],float(row['dispatch_advance_days']),float(row['reserve_fraction']))
    hold_s=replace(s,seed=(int(s.seed)+104729)%(2**32))
    return table,p,evaluate(hold_s,p)

def sensitivity(s,p,q):
    rows=[]
    cases=[('Baseline',s)]
    changes=[('LEO launch price','launch_per_kg',(.75,1.25)),('Transit loss rate','transit_cargo_boil_daily',(.5,2.)),('Depot loss rate','depot_boil_daily',(.5,2.)),('Cargo dry mass','cargo_dry_kg',(.8,1.2)),('Electric transit duration','electric_tanker_days',(.8,1.2))]
    for title,key,factors in changes:
        for factor in factors:
            value=getattr(s,key)*factor
            if key in ('transit_cargo_boil_daily','depot_boil_daily'):value=min(value,.95)
            cases.append((f'{title} × {factor:g}',replace(s,**{key:value})))
    for label,ss in cases:
        rd=samples(ss);a=evaluate(ss,Policy('Direct','Chemical',0.,0.),rd);b=evaluate(ss,p,rd)
        rows.append({'Case':label,'Direct mean USD':a['mean_cost'],'Selected mean USD':b['mean_cost'],
                     'Difference USD':a['mean_cost']-b['mean_cost'],
                     'Direct failure upper95':a['failure_upper95'],'Selected failure upper95':b['failure_upper95'],
                     'Both pass confidence screen':a['failure_upper95']<=q.max_failure_probability and b['failure_upper95']<=q.max_failure_probability})
    return pd.DataFrame(rows)

def fingerprint(s,p):
    return hashlib.sha256(json.dumps({'model':MODEL_ID,'scenario':asdict(s),'policy':asdict(p)},sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]

def export_json(s,p,destination,provenance,business=None):
    return json.dumps({'model_id':MODEL_ID,'version':VERSION,'fingerprint':fingerprint(s,p),'scenario':asdict(s),'policy':asdict(p),'destination':destination,'provenance':provenance,'business_eur':business or {},'limitations':LIMITS},indent=2,ensure_ascii=False,allow_nan=False)

def import_json(text):
    data=json.loads(text)
    if not isinstance(data,dict) or data.get('model_id')!=MODEL_ID:raise ValueError('Incompatible scenario/model ID.')
    if set(data.get('scenario',{}))!=set(asdict(Scenario())):raise ValueError('Scenario fields differ from this version.')
    if set(data.get('policy',{}))!=set(asdict(Policy())):raise ValueError('Policy fields differ from this version.')
    s=Scenario(**data['scenario']);p=Policy(**data['policy']);validate(s,p)
    if not isinstance(data.get('destination',''),str):raise ValueError('Destination must be text.')
    prov=data.get('provenance',PROVENANCE)
    if not isinstance(prov,list) or any(not isinstance(x,dict) or set(x)!=set(PROVENANCE[0]) for x in prov):raise ValueError('Invalid provenance table.')
    for row in prov:
        if any(not isinstance(x,str) for x in row.values()):raise ValueError('Provenance values must be text.')
    return s,p,data

def business_model(clients,price,pilot_revenue,direct_cost,opex,cash):
    if any(not math.isfinite(float(x)) or x<0 for x in (clients,price,pilot_revenue,direct_cost,opex,cash)):raise ValueError('Invalid business input.')
    arr=clients*price;revenue=arr+pilot_revenue;net=revenue-direct_cost-opex
    return {'arr':arr,'revenue':revenue,'net':net,'gross_margin':(revenue-direct_cost)/revenue if revenue else None,'runway_months':cash/(-net/12) if net<0 else None}

class VerificationTests(unittest.TestCase):
    def setUp(self):self.s=Scenario(draws=150)
    def test_legacy(self):self.assertAlmostEqual(legacy_reference()['savings_pct'],37.872210,places=5)
    def test_zero_burn(self):self.assertEqual(wet_mass(123,0,450),123)
    def test_known_burn(self):self.assertAlmostEqual(wet_mass(100,G0*450*math.log(2),450),200)
    def test_invalid_burn(self):
        with self.assertRaises(ValueError):wet_mass(1,1,0)
    def test_mass_direct(self):self.assertLess(manifest(self.s,Policy('Direct'))['mass_balance_error'],1e-6)
    def test_mass_hub(self):self.assertLess(manifest(self.s,Policy())['mass_balance_error'],1e-6)
    def test_no_downstream_no_tanker(self):self.assertEqual(manifest(replace(self.s,cargo_downstream_dv=0),Policy())['tankers'],0)
    def test_integer_tankers(self):self.assertIsInstance(manifest(self.s,Policy())['tankers'],int)
    def test_reserve_adds_mass(self):self.assertGreater(manifest(self.s,Policy(reserve_fraction=.5))['leo_mass'],manifest(self.s,Policy(reserve_fraction=0))['leo_mass'])
    def test_transit_losses_raise_loaded_fuel(self):self.assertGreater(manifest(self.s,Policy('Hub','Electric',100,.1))['fuel_loaded'],manifest(replace(self.s,transit_cargo_boil_daily=0),Policy('Hub','Electric',100,.1))['fuel_loaded'])
    def test_zero_losses(self):
        r=evaluate(replace(self.s,depot_boil_daily=0,transit_cargo_boil_daily=0),Policy())
        self.assertEqual(r['data']['Transit fuel loss kg'].sum()+r['data']['Depot fuel loss kg'].sum(),0)
    def test_flow_conservation(self):self.assertLess(evaluate(self.s,Policy())['fuel_balance_error'],1e-5)
    def test_no_nominal_shortage(self):self.assertEqual(evaluate(replace(self.s,loss_cv=0,delay_probability=0),Policy())['data']['Missing fuel kg'].sum(),0)
    def test_seed(self):np.testing.assert_array_equal(samples(self.s)['price'],samples(self.s)['price'])
    def test_zero_delays(self):self.assertEqual(samples(replace(self.s,delay_probability=0))['tanker_delay'].sum(),0)
    def test_bad_depot(self):self.assertEqual(evaluate(replace(self.s,depot_capacity_kg=1),Policy())['service_probability'],0)
    def test_bad_cargo_tank(self):self.assertEqual(evaluate(replace(self.s,chemical_cargo_prop_capacity_kg=1),Policy('Direct'))['service_probability'],0)
    def test_bad_tanker_tank(self):self.assertEqual(evaluate(replace(self.s,chemical_tanker_prop_capacity_kg=1),Policy())['service_probability'],0)
    def test_power_fail(self):self.assertEqual(evaluate(replace(self.s,electric_power_kw=.001),Policy('Hub','Electric',100,.1))['service_probability'],0)
    def test_burn_time_fail(self):self.assertEqual(evaluate(replace(self.s,electric_thrust_n=.00001),Policy('Hub','Electric',100,.1))['service_probability'],0)
    def test_deadline(self):self.assertEqual(evaluate(replace(self.s,deadline_days=0),Policy('Direct'))['service_probability'],0)
    def test_price_deterministic(self):self.assertEqual(evaluate(replace(self.s,launch_cv=0,delay_probability=0),Policy('Direct'))['data']['Cost USD'].nunique(),1)
    def test_confidence_nonzero(self):self.assertGreater(failure_upper(0,150),0)
    def test_all_failure_bound(self):self.assertAlmostEqual(failure_upper(150,150),1)
    def test_invalid_input(self):
        with self.assertRaises(ValueError):validate(replace(self.s,transfer_efficiency=0))
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):validate(replace(self.s,payload_kg=float('nan')))
    def test_hash(self):self.assertEqual(fingerprint(self.s,Policy()),fingerprint(self.s,Policy()))
    def test_roundtrip(self):
        s,p,d=import_json(export_json(self.s,Policy(),'Test orbit',PROVENANCE));self.assertEqual(s,self.s);self.assertEqual(p,Policy())
    def test_bad_model_import(self):
        with self.assertRaises(ValueError):import_json('{"model_id":"wrong"}')
    def test_search_count(self):self.assertEqual(len(candidates()),81)
    def test_optimizer(self):
        table,p,r=optimize(self.s,Requirements());self.assertEqual(len(table),81);self.assertIsNotNone(p);self.assertTrue(table.iloc[0]['Training feasible'])
    def test_no_feasible(self):self.assertIsNone(optimize(replace(self.s,deadline_days=0),Requirements())[1])
    def test_sensitivity(self):self.assertEqual(len(sensitivity(self.s,Policy(),Requirements())),11)
    def test_business(self):
        b=business_model(3,12000,0,10000,58000,10000);self.assertEqual(b['arr'],36000);self.assertEqual(b['net'],-32000)
    def test_zero_revenue_margin(self):self.assertIsNone(business_model(0,0,0,0,0,0)['gross_margin'])
    def test_randomized_mass_invariants(self):
        rng=np.random.default_rng(7)
        for _ in range(25):
            s=replace(self.s,payload_kg=float(rng.uniform(100,10000)),cargo_dry_kg=float(rng.uniform(100,5000)))
            p=Policy('Hub','Electric' if rng.random()<.5 else 'Chemical',float(rng.uniform(0,100)),float(rng.uniform(0,.75)))
            r=evaluate(s,p);self.assertLess(r['fuel_balance_error'],1e-5);self.assertLess(r['manifest']['mass_balance_error'],1e-5)

def run_tests():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VerificationTests))
    return result.wasSuccessful(),stream.getvalue()


def main():
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    st.set_page_config(page_title='Astraia Assurance Lab 4.1',layout='wide')
    st.markdown('<style>.stApp{background:#091321;color:#e9eff8}section[data-testid="stSidebar"]{background:#12223a}div[data-testid="stMetric"]{padding:14px;border:1px solid #365272;border-top:3px solid #498dff;border-radius:8px}</style>',unsafe_allow_html=True)
    st.title('ASTRAIA · ASSURANCE LAB 4.1')
    st.caption('Space logistics screening | Service constraints | Transit losses | Sensitivity | Reproducible evidence')
    st.warning('Ερευνητικό πρωτότυπο. Δεν είναι επικυρωμένη λύση τροχιών, επιχειρησιακό σύστημα ή εγγύηση χρηματοδότησης. Όλες οι προεπιλογές χρειάζονται ανεξάρτητη τεκμηρίωση.')
    def usd(x):return f'${x/1e6:,.2f}M'
    def chart(fig,title):
        fig.update_layout(title=title,template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,use_container_width=True)
    @st.cache_data(show_spinner=False)
    def cached_eval(s,p):return evaluate(s,p)
    @st.cache_data(show_spinner=False)
    def cached_opt(s,q):return optimize(s,q)
    @st.cache_data(show_spinner=False)
    def cached_sensitivity(s,p,q):return sensitivity(s,p,q)
    @st.cache_data(show_spinner=False)
    def cached_tests():return run_tests()
    default=Scenario();initial_policy=Policy();imported={}
    with st.sidebar:
        st.header('Scenario controls')
        uploaded=st.file_uploader('Import v4.1 scenario JSON',type=['json'])
        if uploaded:
            try:default,initial_policy,imported=import_json(uploaded.getvalue().decode('utf-8'))
            except Exception as exc:st.error(f'Import rejected: {exc}');st.stop()
        ident=fingerprint(default,initial_policy)
        destination=st.text_input('Common final destination',value=imported.get('destination','Specified lunar orbit — unvalidated'),key='dest'+ident)
        st.caption('Same delivered payload and cargo dry mass; cargo starts in LEO at t=0. Fuel tankers may depart earlier.')
        groups={
          'Cargo and burn budgets':['payload_kg','cargo_dry_kg','chemical_isp_s','direct_dv','cargo_upstream_dv','cargo_downstream_dv'],
          'Tanker design and limits':['chemical_tanker_dv','electric_tanker_dv','chemical_tanker_dry_kg','electric_tanker_dry_kg','electric_isp_s','tanker_capacity_kg','depot_capacity_kg','chemical_cargo_prop_capacity_kg','chemical_tanker_prop_capacity_kg','electric_tanker_prop_capacity_kg'],
          'Electric propulsion screening':['electric_thrust_n','electric_power_kw','electric_efficiency','electric_duty_fraction'],
          'Timeline and thermal assumptions':['direct_days','cargo_upstream_days','cargo_downstream_days','chemical_tanker_days','electric_tanker_days','transit_cargo_boil_daily','depot_boil_daily','transfer_efficiency','deadline_days'],
          'Customer USD costs':['launch_per_kg','chemical_propellant_per_kg','electric_propellant_per_kg','cargo_vehicle_service','chemical_tanker_service','electric_tanker_service','depot_service_per_mission','depot_capital_allocation','thermal_service_per_kg_day','delay_cost_per_day'],
          'Uncertainty':['max_extra_delay_days','delay_probability','loss_cv','launch_cv','seed','draws']}
        labels={
          'payload_kg':'Delivered payload (kg)','cargo_dry_kg':'Cargo vehicle dry mass (kg)',
          'chemical_isp_s':'Chemical Isp (s)','electric_isp_s':'Electric Isp (s)',
          'direct_dv':'Direct delta-v (m/s)','cargo_upstream_dv':'Cargo LEO → hub delta-v (m/s)',
          'cargo_downstream_dv':'Cargo hub → destination delta-v (m/s)',
          'chemical_tanker_dv':'Chemical tanker delta-v (m/s)','electric_tanker_dv':'Electric effective delta-v (m/s)',
          'chemical_tanker_dry_kg':'Chemical tanker dry mass / trip (kg)',
          'electric_tanker_dry_kg':'Electric tanker dry mass incl. power hardware / trip (kg)',
          'tanker_capacity_kg':'Tanker delivered-fuel cargo capacity / trip (kg)',
          'depot_capacity_kg':'Depot batch capacity (kg)',
          'chemical_cargo_prop_capacity_kg':'Cargo propulsion tank capacity (kg)',
          'chemical_tanker_prop_capacity_kg':'Chemical tanker own propulsion capacity (kg)',
          'electric_tanker_prop_capacity_kg':'Electric tanker own propulsion capacity (kg)',
          'electric_thrust_n':'Thrust per electric tanker (N)','electric_power_kw':'Available electric propulsion power (kW)',
          'electric_efficiency':'Electric efficiency (fraction)','electric_duty_fraction':'Electric duty fraction',
          'transit_cargo_boil_daily':'Daily loss of chemical fuel cargo in transit (fraction)',
          'depot_boil_daily':'Depot daily boil-off fraction','transfer_efficiency':'Depot transfer efficiency',
          'loss_cv':'Loss-rate coefficient of variation','launch_cv':'Launch-price coefficient of variation',
          'launch_per_kg':'LEO transport price ($/kg, excluding commodity procurement)',
          'depot_capital_allocation':'Depot capital/deployment allocation per mission ($)',
          'thermal_service_per_kg_day':'Thermal service proxy ($ / loaded kg / day)',
          'draws':'Scenarios (100–20,000)','seed':'Reproducible random seed'}
        values={}
        for group,keys in groups.items():
            with st.expander(group,expanded=group=='Cargo and burn budgets'):
                for key in keys:
                    value=getattr(default,key);label=labels.get(key,key.replace('_',' '))
                    if key in ('seed','draws'):
                        lo=100 if key=='draws' else 0;hi=20000 if key=='draws' else 2**32-1
                        values[key]=st.number_input(label,min_value=lo,max_value=hi,value=int(value),step=1,key=key+ident)
                    else:
                        hi=.99 if key in ('transit_cargo_boil_daily','depot_boil_daily') else (1. if key in ('electric_efficiency','electric_duty_fraction','transfer_efficiency','delay_probability') else 1e12)
                        values[key]=st.number_input(label,min_value=0.,max_value=hi,value=float(value),format='%.5f' if key in ('transit_cargo_boil_daily','depot_boil_daily','transfer_efficiency','electric_efficiency','electric_duty_fraction','delay_probability') else '%.2f',key=key+ident)
    s=Scenario(**values)
    try:validate(s)
    except ValueError as exc:st.error(str(exc));st.stop()
    comparison,optimization,stress,assurance,business_tab,pilot_tab,export_tab=st.tabs(['Mission comparison','Policy optimization','Sensitivity','Verification / provenance','SaaS business','Pilot evidence','Exports / limitations'])
    with comparison:
        st.subheader('Reference mission — declared common endpoint')
        a,b,c=st.columns(3)
        mode=a.selectbox('Tanker mode',['Chemical','Electric'],index=1 if initial_policy.tanker_mode=='Electric' else 0,key='mode'+ident)
        advance=b.number_input('Tanker dispatch before cargo launch (days)',min_value=0.,max_value=3650.,value=float(initial_policy.dispatch_advance_days),step=5.,key='advance'+ident)
        reserve=c.number_input('Fuel reserve fraction',min_value=0.,max_value=10.,value=float(initial_policy.reserve_fraction),step=.05,key='reserve'+ident)
        p=Policy('Hub',mode,advance,reserve);dp=Policy('Direct','Chemical',0.,0.)
        try:direct=cached_eval(s,dp);hub=cached_eval(s,p)
        except (ValueError,OverflowError) as exc:st.error(str(exc));st.stop()
        cols=st.columns(4)
        cols[0].metric('Direct mean attempted-service cost',usd(direct['mean_cost']))
        cols[1].metric('Hub mean attempted-service cost',usd(hub['mean_cost']))
        cols[2].metric('Hub modeled service probability',f"{100*hub['service_probability']:.2f}%")
        delta=100*(1-hub['manifest']['leo_mass']/direct['manifest']['leo_mass'])
        cols[3].metric('Total LEO mass difference',f'{delta:.2f}%')
        st.caption('Attempted-service cost is spent even when modeled service fails. Negative mass difference means more initial mass for the hub. No cost-saving headline implies successful delivery automatically.')
        rows=[{'Architecture':label,'Component':key,'Mass kg':value} for label,r in [('Direct',direct),('Hub',hub)] for key,value in r['manifest']['components'].items()]
        chart(px.bar(pd.DataFrame(rows),x='Architecture',y='Mass kg',color='Component'),'Initial LEO manifests — downstream fuel counted once')
        reasons=pd.DataFrame([{'Constraint':key,'Direct failure fraction':direct['failure_reasons'][key],'Hub failure fraction':hub['failure_reasons'][key]} for key in direct['failure_reasons']])
        st.subheader('Why a policy fails')
        st.dataframe(reasons,hide_index=True,use_container_width=True)
        st.caption('Failure fractions overlap; do not sum them as a failure probability.')
        fate=pd.DataFrame({'Flow':['Loss in transit','Loss in depot','Transfer loss','Ending depot reserve'],
                           'Mean kg':[hub['data'][col].mean() for col in ('Transit fuel loss kg','Depot fuel loss kg','Transfer fuel loss kg','Depot residual kg')]})
        chart(px.bar(fate,x='Flow',y='Mean kg',color='Flow'),'Disposition of delivered-fuel cargo, excluding allocated burn fuel')
        st.write('Maximum depot flow-balance error:',f"{hub['fuel_balance_error']:.9f} kg")
        if mode=='Electric':
            m=hub['manifest'];a,b,c=st.columns(3)
            a.metric('Required propulsion power / tanker',f"{m['required_power_kw']:.2f} kW")
            b.metric('Ideal powered burn time',f"{m['electric_burn_days']:.2f} days")
            c.metric('Power / burn screen','PASS' if m['power_ok'] and m['burn_ok'] else 'FAIL')
            st.warning('These are necessary screening conditions only. No low-thrust trajectory, eclipse, thermal hardware sizing or chemical thrust solution is generated.')
        times=pd.DataFrame({'Direct nominal arrival':direct['data']['Nominal arrival day'],'Hub nominal arrival':hub['data']['Nominal arrival day']}).melt(var_name='Route',value_name='Arrival day')
        chart(px.histogram(times,x='Arrival day',color='Route',barmode='overlay',opacity=.6),'Arrival distributions — failed-service arrivals are nominal only')
        with st.expander('Original 37.87% reproduction — not a target or validated saving'):
            st.json(legacy_reference());st.write('Fixed 100 t, 450 s, 5900 vs 3800 m/s partial-route reference. Not used by this engine.')
    with optimization:
        st.subheader('Service-constrained search with independent-seed evaluation')
        a,b=st.columns(2)
        limit=a.slider('Maximum modeled failure probability (%)',0.,30.,5.,.5)/100
        weight=b.slider('CVaR tail-risk weight',0.,1.,.25,.05)
        q=Requirements(limit,weight);signature=(s,q)
        st.caption('81 policies: direct, or chemical/electric fuel tankers with dispatch advances 0/5/15/60/100 days and reserves 0/5/10/15/25/35/50/75%. Finite search, not a globally optimal trajectory solution.')
        if st.button('Search policies and run holdout',type='primary',key='optimize'):
            with st.spinner('Searching and evaluating...'):table,best,holdout=cached_opt(s,q)
            st.session_state['optimization_v41']=(signature,table,best,holdout)
        stored=st.session_state.get('optimization_v41')
        if stored and stored[0]==signature:
            table,best,holdout=stored[1:]
            if best is None:st.error('No candidate satisfies the training service requirement. No recommendation issued.')
            else:
                st.write('Selected policy:',asdict(best))
                passed=holdout['failure_upper95']<=limit
                if passed:st.success('Selected policy passes the independent-seed confidence screen under supplied assumptions.')
                else:st.warning('Selected policy does not establish the requested service level on the independent-seed screen.')
                a,b=st.columns(2)
                a.metric('Holdout modeled service probability',f"{100*holdout['service_probability']:.2f}%")
                b.metric('One-sided 95% failure upper bound',f"{100*holdout['failure_upper95']:.2f}%")
                hs=replace(s,seed=(int(s.seed)+104729)%(2**32));hd=evaluate(hs,dp)
                fee=st.number_input('Illustrative annual software fee — customer USD',min_value=0.,value=50000.,step=10000.)
                studies=st.number_input('Comparable studies per year for fee allocation',min_value=1,value=1,step=1)
                if passed and hd['failure_upper95']<=limit:
                    paired=hd['data']['Cost USD'].to_numpy()-holdout['data']['Cost USD'].to_numpy()
                    st.metric('Modeled advantage vs qualified direct baseline, after fee allocation',usd(float(paired.mean())-fee/studies))
                    st.caption('Architecture difference is not proof of incremental software benefit. A customer pilot must also compare with its current workflow and policy.')
                else:st.info('Equivalent-service savings headline withheld because both policies have not passed the independent confidence screen.')
                st.caption('One-sided Wilson bound applies to a fixed selected policy under assumed distributions. It is not physical validation. Repeated tuning to the same holdout seed compromises independence.')
            st.dataframe(table,hide_index=True,use_container_width=True)
            front=table[table['Pareto among training-feasible']]
            if not front.empty:
                chart(px.scatter(front,x='LEO mass kg',y='Mean cost USD',color='tanker_mode',symbol='route',hover_data=['dispatch_advance_days','reserve_fraction','Failure probability']),'Non-dominated training-feasible candidates: cost / failure / mass')
            st.download_button('Export complete search table',table.to_csv(index=False).encode(),file_name='policy_search.csv',mime='text/csv')
        elif stored:st.info('Inputs changed. Run the search again.')
    with stress:
        st.subheader('Sensitivity of the manually selected hub policy')
        st.write('Fixed policy; no re-optimization. Paired seeds isolate changes in assumptions. Input-specific qualification must still be checked.')
        if st.button('Run sensitivity cases',key='sensitivity'):
            try:
                with st.spinner('Evaluating 11 cases...'):sens=cached_sensitivity(s,p,q)
                st.session_state['sensitivity_v41']=((s,p,q),sens)
            except (ValueError,OverflowError) as exc:st.error(str(exc))
        saved=st.session_state.get('sensitivity_v41')
        if saved and saved[0]==(s,p,q):
            sens=saved[1];view=sens.copy();view['Difference USD M']=view['Difference USD']/1e6
            chart(px.bar(view,x='Difference USD M',y='Case',orientation='h',color='Both pass confidence screen'),'Direct minus selected cost — NOT a valid saving where service screen fails')
            st.dataframe(sens,hide_index=True,use_container_width=True)
            st.download_button('Export sensitivity CSV',sens.to_csv(index=False).encode(),file_name='sensitivity.csv',mime='text/csv')
        elif saved:st.info('Inputs changed; rerun sensitivity.')
    with assurance:
        st.subheader('Verification, provenance and review gates')
        if st.button('Run engine verification suite',key='run_tests'):
            ok,report=cached_tests();st.session_state['verification_v41']=(ok,report)
        checked=st.session_state.get('verification_v41')
        if checked:
            ok,report=checked
            if ok:st.success('All engine verification tests passed.')
            else:st.error('Test failure: resolve before relying on outputs.')
            st.code(report)
            st.download_button('Export test report',report.encode(),file_name='verification.txt',mime='text/plain')
        st.caption('Engine tests do not verify astrodynamic truth, customer demand or browser deployment. The package also includes separate Streamlit AppTest tests for execution in an environment with Streamlit installed.')
        provenance=st.data_editor(pd.DataFrame(imported.get('provenance',PROVENANCE)),num_rows='dynamic',hide_index=True,use_container_width=True,key='provenance'+ident)
        st.caption('Do not mark reviewed without the actual reviewer and evidence. The application assigns no automatic TRL or certification.')
    with business_tab:
        st.subheader('Astraia SaaS business — EUR, independent of customer USD')
        a,b=st.columns(2)
        clients=a.number_input('Average active annual contracts',min_value=0,value=3,key='business_clients')
        price=b.number_input('Annual contract price (€)',min_value=0.,value=12000.,step=1000.)
        pilot=a.number_input('Paid pilot revenue in year (€)',min_value=0.,value=0.,step=1000.)
        delivery=b.number_input('Cloud and direct support (€ / year)',min_value=0.,value=10000.,step=1000.)
        opex=a.number_input('Other development / sales / admin (€ / year)',min_value=0.,value=58000.,step=1000.)
        cash=b.number_input('Unrestricted opening cash (€)',min_value=0.,value=10000.,step=1000.)
        br=business_model(clients,price,pilot,delivery,opex,cash)
        business={'inputs':{'clients':clients,'annual_price_eur':price,'pilot_revenue_eur':pilot,'direct_cost_eur':delivery,'other_opex_eur':opex,'unrestricted_cash_eur':cash},'results':br}
        a,b,c=st.columns(3)
        a.metric('ARR assumption',f"€{br['arr']:,.0f}");b.metric('Annual operating cash proxy',f"€{br['net']:,.0f}")
        runway=br['runway_months'];c.metric('Uniform-flow runway',f'{runway:.1f} months' if runway is not None else 'No modeled burn')
        cashdf=pd.DataFrame({'Month':range(13),'Cash EUR':[cash+month*br['net']/12 for month in range(13)]})
        chart(px.line(cashdf,x='Month',y='Cash EUR',markers=True),'Illustrative cash path — no grant assumed received')
        st.caption('Hypotheses, not signed revenue. No taxes, sales ramp or restricted-grant milestone cash timing. Avoid double-counting development and support expenses.')
        st.download_button('Export company cash projection',cashdf.to_csv(index=False).encode(),file_name='company_cash_eur.csv',mime='text/csv')
    with pilot_tab:
        st.subheader('External pilot evidence worksheet')
        st.caption('Session-only, not a secure CRM. Use anonymized data in public deployments.')
        evidence=st.data_editor(pd.DataFrame({'Prospect ID':[''],'User / buyer role':[''],'Problem':[''],'Current workflow':[''],'Evidence status':['Unverified'],'Next action':['']}),num_rows='dynamic',hide_index=True,use_container_width=True,key='pilot_table')
        st.download_button('Export pilot worksheet',evidence.to_csv(index=False).encode(),file_name='pilot_evidence.csv',mime='text/csv')
        st.markdown('1. Agree a customer task and baseline before evaluation.\n2. Independently review a common-endpoint scenario.\n3. Define service requirements and failure handling.\n4. Lock unseen evaluation cases.\n5. Measure analyst time, reproducibility and service constraints.\n6. Record willingness to pay separately from simulated technical benefit.')
    with export_tab:
        st.subheader('Assumptions and limitations')
        for text in LIMITS:st.write('• '+text)
        fp=fingerprint(s,p);st.write('Scenario fingerprint:',fp)
        prov=provenance.fillna('').astype(str).to_dict('records')
        blob=export_json(s,p,destination,prov,business)
        st.download_button('Export scenario JSON',blob.encode(),file_name='astraia_v41_scenario.json',mime='application/json')
        paired=direct['data'].add_prefix('Direct: ').join(hub['data'].add_prefix('Hub: '))
        st.download_button('Export paired Monte Carlo CSV',paired.to_csv(index=False).encode(),file_name='paired_scenarios.csv',mime='text/csv')
        st.download_button('Export initial LEO manifests',pd.DataFrame(rows).to_csv(index=False).encode(),file_name='mass_manifest.csv',mime='text/csv')
        st.json(json.loads(blob),expanded=False)
        st.caption('JSON restores physical scenario, selected policy, destination and provenance. Business results are archived but business widgets are not automatically restored. Current risk settings and candidate-selection state are not restored.')

if __name__=='__main__':
    if '--test' in sys.argv:
        passed,report=run_tests();print(report);sys.exit(0 if passed else 1)
    else:main()
