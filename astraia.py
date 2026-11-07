"""
ASTRAIA Assurance Lab 4.0 — single-file research application.
Run: python -m streamlit run astraia.py
Tests: python astraia.py --test
No flight certification, independent validation or funding guarantee is claimed.
"""
from dataclasses import dataclass, asdict, replace, fields
import math
import json
import hashlib
import sys
import unittest
import io
import numpy as np
import pandas as pd

VERSION = 'Astraia Assurance Lab 4.0'
G0 = 9.80665
MODEL_ID = 'two-leg-refuel-screening-v4'

@dataclass(frozen=True)
class Scenario:
    payload_kg: float = 2500.0
    cargo_dry_kg: float = 2000.0
    chemical_isp_s: float = 450.0
    electric_isp_s: float = 2000.0
    direct_dv: float = 5900.0
    cargo_upstream_dv: float = 3100.0
    cargo_downstream_dv: float = 2800.0
    chemical_tanker_dv: float = 3100.0
    electric_tanker_dv: float = 4000.0
    chemical_tanker_dry_kg: float = 1500.0
    electric_tanker_dry_kg: float = 2500.0
    tanker_capacity_kg: float = 5000.0
    depot_capacity_kg: float = 10000.0
    direct_days: float = 5.0
    cargo_upstream_days: float = 4.0
    cargo_downstream_days: float = 3.0
    chemical_tanker_days: float = 4.0
    electric_tanker_days: float = 90.0
    electric_thrust_n: float = 5.0
    electric_power_kw: float = 120.0
    electric_efficiency: float = 0.65
    electric_duty_fraction: float = 0.80
    depot_boil_daily: float = 0.001
    transfer_efficiency: float = 0.99
    deadline_days: float = 20.0
    max_extra_delay_days: float = 5.0
    delay_probability: float = 0.15
    boil_cv: float = 0.25
    launch_cv: float = 0.15
    launch_per_kg: float = 4000.0
    chemical_propellant_per_kg: float = 10.0
    electric_propellant_per_kg: float = 1000.0
    cargo_vehicle_service: float = 10e6
    chemical_tanker_service: float = 3e6
    electric_tanker_service: float = 6e6
    depot_service_per_mission: float = 2e6
    initial_depot_deployment_allocation: float = 0.0
    delay_cost_per_day: float = 80000.0
    seed: int = 42
    draws: int = 2000

@dataclass(frozen=True)
class Policy:
    route: str = 'Hub'
    tanker_mode: str = 'Chemical'
    dispatch_advance_days: float = 5.0
    reserve_fraction: float = 0.10

@dataclass(frozen=True)
class Qualification:
    maximum_failure_probability: float = 0.05
    risk_weight: float = 0.25
    annual_software_fee_usd: float = 50000.0
    annual_studies: int = 1

PROVENANCE_DEFAULT = [
    {'Input group':'Destination and all delta-v budgets','Status':'Illustrative / unreviewed','Source':'','Reviewer':''},
    {'Input group':'Vehicle masses, capacities and propulsion','Status':'Illustrative / unreviewed','Source':'','Reviewer':''},
    {'Input group':'Transfer times, deadlines and delays','Status':'Illustrative / unreviewed','Source':'','Reviewer':''},
    {'Input group':'Storage loss and transfer efficiency','Status':'Illustrative / unreviewed','Source':'','Reviewer':''},
    {'Input group':'Customer prices and depot allocation','Status':'Illustrative / unreviewed','Source':'','Reviewer':''},
]

def validate(s, p=None):
    for key,value in asdict(s).items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'{key}: a finite numeric value is required.')
        if value<0: raise ValueError(f'{key}: cannot be negative.')
    for key in ('payload_kg','cargo_dry_kg','chemical_isp_s','electric_isp_s','chemical_tanker_dry_kg','electric_tanker_dry_kg','tanker_capacity_kg','depot_capacity_kg','direct_days','cargo_upstream_days','cargo_downstream_days','chemical_tanker_days','electric_tanker_days','electric_thrust_n','electric_power_kw'):
        if getattr(s,key)<=0: raise ValueError(f'{key}: must be positive.')
    if not 0<=s.depot_boil_daily<1: raise ValueError('Daily boil-off must be below 1.')
    for key in ('transfer_efficiency','electric_efficiency','electric_duty_fraction'):
        if not 0<getattr(s,key)<=1: raise ValueError(f'{key}: must be in (0, 1].')
    if not 0<=s.delay_probability<=1: raise ValueError('Delay probability must be in [0,1].')
    if int(s.seed)!=s.seed or not 0<=s.seed<2**32: raise ValueError('Seed must be an integer in [0, 2^32).')
    if int(s.draws)!=s.draws or not 100<=s.draws<=50000: raise ValueError('Draw count must be an integer from 100 to 50000.')
    for dv,isp in ((s.direct_dv,s.chemical_isp_s),(s.cargo_upstream_dv,s.chemical_isp_s),(s.cargo_downstream_dv,s.chemical_isp_s),(s.chemical_tanker_dv,s.chemical_isp_s),(s.electric_tanker_dv,s.electric_isp_s)):
        if dv/(G0*isp)>20: raise ValueError('Extreme delta-v / Isp ratio; revise inputs.')
    if p:
        if p.route not in ('Direct','Hub'): raise ValueError('Route must be Direct or Hub.')
        if p.tanker_mode not in ('Chemical','Electric'): raise ValueError('Unsupported tanker mode.')
        for value in (p.dispatch_advance_days,p.reserve_fraction):
            if not math.isfinite(value) or value<0: raise ValueError('Invalid policy parameter.')
        if p.dispatch_advance_days>3650 or p.reserve_fraction>10: raise ValueError('Policy outside screening limits.')

def wet_mass(final_kg,dv,isp):
    if min(final_kg,dv)<0 or isp<=0: raise ValueError('Invalid burn inputs.')
    return final_kg*math.exp(dv/(G0*isp))

def mean_one_lognormal(rng,cv,n):
    sigma=math.sqrt(math.log1p(cv*cv))
    return rng.lognormal(-.5*sigma*sigma,sigma,n)

def random_scenarios(s):
    validate(s)
    rng=np.random.default_rng(int(s.seed)); n=int(s.draws)
    def delay():
        return np.where(rng.random(n)<s.delay_probability,rng.uniform(0,s.max_extra_delay_days,n),0.)
    return {'launch_multiplier':mean_one_lognormal(rng,s.launch_cv,n),
            'boil_multiplier':mean_one_lognormal(rng,s.boil_cv,n),
            'cargo_delay':delay(),'tanker_delay':delay(),'direct_delay':delay()}

def legacy_reference():
    p=100000.;ve=450*9.81
    direct=p*(1+(math.exp(5900/ve)-1)*1.1)
    fuel=p*(math.exp(3800/ve)-1)
    hub=p+fuel*(2-.995**5)*1.1
    return {'direct_t':direct/1000,'hub_t':hub/1000,'savings_pct':100*(1-hub/direct)}

def manifest(s,p):
    validate(s,p)
    final=s.payload_kg+s.cargo_dry_kg
    if p.route=='Direct':
        wet=wet_mass(final,s.direct_dv,s.chemical_isp_s)
        rows={'Payload':s.payload_kg,'Cargo vehicle dry mass':s.cargo_dry_kg,'Direct burn propellant':wet-final}
        return {'components':rows,'leo_mass':wet,'chemical_procured':wet-final,'electric_procured':0.,
                'depot_loaded':0.,'downstream_required':0.,'tankers':0,'capacity_ok':True,'electric_ok':True,
                'electric_burn_days':0.,'electric_power_required_kw':0.}
    upstream_wet=wet_mass(final,s.cargo_upstream_dv,s.chemical_isp_s)
    need=wet_mass(final,s.cargo_downstream_dv,s.chemical_isp_s)-final
    mode=p.tanker_mode
    transit=s.chemical_tanker_days if mode=='Chemical' else s.electric_tanker_days
    nominal_arrival=-p.dispatch_advance_days+transit
    nominal_hold=max(s.cargo_upstream_days-nominal_arrival,0.)
    retained=(1-s.depot_boil_daily)**nominal_hold
    if retained<1e-12: raise ValueError('Near-total storage loss: reduce prepositioning or boil-off.')
    loaded=need*(1+p.reserve_fraction)/(retained*s.transfer_efficiency)
    n=math.ceil(loaded/s.tanker_capacity_kg) if loaded>0 else 0
    dry_per=s.chemical_tanker_dry_kg if mode=='Chemical' else s.electric_tanker_dry_kg
    isp=s.chemical_isp_s if mode=='Chemical' else s.electric_isp_s
    dv=s.chemical_tanker_dv if mode=='Chemical' else s.electric_tanker_dv
    tanker_final=loaded+n*dry_per
    tanker_wet=wet_mass(tanker_final,dv,isp)
    tanker_prop=tanker_wet-tanker_final
    rows={'Payload':s.payload_kg,'Cargo vehicle dry mass':s.cargo_dry_kg,
          'Cargo upstream propellant':upstream_wet-final,'Depot propellant cargo':loaded,
          'Tanker dry mass':n*dry_per,'Tanker propulsion propellant':tanker_prop}
    power=s.electric_thrust_n*(s.electric_isp_s*G0)/(2*s.electric_efficiency)/1000
    per_tanker_prop=tanker_prop/n if n else 0.
    mdot=s.electric_thrust_n/(s.electric_isp_s*G0)
    burn_days=per_tanker_prop/mdot/86400 if mode=='Electric' else 0.
    electric_ok=(mode!='Electric' or n==0 or (power<=s.electric_power_kw and burn_days<=s.electric_tanker_days*s.electric_duty_fraction))
    result={'components':rows,'leo_mass':upstream_wet+tanker_wet,
            'chemical_procured':upstream_wet-final+loaded+(tanker_prop if mode=='Chemical' else 0.),
            'electric_procured':tanker_prop if mode=='Electric' else 0.,
            'depot_loaded':loaded,'downstream_required':need,'tankers':n,
            'capacity_ok':loaded<=s.depot_capacity_kg+1e-8,'electric_ok':electric_ok,
            'electric_burn_days':burn_days,'electric_power_required_kw':power if mode=='Electric' else 0.}
    if not math.isclose(sum(rows.values()),result['leo_mass'],rel_tol=1e-10,abs_tol=1e-6):
        raise ValueError('Launch manifest does not balance.')
    return result

def wilson_failure_upper(failures,n,z=1.6448536269514722):
    # One-sided 95% Wilson upper bound for an independently evaluated FIXED policy.
    ph=failures/n;den=1+z*z/n
    return (ph+z*z/(2*n)+z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n)))/den

def evaluate(s,p,rd=None):
    b=manifest(s,p);rd=random_scenarios(s) if rd is None else rd;n=int(s.draws)
    if p.route=='Direct':
        arrival=s.direct_days+rd['direct_delay']
        sufficient=np.ones(n,dtype=bool);hold=np.zeros(n);remaining=np.zeros(n);loss=np.zeros(n)
        missing=np.zeros(n);residual=np.zeros(n);delivered=np.zeros(n)
        fixed=s.cargo_vehicle_service
    else:
        cargo_arrival=s.cargo_upstream_days+rd['cargo_delay']
        tanker_arrival=-p.dispatch_advance_days+(s.chemical_tanker_days if p.tanker_mode=='Chemical' else s.electric_tanker_days)+rd['tanker_delay']
        if b['tankers']==0: tanker_arrival=cargo_arrival.copy()
        departure=np.maximum(cargo_arrival,tanker_arrival)
        hold=departure-tanker_arrival
        rate=np.clip(s.depot_boil_daily*rd['boil_multiplier'],0.,.95)
        remaining=b['depot_loaded']*(1-rate)**hold
        loss=b['depot_loaded']-remaining
        usable=remaining*s.transfer_efficiency
        missing=np.maximum(b['downstream_required']-usable,0.)
        sufficient=missing<=1e-7
        # Only the required burn fuel is loaded into the cargo vehicle.
        delivered=np.minimum(usable,b['downstream_required'])
        withdrawn=delivered/s.transfer_efficiency
        transfer_loss=withdrawn-delivered
        residual=remaining-withdrawn
        check=b['depot_loaded']-loss-transfer_loss-delivered-residual
        if not np.allclose(check,0,atol=1e-5): raise ValueError('Depot mass accounting failed.')
        arrival=departure+s.cargo_downstream_days
        fixed=(s.cargo_vehicle_service+b['tankers']*(s.chemical_tanker_service if p.tanker_mode=='Chemical' else s.electric_tanker_service)
               +s.depot_service_per_mission+s.initial_depot_deployment_allocation)
    late=np.maximum(arrival-s.deadline_days,0.)
    success=sufficient&(late<=1e-9)&b['capacity_ok']&b['electric_ok']
    procurement=b['chemical_procured']*s.chemical_propellant_per_kg+b['electric_procured']*s.electric_propellant_per_kg
    launch=b['leo_mass']*s.launch_per_kg*rd['launch_multiplier']
    cost=launch+procurement+fixed+late*s.delay_cost_per_day
    p95=float(np.quantile(cost,.95));cvar=float(cost[cost>=p95].mean());fail=int((~success).sum())
    return {'policy':asdict(p),'manifest':b,'mean_cost':float(cost.mean()),'median_cost':float(np.median(cost)),
            'p95_cost':p95,'cvar95':cvar,'success_probability':float(success.mean()),
            'failure_upper95':wilson_failure_upper(fail,n),'mean_arrival_days':float(arrival.mean()),
            'p95_arrival_days':float(np.quantile(arrival,.95)),
            'mean_missing_kg':float(missing.mean()),'mean_boiloff_kg':float(loss.mean()),
            'mean_residual_kg':float(residual.mean()),'fixed_cost':fixed,'procurement':procurement,
            'data':pd.DataFrame({'Cost USD':cost,'Nominal arrival day':arrival,'Fuel missing kg':missing,
                                 'Storage days':hold,'Depot boiloff kg':loss,'Depot residual kg':residual,
                                 'Service met':success,'Launch USD':launch})}

def optimize(s,q):
    if not 0<=q.maximum_failure_probability<=1 or not 0<=q.risk_weight<=1: raise ValueError('Invalid risk constraint.')
    rd=random_scenarios(s)
    policies=[Policy('Direct','Chemical',0.,0.)]
    policies += [Policy('Hub',mode,lead,reserve)
                 for mode in ('Chemical','Electric') for lead in (0.,5.,15.,60.,100.) for reserve in (0.,.05,.1,.15,.25,.35,.5,.75)]
    rows=[]
    for p in policies:
        try:
            r=evaluate(s,p,rd)
            score=r['mean_cost']+q.risk_weight*(r['cvar95']-r['mean_cost'])
            feasible=1-r['success_probability']<=q.maximum_failure_probability+1e-12
            rows.append({**asdict(p),'Mean cost USD':r['mean_cost'],'P95 cost USD':r['p95_cost'],
                         'CVaR95 USD':r['cvar95'],'Service probability':r['success_probability'],
                         'Risk score USD':score,'LEO mass kg':r['manifest']['leo_mass'],
                         'Feasible in training':feasible,'Issue':''})
        except (ValueError,OverflowError) as exc:
            rows.append({**asdict(p),'Mean cost USD':np.nan,'P95 cost USD':np.nan,'CVaR95 USD':np.nan,
                         'Service probability':0.,'Risk score USD':np.nan,'LEO mass kg':np.nan,
                         'Feasible in training':False,'Issue':str(exc)})
    table=pd.DataFrame(rows).sort_values(['Feasible in training','Risk score USD'],ascending=[False,True],na_position='last').reset_index(drop=True)
    f=table[table['Feasible in training']]
    if f.empty: return table,None,None
    best=f.iloc[0]
    p=Policy(str(best['route']),str(best['tanker_mode']),float(best['dispatch_advance_days']),float(best['reserve_fraction']))
    holdout=replace(s,seed=(int(s.seed)+104729)%(2**32))
    evaluation=evaluate(holdout,p)
    return table,p,evaluation

def scenario_hash(s,p):
    blob=json.dumps({'model':MODEL_ID,'scenario':asdict(s),'policy':asdict(p)},sort_keys=True,allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

class VerificationTests(unittest.TestCase):
    def setUp(self): self.s=Scenario(draws=200)
    def test_reference(self): self.assertAlmostEqual(legacy_reference()['savings_pct'],37.872210,places=5)
    def test_zero_dv(self): self.assertEqual(wet_mass(100,0,450),100)
    def test_rocket_equation(self): self.assertAlmostEqual(wet_mass(100,G0*450*math.log(2),450),200)
    def test_direct_manifest(self):
        b=manifest(self.s,Policy('Direct'));self.assertAlmostEqual(sum(b['components'].values()),b['leo_mass'])
    def test_hub_manifest(self):
        b=manifest(self.s,Policy());self.assertAlmostEqual(sum(b['components'].values()),b['leo_mass'])
    def test_zero_downstream_no_tanker(self): self.assertEqual(manifest(replace(self.s,cargo_downstream_dv=0),Policy())['tankers'],0)
    def test_integer_tankers(self): self.assertIsInstance(manifest(self.s,Policy())['tankers'],int)
    def test_reserve_costs_mass(self): self.assertGreater(manifest(self.s,Policy(reserve_fraction=.5))['leo_mass'],manifest(self.s,Policy(reserve_fraction=0))['leo_mass'])
    def test_fixed_seed(self): np.testing.assert_equal(random_scenarios(self.s)['cargo_delay'],random_scenarios(self.s)['cargo_delay'])
    def test_no_boiloff(self): self.assertEqual(evaluate(replace(self.s,depot_boil_daily=0),Policy())['mean_boiloff_kg'],0)
    def test_capacity_failure(self): self.assertEqual(evaluate(replace(self.s,depot_capacity_kg=1),Policy())['success_probability'],0)
    def test_power_failure(self): self.assertEqual(evaluate(replace(self.s,electric_power_kw=1),Policy('Hub','Electric',100,.1))['success_probability'],0)
    def test_electric_thrust_time(self): self.assertEqual(evaluate(replace(self.s,electric_thrust_n=.0001),Policy('Hub','Electric',100,.1))['success_probability'],0)
    def test_deadline_failure(self): self.assertEqual(evaluate(replace(self.s,deadline_days=0),Policy('Direct'))['success_probability'],0)
    def test_deterministic_direct(self):
        s=replace(self.s,launch_cv=0,delay_probability=0)
        r=evaluate(s,Policy('Direct'));self.assertEqual(r['success_probability'],1);self.assertEqual(r['data']['Cost USD'].nunique(),1)
    def test_wilson(self): self.assertGreater(wilson_failure_upper(0,200),0)
    def test_zero_transfer_rejected(self):
        with self.assertRaises(ValueError):validate(replace(self.s,transfer_efficiency=0))
    def test_negative_payload_rejected(self):
        with self.assertRaises(ValueError):validate(replace(self.s,payload_kg=-1))
    def test_fingerprint(self): self.assertEqual(scenario_hash(self.s,Policy()),scenario_hash(self.s,Policy()))
    def test_search(self):
        table,p,hold=optimize(self.s,Qualification());self.assertEqual(len(table),81);self.assertIsNotNone(p)
    def test_no_feasible_policy(self):
        table,p,hold=optimize(replace(self.s,deadline_days=0),Qualification());self.assertIsNone(p)
    def test_no_delay(self):self.assertEqual(random_scenarios(replace(self.s,delay_probability=0))['tanker_delay'].sum(),0)
    def test_nominal_fuel_buffer(self):
        s=replace(self.s,boil_cv=0,delay_probability=0)
        self.assertEqual(evaluate(s,Policy())['mean_missing_kg'],0)

def verification_report():
    out=io.StringIO()
    result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VerificationTests))
    return result.wasSuccessful(),out.getvalue()

def json_clean(obj):
    if isinstance(obj,dict):return {k:json_clean(v) for k,v in obj.items()}
    if isinstance(obj,list):return [json_clean(v) for v in obj]
    if isinstance(obj,(np.integer,np.floating,np.bool_)):return obj.item()
    return obj

def main():
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    st.set_page_config(page_title='Astraia Assurance Lab',layout='wide')
    st.markdown('<style>.stApp{background:#09111d;color:#e8edf6}section[data-testid="stSidebar"]{background:#101e31}div[data-testid="stMetric"]{background:#14233a;border:1px solid #314866;border-top:3px solid #468bff;border-radius:7px;padding:14px}</style>',unsafe_allow_html=True)
    st.title('ASTRAIA · ASSURANCE LAB')
    st.caption(VERSION+' | Propellant logistics | Service constraints | Independent scenario evaluation')
    st.warning('Ερευνητικό πρωτότυπο — όχι επικυρωμένο μοντέλο αποστολής ή εγγύηση χρηματοδότησης. Οι προεπιλογές είναι ενδεικτικές. Δεν έχει αποδοθεί TRL ή πιστοποίηση NASA/ESA.')
    def money(x):return f'${x/1e6:,.2f}M'
    def plot(fig,title):
        fig.update_layout(title=title,template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,use_container_width=True)
    @st.cache_data(show_spinner=False)
    def cached_eval(s,p):return evaluate(s,p)
    @st.cache_data(show_spinner=False)
    def cached_search(s,q):return optimize(s,q)
    @st.cache_data(show_spinner=False)
    def cached_tests():return verification_report()
    defaults=asdict(Scenario())
    with st.sidebar:
        st.header('Scenario / Σενάριο')
        incoming=st.file_uploader('Import scenario JSON',type='json')
        if incoming:
            try:
                data=json.load(incoming)
                if data.get('model_id')!=MODEL_ID:raise ValueError('Incompatible model ID.')
                values=data['scenario']
                if not isinstance(values,dict) or set(values)!=set(defaults):raise ValueError('Scenario fields do not match.')
                imported=Scenario(**values);validate(imported);defaults=asdict(imported)
            except Exception as exc:
                st.error(f'Import rejected: {exc}');st.stop()
        input_id=hashlib.sha256(json.dumps(defaults,sort_keys=True).encode()).hexdigest()[:8]
        destination=st.text_input('Common destination / κοινός προορισμός',value=data.get('destination','Specified lunar orbit — unvalidated') if incoming else 'Specified lunar orbit — unvalidated',key='destination_'+input_id)
        st.caption('LEO at t=0 for cargo. Tanker dispatch can occur earlier. Same payload and cargo dry mass at the same final endpoint.')
        groups={
            'Payload and cargo vehicle':['payload_kg','cargo_dry_kg','chemical_isp_s','direct_dv','cargo_upstream_dv','cargo_downstream_dv'],
            'Tankers and electric screening':['chemical_tanker_dv','electric_tanker_dv','chemical_tanker_dry_kg','electric_tanker_dry_kg','electric_isp_s','tanker_capacity_kg','electric_thrust_n','electric_power_kw','electric_efficiency','electric_duty_fraction'],
            'Timeline / storage':['direct_days','cargo_upstream_days','cargo_downstream_days','chemical_tanker_days','electric_tanker_days','depot_capacity_kg','depot_boil_daily','transfer_efficiency','deadline_days'],
            'Costs — customer USD':['launch_per_kg','chemical_propellant_per_kg','electric_propellant_per_kg','cargo_vehicle_service','chemical_tanker_service','electric_tanker_service','depot_service_per_mission','initial_depot_deployment_allocation','delay_cost_per_day'],
            'Uncertainty':['max_extra_delay_days','delay_probability','boil_cv','launch_cv','seed','draws']}
        labels={
            'payload_kg':'Delivered payload (kg)','cargo_dry_kg':'Cargo vehicle dry mass (kg)',
            'chemical_isp_s':'Chemical Isp (s)','electric_isp_s':'Electric Isp (s)',
            'direct_dv':'Direct delta-v (m/s)','cargo_upstream_dv':'Cargo LEO → hub delta-v (m/s)',
            'cargo_downstream_dv':'Cargo hub → destination delta-v (m/s)',
            'chemical_tanker_dv':'Chemical tanker delta-v (m/s)','electric_tanker_dv':'Electric tanker effective delta-v (m/s)',
            'chemical_tanker_dry_kg':'Chemical tanker dry mass / trip (kg)',
            'electric_tanker_dry_kg':'Electric tanker dry mass / trip, INCLUDING power system (kg)',
            'tanker_capacity_kg':'Tanker propellant cargo capacity (kg)',
            'electric_thrust_n':'Electric thrust per tanker (N)','electric_power_kw':'Available electric propulsion power per tanker (kW)',
            'electric_efficiency':'Electric efficiency (fraction)','electric_duty_fraction':'Electric duty fraction',
            'depot_boil_daily':'Depot daily boil-off fraction','transfer_efficiency':'Depot-to-cargo transfer efficiency',
            'initial_depot_deployment_allocation':'Allocated depot deployment / capital cost per mission ($)',
            'delay_probability':'Probability of additional transfer delay','draws':'Monte Carlo samples'}
        values={}
        for group,keys in groups.items():
            with st.expander(group,expanded=group=='Payload and cargo vehicle'):
                for key in keys:
                    label=labels.get(key,key.replace('_',' '))
                    if key in ('seed','draws'):
                        lower=100 if key=='draws' else 0;upper=10000 if key=='draws' else 2**32-1
                        # Imported larger simulations are valid in engine but capped for the public UI.
                        default=min(int(defaults[key]),upper)
                        values[key]=st.number_input(label,min_value=lower,max_value=upper,value=default,step=1,key=key+input_id)
                    else:
                        upper=1e12
                        if key in ('electric_efficiency','electric_duty_fraction','transfer_efficiency','delay_probability'):upper=1.
                        if key=='depot_boil_daily':upper=.99
                        values[key]=st.number_input(label,min_value=0.,max_value=upper,value=float(defaults[key]),format='%.5f' if key in ('depot_boil_daily','transfer_efficiency','electric_efficiency','electric_duty_fraction','delay_probability') else '%.2f',key=key+input_id)
    s=Scenario(**values)
    try:validate(s)
    except ValueError as exc:st.error(str(exc));st.stop()
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(['Mission comparison','Policy optimization','Assurance / tests','SaaS business','Pilot evidence','Exports / model limits'])
    with tab1:
        st.subheader('Complete declared architecture — one cargo mission')
        a,b,c=st.columns(3)
        imported_policy=data.get('policy',{}) if incoming else {}
        mode=a.selectbox('Tanker propulsion',['Chemical','Electric'],index=1 if imported_policy.get('tanker_mode')=='Electric' else 0,key='mode'+input_id)
        lead=b.number_input('Tanker dispatch BEFORE cargo launch (days)',min_value=0.,max_value=3650.,value=float(imported_policy.get('dispatch_advance_days',5.)),step=5.,key='lead'+input_id)
        reserve=c.slider('Fuel reserve fraction',0.,1.,min(1.,float(imported_policy.get('reserve_fraction',.1))),.05,key='reserve'+input_id)
        p=Policy('Hub',mode,lead,reserve);dp=Policy('Direct','Chemical',0.,0.)
        try:
            direct=cached_eval(s,dp);hub=cached_eval(s,p)
        except (ValueError,OverflowError) as exc:st.error(str(exc));st.stop()
        cols=st.columns(4)
        cols[0].metric('Direct attempted-service cost',money(direct['mean_cost']))
        cols[1].metric('Hub attempted-service cost',money(hub['mean_cost']))
        cols[2].metric('Hub modeled service probability',f"{hub['success_probability']*100:.2f}%")
        mass_savings=100*(1-hub['manifest']['leo_mass']/direct['manifest']['leo_mass'])
        cols[3].metric('Total initial LEO-mass difference',f'{mass_savings:.2f}%')
        st.caption('Costs include attempted service even in failed scenarios. A cheaper failed policy is NOT equivalent delivery. Service = enough downstream fuel, deadline met, storage capacity and electric screening passed. No launch/docking failure model.')
        rows=[]
        for name,r in [('Direct',direct),('Hub',hub)]:
            for component,mass in r['manifest']['components'].items():rows.append({'Architecture':name,'Component':component,'kg':mass})
        plot(px.bar(pd.DataFrame(rows),x='Architecture',y='kg',color='Component'),'Initial LEO launch manifest — no double-counted downstream fuel')
        stats=pd.DataFrame([{'Architecture':name,'Mean USD':r['mean_cost'],'P95 USD':r['p95_cost'],'CVaR95 USD':r['cvar95'],'Service probability':r['success_probability'],'Mean missing fuel kg':r['mean_missing_kg'],'P95 arrival day':r['p95_arrival_days']} for name,r in [('Direct',direct),('Hub',hub)]])
        st.dataframe(stats,hide_index=True,use_container_width=True)
        if mode=='Electric':
            m=hub['manifest'];a,b,c=st.columns(3)
            a.metric('Power required per tanker',f"{m['electric_power_required_kw']:.2f} kW")
            b.metric('Ideal powered burn time',f"{m['electric_burn_days']:.2f} days")
            c.metric('Power / burn-time screening','PASS' if m['electric_ok'] else 'FAIL')
            st.warning('These are necessary screening conditions, not a low-thrust trajectory solution. User-supplied delta-v, transit time and dry mass must be independently validated.')
        if not hub['manifest']['capacity_ok']:st.error('Depot capacity is insufficient for the supplied propellant batch.')
        timeline=[]
        for label,event in [('Direct arrival',direct['data']['Nominal arrival day']),('Hub destination arrival',hub['data']['Nominal arrival day'])]:
            timeline.extend({'Path':label,'Arrival day':float(x)} for x in event)
        plot(px.histogram(pd.DataFrame(timeline),x='Arrival day',color='Path',barmode='overlay',opacity=.6),'Arrival-time distributions')
        with st.expander('Fixed original 38% reproduction — NOT a validated claim'):
            st.json(legacy_reference());st.write('Original 100 t partial-route equations only. Not used to constrain the new model.')
    with tab2:
        st.subheader('Risk-aware policy selection and independent-seed service check')
        a,b,c=st.columns(3)
        failure=a.slider('Maximum modeled service-failure probability (%)',0.,30.,5.,.5)/100
        risk=b.slider('CVaR tail weight',0.,1.,.25,.05)
        annual_studies=c.number_input('Studies/year for illustrative fee allocation',min_value=1,value=1,step=1)
        fee=st.number_input('Annual software fee — customer USD, illustrative',min_value=0.,value=50000.,step=10000.)
        q=Qualification(failure,risk,fee,int(annual_studies));signature=(s,q)
        st.caption('81 candidates: direct plus chemical/electric tanker policies with dispatch advances 0/5/15/60/100 days and fuel reserves 0/5/10/15/25/35/50/75%. No autonomous route optimization. Objective: mean + weight × (CVaR95 − mean).')
        if st.button('Search and independently evaluate',type='primary'):
            with st.spinner('Evaluating feasible candidate policies...'):
                table,best,holdout=cached_search(s,q)
            st.session_state['search_v4']=(signature,table,best,holdout)
        stored=st.session_state.get('search_v4')
        if stored and stored[0]==signature:
            table,best,holdout=stored[1:]
            if best is None:st.error('No sampled policy satisfies the training service constraint. Revise the architecture or requirements; no savings recommendation is issued.')
            else:
                st.write('Selected policy',asdict(best))
                acceptable=holdout['failure_upper95']<=failure
                if acceptable:st.success('Independent-seed service screen passes the selected statistical confidence criterion.')
                else:st.warning('Independent-seed screen does NOT establish the requested service level. Do not present this policy as qualified.')
                a,b=st.columns(2)
                a.metric('Holdout service probability',f"{holdout['success_probability']*100:.2f}%")
                b.metric('One-sided 95% failure upper bound',f"{holdout['failure_upper95']*100:.2f}%")
                st.caption('Confidence bound applies to this fixed policy under the assumed distribution. It does not validate that distribution, the mission physics, or operational reliability. Do not repeatedly tune to the holdout seed and call it independent validation.')
                hs=replace(s,seed=(int(s.seed)+104729)%(2**32))
                hd=evaluate(hs,dp)
                direct_qualified=hd['failure_upper95']<=failure
                if acceptable and direct_qualified:
                    paired=hd['data']['Cost USD'].to_numpy()-holdout['data']['Cost USD'].to_numpy()
                    benefit=float(paired.mean())-fee/annual_studies
                    st.metric('Modeled expected benefit vs qualified direct baseline, after fee allocation',money(benefit))
                    st.caption('This estimates architecture/policy difference, NOT proven causal SaaS value. Real pilots must compare against the customer\'s current planning policy and measure study effort separately.')
                else:st.info('Equivalent-service savings headline withheld: both policies must pass the independent service screen.')
            st.dataframe(table,hide_index=True,use_container_width=True)
            st.download_button('Export candidate table',table.to_csv(index=False).encode(),file_name='candidate_policies.csv',mime='text/csv')
        elif stored:st.info('Inputs changed. Search again.')
    with tab3:
        st.subheader('Verification is not physical validation')
        st.write('The engine uses the ideal rocket equation. Software tests check mathematical and accounting behavior, not a NASA/ESA endorsement or flight readiness.')
        if st.button('Run built-in verification tests'):
            ok,report=cached_tests()
            if ok:st.success('All built-in verification tests passed.')
            else:st.error('Verification failures. Do not rely on outputs until corrected.')
            st.code(report)
            st.download_button('Export test report',report.encode(),file_name='verification_report.txt',mime='text/plain')
        st.subheader('Parameter provenance register')
        base=data.get('provenance',PROVENANCE_DEFAULT) if incoming else PROVENANCE_DEFAULT
        if not isinstance(base,list):base=PROVENANCE_DEFAULT
        provenance=st.data_editor(pd.DataFrame(base),num_rows='dynamic',hide_index=True,use_container_width=True,key='provenance'+input_id)
        st.caption('Do not mark a group reviewed without the actual source/reviewer. Missing provenance remains visible; no automatic maturity rating is generated.')
    with tab4:
        st.subheader('Astraia software business — EUR, separate from customer USD costs')
        a,b=st.columns(2)
        clients=a.number_input('Average active annual contracts',min_value=0,value=3)
        price=b.number_input('Annual contract value (€)',min_value=0.,value=12000.,step=1000.)
        pilots=a.number_input('Total paid pilot revenue in year (€)',min_value=0.,value=0.,step=1000.)
        delivery=b.number_input('Annual cloud / direct support (€)',min_value=0.,value=10000.,step=1000.)
        operating=a.number_input('Annual development, sales and administration (€)',min_value=0.,value=58000.,step=1000.)
        cash=b.number_input('Available unrestricted opening cash (€)',min_value=0.,value=10000.,step=1000.)
        arr=clients*price;net=arr+pilots-delivery-operating
        c1,c2,c3=st.columns(3)
        c1.metric('Recurring revenue assumption',f'€{arr:,.0f}')
        c2.metric('Annual operating cash proxy',f'€{net:,.0f}')
        runway=cash/(-net/12) if net<0 else None
        c3.metric('Uniform-cash-flow runway',f'{runway:.1f} months' if runway is not None else 'No modeled burn')
        business={'average_contracts':clients,'annual_price_eur':price,'pilot_revenue_eur':pilots,'direct_cost_eur':delivery,'other_opex_eur':operating,'unrestricted_opening_cash_eur':cash,'arr_eur':arr,'net_operating_eur':net}
        plot(px.line(pd.DataFrame({'Month':range(13),'Cash EUR':[cash+i*net/12 for i in range(13)]}),x='Month',y='Cash EUR',markers=True),'Illustrative cash path — no grant assumed received')
        st.caption('Hypothetical subscriptions, not contracts or a sales forecast. No automatic grant amount, salary eligibility, tax or exchange-rate assumption. Restricted incubation funding and milestone cash timing require a separate approved budget.')
    with tab5:
        st.subheader('Pilot evidence — record, never fabricate')
        st.caption('Session-only worksheet. Do not enter confidential data into an unsecured public deployment.')
        evidence=st.data_editor(pd.DataFrame({'Anonymized prospect':[''],'User / buyer role':[''],'Planning problem':[''],'Current alternative':[''],'Evidence status':['Unverified'],'Next action':['']}),num_rows='dynamic',hide_index=True,use_container_width=True,key='pilot')
        st.download_button('Export evidence worksheet',evidence.to_csv(index=False).encode(),file_name='pilot_evidence.csv',mime='text/csv')
        st.markdown('1. Agree customer task and baseline before evaluation.\n2. Obtain independently reviewed inputs.\n3. Define service requirements and failure handling.\n4. Lock held-out evaluation cases.\n5. Measure analyst time, reproducibility and constraints.\n6. Document purchase interest separately from measured technical benefit.')
    with tab6:
        st.subheader('Scope and physical limitations')
        limits=[
            'Single cargo mission; both alternatives deliver the same payload and cargo dry mass to the same declared endpoint.',
            'Cargo vehicle is the SAME physical chemical vehicle on both hub legs: it arrives empty of upstream fuel, refuels, and continues. No additional free downstream spacecraft.',
            'Tanker propellant cargo is accounted once in initial LEO manifests. Integer tankers include dry mass and their own propulsion fuel.',
            'Electric propulsion is used only for PREPOSITIONING depot fuel. Time, power and burn checks are necessary conditions, NOT a validated low-thrust trajectory.',
            'User-supplied delta-v and durations are NOT solved astrodynamics. Chemical and electric tanker budgets are independently supplied.',
            'All propulsion mass budgets use ideal constant-Isp burns. No gravity loss, drag, finite-burn steering, residuals or mixture-ratio model.',
            'Cargo chemical tank capacity and thrust adequacy are NOT validated. Tanker capacity refers to propellant cargo, not the tanker propulsion tank.',
            'Boil-off applies only after tanker arrival at depot. ZERO in-transit propellant loss and zero cargo-vehicle pre-burn loss are assumed. Validate thermal management or extend the model before use.',
            'Depot volume/mixture handling and docking/transfer duration are not modeled. Transfer efficiency is a lumped supplied assumption.',
            'Electric dry mass must include propulsion and power hardware. No automatic dry-mass discount. Power availability is assumed during thrust duty time.',
            'All tankers in one prepositioning batch share its sampled delay. No launch/docking failures or independent tanker failure model.',
            'A fuel shortage is a failed service: no partial delivery or imaginary penalty-based recovery. Cost still records money spent attempting that service.',
            'Arrival time in failed-fuel cases is nominal only, NOT actual completed delivery.',
            'Vehicle/tanker service prices exclude the separately itemized LEO launch and propulsion commodity charges. Verify no double counting in supplier quotes.',
            'Depot service and deployment allocation are per-mission user costs. No full depot investment, fleet reuse or annual schedule is optimized.',
            'No salvage credit, return missions, reusable fleet economics, autonomous commands, live telemetry or operational certification.',
            'Input sources and customer traction are not supplied by simulation. All defaults are illustrative. No guaranteed 38% savings, TRL, patentability or ESA BIC admission.',
            'No authentication, secure multi-tenant storage or production security certification. Do not upload confidential customer datasets.',
        ]
        for x in limits:st.write('• '+x)
        fingerprint=scenario_hash(s,p)
        export={'model_id':MODEL_ID,'version':VERSION,'fingerprint':fingerprint,'scenario':asdict(s),'policy':asdict(p),'destination':destination,'provenance':provenance.fillna('').to_dict('records'),'business_eur':business,'limitations':limits}
        st.write('Scenario fingerprint:',fingerprint)
        st.download_button('Export reproducible scenario JSON',json.dumps(json_clean(export),indent=2,allow_nan=False).encode(),file_name='astraia_scenario.json',mime='application/json')
        paired=pd.DataFrame({'Direct cost USD':direct['data']['Cost USD'],'Hub cost USD':hub['data']['Cost USD'],'Direct service met':direct['data']['Service met'],'Hub service met':hub['data']['Service met'],'Hub missing fuel kg':hub['data']['Fuel missing kg'],'Hub nominal arrival day':hub['data']['Nominal arrival day']})
        st.download_button('Export paired scenarios CSV',paired.to_csv(index=False).encode(),file_name='paired_scenarios.csv',mime='text/csv')
        manifests=pd.DataFrame(rows)
        st.download_button('Export initial mass manifests',manifests.to_csv(index=False).encode(),file_name='mass_manifests.csv',mime='text/csv')
        st.json(export,expanded=False)

if __name__=='__main__':
    if '--test' in sys.argv:
        ok,report=verification_report();print(report);sys.exit(0 if ok else 1)
    else:
        main()
