"""Astraia 6.0 — Orbital Propellant Service-Level Engine.

Treats cryogenic/storable propellant delivery in orbit as a PERISHABLE-INVENTORY SUPPLY CHAIN:
make-to-order (dedicated tanker per request) vs make-to-stock (pre-positioned stock with
base-stock replenishment) vs hybrid, evaluated on supply-chain service metrics
(on-time fill rate, OTIF, safety stock, launches, cost per kg delivered on time).

Run:   python -m streamlit run astraia6.py
Tests: python astraia6.py --test
Research screening model. Not a trajectory solver; no flight qualification, no funding guarantee.
Default numbers are ILLUSTRATIVE and must be replaced by cited data (see provenance table).
"""
from dataclasses import dataclass, asdict, replace
import math, json, hashlib, sys, io, unittest
import numpy as np
import pandas as pd

VERSION = 'Astraia 6.0 — Orbital Propellant Service-Level Engine'
MODEL_ID = 'astraia-service-network-6.0'
G0 = 9.80665
ARCHS = ('MTO', 'MTS', 'MTS_MTO')


@dataclass(frozen=True)
class NetScenario:
    # demand side
    horizon_days: float = 365.0
    missions_per_year: float = 6.0          # refuel requests per year (Poisson)
    mean_demand_kg: float = 6000.0          # propellant per request, delivered at customer vehicle
    demand_cv: float = 0.4
    promised_days: float = 30.0             # service promise: request -> fuel delivered
    # tanker / launch
    tanker_capacity_kg: float = 5000.0
    tanker_dry_kg: float = 1500.0
    chemical_isp_s: float = 450.0
    tanker_leo_dv: float = 150.0
    leo_rendezvous_days: float = 1.5
    tanker_cadence_days: float = 3.0        # spacing between launches of one multi-tanker order
    mto_prep_days: float = 40.0             # request -> launch for a dedicated (on-demand) launch
    rep_prep_days: float = 40.0             # replenishment order -> launch (planned launch)
    min_order_fraction: float = 0.25        # do not launch below this fraction of one tanker
    # thermal / transfer
    storage_boil_daily: float = 0.002       # perishable stock loss in orbital storage
    transit_cargo_boil_daily: float = 0.0005
    transfer_efficiency: float = 0.99
    # risk
    launch_reliability: float = 0.98
    delay_probability: float = 0.15
    max_extra_delay_days: float = 5.0
    loss_cv: float = 0.25
    launch_cv: float = 0.15
    # USD costs
    launch_per_kg: float = 4000.0
    tanker_launch_price_factor: float = 1.0
    tanker_service: float = 3e6             # per tanker flight (vehicle/ops/amortization)
    propellant_per_kg: float = 10.0
    depot_service_per_year: float = 4e6     # storage-vehicle operations (stock policies only)
    depot_setup_allocation: float = 0.0
    penalty_unserved_per_kg: float = 100.0  # customer cost of fuel never delivered in horizon
    penalty_late_per_kg: float = 40.0       # customer cost of fuel delivered after promise
    seed: int = 42
    draws: int = 500


@dataclass(frozen=True)
class NetPolicy:
    arch: str = 'MTS_MTO'
    stock_multiple: float = 2.0     # base-stock level S as multiple of mean demand per request
    review_days: float = 14.0


@dataclass(frozen=True)
class Target:
    min_fill_rate: float = 0.95     # required lower confidence bound of on-time fill rate


PROVENANCE = [{'Group': g, 'Status': 'Unreviewed', 'Source / evidence': '', 'Reviewer': ''} for g in (
    'Demand: request rate, size, promise (customer interviews, market studies)',
    'Launch: price, capacity, reliability, lead time to launch slot (provider documents, flight records)',
    'Tanker: dry mass, capacity, Isp, rendezvous dv',
    'Thermal: storage and transit boil-off, transfer efficiency (flight demonstrations, test reports)',
    'Costs: tanker service, depot operations, penalties (quotes, contracts)')]

LIMITS = [
    'Research screening simulation of ONE service area over one horizon; not a trajectory solver, vehicle design tool or flight-operations system.',
    'Propellant is modeled as perishable inventory at one orbital storage location; the existence, cost and TRL of such storage must be evidenced by the user. Stock policies are scenarios, not claims that a service exists.',
    'Requests are Poisson arrivals with lognormal sizes; real demand is correlated and may be far lumpier. Demand parameters need customer evidence.',
    'Unmet stock demand is lost (MTS) or served by an expedited dedicated launch (MTS_MTO), which is usually late. No backorder queue beyond that.',
    'Launches fail independently; failed launches cost money and deliver nothing; no recovery mission, insurance or relaunch is modeled.',
    'Delivered-fuel sizing uses nominal boil-off; actual loss multipliers are random, so shortages arise. Tanker propulsion for the LEO rendezvous uses the ideal rocket equation.',
    'Service confidence uses a one-sided normal bound over Monte Carlo replications under assumed distributions. It is not physical validation. Re-tuning on the holdout seed compromises independence.',
    'Costs exclude financing, taxes, insurance, salvage and reusable-vehicle refurbishment unless included in input prices.',
    'No ESA, NASA or provider endorsement, certification or validated TRL is claimed.',
    'No authentication or secure persistence; do not enter confidential data in a public deployment.']


# ---------------------------------------------------------------- helpers
def retention(rate, days):
    return np.exp(np.log1p(-np.asarray(rate)) * np.asarray(days))


def validate(s, p=None, t=None):
    for k, v in asdict(s).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f'{k}: finite numeric value required.')
        if v < 0:
            raise ValueError(f'{k}: cannot be negative.')
    for k in ('horizon_days', 'mean_demand_kg', 'promised_days', 'tanker_capacity_kg', 'tanker_dry_kg',
              'chemical_isp_s', 'leo_rendezvous_days', 'launch_per_kg'):
        if getattr(s, k) <= 0:
            raise ValueError(f'{k}: must be positive.')
    for k in ('storage_boil_daily', 'transit_cargo_boil_daily'):
        if not 0 <= getattr(s, k) < 1:
            raise ValueError(f'{k}: must be in [0,1).')
    if not 0 < s.transfer_efficiency <= 1:
        raise ValueError('transfer_efficiency: must be in (0,1].')
    for k in ('launch_reliability', 'delay_probability'):
        if getattr(s, k) > 1:
            raise ValueError(f'{k}: cannot exceed 1.')
    if not 30 <= s.horizon_days <= 1825 or s.horizon_days != int(s.horizon_days):
        raise ValueError('horizon_days: integer in [30,1825].')
    if s.draws != int(s.draws) or not 100 <= s.draws <= 5000:
        raise ValueError('draws: integer in [100,5000].')
    if s.seed != int(s.seed) or not 0 <= s.seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2^32).')
    if s.missions_per_year > 400 or s.tanker_leo_dv / (G0 * s.chemical_isp_s) > 5:
        raise ValueError('Input outside screening bounds.')
    if p:
        if p.arch not in ARCHS:
            raise ValueError('Invalid architecture.')
        if not (math.isfinite(p.stock_multiple) and 0 <= p.stock_multiple <= 20):
            raise ValueError('stock_multiple must be in [0,20].')
        if not (math.isfinite(p.review_days) and 1 <= p.review_days <= 365):
            raise ValueError('review_days must be in [1,365].')
    if t and not 0 <= t.min_fill_rate <= 1:
        raise ValueError('min_fill_rate must be in [0,1].')


def samples(s):
    """Common random numbers shared by all policies."""
    validate(s)
    rng = np.random.default_rng(int(s.seed)); n = int(s.draws); h = int(s.horizon_days)

    def logn(cv, shape=None):
        sg = math.sqrt(math.log1p(cv * cv)); return rng.lognormal(-sg * sg / 2, sg, shape or n)

    def slip():
        return np.where(rng.random((n, h)) < s.delay_probability, rng.uniform(0, s.max_extra_delay_days, (n, h)), 0.)

    return {'event': rng.random((n, h)) < 1 - math.exp(-s.missions_per_year / 365.),
            'size': s.mean_demand_kg * logn(s.demand_cv, (n, h)),
            'slip': slip(), 'slip2': slip(), 'fail_u': rng.random((n, h)), 'fail_u2': rng.random((n, h)),
            'init_u': rng.random(n), 'price': logn(s.launch_cv), 'loss': logn(s.loss_cv)}


# ---------------------------------------------------------------- simulation
def simulate(s, p, rd=None):
    validate(s, p)
    rd = samples(s) if rd is None else rd
    n = int(s.draws); h = int(s.horizon_days); rows = np.arange(n)
    mto = p.arch == 'MTO'; hybrid = p.arch in ('MTO', 'MTS_MTO')
    cap = s.tanker_capacity_kg; dry = s.tanker_dry_kg; price = rd['price']
    tr = np.clip(s.transit_cargo_boil_daily * rd['loss'], 0, .95)
    sg = np.clip(s.storage_boil_daily * rd['loss'], 0, .95)
    keep_nom = float(retention(s.transit_cargo_boil_daily, s.leo_rendezvous_days)) * s.transfer_efficiency
    keep_act = retention(tr, s.leo_rendezvous_days) * s.transfer_efficiency
    prop_ratio = math.exp(s.tanker_leo_dv / (G0 * s.chemical_isp_s)) - 1
    S = 0. if mto else p.stock_multiple * s.mean_demand_kg
    review = int(round(p.review_days))
    cost_launch = np.zeros(n); proc = np.zeros(n); launches = np.zeros(n)

    def order(qnet, mask, u, prep, slip):
        qn = np.where(mask, qnet, 0.); loaded = qn / keep_nom
        k = np.where(qn > 0, np.ceil(loaded / cap - 1e-12), 0.)
        per = np.where(k > 0, loaded / np.maximum(k, 1.), 0.)
        rp = np.where(k > 0, (dry + per) * prop_ratio, 0.)
        cost_launch[:] += (k * (dry + per) + k * rp) * s.launch_per_kg * price * s.tanker_launch_price_factor + k * s.tanker_service
        proc[:] += (loaded + k * rp) * s.propellant_per_kg
        launches[:] += k
        ok = u < s.launch_reliability ** k
        lead = prep + s.leo_rendezvous_days + slip + np.maximum(k - 1, 0) * s.tanker_cadence_days
        return ok, lead, qn * keep_act / keep_nom

    stock = np.zeros(n)
    if not mto and S > 0:
        ok, _, net = order(np.full(n, S), np.ones(n, dtype=bool), rd['init_u'], 0., 0.)
        stock = np.where(ok, net, 0.)
    init_stock = stock.copy()
    arr = np.zeros((n, h + 1)); arr_nom = np.zeros((n, h + 1)); pipe = np.zeros(n)
    ontime = np.zeros(n); late = np.zeros(n); unserved = np.zeros(n); dem = np.zeros(n)
    miss = np.zeros(n); otif = np.zeros(n); stock_sum = np.zeros(n)
    boil = np.zeros(n); arrivals = np.zeros(n); from_stock = np.zeros(n)
    for t in range(h):
        stock += arr[:, t]; arrivals += arr[:, t]; pipe -= arr_nom[:, t]
        loss_t = stock * sg; stock -= loss_t; boil += loss_t
        stock_sum += stock
        d = np.where(rd['event'][:, t], rd['size'][:, t], 0.); ev = d > 0
        take = np.zeros(n) if mto else np.minimum(stock, d)
        stock -= take; from_stock += take; ontime += take
        rem = d - take; need = rem > 1e-9; full = ev & ~need
        if hybrid and need.any():
            ok, lead, net = order(rem, need, rd['fail_u2'][:, t], s.mto_prep_days, rd['slip2'][:, t])
            delivered = np.where(ok & need, np.minimum(net, rem), 0.)
            inwin = lead <= s.promised_days
            ontime += np.where(inwin, delivered, 0.); late += np.where(inwin, 0., delivered)
            unserved += rem - delivered
            full |= need & inwin & ok & (delivered >= rem - 1e-6)
        else:
            unserved += rem
        dem += d; miss += ev; otif += full
        if not mto and t % review == 0:
            qnet = S - (stock + pipe)
            mask = qnet >= s.min_order_fraction * cap
            ok, lead, net = order(qnet, mask, rd['fail_u'][:, t], s.rep_prep_days, rd['slip'][:, t])
            go = mask & ok
            idx = np.minimum(t + np.ceil(lead).astype(int), h)
            arr[rows, idx] += np.where(go, net, 0.); arr_nom[rows, idx] += np.where(go, qnet, 0.)
            pipe += np.where(go, qnet, 0.)
    balance = init_stock + arrivals - from_stock - boil - stock
    depot = 0. if mto else s.depot_service_per_year * h / 365. + s.depot_setup_allocation
    cost = (cost_launch + proc + depot + unserved * s.penalty_unserved_per_kg + late * s.penalty_late_per_kg)
    return dict(cost=cost, fill=np.where(dem > 0, ontime / np.maximum(dem, 1e-12), 1.),
                otif=np.where(miss > 0, otif / np.maximum(miss, 1), 1.), late=late, unserved=unserved, demand=dem,
                ontime=ontime, launches=launches, avg_stock=stock_sum / h, boil=boil,
                balance_error=float(np.max(np.abs(balance))), S=S, depot=depot)


def evaluate(s, p, rd=None, target=None):
    r = simulate(s, p, rd); n = len(r['cost']); z = 1.6448536269514722
    fill = r['fill']; lb = float(fill.mean() - z * fill.std(ddof=1) / math.sqrt(n))
    p95 = float(np.quantile(r['cost'], .95)); ontime_kg = float(r['ontime'].mean())
    per = s_period(s, p)
    return {'policy': asdict(p), 'mean_cost': float(r['cost'].mean()), 'p95_cost': p95,
            'cvar95': float(r['cost'][r['cost'] >= p95].mean()), 'fill_mean': float(fill.mean()), 'fill_lb95': lb,
            'otif_mean': float(r['otif'].mean()), 'late_kg': float(r['late'].mean()), 'unserved_kg': float(r['unserved'].mean()),
            'launches': float(r['launches'].mean()), 'avg_stock_kg': float(r['avg_stock'].mean()),
            'boil_kg': float(r['boil'].mean()), 'base_stock_kg': r['S'],
            'safety_stock_kg': r['S'] - per if p.arch != 'MTO' else 0.,
            'cost_per_ontime_kg': float(r['cost'].mean() / ontime_kg) if ontime_kg > 0 else float('nan'),
            'balance_error': r['balance_error'], 'raw': r}


def s_period(s, p):
    """Expected demand over review period + replenishment lead time (cycle-stock reference)."""
    lt = s.rep_prep_days + s.leo_rendezvous_days
    return s.missions_per_year / 365. * s.mean_demand_kg * (p.review_days + lt)


# ---------------------------------------------------------------- search, frontier, sensitivity
def candidates():
    out = [NetPolicy('MTO', 0., 1.)]
    out += [NetPolicy(a, m, r) for a in ('MTS', 'MTS_MTO') for m in (.5, 1., 1.5, 2., 3., 4.) for r in (7., 14., 30., 60.)]
    return out


def search(s, t, rd=None):
    rd = samples(s) if rd is None else rd; rows = []
    for p in candidates():
        e = evaluate(s, p, rd)
        rows.append({**asdict(p), 'Mean cost USD': e['mean_cost'], 'P95 cost USD': e['p95_cost'], 'On-time fill mean': e['fill_mean'],
                     'Fill lower95': e['fill_lb95'], 'OTIF mean': e['otif_mean'], 'Launches': e['launches'],
                     'Avg stock kg': e['avg_stock_kg'], 'Base stock kg': e['base_stock_kg'], 'Safety stock kg': e['safety_stock_kg'],
                     'Boil-off kg': e['boil_kg'], 'Late kg': e['late_kg'], 'Unserved kg': e['unserved_kg'],
                     'Cost / on-time kg USD': e['cost_per_ontime_kg'], 'Feasible': e['fill_lb95'] >= t.min_fill_rate})
    return pd.DataFrame(rows).sort_values(['Feasible', 'Mean cost USD'], ascending=[False, True]).reset_index(drop=True)


def optimize(s, t):
    validate(s, t=t); table = search(s, t); ok = table[table['Feasible']]
    if ok.empty:
        return table, None, None
    r = ok.iloc[0]; p = NetPolicy(r['arch'], float(r['stock_multiple']), float(r['review_days']))
    hs = replace(s, seed=(int(s.seed) + 104729) % 2 ** 32)
    return table, p, evaluate(hs, p)


def best_by_arch(table):
    rows = []
    for a in ARCHS:
        sub = table[(table['arch'] == a) & table['Feasible']]
        if sub.empty:
            rows.append({'Architecture': a, 'Status': 'No feasible policy', 'Mean cost USD': np.nan})
        else:
            r = sub.iloc[0]
            rows.append({'Architecture': a, 'Status': 'Feasible', 'Mean cost USD': r['Mean cost USD'], 'Fill lower95': r['Fill lower95'],
                         'Base stock kg': r['Base stock kg'], 'Safety stock kg': r['Safety stock kg'], 'Launches': r['Launches']})
    return pd.DataFrame(rows)


def service_frontier(s, t, promises=(15., 30., 45., 60., 90.)):
    """How the cheapest feasible architecture changes with the promised service time."""
    rows = []
    for pr in promises:
        ss = replace(s, promised_days=pr, draws=min(int(s.draws), 300)); b = best_by_arch(search(ss, t))
        ok = b.dropna(subset=['Mean cost USD'])
        rec = {'Promised days': pr, 'Winner': ok.sort_values('Mean cost USD').iloc[0]['Architecture'] if len(ok) else 'None feasible'}
        for _, r in b.iterrows():
            rec[r['Architecture'] + ' cost USD'] = r['Mean cost USD']
        rows.append(rec)
    return pd.DataFrame(rows)


def sensitivity(s, p):
    cases = [('Baseline', s)]
    for title, key, fs in (('Request rate', 'missions_per_year', (.5, 2.)), ('Request size', 'mean_demand_kg', (.5, 1.5)),
                           ('Storage boil-off', 'storage_boil_daily', (.5, 3.)), ('Launch price', 'launch_per_kg', (.75, 1.25)),
                           ('Launch lead time', 'rep_prep_days', (.5, 1.5))):
        for f in fs:
            v = getattr(s, key) * f
            if key == 'storage_boil_daily': v = min(v, .95)
            cases.append((f'{title} × {f:g}', replace(s, **{key: v})))
    for f in (.5, 2.):
        cases.append((f'Launch failure rate × {f:g}', replace(s, launch_reliability=max(0., 1 - (1 - s.launch_reliability) * f))))
    rows = []
    for label, ss in cases:
        rd = samples(ss); a = evaluate(ss, NetPolicy('MTO', 0., 1.), rd); b = evaluate(ss, p, rd)
        rows.append({'Case': label, 'MTO mean USD': a['mean_cost'], 'Selected mean USD': b['mean_cost'], 'MTO fill lower95': a['fill_lb95'],
                     'Selected fill lower95': b['fill_lb95'], 'Selected cheaper': b['mean_cost'] < a['mean_cost']})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- persistence, business
def fingerprint(s, p):
    return hashlib.sha256(json.dumps({'m': MODEL_ID, 's': asdict(s), 'p': asdict(p)}, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def export_json(s, p, t, provenance):
    return json.dumps({'model_id': MODEL_ID, 'version': VERSION, 'fingerprint': fingerprint(s, p), 'scenario': asdict(s),
                       'policy': asdict(p), 'target': asdict(t), 'provenance': provenance, 'limitations': LIMITS},
                      indent=2, ensure_ascii=False, allow_nan=False)


def import_json(text):
    d = json.loads(text)
    if not isinstance(d, dict) or d.get('model_id') != MODEL_ID:
        raise ValueError('Incompatible scenario/model ID.')
    if set(d.get('scenario', {})) != set(asdict(NetScenario())) or set(d.get('policy', {})) != set(asdict(NetPolicy())):
        raise ValueError('Fields differ from this version.')
    s = NetScenario(**d['scenario']); p = NetPolicy(**d['policy']); t = Target(**d.get('target', asdict(Target())))
    validate(s, p, t)
    prov = d.get('provenance', PROVENANCE)
    if not isinstance(prov, list) or any(not isinstance(x, dict) or set(x) != set(PROVENANCE[0]) or any(not isinstance(v, str) for v in x.values()) for x in prov):
        raise ValueError('Invalid provenance table.')
    return s, p, t, prov


# ---------------------------------------------------------------- tests
class Tests(unittest.TestCase):
    def setUp(self):
        self.s = NetScenario(draws=200, horizon_days=365)
        self.mts = NetPolicy('MTS', 2., 14.); self.hyb = NetPolicy('MTS_MTO', 2., 14.); self.mto = NetPolicy('MTO', 0., 1.)

    def test_stock_conservation(self):
        for p in (self.mts, self.hyb):
            self.assertLess(simulate(self.s, p)['balance_error'], 1e-6)

    def test_stock_conservation_stress(self):
        s = replace(self.s, storage_boil_daily=.05, loss_cv=1., delay_probability=1., launch_reliability=.7)
        self.assertLess(simulate(s, self.hyb)['balance_error'], 1e-6)

    def test_zero_boil(self):
        self.assertEqual(simulate(replace(self.s, storage_boil_daily=0.), self.mts)['boil'].sum(), 0)

    def test_no_demand_full_service(self):
        r = simulate(replace(self.s, missions_per_year=0.), self.mts)
        self.assertTrue(np.all(r['fill'] == 1)); self.assertTrue(np.all(r['otif'] == 1))

    def test_mto_cannot_beat_prep_time(self):
        self.assertEqual(simulate(replace(self.s, promised_days=10.), self.mto)['fill'].max(), 0)

    def test_mto_meets_long_promise(self):
        s = replace(self.s, promised_days=200., launch_reliability=1., loss_cv=0., delay_probability=0.)
        self.assertGreater(simulate(s, self.mto)['fill'].mean(), .99)

    def test_zero_reliability(self):
        s = replace(self.s, launch_reliability=0.)
        for p in (self.mto, self.mts): self.assertEqual(simulate(s, p)['fill'].max(), 0)

    def test_stock_improves_service_under_tight_promise(self):
        self.assertGreater(evaluate(self.s, self.mts)['fill_mean'], evaluate(self.s, self.mto)['fill_mean'])

    def test_more_stock_not_worse(self):
        s = replace(self.s, storage_boil_daily=0., launch_reliability=1., loss_cv=0., delay_probability=0.)
        lo = evaluate(s, NetPolicy('MTS', 1., 14.))['fill_mean']; hi = evaluate(s, NetPolicy('MTS', 4., 14.))['fill_mean']
        self.assertGreaterEqual(hi + 1e-9, lo)

    def test_mto_has_no_depot_cost(self):
        self.assertEqual(simulate(self.s, self.mto)['depot'], 0)

    def test_safety_stock_definition(self):
        e = evaluate(self.s, self.mts); self.assertAlmostEqual(e['safety_stock_kg'], e['base_stock_kg'] - s_period(self.s, self.mts))

    def test_seed(self):
        a = evaluate(self.s, self.hyb)['mean_cost']; b = evaluate(self.s, self.hyb)['mean_cost']; self.assertEqual(a, b)

    def test_invalid(self):
        for bad in (replace(self.s, mean_demand_kg=0.), replace(self.s, launch_reliability=1.5), replace(self.s, draws=10),
                    replace(self.s, storage_boil_daily=float('nan'))):
            with self.assertRaises(ValueError): validate(bad)
        with self.assertRaises(ValueError): validate(self.s, NetPolicy('X'))

    def test_roundtrip(self):
        s, p, t, prov = import_json(export_json(self.s, self.hyb, Target(), PROVENANCE))
        self.assertEqual((s, p, t), (self.s, self.hyb, Target()))

    def test_bad_import(self):
        with self.assertRaises(ValueError): import_json('{"model_id":"x"}')

    def test_candidates(self): self.assertEqual(len(candidates()), 49)

    def test_optimizer(self):
        table, p, h = optimize(self.s, Target(.3)); self.assertIsNotNone(p); self.assertTrue(table.iloc[0]['Feasible'])

    def test_no_feasible(self): self.assertIsNone(optimize(self.s, Target(1.0))[1])

    def test_frontier(self):
        f = service_frontier(replace(self.s, draws=100), Target(.5), (30., 90.)); self.assertEqual(len(f), 2)

    def test_sensitivity(self): self.assertEqual(len(sensitivity(replace(self.s, draws=100), self.hyb)), 13)


def run_tests():
    stream = io.StringIO()
    r = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return r.wasSuccessful(), stream.getvalue()


# ---------------------------------------------------------------- UI
def main():
    import streamlit as st
    import plotly.express as px
    st.set_page_config(page_title='Astraia 6.0', layout='wide')
    st.title('ASTRAIA 6.0 · Orbital Propellant Service-Level Engine')
    st.caption('Propellant delivery as a perishable-inventory supply chain: make-to-order vs make-to-stock vs hybrid · service level, safety stock, boil-off, launch risk')
    st.warning('Research prototype. Defaults are illustrative, not evidence. Replace them with cited data in the provenance tab. '
               'Ερευνητικό πρωτότυπο: τα προεπιλεγμένα νούμερα δεν είναι τεκμηριωμένα.')

    def usd(x): return f'${x / 1e6:,.2f}M'

    def chart(fig, title):
        fig.update_layout(title=title, template='plotly_dark'); st.plotly_chart(fig, use_container_width=True)

    @st.cache_data(show_spinner=False)
    def c_opt(s, t): return optimize(s, t)
    @st.cache_data(show_spinner=False)
    def c_front(s, t): return service_frontier(s, t)
    @st.cache_data(show_spinner=False)
    def c_sens(s, p): return sensitivity(s, p)

    default, ipol, itarget, iprov = NetScenario(), NetPolicy(), Target(), PROVENANCE
    with st.sidebar:
        st.header('Scenario')
        up = st.file_uploader('Import v6.0 scenario JSON', type=['json'])
        if up:
            try: default, ipol, itarget, iprov = import_json(up.getvalue().decode('utf-8'))
            except Exception as exc: st.error(f'Import rejected: {exc}'); st.stop()
        ident = fingerprint(default, ipol)
        groups = {'Demand and promise': ['missions_per_year', 'mean_demand_kg', 'demand_cv', 'promised_days', 'horizon_days'],
                  'Tanker and launch': ['tanker_capacity_kg', 'tanker_dry_kg', 'chemical_isp_s', 'tanker_leo_dv', 'leo_rendezvous_days', 'tanker_cadence_days', 'mto_prep_days', 'rep_prep_days', 'min_order_fraction'],
                  'Thermal': ['storage_boil_daily', 'transit_cargo_boil_daily', 'transfer_efficiency'],
                  'Risk': ['launch_reliability', 'delay_probability', 'max_extra_delay_days', 'loss_cv', 'launch_cv', 'seed', 'draws'],
                  'USD costs': ['launch_per_kg', 'tanker_launch_price_factor', 'tanker_service', 'propellant_per_kg', 'depot_service_per_year', 'depot_setup_allocation', 'penalty_unserved_per_kg', 'penalty_late_per_kg']}
        frac = ('storage_boil_daily', 'transit_cargo_boil_daily', 'transfer_efficiency', 'launch_reliability', 'delay_probability', 'min_order_fraction')
        vals = {}
        for g, keys in groups.items():
            with st.expander(g, expanded=g == 'Demand and promise'):
                for k in keys:
                    v = getattr(default, k)
                    if k in ('seed', 'draws', 'horizon_days'):
                        lo, hi = {'seed': (0, 2 ** 32 - 1), 'draws': (100, 5000), 'horizon_days': (30, 1825)}[k]
                        vals[k] = st.number_input(k.replace('_', ' '), lo, hi, int(v), 1, key=k + ident)
                    else:
                        vals[k] = st.number_input(k.replace('_', ' '), 0., 1. if k in frac else 1e12, float(v), format='%.5f' if k in frac else '%.2f', key=k + ident)
    s = NetScenario(**vals)
    try: validate(s)
    except ValueError as exc: st.error(str(exc)); st.stop()
    tab_opt, tab_front, tab_sens, tab_ver, tab_exp = st.tabs(['Policy optimization', 'Service-promise frontier', 'Sensitivity', 'Verification / provenance', 'Exports / limitations'])
    with tab_opt:
        target = Target(st.slider('Required on-time fill rate (lower 95% bound)', .5, 1., float(itarget.min_fill_rate), .01))
        st.caption(f'{len(candidates())} policies: MTO (dedicated tanker per request), MTS (base-stock, lost sales), MTS_MTO (base-stock + expedited launch for shortfall). Base stock in multiples of mean request; review period in days.')
        if st.button('Search policies and run holdout', type='primary'):
            with st.spinner('Simulating...'): st.session_state['o6'] = ((s, target), c_opt(s, target))
        o = st.session_state.get('o6')
        if o and o[0] == (s, target):
            table, best, hold = o[1]
            st.dataframe(best_by_arch(table), hide_index=True, use_container_width=True)
            if best is None: st.error('No policy meets the service requirement under these assumptions. No recommendation issued.')
            else:
                st.write('Selected policy:', asdict(best))
                ok = hold['fill_lb95'] >= target.min_fill_rate
                (st.success if ok else st.warning)('Passes the independent-seed service screen.' if ok else 'Does NOT establish the required service level on the independent seed.')
                a, b, c, d = st.columns(4)
                a.metric('Holdout on-time fill', f"{100 * hold['fill_mean']:.1f}%"); b.metric('Lower 95% bound', f"{100 * hold['fill_lb95']:.1f}%")
                c.metric('Mean cost', usd(hold['mean_cost'])); d.metric('Launches / horizon', f"{hold['launches']:.1f}")
                a, b, c = st.columns(3)
                a.metric('Base stock (kg)', f"{hold['base_stock_kg']:,.0f}"); b.metric('Safety stock (kg)', f"{hold['safety_stock_kg']:,.0f}"); c.metric('Boil-off per year (kg)', f"{hold['boil_kg']:,.0f}")
                chart(px.histogram(pd.DataFrame({'On-time fill rate': hold['raw']['fill']}), x='On-time fill rate', nbins=30), 'Distribution of on-time fill rate across replications')
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button('Export search table', table.to_csv(index=False).encode(), file_name='policy_search.csv', mime='text/csv')
    with tab_front:
        st.write('Which architecture is cheapest at the required service level as the **promised delivery time** changes?')
        ft = Target(st.slider('Required fill rate for frontier', .5, 1., .9, .01, key='ft'))
        if st.button('Compute frontier'):
            with st.spinner('Computing...'): st.session_state['f6'] = ((s, ft), c_front(s, ft))
        f = st.session_state.get('f6')
        if f and f[0] == (s, ft):
            st.dataframe(f[1], hide_index=True, use_container_width=True)
            cols = [c for c in f[1].columns if c.endswith('cost USD')]
            chart(px.line(f[1].melt(id_vars='Promised days', value_vars=cols, var_name='Architecture', value_name='Cost USD'), x='Promised days', y='Cost USD', color='Architecture', markers=True), 'Cheapest feasible cost by promised service time')
    with tab_sens:
        sp = st.selectbox('Policy to stress', ['MTS_MTO', 'MTS'])
        pol = NetPolicy(sp, st.number_input('Stock multiple', 0., 20., 2., .5), st.number_input('Review days', 1., 365., 14., 1.))
        if st.button('Run sensitivity'):
            with st.spinner('Running...'): st.session_state['s6'] = ((s, pol), c_sens(s, pol))
        sv = st.session_state.get('s6')
        if sv and sv[0] == (s, pol): st.dataframe(sv[1], hide_index=True, use_container_width=True)
    with tab_ver:
        if st.button('Run verification suite'): st.session_state['v6'] = run_tests()
        v = st.session_state.get('v6')
        if v:
            (st.success if v[0] else st.error)('All tests passed.' if v[0] else 'Test failure.'); st.code(v[1])
        st.caption('Tests verify inventory conservation, limiting cases and reproducibility. They do not verify demand, physics beyond the stated approximations, or any provider claim.')
        prov = st.data_editor(pd.DataFrame(iprov), num_rows='dynamic', hide_index=True, use_container_width=True, key='prov' + ident)
    with tab_exp:
        for text in LIMITS: st.write('• ' + text)
        pol_exp = ipol
        blob = export_json(s, pol_exp, Target(), prov.fillna('').astype(str).to_dict('records'))
        st.write('Fingerprint:', fingerprint(s, pol_exp))
        st.download_button('Export scenario JSON', blob.encode(), file_name='astraia6_scenario.json', mime='application/json')


if __name__ == '__main__':
    if '--test' in sys.argv:
        ok, rep = run_tests(); print(rep); sys.exit(0 if ok else 1)
    main()
