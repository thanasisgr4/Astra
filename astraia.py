"""Astraia Assurance Lab 5.0 — hub-agnostic space-logistics policy screening.
Install: python -m pip install -r requirements.txt
Run:     python -m streamlit run astraia.py
Tests:   python astraia.py --test
Research screening tool. No flight qualification, validated trajectory or funding guarantee.
"""
from dataclasses import dataclass, asdict, replace
import math, json, hashlib, sys, io, unittest
import numpy as np
import pandas as pd

VERSION = 'Astraia Assurance Lab 5.0 — Hub-Agnostic Space Logistics'
MODEL_ID = 'astraia-hubless-screening-5.0'
G0 = 9.80665          # standard gravity, m/s^2 (exact by definition)
MAXV = 24             # maximum tanker launches considered per campaign
ROUTES = ('Direct', 'LEO_Refuel', 'Hub')


@dataclass(frozen=True)
class Scenario:
    # --- mission and burn budgets (ONE cargo mission, common final endpoint) ---
    payload_kg: float = 2500.0
    cargo_dry_kg: float = 2000.0
    chemical_isp_s: float = 450.0
    electric_isp_s: float = 2000.0
    direct_dv: float = 5900.0              # LEO -> final endpoint, single burn sequence
    cargo_upstream_dv: float = 3100.0      # LEO -> hub (Hub route only)
    cargo_downstream_dv: float = 2800.0    # hub -> final endpoint (Hub route only)
    chemical_tanker_dv: float = 3100.0     # chemical tanker LEO -> hub
    electric_tanker_dv: float = 4000.0     # effective electric tanker dv LEO -> hub
    tanker_leo_dv: float = 150.0           # tanker phasing/rendezvous dv (LEO_Refuel)
    # --- vehicle masses, capacities ---
    chemical_tanker_dry_kg: float = 1500.0
    electric_tanker_dry_kg: float = 2500.0
    tanker_capacity_kg: float = 5000.0          # delivered-fuel cargo per tanker
    depot_capacity_kg: float = 10000.0          # hub depot batch capacity
    launcher_capacity_kg: float = 15000.0       # max mass per launch to the injection orbit
    chemical_cargo_prop_capacity_kg: float = 15000.0
    chemical_tanker_prop_capacity_kg: float = 15000.0
    electric_tanker_prop_capacity_kg: float = 3000.0
    # --- durations (days) ---
    direct_days: float = 5.0
    cargo_upstream_days: float = 4.0
    cargo_downstream_days: float = 3.0
    chemical_tanker_days: float = 4.0
    electric_tanker_days: float = 90.0
    leo_rendezvous_days: float = 1.5
    transfer_days: float = 1.0                  # docking + fuel transfer before departure
    # --- electric propulsion screening ---
    electric_thrust_n: float = 5.0
    electric_power_kw: float = 120.0
    electric_efficiency: float = 0.65
    electric_duty_fraction: float = 0.8
    # --- thermal / transfer ---
    depot_boil_daily: float = 0.001
    leo_boil_daily: float = 0.002
    transit_cargo_boil_daily: float = 0.0005
    transfer_efficiency: float = 0.99
    # --- time and risk ---
    deadline_days: float = 20.0                 # from PLANNED cargo launch to arrival
    campaign_deadline_days: float = 120.0       # from first launch of any vehicle to arrival
    max_extra_delay_days: float = 5.0
    delay_probability: float = 0.15
    launch_reliability: float = 0.98            # per-launch success probability
    hub_availability: float = 1.0               # probability the hub/depot service exists and is usable
    loss_cv: float = 0.25
    launch_cv: float = 0.15
    # --- customer USD costs ---
    launch_per_kg: float = 4000.0
    tanker_launch_price_factor: float = 1.0     # <1 if tankers fly on cheaper/reusable launchers
    chemical_propellant_per_kg: float = 10.0
    electric_propellant_per_kg: float = 1000.0
    cargo_vehicle_service: float = 10e6
    chemical_tanker_service: float = 3e6
    electric_tanker_service: float = 6e6
    depot_service_per_mission: float = 2e6
    depot_capital_allocation: float = 0.0
    hub_access_fee_usd: float = 0.0
    leo_transfer_service: float = 3e6
    thermal_service_per_kg_day: float = 0.0
    delay_cost_per_day: float = 80000.0
    seed: int = 42
    draws: int = 2000


@dataclass(frozen=True)
class Policy:
    route: str = 'Hub'                  # Direct | LEO_Refuel | Hub
    tanker_mode: str = 'Chemical'       # Chemical | Electric (Electric: Hub only)
    cargo_launch_day: float = 0.0       # cargo launch relative to first tanker launch (day 0)
    launch_cadence_days: float = 2.0    # spacing between tanker launches
    reserve_fraction: float = 0.10
    initial_fill: float = 0.5           # LEO_Refuel: fraction of target fuel loaded at cargo launch


@dataclass(frozen=True)
class Requirements:
    max_failure_probability: float = 0.05
    risk_weight: float = 0.25


PRESETS = {
    'Legacy baseline (v4.1 values)': {},
    'Storable bipropellant (illustrative)': dict(chemical_isp_s=320., leo_boil_daily=0., depot_boil_daily=0., transit_cargo_boil_daily=0.),
    'LOX/methane (illustrative)': dict(chemical_isp_s=360., leo_boil_daily=.003, depot_boil_daily=.002, transit_cargo_boil_daily=.001),
    'LOX/LH2 (illustrative)': dict(chemical_isp_s=450., leo_boil_daily=.006, depot_boil_daily=.004, transit_cargo_boil_daily=.002),
}

PROVENANCE = [
    {'Group': g, 'Status': 'Unreviewed', 'Source / evidence': '', 'Reviewer': ''} for g in (
        'Destination, delta-v budgets and durations (e.g. mission reports, trajectory tool output)',
        'Vehicle dry masses, Isp and tank capacities (vendor data, Sutton & Biblarz-class references)',
        'Launcher capacity, price and per-launch reliability (published user guides, flight records)',
        'Thermal / boil-off rates and transfer efficiency (flight demonstrations, ground test reports)',
        'Hub / depot programme availability (programme status documents with dates)',
        'Cost and service definitions (quotes, contracts)',
        'Delay and price distributions (launch manifest history)')]

LIMITS = [
    'Research screening of ONE cargo mission; not a trajectory solver, fleet optimizer, annual scheduler or flight-control system.',
    'All routes share the supplied payload, cargo dry mass and final endpoint. The user must validate that delta-v and durations represent that endpoint.',
    'Routes: Direct (single launch), LEO_Refuel (hub-free: tankers rendezvous with the cargo vehicle in LEO), Hub (legacy chain: cargo refuels at an intermediate depot). Hub is a comparator, not an assumption.',
    'Hub availability is an explicit probability input. It is NOT an evidence-based forecast of any programme; cite programme documents in the provenance table.',
    'Ideal constant-Isp rocket equation. Tanker propulsion is sized as if all loaded fuel is carried through the full delta-v (conservative). No finite-burn, gravity-loss, steering-loss or trajectory analysis.',
    'Launch campaign: each vehicle is one launch with independent success probability and independent slip; a launch failure fails the service (no recovery mission is modeled). Launcher capacity is a per-launch mass limit.',
    'Fuel boil-off: tanker cargo loses fuel in transit/rendezvous, then at the storage rate while waiting for the last vehicle. In LEO_Refuel the cargo vehicle initial fuel also boils from its own launch until departure. Own propulsion fuel of tankers is assumed thermally managed.',
    'Reserve fuel is sized into tanker loading but is not added to the burn mass of the final departure (small optimistic bias for large reserves).',
    'Thermal service price is a user commercial proxy, NOT a physical cooling model or evidence of zero-boil-off capability.',
    'Electric dry mass includes power hardware. Power and burn-time checks are necessary screening conditions; low-thrust trajectory, eclipses and radiation are not solved.',
    'Chemical thrust, pressure, mixture ratio, tank geometry, docking-system compatibility and launch windows are not solved. Refueling TRL must be established by the user from public evidence.',
    'Depot capacity (Hub) is checked on arrival before storage losses.',
    'Service met = fuel sufficiency, both deadlines, capacities, electric screening, launch success and hub availability only. Arrival in failed cases is nominal.',
    'Cost is attempted-service expenditure including failed samples. No recovery purchase. Savings headlines require both alternatives to pass the independent statistical screen.',
    'Common random numbers are used across policies (paired comparison). One-sided Wilson bound covers a fixed policy under assumed distributions, not physical validation.',
    'No reusable-vehicle salvage, financing or detailed depot deployment model. Include them via input prices; check supplier boundaries to avoid double counting.',
    'The legacy 37.87% figure is preserved only as an isolated reproduction of older equations, not a target.',
    'No certified TRL, independent validation, patent novelty or ESA/NASA endorsement is claimed. Default numbers are illustrative.',
    'No authentication or secure persistence. Do not enter confidential customer data in an unsecured public deployment.']


# ----------------------------------------------------------------------------- validation
def validate(s, p=None, q=None):
    for key, value in asdict(s).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key}: finite numeric value required.')
        if value < 0:
            raise ValueError(f'{key}: cannot be negative.')
    positive = ('payload_kg', 'cargo_dry_kg', 'chemical_isp_s', 'electric_isp_s', 'chemical_tanker_dry_kg',
                'electric_tanker_dry_kg', 'tanker_capacity_kg', 'depot_capacity_kg', 'launcher_capacity_kg',
                'chemical_cargo_prop_capacity_kg', 'chemical_tanker_prop_capacity_kg', 'electric_tanker_prop_capacity_kg',
                'direct_days', 'cargo_upstream_days', 'cargo_downstream_days', 'chemical_tanker_days',
                'electric_tanker_days', 'leo_rendezvous_days', 'electric_thrust_n', 'electric_power_kw',
                'deadline_days', 'campaign_deadline_days')
    for k in positive:
        if getattr(s, k) <= 0:
            raise ValueError(f'{k}: must be positive.')
    for k in ('depot_boil_daily', 'leo_boil_daily', 'transit_cargo_boil_daily'):
        if not 0 <= getattr(s, k) < 1:
            raise ValueError(f'{k}: must be in [0,1).')
    for k in ('transfer_efficiency', 'electric_efficiency', 'electric_duty_fraction'):
        if not 0 < getattr(s, k) <= 1:
            raise ValueError(f'{k}: must be in (0,1].')
    for k in ('delay_probability', 'launch_reliability', 'hub_availability'):
        if getattr(s, k) > 1:
            raise ValueError(f'{k}: cannot exceed 1.')
    if s.tanker_launch_price_factor > 10:
        raise ValueError('tanker_launch_price_factor outside screening bounds.')
    if s.draws != int(s.draws) or not 100 <= s.draws <= 20000:
        raise ValueError('Use 100–20,000 integer samples.')
    if s.seed != int(s.seed) or not 0 <= s.seed < 2 ** 32:
        raise ValueError('Seed must be an integer in [0, 2^32).')
    if max(s.direct_days, s.cargo_upstream_days, s.cargo_downstream_days, s.chemical_tanker_days,
           s.electric_tanker_days, s.max_extra_delay_days, s.leo_rendezvous_days, s.transfer_days) > 3650:
        raise ValueError('Transfer-time inputs exceed screening bounds.')
    for dv, isp in ((s.direct_dv, s.chemical_isp_s), (s.cargo_upstream_dv, s.chemical_isp_s),
                    (s.cargo_downstream_dv, s.chemical_isp_s), (s.chemical_tanker_dv, s.chemical_isp_s),
                    (s.tanker_leo_dv, s.chemical_isp_s), (s.electric_tanker_dv, s.electric_isp_s)):
        if dv / (G0 * isp) > 20:
            raise ValueError('Extreme delta-v/Isp combination.')
    if p:
        if p.route not in ROUTES or p.tanker_mode not in ('Chemical', 'Electric'):
            raise ValueError('Invalid policy type.')
        if p.route == 'LEO_Refuel' and p.tanker_mode == 'Electric':
            raise ValueError('Electric tankers are modeled for the Hub route only.')
        for v in (p.cargo_launch_day, p.launch_cadence_days, p.reserve_fraction, p.initial_fill):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError('Policy values must be finite and nonnegative.')
        if p.cargo_launch_day > 3650 or p.launch_cadence_days > 3650 or p.reserve_fraction > 10:
            raise ValueError('Policy outside screening bounds.')
        if p.initial_fill > 1:
            raise ValueError('initial_fill must be in [0,1].')
    if q:
        if not math.isfinite(q.max_failure_probability) or not 0 <= q.max_failure_probability <= 1:
            raise ValueError('Invalid service threshold.')
        if not math.isfinite(q.risk_weight) or not 0 <= q.risk_weight <= 1:
            raise ValueError('Invalid risk weight.')


# ----------------------------------------------------------------------------- physics helpers
def wet_mass(final, dv, isp):
    if not all(math.isfinite(x) for x in (final, dv, isp)) or min(final, dv) < 0 or isp <= 0:
        raise ValueError('Invalid burn parameters.')
    e = dv / (G0 * isp)
    if e > 20:
        raise ValueError('Mass ratio exceeds screening limit.')
    return final * math.exp(e)


def retention(rate, days):
    return np.exp(np.log1p(-np.asarray(rate)) * np.asarray(days))


def legacy_reference():
    p = 100000.; v = 450 * 9.81
    d = p * (1 + (math.exp(5900 / v) - 1) * 1.1)
    h = p + p * (math.exp(3800 / v) - 1) * (2 - .995 ** 5) * 1.1
    return {'direct_t': d / 1000, 'hub_t': h / 1000, 'savings_pct': 100 * (1 - h / d)}


def samples(s):
    """Common random numbers shared by every policy (paired comparison)."""
    validate(s)
    rng = np.random.default_rng(int(s.seed)); n = int(s.draws)

    def lognormal(cv):
        sig = math.sqrt(math.log1p(cv * cv)); return rng.lognormal(-sig * sig / 2, sig, n)

    def delay(shape):
        return np.where(rng.random(shape) < s.delay_probability, rng.uniform(0, s.max_extra_delay_days, shape), 0.)

    return {'price': lognormal(s.launch_cv), 'loss': lognormal(s.loss_cv), 'cargo_delay': delay(n),
            'tanker_delay': delay((MAXV, n)), 'direct_delay': delay(n),
            'launch_u': rng.random((MAXV + 1, n)), 'hub_u': rng.random(n)}


# ----------------------------------------------------------------------------- manifest / sizing
def legs(s, p):
    """(tanker transit days, cargo days to transfer point, final-leg days, storage boil rate)."""
    if p.route == 'Hub':
        t = s.electric_tanker_days if p.tanker_mode == 'Electric' else s.chemical_tanker_days
        return t, s.cargo_upstream_days, s.cargo_downstream_days, s.depot_boil_daily
    return s.leo_rendezvous_days, s.leo_rendezvous_days, s.direct_days, s.leo_boil_daily


def nominal_timeline(s, p, k):
    t, cl, fl, sr = legs(s, p)
    arr = np.arange(k) * p.launch_cadence_days + t
    ready = p.cargo_launch_day + cl
    x = max(float(arr.max()), ready) if k else ready
    return arr, ready, x, x + s.transfer_days


def _blank():
    return dict(tankers=0, fuel_per_tanker=0., cargo_launch_fuel=0., fuel_loaded=0., chemical_procured=0.,
                electric_procured=0., required_fuel=0., target_fuel=0., tanker_launch_mass=0., transit_days=0.,
                cargo_leg_days=0., final_leg_days=0., storage_rate=0., electric_burn_days=0., required_power_kw=0.,
                cargo_tank_ok=True, tanker_tank_ok=True, power_ok=True, burn_ok=True, launcher_ok=True)


def manifest(s, p):
    validate(s, p)
    final = s.payload_kg + s.cargo_dry_kg
    base = {'Payload': s.payload_kg, 'Cargo dry mass': s.cargo_dry_kg}
    out = _blank()
    if p.route == 'Direct':
        prop = wet_mass(final, s.direct_dv, s.chemical_isp_s) - final
        base['Direct propulsion fuel'] = prop
        out.update(leo_mass=final + prop, cargo_launch_mass=final + prop, chemical_procured=prop, required_fuel=prop,
                   target_fuel=prop, cargo_launch_fuel=prop, final_leg_days=s.direct_days,
                   cargo_tank_ok=prop <= s.chemical_cargo_prop_capacity_kg,
                   launcher_ok=final + prop <= s.launcher_capacity_kg)
    else:
        electric = p.tanker_mode == 'Electric'
        t, cl, fl, sr = legs(s, p)
        isp = s.electric_isp_s if electric else s.chemical_isp_s
        if p.route == 'Hub':
            upstream = wet_mass(final, s.cargo_upstream_dv, s.chemical_isp_s) - final
            required = wet_mass(final, s.cargo_downstream_dv, s.chemical_isp_s) - final
            dv = s.electric_tanker_dv if electric else s.chemical_tanker_dv
            f0 = 0.; cargo_fuel = upstream; target = required * (1 + p.reserve_fraction)
            cargo_ok = max(upstream, required) <= s.chemical_cargo_prop_capacity_kg
            label = 'Cargo upstream fuel'
        else:
            required = wet_mass(final, s.direct_dv, s.chemical_isp_s) - final
            dv = s.tanker_leo_dv; target = required * (1 + p.reserve_fraction)
            f0 = min(p.initial_fill * target, max(s.launcher_capacity_kg - final, 0.), s.chemical_cargo_prop_capacity_kg)
            cargo_fuel = f0
            cargo_ok = target <= s.chemical_cargo_prop_capacity_kg
            label = 'Cargo fuel at launch'
        dry = s.electric_tanker_dry_kg if electric else s.chemical_tanker_dry_kg
        chosen = None
        for k in range(MAXV + 1):
            arr, ready, x, dep = nominal_timeline(s, p, k)
            r0 = float(retention(sr, dep - p.cargo_launch_day)) if p.route == 'LEO_Refuel' else 0.
            need = target - f0 * r0
            if need <= 1e-9:
                chosen = (k, 0.); break
            if k == 0:
                continue
            ri = retention(s.transit_cargo_boil_daily, t) * retention(sr, x - arr)
            qq = need / (s.transfer_efficiency * float(np.sum(ri)))
            if qq <= s.tanker_capacity_kg:
                chosen = (k, qq); break
        if chosen is None:
            raise ValueError(f'Fuel need exceeds tanker capacity within {MAXV} launches.')
        k, qd = chosen
        if k:
            twet = wet_mass(qd + dry, dv, isp); tprop = twet - qd - dry
        else:
            twet = tprop = 0.
        mdot = s.electric_thrust_n / (G0 * s.electric_isp_s)
        burn = tprop / mdot / 86400 if electric and k else 0.
        power = s.electric_thrust_n * (G0 * s.electric_isp_s) / (2 * s.electric_efficiency) / 1000 if electric and k else 0.
        tcap = s.electric_tanker_prop_capacity_kg if electric else s.chemical_tanker_prop_capacity_kg
        base.update({label: cargo_fuel, 'Tanker dry mass': k * dry, 'Tanker delivered fuel': k * qd,
                     'Tanker propulsion fuel': k * tprop})
        out.update(leo_mass=final + cargo_fuel + k * twet, cargo_launch_mass=final + cargo_fuel,
                   tanker_launch_mass=twet, tankers=k, fuel_per_tanker=qd, cargo_launch_fuel=cargo_fuel,
                   fuel_loaded=k * qd, chemical_procured=cargo_fuel + k * qd + (0. if electric else k * tprop),
                   electric_procured=k * tprop if electric else 0., required_fuel=required, target_fuel=target,
                   transit_days=t, cargo_leg_days=cl, final_leg_days=fl, storage_rate=sr,
                   electric_burn_days=burn, required_power_kw=power, cargo_tank_ok=cargo_ok,
                   tanker_tank_ok=(tprop <= tcap) if k else True, power_ok=power <= s.electric_power_kw,
                   burn_ok=(burn <= t * s.electric_duty_fraction) if electric else True,
                   launcher_ok=(final + cargo_fuel <= s.launcher_capacity_kg) and (twet <= s.launcher_capacity_kg))
        out['downstream_fuel'] = required
    out['components'] = base
    out['mass_balance_error'] = abs(sum(base.values()) - out['leo_mass'])
    if not math.isclose(sum(base.values()), out['leo_mass'], rel_tol=1e-10, abs_tol=1e-5):
        raise ValueError('LEO manifest conservation failure.')
    out['launches'] = 1 + out['tankers']
    return out


def failure_upper(failures, n):
    z = 1.6448536269514722; p = failures / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


# ----------------------------------------------------------------------------- Monte Carlo evaluation
def evaluate(s, p, rd=None):
    b = manifest(s, p); rd = samples(s) if rd is None else rd; n = int(s.draws)
    zero = np.zeros(n); k = b['tankers']; L = p.cargo_launch_day
    price = rd['price']
    launch_fail = (rd['launch_u'][:k + 1] < 1 - s.launch_reliability).any(axis=0)
    hub_fail = (rd['hub_u'] >= s.hub_availability) if p.route == 'Hub' else np.zeros(n, dtype=bool)
    if p.route == 'Direct':
        cargo_launch = L + rd['direct_delay']
        arrival = cargo_launch + s.direct_days
        storage = transit = zero; transit_loss = storage_loss = transfer_loss = residual = zero
        missing = zero; fuel_balance = 0.; thermal = zero; cap_ok = np.ones(n, dtype=bool)
        first = 0.
        fixed = s.cargo_vehicle_service
        launch = b['cargo_launch_mass'] * s.launch_per_kg * price
    else:
        t = b['transit_days']; sr = b['storage_rate']; f0 = b['cargo_launch_fuel'] if p.route == 'LEO_Refuel' else 0.
        cargo_launch = L + rd['cargo_delay']
        ready = cargo_launch + b['cargo_leg_days']
        tl = np.arange(k)[:, None] * p.launch_cadence_days + rd['tanker_delay'][:k]
        arr = tl + t
        x = np.maximum(arr.max(axis=0), ready) if k else ready
        depart = x + s.transfer_days
        wait = x - arr
        tr = np.clip(s.transit_cargo_boil_daily * rd['loss'], 0, .95)
        sg = np.clip(sr * rd['loss'], 0, .95)
        q = b['fuel_per_tanker']
        on_arr = q * retention(tr, t)
        avail_t = (on_arr * retention(sg, wait)) if k else np.zeros((1, n))
        avail_sum = avail_t.sum(axis=0) if k else zero
        transit_loss = k * (q - on_arr) if k else zero
        storage_loss_t = (k * on_arr - avail_sum) if k else zero
        cargo_avail = f0 * retention(sg, depart - cargo_launch) if f0 > 0 else zero
        cargo_storage_loss = f0 - cargo_avail
        required = b['required_fuel']
        need_t = np.maximum(required - cargo_avail, 0.)
        withdrawn = np.minimum(avail_sum, need_t / s.transfer_efficiency)
        transferred = withdrawn * s.transfer_efficiency
        transfer_loss = withdrawn - transferred
        used_cargo = np.minimum(cargo_avail, required)
        residual = (avail_sum - withdrawn) + (cargo_avail - used_cargo)
        used = used_cargo + transferred
        missing = np.maximum(required - used, 0.)
        storage_loss = storage_loss_t + cargo_storage_loss
        balance = (k * q + f0) - transit_loss - storage_loss - transfer_loss - residual - used
        fuel_balance = float(np.max(np.abs(balance)))
        if fuel_balance > 1e-5:
            raise ValueError('Fuel flow conservation failure.')
        arrival = depart + b['final_leg_days']
        transit = np.full(n, float(t)) if k else zero
        storage = wait.mean(axis=0) if k else zero
        cap_ok = (k * on_arr <= s.depot_capacity_kg + 1e-7) if p.route == 'Hub' else np.ones(n, dtype=bool)
        wait_sum = wait.sum(axis=0) if k else zero
        thermal = s.thermal_service_per_kg_day * (q * (k * t + wait_sum) + f0 * (depart - cargo_launch))
        tf = s.tanker_launch_price_factor
        launch = (b['cargo_launch_mass'] + k * b['tanker_launch_mass'] * tf) * s.launch_per_kg * price
        if p.route == 'Hub':
            unit = s.electric_tanker_service if p.tanker_mode == 'Electric' else s.chemical_tanker_service
            fixed = (s.cargo_vehicle_service + k * unit + s.depot_service_per_mission + s.depot_capital_allocation
                     + s.hub_access_fee_usd)
        else:
            fixed = s.cargo_vehicle_service + k * s.chemical_tanker_service + s.leo_transfer_service
    payload_time = arrival - L
    late = np.maximum(payload_time - s.deadline_days, 0.)
    campaign_late = arrival > s.campaign_deadline_days + 1e-9
    reasons = {'Fuel shortage': missing > 1e-7, 'Deadline missed (payload clock)': late > 1e-9,
               'Campaign time exceeded': campaign_late, 'Depot capacity': ~cap_ok,
               'Launcher capacity': np.full(n, not b['launcher_ok']),
               'Cargo propulsion tank': np.full(n, not b['cargo_tank_ok']),
               'Tanker propulsion tank': np.full(n, not b['tanker_tank_ok']),
               'Electric power': np.full(n, not b['power_ok']), 'Electric burn duration': np.full(n, not b['burn_ok']),
               'Launch failure': launch_fail, 'Hub unavailable': hub_fail}
    fail = np.logical_or.reduce(list(reasons.values())); success = ~fail
    procurement = b['chemical_procured'] * s.chemical_propellant_per_kg + b['electric_procured'] * s.electric_propellant_per_kg
    delay = late * s.delay_cost_per_day
    cost = launch + fixed + procurement + thermal + delay
    p95 = float(np.quantile(cost, .95)); cvar = float(cost[cost >= p95].mean())
    data = pd.DataFrame({'Cost USD': cost, 'Service met': success, 'Nominal arrival day': arrival,
                         'Payload clock days': payload_time, 'Missing fuel kg': missing,
                         'Transit fuel loss kg': transit_loss, 'Storage fuel loss kg': storage_loss,
                         'Transfer fuel loss kg': transfer_loss, 'Fuel residual kg': residual,
                         'Launch USD': launch, 'Delay cost USD': delay, 'Thermal service USD': thermal})
    for name, mask in reasons.items():
        data['Failure: ' + name] = mask
    return {'policy': asdict(p), 'manifest': b, 'mean_cost': float(cost.mean()), 'median_cost': float(np.median(cost)),
            'p95_cost': p95, 'cvar95': cvar, 'service_probability': float(success.mean()),
            'failure_upper95': failure_upper(int(fail.sum()), n), 'p95_arrival': float(np.quantile(arrival, .95)),
            'fuel_balance_error': fuel_balance, 'failure_reasons': {m: float(v.mean()) for m, v in reasons.items()},
            'fixed_cost': float(fixed), 'commodity_cost': procurement, 'data': data}


# ----------------------------------------------------------------------------- policy search
def candidates(full=True):
    out = [Policy('Direct', 'Chemical', 0., 0., 0., 0.)]
    cad = (3., 7., 14.) if full else (3., 14.)
    out += [Policy('LEO_Refuel', 'Chemical', L, c, r, f) for c in cad for L in (0., 7., 14., 28.)
            for f in (.25, .5, .75) for r in (.05, .15)]
    out += [Policy('Hub', 'Chemical', L, c, r, 0.) for c in (2., 7.) for L in (0., 5., 15.) for r in (.05, .15, .35)]
    out += [Policy('Hub', 'Electric', L, 7., r, 0.) for L in (60., 100.) for r in (.1, .25)]
    return out


def is_pareto(table):
    answer = np.zeros(len(table), dtype=bool)
    ids = np.flatnonzero(table['Training feasible'].to_numpy())
    vals = table[['Mean cost USD', 'Failure probability', 'LEO mass kg']].to_numpy()
    for i in ids:
        c = vals[ids]
        answer[i] = not np.any(np.all(c <= vals[i], axis=1) & np.any(c < vals[i], axis=1))
    return answer


def search(s, q, rd=None, full=True):
    rd = samples(s) if rd is None else rd; rows = []
    for p in candidates(full):
        try:
            r = evaluate(s, p, rd); failure = 1 - r['service_probability']
            issues = '; '.join(k for k, v in r['failure_reasons'].items() if v > q.max_failure_probability)
            rows.append({**asdict(p), 'Mean cost USD': r['mean_cost'], 'P95 cost USD': r['p95_cost'],
                         'CVaR95 USD': r['cvar95'], 'Failure probability': failure,
                         'LEO mass kg': r['manifest']['leo_mass'], 'Launches': r['manifest']['launches'],
                         'Score USD': r['mean_cost'] + q.risk_weight * (r['cvar95'] - r['mean_cost']),
                         'Training feasible': failure <= q.max_failure_probability + 1e-12,
                         'Rejection / warning': issues})
        except (ValueError, OverflowError) as exc:
            rows.append({**asdict(p), 'Mean cost USD': np.nan, 'P95 cost USD': np.nan, 'CVaR95 USD': np.nan,
                         'Failure probability': 1., 'LEO mass kg': np.nan, 'Launches': np.nan, 'Score USD': np.nan,
                         'Training feasible': False, 'Rejection / warning': str(exc)})
    table = pd.DataFrame(rows).sort_values(['Training feasible', 'Score USD'], ascending=[False, True],
                                           na_position='last').reset_index(drop=True)
    table['Pareto among training-feasible'] = is_pareto(table)
    return table


def optimize(s, q):
    validate(s, q=q)
    table = search(s, q)
    feasible = table[table['Training feasible']]
    if feasible.empty:
        return table, None, None
    row = feasible.iloc[0]
    p = Policy(row['route'], row['tanker_mode'], float(row['cargo_launch_day']), float(row['launch_cadence_days']),
               float(row['reserve_fraction']), float(row['initial_fill']))
    return table, p, evaluate(replace(s, seed=(int(s.seed) + 104729) % 2 ** 32), p)


def best_by_route(table):
    rows = []
    for route in ROUTES:
        sub = table[(table['route'] == route) & table['Training feasible']]
        if sub.empty:
            rows.append({'Route': route, 'Best feasible score USD': np.nan, 'Mean cost USD': np.nan,
                         'Failure probability': np.nan, 'Launches': np.nan, 'Status': 'No feasible policy'})
        else:
            r = sub.iloc[0]
            rows.append({'Route': route, 'Best feasible score USD': r['Score USD'], 'Mean cost USD': r['Mean cost USD'],
                         'Failure probability': r['Failure probability'], 'Launches': r['Launches'], 'Status': 'Feasible'})
    return pd.DataFrame(rows)


def decision_map(s, q, reliabilities=(.90, .95, .98, .995), availabilities=(1., .9, .7, .5)):
    """Which architecture wins as launch reliability and hub availability change (re-optimized per cell)."""
    rows = []
    for rel in reliabilities:
        for av in availabilities:
            ss = replace(s, launch_reliability=rel, hub_availability=av, draws=min(int(s.draws), 500))
            summary = best_by_route(search(ss, q, full=False))
            ok = summary.dropna(subset=['Best feasible score USD'])
            rec = {'Launch reliability': rel, 'Hub availability': av,
                   'Winner': ok.sort_values('Best feasible score USD').iloc[0]['Route'] if len(ok) else 'None feasible'}
            for _, r in summary.iterrows():
                rec[r['Route'] + ' score USD'] = r['Best feasible score USD']
            rows.append(rec)
    return pd.DataFrame(rows)


def sensitivity(s, p, q):
    cases = [('Baseline', s)]
    scaled = [('Launch price', 'launch_per_kg', (.75, 1.25)), ('Transit loss rate', 'transit_cargo_boil_daily', (.5, 2.)),
              ('Depot loss rate', 'depot_boil_daily', (.5, 2.)), ('LEO storage loss rate', 'leo_boil_daily', (.5, 2.)),
              ('Cargo dry mass', 'cargo_dry_kg', (.8, 1.2)), ('Electric transit duration', 'electric_tanker_days', (.8, 1.2))]
    for title, key, factors in scaled:
        for f in factors:
            v = getattr(s, key) * f
            if key.endswith('boil_daily'):
                v = min(v, .95)
            cases.append((f'{title} × {f:g}', replace(s, **{key: v})))
    for title, key, factors in (('Launch failure rate', 'launch_reliability', (.5, 2.)),
                                ('Hub unavailability', 'hub_availability', (.5, 2.))):
        for f in factors:
            cases.append((f'{title} × {f:g}', replace(s, **{key: max(0., 1 - (1 - getattr(s, key)) * f)})))
    rows = []
    for label, ss in cases:
        rd = samples(ss); a = evaluate(ss, Policy('Direct', 'Chemical', 0., 0., 0., 0.), rd); b = evaluate(ss, p, rd)
        rows.append({'Case': label, 'Direct mean USD': a['mean_cost'], 'Selected mean USD': b['mean_cost'],
                     'Difference USD': a['mean_cost'] - b['mean_cost'],
                     'Direct failure upper95': a['failure_upper95'], 'Selected failure upper95': b['failure_upper95'],
                     'Both pass confidence screen': a['failure_upper95'] <= q.max_failure_probability
                     and b['failure_upper95'] <= q.max_failure_probability})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- persistence
def fingerprint(s, p):
    return hashlib.sha256(json.dumps({'model': MODEL_ID, 'scenario': asdict(s), 'policy': asdict(p)},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def export_json(s, p, destination, provenance, business=None):
    return json.dumps({'model_id': MODEL_ID, 'version': VERSION, 'fingerprint': fingerprint(s, p),
                       'scenario': asdict(s), 'policy': asdict(p), 'destination': destination,
                       'provenance': provenance, 'business_eur': business or {}, 'limitations': LIMITS},
                      indent=2, ensure_ascii=False, allow_nan=False)


def import_json(text):
    data = json.loads(text)
    if not isinstance(data, dict) or data.get('model_id') != MODEL_ID:
        raise ValueError('Incompatible scenario/model ID.')
    if set(data.get('scenario', {})) != set(asdict(Scenario())):
        raise ValueError('Scenario fields differ from this version.')
    if set(data.get('policy', {})) != set(asdict(Policy())):
        raise ValueError('Policy fields differ from this version.')
    s = Scenario(**data['scenario']); p = Policy(**data['policy']); validate(s, p)
    if not isinstance(data.get('destination', ''), str):
        raise ValueError('Destination must be text.')
    prov = data.get('provenance', PROVENANCE)
    if not isinstance(prov, list) or any(not isinstance(x, dict) or set(x) != set(PROVENANCE[0]) for x in prov):
        raise ValueError('Invalid provenance table.')
    if any(not isinstance(v, str) for row in prov for v in row.values()):
        raise ValueError('Provenance values must be text.')
    return s, p, data


def business_model(clients, price, pilot_revenue, direct_cost, opex, cash):
    if any(not math.isfinite(float(x)) or x < 0 for x in (clients, price, pilot_revenue, direct_cost, opex, cash)):
        raise ValueError('Invalid business input.')
    arr = clients * price; revenue = arr + pilot_revenue; net = revenue - direct_cost - opex
    return {'arr': arr, 'revenue': revenue, 'net': net,
            'gross_margin': (revenue - direct_cost) / revenue if revenue else None,
            'runway_months': cash / (-net / 12) if net < 0 else None}


# ----------------------------------------------------------------------------- tests
class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.s = Scenario(draws=150)
        self.leo = Policy('LEO_Refuel', 'Chemical', 7., 3., .15, .5)
        self.hub = Policy('Hub', 'Chemical', 0., 2., .1, 0.)
        self.eh = Policy('Hub', 'Electric', 100., 7., .1, 0.)
        self.direct = Policy('Direct', 'Chemical', 0., 0., 0., 0.)

    def test_legacy(self): self.assertAlmostEqual(legacy_reference()['savings_pct'], 37.872210, places=5)
    def test_zero_burn(self): self.assertEqual(wet_mass(123, 0, 450), 123)
    def test_known_burn(self): self.assertAlmostEqual(wet_mass(100, G0 * 450 * math.log(2), 450), 200)

    def test_invalid_burn(self):
        with self.assertRaises(ValueError): wet_mass(1, 1, 0)

    def test_mass_all_routes(self):
        for p in (self.direct, self.leo, self.hub, self.eh):
            self.assertLess(manifest(self.s, p)['mass_balance_error'], 1e-6)

    def test_integer_tankers(self):
        self.assertIsInstance(manifest(self.s, self.leo)['tankers'], int)
        self.assertGreaterEqual(manifest(self.s, self.leo)['tankers'], 1)

    def test_no_downstream_no_tanker(self):
        self.assertEqual(manifest(replace(self.s, cargo_downstream_dv=0), self.hub)['tankers'], 0)

    def test_leo_full_fill_zero_loss_no_tanker(self):
        s = replace(self.s, leo_boil_daily=0., launcher_capacity_kg=1e6)
        self.assertEqual(manifest(s, replace(self.leo, reserve_fraction=0., initial_fill=1.))['tankers'], 0)

    def test_reserve_adds_mass(self):
        for r in ('LEO_Refuel', 'Hub'):
            lo = replace(self.leo if r == 'LEO_Refuel' else self.hub, reserve_fraction=0.)
            hi = replace(lo, reserve_fraction=.5)
            self.assertGreater(manifest(self.s, hi)['leo_mass'], manifest(self.s, lo)['leo_mass'])

    def test_losses_raise_loaded_fuel(self):
        self.assertGreater(manifest(self.s, self.eh)['fuel_loaded'],
                           manifest(replace(self.s, transit_cargo_boil_daily=0), self.eh)['fuel_loaded'])

    def test_longer_wait_raises_leo_fuel(self):
        a = manifest(self.s, replace(self.leo, cargo_launch_day=0.))['fuel_loaded']
        b = manifest(self.s, replace(self.leo, cargo_launch_day=28.))['fuel_loaded']
        self.assertNotEqual(a, b)

    def test_zero_losses(self):
        for p in (self.leo, self.hub):
            r = evaluate(replace(self.s, depot_boil_daily=0, leo_boil_daily=0, transit_cargo_boil_daily=0), p)
            self.assertEqual(r['data']['Transit fuel loss kg'].sum() + r['data']['Storage fuel loss kg'].sum(), 0)

    def test_flow_conservation(self):
        for p in (self.leo, self.hub, self.eh): self.assertLess(evaluate(self.s, p)['fuel_balance_error'], 1e-5)

    def test_no_nominal_shortage(self):
        s = replace(self.s, loss_cv=0, delay_probability=0, launch_reliability=1.)
        for p in (self.leo, self.hub):
            self.assertEqual(evaluate(s, p)['data']['Missing fuel kg'].sum(), 0)

    def test_seed(self): np.testing.assert_array_equal(samples(self.s)['price'], samples(self.s)['price'])
    def test_zero_delays(self): self.assertEqual(samples(replace(self.s, delay_probability=0))['tanker_delay'].sum(), 0)

    def test_launcher_capacity_blocks_direct(self):
        self.assertEqual(evaluate(replace(self.s, launcher_capacity_kg=10000.), self.direct)['service_probability'], 0)

    def test_bad_depot(self):
        self.assertEqual(evaluate(replace(self.s, depot_capacity_kg=1), self.hub)['service_probability'], 0)

    def test_bad_cargo_tank(self):
        self.assertEqual(evaluate(replace(self.s, chemical_cargo_prop_capacity_kg=1), self.direct)['service_probability'], 0)

    def test_bad_tanker_tank(self):
        self.assertEqual(evaluate(replace(self.s, chemical_tanker_prop_capacity_kg=1), self.hub)['service_probability'], 0)

    def test_power_fail(self):
        self.assertEqual(evaluate(replace(self.s, electric_power_kw=.001), self.eh)['service_probability'], 0)

    def test_burn_time_fail(self):
        self.assertEqual(evaluate(replace(self.s, electric_thrust_n=.00001), self.eh)['service_probability'], 0)

    def test_deadline(self):
        self.assertEqual(evaluate(replace(self.s, deadline_days=1e-6), self.direct)['service_probability'], 0)

    def test_campaign_deadline(self):
        self.assertEqual(evaluate(replace(self.s, campaign_deadline_days=1e-6), self.leo)['service_probability'], 0)

    def test_hub_unavailable_only_hurts_hub(self):
        s = replace(self.s, hub_availability=0.)
        self.assertEqual(evaluate(s, self.hub)['service_probability'], 0)
        self.assertEqual(evaluate(s, self.leo)['failure_reasons']['Hub unavailable'], 0)

    def test_hub_available_flag_one(self):
        self.assertEqual(evaluate(self.s, self.hub)['failure_reasons']['Hub unavailable'], 0)

    def test_launch_reliability_zero(self):
        for p in (self.direct, self.leo, self.hub):
            self.assertEqual(evaluate(replace(self.s, launch_reliability=0.), p)['service_probability'], 0)

    def test_more_launches_more_launch_risk(self):
        s = replace(self.s, draws=2000, launch_reliability=.9)
        a = evaluate(s, self.direct)['failure_reasons']['Launch failure']
        b = evaluate(s, self.leo)['failure_reasons']['Launch failure']
        self.assertGreater(b, a)

    def test_price_deterministic(self):
        s = replace(self.s, launch_cv=0, delay_probability=0)
        self.assertEqual(evaluate(s, self.direct)['data']['Cost USD'].nunique(), 1)

    def test_confidence_nonzero(self): self.assertGreater(failure_upper(0, 150), 0)
    def test_all_failure_bound(self): self.assertAlmostEqual(failure_upper(150, 150), 1)

    def test_invalid_input(self):
        with self.assertRaises(ValueError): validate(replace(self.s, transfer_efficiency=0))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError): validate(replace(self.s, payload_kg=float('nan')))

    def test_electric_leo_rejected(self):
        with self.assertRaises(ValueError): validate(self.s, Policy('LEO_Refuel', 'Electric'))

    def test_hash(self): self.assertEqual(fingerprint(self.s, self.hub), fingerprint(self.s, self.hub))

    def test_roundtrip(self):
        s, p, d = import_json(export_json(self.s, self.leo, 'Test orbit', PROVENANCE))
        self.assertEqual(s, self.s); self.assertEqual(p, self.leo)

    def test_bad_model_import(self):
        with self.assertRaises(ValueError): import_json('{"model_id":"wrong"}')

    def test_search_count(self): self.assertEqual(len(candidates()), 1 + 72 + 18 + 4)

    def test_optimizer(self):
        table, p, r = optimize(self.s, Requirements(.5, .25))
        self.assertEqual(len(table), len(candidates())); self.assertIsNotNone(p)
        self.assertTrue(table.iloc[0]['Training feasible'])

    def test_no_feasible(self): self.assertIsNone(optimize(replace(self.s, deadline_days=1e-6), Requirements())[1])
    def test_best_by_route(self): self.assertEqual(len(best_by_route(search(self.s, Requirements(.5, .25)))), 3)

    def test_decision_map(self):
        m = decision_map(replace(self.s, draws=100), Requirements(.5, .25), (.95,), (1., .5))
        self.assertEqual(len(m), 2)

    def test_sensitivity(self): self.assertEqual(len(sensitivity(self.s, self.leo, Requirements())), 17)

    def test_business(self):
        b = business_model(3, 12000, 0, 10000, 58000, 10000); self.assertEqual(b['arr'], 36000); self.assertEqual(b['net'], -32000)

    def test_zero_revenue_margin(self): self.assertIsNone(business_model(0, 0, 0, 0, 0, 0)['gross_margin'])

    def test_randomized_invariants(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            s = replace(self.s, payload_kg=float(rng.uniform(100, 6000)), cargo_dry_kg=float(rng.uniform(100, 3000)))
            for p in (Policy('LEO_Refuel', 'Chemical', float(rng.uniform(0, 30)), float(rng.uniform(1, 14)), float(rng.uniform(0, .5)), float(rng.uniform(0, 1))),
                      Policy('Hub', 'Electric' if rng.random() < .5 else 'Chemical', float(rng.uniform(0, 100)), float(rng.uniform(1, 10)), float(rng.uniform(0, .75)))):
                try: r = evaluate(s, p)
                except ValueError as exc:
                    self.assertIn('exceeds', str(exc)); continue
                self.assertLess(r['fuel_balance_error'], 1e-5); self.assertLess(r['manifest']['mass_balance_error'], 1e-5)


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(VerificationTests))
    return result.wasSuccessful(), stream.getvalue()


# ----------------------------------------------------------------------------- Streamlit UI
def main():
    import streamlit as st
    import plotly.express as px
    st.set_page_config(page_title='Astraia Assurance Lab 5.0', layout='wide')
    st.markdown('<style>.stApp{background:#091321;color:#e9eff8}section[data-testid="stSidebar"]{background:#12223a}'
                'div[data-testid="stMetric"]{padding:14px;border:1px solid #365272;border-top:3px solid #498dff;border-radius:8px}</style>',
                unsafe_allow_html=True)
    st.title('ASTRAIA · ASSURANCE LAB 5.0')
    st.caption('Hub-agnostic space-logistics screening | Direct vs in-orbit refueling vs hub | launch-campaign risk | boil-off timeline | '
               'capacity, deadline and programme-availability constraints')
    st.warning('Research prototype — not a validated trajectory solver, operational system or funding guarantee. '
               'Ερευνητικό πρωτότυπο: όλες οι προεπιλογές χρειάζονται ανεξάρτητη τεκμηρίωση.')

    def usd(x): return f'${x / 1e6:,.2f}M'

    def chart(fig, title):
        fig.update_layout(title=title, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    @st.cache_data(show_spinner=False)
    def c_eval(s, p): return evaluate(s, p)
    @st.cache_data(show_spinner=False)
    def c_opt(s, q): return optimize(s, q)
    @st.cache_data(show_spinner=False)
    def c_sens(s, p, q): return sensitivity(s, p, q)
    @st.cache_data(show_spinner=False)
    def c_map(s, q): return decision_map(s, q)
    @st.cache_data(show_spinner=False)
    def c_tests(): return run_tests()

    default = Scenario(); init_policy = Policy(); imported = {}
    with st.sidebar:
        st.header('Scenario controls')
        uploaded = st.file_uploader('Import v5.0 scenario JSON', type=['json'])
        if uploaded:
            try: default, init_policy, imported = import_json(uploaded.getvalue().decode('utf-8'))
            except Exception as exc: st.error(f'Import rejected: {exc}'); st.stop()
        preset = st.selectbox('Propellant preset (illustrative — document sources)', list(PRESETS))
        default = replace(default, **PRESETS[preset])
        ident = fingerprint(default, init_policy)
        destination = st.text_input('Common final destination', value=imported.get('destination', 'Specified destination — unvalidated'), key='dest' + ident)
        groups = {
            'Cargo and burn budgets': ['payload_kg', 'cargo_dry_kg', 'chemical_isp_s', 'direct_dv', 'cargo_upstream_dv', 'cargo_downstream_dv', 'tanker_leo_dv'],
            'Vehicles, launcher and capacities': ['launcher_capacity_kg', 'chemical_tanker_dv', 'electric_tanker_dv', 'chemical_tanker_dry_kg', 'electric_tanker_dry_kg', 'electric_isp_s', 'tanker_capacity_kg', 'depot_capacity_kg', 'chemical_cargo_prop_capacity_kg', 'chemical_tanker_prop_capacity_kg', 'electric_tanker_prop_capacity_kg'],
            'Electric propulsion screening': ['electric_thrust_n', 'electric_power_kw', 'electric_efficiency', 'electric_duty_fraction'],
            'Timeline, thermal and deadlines': ['direct_days', 'cargo_upstream_days', 'cargo_downstream_days', 'chemical_tanker_days', 'electric_tanker_days', 'leo_rendezvous_days', 'transfer_days', 'transit_cargo_boil_daily', 'leo_boil_daily', 'depot_boil_daily', 'transfer_efficiency', 'deadline_days', 'campaign_deadline_days'],
            'Customer USD costs': ['launch_per_kg', 'tanker_launch_price_factor', 'chemical_propellant_per_kg', 'electric_propellant_per_kg', 'cargo_vehicle_service', 'chemical_tanker_service', 'electric_tanker_service', 'depot_service_per_mission', 'depot_capital_allocation', 'hub_access_fee_usd', 'leo_transfer_service', 'thermal_service_per_kg_day', 'delay_cost_per_day'],
            'Risk and uncertainty': ['launch_reliability', 'hub_availability', 'max_extra_delay_days', 'delay_probability', 'loss_cv', 'launch_cv', 'seed', 'draws']}
        frac = ('transit_cargo_boil_daily', 'leo_boil_daily', 'depot_boil_daily', 'transfer_efficiency', 'electric_efficiency',
                'electric_duty_fraction', 'delay_probability', 'launch_reliability', 'hub_availability')
        values = {}
        for group, keys in groups.items():
            with st.expander(group, expanded=group == 'Cargo and burn budgets'):
                for key in keys:
                    value = getattr(default, key); label = key.replace('_', ' ')
                    if key in ('seed', 'draws'):
                        values[key] = st.number_input(label, min_value=100 if key == 'draws' else 0, max_value=20000 if key == 'draws' else 2 ** 32 - 1,
                                                      value=int(value), step=1, key=key + ident)
                    else:
                        hi = .99 if 'boil' in key else (1. if key in frac else 1e12)
                        values[key] = st.number_input(label, min_value=0., max_value=hi, value=float(value),
                                                      format='%.5f' if key in frac else '%.2f', key=key + ident)
    s = Scenario(**values)
    try: validate(s)
    except ValueError as exc: st.error(str(exc)); st.stop()
    t_cmp, t_opt, t_rob, t_ver, t_biz, t_pilot, t_exp = st.tabs(['Architecture comparison', 'Policy optimization', 'Robustness / decision map', 'Verification / provenance', 'SaaS business', 'Pilot evidence', 'Exports / limitations'])

    with t_cmp:
        st.subheader('Reference mission — common declared endpoint')
        c = st.columns(3)
        route = c[0].selectbox('Architecture', ['LEO_Refuel', 'Hub'], index=1 if init_policy.route == 'Hub' else 0, key='route' + ident)
        mode = c[1].selectbox('Tanker mode', ['Chemical', 'Electric'] if route == 'Hub' else ['Chemical'], key='mode' + ident)
        reserve = c[2].number_input('Fuel reserve fraction', 0., 10., float(init_policy.reserve_fraction), .05, key='reserve' + ident)
        c = st.columns(3)
        lday = c[0].number_input('Cargo launch day (first tanker = day 0)', 0., 3650., float(init_policy.cargo_launch_day), 1., key='lday' + ident)
        cad = c[1].number_input('Tanker launch cadence (days)', 0., 3650., float(init_policy.launch_cadence_days), 1., key='cad' + ident)
        fill = c[2].number_input('Cargo initial fuel fraction (LEO_Refuel)', 0., 1., float(init_policy.initial_fill), .05, key='fill' + ident, disabled=route != 'LEO_Refuel')
        p = Policy(route, mode, lday, cad, reserve, fill if route == 'LEO_Refuel' else 0.)
        dp = Policy('Direct', 'Chemical', 0., 0., 0., 0.)
        try: direct = c_eval(s, dp); alt = c_eval(s, p)
        except (ValueError, OverflowError) as exc: st.error(str(exc)); st.stop()
        m = st.columns(5)
        m[0].metric('Direct mean cost', usd(direct['mean_cost'])); m[1].metric(f'{route} mean cost', usd(alt['mean_cost']))
        m[2].metric('Direct service prob.', f"{100 * direct['service_probability']:.1f}%")
        m[3].metric(f'{route} service prob.', f"{100 * alt['service_probability']:.1f}%")
        m[4].metric('Launches (alt / direct)', f"{alt['manifest']['launches']} / 1")
        st.caption('Attempted-service cost is spent even when service fails. A cost advantage is meaningful only where service probability is adequate.')
        rows = [{'Architecture': lab, 'Component': k, 'Mass kg': v} for lab, r in [('Direct', direct), (route, alt)] for k, v in r['manifest']['components'].items()]
        chart(px.bar(pd.DataFrame(rows), x='Architecture', y='Mass kg', color='Component'), 'Total launched mass — each kg counted once')
        st.subheader('Why a policy fails')
        st.dataframe(pd.DataFrame([{'Constraint': k, 'Direct failure fraction': direct['failure_reasons'][k], f'{route} failure fraction': alt['failure_reasons'][k]} for k in direct['failure_reasons']]), hide_index=True, use_container_width=True)
        st.caption('Failure fractions overlap; do not sum them.')
        fate = pd.DataFrame({'Flow': ['Transit loss', 'Storage loss', 'Transfer loss', 'Residual'], 'Mean kg': [alt['data'][c_].mean() for c_ in ('Transit fuel loss kg', 'Storage fuel loss kg', 'Transfer fuel loss kg', 'Fuel residual kg')]})
        chart(px.bar(fate, x='Flow', y='Mean kg', color='Flow'), 'Disposition of delivered fuel (excluding burn fuel)')
        st.write('Maximum fuel-flow balance error (kg):', f"{alt['fuel_balance_error']:.9f}")
        if mode == 'Electric':
            mm = alt['manifest']; a, b_, c_ = st.columns(3)
            a.metric('Power / tanker', f"{mm['required_power_kw']:.2f} kW"); b_.metric('Ideal burn time', f"{mm['electric_burn_days']:.2f} d")
            c_.metric('Power / burn screen', 'PASS' if mm['power_ok'] and mm['burn_ok'] else 'FAIL')
        times = pd.DataFrame({'Direct': direct['data']['Payload clock days'], route: alt['data']['Payload clock days']}).melt(var_name='Route', value_name='Payload clock (days)')
        chart(px.histogram(times, x='Payload clock (days)', color='Route', barmode='overlay', opacity=.6), 'Planned-launch-to-arrival distributions (nominal; failed samples included)')
        with st.expander('Legacy 37.87% reproduction — not a target or validated saving'):
            st.json(legacy_reference())

    with t_opt:
        st.subheader('Service-constrained search across all architectures, independent-seed holdout')
        a, b = st.columns(2)
        limit = a.slider('Maximum modeled failure probability (%)', 0., 30., 5., .5) / 100
        weight = b.slider('CVaR tail-risk weight', 0., 1., .25, .05)
        q = Requirements(limit, weight); signature = (s, q)
        st.caption(f'{len(candidates())} policies: Direct, LEO_Refuel (cadence × cargo launch day × initial fill × reserve), Hub chemical and Hub electric. Finite search; not a globally optimal solution.')
        if st.button('Search policies and run holdout', type='primary', key='opt'):
            with st.spinner('Searching...'): res = c_opt(s, q)
            st.session_state['opt50'] = (signature,) + tuple(res)
        stored = st.session_state.get('opt50')
        if stored and stored[0] == signature:
            table, best, holdout = stored[1:]
            st.subheader('Best feasible policy per architecture (training seed)')
            st.dataframe(best_by_route(table), hide_index=True, use_container_width=True)
            if best is None: st.error('No candidate satisfies the training service requirement. No recommendation issued.')
            else:
                st.write('Selected policy:', asdict(best))
                passed = holdout['failure_upper95'] <= limit
                (st.success if passed else st.warning)('Selected policy passes the independent-seed confidence screen.' if passed else 'Selected policy does not establish the requested service level on the independent-seed screen.')
                a, b = st.columns(2)
                a.metric('Holdout service probability', f"{100 * holdout['service_probability']:.2f}%")
                b.metric('One-sided 95% failure upper bound', f"{100 * holdout['failure_upper95']:.2f}%")
                hs = replace(s, seed=(int(s.seed) + 104729) % 2 ** 32); hd = evaluate(hs, Policy('Direct', 'Chemical', 0., 0., 0., 0.), samples(hs))
                if passed and hd['failure_upper95'] <= limit:
                    paired = hd['data']['Cost USD'].to_numpy() - holdout['data']['Cost USD'].to_numpy()
                    st.metric('Modeled advantage vs qualified direct baseline', usd(float(paired.mean())))
                else: st.info('Equivalent-service savings headline withheld: Direct and selected policy must both pass the screen.')
            st.dataframe(table, hide_index=True, use_container_width=True)
            front = table[table['Pareto among training-feasible']]
            if not front.empty:
                chart(px.scatter(front, x='LEO mass kg', y='Mean cost USD', color='route', symbol='tanker_mode', hover_data=['cargo_launch_day', 'launch_cadence_days', 'reserve_fraction', 'initial_fill', 'Failure probability']), 'Non-dominated feasible candidates')
            st.download_button('Export search table', table.to_csv(index=False).encode(), file_name='policy_search.csv', mime='text/csv')
        elif stored: st.info('Inputs changed. Run the search again.')

    with t_rob:
        st.subheader('Architecture decision map — does the winner depend on hub availability or launch reliability?')
        st.caption('Each cell re-optimizes all architectures (reduced candidate set, ≤500 draws). Hub availability is a user-supplied programme-risk input, not a forecast.')
        qm = Requirements(st.slider('Service threshold for map (%)', 0., 30., 10., .5, key='mapthr') / 100, .25)
        if st.button('Compute decision map', key='map'):
            with st.spinner('Computing 16 cells...'): st.session_state['map50'] = ((s, qm), c_map(s, qm))
        mp = st.session_state.get('map50')
        if mp and mp[0] == (s, qm):
            chart(px.density_heatmap(mp[1].assign(code=mp[1]['Winner'].astype('category').cat.codes), x='Hub availability', y='Launch reliability', z='code', text_auto=False), 'Winner code per cell (see table)')
            st.dataframe(mp[1], hide_index=True, use_container_width=True)
            st.download_button('Export decision map', mp[1].to_csv(index=False).encode(), file_name='decision_map.csv', mime='text/csv')
        st.subheader('Sensitivity of the manually selected policy (no re-optimization)')
        if st.button('Run sensitivity cases', key='sens'):
            try:
                with st.spinner('Evaluating 17 cases...'): st.session_state['sens50'] = ((s, p, qm), c_sens(s, p, qm))
            except (ValueError, OverflowError) as exc: st.error(str(exc))
        sv = st.session_state.get('sens50')
        if sv and sv[0] == (s, p, qm):
            v = sv[1].copy(); v['Difference USD M'] = v['Difference USD'] / 1e6
            chart(px.bar(v, x='Difference USD M', y='Case', orientation='h', color='Both pass confidence screen'), 'Direct minus selected cost — not a valid saving where the service screen fails')
            st.dataframe(sv[1], hide_index=True, use_container_width=True)

    with t_ver:
        st.subheader('Verification, provenance and review gates')
        if st.button('Run engine verification suite', key='tests'): st.session_state['ver50'] = c_tests()
        chk = st.session_state.get('ver50')
        if chk:
            (st.success if chk[0] else st.error)('All engine verification tests passed.' if chk[0] else 'Test failure: resolve before relying on outputs.')
            st.code(chk[1]); st.download_button('Export test report', chk[1].encode(), file_name='verification.txt', mime='text/plain')
        st.caption('Engine tests do not verify astrodynamic truth, customer demand or any programme status.')
        provenance = st.data_editor(pd.DataFrame(imported.get('provenance', PROVENANCE)), num_rows='dynamic', hide_index=True, use_container_width=True, key='prov' + ident)
        st.caption('Do not mark reviewed without the actual reviewer and evidence. No automatic TRL or certification is assigned.')

    with t_biz:
        st.subheader('Astraia SaaS business — EUR, independent of customer USD')
        a, b = st.columns(2)
        clients = a.number_input('Average active annual contracts', min_value=0, value=3, key='clients')
        price = b.number_input('Annual contract price (€)', min_value=0., value=12000., step=1000.)
        pilot = a.number_input('Paid pilot revenue in year (€)', min_value=0., value=0., step=1000.)
        delivery = b.number_input('Cloud and direct support (€ / year)', min_value=0., value=10000., step=1000.)
        opex = a.number_input('Other development / sales / admin (€ / year)', min_value=0., value=58000., step=1000.)
        cash = b.number_input('Unrestricted opening cash (€)', min_value=0., value=10000., step=1000.)
        br = business_model(clients, price, pilot, delivery, opex, cash)
        business = {'inputs': {'clients': clients, 'annual_price_eur': price, 'pilot_revenue_eur': pilot, 'direct_cost_eur': delivery, 'other_opex_eur': opex, 'unrestricted_cash_eur': cash}, 'results': br}
        a, b, c = st.columns(3)
        a.metric('ARR assumption', f"€{br['arr']:,.0f}"); b.metric('Annual operating cash proxy', f"€{br['net']:,.0f}")
        c.metric('Uniform-flow runway', f"{br['runway_months']:.1f} months" if br['runway_months'] is not None else 'No modeled burn')
        cashdf = pd.DataFrame({'Month': range(13), 'Cash EUR': [cash + m_ * br['net'] / 12 for m_ in range(13)]})
        chart(px.line(cashdf, x='Month', y='Cash EUR', markers=True), 'Illustrative cash path — no grant assumed received')
        st.caption('Hypotheses, not signed revenue. No taxes, sales ramp or grant milestone timing.')

    with t_pilot:
        st.subheader('External pilot evidence worksheet')
        st.caption('Session-only, not a secure CRM. Use anonymized data in public deployments.')
        ev = st.data_editor(pd.DataFrame({'Prospect ID': [''], 'User / buyer role': [''], 'Problem': [''], 'Current workflow': [''], 'Evidence status': ['Unverified'], 'Next action': ['']}), num_rows='dynamic', hide_index=True, use_container_width=True, key='pilot')
        st.download_button('Export pilot worksheet', ev.to_csv(index=False).encode(), file_name='pilot_evidence.csv', mime='text/csv')

    with t_exp:
        st.subheader('Assumptions and limitations')
        for text in LIMITS: st.write('• ' + text)
        st.write('Scenario fingerprint:', fingerprint(s, p))
        prov = provenance.fillna('').astype(str).to_dict('records')
        blob = export_json(s, p, destination, prov, business)
        st.download_button('Export scenario JSON', blob.encode(), file_name='astraia_v50_scenario.json', mime='application/json')
        st.download_button('Export paired Monte Carlo CSV', direct['data'].add_prefix('Direct: ').join(alt['data'].add_prefix('Alt: ')).to_csv(index=False).encode(), file_name='paired_scenarios.csv', mime='text/csv')
        st.download_button('Export launched-mass manifests', pd.DataFrame(rows).to_csv(index=False).encode(), file_name='mass_manifest.csv', mime='text/csv')
        st.json(json.loads(blob), expanded=False)


if __name__ == '__main__':
    if '--test' in sys.argv:
        ok, report = run_tests(); print(report); sys.exit(0 if ok else 1)
    else:
        main()
