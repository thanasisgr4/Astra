import json
import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# ASTRAIA OS — SINGLE-FILE APPLICATION
#
# Install:
# pip install streamlit numpy pandas plotly
#
# Run:
# streamlit run app.py
# ============================================================

st.set_page_config(
    page_title="Astraia OS | Decision Lab",
    page_icon="🛰️",
    layout="wide",
)

G0 = 9.80665
MODEL_VERSION = "Astraia OS 2.0.0"


# ============================================================
# 1. DATA MODELS
# ============================================================

@dataclass(frozen=True)
class Mission:
    annual_payload_t: float = 100.0
    payload_capacity_t: float = 10.0
    vehicle_dry_t: float = 2.0
    isp_s: float = 450.0
    direct_dv: float = 5900.0
    upstream_dv: float = 3800.0
    downstream_dv: float = 2100.0
    tanker_dry_ratio: float = 0.10
    boiloff_daily: float = 0.005
    module_capacity_t: float = 50.0


@dataclass(frozen=True)
class Costs:
    launch_per_kg: float = 5000.0
    vehicle_per_mission: float = 15e6
    direct_ops_per_mission: float = 2e6
    hub_ops_per_mission: float = 3e6
    tanker_service_per_kg: float = 1000.0
    depot_propellant_per_kg: float = 10.0
    module_capex: float = 250e6
    shared_capex: float = 15e6
    module_opex: float = 20e6
    shared_opex: float = 5e6
    asset_life_years: int = 10
    capital_rate: float = 0.10
    shortage_penalty_per_kg: float = 20000.0


@dataclass(frozen=True)
class Uncertainty:
    launch_cv: float = 0.15
    operating_cv: float = 0.20
    boiloff_cv: float = 0.30
    seed: int = 42
    draws: int = 10000


# ============================================================
# 2. VALIDATION
# ============================================================

def validate(m, c, u):
    for obj in (m, c, u):
        for name, value in asdict(obj).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

    positive_fields = (
        "annual_payload_t",
        "payload_capacity_t",
        "vehicle_dry_t",
        "isp_s",
        "module_capacity_t",
    )

    for name in positive_fields:
        if getattr(m, name) <= 0:
            raise ValueError(f"{name} must be positive.")

    if not 0 <= m.boiloff_daily < 1:
        raise ValueError(
            "Daily boil-off must be greater than or equal to "
            "zero and less than one."
        )

    nonnegative_mission_fields = (
        "direct_dv",
        "upstream_dv",
        "downstream_dv",
        "tanker_dry_ratio",
    )

    for name in nonnegative_mission_fields:
        if getattr(m, name) < 0:
            raise ValueError(f"{name} cannot be negative.")

    for name, value in asdict(c).items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")

    if c.asset_life_years < 1:
        raise ValueError(
            "Infrastructure economic life must be at least one year."
        )

    if u.draws < 100:
        raise ValueError("Use at least 100 simulation scenarios.")

    if min(
        u.launch_cv,
        u.operating_cv,
        u.boiloff_cv,
        u.seed,
    ) < 0:
        raise ValueError(
            "Uncertainty parameters cannot be negative."
        )


# ============================================================
# 3. PHYSICS ENGINE
# ============================================================

def burn_initial_mass(final_mass_kg, delta_v_m_s, isp_s):
    if final_mass_kg < 0:
        raise ValueError("Final mass cannot be negative.")

    if delta_v_m_s < 0:
        raise ValueError("Delta-v cannot be negative.")

    if isp_s <= 0:
        raise ValueError("Specific impulse must be positive.")

    exponent = delta_v_m_s / (G0 * isp_s)

    if exponent > 100:
        raise ValueError(
            "The selected delta-v / Isp combination "
            "produces an impractically large mass ratio."
        )

    return final_mass_kg * math.exp(exponent)


def capital_recovery_factor(rate, years):
    if years <= 0:
        raise ValueError("Economic life must be positive.")

    if rate < 0:
        raise ValueError("Capital rate cannot be negative.")

    if rate == 0:
        return 1.0 / years

    return rate / (
        1.0 - (1.0 + rate) ** (-years)
    )


def original_reference(
    payload_t=100.0,
    isp_s=450.0,
    boiloff_daily=0.005,
    storage_days=5,
):
    """
    Exact reproduction of the original prototype equations.

    This is a historical reference, not a complete
    common-endpoint mission comparison.
    """
    payload_kg = payload_t * 1000.0

    direct_ratio = math.exp(
        5900.0 / (isp_s * 9.81)
    )

    hub_ratio = math.exp(
        3800.0 / (isp_s * 9.81)
    )

    direct_mass = payload_kg * (
        1.0 + (direct_ratio - 1.0) * 1.10
    )

    hub_propellant = payload_kg * (
        hub_ratio - 1.0
    )

    retention = (
        1.0 - boiloff_daily
    ) ** storage_days

    hub_mass = (
        payload_kg
        + hub_propellant * (2.0 - retention) * 1.10
    )

    savings_pct = (
        100.0 * (1.0 - hub_mass / direct_mass)
    )

    return {
        "direct_t": direct_mass / 1000.0,
        "hub_t": hub_mass / 1000.0,
        "savings_pct": savings_pct,
    }


def mass_budget(
    m,
    route="Direct",
    storage_days=5.0,
    fuel_buffer=0.0,
):
    """
    Both architectures start in LEO and end at the same
    declared destination.

    Direct:
        LEO -> destination.

    Hub:
        Cargo vehicle: LEO -> hub -> destination.
        Separate tanker flow: LEO -> hub.

    Assumptions:
        - Expendable cargo vehicles.
        - Expendable tankers.
        - One effective Isp for all burns.
        - Tanker dry mass scales with delivered fuel.
        - No return legs.
        - No discrete tanker capacity model.
    """
    if route not in ("Direct", "Hub"):
        raise ValueError("Unknown route.")

    if storage_days < 0 or fuel_buffer < 0:
        raise ValueError(
            "Storage days and fuel buffer cannot be negative."
        )

    missions = math.ceil(
        m.annual_payload_t / m.payload_capacity_t
    )

    annual_payload_kg = (
        m.annual_payload_t * 1000.0
    )

    annual_vehicle_dry_kg = (
        missions * m.vehicle_dry_t * 1000.0
    )

    final_cargo_mass = (
        annual_payload_kg + annual_vehicle_dry_kg
    )

    components = {
        "Payload": annual_payload_kg,
        "Cargo vehicle dry mass": annual_vehicle_dry_kg,
    }

    if route == "Direct":
        initial_mass = burn_initial_mass(
            final_cargo_mass,
            m.direct_dv,
            m.isp_s,
        )

        components["Direct propulsion fuel"] = (
            initial_mass - final_cargo_mass
        )

        required_fuel = 0.0
        loaded_fuel = 0.0
        design_inventory = 0.0
        modules = 0

    else:
        upstream_initial = burn_initial_mass(
            final_cargo_mass,
            m.upstream_dv,
            m.isp_s,
        )

        downstream_initial = burn_initial_mass(
            final_cargo_mass,
            m.downstream_dv,
            m.isp_s,
        )

        required_fuel = (
            downstream_initial - final_cargo_mass
        )

        retention = (
            1.0 - m.boiloff_daily
        ) ** storage_days

        if retention <= 0:
            raise ValueError(
                "Fuel retention must be positive."
            )

        loaded_fuel = (
            required_fuel
            * (1.0 + fuel_buffer)
            / retention
        )

        tanker_dry_mass = (
            loaded_fuel * m.tanker_dry_ratio
        )

        tanker_initial = burn_initial_mass(
            loaded_fuel + tanker_dry_mass,
            m.upstream_dv,
            m.isp_s,
        )

        initial_mass = (
            upstream_initial + tanker_initial
        )

        components.update({
            "Cargo upstream fuel": (
                upstream_initial - final_cargo_mass
            ),
            "Downstream nominal fuel": required_fuel,
            "Retained fuel buffer": (
                required_fuel * fuel_buffer
            ),
            "Nominal storage loss": (
                loaded_fuel * (1.0 - retention)
            ),
            "Tanker dry mass": tanker_dry_mass,
            "Tanker upstream fuel": (
                tanker_initial
                - loaded_fuel
                - tanker_dry_mass
            ),
        })

        design_inventory = loaded_fuel / missions

        modules = max(
            1,
            math.ceil(
                design_inventory
                / (m.module_capacity_t * 1000.0)
            ),
        )

    if not math.isclose(
        sum(components.values()),
        initial_mass,
        rel_tol=1e-9,
        abs_tol=1e-5,
    ):
        raise ValueError(
            "Mass conservation check failed."
        )

    return {
        "route": route,
        "storage_days": storage_days,
        "fuel_buffer": fuel_buffer,
        "missions": missions,
        "initial_kg": initial_mass,
        "fuel_required_kg": required_fuel,
        "fuel_loaded_kg": loaded_fuel,
        "modules": modules,
        "design_inventory_kg": design_inventory,
        "components": components,
    }


# ============================================================
# 4. CUSTOMER ECONOMICS
# ============================================================

def nominal_cost(m, c, budget):
    launch = (
        budget["initial_kg"]
        * c.launch_per_kg
    )

    vehicles = (
        budget["missions"]
        * c.vehicle_per_mission
    )

    if budget["route"] == "Direct":
        operations = (
            budget["missions"]
            * c.direct_ops_per_mission
        )

        depot_fuel = 0.0
        tanker_service = 0.0
        depot_opex = 0.0
        capex = 0.0

    else:
        operations = (
            budget["missions"]
            * c.hub_ops_per_mission
        )

        depot_fuel = (
            budget["fuel_loaded_kg"]
            * c.depot_propellant_per_kg
        )

        tanker_service = (
            budget["fuel_loaded_kg"]
            * c.tanker_service_per_kg
        )

        depot_opex = (
            budget["modules"] * c.module_opex
            + c.shared_opex
        )

        capex = (
            budget["modules"] * c.module_capex
            + c.shared_capex
        )

    annual_capital_charge = (
        capex
        * capital_recovery_factor(
            c.capital_rate,
            c.asset_life_years,
        )
    )

    nonlaunch = (
        vehicles
        + operations
        + depot_fuel
        + tanker_service
        + depot_opex
    )

    return {
        "LEO launch": launch,
        "Cargo vehicles": vehicles,
        "Mission operations": operations,
        "Depot fuel procurement": depot_fuel,
        "Tanker service": tanker_service,
        "Depot fixed OPEX": depot_opex,
        "Annualized infrastructure capital": (
            annual_capital_charge
        ),
        "capex": capex,
        "nonlaunch": nonlaunch,
        "total": (
            launch
            + nonlaunch
            + annual_capital_charge
        ),
    }


# ============================================================
# 5. UNCERTAINTY ENGINE
# ============================================================

def positive_multiplier(rng, cv, size):
    sigma = math.sqrt(
        math.log1p(cv * cv)
    )

    mu = -0.5 * sigma * sigma

    return rng.lognormal(
        mean=mu,
        sigma=sigma,
        size=size,
    )


def generate_scenarios(u):
    rng = np.random.default_rng(u.seed)

    return {
        "launch": positive_multiplier(
            rng,
            u.launch_cv,
            u.draws,
        ),
        "operating": positive_multiplier(
            rng,
            u.operating_cv,
            u.draws,
        ),
        "boiloff": positive_multiplier(
            rng,
            u.boiloff_cv,
            u.draws,
        ),
    }


def evaluate_policy(m, c, budget, scenarios):
    costs = nominal_cost(
        m,
        c,
        budget,
    )

    scenario_cost = (
        costs["LEO launch"]
        * scenarios["launch"]
        + costs["nonlaunch"]
        * scenarios["operating"]
        + costs["Annualized infrastructure capital"]
    )

    if (
        budget["route"] == "Hub"
        and budget["fuel_required_kg"] > 0
    ):
        actual_daily_rate = np.clip(
            m.boiloff_daily
            * scenarios["boiloff"],
            0.0,
            0.95,
        )

        available_fuel = (
            budget["fuel_loaded_kg"]
            * (1.0 - actual_daily_rate)
            ** budget["storage_days"]
        )

        shortfall = np.maximum(
            budget["fuel_required_kg"]
            - available_fuel,
            0.0,
        )

        scenario_cost = (
            scenario_cost
            + shortfall
            * c.shortage_penalty_per_kg
        )

    else:
        shortfall = np.zeros_like(
            scenario_cost
        )

    p95 = float(
        np.quantile(scenario_cost, 0.95)
    )

    tail = scenario_cost[
        scenario_cost >= p95
    ]

    cvar95 = float(tail.mean())

    return {
        "cost": scenario_cost,
        "shortfall": shortfall,
        "expected_cost": float(
            scenario_cost.mean()
        ),
        "p95_cost": p95,
        "cvar95": cvar95,
        "shortfall_probability": float(
            (shortfall > 1e-6).mean()
        ),
    }


# ============================================================
# 6. POLICY OPTIMIZER
# ============================================================

def optimize_policies(
    m,
    c,
    u,
    minimum_days,
    maximum_days,
    maximum_buffer,
    risk_weight,
    maximum_shortfall_probability,
):
    validate(m, c, u)

    if minimum_days > maximum_days:
        raise ValueError(
            "Minimum storage days cannot exceed maximum."
        )

    scenarios = generate_scenarios(u)

    day_grid = sorted(
        set(
            np.linspace(
                minimum_days,
                maximum_days,
                7,
            ).round(2)
        )
    )

    buffer_grid = sorted(
        set(
            np.linspace(
                0.0,
                maximum_buffer,
                9,
            ).round(4)
        )
    )

    candidates = [
        ("Direct", 0.0, 0.0)
    ]

    candidates.extend(
        ("Hub", float(days), float(buffer))
        for days in day_grid
        for buffer in buffer_grid
    )

    rows = []

    for route, days, buffer in candidates:
        budget = mass_budget(
            m,
            route,
            days,
            buffer,
        )

        result = evaluate_policy(
            m,
            c,
            budget,
            scenarios,
        )

        score = (
            result["expected_cost"]
            + risk_weight
            * (
                result["cvar95"]
                - result["expected_cost"]
            )
        )

        feasible = (
            result["shortfall_probability"]
            <= maximum_shortfall_probability
        )

        rows.append({
            "Route": route,
            "Storage days": days,
            "Fuel buffer": buffer,
            "Launch mass t": (
                budget["initial_kg"] / 1000.0
            ),
            "Modules": budget["modules"],
            "Expected cost USD": (
                result["expected_cost"]
            ),
            "P95 cost USD": result["p95_cost"],
            "CVaR95 USD": result["cvar95"],
            "Fuel shortfall probability": (
                result["shortfall_probability"]
            ),
            "Risk score USD": score,
            "Feasible": feasible,
        })

    table = pd.DataFrame(rows)

    table = table.sort_values(
        ["Feasible", "Risk score USD"],
        ascending=[False, True],
    ).reset_index(drop=True)

    feasible_table = table[
        table["Feasible"]
    ]

    if feasible_table.empty:
        raise ValueError(
            "No candidate meets the selected constraint."
        )

    best = (
        feasible_table.iloc[0].to_dict()
    )

    return table, best


@st.cache_data(show_spinner=False)
def cached_optimize(
    m,
    c,
    u,
    minimum_days,
    maximum_days,
    maximum_buffer,
    risk_weight,
    maximum_shortfall_probability,
):
    return optimize_policies(
        m,
        c,
        u,
        minimum_days,
        maximum_days,
        maximum_buffer,
        risk_weight,
        maximum_shortfall_probability,
    )


# ============================================================
# 7. SAAS COMPANY ECONOMICS
# ============================================================

def calculate_saas_finances(
    subscriptions,
    monthly_price,
    enterprise_contracts,
    annual_enterprise_price,
    services_revenue,
    cloud_cost,
    support_cost,
    personnel_cost,
    sales_cost,
    admin_cost,
    opening_cash,
    upfront_build_cost,
):
    arr = (
        subscriptions * monthly_price * 12.0
        + enterprise_contracts
        * annual_enterprise_price
    )

    revenue = (
        arr + services_revenue
    )

    direct_costs = (
        cloud_cost + support_cost
    )

    operating_expenses = (
        personnel_cost
        + sales_cost
        + admin_cost
    )

    annual_operating_cash = (
        revenue
        - direct_costs
        - operating_expenses
    )

    cash_after_build = (
        opening_cash
        - upfront_build_cost
    )

    if cash_after_build < 0:
        runway_months = 0.0
    elif annual_operating_cash < 0:
        runway_months = (
            cash_after_build
            / (-annual_operating_cash / 12.0)
        )
    else:
        runway_months = None

    gross_margin = (
        (revenue - direct_costs) / revenue
        if revenue > 0
        else 0.0
    )

    return {
        "ARR": arr,
        "Revenue": revenue,
        "Direct service costs": direct_costs,
        "Operating expenses": operating_expenses,
        "Annual operating cash proxy": (
            annual_operating_cash
        ),
        "Cash after build": cash_after_build,
        "Runway months": runway_months,
        "Gross margin": gross_margin,
    }


# ============================================================
# 8. PRESENTATION HELPERS
# ============================================================

def usd_m(value):
    return f"${value / 1e6:,.2f}M"


def eur(value):
    return f"€{value:,.0f}"


def show_chart(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#BAC3D1"),
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=30,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


st.markdown(
    """
    <style>
        .stApp {
            background-color: #080D16;
            color: #E8EDF6;
        }

        section[data-testid="stSidebar"] {
            background-color: #101827;
        }

        div[data-testid="stMetric"] {
            background-color: #121E30;
            border: 1px solid #293A54;
            border-top: 3px solid #4B85FF;
            padding: 16px;
            border-radius: 8px;
        }

        h1, h2, h3 {
            color: #F3F6FC;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ASTRAIA OS")
st.caption(
    f"{MODEL_VERSION} | "
    "Space Logistics Decision Lab | "
    "Customer mission economics and SaaS finances"
)

st.warning(
    "Research prototype. No flight qualification, "
    "validated trajectory solution, guaranteed savings, "
    "or investment recommendation. Defaults are illustrative."
)


# ============================================================
# 9. SIDEBAR INPUTS
# ============================================================

with st.sidebar:
    st.header("Customer mission assumptions")

    destination = st.text_input(
        "Common destination",
        value="Specified lunar orbit — not yet validated",
    )

    st.caption(
        "Origin: LEO. Both complete architectures "
        "must terminate at the same destination."
    )

    with st.expander(
        "Demand and spacecraft",
        expanded=True,
    ):
        annual_payload_t = st.number_input(
            "Annual payload (tonnes/year)",
            min_value=1.0,
            value=100.0,
            step=10.0,
        )

        payload_capacity_t = st.number_input(
            "Payload capacity per mission (tonnes)",
            min_value=0.1,
            value=10.0,
            step=1.0,
        )

        vehicle_dry_t = st.number_input(
            "Vehicle dry mass per mission (tonnes)",
            min_value=0.01,
            value=2.0,
            step=0.25,
        )

        isp_s = st.number_input(
            "Specific impulse, all burns (seconds)",
            min_value=100.0,
            value=450.0,
            step=10.0,
        )

        tanker_dry_ratio = st.number_input(
            "Tanker dry mass / delivered fuel mass",
            min_value=0.0,
            value=0.10,
            step=0.02,
        )

    with st.expander(
        "Common-endpoint delta-v",
        expanded=True,
    ):
        direct_dv = st.number_input(
            "Direct LEO → destination (m/s)",
            min_value=0.0,
            max_value=20000.0,
            value=5900.0,
            step=100.0,
        )

        upstream_dv = st.number_input(
            "LEO → hub (m/s)",
            min_value=0.0,
            max_value=20000.0,
            value=3800.0,
            step=100.0,
        )

        downstream_dv = st.number_input(
            "Hub → destination (m/s)",
            min_value=0.0,
            max_value=20000.0,
            value=2100.0,
            step=100.0,
        )

        st.caption(
            "The 2,100 m/s downstream default is an "
            "arithmetic placeholder, not a calculated transfer. "
            "No landing is modeled."
        )

    with st.expander("Selected hub policy"):
        boiloff_daily = st.slider(
            "Daily boil-off (%)",
            0.0,
            5.0,
            0.5,
            0.1,
        ) / 100.0

        storage_days = st.slider(
            "Fuel storage duration (days)",
            0,
            60,
            5,
        )

        fuel_buffer = st.slider(
            "Retained fuel buffer (%)",
            0,
            100,
            10,
        ) / 100.0

        module_capacity_t = st.number_input(
            "Storage capacity per module (tonnes)",
            min_value=0.1,
            value=50.0,
            step=5.0,
        )

        st.caption(
            "Storage is sized for one isolated fuel batch "
            "per cargo mission. Batch overlaps and transfer-rate "
            "limits are not modeled."
        )

    with st.expander(
        "Customer mission costs — USD"
    ):
        launch_per_kg = st.number_input(
            "LEO launch ($/kg)",
            min_value=1.0,
            value=5000.0,
            step=500.0,
        )

        vehicle_per_mission = st.number_input(
            "Cargo vehicle procurement ($M/mission)",
            min_value=0.0,
            value=15.0,
            step=1.0,
        ) * 1e6

        direct_ops_per_mission = st.number_input(
            "Direct operations ($M/mission)",
            min_value=0.0,
            value=2.0,
            step=0.5,
        ) * 1e6

        hub_ops_per_mission = st.number_input(
            "Hub operations ($M/mission)",
            min_value=0.0,
            value=3.0,
            step=0.5,
        ) * 1e6

        tanker_service_per_kg = st.number_input(
            "Tanker service excluding launch ($/kg fuel)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )

        depot_propellant_per_kg = st.number_input(
            "Depot propellant procurement ($/kg)",
            min_value=0.0,
            value=10.0,
            step=5.0,
        )

        module_capex = st.number_input(
            "Installed module CAPEX ($M)",
            min_value=0.0,
            value=250.0,
            step=25.0,
        ) * 1e6

        shared_capex = st.number_input(
            "Shared setup CAPEX ($M)",
            min_value=0.0,
            value=15.0,
            step=5.0,
        ) * 1e6

        module_opex = st.number_input(
            "Annual module OPEX ($M)",
            min_value=0.0,
            value=20.0,
            step=1.0,
        ) * 1e6

        shared_opex = st.number_input(
            "Annual shared OPEX ($M)",
            min_value=0.0,
            value=5.0,
            step=1.0,
        ) * 1e6

        asset_life_years = st.slider(
            "Infrastructure economic life (years)",
            1,
            30,
            10,
        )

        capital_rate = st.slider(
            "Capital recovery rate (%)",
            0.0,
            25.0,
            10.0,
            0.5,
        ) / 100.0

        shortage_penalty_per_kg = st.number_input(
            "Fuel-shortfall penalty ($/kg missing fuel)",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
        )

    with st.expander("Simulation assumptions"):
        draws = st.selectbox(
            "Scenario count",
            [1000, 10000, 25000],
            index=1,
        )

        seed = int(
            st.number_input(
                "Reproducible random seed",
                min_value=0,
                max_value=2147483647,
                value=42,
                step=1,
            )
        )

        launch_cv = st.slider(
            "Launch-price coefficient of variation (%)",
            0,
            60,
            15,
        ) / 100.0

        operating_cv = st.slider(
            "Operating-cost coefficient of variation (%)",
            0,
            60,
            20,
        ) / 100.0

        boiloff_cv = st.slider(
            "Boil-off coefficient of variation (%)",
            0,
            100,
            30,
        ) / 100.0


# ============================================================
# 10. EXECUTION
# ============================================================

mission = Mission(
    annual_payload_t=annual_payload_t,
    payload_capacity_t=payload_capacity_t,
    vehicle_dry_t=vehicle_dry_t,
    isp_s=isp_s,
    direct_dv=direct_dv,
    upstream_dv=upstream_dv,
    downstream_dv=downstream_dv,
    tanker_dry_ratio=tanker_dry_ratio,
    boiloff_daily=boiloff_daily,
    module_capacity_t=module_capacity_t,
)

cost_inputs = Costs(
    launch_per_kg=launch_per_kg,
    vehicle_per_mission=vehicle_per_mission,
    direct_ops_per_mission=direct_ops_per_mission,
    hub_ops_per_mission=hub_ops_per_mission,
    tanker_service_per_kg=tanker_service_per_kg,
    depot_propellant_per_kg=depot_propellant_per_kg,
    module_capex=module_capex,
    shared_capex=shared_capex,
    module_opex=module_opex,
    shared_opex=shared_opex,
    asset_life_years=asset_life_years,
    capital_rate=capital_rate,
    shortage_penalty_per_kg=shortage_penalty_per_kg,
)

uncertainty = Uncertainty(
    launch_cv=launch_cv,
    operating_cv=operating_cv,
    boiloff_cv=boiloff_cv,
    seed=seed,
    draws=int(draws),
)

try:
    validate(
        mission,
        cost_inputs,
        uncertainty,
    )

    direct_budget = mass_budget(
        mission,
        route="Direct",
    )

    hub_budget = mass_budget(
        mission,
        route="Hub",
        storage_days=storage_days,
        fuel_buffer=fuel_buffer,
    )

    direct_cost = nominal_cost(
        mission,
        cost_inputs,
        direct_budget,
    )

    hub_cost = nominal_cost(
        mission,
        cost_inputs,
        hub_budget,
    )

    scenario_data = generate_scenarios(
        uncertainty
    )

    direct_result = evaluate_policy(
        mission,
        cost_inputs,
        direct_budget,
        scenario_data,
    )

    hub_result = evaluate_policy(
        mission,
        cost_inputs,
        hub_budget,
        scenario_data,
    )

except (
    ValueError,
    OverflowError,
    ZeroDivisionError,
) as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# 11. TABS
# ============================================================

(
    mission_tab,
    optimizer_tab,
    uncertainty_tab,
    reference_tab,
    company_tab,
    network_tab,
    audit_tab,
) = st.tabs([
    "Mission Economics",
    "Policy Optimizer",
    "Uncertainty",
    "Original 38% Reference",
    "SaaS Company",
    "Network",
    "Audit & Exports",
])


# ============================================================
# 12. MISSION ECONOMICS
# ============================================================

with mission_tab:
    st.subheader(
        "Customer architecture comparison"
    )

    st.caption(
        "Customer mission costs are in USD. "
        "They are not Astraia subscription revenue "
        "or software-company expenses."
    )

    mass_savings_pct = (
        100.0
        * (
            1.0
            - hub_budget["initial_kg"]
            / direct_budget["initial_kg"]
        )
    )

    customer_advantage = (
        direct_result["expected_cost"]
        - hub_result["expected_cost"]
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Hub launch-mass savings",
        f"{mass_savings_pct:.2f}%",
    )

    b.metric(
        "Direct expected annual cost",
        usd_m(
            direct_result["expected_cost"]
        ),
    )

    c.metric(
        "Hub expected annual cost",
        usd_m(
            hub_result["expected_cost"]
        ),
    )

    hub_shortfall_pct = (
        hub_result["shortfall_probability"]
        * 100.0
    )

    d.metric(
        "Hub fuel-shortfall probability",
        f"{hub_shortfall_pct:.1f}%",
    )

    if mass_savings_pct < 0:
        st.info(
            "These assumptions do not produce a hub mass "
            "advantage after depot-fuel supply is included."
        )

    mass_rows = []

    for budget in (
        direct_budget,
        hub_budget,
    ):
        for component, mass in (
            budget["components"].items()
        ):
            mass_rows.append({
                "Architecture": budget["route"],
                "Component": component,
                "Mass t": mass / 1000.0,
            })

    mass_df = pd.DataFrame(
        mass_rows
    )

    mass_chart = px.bar(
        mass_df,
        x="Architecture",
        y="Mass t",
        color="Component",
        barmode="stack",
    )

    show_chart(
        mass_chart,
        "Annual mass launched to LEO",
    )

    cost_keys = [
        "LEO launch",
        "Cargo vehicles",
        "Mission operations",
        "Depot fuel procurement",
        "Tanker service",
        "Depot fixed OPEX",
        "Annualized infrastructure capital",
    ]

    cost_table = pd.DataFrame([
        {
            "Cost component": key,
            "Direct USD": direct_cost[key],
            "Hub USD": hub_cost[key],
        }
        for key in cost_keys
    ])

    st.dataframe(
        cost_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Annualized infrastructure capital is a recovery "
        "charge, not annual cash CAPEX. Installed CAPEX must "
        "include depot deployment and commissioning. "
        "Infrastructure deployment mass is excluded from "
        "recurring launch mass."
    )

    x, y, z = st.columns(3)

    x.metric(
        "Hub installed CAPEX",
        usd_m(hub_cost["capex"]),
    )

    y.metric(
        "Storage modules",
        int(hub_budget["modules"]),
    )

    fuel_batch_t = (
        hub_budget["design_inventory_kg"]
        / 1000.0
    )

    z.metric(
        "Nominal fuel batch",
        f"{fuel_batch_t:.2f} t",
    )

    st.metric(
        "Hub customer cost advantage before software fee",
        usd_m(customer_advantage),
    )

    st.caption(
        "Architecture advantage is different from the "
        "incremental value attributable to Astraia software."
    )


# ============================================================
# 13. POLICY OPTIMIZATION
# ============================================================

with optimizer_tab:
    st.subheader(
        "Scenario-based policy search"
    )

    st.write(
        "Search direct delivery and candidate hub policies "
        "using identical random scenarios. The engine chooses "
        "a route, storage duration, and fuel buffer. "
        "It does not optimize physical trajectories."
    )

    a, b, c = st.columns(3)

    minimum_days = a.number_input(
        "Minimum feasible storage days",
        min_value=0,
        max_value=60,
        value=2,
        step=1,
    )

    maximum_days = b.number_input(
        "Maximum feasible storage days",
        min_value=0,
        max_value=60,
        value=20,
        step=1,
    )

    maximum_buffer = c.slider(
        "Maximum buffer searched (%)",
        0,
        100,
        50,
    ) / 100.0

    a, b = st.columns(2)

    risk_weight = a.slider(
        "Tail-risk weight",
        0.0,
        1.0,
        0.25,
        0.05,
    )

    maximum_shortfall_probability = b.slider(
        "Maximum fuel-shortfall probability (%)",
        0.0,
        100.0,
        5.0,
        0.5,
    ) / 100.0

    st.caption(
        "Objective = expected cost + risk weight × "
        "(CVaR95 − expected cost). The storage-time window "
        "must reflect operational feasibility. No value-of-delay "
        "trade-off is modeled."
    )

    optimization_signature = (
        mission,
        cost_inputs,
        uncertainty,
        int(minimum_days),
        int(maximum_days),
        maximum_buffer,
        risk_weight,
        maximum_shortfall_probability,
    )

    if st.button(
        "Optimize candidate policies",
        type="primary",
    ):
        try:
            with st.spinner(
                "Evaluating policies..."
            ):
                policy_table, best_policy = (
                    cached_optimize(
                        *optimization_signature
                    )
                )

            st.session_state[
                "astraia_optimization"
            ] = (
                optimization_signature,
                policy_table,
                best_policy,
            )

        except (
            ValueError,
            OverflowError,
            ZeroDivisionError,
        ) as exc:
            st.error(str(exc))

    stored_optimization = (
        st.session_state.get(
            "astraia_optimization"
        )
    )

    if (
        stored_optimization is not None
        and stored_optimization[0]
        == optimization_signature
    ):
        policy_table = (
            stored_optimization [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/36404152/)
        )

        best_policy = (
            stored_optimization [arxiv](https://arxiv.org/pdf/2501.05046.pdf)
        )

        best_route = best_policy["Route"]
        best_days = best_policy["Storage days"]
        best_buffer = best_policy["Fuel buffer"]

        st.success(
            f"Selected policy: {best_route} | "
            f"Storage: {best_days:.2f} days | "
            f"Fuel buffer: {best_buffer * 100.0:.1f}%"
        )

        best_budget = mass_budget(
            mission,
            route=best_route,
            storage_days=best_days,
            fuel_buffer=best_buffer,
        )

        best_result = evaluate_policy(
            mission,
            cost_inputs,
            best_budget,
            scenario_data,
        )

        benchmark_name = st.selectbox(
            "Software-value benchmark",
            [
                "Selected hub policy",
                "Direct policy",
            ],
        )

        if benchmark_name == "Selected hub policy":
            benchmark_result = hub_result
        else:
            benchmark_result = direct_result

        improvement = (
            benchmark_result["expected_cost"]
            - best_result["expected_cost"]
        )

        a, b = st.columns(2)

        a.metric(
            "Expected cost reduction versus benchmark",
            usd_m(improvement),
        )

        best_shortfall_pct = (
            best_result["shortfall_probability"]
            * 100.0
        )

        b.metric(
            "Selected-policy fuel-shortfall probability",
            f"{best_shortfall_pct:.2f}%",
        )

        annual_software_fee = st.number_input(
            "Illustrative annual software fee (USD)",
            min_value=0.0,
            value=50000.0,
            step=10000.0,
        )

        st.metric(
            "Modeled customer net benefit after fee",
            usd_m(
                improvement
                - annual_software_fee
            ),
        )

        st.caption(
            "This is modeled improvement versus a declared "
            "benchmark, not proven commercial savings. "
            "A pilot requires an agreed baseline and "
            "independent validation."
        )

        display_table = (
            policy_table.copy()
        )

        money_columns = [
            "Expected cost USD",
            "P95 cost USD",
            "CVaR95 USD",
            "Risk score USD",
        ]

        for column in money_columns:
            display_table[column] = (
                display_table[column]
                / 1e6
            )

        display_table = (
            display_table.rename(
                columns={
                    column: column.replace(
                        "USD",
                        "USD M",
                    )
                    for column in money_columns
                }
            )
        )

        st.dataframe(
            display_table,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download candidate policies CSV",
            data=policy_table.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="astraia_policy_candidates.csv",
            mime="text/csv",
        )

        holdout_uncertainty = replace(
            uncertainty,
            seed=(
                uncertainty.seed + 1
            ) % 2147483647,
        )

        holdout_scenarios = (
            generate_scenarios(
                holdout_uncertainty
            )
        )

        holdout_result = evaluate_policy(
            mission,
            cost_inputs,
            best_budget,
            holdout_scenarios,
        )

        holdout_shortfall_pct = (
            holdout_result[
                "shortfall_probability"
            ] * 100.0
        )

        st.subheader(
            "Independent-seed check"
        )

        a, b = st.columns(2)

        a.metric(
            "Expected cost",
            usd_m(
                holdout_result[
                    "expected_cost"
                ]
            ),
        )

        b.metric(
            "Fuel-shortfall probability",
            f"{holdout_shortfall_pct:.2f}%",
        )

        st.caption(
            "This checks sensitivity to sampled scenarios. "
            "It does not validate the physical assumptions "
            "or chosen probability distributions."
        )

    elif stored_optimization is not None:
        st.info(
            "Inputs changed. Run optimization again "
            "to refresh the results."
        )


# ============================================================
# 14. UNCERTAINTY DASHBOARD
# ============================================================

with uncertainty_tab:
    st.subheader(
        "Paired economic uncertainty"
    )

    st.caption(
        "Independent, mean-one lognormal factors are used. "
        "Coefficient of variation means standard deviation "
        "divided by mean—not a bounded ± range. "
        "Fuel-shortfall probability is not mission-failure "
        "probability."
    )

    scenario_advantage = (
        direct_result["cost"]
        - hub_result["cost"]
    )

    probability_hub_cheaper = (
        100.0
        * (scenario_advantage > 0).mean()
    )

    p5_advantage = float(
        np.quantile(
            scenario_advantage,
            0.05,
        )
    )

    p95_advantage = float(
        np.quantile(
            scenario_advantage,
            0.95,
        )
    )

    a, b, c = st.columns(3)

    a.metric(
        "P(hub costs less than direct)",
        f"{probability_hub_cheaper:.1f}%",
    )

    b.metric(
        "P5 hub cost advantage",
        usd_m(p5_advantage),
    )

    c.metric(
        "P95 hub cost advantage",
        usd_m(p95_advantage),
    )

    scenario_df = pd.DataFrame({
        "Direct annual cost USD": (
            direct_result["cost"]
        ),
        "Hub annual cost USD": (
            hub_result["cost"]
        ),
        "Hub cost advantage USD": (
            scenario_advantage
        ),
        "Hub missing fuel kg": (
            hub_result["shortfall"]
        ),
    })

    histogram_df = pd.DataFrame({
        "Hub cost advantage USD M": (
            scenario_advantage / 1e6
        )
    })

    histogram = px.histogram(
        histogram_df,
        x="Hub cost advantage USD M",
        nbins=70,
        color_discrete_sequence=[
            "#4B85FF"
        ],
    )

    histogram.add_vline(
        x=0,
        line_dash="dash",
        line_color="white",
    )

    show_chart(
        histogram,
        "Hub versus direct annual cost advantage",
    )

    risk_rows = []

    for route, result in (
        ("Direct", direct_result),
        ("Hub", hub_result),
    ):
        risk_rows.append({
            "Route": route,
            "Mean cost USD": (
                result["expected_cost"]
            ),
            "P95 cost USD": (
                result["p95_cost"]
            ),
            "CVaR95 USD": (
                result["cvar95"]
            ),
            "Fuel-shortfall probability": (
                result[
                    "shortfall_probability"
                ]
            ),
        })

    st.dataframe(
        pd.DataFrame(risk_rows),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download paired scenarios CSV",
        data=scenario_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="astraia_paired_scenarios.csv",
        mime="text/csv",
    )


# ============================================================
# 15. ORIGINAL 38% REFERENCE
# ============================================================

with reference_tab:
    st.subheader(
        "Original baseline reproduction"
    )

    st.warning(
        "This panel reproduces the original prototype's "
        "partial-route equations. It is not a validated "
        "end-to-end mission saving or measured software benefit."
    )

    reference = original_reference()

    a, b, c = st.columns(3)

    reference_direct_t = (
        reference["direct_t"]
    )

    reference_hub_t = (
        reference["hub_t"]
    )

    reference_savings_pct = (
        reference["savings_pct"]
    )

    a.metric(
        "Original direct modeled mass",
        f"{reference_direct_t:.2f} t",
    )

    b.metric(
        "Original hub modeled mass",
        f"{reference_hub_t:.2f} t",
    )

    c.metric(
        "Original modeled mass savings",
        f"{reference_savings_pct:.2f}%",
    )

    st.code(
        """
payload = 100_000
isp = 450
g0 = 9.81
dry_ratio = 0.10

retention = (1 - 0.005) ** 5

mr_direct = math.exp(
    5900 / (isp * g0)
)

mr_hub = math.exp(
    3800 / (isp * g0)
)

direct_mass = payload * (
    1 + (mr_direct - 1) * (1 + dry_ratio)
)

hub_propellant = payload * (
    mr_hub - 1
)

hub_mass = (
    payload
    + hub_propellant
    * (2 - retention)
    * (1 + dry_ratio)
)

mass_savings_percent = (
    100 * (1 - hub_mass / direct_mass)
)
        """,
        language="python",
    )

    st.write(
        "The fixed original reference uses 100 tonnes, "
        "450 seconds Isp, 5,900 m/s direct delta-v, "
        "3,800 m/s to L1, 0.5% daily boil-off, "
        "and five storage days."
    )

    st.write(
        "The complete architecture model differs because "
        "it includes a downstream leg, explicit cargo-vehicle "
        "dry mass, a separate depot-fuel supply flow, "
        "and tanker hardware and propulsion."
    )


# ============================================================
# 16. SAAS COMPANY FINANCES
# ============================================================

with company_tab:
    st.subheader(
        "Astraia SaaS company — EUR"
    )

    st.caption(
        "Separate from customer USD mission economics. "
        "Prices and sales assumptions are illustrative. "
        "No automatic currency conversion."
    )

    left, right = st.columns(2)

    with left:
        subscriptions = st.number_input(
            "Planning subscriptions: annual-average active count",
            min_value=0,
            value=3,
            step=1,
        )

        monthly_price = st.number_input(
            "Monthly planning subscription price (€)",
            min_value=0.0,
            value=500.0,
            step=100.0,
        )

        enterprise_contracts = st.number_input(
            "Enterprise contracts: annual-average count",
            min_value=0,
            value=1,
            step=1,
        )

        annual_enterprise_price = st.number_input(
            "Annual enterprise contract price (€)",
            min_value=0.0,
            value=15000.0,
            step=1000.0,
        )

        services_revenue = st.number_input(
            "Annual onboarding/services revenue (€)",
            min_value=0.0,
            value=5000.0,
            step=1000.0,
        )

        opening_cash = st.number_input(
            "Opening grant/cash (€)",
            min_value=0.0,
            value=50000.0,
            step=5000.0,
        )

    with right:
        cloud_cost = st.number_input(
            "Annual cloud/data service costs (€)",
            min_value=0.0,
            value=4000.0,
            step=1000.0,
        )

        support_cost = st.number_input(
            "Annual direct customer support (€)",
            min_value=0.0,
            value=6000.0,
            step=1000.0,
        )

        personnel_cost = st.number_input(
            "Annual development/personnel (€)",
            min_value=0.0,
            value=48000.0,
            step=1000.0,
        )

        sales_cost = st.number_input(
            "Annual sales/customer discovery (€)",
            min_value=0.0,
            value=6000.0,
            step=1000.0,
        )

        admin_cost = st.number_input(
            "Annual security/legal/admin (€)",
            min_value=0.0,
            value=4000.0,
            step=1000.0,
        )

        upfront_build_cost = st.number_input(
            "One-time build payment at start (€)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
        )

    st.caption(
        "Avoid counting the same development work in both "
        "the one-time build payment and annual personnel "
        "expenses. Subscription counts represent steady "
        "annual-average activity, not a sales ramp."
    )

    saas_inputs = {
        "subscriptions": subscriptions,
        "monthly_price": monthly_price,
        "enterprise_contracts": (
            enterprise_contracts
        ),
        "annual_enterprise_price": (
            annual_enterprise_price
        ),
        "services_revenue": services_revenue,
        "cloud_cost": cloud_cost,
        "support_cost": support_cost,
        "personnel_cost": personnel_cost,
        "sales_cost": sales_cost,
        "admin_cost": admin_cost,
        "opening_cash": opening_cash,
        "upfront_build_cost": (
            upfront_build_cost
        ),
    }

    saas_result = (
        calculate_saas_finances(
            **saas_inputs
        )
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Annual recurring revenue",
        eur(saas_result["ARR"]),
    )

    b.metric(
        "Annual operating cash proxy",
        eur(
            saas_result[
                "Annual operating cash proxy"
            ]
        ),
    )

    gross_margin_pct = (
        saas_result["Gross margin"]
        * 100.0
    )

    c.metric(
        "Gross margin proxy",
        f"{gross_margin_pct:.1f}%",
    )

    runway = (
        saas_result["Runway months"]
    )

    if saas_result["Cash after build"] < 0:
        runway_text = (
            "Cash exhausted at start"
        )
    elif runway is None:
        runway_text = (
            "No operating burn"
        )
    else:
        runway_text = (
            f"{runway:.1f} months"
        )

    d.metric(
        "Steady-state runway proxy",
        runway_text,
    )

    cash_rows = []

    for month in range(13):
        cash_balance = (
            saas_result["Cash after build"]
            + month
            * saas_result[
                "Annual operating cash proxy"
            ]
            / 12.0
        )

        cash_rows.append({
            "Month": month,
            "Cash EUR": cash_balance,
        })

    cash_df = pd.DataFrame(
        cash_rows
    )

    cash_chart = px.line(
        cash_df,
        x="Month",
        y="Cash EUR",
        markers=True,
    )

    cash_chart.add_hline(
        y=0,
        line_dash="dash",
        line_color="white",
    )

    show_chart(
        cash_chart,
        "Simplified cash path",
    )

    st.caption(
        "Revenue and operating expenses are spread uniformly "
        "across months. Taxes, working-capital timing, "
        "funding restrictions, and sales-ramp delays are excluded."
    )

    st.download_button(
        "Download SaaS cash projection CSV",
        data=cash_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="astraia_saas_cash_eur.csv",
        mime="text/csv",
    )

    st.subheader(
        "Illustrative €50,000 allocation"
    )

    funding_df = pd.DataFrame({
        "Use": [
            "Product development",
            "External technical review",
            "Customer pilots",
            "Cloud and tools",
            "Security and legal",
            "Documentation",
            "Contingency",
        ],
        "EUR": [
            18000,
            10000,
            7000,
            4000,
            4000,
            2000,
            5000,
        ],
    })

    st.dataframe(
        funding_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "This allocation is a separate planning view. "
        "It is not automatically added to the cash model. "
        "Grant eligibility depends on the funding terms."
    )


# ============================================================
# 17. NETWORK SCHEMATIC
# ============================================================

with network_tab:
    st.subheader(
        "Illustrative cislunar logistics network"
    )

    st.caption(
        "Not to scale. No live telemetry, propagated "
        "orbits, flight times, or calculated trajectories."
    )

    positions = {
        "LEO": (0.0, 0.0),
        "L1 hub": (1.0, 0.7),
        "Destination": (2.0, 0.0),
    }

    network_figure = go.Figure()

    edges = [
        (
            "LEO",
            "Destination",
            "Direct cargo route",
            "#8A96A8",
            "dash",
        ),
        (
            "LEO",
            "L1 hub",
            "Cargo + separate fuel supply",
            "#4B85FF",
            "solid",
        ),
        (
            "L1 hub",
            "Destination",
            "Refueled downstream cargo",
            "#27C4A8",
            "solid",
        ),
    ]

    for (
        start,
        end,
        label,
        color,
        dash,
    ) in edges:
        x0, y0 = positions[start]
        x1, y1 = positions[end]

        network_figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                name=label,
                line=dict(
                    color=color,
                    width=4,
                    dash=dash,
                ),
                hoverinfo="name",
            )
        )

    network_figure.add_trace(
        go.Scatter(
            x=[
                position[0]
                for position
                in positions.values()
            ],
            y=[
                position [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/36404152/)
                for position
                in positions.values()
            ],
            mode="markers+text",
            text=list(
                positions.keys()
            ),
            textposition="top center",
            marker=dict(
                size=[24, 20, 24],
                color=[
                    "#5CB8FF",
                    "#4B85FF",
                    "#CBD5E1",
                ],
            ),
            showlegend=False,
            hoverinfo="text",
        )
    )

    network_figure.update_xaxes(
        visible=False,
        range=[-0.3, 2.3],
    )

    network_figure.update_yaxes(
        visible=False,
        range=[-0.35, 1.05],
    )

    network_figure.update_layout(
        height=450
    )

    show_chart(
        network_figure
    )

    st.write(
        "Declared destination:",
        destination,
    )

    st.info(
        "Cargo vehicles and tankers are expendable in this "
        "version. Reuse requires explicit return burns, "
        "residual propellant, refurbishment, and scheduling."
    )


# ============================================================
# 18. AUDIT, CHECKS, AND EXPORTS
# ============================================================

with audit_tab:
    st.subheader(
        "Model boundaries"
    )

    st.markdown(
        """
        - Both complete architectures share a declared origin
          and destination.
        - Delta-v values are user assumptions, not computed
          trajectories.
        - The ideal rocket equation is used with one effective
          specific impulse.
        - No lunar landing or ascent is modeled.
        - Cargo vehicles and tankers are expendable.
        - Annual demand is split across equally loaded cargo missions.
        - Depot fuel is supplied from Earth.
        - Tanker hardware scales linearly with fuel cargo.
        - Discrete tanker capacity and flight counts are excluded.
        - Storage is sized for one isolated fuel batch.
        - Overlapping batches and transfer-rate limits are excluded.
        - Boil-off uncertainty is shared across the annual aggregate.
        - Missing fuel incurs a penalty proxy, not a modeled
          emergency-resupply mission.
        - Fuel-shortfall probability is not mission-failure probability.
        - Launch price is a linear mass-based proxy.
        - Installed infrastructure CAPEX must include deployment.
        - Depot cargo fuel procurement is itemized separately.
        - Other propulsion-fuel procurement is assumed bundled
          into vehicle/service inputs.
        - Distribution assumptions require calibration.
        - Customer mission costs are USD.
        - SaaS company finances are EUR.
        - No automatic currency conversion is applied.
        - Original 38% savings are retained only as a labeled reference.
        """
    )

    st.subheader(
        "Calculation checks"
    )

    direct_mass_error = abs(
        sum(
            direct_budget[
                "components"
            ].values()
        )
        - direct_budget["initial_kg"]
    )

    hub_mass_error = abs(
        sum(
            hub_budget[
                "components"
            ].values()
        )
        - hub_budget["initial_kg"]
    )

    fixed_reference = (
        original_reference()
    )

    checks = pd.DataFrame([
        {
            "Check": (
                "Direct mass conservation"
            ),
            "Passed": (
                direct_mass_error < 1e-5
            ),
            "Value": (
                f"{direct_mass_error:.8f} kg residual"
            ),
        },
        {
            "Check": (
                "Hub mass conservation"
            ),
            "Passed": (
                hub_mass_error < 1e-5
            ),
            "Value": (
                f"{hub_mass_error:.8f} kg residual"
            ),
        },
        {
            "Check": (
                "Original baseline reproduction"
            ),
            "Passed": (
                abs(
                    fixed_reference[
                        "savings_pct"
                    ]
                    - 37.872210
                ) < 1e-5
            ),
            "Value": (
                f"{fixed_reference['savings_pct']:.6f}%"
            ),
        },
    ])

    st.dataframe(
        checks,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Accounting checks do not validate trajectories, "
        "hardware designs, operational feasibility, "
        "or market assumptions."
    )

    scenario_export = {
        "model_version": MODEL_VERSION,
        "destination": destination,
        "mission_inputs": asdict(mission),
        "customer_cost_inputs_usd": (
            asdict(cost_inputs)
        ),
        "uncertainty": (
            asdict(uncertainty)
        ),
        "selected_hub_policy": {
            "storage_days": storage_days,
            "fuel_buffer": fuel_buffer,
        },
        "original_reference": (
            fixed_reference
        ),
        "nominal_direct_cost_usd": (
            direct_cost
        ),
        "nominal_hub_cost_usd": (
            hub_cost
        ),
        "saas_inputs_eur": saas_inputs,
        "saas_results_eur": (
            saas_result
        ),
    }

    st.download_button(
        "Download scenario JSON",
        data=json.dumps(
            scenario_export,
            indent=2,
            allow_nan=False,
        ).encode("utf-8"),
        file_name="astraia_scenario.json",
        mime="application/json",
    )

    summary_rows = []

    for route, budget, result in (
        (
            "Direct",
            direct_budget,
            direct_result,
        ),
        (
            "Hub",
            hub_budget,
            hub_result,
        ),
    ):
        summary_rows.append({
            "Route": route,
            "Launch mass kg": (
                budget["initial_kg"]
            ),
            "Expected annual cost USD": (
                result["expected_cost"]
            ),
            "P95 annual cost USD": (
                result["p95_cost"]
            ),
            "CVaR95 annual cost USD": (
                result["cvar95"]
            ),
            "Fuel-shortfall probability": (
                result[
                    "shortfall_probability"
                ]
            ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    st.download_button(
        "Download summary CSV",
        data=summary_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="astraia_summary.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download mass budget CSV",
        data=mass_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="astraia_mass_budget.csv",
        mime="text/csv",
    )

    st.json(
        scenario_export,
        expanded=False,
    )

    st.info(
        "Before operational or commercial use: independently "
        "validate trajectories, vehicle masses, replenishment "
        "schedules, uncertainty assumptions, and customer "
        "benchmarks. Add authentication, tenant isolation, "
        "secure persistence, access controls, monitoring, "
        "and a deployment security review."
    )
```
