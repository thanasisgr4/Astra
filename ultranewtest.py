import math
from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Astraia Dynamics | Logistics Screening",
    layout="wide",
)

G0 = 9.80665
DAYS_PER_YEAR = 365.0


# ============================================================
# 2. DATA MODELS
# ============================================================

@dataclass(frozen=True)
class MissionInputs:
    annual_demand_kg: float
    mission_capacity_kg: float

    # One-way cargo vehicle dry mass at full capacity.
    vehicle_dry_mass_kg: float

    isp_s: float

    # All routes share the same origin and final destination.
    dv_direct_m_s: float
    dv_leo_hub_m_s: float
    dv_hub_destination_m_s: float

    # Dry tanker mass / propellant cargo delivered to hub.
    tanker_dry_to_cargo_ratio: float

    boil_off_fraction_per_day: float
    storage_days: float

    # Explicit simplifying assumption:
    # cargo vehicles and tankers are expendable.
    # No vehicle return or reuse is assumed.


@dataclass(frozen=True)
class FinancialInputs:
    launch_cost_per_kg: float
    customer_discount_fraction: float
    reserve_fraction_of_revenue: float

    cargo_vehicle_cost_per_mission: float
    direct_operations_cost_per_mission: float
    hub_operations_cost_per_mission: float

    # Procurement/production cost of depot cargo propellant.
    # Tanker ascent propellant is included in the launch-mass
    # proxy but is not separately priced as a commodity.
    depot_propellant_cost_per_kg: float

    # Aggregate tanker-service cost, excluding LEO launch.
    # This proxy covers tanker hardware and service operations.
    tanker_service_cost_per_kg_delivered: float

    module_storage_capacity_kg: float
    inventory_peak_factor: float

    module_capex: float
    shared_setup_capex: float
    annual_opex_per_module: float
    annual_shared_opex: float

    project_years: int
    discount_rate: float


# ============================================================
# 3. PHYSICS AND ACCOUNTING
# ============================================================

def burn_budget(
    final_mass_kg: float,
    delta_v_m_s: float,
    isp_s: float,
) -> Dict[str, float]:
    """Ideal single-stage burn with explicit final mass."""
    if final_mass_kg <= 0:
        raise ValueError("Final mass must be positive.")
    if delta_v_m_s < 0:
        raise ValueError("Delta-v cannot be negative.")
    if isp_s <= 0:
        raise ValueError("Specific impulse must be positive.")

    mass_ratio = math.exp(delta_v_m_s / (isp_s * G0))
    initial_mass = final_mass_kg * mass_ratio

    return {
        "final_mass_kg": final_mass_kg,
        "propellant_kg": initial_mass - final_mass_kg,
        "initial_mass_kg": initial_mass,
        "mass_ratio": mass_ratio,
    }


def validate_inputs(m: MissionInputs, f: FinancialInputs) -> None:
    if m.annual_demand_kg <= 0:
        raise ValueError("Annual demand must be positive.")
    if m.mission_capacity_kg <= 0:
        raise ValueError("Mission capacity must be positive.")
    if m.vehicle_dry_mass_kg <= 0:
        raise ValueError("Vehicle dry mass must be positive.")
    if m.isp_s <= 0:
        raise ValueError("Specific impulse must be positive.")
    if not 0 <= m.boil_off_fraction_per_day < 1:
        raise ValueError("Daily boil-off must be between 0 and 1.")
    if m.storage_days < 0:
        raise ValueError("Storage duration cannot be negative.")
    if m.tanker_dry_to_cargo_ratio < 0:
        raise ValueError("Tanker dry-to-cargo ratio cannot be negative.")

    for dv in (
        m.dv_direct_m_s,
        m.dv_leo_hub_m_s,
        m.dv_hub_destination_m_s,
    ):
        if dv < 0:
            raise ValueError("Delta-v cannot be negative.")

    if f.launch_cost_per_kg <= 0:
        raise ValueError("Launch price must be positive.")
    if f.module_storage_capacity_kg <= 0:
        raise ValueError("Module storage capacity must be positive.")
    if f.inventory_peak_factor < 1:
        raise ValueError("Inventory peak factor must be at least 1.")
    if not 0 <= f.customer_discount_fraction < 1:
        raise ValueError("Customer discount must be between 0 and 1.")
    if not 0 <= f.reserve_fraction_of_revenue < 1:
        raise ValueError("Reserve fraction must be between 0 and 1.")
    if f.project_years < 1:
        raise ValueError("Project duration must be at least one year.")
    if f.discount_rate < 0:
        raise ValueError("Discount rate cannot be negative.")

    cost_fields = (
        f.cargo_vehicle_cost_per_mission,
        f.direct_operations_cost_per_mission,
        f.hub_operations_cost_per_mission,
        f.depot_propellant_cost_per_kg,
        f.tanker_service_cost_per_kg_delivered,
        f.module_capex,
        f.shared_setup_capex,
        f.annual_opex_per_module,
        f.annual_shared_opex,
    )
    if any(value < 0 for value in cost_fields):
        raise ValueError("Cost assumptions cannot be negative.")


def calculate_architectures(m: MissionInputs) -> Dict[str, float]:
    """
    Common endpoint: a user-specified lunar destination orbit.

    Direct:
        LEO -> destination.

    Hub:
        Cargo vehicle: LEO -> hub -> destination.
        Separate tanker flow: LEO -> hub carrying downstream fuel.

    Simplifications:
        - Expendable cargo vehicles and tankers.
        - Cargo vehicle dry mass is fixed per mission.
        - Downstream fuel is supplied by a separate tanker flow.
        - Same Isp for all burns.
        - Tanker hardware scales linearly with delivered cargo.
        - No discrete tanker flight schedule.
        - No trajectory propagation or gravity-loss modeling.
    """
    missions = math.ceil(
        m.annual_demand_kg / m.mission_capacity_kg
    )

    payload_per_mission = m.annual_demand_kg / missions
    dry_per_mission = m.vehicle_dry_mass_kg
    final_cargo_mass = payload_per_mission + dry_per_mission

    direct = burn_budget(
        final_cargo_mass,
        m.dv_direct_m_s,
        m.isp_s,
    )

    downstream = burn_budget(
        final_cargo_mass,
        m.dv_hub_destination_m_s,
        m.isp_s,
    )
    downstream_fuel_per_mission = downstream["propellant_kg"]

    # The cargo vehicle reaches the hub without downstream fuel.
    upstream = burn_budget(
        final_cargo_mass,
        m.dv_leo_hub_m_s,
        m.isp_s,
    )

    retention = (
        1.0 - m.boil_off_fraction_per_day
    ) ** m.storage_days

    fuel_loaded_per_mission = (
        downstream_fuel_per_mission / retention
    )
    boil_off_per_mission = (
        fuel_loaded_per_mission - downstream_fuel_per_mission
    )

    tanker_dry_per_mission_equivalent = (
        fuel_loaded_per_mission * m.tanker_dry_to_cargo_ratio
    )

    if fuel_loaded_per_mission > 0:
        tanker = burn_budget(
            fuel_loaded_per_mission
            + tanker_dry_per_mission_equivalent,
            m.dv_leo_hub_m_s,
            m.isp_s,
        )
    else:
        tanker = {
            "initial_mass_kg": 0.0,
            "propellant_kg": 0.0,
            "final_mass_kg": 0.0,
        }

    annual_direct_mass = direct["initial_mass_kg"] * missions
    annual_cargo_launch_mass = (
        upstream["initial_mass_kg"] * missions
    )
    annual_tanker_launch_mass = (
        tanker["initial_mass_kg"] * missions
    )
    annual_hub_mass = (
        annual_cargo_launch_mass + annual_tanker_launch_mass
    )

    result = {
        "missions": missions,
        "payload_per_mission_kg": payload_per_mission,
        "annual_payload_kg": m.annual_demand_kg,
        "annual_cargo_dry_mass_kg": dry_per_mission * missions,

        "direct_initial_mass_kg": annual_direct_mass,
        "direct_propellant_kg": (
            direct["propellant_kg"] * missions
        ),

        "hub_initial_mass_kg": annual_hub_mass,
        "hub_cargo_initial_mass_kg": annual_cargo_launch_mass,
        "hub_tanker_initial_mass_kg": annual_tanker_launch_mass,

        "hub_upstream_propellant_kg": (
            upstream["propellant_kg"] * missions
        ),
        "hub_downstream_propellant_kg": (
            downstream_fuel_per_mission * missions
        ),
        "hub_boil_off_kg": boil_off_per_mission * missions,
        "hub_fuel_delivered_kg": (
            fuel_loaded_per_mission * missions
        ),
        "hub_tanker_dry_mass_kg": (
            tanker_dry_per_mission_equivalent * missions
        ),
        "hub_tanker_propellant_kg": (
            tanker["propellant_kg"] * missions
        ),

        "hub_fuel_batch_kg": fuel_loaded_per_mission,
        "retention": retention,
        "mass_savings_pct": (
            100.0 * (1.0 - annual_hub_mass / annual_direct_mass)
        ),
    }

    direct_components = (
        result["annual_payload_kg"]
        + result["annual_cargo_dry_mass_kg"]
        + result["direct_propellant_kg"]
    )

    hub_components = (
        result["annual_payload_kg"]
        + result["annual_cargo_dry_mass_kg"]
        + result["hub_upstream_propellant_kg"]
        + result["hub_downstream_propellant_kg"]
        + result["hub_boil_off_kg"]
        + result["hub_tanker_dry_mass_kg"]
        + result["hub_tanker_propellant_kg"]
    )

    if not math.isclose(
        direct_components,
        annual_direct_mass,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("Direct architecture mass balance failed.")

    if not math.isclose(
        hub_components,
        annual_hub_mass,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("Hub architecture mass balance failed.")

    return result


# ============================================================
# 4. ECONOMICS
# ============================================================

def calculate_economics(
    m: MissionInputs,
    f: FinancialInputs,
    a: Dict[str, float],
) -> Dict[str, float]:

    # Preliminary inventory sizing:
    # size for average resident inventory OR one mission batch,
    # whichever is larger, then apply a peak/reserve allowance.
    average_inventory = (
        a["hub_fuel_delivered_kg"]
        * m.storage_days
        / DAYS_PER_YEAR
    )

    design_inventory = (
        max(average_inventory, a["hub_fuel_batch_kg"])
        * f.inventory_peak_factor
    )

    modules = max(
        1,
        math.ceil(
            design_inventory / f.module_storage_capacity_kg
        ),
    )

    capex = (
        modules * f.module_capex + f.shared_setup_capex
    )

    direct_launch_cost = (
        a["direct_initial_mass_kg"] * f.launch_cost_per_kg
    )
    direct_vehicle_cost = (
        a["missions"] * f.cargo_vehicle_cost_per_mission
    )
    direct_operations_cost = (
        a["missions"] * f.direct_operations_cost_per_mission
    )

    # Modeled reference cost, not a market quotation.
    direct_reference_cost = (
        direct_launch_cost
        + direct_vehicle_cost
        + direct_operations_cost
    )

    revenue = (
        direct_reference_cost
        * (1.0 - f.customer_discount_fraction)
    )

    hub_launch_cost = (
        a["hub_initial_mass_kg"] * f.launch_cost_per_kg
    )
    hub_vehicle_cost = (
        a["missions"] * f.cargo_vehicle_cost_per_mission
    )
    hub_mission_operations = (
        a["missions"] * f.hub_operations_cost_per_mission
    )
    depot_propellant_cost = (
        a["hub_fuel_delivered_kg"]
        * f.depot_propellant_cost_per_kg
    )
    tanker_service_cost = (
        a["hub_fuel_delivered_kg"]
        * f.tanker_service_cost_per_kg_delivered
    )
    fixed_opex = (
        modules * f.annual_opex_per_module
        + f.annual_shared_opex
    )

    nonlaunch_cost = (
        hub_vehicle_cost
        + hub_mission_operations
        + depot_propellant_cost
        + tanker_service_cost
        + fixed_opex
    )

    reserve = revenue * f.reserve_fraction_of_revenue

    annual_surplus = (
        revenue - hub_launch_cost - nonlaunch_cost - reserve
    )

    payback = (
        capex / annual_surplus
        if annual_surplus > 0
        else None
    )

    years = np.arange(1, f.project_years + 1)
    discount_factors = (
        1.0 / (1.0 + f.discount_rate) ** years
    )
    annuity_factor = float(discount_factors.sum())

    # Simplified project cash-flow proxy:
    # upfront CAPEX and constant annual surplus.
    npv = -capex + annual_surplus * annuity_factor

    break_even_revenue = (
        hub_launch_cost + nonlaunch_cost
    ) / (1.0 - f.reserve_fraction_of_revenue)

    return {
        "modules": modules,
        "average_inventory_kg": average_inventory,
        "design_inventory_kg": design_inventory,
        "capex": capex,

        "direct_launch_cost": direct_launch_cost,
        "direct_reference_cost": direct_reference_cost,

        "revenue": revenue,
        "hub_launch_cost": hub_launch_cost,
        "hub_vehicle_cost": hub_vehicle_cost,
        "hub_mission_operations": hub_mission_operations,
        "depot_propellant_cost": depot_propellant_cost,
        "tanker_service_cost": tanker_service_cost,
        "fixed_opex": fixed_opex,
        "nonlaunch_cost": nonlaunch_cost,
        "reserve": reserve,

        "annual_surplus": annual_surplus,
        "surplus_margin_pct": (
            annual_surplus / revenue * 100.0
            if revenue > 0
            else 0.0
        ),
        "payback_years": payback,
        "npv": npv,
        "annuity_factor": annuity_factor,
        "break_even_revenue": break_even_revenue,
    }


# ============================================================
# 5. UNCERTAINTY MODEL
# ============================================================

def positive_multiplier(
    rng: np.random.Generator,
    coefficient_of_variation: float,
    size: int,
) -> np.ndarray:
    """Mean-one lognormal multiplier."""
    sigma = math.sqrt(
        math.log(1.0 + coefficient_of_variation ** 2)
    )
    mu = -0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size)


def simulate_economics(
    a: Dict[str, float],
    e: Dict[str, float],
    f: FinancialInputs,
    draws: int,
    seed: int,
    launch_cv: float,
    opex_cv: float,
    capex_cv: float,
    pricing_mode: str,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    launch_multiplier = positive_multiplier(
        rng, launch_cv, draws
    )
    opex_multiplier = positive_multiplier(
        rng, opex_cv, draws
    )
    capex_multiplier = positive_multiplier(
        rng, capex_cv, draws
    )

    simulated_launch_price = (
        f.launch_cost_per_kg * launch_multiplier
    )

    if pricing_mode == "Fixed customer price":
        revenue = np.full(draws, e["revenue"])
    else:
        # Explicit indexed-pricing assumption:
        # reference nonlaunch costs remain fixed.
        reference_nonlaunch = (
            e["direct_reference_cost"]
            - e["direct_launch_cost"]
        )
        revenue = (
            a["direct_initial_mass_kg"]
            * simulated_launch_price
            + reference_nonlaunch
        ) * (1.0 - f.customer_discount_fraction)

    launch_cost = (
        a["hub_initial_mass_kg"] * simulated_launch_price
    )
    nonlaunch_cost = (
        e["nonlaunch_cost"] * opex_multiplier
    )
    reserve = revenue * f.reserve_fraction_of_revenue

    annual_surplus = (
        revenue - launch_cost - nonlaunch_cost - reserve
    )

    simulated_capex = e["capex"] * capex_multiplier

    # Each draw is a constant operating regime across the
    # project, not an independently simulated annual path.
    project_npv = (
        -simulated_capex
        + annual_surplus * e["annuity_factor"]
    )

    return pd.DataFrame({
        "Annual surplus USD": annual_surplus,
        "Project NPV USD": project_npv,
        "Revenue USD": revenue,
        "Launch price USD per kg": simulated_launch_price,
        "CAPEX USD": simulated_capex,
    })


# ============================================================
# 6. PRESENTATION HELPERS
# ============================================================

def style_figure(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B8C0CC"),
        margin=dict(l=20, r=20, t=55, b=30),
    )
    return fig


def money(value):
    return f"${value / 1e6:,.1f}M"


st.markdown(
    """
    <style>
        .stApp {
            background-color: #070A10;
            color: #E5E7EB;
        }
        section[data-testid="stSidebar"] {
            background-color: #0D121C;
        }
        div[data-testid="stMetric"] {
            background-color: #101723;
            border: 1px solid #253044;
            border-top: 3px solid #3366FF;
            padding: 16px;
            border-radius: 7px;
        }
        h1, h2, h3 {
            color: #F3F4F6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ASTRAIA DYNAMICS")
st.caption(
    "Cislunar logistics screening | Explicit mass budgets | "
    "Scenario economics | Uncertainty analysis"
)

st.warning(
    "Conceptual screening model—not flight-design software or a "
    "validated business case. All trajectory, hardware, cost, and "
    "demand inputs require independent validation."
)


# ============================================================
# 7. SIDEBAR INPUTS
# ============================================================

with st.sidebar:
    st.header("Scenario assumptions")

    st.caption(
        "Common service: deliver cargo from LEO to the same "
        "user-specified lunar destination orbit."
    )

    with st.expander("Demand and spacecraft", expanded=True):
        annual_tons = st.number_input(
            "Annual delivered cargo (tonnes/year)",
            min_value=1.0,
            value=100.0,
            step=10.0,
        )
        mission_tons = st.number_input(
            "Cargo capacity per mission (tonnes)",
            min_value=0.1,
            value=10.0,
            step=1.0,
        )
        dry_tons = st.number_input(
            "Cargo vehicle dry mass (tonnes/mission)",
            min_value=0.01,
            value=2.0,
            step=0.25,
        )
        isp = st.number_input(
            "Specific impulse, all burns (seconds)",
            min_value=100.0,
            value=450.0,
            step=10.0,
        )
        tanker_ratio = st.number_input(
            "Tanker dry mass / delivered fuel mass",
            min_value=0.0,
            value=0.10,
            step=0.02,
        )

    with st.expander("Route delta-v assumptions", expanded=True):
        st.caption(
            "Illustrative inputs. No landing is modeled. "
            "Both alternatives must terminate in the same orbit."
        )
        dv_direct = st.number_input(
            "LEO → destination (m/s)",
            min_value=0.0,
            value=4100.0,
            step=100.0,
        )
        dv_hub = st.number_input(
            "LEO → hub (m/s)",
            min_value=0.0,
            value=3800.0,
            step=100.0,
        )
        dv_downstream = st.number_input(
            "Hub → destination (m/s)",
            min_value=0.0,
            value=800.0,
            step=100.0,
        )

    with st.expander("Depot and inventory"):
        boil_pct = st.slider(
            "Boil-off (% per day)",
            0.0, 5.0, 0.5, 0.1,
        )
        storage_days = st.slider(
            "Fuel storage duration (days)",
            0, 90, 5,
        )
        module_tons = st.number_input(
            "Storage capacity per module (tonnes)",
            min_value=0.1,
            value=50.0,
            step=5.0,
        )
        inventory_peak = st.number_input(
            "Inventory peak/reserve factor",
            min_value=1.0,
            value=1.25,
            step=0.05,
        )

    with st.expander("Pricing and mission costs"):
        launch_price = st.number_input(
            "LEO launch price ($/kg)",
            min_value=1.0,
            value=5000.0,
            step=500.0,
        )
        discount_pct = st.slider(
            "Discount to modeled direct reference (%)",
            0.0, 50.0, 20.0, 1.0,
        )
        reserve_pct = st.slider(
            "Revenue held as annual reserve (%)",
            0.0, 30.0, 10.0, 1.0,
        )
        vehicle_cost_m = st.number_input(
            "Cargo vehicle procurement ($M/mission)",
            min_value=0.0,
            value=15.0,
            step=1.0,
        )
        direct_ops_m = st.number_input(
            "Direct operations ($M/mission)",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )
        hub_ops_m = st.number_input(
            "Hub-route operations ($M/mission)",
            min_value=0.0,
            value=3.0,
            step=0.5,
        )
        propellant_price = st.number_input(
            "Depot cargo propellant procurement ($/kg)",
            min_value=0.0,
            value=10.0,
            step=5.0,
        )
        tanker_service_price = st.number_input(
            "Tanker hardware/service, excluding launch "
            "($/kg delivered to hub)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )

    with st.expander("Infrastructure and project"):
        module_capex_m = st.number_input(
            "Installed module CAPEX ($M)",
            min_value=0.0,
            value=250.0,
            step=25.0,
        )
        setup_capex_m = st.number_input(
            "Shared setup CAPEX ($M)",
            min_value=0.0,
            value=15.0,
            step=5.0,
        )
        module_opex_m = st.number_input(
            "Annual OPEX per module ($M)",
            min_value=0.0,
            value=20.0,
            step=1.0,
        )
        shared_opex_m = st.number_input(
            "Annual shared OPEX ($M)",
            min_value=0.0,
            value=5.0,
            step=1.0,
        )
        project_years = st.slider(
            "Project duration (years)", 1, 30, 10
        )
        discount_rate_pct = st.slider(
            "Annual discount rate (%)",
            0.0, 30.0, 10.0, 0.5,
        )


mission = MissionInputs(
    annual_demand_kg=annual_tons * 1000.0,
    mission_capacity_kg=mission_tons * 1000.0,
    vehicle_dry_mass_kg=dry_tons * 1000.0,
    isp_s=isp,
    dv_direct_m_s=dv_direct,
    dv_leo_hub_m_s=dv_hub,
    dv_hub_destination_m_s=dv_downstream,
    tanker_dry_to_cargo_ratio=tanker_ratio,
    boil_off_fraction_per_day=boil_pct / 100.0,
    storage_days=storage_days,
)

financial = FinancialInputs(
    launch_cost_per_kg=launch_price,
    customer_discount_fraction=discount_pct / 100.0,
    reserve_fraction_of_revenue=reserve_pct / 100.0,
    cargo_vehicle_cost_per_mission=vehicle_cost_m * 1e6,
    direct_operations_cost_per_mission=direct_ops_m * 1e6,
    hub_operations_cost_per_mission=hub_ops_m * 1e6,
    depot_propellant_cost_per_kg=propellant_price,
    tanker_service_cost_per_kg_delivered=tanker_service_price,
    module_storage_capacity_kg=module_tons * 1000.0,
    inventory_peak_factor=inventory_peak,
    module_capex=module_capex_m * 1e6,
    shared_setup_capex=setup_capex_m * 1e6,
    annual_opex_per_module=module_opex_m * 1e6,
    annual_shared_opex=shared_opex_m * 1e6,
    project_years=project_years,
    discount_rate=discount_rate_pct / 100.0,
)

try:
    validate_inputs(mission, financial)
    architecture = calculate_architectures(mission)
    economics = calculate_economics(
        mission, financial, architecture
    )
except (ValueError, OverflowError, ZeroDivisionError) as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# 8. TABS
# ============================================================

tab_dashboard, tab_mass, tab_network, tab_risk, tab_assumptions = (
    st.tabs([
        "Dashboard",
        "Mass and storage",
        "Network schematic",
        "Uncertainty",
        "Assumptions and export",
    ])
)


# ============================================================
# 9. DASHBOARD
# ============================================================

with tab_dashboard:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Hub launch-mass savings",
        f"{architecture['mass_savings_pct']:.1f}%",
    )
    c2.metric(
        "Annual operating surplus",
        money(economics["annual_surplus"]),
    )
    c3.metric(
        "Upfront infrastructure CAPEX",
        money(economics["capex"]),
    )
    c4.metric(
        "Simple payback",
        (
            f"{economics['payback_years']:.1f} years"
            if economics["payback_years"] is not None
            else "No positive payback"
        ),
    )

    st.caption(
        "Negative mass savings means the hub architecture "
        "requires more total mass launched to LEO. "
        "Operating surplus is not accounting net profit."
    )

    if architecture["mass_savings_pct"] < 0:
        st.info(
            "Under these inputs, the hub route increases launched "
            "mass after depot fuel delivery is included."
        )

    c5, c6, c7 = st.columns(3)
    c5.metric(
        f"{financial.project_years}-year project NPV proxy",
        money(economics["npv"]),
    )
    c6.metric(
        "Cargo missions per year",
        int(architecture["missions"]),
    )
    c7.metric(
        "Storage modules",
        int(economics["modules"]),
    )

    waterfall = go.Figure(
        go.Waterfall(
            measure=[
                "absolute",
                "relative",
                "relative",
                "relative",
                "total",
            ],
            x=[
                "Revenue",
                "LEO launch",
                "Other operating costs",
                "Reserve",
                "Operating surplus",
            ],
            y=[
                economics["revenue"] / 1e6,
                -economics["hub_launch_cost"] / 1e6,
                -economics["nonlaunch_cost"] / 1e6,
                -economics["reserve"] / 1e6,
                0,
            ],
            increasing={"marker": {"color": "#3366FF"}},
            decreasing={"marker": {"color": "#F15B6C"}},
            totals={"marker": {"color": "#27C4A8"}},
            connector={"line": {"color": "#576579"}},
        )
    )
    waterfall.update_yaxes(title="USD millions")
    st.plotly_chart(
        style_figure(
            waterfall,
            "Annual cash operating bridge",
        ),
        use_container_width=True,
    )

    cost_table = pd.DataFrame({
        "Item": [
            "Modeled direct reference cost",
            "Customer revenue",
            "Hub architecture LEO launch",
            "Cargo vehicle procurement",
            "Hub mission operations",
            "Depot propellant procurement",
            "Tanker hardware/service",
            "Fixed infrastructure OPEX",
            "Annual reserve",
            "Annual operating surplus",
            "Break-even annual revenue",
        ],
        "USD millions": [
            economics["direct_reference_cost"] / 1e6,
            economics["revenue"] / 1e6,
            economics["hub_launch_cost"] / 1e6,
            economics["hub_vehicle_cost"] / 1e6,
            economics["hub_mission_operations"] / 1e6,
            economics["depot_propellant_cost"] / 1e6,
            economics["tanker_service_cost"] / 1e6,
            economics["fixed_opex"] / 1e6,
            economics["reserve"] / 1e6,
            economics["annual_surplus"] / 1e6,
            economics["break_even_revenue"] / 1e6,
        ],
    })
    st.dataframe(
        cost_table,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# 10. MASS AND STORAGE
# ============================================================

with tab_mass:
    st.subheader("Annual launched-mass accounting")

    mass_rows = [
        {
            "Architecture": "Direct",
            "Component": "Payload",
            "Tonnes": architecture["annual_payload_kg"] / 1000,
        },
        {
            "Architecture": "Direct",
            "Component": "Cargo vehicle dry mass",
            "Tonnes": (
                architecture["annual_cargo_dry_mass_kg"] / 1000
            ),
        },
        {
            "Architecture": "Direct",
            "Component": "Direct propulsion fuel",
            "Tonnes": (
                architecture["direct_propellant_kg"] / 1000
            ),
        },
        {
            "Architecture": "Hub",
            "Component": "Payload",
            "Tonnes": architecture["annual_payload_kg"] / 1000,
        },
        {
            "Architecture": "Hub",
            "Component": "Cargo vehicle dry mass",
            "Tonnes": (
                architecture["annual_cargo_dry_mass_kg"] / 1000
            ),
        },
        {
            "Architecture": "Hub",
            "Component": "Cargo vehicle upstream fuel",
            "Tonnes": (
                architecture["hub_upstream_propellant_kg"] / 1000
            ),
        },
        {
            "Architecture": "Hub",
            "Component": "Downstream fuel",
            "Tonnes": (
                architecture["hub_downstream_propellant_kg"]
                / 1000
            ),
        },
        {
            "Architecture": "Hub",
            "Component": "Storage boil-off",
            "Tonnes": architecture["hub_boil_off_kg"] / 1000,
        },
        {
            "Architecture": "Hub",
            "Component": "Tanker dry mass",
            "Tonnes": (
                architecture["hub_tanker_dry_mass_kg"] / 1000
            ),
        },
        {
            "Architecture": "Hub",
            "Component": "Tanker propulsion fuel",
            "Tonnes": (
                architecture["hub_tanker_propellant_kg"] / 1000
            ),
        },
    ]

    mass_df = pd.DataFrame(mass_rows)

    fig_mass = px.bar(
        mass_df,
        x="Architecture",
        y="Tonnes",
        color="Component",
        barmode="stack",
    )
    st.plotly_chart(
        style_figure(
            fig_mass,
            "Mass launched to LEO, including depot fuel supply",
        ),
        use_container_width=True,
    )
    st.caption(
        "Infrastructure deployment mass is not included in these "
        "annual totals. Its deployment is assumed to be included "
        "in the user-entered installed CAPEX."
    )

    left, right = st.columns(2)

    with left:
        st.metric(
            "Annual depot fuel delivered",
            f"{architecture['hub_fuel_delivered_kg'] / 1000:,.2f} t",
        )
        st.metric(
            "Annual storage loss",
            f"{architecture['hub_boil_off_kg'] / 1000:,.2f} t",
        )
        st.metric(
            "Fuel retained after storage",
            f"{architecture['retention'] * 100:.2f}%",
        )

    with right:
        st.metric(
            "Average resident inventory",
            f"{economics['average_inventory_kg'] / 1000:,.2f} t",
        )
        st.metric(
            "Design inventory",
            f"{economics['design_inventory_kg'] / 1000:,.2f} t",
        )
        st.metric(
            "Installed storage capacity",
            (
                f"{economics['modules'] * "
                f"financial.module_storage_capacity_kg / 1000:,.2f} t"
            ),
        )

    st.caption(
        "Inventory sizing uses the larger of average resident "
        "inventory and one fuel batch, multiplied by the peak "
        "factor. This is not a discrete-event scheduling model."
    )

    st.subheader("Storage-duration sensitivity")

    sensitivity_rows = []
    for day in range(0, 91, 3):
        varied_mission = replace(mission, storage_days=day)
        varied_architecture = calculate_architectures(
            varied_mission
        )
        varied_economics = calculate_economics(
            varied_mission,
            financial,
            varied_architecture,
        )
        sensitivity_rows.append({
            "Storage days": day,
            "Annual surplus USD millions": (
                varied_economics["annual_surplus"] / 1e6
            ),
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    fig_sensitivity = px.line(
        sensitivity_df,
        x="Storage days",
        y="Annual surplus USD millions",
        markers=True,
    )
    fig_sensitivity.add_hline(
        y=0,
        line_dash="dash",
        line_color="#FFFFFF",
    )
    st.plotly_chart(
        style_figure(
            fig_sensitivity,
            "Storage duration versus modeled operating surplus",
        ),
        use_container_width=True,
    )


# ============================================================
# 11. NETWORK SCHEMATIC
# ============================================================

with tab_network:
    st.subheader("Illustrative logistics network")
    st.caption(
        "Not to scale. No live telemetry, propagated orbits, "
        "flight times, or physical transfer trajectories."
    )

    positions = {
        "LEO": (0.0, 0.0),
        "L1 hub": (1.0, 0.7),
        "Destination orbit": (2.0, 0.0),
    }

    fig_network = go.Figure()

    def add_edge(start, end, name, color, dash):
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        fig_network.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=4, dash=dash),
                name=name,
                hoverinfo="name",
            )
        )

    add_edge(
        "LEO",
        "Destination orbit",
        "Direct cargo route",
        "#8A96A8",
        "dash",
    )
    add_edge(
        "LEO",
        "L1 hub",
        "Cargo vehicle + separate fuel supply",
        "#3366FF",
        "solid",
    )
    add_edge(
        "L1 hub",
        "Destination orbit",
        "Refueled downstream cargo leg",
        "#27C4A8",
        "solid",
    )

    fig_network.add_trace(
        go.Scatter(
            x=[p[0] for p in positions.values()],
            y=[p[1] for p in positions.values()],
            mode="markers+text",
            text=list(positions.keys()),
            textposition="top center",
            marker=dict(
                size=[24, 20, 24],
                color=["#5CB8FF", "#3366FF", "#C5CBD4"],
            ),
            showlegend=False,
            hoverinfo="text",
        )
    )

    fig_network.update_xaxes(
        visible=False,
        range=[-0.3, 2.3],
    )
    fig_network.update_yaxes(
        visible=False,
        range=[-0.35, 1.05],
    )
    fig_network.update_layout(height=450)

    st.plotly_chart(
        style_figure(fig_network),
        use_container_width=True,
    )

    st.info(
        "Both cargo vehicles and tankers are expendable in this "
        "version. Reuse requires explicit return burns, residual "
        "propellant, refurbishment, and vehicle-cycle modeling."
    )


# ============================================================
# 12. MONTE CARLO
# ============================================================

with tab_risk:
    st.subheader("Economic uncertainty—not mission reliability")

    st.write(
        "This simulation varies launch price, nonlaunch operating "
        "costs, and infrastructure CAPEX. It does not simulate "
        "launch failures, docking failures, demand shortfalls, "
        "schedule delays, or technical reliability."
    )

    r1, r2, r3 = st.columns(3)

    launch_cv_pct = r1.slider(
        "Launch price coefficient of variation (%)",
        0, 60, 15,
    )
    opex_cv_pct = r2.slider(
        "Nonlaunch cost coefficient of variation (%)",
        0, 60, 20,
    )
    capex_cv_pct = r3.slider(
        "CAPEX coefficient of variation (%)",
        0, 60, 20,
    )

    r4, r5, r6 = st.columns(3)

    pricing_mode = r4.selectbox(
        "Customer pricing",
        [
            "Fixed customer price",
            "Indexed to simulated direct reference",
        ],
    )
    draws = r5.selectbox(
        "Simulation draws",
        [1000, 10000, 50000],
        index=1,
    )
    seed = int(
        r6.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )
    )

    st.caption(
        "Positive-only, mean-one lognormal multipliers are used. "
        "The three cost multipliers are independent. Each draw "
        "represents one constant project operating regime."
    )

    simulation_signature = (
        mission,
        financial,
        draws,
        seed,
        launch_cv_pct,
        opex_cv_pct,
        capex_cv_pct,
        pricing_mode,
    )

    if st.button("Run uncertainty simulation", type="primary"):
        with st.spinner("Running simulations..."):
            simulation = simulate_economics(
                architecture,
                economics,
                financial,
                draws=int(draws),
                seed=seed,
                launch_cv=launch_cv_pct / 100.0,
                opex_cv=opex_cv_pct / 100.0,
                capex_cv=capex_cv_pct / 100.0,
                pricing_mode=pricing_mode,
            )

        st.session_state["simulation_result"] = simulation
        st.session_state["simulation_signature"] = (
            simulation_signature
        )

    saved_simulation = st.session_state.get(
        "simulation_result"
    )
    saved_signature = st.session_state.get(
        "simulation_signature"
    )

    if (
        saved_simulation is not None
        and saved_signature == simulation_signature
    ):
        simulation = saved_simulation
        surplus = simulation["Annual surplus USD"]

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Probability of positive surplus",
            f"{100 * (surplus > 0).mean():.1f}%",
        )
        s2.metric(
            "P5 annual surplus",
            money(float(surplus.quantile(0.05))),
        )
        s3.metric(
            "Median annual surplus",
            money(float(surplus.median())),
        )
        s4.metric(
            "P95 annual surplus",
            money(float(surplus.quantile(0.95))),
        )

        st.metric(
            "Probability of positive project NPV",
            (
                f"{100 * "
                f"(simulation['Project NPV USD'] > 0).mean():.1f}%"
            ),
        )

        hist_data = pd.DataFrame({
            "Annual surplus USD millions": surplus / 1e6
        })

        fig_hist = px.histogram(
            hist_data,
            x="Annual surplus USD millions",
            nbins=70,
            color_discrete_sequence=["#3366FF"],
        )
        fig_hist.add_vline(
            x=0,
            line_dash="dash",
            line_color="#FFFFFF",
        )

        st.plotly_chart(
            style_figure(
                fig_hist,
                "Simulated annual operating surplus",
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download simulation CSV",
            data=simulation.to_csv(index=False).encode("utf-8"),
            file_name="astraia_uncertainty.csv",
            mime="text/csv",
        )

    elif saved_simulation is not None:
        st.info(
            "Inputs changed. Run the simulation again to refresh "
            "the results."
        )


# ============================================================
# 13. ASSUMPTIONS AND EXPORT
# ============================================================

with tab_assumptions:
    st.subheader("Model boundaries")

    st.markdown(
        """
        - Origin: a common LEO staging point.
        - Destination: the same user-specified lunar orbit.
        - No lunar landing or ascent is modeled.
        - Delta-v inputs are assumptions, not calculated trajectories.
        - Cargo vehicles and tankers are expendable.
        - Downstream fuel is delivered separately to the hub.
        - All propulsion uses the same specific impulse.
        - Tanker hardware scales linearly with delivered fuel.
        - Tanker flight-count and tank-capacity constraints are omitted.
        - Cargo missions are evenly loaded up to capacity.
        - Demand is fully served and fully paid.
        - Inventory sizing is a screening approximation.
        - Installed CAPEX must include depot deployment and commissioning.
        - Annual operating surplus excludes taxes and financing.
        - The reserve is withheld each year with no later release modeled.
        - Project NPV uses constant annual surplus with no terminal value.
        - No infrastructure replacement or vehicle return is modeled.
        - No technical failure probabilities are modeled.
        - No flight-ready mission-design claim is made.
        """
    )

    st.subheader("Recommended validation work")

    st.markdown(
        """
        1. Validate common-endpoint trajectories and reference frames.
        2. Replace dry-mass assumptions with spacecraft designs.
        3. Add tank capacity, mixture ratio, residuals, and transfer losses.
        4. Add discrete tanker flights and delivery scheduling.
        5. Add reusable vehicle return legs if reuse is proposed.
        6. Validate installed infrastructure and service costs.
        7. Add demand contracts, utilization, and delivery penalties.
        8. Add reliability, downtime, and liquidity modeling.
        9. Replace constant annual cash flow with a project timeline.
        """
    )

    export_rows = []

    for name, value in architecture.items():
        export_rows.append({
            "Category": "Architecture",
            "Metric": name,
            "Value": value,
        })

    for name, value in economics.items():
        export_rows.append({
            "Category": "Economics",
            "Metric": name,
            "Value": value,
        })

    export_df = pd.DataFrame(export_rows)

    st.download_button(
        "Download model results CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="astraia_model_results.csv",
        mime="text/csv",
    )

    st.dataframe(
        export_df,
        hide_index=True,
        use_container_width=True,
    )
