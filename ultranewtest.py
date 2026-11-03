import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Astraia Dynamics | Logistics Screening",
    layout="wide",
)

G0 = 9.80665
DAYS_PER_YEAR = 365.0


# ============================================================
# 2. INPUT DATA MODELS
# ============================================================

@dataclass(frozen=True)
class MissionInputs:
    annual_demand_kg: float
    mission_capacity_kg: float
    vehicle_dry_mass_kg: float
    isp_s: float
    dv_direct_m_s: float
    dv_leo_hub_m_s: float
    dv_hub_destination_m_s: float
    tanker_dry_to_cargo_ratio: float
    boil_off_fraction_per_day: float
    storage_days: float


@dataclass(frozen=True)
class FinancialInputs:
    launch_cost_per_kg: float
    customer_discount_fraction: float
    reserve_fraction_of_revenue: float
    cargo_vehicle_cost_per_mission: float
    direct_operations_cost_per_mission: float
    hub_operations_cost_per_mission: float
    depot_propellant_cost_per_kg: float
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
# 3. VALIDATION AND PHYSICS
# ============================================================

def validate_inputs(m, f):
    positive_values = {
        "Annual demand": m.annual_demand_kg,
        "Mission capacity": m.mission_capacity_kg,
        "Vehicle dry mass": m.vehicle_dry_mass_kg,
        "Specific impulse": m.isp_s,
        "Launch price": f.launch_cost_per_kg,
        "Module capacity": f.module_storage_capacity_kg,
    }

    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    if not 0 <= m.boil_off_fraction_per_day < 1:
        raise ValueError("Boil-off fraction must be between 0 and 1.")

    if not 0 <= f.customer_discount_fraction < 1:
        raise ValueError("Customer discount must be between 0 and 1.")

    if not 0 <= f.reserve_fraction_of_revenue < 1:
        raise ValueError("Reserve fraction must be between 0 and 1.")

    if f.inventory_peak_factor < 1:
        raise ValueError("Inventory peak factor must be at least 1.")

    if f.project_years < 1:
        raise ValueError("Project duration must be at least one year.")

    nonnegative_values = {
        "Storage duration": m.storage_days,
        "Tanker dry-to-cargo ratio": m.tanker_dry_to_cargo_ratio,
        "Direct delta-v": m.dv_direct_m_s,
        "LEO-to-hub delta-v": m.dv_leo_hub_m_s,
        "Hub-to-destination delta-v": m.dv_hub_destination_m_s,
        "Vehicle procurement cost": f.cargo_vehicle_cost_per_mission,
        "Direct operations cost": f.direct_operations_cost_per_mission,
        "Hub operations cost": f.hub_operations_cost_per_mission,
        "Depot propellant cost": f.depot_propellant_cost_per_kg,
        "Tanker service cost": f.tanker_service_cost_per_kg_delivered,
        "Module CAPEX": f.module_capex,
        "Shared CAPEX": f.shared_setup_capex,
        "Module OPEX": f.annual_opex_per_module,
        "Shared OPEX": f.annual_shared_opex,
        "Discount rate": f.discount_rate,
    }

    for name, value in nonnegative_values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be finite and nonnegative."
            )


def burn_budget(final_mass_kg, delta_v_m_s, isp_s):
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
    }


def calculate_architectures(m):
    """
    Common endpoint: the same user-specified lunar orbit.

    Direct:
        LEO -> destination.

    Hub:
        Cargo vehicle: LEO -> hub -> destination.
        Separate tanker: LEO -> hub with downstream fuel.

    Cargo vehicles and tankers are expendable.
    Tanker hardware scales linearly with fuel cargo.
    No vehicle return, flight schedule, or landing is modeled.
    """
    missions = math.ceil(
        m.annual_demand_kg / m.mission_capacity_kg
    )
    payload_per_mission = m.annual_demand_kg / missions
    final_cargo_mass = (
        payload_per_mission + m.vehicle_dry_mass_kg
    )

    direct = burn_budget(
        final_cargo_mass,
        m.dv_direct_m_s,
        m.isp_s,
    )
    upstream = burn_budget(
        final_cargo_mass,
        m.dv_leo_hub_m_s,
        m.isp_s,
    )
    downstream = burn_budget(
        final_cargo_mass,
        m.dv_hub_destination_m_s,
        m.isp_s,
    )

    retention = (
        1.0 - m.boil_off_fraction_per_day
    ) ** m.storage_days

    if retention <= 0:
        raise ValueError("Fuel retention must be positive.")

    downstream_fuel = downstream["propellant_kg"]
    fuel_loaded = downstream_fuel / retention
    boil_off = fuel_loaded - downstream_fuel

    tanker_dry_mass = (
        fuel_loaded * m.tanker_dry_to_cargo_ratio
    )

    if fuel_loaded > 0:
        tanker = burn_budget(
            fuel_loaded + tanker_dry_mass,
            m.dv_leo_hub_m_s,
            m.isp_s,
        )
    else:
        tanker = {
            "initial_mass_kg": 0.0,
            "propellant_kg": 0.0,
        }

    direct_mass = direct["initial_mass_kg"] * missions
    hub_cargo_mass = upstream["initial_mass_kg"] * missions
    hub_tanker_mass = tanker["initial_mass_kg"] * missions
    hub_total_mass = hub_cargo_mass + hub_tanker_mass

    result = {
        "missions": missions,
        "payload_per_mission_kg": payload_per_mission,
        "annual_payload_kg": m.annual_demand_kg,
        "annual_cargo_dry_mass_kg": (
            m.vehicle_dry_mass_kg * missions
        ),
        "direct_initial_mass_kg": direct_mass,
        "direct_propellant_kg": (
            direct["propellant_kg"] * missions
        ),
        "hub_initial_mass_kg": hub_total_mass,
        "hub_cargo_initial_mass_kg": hub_cargo_mass,
        "hub_tanker_initial_mass_kg": hub_tanker_mass,
        "hub_upstream_propellant_kg": (
            upstream["propellant_kg"] * missions
        ),
        "hub_downstream_propellant_kg": (
            downstream_fuel * missions
        ),
        "hub_boil_off_kg": boil_off * missions,
        "hub_fuel_delivered_kg": fuel_loaded * missions,
        "hub_tanker_dry_mass_kg": tanker_dry_mass * missions,
        "hub_tanker_propellant_kg": (
            tanker["propellant_kg"] * missions
        ),
        "hub_fuel_batch_kg": fuel_loaded,
        "retention": retention,
        "mass_savings_pct": (
            100.0 * (1.0 - hub_total_mass / direct_mass)
        ),
    }

    direct_check = (
        result["annual_payload_kg"]
        + result["annual_cargo_dry_mass_kg"]
        + result["direct_propellant_kg"]
    )

    hub_check = (
        result["annual_payload_kg"]
        + result["annual_cargo_dry_mass_kg"]
        + result["hub_upstream_propellant_kg"]
        + result["hub_downstream_propellant_kg"]
        + result["hub_boil_off_kg"]
        + result["hub_tanker_dry_mass_kg"]
        + result["hub_tanker_propellant_kg"]
    )

    if not math.isclose(
        direct_check,
        direct_mass,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("Direct mass balance failed.")

    if not math.isclose(
        hub_check,
        hub_total_mass,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("Hub mass balance failed.")

    return result


# ============================================================
# 4. ECONOMICS
# ============================================================

def calculate_economics(m, f, a):
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
    direct_ops_cost = (
        a["missions"] * f.direct_operations_cost_per_mission
    )

    direct_reference_cost = (
        direct_launch_cost
        + direct_vehicle_cost
        + direct_ops_cost
    )

    revenue = (
        direct_reference_cost
        * (1.0 - f.customer_discount_fraction)
    )

    hub_launch_cost = (
        a["hub_initial_mass_kg"] * f.launch_cost_per_kg
    )
    vehicle_cost = (
        a["missions"] * f.cargo_vehicle_cost_per_mission
    )
    mission_operations = (
        a["missions"] * f.hub_operations_cost_per_mission
    )
    propellant_cost = (
        a["hub_fuel_delivered_kg"]
        * f.depot_propellant_cost_per_kg
    )
    tanker_cost = (
        a["hub_fuel_delivered_kg"]
        * f.tanker_service_cost_per_kg_delivered
    )
    fixed_opex = (
        modules * f.annual_opex_per_module
        + f.annual_shared_opex
    )

    nonlaunch_cost = (
        vehicle_cost
        + mission_operations
        + propellant_cost
        + tanker_cost
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
    annuity_factor = float(
        np.sum(
            1.0 / (1.0 + f.discount_rate) ** years
        )
    )
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
        "hub_vehicle_cost": vehicle_cost,
        "hub_mission_operations": mission_operations,
        "depot_propellant_cost": propellant_cost,
        "tanker_service_cost": tanker_cost,
        "fixed_opex": fixed_opex,
        "nonlaunch_cost": nonlaunch_cost,
        "reserve": reserve,
        "annual_surplus": annual_surplus,
        "surplus_margin_pct": (
            annual_surplus / revenue * 100.0
            if revenue > 0 else 0.0
        ),
        "payback_years": payback,
        "npv": npv,
        "annuity_factor": annuity_factor,
        "break_even_revenue": break_even_revenue,
    }


# ============================================================
# 5. UNCERTAINTY MODEL
# ============================================================

def positive_multiplier(rng, cv, size):
    sigma = math.sqrt(math.log(1.0 + cv ** 2))
    mu = -0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size)


def simulate_economics(
    a,
    e,
    f,
    draws,
    seed,
    launch_cv,
    opex_cv,
    capex_cv,
    pricing_mode,
):
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

    launch_price = (
        f.launch_cost_per_kg * launch_multiplier
    )

    if pricing_mode == "Fixed customer price":
        revenue = np.full(draws, e["revenue"])
    else:
        reference_nonlaunch = (
            e["direct_reference_cost"]
            - e["direct_launch_cost"]
        )
        revenue = (
            a["direct_initial_mass_kg"] * launch_price
            + reference_nonlaunch
        ) * (1.0 - f.customer_discount_fraction)

    launch_cost = (
        a["hub_initial_mass_kg"] * launch_price
    )
    other_cost = e["nonlaunch_cost"] * opex_multiplier
    reserve = revenue * f.reserve_fraction_of_revenue

    surplus = (
        revenue - launch_cost - other_cost - reserve
    )

    simulated_capex = e["capex"] * capex_multiplier
    project_npv = (
        -simulated_capex
        + surplus * e["annuity_factor"]
    )

    return pd.DataFrame({
        "Annual surplus USD": surplus,
        "Project NPV USD": project_npv,
        "Revenue USD": revenue,
        "Launch price USD per kg": launch_price,
        "CAPEX USD": simulated_capex,
    })


# ============================================================
# 6. PRESENTATION
# ============================================================

def style_figure(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#BAC3D1"),
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
    "Cislunar logistics screening | Mass budgets | "
    "Scenario economics | Uncertainty"
)

st.warning(
    "Conceptual screening model—not flight-design software "
    "or a validated investment case. Inputs require independent "
    "engineering and commercial validation."
)


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Scenario assumptions")

    st.caption(
        "Both architectures deliver cargo from LEO "
        "to the same user-specified lunar orbit."
    )

    with st.expander(
        "Demand and spacecraft",
        expanded=True,
    ):
        annual_tons = st.number_input(
            "Annual cargo (tonnes/year)",
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
            "Cargo vehicle dry mass (tonnes)",
            min_value=0.01,
            value=2.0,
            step=0.25,
        )
        isp = st.number_input(
            "Specific impulse (seconds)",
            min_value=100.0,
            value=450.0,
            step=10.0,
        )
        tanker_ratio = st.number_input(
            "Tanker dry mass / delivered fuel",
            min_value=0.0,
            value=0.10,
            step=0.02,
        )

    with st.expander(
        "Route delta-v",
        expanded=True,
    ):
        st.caption(
            "Illustrative assumptions; no lunar landing."
        )
        dv_direct = st.number_input(
            "LEO to destination (m/s)",
            min_value=0.0,
            value=4100.0,
            step=100.0,
        )
        dv_hub = st.number_input(
            "LEO to hub (m/s)",
            min_value=0.0,
            value=3800.0,
            step=100.0,
        )
        dv_downstream = st.number_input(
            "Hub to destination (m/s)",
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
            "Storage per module (tonnes)",
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
            "Discount to direct reference (%)",
            0.0, 50.0, 20.0, 1.0,
        )
        reserve_pct = st.slider(
            "Annual revenue reserve (%)",
            0.0, 30.0, 10.0, 1.0,
        )
        vehicle_cost_m = st.number_input(
            "Cargo vehicle cost ($M/mission)",
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
            "Hub operations ($M/mission)",
            min_value=0.0,
            value=3.0,
            step=0.5,
        )
        propellant_price = st.number_input(
            "Depot propellant procurement ($/kg)",
            min_value=0.0,
            value=10.0,
            step=5.0,
        )
        tanker_service_price = st.number_input(
            "Tanker service excluding launch ($/kg delivered)",
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
            "Annual module OPEX ($M)",
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
            "Project duration (years)",
            1, 30, 10,
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
        mission,
        financial,
        architecture,
    )
except (ValueError, OverflowError, ZeroDivisionError) as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# 8. TABS
# ============================================================

tabs = st.tabs([
    "Dashboard",
    "Mass and storage",
    "Network schematic",
    "Uncertainty",
    "Assumptions and export",
])

tab_dashboard, tab_mass, tab_network, tab_risk, tab_assumptions = tabs


# ============================================================
# 9. DASHBOARD
# ============================================================

with tab_dashboard:
    c1, c2, c3, c4 = st.columns(4)

    mass_savings = architecture["mass_savings_pct"]
    surplus = economics["annual_surplus"]
    payback = economics["payback_years"]

    c1.metric(
        "Hub launch-mass savings",
        f"{mass_savings:.1f}%",
    )
    c2.metric(
        "Annual operating surplus",
        money(surplus),
    )
    c3.metric(
        "Infrastructure CAPEX",
        money(economics["capex"]),
    )

    payback_text = (
        f"{payback:.1f} years"
        if payback is not None
        else "No positive payback"
    )
    c4.metric("Simple payback", payback_text)

    st.caption(
        "Negative mass savings means the hub architecture "
        "requires more total mass launched to LEO. "
        "Operating surplus is not accounting net profit."
    )

    if mass_savings < 0:
        st.info(
            "Under these assumptions, depot fuel delivery "
            "makes the hub architecture heavier than direct delivery."
        )

    c5, c6, c7 = st.columns(3)

    npv_label = (
        f"{financial.project_years}-year project NPV proxy"
    )
    c5.metric(npv_label, money(economics["npv"]))
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
                0.0,
            ],
            increasing={
                "marker": {"color": "#3366FF"}
            },
            decreasing={
                "marker": {"color": "#F15B6C"}
            },
            totals={
                "marker": {"color": "#27C4A8"}
            },
            connector={
                "line": {"color": "#576579"}
            },
        )
    )
    waterfall.update_yaxes(title="USD millions")

    st.plotly_chart(
        style_figure(
            waterfall,
            "Annual operating cash bridge",
        ),
        use_container_width=True,
    )

    items = [
        ("Modeled direct reference cost", "direct_reference_cost"),
        ("Customer revenue", "revenue"),
        ("Hub LEO launch cost", "hub_launch_cost"),
        ("Cargo vehicle procurement", "hub_vehicle_cost"),
        ("Hub mission operations", "hub_mission_operations"),
        ("Depot propellant procurement", "depot_propellant_cost"),
        ("Tanker hardware/service", "tanker_service_cost"),
        ("Fixed infrastructure OPEX", "fixed_opex"),
        ("Annual reserve", "reserve"),
        ("Annual operating surplus", "annual_surplus"),
        ("Break-even annual revenue", "break_even_revenue"),
    ]

    cost_table = pd.DataFrame([
        {
            "Item": label,
            "USD millions": economics[key] / 1e6,
        }
        for label, key in items
    ])

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

    component_definitions = [
        ("Direct", "Payload", "annual_payload_kg"),
        (
            "Direct",
            "Cargo vehicle dry mass",
            "annual_cargo_dry_mass_kg",
        ),
        (
            "Direct",
            "Direct propulsion fuel",
            "direct_propellant_kg",
        ),
        ("Hub", "Payload", "annual_payload_kg"),
        (
            "Hub",
            "Cargo vehicle dry mass",
            "annual_cargo_dry_mass_kg",
        ),
        (
            "Hub",
            "Cargo vehicle upstream fuel",
            "hub_upstream_propellant_kg",
        ),
        (
            "Hub",
            "Downstream fuel",
            "hub_downstream_propellant_kg",
        ),
        ("Hub", "Storage boil-off", "hub_boil_off_kg"),
        (
            "Hub",
            "Tanker dry mass",
            "hub_tanker_dry_mass_kg",
        ),
        (
            "Hub",
            "Tanker propulsion fuel",
            "hub_tanker_propellant_kg",
        ),
    ]

    mass_df = pd.DataFrame([
        {
            "Architecture": route,
            "Component": component,
            "Tonnes": architecture[key] / 1000.0,
        }
        for route, component, key in component_definitions
    ])

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
        "Depot deployment mass is excluded from annual totals. "
        "Deployment and commissioning must be included in "
        "installed infrastructure CAPEX."
    )

    left, right = st.columns(2)

    annual_fuel_t = (
        architecture["hub_fuel_delivered_kg"] / 1000.0
    )
    annual_loss_t = (
        architecture["hub_boil_off_kg"] / 1000.0
    )
    retention_pct = architecture["retention"] * 100.0

    average_inventory_t = (
        economics["average_inventory_kg"] / 1000.0
    )
    design_inventory_t = (
        economics["design_inventory_kg"] / 1000.0
    )

    # Calculate the complete expression before formatting.
    installed_capacity_t = (
        economics["modules"]
        * financial.module_storage_capacity_kg
        / 1000.0
    )

    with left:
        st.metric(
            "Annual depot fuel delivered",
            f"{annual_fuel_t:,.2f} t",
        )
        st.metric(
            "Annual storage loss",
            f"{annual_loss_t:,.2f} t",
        )
        st.metric(
            "Fuel retained after storage",
            f"{retention_pct:.2f}%",
        )

    with right:
        st.metric(
            "Average resident inventory",
            f"{average_inventory_t:,.2f} t",
        )
        st.metric(
            "Design inventory",
            f"{design_inventory_t:,.2f} t",
        )
        st.metric(
            "Installed storage capacity",
            f"{installed_capacity_t:,.2f} t",
        )

    st.caption(
        "Storage sizing uses the larger of average inventory "
        "and one mission fuel batch, multiplied by the peak "
        "factor. This is not a discrete-event scheduling model."
    )

    st.subheader("Storage-duration sensitivity")

    sensitivity_rows = []

    for day in range(0, 91, 3):
        varied_mission = replace(
            mission,
            storage_days=day,
        )
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
            "Storage duration versus operating surplus",
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
        "flight times, or calculated transfer trajectories."
    )

    positions = {
        "LEO": (0.0, 0.0),
        "L1 hub": (1.0, 0.7),
        "Destination orbit": (2.0, 0.0),
    }

    fig_network = go.Figure()

    edges = [
        (
            "LEO",
            "Destination orbit",
            "Direct cargo route",
            "#8A96A8",
            "dash",
        ),
        (
            "LEO",
            "L1 hub",
            "Cargo + separate fuel supply",
            "#3366FF",
            "solid",
        ),
        (
            "L1 hub",
            "Destination orbit",
            "Refueled downstream cargo",
            "#27C4A8",
            "solid",
        ),
    ]

    for start, end, name, color, dash in edges:
        x0, y0 = positions[start]
        x1, y1 = positions[end]

        fig_network.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    color=color,
                    width=4,
                    dash=dash,
                ),
                name=name,
                hoverinfo="name",
            )
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
                color=[
                    "#5CB8FF",
                    "#3366FF",
                    "#C5CBD4",
                ],
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
        "Cargo vehicles and tankers are expendable in this "
        "version. Reuse requires return burns, residual fuel, "
        "refurbishment costs, and vehicle-cycle scheduling."
    )


# ============================================================
# 12. UNCERTAINTY SIMULATION
# ============================================================

with tab_risk:
    st.subheader(
        "Economic uncertainty—not mission reliability"
    )

    st.write(
        "Varies launch prices, nonlaunch operating costs, and "
        "infrastructure CAPEX. Does not simulate launch failures, "
        "docking failures, demand losses, or schedule delays."
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
        "Uses independent, positive-only lognormal cost "
        "multipliers. Each draw represents a constant project "
        "operating regime, not a year-by-year stochastic path."
    )

    simulation_signature = (
        mission,
        financial,
        int(draws),
        seed,
        launch_cv_pct,
        opex_cv_pct,
        capex_cv_pct,
        pricing_mode,
    )

    if st.button(
        "Run uncertainty simulation",
        type="primary",
    ):
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
        simulated_surplus = simulation[
            "Annual surplus USD"
        ]

        probability_positive_surplus = (
            100.0 * (simulated_surplus > 0).mean()
        )
        probability_positive_npv = (
            100.0
            * (simulation["Project NPV USD"] > 0).mean()
        )

        p5_surplus = float(
            simulated_surplus.quantile(0.05)
        )
        median_surplus = float(
            simulated_surplus.median()
        )
        p95_surplus = float(
            simulated_surplus.quantile(0.95)
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Probability of positive surplus",
            f"{probability_positive_surplus:.1f}%",
        )
        s2.metric(
            "P5 annual surplus",
            money(p5_surplus),
        )
        s3.metric(
            "Median annual surplus",
            money(median_surplus),
        )
        s4.metric(
            "P95 annual surplus",
            money(p95_surplus),
        )

        # Complete expression is evaluated before formatting.
        st.metric(
            "Probability of positive project NPV",
            f"{probability_positive_npv:.1f}%",
        )

        histogram_data = pd.DataFrame({
            "Annual surplus USD millions": (
                simulated_surplus / 1e6
            )
        })

        fig_hist = px.histogram(
            histogram_data,
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
            data=simulation.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="astraia_uncertainty.csv",
            mime="text/csv",
        )

    elif saved_simulation is not None:
        st.info(
            "Inputs changed. Run the simulation again "
            "to refresh the results."
        )


# ============================================================
# 13. ASSUMPTIONS AND EXPORT
# ============================================================

with tab_assumptions:
    st.subheader("Model boundaries")

    st.markdown(
        """
        - Both routes start at the same LEO staging point.
        - Both routes terminate at the same specified lunar orbit.
        - No lunar landing or ascent is modeled.
        - Delta-v inputs are assumptions, not calculated trajectories.
        - Cargo vehicles and tankers are expendable.
        - Downstream fuel is delivered separately to the hub.
        - The same specific impulse is used for all burns.
        - Tanker hardware scales linearly with delivered fuel.
        - Tanker capacity and discrete tanker flight counts are omitted.
        - Annual cargo demand is divided among evenly loaded missions.
        - Demand is fully served and fully paid.
        - Storage sizing is a screening approximation.
        - Installed CAPEX must include deployment and commissioning.
        - Launch cost is a linear mass-based proxy.
        - Depot cargo fuel is priced separately as a commodity.
        - Other propulsion fuel procurement is assumed bundled in
          vehicle/service proxies rather than itemized separately.
        - Operating surplus excludes taxes and financing.
        - The annual reserve is withheld with no later release modeled.
        - NPV uses constant annual surplus and no terminal value.
        - No infrastructure replacement or vehicle return is included.
        - No technical failure probabilities are modeled.
        """
    )

    st.subheader("Recommended validation")

    st.markdown(
        """
        1. Validate common-endpoint trajectories.
        2. Replace spacecraft mass assumptions with vehicle designs.
        3. Add tank capacities, residual fuel, and transfer losses.
        4. Model discrete tanker flights and delivery schedules.
        5. Include return legs if reusable vehicles are proposed.
        6. Validate infrastructure and service cost assumptions.
        7. Add demand contracts and delivery penalties.
        8. Model failures, downtime, delays, and liquidity.
        9. Replace constant annual cash flow with a project timeline.
        """
    )

    export_rows = []

    for name, value in vars(mission).items():
        export_rows.append({
            "Category": "Mission inputs",
            "Metric": name,
            "Value": value,
        })

    for name, value in vars(financial).items():
        export_rows.append({
            "Category": "Financial inputs",
            "Metric": name,
            "Value": value,
        })

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
        "Download inputs and results CSV",
        data=export_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="astraia_model_results.csv",
        mime="text/csv",
    )

    st.dataframe(
        export_df,
        hide_index=True,
        use_container_width=True,
    )
