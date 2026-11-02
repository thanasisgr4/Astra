import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
from dataclasses import dataclass
from typing import Tuple

# ==========================================
# 1. ASTRAIA ENGINE (Ο "Εγκέφαλος" Παραγωγής)
# ==========================================
@dataclass
class SpacecraftSpecs:
    payload_kg: float
    isp_seconds: float
    dry_mass_ratio: float = 0.10

@dataclass
class NetworkNodes:
    dv_direct_lunar: float = 5900.0  
    dv_leo_to_l1: float = 3800.0     

@dataclass
class MarketConditions:
    base_launch_cost_per_kg: float
    saas_discount_margin: float
    boil_off_rate_per_day: float
    hub_wait_time_days: int

class AstraiaRoutingEngine:
    G0 = 9.81
    def __init__(self, specs: SpacecraftSpecs, market: MarketConditions):
        self.specs = specs
        self.market = market
        self.nodes = NetworkNodes()

    def _calculate_mass_ratio(self, delta_v: float) -> float:
        return math.exp(delta_v / (self.specs.isp_seconds * self.G0))

    def calculate_direct_route(self) -> float:
        mr = self._calculate_mass_ratio(self.nodes.dv_direct_lunar)
        return self.specs.payload_kg * (1 + (mr - 1) * (1 + self.specs.dry_mass_ratio))

    def calculate_astraia_hub_route(self) -> Tuple[float, float]:
        mr = self._calculate_mass_ratio(self.nodes.dv_leo_to_l1)
        prop_needed = self.specs.payload_kg * (mr - 1)
        retention = (1 - self.market.boil_off_rate_per_day) ** self.market.hub_wait_time_days
        boil_off_loss = prop_needed * (1 - retention)
        prop_total = prop_needed + boil_off_loss
        m0_relay = self.specs.payload_kg + prop_total * (1 + self.specs.dry_mass_ratio)
        return m0_relay, boil_off_loss

# ==========================================
# 2. ENTERPRISE UI CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="Astraia Dynamics | Mission Control", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #050505; color: #E0E0E0; }
        .stSidebar { background-color: #0A0A0A; border-right: 1px solid #1F1F1F; }
        h1 { color: #FFFFFF !important; font-weight: 300 !important; letter-spacing: 2px; font-size: 2.5rem !important; }
        h2, h3 { color: #FFFFFF !important; font-weight: 400 !important; letter-spacing: 1px;}
        p, span, label { color: #8B949E !important; }
        div[data-testid="stMetric"] { background-color: #0A0A0A; border: 1px solid #1F1F1F; padding: 20px; border-radius: 4px; border-top: 2px solid #0052FF; }
        div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 2rem !important; font-weight: 300 !important; }
        div[data-testid="stMetricDelta"] { color: #8B949E !important; font-weight: 400 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #1F1F1F; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; padding-bottom: 10px; color: #8B949E; font-weight: 500; }
        .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom: 2px solid #0052FF !important; }
        hr { border-color: #1F1F1F; }
    </style>
""", unsafe_allow_html=True)

st.title("ASTRAIA DYNAMICS")
st.markdown("<span style='color:#8B949E; font-size:1.1rem; letter-spacing: 1px;'>ORBITAL LOGISTICS & ROUTING ENGINE (v8.0 - Dynamic Simulation)</span>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 3. SIDEBAR: SCENARIO PRESETS & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='color:#FFFFFF;'>SCENARIO PRESETS</h3>", unsafe_allow_html=True)
    scenario = st.selectbox("Επιλέξτε Σενάριο Αγοράς", ["Custom", "Conservative", "Baseline", "Aggressive (Artemis)"])
    
    if scenario == "Conservative":
        def_tons, def_cost, def_disc, def_wait, def_risk = 20, 7000, 15.0, 10, 20
    elif scenario == "Baseline":
        def_tons, def_cost, def_disc, def_wait, def_risk = 100, 5000, 20.0, 5, 15
    elif scenario == "Aggressive (Artemis)":
        def_tons, def_cost, def_disc, def_wait, def_risk = 350, 2000, 25.0, 2, 10
    else:
        def_tons, def_cost, def_disc, def_wait, def_risk = 100, 5000, 20.0, 5, 15

    st.markdown("---")
    st.markdown("<h3 style='color:#FFFFFF;'>OPERATIONAL CONTROLS</h3>", unsafe_allow_html=True)
    annual_tons = st.number_input("Annual Payload (Tons)", min_value=5, max_value=1000, value=def_tons, step=10)
    target_discount = st.slider("Client SaaS Discount (%)", 5.0, 40.0, def_disc, 1.0)
    base_launch_cost = st.number_input("LEO Launch Cost ($/kg)", value=def_cost, step=500)
    
    st.markdown("#### Cryogenic Variables")
    isp = st.slider("Engine Isp (sec)", 300, 500, 450, 10)
    boil_off_rate = st.slider("Boil-Off Rate (%/day)", 0.0, 5.0, 0.5, 0.1)
    wait_days = st.slider("L1 Hub Wait Time (Days)", 0, 30, def_wait)
    
    st.markdown("#### Infrastructure & Risk")
    module_capacity = st.number_input("Hub Module Capacity (Tons)", value=50)
    insurance_rate = st.slider("Risk Contingency Reserve (%)", 0, 30, def_risk, 5)

# ==========================================
# 4. EXECUTION OF THE ROUTING ENGINE
# ==========================================
specs = SpacecraftSpecs(payload_kg=annual_tons * 1000, isp_seconds=isp)
market = MarketConditions(
    base_launch_cost_per_kg=base_launch_cost, 
    saas_discount_margin=target_discount / 100.0, 
    boil_off_rate_per_day=boil_off_rate / 100.0, 
    hub_wait_time_days=wait_days
)
engine = AstraiaRoutingEngine(specs, market)

m0_direct = engine.calculate_direct_route()
m0_relay, boil_off_penalty = engine.calculate_astraia_hub_route()

mass_savings_pct = (1 - (m0_relay / m0_direct)) * 100
required_modules = math.ceil(annual_tons / module_capacity)
total_capex = (required_modules * 250_000_000) + 15_000_000 
total_opex = (required_modules * 20_000_000) + (annual_tons * 150_000)

direct_market_cost = m0_direct * base_launch_cost
astraia_revenue = direct_market_cost * (1 - (target_discount / 100.0))
astraia_launch_cost = m0_relay * base_launch_cost
risk_reserve = astraia_revenue * (insurance_rate / 100.0)
total_annual_cost = astraia_launch_cost + total_opex + risk_reserve
net_profit = astraia_revenue - total_annual_cost
margin = (net_profit / astraia_revenue) * 100 if astraia_revenue > 0 else 0
payback = total_capex / net_profit if net_profit > 0 else 0

# ==========================================
# 5. TABS WITH RICH VISUALS & ANIMATIONS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EXECUTIVE DASHBOARD", 
    "🚀 LIVE ORBITAL SIMULATION", 
    "🪐 CRYOGENIC PHYSICS", 
    "🎲 MONTE CARLO RISK"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MASS EFFICIENCY", f"{mass_savings_pct:.1f}%", f"vs Direct Launch")
    c2.metric("NET PROFIT", f"${net_profit/1e6:.1f}M", f"Margin: {margin:.1f}%")
    c3.metric("CAPEX BURN", f"${total_capex/1e6:.1f}M", f"Modules: {required_modules}")
    c4.metric("ROI PROJECTION", f"{payback:.1f} YRS" if payback > 0 else "N/A", "Risk-Adjusted")
    
    st.markdown("<br>", unsafe_allow_html=True)
    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v", measure=["relative", "relative", "relative", "relative", "total"],
        x=["Revenue (Client)", "Launch Cost", "OpEx", "Risk Reserve", "Net Profit"],
        y=[astraia_revenue/1e6, -astraia_launch_cost/1e6, -total_opex/1e6, -risk_reserve/1e6, net_profit/1e6],
        text=[f"${astraia_revenue/1e6:.0f}M", f"-${astraia_launch_cost/1e6:.0f}M", f"-${total_opex/1e6:.0f}M", f"-${risk_reserve/1e6:.0f}M", f"${net_profit/1e6:.0f}M"],
        textposition="outside", connector={"line":{"color":"#1F1F1F", "width": 2}},
        decreasing={"marker":{"color":"#333333"}}, increasing={"marker":{"color":"#E0E0E0"}}, totals={"marker":{"color":"#0052FF"}}      
    ))
    fig_waterfall.update_layout(title="<b>FINANCIAL FLOW (USD Millions)</b>", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8B949E"))
    st.plotly_chart(fig_waterfall, use_container_width=True)

with tab2:
    st.markdown("### 🌐 Dynamic Hub-and-Spoke Trajectory (Earth ➔ L1 ➔ Moon)")
    st.write("Προσομοίωση διαστημικού cross-docking. Πατήστε το **PLAY** για να δείτε την κίνηση του φορτίου και την ενδιάμεση στάση στον κόμβο.")
    
    # 3D Node Coordinates
    x_nodes = [0, 5, 10]
    y_nodes = [0, 5, 8]
    z_nodes = [0, 8, 4]
    
    fig_sim = go.Figure()
    # Trace 0: Path Lines
    fig_sim.add_trace(go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes, mode='lines', line=dict(color='#333333', width=4, dash='dot'), name='Astraia Route'))
    # Trace 1: Base Nodes (Earth, L1, Moon)
    fig_sim.add_trace(go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes, mode='text+markers', text=["Earth (LEO)", "L1 Hub", "Lunar Base"], marker=dict(size=[15, 20, 12], color=['#8B949E', '#0052FF', '#00FF9D'], opacity=0.8), name='Network Hubs'))
    # Trace 2: Payload (The moving object)
    fig_sim.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers+text', text=["Launch Ready"], textposition="top center", marker=dict(size=12, color='#FF2A6D', symbol='diamond'), name='Payload'))

    # Animation Frames Generation
    frames_data = []
    steps = 25
    
    # Leg 1: Earth to L1
    for i in range(steps + 1):
        frames_data.append({"x": [0 + (5 - 0) * (i / steps)], "y": [0 + (5 - 0) * (i / steps)], "z": [0 + (8 - 0) * (i / steps)], "label": "Transit: LEO to L1"})
    # Wait at L1 (Cross-docking Phase)
    for _ in range(15):
        frames_data.append({"x": [5], "y": [5], "z": [8], "label": "Cross-Docking / Refueling"})
    # Leg 2: L1 to Moon
    for i in range(steps + 1):
        frames_data.append({"x": [5 + (10 - 5) * (i / steps)], "y": [5 + (8 - 5) * (i / steps)], "z": [8 + (4 - 8) * (i / steps)], "label": "Transit: L1 to Lunar Base"})

    frames = [go.Frame(data=[go.Scatter3d(x=fd["x"], y=fd["y"], z=fd["z"], text=[fd["label"]])], traces=[2], name=f"f{k}") for k, fd in enumerate(frames_data)]
    fig_sim.frames = frames

    fig_sim.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8B949E"),
        scene=dict(xaxis=dict(range=[-1, 11]), yaxis=dict(range=[-1, 9]), zaxis=dict(range=[-1, 10])),
        updatemenus=[dict(type="buttons", showactive=False, y=-0.1, x=0, xanchor="left", yanchor="top",
            buttons=[dict(label="▶ PLAY SIMULATION", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True, "mode": "immediate"}])]
        )]
    )
    st.plotly_chart(fig_sim, use_container_width=True)

with tab3:
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Mass Penalty & Boil-Off Analysis")
        st.write(f"- Hub Wait Time: **{wait_days} days**")
        st.write(f"- Evaporation Loss: **{boil_off_penalty/1000:.1f} tons**")
        st.write(f"- Direct Launch Mass: **{m0_direct/1000:.1f} t**")
        st.write(f"- Astraia Hub Mass: **{m0_relay/1000:.1f} t**")
    with colB:
        dry_ratio = specs.dry_mass_ratio
        df_mass = pd.DataFrame({
            "Stage": ["Payload", "Dry Mass", "Propellant", "Boil-off Refill"],
            "Direct": [specs.payload_kg/1000, (m0_direct-specs.payload_kg)*dry_ratio/1000, (m0_direct-specs.payload_kg)*(1-dry_ratio)/1000, 0],
            "Astraia L1": [specs.payload_kg/1000, (m0_relay-specs.payload_kg-boil_off_penalty)*dry_ratio/1000, (m0_relay-specs.payload_kg-boil_off_penalty)*(1-dry_ratio)/1000, boil_off_penalty/1000]
        })
        fig_mass = px.bar(df_mass, x="Stage", y=["Direct", "Astraia L1"], barmode="group", color_discrete_sequence=["#333333", "#0052FF"])
        fig_mass.update_layout(title="<b>MASS BREAKDOWN (Tons)</b>", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8B949E"))
        st.plotly_chart(fig_mass, use_container_width=True)

with tab4:
    st.markdown("### Stochastic Risk Simulator")
    st.write("Executing 10,000 parallel iterations to model launch cost volatility (±15%) and OpEx scaling (±20%).")
    if st.button("RUN SIMULATION"):
        with st.spinner("Processing Data..."):
            sim_launch_costs = np.random.normal(base_launch_cost, base_launch_cost*0.15, 10000)
            sim_opex_multiplier = np.random.normal(1.0, 0.20, 10000)
            sim_profits = np.array([(m0_direct * sl * (1 - target_discount/100.0)) - (m0_relay * sl + total_opex * so + risk_reserve) for sl, so in zip(sim_launch_costs, sim_opex_multiplier)]) / 1e6
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("SYSTEM SURVIVABILITY", f"{(sim_profits > 0).mean() * 100:.1f}%")
            c_res2.metric("MEAN EXPECTED PROFIT", f"${sim_profits.mean():.1f}M")
            c_res3.metric("MAXIMUM DOWNSIDE", f"${sim_profits.min():.1f}M")
            
            fig_hist = px.histogram(sim_profits, nbins=60, color_discrete_sequence=["#0052FF"])
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#FFFFFF")
            fig_hist.update_layout(title="<b>PROFIT PROBABILITY DISTRIBUTION</b>", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8B949E"), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
