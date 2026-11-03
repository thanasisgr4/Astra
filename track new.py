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
st.markdown("<span style='color:#8B949E; font-size:1.1rem; letter-spacing: 1px;'>MISSION CONTROL & TELEMETRY ENGINE (v10.0)</span>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='color:#FFFFFF;'>SCENARIO PRESETS</h3>", unsafe_allow_html=True)
    scenario = st.selectbox("Επιλέξτε Σενάριο Αγοράς", ["Custom", "Baseline", "Aggressive (Artemis)"])
    
    if scenario == "Baseline":
        def_tons, def_cost, def_disc, def_wait, def_risk = 100, 5000, 20.0, 5, 15
    elif scenario == "Aggressive (Artemis)":
        def_tons, def_cost, def_disc, def_wait, def_risk = 350, 2000, 25.0, 2, 10
    else:
        def_tons, def_cost, def_disc, def_wait, def_risk = 100, 5000, 20.0, 5, 15

    st.markdown("---")
    annual_tons = st.number_input("Annual Payload (Tons)", min_value=5, value=def_tons, step=10)
    target_discount = st.slider("Client SaaS Discount (%)", 5.0, 40.0, def_disc, 1.0)
    base_launch_cost = st.number_input("LEO Launch Cost ($/kg)", value=def_cost, step=500)
    isp = st.slider("Engine Isp (sec)", 300, 500, 450, 10)
    boil_off_rate = st.slider("Boil-Off Rate (%/day)", 0.0, 5.0, 0.5, 0.1)
    wait_days = st.slider("L1 Hub Wait Time (Days)", 0, 30, def_wait)
    module_capacity = st.number_input("Hub Module Capacity (Tons)", value=50)
    insurance_rate = st.slider("Risk Contingency Reserve (%)", 0, 30, def_risk, 5)

# ==========================================
# 4. EXECUTION
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
# 5. TABS & HIGH-FIDELITY VISUALIZATION
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EXECUTIVE DASHBOARD", 
    "📡 ORBITAL TELEMETRY MAP", 
    "🪐 CRYOGENIC PHYSICS", 
    "🎲 MONTE CARLO"
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
    st.markdown("### 🛰️ Live Astrodynamics & Satellite Network (Astraia OS)")
    st.write("Τοπολογία δικτύου υψηλής πιστότητας. Περιλαμβάνει τροχιακά επίπεδα (LEO, GEO, LLO) με κλίσεις (inclinations), ενεργούς δορυφόρους και τηλεμετρία ταχυτήτων (km/s).")
    
    # 3D Math for Spheres
    def create_sphere(r, cx, cy, cz, resolution=50):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = r * np.outer(np.cos(u), np.sin(v)) + cx
        y = r * np.outer(np.sin(u), np.sin(v)) + cy
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + cz
        return x, y, z

    # 3D Math for Inclined Orbital Rings
    def create_orbital_ring(r, cx, cy, cz, inc_x, inc_y, num_points=120):
        theta = np.linspace(0, 2*np.pi, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(num_points)
        # Apply Pitch (inc_x)
        y1 = y * np.cos(inc_x) - z * np.sin(inc_x)
        z1 = y * np.sin(inc_x) + z * np.cos(inc_x)
        # Apply Roll (inc_y)
        x2 = x * np.cos(inc_y) + z1 * np.sin(inc_y)
        z2 = -x * np.sin(inc_y) + z1 * np.cos(inc_y)
        return x2 + cx, y1 + cy, z2 + cz

    # Coordinates
    R_earth, x_E, y_E, z_E = 2.0, 0, 0, 0
    R_moon, x_M, y_M, z_M = 0.5, 14, 2, 1
    x_L1, y_L1, z_L1 = 11, 1.5, 0.8

    fig_space = go.Figure()

    # 1. Earth & Moon Surfaces
    x_e, y_e, z_e = create_sphere(R_earth, x_E, y_E, z_E)
    x_m, y_m, z_m = create_sphere(R_moon, x_M, y_M, z_M)
    fig_space.add_trace(go.Surface(x=x_e, y=y_e, z=z_e, colorscale='Tealgrn', showscale=False, name="Earth"))
    fig_space.add_trace(go.Surface(x=x_m, y=y_m, z=z_m, colorscale='Greys', showscale=False, name="Moon"))

    # 2. LEO Constellation Network (e.g. Starlink style)
    leo_inclinations = [(0.2, 0.1), (0.8, -0.4), (-0.5, 0.6)]
    for i, (ix, iy) in enumerate(leo_inclinations):
        rx, ry, rz = create_orbital_ring(2.5, x_E, y_E, z_E, ix, iy)
        fig_space.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode='lines', line=dict(color='#333333', width=1), showlegend=False))
        # Add a satellite on each plane with telemetry
        sat_idx = (i * 30) % 120
        fig_space.add_trace(go.Scatter3d(
            x=[rx[sat_idx]], y=[ry[sat_idx]], z=[rz[sat_idx]], mode='markers+text',
            marker=dict(size=3, color='#E0E0E0'), text=[f"SAT-{i+1} | v: 7.8 km/s"], 
            textposition="top right", textfont=dict(size=9, color='#8B949E'), showlegend=False
        ))

    # 3. GEO Orbit (Geostationary)
    gx, gy, gz = create_orbital_ring(4.5, x_E, y_E, z_E, 0.1, 0)
    fig_space.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode='lines', line=dict(color='#444444', width=2), showlegend=False))
    fig_space.add_trace(go.Scatter3d(
        x=[gx[80]], y=[gy[80]], z=[gz[80]], mode='markers+text',
        marker=dict(size=4, color='#F39C12'), text=["GEO-COM | v: 3.1 km/s"], 
        textposition="bottom right", textfont=dict(size=9, color='#8B949E'), name="GEO Satellites"
    ))

    # 4. LLO Orbit (Lunar Gateway / Orbiters)
    lx, ly, lz = create_orbital_ring(0.9, x_M, y_M, z_M, 0.4, 0.2)
    fig_space.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#555555', width=1, dash='dot'), showlegend=False))
    fig_space.add_trace(go.Scatter3d(
        x=[lx[20]], y=[ly[20]], z=[lz[20]], mode='markers+text',
        marker=dict(size=4, color='#00FF9D'), text=["GATEWAY | v: 1.6 km/s"], 
        textposition="top left", textfont=dict(size=9, color='#8B949E'), name="Lunar Infrastructure"
    ))

    # 5. Astraia L1 Hub Node
    fig_space.add_trace(go.Scatter3d(
        x=[x_L1], y=[y_L1], z=[z_L1], mode='markers+text',
        marker=dict(size=7, color='#0052FF', symbol='diamond'),
        text=["Astraia L1 Depot | v: 0.0 km/s (Rel)"], textposition="top center", 
        textfont=dict(color="#0052FF", size=11, family="Courier New"), name='Astraia Hub'
    ))

    # 6. Astraia Transfer Trajectory (Earth to L1)
    t = np.linspace(0, 1, 100)
    traj_x = 2.5 + (x_L1 - 2.5) * t
    traj_y = y_E + (y_L1 - y_E) * t + 2 * np.sin(t * np.pi) # Orbital curvature
    traj_z = z_E + (z_L1 - z_E) * t
    
    fig_space.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z, mode='lines',
        line=dict(color='#0052FF', width=3), name='Active Transfer Route'
    ))

    # 7. Payload in transit
    fig_space.add_trace(go.Scatter3d(
        x=[traj_x[45]], y=[traj_y[45]], z=[traj_z[45]], mode='markers+text',
        marker=dict(size=6, color='#FF2A6D'), text=["PAYLOAD | v: 10.4 km/s"],
        textposition="bottom center", textfont=dict(color="#FF2A6D", size=10, family="Courier New"), name='Client Payload'
    ))

    # Layout tuning for absolute dark space and perfect spheres
    fig_space.update_layout(
        template="plotly_dark", paper_bgcolor="#050505", plot_bgcolor="#050505",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            aspectmode='data', # ΑΥΤΟ ΚΛΕΙΔΩΝΕΙ ΤΙΣ ΣΦΑΙΡΕΣ ΝΑ ΕΙΝΑΙ ΣΤΡΟΓΓΥΛΕΣ
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, visible=False),
            camera=dict(eye=dict(x=0.2, y=-1.8, z=0.4)) 
        ),
        showlegend=True,
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1, font=dict(color="#8B949E"), bgcolor="rgba(0,0,0,0)")
    )
    
    st.plotly_chart(fig_space, use_container_width=True)

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
