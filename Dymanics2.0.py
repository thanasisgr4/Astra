import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math

# ==========================================
# 1. ENTERPRISE CONFIGURATION & RICH VISUALS CSS
# ==========================================
st.set_page_config(page_title="Astraia Dynamics | Mission Control", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #050505; color: #E0E0E0; }
        .stSidebar { background-color: #0A0A0A; border-right: 1px solid #1F1F1F; }
        h1 { color: #FFFFFF !important; font-weight: 300 !important; letter-spacing: 2px; font-size: 2.5rem !important; }
        h2, h3 { color: #FFFFFF !important; font-weight: 400 !important; letter-spacing: 1px;}
        p, span, label { color: #8B949E !important; }
        div[data-testid="stMetric"] {
            background-color: #0A0A0A;
            border: 1px solid #1F1F1F;
            padding: 20px;
            border-radius: 4px;
            border-top: 2px solid #0052FF;
        }
        div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 2rem !important; font-weight: 300 !important; }
        div[data-testid="stMetricDelta"] { color: #8B949E !important; font-weight: 400 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; border-bottom: 1px solid #1F1F1F; }
        .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; padding-bottom: 10px; color: #8B949E; font-weight: 500; }
        .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom: 2px solid #0052FF !important; }
        hr { border-color: #1F1F1F; }
    </style>
""", unsafe_allow_html=True)

st.title("ASTRAIA DYNAMICS")
st.markdown("<span style='color:#8B949E; font-size:1.1rem; letter-spacing: 1px;'>ORBITAL LOGISTICS & ROUTING ENGINE (v6.0 - Visual Hub & Routes)</span>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. SIDEBAR: SCENARIO PRESETS & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("<h3 style='color:#FFFFFF;'>SCENARIO PRESETS</h3>", unsafe_allow_html=True)
    scenario = st.selectbox(
        "Επιλέξτε Σενάριο Αγοράς",
        ["Custom (Προσαρμοσμένο)", "Conservative (Συντηρητικό Start)", "Baseline (Εμπορική Ανάπτυξη)", "Aggressive (Εποχή Artemis)"]
    )
    
    if scenario == "Conservative (Συντηρητικό Start)":
        def_tons, def_cost, def_disc, def_wait, def_risk = 20, 7000, 15.0, 10, 20
    elif scenario == "Baseline (Εμπορική Ανάπτυξη)":
        def_tons, def_cost, def_disc, def_wait, def_risk = 100, 5000, 20.0, 5, 15
    elif scenario == "Aggressive (Εποχή Artemis)":
        def_tons, def_cost, def_disc, def_wait, def_risk = 350, 5000, 25.0, 2, 10
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
# 3. MATHEMATICAL CORE
# ==========================================
g0, dry_ratio, dv_direct, dv_stage1 = 9.81, 0.10, 5900, 3800

def get_mass_ratio(dv, specific_impulse): return math.exp(dv / (specific_impulse * g0))

payload_kg = annual_tons * 1000
m0_direct = payload_kg * (1 + (get_mass_ratio(dv_direct, isp) - 1) * (1 + dry_ratio))

propellant_needed_relay = payload_kg * (get_mass_ratio(dv_stage1, isp) - 1)
boil_off_penalty = propellant_needed_relay * ((1 - (boil_off_rate/100))**wait_days)
propellant_launch_relay = propellant_needed_relay + (propellant_needed_relay - boil_off_penalty)
m0_relay = payload_kg + propellant_launch_relay * (1 + dry_ratio)

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
# 4. TABS WITH RICH VISUALS & ROUTES
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EXECUTIVE DASHBOARD", 
    "🗺️ ORBITAL ROUTES & NETWORK", 
    "🪐 CRYOGENIC PHYSICS", 
    "🎲 MONTE CARLO SIMULATION"
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
        textposition="outside",
        connector={"line":{"color":"#1F1F1F", "width": 2}},
        decreasing={"marker":{"color":"#333333"}}, 
        increasing={"marker":{"color":"#E0E0E0"}}, 
        totals={"marker":{"color":"#0052FF"}}      
    ))
    fig_waterfall.update_layout(
        title="<b>FINANCIAL FLOW & REVENUE STREAM (USD Millions)</b>",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B949E", size=13),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1F1F1F", zerolinecolor="#333333")
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

with tab2:
    st.markdown("### 🌐 Hub-and-Spoke Network Architecture ($L_1$ Routing)")
    st.write("Οπτικοποίηση της τοπολογίας δικτύου: Σύνδεση επιφανείας Γης με τον κεντρικό κόμβο διαμετακόμισης στο σημείο Lagrange 1 ($L_1$) και ανακατεύθυνση προς Σελήνη/Βάσεις.")
    
    # 3D Orbital Network Scatter & Lines Visualization
    nodes = {
        "Node": ["Earth (LEO)", "Astraia L1 Hub", "Lunar Surface", "Mars Gateway"],
        "X": [0, 5, 10, 15],
        "Y": [0, 5, 8, 12],
        "Z": [0, 8, 4, 15],
        "Type": ["Origin", "Central Relay (Hub)", "Destination", "Deep Space"]
    }
    df_nodes = pd.DataFrame(nodes)
    
    fig_network = go.Figure()
    # Προσθήκη κόμβων
    fig_network.add_trace(go.Scatter3d(
        x=df_nodes["X"], y=df_nodes["Y"], z=df_nodes["Z"],
        mode='text+markers',
        text=df_nodes["Node"],
        marker=dict(size=[12, 18, 10, 10], color=['#8B949E', '#0052FF', '#00FF9D', '#FF2A6D'], opacity=0.9)
    ))
    # Προσθήκη διαδρομών (Spokes)
    fig_network.add_trace(go.Scatter3d(
        x=[0, 5, 5, 10], y=[0, 5, 5, 8], z=[0, 8, 8, 4],
        mode='lines',
        line=dict(color='#0052FF', width=5),
        name='Astraia Optimized Routes'
    ))
    fig_network.update_layout(
        title="<b>3D Hub-and-Spoke Routing Map</b>",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1F1F1F"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1F1F1F"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#1F1F1F"),
        ),
        font=dict(color="#8B949E")
    )
    st.plotly_chart(fig_network, use_container_width=True)
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.info("📌 **Σκέλος 1 (Γη ➔ L1 Hub):** Βελτιστοποιημένο Δέλτα-V (3,800 m/s) με τυποποιημένες θύρες UDI.")
    with col_r2:
        st.success("🚀 **Σκέλος 2 (L1 Hub ➔ Προορισμός):** Αποθήκευση καυσίμων στον κόμβο και cross-docking χωρίς επαναλαμβανόμενες βαρυτικές εκκινήσεις από τη Γη.")

with tab3:
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Mass Penalty & Boil-Off Analysis")
        st.markdown(f"<span style='color:#0052FF; font-weight:bold;'>Hub Wait Time:</span> {wait_days} days <br> <span style='color:#0052FF; font-weight:bold;'>Evaporation Loss:</span> {boil_off_penalty/1000:.1f} tons", unsafe_allow_html=True)
        st.write("")
        st.write(f"- Direct Launch Mass Requirement: **{m0_direct/1000:.1f} t**")
        st.write(f"- Astraia Relay Mass Requirement: **{m0_relay/1000:.1f} t**")
        
    with colB:
        df_mass = pd.DataFrame({
            "Stage": ["Payload", "Dry Mass", "Propellant", "Boil-off Refill"],
            "Direct": [payload_kg/1000, (m0_direct-payload_kg)*dry_ratio/1000, (m0_direct-payload_kg)*(1-dry_ratio)/1000, 0],
            "Astraia L1": [payload_kg/1000, (m0_relay-payload_kg-boil_off_penalty)*dry_ratio/1000, (m0_relay-payload_kg-boil_off_penalty)*(1-dry_ratio)/1000, boil_off_penalty/1000]
        })
        fig_mass = px.bar(df_mass, x="Stage", y=["Direct", "Astraia L1"], barmode="group", color_discrete_sequence=["#333333", "#0052FF"])
        fig_mass.update_layout(
            title="<b>MASS ALLOCATION BREAKDOWN (Tons)</b>", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8B949E")
        )
        st.plotly_chart(fig_mass, use_container_width=True)

with tab4:
    st.markdown("### Stochastic Risk Simulator")
    st.write("Executing 10,000 parallel iterations to model launch cost volatility (±15%) and OpEx scaling (±20%).")
    
    if st.button("RUN SIMULATION"):
        with st.spinner("Processing Data..."):
            sim_launch_costs = np.random.normal(base_launch_cost, base_launch_cost*0.15, 10000)
            sim_opex_multiplier = np.random.normal(1.0, 0.20, 10000)
            
            sim_profits = []
            for i in range(10000):
                s_direct_cost = m0_direct * sim_launch_costs[i]
                s_revenue = s_direct_cost * (1 - (target_discount / 100.0))
                s_astraia_cost = m0_relay * sim_launch_costs[i]
                s_total_cost = s_astraia_cost + (total_opex * sim_opex_multiplier[i]) + risk_reserve
                sim_profits.append((s_revenue - s_total_cost) / 1e6)
                
            sim_profits = np.array(sim_profits)
            success_rate = (sim_profits > 0).mean() * 100
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("SYSTEM SURVIVABILITY", f"{success_rate:.1f}%")
            c_res2.metric("MEAN EXPECTED PROFIT", f"${sim_profits.mean():.1f}M")
            c_res3.metric("MAXIMUM DOWNSIDE", f"${sim_profits.min():.1f}M")
            
            fig_hist = px.histogram(sim_profits, nbins=60, color_discrete_sequence=["#0052FF"])
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#FFFFFF")
            fig_hist.update_layout(
                title="<b>PROFIT PROBABILITY DISTRIBUTION</b>", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#8B949E"),
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# ==========================================
# 1. CUSTOM EXCEPTIONS (Διαχείριση Σφαλμάτων)
# ==========================================
class OrbitalPhysicsError(Exception):
    """Εγείρεται όταν οι παράμετροι τροχιάς παραβιάζουν τους νόμους της φυσικής."""
    pass

class EconomicViabilityError(Exception):
    """Εγείρεται όταν το μοντέλο αποτυγχάνει στα οικονομικά κριτήρια."""
    pass

# ==========================================
# 2. DATA MODELS (Δομές Δεδομένων Πελάτη & Συστήματος)
# ==========================================
@dataclass
class SpacecraftSpecs:
    payload_kg: float
    isp_seconds: float
    dry_mass_ratio: float = 0.10  # Το 10% της συνολικής μάζας είναι η άδεια δομή (dry mass)

@dataclass
class NetworkNodes:
    dv_direct_lunar: float = 5900.0  # Δv για απευθείας LEO -> Lunar Surface (m/s)
    dv_leo_to_l1: float = 3800.0     # Δv για LEO -> L1 Hub (m/s)
    dv_l1_to_lunar: float = 2100.0   # Δv για L1 Hub -> Lunar Surface (m/s)

@dataclass
class MarketConditions:
    base_launch_cost_per_kg: float
    saas_discount_margin: float
    boil_off_rate_per_day: float = 0.005  # 0.5% απώλεια την ημέρα
    hub_wait_time_days: int = 5

# ==========================================
# 3. CORE ROUTING ENGINE (Ο Ψηφιακός Εγκέφαλος)
# ==========================================
class AstraiaRoutingEngine:
    """
    Enterprise-grade μηχανή βελτιστοποίησης τροχιακών δικτύων (Hub-and-Spoke στο L1).
    Υπολογίζει μάζες, θερμοδυναμικές απώλειες και οικονομικά SaaS περιθώρια.
    """
    G0 = 9.80665  # Σταθερά επιτάχυνσης βαρύτητας (m/s^2)

    def __init__(self, specs: SpacecraftSpecs, market: MarketConditions):
        self.specs = specs
        self.market = market
        self.nodes = NetworkNodes()

    def _calculate_mass_ratio(self, delta_v: float) -> float:
        """Εφαρμογή της εξίσωσης πυραύλων του Tsiolkovsky."""
        if self.specs.isp_seconds <= 0:
            raise OrbitalPhysicsError("Η Ειδική Ώση (Isp) πρέπει να είναι θετική.")
        return math.exp(delta_v / (self.specs.isp_seconds * self.G0))

    def _simulate_thermodynamics(self, required_propellant: float) -> float:
        """Υπολογίζει τις απώλειες καυσίμου (Boil-off) λόγω παραμονής στον κόμβο L1."""
        retention_factor = (1 - self.market.boil_off_rate_per_day) ** self.market.hub_wait_time_days
        boil_off_loss = required_propellant * (1 - retention_factor)
        return boil_off_loss

    def calculate_direct_route(self) -> float:
        """Υπολογίζει την απαιτούμενη αρχική μάζα (m0) για απευθείας αποστολή."""
        mass_ratio = self._calculate_mass_ratio(self.nodes.dv_direct_lunar)
        
        # Επίλυση: m0 = payload + propellant + dry_mass
        # propellant = (payload + dry_mass) * (mass_ratio - 1)
        m0_direct = self.specs.payload_kg * (1 + (mass_ratio - 1) * (1 + self.specs.dry_mass_ratio))
        return m0_direct

    def calculate_astraia_hub_route(self) -> Tuple[float, float]:
        """
        Υπολογίζει την απαιτούμενη μάζα (m0) μέσω του L1 Hub, 
        συνυπολογίζοντας το boil-off και το cross-docking.
        """
        # Σκέλος 1: LEO -> L1
        mr_stage1 = self._calculate_mass_ratio(self.nodes.dv_leo_to_l1)
        propellant_needed = self.specs.payload_kg * (mr_stage1 - 1)
        
        # Θερμοδυναμική ποινή στο L1
        boil_off_penalty = self._simulate_thermodynamics(propellant_needed)
        propellant_total = propellant_needed + boil_off_penalty
        
        # Συνολική μάζα εκκίνησης για το σύστημα Astraia
        m0_relay = self.specs.payload_kg + propellant_total * (1 + self.specs.dry_mass_ratio)
        return m0_relay, boil_off_penalty

    def generate_mission_manifest(self) -> Dict[str, Any]:
        """
        Παράγει το τελικό JSON/Dict Report με τα αποτελέσματα της δρομολόγησης.
        Αυτό το αντικείμενο επιστρέφεται στο API ή στο Web UI.
        """
        m0_direct = self.calculate_direct_route()
        m0_relay, boil_off = self.calculate_astraia_hub_route()

        mass_savings_kg = m0_direct - m0_relay
        mass_savings_pct = (mass_savings_kg / m0_direct) * 100

        if mass_savings_pct < 0:
            raise EconomicViabilityError("Το μοντέλο απέτυχε: Η απευθείας εκτόξευση είναι πιο αποδοτική.")

        # Οικονομικά στοιχεία
        direct_cost = m0_direct * self.market.base_launch_cost_per_kg
        astraia_client_price = direct_cost * (1 - self.market.saas_discount_margin)
        astraia_actual_cost = m0_relay * self.market.base_launch_cost_per_kg
        gross_profit = astraia_client_price - astraia_actual_cost

        return {
            "mission_status": "OPTIMIZED",
            "logistics_metrics": {
                "payload_kg": round(self.specs.payload_kg, 2),
                "direct_launch_mass_kg": round(m0_direct, 2),
                "astraia_hub_mass_kg": round(m0_relay, 2),
                "boil_off_penalty_kg": round(boil_off, 2),
                "mass_savings_percentage": round(mass_savings_pct, 2)
            },
            "financial_metrics_usd": {
                "competitor_direct_cost": round(direct_cost, 2),
                "astraia_client_invoice": round(astraia_client_price, 2),
                "astraia_operational_cost": round(astraia_actual_cost, 2),
                "gross_profit_per_mission": round(gross_profit, 2)

