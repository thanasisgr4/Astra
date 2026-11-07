import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict

# ==============================================================================
# 1. ASTRAIA DYNAMICS - VALIDATED MULTI-MODAL ENGINE (BACKEND)
# ==============================================================================

@dataclass
class MissionParameters:
    payload_kg: float
    dry_mass_kg: float
    isp_chem_s: float
    isp_elec_s: float
    electric_split_ratio: float
    base_boil_off_rate_daily: float
    
@dataclass
class MarketConditions:
    fuel_cost_per_kg: float
    delay_penalty_per_day: float

class AstraiaValidatedEngine:
    """
    Τεχνικά Επικυρωμένος Κινητήρας Monte Carlo (NASA Mass Standards).
    Ενσωματώνει Multi-Modal Logistics (Ηλεκτρικό & Χημικό Ρυμουλκό).
    """
    def __init__(self, iterations: int = 10000, seed: int = 42):
        self.iterations = iterations
        self.g0 = 9.80665
        self.rng = np.random.default_rng(seed)

    def run_simulation(self, params: MissionParameters, market: MarketConditions) -> Dict:
        # --- ΣΤΟΧΑΣΤΙΚΕΣ ΚΑΤΑΝΟΜΕΣ ---
        dv_direct = self.rng.normal(5900, 100, self.iterations)
        dv_leo_l1 = self.rng.normal(3100, 50, self.iterations)
        dv_l1_lunar = self.rng.normal(2800, 50, self.iterations)
        wait_days_l1 = self.rng.triangular(5, 15, 30, self.iterations)
        
        fuel_price = self.rng.uniform(market.fuel_cost_per_kg * 0.9, market.fuel_cost_per_kg * 1.3, self.iterations)
        boil_off_rate = np.clip(self.rng.normal(params.base_boil_off_rate_daily, 0.0002, self.iterations), 0.0001, 0.005)

        # --- ΣΕΝΑΡΙΟ 1: DIRECT LEO -> MOON (Μόνο Χημικό) ---
        final_mass_direct = params.payload_kg + params.dry_mass_kg
        initial_mass_direct = final_mass_direct * np.exp(dv_direct / (params.isp_chem_s * self.g0))
        propellant_direct = initial_mass_direct - final_mass_direct
        cost_direct = propellant_direct * fuel_price

        # --- ΣΕΝΑΡΙΟ 2: STAGED L1 (MULTI-MODAL LOGISTICS) ---
        # Φάση B (L1 -> Moon): Όλο το φορτίο πάει με Χημικό (επείγον)
        final_mass_b = params.payload_kg + params.dry_mass_kg
        initial_mass_b = final_mass_b * np.exp(dv_l1_lunar / (params.isp_chem_s * self.g0))
        prop_needed_b = initial_mass_b - final_mass_b
        
        # Boil-off Retention
        retention_factor = (1 - boil_off_rate)**wait_days_l1
        prop_stored_at_l1 = prop_needed_b / retention_factor
        
        # Φάση Α (LEO -> L1): Multi-Modal Split (Χημικό vs Ηλεκτρικό)
        payload_total_phase_a = params.payload_kg + prop_stored_at_l1
        payload_chem = payload_total_phase_a * (1 - params.electric_split_ratio)
        payload_elec = payload_total_phase_a * params.electric_split_ratio
        
        # Υπολογισμός Χημικού
        final_mass_chem = payload_chem + params.dry_mass_kg
        init_mass_chem = final_mass_chem * np.exp(dv_leo_l1 / (params.isp_chem_s * self.g0))
        prop_chem = init_mass_chem - final_mass_chem
        
        # Υπολογισμός Ηλεκτρικού (Μικρότερο dry mass λόγω απουσίας μεγάλων δεξαμενών)
        final_mass_elec = payload_elec + (params.dry_mass_kg * 0.5)
        init_mass_elec = final_mass_elec * np.exp(dv_leo_l1 / (params.isp_elec_s * self.g0))
        prop_elec = init_mass_elec - final_mass_elec
        
        # Συνολικά κόστη Συναρίου 2
        total_prop_l1 = prop_chem + prop_elec + prop_stored_at_l1
        cost_fuel_l1 = total_prop_l1 * fuel_price
        cost_penalties_l1 = wait_days_l1 * market.delay_penalty_per_day
        cost_total_l1 = cost_fuel_l1 + cost_penalties_l1

        # --- ΑΝΑΛΥΣΗ ---
        savings = cost_direct - cost_total_l1
        prob_positive = (np.sum(savings > 0) / self.iterations) * 100

        df = pd.DataFrame({
            "Direct_Cost_M": cost_direct / 1e6,
            "L1_Cost_M": cost_total_l1 / 1e6,
            "Savings_M": savings / 1e6,
            "Wait_Days": wait_days_l1
        })

        return {
            "Probability_Positive": round(prob_positive, 1),
            "Median_Savings_M": round(np.median(savings) / 1e6, 2),
            "P5_Worst_Savings_M": round(np.percentile(savings, 5) / 1e6, 2),
            "P95_Best_Savings_M": round(np.percentile(savings, 95) / 1e6, 2),
            "Median_Prop_Saved_Tons": round(np.median(propellant_direct - total_prop_l1) / 1000, 1),
            "Data": df
        }

# ==============================================================================
# 2. PLOTLY VISUALIZATIONS
# ==============================================================================

def build_illustrative_map():
    fig = go.Figure()
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    fig.add_trace(go.Surface(x=3*np.cos(u)*np.sin(v), y=3*np.sin(u)*np.sin(v), z=3*np.cos(v), colorscale='Blues', showscale=False, name="Earth"))
    fig.add_trace(go.Surface(x=35+1.2*np.cos(u)*np.sin(v), y=1.2*np.sin(u)*np.sin(v), z=1.2*np.cos(v), colorscale='Greys', showscale=False, name="Moon"))
    fig.add_trace(go.Scatter3d(x=[26], y=[0], z=[0], mode='markers+text', marker=dict(size=8, color='gold', symbol='diamond'), text=["L1 Hub"], textposition="top center"))
    
    # Paths
    fig.add_trace(go.Scatter3d(x=[5, 26, 35], y=[0, 0, 0], z=[0, 2, 0], mode='lines', line=dict(color='#00FF66', width=4), name="Chemical Path"))
    
    t = np.linspace(0, 4*np.pi, 100)
    x_elec = np.linspace(5, 26, 100)
    y_elec, z_elec = 2 * np.sin(t) * (1 - x_elec/26), 2 * np.cos(t) * (1 - x_elec/26)
    fig.add_trace(go.Scatter3d(x=x_elec, y=y_elec, z=z_elec, mode='lines', line=dict(color='#00E5FF', width=2, dash='dot'), name="Electric Tug Spiral"))

    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='#0b0f19'), paper_bgcolor='#0b0f19', font=dict(color='white'), margin=dict(l=0, r=0, b=0, t=0))
    return fig

# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="Astraia OS | Multi-Modal Validated", layout="wide")
st.markdown("<style>.main {background-color: #0b0f19;} .stMetric {background-color: #161f33; padding: 15px; border-radius: 8px; border: 1px solid #263554;}</style>", unsafe_allow_html=True)

st.title("Astraia OS: Space Logistics & Financial Modeler")
st.info("**Proof of Concept (TRL 4):** Επικυρωμένο μοντέλο Multi-Modal εφοδιαστικής αλυσίδας. Τα οικονομικά εκτόξευσης αφορούν τον πελάτη, ενώ τα έσοδα αφορούν το B2B SaaS μοντέλο της Astraia.")
st.divider()

st.sidebar.header("Mission Specs (LEO to Moon)")
payload = st.sidebar.number_input("Payload (kg)", value=2500, step=250)
dry_mass = st.sidebar.number_input("Dry Mass (kg)", value=2000, step=250)
isp_chem = st.sidebar.number_input("Chemical Isp (s)", value=450, step=10)
isp_elec = st.sidebar.number_input("Electric Isp (s)", value=2000, step=100)
electric_split = st.sidebar.slider("Φορτίο με Ηλεκτρικό Ρυμουλκό (%)", min_value=0, max_value=100, value=60, step=5) / 100.0
boil_off = st.sidebar.slider("Boil-off Rate (%/day)", min_value=0.01, max_value=0.5, value=0.1, step=0.01) / 100.0

st.sidebar.header("Market Variables")
fuel_cost = st.sidebar.number_input("Fuel Cost ($/kg)", value=4000, step=500)
penalty = st.sidebar.number_input("Delay Penalty ($/day)", value=80000, step=10000)

engine = AstraiaValidatedEngine()
params = MissionParameters(payload_kg=payload, dry_mass_kg=dry_mass, isp_chem_s=isp_chem, isp_elec_s=isp_elec, electric_split_ratio=electric_split, base_boil_off_rate_daily=boil_off)
market = MarketConditions(fuel_cost_per_kg=fuel_cost, delay_penalty_per_day=penalty)
results = engine.run_simulation(params, market)
df = results["Data"]

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Customer ROI", "💼 Astraia Business Model", "🌐 Multi-Modal Route", "📊 Analytics"])

with tab1:
    st.subheader("Εξοικονόμηση Πελάτη (Apples-to-Apples)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Πιθανότητα Επιτυχίας", f"{results['Probability_Positive']}%")
    c2.metric("Διάμεση Εξοικονόμηση", f"${results['Median_Savings_M']}M")
    c3.metric("Worst Case (P5)", f"${results['P5_Worst_Savings_M']}M")
    c4.metric("Εξοικονόμηση Μάζας", f"{results['Median_Prop_Saved_Tons']} Tons")
    
    st.plotly_chart(go.Figure(data=go.Scatter(x=df["Wait_Days"], y=df["Savings_M"], mode='markers', marker=dict(size=4, color=df["Savings_M"], colorscale='Viridis'))).update_layout(xaxis_title="Days at L1", yaxis_title="Savings ($M)", paper_bgcolor='#0b0f19', plot_bgcolor='#161f33', font=dict(color='white')), use_container_width=True)

with tab2:
    st.subheader("Astraia OS - SaaS Revenue Plan")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Projected Revenue (Year 1-2): $700,000**\n\n- Paid Pilots: $100k\n- B2B Subscriptions: $480k\n- Custom APIs: $120k")
    with col2:
        st.error("**Projected OPEX (Year 1-2): $220,000**\n\n- Cloud/Servers: $30k\n- R&D/Payroll: $120k\n- Data Feeds: $25k\n- Sales/Marketing: $45k")

with tab3:
    st.plotly_chart(build_illustrative_map(), use_container_width=True)

with tab4:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df["Direct_Cost_M"], name="Direct", marker_color='#FF5252', opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=df["L1_Cost_M"], name="L1 Hybrid", marker_color='#00E5FF', opacity=0.7))
    st.plotly_chart(fig_hist.update_layout(barmode='overlay', paper_bgcolor='#0b0f19', plot_bgcolor='#161f33', font=dict(color='white')), use_container_width=True)
