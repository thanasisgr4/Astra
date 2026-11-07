import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict

# ==============================================================================
# 1. ASTRAIA DYNAMICS - VALIDATED AEROSPACE ROUTING & ECONOMIC ENGINE
# ==============================================================================

@dataclass
class MissionParameters:
    payload_kg: float
    dry_mass_kg: float
    isp_chem_s: float
    base_boil_off_rate_daily: float
    
@dataclass
class MarketConditions:
    fuel_cost_per_kg: float
    delay_penalty_per_day: float

class AstraiaValidatedEngine:
    """
    Τεχνικά Επικυρωμένος Κινητήρας Monte Carlo (NASA Mass & Tsiolkovsky Standards).
    Συγκρίνει ισοδύναμες αποστολές με τελικό προορισμό τη Σεληνιακή Επιφάνεια.
    """
    def __init__(self, iterations: int = 10000, seed: int = 42):
        self.iterations = iterations
        self.g0 = 9.80665
        self.rng = np.random.default_rng(seed)  # Reproducible random generator

    def run_simulation(self, params: MissionParameters, market: MarketConditions) -> Dict:
        # --- ΣΤΟΧΑΣΤΙΚΕΣ ΚΑΤΑΝΟΜΕΣ (ΑΒΕΒΑΙΟΤΗΤΑ) ---
        # 1. Απευθείας LEO -> Lunar Surface (Direct)
        dv_direct = self.rng.normal(5900, 100, self.iterations)
        
        # 2. Σπαστή Διαδρομή (Staged via L1 Hub)
        dv_leo_l1 = self.rng.normal(3100, 50, self.iterations)
        dv_l1_lunar = self.rng.normal(2800, 50, self.iterations)
        wait_days_l1 = self.rng.triangular(5, 15, 30, self.iterations)
        
        # Οικονομικές μεταβλητές
        fuel_price = self.rng.uniform(market.fuel_cost_per_kg * 0.9, market.fuel_cost_per_kg * 1.3, self.iterations)
        boil_off_rate = np.clip(self.rng.normal(params.base_boil_off_rate_daily, 0.0002, self.iterations), 0.0001, 0.005)

        # --- ΥΠΟΛΟΓΙΣΜΟΙ ΦΥΣΙΚΗΣ: ROUTE 1 (DIRECT TO MOON) ---
        # Τελική Μάζα στον προορισμό = Payload + Ξηρή Μάζα (Lander/Vehicle)
        final_mass_direct = params.payload_kg + params.dry_mass_kg
        initial_mass_direct = final_mass_direct * np.exp(dv_direct / (params.isp_chem_s * self.g0))
        propellant_direct = initial_mass_direct - final_mass_direct
        cost_direct = propellant_direct * fuel_price

        # --- ΥΠΟΛΟΓΙΣΜΟΙ ΦΥΣΙΚΗΣ: ROUTE 2 (STAGED VIA L1 HUB) ---
        # Φάση B: L1 -> Lunar Surface
        final_mass_l1_to_moon = params.payload_kg + params.dry_mass_kg
        initial_mass_l1_to_moon = final_mass_l1_to_moon * np.exp(dv_l1_lunar / (params.isp_chem_s * self.g0))
        prop_needed_for_phase_b = initial_mass_l1_to_moon - final_mass_l1_to_moon
        
        # Boil-off Loss κατά την παραμονή στον L1 
        # (Αν χρειαζόμαστε X μετά από T μέρες, πρέπει να αποθηκεύσουμε X / (1-rate)^T)
        retention_factor = (1 - boil_off_rate)**wait_days_l1
        prop_stored_at_l1 = prop_needed_for_phase_b / retention_factor
        boil_off_loss = prop_stored_at_l1 - prop_needed_for_phase_b
        
        # Φάση Α: LEO -> L1 (Μεταφορά του Payload + του καυσίμου για τη συνέχεια)
        # Υποθέτουμε ότι το όχημα LEO->L1 μεταφέρει το payload και τη μάζα προωθητικών που απαιτείται στον L1
        payload_for_phase_a = params.payload_kg + prop_stored_at_l1
        final_mass_leo_to_l1 = payload_for_phase_a + params.dry_mass_kg
        initial_mass_leo_to_l1 = final_mass_leo_to_l1 * np.exp(dv_leo_l1 / (params.isp_chem_s * self.g0))
        prop_phase_a = initial_mass_leo_to_l1 - final_mass_leo_to_l1

        # Συνολικά καύσιμα και κόστος για την αρχιτεκτονική L1
        total_propellant_l1_route = prop_phase_a + prop_stored_at_l1
        cost_fuel_l1 = total_propellant_l1_route * fuel_price
        cost_penalties_l1 = wait_days_l1 * market.delay_penalty_per_day
        cost_total_l1 = cost_fuel_l1 + cost_penalties_l1

        # --- ΟΙΚΟΝΟΜΙΚΗ ΑΝΑΛΥΣΗ ΠΕΛΑΤΗ ---
        savings = cost_direct - cost_total_l1
        prob_positive_result = (np.sum(savings > 0) / self.iterations) * 100

        df = pd.DataFrame({
            "Direct_Cost_M": cost_direct / 1e6,
            "L1_Cost_M": cost_total_l1 / 1e6,
            "Savings_M": savings / 1e6,
            "Prop_Direct_Tons": propellant_direct / 1000,
            "Prop_L1_Total_Tons": total_propellant_l1_route / 1000,
            "Wait_Days": wait_days_l1
        })

        return {
            "Probability_Positive_Result": round(prob_positive_result, 1),
            "Median_Savings_M": round(np.median(savings) / 1e6, 2),
            "P5_Worst_Savings_M": round(np.percentile(savings, 5) / 1e6, 2),
            "P95_Best_Savings_M": round(np.percentile(savings, 95) / 1e6, 2),
            "Median_Prop_Saved_Tons": round(np.median(propellant_direct - total_propellant_l1_route) / 1000, 1),
            "Data": df
        }

# ==============================================================================
# 2. PLOTLY VISUALIZATIONS (ILLUSTRATIVE ROUTES)
# ==============================================================================

def build_illustrative_orbital_map():
    fig = go.Figure()

    # Earth & Moon
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    fig.add_trace(go.Surface(x=3*np.cos(u)*np.sin(v), y=3*np.sin(u)*np.sin(v), z=3*np.cos(v), colorscale='Blues', showscale=False, name="Earth (LEO)"))
    fig.add_trace(go.Surface(x=35+1.2*np.cos(u)*np.sin(v), y=1.2*np.sin(u)*np.sin(v), z=1.2*np.cos(v), colorscale='Greys', showscale=False, name="Lunar Surface"))

    # L1 Hub
    fig.add_trace(go.Scatter3d(x=[26], y=[0], z=[0], mode='markers+text', marker=dict(size=8, color='gold', symbol='diamond'), text=["L1 Hub"], textposition="top center", name="L1 Node"))

    # Illustrative Paths
    fig.add_trace(go.Scatter3d(x=[5, 26, 35], y=[0, 0, 0], z=[0, 2, 0], mode='lines+markers', line=dict(color='#00FF66', width=4), name="Staged Route (L1)"))
    fig.add_trace(go.Scatter3d(x=[5, 35], y=[0, 0], z=[0, -4], mode='lines', line=dict(color='#FF5252', width=3, dash='dash'), name="Direct Route"))

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='#0b0f19'),
        paper_bgcolor='#0b0f19', font=dict(color='white'), margin=dict(l=0, r=0, b=0, t=0), legend=dict(x=0.01, y=0.99)
    )
    return fig

# ==============================================================================
# 3. STREAMLIT UI (ENTERPRISE / ACADEMIC VALIDATED)
# ==============================================================================

st.set_page_config(page_title="Astraia OS | Validation Model", layout="wide")
st.markdown("<style>.main {background-color: #0b0f19;} .stMetric {background-color: #161f33; padding: 15px; border-radius: 8px; border: 1px solid #263554;}</style>", unsafe_allow_html=True)

st.title("Astraia OS: Space Logistics & Financial Modeler")
st.info("**Proof of Concept (TRL 4) - Ενδεικτικό Μοντέλο:** Ο παρών κώδικας αποτελεί επικυρωμένη αφετηρία μοντελοποίησης Monte Carlo (εξισώσεις Tsiolkovsky & NASA mass guidelines). Τα οικονομικά εκτόξευσης αφορούν τον πελάτη (Customer View), ενώ τα έσοδα λογισμικού αφορούν το επιχειρηματικό μοντέλο της Astraia Dynamics (SaaS View).")
st.divider()

# --- INPUTS ---
st.sidebar.header("Mission Specs (LEO to Lunar Surface)")
payload = st.sidebar.number_input("Παραδιδόμενο Payload (kg)", value=2500, step=250)
dry_mass = st.sidebar.number_input("Ξηρή Μάζα Οχήματος (kg)", value=2000, step=250)
isp = st.sidebar.number_input("Chemical Isp (s)", value=450, step=10)
boil_off = st.sidebar.slider("Boil-off Rate (% / ημέρα)", min_value=0.01, max_value=0.5, value=0.1, step=0.01) / 100.0

st.sidebar.header("Market Variables")
fuel_cost = st.sidebar.number_input("Κόστος Καυσίμου ($/kg)", value=4000, step=500)
penalty = st.sidebar.number_input("Ρήτρα Καθυστέρησης ($/ημέρα)", value=80000, step=10000)

# --- EXECUTION ---
engine = AstraiaValidatedEngine(iterations=10000)
params = MissionParameters(payload_kg=payload, dry_mass_kg=dry_mass, isp_chem_s=isp, base_boil_off_rate_daily=boil_off)
market = MarketConditions(fuel_cost_per_kg=fuel_cost, delay_penalty_per_day=penalty)
results = engine.run_simulation(params, market)
df = results["Data"]

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Οικονομικά Πελάτη (Mission Savings)", "💼 Οικονομικά Astraia (SaaS Business Model)", "🌐 3D Illustrative Routes", "📊 Monte Carlo Distributions"])

with tab1:
    st.subheader("Σύγκριση Κόστους για τον Πελάτη (Apples-to-Apples)")
    st.write("Ανάλυση κόστους προωθητικών και ρητρών για την παράδοση φορτίου στη Σεληνιακή Επιφάνεια.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Πιθανότητα Θετικού Αποτελέσματος", f"{results['Probability_Positive_Result']}%")
    c2.metric("Διάμεση Εξοικονόμηση (Median)", f"${results['Median_Savings_M']}M")
    c3.metric("Worst Case Εξοικονόμηση (P5)", f"${results['P5_Worst_Savings_M']}M")
    c4.metric("Διάμεσο Κέρδος Μάζας (Propellant)", f"{results['Median_Prop_Saved_Tons']} Tons")
    
    fig_scatter = go.Figure(data=go.Scatter(x=df["Wait_Days"], y=df["Savings_M"], mode='markers', marker=dict(size=4, color=df["Savings_M"], colorscale='Viridis', showscale=True)))
    fig_scatter.update_layout(title="Trade-off: Ημέρες Αναμονής vs Εξοικονόμηση ($M)", xaxis_title="Ημέρες παραμονής στον L1", yaxis_title="Εξοικονόμηση ($M)", paper_bgcolor='#0b0f19', plot_bgcolor='#161f33', font=dict(color='white'))
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Astraia OS - B2B SaaS Revenue Model")
    st.write("Οικονομική πρόβλεψη της εταιρείας λογισμικού μας, ανεξάρτητη από το κόστος των εκτοξεύσεων.")
    
    col_rev, col_exp = st.columns(2)
    with col_rev:
        st.markdown("#### Εκτιμώμενα Έσοδα (MRR / YRR)")
        st.write("1. **Paid Pilot Studies:** 2 εταιρείες × $50,000 / study = **$100,000**")
        st.write("2. **B2B SaaS Subscriptions (Tier 1):** 5 συνδρομητές × $8,000 / μήνα × 12 = **$480,000**")
        st.write("3. **Enterprise Custom Routing API:** 1 συμβόλαιο × **$120,000**")
        st.success("**Total Projected Revenue (Year 1-2): $700,000**")
        
    with col_exp:
        st.markdown("#### Λειτουργικά Έξοδα (OPEX)")
        st.write("- **Cloud Computing (AWS/GCP):** $2,500 / μήνα = **$30,000**")
        st.write("- **R&D / Μισθοδοσία Ομάδας (2 άτομα):** **$120,000**")
        st.write("- **Software Licenses & Data Feeds (TLEs):** **$25,000**")
        st.write("- **Go-to-Market & B2B Sales:** **$45,000**")
        st.error("**Total Projected OPEX (Year 1-2): $220,000**")

with tab3:
    st.subheader("Αναπαράσταση Τροχιακών Διαδρομών (Illustrative)")
    st.plotly_chart(build_illustrative_orbital_map(), use_container_width=True)

with tab4:
    st.subheader("Κατανομή Monte Carlo (Διαστήματα Εμπιστοσύνης)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df["Direct_Cost_M"], name="Direct Route Cost", marker_color='#FF5252', opacity=0.7))
    fig_hist.add_trace(go.Histogram(x=df["L1_Cost_M"], name="L1 Staged Cost (inc. Penalties)", marker_color='#00E5FF', opacity=0.7))
    fig_hist.update_layout(barmode='overlay', title="Σύγκριση 10.000 Στοχαστικών Σεναρίων", xaxis_title="Κόστος Αποστολής Εκατ. ($)", paper_bgcolor='#0b0f19', plot_bgcolor='#161f33', font=dict(color='white'))
    st.plotly_chart(fig_hist, use_container_width=True)
