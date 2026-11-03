import streamlit as stβ
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict

# ==============================================================================
# 1. ASTRAIA DYNAMICS - ENTERPRISE STOCHASTIC & PHYSICAL ENGINE (BACKEND)
# ==============================================================================

@dataclass
class SpacecraftSpecs:
    dry_mass_kg: float
    isp_s: float
    base_boil_off_rate: float  # Ημερήσιος ρυθμός εξάτμισης (π.χ. 0.001 = 0.1%/ημέρα)

@dataclass
class MarketConditions:
    base_fuel_cost_kg: float
    delay_penalty_per_day: float  # Ρήτρα καθυστέρησης παράδοσης σε $/ημέρα

class AstraiaMonteCarloEngine:
    """
    Διανυσματική Μηχανική Προσομοίωσης Monte Carlo (Vectorized NumPy)
    Υπολογίζει Tsiolkovsky Δv, Απώλειες Boil-off και Οικονομικά Trade-offs.
    """
    def __init__(self, iterations: int = 10000):
        self.iterations = iterations
        self.g0 = 9.80665  # Σταθερά βαρύτητας (m/s^2)

def build_3d_orbital_map():
    """Δημιουργεί διαδραστικό 3D χάρτη του τροχιακού δικτύου Earth-L1-Moon."""
    fig = go.Figure()

    # 1. Γη (Earth Sphere)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth = 3 * np.cos(u) * np.sin(v)
    y_earth = 3 * np.sin(u) * np.sin(v)
    z_earth = 3 * np.cos(v)
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, name="Earth"))

    # 2. Σελήνη (Moon Sphere) - ΔΙΟΡΘΩΘΗΚΕ ΤΟ 'Greys'
    x_moon = 35 + 1.2 * np.cos(u) * np.sin(v)
    y_moon = 0 + 1.2 * np.sin(u) * np.sin(v)
    z_moon = 1.2 * np.cos(v)
    fig.add_trace(go.Surface(x=x_moon, y=y_moon, z=z_moon, colorscale='Greys', showscale=False, name="Moon"))

    # 3. Κόμβος L1 Lagrange Node
    fig.add_trace(go.Scatter3d(
        x=[26], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=10, color='gold', symbol='diamond'),
        text=["<b>L1 Logistics Hub</b>"],
        textposition="top center",
        name="L1 Hub"
    ))

    # 4. Τροχιακοί Κύκλοι (LEO & GEO)
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter3d(x=5*np.cos(theta), y=5*np.sin(theta), z=np.zeros(100), mode='lines', line=dict(color='cyan', width=2, dash='dot'), name="LEO Orbit"))
    fig.add_trace(go.Scatter3d(x=10*np.cos(theta), y=10*np.sin(theta), z=np.zeros(100), mode='lines', line=dict(color='magenta', width=2, dash='dot'), name="GEO Orbit"))

    # 5. Τροχιές Μεταφοράς (Trajectories)
    # LEO -> L1 -> Moon (Staged Path)
    fig.add_trace(go.Scatter3d(x=[5, 26, 35], y=[0, 0, 0], z=[0, 2, 0], mode='lines+markers', line=dict(color='#00FF66', width=5), name="Astraia Staged Route (L1)"))
    # Direct Path
    fig.add_trace(go.Scatter3d(x=[5, 35], y=[0, 0], z=[0, -4], mode='lines', line=dict(color='red', width=3, dash='dash'), name="Direct Trajectory"))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='rgb(10, 15, 30)'
        ),
        paper_bgcolor='rgb(10, 15, 30)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
    )
    return fig

# ==============================================================================
# 3. STREAMLIT ENTERPRISE UI (FRONTEND)
# ==============================================================================

st.set_page_config(page_title="Astraia OS | Mission Control & Risk Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom Dark Theme Injection
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    .stMetric { background-color: #161f33; padding: 15px; border-radius: 10px; border: 1px solid #263554; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00E5FF; }
    </style>
""", unsafe_allow_html=True)

# Header Title Block
st.markdown("<h1 style='text-align: center; color: #00E5FF;'>ASTRAIA OS v2.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8C9EFF;'>Space Logistics Optimization & Financial Risk Engine (B2B SaaS)</p>", unsafe_allow_html=True)
st.divider()

# Sidebar: Mission Parameters & Inputs
st.sidebar.image("https://img.icons8.com/isometric-folders/100/space-shuttle.png", width=60)
st.sidebar.title("Mission Controls")
dry_mass = st.sidebar.number_input("Spacecraft Dry Mass (kg)", value=5000, step=250)
isp = st.sidebar.number_input("Propulsion Specific Impulse Isp (s)", value=450, step=10)
fuel_cost = st.sidebar.number_input("Propellant Cost ($/kg)", value=5000, step=500)
delay_penalty = st.sidebar.number_input("SLA Delay Penalty ($/day)", value=100000, step=25000)
boiloff_rate = st.sidebar.slider("Daily Boil-off Loss Rate (%)", min_value=0.01, max_value=0.50, value=0.10, step=0.01) / 100.0

# Run Simulation Engine
engine = AstraiaMonteCarloEngine(iterations=10000)
sc_specs = SpacecraftSpecs(dry_mass_kg=dry_mass, isp_s=isp, base_boil_off_rate=boiloff_rate)
market_env = MarketConditions(base_fuel_cost_kg=fuel_cost, delay_penalty_per_day=delay_penalty)
results = engine.simulate_architectures(sc_specs, market_env)
df = results["Data"]

# Top Level KPI Banner
c1, c2, c3, c4 = st.columns(4)
c1.metric("L1 Win Probability", f"{results['L1_Success_Probability']}%")
c2.metric("Mean Net Savings", f"${results['Mean_Savings_Millions']}M")
c3.metric("P90 Confident Savings", f"${results['P90_Savings_Millions']}M")
c4.metric("Worst-Case Loss", f"${results['Max_Risk_Loss_Millions']}M", delta="Risk Scenario", delta_color="inverse")

st.write("")

# Main Workspace Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌐 3D Mission Control", "📊 Monte Carlo Risk Analytics", "⚖️ Operational Trade-offs", "📄 Executive Report"])

with tab1:
    st.subheader("Orbital Routing Architecture (LEO ➔ L1 Hub ➔ Lunar Target)")
    st.plotly_chart(build_3d_orbital_map(), use_container_width=True)
    st.caption("Interactive 3D Map: Left click to rotate, scroll to zoom. Compares staged L1 routing against direct trajectory.")

with tab2:
    st.subheader("Financial Risk Distribution (10,000 Stochastic Scenarios)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df["Direct_Cost_M"], name="Direct Launch Cost ($M)", marker_color='#FF5252', opacity=0.65))
    fig_hist.add_trace(go.Histogram(x=df["L1_Cost_M"], name="Astraia L1 Hub Cost ($M)", marker_color='#00E5FF', opacity=0.65))
    fig_hist.update_layout(
        barmode='overlay',
        xaxis_title="Total Mission Cost in Millions ($)",
        yaxis_title="Simulation Frequency",
        paper_bgcolor='rgb(10, 15, 30)',
        plot_bgcolor='rgb(16, 24, 45)',
        font=dict(color='white'),
        legend=dict(x=0.7, y=0.95)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("Trade-off Analysis: Waiting Time vs Financial Benefit")
    st.write("Demonstrates the impact of boil-off fuel loss and SLA delay penalties as waiting time increases at the L1 Hub.")
    fig_scatter = go.Figure(data=go.Scatter(
        x=df["L1_Wait_Days"],
        y=df["Savings_M"],
        mode='markers',
        marker=dict(size=5, color=df["Savings_M"], colorscale='Tealgrn', showscale=True, colorbar=dict(title="Savings $M"))
    ))
    fig_scatter.update_layout(
        xaxis_title="Days Waiting at L1 Node (Consolidation Delay)",
        yaxis_title="Net Cost Savings ($M)",
        paper_bgcolor='rgb(10, 15, 30)',
        plot_bgcolor='rgb(16, 24, 45)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    st.subheader("Mission Decision Summary & Dataset Export")
    st.dataframe(df.head(100), use_container_width=True)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Monte Carlo Dataset (CSV)",
        data=csv_data,
        file_name="astraia_monte_carlo_results.csv",
        mime="text/csv"
    )
