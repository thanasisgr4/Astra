import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

# --- 1. BACKEND ENGINE (Αστροδυναμική & Monte Carlo) ---
@dataclass
class SpacecraftSpecs:
    dry_mass_kg: float
    isp_s: float
    base_boil_off_rate: float

@dataclass
class MarketConditions:
    base_fuel_cost_kg: float
    delay_penalty_per_day: float

class AstraiaMonteCarloEngine:
    def __init__(self, iterations: int = 10000):
        self.iterations = iterations
        self.g0 = 9.80665 

    def simulate_architectures(self, sc: SpacecraftSpecs, market: MarketConditions) -> Dict:
        # Κατανομές
        dv_direct = np.random.normal(3800, 150, self.iterations) 
        dv_l1 = np.random.normal(3100, 100, self.iterations) 
        wait_days_l1 = np.random.triangular(5, 15, 30, self.iterations) 
        fuel_price = np.random.uniform(market.base_fuel_cost_kg * 0.9, market.base_fuel_cost_kg * 1.5, self.iterations)
        boil_off_rate = np.random.normal(sc.base_boil_off_rate, 0.0002, self.iterations)

        # Υπολογισμοί Tsiolkovsky
        mass_ratio_direct = np.exp(dv_direct / (sc.isp_s * self.g0))
        fuel_direct = sc.dry_mass_kg * (mass_ratio_direct - 1)
        cost_direct = fuel_direct * fuel_price

        mass_ratio_l1 = np.exp(dv_l1 / (sc.isp_s * self.g0))
        fuel_l1_base = sc.dry_mass_kg * (mass_ratio_l1 - 1)
        
        # Trade-offs (Boil-off & Penalties)
        boil_off_loss = fuel_l1_base * (1 - (1 - boil_off_rate)**wait_days_l1)
        total_fuel_l1 = fuel_l1_base + boil_off_loss
        cost_l1 = (total_fuel_l1 * fuel_price) + (wait_days_l1 * market.delay_penalty_per_day)

        savings = cost_direct - cost_l1
        prob_l1_better = (np.sum(savings > 0) / self.iterations) * 100

        results_df = pd.DataFrame({
            "Direct_Cost_M": cost_direct / 1e6,
            "L1_Cost_M": cost_l1 / 1e6,
            "Savings_M": savings / 1e6,
            "L1_Wait_Days": wait_days_l1
        })

        return {
            "L1_Success_Probability": round(prob_l1_better, 2),
            "Mean_Savings_Millions": round(np.mean(savings) / 1e6, 2),
            "Max_Risk_Loss_Millions": round(np.min(savings) / 1e6, 2),
            "Data": results_df
        }

# --- 2. FRONTEND UI (Streamlit Dashboard) ---
st.set_page_config(page_title="Astraia OS | L1 Logistics", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Astraia OS: Orbital Logistics Dashboard</h1>", unsafe_allow_html=True)

st.sidebar.header("Mission Parameters")
dry_mass = st.sidebar.number_input("Spacecraft Dry Mass (kg)", value=5000, step=100)
isp = st.sidebar.number_input("Propulsion Isp (sec)", value=450, step=10)
fuel_price = st.sidebar.number_input("Fuel Cost ($/kg)", value=5000, step=500)
delay_penalty = st.sidebar.number_input("Delay Penalty ($/day)", value=100000, step=10000)

if st.sidebar.button("Run Stochastic Optimization"):
    with st.spinner("Executing 10,000 Monte Carlo simulations..."):
        sc_specs = SpacecraftSpecs(dry_mass_kg=dry_mass, isp_s=isp, base_boil_off_rate=0.001)
        market_env = MarketConditions(base_fuel_cost_kg=fuel_price, delay_penalty_per_day=delay_penalty)
        
        engine = AstraiaMonteCarloEngine(iterations=10000)
        report = engine.simulate_architectures(sc_specs, market_env)
        df_results = report["Data"]

        st.markdown("### Executive Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("L1 Success Probability", f"{report['L1_Success_Probability']}%")
        col2.metric("Mean Savings", f"${report['Mean_Savings_Millions']}M")
        col3.metric("Worst-Case Exposure", f"${report['Max_Risk_Loss_Millions']}M", delta="Risk", delta_color="inverse")

        st.divider()

        st.markdown("### Financial Risk Distribution (Direct vs L1)")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_results["Direct_Cost_M"], name="Direct Launch Cost", marker_color='red', opacity=0.6))
        fig.add_trace(go.Histogram(x=df_results["L1_Cost_M"], name="L1 Hub Cost", marker_color='blue', opacity=0.6))
        fig.update_layout(barmode='overlay', xaxis_title_text='Cost in Millions ($)', yaxis_title_text='Probability (Frequency)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Operational Trade-off: Wait Time vs Savings")
        fig_scatter = go.Figure(data=go.Scatter(
            x=df_results["L1_Wait_Days"],
            y=df_results["Savings_M"],
            mode='markers',
            marker=dict(size=4, color=df_results["Savings_M"], colorscale='Viridis', showscale=True)
        ))
        fig_scatter.update_layout(xaxis_title="Days Waiting at L1 Node", yaxis_title="Total Savings ($M)")
        st.plotly_chart(fig_scatter, use_container_width=True)
