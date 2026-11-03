import streamlit as st
import plotly.graph_objects as go
# Εδώ κάνουμε import τον κινητήρα που γράψαμε στο προηγούμενο αρχείο (έστω astraia_engine.py)
from astraia_engine import SpacecraftSpecs, MarketConditions, AstraiaMonteCarloEngine

# 1. SETUP ΣΕΛΙΔΑΣ (Enterprise Look)
st.set_page_config(page_title="Astraia OS | L1 Logistics", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Astraia OS: Orbital Logistics Dashboard</h1>", unsafe_allow_html=True)

# 2. SIDEBAR - ΠΑΡΑΜΕΤΡΟΙ ΠΕΛΑΤΗ (Inputs)
st.sidebar.header("Mission Parameters")
dry_mass = st.sidebar.number_input("Spacecraft Dry Mass (kg)", value=5000, step=100)
isp = st.sidebar.number_input("Propulsion Isp (sec)", value=450, step=10)
fuel_price = st.sidebar.number_input("Fuel Cost ($/kg)", value=5000, step=500)
delay_penalty = st.sidebar.number_input("Delay Penalty ($/day)", value=100000, step=10000)

# 3. ΕΚΤΕΛΕΣΗ ΤΟΥ ΚΙΝΗΤΗΡΑ (Backend Call)
if st.sidebar.button("Run Stochastic Optimization"):
    with st.spinner("Executing 10,000 Monte Carlo simulations..."):
        
        # Τροφοδοτούμε τα δεδομένα στα Data Classes
        sc_specs = SpacecraftSpecs(dry_mass_kg=dry_mass, isp_s=isp, base_boil_off_rate=0.001)
        market_env = MarketConditions(base_fuel_cost_kg=fuel_price, delay_penalty_per_day=delay_penalty)
        
        # Καλούμε τη μηχανή προσομοίωσης
        engine = AstraiaMonteCarloEngine(iterations=10000)
        report = engine.simulate_architectures(sc_specs, market_env)
        df_results = report["Data"]

        # 4. ΑΠΕΙΚΟΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ (Metrics & KPIs)
        st.markdown("### Executive Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("L1 Success Probability", f"{report['L1_Success_Probability']}%")
        col2.metric("Mean Savings", f"${report['Mean_Savings_Millions']}M")
        col3.metric("Worst-Case Exposure", f"${report['Max_Risk_Loss_Millions']}M", delta="Risk", delta_color="inverse")

        st.divider()

        # 5. ΟΠΤΙΚΟΠΟΙΗΣΗ ΜΕ PLOTLY (Risk Distribution)
        st.markdown("### Financial Risk Distribution (Direct vs L1)")
        fig = go.Figure()
        
        # Κατανομή Κόστους για Απευθείας Εκτόξευση
        fig.add_trace(go.Histogram(
            x=df_results["Direct_Cost_M"], 
            name="Direct Launch Cost", 
            marker_color='red', 
            opacity=0.6
        ))
        
        # Κατανομή Κόστους για L1 Hub
        fig.add_trace(go.Histogram(
            x=df_results["L1_Cost_M"], 
            name="L1 Hub Cost", 
            marker_color='blue', 
            opacity=0.6
        ))

        fig.update_layout(
            barmode='overlay',
            xaxis_title_text='Cost in Millions ($)',
            yaxis_title_text='Probability (Frequency)',
            title="Monte Carlo Cost Assessment (10,000 Scenarios)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 6. TRADE-OFF ANALYSIS (Boil-off vs Penalty)
        st.markdown("### Operational Trade-off: Wait Time vs Savings")
        fig_scatter = go.Figure(data=go.Scatter(
            x=df_results["L1_Wait_Days"],
            y=df_results["Savings_M"],
            mode='markers',
            marker=dict(size=4, color=df_results["Savings_M"], colorscale='Viridis', showscale=True)
        ))
        fig_scatter.update_layout(
            xaxis_title="Days Waiting at L1 Node",
            yaxis_title="Total Savings ($M)",
            title="Impact of Consolidation Delay on Financial Performance"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
