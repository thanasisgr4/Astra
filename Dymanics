import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math

# ==========================================
# 1. CORE CONFIGURATION & CUSTOM CSS (Enterprise UI)
# ==========================================
st.set_page_config(page_title="Astraia OS | Enterprise Engine", layout="wide", page_icon="🌌")

st.markdown("""
    <style>
        /* Space-Tech UI Theme */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #F8FAFC !important; font-family: 'Inter', sans-serif; }
        .stMetric { 
            background-color: #0F172A; 
            padding: 20px; 
            border-radius: 12px; 
            border-left: 5px solid #3B82F6;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        }
        .stMetric label { color: #94A3B8 !important; font-size: 1.1rem; }
        .stMetric div { color: #E2E8F0 !important; }
        hr { border-color: #334155; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Astraia OS : Mission Control & Logistics Engine")
st.markdown("*Advanced Orbital Routing, Cryogenic Physics & Stochastic Financial Modeling*")
st.divider()

# ==========================================
# 2. SIDEBAR: MISSION PARAMETERS (Input Engine)
# ==========================================
with st.sidebar:
    st.header("⚙️ Παράμετροι Αποστολής")
    
    st.subheader("1. Ζήτηση & Αγορά")
    annual_tons = st.slider("Ετήσιο Φορτίο (Τόνοι)", 10, 500, 100, 10)
    target_discount = st.slider("Έκπτωση Πελάτη (%)", 5.0, 40.0, 20.0, 1.0)
    base_launch_cost = st.number_input("Βασικό Κόστος LEO ($/kg)", value=5000, step=500)
    
    st.subheader("2. Φυσική & Κρυογονική")
    isp = st.slider("Ειδική Ώση (Isp - sec)", 300, 500, 450, 10, help="LOX/LH2 Κινητήρες")
    boil_off_rate = st.slider("Ρυθμός Εξάτμισης (% / ημέρα)", 0.0, 5.0, 0.5, 0.1)
    wait_days = st.slider("Ημέρες Αναμονής στο L1", 0, 30, 5)
    
    st.subheader("3. Υποδομές & Ρίσκο")
    module_capacity = st.number_input("Max Χωρητικότητα Module (t)", value=50)
    insurance_rate = st.slider("Ασφάλιστρο & Contingency (%)", 0, 30, 15, 5)

# ==========================================
# 3. MATHEMATICAL & PHYSICS MODELS
# ==========================================
# Σταθερές
g0 = 9.81
dry_ratio = 0.10
dv_direct = 5900
dv_stage1 = 3800

# Εξίσωση Tsiolkovsky
def get_mass_ratio(dv, specific_impulse):
    return math.exp(dv / (specific_impulse * g0))

# Βασικοί Υπολογισμοί Μάζας
payload_kg = annual_tons * 1000
m0_direct = payload_kg * (1 + (get_mass_ratio(dv_direct, isp) - 1) * (1 + dry_ratio))

# Υπολογισμός με Boil-off penalty
propellant_needed_relay = payload_kg * (get_mass_ratio(dv_stage1, isp) - 1)
boil_off_penalty = propellant_needed_relay * ((1 - (boil_off_rate/100))**wait_days)
propellant_launch_relay = propellant_needed_relay + (propellant_needed_relay - boil_off_penalty)
m0_relay = payload_kg + propellant_launch_relay * (1 + dry_ratio)

mass_savings_pct = (1 - (m0_relay / m0_direct)) * 100

# ==========================================
# 4. FINANCIAL & SCALING MODELS
# ==========================================
required_modules = math.ceil(annual_tons / module_capacity)
capex_per_module = 250_000_000
total_capex = (required_modules * capex_per_module) + 15_000_000 # +15M IP & Certification

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
# 5. UI TABS & VISUALIZATIONS (The Dashboard)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Executive Dashboard", "🪐 Orbital Physics & Mass", "🎲 Monte Carlo Risk Analysis"])

with tab1:
    st.subheader("Βασικοί Δείκτες Απόδοσης (Risk-Adjusted KPIs)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Εξοικονόμηση Μάζας", f"{mass_savings_pct:.1f}%", f"Έναντι {m0_direct/1000:.0f}t (Direct)")
    c2.metric("Καθαρό Κέρδος (Astraia)", f"${net_profit/1e6:.1f}M", f"Περιθώριο {margin:.1f}%")
    c3.metric("Συνολικό CapEx", f"${total_capex/1e6:.1f}M", f"{required_modules} L1 Modules")
    c4.metric("ROI (Απόσβεση)", f"{payback:.1f} έτη" if payback > 0 else "N/A", "Μετά την ασφάλιση")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Πρωτοποριακό γράφημα Waterfall για τα οικονομικά
    fig_waterfall = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["Έσοδα (Πελάτης)", "Κόστος Εκτόξευσης (Astraia)", "OpEx (Λειτουργικά)", "Risk Reserve (Ασφάλεια)", "Καθαρό Κέρδος"],
        textposition="outside",
        y=[astraia_revenue/1e6, -astraia_launch_cost/1e6, -total_opex/1e6, -risk_reserve/1e6, net_profit/1e6],
        connector={"line":{"color":"#475569"}},
        decreasing={"marker":{"color":"#EF4444"}},
        increasing={"marker":{"color":"#10B981"}},
        totals={"marker":{"color":"#3B82F6"}}
    ))
    fig_waterfall.update_layout(title="Ροή Κερδοφορίας ($ Εκατομμύρια)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
    st.plotly_chart(fig_waterfall, use_container_width=True)

with tab2:
    st.subheader("Δυναμική Μάζας & Φαινόμενο Boil-Off")
    colA, colB = st.columns(2)
    
    with colA:
        st.write("Ανάλυση Εξάτμισης Κρυογονικών:")
        st.info(f"Για παραμονή **{wait_days} ημερών** στο L1 με ρυθμό εξάτμισης **{boil_off_rate}%/ημέρα**, το σύστημα αντισταθμίζει απώλεια **{boil_off_penalty/1000:.1f} τόνων** καυσίμου προσθέτοντάς τα στην αρχική εκτόξευση.")
        st.write(f"- Απαιτούμενη Μάζα Γήινης Εκτόξευσης (Direct): **{m0_direct/1000:.1f} t**")
        st.write(f"- Απαιτούμενη Μάζα Γήινης Εκτόξευσης (Astraia L1): **{m0_relay/1000:.1f} t**")
        
    with colB:
        # Διάγραμμα Σύγκρισης Μάζας
        df_mass = pd.DataFrame({
            "Στάδιο": ["Payload", "Δομή (Dry Mass)", "Καύσιμα (Propellant)", "Boil-off Αναπλήρωση"],
            "Direct (t)": [payload_kg/1000, (m0_direct-payload_kg)*dry_ratio/1000, (m0_direct-payload_kg)*(1-dry_ratio)/1000, 0],
            "Astraia L1 (t)": [payload_kg/1000, (m0_relay-payload_kg-boil_off_penalty)*dry_ratio/1000, (m0_relay-payload_kg-boil_off_penalty)*(1-dry_ratio)/1000, boil_off_penalty/1000]
        })
        fig_mass = px.bar(df_mass, x="Στάδιο", y=["Direct (t)", "Astraia L1 (t)"], barmode="group", color_discrete_sequence=["#EF4444", "#3B82F6"])
        fig_mass.update_layout(title="Σύγκριση Απαιτούμενης Μάζας Ανά Κατηγορία", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
        st.plotly_chart(fig_mass, use_container_width=True)

with tab3:
    st.subheader("Στοχαστική Προσομοίωση Μόντε Κάρλο (1.000 Σενάρια)")
    st.write("Στην πραγματική διαστημική αγορά, το κόστος εκτόξευσης και το λειτουργικό κόστος υπόκεινται σε διακυμάνσεις. Ο αλγόριθμος τρέχει 1.000 εναλλακτικά σύμπαντα (scenarios) για να δοκιμάσει την αντοχή της εταιρείας.")
    
    if st.button("▶️ Εκτέλεση Προσομοίωσης Monte Carlo"):
        with st.spinner("Υπολογισμός 1.000 σεναρίων..."):
            # Προσομοίωση 1000 σεναρίων
            sim_launch_costs = np.random.normal(base_launch_cost, base_launch_cost*0.15, 1000) # Διακύμανση 15%
            sim_opex_multiplier = np.random.normal(1.0, 0.20, 1000) # Διακύμανση OpEx 20%
            
            sim_profits = []
            for i in range(1000):
                s_direct_cost = m0_direct * sim_launch_costs[i]
                s_revenue = s_direct_cost * (1 - (target_discount / 100.0))
                s_astraia_cost = m0_relay * sim_launch_costs[i]
                s_total_cost = s_astraia_cost + (total_opex * sim_opex_multiplier[i]) + risk_reserve
                sim_profits.append((s_revenue - s_total_cost) / 1e6)
                
            sim_profits = np.array(sim_profits)
            success_rate = (sim_profits > 0).mean() * 100
            
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Πιθανότητα Επιβίωσης (Κέρδος > 0)", f"{success_rate:.1f}%")
            c_res2.metric("Μέσο (Αναμενόμενο) Κέρδος", f"${sim_profits.mean():.1f}M")
            c_res3.metric("Χειρότερο Σενάριο (Max Ζημιά)", f"${sim_profits.min():.1f}M")
            
            fig_hist = px.histogram(sim_profits, nbins=50, title="Κατανομή Πιθανού Κέρδους (Σε $ Εκατομμύρια)", color_discrete_sequence=["#10B981"])
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Νεκρό Σημείο")
            fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
            st.plotly_chart(fig_hist, use_container_width=True)
            
            if success_rate > 85:
                st.success("✅ Εξαιρετικά ανθεκτικό μοντέλο. Η εταιρεία επιβιώνει σχεδόν σε όλες τις αναταράξεις της αγοράς.")
            else:
                st.warning("⚠️ Υψηλό ρίσκο. Συνιστάται η μείωση της έκπτωσης στον πελάτη ή η αύξηση της χωρητικότητας του σταθμού.")
