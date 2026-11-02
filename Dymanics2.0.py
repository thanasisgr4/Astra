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
        # Θερμοδυναμική εξάτμιση
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
st.markdown("<span style='color:#8B949E; font-size:1.1rem; letter-spacing: 1px;'>ORBITAL LOGISTICS & ROUTING ENGINE (v7.0 - Integrated Core)</span>", unsafe_allow_html=True)
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
# Τροφοδοτούμε την Enterprise Μηχανή μας με τα δεδομένα από το Sidebar
specs = SpacecraftSpecs(payload_kg=annual_tons * 1000, isp_seconds=isp)
market = MarketConditions(
    base_launch_cost_per_kg=base_launch_cost, 
    saas_
