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
st.markdown("<span style='color:#8B949E; font-size:1.1rem; letter-
