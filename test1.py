import streamlit as st
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

    def simulate_architectures(self, sc: SpacecraftSpecs, market: MarketConditions) -> Dict:
        np.random.seed(42)  # Σταθερότητα για επαναληψιμότητα παρουσίασης
        
        # 1. Στοχαστικές Κατανομές (Uncertainty Distribution)
        dv_direct = np.random.normal(3800, 150, self.iterations)
        dv_l1 = np.random.normal(3100, 100, self.iterations)
        wait_days_l1 = np.random.triangular(5, 15, 30, self.iterations)
        fuel_price = np.random.uniform(market.base_fuel_cost_kg * 0.9, market.base_fuel_cost_kg * 1.4, self.iterations)
        boil_off_rate = np.random.normal(sc.base_boil_off_rate, 0.0002, self.iterations)
        boil_off_rate = np.clip(boil_off_rate, 0.0001, 0.005)

        # 2. Φυσική Tsiolkovsky (Rocket Equation)
        mass_ratio_direct = np.exp(dv_direct / (sc.isp_s * self.g0))
        fuel_direct = sc.dry_mass_kg * (mass_ratio_direct - 1)
        cost_direct = fuel_direct * fuel_price

        mass_ratio_l1 = np.exp(dv_l1 / (sc.isp_s * self.g0))
        fuel_l1_base = sc.dry_mass_kg * (mass_ratio_l1 - 1)
        
        # 3. Μοντελοποίηση Trade-offs (Boil-off vs SLA Delay Penalties)
        boil_off_loss = fuel_l1_base * (1 - (1 - boil_off_rate)**wait_days_l1)
        total_fuel_l1 = fuel_l1_base + boil_off_loss
        cost_l1 = (total_fuel_l1 * fuel_price) + (wait_days_l1 * market.delay_penalty_per_day)

        # 4. Στατιστική Ανάλυση & Metrics
        savings = cost_direct - cost_l1
        prob_l1_better = (np.sum(savings > 0) / self.iterations) * 100

        results_df = pd.DataFrame({
            "Direct_
