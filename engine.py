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
            }
        }

# ==========================================
# 4. ΕΚΤΕΛΕΣΗ (Mock API Call)
# ==========================================
if __name__ == "__main__":
    # 1. Ορίζουμε το διαστημόπλοιο (π.χ. Φορτίο 100 τόνων)
    spacecraft = SpacecraftSpecs(payload_kg=100_000, isp_seconds=450)
    
    # 2. Ορίζουμε την αγορά (π.χ. $5,000/kg LEO, 20% έκπτωση στον πελάτη)
    market_env = MarketConditions(base_launch_cost_per_kg=5000, saas_discount_margin=0.20)
    
    # 3. Αρχικοποιούμε τη μηχανή
    engine = AstraiaRoutingEngine(specs=spacecraft, market=market_env)
    
    # 4. Εξάγουμε το μανιφέστο (Αποτέλεσμα Δρομολόγησης)
    try:
        report = engine.generate_mission_manifest()
        import json
        print(json.dumps(report, indent=4, ensure_ascii=False))
    except (OrbitalPhysicsError, EconomicViabilityError) as e:
        print(f"Σφάλμα Αποστολής: {e}")
