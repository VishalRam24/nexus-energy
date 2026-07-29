"""
EC206 — CO2 Mineralization (Accelerated Carbonation) — F1b Part-Load + Reactant Degradation

Extends F1a capture-rate/energy model with:
  1. Conversion efficiency degradation: surface passivation over time.
     eta(t) = eta0 * (1 - k_deact * t), floored at eta_min = 0.4.
     Rate: ~0.03% per operating hour (surface fouling with carbonate layer).
  2. Part-load SEC penalty: at lower throughput, grinding + slurry prep
     specific energy increases. SEC(PLR) = SEC_design * (a + b/PLR).
  3. CO2 permanently stored as mineral carbonate (MgCO3/CaCO3).
  4. Mineral feed requirement scales with conversion efficiency.
  5. By-product (carbonate) production rate.

References:
    Sanna, A. et al. (2014). Prog. Energy Combust. Sci., 44, 40-82.
    Huijgen, W.J.J. & Comans, R.N.J. (2005). ECN-C--05-022 report.
    Lackner, K.S. et al. (1995). Energy, 20(11), 1153-1170.
"""

import numpy as np


class CO2MineralizationF1b:
    """CO2 accelerated mineral carbonation — part-load + conversion degradation model."""

    ETA_MIN = 0.40  # minimum achievable conversion (permanent surface fouling)

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta0 = u["conversion_efficiency_design"]["value"]
        self.SEC_design = u["SEC_design"]["value"]              # kWh/tCO2
        self.E_grind = u["grinding_energy"]["value"]            # kWh/t_mineral
        self.T_rxn = u["reaction_T_degC"]["value"]
        self.P_rxn = u["reaction_P_bar"]["value"]
        self.stoich = u["stoichiometric_ratio"]["value"]        # t_mineral/tCO2
        self.k_deg = u["k_deactivation"]["value"]               # per hour
        self.PLR_coeffs = u["PLR_energy_coeffs"]["value"]
        self.MW_CO2 = u["MW_CO2"]["value"]                      # g/mol

    # ── Conversion efficiency ──────────────────────────────────────────────────

    def conversion_efficiency(self, operating_hours):
        """Mineral conversion efficiency eta(t) = eta0 * (1 - k_deg * t)."""
        t = np.asarray(operating_hours, dtype=float)
        eta = self.eta0 * (1.0 - self.k_deg * t)
        return np.clip(eta, self.ETA_MIN, self.eta0)

    def conversion_relative_pct(self, operating_hours):
        """Conversion as % of fresh value."""
        return self.conversion_efficiency(operating_hours) / self.eta0 * 100.0

    # ── Energy ────────────────────────────────────────────────────────────────

    def _plr_sec_factor(self, plr):
        """SEC part-load factor."""
        plr = np.clip(np.asarray(plr, dtype=float), 0.1, 1.0)
        a, b = self.PLR_coeffs
        return a + b * plr

    def sec_kwh_tco2(self, plr, operating_hours):
        """Specific energy consumption (kWh/tCO2 captured).
        As conversion drops, more mineral must be processed per tCO2 → higher grinding energy.
        """
        plr_f = self._plr_sec_factor(plr)
        eta = self.conversion_efficiency(operating_hours)
        # Mineral feed per tCO2 = stoich/eta (more feed needed at low efficiency)
        mineral_factor = (self.stoich / self.eta0) / (self.stoich / eta)
        # Correction: higher mineral per tCO2 → more grinding
        mineral_per_tco2 = self.stoich / eta
        mineral_per_tco2_design = self.stoich / self.eta0
        grind_correction = mineral_per_tco2 / mineral_per_tco2_design

        SEC = self.SEC_design * plr_f * grind_correction
        return np.clip(SEC, 100.0, 1000.0)

    def mineral_feed_t_per_tco2(self, operating_hours):
        """Mineral feed required (t mineral / tCO2 captured)."""
        eta = self.conversion_efficiency(operating_hours)
        return self.stoich / eta

    def co2_stored_kg_h(self, co2_flow_kg_h, plr, operating_hours):
        """CO2 permanently stored as mineral carbonate (kg/h).
        Accounts for actual conversion efficiency and PLR.
        """
        flow = np.asarray(co2_flow_kg_h, dtype=float)
        plr = np.asarray(plr, dtype=float)
        eta = self.conversion_efficiency(operating_hours)
        return flow * plr * eta

    def carbonate_product_kg_h(self, co2_flow_kg_h, plr, operating_hours):
        """Carbonate by-product produced (kg/h).
        MgCO3: MW = 84.31, CO2 fraction = 44.01/84.31 = 0.522
        Mass of carbonate = co2_stored / 0.522
        """
        co2_stored = self.co2_stored_kg_h(co2_flow_kg_h, plr, operating_hours)
        co2_fraction = 44.01 / 84.31
        return co2_stored / co2_fraction

    def compute(self, co2_flow_kg_h, plr, operating_hours):
        """Full computation returning all outputs."""
        eta = self.conversion_efficiency(operating_hours)
        eta_pct = self.conversion_relative_pct(operating_hours)
        sec = self.sec_kwh_tco2(plr, operating_hours)
        co2_stored = self.co2_stored_kg_h(co2_flow_kg_h, plr, operating_hours)
        carbonate = self.carbonate_product_kg_h(co2_flow_kg_h, plr, operating_hours)
        mineral_feed = self.mineral_feed_t_per_tco2(operating_hours)

        return {
            "co2_stored_kg_h": co2_stored,
            "sec_kwh_tco2": sec,
            "conversion_efficiency": eta,
            "conversion_relative_pct": eta_pct,
            "carbonate_product_kg_h": carbonate,
            "mineral_feed_t_per_tco2": mineral_feed,
        }
