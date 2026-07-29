"""
EC198 — Post-Combustion Capture (Amine Scrubbing) — F1b Part-Load + Degradation Model

Extends F1a energy model with:
  1. Part-load reboiler duty penalty: off-design L/G ratio increases specific duty.
     q_reb(PLR) = q_design * (a + b*PLR) where a=1.3, b=-0.3 → at PLR=1: q=q_design,
     at PLR=0.3: q = 1.21*q_design (~21% penalty).
  2. Solvent degradation: capacity loss = 0.02 * operating_hours/1000 (2% per 1000h).
     Degraded solvent requires more circulation → higher reboiler duty.
  3. Electrical consumption scales with PLR (fans, pumps) + degradation penalty.
  4. Total energy penalty as fraction of plant output.

Reference:
    Abu-Zahra, M.R.M. et al. (2007). Int. J. Greenhouse Gas Control, 1(1), 37-46.
    Rochelle, G.T. (2009). Science, 325(5948), 1652-1654.
"""

import numpy as np


class AmineCaptureF1b:
    """
    Post-combustion CO2 capture with MEA — part-load + degradation model.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.reboiler_duty_design = u["reboiler_duty_design"]["value"]  # GJ/tCO2
        self.electrical_design = u["electrical_design"]["value"]  # kWh/tCO2
        self.L_G_design = u["L_G_design"]["value"]
        self.solvent_capacity_initial = u["solvent_capacity_initial"]["value"]
        self.degradation_rate = u["solvent_degradation_rate"]["value"]
        self.MW_CO2 = u["MW_CO2"]["value"] / 1000.0  # kg/mol
        self.MW_air = u["MW_air"]["value"] / 1000.0  # kg/mol
        self.PLR_coeffs = u["PLR_reboiler_coeffs"]["value"]

    def solvent_degradation_pct(self, operating_hours):
        """Solvent capacity loss (%) due to thermal/oxidative degradation.
        capacity_loss = degradation_rate * hours/1000 * 100
        """
        hours = np.asarray(operating_hours, dtype=float)
        return np.clip(self.degradation_rate * hours / 1000.0 * 100.0, 0.0, 50.0)

    def _solvent_capacity_factor(self, operating_hours):
        """Remaining solvent capacity as fraction of initial.
        Lower capacity → need more circulation → higher reboiler duty.
        """
        deg_pct = self.solvent_degradation_pct(operating_hours)
        return np.clip(1.0 - deg_pct / 100.0, 0.5, 1.0)

    def _plr_reboiler_factor(self, plr):
        """Part-load reboiler duty multiplier.
        At part-load, L/G ratio deviates from optimal → higher specific duty.
        factor = a + b*PLR → at PLR=1: factor=1.0, at PLR=0.3: factor~1.21
        """
        plr = np.asarray(plr, dtype=float)
        a, b = self.PLR_coeffs
        return a + b * plr

    def reboiler_duty_gj_ton(self, capture_rate, plr, operating_hours):
        """Specific reboiler duty (GJ/tCO2) accounting for part-load and degradation.

        q = q_design * PLR_factor * (1/capacity_factor) * capture_rate_factor
        """
        cr = np.asarray(capture_rate, dtype=float)
        plr_f = self._plr_reboiler_factor(plr)
        cap_f = self._solvent_capacity_factor(operating_hours)

        # Capture rate correction (F1a-style): higher capture rate increases duty
        dCR = cr - 0.90
        cr_factor = 1.0 + 2.5 * dCR ** 2 + 1.5 * dCR

        q = self.reboiler_duty_design * plr_f * (1.0 / cap_f) * cr_factor
        return np.clip(q, 2.0, 8.0)

    def electrical_kwh_ton(self, capture_rate, plr, operating_hours):
        """Specific electrical consumption (kWh/tCO2).
        Fans scale with PLR, pumps scale with degradation.
        """
        cr = np.asarray(capture_rate, dtype=float)
        plr = np.asarray(plr, dtype=float)
        cap_f = self._solvent_capacity_factor(operating_hours)

        # Fan power scales inversely with PLR (fixed speed fans at part-load)
        fan_factor = 0.7 + 0.3 / (plr + 1e-6)
        # Pump power scales with 1/capacity (more circulation needed)
        pump_factor = 1.0 / cap_f
        # Compression scales with capture rate
        comp_factor = 1.0 + 0.5 * (cr - 0.90)

        E = self.electrical_design * fan_factor * pump_factor * comp_factor
        return np.clip(E, 20.0, 200.0)

    def co2_captured_kg_h(self, flue_gas_flow_mol_s, co2_concentration, capture_rate, plr):
        """CO2 captured (kg/h).
        CO2_flow = flue_gas * x_CO2 * MW_CO2 * capture_rate * PLR
        """
        fg = np.asarray(flue_gas_flow_mol_s, dtype=float)
        xCO2 = np.asarray(co2_concentration, dtype=float)
        cr = np.asarray(capture_rate, dtype=float)
        plr = np.asarray(plr, dtype=float)

        co2_mol_s = fg * xCO2 * plr
        co2_kg_s = co2_mol_s * self.MW_CO2
        co2_captured = co2_kg_s * cr * 3600.0  # kg/h
        return co2_captured

    def total_energy_penalty_pct(self, capture_rate, plr, operating_hours):
        """Total energy penalty as % of reference plant output.
        Assumes 500 MW reference plant.
        Penalty = (Q_reb [GJ/t] * CO2_rate [t/h] / 3.6 + E_elec [kWh/t] * CO2_rate / 1000) / P_plant * 100
        Simplified: penalty scales with specific energy relative to design.
        """
        q_reb = self.reboiler_duty_gj_ton(capture_rate, plr, operating_hours)
        e_elec = self.electrical_kwh_ton(capture_rate, plr, operating_hours)
        # Total specific energy in GJ/tCO2 (convert electrical: kWh -> GJ: /277.78)
        e_total = q_reb + e_elec / 277.78
        # Reference total at design: ~3.5 + 40/277.78 = ~3.644 GJ/tCO2
        e_ref = self.reboiler_duty_design + self.electrical_design / 277.78
        # Typical penalty at design ~ 25-30% for coal plant
        penalty_design = 28.0  # %
        penalty = penalty_design * e_total / e_ref
        return np.clip(penalty, 10.0, 60.0)

    def compute(self, flue_gas_flow_mol_s, co2_concentration, capture_rate, plr,
                operating_hours):
        """Full computation returning all outputs."""
        co2 = self.co2_captured_kg_h(flue_gas_flow_mol_s, co2_concentration,
                                     capture_rate, plr)
        q_reb = self.reboiler_duty_gj_ton(capture_rate, plr, operating_hours)
        e_elec = self.electrical_kwh_ton(capture_rate, plr, operating_hours)
        deg = self.solvent_degradation_pct(operating_hours)
        penalty = self.total_energy_penalty_pct(capture_rate, plr, operating_hours)

        return {
            "co2_captured_kg_h": co2,
            "reboiler_duty_gj_ton": q_reb,
            "electrical_kwh_ton": e_elec,
            "solvent_degradation_pct": deg,
            "total_energy_penalty_pct": penalty,
        }
