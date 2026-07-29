"""
EC205 — CO2 Electrolyzer (CO2RR to CO/Fuels) — F1b Part-Load + Electrode Degradation Model

Extends F1a capture-rate/energy model with:
  1. Electrode degradation: FE(t) = FE0 * (1 - k_deg * t)
     Faradaic efficiency degrades ~1% per 1000 operating hours.
  2. Part-load voltage penalty: V(PLR) = V_design * (a + b/PLR)
     At lower current density, overpotentials change (both Butler-Volmer and ohmic).
  3. Specific energy consumption = V_cell * n_e * F / (MW_CO * FE) [kWh/tCO].
  4. CO production rate scales with current, FE, and electrode area.
  5. Energy penalty from FE loss and voltage penalty captured jointly.

References:
    Jouny, M. et al. (2018). Ind. Eng. Chem. Res., 57(6), 2165-2177.
    Higgins, D. et al. (2019). Nature Energy, 4, 522-528.
    Bushuyev, O.S. et al. (2018). Joule, 2(5), 825-832.
"""

import numpy as np

FARADAY = 96485.0  # C/mol


class CO2ElectrolyzerF1b:
    """CO2 electrolyzer to CO — part-load + electrode degradation model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_design = u["V_cell_design"]["value"]         # V
        self.j_design = u["j_design"]["value"]              # mA/cm2
        self.FE0 = u["faradaic_efficiency_design"]["value"]
        self.SEC_design = u["SEC_design"]["value"]          # kWh/tCO
        self.k_deg = u["degradation_rate"]["value"]         # per hour
        self.V_slope = u["V_cell_slope"]["value"]           # V/(mA/cm2)
        self.n_e = u["n_electrons_CO"]["value"]
        self.MW_CO = u["MW_CO"]["value"] / 1000.0          # kg/mol
        self.MW_CO2 = u["MW_CO2"]["value"] / 1000.0        # kg/mol
        self.A_cm2 = u["electrode_area_cm2"]["value"]
        self.PLR_coeffs = u["PLR_voltage_coeffs"]["value"]

    # ── Degradation & voltage ──────────────────────────────────────────────────

    def faradaic_efficiency(self, operating_hours):
        """FE(t) = FE0 * (1 - k_deg * t), floored at 50%."""
        t = np.asarray(operating_hours, dtype=float)
        FE = self.FE0 * (1.0 - self.k_deg * t)
        return np.clip(FE, 0.50, self.FE0)

    def cell_voltage(self, plr):
        """Cell voltage at part-load.
        V(PLR) = V_design * (a + b/PLR)
        At PLR=1: V = V_design; at lower PLR current: voltage drops slightly
        but SEC per product increases due to lower throughput.
        At higher PLR (above 1 capped): overpotentials rise.
        """
        plr = np.clip(np.asarray(plr, dtype=float), 0.1, 1.0)
        a, b = self.PLR_coeffs
        V = self.V_design * (a + b / plr)
        return np.clip(V, 1.5, 5.0)

    def current_density(self, plr):
        """Effective current density [mA/cm2] = j_design * PLR."""
        plr = np.asarray(plr, dtype=float)
        return self.j_design * plr

    # ── Production and energy ─────────────────────────────────────────────────

    def co_production_rate_g_h(self, plr, operating_hours):
        """CO production rate [g/h] from electrode area.
        m_CO = I * FE * MW_CO / (n_e * F)  [mol/s] * MW → [kg/s]
        """
        plr = np.asarray(plr, dtype=float)
        t = np.asarray(operating_hours, dtype=float)
        j = self.current_density(plr)          # mA/cm2
        I_mA = j * self.A_cm2                  # mA total
        I_A = I_mA / 1000.0                    # A
        FE = self.faradaic_efficiency(t)
        # mol CO per second
        n_mol_co_s = I_A * FE / (self.n_e * FARADAY)
        # g/h
        return n_mol_co_s * self.MW_CO * 1e3 * 3600.0

    def sec_kwh_t_co(self, plr, operating_hours):
        """Specific energy consumption [kWh/tCO].
        SEC = V_cell * n_e * F / (MW_CO * FE) [J/kg] / 3.6e6 * 1e6
        """
        plr = np.asarray(plr, dtype=float)
        t = np.asarray(operating_hours, dtype=float)
        V = self.cell_voltage(plr)
        FE = self.faradaic_efficiency(t)
        # J/mol CO → kWh/tCO
        energy_J_mol = V * self.n_e * FARADAY / FE
        # kWh/mol / (MW_CO kg/mol) * 1000 kg/t / 3.6e6 J/kWh
        sec = energy_J_mol / (self.MW_CO * 1e3) / 3.6e6 * 1e6
        # Clip to physical range: 200-2000 kWh/tCO
        return np.clip(sec, 200.0, 2000.0)

    def co2_consumed_kg_h(self, plr, operating_hours):
        """CO2 consumed [kg/h] = CO_produced * MW_CO2/MW_CO."""
        co_g_h = self.co_production_rate_g_h(plr, operating_hours)
        return co_g_h * self.MW_CO2 / (self.MW_CO * 1e3) * self.MW_CO2 / self.MW_CO2

    def fe_degradation_pct(self, operating_hours):
        """Faradaic efficiency relative to fresh [%]."""
        t = np.asarray(operating_hours, dtype=float)
        FE = self.faradaic_efficiency(t)
        return FE / self.FE0 * 100.0

    def compute(self, plr, operating_hours):
        """Full computation returning all outputs."""
        co_rate = self.co_production_rate_g_h(plr, operating_hours)
        sec = self.sec_kwh_t_co(plr, operating_hours)
        FE = self.faradaic_efficiency(operating_hours)
        V = self.cell_voltage(plr)
        fe_pct = self.fe_degradation_pct(operating_hours)

        return {
            "co_production_rate_g_h": co_rate,
            "sec_kwh_t_co": sec,
            "cell_voltage_V": V,
            "faradaic_efficiency": FE,
            "fe_relative_pct": fe_pct,
        }
