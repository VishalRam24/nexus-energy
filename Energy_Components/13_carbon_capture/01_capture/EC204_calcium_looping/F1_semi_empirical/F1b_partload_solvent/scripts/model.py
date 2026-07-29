"""
EC204 — Calcium Looping (CaL) — F1b Part-Load + Sorbent Degradation Model

Extends F1a capture-rate/energy model with:
  1. Sorbent activity decay: X(N) = X_inf + (X0 - X_inf)*exp(-k_deact*N)
     where X_inf ~ 0.075 (residual activity), k_deact ~ 0.52 (Grasa 2006).
     Activity determines CO2 uptake per kg CaO.
  2. Part-load calcination duty: q_calc(PLR) = q_design * (a + b*PLR)
     At PLR=0.3: +25% calcination energy penalty.
  3. Sorbent makeup: fresh CaO added to maintain average activity.
     Effective activity = weighted average of aged + fresh CaO.
  4. Heat integration: exothermic carbonation partially offsets calcination.
  5. Capture rate vs energy penalty curve.

References:
    Grasa, G.S. & Abanades, J.C. (2006). Ind. Eng. Chem. Res., 45(26), 8846-8851.
    Charitos, A. et al. (2011). Ind. Eng. Chem. Res., 50(17), 9685-9695.
    Romano, M.C. (2012). Int. J. Greenhouse Gas Control, 10, 399-417.
"""

import numpy as np


class CalciumLoopingF1b:
    """Calcium looping CO2 capture — part-load + cyclic sorbent degradation model."""

    # Grasa & Abanades (2006) deactivation parameters
    X_INF = 0.075   # residual activity after many cycles
    K_DEACT = 0.52  # deactivation constant

    def __init__(self, params: dict):
        u = params["unit"]
        self.capture_rate_design = u["capture_rate_design"]["value"]
        self.T_carb = u["carbonation_T_degC"]["value"]
        self.T_calc = u["calcination_T_degC"]["value"]
        self.X0 = u["CaO_activity_fresh"]["value"]
        self.k_deg = u["k_deactivation"]["value"]
        self.f_makeup = u["sorbent_makeup_rate"]["value"]
        self.q_calc_design = u["heat_calcination_design"]["value"]  # GJ/tCO2
        self.E_el_design = u["electrical_design"]["value"]         # kWh/tCO2
        self.PLR_coeffs = u["PLR_calcination_coeffs"]["value"]
        self.MW_CO2 = u["MW_CO2"]["value"] / 1000.0  # kg/mol

    # ── Sorbent activity ──────────────────────────────────────────────────────

    def sorbent_activity(self, n_cycles):
        """CaO conversion activity X(N) using Grasa & Abanades (2006) model.
        X(N) = X_inf + (X0 - X_inf) * exp(-K_DEACT * N)
        Makeup rate shifts effective activity toward fresh (X0).
        """
        N = np.asarray(n_cycles, dtype=float)
        # Activity without makeup
        X_no_makeup = self.X_INF + (self.X0 - self.X_INF) * np.exp(-self.K_DEACT * N)
        # With makeup: effective activity = (1-f_makeup)*X_no_makeup + f_makeup*X0
        X_eff = (1.0 - self.f_makeup) * X_no_makeup + self.f_makeup * self.X0
        return np.clip(X_eff, self.X_INF, self.X0)

    def sorbent_activity_pct(self, n_cycles):
        """Activity as % of fresh value."""
        return self.sorbent_activity(n_cycles) / self.X0 * 100.0

    # ── Part-load calcination duty ────────────────────────────────────────────

    def _plr_calc_factor(self, plr):
        """Part-load calcination duty multiplier.
        At PLR=1: factor=1.0; at PLR=0.3: factor~1.25.
        """
        plr = np.asarray(plr, dtype=float)
        a, b = self.PLR_coeffs
        return a + b * plr

    # ── Main energy calculations ───────────────────────────────────────────────

    def calcination_duty_gj_ton(self, capture_rate, plr, n_cycles):
        """Specific calcination thermal duty (GJ/tCO2).
        q = q_design * PLR_factor * activity_correction
        As activity degrades, more CaO must circulate → higher heat demand.
        """
        cr = np.asarray(capture_rate, dtype=float)
        plr_f = self._plr_calc_factor(plr)
        X = self.sorbent_activity(n_cycles)
        # Activity correction: need more CaO when activity drops
        activity_correction = self.X0 / (X + 1e-6)
        activity_correction = np.clip(activity_correction, 1.0, 3.0)

        # Capture rate correction (higher CR = more difficult, higher duty)
        dCR = cr - self.capture_rate_design
        cr_factor = 1.0 + 2.0 * dCR ** 2 + 1.2 * dCR

        q = self.q_calc_design * plr_f * activity_correction * cr_factor
        return np.clip(q, 2.0, 10.0)

    def electrical_kwh_ton(self, plr, n_cycles):
        """Electrical energy (kWh/tCO2) for solids handling and fans."""
        plr = np.asarray(plr, dtype=float)
        X = self.sorbent_activity(n_cycles)
        # Fans scale with PLR
        fan_factor = 0.7 + 0.3 / (plr + 1e-6)
        # More sorbent circulation at lower activity
        circ_factor = self.X0 / (X + 1e-6)
        circ_factor = np.clip(circ_factor, 1.0, 3.0)
        E = self.E_el_design * fan_factor * circ_factor
        return np.clip(E, 10.0, 150.0)

    def co2_captured_kg_h(self, flue_gas_flow_mol_s, co2_concentration,
                           capture_rate, plr):
        """CO2 captured (kg/h)."""
        fg = np.asarray(flue_gas_flow_mol_s, dtype=float)
        xCO2 = np.asarray(co2_concentration, dtype=float)
        cr = np.asarray(capture_rate, dtype=float)
        plr = np.asarray(plr, dtype=float)
        co2_mol_s = fg * xCO2 * plr
        return co2_mol_s * cr * self.MW_CO2 * 3600.0

    def total_energy_penalty_pct(self, capture_rate, plr, n_cycles):
        """Total energy penalty as % of reference 500 MW plant output."""
        q_calc = self.calcination_duty_gj_ton(capture_rate, plr, n_cycles)
        e_elec = self.electrical_kwh_ton(plr, n_cycles)
        e_total = q_calc + e_elec / 277.78
        e_ref = self.q_calc_design + self.E_el_design / 277.78
        penalty_design = 22.0  # % (CaL slightly lower penalty than MEA due to heat integration)
        penalty = penalty_design * e_total / e_ref
        return np.clip(penalty, 8.0, 55.0)

    def compute(self, flue_gas_flow_mol_s, co2_concentration, capture_rate,
                plr, n_cycles):
        """Full computation returning all outputs."""
        co2 = self.co2_captured_kg_h(flue_gas_flow_mol_s, co2_concentration,
                                     capture_rate, plr)
        q_calc = self.calcination_duty_gj_ton(capture_rate, plr, n_cycles)
        e_elec = self.electrical_kwh_ton(plr, n_cycles)
        activity_pct = self.sorbent_activity_pct(n_cycles)
        penalty = self.total_energy_penalty_pct(capture_rate, plr, n_cycles)

        return {
            "co2_captured_kg_h": co2,
            "calcination_duty_gj_ton": q_calc,
            "electrical_kwh_ton": e_elec,
            "sorbent_activity_pct": activity_pct,
            "total_energy_penalty_pct": penalty,
        }
