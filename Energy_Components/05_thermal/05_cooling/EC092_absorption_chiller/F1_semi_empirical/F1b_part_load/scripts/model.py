"""
EC092 — Absorption Chiller — F1b Part-Load Model

Extends F1a (characteristic equation) with:
  1. Part-load COP curve: COP_PL = COP_ref * (f1 + f2*PLR + f3*PLR^2)
     LiBr-H2O absorption: COP improves slightly at moderate part-load
     then drops sharply below PLR~0.3 due to crystallization risk.
  2. Driving heat temperature effect: f_Thot = g1 + g2*(T_hot - T_hot_rated)
     Higher generator temperature improves COP up to a point.
  3. Crystallization protection: minimum PLR enforced.

References:
    Herold, Radermacher & Klein (2016), Absorption Chillers and Heat Pumps, 2nd ed.
    Gordon & Ng (2000), Cool Thermodynamics.
    ASHRAE Handbook — HVAC Systems and Equipment (2020), Chapter 2.
"""

import numpy as np


class AbsorptionChillerF1b:
    """Single-effect LiBr-H2O absorption chiller with part-load and temperature correction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_cool_rated = u["Q_cool_rated"]["value"]
        self.COP_rated = u["COP_rated"]["value"]
        self.T_hot_rated = u["T_hot_rated"]["value"]
        self.T_cw_rated = u["T_cw_rated"]["value"]
        self.T_chw_rated = u["T_chw_rated"]["value"]
        self.PLR_min = u["PLR_min"]["value"]
        self.PLR_cryst = u["crystallization_limit_PLR"]["value"]

        pc = u["plr_curve"]
        self.f1 = pc["f1"]["value"]
        self.f2 = pc["f2"]["value"]
        self.f3 = pc["f3"]["value"]

        tc = u["temp_curve"]
        self.g1 = tc["g1"]["value"]
        self.g2 = tc["g2"]["value"]

    # ------------------------------------------------------------------
    # Part-load factor
    # ------------------------------------------------------------------

    def f_plr(self, plr):
        """
        Part-load COP correction: f_PLR = f1 + f2*PLR + f3*PLR^2

        At PLR=1: f1+f2+f3 = 1.0 (rated conditions).
        Peak around PLR=0.5-0.8 for absorption systems.
        """
        plr = np.asarray(plr, dtype=float)
        plr_eff = np.maximum(plr, self.PLR_min)
        f = self.f1 + self.f2 * plr_eff + self.f3 * plr_eff ** 2
        return np.clip(f, 0.1, 1.5)

    # ------------------------------------------------------------------
    # Temperature correction
    # ------------------------------------------------------------------

    def f_temp(self, T_hot):
        """
        COP correction for driving heat temperature.
        f_Thot = g1 + g2*(T_hot - T_hot_rated)
        """
        T_hot = np.asarray(T_hot, dtype=float)
        return np.clip(
            self.g1 + self.g2 * (T_hot - self.T_hot_rated),
            0.5, 1.3,
        )

    # ------------------------------------------------------------------
    # COP
    # ------------------------------------------------------------------

    def cop(self, T_hot, T_cw, T_chw, plr=1.0):
        """
        COP at operating conditions.

        COP = COP_rated * f_PLR(PLR) * f_Thot(T_hot)

        Capped at 0.80 for single-effect LiBr-H2O.
        """
        plr = np.asarray(plr, dtype=float)
        f_p = self.f_plr(plr)
        f_t = self.f_temp(T_hot)

        cop_val = self.COP_rated * f_p * f_t
        return np.clip(cop_val, 0.0, 0.80)

    # ------------------------------------------------------------------
    # Heat flows
    # ------------------------------------------------------------------

    def cooling_capacity(self, plr=1.0):
        """Cooling output in kW."""
        plr = np.asarray(plr, dtype=float)
        plr_eff = np.maximum(plr, self.PLR_min)
        return self.Q_cool_rated * plr_eff

    def heat_input(self, T_hot, T_cw, T_chw, plr=1.0):
        """Generator heat input in kW."""
        Q_cool = self.cooling_capacity(plr)
        c = self.cop(T_hot, T_cw, T_chw, plr)
        safe_cop = np.where(c > 0.01, c, 0.01)
        return Q_cool / safe_cop

    def heat_rejection(self, T_hot, T_cw, T_chw, plr=1.0):
        """Heat rejected to cooling tower (kW). Q_reject = Q_gen + Q_cool."""
        Q_cool = self.cooling_capacity(plr)
        Q_gen = self.heat_input(T_hot, T_cw, T_chw, plr)
        return Q_gen + Q_cool
