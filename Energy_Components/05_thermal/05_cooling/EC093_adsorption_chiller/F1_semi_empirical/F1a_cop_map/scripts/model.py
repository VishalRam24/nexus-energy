"""
EC093 — Adsorption Chiller — F1a COP Map Model

Single-stage silica-gel/water adsorption chiller (also applicable to
zeolite-water).  Driven by low-grade heat at T_hot (typical 60–95 C),
rejects to a cooling stream at T_cool (~25–35 C), and produces chilled
water at T_chilled (~6–18 C).  Reversible (cascade-Carnot) cooling COP:

    COP_rev_c = (T_gen - T_cool) / T_gen * T_chilled / (T_cool - T_chilled)

The actual COP is a fraction of this reversible limit:

    COP_c = eta_rev * COP_rev_c

with eta_rev ~ 0.30–0.40 for silica-gel/water cycles.  Driving and
rejected heat duties follow from the energy balance:

    Q_cool   = Q_rated * PLR
    Q_drive  = Q_cool / COP_c
    Q_reject = Q_cool + Q_drive

References:
    Saha, B.B., Boelman, E.C., Kashiwagi, T. (1995).
        Heat Recovery Systems & CHP 15, 581-590.
    Wang, R.Z., Oliveira, R.G. (2006). Prog. Energy Combust. Sci. 32, 424-458.
    Herold, Radermacher & Klein (2016). Absorption Chillers and Heat Pumps.
"""

import numpy as np


class AdsorptionChillerF1a:
    """Silica-gel / water adsorption chiller — characteristic-eq COP."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated = u["rated_cooling_capacity"]["value"]   # kW_th
        self.eta_rev = u["carnot_fraction"]["value"]
        self.aux_power = u["auxiliary_power"]["value"]

    def cop_reversible_cooling(self, T_hot_c, T_cool_c, T_chilled_c):
        """Reversible cooling COP (cascade Carnot engine + Carnot chiller)."""
        T_gen  = np.asarray(T_hot_c,    dtype=float) + 273.15
        T_cool = np.asarray(T_cool_c,   dtype=float) + 273.15
        T_chw  = np.asarray(T_chilled_c, dtype=float) + 273.15
        dT_drive = T_gen - T_cool
        dT_lift  = T_cool - T_chw
        dT_drive_safe = np.where(dT_drive > 1e-6, dT_drive, 1e-6)
        dT_lift_safe  = np.where(dT_lift  > 1e-6, dT_lift,  1e-6)
        eta_engine = dT_drive_safe / T_gen
        cop_carnot_chiller = T_chw / dT_lift_safe
        return np.maximum(eta_engine * cop_carnot_chiller, 0.0)

    def cop(self, T_hot_c, T_cool_c, T_chilled_c):
        cop_rev = self.cop_reversible_cooling(T_hot_c, T_cool_c, T_chilled_c)
        cop = self.eta_rev * cop_rev
        return np.clip(cop, 0.05, 0.85)

    def cooling_power(self, plr=1.0):
        return self.Q_rated * np.asarray(plr, dtype=float)

    def driving_heat(self, T_hot_c, T_cool_c, T_chilled_c, plr=1.0):
        q = self.cooling_power(plr)
        c = self.cop(T_hot_c, T_cool_c, T_chilled_c)
        return q / c

    def heat_rejection(self, T_hot_c, T_cool_c, T_chilled_c, plr=1.0):
        q = self.cooling_power(plr)
        h = self.driving_heat(T_hot_c, T_cool_c, T_chilled_c, plr)
        return q + h

    def electrical_input(self, plr=1.0):
        plr = np.asarray(plr, dtype=float)
        return self.aux_power * np.where(plr > 0, 1.0, 0.0) + 0.0 * plr
