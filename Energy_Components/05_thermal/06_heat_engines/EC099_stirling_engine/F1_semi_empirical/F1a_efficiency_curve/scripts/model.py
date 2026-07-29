"""
EC099 — Stirling Engine — F1a Efficiency Curve

Semi-empirical efficiency model for a small kinematic Stirling engine
(micro-CHP / small distributed power scale, 1-100 kW class).

Model:
    eta_carnot  = 1 - T_c / T_h               [Carnot upper bound, K]
    eta_design  = f_carnot * eta_carnot       [achievable design eta]
    eta(PLR)    = eta_design * (1 - a*(1-PLR)^2)
    eta(PLR<PLR_min) = 0                      [stalled / minimum load]
    eta         = min(eta, eta_carnot)         [hard Carnot cap]

The fraction-of-Carnot factor f_carnot lumps together regenerator
effectiveness, dead volume, mechanical friction, hysteresis, and
windage; for real Stirling engines f_carnot ~ 0.4-0.6.

Reference:
    Kongtragool & Wongwises (2003), Renew. Sustain. Energy Rev. 7, 131-154.
    Cinar et al. (2005), Appl. Thermal Eng. 25, 1845-1854.
    Walker (1980), Stirling Engines, Oxford University Press.
"""

import numpy as np


class StirlingEngineF1a:
    """Semi-empirical Stirling engine efficiency-curve model."""

    def __init__(self, params: dict):
        e = params["engine"]
        self.P_rated    = e["P_rated"]["value"]      # W
        self.f_carnot   = e["f_carnot"]["value"]     # -
        self.a_partload = e["a_partload"]["value"]   # -
        self.T_h        = e["T_h"]["value"]          # degC
        self.T_c        = e["T_c"]["value"]          # degC
        self.PLR_min    = e["PLR_min"]["value"]      # -

    # ------------------------------------------------------------------

    def carnot_efficiency(self, T_h_c, T_c_c):
        """Carnot efficiency from inputs in deg C."""
        T_h = np.asarray(T_h_c, dtype=float) + 273.15
        T_c = np.asarray(T_c_c, dtype=float) + 273.15
        return 1.0 - T_c / T_h

    def cycle_efficiency(self, PLR, T_h_c=None, T_c_c=None):
        """
        Stirling cycle efficiency at given part-load ratio and reservoir
        temperatures.
        """
        PLR = np.asarray(PLR, dtype=float)
        T_h = self.T_h if T_h_c is None else np.asarray(T_h_c, dtype=float)
        T_c = self.T_c if T_c_c is None else np.asarray(T_c_c, dtype=float)

        eta_carnot = self.carnot_efficiency(T_h, T_c)
        eta_design = self.f_carnot * eta_carnot
        eta = eta_design * (1.0 - self.a_partload * (1.0 - PLR) ** 2)

        # Apply Carnot cap
        eta = np.minimum(eta, eta_carnot)
        # Below minimum load
        eta = np.where(PLR < self.PLR_min, 0.0, eta)
        eta = np.where(PLR <= 0.0, 0.0, eta)
        return np.clip(eta, 0.0, 1.0)

    def power_output(self, PLR):
        """Mechanical / electrical output power [W]."""
        PLR = np.asarray(PLR, dtype=float)
        out = np.where(PLR < self.PLR_min, 0.0, np.clip(PLR, 0.0, 1.0)) * self.P_rated
        return out

    def heat_input(self, PLR, T_h_c=None, T_c_c=None):
        """Thermal heat input from hot reservoir [W]."""
        P_out = self.power_output(PLR)
        eta = self.cycle_efficiency(PLR, T_h_c, T_c_c)
        return np.where(eta > 0, P_out / eta, 0.0)

    def heat_rejected(self, PLR, T_h_c=None, T_c_c=None):
        """Heat rejected to cold reservoir [W]."""
        return self.heat_input(PLR, T_h_c, T_c_c) - self.power_output(PLR)
