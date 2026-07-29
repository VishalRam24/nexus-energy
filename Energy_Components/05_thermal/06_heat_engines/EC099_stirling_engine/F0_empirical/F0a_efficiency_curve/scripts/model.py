"""F0a empirical efficiency-curve lookup for EC099 Stirling Engine (micro-CHP scale).

Simplest fidelity: a 1-D np.interp lookup of net electrical (cycle) efficiency
versus part-load ratio (PLR = P_out / P_rated). Breakpoints tabulated from the
component's F1a part-load relation. Pure NumPy, no scipy/ODEs.

Data source: Kongtragool & Wongwises (2003) RSER 7, 131-154; Cinar et al. (2005) ATE 25, 1845
"""
import numpy as np


class EfficiencyCurve:
    def __init__(self, plr_bp, eta_bp, P_rated, plr_min):
        self.plr_bp = np.asarray(plr_bp, dtype=float)
        self.eta_bp = np.asarray(eta_bp, dtype=float)
        self.P_rated = float(P_rated)
        self.plr_min = float(plr_min)

    def efficiency(self, plr):
        """Net electrical efficiency at a given part-load ratio (clipped to [0,1])."""
        plr = np.clip(np.asarray(plr, dtype=float), 0.0, 1.0)
        eta = np.interp(plr, self.plr_bp, self.eta_bp)
        eta = np.where(plr < self.plr_min, 0.0, eta)
        return eta

    def power_out(self, plr):
        return np.clip(np.asarray(plr, dtype=float), 0.0, 1.0) * self.P_rated

    def fuel_heat_in(self, plr):
        """Fuel/heat input power = P_out / eta (0 when off)."""
        eta = self.efficiency(plr)
        pout = self.power_out(plr)
        return np.where(eta > 0.0, pout / np.where(eta > 0.0, eta, 1.0), 0.0)
