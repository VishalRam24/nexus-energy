"""F0a empirical part-load efficiency table for EC104 Gas Engine CHP (CHP).

Simplest fidelity: two 1-D np.interp lookups giving electrical and useful-thermal
efficiency versus part-load ratio (PLR = P_el / P_el_rated). Breakpoints tabulated
from the component's F1a part-load relations. Pure NumPy, no scipy/ODEs.

Data source: US EPA CHP Catalog (2017); ASUE BHKW-Kenndaten (2011)
"""
import numpy as np


class EtaPartLoadTable:
    def __init__(self, plr_bp, eta_e_bp, eta_th_bp, P_el_rated, plr_min):
        self.plr_bp = np.asarray(plr_bp, dtype=float)
        self.eta_e_bp = np.asarray(eta_e_bp, dtype=float)
        self.eta_th_bp = np.asarray(eta_th_bp, dtype=float)
        self.P_el_rated = float(P_el_rated)
        self.plr_min = float(plr_min)

    def _lookup(self, table, plr):
        plr = np.clip(np.asarray(plr, dtype=float), 0.0, 1.0)
        val = np.interp(plr, self.plr_bp, table)
        return np.where(plr < self.plr_min, 0.0, val)

    def eta_el(self, plr):
        return self._lookup(self.eta_e_bp, plr)

    def eta_th(self, plr):
        return self._lookup(self.eta_th_bp, plr)

    def power_el(self, plr):
        return np.clip(np.asarray(plr, dtype=float), 0.0, 1.0) * self.P_el_rated
