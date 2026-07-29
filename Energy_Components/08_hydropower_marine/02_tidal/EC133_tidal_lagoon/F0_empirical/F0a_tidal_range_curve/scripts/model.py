"""EC133 — Tidal Lagoon — F0a empirical tidal-range power curve.

Simplest fidelity: tabulated mean power vs tidal-range amplitude h, P_avg ~ h^2 (bidirectional ebb+flood),
interpolated with numpy.interp. Below h_min_operation power is zero.
Data source: Aggidis & Feather (2012); Xiao et al. (2020); reuses EC133 F1a parameters.
NumPy only.
"""
import numpy as np


class TidalRangeF0a:
    def __init__(self, params):
        r = params["rated"]
        c = params["power_curve"]
        self.h_design = r["h_design"]["value"]
        self.P_avg_rated = r["P_avg_rated"]["value"]
        self.h_min = r["h_min_operation"]["value"]
        self.h_tab = np.asarray(c["h_amplitude_m"]["value"], dtype=float)
        self.P_tab = np.asarray(c["P_avg_mw"]["value"], dtype=float)

    def mean_power_mw(self, h):
        """Mean electrical power (MW) for tidal-range amplitude h (m)."""
        h = np.asarray(h, dtype=float)
        P = np.interp(h, self.h_tab, self.P_tab)
        return np.where(h < self.h_min, 0.0, P)

    def capacity_factor(self, h):
        return self.mean_power_mw(h) / self.P_avg_rated
