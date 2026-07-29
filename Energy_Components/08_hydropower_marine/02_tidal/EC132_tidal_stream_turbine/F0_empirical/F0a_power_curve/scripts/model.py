"""EC132 — Tidal Stream Turbine — F0a empirical power curve.

Simplest fidelity: a tabulated power curve P(v) vs current speed, interpolated with
numpy.interp. Zero below cut-in, cube-law to rated, flat at rated to cut-out, zero above.
Data source: EMEC Tidal Resource Guide; Fraenkel (2002); reuses EC132 F1a parameters.
NumPy only.
"""
import numpy as np


class TidalStreamF0a:
    def __init__(self, params):
        r = params["rated"]
        c = params["power_curve"]
        self.rated_power = r["rated_power_kw"]["value"]
        self.cut_in = r["cut_in_speed_ms"]["value"]
        self.cut_out = r["cut_out_speed_ms"]["value"]
        self.v_tab = np.asarray(c["current_speed_ms"]["value"], dtype=float)
        self.P_tab = np.asarray(c["power_kw"]["value"], dtype=float)

    def power_kw(self, v):
        """Electrical power (kW) at current speed v (m/s)."""
        v = np.asarray(v, dtype=float)
        P = np.interp(v, self.v_tab, self.P_tab)
        P = np.where(v < self.cut_in, 0.0, P)
        return np.where(v > self.cut_out, 0.0, P)

    def capacity_factor(self, v):
        return self.power_kw(v) / self.rated_power
