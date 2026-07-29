"""EC134 — Oscillating Water Column (OWC) — F0a empirical wave-power curve.

Simplest fidelity: the device power matrix collapsed to a 1-D lookup of electrical
power vs significant wave height Hs (at a representative energy period), interpolated
with numpy.interp and clipped at rated power.
Data source: Falnes (2002); EMEC data; reuses EC134 F1a parameters. NumPy only.
"""
import numpy as np


class WavePowerF0a:
    def __init__(self, params):
        r = params["rated"]
        c = params["power_curve"]
        self.P_rated = r["P_rated_kw"]["value"]
        self.Te_ref = r["Te_ref_s"]["value"]
        self.Hs_tab = np.asarray(c["Hs_m"]["value"], dtype=float)
        self.P_tab = np.asarray(c["power_kw"]["value"], dtype=float)

    def power_kw(self, Hs):
        """Electrical power (kW) at significant wave height Hs (m)."""
        Hs = np.asarray(Hs, dtype=float)
        P = np.interp(Hs, self.Hs_tab, self.P_tab)
        return np.minimum(P, self.P_rated)

    def capacity_factor(self, Hs):
        return self.power_kw(Hs) / self.P_rated
