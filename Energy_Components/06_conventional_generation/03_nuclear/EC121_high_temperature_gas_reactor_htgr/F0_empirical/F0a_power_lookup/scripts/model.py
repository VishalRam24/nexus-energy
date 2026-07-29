"""F0a empirical power-map lookup for a nuclear reactor.

Net electrical power is stored as a 1-D breakpoint table over the load factor
(fraction of rated thermal power) and evaluated with ``numpy.interp``:

    P_elec(L) = interp(L, load_bp, pelec_bp)
    eta_net(L) = P_elec(L) / (P_thermal * L)

Breakpoints are built from the F1a power map P_elec = eta * P_thermal * L,
with a linear efficiency derate below the minimum stable load. NumPy only.
Data source: see ``data/parameters.json``.
"""
import numpy as np


class PowerLookup:
    def __init__(self, load_bp, pelec_bp, P_thermal_mw, eta_rated, load_min):
        self.load_bp = np.asarray(load_bp, dtype=float)
        self.pelec_bp = np.asarray(pelec_bp, dtype=float)
        self.P_thermal_mw = float(P_thermal_mw)
        self.eta_rated = float(eta_rated)
        self.load_min = float(load_min)

    def power_elec(self, load):
        """Net electrical power (MW_e) at the given load factor."""
        load = np.clip(np.asarray(load, dtype=float), self.load_bp[0], self.load_bp[-1])
        return np.interp(load, self.load_bp, self.pelec_bp)

    def efficiency(self, load):
        """Net thermal-to-electric efficiency at the given load factor."""
        load = np.clip(np.asarray(load, dtype=float), self.load_bp[0], self.load_bp[-1])
        p = self.power_elec(load)
        return np.where(load > 0, p / (self.P_thermal_mw * load), 0.0)
