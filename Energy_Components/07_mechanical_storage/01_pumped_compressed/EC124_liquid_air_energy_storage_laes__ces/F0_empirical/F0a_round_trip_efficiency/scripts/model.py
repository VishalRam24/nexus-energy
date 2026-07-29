"""F0a empirical round-trip-efficiency lookup for EC124 Liquid Air Energy Storage (LAES / CES).

F0 = simplest fidelity: an empirical efficiency curve. NumPy only.

A 1-D part-load round-trip-efficiency (RTE) lookup plus self-discharge.
RTE_rated from F1 specific energies: net discharge = w_disch*eta_p*eta_e*eta_g
    = 0.36*0.85*0.85*0.97 = 0.25232 kWh/kg; charge = 0.40 kWh/kg;
    RTE = 0.25232/0.40 = 0.6308 (stand-alone, no external cold/heat integration).

Source (reused from F1a): Morgan et al. (2015), Applied Energy, 137, 845-853; Sciacovelli et al. (2017), Applied Energy, 190, 84-98 (reused from F1a)
"""
import numpy as np


class StorageRTECurve:
    """Generic part-load round-trip-efficiency lookup for a storage device."""

    def __init__(self, frac_breakpoints, rte_breakpoints, rte_rated,
                 self_discharge_per_hr=0.0):
        self.frac = np.asarray(frac_breakpoints, dtype=float)
        self.rte = np.asarray(rte_breakpoints, dtype=float)
        self.rte_rated = float(rte_rated)
        self.self_discharge_per_hr = float(self_discharge_per_hr)

    def round_trip_efficiency(self, power_fraction):
        """RTE at a given fraction of rated power (np.interp, clamped)."""
        pf = np.clip(np.asarray(power_fraction, dtype=float), self.frac[0],
                     self.frac[-1])
        return np.interp(pf, self.frac, self.rte)

    def retained_fraction(self, hours):
        """Energy fraction retained after idle `hours` of self-discharge."""
        h = np.asarray(hours, dtype=float)
        return (1.0 - self.self_discharge_per_hr) ** h
