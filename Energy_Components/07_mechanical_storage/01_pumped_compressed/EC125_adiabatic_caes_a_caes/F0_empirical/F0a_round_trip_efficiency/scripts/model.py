"""F0a empirical round-trip-efficiency lookup for EC125 Adiabatic CAES (A-CAES).

F0 = simplest fidelity: an empirical efficiency curve. NumPy only.

A 1-D part-load round-trip-efficiency (RTE) lookup plus self-discharge.
RTE_rated from F1: E_out = 335*0.86*0.97 = 279.46 kJ/kg;
    E_in = 310/(0.80*0.97) = 399.48 kJ/kg; RTE = 279.46/399.48 = 0.6996
    (~0.70; heat recovered from TES, no supplemental fuel).

Source (reused from F1a): Barbour et al. (2015), Renew. Sustain. Energy Rev. 45, 598-614; Wolf & Budt (2014), Applied Energy, 125, 158-164 (reused from F1a)
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
