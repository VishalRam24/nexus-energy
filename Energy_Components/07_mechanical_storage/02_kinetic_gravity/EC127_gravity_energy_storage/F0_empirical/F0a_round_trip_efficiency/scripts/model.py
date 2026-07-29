"""F0a empirical round-trip-efficiency lookup for EC127 Gravity Energy Storage.

F0 = simplest fidelity: an empirical efficiency curve. NumPy only.

A 1-D part-load round-trip-efficiency (RTE) lookup plus self-discharge.
RTE_rated = (eta_motor*eta_drive)_charge * (eta_drive*eta_generator)_discharge
    = (0.96*0.95)*(0.95*0.96) = 0.8318. Self-discharge negligible (potential
    energy stored mechanically).

Source (reused from F1a): Botha & Kamper (2019), J. Energy Storage, 23, 159-174; Berrada et al. (2017), Energy Convers. Manag., 137, 191-200 (reused from F1a)
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
