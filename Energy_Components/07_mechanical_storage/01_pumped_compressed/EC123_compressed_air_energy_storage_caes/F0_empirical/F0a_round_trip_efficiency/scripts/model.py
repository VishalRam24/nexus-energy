"""F0a empirical round-trip-efficiency lookup for EC123 Compressed Air Energy Storage (CAES).

F0 = simplest fidelity: an empirical efficiency curve. NumPy only.

A 1-D part-load round-trip-efficiency (RTE) lookup plus self-discharge.
RTE_rated computed from F1 specific work: E_out = w_exp*eta_e*eta_g = 470*0.82*0.97;
    E_in = w_comp/(eta_c*eta_m) = 280/(0.72*0.97); electrical-only RTE = 0.5363
    (diabatic; supplemental fuel ~4200 kJ/kWh_e not counted in electrical RTE).

Source (reused from F1a): Budt et al. (2016), Applied Energy, 170, 250-268; Luo et al. (2015), Applied Energy, 137, 511-536 (reused from F1a)
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
