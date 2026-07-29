"""F0a empirical round-trip-efficiency lookup for EC126 Flywheel Energy Storage (FESS).

F0 = simplest fidelity: an empirical efficiency curve. NumPy only.

A 1-D part-load round-trip-efficiency (RTE) lookup plus self-discharge.
RTE_rated = eta_motor*eta_gen = 0.95*0.95 = 0.9025.
    Self-discharge ~1%/h (windage + bearing drag) from F1 k_sd_per_hr.

Source (reused from F1a): Arani et al. (2017). Energies, 10, 1361. doi:10.3390/en10091361 (reused from F1a)
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
