"""F0a empirical round-trip-efficiency lookup for EC122 Pumped Hydro Storage.

F0 = simplest fidelity: an empirical efficiency curve. NumPy only.

The model is a 1-D part-load efficiency lookup. Round-trip efficiency (RTE)
of PHS varies with the part-load (fraction of rated power) at which the units
operate: pumps/turbines run below their best-efficiency point at part load.
We store a small tabulated curve of RTE vs power fraction whose rated point
(fraction = 1.0) equals the product of the component efficiencies in the F1
datasheet:

    RTE_rated = eta_turbine * eta_generator * eta_pump * eta_motor
              = 0.90 * 0.97 * 0.88 * 0.97 = 0.7454

Source (reused from F1a): Rehman et al. (2015), Renewable and Sustainable
Energy Reviews, 44, 586-598.
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
