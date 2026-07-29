"""F0a empirical lookup model for a combustion genset / turbine.

Net LHV electrical efficiency is stored as a 1-D breakpoint table over the
part-load ratio (PLR) and evaluated with ``numpy.interp``. A linear ambient
derate scales efficiency away from the ISO reference temperature.

    eta(PLR, T_amb) = interp(PLR, plr_bp, eta_bp) * (1 - f_amb*(T_amb - T_ref))

Data source: see ``data/parameters.json`` (breakpoints generated from the
F1a semi-empirical quadratic part-load curve). NumPy only.
"""
import numpy as np


class EfficiencyLookup:
    def __init__(self, plr_bp, eta_bp, eta_rated, T_amb_ref, f_amb_coeff, PLR_min):
        self.plr_bp = np.asarray(plr_bp, dtype=float)
        self.eta_bp = np.asarray(eta_bp, dtype=float)
        self.eta_rated = float(eta_rated)
        self.T_amb_ref = float(T_amb_ref)
        self.f_amb_coeff = float(f_amb_coeff)
        self.PLR_min = float(PLR_min)

    def efficiency(self, plr, T_amb=None):
        """Return net LHV electrical efficiency at the given PLR (and ambient)."""
        plr = np.clip(np.asarray(plr, dtype=float), self.plr_bp[0], self.plr_bp[-1])
        eta = np.interp(plr, self.plr_bp, self.eta_bp)
        if T_amb is not None:
            eta = eta * (1.0 - self.f_amb_coeff * (np.asarray(T_amb, dtype=float) - self.T_amb_ref))
        return eta
